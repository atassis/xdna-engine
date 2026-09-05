#!/usr/bin/env python3
"""Device gate for the upscaler conv2d PATH (M1 Task 4): prove that
   host im2col -> the validated gemm-int8 brick -> reshape  ==  a reference conv2d,
on aie2p. This is the linchpin of Option A (conv2d = im2col -> whole-array GEMM).

We pick an overlapping-window conv that maps EXACTLY to one 64x64x64 gemm-int8 tile:
  2x2 conv, Cin=16, Cout=64, H=W=9, pad=0, stride=1
   -> Ho=Wo=8  => M = Ho*Wo = 64
   -> K = Cin*k*k = 16*4 = 64
   -> N = Cout = 64
int8 x int8 -> int32 is EXACT, so device must match the host int8 conv bit-for-bit.
(bf16 precision path = validate the parked bf16 GEMM brick; a separate follow-up.)

Run under the NPU lock:  ./run.sh verify_upscaler_conv2d.py
"""
import numpy as np
import bricklib
from verify_f2 import tile_pack, tile_unpack, golden_mod

# --- conv config chosen to fit ONE 64x64x64 gemm-int8 tile -------------------
Cin, Cout, k, stride, pad, H, W = 16, 64, 2, 1, 0, 9, 9
Ho = (H + 2 * pad - k) // stride + 1
Wo = (W + 2 * pad - k) // stride + 1
M, K, N = Ho * Wo, Cin * k * k, Cout
assert (M, K, N) == (64, 64, 64), (M, K, N)

rng = np.random.default_rng(7)
x = rng.integers(-127, 128, size=(Cin, H, W), dtype=np.int64).astype(np.int8)   # int8 activation
w = rng.integers(-127, 128, size=(Cout, Cin, k, k), dtype=np.int64).astype(np.int8)  # int8 weights


def im2col(x, k, stride, pad):
    Cin, H, W = x.shape
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    Ho = (H + 2 * pad - k) // stride + 1
    Wo = (W + 2 * pad - k) // stride + 1
    cols = np.zeros((Ho * Wo, Cin * k * k), np.int8)
    idx = 0
    for oy in range(Ho):
        for ox in range(Wo):
            cols[idx] = xp[:, oy*stride:oy*stride+k, ox*stride:ox*stride+k].reshape(-1)  # [Cin,ky,kx] C-order
            idx += 1
    return cols


A = im2col(x, k, stride, pad)                 # [M, K] int8, overlapping-window gather
B = w.reshape(Cout, Cin*k*k).T.copy()         # [K, Cout] int8  (== vit.rs patch_proj relabel-transpose)

g, cc = golden_mod("gemm-int8", "gemm_int8.cc")
ref = g.gemm_int8_ref(A, B)                    # [M,N] int32 == conv output (im2col GEMM)

# cross-check the lowering vs a DIRECT reference conv2d (host, int32) before the device run
xp = np.pad(x.astype(np.int64), ((0,0),(pad,pad),(pad,pad)))
direct = np.zeros((Ho, Wo, Cout), np.int64)
for oy in range(Ho):
    for ox in range(Wo):
        patch = xp[:, oy:oy+k, ox:ox+k]
        direct[oy, ox] = np.tensordot(patch, w.astype(np.int64), axes=([0,1,2],[1,2,3]))
assert np.array_equal(ref.reshape(Ho, Wo, Cout), direct), "im2col-GEMM != direct conv2d (host)"
print(f"host: im2col-GEMM == direct conv2d  (M,K,N={M},{K},{N}, 2x2 overlapping window) OK")

res = bricklib.verify_oneshot(
    "conv2d-im2col-gemm-int8", cc, "", "gemm_int8_64x64x64",
    inputs=[(tile_pack(A, 8, 8), np.int8), (tile_pack(B, 8, 8), np.int8)],
    out_numel=M * N, out_shape=(M, N),
    unpack=lambda flat: tile_unpack(flat, M, N, 8, 8),
    golden=ref, gate=3e-2, out_dt=np.int32)
print("\nconv2d device path:", "PASS" if res["ok"] else "FAIL",
      f"(rel_l2={res['rel_l2']:.3e}, run2run={res['run2run']:.2e})")
