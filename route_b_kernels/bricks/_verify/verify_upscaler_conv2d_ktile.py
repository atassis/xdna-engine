#!/usr/bin/env python3
"""Device gate for K-TILED conv2d at a REAL ESPCN 3x3 shape (M1 Task 4, extension).

ESPCN conv2: Cin=64, Cout=32, k=3, stride=1, pad=1, H=W=8 -> Ho=Wo=8.
  M = Ho*Wo = 64,  K = Cin*k*k = 64*9 = 576 = 9*64,  N = Cout = 32 (pad -> 64).
The 64x64x64 gemm-int8 brick handles ONE tile; a real conv needs K-tiling: run the
brick over the 9 K-blocks and accumulate the int32 partials on host. int8xint8->int32
is exact, so the K-tiled sum is bit-exact vs a reference conv2d.

Run under the NPU lock:  ./run.sh verify_upscaler_conv2d_ktile.py
"""
import numpy as np
import bricklib
from bricklib import _build_oneshot, iron, GEN
from verify_f2 import tile_pack, tile_unpack, golden_mod

Cin, Cout, k, stride, pad, H, W = 64, 32, 3, 1, 1, 8, 8
Ho = (H + 2*pad - k)//stride + 1
Wo = (W + 2*pad - k)//stride + 1
M, Kfull, N = Ho*Wo, Cin*k*k, Cout
NPAD, T = 64, 64                       # pad N 32->64; tile size 64
assert (M, Kfull) == (64, 576) and Kfull % T == 0, (M, Kfull)
nkb = Kfull // T                       # 9 K-blocks

rng = np.random.default_rng(11)
x = rng.integers(-127, 128, size=(Cin, H, W), dtype=np.int64).astype(np.int8)
w = rng.integers(-127, 128, size=(Cout, Cin, k, k), dtype=np.int64).astype(np.int8)


def im2col(x, k, stride, pad):
    Cin, H, W = x.shape
    xp = np.pad(x, ((0,0),(pad,pad),(pad,pad)))
    Ho = (H+2*pad-k)//stride+1; Wo = (W+2*pad-k)//stride+1
    cols = np.zeros((Ho*Wo, Cin*k*k), np.int8); idx = 0
    for oy in range(Ho):
        for ox in range(Wo):
            cols[idx] = xp[:, oy*stride:oy*stride+k, ox*stride:ox*stride+k].reshape(-1); idx += 1
    return cols


A = im2col(x, k, stride, pad)                              # [M=64, K=576] int8
B = np.zeros((Kfull, NPAD), np.int8)                       # [K=576, N=64] int8, N zero-padded 32->64
B[:, :Cout] = w.reshape(Cout, Cin*k*k).T                   # [K, Cout]  (vit.rs patch_proj relabel-transpose)

# host reference conv2d (direct), int32
xp = np.pad(x.astype(np.int64), ((0,0),(pad,pad),(pad,pad)))
direct = np.zeros((Ho, Wo, Cout), np.int64)
for oy in range(Ho):
    for ox in range(Wo):
        direct[oy, ox] = np.tensordot(xp[:, oy:oy+k, ox:ox+k], w.astype(np.int64), axes=([0,1,2],[1,2,3]))
ref = direct.reshape(M, Cout).astype(np.int32)

# --- build the 64x64x64 gemm-int8 device design ONCE ------------------------
g, cc = golden_mod("gemm-int8", "gemm_int8.cc")
shim = GEN / "gemm_int8_ktile_shim.cc"
shim.write_text(f'#include <stdint.h>\n#include "{cc}"\n')
design = _build_oneshot("gemm_int8_64x64x64", shim, [T*T, T*T], T*T,
                        [np.int8, np.int8], np.int32, [])

def dev_gemm(a_blk, b_blk):
    ta = iron.tensor(np.ascontiguousarray(tile_pack(a_blk, 8, 8)), dtype=np.int8, device="npu")
    tb = iron.tensor(np.ascontiguousarray(tile_pack(b_blk, 8, 8)), dtype=np.int8, device="npu")
    out = iron.zeros((T*T,), dtype=np.int32, device="npu")
    design(ta, tb, out)
    return tile_unpack(out.numpy().reshape(-1).copy(), T, T, 8, 8).astype(np.int64)

# --- K-tile on device, accumulate on host ----------------------------------
acc = np.zeros((M, NPAD), np.int64)
for kb in range(nkb):
    A_kb = A[:, kb*T:(kb+1)*T]                             # [64,64] int8
    B_kb = B[kb*T:(kb+1)*T, :]                             # [64,64] int8
    acc += dev_gemm(A_kb, B_kb)
    print(f"  K-block {kb+1}/{nkb} done")

got = acc[:, :Cout].astype(np.int32)                      # drop the N-pad -> [M, Cout]
rl2 = float(np.linalg.norm((got - ref).ravel().astype(np.float64)) /
            (np.linalg.norm(ref.ravel().astype(np.float64)) + 1e-30))
bitexact = np.array_equal(got, ref)
print(f"\nK-tiled conv2d (ESPCN conv2 3x3, Cin=64 Cout=32, {nkb} K-blocks): "
      f"rel_l2={rl2:.3e} bit_exact={bitexact} -> {'PASS' if bitexact else 'FAIL'}")
