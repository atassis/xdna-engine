#!/usr/bin/env python3
"""Device gate: a full conv2d LAYER in bf16 (M1 path (b), the precision a clean
whole-net ESPCN parity needs). Combines the three validated pieces:
  host im2col  +  K-tiled 64x64x64 gemm-bfp16-ebs8 (B block-transposed)  +  accumulate.

ESPCN conv2: Cin=64, Cout=32, k=3, pad=1, H=W=8 -> M=64, K=576=9*64, N=32(pad 64).

Gates:
  (1) device bf16 conv  vs  the host bfp16 K-tiled MODEL  -> device correctness (~1e-3).
  (2) device bf16 conv  vs  the fp32 reference conv2d     -> the bf16 accuracy (context).

Run under the NPU lock:  ./run.sh verify_upscaler_bf16_conv2d.py
"""
import numpy as np
import ml_dtypes
import bricklib
from bricklib import _build_oneshot, iron, GEN
from verify_f2 import tile_pack, tile_unpack, golden_mod

Cin, Cout, k, stride, pad, H, W = 64, 32, 3, 1, 1, 8, 8
Ho = (H + 2*pad - k)//stride + 1
Wo = (W + 2*pad - k)//stride + 1
M, Kfull, N, T = Ho*Wo, Cin*k*k, 64, 64
nkb = Kfull // T
assert (M, Kfull) == (64, 576)

g, cc = golden_mod("gemm-bfp16-ebs8", "gemm_bfp16_ebs8.cc")
rng = np.random.default_rng(21)
x = rng.standard_normal((Cin, H, W)).astype(np.float32)
w = (rng.standard_normal((Cout, Cin, k, k)).astype(np.float32) * (1.0 / np.sqrt(Cin*k*k)))

def im2col(x, k, stride, pad):
    Cin, H, W = x.shape
    xp = np.pad(x, ((0,0),(pad,pad),(pad,pad)))
    Ho = (H+2*pad-k)//stride+1; Wo = (W+2*pad-k)//stride+1
    cols = np.zeros((Ho*Wo, Cin*k*k), np.float32); idx = 0
    for oy in range(Ho):
        for ox in range(Wo):
            cols[idx] = xp[:, oy*stride:oy*stride+k, ox*stride:ox*stride+k].reshape(-1); idx += 1
    return cols

A = im2col(x, k, stride, pad)                    # [M=64, K=576] fp32
B = np.zeros((Kfull, N), np.float32)             # [K=576, N=64] fp32, N padded 32->64
B[:, :Cout] = w.reshape(Cout, Cin*k*k).T         # [K, Cout]

# references
ref_fp32 = (A @ B)[:, :Cout]                     # true conv2d (fp32) [M, Cout]
ref_model = np.zeros((M, N), np.float32)         # host bfp16 K-tiled model (what the device computes)
for kb in range(nkb):
    ref_model += g.gemm_bfp16_ebs8_ref(A[:, kb*T:(kb+1)*T], B[kb*T:(kb+1)*T, :]).astype(np.float32)
ref_model = ref_model[:, :Cout]

# --- build the bf16 gemm design once ----------------------------------------
shim = GEN / "gemm_bfp16_ebs8_conv_shim.cc"
shim.write_text(f'#include <stdint.h>\n#include "{cc}"\n')
design = _build_oneshot("gemm_bfp16_ebs8_64x64x64", shim, [T*T, T*T], T*T,
                        [ml_dtypes.bfloat16, ml_dtypes.bfloat16], ml_dtypes.bfloat16, [])

def pack_B_blockT(Bblk):                          # 64x64 block -> per-8x8 transposed (mac_8x8_8x8T)
    return Bblk.reshape(8, 8, 8, 8).transpose(0, 2, 3, 1).reshape(-1)

def dev_gemm(A_kb, B_kb):
    ta = iron.tensor(np.ascontiguousarray(tile_pack(A_kb.astype(ml_dtypes.bfloat16), 8, 8)),
                     dtype=ml_dtypes.bfloat16, device="npu")
    tb = iron.tensor(np.ascontiguousarray(pack_B_blockT(B_kb.astype(ml_dtypes.bfloat16))),
                     dtype=ml_dtypes.bfloat16, device="npu")
    tc = iron.zeros((T*T,), dtype=ml_dtypes.bfloat16, device="npu")
    design(ta, tb, tc)
    return tile_unpack(np.array(tc.numpy(), copy=True), T, T, 8, 8).astype(np.float32)

# --- K-tile the bf16 conv on device -----------------------------------------
acc = np.zeros((M, N), np.float32)
for kb in range(nkb):
    acc += dev_gemm(A[:, kb*T:(kb+1)*T], B[kb*T:(kb+1)*T, :])
    print(f"  bf16 K-block {kb+1}/{nkb} done")
got = acc[:, :Cout]

def rel_l2(ref, go):
    ref, go = np.asarray(ref, np.float64).ravel(), np.asarray(go, np.float64).ravel()
    return float(np.linalg.norm(go - ref) / (np.linalg.norm(ref) + 1e-12))

r_model = rel_l2(ref_model, got)
r_fp32 = rel_l2(ref_fp32, got)
ok = r_model <= 3e-2 and r_fp32 <= 0.08
print(f"\nbf16 conv2d (ESPCN conv2 3x3, {nkb} K-blocks): dev_vs_bfp16model={r_model:.3e} (gate 3e-2) "
      f"dev_vs_fp32conv={r_fp32:.3e} (gate 8e-2) -> {'PASS' if ok else 'FAIL'}")
