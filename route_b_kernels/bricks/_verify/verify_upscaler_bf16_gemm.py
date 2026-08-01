#!/usr/bin/env python3
"""Device-validate the PARKED gemm-bfp16-ebs8 brick (M1 path (b): the bf16 precision
path conv2d needs for clean whole-net parity vs the f32 oracle).

C[M,N] bf16 = A[M,K] bf16 @ B[K,N] bf16, A,B quantized to bfp16ebs8 (block-float,
blocks of 8 along K), fp32-accumulated. Tile-packed (8x8) layout, same as gemm-int8;
iron handles bf16 via ml_dtypes.bfloat16 (already used for bf16-out bricks).

Two gates:
  (1) device vs the bfp16 HOST MODEL (gemm_bfp16_ebs8_ref) -- should be small: the
      host model IS the device datapath. This is the brick-correctness gate.
  (2) device vs fp32 GEMM -- the bfp16 FORMAT accuracy (golden's own <=0.08 bar).

Run under the NPU lock:  ./run.sh verify_upscaler_bf16_gemm.py
"""
import numpy as np
import ml_dtypes
import bricklib
from bricklib import _build_oneshot, iron, GEN
from verify_f2 import tile_pack, tile_unpack, golden_mod

M = K = N = 64
g, cc = golden_mod("gemm-bfp16-ebs8", "gemm_bfp16_ebs8.cc")

rng = np.random.default_rng(3)
A = rng.standard_normal((M, K)).astype(np.float32)
B = (rng.standard_normal((K, N)).astype(np.float32) / np.sqrt(K))

ref_bfp16 = g.gemm_bfp16_ebs8_ref(A, B).astype(np.float32)   # host bfp16-datapath model
ref_fp32 = g.gemm_fp32(A, B).astype(np.float32)              # fp32 reference

def rel_l2(ref, got):
    ref, got = np.asarray(ref, np.float64).ravel(), np.asarray(got, np.float64).ravel()
    return float(np.linalg.norm(got - ref) / (np.linalg.norm(ref) + 1e-12))

shim = GEN / "gemm_bfp16_ebs8_shim.cc"
shim.write_text(f'#include <stdint.h>\n#include "{cc}"\n')
design = _build_oneshot("gemm_bfp16_ebs8_64x64x64", shim, [M*K, K*N], M*N,
                        [ml_dtypes.bfloat16, ml_dtypes.bfloat16], ml_dtypes.bfloat16, [])

def pack_B_blockT(B):
    # gemm-bfp16-ebs8 uses mac_8x8_8x8T (A @ B_tile^T), so each 8x8 B block is fed TRANSPOSED
    # (a DIFFERENT convention than gemm-int8's plain tile_pack). Verified in numpy: rel-L2 5e-8.
    return B.reshape(K // 8, 8, N // 8, 8).transpose(0, 2, 3, 1).reshape(-1)

def run():
    ta = iron.tensor(np.ascontiguousarray(tile_pack(A.astype(ml_dtypes.bfloat16), 8, 8)),
                     dtype=ml_dtypes.bfloat16, device="npu")
    tb = iron.tensor(np.ascontiguousarray(pack_B_blockT(B.astype(ml_dtypes.bfloat16))),
                     dtype=ml_dtypes.bfloat16, device="npu")
    tc = iron.zeros((M*N,), dtype=ml_dtypes.bfloat16, device="npu")
    design(ta, tb, tc)
    return tile_unpack(np.array(tc.numpy(), copy=True), M, N, 8, 8).astype(np.float32)

dev1 = run(); dev2 = run()
r_model = rel_l2(ref_bfp16, dev1)     # device vs host bfp16 model (brick correctness)
r_fp32 = rel_l2(ref_fp32, dev1)       # device vs fp32 (format accuracy)
run2run = float(np.linalg.norm((dev1 - dev2).ravel()))
nz = float(np.abs(dev1).sum())
ok = (nz > 0) and (r_model <= 3e-2) and (r_fp32 <= 0.08)
print(f"[gemm-bfp16-ebs8 64x64x64] dev_vs_bfp16model={r_model:.3e} (gate 3e-2) "
      f"dev_vs_fp32={r_fp32:.3e} (gate 8e-2) run2run={run2run:.2e} nz={nz:.2e} "
      f"-> {'PASS' if ok else 'FAIL'}")
# Without this the gate printed FAIL and still exited 0, so no drain could ever fail on it.
assert ok, (f"gemm-bfp16-ebs8 gate failed: dev_vs_bfp16model={r_model:.3e} "
            f"dev_vs_fp32={r_fp32:.3e} nz={nz:.2e}")
