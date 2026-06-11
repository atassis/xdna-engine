#!/usr/bin/env python3
"""Fused FFN sub-block on XDNA2, verified vs ONNX — the first assembled fused slice.

FFN (macaron half) = LN -> linear1[768->3072] -> SiLU -> linear2[3072->768], then
x + 0.5*out. We run it as TWO whole-array (8-col) fused-epilogue dispatches:
  mm1 = silu(norm @ W1' + bias1')     [whole_array silu xclbin]
  mm2 =       silu_out @ W2 + b2       [whole_array bias xclbin]
with the LayerNorm affine FOLDED into mm1 (so the NPU LN is normalize-only):
  W1'   = g[:,None] * W1           (g = norm_feed_forward1.weight)
  bias1' = beta @ W1 + b1          (beta = norm_feed_forward1.bias, b1 = linear1.bias)
and each matmul's bias carried via K-augmentation (one extra k-block; see
run_npu_mm_silu_wa.py). Only ~2 NPU dispatches for the whole FFN vs ~4-5 ops unfused.

Verifies the linear output vs ONNX `ffn1_l2` and the residual vs `after_ffn1`.
Run on a freed NPU:  .venv-iron/bin/python scripts/verify_fused_ffn.py
"""
import os, sys, time
import numpy as np
from ml_dtypes import bfloat16

A = "artifacts"
WA = "mlir-aie/programming_examples/basic/matrix_multiplication/whole_array/build"
M_PAD, K1, K2, FF, D = 512, 768, 3072, 3072, 768
f32 = lambda x: np.asarray(x, np.float32)
bf = lambda x: f32(x).astype(bfloat16)
W = lambda k: np.load(f"{A}/weights/{k}.npy")
R = lambda k: np.load(f"{A}/refs/{k}.npy")


def wa_epilogue(mode, M, K, N, A_real, B_real, bias):
    """Run one whole-array fused-epilogue matmul on the NPU (K-augmented bias).
    A_real[M,K], B_real[K,N], bias[N] -> C[M,N] bf16 (silu if mode=='silu')."""
    import pyxrt
    m = k = n = 32
    Kaug = K + k
    suffix = f"{M}x{Kaug}x{N}_{m}x{k}x{n}_8c_{mode}"
    xclbin = f"{WA}/final_{suffix}.xclbin"; insts = f"{WA}/insts_{suffix}.txt"
    for p in (xclbin, insts):
        if not os.path.exists(p):
            sys.exit(f"missing {p}")
    A_aug = np.zeros((M, Kaug), bfloat16); A_aug[:, :K] = bf(A_real); A_aug[:, K] = bfloat16(1.0)
    B_aug = np.zeros((Kaug, N), bfloat16); B_aug[:K, :] = bf(B_real); B_aug[K, :] = bf(bias)
    instr = np.fromfile(insts, np.uint32)
    xb = pyxrt.xclbin(xclbin); kname = xb.get_kernels()[0].get_name()
    d = pyxrt.device(0); d.register_xclbin(xb)
    kk = pyxrt.kernel(pyxrt.hw_context(d, xb.get_uuid()), kname)
    TO = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
    FROM = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
    Ab = np.ascontiguousarray(A_aug).view(np.uint16); Bb = np.ascontiguousarray(B_aug).view(np.uint16)
    bi = pyxrt.bo(d, instr.nbytes, pyxrt.bo.cacheable, kk.group_id(1))
    ba = pyxrt.bo(d, Ab.nbytes, pyxrt.bo.host_only, kk.group_id(3))
    bb = pyxrt.bo(d, Bb.nbytes, pyxrt.bo.host_only, kk.group_id(4))
    bc = pyxrt.bo(d, M * N * 2, pyxrt.bo.host_only, kk.group_id(5))
    bt = pyxrt.bo(d, 1, pyxrt.bo.host_only, kk.group_id(6)); btr = pyxrt.bo(d, 4, pyxrt.bo.host_only, kk.group_id(7))
    bi.write(instr.tobytes(), 0); bi.sync(TO)
    ba.write(Ab.tobytes(), 0); ba.sync(TO)
    bb.write(Bb.tobytes(), 0); bb.sync(TO)
    t0 = time.perf_counter()
    kk(3, bi, instr.size, ba, bb, bc, bt, btr).wait()
    dt = time.perf_counter() - t0
    bc.sync(FROM)
    C = np.frombuffer(bc.read(M * N * 2, 0), np.uint16).view(bfloat16).reshape(M, N)
    return f32(C), dt


def rel(a, b):
    a, b = f32(a), f32(b)
    return np.abs(a - b).max() / (np.abs(b).max() + 1e-9)


def main():
    x = f32(R("block_in")[0])                              # [400,768]
    Tn = x.shape[0]
    # LayerNorm normalize-only (host); affine folded into mm1
    mu = x.mean(-1, keepdims=True); var = x.var(-1, keepdims=True)
    norm = (x - mu) / np.sqrt(var + 1e-5)
    g, beta = W("norm_feed_forward1.weight"), W("norm_feed_forward1.bias")
    W1, b1 = W("feed_forward1.linear1.weight"), W("feed_forward1.linear1.bias")
    W2, b2 = W("feed_forward1.linear2.weight"), W("feed_forward1.linear2.bias")
    W1p = g[:, None] * W1                                  # fold LN scale into W1
    b1p = beta @ W1 + b1                                   # fold LN shift + linear1 bias

    # pad rows 400->512 for the M=512 xclbins
    def pad(a): p = np.zeros((M_PAD, a.shape[1]), np.float32); p[:Tn] = a; return p

    h1, t1 = wa_epilogue("silu", M_PAD, K1, FF, pad(norm), W1p, b1p)   # silu(norm@W1'+b1')
    h1 = h1[:Tn]
    l2, t2 = wa_epilogue("bias", M_PAD, K2, D, pad(h1), W2, b2)        # h1@W2 + b2
    l2 = l2[:Tn]

    r_l2 = rel(l2, R("ffn1_l2")[0])
    x_out = bf(x + 0.5 * l2)
    r_res = rel(x_out, R("after_ffn1")[0])
    print(f"[fused FFN] 2 NPU dispatches (whole_array 8-col, bias+SiLU epilogues, LN affine folded)")
    print(f"  device time: mm1={t1*1e3:.3f} ms  mm2={t2*1e3:.3f} ms  (LN+residual host)")
    print(f"  linear out (ffn1_l2) rel vs ONNX = {r_l2:.2e}  ({'PASS' if r_l2 < 0.05 else 'FAIL'})")
    print(f"  after_ffn1 (+residual) rel vs ONNX = {r_res:.2e}  ({'PASS' if r_res < 0.05 else 'FAIL'})")
    return 0 if (r_l2 < 0.05 and r_res < 0.05) else 1


if __name__ == "__main__":
    sys.exit(main())
