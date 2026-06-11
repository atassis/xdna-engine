#!/usr/bin/env python3
"""Fused Conformer ConvModule (GigaAM block 0) on XDNA2, verified vs ONNX.

The conv module (recipe in scripts/block0_numpy.py + npu_asr/block.py):
  x[400,768] = after_mhsa
  -> norm_conv LayerNorm                          -> conv_ln  [400,768]
  -> pointwise_conv1 (1x1 = matmul + bias)        -> conv_pw1 [1536,400]   (NPU)
  -> GLU: a*sigmoid(g) over the 1536 split        -> conv_glu [768,400]    (host)
  -> depthwise_conv k5 'same' + bias              -> conv_dw  [768,400]    (NPU)
  -> batch_norm (exported as LayerNorm/channel)   -> conv_bn  [400,768]    (host)
  -> SiLU                                          -> conv_swish [768,400] (host)
  -> pointwise_conv2 (1x1 = matmul + bias)        -> conv_pw2 [768,400]    (NPU)
  -> residual: after_mhsa + pw2.T                 -> after_conv [400,768]

FUSION (matmuls + their bias epilogue on NPU; cheap glue on host):
  pw1 = (ln @ W1' + b1)                whole_array BIAS xclbin  512x800x1536  (Kaug=768+32)
  pw2 = (swish @ W2' + b2)             whole_array BIAS xclbin  512x800x768   (Kaug=768+32)
  W*' = conv.pointwise_conv*.weight[:, :, 0].T   ([out,in,1] -> [in,out] matmul orient)
Bias rides one K-augmented k-block (see run_npu_mm_silu_wa.py). GLU, batch_norm-LN,
SiLU and the residual stay on host; dwconv uses the existing NPU k=5 kernel.

The ConvModule has NO matmul-fusable activation chain (GLU sits between pw1 and dw;
SiLU sits before pw2 not after) so both matmuls use the plain BIAS (narrow) epilogue.

--dry mirrors the device math in numpy (bf16-rounding epilogues + bf16 dwconv) WITHOUT
touching the NPU, and reports per-stage rel vs the conv_* ONNX refs. Drop --dry to
dispatch the two matmuls + dwconv on a freed NPU (single-tenant; main session validates).

Build the xclbins (no NPU; see run_npu_mm_silu_wa.py header for the trap):
  source scripts/iron_env.sh
  MM=$PWD/mlir-aie/programming_examples/basic/matrix_multiplication/whole_array
  rm -f $MM/build/mm_32x32x32.o $MM/build/mm_silu_epilogue_32x32x32.o
  make -f $MM/Makefile.silu -C $MM NPU2=1 M=512 K=800 N=1536 n_aie_cols=8 no_silu=1 \
       build/final_512x800x1536_32x32x32_8c_bias.xclbin
  rm -f $MM/build/mm_32x32x32.o $MM/build/mm_silu_epilogue_32x32x32.o
  make -f $MM/Makefile.silu -C $MM NPU2=1 M=512 K=800 N=768 n_aie_cols=8 no_silu=1 \
       build/final_512x800x768_32x32x32_8c_bias.xclbin

Usage:
  .venv-iron/bin/python scripts/verify_fused_conv.py --dry   # wiring + numpy, no NPU
  .venv-iron/bin/python scripts/verify_fused_conv.py         # real run (NPU must be free)
"""
import argparse, os, sys, time
import numpy as np
from ml_dtypes import bfloat16

A = "artifacts"
WA = "mlir-aie/programming_examples/basic/matrix_multiplication/whole_array/build"
DW = "mlir-aie/programming_examples/ml/dwconv1d/build"
M_PAD, K, N1, N2, D, T = 512, 768, 1536, 768, 768, 400
EPS = 1e-5
f32 = lambda x: np.asarray(x, np.float32)
bf = lambda x: f32(x).astype(bfloat16)
bfr = lambda x: f32(f32(x).astype(bfloat16))   # f32 holding bf16-rounded values
W = lambda k: np.load(f"{A}/weights/{k}.npy")
R = lambda k: np.load(f"{A}/refs/{k}.npy")


def rel(a, b):
    a, b = f32(a), f32(b)
    return np.abs(a - b).max() / (np.abs(b).max() + 1e-9)


# ---- device-matched numpy mirrors (for --dry) ----
def narrow_ref_bf16(z_f32):
    """bias/narrow epilogue: bf16-round the f32 accumulator (matches device)."""
    return f32(z_f32.astype(bfloat16))


def mm_bias_numpy(A_real, B_real, bias):
    """numpy mirror of the whole_array BIAS matmul: bf16 inputs, f32 accumulate,
    bf16-round out. A_real[M,K], B_real[K,N], bias[N] -> [M,N] f32(bf16)."""
    z = bf(A_real).astype(np.float32) @ bf(B_real).astype(np.float32) + f32(bias)
    return narrow_ref_bf16(z)


def dwconv_numpy(x_bf16, taps, bias):
    """numpy mirror of the NPU dwconv k=5 'same' kernel: bf16 in, fp32 5-tap MAC,
    bf16-round (then host adds bias, bf16-rounded — matches Ops.dwconv). x[C,T]."""
    xf = bf(x_bf16).astype(np.float32)
    w = bf(np.pad(f32(taps), ((0, 0), (0, 11))))[:, :5].astype(np.float32)  # bf16 taps
    C, Tt = xf.shape
    pad = np.pad(xf, ((0, 0), (2, 2)))
    out = np.zeros_like(xf)
    for i in range(5):
        out += w[:, i:i + 1] * pad[:, i:i + Tt]
    out = narrow_ref_bf16(out)                # kernel stores one bf16 round
    return bfr(out + f32(bias)[:, None])      # host bias add, bf16 round (Ops.dwconv)


# ---- NPU dispatchers ----
def wa_bias_npu(M, Kreal, Nn, A_real, B_real, bias):
    """One whole_array BIAS matmul on the NPU (K-augmented bias). -> [M,N] f32."""
    import pyxrt
    m = k = n = 32
    Kaug = Kreal + k
    suffix = f"{M}x{Kaug}x{Nn}_{m}x{k}x{n}_8c_bias"
    xclbin = f"{WA}/final_{suffix}.xclbin"; insts = f"{WA}/insts_{suffix}.txt"
    for p in (xclbin, insts):
        if not os.path.exists(p):
            sys.exit(f"missing {p}")
    A_aug = np.zeros((M, Kaug), bfloat16); A_aug[:, :Kreal] = bf(A_real); A_aug[:, Kreal] = bfloat16(1.0)
    B_aug = np.zeros((Kaug, Nn), bfloat16); B_aug[:Kreal, :] = bf(B_real); B_aug[Kreal, :] = bf(bias)
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
    bc = pyxrt.bo(d, M * Nn * 2, pyxrt.bo.host_only, kk.group_id(5))
    bt = pyxrt.bo(d, 1, pyxrt.bo.host_only, kk.group_id(6)); btr = pyxrt.bo(d, 4, pyxrt.bo.host_only, kk.group_id(7))
    bi.write(instr.tobytes(), 0); bi.sync(TO)
    ba.write(Ab.tobytes(), 0); ba.sync(TO)
    bb.write(Bb.tobytes(), 0); bb.sync(TO)
    t0 = time.perf_counter()
    kk(3, bi, instr.size, ba, bb, bc, bt, btr).wait()
    dt = time.perf_counter() - t0
    bc.sync(FROM)
    C = np.frombuffer(bc.read(M * Nn * 2, 0), np.uint16).view(bfloat16).reshape(M, Nn)
    return f32(C), dt


def dwconv_npu(x_bf16, taps, bias):
    """NPU depthwise conv1d k=5 'same' on [C,T], host bf16 bias add. -> [C,T] f32."""
    import pyxrt
    C, Tt = x_bf16.shape
    w = np.zeros((C, 16), np.float32); w[:, :5] = f32(taps)
    xclbin = f"{DW}/final.xclbin"; insts = f"{DW}/insts.bin"
    instr = np.fromfile(insts, np.uint32)
    xb = pyxrt.xclbin(xclbin); kname = xb.get_kernels()[0].get_name()
    d = pyxrt.device(0); d.register_xclbin(xb)
    kk = pyxrt.kernel(pyxrt.hw_context(d, xb.get_uuid()), kname)
    TO = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
    FROM = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
    X = np.ascontiguousarray(bf(x_bf16)).reshape(-1).view(np.uint16)
    Wv = np.ascontiguousarray(bf(w)).reshape(-1).view(np.uint16)
    bi = pyxrt.bo(d, instr.nbytes, pyxrt.bo.cacheable, kk.group_id(1))
    bx = pyxrt.bo(d, X.nbytes, pyxrt.bo.host_only, kk.group_id(3))
    bw = pyxrt.bo(d, Wv.nbytes, pyxrt.bo.host_only, kk.group_id(4))
    nb = C * Tt * 2
    by = pyxrt.bo(d, nb, pyxrt.bo.host_only, kk.group_id(5))
    bi.write(instr.tobytes(), 0); bi.sync(TO)
    bx.write(X.tobytes(), 0); bx.sync(TO)
    bw.write(Wv.tobytes(), 0); bw.sync(TO)
    t0 = time.perf_counter()
    kk(3, bi, instr.size, bx, bw, by).wait()
    dt = time.perf_counter() - t0
    by.sync(FROM)
    Y = np.frombuffer(by.read(nb, 0), np.uint16).view(bfloat16).reshape(C, Tt)
    return bfr(f32(Y) + f32(bias)[:, None]), dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="numpy mirror of device math, no NPU")
    a = ap.parse_args()

    # Conv-module input: after_mhsa (for the residual) and conv_ln (== LN(after_mhsa),
    # the matmul input). Start from conv_ln directly (its own LN already verified op-by-op
    # in block0_numpy.py); residual uses after_mhsa.
    res_in = bfr(R("after_mhsa")[0])           # [400,768] residual base
    ln = bfr(R("conv_ln")[0])                   # [400,768] = norm_conv(after_mhsa)

    # weights -> matmul orientation [in, out]
    W1 = W("conv.pointwise_conv1.weight")[:, :, 0].T   # [768,1536]
    b1 = W("conv.pointwise_conv1.bias")                # [1536]
    W2 = W("conv.pointwise_conv2.weight")[:, :, 0].T   # [768,768]
    b2 = W("conv.pointwise_conv2.bias")                # [768]
    dww = W("conv.depthwise_conv.weight")[:, 0, :]     # [768,5]
    dwb = W("conv.depthwise_conv.bias")                # [768]
    bn_g, bn_b = W("conv.batch_norm.weight"), W("conv.batch_norm.bias")

    def pad(x):  # rows T -> M_PAD for the M=512 xclbins
        p = np.zeros((M_PAD, x.shape[1]), np.float32); p[:T] = f32(x); return p

    mm = mm_bias_numpy if a.dry else (lambda Ar, Br, bb: wa_bias_npu(M_PAD, K,
                                      Br.shape[1], Ar, Br, bb)[0])
    dwc = dwconv_numpy if a.dry else (lambda x, t, b: dwconv_npu(x, t, b)[0])
    tag = "dry/numpy(device-matched)" if a.dry else "NPU"
    print(f"[fused ConvModule] backend={tag}  (matmuls+dwconv offloaded, glue on host)")
    print(f"  pw1 -> {WA}/final_512x800x1536_32x32x32_8c_bias.xclbin (bias)")
    print(f"  pw2 -> {WA}/final_512x800x768_32x32x32_8c_bias.xclbin  (bias)")
    print(f"  dw  -> {DW}/final.xclbin (k=5 same)")
    print(f"  glue (host): GLU split+gate, batch_norm-LN, SiLU, residual\n")

    # ---- pointwise_conv1 (NPU matmul + bias) ----  [T,768]@[768,1536] -> [T,1536]
    pw1 = mm(pad(ln), W1, b1)[:T]               # [400,1536]
    r_pw1 = rel(pw1.T, R("conv_pw1")[0])        # ref is [1536,400]
    print(f"  conv_pw1  (NPU mm+bias)   rel vs ONNX = {r_pw1:.2e}")

    # ---- GLU (host): split 1536 -> a[768],g[768]; a*sigmoid(g) ----  on [1536,T]
    p = pw1.T                                    # [1536,400]
    aa, gg = p[:D], p[D:]
    glu = bfr(aa / (1.0 + np.exp(-gg)))          # [768,400]
    r_glu = rel(glu, R("conv_glu")[0])
    print(f"  conv_glu  (host)          rel vs ONNX = {r_glu:.2e}")

    # ---- depthwise_conv k5 + bias (NPU) ----  [768,400] -> [768,400]
    dw = dwc(glu, dww, dwb)
    r_dw = rel(dw, R("conv_dw")[0])
    print(f"  conv_dw   (NPU dwconv)    rel vs ONNX = {r_dw:.2e}")

    # ---- batch_norm == LayerNorm over channels (host) ----  on [T,768]
    bnx = dw.T                                   # [400,768]
    mu = bnx.mean(-1, keepdims=True); var = bnx.var(-1, keepdims=True)
    bn = bfr((bnx - mu) / np.sqrt(var + EPS) * bn_g + bn_b)   # [400,768]
    r_bn = rel(bn, R("conv_bn")[0])
    print(f"  conv_bn   (host LN)       rel vs ONNX = {r_bn:.2e}")

    # ---- SiLU (host) ----  on [768,400]
    sw = bn.T                                    # [768,400]
    swish = bfr(sw / (1.0 + np.exp(-sw)))
    r_sw = rel(swish, R("conv_swish")[0])
    print(f"  conv_swish (host SiLU)    rel vs ONNX = {r_sw:.2e}")

    # ---- pointwise_conv2 (NPU matmul + bias) ----  [T,768]@[768,768] -> [T,768]
    pw2 = mm(pad(swish.T), W2, b2)[:T]           # [400,768]
    r_pw2 = rel(pw2.T, R("conv_pw2")[0])         # ref is [768,400]
    print(f"  conv_pw2  (NPU mm+bias)   rel vs ONNX = {r_pw2:.2e}")

    # ---- residual (host): after_mhsa + pw2 ----
    out = bfr(f32(res_in) + pw2)                 # [400,768]
    r_out = rel(out, R("after_conv")[0])
    print(f"\n  after_conv (+residual)    rel vs ONNX = {r_out:.2e}  ({'PASS' if r_out < 0.05 else 'FAIL'})")

    rels = [r_pw1, r_glu, r_dw, r_bn, r_sw, r_pw2, r_out]
    ok = all(x < 0.05 for x in rels)
    print(f"  ALL STAGES: {'PASS' if ok else 'FAIL'} (threshold 5e-2)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
