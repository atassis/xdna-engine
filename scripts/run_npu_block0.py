#!/usr/bin/env python3
"""Host-orchestrated GigaAM-v3 Conformer block 0, with heavy ops on the XDNA2 NPU.

Runs in .venv-iron (pyxrt + numpy + ml_dtypes). Reconstructs the verified block
recipe (see scripts/block0_numpy.py) in a bf16 dataflow — each op takes bf16
inputs, accumulates in fp32, rounds to bf16 — and offloads selected ops to the
NPU via their xclbins, driven op-by-op (pyxrt). Verifies every stage against the
ONNX intermediates in artifacts/refs (now within bf16 tolerance) and reports
per-op placement + end-to-end error vs the ONNX block output.

Currently on NPU: depthwise-conv1d (validated primitive). Others on host, with a
clear path to offload (LayerNorm, FFN matmuls) next.

Usage:
  .venv-iron/bin/python scripts/run_npu_block0.py --dry   # host-only, no NPU
  .venv-iron/bin/python scripts/run_npu_block0.py         # dwconv on NPU
"""
import argparse, json, os, sys
import numpy as np
from ml_dtypes import bfloat16

A = "artifacts"
NH, HD, T, C = 16, 48, 400, 768
EPS = 1e-5
DW_EX = "mlir-aie/programming_examples/ml/dwconv1d/build"


def bf16(x):
    return np.asarray(x, np.float32).astype(bfloat16)


def f32(x):
    return np.asarray(x, np.float32)


def chk(name, got, ref, tol):
    got, ref = f32(got), f32(ref)
    if got.shape != ref.shape:
        print(f"  {name:14s} SHAPE {got.shape} vs {ref.shape} ***"); return 1
    d = np.abs(got - ref)
    rel = d.max() / (np.abs(ref).max() + 1e-9)
    ok = rel < tol
    print(f"  {name:14s} max|Δ|={d.max():.4e}  mean={d.mean():.4e}  rel={rel:.2e}  {'ok' if ok else '**OFF**'}")
    return 0 if ok else 1


# ---- host ops (bf16 in -> fp32 accumulate -> bf16 out); NPU if eng given ----
def layernorm(x, w, b, eng=None):
    if eng is not None:
        norm = f32(eng.normalize(bf16(x)))          # NPU normalize-only
    else:
        xf = f32(x)
        norm = (xf - xf.mean(-1, keepdims=True)) / np.sqrt(xf.var(-1, keepdims=True) + EPS)
    return bf16(norm * f32(w) + f32(b))              # learned affine on host


def silu(x, eng=None):
    if eng is not None:
        return eng.run(bf16(x))
    xf = f32(x); return bf16(xf / (1.0 + np.exp(-xf)))


MM_EX = "mlir-aie/programming_examples/basic/matrix_multiplication/single_core/build"


def matmul(x, w, b=None, eng=None):
    """x[M,K] @ w[K,N] (+b) -> bf16. On NPU if eng given, else host."""
    y = eng.mm(f32(x), f32(w)) if eng is not None else (f32(x) @ f32(w))
    if b is not None: y = y + f32(b)
    return bf16(y)


class NpuMatmul:
    """bf16->f32 single_core matmuls on the NPU. Pads M->512; picks the xclbin by
    (K,N); tiles N=3072 (FFN linear1) as 2x the 768x1536 xclbin. Covers all block
    matmuls with 3 validated xclbins: (768,768),(3072,768),(768,1536)."""
    PAD_M = 512
    SHAPES = {(768, 768), (3072, 768), (768, 1536)}

    def __init__(self):
        import pyxrt
        self.pyxrt = pyxrt
        self.d = pyxrt.device(0)
        self.k = {}; self.instr = {}
        self.TO = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
        self.FROM = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
        for (K, N) in self.SHAPES:
            self._load(K, N)

    def _load(self, K, N):
        pyxrt = self.pyxrt
        suf = f"{self.PAD_M}x{K}x{N}_32x32x32"
        xb = pyxrt.xclbin(f"{MM_EX}/final_{suf}.xclbin")
        self.instr[(K, N)] = np.fromfile(f"{MM_EX}/insts_{suf}.txt", dtype=np.uint32)
        self.d.register_xclbin(xb)
        ctx = pyxrt.hw_context(self.d, xb.get_uuid())
        self.k[(K, N)] = pyxrt.kernel(ctx, xb.get_kernels()[0].get_name())

    def _one(self, A, B):
        """A[512,K] bf16, B[K,N] bf16 with (K,N) in SHAPES -> C[512,N] f32."""
        pyxrt = self.pyxrt
        K, N = B.shape
        k = self.k[(K, N)]; instr = self.instr[(K, N)]
        Ab = np.ascontiguousarray(A).view(np.uint16)
        Bb = np.ascontiguousarray(B).view(np.uint16)
        bi = pyxrt.bo(self.d, instr.nbytes, pyxrt.bo.cacheable, k.group_id(1))
        ba = pyxrt.bo(self.d, Ab.nbytes, pyxrt.bo.host_only, k.group_id(3))
        bb = pyxrt.bo(self.d, Bb.nbytes, pyxrt.bo.host_only, k.group_id(4))
        bc = pyxrt.bo(self.d, self.PAD_M * N * 4, pyxrt.bo.host_only, k.group_id(5))
        bt = pyxrt.bo(self.d, 1, pyxrt.bo.host_only, k.group_id(6))
        btr = pyxrt.bo(self.d, 4, pyxrt.bo.host_only, k.group_id(7))
        bi.write(instr.tobytes(), 0); bi.sync(self.TO)
        ba.write(Ab.tobytes(), 0); ba.sync(self.TO)
        bb.write(Bb.tobytes(), 0); bb.sync(self.TO)
        k(3, bi, instr.size, ba, bb, bc, bt, btr).wait()
        bc.sync(self.FROM)
        return np.frombuffer(bc.read(self.PAD_M * N * 4, 0), np.float32).reshape(self.PAD_M, N)

    def mm(self, A, B):
        M, K = A.shape; K2, N = B.shape
        assert K == K2, (A.shape, B.shape)
        Ap = np.zeros((self.PAD_M, K), np.float32); Ap[:M] = A
        Ap = bf16(Ap); Bb = bf16(B)
        if (K, N) in self.SHAPES:
            C = self._one(Ap, Bb)
        elif K == 768 and N == 3072:                       # FFN linear1: 2x N=1536
            C = np.concatenate([self._one(Ap, Bb[:, :1536]),
                                self._one(Ap, Bb[:, 1536:])], axis=1)
        else:
            raise ValueError(f"no xclbin for (K,N)=({K},{N})")
        return C[:M]


LN_EX = "mlir-aie/programming_examples/ml/layernorm/build"
SILU_EX = "mlir-aie/programming_examples/ml/silu/build"


class NpuLayerNorm:
    """normalize-only LayerNorm [400,768] on NPU (kernel hardcodes gamma=1,beta=0);
    the learned affine is applied on host. ABI: gid1 instr, gid3 in, gid4 out, 5/6/7 dummy."""
    def __init__(self):
        import pyxrt
        self.pyxrt = pyxrt
        xb = pyxrt.xclbin(f"{LN_EX}/final.xclbin")
        self.instr = np.fromfile(f"{LN_EX}/insts.bin", np.uint32)
        self.d = pyxrt.device(0); self.d.register_xclbin(xb)
        ctx = pyxrt.hw_context(self.d, xb.get_uuid())
        self.k = pyxrt.kernel(ctx, xb.get_kernels()[0].get_name())
        self.TO = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
        self.FROM = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE

    def normalize(self, x_bf16):                      # [400,768] bf16 -> [400,768] bf16 normalized
        pyxrt = self.pyxrt
        X = np.ascontiguousarray(x_bf16).reshape(-1).view(np.uint16)
        bi = pyxrt.bo(self.d, self.instr.nbytes, pyxrt.bo.cacheable, self.k.group_id(1))
        bx = pyxrt.bo(self.d, X.nbytes, pyxrt.bo.host_only, self.k.group_id(3))
        by = pyxrt.bo(self.d, X.nbytes, pyxrt.bo.host_only, self.k.group_id(4))
        bt = pyxrt.bo(self.d, 1, pyxrt.bo.host_only, self.k.group_id(5))
        bcp = pyxrt.bo(self.d, 8, pyxrt.bo.host_only, self.k.group_id(6))
        btr = pyxrt.bo(self.d, 1, pyxrt.bo.host_only, self.k.group_id(7))
        bi.write(self.instr.tobytes(), 0); bi.sync(self.TO)
        bx.write(X.tobytes(), 0); bx.sync(self.TO)
        self.k(3, bi, self.instr.size, bx, by, bt, bcp, btr).wait()
        by.sync(self.FROM)
        return np.frombuffer(by.read(X.nbytes, 0), np.uint16).view(bfloat16).reshape(T, C)


class NpuSilu:
    """SiLU on NPU (tanh-approx sigmoid). One xclbin per element count; covers the
    block's two sizes 400*768 and 400*3072. ABI: gid1 instr, gid3 in, gid4 out."""
    def __init__(self):
        import pyxrt
        self.pyxrt = pyxrt
        self.k = {}; self.instr = {}
        self.d = pyxrt.device(0)
        self.TO = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
        self.FROM = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
        for L in (T * C, T * 4 * C):                  # 307200, 1228800
            xb = pyxrt.xclbin(f"{SILU_EX}/final_{L}.xclbin")
            self.instr[L] = np.fromfile(f"{SILU_EX}/insts_{L}.bin", np.uint32)
            self.d.register_xclbin(xb)
            ctx = pyxrt.hw_context(self.d, xb.get_uuid())
            self.k[L] = pyxrt.kernel(ctx, xb.get_kernels()[0].get_name())

    def run(self, x_bf16):
        pyxrt = self.pyxrt
        shape = x_bf16.shape
        X = np.ascontiguousarray(x_bf16).reshape(-1).view(np.uint16)
        L = X.size
        k, instr = self.k[L], self.instr[L]
        bi = pyxrt.bo(self.d, instr.nbytes, pyxrt.bo.cacheable, k.group_id(1))
        bx = pyxrt.bo(self.d, X.nbytes, pyxrt.bo.host_only, k.group_id(3))
        by = pyxrt.bo(self.d, X.nbytes, pyxrt.bo.host_only, k.group_id(4))
        bi.write(instr.tobytes(), 0); bi.sync(self.TO)
        bx.write(X.tobytes(), 0); bx.sync(self.TO)
        k(3, bi, instr.size, bx, by).wait()
        by.sync(self.FROM)
        return np.frombuffer(by.read(X.nbytes, 0), np.uint16).view(bfloat16).reshape(shape)


# ---- NPU dwconv (the validated primitive) ----
class NpuDwconv:
    def __init__(self):
        import pyxrt
        self.pyxrt = pyxrt
        xb = pyxrt.xclbin(f"{DW_EX}/final.xclbin")
        self.instr = np.fromfile(f"{DW_EX}/insts.bin", dtype=np.uint32)
        self.kname = xb.get_kernels()[0].get_name()
        d = pyxrt.device(0); d.register_xclbin(xb)
        ctx = pyxrt.hw_context(d, xb.get_uuid())
        self.k = pyxrt.kernel(ctx, self.kname); self.d = d
        self.TO = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
        self.FROM = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE

    def __call__(self, x_bf16, taps_bf16):
        """x_bf16: [768,400], taps_bf16: [768,5] -> [768,400] bf16 (no bias)."""
        pyxrt = self.pyxrt
        X = np.ascontiguousarray(x_bf16).reshape(-1)
        w = np.zeros((C, 16), np.float32); w[:, :5] = f32(taps_bf16)
        W = bf16(w).reshape(-1)
        bo_i = pyxrt.bo(self.d, self.instr.nbytes, pyxrt.bo.cacheable, self.k.group_id(1))
        bo_x = pyxrt.bo(self.d, X.nbytes, pyxrt.bo.host_only, self.k.group_id(3))
        bo_w = pyxrt.bo(self.d, W.nbytes, pyxrt.bo.host_only, self.k.group_id(4))
        bo_y = pyxrt.bo(self.d, X.nbytes, pyxrt.bo.host_only, self.k.group_id(5))
        bo_i.write(self.instr.tobytes(), 0); bo_i.sync(self.TO)
        bo_x.write(X.view(np.uint16).tobytes(), 0); bo_x.sync(self.TO)
        bo_w.write(W.view(np.uint16).tobytes(), 0); bo_w.sync(self.TO)
        self.k(3, bo_i, self.instr.size, bo_x, bo_w, bo_y).wait()
        bo_y.sync(self.FROM)
        return np.frombuffer(bo_y.read(X.nbytes, 0), np.uint16).view(bfloat16).reshape(C, T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="host-only, no NPU")
    ap.add_argument("--host-norm-act", action="store_true",
                    help="keep LayerNorm+SiLU on host (tighter: approx kernels stay off NPU)")
    a = ap.parse_args()

    man = json.load(open(f"{A}/manifest.json"))
    W = {k: np.load(f"{A}/weights/{k}.npy") for k in man["weights"]}
    R = {k: np.load(f"{A}/refs/{k}.npy") for k in man["refs"]}
    Wb = {k: bf16(v) for k, v in W.items()}  # bf16 weights

    dw = None if a.dry else NpuDwconv()
    mm = None if a.dry else NpuMatmul()
    na = a.dry or a.host_norm_act
    ln_e = None if na else NpuLayerNorm()
    si_e = None if na else NpuSilu()
    where = "HOST (dry)" if a.dry else "NPU"
    na_where = "HOST" if na else "NPU"
    print(f"on {where}: dwconv + matmuls. LayerNorm+SiLU on {na_where}. "
          f"HOST: RoPE, softmax, attn score/ctx, GLU sigmoid, residuals. tol=bf16-level.\n")
    tol = 0.05
    fails = 0

    x = bf16(R["block_in"][0])  # [400,768]

    # FFN1
    ln = layernorm(x, Wb["norm_feed_forward1.weight"], Wb["norm_feed_forward1.bias"], eng=ln_e)
    h = silu(matmul(ln, Wb["feed_forward1.linear1.weight"], Wb["feed_forward1.linear1.bias"], eng=mm), eng=si_e)
    h2 = matmul(h, Wb["feed_forward1.linear2.weight"], Wb["feed_forward1.linear2.bias"], eng=mm)
    x = bf16(f32(x) + 0.5 * f32(h2))
    fails += chk("after_ffn1", x, R["after_ffn1"][0], tol)

    # MHSA
    ln = layernorm(x, Wb["norm_self_att.weight"], Wb["norm_self_att.bias"], eng=ln_e)
    xr = f32(ln).reshape(T, 1, NH, HD)
    cos, sin = R["pos_cos"], R["pos_sin"]; half = HD // 2
    rot = np.concatenate([-xr[..., half:], xr[..., :half]], -1)
    rope = bf16(xr * cos + rot * sin).reshape(T, C)[None]
    q = matmul(rope[0], Wb["self_attn.linear_q.weight"], Wb["self_attn.linear_q.bias"], eng=mm)
    k = matmul(rope[0], Wb["self_attn.linear_k.weight"], Wb["self_attn.linear_k.bias"], eng=mm)
    v = matmul(ln, Wb["self_attn.linear_v.weight"], Wb["self_attn.linear_v.bias"], eng=mm)
    qh = f32(q).reshape(T, NH, HD).transpose(1, 0, 2)
    kh = f32(k).reshape(T, NH, HD).transpose(1, 0, 2)
    vh = f32(v).reshape(T, NH, HD).transpose(1, 0, 2)
    sc = (qh @ kh.transpose(0, 2, 1)) / np.sqrt(HD)
    p = np.exp(sc - sc.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
    ctx = (bf16(p).astype(np.float32) @ vh).transpose(1, 0, 2).reshape(T, C)
    ao = matmul(bf16(ctx), Wb["self_attn.linear_out.weight"], Wb["self_attn.linear_out.bias"], eng=mm)
    x = bf16(f32(x) + f32(ao))
    fails += chk("after_mhsa", x, R["after_mhsa"][0], tol)

    # ConvModule
    ln = layernorm(x, Wb["norm_conv.weight"], Wb["norm_conv.bias"], eng=ln_e)
    xct = bf16(f32(ln).T)  # [768,400]
    pw1 = matmul(xct.T, bf16(W["conv.pointwise_conv1.weight"][:, :, 0].T),
                 W["conv.pointwise_conv1.bias"], eng=mm).T  # [1536,400]
    a_, g_ = f32(pw1[:768]), f32(pw1[768:])
    glu = bf16(a_ / (1.0 + np.exp(-g_)))  # [768,400]
    taps = bf16(W["conv.depthwise_conv.weight"][:, 0, :])  # [768,5]
    if a.dry:
        pad = np.pad(f32(glu), ((0, 0), (2, 2))); dwout = np.zeros((C, T), np.float32)
        for i in range(5): dwout += f32(taps)[:, i:i+1] * pad[:, i:i+T]
        dwout = bf16(dwout)
    else:
        dwout = dw(glu, taps)  # NPU
    dwout = bf16(f32(dwout) + f32(W["conv.depthwise_conv.bias"])[:, None])
    fails += chk("conv_dw(+bias)", dwout, R["conv_dw"][0], tol)
    bn = layernorm(f32(dwout).T, Wb["conv.batch_norm.weight"], Wb["conv.batch_norm.bias"], eng=ln_e)  # [400,768]
    sw = silu(f32(bn).T, eng=si_e)  # [768,400]
    pw2 = matmul(sw.T, bf16(W["conv.pointwise_conv2.weight"][:, :, 0].T),
                 W["conv.pointwise_conv2.bias"], eng=mm).T  # [768,400]
    x = bf16(f32(x) + f32(pw2).T)
    fails += chk("after_conv", x, R["after_conv"][0], tol)

    # FFN2
    ln = layernorm(x, Wb["norm_feed_forward2.weight"], Wb["norm_feed_forward2.bias"], eng=ln_e)
    h = silu(matmul(ln, Wb["feed_forward2.linear1.weight"], Wb["feed_forward2.linear1.bias"], eng=mm), eng=si_e)
    h2 = matmul(h, Wb["feed_forward2.linear2.weight"], Wb["feed_forward2.linear2.bias"], eng=mm)
    x = bf16(f32(x) + 0.5 * f32(h2))
    fails += chk("after_ffn2", x, R["after_ffn2"][0], tol)

    out = layernorm(x, Wb["norm_out.weight"], Wb["norm_out.bias"], eng=ln_e)
    fails += chk("block_out", out, R["block_out"][0], tol)

    d = np.abs(f32(out) - f32(R["block_out"][0]))
    print(f"\n[block] end-to-end vs ONNX: max|Δ|={d.max():.4e}  mean={d.mean():.4e}  "
          f"rel={d.max()/(np.abs(R['block_out'][0]).max()+1e-9):.2e}")
    print(f"[block] dwconv + matmuls ran on {where}; {'PASS' if fails == 0 else f'FAIL ({fails} stages off)'}")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
