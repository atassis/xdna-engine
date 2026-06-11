"""Fused-backend engines: ObjectFifo-fused sub-graphs behind the same Ops idea.

Where `ops.py` runs one xclbin per op (host-orchestrated), this runs **fused**
xclbins — whole-array (8-col) matmuls with the bias (+SiLU) epilogue applied
on-chip — so a sub-graph is a couple of dispatches instead of many.

PERF (task 15): `WAEpilogue` is **weight-bound** — the constant K-augmented weight
`B_aug` is built ONCE and its device buffer + the instruction buffer are allocated
and synced ONCE in __init__; the activation/output buffers are allocated once and
REUSED across calls. So steady-state `forward()` only writes the activation tile
and dispatches — no per-call weight rebuild, no buffer (re)allocation, no weight
re-sync. (Levers 1+2 of task 15.)

Correctness oracle stays block.py; verified vs ONNX (docs/10).
"""
import os
import time
import numpy as np

from . import config as C
from .dtypes import bf16, f32, bfloat16

WA = C.MM_WHOLE_DIR

# lightweight profiling: total seconds spent inside NPU dispatch .wait() calls
NPU_DISPATCH_S = [0.0]
NPU_DISPATCH_N = [0]


def reset_npu_prof():
    NPU_DISPATCH_S[0] = 0.0; NPU_DISPATCH_N[0] = 0


def fold_ln_into_mm1(g, beta, W1, b1):
    """Fold a LayerNorm affine (scale g, shift beta) into the following matmul, so
    the NPU LayerNorm can be normalize-only: (norm*g+beta)@W1 + b1 == norm@W1' + b1'."""
    W1p = f32(g)[:, None] * f32(W1)        # scale W1 rows by g
    b1p = f32(beta) @ f32(W1) + f32(b1)    # LN shift through W1 + linear1 bias
    return W1p, b1p


class WAEpilogue:
    """A WEIGHT-BOUND whole-array (8-col) fused matmul with on-chip bias(+SiLU).
    Bias rides a K-augmented extra k-block. Weight + instr buffers are built/synced
    once; activation/output buffers reused. forward(A_real) -> f32 [M',N]."""
    M = C.PAD_M
    TILE = 32

    def __init__(self, dev, mode, K, N, B_real, bias):
        assert mode in ("silu", "bias")
        self.dev, self.K, self.N = dev, K, N
        self.Kaug = K + self.TILE
        suffix = f"{self.M}x{self.Kaug}x{N}_{self.TILE}x{self.TILE}x{self.TILE}_8c_{mode}"
        self.k = dev.kernel(os.path.join(WA, f"final_{suffix}.xclbin"))
        instr = np.fromfile(os.path.join(WA, f"insts_{suffix}.txt"), np.uint32)
        px, d = dev.pyxrt, dev.d

        # constant K-augmented weight, built ONCE
        B_aug = np.zeros((self.Kaug, N), np.float32)
        B_aug[:K, :] = f32(B_real); B_aug[K, :] = f32(bias)
        Bb = np.ascontiguousarray(bf16(B_aug)).view(np.uint16)

        # buffers: instr + weight synced once; activation/output reused
        self.bo_i = px.bo(d, instr.nbytes, px.bo.cacheable, self.k.group_id(1))
        self.bo_i.write(instr.tobytes(), 0); self.bo_i.sync(dev.TO)
        self.bo_b = px.bo(d, Bb.nbytes, px.bo.host_only, self.k.group_id(4))
        self.bo_b.write(Bb.tobytes(), 0); self.bo_b.sync(dev.TO)
        self.bo_a = px.bo(d, self.M * self.Kaug * 2, px.bo.host_only, self.k.group_id(3))
        self.bo_c = px.bo(d, self.M * N * 2, px.bo.host_only, self.k.group_id(5))
        self.bo_t = px.bo(d, 1, px.bo.host_only, self.k.group_id(6))
        self.bo_tr = px.bo(d, 4, px.bo.host_only, self.k.group_id(7))
        self.n_instr = instr.size
        # reusable A_aug host buffer (ones column preset)
        self._A = np.zeros((self.M, self.Kaug), bfloat16); self._A[:, K] = bfloat16(1.0)

    def __call__(self, A_real):
        Mp = A_real.shape[0]
        self._A[:Mp, :self.K] = bf16(A_real)
        Ab = np.ascontiguousarray(self._A).view(np.uint16)
        self.bo_a.write(Ab.tobytes(), 0); self.bo_a.sync(self.dev.TO)
        _t = time.perf_counter()
        self.k(3, self.bo_i, self.n_instr, self.bo_a, self.bo_b, self.bo_c, self.bo_t, self.bo_tr).wait()
        NPU_DISPATCH_S[0] += time.perf_counter() - _t; NPU_DISPATCH_N[0] += 1
        self.bo_c.sync(self.dev.FROM)
        Cf = np.frombuffer(self.bo_c.read(self.M * self.N * 2, 0), np.uint16).view(bfloat16)
        return f32(Cf.reshape(self.M, self.N))[:Mp]


class FusedFFN:
    """Macaron-half FFN as 2 weight-bound fused dispatches (LN affine folded into
    mm1, biases K-augmented, SiLU on-chip). Verified vs ONNX (docs/10)."""
    def __init__(self, dev, w, pfx="feed_forward1", norm="norm_feed_forward1"):
        W1p, b1p = fold_ln_into_mm1(w[f"{norm}.weight"], w[f"{norm}.bias"],
                                    w[f"{pfx}.linear1.weight"], w[f"{pfx}.linear1.bias"])
        self.mm1 = WAEpilogue(dev, "silu", C.D_MODEL, C.D_FF, W1p, b1p)
        self.mm2 = WAEpilogue(dev, "bias", C.D_FF, C.D_MODEL,
                              w[f"{pfx}.linear2.weight"], w[f"{pfx}.linear2.bias"])
        self.eps = C.LN_EPS

    def forward(self, x):
        xf = f32(x)
        norm = (xf - xf.mean(-1, keepdims=True)) / np.sqrt(xf.var(-1, keepdims=True) + self.eps)
        return self.mm2(self.mm1(norm))


class FusedBlock:
    """One GigaAM-v3 Conformer block, matmul-heavy ops fused on the NPU (FFN×2,
    q/k/v/out, pointwise1/2 weight-bound; dwconv on NPU); LN/RoPE/GLU/softmax/
    residual on host. Verified vs ONNX out_L0 (docs/10)."""
    NH, HD, T, D = C.N_HEADS, C.HEAD_DIM, C.T_OUT, C.D_MODEL

    def __init__(self, dev, w, cos, sin):
        self.w, self.cos, self.sin, self.eps = w, cos, sin, C.LN_EPS
        self.ffn1 = FusedFFN(dev, w, "feed_forward1", "norm_feed_forward1")
        self.ffn2 = FusedFFN(dev, w, "feed_forward2", "norm_feed_forward2")
        P = lambda key: WAEpilogue(dev, "bias", 768, 768, w[f"self_attn.{key}.weight"], w[f"self_attn.{key}.bias"])
        self.q, self.k, self.v, self.o = P("linear_q"), P("linear_k"), P("linear_v"), P("linear_out")
        self.pw1 = WAEpilogue(dev, "bias", 768, 1536, w["conv.pointwise_conv1.weight"][:, :, 0].T,
                              w["conv.pointwise_conv1.bias"])
        self.pw2 = WAEpilogue(dev, "bias", 768, 768, w["conv.pointwise_conv2.weight"][:, :, 0].T,
                              w["conv.pointwise_conv2.bias"])
        from .ops import DwconvEngine
        self.dw = DwconvEngine(dev)

    def _ln(self, x, key):
        xf = f32(x)
        n = (xf - xf.mean(-1, keepdims=True)) / np.sqrt(xf.var(-1, keepdims=True) + self.eps)
        return n * f32(self.w[f"{key}.weight"]) + f32(self.w[f"{key}.bias"])

    def forward(self, x):
        w, T, D, NH, HD = self.w, self.T, self.D, self.NH, self.HD
        x = bf16(x)
        x = bf16(f32(x) + 0.5 * f32(self.ffn1.forward(x)))           # FFN1
        # MHSA
        ln = self._ln(x, "norm_self_att")
        xr = f32(ln).reshape(T, 1, NH, HD); half = HD // 2
        rope = (xr * self.cos + np.concatenate([-xr[..., half:], xr[..., :half]], -1) * self.sin).reshape(T, D)
        qh = self.q(rope).reshape(T, NH, HD).transpose(1, 0, 2)
        kh = self.k(rope).reshape(T, NH, HD).transpose(1, 0, 2)
        vh = self.v(ln).reshape(T, NH, HD).transpose(1, 0, 2)
        sc = (qh @ kh.transpose(0, 2, 1)) / np.sqrt(HD)
        p = np.exp(sc - sc.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
        ctx = (f32(bf16(p)) @ vh).transpose(1, 0, 2).reshape(T, D)
        x = bf16(f32(x) + f32(self.o(ctx)))
        # ConvModule
        ln = self._ln(x, "norm_conv")
        pw1 = self.pw1(ln).T                                          # [1536,400]
        a_, g_ = pw1[:D], pw1[D:]
        glu = a_ / (1.0 + np.exp(-g_))
        dwout = f32(self.dw.dwconv(glu, w["conv.depthwise_conv.weight"][:, 0, :])) \
            + f32(w["conv.depthwise_conv.bias"])[:, None]
        bn = self._ln(f32(dwout).T, "conv.batch_norm")
        sw = f32(bn).T; sw = sw / (1.0 + np.exp(-sw))
        x = bf16(f32(x) + f32(self.pw2(f32(sw).T)))
        x = bf16(f32(x) + 0.5 * f32(self.ffn2.forward(x)))           # FFN2
        return bf16(self._ln(x, "norm_out"))
