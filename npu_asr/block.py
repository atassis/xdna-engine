"""One GigaAM-v3 Conformer block (macaron), backend-agnostic via `ops`.

Recipe verified op-by-op vs ONNX (docs/08). Heavy ops (matmul/layernorm/silu/dwconv)
go through `ops` (NPU or host); the glue stays host: per-head RoPE applied to the LN
output BEFORE the q/k projection (v uses plain LN), scaled-dot-product attention with
softmax, GLU (split+sigmoid gate), residuals. "batch_norm" is a LayerNorm over channels.
"""
import numpy as np

from . import config as C
from .dtypes import bf16, f32

NH, HD, T, D = C.N_HEADS, C.HEAD_DIM, C.T_OUT, C.D_MODEL


def _ffn(x, w, ops, pfx):
    h = ops.matmul(x, w[f"{pfx}.linear1.weight"], w[f"{pfx}.linear1.bias"])
    h = ops.silu(h)
    return ops.matmul(h, w[f"{pfx}.linear2.weight"], w[f"{pfx}.linear2.bias"])


def conformer_block(x, w, ops, cos, sin):
    """x: bf16 [T, D]; w: this block's weight dict; -> bf16 [T, D]."""
    # ---- FFN1 (macaron ½) ----
    h = ops.layernorm(x, w["norm_feed_forward1.weight"], w["norm_feed_forward1.bias"])
    h = _ffn(h, w, ops, "feed_forward1")
    x = bf16(f32(x) + 0.5 * f32(h))

    # ---- MHSA: RoPE on LN output before q/k projection; v from plain LN ----
    ln = ops.layernorm(x, w["norm_self_att.weight"], w["norm_self_att.bias"])
    xr = f32(ln).reshape(T, 1, NH, HD)
    half = HD // 2
    rot = np.concatenate([-xr[..., half:], xr[..., :half]], axis=-1)
    rope = bf16((xr * cos + rot * sin).reshape(T, D))
    q = ops.matmul(rope, w["self_attn.linear_q.weight"], w["self_attn.linear_q.bias"])
    k = ops.matmul(rope, w["self_attn.linear_k.weight"], w["self_attn.linear_k.bias"])
    v = ops.matmul(ln, w["self_attn.linear_v.weight"], w["self_attn.linear_v.bias"])
    qh = f32(q).reshape(T, NH, HD).transpose(1, 0, 2)
    kh = f32(k).reshape(T, NH, HD).transpose(1, 0, 2)
    vh = f32(v).reshape(T, NH, HD).transpose(1, 0, 2)
    scores = (qh @ kh.transpose(0, 2, 1)) / np.sqrt(HD)
    probs = f32(ops.softmax(scores, axis=-1))
    ctx = (probs @ vh).transpose(1, 0, 2).reshape(T, D)
    ao = ops.matmul(bf16(ctx), w["self_attn.linear_out.weight"], w["self_attn.linear_out.bias"])
    x = bf16(f32(x) + f32(ao))

    # ---- ConvModule ----
    ln = ops.layernorm(x, w["norm_conv.weight"], w["norm_conv.bias"])
    pw1 = ops.matmul(ln, bf16(w["conv.pointwise_conv1.weight"][:, :, 0].T),
                     w["conv.pointwise_conv1.bias"])              # [T, 1536]
    pw1 = f32(pw1).T                                              # [1536, T]
    a_, g_ = pw1[:D], pw1[D:]
    glu = bf16(a_ / (1.0 + np.exp(-g_)))                          # GLU gate, [D, T]
    dwout = ops.dwconv(glu, w["conv.depthwise_conv.weight"][:, 0, :],
                       bias=w["conv.depthwise_conv.bias"])        # [D, T]
    bn = ops.layernorm(f32(dwout).T, w["conv.batch_norm.weight"], w["conv.batch_norm.bias"])  # [T,D]
    sw = ops.silu(f32(bn).T)                                      # [D, T]
    pw2 = ops.matmul(f32(sw).T, bf16(w["conv.pointwise_conv2.weight"][:, :, 0].T),
                     w["conv.pointwise_conv2.bias"])              # [T, D]
    x = bf16(f32(x) + f32(pw2))

    # ---- FFN2 (macaron ½) + final LN ----
    h = ops.layernorm(x, w["norm_feed_forward2.weight"], w["norm_feed_forward2.bias"])
    h = _ffn(h, w, ops, "feed_forward2")
    x = bf16(f32(x) + 0.5 * f32(h))
    return ops.layernorm(x, w["norm_out.weight"], w["norm_out.bias"])
