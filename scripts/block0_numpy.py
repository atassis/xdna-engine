#!/usr/bin/env python3
"""Pure-numpy reconstruction of GigaAM-v3 Conformer block 0, verified op-by-op
against the captured ONNX intermediates (artifacts/refs/*.npy).

This proves the block *recipe* (layouts, RoPE, scaling, GLU, the LN-as-batchnorm,
residual weights) before any op is moved onto the NPU. fp32 throughout here; the
bf16/NPU version (host-orchestrated) builds on this once it matches.
"""
import json, sys
import numpy as np

A = "artifacts"
man = json.load(open(f"{A}/manifest.json"))
W = {k: np.load(f"{A}/weights/{k}.npy") for k in man["weights"]}
R = {k: np.load(f"{A}/refs/{k}.npy") for k in man["refs"]}

NH, HD, T, C = 16, 48, 400, 768
EPS = 1e-5


def chk(name, got, ref):
    got = np.asarray(got, np.float32); ref = np.asarray(ref, np.float32)
    if got.shape != ref.shape:
        print(f"  {name:14s} SHAPE {got.shape} vs ref {ref.shape}  *** MISMATCH ***")
        return
    d = np.abs(got - ref)
    rel = d.max() / (np.abs(ref).max() + 1e-9)
    flag = "ok" if rel < 1e-3 else "**OFF**"
    print(f"  {name:14s} max|Δ|={d.max():.3e}  mean={d.mean():.3e}  rel={rel:.2e}  {flag}")


def layernorm(x, w, b, eps=EPS):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def silu(x):
    return x * (1.0 / (1.0 + np.exp(-x)))


def ffn(x, pfx):
    h = x @ W[f"{pfx}.linear1.weight"] + W[f"{pfx}.linear1.bias"]
    h = silu(h)
    h = h @ W[f"{pfx}.linear2.weight"] + W[f"{pfx}.linear2.bias"]
    return h


def main():
    x = R["block_in"][0]  # [400,768]
    print("== FFN1 ==")
    ln = layernorm(x, W["norm_feed_forward1.weight"], W["norm_feed_forward1.bias"])
    chk("ffn1_ln", ln, R["ffn1_ln"][0])
    h1 = ln @ W["feed_forward1.linear1.weight"] + W["feed_forward1.linear1.bias"]
    chk("ffn1_l1", h1, R["ffn1_l1"][0])
    sw = silu(h1)
    chk("ffn1_swish", sw, R["ffn1_swish"][0])
    h2 = sw @ W["feed_forward1.linear2.weight"] + W["feed_forward1.linear2.bias"]
    chk("ffn1_l2", h2, R["ffn1_l2"][0])
    x = x + 0.5 * h2
    chk("after_ffn1", x, R["after_ffn1"][0])

    print("== MHSA ==")
    ln = layernorm(x, W["norm_self_att.weight"], W["norm_self_att.bias"])
    chk("att_ln", ln, R["att_ln"][0])

    # RoPE on the LN output, per head (head_dim=48), BEFORE q/k projection.
    xr = ln.reshape(T, 1, NH, HD)            # [T,1,NH,HD] == att_reshape layout
    chk("att_reshape", xr, R["att_reshape"])
    cos, sin = R["pos_cos"], R["pos_sin"]    # [T,1,1,HD]
    half = HD // 2
    rot = np.concatenate([-xr[..., half:], xr[..., :half]], axis=-1)
    rope = xr * cos + rot * sin
    chk("att_rope", rope, R["att_rope"])
    qk_in = rope.reshape(T, NH * HD)[None]   # [1,T,768]
    chk("qk_in", qk_in, R["qk_in"])
    v_in = R["v_in"]                          # plain LN, rearranged (== ln)

    q = qk_in[0] @ W["self_attn.linear_q.weight"] + W["self_attn.linear_q.bias"]
    k = qk_in[0] @ W["self_attn.linear_k.weight"] + W["self_attn.linear_k.bias"]
    v = v_in[0] @ W["self_attn.linear_v.weight"] + W["self_attn.linear_v.bias"]
    chk("q", q, R["q"][0]); chk("k", k, R["k"][0]); chk("v", v, R["v"][0])

    # to heads [NH,T,HD], scaled-dot-product attention
    qh = q.reshape(T, NH, HD).transpose(1, 0, 2)
    kh = k.reshape(T, NH, HD).transpose(1, 0, 2)
    vh = v.reshape(T, NH, HD).transpose(1, 0, 2)
    scale = (1.0 / np.sqrt(HD))
    scores = (qh @ kh.transpose(0, 2, 1)) * scale      # [NH,T,T]
    chk("scores", scores[None], R["scores"])
    probs = scores - scores.max(-1, keepdims=True)
    probs = np.exp(probs); probs /= probs.sum(-1, keepdims=True)
    chk("attn_probs", probs[None], R["attn_probs"])
    ctx = probs @ vh                                    # [NH,T,HD]
    chk("attn_ctx", ctx[None], R["attn_ctx"])
    ctx = ctx.transpose(1, 0, 2).reshape(T, NH * HD)    # [T,768]
    attn_out = ctx @ W["self_attn.linear_out.weight"] + W["self_attn.linear_out.bias"]
    chk("attn_out", attn_out, R["attn_out"][0])
    x = x + attn_out
    chk("after_mhsa", x, R["after_mhsa"][0])

    print("== ConvModule ==")
    ln = layernorm(x, W["norm_conv.weight"], W["norm_conv.bias"])
    chk("conv_ln", ln, R["conv_ln"][0])
    xct = ln.T                                          # [C,T]=[768,400]
    pw1w = W["conv.pointwise_conv1.weight"][:, :, 0]    # [1536,768]
    pw1 = pw1w @ xct + W["conv.pointwise_conv1.bias"][:, None]   # [1536,400]
    chk("conv_pw1", pw1[None], R["conv_pw1"])
    a, g = pw1[:768], pw1[768:]                         # GLU split on channels
    glu = a * (1.0 / (1.0 + np.exp(-g)))                # [768,400]
    chk("conv_glu", glu[None], R["conv_glu"])
    # depthwise conv1d k=5 same pad
    dww = W["conv.depthwise_conv.weight"][:, 0, :]      # [768,5]
    dwb = W["conv.depthwise_conv.bias"]
    pad = np.pad(glu, ((0, 0), (2, 2)))
    dw = np.zeros_like(glu)
    for i in range(5):
        dw += dww[:, i:i+1] * pad[:, i:i+T]
    dw += dwb[:, None]
    chk("conv_dw", dw[None], R["conv_dw"])
    bn = layernorm(dw.T, W["conv.batch_norm.weight"], W["conv.batch_norm.bias"])  # [400,768]
    chk("conv_bn", bn[None], R["conv_bn"])
    sw = (bn.T) * (1.0 / (1.0 + np.exp(-bn.T)))         # swish on [768,400]
    chk("conv_swish", sw[None], R["conv_swish"])
    pw2w = W["conv.pointwise_conv2.weight"][:, :, 0]    # [768,768]
    pw2 = pw2w @ sw + W["conv.pointwise_conv2.bias"][:, None]
    chk("conv_pw2", pw2[None], R["conv_pw2"])
    x = x + pw2.T
    chk("after_conv", x, R["after_conv"][0])

    print("== FFN2 ==")
    ln = layernorm(x, W["norm_feed_forward2.weight"], W["norm_feed_forward2.bias"])
    h2 = ffn(ln, "feed_forward2")
    chk("ffn2_l2", h2, R["ffn2_l2"][0])
    x = x + 0.5 * h2
    chk("after_ffn2", x, R["after_ffn2"][0])

    print("== norm_out ==")
    out = layernorm(x, W["norm_out.weight"], W["norm_out.bias"])
    chk("block_out", out, R["block_out"][0])


if __name__ == "__main__":
    main()
