#!/usr/bin/env python3
"""Stack Conformer blocks 0..N-1 using the verified recipe; check vs ONNX.

Two modes:
  fp32 teacher-forced  -- each block fed the ONNX output of the previous block;
                          proves the recipe + weight extraction generalize beyond
                          block 0 (the basis for scaling to all 16 blocks).
  bf16 free-running    -- chain our own bf16 outputs from the block-0 input;
                          measures how bf16 error accumulates across the stack.

Runs in .venv (numpy only here). Mirrors scripts/block0_numpy.py op semantics.
"""
import json
import numpy as np
from ml_dtypes import bfloat16

S = "artifacts/stack"
NH, HD, T, C = 16, 48, 400, 768
EPS = 1e-5
man = json.load(open(f"{S}/manifest.json"))
NB = man["nblocks"]
cos = np.load(f"{S}/refs/pos_cos.npy"); sin = np.load(f"{S}/refs/pos_sin.npy")


def loadW(blk):
    return {k: np.load(f"{S}/L{blk}/{k}.npy") for k in man["blocks"][str(blk)]}


def block_forward(x, W, q):
    f = lambda a: np.asarray(a, np.float32)
    def ln(z, w, b):
        z = f(z); mu = z.mean(-1, keepdims=True); var = z.var(-1, keepdims=True)
        return q((z - mu) / np.sqrt(var + EPS) * f(w) + f(b))
    def silu(z): z = f(z); return q(z / (1.0 + np.exp(-z)))
    def mm(z, w, b): return q(f(z) @ f(w) + f(b))

    # FFN1
    h = ln(x, W["norm_feed_forward1.weight"], W["norm_feed_forward1.bias"])
    h = silu(mm(h, W["feed_forward1.linear1.weight"], W["feed_forward1.linear1.bias"]))
    h = mm(h, W["feed_forward1.linear2.weight"], W["feed_forward1.linear2.bias"])
    x = q(f(x) + 0.5 * f(h))
    # MHSA
    h = ln(x, W["norm_self_att.weight"], W["norm_self_att.bias"])
    xr = f(h).reshape(T, 1, NH, HD); half = HD // 2
    rope = q(xr * cos + np.concatenate([-xr[..., half:], xr[..., :half]], -1) * sin).reshape(T, C)
    qq = mm(rope, W["self_attn.linear_q.weight"], W["self_attn.linear_q.bias"])
    kk = mm(rope, W["self_attn.linear_k.weight"], W["self_attn.linear_k.bias"])
    vv = mm(h, W["self_attn.linear_v.weight"], W["self_attn.linear_v.bias"])
    qh = f(qq).reshape(T, NH, HD).transpose(1, 0, 2)
    kh = f(kk).reshape(T, NH, HD).transpose(1, 0, 2)
    vh = f(vv).reshape(T, NH, HD).transpose(1, 0, 2)
    sc = (qh @ kh.transpose(0, 2, 1)) / np.sqrt(HD)
    p = np.exp(sc - sc.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
    ctx = (q(p).astype(np.float32) @ vh).transpose(1, 0, 2).reshape(T, C)
    ao = mm(ctx, W["self_attn.linear_out.weight"], W["self_attn.linear_out.bias"])
    x = q(f(x) + f(ao))
    # ConvModule
    h = ln(x, W["norm_conv.weight"], W["norm_conv.bias"])
    xct = f(h).T
    pw1 = (f(W["conv.pointwise_conv1.weight"][:, :, 0]) @ xct
           + f(W["conv.pointwise_conv1.bias"])[:, None])
    a_, g_ = pw1[:768], pw1[768:]
    glu = q(a_ / (1.0 + np.exp(-g_)))
    dww = f(W["conv.depthwise_conv.weight"][:, 0, :]); pad = np.pad(f(glu), ((0, 0), (2, 2)))
    dw = sum(dww[:, i:i+1] * pad[:, i:i+T] for i in range(5)) + f(W["conv.depthwise_conv.bias"])[:, None]
    dw = q(dw)
    bn = ln(f(dw).T, W["conv.batch_norm.weight"], W["conv.batch_norm.bias"])
    sw = silu(f(bn).T)
    pw2 = q(f(W["conv.pointwise_conv2.weight"][:, :, 0]) @ f(sw)
            + f(W["conv.pointwise_conv2.bias"])[:, None])
    x = q(f(x) + f(pw2).T)
    # FFN2
    h = ln(x, W["norm_feed_forward2.weight"], W["norm_feed_forward2.bias"])
    h = silu(mm(h, W["feed_forward2.linear1.weight"], W["feed_forward2.linear1.bias"]))
    h = mm(h, W["feed_forward2.linear2.weight"], W["feed_forward2.linear2.bias"])
    x = q(f(x) + 0.5 * f(h))
    return ln(x, W["norm_out.weight"], W["norm_out.bias"])


def rel(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return np.abs(a - b).max() / (np.abs(b).max() + 1e-9)


def main():
    Ws = [loadW(b) for b in range(NB)]
    refs = [np.load(f"{S}/refs/out_L{b}.npy")[0] for b in range(NB)]
    x0 = np.load(f"{S}/refs/block_in.npy")[0]
    idf = lambda z: np.asarray(z, np.float32)
    bf = lambda z: np.asarray(z, np.float32).astype(bfloat16)

    print("== fp32, teacher-forced (each block fed ONNX prev output) ==")
    for b in range(NB):
        xin = x0 if b == 0 else refs[b - 1]
        out = block_forward(idf(xin), Ws[b], idf)
        print(f"  block {b}: rel vs ONNX = {rel(out, refs[b]):.2e}")

    print("== bf16, free-running (chained own outputs from block_in) ==")
    x = bf(x0)
    for b in range(NB):
        x = block_forward(x, Ws[b], bf)
        print(f"  block {b}: rel vs ONNX = {rel(x, refs[b]):.2e}  (accumulated)")


if __name__ == "__main__":
    main()
