#!/usr/bin/env python3
"""Phase 2 — NumPy reference of the Parakeet FastConformer encoder, verified
block-by-block against the ONNX reference activations from Phase 1
(artifacts/parakeet/encoder/refs/). This nails the rel-pos (Transformer-XL)
attention + conv2D ÷8 subsample + dwconv-k9 math against the ONNX oracle BEFORE
the Rust port (per internal notes).

f32 throughout (the reference is the correctness oracle; bf16 is a later Rust/NPU
concern). Single full-length sequence (valid_len = T), so attention/conv masking
is a no-op and omitted.

Gates (rel = ||a-b|| / ||b||):
  1. pos_enc regen vs ref          (<= 1e-4)
  2. pre_encode subsample vs block_in (<= 0.08)
  3. block-0 vs out_L0             (<= 0.08)   <- the rel-pos gate
  4. full 24-block vs out_L{b}, encoded (<= 0.08)

Usage: ~/npuvox-asr-bench/.venv/bin/python scripts/parakeet_ref_encoder.py
"""
import os, sys, json
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENC = os.path.join(REPO, "artifacts", "parakeet", "encoder")
TOL = 0.08
D, DFF, H, DK, NB = 1024, 4096, 8, 128, 24

def W(blk, name):  # block weight
    return np.load(f"{ENC}/L{blk}/{name}.npy")
def PE(name):      # pre_encode weight
    return np.load(f"{ENC}/pre_encode/{name}.npy")
def REF(name):
    return np.load(f"{ENC}/refs/{name}.npy")

def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a.ravel() - b.ravel()) / (np.linalg.norm(b.ravel()) + 1e-12))

def layernorm(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True); var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * g + b

def silu(x):
    return x / (1.0 + np.exp(-x))

# ---------------- rel-pos positional encoding (NeMo RelPositionalEncoding) ----------------
def rel_pos_encoding(T, d):
    """Length 2T-1, positions [T-1, T-2, ..., 0, ..., -(T-1)] (positive first)."""
    pe_pos = np.zeros((T, d), np.float64)
    pe_neg = np.zeros((T, d), np.float64)
    pos = np.arange(0, T, dtype=np.float64)[:, None]
    div = np.exp(np.arange(0, d, 2, dtype=np.float64) * (-np.log(10000.0) / d))
    pe_pos[:, 0::2] = np.sin(pos * div);  pe_pos[:, 1::2] = np.cos(pos * div)
    pe_neg[:, 0::2] = np.sin(-pos * div); pe_neg[:, 1::2] = np.cos(-pos * div)
    # positive reversed (T-1..0) then negative (1..T-1)  -> center index T-1 is pos 0
    pe = np.concatenate([pe_pos[::-1], pe_neg[1:]], axis=0)   # [2T-1, d]
    return pe

# ---------------- conv2D ÷8 subsample (dw_striding) ----------------
def conv2d(x, w, b, stride, pad, groups):
    # x [C_in, Hin, Win], w [C_out, C_in/groups, kh, kw] -> [C_out, Hout, Wout]
    Ci, Hin, Win = x.shape
    Co, Cig, kh, kw = w.shape
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    Hout = (Hin + 2 * pad - kh) // stride + 1
    Wout = (Win + 2 * pad - kw) // stride + 1
    out = np.zeros((Co, Hout, Wout), np.float64)
    gci = Ci // groups; gco = Co // groups
    for g in range(groups):
        for oc in range(gco):
            co = g * gco + oc
            acc = np.full((Hout, Wout), b[co], np.float64)
            for ic in range(Cig):
                ci = g * gci + ic
                ker = w[co, ic]
                for i in range(kh):
                    for j in range(kw):
                        sub = xp[ci, i:i + stride * Hout:stride, j:j + stride * Wout:stride]
                        acc += ker[i, j] * sub
            out[co] = acc
    return out

def subsample(audio):  # audio [1,128,T] -> [T/8, 1024]
    # ONNX feeds conv as [b,1,time,freq] (Transpose perm [0,2,1,3] before out/MatMul
    # confirms H=time, W=freq), so transpose mel [freq=128, T] -> [time=T, freq=128]
    x = audio[0].T[None]                    # [1, T(time), 128(freq)]
    x = conv2d(x, PE("conv.0.weight"), PE("conv.0.bias"), 2, 1, 1); x = np.maximum(x, 0)
    x = conv2d(x, PE("conv.2.weight"), PE("conv.2.bias"), 2, 1, 256)   # depthwise
    x = conv2d(x, PE("conv.3.weight"), PE("conv.3.bias"), 1, 0, 1); x = np.maximum(x, 0)  # pointwise
    x = conv2d(x, PE("conv.5.weight"), PE("conv.5.bias"), 2, 1, 256)   # depthwise
    x = conv2d(x, PE("conv.6.weight"), PE("conv.6.bias"), 1, 0, 1); x = np.maximum(x, 0)  # pointwise
    return x  # [256, 16, T/8]

def subsample_flatten(x, target):
    # x [C=256, F=16, Tp]; find the flatten/transpose that matches block_in [Tp, 4096]
    # x is now [C=256, H=time, W=freq=16]; ONNX Transpose [0,2,1,3] -> [H, C, W] then reshape
    C, Hh, Wf = x.shape
    Wout = PE("out.weight"); Bout = PE("out.bias")  # [4096,1024]
    cands = {
        "h,c,w": np.transpose(x, (1, 0, 2)).reshape(Hh, C * Wf),   # ONNX order (time, C*freq)
        "h,w,c": np.transpose(x, (1, 2, 0)).reshape(Hh, Wf * C),
    }
    best = None
    for name, flat in cands.items():
        if flat.shape[1] != Wout.shape[0]:
            continue
        out = flat @ Wout + Bout
        r = rel(out, target)
        if best is None or r < best[1]:
            best = (name, r, out)
    return best  # (order_name, rel, out[Tp,1024])

# ---------------- rel-pos multi-head attention ----------------
def rel_shift(bd):  # bd [H, T, 2T-1] -> [H, T, T]
    Hh, T, P = bd.shape
    x = np.pad(bd, ((0, 0), (0, 0), (1, 0)))      # [H, T, P+1]
    x = x.reshape(Hh, P + 1, T)                   # [H, P+1, T]
    x = x[:, 1:].reshape(Hh, T, P)                # drop first row -> [H, T, P]
    return x[:, :, :T]                            # keep first T cols

def mhsa(x, blk, pos_enc):
    T = x.shape[0]
    q = (x @ W(blk, "self_attn.linear_q.weight")).reshape(T, H, DK)
    k = (x @ W(blk, "self_attn.linear_k.weight")).reshape(T, H, DK)
    v = (x @ W(blk, "self_attn.linear_v.weight")).reshape(T, H, DK)
    p = (pos_enc @ W(blk, "self_attn.linear_pos.weight")).reshape(-1, H, DK)  # [2T-1,H,DK]
    u = W(blk, "self_attn.pos_bias_u"); vv = W(blk, "self_attn.pos_bias_v")   # [H,DK]
    qu = (q + u).transpose(1, 0, 2)   # [H,T,DK]
    qv = (q + vv).transpose(1, 0, 2)  # [H,T,DK]
    kt = k.transpose(1, 2, 0)         # [H,DK,T]
    pt = p.transpose(1, 2, 0)         # [H,DK,2T-1]
    ac = qu @ kt                      # [H,T,T]
    bd = rel_shift(qv @ pt)           # [H,T,T]
    scores = (ac + bd) / np.sqrt(DK)
    scores = scores - scores.max(-1, keepdims=True)
    a = np.exp(scores); a /= a.sum(-1, keepdims=True)
    ctx = a @ v.transpose(1, 0, 2)    # [H,T,DK]
    ctx = ctx.transpose(1, 0, 2).reshape(T, H * DK)
    return ctx @ W(blk, "self_attn.linear_out.weight")

def conv_module(x, blk):  # x [T, D]
    pw1 = W(blk, "conv.pointwise_conv1.weight")[:, :, 0]  # [2D, D] (squeeze k=1)
    pw2 = W(blk, "conv.pointwise_conv2.weight")[:, :, 0]  # [D, D]
    dw = W(blk, "conv.depthwise_conv.weight")[:, 0, :]    # [D, 9]
    dwb = W(blk, "conv.depthwise_conv.bias")              # [D]
    h = x @ pw1.T                                         # [T, 2D]
    a, g = h[:, :D], h[:, D:]
    h = a * (1.0 / (1.0 + np.exp(-g)))                    # GLU [T, D]
    # depthwise k=9 pad=4 along time, BN folded into dw bias
    ht = h.T                                              # [D, T]
    hp = np.pad(ht, ((0, 0), (4, 4)))
    out = np.zeros_like(ht)
    for j in range(9):
        out += dw[:, j:j + 1] * hp[:, j:j + ht.shape[1]]
    out += dwb[:, None]
    h = silu(out.T)                                       # [T, D]
    return h @ pw2.T

def block(x, blk, pos_enc):
    x = x + 0.5 * (silu(layernorm(x, W(blk, "norm_feed_forward1.weight"), W(blk, "norm_feed_forward1.bias")) @
                        W(blk, "feed_forward1.linear1.weight")) @ W(blk, "feed_forward1.linear2.weight"))
    x = x + mhsa(layernorm(x, W(blk, "norm_self_att.weight"), W(blk, "norm_self_att.bias")), blk, pos_enc)
    x = x + conv_module(layernorm(x, W(blk, "norm_conv.weight"), W(blk, "norm_conv.bias")), blk)
    x = x + 0.5 * (silu(layernorm(x, W(blk, "norm_feed_forward2.weight"), W(blk, "norm_feed_forward2.bias")) @
                        W(blk, "feed_forward2.linear1.weight")) @ W(blk, "feed_forward2.linear2.weight"))
    return layernorm(x, W(blk, "norm_out.weight"), W(blk, "norm_out.bias"))

def main():
    fails = []
    # ---- gate 1: pos_enc ----
    pe_ref = REF("pos_enc")[0]            # [2T-1, D]
    T = (pe_ref.shape[0] + 1) // 2
    pe = rel_pos_encoding(T, D)
    r1 = rel(pe, pe_ref)
    print(f"[gate1] pos_enc regen vs ref: rel={r1:.2e}  (T={T})  {'OK' if r1 <= 1e-4 else 'FAIL'}")
    if r1 > 1e-4:
        fails.append("pos_enc"); pe = pe_ref  # fall back to ref so later gates still run

    # ---- gate 2: subsample ----
    block_in = REF("block_in")[0]         # [T, D]
    sub = subsample(REF("audio_signal"))
    order, r2, x0 = subsample_flatten(sub, block_in)
    print(f"[gate2] subsample vs block_in: rel={r2:.2e}  flatten={order}  {'OK' if r2 <= TOL else 'FAIL'}")
    if r2 > TOL:
        fails.append("subsample"); x0 = block_in  # fall back

    # ---- gate 3: block 0 ----
    o0 = block(block_in, 0, pe)
    r3 = rel(o0, REF("out_L0")[0])
    print(f"[gate3] block-0 vs out_L0: rel={r3:.2e}  {'OK' if r3 <= TOL else 'FAIL'}  <- rel-pos gate")
    if r3 > TOL:
        fails.append("block0")

    # ---- gate 4: full stack ----
    x = block_in.copy()
    worst = 0.0
    for b in range(NB):
        x = block(x, b, pe)
        rb = rel(x, REF(f"out_L{b}")[0])
        worst = max(worst, rb)
        if rb > TOL:
            print(f"  [block {b}] rel={rb:.2e}  FAIL"); fails.append(f"block{b}")
    enc_ref = REF("encoded")[0].T          # [T, D] (ref stored [D, T])
    r_enc = rel(x, enc_ref)
    print(f"[gate4] full 24-block: worst per-block rel={worst:.2e}; final vs encoded rel={r_enc:.2e}  "
          f"{'OK' if (worst <= TOL and r_enc <= TOL) else 'FAIL'}")
    if r_enc > TOL: fails.append("encoded")

    print("\n" + ("ALL GATES PASS" if not fails else f"FAILED: {sorted(set(fails))}"))
    return 0 if not fails else 1

if __name__ == "__main__":
    sys.exit(main())
