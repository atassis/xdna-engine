#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Emit the bf16 host oracle: the token sequence a FAITHFUL bf16 forward produces.

Greedy decode is chaotic wherever the top-2 logits are close, and Qwen3-0.6B has such a step on the
stock prompt: at step 5 the margin is 0.0203, and a faithful bf16 forward already disagrees with HF's
f32 `generate` there. Gating a bf16 device against an f32 reference therefore charges it for a tie it
cannot win, and caps a PERFECT implementation below 8/8.

So the gate is device-vs-CPU-oracle at MATCHING precision, which is what the doctrine asks for
(1:1 determinism against the host at temp 0, not an error metric). This writes that oracle, and
records the per-step margin so a future mismatch can be read as "knife-edge" or "real" instead of
being argued about.
"""
import argparse
import json
import os

import numpy as np
import ml_dtypes

BF16 = ml_dtypes.bfloat16


def BF(a):
    return np.asarray(a, BF16).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="artifacts/qwen3-0.6b/weights")
    ap.add_argument("--ref", default="tests/refs/qwen3-0.6b/greedy_ref.json")
    ap.add_argument("--out", default="tests/refs/qwen3-0.6b/bf16_oracle.json")
    ap.add_argument("--steps", type=int, default=8)
    a = ap.parse_args()

    NL, D, Hq, Hkv, HD, EPS, THETA = 28, 1024, 16, 8, 128, 1e-6, 1e6
    W = a.weights
    ref = json.load(open(a.ref))

    def npy(n):
        return np.load(os.path.join(W, f"{n}.npy")).astype(np.float32)

    def rms(x, w):
        return x / np.sqrt((x * x).mean(-1, keepdims=True) + EPS) * w

    def silu(x):
        return x / (1.0 + np.exp(-np.clip(x, -60, 60)))

    def rope(v, pos):
        h = HD // 2
        inv = 1.0 / (THETA ** (np.arange(0, HD, 2, dtype=np.float64)[:h] / HD))
        c, s = np.cos(pos * inv).astype(np.float32), np.sin(pos * inv).astype(np.float32)
        v = v.reshape(-1, HD)
        x1, x2 = v[:, :h], v[:, h:]
        return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], -1).reshape(-1)

    embed, n_final = npy("model.embed_tokens.weight"), npy("model.norm.weight")
    Wt = {l: {k: npy(f"model.layers.{l}." + v) for k, v in (
        ("n_in", "input_layernorm.weight"), ("n_pf", "post_attention_layernorm.weight"),
        ("n_qn", "self_attn.q_norm.weight"), ("n_kn", "self_attn.k_norm.weight"),
        ("Wq", "self_attn.q_proj.weight"), ("Wk", "self_attn.k_proj.weight"),
        ("Wv", "self_attn.v_proj.weight"), ("Wo", "self_attn.o_proj.weight"),
        ("Wg", "mlp.gate_proj.weight"), ("Wu", "mlp.up_proj.weight"),
        ("Wd", "mlp.down_proj.weight"))} for l in range(NL)}

    S = 64
    kc = [np.zeros((Hkv, S, HD), np.float32) for _ in range(NL)]
    vc = [np.zeros((Hkv, S, HD), np.float32) for _ in range(NL)]
    fed, produced, margins = list(ref["prompt_ids"]), [], []
    tok = fed[0]
    for pos in range(len(fed) + a.steps - 1):
        x = BF(embed[tok])
        for l in range(NL):
            w = Wt[l]
            h = BF(rms(x, w["n_in"]))
            q, k, v = BF(w["Wq"] @ h), BF(w["Wk"] @ h), BF(w["Wv"] @ h)
            q = BF(np.concatenate([rms(q.reshape(Hq, HD)[i], w["n_qn"]) for i in range(Hq)]))
            k = BF(np.concatenate([rms(k.reshape(Hkv, HD)[i], w["n_kn"]) for i in range(Hkv)]))
            q, k = BF(rope(q, pos)), BF(rope(k, pos))
            kc[l][:, pos, :] = k.reshape(Hkv, HD)
            vc[l][:, pos, :] = v.reshape(Hkv, HD)
            qh = q.reshape(Hq, HD)
            ctx = np.empty((Hq, HD), np.float32)
            for hh in range(Hq):
                kv = hh // (Hq // Hkv)
                sc = (kc[l][kv, :pos + 1] @ qh[hh]) * (HD ** -0.5)
                sc = np.exp(sc - sc.max())
                sc /= sc.sum()
                ctx[hh] = sc @ vc[l][kv, :pos + 1]
            x = BF(x + BF(w["Wo"] @ BF(ctx.reshape(-1))))
            hf = BF(rms(x, w["n_pf"]))
            x = BF(x + BF(w["Wd"] @ BF(BF(silu(BF(w["Wg"] @ hf))) * BF(w["Wu"] @ hf))))
        lg = embed @ BF(rms(x, n_final))
        top = np.argsort(lg)[::-1]
        if pos + 1 < len(fed):
            tok = fed[pos + 1]
        else:
            produced.append(int(top[0]))
            margins.append(float(lg[top[0]] - lg[top[1]]))
            tok = int(top[0])
        if len(produced) >= a.steps:
            break

    out = dict(model=ref["model"], prompt=ref["prompt"], prompt_ids=ref["prompt_ids"],
               gen_ids=produced, margins=margins, hf_f32_gen_ids=ref["gen_ids"][:len(produced)],
               note="Faithful bf16 host forward -- the oracle a bf16 DEVICE should match 1:1. "
                    "`margins` is top1-top2 per step: a mismatch at a margin near the device's own "
                    "logit error is a knife-edge, not a defect. hf_f32_gen_ids is kept for contrast; "
                    "it is NOT the gate.")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"bf16 oracle : {produced}")
    print(f"HF f32      : {ref['gen_ids'][:len(produced)]}")
    print(f"margins     : {['%.4f' % m for m in margins]}")
    agree = sum(1 for i, t in enumerate(produced) if t == ref["gen_ids"][i])
    print(f"\nbf16 oracle agrees with HF f32 on {agree}/{len(produced)} "
          f"-- the ceiling any bf16 device can score against the f32 reference")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
