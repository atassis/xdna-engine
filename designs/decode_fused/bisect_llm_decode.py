#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Per-node rel-L2 bisect of a fused decode ELF against an f32 host reference.

One dispatch runs the whole stack and every intermediate is its OWN named buffer in the arena, so a
single device run yields every node. Compare each against the host value computed from the SAME
weights, walk forward, and report the first node that leaves the bf16 noise floor -- that names the
kernel to look at instead of the model.

Node order per layer mirrors the runlist in gen_llm_decode.py. In-place ops (q/k through qk-norm and
RoPE, sc through scale, g through the activation) can only be read at their FINAL value, so those
rows compare the end of the chain, not each step. Run at pos 0 so attention is a no-op rotation and
softmax is over one element: that isolates the projections, norms and FFN from the KV path.
"""
import argparse
import os
import sys

import numpy as np
import ml_dtypes

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import newstack_compat  # noqa: F401,E402
from gen_llm_decode import build_graph  # noqa: E402

BF16 = ml_dtypes.bfloat16


def rel_l2(ref, got):
    ref = np.asarray(ref, np.float64).ravel()
    got = np.asarray(got, np.float64).ravel()
    n = min(len(ref), len(got))
    d = np.linalg.norm(ref[:n] - got[:n])
    r = np.linalg.norm(ref[:n])
    return float(d / r) if r else float(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="qwen3-0.6b")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--token", type=int, default=785)
    ap.add_argument("--detail-layers", type=int, default=2)
    ap.add_argument("--detail-from", type=int, default=None,
                    help="also print node detail for layers >= this index")
    a = ap.parse_args()

    sp, fused, weights, md = build_graph(a.spec, a.weights, a.layers)
    NL, HD, D, FF = md["NL"], sp.head_dim, sp.d_model, sp.ffn
    Hq, Hkv, EPS = sp.n_q_heads, sp.n_kv_heads, sp.eps
    print(f"[bisect] {sp.name}: {NL} layers, token {a.token}, pos 0")

    c = fused.get_callable()
    params = c.params
    for n, arr in weights.items():
        np.copyto(c.get_buffer(n).data, np.asarray(arr, BF16).reshape(-1))

    def npy(n):
        return np.load(os.path.join(a.weights, f"{n}.npy")).astype(np.float32)

    embed = npy("model.embed_tokens.weight")
    x0 = embed[a.token].copy()
    np.copyto(c.get_buffer("x").data, np.asarray(x0, BF16).reshape(-1))
    half = HD // 2
    inv = 1.0 / (sp.rope_theta_global ** (np.arange(0, HD, 2, dtype=np.float64)[:half] / HD))
    row = np.empty(HD, np.float32)
    row[0::2] = np.cos(0 * inv)
    row[1::2] = np.sin(0 * inv)
    np.copyto(c.get_buffer("rope_global").data, np.asarray(row, BF16))
    params.write("kv_off", 0)
    params.write("sm_mask", 1)
    params.sync()
    c()
    print("[bisect] dispatch complete\n")

    def dev(name):
        return np.asarray(c.get_buffer(name).data, np.float32)

    def rms(x, w):
        x = np.asarray(x, np.float32)
        return x / np.sqrt((x * x).mean(-1, keepdims=True) + EPS) * w

    def silu(x):
        return x / (1.0 + np.exp(-np.clip(x, -60, 60)))

    print(f"{'node':14} {'rel-L2':>11}   {'|host|':>10} {'|dev|':>10}")
    print("-" * 52)
    x = x0
    first_bad = None
    prev_e = 0.0
    for l in range(NL):
        p = f"L{l}_"
        detail = l < a.detail_layers or (a.detail_from is not None and l >= a.detail_from)
        w = {k: npy(f"model.layers.{l}.{v}") for k, v in (
            ("n_in", "input_layernorm.weight"), ("n_pf", "post_attention_layernorm.weight"),
            ("n_qn", "self_attn.q_norm.weight"), ("n_kn", "self_attn.k_norm.weight"),
            ("Wq", "self_attn.q_proj.weight"), ("Wk", "self_attn.k_proj.weight"),
            ("Wv", "self_attn.v_proj.weight"), ("Wo", "self_attn.o_proj.weight"),
            ("Wg", "mlp.gate_proj.weight"), ("Wu", "mlp.up_proj.weight"),
            ("Wd", "mlp.down_proj.weight"))}
        rows = []
        hn = rms(x, w["n_in"]);                                  rows.append(("hn", hn))
        q = w["Wq"] @ hn; k = w["Wk"] @ hn; v = w["Wv"] @ hn
        q = np.concatenate([rms(q.reshape(Hq, HD)[i], w["n_qn"]) for i in range(Hq)])
        k = np.concatenate([rms(k.reshape(Hkv, HD)[i], w["n_kn"]) for i in range(Hkv)])
        rows += [("q", q), ("k", k), ("v", v)]
        ctx = np.stack([v.reshape(Hkv, HD)[h // (Hq // Hkv)] for h in range(Hq)]).reshape(-1)
        rows.append(("cx", ctx))
        av = w["Wo"] @ ctx;                                      rows.append(("a", av))
        x1 = x + av;                                             rows.append(("x1", x1))
        hf = rms(x1, w["n_pf"]);                                 rows.append(("hf", hf))
        g = silu(w["Wg"] @ hf); u = w["Wu"] @ hf
        rows += [("g", g), ("u", u)]
        gh = g * u;                                              rows.append(("gh", gh))
        d = w["Wd"] @ gh;                                        rows.append(("d", d))
        x = x1 + d
        for nm, ref in rows:
            if not detail:
                continue
            try:
                got = dev(p + nm)
            except Exception as e:
                print(f"{p+nm:14} unreadable: {type(e).__name__}")
                continue
            e = rel_l2(ref, got)
            gn = np.linalg.norm(np.asarray(got, np.float64).ravel()[:ref.size])
            print(f"{p+nm:14} {e:11.4e}   {np.linalg.norm(ref):10.3f} {gn:10.3f}")
            if first_bad is None and e > 0.05:
                first_bad = (p + nm, e)
        nxt = f"x{l+1}"
        try:
            gx = dev(nxt)
            e = rel_l2(x, gx)
            print(f"{'  -> ' + nxt:14} {e:11.4e}")
            # A jump in this curve has two very different causes and the per-quarter profile tells
            # them apart: a PARTIAL WRITE leaves one region at the previous value (or zero) while the
            # rest is exact, whereas a wrong computation is wrong roughly uniformly. Print the
            # profile only when the step is anomalous, so the normal curve stays readable.
            if e > 2.5 * prev_e and prev_e > 0:
                q = len(x) // 4
                parts = " ".join(f"q{i}={rel_l2(x[i*q:(i+1)*q], gx[i*q:(i+1)*q]):.3e}"
                                 for i in range(4))
                zeros = int((np.abs(np.asarray(gx, np.float64).ravel()[:len(x)]) == 0).sum())
                print(f"{'':14} JUMP {e/prev_e:.1f}x -> {parts}  exact-zeros={zeros}/{len(x)}")
            prev_e = e
            if first_bad is None and e > 0.05:
                first_bad = (nxt, e)
        except Exception:
            pass
        # No early exit: the whole curve is the evidence. A single node above a threshold says
        # little when the error accumulates -- the SHAPE of the growth is what distinguishes a
        # biased per-op error from faithful bf16 rounding.

    # final norm + tied lm-head
    xf = rms(x, npy("model.norm.weight"))
    try:
        print(f"{'  -> xf':14} {rel_l2(xf, dev('xf')):11.4e}")
    except Exception:
        pass
    host_logits = embed @ xf
    try:
        dl = dev("logits")[:len(host_logits)]
        print(f"{'  -> logits':14} {rel_l2(host_logits, dl):11.4e}   "
              f"host argmax {int(np.argmax(host_logits))}  dev argmax {int(np.argmax(dl))}")
    except Exception:
        pass
    print()
    print(f"*** first node above 0.05: {first_bad[0]} at {first_bad[1]:.4e} ***" if first_bad
          else "*** no node above 0.05 ***")


if __name__ == "__main__":
    main()
