#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Isolate the attention path at pos > 0 -- the half a pos-0 probe structurally cannot see.

At pos 0 attention is a NO-OP: softmax runs over one element, RoPE is the identity rotation, and the
KV cache holds exactly one row. Every per-op number measured there says nothing about `kv_off`
addressing, the GQA `Repeat` head mapping, the `sm_mask` width, or the V-transpose over the padded S.
This runs two dispatches and checks each of those against the device's OWN q/k/v.

The GQA mapping is checked structurally rather than numerically, because it is the one thing Gemma
could never have exercised: Gemma has n_kv_heads=1, so ANY head mapping is correct there, and Qwen3
(n_kv_heads=8, group 2) is the first model where a wrong mapping is observable.
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


def rel(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()[:a.size]
    r = np.linalg.norm(a)
    return float(np.linalg.norm(a - b) / r) if r else float(np.linalg.norm(a - b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="qwen3-0.6b")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--tokens", default="785,6722")
    ap.add_argument("--layer", type=int, default=0)
    a = ap.parse_args()

    sp, fused, weights, md = build_graph(a.spec, a.weights, a.layers)
    S, HD, Hq, Hkv = md["S"], sp.head_dim, sp.n_q_heads, sp.n_kv_heads
    grp = Hq // Hkv
    toks = [int(t) for t in a.tokens.split(",")]
    L = a.layer
    pf = f"L{L}_"

    c = fused.get_callable()
    params = c.params
    for n, arr in weights.items():
        np.copyto(c.get_buffer(n).data, np.asarray(arr, BF16).reshape(-1))

    def npy(n):
        return np.load(os.path.join(a.weights, f"{n}.npy")).astype(np.float32)

    embed = npy("model.embed_tokens.weight")
    half = HD // 2
    inv = 1.0 / (sp.rope_theta_global ** (np.arange(0, HD, 2, dtype=np.float64)[:half] / HD))

    def dev(n):
        return np.asarray(c.get_buffer(n).data, np.float32)

    kv_hist = []
    for pos, tok in enumerate(toks):
        np.copyto(c.get_buffer("x").data, np.asarray(embed[tok], BF16).reshape(-1))
        row = np.empty(HD, np.float32)
        row[0::2] = np.cos(pos * inv)
        row[1::2] = np.sin(pos * inv)
        np.copyto(c.get_buffer("rope_global").data, np.asarray(row, BF16))
        params.write("kv_off", int(pos * HD))
        params.write("sm_mask", int(pos + 1))
        params.sync()
        c()
        kv_hist.append((dev(pf + "k").copy(), dev(pf + "v").copy(), dev(pf + "q").copy()))

    P = len(toks) - 1                       # the last position dispatched
    kc = dev(pf + "kc").reshape(Hkv, S, HD)
    vc = dev(pf + "vc").reshape(Hkv, S, HD)
    kr = dev(pf + "kr").reshape(Hq, S, HD)
    vr = dev(pf + "vr").reshape(Hq, S, HD)
    sc = dev(pf + "sc").reshape(Hq, S)
    sw = dev(pf + "sw").reshape(Hq, S)
    vt = dev(pf + "vt").reshape(Hq, S, HD)
    cx = dev(pf + "cx").reshape(Hq, HD)
    q = kv_hist[P][2].reshape(Hq, HD)

    print(f"[attn] layer {L}, {len(toks)} positions, Hq={Hq} Hkv={Hkv} group={grp}, S={S}\n")

    print("1. KV-CACHE APPEND (kv_off addressing) -- each position's k/v must land at its own row")
    for p in range(len(toks)):
        kh = kv_hist[p][0].reshape(Hkv, HD)
        vh = kv_hist[p][1].reshape(Hkv, HD)
        print(f"   pos {p}: kc[:, {p}, :] {rel(kh, kc[:, p, :]):.4e}   "
              f"vc[:, {p}, :] {rel(vh, vc[:, p, :]):.4e}")
    stale = float(np.abs(kc[:, len(toks):len(toks) + 4, :]).max())
    print(f"   rows beyond the written ones are {'ZERO (clean)' if stale == 0 else f'NONZERO max={stale:.3e}'}")

    print("\n2. GQA REPEAT head mapping -- repeat_interleave means kr[h] == kc[h // group]")
    bad = []
    for h in range(Hq):
        e_int = rel(kc[h // grp, :P + 1, :], kr[h, :P + 1, :])
        e_mod = rel(kc[h % Hkv, :P + 1, :], kr[h, :P + 1, :])
        if e_int > 1e-6:
            bad.append((h, e_int, e_mod))
    if not bad:
        print(f"   all {Hq} heads match kc[h // {grp}] exactly -- interleaved mapping CONFIRMED")
    else:
        print(f"   {len(bad)}/{Hq} heads do NOT match kc[h // {grp}]:")
        for h, ei, em in bad[:6]:
            better = "h % Hkv fits better" if em < ei else ""
            print(f"     head {h:2}: vs kc[h//{grp}] {ei:.4e}   vs kc[h%{Hkv}] {em:.4e}  {better}")

    print("\n3. SCORES + SOFTMAX (sm_mask width) -- valid width is P+1 = %d" % (P + 1))
    scale = sp.attn_scale
    ref_sc = np.stack([kc[h // grp, :P + 1, :] @ q[h] * scale for h in range(Hq)])
    print(f"   scores[:, :{P+1}]      {rel(ref_sc, sc[:, :P + 1]):.4e}")
    e = np.exp(ref_sc - ref_sc.max(-1, keepdims=True))
    ref_sw = e / e.sum(-1, keepdims=True)
    print(f"   softmax[:, :{P+1}]     {rel(ref_sw, sw[:, :P + 1]):.4e}")
    tail = float(np.abs(sw[:, P + 1:]).max())
    tsum = float(sw[:, :P + 1].sum(-1).mean())
    print(f"   softmax rows sum to {tsum:.6f} over the valid width (want 1.0)")
    print(f"   softmax beyond the mask: max {tail:.3e} {'(masked)' if tail < 1e-6 else '<-- LEAKING'}")

    print("\n4. V-TRANSPOSE + CONTEXT")
    print(f"   vt[h][:HD, :P+1] vs vr[h][:P+1, :HD]  "
          f"{rel(np.stack([vr[h, :P + 1, :].T for h in range(Hq)]), np.stack([vt[h].reshape(HD, S)[:, :P + 1] for h in range(Hq)])):.4e}")
    ref_cx = np.stack([sw[h, :P + 1] @ vc[h // grp, :P + 1, :] for h in range(Hq)])
    print(f"   context                {rel(ref_cx, cx):.4e}")


if __name__ == "__main__":
    main()
