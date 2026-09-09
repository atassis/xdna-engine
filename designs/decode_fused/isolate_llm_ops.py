#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Per-op ISOLATED error: apply the host op to the DEVICE's own inputs.

The cumulative bisect tells you where error has grown to; it cannot tell you which op MAKES it,
because every node inherits its predecessor's error. This feeds each host op the device's own input
buffer, so the number reported is that op's contribution alone.

The load-bearing pair is `u` and `g`: both are a GEMV over the same `hf` with the same kernel, and
`g` additionally passes through the gated activation. If g's isolated error is far above u's, the
activation is the source and the GEMV is exonerated -- which no cumulative measurement can establish.

`a` (o_proj) and `d` (the FFN down projection) reproduce the SAME split-and-fold arithmetic the
device runs when a GEMV's K does not fit L1 (llm_decode_spec.k_chunks_for picks the chunk count --
Gemma-4-12B splits o_proj 2-way on its global layers only and down 4-way on every layer): the split
is itself a suspect, and mv.cc rounds its f32 accumulator to bf16 once PER PARTIAL, so an unsplit
host GEMV is not the same function as what ran. Where `sandwich_norms` applies, the device's `a`/`d`
buffer additionally carries an RMSNorm run IN PLACE on top (same caveat as q/k below), so the
isolated reference applies that too -- otherwise it compares two different functions.

Some per-node buffers exist only under certain dataflow-fusion flags (FUSE_QKV_GEMV folds q/k/v
into one `qkv` buffer with no standalone name; FUSE_QKV_DP additionally folds the norm and
projection into one design with no `hn`). A row whose buffer is unavailable prints "unreadable" and
the probe continues -- that is a property of the CURRENT build's flags, not a model-port defect.
"""
import argparse
import os
import sys

import numpy as np
import ml_dtypes

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import newstack_compat  # noqa: F401,E402
from gen_llm_decode import build_graph, COLS  # noqa: E402
from llm_decode_spec import k_chunks_for  # noqa: E402

BF16 = ml_dtypes.bfloat16


def rel(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()[:a.size]
    r = np.linalg.norm(a)
    return float(np.linalg.norm(a - b) / r) if r else 0.0


def identity_rope_row(width):
    """RoPE row at pos 0: ang = pos*inv is zero for every frequency regardless of theta or
    partial-rotary width, so cos=1/sin=0 everywhere -- only the buffer's own WIDTH is
    table-specific (Gemma-4's rope_global is 512 wide, rope_local 256).
    """
    row = np.zeros(width, np.float32)
    row[0::2] = 1.0
    return np.asarray(row, BF16)


def split_matvec(W, x, n):
    """Host mirror of gen_llm_decode.py's split_over_k -- see bisect_llm_decode.py's copy for the
    full derivation. K cut into `n` chunks, each partial rounded to bf16 and folded PAIRWISE in
    the same tree order the runlist builds. `n == 1` is the ordinary unsplit GEMV.
    """
    if n == 1:
        return W @ x
    rnd = lambda v: np.asarray(v, BF16).astype(np.float32)  # noqa: E731
    K = x.shape[-1]
    cw = K // n
    level = [rnd(W[:, i * cw:(i + 1) * cw] @ x[i * cw:(i + 1) * cw]) for i in range(n)]
    while len(level) > 1:
        nxt = [rnd(level[i] + level[i + 1]) for i in range(0, len(level) - 1, 2)]
        if len(level) % 2:
            nxt.append(level[-1])
        level = nxt
    return level[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="qwen3-0.6b")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--token", type=int, default=785)
    ap.add_argument("--probe-layers", type=int, default=3)
    ap.add_argument("--probe-from", type=int, default=None,
                    help="probe layers [probe-from, NL) instead of the first few")
    a = ap.parse_args()

    sp, fused, weights, md = build_graph(a.spec, a.weights, a.layers)
    NL, D, FF, Hq, EPS = md["NL"], sp.d_model, sp.ffn, sp.n_q_heads, sp.eps
    down_chunks = k_chunks_for(D, FF, COLS)
    c = fused.get_callable()
    params = c.params
    for n, arr in weights.items():
        with c.get_buffer(n).overwrite() as _buf:
            _buf[:] = np.asarray(arr, BF16).reshape(-1)

    def npy(n):
        return np.load(os.path.join(a.weights, f"{n}.npy")).astype(np.float32)

    def load_norm(name):
        w = npy(name)
        return 1.0 + w if sp.norm_gain == "one_plus_w" else w

    embed = npy(f"{sp.weight_prefix}embed_tokens.weight")
    scale = np.sqrt(D) if sp.embed_scale == "sqrt_d_model" else 1.0
    with c.get_buffer("x").overwrite() as _buf:
        _buf[:] = np.asarray(embed[a.token].astype(np.float32) * scale, BF16).reshape(-1)
    for slot_name, _ in md["kv_slots"]:
        params.write(slot_name, 0)
    for ang_name in ("rope_global", "rope_local"):
        if ang_name in md["inputs"]:
            with c.get_buffer(ang_name).overwrite() as _buf:
                _buf[:] = identity_rope_row(_buf.size)
    params.write("sm_mask", 1)
    params.sync()
    c()

    def dev(n):
        return np.asarray(c.get_buffer(n).data, np.float32)

    def rms(x, w=None):
        x = np.asarray(x, np.float32)
        xn = x / np.sqrt((x * x).mean(-1, keepdims=True) + EPS)
        return xn if w is None else xn * w

    def silu(x):
        return x / (1.0 + np.exp(-np.clip(x, -60, 60)))

    def gelu_tanh(x):
        x = np.asarray(x, np.float64)
        return (0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))).astype(np.float32)

    def act(x):
        return silu(x) if sp.act == "silu" else gelu_tanh(x)

    def bf(a_):
        return np.asarray(a_, BF16).astype(np.float32)

    def show(label, compute, tag):
        """One row: run `compute` -> (host_ref, device_val) and print, or note the device buffer
        unreadable and move on -- which per-node buffers exist depends on the build's fusion
        flags (see module docstring), and one missing buffer must not abort the whole probe.
        """
        try:
            h, d = compute()
        except Exception as e:
            print(f"{label:22} unreadable: {type(e).__name__}")
            return
        print(f"{label:22} {rel(h, d):11.4e}   {tag}")

    print(f"{'op (isolated)':22} {'rel-L2':>11}   what it isolates")
    print("-" * 74)
    probe = range(a.probe_from, NL) if a.probe_from is not None else range(min(a.probe_layers, NL))
    for l in probe:
        wp = f"{sp.weight_prefix}layers.{l}."
        pf = f"L{l}_"
        hd, hkv = sp.head_dim_for(l), sp.n_kv_heads_for(l)
        has_v = sp.has_v_proj(l)
        qd = Hq * hd
        o_chunks = k_chunks_for(D, qd, COLS)
        nm = sp.norm_weight_names(l)

        def w(name):
            return npy(wp + name)

        xin = dev("x") if l == 0 else dev(f"x{l}")
        hn_h = bf(rms(xin, load_norm(nm["n_in"])))
        show(pf + "hn", lambda: (hn_h, dev(pf + "hn")), "RMSNorm on the device's own x")

        def qk_ref():
            hn_d = dev(pf + "hn")
            q_ = w("self_attn.q_proj.weight") @ hn_d
            k_ = w("self_attn.k_proj.weight") @ hn_d
            if sp.qk_norm:
                q_ = np.concatenate([rms(q_.reshape(Hq, hd)[i], load_norm(nm["n_qn"]))
                                     for i in range(Hq)])
                k_ = np.concatenate([rms(k_.reshape(hkv, hd)[i], load_norm(nm["n_kn"]))
                                     for i in range(hkv)])
            return q_, k_

        qk_tag = f"GEMV{'+qk-norm' if sp.qk_norm else ' only'} on the device's own hn"
        show(pf + "q", lambda: (bf(qk_ref()[0]), dev(pf + "q")), qk_tag)
        show(pf + "k", lambda: (bf(qk_ref()[1]), dev(pf + "k")), qk_tag)

        if sp.v_norm or has_v:
            def v_ref():
                hn_d = dev(pf + "hn")
                # attention_k_eq_v: layers with no v_proj tensor read v_norm's INPUT off the raw
                # k projection, before qk-norm touches it -- the same GEMV the `k` row above
                # computes pre-norm, since there is no v_proj tensor for these layers at all.
                pre = ((w("self_attn.v_proj.weight") @ hn_d) if has_v
                       else (w("self_attn.k_proj.weight") @ hn_d))
                if sp.v_norm:
                    pre = np.concatenate([rms(pre.reshape(hkv, hd)[i]) for i in range(hkv)])
                return bf(pre), dev(pf + "v")
            if has_v and sp.v_norm:
                v_tag = "GEMV+v-norm"
            elif sp.v_norm:
                v_tag = "v-norm on k's own GEMV (no v_proj tensor)"
            else:
                v_tag = "GEMV only"
            show(pf + "v", v_ref, v_tag + " on the device's own hn")

        def a_ref():
            cx_d = dev(pf + "cx")
            av = split_matvec(w("self_attn.o_proj.weight"), cx_d, o_chunks)
            if sp.sandwich_norms:
                av = rms(av, load_norm(nm["n_pa"]))
            return bf(av), dev(pf + "a")
        a_tag = "o_proj GEMV" + (f" ({o_chunks}-way K-split)" if o_chunks > 1 else "")
        a_tag += "+n_pa" if sp.sandwich_norms else ""
        show(pf + "a", a_ref, a_tag + " on the device's own cx")

        def u_ref():
            return bf(w("mlp.up_proj.weight") @ dev(pf + "hf")), dev(pf + "u")
        show(pf + "u", u_ref, "GEMV ONLY, on the device's own hf")

        def g_ref():
            hf_d = dev(pf + "hf")
            return bf(act(bf(w("mlp.gate_proj.weight") @ hf_d))), dev(pf + "g")
        show(pf + "g", g_ref, f"SAME GEMV + {sp.act}, same hf  <-- the pair")

        show(pf + "gh", lambda: (bf(dev(pf + "g") * dev(pf + "u")), dev(pf + "gh")),
             "elementwise mul on the device's own g,u")

        def d_ref():
            dv = split_matvec(w("mlp.down_proj.weight"), dev(pf + "gh"), down_chunks)
            if sp.sandwich_norms:
                dv = rms(dv, load_norm(nm["n_pff"]))
            return bf(dv), dev(pf + "d")
        d_tag = "down_proj GEMV" + (f" ({down_chunks}-way K-split)" if down_chunks > 1 else "")
        d_tag += "+n_pff" if sp.sandwich_norms else ""
        show(pf + "d", d_ref, d_tag + " on the device's own gh")

        show(pf + "x1", lambda: (bf(xin + dev(pf + "a")), dev(pf + "x1")),
             "attn residual add on the device's own x,a")

        def xnext_ref():
            raw = dev(pf + "x1") + dev(pf + "d")
            if sp.layer_scalar:
                raw = float(npy(sp.layer_scalar_name(l))[0]) * raw
            return bf(raw), dev("x" + str(l + 1))
        xnext_tag = "FFN residual add" + (" + layer_scalar" if sp.layer_scalar else "")
        show("x" + str(l + 1), xnext_ref, xnext_tag + " on the device's own x1,d")
        print()


if __name__ == "__main__":
    main()
