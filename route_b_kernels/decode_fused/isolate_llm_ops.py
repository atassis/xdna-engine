#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Per-op ISOLATED error: apply the host op to the DEVICE's own inputs.

The cumulative bisect tells you where error has grown to; it cannot tell you which op MAKES it,
because every node inherits its predecessor's error. This feeds each host op the device's own input
buffer, so the number reported is that op's contribution alone.

The load-bearing pair is `u` and `g`: both are a GEMV over the same `hf` with the same kernel, and
`g` additionally passes through SiLU. If g's isolated error is far above u's, the activation is the
source and the GEMV is exonerated -- which no cumulative measurement can establish.
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
    return float(np.linalg.norm(a - b) / r) if r else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="qwen3-0.6b")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--token", type=int, default=785)
    ap.add_argument("--probe-layers", type=int, default=3)
    a = ap.parse_args()

    sp, fused, weights, md = build_graph(a.spec, a.weights, a.layers)
    NL, HD, Hq, Hkv, EPS = md["NL"], sp.head_dim, sp.n_q_heads, sp.n_kv_heads, sp.eps
    c = fused.get_callable()
    params = c.params
    for n, arr in weights.items():
        np.copyto(c.get_buffer(n).data, np.asarray(arr, BF16).reshape(-1))

    def npy(n):
        return np.load(os.path.join(a.weights, f"{n}.npy")).astype(np.float32)

    embed = npy("model.embed_tokens.weight")
    np.copyto(c.get_buffer("x").data, np.asarray(embed[a.token], BF16).reshape(-1))
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

    def dev(n):
        return np.asarray(c.get_buffer(n).data, np.float32)

    def rms(x, w):
        x = np.asarray(x, np.float32)
        return x / np.sqrt((x * x).mean(-1, keepdims=True) + EPS) * w

    def silu(x):
        return x / (1.0 + np.exp(-np.clip(x, -60, 60)))

    def bf(a_):
        return np.asarray(a_, BF16).astype(np.float32)

    print(f"{'op (isolated)':22} {'rel-L2':>11}   what it isolates")
    print("-" * 74)
    for l in range(min(a.probe_layers, NL)):
        p = f"model.layers.{l}."
        pf = f"L{l}_"
        xin = dev("x") if l == 0 else dev(f"x{l}")
        hn_h = bf(rms(xin, npy(p + "input_layernorm.weight")))
        print(f"{pf+'hn':22} {rel(hn_h, dev(pf+'hn')):11.4e}   RMSNorm on the device's own x")
        hn_d = dev(pf + "hn")
        for nm, wn in (("q", "self_attn.q_proj.weight"), ("k", "self_attn.k_proj.weight"),
                       ("v", "self_attn.v_proj.weight")):
            ref = bf(npy(p + wn) @ hn_d)
            if nm == "q":
                ref = bf(np.concatenate([rms(ref.reshape(Hq, HD)[i],
                                             npy(p + "self_attn.q_norm.weight")) for i in range(Hq)]))
            if nm == "k":
                ref = bf(np.concatenate([rms(ref.reshape(Hkv, HD)[i],
                                             npy(p + "self_attn.k_norm.weight")) for i in range(Hkv)]))
            tag = "GEMV" + ("+qk-norm" if nm in "qk" else " only")
            print(f"{pf+nm:22} {rel(ref, dev(pf+nm)):11.4e}   {tag} on the device's own hn")
        hf_d = dev(pf + "hf")
        u_h = bf(npy(p + "mlp.up_proj.weight") @ hf_d)
        g_pre = npy(p + "mlp.gate_proj.weight") @ hf_d
        g_h = bf(silu(bf(g_pre)))
        print(f"{pf+'u':22} {rel(u_h, dev(pf+'u')):11.4e}   GEMV ONLY, on the device's own hf")
        print(f"{pf+'g':22} {rel(g_h, dev(pf+'g')):11.4e}   SAME GEMV + SiLU, same hf  <-- the pair")
        gh_h = bf(dev(pf + "g") * dev(pf + "u"))
        print(f"{pf+'gh':22} {rel(gh_h, dev(pf+'gh')):11.4e}   elementwise mul on the device's own g,u")
        d_h = bf(npy(p + "mlp.down_proj.weight") @ dev(pf + "gh"))
        print(f"{pf+'d':22} {rel(d_h, dev(pf+'d')):11.4e}   GEMV on the device's own gh")
        x1_h = bf(xin + dev(pf + "a"))
        print(f"{pf+'x1':22} {rel(x1_h, dev(pf+'x1')):11.4e}   residual add on the device's own x,a")
        print()


if __name__ == "__main__":
    main()
