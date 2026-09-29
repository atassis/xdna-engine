#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Argmax gate for the rail-built Whisper decode ELF against the host ONNX decode, on device.

Drives the graph gen_llm_decode.py builds for a `cross_len` spec. Per clip the host writes each
layer's cross K/V once (from the reference's encoder hidden and the dumped cross weights), then per
token writes `x = embed[tok] + embed_positions[pos]`, `kv_off` and `sm_mask` and dispatches once.
Logits come back padded; the host trims them and adds the folded final-LayerNorm bias.

Two readings per clip against scripts/whisper_host_decode_ref.py's npz:
  teacher-forced  the ONNX tokens are fed, and each step's argmax is compared to the next one.
  free-running    the device's own tokens are fed until <|endoftext|>; written to --out for WER.

Single-tenant: take npu_lock.sh and stop xdna-engine.service first.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import ml_dtypes

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import newstack_compat  # noqa: F401,E402
from gen_llm_decode import build_graph, isolate_build_dir, load_weight_buffer  # noqa: E402
from iron.common.kv_layout import KVLayout  # noqa: E402

BF16 = ml_dtypes.bfloat16
EOT = 50257


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="whisper-small")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--ref", required=True, help="dir of <clip>.npz from whisper_host_decode_ref.py")
    ap.add_argument("--max-seq", type=int, default=512)
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--clips", default=None, help="comma-separated stems; default every npz")
    ap.add_argument("--max-new", type=int, default=120)
    ap.add_argument("--out", default=None, help="JSON of per-clip device ids and timings")
    a = ap.parse_args()
    isolate_build_dir("verify-whisper")

    sp, fused, weights, md = build_graph(a.spec, a.weights, a.layers, a.max_seq)
    NL, S = md["NL"], md["S"]
    H, HD, D, V = sp.n_q_heads, sp.head_dim, sp.d_model, sp.vocab
    TP = sp.cross_len_padded()
    c = fused.get_callable()
    for name, arr in weights.items():
        load_weight_buffer(c.get_buffer(name), arr)
    c.scratch_buffer.device = "cpu"
    c.scratch_buffer.to("npu")
    params, xin, out = c.params, c.get_buffer("x"), c.get_buffer("logits")
    (slot, slot_hd, slot_w, mask_slot, slot_blk, slot_hkv), = md["geom_slots"]
    kvl = KVLayout(Hkv=slot_hkv, S=slot_w, HD=slot_hd, T=slot_blk)
    bias = md["logit_bias"]

    def w(name):
        return np.load(os.path.join(a.weights, f"{sp.weight_prefix}{name}.npy")).astype(np.float32)

    embed, pos_tab = w("embed_tokens.weight"), w("embed_positions.weight")
    xw = [{k: w(f"layers.{l}.encoder_attn.{k}") for k in
           ("k_proj.weight", "v_proj.weight", "v_proj.bias")} for l in range(NL)]

    def heads(m):
        o = np.zeros((H, TP, HD), np.float32)
        o[:, :m.shape[0]] = m.reshape(m.shape[0], H, HD).transpose(1, 0, 2)
        return o.reshape(-1)

    def load_clip(enc):
        for l in range(NL):
            p = f"L{l}_"
            load_weight_buffer(c.get_buffer(p + "kx"), heads(enc @ xw[l]["k_proj.weight"].T))
            load_weight_buffer(c.get_buffer(p + "vx"), heads(enc @ xw[l]["v_proj.weight"].T
                                                             + xw[l]["v_proj.bias"]))
            for k in ("kc", "vc"):
                load_weight_buffer(c.get_buffer(p + k), np.zeros(kvl.total_elems, np.float32))
        c.scratch_buffer.device = "cpu"
        c.scratch_buffer.to("npu")

    def step(tok, pos, times):
        with xin.overwrite() as b:
            b[:] = np.asarray(embed[tok].astype(BF16).astype(np.float32) + pos_tab[pos], BF16)
        params.write(slot, int(kvl.kv_off(pos)))
        params.write(mask_slot, pos + 1)
        params.sync()
        t0 = time.perf_counter()
        c()
        times.append(time.perf_counter() - t0)
        return np.asarray(out.data[:V], np.float32) + bias

    stems = a.clips.split(",") if a.clips else sorted(p.stem for p in Path(a.ref).glob("*.npz"))
    report, tf_hit, tf_n, fr_same = {}, 0, 0, 0
    for stem in stems:
        r = np.load(Path(a.ref) / f"{stem}.npz")
        ids, n_p = [int(i) for i in r["ids"]], int(r["n_prompt"])
        load_clip(r["enc"])
        times, hits, first_miss = [], 0, None
        for pos in range(len(ids) - 1):
            lg = step(ids[pos], pos, times)
            if pos >= n_p - 1:
                top = int(np.argmax(lg))
                if top == ids[pos + 1]:
                    hits += 1
                elif first_miss is None:
                    s2 = np.sort(lg)[-2:]
                    first_miss = dict(pos=pos, want=ids[pos + 1], got=top,
                                      margin=float(lg[top] - lg[ids[pos + 1]]),
                                      top2_gap=float(s2[1] - s2[0]))
        n = len(ids) - n_p
        load_clip(r["enc"])
        free, tok = list(ids[:n_p]), None
        for pos in range(a.max_new + n_p - 1):
            lg = step(free[pos], pos, [])
            if pos >= n_p - 1:
                tok = int(np.argmax(lg))
                free.append(tok)
                if tok == EOT:
                    break
        same = free == ids
        tf_hit, tf_n, fr_same = tf_hit + hits, tf_n + n, fr_same + same
        report[stem] = dict(tf_hits=hits, tf_steps=n, first_miss=first_miss, free_ids=free,
                            free_identical=same, step_ms=[t * 1e3 for t in times])
        print(f"[{stem}] teacher-forced {hits}/{n}  free-run {'IDENTICAL' if same else 'DIFFERS'}"
              f"  median step {1e3 * float(np.median(times)):.2f} ms"
              + (f"  first miss {first_miss}" if first_miss else ""), flush=True)
    print(f"[verify-whisper] teacher-forced argmax {tf_hit}/{tf_n}, free-run identical "
          f"{fr_same}/{len(stems)} clips")
    if a.out:
        json.dump(report, open(a.out, "w"))


if __name__ == "__main__":
    main()
