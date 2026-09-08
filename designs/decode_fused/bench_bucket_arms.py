#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Interleaved A/B timing across several `max_seq` builds of the fused decode, at ONE shared n_past.

Sibling to bench_llm_decode.py, built for llm-decode-attention-pads-to-full-window.md gate 3:
"interleave arms within ONE session, 3+ alternating reps ... any effect under ~15% is only real if
measured alternating" (decode-gemv-dispatch-floor's 2.2%-vs-8.0% within/cross-session spread).

Each --arms value is a separate `max_seq` the WHOLE graph (kc/vc/kr/vr/vt/sc/sw, op_scores,
op_rep_k/v, op_trv, op_ctx, softmax -- see gen_llm_decode.py's comment above op_rep_k for why this
must be uniform, not the op_ctx-excluded 4-of-5 split the task record scoped out) is built and
compiled at. All arms are held resident (separate hw_contexts, all alive at once) and dispatched in
round-robin order so any thermal/DVFS drift across the session lands on every arm equally. Each
arm's own hw_context stays loaded for its own dispatch (c.last_elapsed is run.start()+wait() only),
so a per-arm-switch context-reload cost, if there is one, shows up as WIDER spread on every arm
alike rather than being charged to one arm's median -- report the spread column, not just the median.

  python designs/decode_fused/bench_bucket_arms.py --spec qwen3-0.6b \
      --weights artifacts/qwen3-0.6b/weights --arms 2048 256 --pos 7 --reps 30
"""
import argparse
import json
import os
import statistics
import sys
import time

import numpy as np
import ml_dtypes

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import newstack_compat  # noqa: F401,E402
from gen_llm_decode import build_graph, report_artifact_freshness  # noqa: E402
from bench_llm_decode import rope_row  # noqa: E402

BF16 = ml_dtypes.bfloat16


def median_spread(xs):
    xs = sorted(xs)
    n = len(xs)
    med = statistics.median(xs)
    spread = (xs[-1] - xs[0]) / med * 100.0 if med else float("nan")
    return med, spread


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--arms", type=int, nargs="+", required=True,
                     help="max_seq values to build, one arm each (first is the reference/baseline)")
    ap.add_argument("--pos", type=int, required=True,
                     help="n_past to dispatch at -- must be < every arm's max_seq (sm_mask=pos+1)")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--out-json", default=None)
    a = ap.parse_args()

    report_artifact_freshness(a.weights)

    for s in a.arms:
        if a.pos + 1 > s:
            raise SystemExit(f"--pos {a.pos} needs sm_mask={a.pos + 1} <= every arm's max_seq; "
                              f"arm {s} is too small")

    embed = np.load(os.path.join(a.weights, "model.embed_tokens.weight.npy")).astype(np.float32)
    TOK = 100

    arms = []
    for s in a.arms:
        t0 = time.perf_counter()
        sp, fused, weights, md = build_graph(a.spec, a.weights, a.layers, s)
        print(f"[bucket-arms] built max_seq={s} in {time.perf_counter() - t0:.1f}s "
              f"({md['NL']} layers)", flush=True)
        c = fused.get_callable()
        params = c.params
        if params is None:
            raise SystemExit(f"[bucket-arms] arm max_seq={s}: no runtime parameters bound")
        for name, arr in weights.items():
            with c.get_buffer(name).overwrite() as _buf:
                _buf[:] = np.asarray(arr, BF16).reshape(-1)
        scale = np.sqrt(sp.d_model) if sp.embed_scale == "sqrt_d_model" else 1.0
        arms.append(dict(s=s, sp=sp, c=c, params=params,
                          xin=c.get_buffer("x"), rope_buf=c.get_buffer("rope_global"),
                          out=c.get_buffer("logits"), scale=scale, vocab=sp.vocab))
    print(f"[bucket-arms] {len(arms)} arms resident concurrently, dispatching at pos={a.pos}",
          flush=True)

    def one(arm):
        with arm["xin"].overwrite() as _buf:
            _buf[:] = np.asarray(embed[TOK] * arm["scale"], BF16).reshape(-1)
        with arm["rope_buf"].overwrite() as _buf:
            _buf[:] = rope_row(a.pos, arm["sp"].head_dim, arm["sp"].rope_theta_global).reshape(-1)
        arm["params"].write("kv_off", int(a.pos * arm["sp"].head_dim))
        arm["params"].write("sm_mask", int(a.pos + 1))
        arm["params"].sync()
        arm["c"]()
        return float(arm["c"].last_elapsed)

    for arm in arms:
        for _ in range(a.warmup):
            one(arm)

    # Round-robin: rep 0 touches every arm once, then rep 1, etc. -- this is the "interleave" gate 3
    # asks for, as opposed to finishing one arm's whole rep budget before starting the next.
    samples = {arm["s"]: [] for arm in arms}
    for rep in range(a.reps):
        for arm in arms:
            samples[arm["s"]].append(one(arm))

    print(f"\n{'max_seq':>8} {'n':>4} {'median_ms':>10} {'spread_%':>9} {'min_ms':>8} {'max_ms':>8}")
    report = {}
    for arm in arms:
        xs = [t * 1e3 for t in samples[arm["s"]]]
        med, spread = median_spread(xs)
        report[arm["s"]] = {"reps": xs, "median_ms": med, "spread_pct": spread}
        print(f"{arm['s']:8} {len(xs):4} {med:10.3f} {spread:9.2f} {min(xs):8.3f} {max(xs):8.3f}")

    base = a.arms[0]
    if len(arms) > 1:
        print()
        for s in a.arms[1:]:
            d = (1 - report[s]["median_ms"] / report[base]["median_ms"]) * 100
            print(f"max_seq={s} vs max_seq={base}: {d:+.1f}% ms/token "
                  f"(median {report[s]['median_ms']:.3f} vs {report[base]['median_ms']:.3f} ms)")

    if a.out_json:
        json.dump({"spec": a.spec, "pos": a.pos, "arms": report}, open(a.out_json, "w"), indent=2)
        print(f"\n[bucket-arms] wrote {a.out_json}")


if __name__ == "__main__":
    main()
