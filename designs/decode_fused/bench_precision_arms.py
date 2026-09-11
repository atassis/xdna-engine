#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Interleaved A/B across several PRECISION arms of the fused decode, at one shared n_past.

Sibling to bench_bucket_arms.py, which varies `max_seq` (KV bytes) at fixed format. This varies
the weight FORMAT (weight bytes) at fixed shape, and it exists to answer two questions at once.

  1. Does a narrow weight stream pay on LATENCY? The standing objection is that int8's dequant has
     no zero-overhead loop where int4's does, so its byte win carries an unpriced compute cost.
     That has never been measured in either direction.

  2. What does a byte cut CONVERT at on this base? Two measured models disagree by 1.5x -- the
     transport law's fitted marginal rate (18.28 us/MB) against the ~2/3 of floor arithmetic the
     only device-measured byte lever returned across three bases. That factor sits underneath
     every byte-lever prediction in this project, so it is worth more than the arm.

Question 2 is why this takes SEVERAL arms rather than one A/B: three byte deltas spanning 3.08x
give a fitted SLOPE with an intercept term, where one pair gives a difference of two absolutes
(D028). A non-zero intercept is itself the finding -- it is what the dequant's own cost looks
like, a term that moves with the ARM and not with its bytes.

All arms are held resident (separate hw_contexts, alive at once) and dispatched round-robin, so
thermal/DVFS drift over the session lands on every arm equally -- bench_bucket_arms.py's reason,
unchanged. Report the SPREAD column, not just the median.

  python designs/decode_fused/bench_precision_arms.py --spec qwen3-0.6b \
      --weights artifacts/qwen3-0.6b/weights --layers 28 --max-seq 4096 --pos 1024 \
      --arms bf16 '{"head":"int8a/g128"}' mlp-int8 mlp-head-int8
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
import precision as P  # noqa: E402
from gen_llm_decode import build_graph, report_artifact_freshness, load_weight_buffer  # noqa: E402
from iron.common.kv_layout import KVLayout  # noqa: E402
from bench_llm_decode import rope_row  # noqa: E402

BF16 = ml_dtypes.bfloat16


def median_spread(xs):
    lo, hi = min(xs), max(xs)
    return dict(median=statistics.median(xs), mean=statistics.fmean(xs), min=lo, max=hi,
                spread_pct=100.0 * (hi - lo) / statistics.median(xs), n=len(xs))


def resolve(text):
    """An --arms value: a preset name or inline JSON, the same vocabulary PRECISION takes."""
    raw = P.PRESETS[text][0] if text in P.PRESETS else None
    return P.parse_plan(json.dumps(raw)) if raw is not None else P.parse_plan(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--max-seq", type=int, default=4096)
    ap.add_argument("--arms", nargs="+", required=True,
                    help="preset names or inline JSON plans; the first is the control")
    ap.add_argument("--pos", type=int, required=True, help="n_past every arm is dispatched at")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--out-json", default=None)
    a = ap.parse_args()

    report_artifact_freshness(a.weights)
    plans = [(t, resolve(t)) for t in a.arms]
    if a.pos + 1 > a.max_seq:
        raise SystemExit(f"--pos {a.pos} needs sm_mask={a.pos + 1} <= max_seq {a.max_seq}")

    embed = np.load(os.path.join(a.weights, "model.embed_tokens.weight.npy")).astype(np.float32)
    TOK = 100

    arms = []
    for tag, plan in plans:
        t0 = time.perf_counter()
        sp, fused, weights, md = build_graph(a.spec, a.weights, a.layers, a.max_seq,
                                             precision_plan=plan)
        mb = P.token_mb(plan)["total"]
        print(f"[prec-arms] built {tag} in {time.perf_counter() - t0:.1f}s "
              f"({md['NL']} layers, {mb:.1f} MB/token projected)", flush=True)
        c = fused.get_callable()
        params = c.params
        if params is None:
            raise SystemExit(f"[prec-arms] arm {tag}: no runtime parameters bound")
        for name, arr in weights.items():
            # load_weight_buffer, NOT a bf16 cast: a packed weight is opaque on-wire bytes and
            # value-casting it to bf16 reinterprets the payload. bench_bucket_arms.py can cast
            # because every arm it builds is bf16; this one cannot.
            load_weight_buffer(c.get_buffer(name), arr)
        scale = np.sqrt(sp.d_model) if sp.embed_scale == "sqrt_d_model" else 1.0
        arms.append(dict(tag=tag, plan=plan, mb=mb, sp=sp, c=c, params=params,
                         kv_layout=KVLayout(Hkv=sp.n_kv_heads, S=md["S"], HD=sp.head_dim,
                                            T=md["T"]),
                         xin=c.get_buffer("x"), rope_buf=c.get_buffer("rope_global"),
                         scale=scale))
    print(f"[prec-arms] {len(arms)} arms resident concurrently, dispatching at pos={a.pos}",
          flush=True)

    def one(arm):
        with arm["xin"].overwrite() as buf:
            buf[:] = np.asarray(embed[TOK] * arm["scale"], BF16).reshape(-1)
        with arm["rope_buf"].overwrite() as buf:
            buf[:] = rope_row(a.pos, arm["sp"].head_dim, arm["sp"].rope_theta_global).reshape(-1)
        arm["params"].write("kv_off", int(arm["kv_layout"].kv_off(a.pos)))
        arm["params"].write("sm_mask", int(a.pos + 1))
        arm["params"].sync()
        arm["c"]()
        return float(arm["c"].last_elapsed)

    for arm in arms:
        for _ in range(a.warmup):
            one(arm)

    # Round-robin: rep 0 touches every arm once, then rep 1. Not one arm's whole budget first.
    samples = {arm["tag"]: [] for arm in arms}
    for _ in range(a.reps):
        for arm in arms:
            samples[arm["tag"]].append(one(arm))

    ctl = arms[0]
    stats = {t: median_spread(v) for t, v in samples.items()}
    print(f"\n{'arm':22} {'MB/token':>9} {'ms median':>10} {'spread':>8} {'d_ms':>9} {'us/MB':>8}")
    pts = []
    for arm in arms:
        st = stats[arm["tag"]]
        dmb = arm["mb"] - ctl["mb"]
        # Paired WITHIN rep before aggregating -- never a difference of two medians.
        dms = statistics.median([samples[arm["tag"]][i] - samples[ctl["tag"]][i]
                                 for i in range(a.reps)])
        conv = f"{1e3 * dms / dmb:8.2f}" if dmb else " " * 8
        print(f"{arm['tag']:22} {arm['mb']:9.1f} {st['median']:10.3f} "
              f"{st['spread_pct']:7.1f}% {dms:+9.3f} {conv}")
        if dmb:
            pts.append((dmb, dms))

    fit = None
    if len(pts) >= 2:
        x = np.array([p[0] for p in pts]); y = np.array([p[1] for p in pts])
        slope, intercept = np.polyfit(x, y, 1)
        resid = float(np.abs(y - (slope * x + intercept)).max())
        fit = dict(us_per_mb=float(slope * 1e3), intercept_ms=float(intercept), max_resid_ms=resid)
        print(f"\nfit  d_ms = {slope * 1e3:.2f} us/MB * dMB {intercept:+.3f} ms"
              f"   max|resid| {resid:.3f} ms")
        print(f"  transport law's marginal rate  18.28 us/MB")
        print(f"  two-thirds of floor arithmetic 12.19 us/MB")
        if abs(intercept) > 0.5:
            print(f"  the {intercept:+.3f} ms intercept moves with the ARM and not with its "
                  "bytes -- that is the shape the dequant's own core cost would have")

    if a.out_json:
        json.dump({"pos": a.pos, "reps": a.reps, "max_seq": a.max_seq,
                   "arms": [{"tag": arm["tag"], "mb": arm["mb"],
                             "plan": {k: str(v) for k, v in arm["plan"].items()},
                             "stats": stats[arm["tag"]], "samples": samples[arm["tag"]]}
                            for arm in arms],
                   "fit": fit}, open(a.out_json, "w"), indent=1)
        print(f"[prec-arms] wrote {a.out_json}", flush=True)


if __name__ == "__main__":
    main()
