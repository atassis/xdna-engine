#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Interleaved timing across several LAYER-COUNT builds of the fused decode, at one shared max_seq.

Sibling to bench_bucket_arms.py, which varies `max_seq` (KV bytes) at fixed depth. This varies
DEPTH at fixed max_seq, which is the only axis that separates a per-LAYER cost from a per-TOKEN
one -- every arm measured so far has been 28 layers, so the whole decode graph's structure moved
together and nothing could tell the two apart.

The graph is exactly affine in depth: configures = 6*L + 2 and bytes = 44.465*L + 311.49 MB, so

    t(L) = alpha * L + beta

splits the step into a per-layer term (bytes + configures + whatever else scales with depth) and a
per-token term (lm_head bytes, the final norm, and any fixed per-dispatch lump). Subtracting the
independently measured byte and configure rates from each leaves the part of the 24.74 ms floor
that lives on each side -- which is what decides whether fusing a whole layer can reach the floor
at all.

Arms are held resident and dispatched ROUND-ROBIN for the same reason bench_bucket_arms.py does it:
any thermal/DVFS drift over the session lands on every arm equally. Logits are garbage at L < full
depth -- this measures time only.

  python designs/decode_fused/bench_layer_arms.py --spec qwen3-0.6b \
      --weights artifacts/qwen3-0.6b/weights --arms 28 14 8 4 --max-seq 512 --pos 7 --reps 25
"""
import argparse
import json
import re
import os
import statistics
import sys
import time

import numpy as np
import ml_dtypes

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import newstack_compat  # noqa: F401,E402
import gen_llm_decode as G  # noqa: E402
from gen_llm_decode import build_graph, report_artifact_freshness  # noqa: E402
from bench_llm_decode import rope_row  # noqa: E402

BF16 = ml_dtypes.bfloat16


def census_from_mlir(fused, md):
    """The arm's CONTROL variables, read off its own fused MLIR rather than assumed.

    A fit's control variables need the same evidence as its result: the 39.3 us/configure fit this
    file supersedes broke because one arm's `runs` differed from its `configures` and nobody
    printed both before fitting.
    """
    out = {"NL": md.get("NL"), "S": md.get("S")}
    try:
        src = open(fused.artifacts[0].mlir_input.filename).read()
        out["configures"] = len(re.findall(r"aiex\.configure\s+@", src))
        out["runs"] = len(re.findall(r"aiex\.run\s+@", src))
        out["designs"] = len(re.findall(r"aie\.device\(", src))
    except Exception as e:
        out["census_error"] = repr(e)
    return out


def median_spread(xs):
    xs = sorted(xs)
    med = statistics.median(xs)
    return med, ((xs[-1] - xs[0]) / med * 100.0 if med else float("nan"))


def fit_affine(pairs):
    """Least-squares t = alpha*L + beta over (L, t_ms). Returns alpha, beta, max |residual|."""
    n = len(pairs)
    sx = sum(L for L, _ in pairs)
    sy = sum(t for _, t in pairs)
    sxx = sum(L * L for L, _ in pairs)
    sxy = sum(L * t for L, t in pairs)
    den = n * sxx - sx * sx
    alpha = (n * sxy - sx * sy) / den
    beta = (sy - alpha * sx) / n
    resid = [t - (alpha * L + beta) for L, t in pairs]
    return alpha, beta, resid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--arms", nargs="+", required=True,
                    help="arm spec L[f]: layer count, optional 'f' = FUSE_MLP_O=1 (folds op_o into "
                         "the MLP design: 5 runs/layer instead of 6, bytes ~unchanged). Fitting "
                         "both variants gives alpha at two runs-per-layer values, which is what "
                         "prices a whole-layer fusion.")
    ap.add_argument("--max-seq", type=int, default=512)
    ap.add_argument("--pos", type=int, required=True)
    ap.add_argument("--reps", type=int, default=25)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--out-json", default=None)
    a = ap.parse_args()

    report_artifact_freshness(a.weights)
    if a.pos + 1 > a.max_seq:
        raise SystemExit(f"--pos {a.pos} needs sm_mask={a.pos+1} <= max_seq {a.max_seq}")

    embed = np.load(os.path.join(a.weights, "model.embed_tokens.weight.npy")).astype(np.float32)
    TOK = 100

    arms = []
    census = {}
    for spec in a.arms:
        fmo = spec.endswith("f")
        L = int(spec[:-1] if fmo else spec)
        # FUSE_MLP_O is captured at gen_llm_decode IMPORT time, so flipping os.environ here would
        # be silently ignored -- set the module global the generator actually reads.
        G.FUSE_MLP_O = fmo
        t0 = time.perf_counter()
        sp, fused, weights, md = build_graph(a.spec, a.weights, L, a.max_seq)
        # A fit's CONTROL variables need the same evidence as its result: print every quantity
        # that differs between arms BEFORE fitting, not just the one being varied.
        census[spec] = census_from_mlir(fused, md)
        census[spec]["fuse_mlp_o"] = fmo
        print(f"[layer-arms] built {spec} in {time.perf_counter()-t0:.1f}s  census={census[spec]}",
              flush=True)
        c = fused.get_callable()
        params = c.params
        if params is None:
            raise SystemExit(f"[layer-arms] arm {spec}: no runtime parameters bound")
        for name, arr in weights.items():
            np.copyto(c.get_buffer(name).data, np.asarray(arr, BF16).reshape(-1))
        del weights
        scale = np.sqrt(sp.d_model) if sp.embed_scale == "sqrt_d_model" else 1.0
        arms.append(dict(spec=spec, L=L, fmo=fmo, sp=sp, c=c, params=params,
                         xin=c.get_buffer("x"), rope_buf=c.get_buffer("rope_global"),
                         scale=scale))
    print(f"[layer-arms] {len(arms)} arms resident, dispatching at pos={a.pos}", flush=True)

    def one(arm):
        np.copyto(arm["xin"].data, np.asarray(embed[TOK] * arm["scale"], BF16).reshape(-1))
        np.copyto(arm["rope_buf"].data,
                  rope_row(a.pos, arm["sp"].head_dim, arm["sp"].rope_theta_global).reshape(-1))
        arm["params"].write("kv_off", int(a.pos * arm["sp"].head_dim))
        arm["params"].write("sm_mask", int(a.pos + 1))
        arm["params"].sync()
        arm["c"]()
        return float(arm["c"].last_elapsed)

    for arm in arms:
        for _ in range(a.warmup):
            one(arm)

    samples = {arm["spec"]: [] for arm in arms}
    for _ in range(a.reps):
        for arm in arms:
            samples[arm["spec"]].append(one(arm))

    print(f"\n{'arm':>6} {'L':>4} {'cfg':>5} {'n':>4} {'median_ms':>10} {'spread_%':>9} "
          f"{'min_ms':>9} {'max_ms':>9}")
    report = {}
    for arm in sorted(arms, key=lambda x: (x["fmo"], x["L"])):
        sp_ = arm["spec"]
        xs = [t * 1e3 for t in samples[sp_]]
        med, spread = median_spread(xs)
        report[sp_] = {"reps": xs, "median_ms": med, "spread_pct": spread, "L": arm["L"],
                       "fuse_mlp_o": arm["fmo"], "min_ms": min(xs), "census": census[sp_]}
        print(f"{sp_:>6} {arm['L']:4} {census[sp_].get('configures', 0):5} {len(xs):4} "
              f"{med:10.3f} {spread:9.2f} {min(xs):9.3f} {max(xs):9.3f}")

    # Fit on MEDIANS and again on MINIMA. A sustained downclock inflates a whole cell with LOW
    # variance (four-configures-a-layer-came-off-without-a-new-kernel: one cell read 110.261 ms at
    # sd 0.147), so a spread filter cannot catch it -- agreement between the two fits is the check.
    for fmo in (False, True):
        group = sorted((s_ for s_ in report if report[s_]["fuse_mlp_o"] == fmo),
                       key=lambda s_: report[s_]["L"])
        if len(group) < 2:
            continue
        tag = "FUSE_MLP_O=1 (5 runs/layer)" if fmo else "FUSE_MLP_O=0 (6 runs/layer)"
        for label, key in (("median", "median_ms"), ("min", "min_ms")):
            pairs = [(report[s_]["L"], report[s_][key]) for s_ in group]
            alpha, beta, resid = fit_affine(pairs)
            print(f"\n{tag} -- fit on {label}s:  t = {alpha:.4f} ms/layer * L + {beta:.4f} ms")
            for (L, t), r in zip(pairs, resid):
                print(f"   L={L:3}  {t:9.3f}   residual {r:+7.3f}")
            print(f"   extrapolated to L=28: {alpha*28 + beta:.3f} ms")

    if a.out_json:
        json.dump({"spec": a.spec, "pos": a.pos, "max_seq": a.max_seq, "arms": report},
                  open(a.out_json, "w"), indent=2)
        print(f"\n[layer-arms] wrote {a.out_json}")


if __name__ == "__main__":
    main()
