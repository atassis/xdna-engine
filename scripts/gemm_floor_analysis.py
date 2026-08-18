#!/usr/bin/env python3
# Decompose the whole_array GEMM command into per-command floor, DDR bytes and compute, and then
# split the floor itself into host round-trip and device-serial residue
# (task gemm-offcore-residue-occupancy, item 1).
#
# Reads the artifacts the device windows wrote; no device needed. Two stages:
#
#   1. SHAPE REGRESSION. Over arms at a FIXED cols, fit T = c + x*DDR_MiB + y*t_peak. Fixing cols is
#      the point: the width series varies cores AND bytes together (A's shim BD outer repeat falls
#      8 -> 2 -> 1 as cols goes 1 -> 4 -> 8, so DDR traffic falls 14 -> 8 -> 7 MiB for the identical
#      operands), so a fit across widths cannot attribute anything to either.
#   2. FLOOR SPLIT. The regression's intercept is an extrapolation, so it is checked against a
#      directly measured near-zero-work arm, and then split by the pipeline-depth sweep into the
#      part that amortizes when commands are queued (host/driver round-trip) and the part that does
#      not (device-serial per-command work).
import argparse
import json
import os

import numpy as np


def load(path):
    return json.load(open(path)) if os.path.exists(path) else []


def regress(rows):
    b = np.array([r["ddr_mib"] for r in rows])
    p = np.array([r["t_peak_us"] for r in rows])
    m = np.array([r["median_us"] for r in rows])
    X = np.column_stack([np.ones(len(m)), b, p])
    beta, *_ = np.linalg.lstsq(X, m, rcond=None)
    pred = X @ beta
    n, k = len(m), X.shape[1]
    se = np.sqrt(np.diag((np.sum((pred - m) ** 2) / (n - k)) * np.linalg.inv(X.T @ X)))
    return beta, se, pred, m, b, p


def main(o):
    rows = [r for f in o.accounting for r in load(f)]
    rows = [r for r in rows if r["cols"] == o.cols]
    if len(rows) < 4:
        raise SystemExit(f"need >=4 arms at cols={o.cols}, got {len(rows)}")
    beta, se, pred, m, b, p = regress(rows)
    c, x, y = beta
    print(f"=== shape regression, cols={o.cols}, {len(rows)} arms ===")
    print(f"  T = {c:.1f}(+-{se[0]:.1f}) + {x:.2f}(+-{se[1]:.2f})*DDR_MiB "
          f"+ {y:.2f}(+-{se[2]:.2f})*t_peak")
    print(f"  marginal DDR {2**20/(x*1e-6)/1e9:.1f} GB/s   "
          f"marginal compute {100/y:.1f}% of bfp16 peak   "
          f"RMSE {np.sqrt(np.mean((pred-m)**2)):.1f} us   "
          f"max|res| {np.max(np.abs(100*(pred-m)/m)):.1f}%")

    print(f"\n=== floor split, from the pipeline-depth sweep ===")
    pipe = load(o.pipeline)
    if not pipe:
        return
    print(f"  {'shape':16s} {'DDR':>6s} {'submit':>7s} {'d1':>8s} {'deep':>8s} {'saved':>8s} {'ratio':>6s}")
    saved, byts = [], []
    for r in pipe:
        d = r["depths"]
        d1, dn = d[str(min(int(k) for k in d))], d[str(max(int(k) for k in d))]
        s = d1["per_cmd_us"] - dn["per_cmd_us"]
        saved.append(s)
        byts.append(r["ddr_mib"])
        print(f"  {r['M']}x{r['K']}x{r['N']:<8d} {r['ddr_mib']:6.2f} "
              f"{r['submit_us_median']:7.2f} {d1['per_cmd_us']:8.1f} {dn['per_cmd_us']:8.1f} "
              f"{s:8.1f} {d1['per_cmd_us']/dn['per_cmd_us']:6.2f}x")

    # What amortizes is not one number: a fixed host round-trip plus a byte-proportional overlap of
    # adjacent commands' DMA phases. Separating them is what says how much is dispatch cost.
    A = np.column_stack([np.ones(len(saved)), byts])
    (fixed, per_mib), *_ = np.linalg.lstsq(A, np.array(saved), rcond=None)
    submits = [r["submit_us_median"] for r in pipe]
    print(f"\n  amortizable = {fixed:.1f} us fixed + {per_mib:.2f} us/MiB "
          f"(adjacent-command DMA overlap)")
    print(f"  host submit call = {np.median(submits):.2f} us "
          f"(range {min(submits):.2f}..{max(submits):.2f} over a "
          f"{max(byts)/min(byts):.0f}x byte range)")
    print(f"  => completion round-trip = {fixed - np.median(submits):.1f} us")

    tiny = min(pipe, key=lambda r: r["ddr_mib"])
    d = tiny["depths"]
    d1 = d[str(min(int(k) for k in d))]["per_cmd_us"]
    dn = d[str(max(int(k) for k in d))]["per_cmd_us"]
    work = x * tiny["ddr_mib"] + y * tiny["t_peak_us"]
    print(f"\n  DIRECT floor read at {tiny['M']}x{tiny['K']}x{tiny['N']} "
          f"({tiny['ddr_mib']} MiB, t_peak {tiny['t_peak_us']} us):")
    print(f"    depth-1 {d1:.1f} us - modelled work {work:.1f} us = floor {d1-work:.1f} us")
    print(f"    of which device-serial (survives queueing) = {dn-work:.1f} us, "
          f"host round-trip = {d1-dn:.1f} us")
    print(f"    regression intercept {c:.1f} us OVERSTATES this by "
          f"{100*(c-(d1-work))/(d1-work):+.0f}% -- the response curves at the small end")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--accounting", nargs="+",
                    default=["artifacts/gemm_floor_shape_series.json",
                             "artifacts/gemm_floor_tiny.json"])
    ap.add_argument("--pipeline", default="artifacts/gemm_dispatch_pipeline_deep.json")
    ap.add_argument("--cols", type=int, default=8)
    main(ap.parse_args())
