#!/usr/bin/env python3
"""Collapse a restream_activity_sweep run into one table per axis, against the positive control.

The question is a THRESHOLD one -- at which interleaved-ballast count, if any, does the isolated
same-context restream price stop being zero -- so the useful view is the price and its CI as a
column against N, with the control's +2.3 ms printed alongside as the scale the sweep must be read
on. A point is called ON only when its CI clears zero; a point estimate that wanders is what the
dissimilarity ladder already produced while being flat.
"""
import argparse
import glob
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="artifacts/restream_activity")
a = ap.parse_args()

ctl = {}
ctl_path = os.path.join(a.dir, "control_twoctx.json")
if os.path.exists(ctl_path):
    for r in json.load(open(ctl_path))["results"]:
        ctl[r["pair"]] = r

by_axis = {}
for p in sorted(glob.glob(os.path.join(a.dir, "*_n*.json"))):
    base = os.path.basename(p)[:-5]
    axis, _, n = base.rpartition("_n")
    d = json.load(open(p))
    for r in d["results"]:
        by_axis.setdefault((axis, r["pair"]), []).append((int(n), r))

if ctl:
    print("positive control -- two hw_contexts, no ballast")
    for name, r in ctl.items():
        print(f"  {name:30s} {r['mean_us']:10.2f} us/change  "
              f"[{r['mean_us'] - r['ci95_us']:.2f}, {r['mean_us'] + r['ci95_us']:.2f}]  "
              f"{r['pos']}/{r['reps']} pos")
    print()

AXIS = {"ctx": "ballast on its OWN hw_context", "bos": "ballast on the measured context"}
for (axis, pair), rows in sorted(by_axis.items()):
    print(f"### {axis} -- {AXIS.get(axis, axis)} -- {pair}")
    print(f"  {'N':>3s} {'us/change':>11s} {'95% CI':>24s} {'pos':>7s} {'arm ms':>9s}   verdict")
    for n, r in sorted(rows):
        lo, hi = r["mean_us"] - r["ci95_us"], r["mean_us"] + r["ci95_us"]
        on = "ON" if lo > 0 else ("negative" if hi < 0 else "off")
        print(f"  {n:3d} {r['mean_us']:11.2f}   [{lo:9.2f}, {hi:9.2f}] {r['pos']:3d}/{r['reps']:<3d}"
              f" {r['alt_ms']:9.1f}   {on}")
    print()
