#!/usr/bin/env python3
"""Paired per-clip wall clock + dispatch ledger for the one-dispatch fc2 A/B.

Pairing unit is (rep, clip), not the rep mean: the probe times each clip, the arms interleave inside
a rep, and clip length varies by 2x across the set -- so pairing per clip removes both box drift and
the clip-length term that a rep mean would leave in.

Reports the warm column separately because `mean encode` is `(one-time + n x per-clip)/n`: a delta
living in process setup lands entirely on clip 1 and is then divided by n. At 17 clips pooled and
warm should agree closely; a gap between them means the effect is in setup, not per-clip.

Usage: krtp_onedispatch_stats.py <dir>
"""
import re
import statistics
import sys
from pathlib import Path

ARMS = ["k0", "k1"]
LABEL = {"k0": "K-split fc2 (default)", "k1": "one-dispatch fc2 (krtp)"}

CLIP = re.compile(r"^\[enc\]\s+(\S+)\s+T'=\d+\s+([\d.]+)s", re.M)
MEAN = re.compile(r"^mean encode ([\d.]+)s/clip over (\d+) clips", re.M)
TOTALS = re.compile(r"^dispatches (\d+) \| transitions (\d+)", re.M)
BLOCK = re.compile(r"total BLOCKING dispatch time ([\d.]+) s")
CTX = re.compile(r"hw_contexts: (\d+)/(\d+)")
PERKERNEL = re.compile(r"^\s{4}(\S+)\s+x(\d+)\s+([\d.]+)s\s+([\d.]+) ms$", re.M)


def load(path):
    txt = path.read_text()
    m = MEAN.search(txt)
    if not m:
        return None
    tot = TOTALS.search(txt)
    blk = BLOCK.search(txt)
    ctx = CTX.search(txt)
    return {
        "clips": {c: float(t) for c, t in CLIP.findall(txt)},
        "mean": float(m.group(1)),
        "n_clips": int(m.group(2)),
        "dispatches": int(tot.group(1)) if tot else None,
        "transitions": int(tot.group(2)) if tot else None,
        "blocking_s": float(blk.group(1)) if blk else None,
        "ctx": (int(ctx.group(1)), int(ctx.group(2))) if ctx else None,
        "per_kernel": {k: (int(n), float(s), float(ms))
                       for k, n, s, ms in PERKERNEL.findall(txt)},
    }


def ci95(xs):
    """Mean and a normal-approx 95% CI. n is per-clip pairs, so it is comfortably large."""
    n = len(xs)
    mu = statistics.fmean(xs)
    if n < 2:
        return mu, mu, mu
    half = 1.96 * statistics.stdev(xs) / (n ** 0.5)
    return mu, mu - half, mu + half


def report_delta(name, pairs):
    if not pairs:
        print(f"  {name:<28} no pairs")
        return
    ms = [1000.0 * d for d in pairs]
    mu, lo, hi = ci95(ms)
    agree = sum(1 for x in ms if (x < 0) == (mu < 0))
    print(f"  {name:<28} {mu:+8.1f} ms/clip  CI [{lo:+8.1f}, {hi:+8.1f}]  {agree}/{len(ms)}")


d = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/krtp_onedispatch")
reps = {}
for arm in ARMS:
    for f in d.glob(f"{arm}_rep*.txt"):
        rep = int(re.search(r"_rep(\d+)\.txt$", f.name).group(1))
        r = load(f)
        if r:
            reps.setdefault(rep, {})[arm] = r

complete = sorted(r for r, a in reps.items() if all(x in a for x in ARMS))
if not complete:
    sys.exit("no complete reps")
print(f"complete reps: {complete}  ({len(complete)} x {reps[complete[0]]['k0']['n_clips']} clips)\n")

print("per-arm mean encode (s/clip), by rep")
for rep in complete:
    row = "  ".join(f"{a}={reps[rep][a]['mean']:.4f}" for a in ARMS)
    print(f"  rep{rep}  {row}")

print("\nstructural counts (identical every rep, or the arms are not comparable)")
for arm in ARMS:
    s = {(reps[r][arm]["dispatches"], reps[r][arm]["transitions"]) for r in complete}
    ctxs = {reps[r][arm]["ctx"] for r in complete}
    blk = statistics.fmean([reps[r][arm]["blocking_s"] for r in complete
                            if reps[r][arm]["blocking_s"] is not None] or [float("nan")])
    print(f"  {arm} {LABEL[arm]:<26} dispatches/transitions {sorted(s)}  "
          f"hw_contexts {sorted(ctxs)}  blocking {blk:.3f} s")

k0d = reps[complete[0]]["k0"]["dispatches"]
k1d = reps[complete[0]]["k1"]["dispatches"]
if k0d and k1d:
    n = reps[complete[0]]["k0"]["n_clips"]
    print(f"  => dispatches/clip {k0d / n:.1f} -> {k1d / n:.1f}  "
          f"({(k1d - k0d) / n:+.1f}/clip, {k1d - k0d:+d} over {n} clips)")

print("\nkernels present in k0 but NOT in k1 (the collapse should delete accadd outright)")
only0 = set(reps[complete[0]]["k0"]["per_kernel"]) - set(reps[complete[0]]["k1"]["per_kernel"])
for k in sorted(only0):
    n, s, ms = reps[complete[0]]["k0"]["per_kernel"][k]
    print(f"  {k:<52} x{n:<5} {s:.3f}s  {ms:.3f} ms")
if not only0:
    print("  (none)")

print("\npaired per-(rep, clip) delta, k1 - k0   [negative = one-dispatch is faster]")
pooled, warm = [], []
clip_order = sorted(reps[complete[0]]["k0"]["clips"])
for rep in complete:
    for ci, c in enumerate(clip_order):
        a, b = reps[rep]["k1"]["clips"].get(c), reps[rep]["k0"]["clips"].get(c)
        if a is None or b is None:
            continue
        pooled.append(a - b)
        if ci > 0:
            warm.append(a - b)
report_delta("pooled (all clips)", pooled)
report_delta("warm (clip 1 dropped)", warm)

print("\ndevice BLOCKING dispatch time, paired per rep [the non-wall-clock instrument]")
blk = [reps[r]["k1"]["blocking_s"] - reps[r]["k0"]["blocking_s"] for r in complete
       if reps[r]["k1"]["blocking_s"] is not None and reps[r]["k0"]["blocking_s"] is not None]
if blk:
    n = reps[complete[0]]["k0"]["n_clips"]
    report_delta("blocking (last clip only)", blk)
    print(f"  NOTE: the ledger covers the LAST clip only, so n is {len(blk)} single-clip observations.")
