#!/usr/bin/env python3
"""Paired per-clip wall clock for the six fold/cache arms.

Pairing unit is (rep, clip), not the rep mean: the probe times each clip, the arms interleave inside
a rep, and clip 1 carries the cold weight-BO load in EVERY arm, so pairing on the clip index keeps
that cost in both sides of the difference instead of averaging it into one arm's mean. The rep-mean
delta is printed too, as the coarser control -- if the two disagree, the per-clip one is the drift-
free reading and the disagreement is the finding.

CI is Student-t at 95% on the paired differences. `neg` counts pairs where the arm beat its base;
a mean whose CI excludes 0 but whose sign flips across pairs is drift, not an effect.
"""
import math
import re
import statistics
import sys
from pathlib import Path

ARMS = ["b0", "b1", "f0", "f1", "p0", "p1"]
LABEL = {
    "b0": "default",
    "b1": "default + cache",
    "f0": "FOLD_FC1",
    "f1": "FOLD_FC1 + cache",
    "p0": "FOLD_FC1+FOLD_GLU",
    "p1": "FOLD_FC1+FOLD_GLU + cache",
}
CTX = re.compile(r"hw_contexts:\s*(\d+)/")
ENC = re.compile(r"^mean encode ([0-9.]+)s/clip", re.M)
CLIP = re.compile(r"^\[enc\] (\S+)\s+T'=\d+\s+([0-9.]+)s", re.M)
DISP = re.compile(r"modal dispatch sites?[^0-9]*([0-9]+)", re.I)
BLOCK = re.compile(r"total BLOCKING dispatch time ([0-9.]+) s")
DTR = re.compile(r"^dispatches (\d+) \| transitions (\d+)", re.M)

# t(0.975, df) for the df we actually reach; falls back to the normal quantile past the table.
TCRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
         9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
         16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 24: 2.064, 29: 2.045,
         39: 2.023, 49: 2.010, 59: 2.001, 99: 1.984}


def tcrit(df):
    if df <= 0:
        return float("nan")
    keys = sorted(TCRIT)
    for k in keys:
        if df <= k:
            return TCRIT[k]
    return 1.960


def ci(xs):
    if len(xs) < 2:
        return (float("nan"), float("nan"))
    m, sd = statistics.mean(xs), statistics.stdev(xs)
    h = tcrit(len(xs) - 1) * sd / math.sqrt(len(xs))
    return (m - h, m + h)


def parse(p):
    txt = p.read_text()
    dtr = DTR.search(txt)
    return {
        "ctx": int(CTX.search(txt).group(1)) if CTX.search(txt) else None,
        "enc": float(ENC.search(txt).group(1)) if ENC.search(txt) else None,
        "clips": {name: float(t) for name, t in CLIP.findall(txt)},
        "block": float(BLOCK.search(txt).group(1)) if BLOCK.search(txt) else None,
        "disp": int(dtr.group(1)) if dtr else None,
        "trans": int(dtr.group(2)) if dtr else None,
    }


d = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/fold_pair_wallclock")
reps = {}
for arm in ARMS:
    for f in d.glob(f"{arm}_rep*.txt"):
        rep = int(re.search(r"rep(\d+)", f.name).group(1))
        reps.setdefault(rep, {})[arm] = parse(f)

# Only reps where EVERY arm landed -- a partial rep would pair an arm against a different box state.
complete = sorted(r for r, a in reps.items() if all(x in a for x in ARMS))
partial = sorted(set(reps) - set(complete))
print(f"complete reps: {len(complete)}{'  (partial, ignored: ' + str(partial) + ')' if partial else ''}")
if not complete:
    sys.exit(0)

print("\narm                            hw_ctx   mean s/clip        95% CI")
for arm in ARMS:
    encs = [reps[r][arm]["enc"] for r in complete]
    ctxs = sorted({reps[r][arm]["ctx"] for r in complete})
    lo, hi = ci(encs)
    print(f"{LABEL[arm]:<28} {str(ctxs):>7}  {statistics.mean(encs):>10.4f}   [{lo:.4f}, {hi:.4f}]")

def paired(arm, base, label):
    pc = []
    for r in complete:
        a, b = reps[r][arm]["clips"], reps[r][base]["clips"]
        for name in sorted(set(a) & set(b)):
            pc.append((a[name] - b[name]) * 1e3)
    if len(pc) < 2:
        return
    rm = [(reps[r][arm]["enc"] - reps[r][base]["enc"]) * 1e3 for r in complete]
    lo, hi = ci(pc)
    neg = sum(1 for x in pc if x < 0)
    print(f"{label:<28} {len(pc):>10}   {statistics.mean(pc):>9.1f}   "
          f"[{lo:>8.1f}, {hi:>8.1f}]  {neg:>3}/{len(pc):<3}  {statistics.mean(rm):>8.1f}")


HDR = "arm                            per-clip pairs   mean      95% CI                neg    rep-mean"
for base in ("b0", "b1"):
    print(f"\npaired vs {LABEL[base]} -- ms/clip, negative = faster")
    print(HDR)
    for arm in ARMS:
        if arm != base:
            paired(arm, base, LABEL[arm])

# One factor at a time. Against a default base both factors move together, which is what confounded
# the fold with the context cache in the first place; each row here holds the other factor fixed.
print("\nsingle-factor contrasts -- ms/clip, negative = faster")
print(HDR)
for arm, base, label in (
    ("b1", "b0", "cache, on default"),
    ("f1", "f0", "cache, on FOLD_FC1"),
    ("p1", "p0", "cache, on the pair"),
    ("p0", "f0", "GLU fold, uncached"),
    ("p1", "f1", "GLU fold, cached"),
    ("f0", "b0", "FOLD_FC1, uncached"),
):
    paired(arm, base, label)


# Clips-per-process decomposition. The pairing above holds the cold clip in BOTH arms, which removes
# it as a bias but not as a TERM: a delta that lives in one-time setup (a context the arm does not
# create, an xclbin it does not load) still lands entirely on clip 1, so the per-clip mean carries it
# divided by CLIPS. That makes the headline a function of how many clips the probe ran, not a
# property of the encoder -- and the shipped banks run 17 clips, not 3. Split it: `warm` is the
# steady-state per-clip effect that transfers, `one-time` is (cold - warm), and the projection is
# one-time/n + warm. Report the warm column as the number; it is also the tighter one, because all of
# the cold clip's variance stays out of it.
def decompose(arm, base, label, clip_order):
    warm, cold = [], []
    for r in complete:
        a, b = reps[r][arm]["clips"], reps[r][base]["clips"]
        for name in clip_order:
            if name in a and name in b:
                (cold if name == clip_order[0] else warm).append((a[name] - b[name]) * 1e3)
    if len(warm) < 2 or len(cold) < 2:
        return
    w, c = statistics.mean(warm), statistics.mean(cold)
    one = c - w
    lo, hi = ci(warm)
    neg = sum(1 for x in warm if x < 0)
    print(f"{label:<28} {w:>8.1f} [{lo:>7.1f},{hi:>7.1f}] {neg:>3}/{len(warm):<3} "
          f"{one:>9.1f} {one / 3 + w:>8.1f} {one / 17 + w:>8.1f}")


clip_order = sorted(reps[complete[0]]["b0"]["clips"])
if len(clip_order) > 1:
    print(f"\nclips-per-process decomposition -- ms/clip, cold clip = {clip_order[0]}")
    print(f"{'contrast':<28} {'warm':>8} {'95% CI':>17} {'neg':>7} {'one-time':>9} "
          f"{'n=3':>8} {'n=17':>8}")
    for arm, base, label in (
        ("f0", "b0", "FOLD_FC1"),
        ("p0", "b0", "FOLD_FC1+FOLD_GLU"),
        ("p0", "f0", "GLU fold, uncached"),
        ("b1", "b0", "cache, on default"),
    ):
        decompose(arm, base, label, clip_order)


# Device-blocking ledger (`NPU_DISPATCH_LOG=1`), the instrument that is NOT whole-clip wall clock.
#
# Why it is here: whole-clip wall clock could never resolve the GLU half. Three runs put it at -99.3,
# -17.3 and -8.5 ms/clip, the last at 43/64 sign consistency -- drift-grade, because a ~23 ms effect
# is 1.3% of a 1.72 s/clip statistic. The ledger sums blocking time over ~800 dispatches and drops
# host work entirely, so the same effect lands at 8/8 sign consistency with a +-5 ms CI.
#
# It is a DIFFERENT quantity from wall clock (device blocking only), so it earns trust by reproducing
# the two contrasts wall clock DID resolve, on the same runs -- see the additivity line below.
#
# One value per (arm, rep): the probe reports the ledger for the LAST clip only, which is a warm clip.
def ledger_pairs(arm, base):
    return [(reps[r][arm]["block"] - reps[r][base]["block"]) * 1e3
            for r in complete
            if reps[r][arm]["block"] is not None and reps[r][base]["block"] is not None]


def ledger_row(label, pairs):
    if len(pairs) < 2:
        return None
    lo, hi = ci(pairs)
    neg = sum(1 for x in pairs if x < 0)
    print(f"{label:<28} {statistics.mean(pairs):>9.1f}   [{lo:>8.1f}, {hi:>8.1f}]  "
          f"{neg:>3}/{len(pairs):<3}")
    return statistics.mean(pairs)


if all(reps[complete[0]][a]["block"] is not None for a in ARMS):
    print("\ndevice-blocking ledger -- ms/clip, negative = faster  (last clip, NPU_DISPATCH_LOG)")
    print(f"{'contrast':<28} {'mean':>9}   {'95% CI':>20}  {'neg':>7}")
    for arm, base, label in (
        ("f0", "b0", "FOLD_FC1, uncached"),
        ("f1", "b1", "FOLD_FC1, cached"),
        ("p0", "f0", "GLU fold, uncached"),
        ("p1", "f1", "GLU fold, cached"),
        ("p0", "b0", "the pair, uncached"),
        ("p1", "b1", "the pair, cached"),
    ):
        ledger_row(label, ledger_pairs(arm, base))

    # Pooled across the cache setting, which the ledger itself shows is inert (the two cache rows
    # above differ by a few ms with CIs containing zero), so the 4+4 reps are 8 replicates of one
    # contrast rather than two contrasts.
    print("\npooled across the cache setting")
    print(f"{'contrast':<28} {'mean':>9}   {'95% CI':>20}  {'neg':>7}")
    fc1 = ledger_row("FOLD_FC1", ledger_pairs("f0", "b0") + ledger_pairs("f1", "b1"))
    glu = ledger_row("GLU fold", ledger_pairs("p0", "f0") + ledger_pairs("p1", "f1"))
    pair = ledger_row("the pair", ledger_pairs("p0", "b0") + ledger_pairs("p1", "b1"))
    if None not in (fc1, glu, pair):
        # The halves add on this instrument. They never did on wall clock, where the GLU half was
        # unresolvable and the residual absorbed it.
        print(f"\n  additivity: {fc1:.1f} + {glu:.1f} = {fc1 + glu:.1f} vs measured pair "
              f"{pair:.1f}  (residual {pair - fc1 - glu:+.2f} ms)")

    print("\ndispatch / transition counts per arm (identical across reps -- they are structural)")
    for arm in ARMS:
        a = reps[complete[0]][arm]
        print(f"  {LABEL[arm]:<28} dispatches {a['disp']}  transitions {a['trans']}")
