#!/usr/bin/env python3
"""Paired bracket for the lnaffcast mode, on whichever composition it is measured in.

Takes the results dir and a composition key (`fold`, `krtpkrl`; see COMPOSITIONS) -- the mode is
worth what fc1's NEIGHBOURS make it worth, so the same statistic has to be readable on each graph.

Two instruments, deliberately, because they answer different questions:

  * whole-clip WALL CLOCK -- what a caller feels, but it carries host work and a ~2 s/clip base, so
    a 170 ms effect is ~8% of the statistic and needs the pairing to resolve;
  * the device-BLOCKING ledger (`NPU_DISPATCH_LOG=1`) -- sums blocking time over ~800 dispatches and
    drops host work entirely. It is the quantity the mode's per-clip claims are made in, so it is
    the one that brackets them. One value per (arm, rep): the probe reports the ledger for the LAST
    clip.

TRANSITION COUNTS are printed as an exactness check, not a statistic -- they are counted at every
dispatch, so any spread across reps within one arm means the arm is not deterministic and every ms
below is suspect.

CI is Student-t at 95% on the paired differences. `neg` counts pairs where the arm beat its base; a
mean whose CI excludes 0 but whose sign flips across pairs is drift, not an effect.
"""
import math
import re
import statistics
import sys
from pathlib import Path

# Two COMPOSITIONS, one instrument. The mode's value is a property of what fc1's neighbours are, so
# the same paired statistic has to be read on more than one graph: `fold` is FOLD_FC1 alone (the -120
# transitions), `krtpkrl` adds FOLD_GLU and the krtpkrl resident carrying the one-dispatch fc2, which
# is the composition the 143-transition ledger was quoted from. The krtpkrl set drops the cache
# control -- it PASSED in two independent windows on `fold`, so those reps buy the contrast instead.
COMPOSITIONS = {
    "fold": (
        ["f0", "f0m", "f1", "f1m"],
        {"f0": "fold", "f0m": "fold + LN_MODE",
         "f1": "fold + cache (control)", "f1m": "fold + LN_MODE + cache"},
        (("f0m", "f0", "MODE on fold (uncached)"),
         ("f1m", "f1", "MODE on fold (cached)"),
         ("f1", "f0", "cache CONTROL, no mode"),
         ("f1m", "f0m", "cache CONTROL, mode")),
        (("f1", "f0"), ("f1m", "f0m")),
    ),
    "krtpkrl": (
        ["k0", "k0m"],
        {"k0": "fold+glu+krtpkrl", "k0m": "  + LN_MODE"},
        (("k0m", "k0", "MODE on krtpkrl"),),
        (),
    ),
}
CTX = re.compile(r"hw_contexts:\s*(\d+)/")
ENC = re.compile(r"^mean encode ([0-9.]+)s/clip", re.M)
CLIP = re.compile(r"^\[enc\] (\S+)\s+T'=\d+\s+([0-9.]+)s", re.M)
BLOCK = re.compile(r"total BLOCKING dispatch time ([0-9.]+) s")
DTR = re.compile(r"^dispatches (\d+) \| transitions (\d+)", re.M)

TCRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
         9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
         16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 24: 2.064, 29: 2.045,
         39: 2.023, 49: 2.010, 59: 2.001, 99: 1.984}


def tcrit(df):
    if df <= 0:
        return float("nan")
    for k in sorted(TCRIT):
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


d = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/lnmode_fold_bracket")
comp = sys.argv[2] if len(sys.argv) > 2 else "fold"
ARMS, LABEL, CONTRASTS, CONTROLS = COMPOSITIONS[comp]
reps = {}
for arm in ARMS:
    for f in d.glob(f"{arm}_rep*.txt"):
        rep = int(re.search(r"rep(\d+)", f.name).group(1))
        reps.setdefault(rep, {})[arm] = parse(f)

complete = sorted(r for r, a in reps.items() if all(x in a for x in ARMS))
partial = sorted(set(reps) - set(complete))
print(f"complete reps: {len(complete)}{'  (partial, ignored: ' + str(partial) + ')' if partial else ''}")
if not complete:
    sys.exit(0)

# Exactness check first: a transition count that moves across reps invalidates everything below.
print("\narm                        hw_ctx   transitions  dispatches   mean s/clip        95% CI")
for arm in ARMS:
    encs = [reps[r][arm]["enc"] for r in complete]
    ctxs = sorted({reps[r][arm]["ctx"] for r in complete})
    trs = sorted({reps[r][arm]["trans"] for r in complete})
    dps = sorted({reps[r][arm]["disp"] for r in complete})
    lo, hi = ci(encs)
    flag = "" if len(trs) == 1 else "  <-- NOT DETERMINISTIC"
    print(f"{LABEL[arm]:<26} {str(ctxs):>6}  {str(trs):>12} {str(dps):>11}  "
          f"{statistics.mean(encs):>10.4f}   [{lo:.4f}, {hi:.4f}]{flag}")


def paired_clips(arm, base, label):
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
    print(f"{label:<26} {len(pc):>7}   {statistics.mean(pc):>9.1f}   "
          f"[{lo:>8.1f}, {hi:>8.1f}]  {neg:>3}/{len(pc):<3}  {statistics.mean(rm):>8.1f}")


def paired_block(arm, base, label):
    pb = [(reps[r][arm]["block"] - reps[r][base]["block"]) * 1e3
          for r in complete
          if reps[r][arm]["block"] is not None and reps[r][base]["block"] is not None]
    if len(pb) < 2:
        return
    lo, hi = ci(pb)
    neg = sum(1 for x in pb if x < 0)
    print(f"{label:<26} {len(pb):>7}   {statistics.mean(pb):>9.1f}   "
          f"[{lo:>8.1f}, {hi:>8.1f}]  {neg:>3}/{len(pb):<3}")


print("\nwhole-clip wall clock -- ms/clip, negative = faster")
print(f"{'contrast':<26} {'pairs':>7}   {'mean':>9}   {'95% CI':>20}  {'neg':>7}  {'rep-mean':>8}")
for arm, base, label in CONTRASTS:
    paired_clips(arm, base, label)

print("\ndevice-blocking ledger -- ms/clip, negative = faster  (last clip, NPU_DISPATCH_LOG)")
print("this is the instrument the mode's per-clip claim is made in; it is the one that brackets it.")
print(f"{'contrast':<26} {'pairs':>7}   {'mean':>9}   {'95% CI':>20}  {'neg':>7}")
for arm, base, label in CONTRASTS:
    paired_block(arm, base, label)

# The control has to be read explicitly -- a silent pass is how the duplicate-context artifact
# survived seven refuted axes the first time.
if CONTROLS:
    print("\ncontrol verdict")
for arm, base in CONTROLS:
    at = sorted({reps[r][arm]["trans"] for r in complete})
    bt = sorted({reps[r][base]["trans"] for r in complete})
    ac = sorted({reps[r][arm]["ctx"] for r in complete})
    bc = sorted({reps[r][base]["ctx"] for r in complete})
    ok = at == bt and ac == bc
    print(f"  {LABEL[arm]} vs {LABEL[base]}: transitions {bt} -> {at}, hw_ctx {bc} -> {ac}  "
          f"{'PASS (cache is a no-op; the fold folds on its own)' if ok else 'FAIL (a duplicate context is still live)'}")
