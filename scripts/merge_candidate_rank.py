#!/usr/bin/env python3
"""Rank every remaining context merge on a banked ledger, with the CORRECTED charge.

WHY THIS EXISTS. The previous ranking pass priced a merge as boundaries-deleted x tax and then
withdrew the green light for one-xclbin-per-block on a compounding term: ~96 surviving arrivals into
the destination x the +0.114 ms the destination got dearer, ~11 ms per merge. Both factors were
wrong. The count double-counted the absorbed brick's own survivors (which the saving side already
charges), and the rate came from a bracket with no untouched-xclbin control, where the merged
program's rise is not separable from an arm-wide level shift (DiD +0.018% [-0.488, +0.524]). The
controlled rate, from the bracket that HAS null arms, is 0.0249 ms.

So the destination-side term is ~1% of a merge's saving and does not gate anything. THE GATE IS THE
OTHER TERM: the absorbed brick's own surviving arrivals, which after the merge enter the big program
instead of the brick's cheap one. On the lnaffcast merge that term was 48 x (3.280 - 1.470) =
+86.9 ms, 47% of the gross saving. This script ranks candidates on it.

WHAT IT COMPUTES, per candidate C absorbed into destination D:

    saving  = n(C->D) * reconfig(D)          MEASURED -- D has both columns, so its tax is observed
            + n(D->C) * reconfig(C)          BOUNDED  -- C is never in-context; work >= 0 bounds it
    charge1 = survivors(C) * (reconfig(D) - reconfig(C))    the gate; survivors(C) = edges X->C, X not in {C,D}
    charge2 = survivors(D) * DEST_DEARNESS                  the corrected destination-side term
    net     = saving_measured - charge1 - charge2

Read the MEASURED column and rank on it. The bounded column charges a brick's whole arrival to
reconfiguration, which is near enough for an 812-word brick and vacuous for one doing real work
(relpos is the worked example: the bound hands it ~5 ms that is its own kernel time).

With reconfig(C) known only as a bound, charge1 is a LOWER limit, so net is an UPPER limit -- the
same convention transition_fold_model.py uses. When survivors(C) is 0 the bound does not matter:
charge1 is exactly 0 whatever the rate.

Usage: merge_candidate_rank.py <bracket_dir> <arm> [--dest SUBSTR]
       merge_candidate_rank.py <report.txt> [<report.txt> ...] [--dest SUBSTR]
"""
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from transition_fold_model import load, reconfig_by_program  # noqa: E402

# The destination-side charge: what absorbing a brick adds to the cost of ENTERING the absorbing
# program, per arrival that still crosses into it. 0.0249 ms [0.0161, 0.0336], dearer in 12/12 reps,
# fitted on artifacts/lnmode_fold_bracket -- the only bracket here with null-control arms, which read
# -0.0053 and +0.0034 ms, and whose untouched xclbins show no arm-wide inflation (-0.361%). It
# SUPERSEDES the 0.114 ms from lnmode_krtpkrl_bracket, which has no control group and where the same
# term is not separable from a +5.19% arm-wide shift. See arrival_control_bracket.py.
DEST_DEARNESS_MS = 0.0249


def ci(xs):
    """Mean and a 95% normal interval, matching the bracket harness's own summary."""
    m = statistics.mean(xs)
    if len(xs) < 2:
        return m, m, m
    h = 1.96 * statistics.stdev(xs) / len(xs) ** 0.5
    return m, m - h, m + h


def short(label):
    return label.split("final_")[-1][:38] or "final"


def reconfig_bounds(rep, reconfig, per_stream):
    """reconfig for programs never seen in-context: arrival - work, bounded above by arrival."""
    bound = {}
    for (lab, _), s in per_stream.items():
        if lab not in reconfig and s["xcl"] is not None:
            bound[lab] = min(bound.get(lab, s["xcl"]), s["xcl"])
    return bound


def rank_one(path, dest_frag):
    """Price every merge candidate on ONE report. Returns (dest, {candidate: terms}, meta)."""
    rep = load(path)
    reconfig, per_stream = reconfig_by_program(rep)
    bound = reconfig_bounds(rep, reconfig, per_stream)

    if dest_frag:
        hits = [k for k in rep["per_kernel"] if dest_frag in k]
        if len(hits) != 1:
            sys.exit(f"--dest '{dest_frag}' matched {len(hits)}: {[short(h) for h in hits]}")
        dest = hits[0]
    else:
        # The resident is the program carrying the most dispatches -- every merge absorbs INTO it.
        dest = max(rep["per_kernel"], key=lambda k: rep["per_kernel"][k][0])
    if dest not in reconfig:
        sys.exit(f"destination {short(dest)} is never dispatched in-context, so its tax is not "
                 f"observed and no saving here would be measured")
    r_d = reconfig[dest]

    inbound = {}   # C -> total arrivals into C, by source
    for x, y, n in rep["trans"]:
        inbound.setdefault(y, {})[x] = inbound.setdefault(y, {}).get(x, 0) + n

    cands = {}
    for c in {x for x, y, _ in rep["trans"] if y == dest} | {y for x, y, _ in rep["trans"] if x == dest}:
        if c == dest:
            continue
        n_cd = inbound.get(dest, {}).get(c, 0)
        n_dc = inbound.get(c, {}).get(dest, 0)
        r_c = reconfig.get(c, bound.get(c))
        fitted = c in reconfig
        # THE GATE: arrivals into C from anywhere but C and D still cross after the merge, and now
        # enter D's bigger program instead of C's cheap one.
        surv_c = sum(n for x, n in inbound.get(c, {}).items() if x not in (c, dest))
        surv_d = sum(n for x, n in inbound.get(dest, {}).items() if x not in (c, dest))
        # With reconfig(C) known only as an upper bound B, surv*(r_d - B) is a LOWER limit on the
        # charge. That limit is informative only while it is positive: once B exceeds r_d it goes
        # negative and would read as a CREDIT for absorbing the brick, which the bound cannot
        # establish -- the true reconfig(C) may sit anywhere below B, on either side of r_d. Report
        # it as indeterminate rather than bank a saving on the sign of a bound.
        if surv_c == 0:
            charge1, kind = 0.0, "exact"      # no survivors: zero whatever the rate
        elif r_c is None:
            charge1, kind = None, "unknown"
        elif fitted:
            charge1, kind = surv_c * (r_d - r_c), "fitted"
        else:
            lower = surv_c * (r_d - r_c)
            charge1, kind = (lower, "bounded") if lower > 0 else (None, "indeterminate")
        cands[c] = {
            "n_cd": n_cd, "n_dc": n_dc,
            "saving_meas": n_cd * r_d,
            "saving_bound": n_dc * r_c if r_c is not None else None,
            "surv_c": surv_c, "surv_d": surv_d,
            "charge1": charge1, "charge1_kind": kind,
            "charge2": surv_d * DEST_DEARNESS_MS,
            "r_c": r_c, "r_c_fitted": fitted,
        }

    # A SWEEP -- absorb every candidate at once -- is only well defined when the candidates do not
    # talk to each other, i.e. the transition graph is a star centred on the destination. Then every
    # boundary in the report is a D<->C boundary and the merged program has no arrivals left at all.
    cross = [(x, y, n) for x, y, n in rep["trans"] if x != dest and y != dest]
    sweep = None
    if not cross:
        n_in = sum(v["n_cd"] for v in cands.values())
        n_out = sum(v["n_dc"] for v in cands.values())
        sweep = {
            "n_cd": n_in, "n_dc": n_out,
            "saving_meas": n_in * r_d,
            "saving_bound": sum(v["saving_bound"] for v in cands.values()
                                if v["saving_bound"] is not None),
            "surv_c": 0, "surv_d": 0, "charge1": 0.0, "charge2": 0.0,
            "r_c": None, "r_c_fitted": False,
        }
    meta = {"dest": dest, "r_d": r_d, "cross": cross,
            "dispatches": rep["dispatches"], "transitions": rep["transitions"],
            "blocking_ms": rep["blocking_s"] * 1e3}
    return dest, cands, sweep, meta


def main():
    args = [a for a in sys.argv[1:]]
    dest_frag = None
    if "--dest" in args:
        i = args.index("--dest")
        dest_frag = args[i + 1]
        del args[i:i + 2]
    if not args:
        sys.exit(__doc__.strip().splitlines()[-1])

    first = Path(args[0])
    if first.is_dir():
        arm = args[1] if len(args) > 1 else sys.exit("give an arm, e.g. k0m")
        paths = sorted(first.glob(f"{arm}_rep*.txt"),
                       key=lambda p: int(re.search(r"rep(\d+)", p.name).group(1)))
        if not paths:
            sys.exit(f"no {arm}_rep*.txt under {first}")
        title = f"{first}  arm {arm}"
    else:
        paths = [Path(a) for a in args]
        title = " ".join(p.name for p in paths)

    runs = [rank_one(p, dest_frag) for p in paths]
    dest, _, _, meta0 = runs[0]
    if any(r[0] != dest for r in runs):
        sys.exit("reports disagree on the destination program; pass --dest")

    print(f"{title}  n={len(paths)} report(s)  (no device time -- re-read of banked ledgers)")
    print(f"  destination: {short(dest)}")
    print(f"  {meta0['dispatches']} dispatches, {meta0['transitions']} transitions, "
          f"{meta0['blocking_ms']:.0f} ms blocking (rep 1)")
    m, lo, hi = ci([r[3]["r_d"] for r in runs])
    print(f"  reconfig(destination) {m:.3f} [{lo:.3f}, {hi:.3f}] ms  -- the MEASURED tax every "
          f"C -> D boundary pays")
    if meta0["cross"]:
        print(f"  graph is NOT a star: {len(meta0['cross'])} candidate-to-candidate edge(s); "
              f"no sweep row, and merges interact")
    else:
        print(f"  graph IS a star on the destination: no candidate-to-candidate edge, so every "
              f"candidate's\n  surviving arrivals are 0 by construction and the merges are "
              f"independent")

    names = sorted(set().union(*[set(r[1]) for r in runs]),
                   key=lambda c: -statistics.mean([r[1][c]["saving_meas"] for r in runs if c in r[1]]))

    print(f"\n  {'candidate':<40} {'in':>4} {'out':>4} {'surv':>5}  {'MEASURED saving':>24}  "
          f"{'charge1':>9} {'charge2':>8}  {'NET (measured)':>22}  {'+departing BOUND':>16}")
    FLAG = {"exact": " ", "fitted": " ", "bounded": "~", "indeterminate": "?", "unknown": "?"}
    for c in names:
        rows = [r[1][c] for r in runs if c in r[1]]
        sm, slo, shi = ci([x["saving_meas"] for x in rows])
        c2 = statistics.mean([x["charge2"] for x in rows])
        kind = rows[0]["charge1_kind"]
        known = [x["charge1"] for x in rows if x["charge1"] is not None]
        c1s = f"{statistics.mean(known):>8.1f}" if known else " " * 8
        bnd = [x["saving_bound"] for x in rows if x["saving_bound"] is not None]
        bs = f"{statistics.mean(bnd):>15.1f}" if bnd else f"{'-':>15}"
        if known:
            nm, nlo, nhi = ci([x["saving_meas"] - x["charge1"] - x["charge2"] for x in rows])
            nets = f"{nm:>9.1f} [{nlo:6.1f},{nhi:7.1f}]"
        else:
            nets = f"{'not determined':>24}"
        print(f"  {short(c):<40} {rows[0]['n_cd']:>4} {rows[0]['n_dc']:>4} {rows[0]['surv_c']:>5}  "
              f"{sm:>10.1f} [{slo:6.1f},{shi:7.1f}]  {c1s}{FLAG[kind]} {c2:>8.2f}  "
              f"{nets}  {bs}")

    if runs[0][2] is not None:
        rows = [r[2] for r in runs]
        sm, slo, shi = ci([x["saving_meas"] for x in rows])
        bm = statistics.mean([x["saving_bound"] for x in rows])
        blk = statistics.mean([r[3]["blocking_ms"] for r in runs])
        print(f"  {'-' * 118}")
        print(f"  {'SWEEP (absorb every candidate)':<40} {rows[0]['n_cd']:>4} {rows[0]['n_dc']:>4} "
              f"{0:>5}  {sm:>10.1f} [{slo:6.1f},{shi:7.1f}]  {0.0:>8.1f} {0.0:>8.2f}  "
              f"{sm:>9.1f} [{slo:6.1f},{shi:7.1f}]")
        print(f"    the sweep leaves ONE program and zero transitions. Measured column is "
              f"{sm / blk * 100:.0f}% of the\n    arm's {blk:.0f} ms blocking; the departing side "
              f"adds up to {bm:.1f} ms more as an upper BOUND.")

    print(f"\n  surv = the absorbed brick's OWN surviving arrivals -- the term that gates a merge "
          f"(48 of them\n  cost +86.9 ms on the lnaffcast merge, 47% of its gross saving). charge2 "
          f"= surviving arrivals into\n  the destination x {DEST_DEARNESS_MS} ms, the controlled "
          f"rate. '~' marks a charge1 computed from a\n  BOUNDED reconfig(C), which makes it a lower "
          f"limit and the net an upper limit; '?' marks one that the bound cannot even sign, because "
          f"the\n  bound on reconfig(C) exceeds reconfig(D) -- no net is quoted there rather than "
          f"bank a credit on it.\n  The departing column (D -> C boundaries) is an "
          f"upper BOUND, not a saving: it charges C's whole\n  arrival to reconfiguration, which is "
          f"fair for an 812-word brick and vacuous for one doing real work.")


if __name__ == "__main__":
    main()
