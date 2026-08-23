#!/usr/bin/env python3
"""Price a proposed context merge from a banked `NPU_DISPATCH_LOG=1` report.

The question this answers is "what is the next fold worth?", and the reason it exists is that the
obvious way to answer it -- transitions deleted x a fleet-average switch cost -- is what produced the
lnaffcast estimate the fold task was told not to quote. The fleet average (1.543 ms, from the
whole-collapse REPLAY/GROUPED pair) is an average over 743 boundaries that differ by 2x, so it is the
wrong rate for any particular one.

The right rate is measurable and is already in the report. One xclbin stream in the encoder is
dispatched BOTH after a switch and in-context (the modal GEMM, which follows itself inside an FFN),
so its two columns are the same work under the two conditions and their difference is the switch tax
with the work divided out -- a within-stream control, not a cross-kernel comparison.

CALIBRATION, and why the model is trusted here at all: the same arithmetic applied to the two folds
that WERE measured on this ledger predicts them to ~1%.

  FOLD_FC1   48 boundaries x 2.263 ms tax  = 108.6 predicted   vs -109.6 measured
  FOLD_GLU   24 deleted dispatches x 0.945 =  22.7 predicted   vs  -23.0 measured

A merge also CHARGES, and leaving that term out is a 43% error. Every arrival into the absorbed
brick that still crosses afterwards now enters the bigger program and pays its reconfiguration
rate instead of the small one's. On the lnaffcast merge that is 48 arrivals x +1.693 ms = +81.3 ms
against 299.1 ms of saving; the saving-only projection reads 310.3 where the ledger pair moves
217.0. See `survivor_charge`.

SCOPE, now measured over the bracket rather than guessed (scripts/arrival_refit_bracket.py, 11
reps, no new device time): the model is SYSTEMATICALLY OPTIMISTIC by 21.8 ms [10.6, 33.1] on this
merge -- -204.9 [-214.0, -195.8] modelled against -183.1 [-203.1, -163.1] measured, inside 10% in
only 3/11 reps. It gets the sign, the transition count (-143 exact) and the magnitude right, so use
it to RANK candidate merges, and discount ~12% before quoting one as a saving. The single pass it
was first fitted on closed to 0.4%, which was luck: that pass moved -217.0 and sits outside the
bracket entirely.

Usage: transition_fold_model.py <report.txt> [--merge A,B] [--delete K]
  --merge   two kernel labels (substring match) whose contexts become one
  --delete  a kernel label whose dispatches disappear entirely
Defaults to modelling the lnaffcast merge, which is the fold task's named next candidate.
"""
import re
import sys
from pathlib import Path

PERKERNEL = re.compile(r"^\s{4}(\S+)\s+x(\d+)\s+([\d.]+)s\s+([\d.]+) ms$", re.M)
BYPRED = re.compile(
    r"^\s{4}(\S+?)#(\d+)@\S+\s+(\d+)\s+(?:x(\d+)\s+([\d.]+)|-)\s+(?:x(\d+)\s+([\d.]+)|-)\s+"
    r"(?:x(\d+)\s+([\d.]+)|-)\s*$", re.M)
TRANS = re.compile(r"^\s{4}(\S+)\s+->\s+(\S+)\s+x(\d+)$", re.M)
TOTALS = re.compile(r"^dispatches (\d+) \| transitions (\d+)", re.M)
BLOCK = re.compile(r"total BLOCKING dispatch time ([\d.]+) s")


def load(path):
    txt = Path(path).read_text()
    body = txt[txt.index("dispatch transitions"):]
    m = TOTALS.search(body)
    return {
        "dispatches": int(m.group(1)),
        "transitions": int(m.group(2)),
        "blocking_s": float(BLOCK.search(body).group(1)),
        "per_kernel": {k: (int(n), float(tot), float(mean))
                       for k, n, tot, mean in PERKERNEL.findall(body)},
        "by_pred": BYPRED.findall(body),
        "trans": [(a, b, int(n)) for a, b, n in TRANS.findall(body)],
    }


def switch_tax(rep):
    """The within-stream switch tax: same stream dispatched in-context vs after a foreign xclbin.

    Only streams observed under BOTH conditions qualify -- everything else in this encoder is always
    preceded by a switch, so its two columns do not exist and any tax read off it would be a
    cross-kernel difference in WORK, not in reconfiguration.
    """
    out = []
    for label, ordn, insts, n_same, t_same, _n_re, _t_re, n_xcl, t_xcl in rep["by_pred"]:
        if n_same and n_xcl:
            out.append((label, int(n_same), float(t_same), int(n_xcl), float(t_xcl),
                        float(t_xcl) - float(t_same)))
    return out


def find(rep, frag):
    hits = [k for k in rep["per_kernel"] if frag in k]
    if len(hits) != 1:
        sys.exit(f"'{frag}' matched {len(hits)} kernels: {hits}")
    return hits[0]


def reconfig_by_program(rep):
    """Fit ONE reconfiguration cost per destination xclbin, and each stream's own work.

    This replaces the `bounded` fallback below, which charged a stream's whole dispatch minus a
    global floor to reconfiguration and was vacuous for anything doing real work. The reason a
    single per-program number is enough: within one xclbin, arrival cost decomposes as

        arrival(stream) = reconfig(xclbin) + work(stream)

    and on the fold+glu+krtpkrl pair that holds to a max residual of 0.037 ms over five streams
    whose arrivals span 3.02-4.08 ms. So a stream seen ONLY after a switch still yields its work
    exactly, as arrival - reconfig, instead of a bound nobody should believe.

    reconfig is NOT one constant across programs, which is the whole reason the flat per-transition
    figure misprices a specific boundary: the krtpkrl panel charges 2.677 ms to enter, the fold
    arm's panel 2.222-2.295, and the 812-word single-brick xclbins arrive in 0.80-1.72 ms TOTAL, so
    theirs is at most that. Fit it per report; do not carry a rate across compositions.
    """
    per_stream, by_prog = {}, {}
    for label, ordn, insts, n_same, t_same, n_re, t_re, n_xcl, t_xcl in rep["by_pred"]:
        # Same-stream is the cleanest in-context column; another stream on the same xclbin is the
        # same reconfiguration state, so it serves when the stream never follows itself.
        incontext = float(t_same) if n_same else (float(t_re) if n_re else None)
        xcl = float(t_xcl) if n_xcl else None
        per_stream[(label, int(ordn))] = {
            "insts": int(insts), "incontext": incontext, "xcl": xcl,
            "n_incontext": int(n_same or 0) + int(n_re or 0), "n_xcl": int(n_xcl or 0),
        }
        if incontext is not None and xcl is not None:
            by_prog.setdefault(label, []).append(xcl - incontext)
    reconfig = {k: sum(v) / len(v) for k, v in by_prog.items()}
    for (label, ordn), s in per_stream.items():
        r = reconfig.get(label)
        if s["incontext"] is not None:
            s["work"], s["work_kind"] = s["incontext"], "measured"
        elif r is not None and s["xcl"] is not None:
            s["work"], s["work_kind"] = s["xcl"] - r, "fitted"
        else:
            s["work"], s["work_kind"] = None, "unknown"
    return reconfig, per_stream


def model_merge(rep, a, b, tax, floor):
    """Contexts of `a` and `b` become one: every a<->b boundary stops being a transition.

    Returns rows tagged `measured` or `bounded`, and the two must be read differently. The saving
    lands on the dispatch that FOLLOWS the deleted boundary, so where that dispatch's stream has an
    observed tax the row is MEASURED. Where it does not, the row falls back to
    (mean after-switch - in-context floor), which is only an upper BOUND -- it charges the whole
    dispatch minus the floor to reconfiguration, which is near enough for a small brick whose cost is
    mostly reconfiguration and badly wrong for a kernel that does real work. `final~ctx3` is the
    worked example: the bound hands it 7.593 of its 7.912 ms, which no one should believe.
    """
    killed = [(x, y, n) for x, y, n in rep["trans"] if {x, y} == {a, b}]
    rows = []
    for x, y, n in killed:
        if y in tax:
            t, kind = tax[y], "measured"
        else:
            t, kind = max(0.0, rep["per_kernel"][y][2] - floor), "bounded"
        rows.append((f"{x[-34:]} -> {y[-34:]}", n, t, n * t, kind))
    return killed, sum(r[3] for r in rows), rows


def survivor_charge(rep, a, b, reconfig, bound):
    """What the merge COSTS: arrivals into `a` from elsewhere now enter the bigger program.

    Deleting boundaries is only half a merge. Every arrival into `a` whose source is neither `a` nor
    `b` still crosses afterwards, but its destination is now inside `b`, so it pays reconfig(b)
    rather than reconfig(a). Omitting this is what made the projection for the lnaffcast merge read
    310.3 ms against a measured 217.0: 48 surviving arrivals were charged nothing when the ledger
    charges them +1.693 ms each.

    reconfig(b) is taken as the merged program's rate BEFORE the merge, which understates it: over
    11 reps the panel's fit moves 2.651 [2.610, 2.693] -> 2.765 [2.724, 2.807] after absorbing
    lnaffcast, +0.114 ms [0.060, 0.168], dearer in 10/11 reps. A single pass reads that delta as
    +0.014 and hides it. So the absorbing program does get dearer to enter, and this term is a
    second, smaller charge the model does not yet carry -- see scripts/arrival_refit_bracket.py.
    """
    r_b = reconfig.get(b)
    r_a = reconfig.get(a, bound.get(a))
    if r_b is None or r_a is None:
        return None, []
    rows = []
    for x, y, n in rep["trans"]:
        if y == a and x not in (a, b):
            rows.append((f"{x[-34:]} -> {y[-34:]}", n, r_b - r_a, n * (r_b - r_a)))
    return sum(r[3] for r in rows), rows


def main():
    args = sys.argv[1:]
    if not args:
        sys.exit(__doc__)
    rep = load(args[0])
    merge = delete = None
    if "--merge" in args:
        merge = args[args.index("--merge") + 1].split(",")
    if "--delete" in args:
        delete = args[args.index("--delete") + 1]
    if not merge and not delete:
        merge = ["lnaffcast", "modal"]

    print(f"report: {args[0]}")
    print(f"  {rep['dispatches']} dispatches, {rep['transitions']} transitions, "
          f"{rep['blocking_s']:.3f} s blocking (last clip)")

    print("\nwithin-stream switch tax (the only streams seen under BOTH conditions)")
    taxes = switch_tax(rep)
    if not taxes:
        sys.exit("  none -- this report cannot calibrate a tax; use a report with an in-context stream")
    tax = {}
    for label, ns, ts, nx, tx, d in taxes:
        print(f"  {label[-46:]:<46} in-context x{ns:<4}{ts:.3f} ms   after-switch x{nx:<4}{tx:.3f} ms"
              f"   tax {d:.3f} ms")
        tax[label] = d
    floor = min(t for _, _, t, _, _, _ in taxes)
    tax_mean = sum(d for *_, d in taxes) / len(taxes)
    print(f"  in-context floor {floor:.3f} ms; tax used for unobserved streams = mean - floor bound")

    reconfig, per_stream = reconfig_by_program(rep)
    print("\nreconfiguration cost per destination PROGRAM (arrival = reconfig + the stream's work)")
    for prog, r in sorted(reconfig.items(), key=lambda kv: -kv[1]):
        n = sum(1 for (lab, _), s in per_stream.items()
                if lab == prog and s["incontext"] is not None and s["xcl"] is not None)
        print(f"  {prog[-56:]:<56} reconfig {r:.3f} ms  (fit on {n} stream(s))")
    unfit = sorted({lab for (lab, _), s in per_stream.items()
                    if lab not in reconfig and s["xcl"] is not None})
    for prog in unfit:
        arr = min(s["xcl"] for (lab, _), s in per_stream.items()
                  if lab == prog and s["xcl"] is not None)
        print(f"  {prog[-56:]:<56} reconfig <= {arr:.3f} ms  (never in-context; work >= 0 bounds it)")
    print("  residuals of arrival - (reconfig + work), streams with BOTH columns:")
    for (lab, ordn), s in sorted(per_stream.items()):
        if s["incontext"] is None or s["xcl"] is None:
            continue
        resid = s["xcl"] - (reconfig[lab] + s["incontext"])
        print(f"    {lab[-46:]:<46}#{ordn} insts {s['insts']:<5} resid {resid:+.3f} ms")

    if delete:
        k = find(rep, delete)
        n, tot, mean = rep["per_kernel"][k]
        print(f"\nELIMINATE {k}")
        print(f"  {n} dispatches x {mean:.3f} ms = {tot * 1e3:.1f} ms/clip deleted outright")
        # Re-link X -> k -> Y as X -> Y. Exact only when one side is a single label; otherwise the
        # pair counts do not say which inbound edge fed which outbound one, so refuse rather than
        # assume a mixing that the recorded sequence could contradict.
        inn = [(x, c) for x, y, c in rep["trans"] if y == k]
        out = [(y, c) for x, y, c in rep["trans"] if x == k]
        relink = 0.0
        if len(out) == 1:
            (dest, _), = out
            for x, c in inn:
                if x == dest:
                    t = tax.get(dest, max(0.0, rep["per_kernel"][dest][2] - floor))
                    relink += c * t
                    print(f"    x{c:<4} @ {t:.3f} ms = {c * t:7.1f} ms   {x[-30:]} -> {dest[-30:]} "
                          f"becomes in-context")
                else:
                    print(f"    x{c:<4} @ 0.000 ms =     0.0 ms   {x[-30:]} -> {dest[-30:]} "
                          f"stays a transition")
            print(f"  PROJECTED saving {tot * 1e3 + relink:.1f} ms/clip "
                  f"({tot * 1e3:.1f} deleted + {relink:.1f} re-linked)")
            print("  EXCLUDES whatever the absorbing kernel pays to do this work instead (>= 0).")
        else:
            print(f"  cannot re-link: {k} has {len(out)} outbound edges, so the pair counts do not "
                  "determine\n  which inbound edge fed which outbound one. Re-run with a dumped "
                  "NPU_DISPATCH_SEQ.")

    if merge:
        a, b = find(rep, merge[0]), find(rep, merge[1])
        killed, saved, rows = model_merge(rep, a, b, tax, floor)
        n_killed = sum(n for _, _, n in killed)
        print(f"\nMERGE {a[-40:]}\n   +  {b[-40:]}")
        print(f"  boundaries deleted: {n_killed}  ({rep['transitions']} -> "
              f"{rep['transitions'] - n_killed} transitions)")
        for name, n, t, s, kind in rows:
            print(f"    x{n:<4} @ {t:.3f} ms = {s:7.1f} ms  [{kind:<8}] {name}")
        meas = sum(r[3] for r in rows if r[4] == "measured")
        bound = saved - meas
        print(f"  PROJECTED saving {saved:.1f} ms/clip  "
              f"({saved / max(n_killed,1):.3f} ms per boundary)")
        print(f"    of which MEASURED tax {meas:.1f} ms  |  upper-BOUND only {bound:.1f} ms")
        print(f"    rank folds on the MEASURED column; the bounded one is an upper limit, and it is "
              f"vacuous\n    for any kernel whose dispatch does real work rather than mostly "
              f"reconfiguring.")
        bound = {}
        for (lab, _), s in per_stream.items():
            if lab not in reconfig and s["xcl"] is not None:
                bound[lab] = min(bound.get(lab, s["xcl"]), s["xcl"])
        charge, crows = survivor_charge(rep, a, b, reconfig, bound)
        if charge is None:
            print("  survivor charge: NOT MODELLED -- no reconfig rate for one side")
        else:
            print(f"  survivor charge (arrivals into {a[-30:]} that still cross,")
            print(f"    now entering the merged program):")
            for name, n, t, s in crows:
                print(f"    x{n:<4} @ {t:+.3f} ms = {s:+7.1f} ms  {name}")
            using = "fitted" if a in reconfig else f"BOUND reconfig({a[-24:]}) <= {bound.get(a):.3f}"
            print(f"    [{using}] -- with a bound this charge is a LOWER limit, so the net is an "
                  f"UPPER limit")
            print(f"  NET {saved - charge:+.1f} ms/clip  "
                  f"({saved:.1f} saved - {charge:.1f} charged)")
        print(f"  fleet-average comparison: {n_killed} x 1.543 = {n_killed * 1.543:.1f} ms "
              f"-- the rate this model exists to replace")
        print("\n  NOTE: this is a PROJECTION. It is calibrated to ~1% on FOLD_FC1 and FOLD_GLU "
              "(see module docstring),\n  but it assumes the merge deletes exactly these boundaries "
              "and changes nothing else about the work.")


if __name__ == "__main__":
    main()
