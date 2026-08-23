#!/usr/bin/env python3
"""Separate a merge-specific charge from an arm-wide level shift, using untouched xclbins.

`arrival_refit_bracket.py` fitted the panel's reconfiguration cost before and after a merge and
found it rose by +0.114 ms, 10/11 reps -- and read that as "absorbing a brick makes its program
dearer to enter". That comparison has no control group. A merge changes the whole arm, so a
before/after on the ONE program that was merged cannot tell a charge on that program from a shift
affecting every program in the run.

This adds the control. Streams whose xclbin the merge never touched must, by construction, be
unaffected by it; whatever they do between the arms is the arm-wide level shift. Two readings:

  * per-stream arrival deltas as a FRACTION of the k0 arrival -- a level shift is proportional,
    a merge charge is not, and on the krtpkrl bracket every untouched stream moves +3.8..+7.0%
  * difference-in-differences, treated minus pooled control, paired per rep -- what survives the
    shift. On krtpkrl that is +0.02% [-0.49, +0.52]: nothing.

It also prints the exact ledger decomposition, which closes and shows nothing else is missing:

    delta(stream) = -n_conv*(xcl0 - ic1)      converted boundaries (the model has this)
                  + n_xcl1*(xcl1 - xcl0)      surviving arrivals at the new cost    (A)
                  + n_ic0*(ic1 - ic0)         drift in work already in-context      (B)

with n_conv = n_xcl0 - n_xcl1. The shipped model carries (A) for the absorbed brick only and no
(B) at all, which is the whole of its 21.8 ms optimism on krtpkrl.

Run it against a bracket that HAS a null-control arm (lnmode_fold_bracket's cache arms are a
verified no-op: 695 -> 695 transitions) to check the instrument reads zero when nothing changed.

Usage: arrival_control_bracket.py <bracket_dir> <arm_before> <arm_after> [--treated SUBSTR]
"""
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from transition_fold_model import load, reconfig_by_program  # noqa: E402


def ci(xs):
    """Mean and a 95% normal interval, matching the bracket harness's own summary."""
    m = statistics.mean(xs)
    if len(xs) < 2:
        return m, m, m
    h = 1.96 * statistics.stdev(xs) / len(xs) ** 0.5
    return m, m - h, m + h


def row(name, xs, unit="ms", width=46):
    m, lo, hi = ci(xs)
    print(f"  {name:<{width}} {m:+9.3f} [{lo:+8.3f}, {hi:+8.3f}] {unit}  n={len(xs)}")


def streams(path):
    rep = load(path)
    reconfig, per = reconfig_by_program(rep)
    return per, reconfig, rep


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__.strip().splitlines()[-1])
    d = Path(sys.argv[1])
    before, after = sys.argv[2], sys.argv[3]
    treated = sys.argv[sys.argv.index("--treated") + 1] if "--treated" in sys.argv else "modal"

    reps = sorted({int(re.search(r"rep(\d+)", p.name).group(1))
                   for p in d.glob(f"{before}_rep*.txt")})
    reps = [r for r in reps if (d / f"{after}_rep{r}.txt").exists()]
    if not reps:
        sys.exit(f"no paired {before}/{after} reps under {d}")
    print(f"bracket {d}  arms {before} -> {after}  reps {min(reps)}..{max(reps)}"
          f"  (no device time -- re-read of banked reports)")

    ladder = {k: [] for k in ("saved", "A_treated", "A_control", "B", "meas")}
    conserve, rel_ctl, rel_trt, recon = [], {}, [], []
    unmatched = set()
    for r in reps:
        A, recA, repA = streams(d / f"{before}_rep{r}.txt")
        B, recB, repB = streams(d / f"{after}_rep{r}.txt")
        pa = next((p for p in recA if treated in p), None)
        pb = next((p for p in recB if treated in p), None)
        if pa and pb:
            recon.append(recB[pb] - recA[pa])
        saved = a_trt = a_ctl = b_term = 0.0
        cerr = 0
        for k in set(A) & set(B):
            s0, s1 = A[k], B[k]
            # An untouched xclbin is the control: the merge cannot have charged it.
            is_treated = treated in k[0]
            cerr += abs((s0["n_xcl"] + s0["n_incontext"]) - (s1["n_xcl"] + s1["n_incontext"]))
            n_conv = max(0, s0["n_xcl"] - s1["n_xcl"])
            if s0["xcl"] is not None and s1["incontext"] is not None:
                saved += n_conv * (s0["xcl"] - s1["incontext"])
            if s0["xcl"] and s1["xcl"] and s1["n_xcl"]:
                t = s1["n_xcl"] * (s1["xcl"] - s0["xcl"])
                pct = (s1["xcl"] - s0["xcl"]) / s0["xcl"] * 100.0
                if is_treated:
                    a_trt += t
                    rel_trt.append(pct)
                else:
                    a_ctl += t
                    rel_ctl.setdefault(k, []).append(pct)
            if s0["incontext"] is not None and s1["incontext"] is not None:
                b_term += s0["n_incontext"] * (s1["incontext"] - s0["incontext"])
        unmatched.update({k[0] for k in set(A) ^ set(B)})
        ladder["saved"].append(-saved)
        ladder["A_treated"].append(a_trt)
        ladder["A_control"].append(a_ctl)
        ladder["B"].append(b_term)
        ladder["meas"].append((repB["blocking_s"] - repA["blocking_s"]) * 1e3)
        conserve.append(cerr)

    n = len(reps)
    print("\nledger decomposition, paired per rep")
    row("-saved (converted boundaries)", ladder["saved"])
    row("+A treated survivors", ladder["A_treated"])
    row("+A control survivors  [untouched xclbins]", ladder["A_control"])
    row("+B in-context drift", ladder["B"])
    print()
    shipped = [ladder["saved"][i] + ladder["A_treated"][i] for i in range(n)]
    exact = [shipped[i] + ladder["A_control"][i] + ladder["B"][i] for i in range(n)]
    row("model as shipped", shipped)
    row("exact (all terms)", exact)
    row("MEASURED", ladder["meas"])
    row("residual: shipped - measured", [shipped[i] - ladder["meas"][i] for i in range(n)])
    row("residual: exact - measured", [exact[i] - ladder["meas"][i] for i in range(n)])
    print(f"    dispatch-count conservation error per rep: {set(conserve)}"
          f"  ({'exact' if set(conserve) == {0} else 'LEAKS -- streams are not matching'})")
    if unmatched:
        # A stream that CHANGES program (the absorbed brick) has a different label in each arm, so
        # (label, ordinal) cannot pair it and its converted boundaries are missing from `saved`
        # above. The control/DiD readings below are unaffected -- they only use paired streams.
        print(f"    UNPAIRED streams, so the ladder is partial: "
              f"{sorted(x.split('final_')[-1][:40] for x in unmatched)}")

    print("\nrelative arrival inflation, % of the before-arm arrival")
    pooled = []
    for k, xs in sorted(rel_ctl.items(), key=lambda kv: -len(kv[1])):
        if len(xs) < max(3, n // 2):
            continue
        row(f"CONTROL {k[0].split('final_')[-1][:34]}#{k[1]}", xs, "%", width=46)
        pooled.append(xs)
    if rel_trt:
        row("TREATED (the merged program)", rel_trt, "%")
    if pooled:
        m = min(len(v) for v in pooled)
        agg = [statistics.mean([v[i] for v in pooled]) for i in range(m)]
        row("POOLED CONTROL", agg, "%")
        if len(rel_trt) >= m:
            dd = [rel_trt[i] - agg[i] for i in range(m)]
            row("DIFFERENCE-IN-DIFFERENCES (treated - control)", dd, "%")
            _, lo, hi = ci(dd)
            print("    -> spans zero: NO merge-specific excess beyond the arm-wide shift"
                  if lo < 0 < hi else
                  "    -> excludes zero: a merge-specific excess survives the control")
    if recon:
        print()
        row("reconfig(treated program) paired delta", recon)
        print(f"    dearer in {sum(1 for x in recon if x > 0)}/{len(recon)} reps")


if __name__ == "__main__":
    main()
