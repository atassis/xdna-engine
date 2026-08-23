#!/usr/bin/env python3
"""Re-fit the per-program reconfiguration rate over a BRACKET's reps, not one pass.

`transition_fold_model.py` fits `arrival = reconfig(program) + work(stream)` from a single
`NPU_DISPATCH_LOG=1` report, and closes the lnaffcast merge to 0.4% on the pass it was fitted on.
That is a statement about the model's SHAPE. Its LEVEL was never bracketed: the fitted pass moved
-217.0 ms while the same merge's 11-rep bracket is -183.1 [-205.8, -160.4], so a per-pass arrival
table was being asked to predict a bracketed quantity.

This closes that. The bracket harness already banks a full dispatch report per (arm, rep), so the
re-fit costs no device time -- it re-reads `artifacts/lnmode_krtpkrl_bracket/{k0,k0m}_rep*.txt`.

Stream identity is matched across arms by the `[dispatch_log] stream k=.. n=.. act=..` header,
which prints the BO address the by-predecessor table keys on. Matching by dispatch count instead
would pair the wrong panel streams -- two of them are x48 -- and matching by ordinal would break
whenever a stream is added, which is exactly what the merge does.

Usage: arrival_refit_bracket.py [bracket_dir]
"""
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from transition_fold_model import load, reconfig_by_program  # noqa: E402

DECL = re.compile(r"^\[dispatch_log\] stream k=(\d+) n=(\d+) act=(\S+) -> (\d+) insts words, "
                  r"bo (0x[0-9a-f]+)", re.M)
PANEL = "modalsilubf16outpanel1024krtpkrl"


def identities(path):
    """stream key -> a name stable across arms: the declared (k,n,act), else insts-width."""
    txt = Path(path).read_text()
    by_bo = {bo: f"k{k}n{n}-{act}" for k, n, act, _insts, bo in DECL.findall(txt)}
    out = {}
    for m in re.finditer(r"^\s{4}(\S+?)#(\d+)@(0x[0-9a-f]+)\s+(\d+)\s", txt, re.M):
        label, ordn, bo, insts = m.group(1), int(m.group(2)), m.group(3), int(m.group(4))
        if PANEL in label:
            # The two undeclared panel streams are told apart by width: the mode-carrying one is
            # wider than the plain GEMM stream it rides with.
            name = by_bo.get(bo) or ("lnaff-mode" if insts > 2676 else "fc1-panel")
        else:
            name = label.split("final_")[-1]
        out[(label, ordn)] = name
    return out


def fit(path):
    rep = load(path)
    reconfig, per_stream = reconfig_by_program(rep)
    names = identities(path)
    panel = next((p for p in reconfig if PANEL in p), None)
    streams = {}
    for (label, ordn), s in per_stream.items():
        streams[names.get((label, ordn), f"{label}#{ordn}")] = {
            "incontext": s["incontext"], "xcl": s["xcl"],
            "n_incontext": s["n_incontext"], "n_xcl": s["n_xcl"], "prog": label,
        }
    return {"blocking_s": rep["blocking_s"], "transitions": rep["transitions"],
            "reconfig_panel": reconfig.get(panel), "streams": streams,
            "lnaff_arrival": next((s["xcl"] for k, s in streams.items()
                                   if k.startswith("lnaffcast")), None)}


def ci(xs):
    """Mean and a 95% normal interval; n is 11, so this matches the harness's own summary."""
    m = statistics.mean(xs)
    if len(xs) < 2:
        return m, m, m
    h = 1.96 * statistics.stdev(xs) / len(xs) ** 0.5
    return m, m - h, m + h


def row(name, xs, unit="ms"):
    m, lo, hi = ci(xs)
    print(f"  {name:<44} {m:8.3f} [{lo:7.3f}, {hi:7.3f}] {unit}  n={len(xs)}")


def main():
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/lnmode_krtpkrl_bracket")
    reps = sorted({int(re.search(r"rep(\d+)", p.name).group(1)) for p in d.glob("k0_rep*.txt")})
    if not reps:
        sys.exit(f"no k0_rep*.txt under {d}")
    print(f"bracket: {d}  reps {min(reps)}..{max(reps)}  (no device time -- re-read of banked reports)")

    k0 = {r: fit(d / f"k0_rep{r}.txt") for r in reps}
    k1 = {r: fit(d / f"k0m_rep{r}.txt") for r in reps}

    print("\nreconfiguration rate, fitted per rep")
    row("k0  panel (pre-merge)", [k0[r]["reconfig_panel"] for r in reps if k0[r]["reconfig_panel"]])
    row("k0m panel (post-merge, absorbed lnaffcast)",
        [k1[r]["reconfig_panel"] for r in reps if k1[r]["reconfig_panel"]])
    d_rec = [k1[r]["reconfig_panel"] - k0[r]["reconfig_panel"] for r in reps
             if k0[r]["reconfig_panel"] and k1[r]["reconfig_panel"]]
    row("paired delta (does absorbing inflate it?)", d_rec)
    print(f"    reps where the merged program is DEARER to enter: "
          f"{sum(1 for x in d_rec if x > 0)}/{len(d_rec)}")

    print("\nledger, paired per rep")
    row("blocking time k0", [k0[r]["blocking_s"] * 1e3 for r in reps])
    row("blocking time k0m", [k1[r]["blocking_s"] * 1e3 for r in reps])
    delta = [(k1[r]["blocking_s"] - k0[r]["blocking_s"]) * 1e3 for r in reps]
    row("delta (the merge, measured)", delta)
    print(f"    negative in {sum(1 for x in delta if x < 0)}/{len(delta)} reps")
    tr = [k1[r]["transitions"] - k0[r]["transitions"] for r in reps]
    print(f"    transitions deleted: {set(tr)} (exact every rep)" if len(set(tr)) == 1
          else f"    transitions deleted: {tr}")

    print("\nmodel, per rep: converted savings - survivor charge")
    modelled, resid = [], []
    for r in reps:
        a, b = k0[r]["streams"], k1[r]["streams"]
        rec0, rec1 = k0[r]["reconfig_panel"], k1[r]["reconfig_panel"]
        if not (rec0 and rec1):
            continue
        saved = charge = 0.0
        # The absorbed brick changes program, so it is matched by hand; every other stream keeps
        # its declared identity across the two arms.
        ln0 = next((s for k, s in a.items() if k.startswith("lnaffcast")), None)
        pairs = [(a.get(n), s1) for n, s1 in b.items() if n != "lnaff-mode"]
        if ln0 and "lnaff-mode" in b:
            pairs.append((ln0, b["lnaff-mode"]))
        # Converted = arrivals that STOPPED crossing, which is the drop in the cross-xclbin count --
        # NOT the post-merge in-context count. Two of the panel's streams were already dispatched
        # in-context before the merge, so counting their in-context column here would bill the merge
        # for boundaries it never deleted (it overstated this decomposition by 104 ms/clip).
        for s0, s1 in pairs:
            if not s0 or s0["xcl"] is None or s1["incontext"] is None:
                continue
            n_conv = max(0, s0["n_xcl"] - s1["n_xcl"])
            saved += n_conv * (s0["xcl"] - s1["incontext"])
        # Charged: the absorbed brick's surviving arrivals now enter the panel, not its own xclbin.
        lnm = b.get("lnaff-mode")
        if ln0 and lnm and lnm["xcl"] is not None and ln0["xcl"] is not None:
            charge += lnm["n_xcl"] * (lnm["xcl"] - ln0["xcl"])
        modelled.append(-(saved - charge))
        resid.append(modelled[-1] - delta[reps.index(r)])
    row("modelled delta", modelled)
    row("model - measured (paired residual)", resid)
    print(f"    |residual| median {statistics.median(abs(x) for x in resid):.1f} ms; "
          f"model within 10% of measured in "
          f"{sum(1 for m, dd in zip(modelled, delta) if abs(m - dd) <= 0.10 * abs(dd))}/{len(modelled)} reps")

    # The model above carries the surviving-arrival charge for the ABSORBED brick only. Two terms
    # are missing, and together they are the whole of its optimism: every OTHER stream's surviving
    # arrivals also move (A), and work already dispatched in-context drifts (B). With
    # n_conv = n_xcl0 - n_xcl1 the per-stream ledger is exact:
    #     delta = -n_conv*(xcl0 - ic1) + n_xcl1*(xcl1 - xcl0) + n_ic0*(ic1 - ic0)
    print("\nthe two terms the model omits, and what they close")
    a_other, b_term = [], []
    for r in reps:
        a, b = k0[r]["streams"], k1[r]["streams"]
        ln0 = next((s for kk, s in a.items() if kk.startswith("lnaffcast")), None)
        pairs = [(n, a.get(n), s1) for n, s1 in b.items() if n != "lnaff-mode"]
        if ln0 and "lnaff-mode" in b:
            pairs.append(("lnaff", ln0, b["lnaff-mode"]))
        ao = bt = 0.0
        for name, s0, s1 in pairs:
            if not s0:
                continue
            if name != "lnaff" and s0["xcl"] and s1["xcl"]:
                ao += s1["n_xcl"] * (s1["xcl"] - s0["xcl"])
            if s0["incontext"] is not None and s1["incontext"] is not None:
                bt += s0["n_incontext"] * (s1["incontext"] - s0["incontext"])
        a_other.append(ao)
        b_term.append(bt)
    row("A: surviving arrivals, streams other than lnaff", a_other)
    row("B: drift in work already in-context", b_term)
    exact = [modelled[i] + a_other[i] + b_term[i] for i in range(len(modelled))]
    row("exact decomposition (model + A + B)", exact)
    row("residual: exact - measured", [exact[i] - delta[i] for i in range(len(exact))])

    # ...but A is NOT a merge charge. Streams on xclbins the merge never touched move by the same
    # FRACTION as the merged one, so this arm pair carries a level shift the treated/control
    # contrast divides out. lnmode_fold_bracket, which has null-control arms that read zero, shows
    # no such shift and isolates the real charge at +0.025 ms (scripts/arrival_control_bracket.py).
    print("\ncontrol: streams whose xclbin the merge never touched (% of the k0 arrival)")
    CONTROL = ["final", "dwconv_silu_t_1024x400",
               "resadd_512x1024_s050_bf16b", "resadd_512x1024_s100_bf16b"]
    TREATED = "k1024n1024-modalid"

    def rel(r, n):
        s0, s1 = k0[r]["streams"].get(n), k1[r]["streams"].get(n)
        if not s0 or not s1 or not s0["xcl"] or not s1["xcl"]:
            return None
        return (s1["xcl"] - s0["xcl"]) / s0["xcl"] * 100.0

    cols = []
    for n in CONTROL:
        xs = [rel(r, n) for r in reps]
        xs = [x for x in xs if x is not None]
        if len(xs) == len(reps):
            row(f"CONTROL {n}", xs, "%")
            cols.append(xs)
    trt = [rel(r, TREATED) for r in reps]
    if all(x is not None for x in trt) and cols:
        row(f"TREATED {TREATED} (the merged program)", trt, "%")
        pooled = [statistics.mean([c[i] for c in cols]) for i in range(len(reps))]
        row("POOLED CONTROL", pooled, "%")
        dd = [trt[i] - pooled[i] for i in range(len(reps))]
        row("DIFFERENCE-IN-DIFFERENCES", dd, "%")
        _, lo, hi = ci(dd)
        print("    -> spans zero: on THIS bracket the merged program's rise is not "
              "separable from the arm-wide shift" if lo < 0 < hi else
              "    -> excludes zero: a merge-specific excess survives the control")

    print("\narrival cost of the absorbed brick, per rep (the cap on what its boundaries return)")
    row("lnaffcast arrival, k0", [k0[r]["lnaff_arrival"] for r in reps if k0[r]["lnaff_arrival"]])
    row("lnaff-mode arrival, k0m (entering the panel)",
        [k1[r]["streams"]["lnaff-mode"]["xcl"] for r in reps if "lnaff-mode" in k1[r]["streams"]])


if __name__ == "__main__":
    main()
