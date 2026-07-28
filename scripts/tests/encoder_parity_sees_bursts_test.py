#!/usr/bin/env python3
"""CPU gate for the encoder-parity gate itself -- a negative control.

`encoder_parity.py` used to report ONE number: the whole-clip mean rel-L2, averaged over clips. A
mean over ~1800 frames cannot see a burst confined to 3 of them, so a change that wrecked a handful
of frames and left everything else alone passed. That blind spot is not hypothetical: the shipped
encoder carries 15-79% per-frame bursts on 16 of 17 clips and the mean gate reported 0.089 and PASS.

This test builds a synthetic candidate whose ONLY defect is a localized burst, and asserts:

  1. the mean check alone PASSES it            (the old gate's blind spot, reproduced)
  2. the worst-frame check FAILS it            (the new gate sees it)
  3. the worst-burst check FAILS it            (the new gate sees the contiguous run)
  4. the localized-regression list names the injected frames exactly

and, as the other half of a control, that a candidate identical to the baseline passes everything.
No device, no NPU, no dumps needed -- it synthesises its own inputs.

    python3 scripts/tests/encoder_parity_sees_bursts_test.py
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GATE = os.path.join(REPO, "scripts", "encoder_parity.py")

T, D, NCLIPS = 120, 1024, 6
BURST_FRAMES = [40, 41, 42]
BURST_CLIP = 2

rng = np.random.default_rng(20260728)


def run_gate(ref, base, cand, extra=()):
    out = subprocess.run(
        [sys.executable, GATE, ref, base, cand, *extra],
        capture_output=True, text=True)
    return out.returncode, out.stdout + out.stderr


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    return cond


def gate_line(text, stat):
    for line in text.splitlines():
        if line.startswith(f"GATE {stat}"):
            return line
    raise AssertionError(f"no GATE line for {stat!r} in:\n{text}")


def main():
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        ref_d = os.path.join(tmp, "ref")
        base_d = os.path.join(tmp, "base")
        cand_d = os.path.join(tmp, "cand")
        same_d = os.path.join(tmp, "same")
        for d in (ref_d, base_d, cand_d, same_d):
            os.makedirs(d)

        for i in range(NCLIPS):
            n = f"c{i:02d}.npy"
            ref = rng.standard_normal((T, D)).astype(np.float32)
            # A baseline that is uniformly a little off truth -- the bf16 matmul noise floor.
            base = ref + rng.standard_normal((T, D)).astype(np.float32) * 0.08
            cand = base.copy()
            if i == BURST_CLIP:
                # The ONLY difference: three frames pushed to ~60% relative error. Everything else
                # is bit-identical to the baseline, so the whole-clip mean barely moves.
                for t in BURST_FRAMES:
                    cand[t] = ref[t] + rng.standard_normal(D).astype(np.float32) * 0.60
            np.save(os.path.join(ref_d, n), ref)
            np.save(os.path.join(base_d, n), base)
            np.save(os.path.join(cand_d, n), cand)
            np.save(os.path.join(same_d, n), base)

        # --- control A: identical candidate must pass every check -------------------------------
        rc, txt = run_gate(ref_d, base_d, same_d)
        ok &= check("identical candidate -> exit 0", rc == 0, f"exit={rc}")
        ok &= check("identical candidate -> OVERALL PASS", "GATE OVERALL" in txt and
                    gate_line(txt, "OVERALL").endswith("PASS"))
        ok &= check("identical candidate -> 0 localized regressions",
                    "localized regressions  : 0 frames" in txt)

        # --- control B: the burst-only candidate ------------------------------------------------
        rc, txt = run_gate(ref_d, base_d, cand_d, extra=("--per-clip",))

        mean_line = gate_line(txt, "mean")
        ok &= check("burst candidate -> mean check still PASSES (the old blind spot)",
                    mean_line.rstrip().endswith("PASS"), mean_line.strip())

        wf_line = gate_line(txt, "worst-frame")
        ok &= check("burst candidate -> worst-frame check FAILS",
                    wf_line.rstrip().endswith("FAIL"), wf_line.strip())

        wb_line = gate_line(txt, "worst-burst[5]")
        ok &= check("burst candidate -> worst-burst check FAILS",
                    wb_line.rstrip().endswith("FAIL"), wb_line.strip())

        nb_line = gate_line(txt, "new-burst")
        ok &= check("burst candidate -> new-burst check FAILS", "FAIL" in nb_line, nb_line.strip())
        ok &= check("new-burst names the injected clip+frame",
                    f"c{BURST_CLIP:02d}[" in nb_line and
                    any(f"[{t}]" in nb_line for t in BURST_FRAMES), nb_line.strip())
        ok &= check("burst-frame churn is reported",
                    f"burst-frame churn      : {len(BURST_FRAMES)} frames crossed INTO" in txt)

        ok &= check("burst candidate -> exit non-zero", rc != 0, f"exit={rc}")

        # The regression list must name exactly the frames we injected, no more.
        named = set()
        for line in txt.splitlines():
            if line.startswith("localized regressions"):
                count = int(line.split(":")[1].strip().split()[0])
                ok &= check("regression count == injected frames", count == len(BURST_FRAMES),
                            f"reported {count}, injected {len(BURST_FRAMES)}")
                for tok in line.split("worst: ")[-1].split(", "):
                    if "[" in tok:
                        named.add(int(tok.split("[")[1].split("]")[0]))
        ok &= check("regression list names the injected frames",
                    named == set(BURST_FRAMES), f"{sorted(named)} vs {BURST_FRAMES}")

        # And the per-clip table must point at the right clip.
        ok &= check("per-clip table is printed", "wF_b" in txt and "burst_c" in txt)

    print()
    print("GATE-SELFTEST", "GREEN" if ok else "RED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
