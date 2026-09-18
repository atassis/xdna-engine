#!/usr/bin/env python3
"""DEVICE HALF of the JIT-cache perturb check (companion to perturb_check.py, which is
device-free and already PASSED: insts.bin/main.pdi/<symbol>.o are byte-identical across an
unperturbed rebuild and differ under a real kernel-content perturbation, in three separate
processes, both with BRICK_JIT_CACHE=1 the new default).

This script is the thing that check could NOT prove by itself: that a real device GATE (rel-L2
against a numpy golden, through `bricklib.verify_rowwise`, the exact rail every brick uses) does
not silently keep reporting a stale result across SEPARATE PROCESSES now that
`BRICK_JIT_CACHE` defaults on and `window_driver._CB` no longer forces a fresh design name every
run. bricklib.py's own docstring records the failure this guards: a reused design name once
returned a stale xclbin at rel-L2 2.052e+01 (output in [-110.9, +124.96]) while a fresh name on
the identical shim returned 1.275e-04 -- a PASSING run on the wrong artifact, not a crash.

THREE SEPARATE PROCESSES (subprocess, not in-process loops -- an in-process rerun would hit
bricklib._DESIGNS / CallableDesign._kernel_cache regardless of the on-disk cache and prove
nothing about it):
  1. baseline   -- coeff=3.0, gate against the coeff=3.0 golden.       Expect: PASS, rel-L2 ~1e-7.
  2. unperturbed rerun -- coeff=3.0 again (same content), same gate.  Expect: bit-identical
     device output vs run 1 (run2run already checked by verify_rowwise INSIDE each process; this
     checks ACROSS processes, which is the part that matters here).
  3. perturbed  -- coeff=5.0, gate STILL run against the coeff=3.0 golden on purpose. Expect:
     rel-L2 MOVES (large, gate fails) -- if it instead reports ~1e-7 again, the cache served a
     stale coeff=3.0 artifact under the coeff=5.0 design's name, exactly the bricklib.py failure
     mode above.

Run under the device lock (announce + fuser), like any other device-touching probe/verify here.
Uses an isolated symbol prefix (`perturbdev_`) and its own scratch dir so it does not collide
with another agent's concurrent device work through the shared bricklib.GEN directory.

Usage:
    python3 perturb_check_device.py                 # runs all three, prints PASS/FAIL
    python3 perturb_check_device.py --worker <coeff> <cache_dir>   # one subprocess (internal)
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
SYMBOL = "perturbdev_op"
M, COLS = 32, 16
GOLDEN_COEFF = 3.0


def _golden(x, coeff):
    return (x * coeff).astype(np.float32)


def worker(coeff, cache_dir):
    """One process: build (or reuse) the design for `coeff`, dispatch on device, gate against
    the FIXED coeff=3.0 golden regardless of what coeff this process actually built with --
    that mismatch, for the coeff=5.0 case, is the point."""
    os.environ.setdefault("NPU_CACHE_HOME", cache_dir)
    sys.path.insert(0, str(HERE))
    import bricklib

    rng = np.random.default_rng(0)   # fixed seed: identical input across all three processes
    x = (rng.random((M, COLS), dtype=np.float32) * 4.0 - 2.0).astype(np.float32)
    ref = _golden(x, GOLDEN_COEFF)

    kernel = bricklib.GEN / f"{SYMBOL}_kernel.cc"
    kernel.write_text(
        "#include <aie_api/aie.hpp>\n"
        f'extern "C" void {SYMBOL}(float *in, float *out) {{\n'
        "  event0();\n"
        "  ::aie::vector<float,16> v = ::aie::load_v<16>(in);\n"
        f"  v = ::aie::mul(v, ::aie::broadcast<float,16>({coeff}f));\n"
        "  ::aie::store_v(out, v);\n"
        "  event1();\n"
        "}\n"
    )

    res = bricklib.verify_rowwise(
        name=SYMBOL, brick_cc=kernel, shim_body="", symbol=SYMBOL,
        m=M, in_cols=COLS, out_cols=COLS, x=x, expected=ref, gate=3e-2,
    )
    print(json.dumps({
        "coeff": coeff, "rel_l2": res["rel_l2"], "ok": res["ok"],
        "run2run": res["run2run"], "got_sample": res["got"][0, :4].tolist(),
    }))


def _run_subprocess(coeff, cache_dir):
    r = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker", str(coeff), cache_dir],
        capture_output=True, text=True, cwd=str(HERE),
    )
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr, file=sys.stderr)
        raise RuntimeError(f"device worker failed (coeff={coeff})")
    # verify_rowwise prints progress lines too; the JSON is always the last line.
    return json.loads(r.stdout.strip().splitlines()[-1])


def main():
    cache_dir = str(Path.home() / ".npu" / "cache")   # the REAL default, deliberately --
    # this check's whole point is the shared, persistent, cross-agent cache, not an isolated one.
    print(f"NPU_CACHE_HOME: {cache_dir}")

    r1 = _run_subprocess(GOLDEN_COEFF, cache_dir)
    print(f"[1] baseline    coeff={GOLDEN_COEFF}: rel_l2={r1['rel_l2']:.3e} ok={r1['ok']}")

    r2 = _run_subprocess(GOLDEN_COEFF, cache_dir)
    print(f"[2] unperturbed coeff={GOLDEN_COEFF}: rel_l2={r2['rel_l2']:.3e} ok={r2['ok']}")

    r3 = _run_subprocess(5.0, cache_dir)
    print(f"[3] PERTURBED   coeff=5.0 (gated vs coeff={GOLDEN_COEFF} golden): "
          f"rel_l2={r3['rel_l2']:.3e} ok={r3['ok']}")

    cross_process_identical = (r1["got_sample"] == r2["got_sample"]) and abs(
        r1["rel_l2"] - r2["rel_l2"]) < 1e-9
    perturbation_detected = r3["rel_l2"] > 0.1   # coeff 5 vs 3 on x in [-2,2]: large, unmissable

    print(f"\ncross-process unperturbed rerun bit-identical: {cross_process_identical}")
    print(f"perturbation moved rel-L2 (not served stale):   {perturbation_detected}")

    ok = cross_process_identical and perturbation_detected
    print(f"\nVERDICT: {'PASS' if ok else 'FAIL'} -- "
          f"{'cache is cross-process safe with BRICK_JIT_CACHE=1 and no _CB' if ok else 'INVESTIGATE: stale-artifact risk reproduced, do NOT ship this default'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--worker":
        worker(float(sys.argv[2]), sys.argv[3])
        sys.exit(0)
    main()
