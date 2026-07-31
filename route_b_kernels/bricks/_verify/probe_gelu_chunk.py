#!/usr/bin/env python3
"""SUPERSEDED ABI -- retained as evidence, NOT runnable against the current bricks.

This probe calls the pre-2026-07-31 entry points `gelu_erf_f32(x, out, n)` /
`sin_f32(x, out, n)`, which took an element count and looped internally. Both bricks now
take ONE 16-wide vector per call with no count and no loop -- that change is what took
`gelu-erf` to 1.138e-03 and `sin` to 1.275e-04, after each had been red.
See kb/kernel-internal-loops-miscompile-put-volume-in-the-worker.

Kept rather than deleted because the KB cites its measurements. To re-run it you would have
to restore the old ABI, which would reintroduce the defect -- so its numbers stand as a
historical record of the broken form, not as a live gate.
"""

"""Is gelu-erf wrong per-CHUNK rather than per-value?

v3's arithmetic is demonstrably right: every landmark in verify_gelu_erf.py comes back correct
(x=+10 -> 9.999732, matching the author's host prediction to 6 digits), yet x=+9.573385 in the same
buffer returns -1.77e+05. Same sign, same magnitude, same clamp branch. Value-dependence cannot
explain that.

What CAN: the 11 landmarks occupy elements 0..10 of the buffer, and the kernel consumes 16 floats
per `aie::load_v` chunk. So all of them sit in CHUNK 0. If chunk 0 is computed correctly and later
chunks are not, every observation so far is explained at once -- including why round 2's oneshot vs
rowwise probe looked like "both rails red" (both rails run many chunks per call).

That is the register-pressure spill sin.cc's header documents: "exact at 64 floats per call and
wrong above that, with results ALIASED across variables ... the trigger is the unroller". v3 is a
degree-9 Horner, a LONGER live chain than sin_v's 6-term one, so it should hit it harder.

Test: run the identical input at three call widths.
  n=16  -- exactly ONE chunk per call
  n=32  -- two chunks
  n=64  -- four chunks (what verify_gelu_erf.py uses)
If n=16 is green and the others degrade with chunk count, the arithmetic is exonerated and this is
a codegen/register-pressure defect, not an approximation problem -- and no amount of polynomial
work will fix it.
"""
import importlib.util
import time
from pathlib import Path

import numpy as np

import bricklib

HERE = Path(__file__).parent
BRICK = (HERE.parent / "gelu-erf").resolve()
GATE = 3e-2
ROWS = 32

spec = importlib.util.spec_from_file_location("gelu_golden", BRICK / "golden.py")
golden = importlib.util.module_from_spec(spec)
spec.loader.exec_module(golden)

rng = np.random.default_rng(7)
# Values that the landmark list does NOT cover: arbitrary reals across the bend and both tails,
# so a pass cannot be an artifact of nice inputs.
pool = np.concatenate([
    rng.uniform(-12.0, -4.0, 4096),
    rng.uniform(-3.0, 3.0, 4096),
    rng.uniform(4.0, 12.0, 4096),
]).astype(np.float32)
rng.shuffle(pool)

print(f"{'width':>6} {'chunks/call':>12} {'rel-L2':>12}  {'max-abs':>12}  verdict")
results = {}
for cols in (16, 32, 64):
    x = pool[:ROWS * cols].reshape(ROWS, cols)
    ref = golden.gelu_erf_ref(x)
    cb = int(time.time() * 1000) % 10**9
    res = bricklib.verify_rowwise(
        name=f"gelu_w{cols}",
        brick_cc=BRICK / "gelu_erf.cc",
        shim_body=(f"// cachebust {cb}\n"
                   f'extern "C" void gelu_w{cols}(float *x, float *out) '
                   f'{{ gelu_erf_f32(x, out, {cols}); }}\n'),
        symbol=f"gelu_w{cols}",
        m=ROWS, in_cols=cols, out_cols=cols,
        x=x, expected=ref, gate=GATE,
    )
    got = np.asarray(res["got"], np.float64)
    max_abs = float(np.max(np.abs(got - ref.astype(np.float64))))
    results[cols] = res["rel_l2"]
    print(f"{cols:6d} {cols // 16:12d} {res['rel_l2']:12.3e}  {max_abs:12.3e}  "
          f"{'PASS' if res['ok'] else 'FAIL'}")

    # Which chunk goes wrong, if any. This is the whole question.
    per_chunk = [float(np.max(np.abs(got[:, c:c + 16] - ref[:, c:c + 16].astype(np.float64))))
                 for c in range(0, cols, 16)]
    print("        per-chunk max-abs: " +
          "  ".join(f"c{i}={v:.3e}" for i, v in enumerate(per_chunk)))
    # WHAT does a wrong element look like? Exactly-zero would mean never written.
    err = np.abs(got - ref.astype(np.float64))
    nzero = int((got == 0.0).sum())
    print(f"        exact zeros in device out: {nzero}/{got.size} "
          f"({100.0*nzero/got.size:.1f}%)   sqrt(frac)={np.sqrt(nzero/got.size):.3f}")
    flat_i = np.argsort(err.ravel())[::-1][:6]
    for fi in flat_i:
        r, c = divmod(int(fi), cols)
        print(f"          x={x[r,c]:+9.4f}  golden={ref[r,c]:+12.5f}  device={got[r,c]:+14.5e}"
              f"  lane={c%16:2d} chunk={c//16}")

print()
if results[16] <= GATE and results[64] > GATE:
    print("VERDICT: n=16 (one chunk) GREEN, n=64 RED -> arithmetic is FINE, this is a")
    print("         per-chunk codegen/register-pressure defect, same family as sin.cc.")
elif all(v <= GATE for v in results.values()):
    print("VERDICT: all widths green -- the defect is elsewhere, re-read verify_gelu_erf.py.")
else:
    print("VERDICT: n=16 already red -> not chunk-count dependent; arithmetic still suspect.")
