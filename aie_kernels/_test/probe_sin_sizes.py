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

"""Does the sin brick fail as a function of tile SIZE? First bad index was exactly 64 at N=1024."""
import importlib.util, time
from pathlib import Path
import numpy as np
import bricklib

B = (Path(__file__).parent.parent / "sin").resolve()
spec = importlib.util.spec_from_file_location("g", B / "golden.py")
g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)

for N in (64, 256, 512, 1024):
    x = (np.random.default_rng(0).random(N, dtype=np.float32) * 128 - 64).astype(np.float32)
    ref = g.sin_ref(x)
    m = int(time.time() * 1000) % 10**9          # cache-bust: the JIT key hashes only the shim text
    r = bricklib.verify_oneshot(
        name=f"sin{N}", brick_cc=B / "sin.cc",
        shim_body=f'// cb {m}\nextern "C" void sv{m}(float *x, float *out) {{ sin_f32(x, out, {N}); }}',
        symbol=f"sv{m}", inputs=[(x, np.float32)], out_numel=N, out_shape=(N,),
        unpack=lambda d: d, golden=ref, gate=3e-2, out_dt=np.float32)
    bad = np.where(~np.isclose(r["got"], ref, atol=1e-3))[0]
    fb = int(bad[0]) if bad.size else None
    print(f"  N={N:5d} rel_l2={r['rel_l2']:.3e} bad={bad.size}/{N} first_bad={fb}")
