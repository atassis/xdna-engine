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

"""Where between 128 (works) and 512 (fails hard) does sin_v break?"""
import time
from pathlib import Path
import numpy as np
import bricklib

BRICK = (Path(__file__).parent.parent / "sin" / "sin.cc").resolve()
for N in (192, 256, 384):
    x = (np.random.default_rng(0).random(N, dtype=np.float32) * 128 - 64).astype(np.float32)
    ref = np.sin(x.astype(np.float64)).astype(np.float32)
    sym = f"sb_{N}_{int(time.time()*1e6) % 10**6}"
    shim = (f"// cb {int(time.time()*1e6) % 10**9}\n"
            f'extern "C" void {sym}(float *x, float *out) {{ sin_f32(x, out, {N}); }}\n')
    try:
        r = bricklib.verify_oneshot(name=f"sb{N}", brick_cc=BRICK, shim_body=shim, symbol=sym,
                                    inputs=[(x, np.float32)], out_numel=N, out_shape=(N,),
                                    unpack=lambda d: d, golden=ref, gate=3e-2, out_dt=np.float32)
        got = np.asarray(r["got"], np.float32)
        bad = np.where(np.abs(got) > 1.0001)[0]      # sin cannot exceed 1: unambiguous corruption
        print(f"  N={N:4d} rel_l2={r['rel_l2']:.3e} |out|>1 count={bad.size} first={int(bad[0]) if bad.size else None} -> {r['status']}")
    except Exception as exc:
        print(f"  N={N:4d} FAILED: {type(exc).__name__}: {str(exc)[:90]}")
