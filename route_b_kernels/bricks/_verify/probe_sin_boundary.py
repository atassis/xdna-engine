#!/usr/bin/env python3
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
