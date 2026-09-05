#!/usr/bin/env python3
"""Bisect the head conv's ERT timeout by c_out, holding every other head parameter fixed.

The head (c_in=1024, c_out=1536, k=7, ci_chunk=128, resident_depth=1) was device-green at
4.119e-07 on 2026-07-31 and returns ERT_CMD_STATE_TIMEOUT today, in isolation, with the device
provably healthy (snake re-gates at its recorded 5.143e-06 either side of the failure). conv-1d
itself still passes its own gate at c=96, so the trigger is somewhere between those two shapes.

Sweeps c_out with a CONTROL at the shape conv-1d's own gate already covers, so a run where
everything fails is distinguishable from a real threshold. Gate here is only "does it return" --
correctness is verify_head_tail.py's job, not this probe's.
"""
import sys, time, traceback
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import window_driver as wd

C_IN, K, CI_CHUNK, RD = 1024, 7, 128, 1
L = 20                      # head_in length verify_whole_decoder solved for V_FINAL=128
rng = np.random.default_rng(0)

print(f"fixed: c_in={C_IN} k={K} ci_chunk={CI_CHUNK} resident_depth={RD} L={L}", flush=True)
for c_out in (96, 256, 512, 1024, 1536):
    x = rng.standard_normal((C_IN, L)).astype(np.float32)
    w = rng.standard_normal((c_out, C_IN, K)).astype(np.float32) * 0.02
    b = np.zeros(c_out, np.float32)
    t0 = time.time()
    try:
        out = wd.conv(x, w, b, K, 1, f"probe{c_out}", ci_chunk=CI_CHUNK, resident_depth=RD)
        print(f"c_out={c_out:5d}  OK    {time.time()-t0:7.1f}s  out{out.shape} "
              f"finite={np.isfinite(out).all()}", flush=True)
    except Exception as e:
        print(f"c_out={c_out:5d}  FAIL  {time.time()-t0:7.1f}s  {type(e).__name__}: "
              f"{str(e)[:90]}", flush=True)
