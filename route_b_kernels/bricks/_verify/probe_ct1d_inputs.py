#!/usr/bin/env python3
"""Which of the two input buffers actually arrives? Echo both to the output."""
import time
from pathlib import Path
import numpy as np
import bricklib

B = (Path(__file__).parent.parent / "conv-transpose-1d").resolve()
a = (np.arange(64, dtype=np.float32) + 100.0)      # in0 marker: 100..163
c = (np.arange(64, dtype=np.float32) + 1000.0)     # in1 marker: 1000..1063
cb = int(time.time() * 1000) % 10**9
shim = f'''// cb {cb}
extern "C" void ct1d_verify(float *x, float *wb, float *out) {{
  for (int i = 0; i < 32; i++) {{ out[i] = x[i]; out[32 + i] = wb[i]; }}
}}
'''
r = bricklib.verify_oneshot(name="conv_transpose_1d", brick_cc=B / "conv_transpose_1d.cc",
    shim_body=shim, symbol="ct1d_verify",
    inputs=[(a, np.float32), (c, np.float32)], out_numel=64, out_shape=(64,),
    unpack=lambda d: d, golden=np.concatenate([a[:32], c[:32]]), gate=1e-6, out_dt=np.float32)
g = np.asarray(r["got"], np.float32)
print("  in0 echo out[0:4]  :", g[0:4].tolist(), "(expect 100,101,102,103)")
print("  in1 echo out[32:36]:", g[32:36].tolist(), "(expect 1000,1001,1002,1003)")
