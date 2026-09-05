#!/usr/bin/env python3
"""Do stores past element 64 land at all? Writes the iteration index, no arithmetic."""
import time
from pathlib import Path
import numpy as np
import bricklib

B = (Path(__file__).parent.parent / "sin").resolve()
M, COLS = 8, 512
cb = int(time.time() * 1000) % 10**9
shim = f'''// cb {cb}
extern "C" void sin_verify(float *x, float *out) {{
  const int W = 16;
  const int chunks = {COLS} / W;
  for (int i = 0; i < chunks; i++) {{
    ::aie::store_v(out + i * W, ::aie::broadcast<float, W>((float) i));
  }}
}}
'''
x = np.zeros((M, COLS), np.float32)
exp = np.tile((np.arange(COLS) // 16).astype(np.float32), (M, 1))
r = bricklib.verify_rowwise(name="sin", brick_cc=B / "sin.cc", shim_body=shim,
                            symbol="sin_verify", m=M, in_cols=COLS, out_cols=COLS,
                            x=x, expected=exp, gate=1e-6)
got = np.asarray(r["got"], np.float32)
bad = np.where(~np.isclose(got[0], exp[0], atol=1e-5))[0]
print("  row0 got[60:70] :", got[0, 60:70].tolist())
print("  row0 exp[60:70] :", exp[0, 60:70].tolist())
print("  row0 bad:", bad.size, "of", COLS, "first_bad:", int(bad[0]) if bad.size else None)
