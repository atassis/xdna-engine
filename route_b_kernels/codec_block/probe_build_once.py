#!/usr/bin/env python3
"""Does building the design ONCE and calling it N times avoid the rebuild?

The driver currently calls bricklib.verify_streamed per window, and that function constructs a
fresh design and jit-wrapper every call -- so the aiecc subprocess (~2 s, 83% of a dispatch) runs
per window. Enabling the JIT cache papers over that. The actual question is whether a HOISTED
design object reuses its compiled kernel across calls, which would make the build a one-time
step instead of a per-dispatch one, with no cache and no correctness question at all.
"""
import os, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import window_driver as wd            # puts bricks/_verify on sys.path
import bricklib
from aie import iron

C_IN, K, T, C_OUT = 128, 7, wd.T, 384
wide = C_IN * K + 1
rng = np.random.default_rng(9)
sym = "probe_bo"
shim = bricklib.GEN / "probe_bo_shim.cc"
shim.write_text(f'#include <stdint.h>\n#include "{wd.CONV_CC}"\n'
                f'extern "C" void {sym}(float *tile, float *resident, float *out) {{\n'
                f"  route_b_bricks::conv_1d_causal_core_vec<16>(resident, tile, tile[{C_IN*K}],\n"
                f"                     out, {C_IN}, {K}, {T}, 1);\n}}\n")

print(f"BRICK_JIT_CACHE={os.environ.get('BRICK_JIT_CACHE','<unset>')}  build ONCE, dispatch 4x",
      flush=True)
t0 = time.perf_counter()
design = bricklib._build_streamed(sym, shim, C_OUT, wide, T, C_IN * T, None,
                                  np.float32, np.float32, np.float32, 1)
print(f"  _build_streamed() returned in {time.perf_counter()-t0:6.3f} s (lazy: no compile yet)",
      flush=True)

for i in range(1, 5):
    tiles = (rng.standard_normal((C_OUT, wide)) * 0.02).astype(np.float32)
    win = rng.standard_normal((C_IN, T)).astype(np.float32)
    t0 = time.perf_counter()
    in_t = iron.tensor(np.ascontiguousarray(tiles.reshape(-1)), dtype=np.float32, device="npu")
    r_t = iron.tensor(np.ascontiguousarray(win.reshape(-1)), dtype=np.float32, device="npu")
    out_t = iron.zeros((C_OUT * T,), dtype=np.float32, device="npu")
    design(in_t, r_t, out_t)          # SAME design object every time
    got = out_t.numpy()
    print(f"  dispatch {i}: {time.perf_counter()-t0:6.3f} s   nz={np.abs(got).sum():.3e}",
          flush=True)
