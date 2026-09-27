#!/usr/bin/env python3
"""Device-time the fsr1_strip kernel at two crop sizes (dispatch-inclusive wall time,
min of N repeats per method-build-run-npu-xrt-test.md). BRICK_JIT_CACHE=1 so repeat
calls hit the cached compiled design instead of re-running aiecc each time."""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("BRICK_JIT_CACHE", "1")

import numpy as np

HERE = Path(__file__).parent
WORKSPACE = HERE.parent.parent.parent.parent
sys.path.insert(0, str(WORKSPACE / "xdna-engine" / "aie_kernels" / "_test"))
import bricklib as bl  # noqa: E402

import aie.iron as iron  # noqa: E402

GEN = HERE / "gen"
GEN.mkdir(exist_ok=True)
SHIM = GEN / "time_fsr1_shim.cc"
SHIM.write_text(
    '#include <stdint.h>\n#include "../fsr1_kernel.cc"\n'
    'extern "C" void fsr1_strip_shim(float *in_rgb, float *out_rgb) {\n'
    '    fsr1_strip(in_rgb, out_rgb);\n}\n'
)
N_REPS = 15


def time_one(in_w, in_h):
    out_w, out_h = in_w * 3, in_h * 3
    compile_flags = [f"-DFSR1_IN_W={in_w}", f"-DFSR1_IN_H={in_h}", "-Oz"]
    design = bl._build_oneshot(
        "fsr1_strip_shim", str(SHIM), [in_w * in_h * 3], out_w * out_h * 3,
        [np.float32], np.float32, compile_flags, stack_size=0x2000,
    )
    rng = np.random.default_rng(0)
    img = (rng.integers(20, 235, size=(in_h * in_w * 3,)).astype(np.float32) / 255.0)
    in_t = iron.tensor(np.ascontiguousarray(img), dtype=np.float32, device="npu")
    out_t = iron.zeros((out_w * out_h * 3,), dtype=np.float32, device="npu")

    times = []
    for _ in range(N_REPS):
        t0 = time.perf_counter()
        design(in_t, out_t)
        times.append(time.perf_counter() - t0)
    n_out_px = out_w * out_h
    tmin = min(times)
    print(f"in={in_w}x{in_h} out={out_w}x{out_h} ({n_out_px} px) "
          f"wall_min={tmin*1e6:.1f}us wall_med={sorted(times)[len(times)//2]*1e6:.1f}us "
          f"({tmin/n_out_px*1e9:.1f} ns/px, dispatch-inclusive host round trip)")
    return tmin, n_out_px


if __name__ == "__main__":
    r1 = time_one(8, 8)
    r2 = time_one(11, 11)
    d_time = r2[0] - r1[0]
    d_px = r2[1] - r1[1]
    print(f"delta: {d_time*1e6:.1f}us over {d_px} extra px "
          f"= {d_time/d_px*1e9:.1f} ns/px marginal (isolates per-pixel cost from fixed "
          f"dispatch overhead, which is ~{r1[0]*1e6 - r1[1]*(d_time/d_px)*1e6:.1f}us)")
