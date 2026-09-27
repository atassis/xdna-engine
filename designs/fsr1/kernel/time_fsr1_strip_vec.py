#!/usr/bin/env python3
"""Device-time the fused fsr1_strip_vec kernel (vectorized EASU+RCAS, one dispatch) at two
crop sizes, same marginal-cost method as time_fsr1_vec.py."""
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
IN_W = 16
N_REPS = 15


def time_one(in_h):
    out_w, out_h = IN_W * 3, in_h * 3
    shim = GEN / f"time_fsr1_strip_vec_shim_{in_h}.cc"
    shim.write_text(
        f'#include <stdint.h>\n#define FSR1_IN_W {IN_W}\n#define FSR1_IN_H {in_h}\n'
        '#include "../fsr1_kernel_vec.cc"\n'
        'extern "C" void fsr1_strip_vec_shim(float *in_rgb, float *out_rgb) {\n'
        '    fsr1_strip_vec(in_rgb, out_rgb);\n}\n'
    )
    compile_flags = [f"-DFSR1_IN_W={IN_W}", f"-DFSR1_IN_H={in_h}", "-Oz"]
    design = bl._build_oneshot(
        "fsr1_strip_vec_shim", str(shim), [IN_W * in_h * 3], out_w * out_h * 3,
        [np.float32], np.float32, compile_flags, stack_size=0x4000,
    )
    rng = np.random.default_rng(0)
    img = (rng.integers(20, 235, size=(in_h * IN_W * 3,)).astype(np.float32) / 255.0)
    in_t = iron.tensor(np.ascontiguousarray(img), dtype=np.float32, device="npu")
    out_t = iron.zeros((out_w * out_h * 3,), dtype=np.float32, device="npu")

    times = []
    for _ in range(N_REPS):
        t0 = time.perf_counter()
        design(in_t, out_t)
        times.append(time.perf_counter() - t0)
    n_out_px = out_w * out_h
    tmin = min(times)
    print(f"in={IN_W}x{in_h} out={out_w}x{out_h} ({n_out_px} px) "
          f"wall_min={tmin*1e6:.1f}us wall_med={sorted(times)[len(times)//2]*1e6:.1f}us "
          f"({tmin/n_out_px*1e9:.1f} ns/px, dispatch-inclusive host round trip)")
    return tmin, n_out_px


if __name__ == "__main__":
    # fsr1_strip_vec carries a THIRD buffer beyond the usual depth-2 in/out objectFifos: the
    # static easu_buf intermediate (output-sized, single copy, no double-buffering). At in_h=12
    # (out 36x48) that overflows the core's data memory the same way RCAS-alone's did (see
    # README) -- in_h=8 is the largest crop that still fits.
    r1 = time_one(4)
    r2 = time_one(8)
    d_time = r2[0] - r1[0]
    d_px = r2[1] - r1[1]
    print(f"delta: {d_time*1e6:.1f}us over {d_px} extra px "
          f"= {d_time/d_px*1e9:.1f} ns/px marginal (isolates per-pixel cost from fixed "
          f"dispatch overhead, which is ~{r1[0]*1e6 - r1[1]*(d_time/d_px)*1e6:.1f}us)")
