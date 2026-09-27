#!/usr/bin/env python3
"""Device-time the vectorized fsr1_rcas_vec kernel (RCAS alone) at two crop sizes, same
marginal-cost method as time_fsr1_vec.py. FSR1_IN_W/FSR1_IN_H are reused only to derive the
RCAS operand's own shape (FSR1_OUT_W/H = 3x); EASU itself is not called here."""
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
    w, h = IN_W * 3, in_h * 3
    shim = GEN / f"time_fsr1_rcas_vec_shim_{in_h}.cc"
    shim.write_text(
        f'#include <stdint.h>\n#define FSR1_IN_W {IN_W}\n#define FSR1_IN_H {in_h}\n'
        '#include "../fsr1_kernel_vec.cc"\n'
        'extern "C" void fsr1_rcas_vec_shim(float *in_rgb, float *out_rgb) {\n'
        '    fsr1_rcas_vec(in_rgb, out_rgb);\n}\n'
    )
    compile_flags = [f"-DFSR1_IN_W={IN_W}", f"-DFSR1_IN_H={in_h}", "-Oz"]
    design = bl._build_oneshot(
        "fsr1_rcas_vec_shim", str(shim), [w * h * 3], w * h * 3,
        [np.float32], np.float32, compile_flags, stack_size=0x4000,
    )
    rng = np.random.default_rng(0)
    img = rng.random(size=(h * w * 3,)).astype(np.float32)
    in_t = iron.tensor(np.ascontiguousarray(img), dtype=np.float32, device="npu")
    out_t = iron.zeros((w * h * 3,), dtype=np.float32, device="npu")

    times = []
    for _ in range(N_REPS):
        t0 = time.perf_counter()
        design(in_t, out_t)
        times.append(time.perf_counter() - t0)
    n_out_px = w * h
    tmin = min(times)
    print(f"in=out={w}x{h} ({n_out_px} px) "
          f"wall_min={tmin*1e6:.1f}us wall_med={sorted(times)[len(times)//2]*1e6:.1f}us "
          f"({tmin/n_out_px*1e9:.1f} ns/px, dispatch-inclusive host round trip)")
    return tmin, n_out_px


if __name__ == "__main__":
    # RCAS is same-size in->out, so (unlike EASU) BOTH objectFifo buffers are output-sized;
    # at depth 2 that's 4 buffers of w*h*3*4 bytes competing for the 48KB left after the
    # 16KB stack reservation -- crop sizes above ~21 output rows (at w=48) overflow the core's
    # data memory ("basic-sequential allocation failed"), so the two probe sizes are smaller
    # than time_fsr1_vec.py's.
    r1 = time_one(4)
    r2 = time_one(7)
    d_time = r2[0] - r1[0]
    d_px = r2[1] - r1[1]
    print(f"delta: {d_time*1e6:.1f}us over {d_px} extra px "
          f"= {d_time/d_px*1e9:.1f} ns/px marginal (isolates per-pixel cost from fixed "
          f"dispatch overhead, which is ~{r1[0]*1e6 - r1[1]*(d_time/d_px)*1e6:.1f}us)")
