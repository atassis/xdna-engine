#!/usr/bin/env python3
"""Device-time warp_kernel.cc at two tile sizes (marginal method, per time_fsr1_vec.py),
for each (channels, dtype) combo actually built. TILE_W fixed at 32; TILE_H 8 vs 16."""
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
TILE_W = 16
HALO = 8
PAD_W = TILE_W + 2 * HALO
N_REPS = 15

def time_one(ch, tile_h):
    pad_h = tile_h + 2 * HALO
    shim = GEN / f"time_warp_shim_c{ch}_f32_{tile_h}.cc"
    shim.write_text(
        f'#include <stdint.h>\n'
        f'#define WARP_TILE_W {TILE_W}\n#define WARP_TILE_H {tile_h}\n'
        f'#define WARP_HALO {HALO}\n#define WARP_CH {ch}\n'
        f'#include "../warp_kernel.cc"\n'
        f'extern "C" void warp_kernel_shim(float *in_padded, float *flow, float *out) {{\n'
        f'    warp_kernel(in_padded, flow, out);\n}}\n'
    )
    compile_flags = [f"-DWARP_TILE_W={TILE_W}", f"-DWARP_TILE_H={tile_h}",
                      f"-DWARP_HALO={HALO}", f"-DWARP_CH={ch}", "-Oz"]
    n_in = pad_h * PAD_W * ch
    n_flow = 2 * tile_h * TILE_W
    n_out = tile_h * TILE_W * ch
    design = bl._build_oneshot(
        "warp_kernel_shim", str(shim), [n_in, n_flow], n_out,
        [np.float32, np.float32], np.float32, compile_flags, stack_size=0x1000,
    )
    rng = np.random.default_rng(0)
    img = rng.standard_normal((n_in,)).astype(np.float32)
    flow = (rng.standard_normal((n_flow,)) * (HALO * 0.5)).astype(np.float32)
    in_t = iron.tensor(np.ascontiguousarray(img), dtype=np.float32, device="npu")
    flow_t = iron.tensor(np.ascontiguousarray(flow), dtype=np.float32, device="npu")
    out_t = iron.zeros((n_out,), dtype=np.float32, device="npu")

    times = []
    for _ in range(N_REPS):
        t0 = time.perf_counter()
        design(in_t, flow_t, out_t)
        times.append(time.perf_counter() - t0)
    n_out_px = tile_h * TILE_W
    tmin = min(times)
    print(f"[C={ch} f32] tile={TILE_W}x{tile_h} ({n_out_px} px) wall_min={tmin*1e6:.1f}us "
          f"wall_med={sorted(times)[len(times)//2]*1e6:.1f}us "
          f"({tmin/n_out_px*1e9:.1f} ns/px, dispatch-inclusive host round trip)")
    return tmin, n_out_px


def sweep(ch):
    r1 = time_one(ch, 8)
    r2 = time_one(ch, 16)
    d_time = r2[0] - r1[0]
    d_px = r2[1] - r1[1]
    ns_px = d_time / d_px * 1e9
    print(f"[C={ch} f32] marginal: {d_time*1e6:.1f}us over {d_px} extra px = "
          f"{ns_px:.1f} ns/px (fixed dispatch overhead ~"
          f"{r1[0]*1e6 - r1[1]*ns_px*1e-3:.1f}us)")
    return ns_px


if __name__ == "__main__":
    for ch in (3, 4):
        sweep(ch)
