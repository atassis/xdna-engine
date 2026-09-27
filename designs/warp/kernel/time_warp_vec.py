#!/usr/bin/env python3
"""Device-time warp_kernel_vec.cc at two tile sizes (marginal method). TILE_W fixed at
VW=16 (the vector width); TILE_H 8 vs 16, halo=8 (same as verify_warp_vec.py)."""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("BRICK_JIT_CACHE", "1")

import ml_dtypes
import numpy as np

HERE = Path(__file__).parent
WORKSPACE = HERE.parent.parent.parent.parent
sys.path.insert(0, str(WORKSPACE / "xdna-engine" / "aie_kernels" / "_test"))
import bricklib as bl  # noqa: E402

import aie.iron as iron  # noqa: E402

BF16 = ml_dtypes.bfloat16
GEN = HERE / "gen"
GEN.mkdir(exist_ok=True)
TILE_W = 16
HALO = 8
N_REPS = 15


def time_one(ch, tile_h):
    pad_h = tile_h + 2 * HALO
    pad_w = TILE_W + 2 * HALO
    shim = GEN / f"time_warp_vec_shim_c{ch}_{tile_h}.cc"
    shim.write_text(
        f'#include <stdint.h>\n'
        f'#define WARP_TILE_W {TILE_W}\n#define WARP_TILE_H {tile_h}\n'
        f'#define WARP_HALO {HALO}\n#define WARP_CH {ch}\n'
        f'#include "../warp_kernel_vec.cc"\n'
        f'extern "C" void warp_kernel_vec_shim(bfloat16 *in_padded, int16_t *flow, bfloat16 *out) {{\n'
        f'    warp_kernel_vec(in_padded, flow, out);\n}}\n'
    )
    compile_flags = [f"-DWARP_TILE_W={TILE_W}", f"-DWARP_TILE_H={tile_h}",
                      f"-DWARP_HALO={HALO}", f"-DWARP_CH={ch}", "-Oz"]
    n_in = pad_h * pad_w * ch
    n_flow = 2 * tile_h * TILE_W
    n_out = tile_h * TILE_W * ch
    design = bl._build_oneshot(
        "warp_kernel_vec_shim", str(shim), [n_in, n_flow], n_out,
        [BF16, np.int16], BF16, compile_flags, stack_size=0x1000,
    )
    rng = np.random.default_rng(0)
    img = rng.standard_normal((n_in,)).astype(BF16)
    flow = (rng.standard_normal((n_flow,)) * (HALO * 0.5 * 128)).astype(np.int16)
    in_t = iron.tensor(np.ascontiguousarray(img), dtype=BF16, device="npu")
    flow_t = iron.tensor(np.ascontiguousarray(flow), dtype=np.int16, device="npu")
    out_t = iron.zeros((n_out,), dtype=BF16, device="npu")

    times = []
    for _ in range(N_REPS):
        t0 = time.perf_counter()
        design(in_t, flow_t, out_t)
        times.append(time.perf_counter() - t0)
    n_out_px = tile_h * TILE_W
    tmin = min(times)
    print(f"[C={ch} vec] tile={TILE_W}x{tile_h} ({n_out_px} px) wall_min={tmin*1e6:.1f}us "
          f"wall_med={sorted(times)[len(times)//2]*1e6:.1f}us "
          f"({tmin/n_out_px*1e9:.1f} ns/px, dispatch-inclusive host round trip)")
    return tmin, n_out_px


def sweep(ch):
    # Wide size gap (8 vs 48 rows): the vectorized kernel's per-pixel cost turned out small
    # enough that an 8-vs-16 gap (time_warp.py's scalar-kernel spacing) is swamped by host
    // dispatch jitter (~300-400us) -- see README for the 8-vs-16 measurement that came back
    # with a NEGATIVE marginal for C=3, i.e. pure noise, not a real per-pixel cost.
    r1 = time_one(ch, 8)
    r2 = time_one(ch, 48)
    d_time = r2[0] - r1[0]
    d_px = r2[1] - r1[1]
    ns_px = d_time / d_px * 1e9
    print(f"[C={ch} vec] marginal: {d_time*1e6:.1f}us over {d_px} extra px = "
          f"{ns_px:.1f} ns/px (fixed dispatch overhead ~"
          f"{r1[0]*1e6 - r1[1]*ns_px*1e-3:.1f}us)")
    return ns_px


if __name__ == "__main__":
    for ch in (3, 4):
        sweep(ch)
