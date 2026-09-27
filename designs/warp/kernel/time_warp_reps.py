#!/usr/bin/env python3
"""Repeat-count instrument: the marginal two-tile-size method broke down on the
vectorized kernel (128 extra px is tens of us of real compute under ~300-400us of host
dispatch jitter -- time_warp_vec.py's C=3 point came back with a NEGATIVE marginal, pure
noise). Fix: bake a REPS loop calling the kernel REPS times into ONE dispatch (same tile,
same dispatch overhead every build); (T(K) - T(1)) / (K-1) isolates device-side per-call
time, since only the compiled trip count differs between the two builds.

Runs BOTH kernels at ONE tile size each (no marginal-tile-size needed once REPS supplies
the isolation) -- scalar warp_kernel.cc as the CONTROL (must reproduce time_warp.py's
~879/1499 ns/px before the vec numbers are trusted) and warp_kernel_vec.cc as the target.
"""
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
N_REPS_HOST = 9  # host-side repeat calls per build, for wall_min


def time_scalar(ch, tile_w, tile_h, halo, reps):
    pad_w, pad_h = tile_w + 2 * halo, tile_h + 2 * halo
    shim = GEN / f"time_scalar_reps_c{ch}_{reps}.cc"
    shim.write_text(
        f'#include <stdint.h>\n#define WARP_TILE_W {tile_w}\n#define WARP_TILE_H {tile_h}\n'
        f'#define WARP_HALO {halo}\n#define WARP_CH {ch}\n#include "../warp_kernel.cc"\n'
        f'extern "C" void warp_scalar_reps_shim(float *in_padded, float *flow, float *out) {{\n'
        f'    for (int r = 0; r < {reps}; r++) warp_kernel(in_padded, flow, out);\n}}\n'
    )
    compile_flags = [f"-DWARP_TILE_W={tile_w}", f"-DWARP_TILE_H={tile_h}",
                      f"-DWARP_HALO={halo}", f"-DWARP_CH={ch}", "-Oz"]
    n_in, n_flow, n_out = pad_h * pad_w * ch, 2 * tile_h * tile_w, tile_h * tile_w * ch
    design = bl._build_oneshot("warp_scalar_reps_shim", str(shim), [n_in, n_flow], n_out,
                               [np.float32, np.float32], np.float32, compile_flags,
                               stack_size=0x1000)
    rng = np.random.default_rng(0)
    img = rng.standard_normal((n_in,)).astype(np.float32)
    flow = (rng.standard_normal((n_flow,)) * (halo * 0.5)).astype(np.float32)
    in_t = iron.tensor(np.ascontiguousarray(img), dtype=np.float32, device="npu")
    flow_t = iron.tensor(np.ascontiguousarray(flow), dtype=np.float32, device="npu")
    out_t = iron.zeros((n_out,), dtype=np.float32, device="npu")
    times = [_time_call(design, in_t, flow_t, out_t) for _ in range(N_REPS_HOST)]
    return min(times), tile_h * tile_w


def time_vec(ch, tile_w, tile_h, halo, reps):
    pad_w, pad_h = tile_w + 2 * halo, tile_h + 2 * halo
    shim = GEN / f"time_vec_reps_c{ch}_{reps}.cc"
    shim.write_text(
        f'#include <stdint.h>\n#define WARP_TILE_W {tile_w}\n#define WARP_TILE_H {tile_h}\n'
        f'#define WARP_HALO {halo}\n#define WARP_CH {ch}\n#include "../warp_kernel_vec.cc"\n'
        f'extern "C" void warp_vec_reps_shim(bfloat16 *in_padded, int16_t *flow, bfloat16 *out) {{\n'
        f'    for (int r = 0; r < {reps}; r++) warp_kernel_vec(in_padded, flow, out);\n}}\n'
    )
    compile_flags = [f"-DWARP_TILE_W={tile_w}", f"-DWARP_TILE_H={tile_h}",
                      f"-DWARP_HALO={halo}", f"-DWARP_CH={ch}", "-Oz"]
    n_in, n_flow, n_out = pad_h * pad_w * ch, 2 * tile_h * tile_w, tile_h * tile_w * ch
    design = bl._build_oneshot("warp_vec_reps_shim", str(shim), [n_in, n_flow], n_out,
                               [BF16, np.int16], BF16, compile_flags, stack_size=0x1000)
    rng = np.random.default_rng(0)
    img = rng.standard_normal((n_in,)).astype(BF16)
    flow = (rng.standard_normal((n_flow,)) * (halo * 0.5 * 128)).astype(np.int16)
    in_t = iron.tensor(np.ascontiguousarray(img), dtype=BF16, device="npu")
    flow_t = iron.tensor(np.ascontiguousarray(flow), dtype=np.int16, device="npu")
    out_t = iron.zeros((n_out,), dtype=BF16, device="npu")
    times = [_time_call(design, in_t, flow_t, out_t) for _ in range(N_REPS_HOST)]
    return min(times), tile_h * tile_w


def _time_call(design, *args):
    t0 = time.perf_counter()
    design(*args)
    return time.perf_counter() - t0


def marginal(fn, ch, tile_w, tile_h, halo, k_lo, k_hi, label):
    t_lo, npx = fn(ch, tile_w, tile_h, halo, k_lo)
    t_hi, _ = fn(ch, tile_w, tile_h, halo, k_hi)
    per_call = (t_hi - t_lo) / (k_hi - k_lo)
    ns_px = per_call / npx * 1e9
    print(f"[{label} C={ch}] K={k_lo}: {t_lo*1e6:.1f}us  K={k_hi}: {t_hi*1e6:.1f}us  "
          f"per-call={per_call*1e6:.2f}us over {npx}px -> {ns_px:.1f} ns/px "
          f"({ns_px*1.8:.0f} cyc/px @1.8GHz)")
    return ns_px


if __name__ == "__main__":
    TILE_W, TILE_H, HALO = 16, 16, 8
    K_LO, K_HI = 4, 800
    print("=== scalar control (must reproduce ~879/1499 ns/px) ===")
    for ch in (3, 4):
        marginal(time_scalar, ch, TILE_W, TILE_H, HALO, K_LO, K_HI, "scalar")
    print("=== vectorized ===")
    for ch in (3, 4):
        marginal(time_vec, ch, TILE_W, TILE_H, HALO, K_LO, K_HI, "vec")
