#!/usr/bin/env python3
"""Repeat-count timing instrument -- SUPERSEDES time_fsr1_vec.py's, time_fsr1_rcas_vec.py's and
time_fsr1_strip_vec.py's marginal-two-crop-size numbers. Per wt-npu-warp's finding
(designs/warp/kernel/time_warp_reps.py, commit e5fa996): the marginal method's per-pixel delta is
tens of us of real compute under ~300-400us of host dispatch jitter, and FSR1's probe crops are
smaller than warp's (which already broke down), so its numbers are worse. Fix: bake a REPS loop
calling the kernel REPS times into ONE dispatch at ONE fixed crop (same dispatch overhead every
build); (T(K_hi) - T(K_lo)) / (K_hi - K_lo) isolates device-side per-call time, since only the
compiled trip count differs between builds. A 3-point linearity check (K=4/200/800) confirms the
slope is a real per-call cost, not a K_lo/K_hi artifact.
"""
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
N_REPS_HOST = 9  # host-side repeat calls per build, for wall_min

# One fixed crop, known to fit every kernel's data-memory budget (verify_fsr1_vec.py /
# verify_fsr1_rcas_vec.py / verify_fsr1_strip_vec.py all build at this size).
IN_W, IN_H = 16, 6
OUT_W, OUT_H = IN_W * 3, IN_H * 3


def _time_call(design, *args):
    t0 = time.perf_counter()
    design(*args)
    return time.perf_counter() - t0


def time_easu(reps):
    shim = GEN / f"time_easu_reps_{reps}.cc"
    shim.write_text(
        f'#include <stdint.h>\n#define FSR1_IN_W {IN_W}\n#define FSR1_IN_H {IN_H}\n'
        '#include "../fsr1_kernel_vec.cc"\n'
        f'extern "C" void fsr1_easu_reps_shim(float *in_rgb, float *out_rgb) {{\n'
        f'    for (int r = 0; r < {reps}; r++) fsr1_easu_vec(in_rgb, out_rgb);\n}}\n'
    )
    compile_flags = [f"-DFSR1_IN_W={IN_W}", f"-DFSR1_IN_H={IN_H}", "-Oz"]
    design = bl._build_oneshot(
        "fsr1_easu_reps_shim", str(shim), [IN_W * IN_H * 3], OUT_W * OUT_H * 3,
        [np.float32], np.float32, compile_flags, stack_size=0x4000,
    )
    rng = np.random.default_rng(0)
    img = (rng.integers(20, 235, size=(IN_H * IN_W * 3,)).astype(np.float32) / 255.0)
    in_t = iron.tensor(np.ascontiguousarray(img), dtype=np.float32, device="npu")
    out_t = iron.zeros((OUT_W * OUT_H * 3,), dtype=np.float32, device="npu")
    times = [_time_call(design, in_t, out_t) for _ in range(N_REPS_HOST)]
    return min(times), OUT_W * OUT_H


def time_rcas(reps):
    shim = GEN / f"time_rcas_reps_{reps}.cc"
    shim.write_text(
        f'#include <stdint.h>\n#define FSR1_IN_W {IN_W}\n#define FSR1_IN_H {IN_H}\n'
        '#include "../fsr1_kernel_vec.cc"\n'
        f'extern "C" void fsr1_rcas_reps_shim(float *in_rgb, float *out_rgb) {{\n'
        f'    for (int r = 0; r < {reps}; r++) fsr1_rcas_vec(in_rgb, out_rgb);\n}}\n'
    )
    compile_flags = [f"-DFSR1_IN_W={IN_W}", f"-DFSR1_IN_H={IN_H}", "-Oz"]
    design = bl._build_oneshot(
        "fsr1_rcas_reps_shim", str(shim), [OUT_W * OUT_H * 3], OUT_W * OUT_H * 3,
        [np.float32], np.float32, compile_flags, stack_size=0x4000,
    )
    rng = np.random.default_rng(0)
    img = rng.random(size=(OUT_H * OUT_W * 3,)).astype(np.float32)
    in_t = iron.tensor(np.ascontiguousarray(img), dtype=np.float32, device="npu")
    out_t = iron.zeros((OUT_W * OUT_H * 3,), dtype=np.float32, device="npu")
    times = [_time_call(design, in_t, out_t) for _ in range(N_REPS_HOST)]
    return min(times), OUT_W * OUT_H


def time_strip(reps):
    shim = GEN / f"time_strip_reps_{reps}.cc"
    shim.write_text(
        f'#include <stdint.h>\n#define FSR1_IN_W {IN_W}\n#define FSR1_IN_H {IN_H}\n'
        '#include "../fsr1_kernel_vec.cc"\n'
        f'extern "C" void fsr1_strip_reps_shim(float *in_rgb, float *out_rgb) {{\n'
        f'    for (int r = 0; r < {reps}; r++) fsr1_strip_vec(in_rgb, out_rgb);\n}}\n'
    )
    compile_flags = [f"-DFSR1_IN_W={IN_W}", f"-DFSR1_IN_H={IN_H}", "-Oz"]
    design = bl._build_oneshot(
        "fsr1_strip_reps_shim", str(shim), [IN_W * IN_H * 3], OUT_W * OUT_H * 3,
        [np.float32], np.float32, compile_flags, stack_size=0x4000,
    )
    rng = np.random.default_rng(0)
    img = (rng.integers(20, 235, size=(IN_H * IN_W * 3,)).astype(np.float32) / 255.0)
    in_t = iron.tensor(np.ascontiguousarray(img), dtype=np.float32, device="npu")
    out_t = iron.zeros((OUT_W * OUT_H * 3,), dtype=np.float32, device="npu")
    times = [_time_call(design, in_t, out_t) for _ in range(N_REPS_HOST)]
    return min(times), OUT_W * OUT_H


def marginal(fn, k_lo, k_hi, label):
    t_lo, npx = fn(k_lo)
    t_hi, _ = fn(k_hi)
    per_call = (t_hi - t_lo) / (k_hi - k_lo)
    ns_px = per_call / npx * 1e9
    print(f"[{label}] K={k_lo}: {t_lo*1e6:.1f}us  K={k_hi}: {t_hi*1e6:.1f}us  "
          f"per-call={per_call*1e6:.2f}us over {npx}px -> {ns_px:.1f} ns/px "
          f"({ns_px*1.8:.0f} cyc/px @1.8GHz)")
    return ns_px


def linearity_check(fn, ks, label):
    pts = [(k, fn(k)[0]) for k in ks]
    print(f"[{label}] " + "  ".join(f"K={k}:{t*1e6:.1f}us" for k, t in pts))
    (k0, t0), (k1, t1), (k2, t2) = pts
    slope_01 = (t1 - t0) / (k1 - k0)
    slope_12 = (t2 - t1) / (k2 - k1)
    ratio = slope_12 / slope_01 if slope_01 else float("nan")
    print(f"[{label}] per-call slope K{k0}->{k1}: {slope_01*1e9:.2f}ns  "
          f"K{k1}->{k2}: {slope_12*1e9:.2f}ns  ratio={ratio:.3f} (1.0 = linear)")


if __name__ == "__main__":
    K_LO, K_HI = 4, 800
    for fn, label in ((time_easu, "easu"), (time_rcas, "rcas"), (time_strip, "strip")):
        print(f"=== {label} ===")
        marginal(fn, K_LO, K_HI, label)
        linearity_check(fn, (4, 200, 800), label)
