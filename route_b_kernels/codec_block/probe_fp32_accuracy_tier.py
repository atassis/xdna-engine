#!/usr/bin/env python3
"""aie2p has no native f32 vector MAC -- does a cheaper emulation tier pay?

The API exposes mac_elem_32_accuracy_{low,fast,safe}; accuracy tiers are the signature of
emulation, and this kernel is `float` throughout so it takes the default. Splitting the
accumulator recurrence bought nothing (NACC=2 -> 0.98x, NACC=4 -> 0.91x), which kills the
latency-chain explanation and leaves the emulated op COUNT as the cost.

mlir-aie's own test config passes -DAIE2_FP32_EMULATION_ACCURACY_FAST, so the tier is a
compile-time choice. This times the same kernel under each tier and reports rel-L2 against the
default-tier result, because a cheaper tier is only interesting if the 3e-2 gate still holds --
the decoder chain currently sits at 3.304e-04, about 100x of headroom.
"""
import sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from aie.utils.hostruntime.xrtruntime import hostruntime as hr
_dev = {"ms": 0.0}
_orig = hr.CachedXRTRuntime.run
def _timed(self, *a, **k):
    t0 = time.perf_counter()
    try:
        return _orig(self, *a, **k)
    finally:
        _dev["ms"] += (time.perf_counter() - t0) * 1e3
hr.CachedXRTRuntime.run = _timed

import window_driver as wd
import bricklib

C_IN, K, T, C_OUT = 128, 7, wd.T, 384
ROW = C_IN * K + 1
rng = np.random.default_rng(71)
tiles = (rng.standard_normal((C_OUT, ROW)) * 0.02).astype(np.float32)
win = rng.standard_normal((C_IN, T)).astype(np.float32)
macs = C_OUT * C_IN * K * T
print(f"c_in={C_IN} k={K} T={T} c_out={C_OUT}  {macs/1e6:.2f} M MACs/dispatch", flush=True)
print(f"{'tier':>34} {'device_ms':>10} {'speedup':>8} {'MAC/cyc':>8} {'rel-L2 vs default':>18}", flush=True)

base_ms, ref = None, None
for label, flags in (
    ("default (safe)", []),
    ("-DAIE2_FP32_EMULATION_ACCURACY_FAST", ["-DAIE2_FP32_EMULATION_ACCURACY_FAST"]),
    ("-DAIE2_FP32_EMULATION_ACCURACY_LOW", ["-DAIE2_FP32_EMULATION_ACCURACY_LOW"]),
):
    sym = "tier" + str(abs(hash(label)) % 10000)
    shim = bricklib.GEN / f"{sym}_shim.cc"
    shim.write_text(f"#include <stdint.h>\n#include \"{wd.CONV_CC}\"\n"
                    f'extern "C" void {sym}(float *tile, float *resident, float *out) {{\n'
                    f"  route_b_bricks::conv_1d_causal_core_vec<16>(resident, tile,\n"
                    f"      tile[{C_IN*K}], out, {C_IN}, {K}, {T}, 1);\n}}\n")
    try:
        d = bricklib._build_streamed(sym, shim, C_OUT, ROW, T, C_IN*T, flags or None,
                                     np.float32, np.float32, np.float32, 1)
        in_t = bricklib.iron.tensor(tiles.reshape(-1), dtype=np.float32, device="npu")
        r_t = bricklib.iron.tensor(win.reshape(-1), dtype=np.float32, device="npu")
        out_t = bricklib.iron.zeros((C_OUT*T,), dtype=np.float32, device="npu")
        d(in_t, r_t, out_t); _dev["ms"] = 0.0
        d(in_t, r_t, out_t)
        got = out_t.numpy().copy().astype(np.float64); ms = _dev["ms"]
        base_ms = base_ms or ms
        if ref is None:
            ref, rl2 = got, 0.0
        else:
            rl2 = float(np.linalg.norm(got - ref) / (np.linalg.norm(ref) or 1.0))
        print(f"{label:>34} {ms:10.1f} {base_ms/ms:8.2f}x {macs/(ms/1e3)/1.8e9:8.4f} {rl2:18.3e}",
              flush=True)
    except Exception as e:
        print(f"{label:>34}  FAILED: {type(e).__name__}: {str(e)[:60]}", flush=True)
