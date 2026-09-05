#!/usr/bin/env python3
"""Does splitting the accumulator recurrence actually buy cycles?

The trace showed 79.5% of a dispatch inside the kernel body with 0.83% stalls and only 25.6% of
the span issuing vector instructions -- a pipeline starved by the single accumulator carried across
all c_in*k taps. conv_1d_causal_core_vec_acc<N, NACC> splits that into NACC independent lanes.

Correctness is already gated on device: NACC=2 and 4 land at ~1e-7 across dilations 1/3/9 and k=1;
NACC=8 returns NaN at dilations 3 and 9 (it spills 8 vector registers inside the hot loop) and is
excluded here rather than timed.
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
rng = np.random.default_rng(61)
tiles = (rng.standard_normal((C_OUT, ROW)) * 0.02).astype(np.float32)
win = rng.standard_normal((C_IN, T)).astype(np.float32)
macs = C_OUT * C_IN * K * T
print(f"c_in={C_IN} k={K} T={T} c_out={C_OUT}  {macs/1e6:.2f} M MACs/dispatch", flush=True)
print(f"{'core':>22} {'device_ms':>10} {'speedup':>8} {'MAC/cyc':>8}", flush=True)

base = None
for label, call in (
    ("vec (1 accumulator)", f"conv_1d_causal_core_vec<16>(resident, tile, tile[{C_IN*K}], out, {C_IN}, {K}, {T}, 1)"),
    ("vec_acc NACC=2",      f"conv_1d_causal_core_vec_acc<16, 2>(resident, tile, tile[{C_IN*K}], out, {C_IN}, {K}, {T}, 1)"),
    ("vec_acc NACC=4",      f"conv_1d_causal_core_vec_acc<16, 4>(resident, tile, tile[{C_IN*K}], out, {C_IN}, {K}, {T}, 1)"),
):
    sym = "acc_" + label.split()[-1].replace("=", "").replace("(", "").replace(")", "")
    shim = bricklib.GEN / f"{sym}_shim.cc"
    shim.write_text(f"#include <stdint.h>\n#include \"{wd.CONV_CC}\"\n"
                    f'extern "C" void {sym}(float *tile, float *resident, float *out) {{\n'
                    f"  route_b_bricks::{call};\n}}\n")
    try:
        d = bricklib._build_streamed(sym, shim, C_OUT, ROW, T, C_IN*T, None,
                                     np.float32, np.float32, np.float32, 1)
        in_t = bricklib.iron.tensor(tiles.reshape(-1), dtype=np.float32, device="npu")
        r_t = bricklib.iron.tensor(win.reshape(-1), dtype=np.float32, device="npu")
        out_t = bricklib.iron.zeros((C_OUT*T,), dtype=np.float32, device="npu")
        d(in_t, r_t, out_t); _dev["ms"] = 0.0
        d(in_t, r_t, out_t)
        got = out_t.numpy().copy(); ms = _dev["ms"]; base = base or ms
        print(f"{label:>22} {ms:10.1f} {base/ms:8.2f}x {macs/(ms/1e3)/1.8e9:8.4f}"
              f"   nz={np.abs(got).sum():.4e}", flush=True)
    except Exception as e:
        print(f"{label:>22}  FAILED: {type(e).__name__}: {str(e)[:70]}", flush=True)
