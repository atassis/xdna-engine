#!/usr/bin/env python3
"""Kernel rate of each SPAN block core at the block's own shape (48->48, W=32), as a fitted slope
of wall time vs in-core REPS (probe_conv3x3_rate's method: dispatch and DMA cancel). plain is
conv3x3_i8, lut the SiLU cores (c1, c2), gate the c3 core; the chained block cannot beat the
slowest. Cycles assume 1.8 GHz, printed. Prints, exits 0."""
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "designs" / "span_sr"))
import aie.iron as iron  # noqa: E402
import bricklib  # noqa: E402
import block_design as B  # noqa: E402
import span_int as S  # noqa: E402
import test_span_int as TS  # noqa: E402

W, ROWS, CLOCK, TRIALS, REPS = 32, 64, 1.8e9, 7, [1, 9, 17, 25]
HALF = B.row_bytes(W)
g = B._golden()
cal, _ = TS.load_split(os.environ["SPAN_DEMO_DIR"])
net = S.Span(os.environ["SPAN_EXPORT"])
net.quantize(cal)
P = net.core_params(2)
rng = np.random.default_rng(0)


def timeit(design, args):
    design(*args)
    ts = []
    for _ in range(TRIALS):
        t0 = time.perf_counter()
        design(*args)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def core_rate(kind):
    p = P["c3" if kind == "gate" else "c1"]
    inc = g.lut_inc(p["table"], bricklib.GEN / f"ctl_{kind}_lut.inc")
    nlines = 4 if kind == "gate" else 3
    xs, ys = [], []
    for reps in REPS:
        sym = f"ctl_{kind}_r{reps}"
        shim = bricklib.GEN / f"{sym}_shim.cc"
        if kind == "plain":
            call = f"conv3x3_i8(t, t + {HALF}, t + {2*HALF}, p, o, {W}, 1, {p['pre']}, {p['shift']}, 0, {W});"
        elif kind == "lut":
            call = f"conv3x3_i8_lut(t, t + {HALF}, t + {2*HALF}, p, o, {W}, 1, {p['pre']}, {p['shift']}, 0, {W});"
        else:
            call = (f"conv3x3_i8_gate(t, t + {HALF}, t + {2*HALF}, t + {3*HALF}, p, o, {W}, 1, {p['pre']}, "
                    f"{p['shift']}, 0, {W}, {p['ga']}, {p['gb']}, {p['gs1']}, {p['gc']}, {p['gs2']});")
        shim.write_text(f'#include <stdint.h>\n#include "{B.KDIR / "conv3x3_u8.cc"}"\n'
                        f'extern "C" void {sym}(int8_t *t, int8_t *p, int8_t *o) {{\n'
                        f'  for (int r = 0; r < {reps}; r++)\n    {call}\n}}\n')
        d = bricklib._build_streamed(sym, shim, ROWS, nlines * HALF, HALF, P["c1"]["blob"].size,
                                     ["-DCONV3X3_CIN=48", "-DCONV3X3_COUT=48", f'-DCONV3X3_LUT_INC="{inc}"'],
                                     np.int8, np.int8, np.int8, resident_depth=1, stack_size=3584)
        tiles = rng.integers(-40, 40, size=ROWS * nlines * HALF, dtype=np.int64).astype(np.int8)
        a = [iron.tensor(tiles, dtype=np.int8, device="npu"),
             iron.tensor(p["blob"], dtype=np.int8, device="npu"),
             iron.zeros((ROWS * HALF,), dtype=np.int8, device="npu")]
        xs.append(reps); ys.append(timeit(d, a))
    slope, icpt = np.polyfit(xs, ys, 1)
    print(f"core {kind:5s}: {[round(y*1e3,3) for y in ys]} ms -> {slope / (ROWS*W) * CLOCK:.0f} cyc/px "
          f"({9*48*48 / (slope / (ROWS*W) * CLOCK):.0f} MAC/cyc)", flush=True)

for k in ("plain", "lut", "gate"):
    core_rate(k)

