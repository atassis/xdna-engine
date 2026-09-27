#!/usr/bin/env python3
"""Compile-only check: pin the join's MemTile off the one shared with weight group 3
(TRACE_RESULTS.md's MemTile(4,1) finding) onto a dedicated candidate tile. No device."""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "designs" / "span_sr"))
import net_design as N  # noqa: E402
import net_layout as NL  # noqa: E402
import span_int as S  # noqa: E402
import test_span_int as TS  # noqa: E402
from aie.iron.device import Tile  # noqa: E402

cal, _ = TS.load_split(os.environ["SPAN_DEMO_DIR"])
net = S.Span(os.environ["SPAN_EXPORT"])
net.quantize(cal)
NP = net.net_params()

CASES = [("default", None), ("tile1_1", Tile(1, 1)), ("tile5_1", Tile(5, 1))]
for label, jt in CASES:
    try:
        design = N.build(32, 64, NP, HERE / "gen" / f"jt_{label}", tag=f"jt{label}", join_tile=jt)
        design.compile()
        print(f"[{label}] OK", flush=True)
    except Exception as e:
        print(f"[{label}] FAIL: {e}", flush=True)
