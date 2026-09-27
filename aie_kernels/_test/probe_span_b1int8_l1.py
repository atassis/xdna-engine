#!/usr/bin/env python3
"""INT8_BLOCK1.md step 2: per-core L1 on tiles (0,3)/(0,4) (b1c1/b1c2), int16 vs int8 block-1,
at W=32 (the current shipped default: DEPTH["b1c1"]=4, DATA_SIZES={"b1c1": 4160},
skip_cons_depths={"conv_1": 3}). Compile-only (design.compile()), no device.

For each mode: (a) compile at the shipped defaults, pass/fail; (b) force a deliberate 1-byte
overflow (inflate DEPTH["b1c1"] by one slot) to make aiecc print tile (0,3)'s full MemoryMap, so
the byte accounting is READ off the compiler, not guessed (same technique TRACE_RESULTS.md uses).
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "designs" / "span_sr"))
import net_design as N  # noqa: E402
import net_layout as NL  # noqa: E402
import span_int as S  # noqa: E402
import test_span_int as TS  # noqa: E402

WIDTH, H = 32, 64

cal, _ = TS.load_split(os.environ["SPAN_DEMO_DIR"])


def run(b1_mode):
    b1_int8 = b1_mode == "int8"
    net = S.Span(os.environ["SPAN_EXPORT"], b1_mode=b1_mode)
    net.quantize(cal)
    NP = net.net_params()
    tag = f"b1l1_{b1_mode}"
    try:
        d = N.build(WIDTH, H, NP, HERE / "gen" / tag, tag=tag, b1_int8=b1_int8)
        d.compile()
        print(f"[{b1_mode}] shipped defaults (depth=4): OK", flush=True)
    except Exception as e:
        print(f"[{b1_mode}] shipped defaults (depth=4): FAIL -- {str(e)[:300]}", flush=True)

    # deliberate 1-slot overflow on b1c1's own output depth to surface the MemoryMap
    tag_of = f"{tag}_of"
    try:
        d = N.build(WIDTH, H, NP, HERE / "gen" / tag_of, tag=tag_of, b1_int8=b1_int8,
                   depths={"b1c1": 5})
        d.compile()
        print(f"[{b1_mode}] depth=5 (deliberate overflow probe): unexpectedly OK", flush=True)
    except Exception as e:
        msg = str(e)
        print(f"[{b1_mode}] depth=5 (deliberate overflow probe) error:\n{msg}", flush=True)


for mode in ("int16", "int8"):
    run(mode)
