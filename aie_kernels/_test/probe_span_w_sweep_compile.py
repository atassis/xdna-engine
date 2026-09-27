#!/usr/bin/env python3
"""Phase 1d compile-only gate: which W in {16,32,48,64} fits L1 for the SPAN_UPTO=b1c3 4-core
prefix, at main's defaults (no depth/skip overrides -- exactly what verify_span_net.py ships).
No device. Reports pass/fail per width and the aiecc error for any failure, mirroring the
compile-only sweep methodology already used throughout TRACE_RESULTS.md (MAIN_DEPTH/b1c1-depth/
SKIP_SLACK sections).
"""
import os
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "designs" / "span_sr"))
import net_design as N  # noqa: E402
import net_layout as NL  # noqa: E402
import span_int as S  # noqa: E402
import test_span_int as TS  # noqa: E402

UPTO = "b1c3"
WIDTHS = [16, 32, 48, 64]
H = 64

cal, _ = TS.load_split(os.environ["SPAN_DEMO_DIR"])
net = S.Span(os.environ["SPAN_EXPORT"])
net.quantize(cal)
NP = net.net_params()

for w in WIDTHS:
    try:
        design = N.build(w, H, NP, HERE / "gen" / f"wsweep_compile_w{w}", tag=f"wsweepc{w}",
                         upto=UPTO)
        design.compile()
        print(f"[w-sweep-compile] W={w:3d} upto={UPTO}: OK", flush=True)
    except Exception as e:
        msg = str(e).replace("\n", " | ")
        print(f"[w-sweep-compile] W={w:3d} upto={UPTO}: FAIL -- {msg[:400]}", flush=True)
