#!/usr/bin/env python3
"""Compile-only (no device) check for net_design.build(split_gate=...): does the b2c3 output-
channel split (BALANCE.md option (a)) place within L1/MemTile/DMA budgets at production W=32.

Env: SPAN_EXPORT, SPAN_DEMO_DIR.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "designs" / "span_sr"))
import net_design as N  # noqa: E402
import span_int as S  # noqa: E402
import test_span_int as TS  # noqa: E402

WIDTH, HEIGHT = 32, 64


def main():
    cal, _ = TS.load_split(os.environ["SPAN_DEMO_DIR"])
    net = S.Span(os.environ["SPAN_EXPORT"])
    net.quantize(cal)
    NP = net.net_params()
    design = N.build(WIDTH, HEIGHT, NP, HERE / "gen" / "span_net_split", tag="spansplit",
                     split_gate={"b2c3"})
    design.compile()
    print("compile-only OK: b2c3 split places at W=32", flush=True)


if __name__ == "__main__":
    main()
