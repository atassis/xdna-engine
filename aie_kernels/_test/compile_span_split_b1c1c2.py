#!/usr/bin/env python3
"""Compile-only (no device) check for net_design.build(split_gate={"b1c1","b1c2"}): the spec's
S5-iii L1 relief for tile (0,3), generalized from BALANCE.md's gate-only split.

Env: SPAN_EXPORT, SPAN_DEMO_DIR; optional SPAN_W (default 32).
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "designs" / "span_sr"))
import net_design as N  # noqa: E402
import span_int as S  # noqa: E402
import test_span_int as TS  # noqa: E402

WIDTH = int(os.environ.get("SPAN_W", 32))
HEIGHT = 64


def main():
    cal, _ = TS.load_split(os.environ["SPAN_DEMO_DIR"])
    net = S.Span(os.environ["SPAN_EXPORT"])
    net.quantize(cal)
    NP = net.net_params()
    design = N.build(WIDTH, HEIGHT, NP, HERE / "gen" / "span_net_split_b1", tag="spanb1",
                     split_gate={"b1c1", "b1c2"})
    design.compile()
    print(f"compile-only OK: b1c1+b1c2 split places at W={WIDTH}", flush=True)


if __name__ == "__main__":
    main()
