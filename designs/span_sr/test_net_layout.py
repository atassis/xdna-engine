"""CPU checks for net_layout and Span.net_params. usage: python3 test_net_layout.py <export> [demo]"""
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import span_int as S  # noqa: E402
import test_span_int as TS  # noqa: E402

W, H = 32, 24


def main():
    export = sys.argv[1]
    demo = sys.argv[2] if len(sys.argv) > 2 else os.environ["SPAN_DEMO_DIR"]
    cal, test = TS.load_split(demo)
    net = S.Span(export)
    net.quantize(cal)
    NP = net.net_params()
    assert len(NP) == 22, sorted(NP)
    assert NP["conv_1"]["blob"].size == 48 * 3 * 8 * 3 + 6 * 64 * 6       # CIN 8
    assert NP["b2c1"]["blob"].size == 48 * 3 * 48 * 3 + 6 * 64 * 6
    assert NP["conv_cat"]["blob"].size == 48 * 4 * 48 + 6 * 64 * 6        # 1x1 over 4 x 48
    assert NP["up"]["blob"].size == 16 * 3 * 48 * 3 + 2 * 64 * 6          # COUT 16
    assert len(NP["b1c1"]["tables"]) == 2 and len(NP["b2c1"]["tables"]) == 1
    assert "tables" not in NP["conv_2"]
    assert {"ga", "gb", "gs1", "gc", "gs2"} <= set(NP["b4c3"])
    print("test_net_layout: OK")


if __name__ == "__main__":
    main()
