#!/usr/bin/env python3
"""SPAN x2 on the array, brought up front to back: the chain up to each stage in SPAN_UPTO
(default: every stage in net_layout.GOLDEN, in order) on a WIDTH x HEIGHT crop of the demo's test
half, treated as the whole frame, must equal span_int on that crop exactly. Stops at the first
stage that disagrees.

Env: SPAN_EXPORT, SPAN_DEMO_DIR; optional SPAN_UPTO (comma list), SPAN_HEIGHT (default 64),
SPAN_B1_MODE (int16 [default] or int8 -- INT8_BLOCK1.md's block-1 quantization option).
"""
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "designs" / "span_sr"))
import aie.iron as iron  # noqa: E402
import net_design as N  # noqa: E402
import net_layout as NL  # noqa: E402
import span_int as S  # noqa: E402
import test_span_int as TS  # noqa: E402

WIDTH = 32
HEIGHT = int(os.environ.get("SPAN_HEIGHT", 64))


def run(net, NP, crop, T, upto, b1_int8):
    names = NL.stage_names(upto)
    design = N.build(WIDTH, HEIGHT, NP, HERE / "gen" / "span_net", upto=upto, b1_int8=b1_int8)
    x = NL.conv1_rows(crop, net.mean255).reshape(-1).view(np.int8)
    xt = iron.tensor(x, dtype=np.int8, device="npu")
    wt = iron.tensor(NL.weights_blob(NP, names), dtype=np.int8, device="npu")
    out_bytes = NL.layout(NL.stages(b1_int8)[upto], WIDTH).out_bytes
    yt = iron.zeros((HEIGHT * out_bytes,), dtype=np.int8, device="npu")
    design(xt, wt, yt)
    got = NL.unpack_out(yt.numpy(), upto, HEIGHT, WIDTH, b1_int8=b1_int8).astype(np.int64)
    ref = T[NL.golden(b1_int8)[upto][0]].astype(np.int64)
    mism = int((got != ref).sum())
    print(f"span net upto {upto} ({len(names)} cores), {WIDTH}x{HEIGHT}: exact "
          f"{ref.size - mism}/{ref.size} -> {'PASS' if mism == 0 else 'FAIL'}", flush=True)
    return mism == 0


def main():
    b1_mode = os.environ.get("SPAN_B1_MODE", "int16")
    b1_int8 = b1_mode == "int8"
    cal, test = TS.load_split(os.environ["SPAN_DEMO_DIR"])
    net = S.Span(os.environ["SPAN_EXPORT"], b1_mode=b1_mode)
    net.quantize(cal)
    crop = test[:, :HEIGHT, :WIDTH]
    T = net.int_tensors(crop)
    NP = net.net_params()
    for upto in os.environ.get("SPAN_UPTO", ",".join(NL.GOLDEN)).split(","):
        if not run(net, NP, crop, T, upto, b1_int8):
            sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
