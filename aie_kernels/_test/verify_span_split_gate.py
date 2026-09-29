#!/usr/bin/env python3
"""Device gate for net_design.build(split_gate={"b2c3"}) (BALANCE.md option (a)): the full net
(upto="up") on a real strip must still equal span_int exactly, same as verify_span_net.py's own
gate for the unsplit net.

Env: SPAN_EXPORT, SPAN_DEMO_DIR; optional SPAN_HEIGHT (default 64).
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


def main():
    cal, test = TS.load_split(os.environ["SPAN_DEMO_DIR"])
    net = S.Span(os.environ["SPAN_EXPORT"])
    net.quantize(cal)
    crop = test[:, :HEIGHT, :WIDTH]
    T = net.int_tensors(crop)
    NP = net.net_params()
    upto = "up"
    names = NL.stage_names(upto)
    design = N.build(WIDTH, HEIGHT, NP, HERE / "gen" / "span_net_split", upto=upto,
                     split_gate={"b2c3"})
    x = NL.conv1_rows(crop, net.mean255).reshape(-1).view(np.int8)
    xt = iron.tensor(x, dtype=np.int8, device="npu")
    wt = iron.tensor(N.weights_blob(NP, names, split_gate={"b2c3"}), dtype=np.int8, device="npu")
    out_bytes = NL.layout(dict(NL.STAGES)[upto], WIDTH).out_bytes
    yt = iron.zeros((HEIGHT * out_bytes,), dtype=np.int8, device="npu")
    design(xt, wt, yt)
    got = NL.unpack_out(yt.numpy(), upto, HEIGHT, WIDTH).astype(np.int64)
    ref = T[NL.GOLDEN[upto][0]].astype(np.int64)
    mism = int((got != ref).sum())
    print(f"split_gate={{'b2c3'}} net upto {upto} ({len(names) + 1} cores), {WIDTH}x{HEIGHT}: "
         f"exact {ref.size - mism}/{ref.size} -> {'PASS' if mism == 0 else 'FAIL'}", flush=True)
    sys.exit(0 if mism == 0 else 1)


if __name__ == "__main__":
    main()
