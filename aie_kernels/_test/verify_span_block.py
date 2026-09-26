#!/usr/bin/env python3
"""SPAN block 2 as one design on a real strip: exact equality with span_int.block_int.

Env: SPAN_EXPORT (SPAN export dir), SPAN_DEMO_DIR (dir with hr.png). The strip is the first
WIDTH columns and HEIGHT rows of block 2's integer input for the demo test half; the golden runs
on that same crop, so both see the crop's edges as zero padding.
"""
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "designs" / "span_sr"))
import aie.iron as iron  # noqa: E402
import block_design as B  # noqa: E402
import span_int as S  # noqa: E402
import test_span_int as TS  # noqa: E402

WIDTH, HEIGHT, BLOCK = 32, 64, 2


def main():
    cal, test = TS.load_split(os.environ["SPAN_DEMO_DIR"])
    net = S.Span(os.environ["SPAN_EXPORT"])
    net.quantize(cal)
    x = net.int_tensors(test)[f"b{BLOCK}.in"][:, :HEIGHT, :WIDTH].astype(np.int8)
    ref, _ = net.block_int(BLOCK, x)
    P = net.core_params(BLOCK)
    g = B._golden()
    rows = g.pack_rows(x).reshape(-1)
    design = B.build(WIDTH, HEIGHT, P, HERE / "gen" / "span_block")
    xt = iron.tensor(rows, dtype=np.int8, device="npu")
    pt = [iron.tensor(P[k]["blob"], dtype=np.int8, device="npu") for k in ("c1", "c2", "c3")]
    yt = iron.zeros((HEIGHT * B.row_bytes(WIDTH),), dtype=np.int8, device="npu")
    design(xt, *pt, yt)
    got = g.unpack_rows(yt.numpy().reshape(-1), 48, HEIGHT, WIDTH).astype(np.int64)
    mism = int((got != ref.astype(np.int64)).sum())
    print(f"span block {BLOCK}, {WIDTH}x{HEIGHT}: exact {ref.size - mism}/{ref.size} -> "
          f"{'PASS' if mism == 0 else 'FAIL'}")
    sys.exit(0 if mism == 0 else 1)


if __name__ == "__main__":
    main()
