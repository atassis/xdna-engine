#!/usr/bin/env python3
"""Compile-only sanity check for trace_iterate_bds.py's design matrix. No device."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import trace_iterate_bds as T  # noqa: E402
from aie.utils.trace.config import TraceConfig  # noqa: E402

H = 8
CASES = [(4, 2304, 4), (4, 2304, 8)]
for nsrc, rb, d in CASES:
    for it in (True, False):
        tag = f"cctr_n{nsrc}_r{rb}_d{d}_{'it' if it else 'sl'}"
        tc = TraceConfig(trace_size=1048576, trace_file=f"/tmp/{tag}.txt")
        try:
            design = T.build(nsrc, rb, d, it, H, tag, trace_config=tc)
            design.compile()
            print(f"[{tag}] OK", flush=True)
        except Exception as e:
            print(f"[{tag}] FAIL: {e}", flush=True)
