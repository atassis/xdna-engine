#!/usr/bin/env python3
"""Compile-only sanity check for bench_iterate_bds.py's design matrix. No device."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bench_iterate_bds as B  # noqa: E402

H = 8
for nsrc, rb, d in B.CASES_A + B.CASES_B:
    for it in (True, False):
        tag = f"cc_n{nsrc}_r{rb}_d{d}_{'it' if it else 'sl'}"
        try:
            design = B.build(nsrc, rb, d, it, H, tag)
            design.compile()
            print(f"[{tag}] OK", flush=True)
        except Exception as e:
            print(f"[{tag}] FAIL: {e}", flush=True)
