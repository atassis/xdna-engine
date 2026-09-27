#!/usr/bin/env python3
"""Phase 1j: re-trace the four pace-setting kinds (b1c2 silu_i16, b1c3 gate, b1c1 silu16,
b2c2 silu) after hoisting the Look/parallel_lookup construction out of apply_lut_inplace/
apply_lut16/the gate epilogue to once per conv3x3_core call. Same method/config as Phase 1i
(W=32 H=128, one stage per dispatch, default net_design.py config) so the numbers are directly
comparable to Phase 1i's row in TRACE_RESULTS.md. run.sh drops argv, so this calls
trace_span_net.main() directly instead of going through its __main__ argparse block."""
import argparse
from pathlib import Path
import trace_span_net as T

HERE = Path(__file__).resolve().parent
STAGES = ["b1c2", "b1c3", "b1c1", "b2c2"]

for stage in STAGES:
    o = argparse.Namespace(stages=stage, width=32, height=128,
                           out=str(HERE / "trace-out-1j"), trace_size=1048576,
                           egress_shim_col=1, clock_ghz=1.8)
    T.main(o)
