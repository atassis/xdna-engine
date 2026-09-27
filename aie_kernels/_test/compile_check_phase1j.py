"""Phase 1i step 0: compile-only sanity check for the conv3x3_u8.cc epilogue-loop restrict/
C3_LOOP_RANGE edits (silu_i16/silu/silu_x LUT gather, silu16 LUT16 gather, gate epilogue).
Full aiecc pipeline through address allocation via design.compile() (no device), at main's
default net_design.py config -- catches any pragma/restrict-triggered compile regression before
spending device time. See designs/span_sr/TRACE_RESULTS.md, Phase 1i.
"""
import os, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "designs" / "span_sr"))
import aie.iron as iron  # noqa
import net_design as N
import span_int as S
import test_span_int as TS

cal, _ = TS.load_split(os.environ["SPAN_DEMO_DIR"])
net = S.Span(os.environ["SPAN_EXPORT"])
net.quantize(cal)
NP = net.net_params()
W = 32

design = N.build(W, 64, NP, HERE / "gen" / "cs1j_default", tag="cs1jdefault")
design.compile()
print("[default] OK -- full net compiles through address allocation with the modified kernel")
