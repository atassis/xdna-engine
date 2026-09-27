"""Phase 1h step 1: compile-only MemTile budget re-check for SKIP_SLACK, current main defaults
(post block-1 fix, post Phase1g). No device.

`N.build()` alone does NOT check address allocation -- it returns a lazily-jitted CallableDesign,
and aiecc's placement/allocation passes (where a MemTile buffer overflow is actually raised) only
run on first invocation (`design(*args)`, needing device tensors) or on an explicit `.compile()`
call. An earlier version of this script called only `N.build()` and reported "OK" for every
skip_slack up to 40, which is a VACUOUS check -- device confirmed slack=40 fails with "iterate_bds
needs one 691200-byte buffer on its MemTile, which has 524288 bytes free". This version calls
`.compile()` (no xclbin_path -- pre-warms the cache, still no device) to force the real aiecc
pipeline, including address allocation. See designs/span_sr/TRACE_RESULTS.md, Phase 1h.
"""
import os, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "designs" / "span_sr"))
import aie.iron as iron  # noqa
import net_design as N
import net_layout as NL
import span_int as S
import test_span_int as TS

cal, _ = TS.load_split(os.environ["SPAN_DEMO_DIR"])
net = S.Span(os.environ["SPAN_EXPORT"])
net.quantize(cal)
NP = net.net_params()
W = 32

def try_build(tag, **kw):
    try:
        design = N.build(W, 64, NP, HERE / "gen" / f"cs1h_{tag}", tag=f"cs1h{tag}", **kw)
        design.compile()  # force the real aiecc pipeline (address allocation included), no device
        print(f"[{tag}] OK  kw={kw}")
        return True
    except Exception as e:
        msg = str(e).splitlines()[-1] if str(e) else repr(e)
        print(f"[{tag}] FAIL kw={kw}  -- {msg}")
        return False

print("== skip_slack sweep (task values + boundary re-check, real aiecc compile) ==")
for sk in (8, 12, 16, 20, 25, 26, 27, 28, 30, 32, 35, 40):
    try_build(f"sk{sk}", skip_slack=sk)
