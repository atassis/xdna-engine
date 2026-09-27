"""Phase 1h step 4: dump the compiled MLIR's cat_in objectfifo placement/size at skip_slack=16 to
confirm the ring is one buffer on one MemTile (tile (4,1), 414720 B of 524288 B free). Needs a
real device dispatch (`design(*args)`) to populate `TraceConfig.physical_mlir_path` -- an
earlier `N.build()`-only compile-only script (compile_sweep_phase1h.py, before its `.compile()`
fix) wrongly reported skip_slack up to 40 as "OK"; slack=40 here correctly FAILS at aiecc's
address-allocation pass ("iterate_bds needs one 691200-byte buffer on its MemTile, which has
524288 bytes free") -- that failure is what caught the vacuous check, not a bug in this script.
See designs/span_sr/TRACE_RESULTS.md, Phase 1h.
"""
import os, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "designs" / "span_sr"))
import aie.iron as iron  # noqa
from aie.utils.trace.config import TraceConfig
from aie.utils.trace.events import CoreEvent
import numpy as np
import net_design as N
import net_layout as NL
import span_int as S
import test_span_int as TS

cal, _ = TS.load_split(os.environ["SPAN_DEMO_DIR"])
net = S.Span(os.environ["SPAN_EXPORT"])
net.quantize(cal)
NP = net.net_params()
W = 32
EVENTS = [CoreEvent.INSTR_EVENT_0, CoreEvent.INSTR_EVENT_1, CoreEvent.LOCK_STALL]

HEIGHT = 64
for sk in (16, 40):
    tc = TraceConfig(trace_size=1048576, trace_file=f"/tmp/ij1h_ss{sk}.txt")
    design = N.build(W, HEIGHT, NP, HERE / "gen" / f"ij1h_{sk}", tag=f"ij1h{sk}",
                     trace_stages=["conv_cat"], trace_config=tc, coretile_events=EVENTS,
                     egress_shim_col=1, skip_slack=sk)
    rng = np.random.default_rng(0)
    x_row, y_row = NL.layout("conv1", W).in_bytes, NL.layout("up", W).out_bytes
    x = rng.integers(0, 256, size=(HEIGHT + 2) * x_row, dtype=np.int64).astype(np.uint8).view(np.int8)
    args = [iron.tensor(x, dtype=np.int8, device="npu"),
           iron.tensor(NL.weights_blob(NP, NL.stage_names("up")), dtype=np.int8, device="npu"),
           iron.zeros((HEIGHT * y_row,), dtype=np.int8, device="npu")]
    design(*args)
    print(f"== skip_slack={sk} physical_mlir_path={tc.physical_mlir_path} ==")
    if tc.physical_mlir_path and Path(tc.physical_mlir_path).exists():
        txt = Path(tc.physical_mlir_path).read_text()
        out = HERE / "gen" / f"ij1h_{sk}_physical.mlir"
        out.write_text(txt)
        print(f"saved {out}")
