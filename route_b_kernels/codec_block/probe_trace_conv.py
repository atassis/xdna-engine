#!/usr/bin/env python3
"""Hardware trace of ONE codec conv dispatch: where does the on-device time actually go?

Established so far, all measured: the harness compiled per dispatch (fixed), and every design the
codec builds uses exactly ONE aie.core -- confirmed by counting `aie.core` in the built MLIR and
the single elfs_main_core_* dir -- against 32 on this array. So the array is ~97% idle by
construction. What that does NOT say is what the ONE active core spends its time on: real MAC work,
DMA wait, or lock stall. If it is already DMA-bound, spreading to 32 cores buys far less than 32x.

bricklib hardcodes iron.jit(design, use_cache=...), so trace_config is injected by wrapping
iron.jit for the duration of the build. Everything else is the ordinary driver path.
"""
import sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import window_driver as wd
import bricklib
import aie.iron as iron
from aie.utils.trace.config import TraceConfig

TRACE_OUT = Path("$TRACE_OUT_DIR/trace_conv.bin")
TRACE_OUT.parent.mkdir(parents=True, exist_ok=True)
tc = TraceConfig(trace_size=262144, trace_file=str(TRACE_OUT))

_orig_jit = iron.jit
def _traced_jit(*a, **k):
    k.setdefault("trace_config", tc)
    return _orig_jit(*a, **k)
iron.jit = _traced_jit
bricklib.iron.jit = _traced_jit
wd._DESIGNS.clear()                       # force a rebuild WITH trace enabled

C_IN, K, L, C_OUT = 128, 7, 136, 384
rng = np.random.default_rng(21)
x = rng.standard_normal((C_IN, L)).astype(np.float32)
w = (rng.standard_normal((C_OUT, C_IN, K)) * 0.02).astype(np.float32)

t0 = time.perf_counter()
out = wd.conv(x, w, np.zeros(C_OUT, np.float32), K, 1, "trc", ci_chunk=C_IN, resident_depth=1)
print(f"conv done in {time.perf_counter()-t0:.2f} s, out{out.shape}, "
      f"dispatches={wd.stats()['dispatches']}", flush=True)
print(f"trace_size={tc.trace_size}  file={tc.trace_file}", flush=True)
print(f"physical_mlir_path={getattr(tc, 'physical_mlir_path', None)}", flush=True)
if TRACE_OUT.exists():
    print(f"trace bytes on disk: {TRACE_OUT.stat().st_size}", flush=True)
