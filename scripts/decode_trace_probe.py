#!/usr/bin/env python3
# Per-op on-NPU trace probe (Option B, route b — standalone per-op).
# Validates the trace build+dispatch+capture+parse pipeline on LayerNorm (the one decode op
# that currently exposes trace_size). Once green, the remaining work is adding trace_size to
# GEMV/transpose/etc. (mirror layer_norm/op.py) and looping this over every decode op.
#
# Run (production stack):
#   PYTHONPATH=route_b_kernels/decode_fused:$HOME/repositories/ns/atassis/xdna-engine-workspace/amd/IRON \
#   .venv-iron/bin/python scripts/decode_trace_probe.py
# Needs the NPU free (stop npu-asr first).
import glob
import json
import os
import sys

import newstack_compat  # noqa: F401  MUST precede iron imports (new-mlir-aie shim)
import torch

from iron.common import AIEContext
from iron.operators.layer_norm.op import LayerNorm
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor
from aie.utils.npukernel import NPUKernel
from aie.utils.trace import TraceConfig, get_cycles_summary
import aie.utils as aie_utils

D = 768
TRACE_SIZE = 8192
TRACE_TXT = os.path.abspath("artifacts/trace_layernorm.txt")
TRACE_JSON = os.path.abspath("artifacts/trace_layernorm.json")
os.makedirs("artifacts", exist_ok=True)

ctx = AIEContext()
op = LayerNorm(size=D, num_aie_columns=1, num_channels=1, tile_size=D,
               trace_size=TRACE_SIZE, context=ctx)
print("[probe] compiling LayerNorm with trace_size=%d ..." % TRACE_SIZE)
op.compile()
bd = ctx.build_dir
print("[probe] build_dir:", bd)
print("[probe] xclbin:", op.xclbin_artifact.filename, "kernel:", op.xclbin_artifact.kernel_name)
print("[probe] insts :", op.insts_artifact.filename)

# Candidate lowered-MLIR files for the parser (needs the write32 addresses).
mlirs = sorted(glob.glob(os.path.join(bd, "**", "*.mlir"), recursive=True))
print("[probe] mlir candidates in build_dir:")
for m in mlirs:
    print("   ", m)

tc = TraceConfig(trace_size=TRACE_SIZE, trace_file=TRACE_TXT, ddr_id=4)
npu_kernel = NPUKernel(
    xclbin_path=op.xclbin_artifact.filename,
    kernel_name=op.xclbin_artifact.kernel_name,
    insts_path=op.insts_artifact.filename,
    trace_config=tc,
)
# Build buffers from arg_spec (random; we only want timing/trace, not correctness).
args = []
for s in op.get_arg_spec():
    if s.direction in ("in", "inout"):
        args.append(XRTTensor.from_torch(torch.rand(*s.shape, dtype=torch.bfloat16)))
    else:
        args.append(XRTTensor(s.shape, dtype=s.dtype))
# NPUKernel.__call__ -> load_and_run, which is the TRACE-AWARE path (prepares the trace BO,
# extracts it after the run, writes trace_config.trace_file). Plain load+run does NOT.
print("[probe] dispatching via load_and_run (trace-aware) ...")
res = npu_kernel(*args)
print("[probe] run result:", res)

print("[probe] trace.txt exists:", os.path.exists(TRACE_TXT),
      "size:", os.path.getsize(TRACE_TXT) if os.path.exists(TRACE_TXT) else 0)

# Parse: the parser needs input_with_addresses.mlir (it reads the lowered
# aiex.npu.write32 trace-config ops). The pre-lowering *.mlir lacks them and
# makes the parser sys.exit(1), so select the right file explicitly.
addr_mlir = [m for m in mlirs if m.endswith("input_with_addresses.mlir")]
ok = False
if addr_mlir:
    m = addr_mlir[0]
    tc.trace_to_json(m, TRACE_JSON)
    ev = json.load(open(TRACE_JSON))
    if ev:
        print(f"[probe] parsed {len(ev)} trace events using {os.path.basename(m)}")
        # INSTR_EVENT_0->1 brackets = kernel compute cycles (per traced tile).
        print("[probe] INSTR_EVENT_0->1 cycle deltas:", get_cycles_summary(TRACE_JSON))
        ts = [e["ts"] for e in ev if "ts" in e]
        if ts:
            print(f"[probe] active span = {max(ts) - min(ts)} cycles "
                  f"({min(ts)}..{max(ts)})")
        ok = True
else:
    print("[probe] no input_with_addresses.mlir found")
aie_utils.DefaultNPURuntime.cleanup()
print("[probe] PIPELINE", "VALIDATED" if ok else "INCOMPLETE (see above)")
sys.exit(0 if ok else 1)
