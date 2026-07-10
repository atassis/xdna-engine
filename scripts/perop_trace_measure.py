#!/usr/bin/env python3
# Per-op on-NPU hardware-trace measurement (route b, standalone per-op).
# Builds each fused-Whisper-decode op at its REAL decode shape (from
# route_b_kernels/decode_fused/gen_decode.py), traces it on the NPU via the
# IRON_TRACE_SIZE env hook (gated in each op's design.py -> no-op for production),
# and extracts per-op on-chip cycles:
#   - span   = full active window (max ts - min ts over all traced tiles' events),
#              incl. input DMA -> compute -> output DMA. The per-dispatch wall cost.
#   - instr  = INSTR_EVENT_0->1 delta (kernel compute bracket) where the kernel has
#              the brackets (LayerNorm/GELU stock kernels do; gemv/transpose/softmax
#              stock kernels do NOT -> instr=None, use span).
#
# Run (production stack, NPU free -> stop npu-asr first):
#   IRON_TRACE_SIZE=65536 \
#   PYTHONPATH=route_b_kernels/decode_fused:$HOME/repositories/ns/amd/IRON \
#   .venv-iron/bin/python scripts/perop_trace_measure.py [op1 op2 ...]
#
# Results appended to artifacts/perop_trace_results.json (keyed by label).
import glob
import json
import os
import sys

import newstack_compat  # noqa: F401  MUST precede iron imports
import torch

from iron.common import AIEContext
from iron.operators.layer_norm.op import LayerNorm
from iron.operators.gelu.op import GELU
from iron.operators.gemv.op import GEMV
from iron.operators.softmax.op import Softmax
from iron.operators.transpose.op import Transpose
from iron.operators.gemm.op import GEMM
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor
from aie.utils.npukernel import NPUKernel
from aie.utils.trace import TraceConfig
import aie.utils as aie_utils

# --- decode constants (gen_decode.py) ---
D, QKV, FF, H, HD, S, TP = 768, 2304, 3072, 12, 64, 448, 1536


def pick_tiling(M, N):
    for s in (8, 4):
        for m in sorted((d for d in range(s, M + 1) if M % d == 0 and d % s == 0), reverse=True):
            for n in sorted((d for d in range(s, N + 1) if N % d == 0 and d % s == 0), reverse=True):
                if m * n <= 8192 and not (s == 8 and (m <= 16 or n <= 16)):
                    return m, n, s
    raise ValueError("no tiling")


tms, tns, tss = pick_tiling(S, HD)
tmc, tnc, tsc = pick_tiling(TP, HD)

TRACE_SIZE = int(os.environ.get("IRON_TRACE_SIZE", "65536"))
os.environ["IRON_TRACE_SIZE"] = str(TRACE_SIZE)  # ensure the design hooks see it
os.makedirs("artifacts", exist_ok=True)
RESULTS = os.path.abspath("artifacts/perop_trace_results.json")


def op_factory(ctx):
    """label -> (builder, note).

    For 8-column GEMV/GELU the production op saturates all 8 shim DMA channels, leaving
    no free route for the trace packet flow (aiecc routing conflict). We measure a
    SINGLE-COLUMN PROXY at M = M_real // 8 = exactly one production column's share: the
    mv.cc/gelu kernel does the identical per-tile compute AND per-tile DMA volume, and
    the 8 real columns run in parallel, so the proxy's traced cycles == the op's
    per-dispatch wall. LayerNorm (1 col) / Softmax (1 col) / Transpose (2 col) trace at
    their real config.
    """
    f = {}
    f["layernorm"] = (lambda: LayerNorm(size=D, num_aie_columns=1, num_channels=1,
                                        tile_size=D, trace_size=TRACE_SIZE, context=ctx),
                      "self/cross/ffn LN, D=768, real 1col/4core")
    f["gelu_ffn"] = (lambda: GELU(size=FF // 8, num_aie_columns=1, num_channels=1,
                                  tile_size=FF // 8, context=ctx),
                     "FFN GELU, per-col proxy size=384 (1 of 8 cols)")

    def gemv1(Mreal, K, nb=1):
        mp = Mreal // 8  # one production column's row share
        return GEMV(M=mp, K=K, num_aie_columns=1, tile_size_input=4,
                    tile_size_output=mp, num_batches=nb, context=ctx)

    f["gemv_proj"] = (lambda: gemv1(D, D), "out proj 768x768 per-col proxy M=96")
    f["gemv_qkv"] = (lambda: gemv1(QKV, D), "qkv 2304x768 per-col proxy M=288")
    f["gemv_fc1"] = (lambda: gemv1(FF, D), "fc1 3072x768 per-col proxy M=384")
    f["gemv_fc2"] = (lambda: gemv1(D, FF), "fc2 768x3072 per-col proxy M=96")
    f["gemv_score_self"] = (lambda: gemv1(S, HD, H), "self QKt 448x64 b12 per-col M=56")
    f["gemv_ctx_self"] = (lambda: gemv1(HD, S, H), "self ctx 64x448 b12 per-col M=8")
    f["gemv_score_cross"] = (lambda: gemv1(TP, HD, H), "cross QKt 1536x64 b12 per-col M=192")
    f["gemv_ctx_cross"] = (lambda: gemv1(HD, TP, H), "cross ctx 64x1536 b12 per-col M=8")
    # Standalone softmax: compile-time (non-scratchpad) variant -> no kv/mask binding.
    f["softmax_self"] = (lambda: Softmax(rows=16, cols=S, num_aie_columns=1, num_channels=1,
                                         rtp_vector_size=S, context=ctx),
                         "self softmax cols=448 real 1col")
    f["softmax_cross"] = (lambda: Softmax(rows=16, cols=TP, num_aie_columns=1, num_channels=1,
                                          rtp_vector_size=TP, context=ctx),
                          "cross softmax cols=1536 real 1col")
    # V-transpose (the #1 suspect). Real op is 2-col (splits N=64 into 2x32) -> trace-flow
    # routing conflict, same as GEMV. Per-col proxy: 1 col at N=HD//2=32 = one production
    # column's share (the other column transposes the other 32 cols in parallel) -> the
    # proxy span == the op's per-dispatch wall. num_batches preserved (per-head unroll).
    def tr1(Mt, m_, n_, s_, nb=1):
        return Transpose(M=Mt, N=HD // 2, num_batches=nb, num_aie_columns=1,
                         num_channels=1, m=m_, n=HD // 2, s=s_, context=ctx)

    f["transpose_self"] = (lambda: tr1(S, tms, tns, tss),
                           "self V-transpose 448x32 per-col proxy (1 of 2 cols)")
    f["transpose_cross"] = (lambda: tr1(TP, tmc, tnc, tsc),
                            "cross V-transpose 1536x32 per-col proxy (1 of 2 cols)")
    f["transpose_self_b12"] = (lambda: tr1(S, tms, tns, tss, H),
                               "self V-transpose head-batched b12 per-col proxy")

    # ---- B=128 (batched decode, gen_decode_batched.py) ----
    # GEMMs replace the projection GEMVs. REAL production config (N=B=128, tile_n=16,
    # num_aie_columns=8). GEMM trace can't route a packet flow (cascade already
    # packet-switches DMA0), so these are measured by DISPATCH-TIME (MEASURE_DISPATCH=1)
    # at the real config, not trace.
    def gemmB(M, K):
        return GEMM(M=M, K=K, N=128, tile_m=64, tile_k=64, tile_n=16, num_aie_columns=8,
                    b_col_maj=True, c_col_maj=True, context=ctx)

    f["gemm_qkv_b128"] = (lambda: gemmB(QKV, D), "B128 qkv GEMM 2304x768 N=128 real 8col")
    f["gemm_proj_b128"] = (lambda: gemmB(D, D), "B128 proj GEMM 768x768 N=128 real 8col")
    f["gemm_fc1_b128"] = (lambda: gemmB(FF, D), "B128 fc1 GEMM 3072x768 N=128 real 8col")
    f["gemm_fc2_b128"] = (lambda: gemmB(D, FF), "B128 fc2 GEMM 768x3072 N=128 real 8col")
    # B=128 V-transpose: REAL op is num_aie_columns=1 already. num_batches=BH=1536 production;
    # measured at nb=128 (tractable) -> scale x12 for the per-token BH=1536 cost (~linear).
    # B=128 attention GEMVs (g_scs/g_cts/g_scc/g_ctc): real op is 8-col, num_batches=BH=1536.
    # 1-col proxy at M=M//8, measured at nb=128 (tractable) -> scale x12 for BH=1536 (~linear).
    f["gemv_score_self_b128"] = (lambda: gemv1(S, HD, 128), "B128 self QKt nb=128 (x12->BH=1536) per-col")
    f["gemv_ctx_self_b128"] = (lambda: gemv1(HD, S, 128), "B128 self ctx nb=128 (x12->BH=1536) per-col")
    f["gemv_score_cross_b128"] = (lambda: gemv1(TP, HD, 128), "B128 cross QKt nb=128 (x12->BH=1536) per-col")
    f["gemv_ctx_cross_b128"] = (lambda: gemv1(HD, TP, 128), "B128 cross ctx nb=128 (x12->BH=1536) per-col")
    f["transpose_self_b128"] = (lambda: Transpose(M=S, N=HD, num_batches=128, num_aie_columns=1,
                                                  num_channels=1, m=tms, n=tns, s=tss, context=ctx),
                                "B128 self V-transpose REAL 1col nb=128 (x12 -> BH=1536)")
    f["transpose_cross_b128"] = (lambda: Transpose(M=TP, N=HD, num_batches=128, num_aie_columns=1,
                                                   num_channels=1, m=tmc, n=tnc, s=tsc, context=ctx),
                                 "B128 cross V-transpose REAL 1col nb=128 (x12 -> BH=1536)")
    return f


def measure(label, builder, note):
    ctx = AIEContext()
    op = builder()
    print(f"\n[{label}] compiling ({note}) ...", flush=True)
    op.compile()
    xcl = op.xclbin_artifact.filename
    stem = xcl[: -len(".xclbin")] if xcl.endswith(".xclbin") else xcl
    prj_mlir = stem + ".mlir.prj/input_with_addresses.mlir"
    if not os.path.exists(prj_mlir):
        cand = glob.glob(os.path.join(os.path.dirname(xcl), "**", "input_with_addresses.mlir"),
                         recursive=True)
        # newest = this op's
        prj_mlir = max(cand, key=os.path.getmtime) if cand else None
    trace_txt = os.path.abspath(f"artifacts/trace_{label}.txt")
    trace_json = os.path.abspath(f"artifacts/trace_{label}.json")
    tc = TraceConfig(trace_size=TRACE_SIZE, trace_file=trace_txt, ddr_id=4)
    kern = NPUKernel(xclbin_path=xcl, kernel_name=op.xclbin_artifact.kernel_name,
                     insts_path=op.insts_artifact.filename, trace_config=tc)
    args = []
    for sp in op.get_arg_spec():
        if sp.direction in ("in", "inout"):
            args.append(XRTTensor.from_torch(torch.rand(*sp.shape, dtype=torch.bfloat16)))
        else:
            args.append(XRTTensor(sp.shape, dtype=sp.dtype))
    print(f"[{label}] dispatching (trace-aware) ...", flush=True)
    kern(*args)
    tc.trace_to_json(prj_mlir, trace_json)
    ev = json.load(open(trace_json))
    ts = [e["ts"] for e in ev if "ts" in e]
    span = (max(ts) - min(ts)) if ts else None
    # INSTR_EVENT_0->1 deltas per tile (kernel compute bracket, if present)
    instr = []
    pids = {m["pid"]: m["args"]["name"] for m in ev if m["name"] == "process_name"}
    open0 = {}
    for e in ev:
        if e.get("name") == "INSTR_EVENT_0" and e.get("ph") == "B":
            open0[e["pid"]] = e["ts"]
        elif e.get("name") == "INSTR_EVENT_1" and e.get("ph") == "B" and e["pid"] in open0:
            instr.append(e["ts"] - open0.pop(e["pid"]))
    tiles = sorted(set(pids.values()))
    res = {"label": label, "note": note, "events": len(ev), "tiles": tiles,
           "span_cycles": span, "instr_event_deltas": instr,
           "instr_max": max(instr) if instr else None}
    print(f"[{label}] span={span} cyc  instr_deltas={instr}  tiles={len(tiles)}", flush=True)
    aie_utils.DefaultNPURuntime.cleanup()
    return res


def measure_dispatch(label, builder, note, iters=30):
    """Fallback: per-op NPU dispatch wall time (no trace) for ops whose dataflow can't
    route a trace packet flow (B=128 GEMM: cascade already packet-switches DMA0). Builds
    WITHOUT IRON_TRACE_SIZE, dispatches `iters` times, reports median wall per dispatch."""
    import time

    env_bak = os.environ.pop("IRON_TRACE_SIZE", None)
    try:
        ctx = AIEContext()
        op = builder()
        print(f"\n[{label}] (dispatch-time fallback) compiling ({note}) ...", flush=True)
        op.compile()
        kern = NPUKernel(xclbin_path=op.xclbin_artifact.filename,
                         kernel_name=op.xclbin_artifact.kernel_name,
                         insts_path=op.insts_artifact.filename)
        args = []
        for sp in op.get_arg_spec():
            if sp.direction in ("in", "inout"):
                args.append(XRTTensor.from_torch(torch.rand(*sp.shape, dtype=torch.bfloat16)))
            else:
                args.append(XRTTensor(sp.shape, dtype=sp.dtype))
        kern(*args)  # warmup
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            kern(*args)
            ts.append((time.perf_counter() - t0) * 1e6)  # us
        ts.sort()
        med = ts[len(ts) // 2]
        res = {"label": label, "note": note, "method": "dispatch_time_us",
               "dispatch_us_median": round(med, 2),
               "dispatch_us_min": round(ts[0], 2)}
        print(f"[{label}] dispatch median={med:.1f}us min={ts[0]:.1f}us", flush=True)
        aie_utils.DefaultNPURuntime.cleanup()
        return res
    finally:
        if env_bak is not None:
            os.environ["IRON_TRACE_SIZE"] = env_bak


def main():
    all_res = {}
    if os.path.exists(RESULTS):
        all_res = json.load(open(RESULTS))
    ctx0 = AIEContext()
    factory = op_factory(ctx0)
    want = sys.argv[1:] or list(factory.keys())
    for label in want:
        if label not in factory:
            print(f"!! unknown op '{label}' (have: {list(factory)})")
            continue
        builder, note = factory[label]
        fn = measure_dispatch if os.environ.get("MEASURE_DISPATCH") == "1" else measure
        try:
            all_res[label] = fn(label, builder, note)
        except Exception as e:
            import traceback
            traceback.print_exc()
            all_res[label] = {"label": label, "note": note, "error": repr(e)}
        json.dump(all_res, open(RESULTS, "w"), indent=2)
    print("\n=== RESULTS so far ===")
    for k, v in all_res.items():
        print(f"  {k}: span={v.get('span_cycles')} instr={v.get('instr_max')} "
              f"{'ERR ' + v['error'] if 'error' in v else ''}")


if __name__ == "__main__":
    main()
