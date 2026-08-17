#!/usr/bin/env python3
# On-device stall breakdown for the whole_array GEMM (task gemm-offcore-residue-occupancy).
#
# perop_trace_measure.py measures the decode ops and reports span + INSTR_EVENT_0->1 only.
# This task needs the OTHER half: where the traced core's cycles GO. The generator traces
# MEMORY_STALL / STREAM_STALL / LOCK_STALL / INSTR_VECTOR plus the two DMA PORT_RUNNING
# events, which is exactly the core-local-vs-off-core split, and nothing here reads them.
#
# Consumes a Makefile.modal build (xclbin + insts_*.txt + the .mlir.prj), not the iron.operators
# op stack, so it runs against the shipped whole_array artifacts directly.
#
# WIDTH CAVEAT: cols=8 (the production dispatch) cannot route a trace flow -- (0,2) and (0,4)
# are both at 4/4 South egress. cols=4 traces with the stock worker. So this measures the inner
# loop's stall structure, NOT the ~2.4%-of-peak headline.
#
# Run (NPU free -- stop xdna-engine and npu-vox first):
#   source scripts/iron_env.sh
#   .venv-iron/bin/python scripts/gemm_trace_probe.py \
#       --build-dir mlir-aie/programming_examples/basic/matrix_multiplication/whole_array/build \
#       --suffix 512x1024x1024_64x32x128_4c_modalid
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
from ml_dtypes import bfloat16

from aie.utils.npukernel import NPUKernel
from aie.utils.tensor_factory import tensor, zeros
from aie.utils.trace import TraceConfig
import aie.utils as aie_utils

# AIE2 traces 8 core events, so which 8 is a CHOICE and the useful runs use different sets
# (--trace_events on the generator). Report whatever is present rather than a fixed table.
#
# ISSUE_EVENTS get a union as well as a sum, because they OVERLAP: a VLIW bundle can issue a load
# and a vector op in the same cycle. Measured at cols=4 the sum is 79.98% of span and the union
# 68.84%, i.e. 11.14pp of double-count -- and the sum happens to land within 0.5pp of ACTIVE
# (79.50%), which reads as a clean "issue accounts for all active time" and is a coincidence.
ISSUE_EVENTS = ["INSTR_VECTOR", "INSTR_LOAD", "INSTR_STORE"]


def insts_bin(insts_txt, artifacts):
    """Rename-only shim. route_b_override.mk pins --npu-insts-name to insts_*.txt because the
    engine and the kernel registry read that name, but aiecc writes RAW BINARY into it either
    way, and the host runtime dispatches on the extension alone (read_insts raises on anything
    but .bin, despite a docstring still promising a text branch). So the bytes are already
    correct; only the suffix is wrong."""
    out = os.path.join(artifacts, os.path.basename(insts_txt)[: -len(".txt")] + ".bin")
    with open(insts_txt, "rb") as f, open(out, "wb") as g:
        g.write(f.read())
    return out


def intervals(events):
    """Chrome-trace B/E pairs -> {(pid, name): [(start, end), ...]}. An unclosed B at the end of
    the buffer is dropped, not extrapolated -- the trace buffer truncates mid-run and a guessed
    tail would silently inflate whichever event happened to be open when it filled."""
    out = defaultdict(list)
    open_at = {}
    for e in events:
        name, ph = e.get("name"), e.get("ph")
        if ph not in ("B", "E") or "ts" not in e:
            continue
        key = (e.get("pid"), name)
        if ph == "B":
            open_at[key] = e["ts"]
        elif key in open_at:
            out[key].append((open_at.pop(key), e["ts"]))
    return out


def union_cycles(ivs):
    """Cycles covered by at least one interval -- the honest total for events that can co-occur."""
    total, cur_s, cur_e = 0, None, None
    for s, e in sorted(ivs):
        if cur_s is None:
            cur_s, cur_e = s, e
        elif s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
    return total + (cur_e - cur_s) if cur_s is not None else 0


def summarize(trace_json):
    ev = json.load(open(trace_json))
    pids = {m["pid"]: m["args"]["name"] for m in ev if m.get("name") == "process_name"}
    ts = [e["ts"] for e in ev if "ts" in e]
    if not ts:
        return {"events": len(ev), "error": "no timestamped events"}

    iv = intervals(ev)
    per_tile = {}
    for pid, tile in pids.items():
        tile_ts = [e["ts"] for e in ev if e.get("pid") == pid and "ts" in e]
        if not tile_ts:
            continue
        span = max(tile_ts) - min(tile_ts)
        pct = lambda c: round(100 * c / span, 2) if span else None
        row = {"span_cycles": span, "events": {}}
        for (p, name), ivs in sorted(iv.items()):
            if p != pid:
                continue
            cyc = sum(e - s for s, e in ivs)
            row["events"][name] = {
                "cycles": cyc,
                "pct_of_span": pct(cyc),
                "count": len(ivs),
                "mean_cycles": round(cyc / len(ivs), 2) if ivs else None,
            }
        issue = [i for n in ISSUE_EVENTS for i in iv.get((pid, n), [])]
        if issue:
            s = sum(e - st for st, e in issue)
            u = union_cycles(issue)
            row["issue_union"] = {"sum_cycles": s, "sum_pct": pct(s),
                                  "union_cycles": u, "union_pct": pct(u),
                                  "overlap_pct": pct(s - u)}
        per_tile[tile] = row

    return {
        "events": len(ev),
        "tiles": sorted(pids.values()),
        "global_span_cycles": max(ts) - min(ts),
        "per_tile": per_tile,
    }


def main(o):
    xclbin = os.path.join(o.build_dir, f"final_{o.suffix}.xclbin")
    insts = os.path.join(o.build_dir, f"insts_{o.suffix}.txt")
    prj_mlir = os.path.join(o.build_dir, f"aie_{o.suffix}.mlir.prj", "input_with_addresses.mlir")
    for p in (xclbin, insts, prj_mlir):
        if not os.path.exists(p):
            sys.exit(f"missing artifact: {p}")

    os.makedirs(o.artifacts, exist_ok=True)
    trace_txt = os.path.abspath(os.path.join(o.artifacts, f"trace_gemm_{o.suffix}.txt"))
    trace_json = os.path.abspath(os.path.join(o.artifacts, f"trace_gemm_{o.suffix}.json"))

    M, K, N = o.M, o.K, o.N
    rng = np.random.default_rng(seed=42)
    # Flat buffers: the runtime_sequence takes memref<MxK>, memref<KxN>, memref<MxN> as 1-D.
    A = tensor(rng.standard_normal((M * K,)).astype(bfloat16), dtype=bfloat16)
    B = tensor(rng.standard_normal((K * N,)).astype(bfloat16), dtype=bfloat16)
    C = zeros((M * N,), dtype=np.float32)

    tc = TraceConfig(trace_size=o.trace_size, trace_file=trace_txt)
    # NPUKernel.__call__ -> load_and_run is the trace-aware path: it appends the trace BO and
    # extracts it after the run. A plain load()+run() leaves trace_file untouched.
    kern = NPUKernel(xclbin_path=xclbin, insts_path=insts_bin(insts, o.artifacts),
                     kernel_name=o.kernel, trace_config=tc)
    print(f"[gemm-trace] {o.suffix} M={M} K={K} N={N} trace_size={o.trace_size}", flush=True)
    kern(A, B, C)

    # An occupancy number off a kernel that computes the wrong thing is worse than no number, and
    # this probe is used to COMPARE builds -- a K-tile or fifo-depth arm changes the compiled kernel
    # object, not just the timing, so a timing variant here can be fast and wrong.
    # 0.08 is the same rel-L2 bar the other whole_array probes use for bf16-in / f32-accumulate.
    ref = (np.asarray(A).reshape(M, K).astype(np.float32)
           @ np.asarray(B).reshape(K, N).astype(np.float32))
    got = np.asarray(C).reshape(M, N).astype(np.float32)
    rel_l2 = float(np.linalg.norm(got - ref) / np.linalg.norm(ref))
    gate = "PASS" if rel_l2 <= 0.08 else "FAIL"
    print(f"[gemm-trace] rel-L2 = {rel_l2:.4e}  {gate} (bar 0.08)", flush=True)

    size = os.path.getsize(trace_txt) if os.path.exists(trace_txt) else 0
    print(f"[gemm-trace] trace.txt {size} B", flush=True)
    if not size:
        aie_utils.DefaultNPURuntime.cleanup()
        sys.exit("empty trace buffer -- the packet never reached the shim")

    tc.trace_to_json(prj_mlir, trace_json)
    res = summarize(trace_json)
    res.update(suffix=o.suffix, M=M, K=K, N=N, trace_size=o.trace_size,
               rel_l2=rel_l2, correctness=gate)
    out = os.path.join(o.artifacts, f"gemm_trace_summary_{o.suffix}.json")
    json.dump(res, open(out, "w"), indent=2)

    print(f"\n=== {o.suffix}: {res['events']} events, span {res.get('global_span_cycles')} cyc ===")
    for tile, row in res.get("per_tile", {}).items():
        print(f"  {tile}  span={row['span_cycles']} cyc")
        for name, d in sorted(row["events"].items(), key=lambda kv: -kv[1]["cycles"]):
            print(f"    {name:<24} {d['cycles']:>9} cyc {d['pct_of_span']:>6}%  "
                  f"n={d['count']:<7} mean={d['mean_cycles']}")
        u = row.get("issue_union")
        if u:
            print(f"    {'issue sum/union':<24} {u['sum_cycles']:>9}/{u['union_cycles']} cyc "
                  f"{u['sum_pct']}/{u['union_pct']}%  overlap {u['overlap_pct']}%")
    print(f"\nwrote {out}")
    aie_utils.DefaultNPURuntime.cleanup()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--build-dir", required=True)
    p.add_argument("--suffix", required=True, help="e.g. 512x1024x1024_64x32x128_4c_modalid")
    p.add_argument("--M", type=int, default=512)
    p.add_argument("--K", type=int, default=1024)
    p.add_argument("--N", type=int, default=1024)
    p.add_argument("--trace-size", type=int, default=65536)
    p.add_argument("--kernel", default="MLIR_AIE")
    p.add_argument("--artifacts", default="artifacts")
    main(p.parse_args())
