#!/usr/bin/env python3
# Command-time accounting for the whole_array GEMM (task gemm-offcore-residue-occupancy, item 3).
#
# The eight trace passes all measured what happens INSIDE the traced core window and none of them
# measured the window's DENOMINATOR: gemm_trace_probe.py never records a wall-clock, so "the core
# spans 407710 cyc" has never been divided by how long the command actually took. The task's own
# item (3) says the ~2.4%-of-peak headline lives outside that window; this is what tests it.
#
# Two things this can do that the trace path cannot:
#   * cols=8. Tracing needs a routable trace flow and cols=8 has none ((0,2) and (0,4) are both at
#     4/4 South egress), which is why every prior pass is cols=4. Timing needs no flow, so the
#     production width is measurable here and the width caveat does not bind.
#   * steady state. load() once, then time run() repeatedly, so xclbin load and hw-context creation
#     land in the warmup rather than in the number.
#
# TRACE OPERAND: a trace-enabled design carries an appended i8 operand and its packet flows expect a
# BO to land in. Supplying 3 buffers to a 4-operand design is accepted by XRT (fewer than declared is
# legal) but leaves those packets without a destination, which perturbs the timing being measured.
# So the operand list is read from the design's own lowered MLIR and the trace BO supplied iff it is
# declared. This also means a traced arm's command time INCLUDES trace cost -- compare it against its
# own span, not against an untraced arm's command time.
#
# npu_time is XRT submit -> completion (perf_counter_ns around kernel() + wait() in the runtime), so
# it excludes host buffer sync and includes everything the command itself does.
#
# Run (NPU free -- stop xdna-engine and npu-vox first):
#   source scripts/iron_env.sh
#   .venv-iron/bin/python scripts/gemm_command_accounting.py --suffixes <a> <b> ...
import argparse
import json
import os
import re
import statistics
import sys

import numpy as np
from ml_dtypes import bfloat16

from aie.utils.npukernel import NPUKernel
from aie.utils.tensor_factory import tensor, zeros
import aie.utils as aie_utils

# bfp16 path: mmul<8,8,8> lowers to one mac_8x8_8x8T_conf = 512 MACs/cycle/core. The brick catalog's
# 128 is plain bf16 and gating against it understates the gap 4x. Core clock from AMD's published
# 50 TOPS / 32 cores / 512 MAC (an on-device trace independently implied ~1.6 GHz).
MAC_PER_CYCLE_PER_CORE = 512
CORE_CLOCK_HZ = 1.53e9
COMPUTE_ROWS = 4  # rows 2..5


def design_operands(build_dir, suffix):
    """Operand type list of the design's aie.runtime_sequence, from its own generated MLIR.

    PROFILE=trace keeps input_with_addresses.mlir in the .prj; PROFILE=production does not, so the
    no-trace control arms only have the top-level generated aie_<suffix>.mlir. Both carry the same
    runtime_sequence signature, which is all this needs.
    """
    for path in (os.path.join(build_dir, f"aie_{suffix}.mlir.prj", "input_with_addresses.mlir"),
                 os.path.join(build_dir, f"aie_{suffix}.mlir")):
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                if "aie.runtime_sequence(" in line:
                    return re.findall(r"memref<(\d+)x(bf16|f32|i8)>", line)
    return []


def insts_bin(insts_txt, artifacts):
    """Rename-only shim: aiecc writes raw binary into insts_*.txt, and the host runtime dispatches
    on the extension alone. Same shim gemm_trace_probe.py uses."""
    out = os.path.join(artifacts, os.path.basename(insts_txt)[: -len(".txt")] + ".bin")
    with open(insts_txt, "rb") as f, open(out, "wb") as g:
        g.write(f.read())
    return out


def measure(suffix, o):
    xclbin = os.path.join(o.build_dir, f"final_{suffix}.xclbin")
    insts = os.path.join(o.build_dir, f"insts_{suffix}.txt")
    for p in (xclbin, insts):
        if not os.path.exists(p):
            sys.exit(f"missing artifact: {p}")

    cols = int(re.search(r"_(\d+)c_", suffix).group(1))
    operands = design_operands(o.build_dir, suffix)
    if not operands:
        sys.exit(f"{suffix}: could not read runtime_sequence operands in {o.build_dir}")
    traced = len(operands) > 3

    M, K, N = o.M, o.K, o.N
    rng = np.random.default_rng(seed=42)
    A = tensor(rng.standard_normal((M * K,)).astype(bfloat16), dtype=bfloat16)
    B = tensor(rng.standard_normal((K * N,)).astype(bfloat16), dtype=bfloat16)
    C = zeros((M * N,), dtype=np.float32)
    args = [A, B, C]
    if traced:
        # Declared size, not a guess -- a short trace BO backpressures the packet flows.
        args.append(zeros((int(operands[3][0]),), dtype=np.int8))

    kern = NPUKernel(xclbin_path=xclbin, insts_path=insts_bin(insts, o.artifacts),
                     kernel_name=o.kernel)
    rt = aie_utils.DefaultNPURuntime
    handle = rt.load(kern)

    # Warmup carries hw-context creation and first-touch of every BO; it is reported, never folded
    # into the steady-state number.
    warm = rt.run(handle, list(args)).npu_time / 1000.0

    samples = []
    for _ in range(o.reps):
        samples.append(rt.run(handle, list(args)).npu_time / 1000.0)

    # An occupancy number off a kernel computing the wrong thing is worse than no number, and the
    # arms differ by compiled kernel object, not just timing. Same 0.08 bar as the other whole_array
    # probes (bf16 in / f32 accumulate).
    ref = (np.asarray(A).reshape(M, K).astype(np.float32)
           @ np.asarray(B).reshape(K, N).astype(np.float32))
    got = np.asarray(C).reshape(M, N).astype(np.float32)
    rel_l2 = float(np.linalg.norm(got - ref) / np.linalg.norm(ref))
    gate = "PASS" if rel_l2 <= 0.08 else "FAIL"

    cores = cols * COMPUTE_ROWS
    macs = M * K * N
    t_peak_us = 1e6 * macs / (cores * MAC_PER_CYCLE_PER_CORE * CORE_CLOCK_HZ)
    med = statistics.median(samples)
    row = {
        "suffix": suffix, "cols": cols, "cores": cores, "traced_design": traced,
        "operands": len(operands), "reps": o.reps,
        "warmup_us": round(warm, 1),
        "samples_us": [round(s, 1) for s in samples],
        "median_us": round(med, 1), "min_us": round(min(samples), 1),
        "max_us": round(max(samples), 1),
        "t_peak_us": round(t_peak_us, 1),
        "pct_of_peak_median": round(100 * t_peak_us / med, 2),
        "pct_of_peak_best": round(100 * t_peak_us / min(samples), 2),
        "rel_l2": rel_l2, "correctness": gate,
    }

    # Accounting: convert this arm's traced core span to microseconds and ask how much of the
    # command the core was even spanning. This is the number item (3) is about. Per-suffix, because
    # the span is a property of the arm -- one number applied to every arm would be a category error.
    span_cycles = o.span.get(suffix)
    if span_cycles:
        span_us = span_cycles / CORE_CLOCK_HZ * 1e6
        row["span_cycles"] = span_cycles
        row["span_us"] = round(span_us, 1)
        row["span_pct_of_command"] = round(100 * span_us / med, 1)
        row["outside_window_us"] = round(med - span_us, 1)
    return row


def main(o):
    o.span = {kv.split("=")[0]: int(kv.split("=")[1]) for kv in o.span}
    os.makedirs(o.artifacts, exist_ok=True)
    rows = []
    for suffix in o.suffixes:
        print(f"\n---------- {suffix} ----------", flush=True)
        r = measure(suffix, o)
        rows.append(r)
        tag = "traced design (+trace BO)" if r["traced_design"] else "no trace operand"
        print(f"  cols={r['cols']} cores={r['cores']}  {tag}", flush=True)
        print(f"  rel-L2 = {r['rel_l2']:.4e}  {r['correctness']} (bar 0.08)", flush=True)
        print(f"  warmup {r['warmup_us']} us", flush=True)
        print(f"  per-rep us: {r['samples_us']}", flush=True)
        print(f"  median {r['median_us']} us   min {r['min_us']}   max {r['max_us']}", flush=True)
        print(f"  peak-time {r['t_peak_us']} us -> {r['pct_of_peak_median']}% of peak "
              f"(best rep {r['pct_of_peak_best']}%)", flush=True)
        if "span_us" in r:
            print(f"  traced span {r['span_cycles']} cyc = {r['span_us']} us = "
                  f"{r['span_pct_of_command']}% of command; "
                  f"{r['outside_window_us']} us outside the window", flush=True)

    out = os.path.join(o.artifacts, "gemm_command_accounting.json")
    json.dump(rows, open(out, "w"), indent=2)
    print(f"\nwrote {out}")
    aie_utils.DefaultNPURuntime.cleanup()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--build-dir", default="mlir-aie/programming_examples/basic/"
                                          "matrix_multiplication/whole_array/build")
    p.add_argument("--suffixes", nargs="+", required=True)
    p.add_argument("--M", type=int, default=512)
    p.add_argument("--K", type=int, default=1024)
    p.add_argument("--N", type=int, default=1024)
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--span", nargs="*", default=[], metavar="SUFFIX=CYCLES",
                   help="per-arm traced core span to account the command against")
    p.add_argument("--kernel", default="MLIR_AIE")
    p.add_argument("--artifacts", default="artifacts")
    main(p.parse_args())
