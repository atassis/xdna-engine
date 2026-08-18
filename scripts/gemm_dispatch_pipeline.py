#!/usr/bin/env python3
# Split the whole_array GEMM's per-command floor into HOST ROUND-TRIP and DEVICE-SERIAL work
# (task gemm-offcore-residue-occupancy, item 1).
#
# WHY: the shape series measured a per-command floor of ~125 us that is invariant in cols, M, K and
# N -- at the production shape it is a third of the command and it is the largest lever left. But
# `npu_time` is perf_counter_ns around BOTH `kernel(...)` (submit) and `h.wait()` (block until
# complete), so a host/driver round-trip is inside every number this task has ever quoted. A floor
# that is submit+wait latency and a floor that is device-serial work are different findings with
# different fixes, and nothing so far distinguishes them.
#
# TWO MEASUREMENTS, no rebuild:
#   * SPLIT: time the submit call and the wait call separately at depth 1. Submit returns a run
#     handle without blocking, so this is the cleanest available cut between the two.
#   * DEPTH SWEEP: submit B commands back to back, THEN wait on all of them, and report total/B.
#     Anything that is per-command host round-trip amortizes as B grows; anything the device must
#     do serially per command does not. The asymptote is the real device-serial floor.
#
# CORRECTNESS, and why the deep arms are not gated: every in-flight command writes the SAME C
# buffer, so at B > 1 the outputs race by construction and rel-L2 is meaningless. Depth 1 runs the
# identical argument list through the identical path and IS gated; the deep arms are timing-only and
# reported as such. Do not read a rel-L2 off them.
#
# Run (NPU free -- the device wrapper stops xdna-engine and npu-vox):
#   .venv-iron/bin/python scripts/gemm_dispatch_pipeline.py --suffixes <a> <b> ...
import argparse
import json
import os
import re
import statistics
import sys
import time

import numpy as np
from ml_dtypes import bfloat16

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gemm_ddr_bytes import account as ddr_account
from gemm_command_accounting import design_operands, insts_bin, shape_of, CORE_CLOCK_HZ, \
    MAC_PER_CYCLE_PER_CORE, COMPUTE_ROWS

import pyxrt
from aie.utils.npukernel import NPUKernel
from aie.utils.tensor_factory import tensor, zeros
import aie.utils as aie_utils


def measure(suffix, o):
    xclbin = os.path.join(o.build_dir, f"final_{suffix}.xclbin")
    insts = os.path.join(o.build_dir, f"insts_{suffix}.txt")
    for p in (xclbin, insts):
        if not os.path.exists(p):
            sys.exit(f"missing artifact: {p}")
    cols = int(re.search(r"_(\d+)c_", suffix).group(1))
    operands = design_operands(o.build_dir, suffix)
    if not operands:
        sys.exit(f"{suffix}: could not read runtime_sequence operands")
    if len(operands) > 3:
        sys.exit(f"{suffix}: traced design -- run the pipeline probe on no-trace arms only")

    M, K, N = shape_of(suffix, o)
    rng = np.random.default_rng(seed=42)
    A = tensor(rng.standard_normal((M * K,)).astype(bfloat16), dtype=bfloat16)
    B = tensor(rng.standard_normal((K * N,)).astype(bfloat16), dtype=bfloat16)
    C = zeros((M * N,), dtype=np.float32)
    args = [A, B, C]

    kern = NPUKernel(xclbin_path=xclbin, insts_path=insts_bin(insts, o.artifacts),
                     kernel_name=o.kernel)
    rt = aie_utils.DefaultNPURuntime
    handle = rt.load(kern)
    rt.run(handle, list(args))  # warmup carries hw-context creation and BO first-touch

    # Reach past the runtime wrapper to the same objects its own timed region uses, so the split
    # measures the identical calls rather than a re-implementation of them. This mirrors run()
    # exactly: to("npu") and buffer_object() sit OUTSIDE its timed region, which is why npu_time
    # excludes host buffer sync -- and why this probe must leave them outside too.
    kh = handle
    while not hasattr(kh, "kernel"):
        kh = kh._handle
    [a.to("npu") for a in args]
    bufs = [a.buffer_object() for a in args]
    insts_bytes = kh.insts.nbytes
    insts_bo = kh.insts_bo
    if not insts_bo:
        insts_bo = rt._tensor_class(kh.insts, flags=pyxrt.bo.cacheable,
                                    group_id=kh.kernel.group_id(1),
                                    xrt_device=rt._device).buffer_object()

    def submit():
        return kh.kernel(3, insts_bo, insts_bytes, *bufs)

    # ---- depth 1, submit and wait timed separately ----
    subs, waits, totals = [], [], []
    for _ in range(o.reps):
        t0 = time.perf_counter_ns()
        h = submit()
        t1 = time.perf_counter_ns()
        r = h.wait()
        t2 = time.perf_counter_ns()
        if r != pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
            sys.exit(f"{suffix}: kernel returned {r}")
        subs.append((t1 - t0) / 1000.0)
        waits.append((t2 - t1) / 1000.0)
        totals.append((t2 - t0) / 1000.0)

    ref = (np.asarray(A).reshape(M, K).astype(np.float32)
           @ np.asarray(B).reshape(K, N).astype(np.float32))
    rel_l2 = float(np.linalg.norm(np.asarray(C).reshape(M, N).astype(np.float32) - ref)
                   / np.linalg.norm(ref))

    # ---- depth sweep: B submits, then B waits ----
    depths = {}
    for d in o.depths:
        per = []
        for _ in range(o.depth_reps):
            t0 = time.perf_counter_ns()
            hs = [submit() for _ in range(d)]
            t1 = time.perf_counter_ns()
            for h in hs:
                h.wait()
            t2 = time.perf_counter_ns()
            per.append(((t2 - t0) / 1000.0 / d, (t1 - t0) / 1000.0 / d))
        depths[d] = {
            "per_cmd_us": round(statistics.median(p[0] for p in per), 1),
            "submit_phase_per_cmd_us": round(statistics.median(p[1] for p in per), 2),
        }

    ddr = ddr_account(o.build_dir, suffix)
    macs = M * K * N
    cores = cols * COMPUTE_ROWS
    return {
        "suffix": suffix, "cols": cols, "M": M, "K": K, "N": N, "macs": macs,
        "ddr_mib": ddr["ddr_mib"],
        "t_peak_us": round(1e6 * macs / (cores * MAC_PER_CYCLE_PER_CORE * CORE_CLOCK_HZ), 2),
        "reps": o.reps,
        "submit_us_median": round(statistics.median(subs), 2),
        "wait_us_median": round(statistics.median(waits), 1),
        "total_us_median": round(statistics.median(totals), 1),
        "rel_l2_depth1": rel_l2, "correctness_depth1": "PASS" if rel_l2 <= 0.08 else "FAIL",
        "depths": depths,
    }


def main(o):
    os.makedirs(o.artifacts, exist_ok=True)
    rows = []
    for s in o.suffixes:
        print(f"\n---------- {s} ----------", flush=True)
        r = measure(s, o)
        rows.append(r)
        print(f"  shape {r['M']}x{r['K']}x{r['N']} cols={r['cols']}  DDR {r['ddr_mib']} MiB  "
              f"t_peak {r['t_peak_us']} us", flush=True)
        print(f"  rel-L2 (depth 1) {r['rel_l2_depth1']:.4e}  {r['correctness_depth1']}", flush=True)
        print(f"  depth 1: submit {r['submit_us_median']} us + wait {r['wait_us_median']} us "
              f"= {r['total_us_median']} us", flush=True)
        base = r["depths"][min(r["depths"])]["per_cmd_us"]
        for d in sorted(r["depths"]):
            v = r["depths"][d]
            print(f"    depth {d:3d}: {v['per_cmd_us']:8.1f} us/cmd   "
                  f"(submit phase {v['submit_phase_per_cmd_us']:6.2f})   "
                  f"{base / v['per_cmd_us']:5.2f}x vs depth {min(r['depths'])}", flush=True)
    out = os.path.join(o.artifacts, o.out)
    json.dump(rows, open(out, "w"), indent=2)
    print(f"\nwrote {out}")
    aie_utils.DefaultNPURuntime.cleanup()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--build-dir", default="mlir-aie/programming_examples/basic/"
                                          "matrix_multiplication/whole_array/build")
    p.add_argument("--suffixes", nargs="+", required=True)
    p.add_argument("--M", type=int, default=None)
    p.add_argument("--K", type=int, default=None)
    p.add_argument("--N", type=int, default=None)
    p.add_argument("--reps", type=int, default=21)
    p.add_argument("--depths", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    p.add_argument("--depth-reps", type=int, default=7)
    p.add_argument("--kernel", default="MLIR_AIE")
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--out", default="gemm_dispatch_pipeline.json")
    main(p.parse_args())
