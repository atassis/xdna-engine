#!/usr/bin/env python3
# Split the whole_array GEMM's per-command floor into HOST ROUND-TRIP and DEVICE-SERIAL work, and
# gate the queueing win on correctness (task gemm-offcore-residue-occupancy, item 1).
#
# WHY: the shape series measured a per-command floor of ~140 us that is invariant in cols, M, K and
# N -- at the production shape it is a third of the command and it is the largest lever left. But
# `npu_time` is perf_counter_ns around BOTH `kernel(...)` (submit) and `h.wait()` (block until
# complete), so a host/driver round-trip is inside every number this task has ever quoted. A floor
# that is submit+wait latency and a floor that is device-serial work are different findings with
# different fixes.
#
# TWO MEASUREMENTS, no rebuild:
#   * SPLIT: time the submit call and the wait call separately at depth 1. Submit returns a run
#     handle without blocking, so this is the cleanest available cut between the two.
#   * DEPTH SWEEP: submit B commands back to back, THEN wait on all of them, and report total/B.
#     Anything that is per-command host round-trip amortizes as B grows; anything the device must
#     do serially per command does not. The asymptote is the real device-serial floor.
#
# CORRECTNESS -- what changed, and why the previous form could not be gated. Pass 10 ran every
# in-flight command through ONE A/B/C triple, so the deep arms were reported timing-only. Sharing C
# alone would not have been enough to fix that: with one A and one B every command computes the
# SAME product, so racing writes deposit identical bytes and rel-L2 passes whether or not the
# commands actually stayed separate. The gate has to be falsifiable, so each command now gets its
# OWN A and its OWN C (B is shared, which is also the real-use shape: one weight matrix, different
# activations). Command i must leave A_i @ B in C_i, so a command whose output lands in the wrong
# buffer, or is clobbered by a neighbour, now FAILS.
#
# The two C modes run interleaved rep by rep rather than in sequence, because the same arm has been
# measured to drift ~+-7% across instrument paths within one session; only a paired comparison can
# say whether per-command buffers cost anything. `shared` reproduces the earlier configuration and
# is kept to measure that delta -- it is NOT to be quoted as a correctness result.
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
    depth_max = max(o.depths)
    rng = np.random.default_rng(seed=42)
    # Distinct A per slot so a misrouted or clobbered output is detectable; one shared B.
    As = [tensor(rng.standard_normal((M * K,)).astype(bfloat16), dtype=bfloat16)
          for _ in range(depth_max)]
    B = tensor(rng.standard_normal((K * N,)).astype(bfloat16), dtype=bfloat16)
    Cs = [zeros((M * N,), dtype=np.float32) for _ in range(depth_max)]

    kern = NPUKernel(xclbin_path=xclbin, insts_path=insts_bin(insts, o.artifacts),
                     kernel_name=o.kernel)
    rt = aie_utils.DefaultNPURuntime
    handle = rt.load(kern)
    rt.run(handle, [As[0], B, Cs[0]])  # warmup carries hw-context creation and BO first-touch

    # Reach past the runtime wrapper to the same objects its own timed region uses, so the split
    # measures the identical calls rather than a re-implementation of them. This mirrors run()
    # exactly: to("npu") and buffer_object() sit OUTSIDE its timed region, which is why npu_time
    # excludes host buffer sync -- and why this probe must leave them outside too.
    kh = handle
    while not hasattr(kh, "kernel"):
        kh = kh._handle
    for t in (*As, B, *Cs):
        t.to("npu")
    a_bos = [t.buffer_object() for t in As]
    b_bo = B.buffer_object()
    c_bos = [t.buffer_object() for t in Cs]
    insts_bytes = kh.insts.nbytes
    insts_bo = kh.insts_bo
    if not insts_bo:
        insts_bo = rt._tensor_class(kh.insts, flags=pyxrt.bo.cacheable,
                                    group_id=kh.kernel.group_id(1),
                                    xrt_device=rt._device).buffer_object()

    def submit(i):
        return kh.kernel(3, insts_bo, insts_bytes, a_bos[i], b_bo, c_bos[i])

    B_np = np.asarray(B).reshape(K, N).astype(np.float32)
    refs = {}

    def check(slots):
        """Worst rel-L2 over `slots`, each against ITS OWN A @ B."""
        worst = 0.0
        for i in slots:
            if i not in refs:
                refs[i] = np.asarray(As[i]).reshape(M, K).astype(np.float32) @ B_np
            got = np.asarray(Cs[i]).reshape(M, N).astype(np.float32)
            worst = max(worst, float(np.linalg.norm(got - refs[i])
                                     / np.linalg.norm(refs[i])))
        return worst

    # ---- depth 1, submit and wait timed separately ----
    subs, waits, totals = [], [], []
    for _ in range(o.reps):
        t0 = time.perf_counter_ns()
        h = submit(0)
        t1 = time.perf_counter_ns()
        r = h.wait()
        t2 = time.perf_counter_ns()
        if r != pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
            sys.exit(f"{suffix}: kernel returned {r}")
        subs.append((t1 - t0) / 1000.0)
        waits.append((t2 - t1) / 1000.0)
        totals.append((t2 - t0) / 1000.0)
    rel_l2 = check([0])

    # ---- depth sweep: B submits, then B waits; per-command and shared-C paired per rep ----
    depths = {}
    for d in o.depths:
        per, shared = [], []
        for _ in range(o.depth_reps):
            for mode, acc in (("per", per), ("shared", shared)):
                slots = range(d) if mode == "per" and not o.negative_control else (0,) * d
                t0 = time.perf_counter_ns()
                hs = [submit(i) for i in slots]
                t1 = time.perf_counter_ns()
                for h in hs:
                    h.wait()
                t2 = time.perf_counter_ns()
                acc.append(((t2 - t0) / 1000.0 / d, (t1 - t0) / 1000.0 / d))
        # Verify after the timing reps: every slot holds the last command that wrote it, and every
        # command at this depth wrote a distinct slot, so all d are live results.
        worst = check(range(d))
        depths[d] = {
            "per_cmd_us": round(statistics.median(p[0] for p in per), 1),
            "submit_phase_per_cmd_us": round(statistics.median(p[1] for p in per), 2),
            "shared_c_per_cmd_us": round(statistics.median(p[0] for p in shared), 1),
            "rel_l2_worst": worst,
            "correctness": "PASS" if worst <= o.rel_l2_max else "FAIL",
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
        "rel_l2_depth1": rel_l2, "correctness_depth1": "PASS" if rel_l2 <= o.rel_l2_max else "FAIL",
        "depths": depths,
    }


def main(o):
    os.makedirs(o.artifacts, exist_ok=True)
    rows, failed = [], []
    for s in o.suffixes:
        print(f"\n---------- {s} ----------", flush=True)
        r = measure(s, o)
        rows.append(r)
        print(f"  shape {r['M']}x{r['K']}x{r['N']} cols={r['cols']}  DDR {r['ddr_mib']} MiB  "
              f"t_peak {r['t_peak_us']} us", flush=True)
        print(f"  rel-L2 (depth 1) {r['rel_l2_depth1']:.4e}  {r['correctness_depth1']}", flush=True)
        print(f"  depth 1: submit {r['submit_us_median']} us + wait {r['wait_us_median']} us "
              f"= {r['total_us_median']} us", flush=True)
        d0 = min(r["depths"])
        base = r["depths"][d0]["per_cmd_us"]
        for d in sorted(r["depths"]):
            v = r["depths"][d]
            if v["correctness"] != "PASS":
                failed.append(f"{s} depth {d} rel-L2 {v['rel_l2_worst']:.4e}")
            print(f"    depth {d:3d}: {v['per_cmd_us']:8.1f} us/cmd   "
                  f"(submit phase {v['submit_phase_per_cmd_us']:6.2f})   "
                  f"{base / v['per_cmd_us']:5.2f}x vs depth {d0}   "
                  f"sharedC {v['shared_c_per_cmd_us']:8.1f}   "
                  f"rel-L2 {v['rel_l2_worst']:.3e} {v['correctness']}", flush=True)
    out = os.path.join(o.artifacts, o.out)
    json.dump(rows, open(out, "w"), indent=2)
    print(f"\nwrote {out}")
    aie_utils.DefaultNPURuntime.cleanup()
    if failed:
        sys.exit("CORRECTNESS FAILED:\n  " + "\n  ".join(failed))


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
    p.add_argument("--rel-l2-max", type=float, default=0.08)
    # Self-test: route every command back to slot 0 while still checking all d slots. A gate that
    # cannot fail proves nothing, and this task has already shipped one instrument whose two
    # counters turned out to be the same counter. Expect FAIL at every depth > 1.
    p.add_argument("--negative-control", action="store_true")
    p.add_argument("--kernel", default="MLIR_AIE")
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--out", default="gemm_dispatch_pipeline_gated.json")
    main(p.parse_args())
