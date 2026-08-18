#!/usr/bin/env python3
# Does the dispatch-queueing win survive a DEPENDENT command chain?
# (task gemm-offcore-residue-occupancy, item 1 / lever3-dispatch-coalesce)
#
# WHY: queueing d whole_array GEMM commands before waiting measured 382.3 -> 240.1 us/cmd, 1.59x, at
# the production shape. Every command in that measurement was INDEPENDENT. The encoder is not: each
# GEMM consumes the previous one's output out of DDR. If a read-after-write between consecutive
# commands forces the device to serialise, the lever does not transfer and the encoder never sees it.
# That is the question this probe answers, and it is the one that decides whether the largest measured
# lever of the day is worth anything to the shipped pipeline.
#
# THE CHAIN IS A REAL RAW HAZARD, not a simulated one: command i is submitted with buffer X[i] as its
# A operand and X[i+1] as its C operand, so command i+1 reads the buffer command i writes, with no
# host touch in between. That is exactly the encoder's shape.
#
# WHY bf16 OUT: C must be re-readable as A for the chain to close, and the shipped f32-out design
# cannot do that -- C is [M,N] f32 and A is [M,K] bf16, so feeding one to the other reinterprets each
# f32 as two bf16s. ~0.4% of those land on an all-ones exponent, so the chain NaNs out within a
# command or two and the correctness gate stops being able to fail. dtype_out=bf16 with K == N makes
# C and A the same 1 MiB layout and the alias exact.
#
# WHY +-1 DATA AND A PERMUTATION B -- the gate is BIT-EXACT, no threshold. B is a single 1024-cycle
# (column rotate by one), so A @ B is a column rotation and a chain of d is a rotation by d. With every
# A entry +-1 the bfp16 path is lossless (one shared exponent per block, every value at 2^0), the f32
# accumulate sums one +-1 against 1023 zeros, and the bf16 drain is exact. So the expected chain output
# is np.roll(A_0, d, axis=1) EXACTLY, and any reordering, dropped command, or partially-written buffer
# is a bit difference rather than a number to argue about against a tolerance.
#   * B is order 1024, so P^d != P^(d-1) for every depth here: the gate distinguishes a chain of d
#     from a chain of d-1. That is checked, not assumed (--self-test prints the off-by-one distance).
#   * --negative-control feeds every command X[0] instead of its predecessor's output, so the chain is
#     broken while the check is unchanged. It must FAIL, and a run that passes it is not a gate.
#
# ARMS, both on the same artifact and interleaved rep by rep (the same arm drifts ~+-7% between
# instrument paths within a session, so only a paired comparison is worth quoting):
#   indep -- distinct A[i] -> distinct C[i], no hazard. Reproduces the banked lever on THIS artifact,
#            which is what makes the chain number comparable to it rather than to a different build.
#   chain -- X[i] -> X[i+1], the RAW hazard above.
#
# Run (NPU free -- use the device wrapper, which stops xdna-engine and npu-vox):
#   bash scripts/gemm_dispatch_chain_device.sh
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
        sys.exit(f"{suffix}: traced design -- run the chain probe on no-trace arms only")

    M, K, N = shape_of(suffix, o)
    if K != N:
        sys.exit(f"{suffix}: chain needs K == N so C aliases A, got K={K} N={N}")
    depth_max = max(o.depths)

    # +-1 A and a single-cycle permutation B: see the header. Both are exact through bfp16, the f32
    # accumulate and the bf16 drain, so the chain reference below is an equality, not a tolerance.
    rng = np.random.default_rng(seed=42)
    a0 = np.where(rng.random((M, K)) < 0.5, -1.0, 1.0).astype(bfloat16)
    b = np.zeros((K, N), dtype=bfloat16)
    b[np.arange(K), (np.arange(K) + 1) % N] = 1.0        # A @ B == np.roll(A, 1, axis=1)

    B = tensor(b.reshape(K * N), dtype=bfloat16)
    # indep arm: distinct A per slot, distinct C per slot -- no hazard between commands.
    As = [tensor(np.roll(a0, i, axis=0).reshape(M * K), dtype=bfloat16) for i in range(depth_max)]
    Cs = [zeros((M * N,), dtype=bfloat16) for _ in range(depth_max)]
    # chain arm: X[0] is the input, X[i+1] is command i's output AND command i+1's input.
    Xs = [tensor(a0.reshape(M * K).copy(), dtype=bfloat16)] + \
         [zeros((M * N,), dtype=bfloat16) for _ in range(depth_max)]

    kern = NPUKernel(xclbin_path=xclbin, insts_path=insts_bin(insts, o.artifacts),
                     kernel_name=o.kernel)
    rt = aie_utils.DefaultNPURuntime
    handle = rt.load(kern)
    rt.run(handle, [As[0], B, Cs[0]])  # warmup carries hw-context creation and BO first-touch

    # Reach past the runtime wrapper to the objects its own timed region uses, so submit/wait here are
    # the identical calls run() makes rather than a re-implementation. to("npu") and buffer_object()
    # stay OUTSIDE the timed region for the same reason npu_time excludes host buffer sync.
    kh = handle
    while not hasattr(kh, "kernel"):
        kh = kh._handle
    for t in (*As, B, *Cs, *Xs):
        t.to("npu")
    a_bos = [t.buffer_object() for t in As]
    b_bo = B.buffer_object()
    c_bos = [t.buffer_object() for t in Cs]
    x_bos = [t.buffer_object() for t in Xs]
    insts_bytes = kh.insts.nbytes
    insts_bo = kh.insts_bo
    if not insts_bo:
        insts_bo = rt._tensor_class(kh.insts, flags=pyxrt.bo.cacheable,
                                    group_id=kh.kernel.group_id(1),
                                    xrt_device=rt._device).buffer_object()

    def submit_indep(i):
        return kh.kernel(3, insts_bo, insts_bytes, a_bos[i], b_bo, c_bos[i])

    def submit_chain(i):
        """Command i reads X[i] -- which command i-1 wrote -- and writes X[i+1].

        X[0] is only ever an A operand, never a C operand, so the chain input is never clobbered and
        no per-rep restore is needed. That matters beyond tidiness: writing X[0] through __array__
        would not mark the buffer dirty in the coherence map, so the following to("npu") would skip
        the transfer and the device would keep reading whatever it already had.
        """
        src = 0 if o.negative_control else i
        return kh.kernel(3, insts_bo, insts_bytes, x_bos[src], b_bo, x_bos[i + 1])

    def poison(mode, d):
        """Zero every buffer this rep is about to write, BEFORE the timed region.

        Without this the gate is unfalsifiable, and in the same way this task has already been
        caught twice. Each rep writes the same buffers with the same values, so from rep 2 on every
        output slot ALREADY holds its correct contents: a command that never ran, or one that raced
        ahead and read its predecessor's buffer early, would still find the right bytes there and
        the check would pass. Zeroing first means a stale or early read returns zeros against +-1
        data, so every element mismatches.

        overwrite() rather than mutate(): every byte is replaced, so it skips the read-back and
        costs one transfer. Both arms are poisoned so the paired comparison stays symmetric -- the
        indep arm's C slots have exactly the same staleness hole.
        """
        for t in (Cs[:d] if mode == "indep" else Xs[1:d + 1]):
            with t.overwrite() as buf:
                buf[:] = 0
            t.to("npu")

    def check_indep(slots):
        """Worst rel-L2 over `slots`, each against ITS OWN A @ B.

        B is the single-cycle permutation, so A @ B is A rotated one column and the reference is a
        roll rather than a 512x1024x1024 matmul per slot -- same reference, ~0 cost, and exact.
        """
        worst = 0.0
        for i in slots:
            ref = np.roll(np.asarray(As[i]).reshape(M, K), 1, axis=1).astype(np.float32)
            got = np.asarray(Cs[i]).reshape(M, N).astype(np.float32)
            worst = max(worst, float(np.linalg.norm(got - ref) / np.linalg.norm(ref)))
        return worst

    def check_chain(d):
        """X[d] must be A_0 rotated d columns, BIT-EXACT. Returns (mismatches, offby1_mismatches)."""
        got = np.asarray(Xs[d]).reshape(M, N)
        ref = np.roll(a0, d, axis=1)
        off = np.roll(a0, d - 1, axis=1)      # the chain one command short
        return int(np.count_nonzero(got != ref)), int(np.count_nonzero(ref != off))

    # ---- depth 1, submit and wait timed separately (same cut as the pipeline probe) ----
    subs, waits, totals = [], [], []
    for _ in range(o.reps):
        t0 = time.perf_counter_ns()
        h = submit_indep(0)
        t1 = time.perf_counter_ns()
        r = h.wait()
        t2 = time.perf_counter_ns()
        if r != pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
            sys.exit(f"{suffix}: kernel returned {r}")
        subs.append((t1 - t0) / 1000.0)
        waits.append((t2 - t1) / 1000.0)
        totals.append((t2 - t0) / 1000.0)
    rel_l2 = check_indep([0])

    # ---- depth sweep: d submits, then d waits; indep and chain paired within each rep ----
    depths = {}
    for d in o.depths:
        indep, chain = [], []
        for _ in range(o.depth_reps):
            for mode, acc in (("indep", indep), ("chain", chain)):
                if o.poison:
                    poison(mode, d)
                sub = submit_indep if mode == "indep" else submit_chain
                t0 = time.perf_counter_ns()
                hs = [sub(i) for i in range(d)]
                t1 = time.perf_counter_ns()
                for h in hs:
                    h.wait()
                t2 = time.perf_counter_ns()
                acc.append(((t2 - t0) / 1000.0 / d, (t1 - t0) / 1000.0 / d))
        # Verify after the timing reps: the indep arm's d slots each hold their own product, and the
        # chain arm's tail holds A_0 rotated exactly d columns.
        worst = check_indep(range(d))
        bad, offby1 = check_chain(d)
        indep_us = statistics.median(p[0] for p in indep)
        chain_us = statistics.median(p[0] for p in chain)
        # The two arms are interleaved WITHIN each rep, so the per-rep ratio is a paired sample and
        # box drift cancels in it. Quote this, not the ratio of the two medians: the medians come
        # from different points in the run and this arm drifts ~+-7% across a session.
        ratios = [c[0] / i[0] for i, c in zip(indep, chain)]
        rmean = statistics.fmean(ratios)
        rsem = statistics.stdev(ratios) / len(ratios) ** 0.5 if len(ratios) > 1 else 0.0
        depths[d] = {
            "indep_per_cmd_us": round(indep_us, 1),
            "chain_per_cmd_us": round(chain_us, 1),
            "indep_submit_phase_per_cmd_us": round(statistics.median(p[1] for p in indep), 2),
            "chain_submit_phase_per_cmd_us": round(statistics.median(p[1] for p in chain), 2),
            "chain_over_indep_paired_mean": round(rmean, 4),
            "chain_over_indep_ci95": [round(rmean - 1.96 * rsem, 4), round(rmean + 1.96 * rsem, 4)],
            "chain_slower_reps": sum(1 for r in ratios if r > 1.0),
            "reps_counted": len(ratios),
            "indep_rel_l2_worst": worst,
            "indep_correctness": "PASS" if worst <= o.rel_l2_max else "FAIL",
            "chain_mismatches": bad,
            # Sensitivity: how many elements a chain one command short would differ by. If this is 0
            # the reference cannot tell depth d from d-1 and the gate proves nothing at this depth.
            "chain_offby1_distance": offby1,
            "chain_correctness": "PASS" if bad == 0 else "FAIL",
        }

    ddr = ddr_account(o.build_dir, suffix)
    macs = M * K * N
    cores = cols * COMPUTE_ROWS
    return {
        "suffix": suffix, "cols": cols, "M": M, "K": K, "N": N, "macs": macs,
        "ddr_mib": ddr["ddr_mib"],
        "t_peak_us": round(1e6 * macs / (cores * MAC_PER_CYCLE_PER_CORE * CORE_CLOCK_HZ), 2),
        "reps": o.reps, "depth_reps": o.depth_reps,
        "negative_control": bool(o.negative_control), "poison": bool(o.poison),
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
        print(f"  rel-L2 (depth 1, indep) {r['rel_l2_depth1']:.4e}  {r['correctness_depth1']}",
              flush=True)
        print(f"  depth 1: submit {r['submit_us_median']} us + wait {r['wait_us_median']} us "
              f"= {r['total_us_median']} us", flush=True)
        d0 = min(r["depths"])
        base_i = r["depths"][d0]["indep_per_cmd_us"]
        base_c = r["depths"][d0]["chain_per_cmd_us"]
        for d in sorted(r["depths"]):
            v = r["depths"][d]
            if v["indep_correctness"] != "PASS":
                failed.append(f"{s} depth {d} indep rel-L2 {v['indep_rel_l2_worst']:.4e}")
            if v["chain_correctness"] != "PASS":
                failed.append(f"{s} depth {d} chain {v['chain_mismatches']} mismatched elements")
            if d > 1 and v["chain_offby1_distance"] == 0:
                failed.append(f"{s} depth {d} chain gate is INSENSITIVE (P^d == P^(d-1))")
            print(f"    depth {d:3d}: indep {v['indep_per_cmd_us']:8.1f} us/cmd "
                  f"({base_i / v['indep_per_cmd_us']:5.2f}x)   "
                  f"chain {v['chain_per_cmd_us']:8.1f} us/cmd "
                  f"({base_c / v['chain_per_cmd_us']:5.2f}x)   "
                  f"chain/indep {v['chain_over_indep_paired_mean']:5.3f} "
                  f"[{v['chain_over_indep_ci95'][0]:.3f},{v['chain_over_indep_ci95'][1]:.3f}] "
                  f"{v['chain_slower_reps']:>2d}/{v['reps_counted']} slower   "
                  f"relL2 {v['indep_rel_l2_worst']:.3e} {v['indep_correctness']}   "
                  f"chain {v['chain_mismatches']:>7d} bad {v['chain_correctness']}", flush=True)
    out = os.path.join(o.artifacts, o.out)
    json.dump(rows, open(out, "w"), indent=2)
    print(f"\nwrote {out}")
    aie_utils.DefaultNPURuntime.cleanup()
    if o.negative_control:
        # Inverted: the broken chain MUST be caught. A clean run here means the gate cannot fail.
        if any("chain" in f for f in failed):
            print("negative control OK -- the broken chain was caught")
            return
        sys.exit("NEGATIVE CONTROL DID NOT FAIL -- the chain gate is not a gate")
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
    p.add_argument("--depths", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    p.add_argument("--depth-reps", type=int, default=7)
    p.add_argument("--rel-l2-max", type=float, default=0.08)
    # Self-test: feed every command X[0] instead of its predecessor's output. The chain is broken
    # while the check is unchanged, so every depth > 1 must FAIL. A gate that cannot fail proves
    # nothing, and this task has already shipped one instrument whose two counters were one counter.
    p.add_argument("--negative-control", action="store_true")
    # Strong-gate control (see poison()). OFF by default because it is not free: zeroing and
    # re-sending every output slot before each rep costs ~64 MiB of host->device traffic per rep per
    # arm at depth 64, and MEASURED that inflates per-command time 222 -> 282 us at that depth, which
    # flattens the depth-1 -> depth-64 speedup from 1.67x to 1.38x. It perturbs both arms equally, so
    # the PAIRED chain/indep ratio survives it and the absolute us/cmd does not. Read the two runs
    # accordingly: unpoisoned for the speedup, poisoned for the correctness verdict.
    p.add_argument("--poison", action="store_true")
    p.add_argument("--kernel", default="MLIR_AIE")
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--out", default="gemm_dispatch_chain.json")
    main(p.parse_args())
