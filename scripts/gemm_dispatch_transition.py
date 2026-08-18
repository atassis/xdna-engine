#!/usr/bin/env python3
# Does the dispatch-queueing win survive a PROGRAM/xclbin TRANSITION?
# (task gemm-offcore-residue-occupancy, item 1b / lever3-dispatch-coalesce)
#
# WHY: queueing d commands before waiting is worth 1.59-1.77x, and a read-after-write between
# consecutive queued commands costs nothing [[dependent-command-chain-keeps-the-dispatch-queueing-win]].
# Both results are bounded by the same scope sentence: every command ran ONE instruction stream inside
# ONE hardware context. The encoder does not. It crosses 743 program boundaries per clip at
# ~1.543 ms each [[switch-cost-per-transition-2026-08-04]], and its block boundaries are exactly
# where the lever has never been tested. If queueing cannot hide a transition, the lever stops at the
# block boundary and the encoder sees it only inside a block; if it can, the largest single line in
# the encoder's overhead is queueable and that is a much bigger result than the 1.77x itself.
#
# THE MEASUREMENT. Two DISTINCT xclbins of the SAME shape, so a command's work is identical and only
# the program differs. Each is loaded into its own pyxrt.hw_context; both stay live. Three arms, all
# interleaved within a rep so box drift cancels:
#   a    -- d commands, all on xclbin A.  0 transitions.
#   b    -- d commands, all on xclbin B.  0 transitions.
#   alt  -- d commands, A,B,A,B,...       d-1 transitions.
# The transition-free counterfactual is a and b at the SAME depth in the SAME rep, mixed in alt's own
# composition (half and half at every even depth, all A at d=1). That is the denominator; the arms do
# NOT need to cost the same, which is why two different tilings are usable here. Each arm is preceded
# by one untimed command on its first kernel, so the only boundaries inside a timed region are the
# ones the arm is there to measure (--no-settle to see what that is worth: ~770 us/cmd at d=2).
#
# WHAT THE TWO HYPOTHESES PREDICT, and they are 5x apart at depth 64:
#   hidden      -- alt(d) ~= (a(d)+b(d))/2, i.e. excess ~= 0 and the win transfers across blocks.
#   serialised  -- every boundary costs the banked ~1543 us, so alt(d) ~= mean(1) + 1543*(d-1)/d,
#                  i.e. ~1520 us/cmd of excess at d=64 against a ~370 us/cmd command.
# The reported number is EXCESS PER TRANSITION = (alt(d) - (a(d)+b(d))/2) * d / (d-1), which is the
# same quantity as the banked per-transition cost and directly comparable to it.
#
# THE GATE IS BIT-EXACT, not thresholded, and it is here because this task has already shipped two
# timing results whose correctness check could not fail. A is +-1 and B is a single-cycle column
# permutation, so bfp16 is lossless (one shared exponent, every value at 2^0), the f32 accumulate
# sums one +-1 against K-1 zeros, and C == np.roll(A, 1, axis=1) EXACTLY. --poison zeroes every
# output slot before the timed region, so a command that never ran, or one that read its slot early,
# returns zeros against +-1 data and every element mismatches; without it each rep rewrites the same
# buffers with the same values and from rep 2 on a dropped command still finds correct bytes.
# --negative-control checks the OTHER direction the gate can be blind: it runs `alt` with B's
# commands skipped, so the arm is not the arm it claims to be, and the run must fail.
#
# Run (NPU free -- use the device wrapper, which stops xdna-engine and npu-vox):
#   bash scripts/gemm_dispatch_transition_device.sh
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
from gemm_command_accounting import design_operands, insts_bin, shape_of, CORE_CLOCK_HZ, \
    MAC_PER_CYCLE_PER_CORE, COMPUTE_ROWS

import pyxrt
from aie.utils.npukernel import NPUKernel
from aie.utils.tensor_factory import tensor, zeros
import aie.utils as aie_utils

DTYPE = {"bf16": bfloat16, "f32": np.float32}


def load_arm(suffix, o):
    """Load one xclbin into its own hw_context and return the raw kernel handle plus its out dtype.

    Reaches past the runtime wrapper to the objects run()'s own timed region uses, so submit and wait
    below are the identical calls rather than a re-implementation of them.
    """
    xclbin = os.path.join(o.build_dir, f"final_{suffix}.xclbin")
    insts = os.path.join(o.build_dir, f"insts_{suffix}.txt")
    for p in (xclbin, insts):
        if not os.path.exists(p):
            sys.exit(f"missing artifact: {p}")
    operands = design_operands(o.build_dir, suffix)
    if not operands:
        sys.exit(f"{suffix}: could not read runtime_sequence operands")
    if len(operands) > 3:
        sys.exit(f"{suffix}: traced design -- run the transition probe on no-trace arms only")

    kern = NPUKernel(xclbin_path=xclbin, insts_path=insts_bin(insts, o.artifacts),
                     kernel_name=o.kernel)
    handle = aie_utils.DefaultNPURuntime.load(kern)
    kh = handle
    while not hasattr(kh, "kernel"):
        kh = kh._handle
    return kh, handle, operands[-1][1]


def main(o):
    os.makedirs(o.artifacts, exist_ok=True)
    rt = aie_utils.DefaultNPURuntime

    M, K, N = shape_of(o.arm_a, o)
    if shape_of(o.arm_b, o) != (M, K, N):
        sys.exit(f"arms differ in shape: {shape_of(o.arm_a, o)} vs {shape_of(o.arm_b, o)}")
    cols_a = int(re.search(r"_(\d+)c_", o.arm_a).group(1))
    cols_b = int(re.search(r"_(\d+)c_", o.arm_b).group(1))
    depth_max = max(o.depths)

    kh_a, h_a, out_a = load_arm(o.arm_a, o)
    kh_b, h_b, out_b = load_arm(o.arm_b, o)
    if out_a != out_b:
        sys.exit(f"arms differ in output dtype: {out_a} vs {out_b}")
    # Two live contexts is the premise of the whole probe. If XRT handed back the same one -- which
    # it does for the same xclbin path -- there is no boundary to cross and the result is vacuous.
    if kh_a.context is kh_b.context:
        sys.exit("both arms share one hw_context -- no program boundary, nothing to measure")

    # +-1 A and a single-cycle permutation B: see the header. Exact through bfp16, the f32
    # accumulate and the drain, so the reference below is an equality rather than a tolerance.
    rng = np.random.default_rng(seed=42)
    a0 = np.where(rng.random((M, K)) < 0.5, -1.0, 1.0).astype(bfloat16)
    b = np.zeros((K, N), dtype=bfloat16)
    b[np.arange(K), (np.arange(K) + 1) % N] = 1.0        # A @ B == np.roll(A, 1, axis=1)

    Bt = tensor(b.reshape(K * N), dtype=bfloat16)
    As = [tensor(np.roll(a0, i, axis=0).reshape(M * K), dtype=bfloat16) for i in range(depth_max)]
    Cs = [zeros((M * N,), dtype=DTYPE[out_a]) for _ in range(depth_max)]

    # Warm both contexts: hw-context creation and BO first-touch are one-time and must not land in a
    # timed region. Running each arm once also proves the shared buffers are legal for both kernels.
    rt.run(h_a, [As[0], Bt, Cs[0]])
    rt.run(h_b, [As[0], Bt, Cs[0]])

    for t in (*As, Bt, *Cs):
        t.to("npu")
    a_bos = [t.buffer_object() for t in As]
    b_bo = Bt.buffer_object()
    c_bos = [t.buffer_object() for t in Cs]

    def insts_of(kh):
        bo = kh.insts_bo
        if not bo:
            bo = rt._tensor_class(kh.insts, flags=pyxrt.bo.cacheable,
                                  group_id=kh.kernel.group_id(1),
                                  xrt_device=rt._device).buffer_object()
        return bo, kh.insts.nbytes

    ibo_a, ib_a = insts_of(kh_a)
    ibo_b, ib_b = insts_of(kh_b)

    def submit(kh, ibo, ib, i):
        return kh.kernel(3, ibo, ib, a_bos[i], b_bo, c_bos[i])

    # Arm -> the kernel each of the d commands runs on. `alt` is the only one that crosses.
    def plan(arm, d):
        if arm == "a":
            return [0] * d
        if arm == "b":
            return [1] * d
        return [i & 1 for i in range(d)]

    def settle(p):
        """Run one untimed command on the arm's FIRST kernel, so the timed region starts with that
        program already the resident one.

        Without this every arm carries one extra boundary INSIDE its timed region -- the arms run
        back to back, so `a` follows the previous rep's `alt` (which ended on B) and `b` follows `a`.
        That inflates a and b by T/d, which is ~770 us/cmd at d=2, and it inflates them into the
        counterfactual, so the excess it produces is biased DOWNWARD. With this, transitions inside
        the timed region are exactly len(set-adjacent-differences) = d-1 for alt and 0 for a and b.
        """
        kh, ibo, ib = kern_of[p[0]]
        kh.kernel(3, ibo, ib, a_bos[0], b_bo, c_bos[0]).wait()

    def poison(d):
        """Zero every output slot this rep is about to write, BEFORE the timed region.

        Without it the gate cannot fail: each rep rewrites the same buffers with the same values, so
        from rep 2 on a command that never ran would find its own correct output already there.
        Costs one transfer per slot per arm and inflates absolute us/cmd, so it stays opt-in and only
        the paired ratio is quoted from a poisoned run.
        """
        for t in Cs[:d]:
            with t.overwrite() as buf:
                buf[:] = 0
            t.to("npu")

    def check(d):
        """Worst mismatch count over the d slots, each against ITS OWN A rotated one column."""
        bad = 0
        for i in range(d):
            ref = np.roll(np.asarray(As[i]).reshape(M, K), 1, axis=1).astype(np.float32)
            got = np.asarray(Cs[i]).reshape(M, N).astype(np.float32)
            bad = max(bad, int(np.count_nonzero(got != ref)))
        return bad

    arms = ("a", "b", "alt")
    kern_of = ((kh_a, ibo_a, ib_a), (kh_b, ibo_b, ib_b))
    depths = {}
    for d in o.depths:
        per = {arm: [] for arm in arms}
        for _ in range(o.depth_reps):
            for arm in arms:
                p = plan(arm, d)
                if o.negative_control and arm == "alt":
                    p = [k for k in p if k == 0]     # drop B's half: the arm is no longer `alt`
                if o.settle:
                    settle(p)                        # before poison: it writes slot 0
                if o.poison:
                    poison(d)
                t0 = time.perf_counter_ns()
                hs = [submit(*kern_of[k], i) for i, k in enumerate(p)]
                t1 = time.perf_counter_ns()
                for h in hs:
                    if h.wait() != pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
                        sys.exit(f"depth {d} arm {arm}: kernel did not complete")
                t2 = time.perf_counter_ns()
                per[arm].append(((t2 - t0) / 1000.0 / d, (t1 - t0) / 1000.0 / d))
        bad = check(d)

        med = {arm: statistics.median(p[0] for p in per[arm]) for arm in arms}
        # Paired WITHIN each rep: the counterfactual for that rep's alt is that rep's own a and b.
        # The medians come from different points in the run and this arm drifts ~+-7% in a session.
        # Weighted by alt's ACTUAL composition, not assumed 50/50 -- at d=1 alt is 100% A, so an
        # (a+b)/2 counterfactual would charge it half of B's cost and report an excess that is
        # nothing but the two arms' cost difference.
        na = sum(1 for k in plan("alt", d) if k == 0)
        wa, wb = na / d, (d - na) / d
        base = [wa * x[0] + wb * y[0] for x, y in zip(per["a"], per["b"])]
        ratios = [c[0] / q for c, q in zip(per["alt"], base)]
        rmean = statistics.fmean(ratios)
        rsem = statistics.stdev(ratios) / len(ratios) ** 0.5 if len(ratios) > 1 else 0.0
        # Excess per transition, in the same units as the banked ~1543 us/transition: alt carries
        # d-1 boundaries that the (a+b)/2 counterfactual carries none of.
        exc = [(c[0] - q) * d / (d - 1) for c, q in zip(per["alt"], base)] if d > 1 else []
        emean = statistics.fmean(exc) if exc else 0.0
        esem = statistics.stdev(exc) / len(exc) ** 0.5 if len(exc) > 1 else 0.0
        depths[d] = {
            "a_per_cmd_us": round(med["a"], 1),
            "b_per_cmd_us": round(med["b"], 1),
            "alt_per_cmd_us": round(med["alt"], 1),
            "counterfactual_per_cmd_us": round(statistics.median(base), 1),
            "alt_submit_phase_per_cmd_us": round(statistics.median(p[1] for p in per["alt"]), 2),
            "alt_over_counterfactual_paired_mean": round(rmean, 4),
            "alt_over_counterfactual_ci95": [round(rmean - 1.96 * rsem, 4),
                                             round(rmean + 1.96 * rsem, 4)],
            "excess_per_transition_us": round(emean, 1),
            "excess_per_transition_ci95": [round(emean - 1.96 * esem, 1),
                                           round(emean + 1.96 * esem, 1)],
            "transitions": max(d - 1, 0),
            "alt_slower_reps": sum(1 for r in ratios if r > 1.0),
            "reps_counted": len(ratios),
            "mismatches": bad,
            "correctness": "PASS" if bad == 0 else "FAIL",
        }

    macs = M * K * N
    row = {
        "arm_a": o.arm_a, "arm_b": o.arm_b, "cols_a": cols_a, "cols_b": cols_b,
        "M": M, "K": K, "N": N, "macs": macs, "out_dtype": out_a,
        "t_peak_a_us": round(1e6 * macs / (cols_a * COMPUTE_ROWS * MAC_PER_CYCLE_PER_CORE
                                           * CORE_CLOCK_HZ), 2),
        "t_peak_b_us": round(1e6 * macs / (cols_b * COMPUTE_ROWS * MAC_PER_CYCLE_PER_CORE
                                           * CORE_CLOCK_HZ), 2),
        "depth_reps": o.depth_reps, "poison": bool(o.poison), "settle": bool(o.settle),
        "negative_control": bool(o.negative_control),
        "depths": depths,
    }

    print(f"\n---------- {o.arm_a}  vs  {o.arm_b} ----------", flush=True)
    print(f"  shape {M}x{K}x{N}  cols {cols_a}/{cols_b}  out {out_a}  "
          f"t_peak {row['t_peak_a_us']}/{row['t_peak_b_us']} us", flush=True)
    failed = []
    for d in sorted(depths):
        v = depths[d]
        if v["correctness"] != "PASS":
            failed.append(f"depth {d}: {v['mismatches']} mismatched elements")
        print(f"    depth {d:3d}: a {v['a_per_cmd_us']:8.1f}  b {v['b_per_cmd_us']:8.1f}  "
              f"alt {v['alt_per_cmd_us']:8.1f} us/cmd   "
              f"alt/(a+b)/2 {v['alt_over_counterfactual_paired_mean']:5.3f} "
              f"[{v['alt_over_counterfactual_ci95'][0]:.3f},"
              f"{v['alt_over_counterfactual_ci95'][1]:.3f}]   "
              f"excess/transition {v['excess_per_transition_us']:8.1f} us "
              f"[{v['excess_per_transition_ci95'][0]:.1f},"
              f"{v['excess_per_transition_ci95'][1]:.1f}]   "
              f"{v['alt_slower_reps']:>2d}/{v['reps_counted']} slower   "
              f"{v['mismatches']:>7d} bad {v['correctness']}", flush=True)

    out = os.path.join(o.artifacts, o.out)
    json.dump([row], open(out, "w"), indent=2)
    print(f"\nwrote {out}")
    rt.cleanup()
    if o.negative_control:
        if failed:
            print("negative control OK -- the truncated arm was caught")
            return
        sys.exit("NEGATIVE CONTROL DID NOT FAIL -- the transition gate is not a gate")
    if failed:
        sys.exit("CORRECTNESS FAILED:\n  " + "\n  ".join(failed))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--build-dir", default="mlir-aie/programming_examples/basic/"
                                          "matrix_multiplication/whole_array/build")
    p.add_argument("--arm-a", required=True)
    p.add_argument("--arm-b", required=True)
    p.add_argument("--M", type=int, default=None)
    p.add_argument("--K", type=int, default=None)
    p.add_argument("--N", type=int, default=None)
    p.add_argument("--depths", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    p.add_argument("--depth-reps", type=int, default=15)
    p.add_argument("--poison", action="store_true")
    p.add_argument("--no-settle", dest="settle", action="store_false",
                   help="do not pre-run the arm's first kernel; leaves one boundary "
                        "transition inside every arm's timed region")
    p.add_argument("--negative-control", action="store_true")
    p.add_argument("--kernel", default="MLIR_AIE")
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--out", default="gemm_dispatch_transition.json")
    main(p.parse_args())
