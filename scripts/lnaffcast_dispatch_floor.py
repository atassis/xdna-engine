#!/usr/bin/env python3
# Is the lnaffcast 1x route's 151.0 us fixed floor its own cost, or the per-command host round-trip
# every command on this box pays? (task mode-switched-multi-program-xclbin, its `next:` item 2)
#
# THE LEAD, AND WHY IT NEEDED RE-FRAMING. The task recorded the 151.0 us as "near the ~170 us
# decode-GEMV dispatch floor -- a lead, not a finding". But a third floor is already measured and
# already DECOMPOSED: the whole_array GEMM's per-command floor is 139.7 us direct-read, of which
# 87.9 us (63%) amortizes by queueing commands instead of submit-then-wait and 51.8 us (37%) is
# device-serial ([[gemm-per-command-floor-is-mostly-host-round-trip]]). Three floors in one band on
# one box is not a coincidence to chase pairwise -- it is a hypothesis with a sharp test.
#
# THE SHARP PREDICTION. If the floor is a generic per-command host round-trip, the AMORTIZABLE part
# is the same absolute microseconds on every arm regardless of what the command computes, while the
# FRACTION it represents differs because the commands differ in size. If instead each op has its
# own floor, the absolute amortizable parts differ. Fractions cannot separate those two; absolutes
# can, which is why this reports both and gates the verdict on the absolutes.
#
# TWO MEASUREMENTS, NO REBUILD, mirroring scripts/gemm_dispatch_pipeline.py so the numbers are
# comparable to the GEMM's by construction:
#   * SPLIT: at depth 1, time the submit call and the wait call separately. Submit returns a run
#     handle without blocking, so this is the cleanest available cut between host and device.
#   * DEPTH SWEEP: submit d commands back to back, THEN wait on all of them, report total/d. What
#     is per-command host round-trip amortizes as d grows; what the device must do serially does
#     not. The asymptote is the device-serial floor.
#
# ONE HARDWARE CONTEXT, THREE STREAMS. lnaffcast_rows is insts-only and the GEMM stream is a mode
# of the same array program, so the 256-row arm, the 512-row arm and the GEMM stream all run on the
# ONE loaded xclbin. The program-transition tax (~1.543 ms) is therefore outside the comparison
# rather than subtracted from it.
#
# CORRECTNESS IS FALSIFIABLE AT EVERY DEPTH. Each in-flight command gets its OWN x (on B, which is
# where the 1x route reads it) and its OWN C; gb shares one A, which is also the real-use shape --
# one norm weight, different activations. Command i must leave lnaffcast(x_i) in C_i, so a command
# whose output lands in the wrong buffer, or is clobbered by a neighbour, FAILS. Sharing one x
# across the batch would make every command compute the same bytes and parity would pass whether
# or not the commands stayed separate -- that is the trap gemm_dispatch_pipeline.py documents, and
# `--negative-control` reproduces it deliberately.
#
# The GEMM arm carries its OWN operand set rather than borrowing the lnaffcast one: under the mode
# layout A holds only gb and is otherwise zero, so a GEMM against it would be gated on a near-zero
# reference. Random A_i @ shared B is well conditioned and is the same shape the GEMM probe uses.
#
# Run (NPU free -- use the device wrapper, which stops xdna-engine and npu-vox):
#   bash scripts/lnaffcast_dispatch_floor_device.sh
import argparse
import json
import os
import statistics
import sys
import time

import numpy as np
from ml_dtypes import bfloat16

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gemm_command_accounting import design_operands, insts_bin

import pyxrt
from aie.utils.npukernel import NPUKernel
from aie.utils.tensor_factory import tensor
import aie.utils as aie_utils

M, K, N = 512, 1024, 1024
COLS = 1024      # the lnaffcast embedding dim
EPS = 1e-5       # mm_mode_lnaffcast.cc's epsilon

# The 4-column set is the same route on a second axis, so the arms are named off --base rather
# than off one design.
BASE = "512x1024x1024_32x32x128_8c_modalidbf16outkrtpkrllnaff1024scat21x"


def lnaffcast_operands(rows, slots, rng):
    """Shared A carrying gb, one B per slot carrying that slot's x, and the per-slot references.

    The 1x route reads x off the per-column B fifo and gb off the per-row A one, both as f32
    reinterpreted inside the GEMM's bf16 operand buffers. C is poisoned with NaN so a command that
    wrote nothing reads as nan rather than as whatever ran before it.
    """
    gamma = rng.normal(1.0, 0.1, COLS).astype(np.float32)
    beta = rng.normal(0.0, 0.1, COLS).astype(np.float32)
    a = np.zeros(M * K, dtype=bfloat16)
    a.view(np.float32)[:COLS] = gamma
    a.view(np.float32)[COLS: 2 * COLS] = beta

    bs, refs = [], []
    for _ in range(slots):
        x = rng.standard_normal((rows, COLS)).astype(np.float32)
        b = np.zeros(K * N, dtype=bfloat16)
        b.view(np.float32)[: rows * COLS] = x.reshape(-1)
        bs.append(b)
        mean = x.mean(axis=1, keepdims=True)
        var = ((x - mean) ** 2).mean(axis=1, keepdims=True)
        y = ((x - mean) / np.sqrt(var + EPS)) * gamma + beta
        refs.append(y.astype(bfloat16).astype(np.float64))
    return a, bs, refs


def gemm_operands(slots, rng):
    """One B per slot is what the lnaffcast arms vary, so vary the same operand here -- the arms
    then differ in what they compute and not in which buffer carries the per-command data."""
    a = rng.standard_normal((M * K,)).astype(bfloat16)
    a_f32 = np.asarray(a).reshape(M, K).astype(np.float32)
    bs, refs = [], []
    for _ in range(slots):
        b = rng.standard_normal((K * N,)).astype(bfloat16)
        bs.append(b)
        refs.append((a_f32 @ np.asarray(b).reshape(K, N).astype(np.float32)))
    return a, bs, refs


def rel_l2(got, ref):
    return float(np.linalg.norm(got - ref) / np.linalg.norm(ref))


def main(o):
    os.makedirs(o.artifacts, exist_ok=True)
    rt = aie_utils.DefaultNPURuntime
    depth_max = max(o.depths)

    host = o.host
    xclbin = os.path.join(o.build_dir, f"final_{host}.xclbin")
    if not os.path.exists(xclbin):
        sys.exit(f"missing host xclbin: {xclbin}")
    kern = NPUKernel(xclbin_path=xclbin,
                     insts_path=insts_bin(os.path.join(o.build_dir, f"insts_{host}.txt"),
                                          o.artifacts),
                     kernel_name=o.kernel)
    kh = rt.load(kern)
    while not hasattr(kh, "kernel"):
        kh = kh._handle

    # ---- every arm's instruction stream, bound to the one loaded kernel ----
    arms = []
    for spec in o.arms:
        name, kind, rows = spec.split(":")
        rows = int(rows)
        suffix = o.gemm_suffix if kind == "gemm" else f"{o.base}rtp18r{rows}g4ctgc"
        insts = os.path.join(o.build_dir, f"insts_{suffix}.txt")
        if not os.path.exists(insts):
            sys.exit(f"missing insts: {insts}")
        if len(design_operands(o.build_dir, suffix)) > 3:
            sys.exit(f"{suffix}: traced design -- this probe runs no-trace arms only")
        arr = np.frombuffer(open(insts, "rb").read(), dtype=np.uint32)
        ibo = rt._tensor_class(arr, flags=pyxrt.bo.cacheable,
                               group_id=kh.kernel.group_id(1),
                               xrt_device=rt._device).buffer_object()
        arms.append({"name": name, "kind": kind, "rows": rows, "suffix": suffix,
                     "arr": arr, "ibo": ibo})

    # ---- operands: one set per arm kind, own-B and own-C per slot ----
    rng = np.random.default_rng(seed=11)
    sets = {}
    # Keyed by KIND, not by row count. lnaffcast normalizes each row independently, so one x of
    # MAX_ROWS serves every arm -- arm `rows` is gated against the first `rows` rows of the same
    # reference. That keeps the device-BO footprint at two operand sets instead of one per arm,
    # and it also means the row sweep varies ONLY the instruction stream.
    max_rows = max([a["rows"] for a in arms if a["kind"] == "lnaff"] or [0])
    for a in arms:
        key = a["kind"]
        if key in sets:
            continue
        if a["kind"] == "gemm":
            A, Bs, refs = gemm_operands(depth_max, np.random.default_rng(seed=42))
        else:
            A, Bs, refs = lnaffcast_operands(max_rows, depth_max,
                                             np.random.default_rng(seed=7))
        At = tensor(A, dtype=bfloat16)
        Bts = [tensor(b, dtype=bfloat16) for b in Bs]
        Cts = [tensor(np.full(M * N, np.nan, dtype=bfloat16), dtype=bfloat16)
               for _ in range(depth_max)]
        for t in (At, *Bts, *Cts):
            t.to("npu")
        sets[key] = {"a_bo": At.buffer_object(),
                     "b_bos": [t.buffer_object() for t in Bts],
                     "c_ts": Cts,
                     "c_bos": [t.buffer_object() for t in Cts],
                     "refs": refs}
        print(f"  operands for {a['kind']} rows={max_rows if a['kind'] == 'lnaff' else 0}: "
              f"{depth_max} slots x (B {Bs[0].nbytes >> 20} MiB + C {M * N * 2 >> 20} MiB)",
              flush=True)

    def check(arm, slots):
        """Worst rel-L2 over `slots`, each against ITS OWN reference."""
        s = sets[arm["kind"]]
        worst, nan = 0.0, 0
        for i in slots:
            s["c_ts"][i].to("cpu")
            got = np.asarray(s["c_ts"][i])
            if arm["kind"] == "gemm":
                got = got.reshape(M, N).astype(np.float32)
                ref = s["refs"][i]
            else:
                got = got[: arm["rows"] * COLS].reshape(arm["rows"], COLS).astype(np.float64)
                ref = s["refs"][i][: arm["rows"]]
            nan += int(np.count_nonzero(~np.isfinite(got)))
            worst = max(worst, rel_l2(got, ref))
            s["c_ts"][i].to("npu")
        return worst, nan

    def submit(arm, i):
        s = sets[arm["kind"]]
        return kh.kernel(3, arm["ibo"], arm["arr"].nbytes,
                         s["a_bo"], s["b_bos"][i], s["c_bos"][i])

    gate = {"gemm": o.rel_l2_gate_gemm, "lnaff": o.rel_l2_gate}
    results = {a["name"]: {"suffix": a["suffix"], "kind": a["kind"], "rows": a["rows"],
                           "blocks": {}} for a in arms}

    for bname, order in (("forward", list(arms)), ("reversed", list(reversed(arms)))):
        print(f"\n=== block '{bname}' ===", flush=True)
        for arm in order:
            # warmup: first touch of this stream's BOs on this context
            submit(arm, 0).wait()

            subs, waits, totals = [], [], []
            for _ in range(o.reps):
                t0 = time.perf_counter_ns()
                h = submit(arm, 0)
                t1 = time.perf_counter_ns()
                r = h.wait()
                t2 = time.perf_counter_ns()
                if r != pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
                    sys.exit(f"{arm['name']}: kernel returned {r}")
                subs.append((t1 - t0) / 1000.0)
                waits.append((t2 - t1) / 1000.0)
                totals.append((t2 - t0) / 1000.0)

            depths = {}
            for d in o.depths:
                per = []
                # A batch of this depth runs untimed first: the first batch at a new depth pays
                # BO first-touch on slots the previous depth never used, which lands entirely in
                # rep 0 and is not part of what the sweep is measuring.
                for h in [submit(arm, i) for i in ((0,) * d if o.negative_control else range(d))]:
                    h.wait()
                for _ in range(o.depth_reps):
                    slots = (0,) * d if o.negative_control else range(d)
                    t0 = time.perf_counter_ns()
                    hs = [submit(arm, i) for i in slots]
                    t1 = time.perf_counter_ns()
                    for h in hs:
                        h.wait()
                    t2 = time.perf_counter_ns()
                    per.append(((t2 - t0) / 1000.0 / d, (t1 - t0) / 1000.0 / d))
                worst, nan = check(arm, range(d))
                depths[d] = {
                    "per_cmd_us": round(statistics.median(p[0] for p in per), 1),
                    "submit_phase_per_cmd_us": round(statistics.median(p[1] for p in per), 2),
                    "rel_l2_worst": worst,
                    "nan_out": nan,
                    "correctness": "PASS" if worst <= gate[arm["kind"]] and not nan else "FAIL",
                }

            d0, dmax = min(o.depths), max(o.depths)
            amort = depths[d0]["per_cmd_us"] - depths[dmax]["per_cmd_us"]
            rec = {
                "submit_us_median": round(statistics.median(subs), 2),
                "wait_us_median": round(statistics.median(waits), 1),
                "total_us_median": round(statistics.median(totals), 1),
                "depths": depths,
                "amortizable_us": round(amort, 1),
                "device_serial_us": depths[dmax]["per_cmd_us"],
                "amortizable_frac": round(amort / depths[d0]["per_cmd_us"], 3),
            }
            results[arm["name"]]["blocks"][bname] = rec
            print(f"\n  {arm['name']:<16s} depth 1: submit {rec['submit_us_median']:.2f} + "
                  f"wait {rec['wait_us_median']:.1f} = {rec['total_us_median']:.1f} us", flush=True)
            for d in o.depths:
                v = depths[d]
                print(f"    depth {d:3d}: {v['per_cmd_us']:8.1f} us/cmd  "
                      f"{depths[d0]['per_cmd_us'] / v['per_cmd_us']:5.2f}x  "
                      f"submit-phase {v['submit_phase_per_cmd_us']:6.2f}  "
                      f"rel-L2 {v['rel_l2_worst']:.4e} {v['correctness']}", flush=True)
            print(f"    -> amortizable {rec['amortizable_us']:.1f} us "
                  f"({rec['amortizable_frac']:.1%})   device-serial "
                  f"{rec['device_serial_us']:.1f} us", flush=True)

    # ---- verdict ----
    print("\n=== verdict: is the amortizable part the same ABSOLUTE cost on every arm? ===")
    print(f"  {'arm':<16s} {'depth1 us':>10s} {'amortizable us':>16s} {'frac':>7s} "
          f"{'device-serial us':>18s}   block spread")
    amorts = []
    ok = True
    for a in arms:
        r = results[a["name"]]["blocks"]
        vals = [r[b]["amortizable_us"] for b in ("forward", "reversed")]
        d1 = [r[b]["depths"][min(o.depths)]["per_cmd_us"] for b in ("forward", "reversed")]
        ser = [r[b]["device_serial_us"] for b in ("forward", "reversed")]
        # Spread is taken against the depth-1 command, not against the amortizable part itself --
        # the latter can sit near zero, where a ratio says nothing about agreement.
        spread = abs(vals[0] - vals[1]) / statistics.mean(d1)
        amorts.append((a["name"], statistics.mean(vals)))
        print(f"  {a['name']:<16s} {statistics.mean(d1):>10.1f} {statistics.mean(vals):>16.1f} "
              f"{statistics.mean(vals) / statistics.mean(d1):>7.1%} {statistics.mean(ser):>18.1f}"
              f"   {spread:>6.1%}")
        for b in ("forward", "reversed"):
            for d, v in r[b]["depths"].items():
                if v["correctness"] != "PASS" and not o.negative_control:
                    ok = False
                    print(f"    FAIL {a['name']} {b} depth {d}: rel-L2 {v['rel_l2_worst']:.4e} "
                          f"nan {v['nan_out']}")
    # ---- the row sweep: is the 151.0 us floor a floor, or a two-point extrapolation? ----
    # The recorded 151.0 us came from fitting a LINE to exactly two points (256 and 512 rows).
    # Two points cannot tell a line from a curve, and this vehicle's sibling task already found an
    # intercept that turned out to be an artifact of exactly that
    # ([[decode-gemv-depth2-intercept-is-not-resolvable]]), so the fit is repeated here over every
    # legal row count -- the generator refuses anything that is not a multiple of the 128-row round,
    # so 128/256/384/512 is the complete set at or below the shipped count.
    rows_pts = sorted((a["rows"], statistics.mean(
        results[a["name"]]["blocks"][b]["depths"][min(o.depths)]["per_cmd_us"]
        for b in ("forward", "reversed"))) for a in arms if a["kind"] == "lnaff")
    fits = {}
    if len(rows_pts) >= 2:
        print("\n=== row sweep at depth 1: does the floor survive more than two points? ===")
        for r, t in rows_pts:
            print(f"  {r:4d} rows  {t:8.1f} us")

        def fit(pts):
            n = len(pts)
            sx = sum(r for r, _ in pts); sy = sum(t for _, t in pts)
            sxx = sum(r * r for r, _ in pts); sxy = sum(r * t for r, t in pts)
            den = n * sxx - sx * sx
            slope = (n * sxy - sx * sy) / den
            return slope, (sy - slope * sx) / n

        m_all, b_all = fit(rows_pts)
        fits["all_points"] = {"rows": [r for r, _ in rows_pts], "slope_us_per_row": round(m_all, 4),
                              "intercept_us": round(b_all, 1)}
        resid = [t - (m_all * r + b_all) for r, t in rows_pts]
        fits["all_points"]["max_abs_resid_us"] = round(max(abs(x) for x in resid), 1)
        print(f"  fit over all {len(rows_pts)}: {m_all:.4f} us/row + {b_all:.1f} us floor, "
              f"max |resid| {max(abs(x) for x in resid):.1f} us")
        pair = [p for p in rows_pts if p[0] in (256, 512)]
        if len(pair) == 2:
            m2, b2 = fit(pair)
            fits["two_point_256_512"] = {"slope_us_per_row": round(m2, 4),
                                         "intercept_us": round(b2, 1)}
            print(f"  the recorded two-point 256/512 fit on THIS session's numbers: "
                  f"{m2:.4f} us/row + {b2:.1f} us floor")
            print(f"  -> the two-point floor {'OVERSTATES' if b2 > b_all else 'understates'} the "
                  f"multi-point one by {abs(b2 - b_all):.1f} us "
                  f"({abs(b2 - b_all) / max(1e-9, b_all):.0%})")

    lo, hi = min(v for _n, v in amorts), max(v for _n, v in amorts)
    print(f"\n  amortizable across arms: {lo:.1f} .. {hi:.1f} us, "
          f"spread {(hi - lo) / statistics.mean([v for _n, v in amorts]):.1%}")
    print("  A GENERIC host round-trip predicts these agree in ABSOLUTE us despite the arms\n"
          "  differing in command size; a per-op floor predicts they do not.")

    with open(o.out, "w") as f:
        json.dump({"host": host, "shape": [M, K, N], "depths": o.depths,
                   "row_fits": fits,
                   "reps": o.reps, "depth_reps": o.depth_reps,
                   "negative_control": o.negative_control,
                   "results": results}, f, indent=2)
    print(f"\nwrote {o.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--build-dir", default="mlir-aie/programming_examples/basic/"
                                          "matrix_multiplication/whole_array/build")
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--kernel", default="MLIR_AIE")
    p.add_argument("--base", default=BASE,
                   help="design prefix the lnaff arms name their streams off")
    p.add_argument("--host", default=None,
                   help="the ONE xclbin every arm's stream runs on (default <base>rtp18r256g4ctgc)")
    p.add_argument("--gemm-suffix", default=None,
                   help="the GEMM stream's insts suffix (default <base>)")
    p.add_argument("--arms", nargs="+",
                   default=["lnaff128:lnaff:128", "lnaff256:lnaff:256",
                            "lnaff384:lnaff:384", "lnaff512:lnaff:512", "gemm:gemm:0"],
                   help="name:kind:rows triples; kind is lnaff or gemm")
    p.add_argument("--depths", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    p.add_argument("--reps", type=int, default=9)
    p.add_argument("--depth-reps", type=int, default=9)
    p.add_argument("--rel-l2-gate", type=float, default=1e-4)
    p.add_argument("--rel-l2-gate-gemm", type=float, default=2e-2,
                   help="the GEMM writes bf16 C against an f32 reference, so its floor is the "
                        "output dtype's epsilon and not the lnaffcast arms' 1e-4")
    p.add_argument("--negative-control", action="store_true",
                   help="run every in-flight command against slot 0 -- they then compute the same "
                        "bytes and parity passes whether or not the commands stayed separate, "
                        "which is what makes the per-slot gate above meaningful")
    p.add_argument("--out", default="artifacts/lnaffcast_dispatch_floor.json")
    o = p.parse_args()
    o.host = o.host or f"{o.base}rtp18r256g4ctgc"
    o.gemm_suffix = o.base if o.gemm_suffix is None else o.gemm_suffix
    sys.exit(main(o))
