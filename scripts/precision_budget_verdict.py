#!/usr/bin/env python3
"""Pre-registered decision rule for mixed-precision-budget-sweep, written BEFORE the remaining
device data exists (task's own instruction: writing the rule down before the data exists is the
point). Consumes precision_budget_sweep.py's CPU JSON (numeric tolerance, no device) and, once an
arm gets a device probe, an optional device-side JSON (real hardware rounding, ms/dispatch, L1
bytes, determinism vs CPU/simulator). Never invents a verdict from the CPU numbers alone -- per
error-metrics-are-notes-not-gates and seventeen-clip-wer-gate-is-chaotic, rel-L2/burst statistics
here decide only whether an arm is WORTH BUILDING on device, never whether to SHIP it.

Two independent axes are threaded through every row, per the task's regime lens:
  lever   -- ACTIVATION-BYTES (streamed activation format, judged on whether the op sits at the
             movement floor so a bytes win becomes a wall-clock win), WEIGHT-RSS (encoder weight
             format is a footprint lever here, not a speed lever -- correction (a)), or
             COMPUTE-FORMAT (unlocks a wider systolic MAC path, judged on compute peak %, not bytes).
  numeric verdict -- SAFE / JUDGMENT-CALL / FAILS, from the CPU harness's four parity statistics
             against the pre-registered thresholds encoder_parity.py already gates at.
A numeric SAFE is necessary but not sufficient to build; a numeric FAILS is sufficient to reject
without spending device time. Ship, in every case, still needs the device gate: 1:1 determinism
against the CPU/simulator reference at temperature 0.

Usage:
  scripts/precision_budget_verdict.py --cpu-json /path/to/sweep.json
  scripts/precision_budget_verdict.py --cpu-json sweep.json --device-json device_probes.json
  scripts/precision_budget_verdict.py --cpu-json sweep.json --json out.json
"""
import argparse
import json
import sys

# Pre-registered thresholds -- copied from encoder_parity.py's own gate and
# precision-budget-sweep-cpu-harness-2026-08-14.md's anchor, not re-derived here.
BURST_FLOOR = 0.15   # encoder_parity.py --burst-floor: a frame counts as "in a burst" above this
CLIFF = 1.0           # burst_sensitivity.py's measured transcript-changing threshold
SHIPPED_BASELINE_MEAN = 0.0891  # whole-block-fusion-t6-t7-2026-07-27: shipped path vs f32 truth
L1_BUDGET_BYTES = 65536          # aie2p core data memory

# Which independent lever each op is being judged on (task's regime-lens instruction: a format is
# TWO levers, say which one). fc1fc2's activation narrowing and bfp16gemm's operand narrowing are
# both ACTIVATION/COMPUTE bytes-in-flight, never a weight-footprint question.
LEVER = {
    "resadd": "activation-bytes",
    "affcast": "activation-bytes",
    "fc1fc2": "activation-bytes",
    "bfp16gemm": "compute-format",
    "wint8": "weight-rss",
}

# Ops measured at the movement floor (section 3): a bytes win here is predicted to convert to
# wall-clock because there is no other lever short of deletion. Off this list, a bytes win needs
# an explicit ms measurement before it is credited as a speed win (dispatch/transition dominate).
AT_MOVEMENT_FLOOR = {"resadd", "affcast"}

# L1 risk carried from section 1(b): the modal matmul's in-place f32 accumulator forces a SEPARATE
# narrow output buffer, which COSTS L1 while a narrower format saves DDR. Elementwise ops (resadd,
# affcast) have no such accumulator; their narrow output IS their only buffer.
L1_ACCUMULATOR_RISK = {"fc1fc2", "bfp16gemm"}


def numeric_verdict(agg):
    """SAFE / JUDGMENT-CALL / FAILS from the CPU harness's own aggregate dict (mean, worst_frame,
    worst_burst -- new_burst is not present here, it needs a device baseline, see module docstring).
    Gates on all three available statistics, not the mean alone -- affcast-int8 is the recorded
    case a mean-only read gets wrong (mean 0.092 looks free, worst-frame 0.907 does not)."""
    worst = max(agg["worst_frame"], agg["worst_burst"])
    if worst >= CLIFF:
        return "FAILS"
    if worst < BURST_FLOOR and agg["mean"] < SHIPPED_BASELINE_MEAN:
        return "SAFE"
    return "JUDGMENT-CALL"


def build_readiness(op, fmt, verdict):
    """Should this arm be BUILT on device at all -- the device-free half's actual question.
    Never a ship decision; ship needs the device determinism gate (see ship_verdict)."""
    if verdict == "FAILS":
        return "REJECT -- do not build; crosses the measured transcript cliff"
    if verdict == "JUDGMENT-CALL":
        return ("HOLD -- needs a finer granularity (per-group scale / rotation) or the on-device "
                "cliff check before committing build effort")
    lever = LEVER.get(op, "unknown")
    if lever == "activation-bytes" and op in AT_MOVEMENT_FLOOR:
        return "BUILD -- highest priority: op is at the movement floor, bytes win predicted to convert to ms"
    if lever == "activation-bytes":
        return "BUILD -- numerically safe, but ms win is NOT assumed (op not at movement floor); measure before crediting speed"
    if lever == "weight-rss":
        return "BUILD -- scored on RSS delta, not ms, per correction (a) (encoder weights are not the wire)"
    if lever == "compute-format":
        return "BUILD -- scored on compute-peak %, not bytes; conditioning failures go to precision-conditioning-for-bfp16"
    return "UNSCOPED op, add it to LEVER/AT_MOVEMENT_FLOOR before verdicting"


def ship_verdict(op, fmt, device):
    """The real gate, once a device probe exists. `device` is one arm's dict with at minimum
    `determinism_1to1` (bool: device path reproduces the CPU/simulator reference exactly, or the
    documented rounding-consistent equivalent -- see bf16-resadd-is-bit-exact-and-halves-l1-but-
    is-not-bytes-bound.md for what "documented rounding-consistent" means for a narrowed format)
    and `l1_bytes`. `ms_f32`/`ms_fmt` are recorded as evidence, never gating -- ms adoption stays
    owner-gated on re-pricing --max-new-burst per the task, independent of this rule."""
    if not device.get("determinism_1to1", False):
        return "DEVICE-REJECTED -- fails 1:1 determinism vs CPU/simulator reference at temp 0"
    l1 = device.get("l1_bytes")
    if l1 is not None and l1 > L1_BUDGET_BYTES:
        return f"DEVICE-REJECTED -- L1 overflow ({l1} > {L1_BUDGET_BYTES} B)"
    return "DEVICE-CONFIRMED -- determinism holds, L1 fits; adoption stays OWNER-GATED on --max-new-burst re-pricing"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu-json", required=True, help="precision_budget_sweep.py --json output")
    ap.add_argument("--device-json", default=None,
                    help="optional: {\"op\": ..., \"fmt\": ..., \"determinism_1to1\": bool, "
                         "\"l1_bytes\": int, \"ms_f32\": float, \"ms_fmt\": float} per arm, keyed "
                         "\"<op>:<fmt>\"")
    ap.add_argument("--json", default=None, help="write the verdict table as JSON")
    args = ap.parse_args()

    with open(args.cpu_json) as f:
        cpu = json.load(f)

    device_probes = {}
    if args.device_json:
        with open(args.device_json) as f:
            device_probes = json.load(f)

    rows = []
    header = f"{'op':10} {'fmt':6} {'lever':16} {'numeric':13} {'build readiness / ship verdict'}"
    print(header)
    print("-" * len(header))
    for r in cpu["results"]:
        op, fmt = r["op"], r["fmt"]
        verdict = numeric_verdict(r["aggregate"])
        readiness = build_readiness(op, fmt, verdict)
        key = f"{op}:{fmt}"
        row = {"op": op, "fmt": fmt, "lever": LEVER.get(op, "unknown"),
               "numeric_verdict": verdict, "build_readiness": readiness,
               "l1_accumulator_risk": op in L1_ACCUMULATOR_RISK,
               "at_movement_floor": op in AT_MOVEMENT_FLOOR}
        if key in device_probes:
            row["ship_verdict"] = ship_verdict(op, fmt, device_probes[key])
            printed = row["ship_verdict"]
        else:
            printed = readiness
        rows.append(row)
        print(f"{op:10} {fmt:6} {row['lever']:16} {verdict:13} {printed}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"rows": rows}, f, indent=2)
        print(f"\nwrote {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
