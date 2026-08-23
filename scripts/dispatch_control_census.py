#!/usr/bin/env python3
# Static census of what a command's CONTROL STREAM contains, for any built design, so a claimed
# per-dispatch floor can be tested against the ops that would have to produce it.
#
# WHY THIS EXISTS. Three per-command floors are on record and have been compared to each other as if
# they were one quantity -- the whole_array GEMM's ~140 us (51.8 us of it device-serial), the
# lnaffcast 1x route's 155.1 us (~103 us device-serial), and decode-GEMV's ~170 us. Every one of
# them is a FITTED INTERCEPT, and what sits at the intercept differs per design: the GEMM's was fit
# across SHAPES, so its whole control stream is still being issued at zero work, while lnaffcast's
# was fit across ROWS, and rows drive its dma tasks, so at its intercept there is no control stream
# left at all. Two intercepts that hold different things constant are not comparable, and no amount
# of re-measuring either one alone surfaces that -- it only shows up next to a census.
#
# WHAT IT IS FOR, concretely: a candidate mechanism for a floor ("the control processor executing
# the prologue") predicts a per-unit cost, and a census over designs whose floors are already
# measured either finds that cost stable or kills it. It killed the prologue model -- the GEMM's
# 64 control writes against a ~62 us fitted floor and lnaffcast's 96 against ~103 us agree at
# ~1 us/write, and decode-GEMV then carries the LARGEST floor of the three with ZERO of them.
#
# It also shows where a sweep cannot attribute its own slope: on the lnaffcast 1x route, +128 rows
# is +20 dma tasks, +20 BDs, +8 awaits, +2752 insts bytes and +800 KB of traffic, exactly each time.
#
# Device-free -- reads only emitted MLIR and insts. Run:
#   python3 scripts/dispatch_control_census.py --design <build-dir>:<suffix> [...]
#   python3 scripts/dispatch_control_census.py --design <dir>:<sfx>=<FLOOR us> --per-unit
# --per-unit wants the design's FLOOR, not a whole command: dividing a time that still carries work
# by a control-stream count tests nothing.
import argparse
import json
import os
import re
import sys

# Ops before the runtime sequence configure the array and are paid once at xclbin load, not per
# command.
OP_RE = re.compile(r"^\s+([a-z_0-9]+\.[a-z_0-9.]+)")
REPEAT_RE = re.compile(r"repeat_count\s*=\s*(\d+)")
COUNTED = ["aiex.dma_start_task", "aie.dma_bd", "aiex.dma_await_task",
           "aiex.set_lock", "aiex.npu.rtp_write", "aiex.npu.write32",
           "aiex.npu.sync", "aiex.npu.dma_memcpy_nd"]


def census(build_dir, suffix):
    mlir = os.path.join(build_dir, f"aie_{suffix}.mlir")
    if not os.path.exists(mlir):
        return None
    counts, bd_issues, in_seq = {}, 0, False
    with open(mlir) as f:
        for line in f:
            if "aie.runtime_sequence(" in line:
                in_seq = True
            if not in_seq:
                continue
            m = OP_RE.match(line)
            if m:
                counts[m.group(1)] = counts.get(m.group(1), 0) + 1
            if "aie.dma_bd(" in line:
                bd_issues += 1
    with open(mlir) as f:
        text = f.read()
    # repeat_count is how many times the shim re-runs a BD, so transfers issued is not BDs written.
    seq = text[text.index("aie.runtime_sequence("):] if "aie.runtime_sequence(" in text else ""
    bd_issues = sum(int(r) + 1 for r in REPEAT_RE.findall(seq)) or bd_issues

    insts = os.path.join(build_dir, f"insts_{suffix}.txt")
    row = {"suffix": suffix, "build_dir": build_dir,
           "insts_bytes": os.path.getsize(insts) if os.path.exists(insts) else None,
           "bd_issues": bd_issues}
    for op in COUNTED:
        row[op.split(".")[-1]] = counts.get(op, 0)
    row["ctrl_writes"] = row["set_lock"] + row["rtp_write"] + row["write32"]
    return row


def main(o):
    rows, measured = [], {}
    for spec in o.design:
        head, _, us = spec.partition("=")
        build_dir, _, suffix = head.rpartition(":")
        if not build_dir:
            sys.exit(f"--design wants <build-dir>:<suffix>[=<measured_us>], got {spec!r}")
        r = census(build_dir, suffix)
        if r is None:
            sys.exit(f"no emitted MLIR for {suffix} in {build_dir}")
        if us:
            measured[suffix] = float(us)
        rows.append(r)

    cols = ["insts_bytes", "dma_start_task", "bd_issues", "dma_await_task",
            "set_lock", "rtp_write", "ctrl_writes"]
    print(f"{'design':<52}" + "".join(f"{c[:11]:>12}" for c in cols))
    for r in rows:
        print(f"{r['suffix'][-52:]:<52}" + "".join(f"{str(r[c]):>12}" for c in cols))

    if o.per_unit and measured:
        # A mechanism that is real gives the same per-unit cost on every design. One that is a
        # coincidence of two designs does not survive a third.
        print(f"\nimplied per-unit cost against the supplied measurement ({len(measured)} designs):")
        print(f"{'design':<52}{'us':>9}" + "".join(f"{c[:11]:>12}" for c in cols[1:]))
        for r in rows:
            if r["suffix"] not in measured:
                continue
            us = measured[r["suffix"]]
            cells = "".join(f"{(us / r[c]):>12.3f}" if r[c] else f"{'--':>12}" for c in cols[1:])
            print(f"{r['suffix'][-52:]:<52}{us:>9.1f}{cells}")
        print("\nA column whose values agree across designs is a candidate mechanism; one that\n"
              "spreads, or divides by zero on a design that still has the cost, is refuted.")

    if o.out:
        with open(o.out, "w") as f:
            json.dump({"designs": rows, "measured_us": measured}, f, indent=1)
        print(f"\nwrote {o.out}")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--design", nargs="+", required=True,
                   help="<build-dir>:<suffix>[=<measured floor us>], repeatable")
    p.add_argument("--per-unit", action="store_true",
                   help="divide each supplied measurement by each count, to falsify a mechanism")
    p.add_argument("--out")
    sys.exit(main(p.parse_args()))
