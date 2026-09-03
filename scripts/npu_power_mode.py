#!/usr/bin/env python3
"""NPU power-mode gate for timed runs.

WHY THIS EXISTS: a sequential parameter sweep on an unpinned NPU aliases the swept
variable with the device's DVFS ramp. A 1/2/4/8-worker DMA sweep once produced a
textbook-monotone 3.05x curve that collapsed to 1.17x (range 0.84-1.51x) once the
identical configs were interleaved -- the curve was the device clocking up, not the
fan-out it appeared to measure. Pin the power mode before believing any timed sweep.

WHAT IT CAN AND CANNOT DO: reading the mode is unprivileged (the amdxdna GET_INFO
ioctl carries no permission flag). SETTING it is not: the driver registers
DRM_IOCTL_DEF_DRV(AMDXDNA_SET_STATE, ..., DRM_ROOT_ONLY), and DRM_ROOT_ONLY is a
CAP_SYS_ADMIN check on the calling process, so no udev rule or file mode grants it.
This module therefore GATES rather than sets: it refuses to let a harness report
numbers taken at the Default (unpinned) mode, and prints the command the owner must
run as root. Fail loud beats silently measuring warm-up.

USE FROM A HARNESS:
    from npu_power_mode import require_pinned
    mode = require_pinned()          # exits nonzero with the fix if unpinned

CLI:
    scripts/npu_power_mode.py                  # print the current mode
    scripts/npu_power_mode.py --require-pinned # exit 1 (with the fix) if Default
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

# xrt-smi --pmode spellings -> the driver's enum amdxdna_power_mode_type.
# powersaver/balanced/performance map to LOW/MEDIUM/HIGH.
PINNED_MODES = ("powersaver", "balanced", "performance", "turbo")
UNPINNED = "default"
# The mode to pin for bandwidth/latency work: turbo. Both turbo and performance pin
# dpm_level to max_dpm_level; turbo ADDITIONALLY disables clock gating, so it is the more
# deterministic pin -- no gating transitions during a timed sweep.
#
# TURBO HAS A PRECONDITION: zero active hardware contexts. Mainline
# drivers/accel/amdxdna/aie2_pm.c, aie2_pm_set_mode():
#     case POWER_MODE_TURBO:
#         if (ndev->hwctx_num) {
#             XDNA_ERR(xdna, "Can not set turbo when there is active hwctx");
#             return -EINVAL;
#         }
# so `xrt-smi configure --pmode turbo` reports a bare "Invalid argument" (err=-22) whenever
# anything holds the device. MEASURED 2026-07-26: fails with xdna-engine.service up, while
# `--pmode performance` succeeds -- POWER_MODE_HIGH carries no such guard. STOP THE NPU
# SERVICES FIRST (which timed runs do anyway) and turbo works.
#
# DRIVER-PROVENANCE TRAP: AMD's out-of-tree xdna-driver checkout has NO hwctx guard on turbo,
# so reading that source predicts success. The driver actually running here is the IN-TREE
# amdxdna (`modinfo amdxdna` -> intree: Y) and mainline is the STRICTER of the two. Check
# `modinfo` before treating a local driver checkout as authoritative for this box.
RECOMMENDED = "turbo"
# No-precondition fallback: pins max DPM but leaves clock gating enabled, so gating
# transitions remain a smaller residual variance source. Use if turbo is refused with the
# device already quiesced.
FALLBACK = "performance"

SET_CMD = "sudo xrt-smi configure --pmode {mode} --force"
# Prepended to the fix hint: turbo's precondition is the overwhelmingly common reason it fails.
QUIESCE_CMD = "systemctl --user stop xdna-engine.service npu-vox.service"


class PowerModeUnavailable(RuntimeError):
    """xrt-smi could not be queried (missing, no device, or unparseable output)."""


def read_power_mode(timeout=30):
    """Return the device's current power mode, lowercased (e.g. 'default', 'turbo').

    Unprivileged. Raises PowerModeUnavailable if the device cannot be queried."""
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "platform.json")
        try:
            subprocess.run(
                ["xrt-smi", "examine", "-r", "platform", "-f", "json", "-o", out, "--force"],
                capture_output=True, timeout=timeout, check=False,
            )
        except FileNotFoundError:
            raise PowerModeUnavailable("xrt-smi not found on PATH")
        except subprocess.TimeoutExpired:
            raise PowerModeUnavailable(f"xrt-smi timed out after {timeout}s")
        if not os.path.exists(out):
            raise PowerModeUnavailable("xrt-smi wrote no platform report (device busy or absent?)")
        try:
            doc = json.load(open(out))
            return doc["devices"][0]["platforms"][0]["status"]["power_mode"].strip().lower()
        except (ValueError, KeyError, IndexError, AttributeError) as e:
            raise PowerModeUnavailable(f"could not parse power_mode out of the report: {e}")


def is_pinned(mode):
    return mode in PINNED_MODES


def require_pinned(recommended=RECOMMENDED, allow_override=True):
    """Assert the NPU power mode is pinned; return it. Exits nonzero if it is not.

    Set NPU_ALLOW_UNPINNED=1 to proceed anyway (prints a loud banner -- for exploratory
    runs whose numbers must NOT be recorded as canonical)."""
    override = allow_override and os.environ.get("NPU_ALLOW_UNPINNED") == "1"
    try:
        mode = read_power_mode()
    except PowerModeUnavailable as e:
        if override:
            print(f"!! power mode UNKNOWN ({e}) and NPU_ALLOW_UNPINNED=1 -- numbers are NOT canonical",
                  file=sys.stderr)
            return "unknown"
        sys.exit(f"FATAL: cannot read the NPU power mode ({e}).\n"
                 f"       Timed runs need it pinned. Fix the query, or set NPU_ALLOW_UNPINNED=1\n"
                 f"       to proceed with numbers that must not be recorded as canonical.")
    if is_pinned(mode):
        print(f"[power] mode={mode} (pinned) -- timed sweep may proceed")
        return mode
    msg = (f"NPU power mode is '{mode}' (UNPINNED). A timed sweep taken now measures the\n"
           f"       DVFS ramp as well as the swept variable -- this is the exact failure that\n"
           f"       turned a 3.05x DMA fan-out result into 1.17x once interleaved.\n\n"
           f"       Pin it first (needs root -- the driver gates SET_STATE on DRM_ROOT_ONLY).\n"
           f"       Quiesce FIRST: turbo is refused while any hardware context is live, and\n"
           f"       xrt-smi reports that only as 'Invalid argument' (err=-22):\n"
           f"           {QUIESCE_CMD}\n"
           f"           {SET_CMD.format(mode=recommended)}\n"
           f"       If it is still refused on a quiesced device, fall back to:\n"
           f"           {SET_CMD.format(mode=FALLBACK)}\n"
           f"       and restore afterwards:\n"
           f"           {SET_CMD.format(mode='default')}")
    if override:
        print(f"!! {msg}\n!! NPU_ALLOW_UNPINNED=1 -- proceeding; numbers are NOT canonical",
              file=sys.stderr)
        return mode
    sys.exit(f"FATAL: {msg}")


def main():
    ap = argparse.ArgumentParser(description="Read/gate the NPU power mode for timed runs.")
    ap.add_argument("--require-pinned", action="store_true",
                    help="exit 1 with the fix command if the mode is unpinned")
    ap.add_argument("--recommended", default=RECOMMENDED, choices=PINNED_MODES,
                    help=f"mode to recommend pinning to (default {RECOMMENDED})")
    a = ap.parse_args()
    if a.require_pinned:
        require_pinned(recommended=a.recommended, allow_override=False)
        return 0
    try:
        mode = read_power_mode()
    except PowerModeUnavailable as e:
        print(f"power mode: UNAVAILABLE ({e})")
        return 2
    state = "pinned" if is_pinned(mode) else "UNPINNED -- timed sweeps are not trustworthy"
    print(f"power mode: {mode} ({state})")
    if not is_pinned(mode):
        print(f"  pin:     {SET_CMD.format(mode=a.recommended)}")
        print(f"  restore: {SET_CMD.format(mode='default')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
