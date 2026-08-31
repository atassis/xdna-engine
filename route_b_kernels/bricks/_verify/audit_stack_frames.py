#!/usr/bin/env python3
"""Which bricks have a stack frame that overruns the default reservation?

This is the check that does not exist upstream, done locally. AIETargetLdScript.cpp reserves
core.getStackSize() bytes -- default 0x400 (AIEOps.td:374) -- immediately below the objectFIFO
buffers with zero clearance, and nothing compares that against what the kernel actually needs. A
kernel that needs more silently overwrites the buffers: prefill-attn returned NaN on all 32 heads
for weeks because of exactly this, and went green at 0x800.

For AIE the requirement is computable rather than a policy choice: every call is a direct `jl` to a
fixed address, there is no recursion, no indirect call and no function pointer, so the call graph is
static and the deepest path is just a sum. LLVM emits per-function sizes under
`-fstack-size-section`.

Scope and honesty about it, all three points measured rather than assumed:
  * this sums EVERY non-zero frame in the object, which is an UPPER BOUND, not the true deepest
    path -- it assumes the functions can nest, and sibling calls are counted twice. layernorm audits
    at 0x480 across FOUR functions (max 0x180) and shows zero device movement between 0x400 and
    0xD00, so it evidently never nests that deep: a conservative flag, not a real one.
  * the -D flags must match the brick's verify script. rope_lut.cc defaults to ROPE_M=64 while the
    real build is ROPE_M=16; auditing the default gave 0x480 AT RISK, and at the real shape it is
    0x400 -- ok, and matching a device result that does not move at all.
  * it counts the BRICK object only. `main` has its own frame on top, measured at 0x40: prefill-attn
    audits at 0x700 and needed 0x740 on device. So this UNDER-reports the true requirement by
    roughly that much, in the unsafe direction.
  * OPEN: rope-lut sits at exactly 0x400 against a 0x400 reservation, so adding main's 0x40 predicts
    a small overflow -- and the device shows none. Either that function is not on the deepest path
    with main, or the accounting is not exact. Do not read the margin here as understood.

So: a brick over 0x400 here is AT RISK and worth re-gating; a brick under it is not proof of
innocence. Confirmed useful once -- it reproduced prefill-attn's 0x700 exactly.

Device-free. Run:  python3 audit_stack_frames.py
"""
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent.resolve()
WS = HERE.parents[3]

# The brick harness lives in TWO checkouts that have diverged (see the KB: wt-s2-codec-bricks is
# stale on rope-lut and carries its pre-fix kernel, while xdna-engine has the fix and 9 fewer
# bricks). Auditing the wrong one measures the wrong kernel's frame, so make the tree explicit:
#   python3 audit_stack_frames.py [<bricks-dir>]
BRICKS = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE.parent

DEFAULT_STACK = 0x400

PEANO = Path(os.environ.get(
    "PEANO_INSTALL_DIR",
    WS / "wt-s2-codec-bricks/.venv-iron/lib/python3.14/site-packages/llvm-aie"))
CLANG = PEANO / "bin/clang++"
READELF = PEANO / "bin/llvm-readelf"

# The adf-free aie_api fork: upstream aie_api pulls <adf.h> from Vitis, which a standalone Peano
# compile does not have. Falls back to the toolchain instance's copy if the fork is absent.
INC_CANDIDATES = [
    WS / "aie_api-fork/include",
    WS / ".cache/instances/7d8a49b5d7a0/src/third_party/aie_api/include",
]

# Per-brick -D flags. These MUST match what the brick's verify script passes, not the .cc's own
# #ifndef defaults -- the defaults are frequently a different shape and the frame scales with it.
# Measured: rope_lut.cc defaults to ROPE_M=64 while verify_rope_lut.py builds at ROPE_M=16, and
# auditing the default reported a 0x480 frame for a kernel 4x wider than the one that runs. That
# was this script's first false positive, and it is the failure mode to watch for when adding a
# brick here: an audit against the wrong configuration is worse than no audit.
EXTRA_FLAGS = {
    "prefill-attn": ["-DPREFILL_HD=128", "-DPREFILL_M=11"],
    "rope-lut": ["-DROPE_D=128", "-DROPE_ROT=128", "-DROPE_M=16", "-DROPE_SCALE_INV=1.0f"],
}


def _uleb(buf, i):
    val, shift = 0, 0
    while True:
        b = buf[i]
        i += 1
        val |= (b & 0x7F) << shift
        if not b & 0x80:
            return val, i
        shift += 7


# aie2p is 32-bit, so a .stack_sizes entry is a 4-BYTE address followed by the ULEB size. Reading
# it as the 8-byte form silently yields an empty list on every object rather than an error.
ADDR_BYTES = 4


def frames(obj):
    """Every non-zero per-function frame in `obj`, from .stack_sizes (4-byte addr + ULEB size)."""
    out = subprocess.run([str(READELF), "--hex-dump=.stack_sizes", str(obj)],
                         capture_output=True, text=True).stdout
    sizes = []
    for block in out.split("Hex dump of section")[1:]:
        hexs = "".join(re.findall(r"^0x[0-9a-f]{8} ((?:[0-9a-f]{2,8} )+)", block, re.M)).replace(" ", "")
        raw = bytes.fromhex(hexs)
        i = 0
        while i + ADDR_BYTES < len(raw):
            i += ADDR_BYTES
            v, i = _uleb(raw, i)
            if v:
                sizes.append(v)
    return sizes


def main():
    inc = next((c for c in INC_CANDIDATES if (c / "aie_api/aie.hpp").exists()), None)
    if inc is None:
        sys.exit(f"no aie_api include found; tried {[str(c) for c in INC_CANDIDATES]}")
    if not CLANG.exists():
        sys.exit(f"no Peano clang++ at {CLANG}")
    print(f"aie_api: {inc}\ndefault reservation: {DEFAULT_STACK:#x}\n")

    tmp = Path(os.environ.get("TMPDIR", "/tmp")) / "brick_frame_audit"
    tmp.mkdir(exist_ok=True)

    rows, skipped = [], []
    for cc in sorted(BRICKS.glob("*/*.cc")):
        brick = cc.parent.name
        if brick == "_verify":  # repro shims, not bricks
            continue
        obj = tmp / f"{brick}.o"
        cmd = [str(CLANG), "--target=aie2p-none-unknown-elf", "-std=c++2b", "-O2",
               f"-I{inc}", "-fstack-size-section", "-c", str(cc), "-o", str(obj)]
        cmd[6:6] = EXTRA_FLAGS.get(brick, [])
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            first = next((l for l in r.stderr.splitlines() if "error:" in l), r.stderr[:80])
            skipped.append((brick, first.strip()[:95]))
            continue
        fs = frames(obj)
        rows.append((brick, sum(fs), max(fs) if fs else 0, len(fs)))

    rows.sort(key=lambda t: -t[1])
    print(f"{'brick':32s} {'sum(frames)':>12} {'max':>8} {'fns':>5}  verdict")
    at_risk = []
    for brick, tot, mx, n in rows:
        bad = tot > DEFAULT_STACK
        if bad:
            at_risk.append(brick)
        print(f"{brick:32s} {tot:>#12x} {mx:>#8x} {n:>5}  "
              f"{'AT RISK' if bad else 'ok'}")

    if skipped:
        print(f"\n{len(skipped)} did not compile standalone (NOT audited -- absence of a verdict, "
              f"not a pass):")
        for brick, err in skipped:
            print(f"  {brick:30s} {err}")

    print()
    if at_risk:
        print(f"AT RISK ({len(at_risk)}): {', '.join(at_risk)}")
        print(f"  Upper-bound sum exceeds the {DEFAULT_STACK:#x} default, so these can overrun the "
              f"objectFIFO\n  buffers. Re-gate each with an adequate stack_size and compare.")
    else:
        print(f"No brick's frame sum exceeds {DEFAULT_STACK:#x}.")


if __name__ == "__main__":
    main()
