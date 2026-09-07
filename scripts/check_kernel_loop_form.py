#!/usr/bin/env python3
"""Assert a kernel's MAC sits in a zero-overhead loop, not a branchy software loop.

Kernel contract K013. A weight-consuming inner loop that does not become a hardware loop pays
its branch and its stalls on every chunk, and nothing in the build reports it: mv_quant.cc was
written as mv.cc's sibling, grew an inner scalar unpack, and compiled to a 181-bundle software
loop where mv.cc's is a 6-bundle ZOL -- 11.7x on the decode step, found only by timing it.

  check_kernel_loop_form.py <object.o|elf> [--reference bf16.o] [--max-ratio 3.0]
"""
import argparse
import os
import re
import shutil
import subprocess
import sys

LABEL = re.compile(r"^([0-9a-f]{8}) <([^>]+)>:")
INSTR = re.compile(r"^\s+([0-9a-f]+):")
BACKREF = re.compile(r"\bj(?:n?z)?\b[^#]*#0x([0-9a-f]+)")
MAC = re.compile(r"\bv(?:mac|mul)\.")


def objdump():
    for cand in (os.environ.get("LLVM_OBJDUMP"),
                 os.path.join(os.environ.get("PEANO_INSTALL_DIR", ""), "bin", "llvm-objdump"),
                 shutil.which("llvm-objdump")):
        if cand and os.path.exists(cand):
            return cand
    sys.exit("no llvm-objdump: set PEANO_INSTALL_DIR or LLVM_OBJDUMP")


def parse(path):
    text = subprocess.run([objdump(), "-d", path], capture_output=True, text=True).stdout
    labels, instrs = [], []
    for line in text.splitlines():
        m = LABEL.match(line)
        if m:
            labels.append((int(m.group(1), 16), m.group(2)))
            continue
        m = INSTR.match(line)
        if m:
            instrs.append((int(m.group(1), 16), line))
    return labels, instrs


def spans(labels, instrs):
    """(start, end, kind) for every loop: ZOL bodies from LEnd labels, sw loops from back-branches."""
    out = []
    for i, (addr, name) in enumerate(labels):
        if "LEnd" in name and i > 0:
            out.append((labels[i - 1][0], addr, "zol"))
    for addr, line in instrs:
        m = BACKREF.search(line)
        if m:
            target = int(m.group(1), 16)
            if target < addr:
                out.append((target, addr, "sw"))
    return out


def bundles(instrs, lo, hi):
    return sum(1 for a, _ in instrs if lo <= a <= hi)


def check(path):
    labels, instrs = parse(path)
    if not instrs:
        sys.exit(f"{path}: no instructions disassembled")
    loops = spans(labels, instrs)
    macs = [a for a, line in instrs if MAC.search(line)]
    if not macs:
        sys.exit(f"{path}: no vector MAC found -- wrong object?")

    # Pipelining hoists prologue/epilogue copies of the MAC outside the loop body, so the
    # invariant is that the STEADY-STATE MAC is in a hardware loop -- not that every MAC is.
    zols = [s for s in loops if s[2] == "zol"]
    steady = [(lo, hi) for lo, hi, _ in zols if any(lo <= a <= hi for a in macs)]
    zol_n = max((bundles(instrs, lo, hi) for lo, hi in steady), default=0)

    sw = None
    if not steady:
        enclosing = [s for s in loops if s[2] == "sw" and any(s[0] <= a <= s[1] for a in macs)]
        if enclosing:
            lo, hi, _ = min(enclosing, key=lambda s: s[1] - s[0])
            sw = (lo, hi, bundles(instrs, lo, hi))

    return {"path": path, "zol_loops": len(zols), "mac_zols": len(steady),
            "zol_bundles": zol_n, "mac_in_sw_loop": sw, "instrs": len(instrs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("object")
    ap.add_argument("--reference", help="known-good kernel to bound bundle count against")
    ap.add_argument("--max-ratio", type=float, default=3.0)
    a = ap.parse_args()

    r = check(a.object)
    print(f"{os.path.basename(r['path'])}: {r['instrs']} instrs, {r['zol_loops']} ZOL, "
          f"{r['mac_zols']} carrying a MAC (body {r['zol_bundles']} bundles)")

    failed = False
    if not r["mac_zols"]:
        where = ""
        if r["mac_in_sw_loop"]:
            lo, hi, n = r["mac_in_sw_loop"]
            where = f" -- it is in a software loop 0x{lo:x}-0x{hi:x} ({n} bundles)"
        print(f"  K013 FAIL: no MAC in a zero-overhead loop{where}")
        failed = True

    if a.reference:
        ref = check(a.reference)
        print(f"  reference {os.path.basename(ref['path'])}: largest ZOL body "
              f"{ref['zol_bundles']} bundles")
        if ref["zol_bundles"] and r["zol_bundles"]:
            ratio = r["zol_bundles"] / ref["zol_bundles"]
            if ratio > a.max_ratio:
                print(f"  K013 FAIL: inner loop is {ratio:.1f}x the reference "
                      f"(max {a.max_ratio})")
                failed = True

    print("  K013 PASS" if not failed else "")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
