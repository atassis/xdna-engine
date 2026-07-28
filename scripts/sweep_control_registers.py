#!/usr/bin/env python3
"""Sweep AIE kernels for LOSSY operations running on an UNSET global control register.

The bfp16 rounding finding was one instance of a family, not a one-off: a lossy operation reads a
core control register, nobody chose a value, and the hardware default is whatever it is. aie_api
documents the rounding default as `floor` (biased low on every product) and describes
`saturation_mode::none` as "allows values to overflow". Neither is a safe default and neither is set
by our kernels.

This sweep enumerates, per kernel source:

  * which lossy op FAMILIES it uses (each reads a control register)
  * whether it sets the corresponding register anywhere
  * the same for the vendored AMD kernels, as a comparison population

The point is coverage, not cleverness: a per-file table of "uses X, never sets X" is exactly the
artifact that turns "we should check saturation" into a list of files to fix.

    python3 scripts/sweep_control_registers.py                 # our kernels + AMD's
    python3 scripts/sweep_control_registers.py --roots a b c   # explicit roots
"""
import argparse
import collections
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Lossy op families and the control register each reads. The `ops` patterns are deliberately
# syntactic (what appears in kernel source) rather than semantic.
FAMILIES = {
    "rounding": {
        "register": "aie::set_rounding / crRoundMode",
        "why": "narrowing and block-float conversion round per this register; documented default "
               "is floor, which biases every value toward -inf and compounds over a reduction",
        "ops": [
            r"\baie::mmul\b", r"\bto_v\d+bfp16", r"\bsrs\b", r"\.to_vector\s*<", r"\bto_fixed\b",
            r"\bbfloat16\s*\(", r"\baie::store_v\b.*bfloat",
        ],
        "setter": [r"aie::set_rounding", r"set_rnd", r"crRoundMode", r"::rounding_mode::"],
    },
    "saturation": {
        "register": "aie::set_saturation / crSat",
        "why": "aie_api documents saturation_mode::none as 'allows values to overflow'; an overflow "
               "that WRAPS is a sign flip, not a small error",
        "ops": [
            r"\.to_vector\s*<", r"\bsrs\b", r"\bto_fixed\b", r"\baie::accum\b",
            r"\baie::mac\b", r"\baie::mmul\b", r"\baie::add\b", r"\baie::mul\b",
        ],
        "setter": [r"aie::set_saturation", r"set_sat", r"crSat", r"::saturation_mode::"],
    },
}

SRC_RE = re.compile(r"\.(cc|cpp|h|hpp)$")


def scan_file(path):
    try:
        text = open(path, errors="replace").read()
    except OSError:
        return None
    # Strip comments so a note ABOUT saturation is not counted as setting it.
    stripped = re.sub(r"//[^\n]*", "", text)
    stripped = re.sub(r"/\*.*?\*/", "", stripped, flags=re.S)
    out = {}
    for fam, spec in FAMILIES.items():
        uses = sorted({m.group(0) for pat in spec["ops"]
                       for m in re.finditer(pat, stripped)})
        sets = any(re.search(pat, stripped) for pat in spec["setter"])
        out[fam] = {"uses": uses, "sets": sets}
    return out


def walk(root):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames
                       if d not in {"build", ".git", "__pycache__", "test", "tests"}]
        for f in filenames:
            if SRC_RE.search(f):
                yield os.path.join(dirpath, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="*", default=None)
    ap.add_argument("--all-files", action="store_true", help="list files with no lossy op too")
    args = ap.parse_args()

    roots = args.roots or [
        os.path.join(REPO, "route_b_kernels"),
        os.path.join(REPO, "mlir-aie", "aie_kernels"),
    ]
    roots = [r for r in roots if os.path.isdir(r)]
    if not roots:
        print("no roots found")
        sys.exit(2)

    groups = collections.OrderedDict()
    for root in roots:
        label = os.path.relpath(root, REPO)
        groups[label] = []
        for p in sorted(walk(root)):
            r = scan_file(p)
            if r is None:
                continue
            interesting = any(v["uses"] for v in r.values())
            if interesting or args.all_files:
                groups[label].append((os.path.relpath(p, REPO), r))

    for label, files in groups.items():
        print(f"\n{'='*100}\n=== {label}   ({len(files)} sources with a lossy op)\n{'='*100}")
        print(f"{'file':<62} {'rounding':>18} {'saturation':>18}")
        for rel, r in files:
            cells = []
            for fam in ("rounding", "saturation"):
                v = r[fam]
                if not v["uses"]:
                    cells.append("-")
                elif v["sets"]:
                    cells.append(f"SET ({len(v['uses'])} ops)")
                else:
                    cells.append(f"UNSET ({len(v['uses'])} ops)")
            print(f"{rel[-62:]:<62} {cells[0]:>18} {cells[1]:>18}")

    print(f"\n{'='*100}\n=== summary\n{'='*100}")
    for label, files in groups.items():
        for fam in ("rounding", "saturation"):
            using = [f for f, r in files if r[fam]["uses"]]
            setting = [f for f, r in files if r[fam]["uses"] and r[fam]["sets"]]
            print(f"{label:<34} {fam:<11} {len(setting):>3} of {len(using):>3} sources that use a "
                  f"{fam}-sensitive op set the register")
    print()
    for fam, spec in FAMILIES.items():
        print(f"  {fam}: {spec['register']}\n      {spec['why']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
