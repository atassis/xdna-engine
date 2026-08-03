#!/usr/bin/env python3
"""Autorun device drain: import each target verify module, run every do_*()
device gate (build xclbin + run on aie2p + rel-L2 vs golden), tally PASS/FAIL.

The per-module __main__ often runs only the CPU self-check; the do_*() functions
are the deferred DEVICE step. This driver runs them. One do_*() crashing (e.g. a
backend codegen bug) is caught and reported, not fatal to the rest of the drain.

Modules to drain are listed in TARGETS (run.sh forwards no argv). TARGETS must
stay the FULL gate set: it listed 7 modules / 12 of the 27 gates until
2026-08-01, which read as full coverage while 15 gates went unrun. Use
drain_mods.py (DRAIN_MODS=) to iterate on a subset; do not trim this list.

TWO KINDS OF MODULE, and conflating them is how a whole second suite went
unmeasured. TARGETS modules expose do_*() functions and are imported. SCRIPT_TARGETS
modules are plain scripts -- module-level code, no functions -- and MUST be run as a
SUBPROCESS. Importing one executes its gate as a side effect and then records nothing,
because there is no do_*() to find, so a red one could not even fail the drain.

SCRIPT_TARGETS lists only script-style modules that are real GATES. Several script-style
modules in this directory are deliberate bisect probes (verify_dequant_f32,
verify_int4_shapes, verify_int4_stride, verify_rounding_ab): they print numbers and exit 0
by design, so an exit-status tally would score them as passes that mean nothing. They are
left out on purpose. verify_conv_1d_realdata is also excluded -- it takes a captured stage
dump as an argv argument, and run.sh forwards no argv.

Exit status is only trustworthy here because all four gates that used to print FAIL and
exit 0 anyway now assert (2026-08-02).
"""
import ast
import importlib
import re
import subprocess
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent

TARGETS = [
    "verify_f1",  # norm + elementwise (6 gates)
    "verify_f2",  # int8 (2)
    "verify_f2b",  # int8 gemv (6)
    "verify_f3",  # transpose (1)
    "verify_bfp16",
    "verify_cast_quant",
    "verify_rope_lut",
    "verify_lm_head_argmax",
    "verify_moe_topk_router",
    "verify_gatedeltanet",
    "verify_dequant_int4_group",
]

SCRIPT_TARGETS = [
    "probe_rope_partial_rotary",
    "verify_rmsnorm",
    "verify_int4_streamed",
    "verify_upscaler_bf16_conv2d",
    "verify_upscaler_bf16_gemm",
    "verify_upscaler_conv2d",
    "verify_upscaler_conv2d_ktile",
    "verify_upscaler_espcn_image",
    "verify_upscaler_espcn_wholenet",
]


def _has_do_fns(modname):
    """True if the module defines a top-level do_*(), by PARSING -- never importing.

    Importing to find out is the bug this whole split exists to fix: for a script-style
    module the import IS the gate run.
    """
    src = HERE / f"{modname}.py"
    if not src.exists():
        return None
    for node in ast.parse(src.read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("do_"):
            return True
    return False


def _check_lists():
    """Fail loud on a misfiled module rather than silently under-covering again."""
    bad = []
    for m in TARGETS:
        if _has_do_fns(m) is False:
            bad.append(f"{m} is in TARGETS but defines no do_*() -- it belongs in SCRIPT_TARGETS")
    for m in SCRIPT_TARGETS:
        if _has_do_fns(m) is True:
            bad.append(f"{m} is in SCRIPT_TARGETS but defines do_*() -- it belongs in TARGETS")
        if _has_do_fns(m) is None:
            bad.append(f"{m} is in SCRIPT_TARGETS but the file does not exist")
    if bad:
        raise SystemExit("drain target lists are wrong:\n  " + "\n  ".join(bad))


_check_lists()

results = []

# SCRIPT PHASE FIRST, AND THE ORDER IS LOAD-BEARING. These run as subprocesses, and a child can
# only create its own hw_context while THIS process has not opened the device: once a do_*() gate
# has run here, the parent holds a context and every child dies with
#   DRM_IOCTL_AMDXDNA_CREATE_HWCTX IOCTL failed (err=-22): Invalid argument
# (measured 2026-08-02 -- all 8 script gates failed that way when this loop ran last, and every one
# of them passes standalone). The NPU lock is held by our parent shell, so the children inherit
# serialisation without re-acquiring it.
for modname in SCRIPT_TARGETS:
    print(f"\n===== script {modname} =====", flush=True)
    p = subprocess.run([sys.executable, "-u", f"{modname}.py"], cwd=HERE,
                       capture_output=True, text=True)
    sys.stdout.write(p.stdout)
    if p.returncode != 0:
        sys.stdout.write(p.stderr[-2000:])
    rl2 = None
    for line in reversed(p.stdout.splitlines()):
        m = re.search(r"rel[-_]?l2[=: ]+([0-9.eE+-]+)", line, re.I)
        if m:
            try:
                rl2 = float(m.group(1))
            except ValueError:
                pass
            break
    results.append((modname, modname, "PASS" if p.returncode == 0 else "FAIL", rl2))

for modname in TARGETS:
    print(f"\n===== module {modname} =====", flush=True)
    try:
        mod = importlib.import_module(modname)
    except Exception as e:
        print(f"  IMPORT-FAIL {modname}: {e}", flush=True)
        traceback.print_exc()
        results.append((modname, "<import>", "IMPORT-FAIL", None))
        continue
    do_fns = sorted(n for n in dir(mod) if n.startswith("do_") and callable(getattr(mod, n)))
    if not do_fns:
        print(f"  (no do_*() device gates in {modname})", flush=True)
        continue
    for fn_name in do_fns:
        fn = getattr(mod, fn_name)
        brick = getattr(fn, "brick_name", fn_name)
        try:
            r = fn()
            if isinstance(r, dict):
                results.append((modname, brick, r.get("status", "?"), r.get("rel_l2")))
            elif isinstance(r, (list, tuple)):
                for sub in r:
                    if isinstance(sub, dict):
                        results.append((modname, sub.get("name", brick),
                                        sub.get("status", "?"), sub.get("rel_l2")))
            elif isinstance(r, bool):
                results.append((modname, brick, "PASS" if r else "FAIL", None))
            else:
                results.append((modname, brick, f"ran({type(r).__name__})", None))
        except Exception as e:
            print(f"  RUN-FAIL {brick} ({fn_name}): {e}", flush=True)
            traceback.print_exc()
            results.append((modname, brick, "RUN-FAIL", None))

print("\n\n========== DEVICE DRAIN SUMMARY ==========", flush=True)
npass = 0
for modname, brick, status, rl2 in results:
    rl2s = f"{rl2:.3e}" if isinstance(rl2, float) else "--"
    mark = "PASS" if status == "PASS" else status
    if status == "PASS":
        npass += 1
    print(f"  {brick:26s} {mark:12s} rel_l2={rl2s}   [{modname}]", flush=True)
print(f"\nDRAIN: {npass}/{len(results)} PASS", flush=True)
