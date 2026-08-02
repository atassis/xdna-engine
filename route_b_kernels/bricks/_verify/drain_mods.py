#!/usr/bin/env python3
"""Drain a SUBSET of verify modules, chosen by the DRAIN_MODS env var.

`drain_device.py` hardcodes the full TARGETS list, which is right for a full
re-verify but wasteful when you are iterating on one module (every run rebuilds
every xclbin). This is the same driver keyed off an env var instead:

    DRAIN_MODS=verify_cast_quant ./run.sh drain_mods.py

run.sh forwards no argv, so the selection has to arrive through the environment.

Only do_*() modules belong here. A SCRIPT-STYLE module (module-level code, no functions)
would run its gate as an import side effect and then record nothing, because there is no
do_*() to find -- so it would silently read as covered. Those are run as subprocesses by
drain_device.py's SCRIPT_TARGETS phase; this driver rejects them rather than pretending.
"""
import ast
import importlib
import os
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent

TARGETS = [m for m in os.environ.get("DRAIN_MODS", "").split(",") if m]
if not TARGETS:
    raise SystemExit("set DRAIN_MODS=mod1,mod2 (comma-separated verify module names)")

_script_style = []
for _m in TARGETS:
    _src = HERE / f"{_m}.py"
    if not _src.exists():
        raise SystemExit(f"no such verify module: {_m}")
    if not any(isinstance(n, ast.FunctionDef) and n.name.startswith("do_")
               for n in ast.parse(_src.read_text()).body):
        _script_style.append(_m)
if _script_style:
    raise SystemExit(
        "these are SCRIPT-STYLE modules with no do_*(); importing one runs its gate as a side\n"
        "effect and records nothing. Run them directly (./run.sh <mod>.py) or via\n"
        f"drain_device.py's SCRIPT_TARGETS phase:\n  " + "\n  ".join(_script_style))

results = []
for modname in TARGETS:
    print(f"\n===== module {modname} =====", flush=True)
    try:
        mod = importlib.import_module(modname)
    except Exception as e:
        print(f"  IMPORT-FAIL {modname}: {e}", flush=True)
        traceback.print_exc()
        results.append((modname, "<import>", "IMPORT-FAIL", None))
        continue
    for fn_name in sorted(n for n in dir(mod) if n.startswith("do_") and callable(getattr(mod, n))):
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
    npass += status == "PASS"
    rl2s = f"rel_l2={rl2:.3e}" if isinstance(rl2, float) else "rel_l2=--"
    print(f"  {brick:26s} {status:12s} {rl2s}   [{modname}]", flush=True)
print(f"\nDRAIN: {npass}/{len(results)} PASS", flush=True)
