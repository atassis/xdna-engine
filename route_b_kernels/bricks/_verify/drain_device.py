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
"""
import importlib
import traceback

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
