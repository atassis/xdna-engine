#!/usr/bin/env python3
"""Re-gate the bricks whose frames exceed the default reservation.

`audit_stack_frames.py` flags three, by summing per-function `.stack_sizes` against the 0x400
default: prefill-attn 0x700 (already re-gated green on all 32 heads), layernorm 0x480 across four
functions, rope-lut 0x480 in a SINGLE function.

rope-lut is the interesting one. Its frame is one function's, so the audit's conservative
sum-assumes-nesting caveat does not apply -- 0x480 > 0x400 unconditionally. And the brick is
currently recorded GREEN at rel-L2 6.129e-03. Both can be true: an overrun corrupts whatever sits
adjacent, and if that is a small part of the output the gate can still pass. So the question is not
pass/fail, it is whether the NUMBER MOVES when the frame fits.

  number moves  -> the recorded green was measured under an active overrun; re-baseline it
  number static -> the frame exceeds the reservation without consequence at this shape, which is
                   worth knowing too (it bounds how alarming the audit's flag is)

layernorm's 0x480 is a four-function SUM, so it may never nest that deep and may be a false flag.
That is exactly what this measures.

Run:  ./run.sh probe_flagged_bricks_stack.py
"""
import importlib
import sys
import traceback
from pathlib import Path

import numpy as np

import bricklib

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

# Default first so any cache/order effect shows up as the default being the odd one out, not last.
STACKS = [None, 0x800, 0xD00]

_orig_oneshot = bricklib.verify_oneshot
_orig_rowwise = bricklib.verify_rowwise
_CUR = {"stack": None}


def _oneshot(*a, **kw):
    kw.setdefault("stack_size", _CUR["stack"])
    return _orig_oneshot(*a, **kw)


def _rowwise(*a, **kw):
    kw.setdefault("stack_size", _CUR["stack"])
    return _orig_rowwise(*a, **kw)


bricklib.verify_oneshot = _oneshot
bricklib.verify_rowwise = _rowwise


def run(label, call):
    print(f"\n=== {label} ===")
    print(f"{'stack':>8} {'rel-L2':>12} {'run2run':>10}  verdict")
    seen = {}
    for st in STACKS:
        _CUR["stack"] = st
        try:
            res = call()
            seen[st] = res["rel_l2"]
            print(f"{('default' if st is None else f'{st:#x}'):>8} {res['rel_l2']:12.3e} "
                  f"{res['run2run']:10.2e}  {'PASS' if res['ok'] else 'FAIL'}")
        except Exception:
            print(f"{('default' if st is None else f'{st:#x}'):>8} ERROR")
            traceback.print_exc(limit=2)
    vals = [v for v in seen.values() if isinstance(v, float)]
    if len(vals) >= 2:
        spread = max(vals) - min(vals)
        rel = spread / max(max(vals), 1e-30)
        print(f"  spread {spread:.3e} ({rel * 100:.2f}% of worst) -> "
              f"{'MOVES: the recorded number was measured under an overrun' if rel > 0.01 else 'static: frame exceeds reservation without consequence at this shape'}")
    return seen


# ---- rope-lut (verify_oneshot) ----
try:
    vrl = importlib.import_module("verify_rope_lut")
    fn = getattr(vrl, "verify_rope_lut", None) or getattr(vrl, "do_rope_lut", None)
    if fn is None:
        print("verify_rope_lut: no callable entry point found; "
              f"module has {[n for n in dir(vrl) if not n.startswith('_')][:12]}")
    else:
        run("rope-lut  (frame 0x480, single function)", fn)
except Exception:
    print("rope-lut: could not import verify_rope_lut")
    traceback.print_exc(limit=3)

# ---- layernorm (verify_rowwise, lives in verify_f1) ----
try:
    vf1 = importlib.import_module("verify_f1")
    run("layernorm (frame 0x480, sum over 4 functions)", vf1.do_layernorm)
except Exception:
    print("layernorm: could not import verify_f1.do_layernorm")
    traceback.print_exc(limit=3)
