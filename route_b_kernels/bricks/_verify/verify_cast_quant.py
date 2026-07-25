#!/usr/bin/env python3
"""cast-quant-bf16-int8 device-verify: bf16<->int8 symmetric quant/dequant + bf16<->f32 casts.

Rowwise FORMAT brick (per-row core_body, cols % N == 0). Four extern "C" entry points, each a
[m, cols] resident stream:
  quantize   bf16 -> int8  (needs `scale`)   q = clamp(round(x/scale), -128, 127)
  dequantize int8 -> bf16  (needs `scale`)   x = q * scale, narrowed to bf16
  cast       bf16 -> f32   (exact widen)
  cast       f32  -> bf16  (round-to-nearest-even narrow)

Scalars (`cols`, `scale`) are BAKED into the verify-shim as literals -- this sidesteps IRON's
scalar ABI (the same trick the rmsnorm/relu2 shims use). `scale` is pre-rounded to f32 so the
host golden and the on-device f32 literal denote the exact same value. Each do_<op> calls the
template entry `<name>_row<16>(...)` (explicit <16> disambiguates from the brick's own extern "C"
wrapper of the same name) through a fresh verify-shim symbol.

bf16 host dtype = ml_dtypes.bfloat16 (matches iron's bf16 tensor path). int8/f32 are numpy-native.

Gate: quantize + casts are numerically ~exact (device f32 datapath vs f64 golden diverges only at
round-half boundaries -> at most +-1 LSB on a few of 512 elements, rel-L2 ~1e-3); 3e-2 is safe.

  CPU-only golden cross-check (default):  python verify_cast_quant.py    (never touches device)
  Device run (opt-in, under the NPU lock): VERIFY_CAST_QUANT_DEVICE=1 ./run.sh verify_cast_quant.py
      (run.sh forwards only the filename, so the device path is gated on an env var it propagates;
       `python verify_cast_quant.py --device` also works for a direct invocation under the lock.)
"""
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path
import numpy as np
import ml_dtypes

import bricklib

BRICKS = Path(__file__).parent.parent
BRICK = "cast-quant-bf16-int8"
CCFILE = "cast_quant_bf16_int8.cc"


def golden_mod(brick, fname):
    """Load a brick's golden.py as a module; return (module, abs path to its .cc)."""
    p = BRICKS / brick / "golden.py"
    spec = importlib.util.spec_from_file_location(f"{brick.replace('-', '_')}_golden", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m, str(BRICKS / brick / fname)


# ---------------------------------------------------------------------------
# CASE BUILDERS -- pure CPU (golden + numpy only, NO device). Each returns a
# dict carrying everything verify_rowwise needs; the __main__ CPU cross-check
# reuses them to check the reference without ever building/running on device.
# ---------------------------------------------------------------------------
def _quant_case():
    g, cc = golden_mod(BRICK, CCFILE)
    m, cols = 8, 64  # 64 = multiple of N=16; 8 rows * 64 * 2B(bf16) well within 64KB L1
    rng = np.random.default_rng(0)
    x_f32 = (rng.standard_normal((m, cols)).astype(np.float32) * 3.0)
    x_bf16 = g.cast_f32_to_bf16(x_f32)                              # f32 carrying bf16 values
    scale = float(np.float32(np.max(np.abs(x_bf16)) / 127.0))      # per-tensor symmetric, f32-exact
    exp = g.quantize_bf16_to_int8(x_bf16, scale)                   # (m, cols) int8
    shim = ('extern "C" void quant_bf16_int8_verify(bfloat16*x,int8_t*o){'
            'quantize_bf16_to_int8_row<16>(x,o,%sf,%d);}' % (repr(scale), cols))
    return dict(name="cq-quant-bf16-int8", cc=cc, symbol="quant_bf16_int8_verify",
                shim=shim, m=m, in_cols=cols, out_cols=cols,
                x=x_bf16.astype(ml_dtypes.bfloat16), exp=exp, gate=3e-2,
                in_dt=ml_dtypes.bfloat16, out_dt=np.int8, ref_dtype=np.int8)


def _dequant_case():
    g, cc = golden_mod(BRICK, CCFILE)
    m, cols = 8, 64
    rng = np.random.default_rng(1)
    q = rng.integers(-128, 128, size=(m, cols)).astype(np.int8)    # full int8 range
    scale = float(np.float32(0.05))
    exp = g.dequantize_int8_to_bf16(q, scale)                     # f32 carrying bf16 values
    shim = ('extern "C" void dequant_int8_bf16_verify(int8_t*x,bfloat16*o){'
            'dequantize_int8_to_bf16_row<16>(x,o,%sf,%d);}' % (repr(scale), cols))
    return dict(name="cq-dequant-int8-bf16", cc=cc, symbol="dequant_int8_bf16_verify",
                shim=shim, m=m, in_cols=cols, out_cols=cols,
                x=q, exp=exp, gate=3e-2,
                in_dt=np.int8, out_dt=ml_dtypes.bfloat16, ref_dtype=np.float32)


def _cast_bf16_f32_case():
    g, cc = golden_mod(BRICK, CCFILE)
    m, cols = 8, 64
    rng = np.random.default_rng(2)
    x_f32 = (rng.standard_normal((m, cols)).astype(np.float32) * 3.0)
    x_bf16 = g.cast_f32_to_bf16(x_f32)                             # f32 carrying bf16 values
    exp = g.cast_bf16_to_f32(x_bf16)                              # exact widen -> f32
    shim = ('extern "C" void cast_bf16_f32_verify(bfloat16*x,float*o){'
            'cast_bf16_to_f32_row<16>(x,o,%d);}' % cols)
    return dict(name="cq-cast-bf16-f32", cc=cc, symbol="cast_bf16_f32_verify",
                shim=shim, m=m, in_cols=cols, out_cols=cols,
                x=x_bf16.astype(ml_dtypes.bfloat16), exp=exp, gate=3e-2,
                in_dt=ml_dtypes.bfloat16, out_dt=np.float32, ref_dtype=np.float32)


def _cast_f32_bf16_case():
    g, cc = golden_mod(BRICK, CCFILE)
    m, cols = 8, 64
    rng = np.random.default_rng(3)
    x_f32 = (rng.standard_normal((m, cols)).astype(np.float32) * 3.0)
    exp = g.cast_f32_to_bf16(x_f32)                              # f32 carrying bf16 values
    shim = ('extern "C" void cast_f32_bf16_verify(float*x,bfloat16*o){'
            'cast_f32_to_bf16_row<16>(x,o,%d);}' % cols)
    return dict(name="cq-cast-f32-bf16", cc=cc, symbol="cast_f32_bf16_verify",
                shim=shim, m=m, in_cols=cols, out_cols=cols,
                x=x_f32, exp=exp, gate=3e-2,
                in_dt=np.float32, out_dt=ml_dtypes.bfloat16, ref_dtype=np.float32)


CASES = [_quant_case, _dequant_case, _cast_bf16_f32_case, _cast_f32_bf16_case]


# ---------------------------------------------------------------------------
# DEVICE do_<op> configs -- each builds+runs the xclbin twice and gates rel-L2.
# DO NOT call these at CPU/import time (they trigger an iron.jit device build).
# ---------------------------------------------------------------------------
def _run_case(build):
    c = build()
    return bricklib.verify_rowwise(
        c["name"], c["cc"], c["shim"], c["symbol"],
        c["m"], c["in_cols"], c["out_cols"], c["x"], c["exp"], c["gate"],
        in_dt=c["in_dt"], out_dt=c["out_dt"])


def do_quantize():
    return _run_case(_quant_case)
do_quantize.brick_name = "cq-quant-bf16-int8"


def do_dequantize():
    return _run_case(_dequant_case)
do_dequantize.brick_name = "cq-dequant-int8-bf16"


def do_cast_bf16_f32():
    return _run_case(_cast_bf16_f32_case)
do_cast_bf16_f32.brick_name = "cq-cast-bf16-f32"


def do_cast_f32_bf16():
    return _run_case(_cast_f32_bf16_case)
do_cast_f32_bf16.brick_name = "cq-cast-f32-bf16"


results = []


def guard(fn):
    try:
        results.append(fn())
    except Exception as e:
        print(f"[{fn.brick_name:22s}] ERROR: {e}")
        traceback.print_exc()
        results.append(dict(name=fn.brick_name, status="ERROR", ok=False, err=str(e)))


# ---------------------------------------------------------------------------
# CPU-only golden cross-check. Recomputes each op's reference from the golden
# module and checks shape + dtype + finiteness. NEVER invokes the device do_
# fns / verify_rowwise. Also does a quant<->dequant round-trip sanity.
# ---------------------------------------------------------------------------
def _cpu_crosscheck():
    ok_all = True
    for build in CASES:
        c = build()
        exp = np.asarray(c["exp"])
        x = np.asarray(c["x"])
        checks = [
            ("exp shape", exp.shape == (c["m"], c["out_cols"])),
            ("exp dtype", exp.dtype == np.dtype(c["ref_dtype"])),
            ("exp finite", bool(np.isfinite(exp.astype(np.float64)).all())),
            ("in shape", x.shape == (c["m"], c["in_cols"])),
            ("in finite", bool(np.isfinite(x.astype(np.float64)).all())),
        ]
        bad = [n for n, v in checks if not v]
        status = "PASS" if not bad else "FAIL(" + ",".join(bad) + ")"
        ok_all = ok_all and not bad
        print(f"[{c['name']:22s}] exp{tuple(exp.shape)} dtype={exp.dtype} -> {status}")

    # round-trip sanity: quantize then dequantize should sit near the bf16 input.
    g, _ = golden_mod(BRICK, CCFILE)
    rng = np.random.default_rng(7)
    x_bf16 = g.cast_f32_to_bf16(rng.standard_normal(256).astype(np.float32) * 3.0)
    scale = float(np.float32(np.max(np.abs(x_bf16)) / 127.0))
    rt = g.dequantize_int8_to_bf16(g.quantize_bf16_to_int8(x_bf16, scale), scale)
    rt_err = g.rel_l2(rt, x_bf16)
    rt_ok = rt_err < 1e-2
    ok_all = ok_all and rt_ok
    print(f"[{'cq-quant-roundtrip':22s}] rel_l2={rt_err:.3e} (<1e-2) -> {'PASS' if rt_ok else 'FAIL'}")
    return ok_all


_DEVICE = ("--device" in sys.argv) or (os.environ.get("VERIFY_CAST_QUANT_DEVICE") == "1")

if __name__ == "__main__" and not _DEVICE:
    # CPU-only path (default): golden cross-check, no device build/run.
    ok = _cpu_crosscheck()
    print(f"\nverify_cast_quant CPU cross-check: {'ALL PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__" and _DEVICE:
    # Device path (opt-in): run every op's xclbin under the NPU lock.
    for fn in (do_quantize, do_dequantize, do_cast_bf16_f32, do_cast_f32_bf16):
        guard(fn)
    print("\n==== cast-quant-bf16-int8 SUMMARY ====")
    for r in results:
        print(f"  {r['name']:22s} {r.get('status', '?'):10s} "
              f"rel_l2={r.get('rel_l2', float('nan')):.3e}")
    passed = sum(1 for r in results if r.get("ok"))
    print(f"cast-quant: {passed}/{len(results)} PASS")
    print("JSON " + json.dumps(results, default=float))
    sys.exit(0 if passed == len(results) else 1)
