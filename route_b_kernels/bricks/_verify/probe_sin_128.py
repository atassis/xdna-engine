#!/usr/bin/env python3
"""At 128 floats/call, which op in sin_v introduces the aliasing? Copy is exact at 1024, so the
fault is inside the arithmetic. Progressive fragments localise it."""
import time
from pathlib import Path
import numpy as np
import bricklib

BRICK = (Path(__file__).parent.parent / "sin" / "sin.cc").resolve()
N = 128
x = (np.random.default_rng(0).random(N, dtype=np.float32) * 128 - 64).astype(np.float32)
TP = 6.28318530717958648
FRAGS = {
    "copy":      ("::aie::vector<float,16> r = v;", lambda a: a),
    "fold":      ("""::aie::vector<float,16> k = ::aie::mul(v, ::aie::broadcast<float,16>(0.15915494309189535f));
      k = ::aie::add(k, ::aie::broadcast<float,16>(8388608.0f));
      k = ::aie::sub(k, ::aie::broadcast<float,16>(8388608.0f));
      ::aie::vector<float,16> kt = ::aie::mul(k, ::aie::broadcast<float,16>(6.28318530717958648f));
      ::aie::vector<float,16> r = ::aie::sub(v, kt);""",
                  lambda a: (a - TP * np.rint(a / TP)).astype(np.float32)),
    "fold_sq":   ("""::aie::vector<float,16> k = ::aie::mul(v, ::aie::broadcast<float,16>(0.15915494309189535f));
      k = ::aie::add(k, ::aie::broadcast<float,16>(8388608.0f));
      k = ::aie::sub(k, ::aie::broadcast<float,16>(8388608.0f));
      ::aie::vector<float,16> kt = ::aie::mul(k, ::aie::broadcast<float,16>(6.28318530717958648f));
      ::aie::vector<float,16> f = ::aie::sub(v, kt);
      ::aie::vector<float,16> r = ::aie::mul(f, f);""",
                  lambda a: (lambda f: (f * f).astype(np.float32))((a - TP * np.rint(a / TP)).astype(np.float32))),
    "full_sin":  ("::aie::vector<float,16> r = route_b_bricks::sin_v<16>(v);",
                  lambda a: np.sin(a.astype(np.float64)).astype(np.float32)),
}
for name, (body, ref_fn) in FRAGS.items():
    sym = f"fr_{name}_{int(time.time()*1e6) % 10**6}"
    shim = (f"// cb {int(time.time()*1e6) % 10**9}\n"
            f'extern "C" void {sym}(float *x, float *out) {{\n'
            f"  for (int i = 0; i + 16 <= {N}; i += 16) {{\n"
            f"    ::aie::vector<float,16> v = ::aie::load_v<16>(x + i);\n"
            f"    {body}\n"
            f"    ::aie::store_v(out + i, r);\n  }}\n}}\n")
    try:
        ref = ref_fn(x)
        r = bricklib.verify_oneshot(name=f"frag_{name}", brick_cc=BRICK, shim_body=shim, symbol=sym,
                                    inputs=[(x, np.float32)], out_numel=N, out_shape=(N,),
                                    unpack=lambda d: d, golden=ref, gate=1e-3, out_dt=np.float32)
        got = np.asarray(r["got"], np.float32)
        bad = np.where(~np.isclose(got, ref, rtol=1e-3, atol=1e-4))[0]
        print(f"  {name:9s} rel_l2={r['rel_l2']:.3e} bad={bad.size}/{N} first={int(bad[0]) if bad.size else None}")
    except Exception as exc:
        print(f"  {name:9s} FAILED: {type(exc).__name__}: {str(exc)[:90]}")
