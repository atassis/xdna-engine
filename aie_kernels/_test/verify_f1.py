#!/usr/bin/env python3
"""F1 family device-verify: norm + elementwise f32 bricks, one device session.
   rmsnorm, layernorm, qk-norm, relu2, geglu, swiglu.  Gate rel-L2 <= brick threshold.
"""
import importlib.util
import json
import sys
import traceback
from pathlib import Path
import numpy as np

import bricklib

BRICKS = Path(__file__).parent.parent


def golden_mod(brick):
    p = BRICKS / brick / "golden.py"
    spec = importlib.util.spec_from_file_location(f"{brick}_golden", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m, str(BRICKS / brick / (brick.replace("-", "_") + ".cc"))


def cc_path(brick, fname):
    return str(BRICKS / brick / fname)


results = []


def guard(fn):
    try:
        results.append(fn())
    except Exception as e:
        name = getattr(fn, "brick_name", fn.__name__)
        print(f"[{name:22s}] ERROR: {e}")
        traceback.print_exc()
        results.append(dict(name=name, status="ERROR", ok=False, err=str(e)))


def do_rmsnorm():
    g, cc = golden_mod("rmsnorm")
    m, cols = 8, 768
    rng = np.random.default_rng(0)
    x = rng.standard_normal((m, cols)).astype(np.float32)
    gamma = (rng.standard_normal((cols,)).astype(np.float32) * 0.1 + 1.0)
    beta = (rng.standard_normal((cols,)).astype(np.float32) * 0.01)
    exp = g.rmsnorm_ref(x, gamma, beta, eps=1e-6)
    shim = ('extern "C" void rmsnorm_verify(float*x,float*gb,float*o){'
            'rmsnorm_f32<16>(x,gb,gb+%d,o,%d,1e-6f);}' % (cols, cols))
    return bricklib.verify_rowwise("rmsnorm", cc, shim, "rmsnorm_verify",
                                   m, cols, cols, x, exp, 3e-2,
                                   const=np.concatenate([gamma, beta]))
do_rmsnorm.brick_name = "rmsnorm"


def do_layernorm():
    g, cc = golden_mod("layernorm")
    m, cols = 8, 768
    rng = np.random.default_rng(0)
    x = rng.standard_normal((m, cols)).astype(np.float32)
    gamma = (rng.standard_normal((cols,)).astype(np.float32) * 0.1 + 1.0)
    beta = (rng.standard_normal((cols,)).astype(np.float32) * 0.01)
    exp = g.layernorm_ln_affine(x, gamma, beta, eps=1e-5)
    shim = ('extern "C" void ln_verify(float*x,float*gb,float*o){'
            'layernorm_ln_affine_f32(x,o,%d,gb,gb+%d);}' % (cols, cols))
    return bricklib.verify_rowwise("layernorm", cc, shim, "ln_verify",
                                   m, cols, cols, x, exp, 3e-2,
                                   const=np.concatenate([gamma, beta]))
do_layernorm.brick_name = "layernorm"


def do_qknorm():
    g, cc = golden_mod("qk-norm")
    cc = cc_path("qk-norm", "qk_norm.cc")
    m, cols = 8, 64
    x, gamma, out = g.make_golden_case(m, cols, seed=0, with_gamma=True, eps=1e-6)
    shim = ('extern "C" void qk_verify(float*x,float*gamma,float*o){'
            'qk_norm_f32_gamma(x,gamma,o,1,%d);}' % cols)
    return bricklib.verify_rowwise("qk-norm", cc, shim, "qk_verify",
                                   m, cols, cols, x, out, 3e-2, const=gamma)
do_qknorm.brick_name = "qk-norm"


def do_relu2():
    g, cc = golden_mod("relu2")
    m, cols = 64, 1024
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((m, cols)).astype(np.float32) * 3.0)
    exp = g.relu2_ref(x.astype(np.float64))
    shim = ('extern "C" void relu2_verify(float*x,float*o){'
            'relu2_row(x,o,%d);}' % cols)
    return bricklib.verify_rowwise("relu2", cc, shim, "relu2_verify",
                                   m, cols, cols, x, exp, 3e-2)
do_relu2.brick_name = "relu2"


def do_geglu():
    g, cc = golden_mod("geglu")
    m, cols = 32, 64  # per-half; input row = 2*cols
    rng = np.random.default_rng(0)
    up = rng.standard_normal((m, cols)).astype(np.float32)
    gate = rng.standard_normal((m, cols)).astype(np.float32)
    row_in = np.concatenate([up, gate], axis=-1).astype(np.float32)
    exp = g.geglu_row_stream(row_in, cols)
    shim = ('extern "C" void geglu_verify(float*x,float*o){'
            'geglu_row(x,o,%d);}' % cols)
    return bricklib.verify_rowwise("geglu", cc, shim, "geglu_verify",
                                   m, 2 * cols, cols, row_in, exp, 3e-2)
do_geglu.brick_name = "geglu"


def do_swiglu():
    g, cc = golden_mod("swiglu")
    m, cols = 64, 1024
    rng = np.random.default_rng(0)
    gate = (rng.standard_normal((m, cols)).astype(np.float32) * 3.0)
    up = (rng.standard_normal((m, cols)).astype(np.float32) * 3.0)
    row_in = np.concatenate([gate, up], axis=-1).astype(np.float32)
    exp = g.swiglu_ref(gate.astype(np.float64), up.astype(np.float64))
    shim = ('extern "C" void swiglu_verify(float*x,float*o){'
            'swiglu_f32_f32(x,x+%d,o);}' % cols)
    return bricklib.verify_rowwise("swiglu", cc, shim, "swiglu_verify",
                                   m, 2 * cols, cols, row_in, exp, 3e-2)
do_swiglu.brick_name = "swiglu"


if __name__ == "__main__":
    for fn in (do_rmsnorm, do_layernorm, do_qknorm, do_relu2, do_geglu, do_swiglu):
        guard(fn)
    print("\n==== F1 SUMMARY ====")
    for r in results:
        print(f"  {r['name']:22s} {r['status']:10s} "
              f"rel_l2={r.get('rel_l2', float('nan')):.3e}")
    passed = sum(1 for r in results if r.get("ok"))
    print(f"F1: {passed}/{len(results)} PASS")
    print("JSON " + json.dumps(results))
    sys.exit(0 if passed == len(results) else 1)
