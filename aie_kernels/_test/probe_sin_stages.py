#!/usr/bin/env python3
"""Bisect the sin brick: dump q, k, r and res per element and compare each against the golden.

Unique symbol per run -- the JIT cache is keyed such that reusing a symbol returns a STALE xclbin,
which silently invalidated two earlier device runs.
"""
import importlib.util
from pathlib import Path
import numpy as np
import bricklib

HERE = Path(__file__).parent
BRICK = HERE.parent / "sin"
N = 256
spec = importlib.util.spec_from_file_location("sin_golden", BRICK / "golden.py")
g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)

rng = np.random.default_rng(0)
x = (rng.random(N, dtype=np.float32) * 128.0 - 64.0).astype(np.float32)

shim = f'''
extern "C" void sin_stages(float *x, float *out) {{
  const int N16 = 16;
  ::aie::vector<float, N16> two_pi = ::aie::broadcast<float, N16>(6.28318530717958648f);
  ::aie::vector<float, N16> inv_two_pi = ::aie::broadcast<float, N16>(0.15915494309189535f);
  ::aie::vector<float, N16> magic = ::aie::broadcast<float, N16>(8388608.0f);
  for (int i = 0; i < {N}; i += N16) {{
    ::aie::vector<float, N16> xv = ::aie::load_v<N16>(x + i);
    ::aie::vector<float, N16> q = ::aie::mul(xv, inv_two_pi);
    ::aie::vector<float, N16> k = ::aie::sub(::aie::add(q, magic), magic);
    ::aie::vector<float, N16> kt = ::aie::mul(k, two_pi);
    ::aie::vector<float, N16> r = ::aie::sub(xv, kt);
    ::aie::store_v(out + i, q);
    ::aie::store_v(out + {N} + i, k);
    ::aie::store_v(out + {2*N} + i, kt);
    ::aie::store_v(out + {3*N} + i, r);
  }}
}}
'''
res = bricklib.verify_oneshot(
    name="sin_stages", brick_cc=BRICK / "sin.cc", shim_body=shim, symbol="sin_stages",
    inputs=[(x, np.float32)], out_numel=4 * N, out_shape=(4 * N,),
    unpack=lambda d: d, golden=np.zeros(4 * N, np.float32), gate=1e9, out_dt=np.float32)
d = res["got"]
q_d, k_d, kt_d, r_d = d[:N], d[N:2*N], d[2*N:3*N], d[3*N:]
q_h = (x / (2*np.pi)).astype(np.float32)
k_h = np.rint(q_h).astype(np.float32)
r_h = g.fold(x)
for nm, dev, host in (("q", q_d, q_h), ("k", k_d, k_h), ("kt", kt_d, (k_h*2*np.pi).astype(np.float32)), ("r", r_d, r_h)):
    bad = int(np.sum(~np.isclose(dev, host, rtol=1e-4, atol=1e-4)))
    print(f"{nm:3s} dev[:4]={np.array2string(dev[:4],precision=4)} host[:4]={np.array2string(host[:4],precision=4)} mismatch={bad}/{N}")
