#!/usr/bin/env python3
"""Dump every Horner intermediate from the sin poly and find the first step that diverges."""
import importlib.util
from pathlib import Path
import numpy as np
import bricklib

HERE = Path(__file__).parent
BRICK = HERE.parent / "sin"
N = 64
spec = importlib.util.spec_from_file_location("sin_golden", BRICK / "golden.py")
g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)

rng = np.random.default_rng(0)
x = (rng.random(N, dtype=np.float32) * 128.0 - 64.0).astype(np.float32)

C = dict(c1=-1.66666666666666657e-01, c2=8.33333333333333322e-03, c3=-1.98412698412698413e-04,
         c4=2.75573192239858907e-06, c5=-2.50521083854417188e-08)
NAMES = ["r", "r2", "m5", "p4", "m4", "p3", "m3", "p2", "m2", "p1", "m1", "p0", "res"]

decl = "\n".join(f'  ::aie::vector<float, W> {k} = ::aie::broadcast<float, W>({v:.17e}f);' for k, v in C.items())
shim = f'''
extern "C" void sin_horner(float *x, float *out) {{
  const int W = 16;
  ::aie::vector<float, W> two_pi = ::aie::broadcast<float, W>(6.28318530717958648f);
  ::aie::vector<float, W> inv_two_pi = ::aie::broadcast<float, W>(0.15915494309189535f);
  ::aie::vector<float, W> magic = ::aie::broadcast<float, W>(8388608.0f);
  ::aie::vector<float, W> one = ::aie::broadcast<float, W>(1.0f);
{decl}
  for (int i = 0; i < {N}; i += W) {{
    ::aie::vector<float, W> xv = ::aie::load_v<W>(x + i);
    ::aie::vector<float, W> q = ::aie::mul(xv, inv_two_pi);
    ::aie::vector<float, W> k = ::aie::sub(::aie::add(q, magic), magic);
    ::aie::vector<float, W> kt = ::aie::mul(k, two_pi);
    ::aie::vector<float, W> r = ::aie::sub(xv, kt);
    ::aie::vector<float, W> r2 = ::aie::mul(r, r);
    ::aie::vector<float, W> m5 = ::aie::mul(c5, r2);
    ::aie::vector<float, W> p4 = ::aie::add(m5, c4);
    ::aie::vector<float, W> m4 = ::aie::mul(p4, r2);
    ::aie::vector<float, W> p3 = ::aie::add(m4, c3);
    ::aie::vector<float, W> m3 = ::aie::mul(p3, r2);
    ::aie::vector<float, W> p2 = ::aie::add(m3, c2);
    ::aie::vector<float, W> m2 = ::aie::mul(p2, r2);
    ::aie::vector<float, W> p1 = ::aie::add(m2, c1);
    ::aie::vector<float, W> m1 = ::aie::mul(p1, r2);
    ::aie::vector<float, W> p0 = ::aie::add(m1, one);
    ::aie::vector<float, W> res = ::aie::mul(p0, r);
'''
for j, nm in enumerate(NAMES):
    shim += f"    ::aie::store_v(out + {j} * {N} + i, {nm});\n"
shim += "  }\n}\n"

res = bricklib.verify_oneshot(
    name="sin_horner", brick_cc=BRICK / "sin.cc", shim_body=shim, symbol="sin_horner",
    inputs=[(x, np.float32)], out_numel=len(NAMES) * N, out_shape=(len(NAMES) * N,),
    unpack=lambda d: d, golden=np.zeros(len(NAMES) * N, np.float32), gate=1e9, out_dt=np.float32)
d = res["got"]

r_h = g.fold(x); r2_h = (r_h * r_h).astype(np.float32)
h = {"r": r_h, "r2": r2_h}
p = (np.float32(C["c5"]) * r2_h + np.float32(C["c4"])).astype(np.float32)
h["m5"] = (np.float32(C["c5"]) * r2_h).astype(np.float32); h["p4"] = p
for nm_m, nm_p, c in (("m4", "p3", "c3"), ("m3", "p2", "c2"), ("m2", "p1", "c1")):
    h[nm_m] = (p * r2_h).astype(np.float32); p = (h[nm_m] + np.float32(C[c])).astype(np.float32); h[nm_p] = p
h["m1"] = (p * r2_h).astype(np.float32); h["p0"] = (h["m1"] + np.float32(1.0)).astype(np.float32)
h["res"] = (h["p0"] * r_h).astype(np.float32)

for j, nm in enumerate(NAMES):
    dev = d[j * N:(j + 1) * N]; host = h[nm]
    bad = int(np.sum(~np.isclose(dev, host, rtol=2e-3, atol=1e-7)))
    flag = "  <-- FIRST DIVERGENCE" if bad and all(int(np.sum(~np.isclose(d[i*N:(i+1)*N], h[NAMES[i]], rtol=2e-3, atol=1e-7))) == 0 for i in range(j)) else ""
    print(f"{nm:4s} mismatch={bad:3d}/{N}  dev[0]={dev[0]:+.6e} host[0]={host[0]:+.6e}{flag}")
