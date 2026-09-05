#!/usr/bin/env python3
"""Is the 64-float per-call limit a real tile bound, or a bricklib oneshot artifact?

A COPY kernel (out[i] = in[i]) cannot be wrong for arithmetic reasons, so the first index that
differs from the input localises the delivery boundary exactly. Swept through BOTH rails.

This gates the whole Stage-2 tiling design: real decoder stages are c_in=1536 wide, far beyond 64,
so whether the block tiles by 64 or by L1 capacity depends on the answer here.

Unique symbol + shim marker per configuration: the JIT cache key hashes only the shim text, so
reusing a symbol silently returns a stale xclbin.
"""
import time
from pathlib import Path

import numpy as np

import bricklib

BRICK = (Path(__file__).parent.parent / "sin" / "sin.cc").resolve()   # included but unused


def copy_shim(sym, n):
    return (f"// cachebust {int(time.time() * 1e6) % 10**9}\n"
            f'extern "C" void {sym}(float *x, float *out) {{\n'
            f"  for (int i = 0; i + 16 <= {n}; i += 16) {{\n"
            f"    ::aie::store_v(out + i, ::aie::load_v<16>(x + i));\n"
            "  }\n}\n")


def first_bad(got, exp):
    bad = np.where(~np.isclose(np.asarray(got, np.float32).ravel(), exp.ravel(), atol=1e-6))[0]
    return int(bad[0]) if bad.size else None


print("=== oneshot: in and out both N floats ===")
for N in (64, 128, 256, 1024, 4096):
    x = (np.arange(N, dtype=np.float32) + 1.0)
    sym = f"cp_os_{N}_{int(time.time() * 1e6) % 10**6}"
    try:
        r = bricklib.verify_oneshot(name=f"copy_os{N}", brick_cc=BRICK, shim_body=copy_shim(sym, N),
                                    symbol=sym, inputs=[(x, np.float32)], out_numel=N,
                                    out_shape=(N,), unpack=lambda d: d, golden=x, gate=1e-6,
                                    out_dt=np.float32)
        print(f"  N={N:5d} rel_l2={r['rel_l2']:.2e} first_bad={first_bad(r['got'], x)} -> {r['status']}")
    except Exception as exc:
        print(f"  N={N:5d} BUILD/RUN FAILED: {type(exc).__name__}: {str(exc)[:110]}")

print("=== rowwise: m rows of COLS floats, one call per row ===")
for M, COLS in ((8, 64), (8, 128), (8, 256), (8, 512), (64, 64)):
    x = (np.arange(M * COLS, dtype=np.float32).reshape(M, COLS) + 1.0)
    sym = f"cp_rw_{COLS}_{int(time.time() * 1e6) % 10**6}"
    try:
        r = bricklib.verify_rowwise(name=f"copy_rw{M}x{COLS}", brick_cc=BRICK,
                                    shim_body=copy_shim(sym, COLS), symbol=sym, m=M,
                                    in_cols=COLS, out_cols=COLS, x=x, expected=x, gate=1e-6)
        print(f"  {M}x{COLS:<4d} rel_l2={r['rel_l2']:.2e} first_bad={first_bad(r['got'], x)} -> {r['status']}")
    except Exception as exc:
        print(f"  {M}x{COLS:<4d} BUILD/RUN FAILED: {type(exc).__name__}: {str(exc)[:110]}")
