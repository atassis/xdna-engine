#!/usr/bin/env python3
"""Which RAIL delivers 512-float tiles intact -- oneshot, rowwise, or neither?

Standing contradiction to resolve. probe_sin_trip_count / probe_sin_sizes_recheck put sin through
`verify_oneshot` at 512 and 1024 and every configuration passed at rel-L2 ~1.3e-04. Then
verify_sin.py at m=32 x 512 through `verify_rowwise` failed with output in [-110.9, +124.96] --
sin cannot leave [-1, 1], so that is not a numerics error -- and verify_conv_transpose_1d.py at
PACK=512 through `verify_oneshot` failed with nz=30.6, which is very close to the sum of |bias|
over its 408 live outputs, i.e. the bias fill survived and every weighted accumulation contributed
nothing. Same kernels, same sizes, different rails, opposite verdicts.

A COPY kernel separates delivery from arithmetic: it cannot be wrong for a numerics reason, so any
divergence localises the transfer. Each case reports the first differing index, which is the number
that matters -- 64 would mean the old "only the first 256 B arrive" reading is real after all and
merely masked while every buffer fit inside 64 floats.

  O_copy512     oneshot copy, 512 in/out        -- if exact, ct1d's failure is arithmetic not DMA
  O_copy512_ct  oneshot copy at ct1d's exact packed layout
  R_copy64      rowwise copy, 32 x 64           -- the known-good baseline
  R_copy512     rowwise copy, 32 x 512          -- verify_sin's failing geometry
  R_copy512_m8  rowwise copy, 8 x 512           -- same tile, fewer tiles: is it tile or total?
  R_sin512_m8   rowwise sin,  8 x 512           -- arithmetic on the same delivery

Run under the device lock with PYTHONPATH at instance 7d8a49b5d7a0.
"""
import time
from pathlib import Path

import numpy as np

import bricklib

BRICK = (Path(__file__).parent.parent / "sin" / "sin.cc").resolve()
GEN = Path(__file__).parent / "gen"
CB = int(time.time() * 1e6) % 10**9


def first_diff(got, exp, rtol=1e-3, atol=1e-4):
    bad = np.where(~np.isclose(np.asarray(got).ravel(), np.asarray(exp).ravel(),
                               rtol=rtol, atol=atol))[0]
    return (int(bad[0]) if bad.size else -1), int(bad.size)


def oneshot_copy(tag, n):
    x = (np.random.default_rng(0).random(n, dtype=np.float32) * 128 - 64).astype(np.float32)
    sym = f"oc_{tag}_{CB}"
    shim = (f"// cachebust {CB}_{tag}\n"
            f'extern "C" void {sym}(float *x, float *out) {{\n'
            f"  for (int i = 0; i < {n}; i++) out[i] = x[i];\n}}\n")
    r = bricklib.verify_oneshot(name=tag, brick_cc=BRICK, shim_body=shim, symbol=sym,
                                inputs=[(x, np.float32)], out_numel=n, out_shape=(n,),
                                unpack=lambda d: d, golden=x, gate=0.0, out_dt=np.float32)
    fd, cnt = first_diff(r["got"], x)
    print(f"  {tag:14s} n={n:5d} rel_l2={r['rel_l2']:.3e} first_diff={fd:5d} n_diff={cnt:5d}",
          flush=True)


def rowwise(tag, m, cols, body, ref_fn):
    x = (np.random.default_rng(0).random((m, cols), dtype=np.float32) * 128 - 64).astype(np.float32)
    ref = ref_fn(x)
    sym = f"rw_{tag}_{CB}"
    shim = (f"// cachebust {CB}_{tag}\n"
            f'extern "C" void {sym}(float *x, float *out) {{ {body} }}\n')
    r = bricklib.verify_rowwise(name=tag, brick_cc=BRICK, shim_body=shim, symbol=sym,
                                m=m, in_cols=cols, out_cols=cols, x=x, expected=ref, gate=3e-2)
    got = np.asarray(r["got"], np.float32)
    fd, cnt = first_diff(got, ref, rtol=1e-3, atol=1e-3)
    # Report the first differing index WITHIN a row too: a per-tile truncation and a whole-stream
    # one look identical in a flat index.
    print(f"  {tag:14s} {m:3d}x{cols:4d} rel_l2={r['rel_l2']:.3e} first_diff={fd:6d} "
          f"(row {fd // cols if fd >= 0 else -1}, col {fd % cols if fd >= 0 else -1}) "
          f"n_diff={cnt:6d} range=[{got.min():+.2f},{got.max():+.2f}]", flush=True)


oneshot_copy("O_copy512", 512)

# ct1d's exact packed layout: x[0:64], w[64:256], bias[256:262], of a 512-float buffer.
oneshot_copy("O_copy512_ct", 512)

rowwise("R_copy64", 32, 64, "for (int i = 0; i < 64; i++) out[i] = x[i];", lambda a: a)
rowwise("R_copy512", 32, 512, "for (int i = 0; i < 512; i++) out[i] = x[i];", lambda a: a)
rowwise("R_copy512_m8", 8, 512, "for (int i = 0; i < 512; i++) out[i] = x[i];", lambda a: a)
rowwise("R_sin512_m8", 8, 512, "sin_f32(x, out, 512);",
        lambda a: np.sin(a.astype(np.float64)).astype(np.float32))
