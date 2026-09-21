#!/usr/bin/env python3
"""conv1d_step on device: one 256-channel tile, three chained steps with the device's hist_out fed back.

The output buffer is [y bf16 C | hist_out bf16 (K-1)*C]. Each step is gated against kernel_model,
which is exact up to accumulation order, so the gate is tight; a golden with the taps reversed is the
negative control the gate must reject.
"""
import importlib.util
import sys
from pathlib import Path

import ml_dtypes
import numpy as np

BRICK_DIR = Path(__file__).parent.parent / "conv1d-step"
BRICK_CC = str(BRICK_DIR / "conv1d_step.cc")
C, K = 256, 4
GATE = 1e-2


def _golden():
    spec = importlib.util.spec_from_file_location("cs_golden", BRICK_DIR / "golden.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def step(bricklib, x, hist, w):
    bf = ml_dtypes.bfloat16
    g = _golden()
    y_ref, h_ref = g.kernel_model(x, hist, w)
    # A core has two input DMA channels: [hist | x] is the K-row window, contiguous, as one input.
    shim = ('extern "C" void cs_verify(bfloat16* win, bfloat16* w, bfloat16* out) {\n'
            f'  conv1d_step(win + {(K - 1) * C}, win, w, out, out + {C});\n'
            '}\n')
    r = bricklib.verify_oneshot(
        name="conv1d_step", brick_cc=BRICK_CC, shim_body=shim, symbol="cs_verify",
        inputs=[(g.bf16(np.concatenate([hist.reshape(-1), x])).astype(bf), bf),
                (g.bf16(w).reshape(-1).astype(bf), bf)],
        out_numel=C * K, out_shape=None, unpack=lambda f: np.asarray(f).astype(np.float32),
        golden=np.concatenate([y_ref, h_ref.reshape(-1)]), gate=GATE,
        compile_flags=[f"-DCS_C={C}", f"-DCS_K={K}"], out_dt=bf)
    got = r["got"]
    return r, got[:C], got[C:].reshape(K - 1, C), y_ref


def main():
    import bricklib
    g = _golden()
    rng = np.random.default_rng(7)
    w = (rng.standard_normal((K, C)) * 0.5).astype(np.float32)
    hist = (rng.standard_normal((K - 1, C))).astype(np.float32)
    ok = True
    for n in range(3):
        x = rng.standard_normal(C).astype(np.float32)
        r, y, hist_dev, y_ref = step(bricklib, x, hist, w)
        wrong, _ = g.kernel_model(x, hist, w[::-1])
        ry, rneg = g.rel_l2(y, y_ref), g.rel_l2(y, wrong)
        rh = g.rel_l2(hist_dev, g.bf16(np.concatenate([hist[1:], x[None]], 0)))
        good = r["ok"] and ry <= GATE and rh == 0.0 and rneg > GATE
        print(f"[conv1d-step {n}] y rel-L2 {ry:.3e} (<= {GATE:.0e})  hist rel-L2 {rh:.1e} (== 0)  "
              f"vs reversed-taps {rneg:.2e} (> gate)  run2run {r['run2run']:.1e} -> {'PASS' if good else 'FAIL'}")
        ok &= good
        hist = hist_dev
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
