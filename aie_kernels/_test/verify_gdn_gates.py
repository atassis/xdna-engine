#!/usr/bin/env python3
"""gdn_gates on device at Qwen3.5's 32 value heads, inputs drawn from the real checkpoint's ranges.

A_log and dt_bias come from layer 0 of Qwen/Qwen3.5-4B when the checkpoint is on disk (a and b span
the projection's output range); otherwise from the ranges HF initialises them in. Negative control:
the same golden with dt_bias dropped, a plausible wiring bug, must fail the gate.
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

import ml_dtypes
import numpy as np

BRICK_DIR = Path(__file__).parent.parent / "gdn-gates"
BRICK_CC = str(BRICK_DIR / "gdn_gates.cc")
N = 32
# Every element within 2^-12 of the f64 golden: 8x under the bf16 resolution of the activations
# around the gates. aie2p emulates f32 vector multiplies through bf16 (kb/aie2p-f32-elementwise-is-
# emulated-via-bf16), and the exp/log1p chain measures 8.5e-5 on device, not the polynomials' 2e-7.
GATE = 2.0 ** -12
CKPT = "/mnt/data/xdna/artifacts/qwen3.5-4b/hf"
STACK = 3392  # aiecc's measurement of this core's frame; it refuses to build below it


def _golden():
    spec = importlib.util.spec_from_file_location("gg_golden", BRICK_DIR / "golden.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def params(rng):
    try:
        from safetensors import safe_open
        idx = json.load(open(os.path.join(CKPT, "model.safetensors.index.json")))["weight_map"]
        out = []
        for leaf in ("A_log", "dt_bias"):
            name = f"model.language_model.layers.0.linear_attn.{leaf}"
            with safe_open(os.path.join(CKPT, idx[name]), framework="numpy") as f:
                out.append(f.get_tensor(name).astype(np.float32))
        return (*out, "checkpoint layer 0")
    except Exception:  # noqa: BLE001 -- no checkpoint: HF's own init ranges
        return (np.log(rng.uniform(0.01, 16, N)).astype(np.float32),
                np.ones(N, np.float32), "HF init ranges")


def main():
    import bricklib
    g = _golden()
    bf = ml_dtypes.bfloat16
    rng = np.random.default_rng(11)
    a_log, dt_bias, src = params(rng)
    ok = True
    for scale in (1.0, 6.0):
        a = (rng.standard_normal(N) * scale).astype(np.float32)
        b = (rng.standard_normal(N) * scale).astype(np.float32)
        ref = g.gates(a, b, a_log, dt_bias)
        neg = g.gates(a, b, a_log, np.zeros(N, np.float32))
        pm = np.concatenate([-np.exp(a_log.astype(np.float64)).astype(np.float32), dt_bias])
        shim = 'extern "C" void gg_verify(bfloat16* ab, float* p, float* out) { gdn_gates(ab, p, out); }\n'
        r = bricklib.verify_oneshot(
            name="gdn_gates", brick_cc=BRICK_CC, shim_body=shim, symbol="gg_verify",
            inputs=[(g.bf16(np.concatenate([a, b])).astype(bf), bf), (pm, np.float32)],
            out_numel=2 * N, out_shape=None, unpack=lambda f: np.asarray(f, np.float64),
            golden=ref.reshape(-1), gate=GATE, compile_flags=[f"-DGG_N={N}"], out_dt=np.float32, stack_size=STACK)
        got = r["got"].reshape(N, 2)
        e_a, e_b = g.max_rel(got[:, 0], ref[:, 0]), g.max_rel(got[:, 1], ref[:, 1])
        e_neg = g.max_rel(got[:, 0], neg[:, 0])
        good = r["ok"] and max(e_a, e_b) <= GATE and e_neg > GATE and np.isfinite(got).all()
        print(f"[gdn-gates |a|~{scale:g}] params from {src}: max rel alpha {e_a:.2e} beta {e_b:.2e} "
              f"(<= {GATE:.0e})  vs no-dt_bias {e_neg:.2e} (> gate)  alpha range "
              f"[{ref[:, 0].min():.3g}, {ref[:, 0].max():.3g}]  run2run {r['run2run']:.1e} "
              f"-> {'PASS' if good else 'FAIL'}")
        ok &= good
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
