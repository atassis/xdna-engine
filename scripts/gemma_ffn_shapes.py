#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Emit shapes.json for the Gemma FFN generator from a captured oracle dir.

Uses the ACTUAL weight tensor shapes (authoritative), not config.intermediate_size -- Gemma-4 E2B's
config says 6144 but the real gate_proj is [12288, 1536] (per-layer MoE-adjacent sizing). The generator
(Task 3) and the movement model consume this.
"""
import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", required=True)
    ap.add_argument("--out", default=None, help="default: <oracle>/../shapes.json")
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.oracle, "meta.json")))
    gate = np.load(os.path.join(a.oracle, "weights/gate_proj.npy"))  # [I, D]
    down = np.load(os.path.join(a.oracle, "weights/down_proj.npy"))  # [D, I]
    I, D = int(gate.shape[0]), int(gate.shape[1])
    assert down.shape == (D, I), f"down_proj {down.shape} != ({D},{I})"
    shapes = {
        "model": meta["model"], "d_model": D, "intermediate": I,
        "act": "gelu_tanh", "gated": True, "norm": "rmsnorm_f32_ssq_effective_gamma",
        "rms_norm_eps": meta["rms_norm_eps"], "norm_convention": meta.get("norm_convention"),
        "gate_proj": [I, D], "up_proj": [I, D], "down_proj": [D, I],
    }
    out = a.out or os.path.join(os.path.dirname(a.oracle.rstrip("/")), "shapes.json")
    json.dump(shapes, open(out, "w"), indent=2)
    print(f"[shapes] {meta['model']}: D={D} I={I} gated GeGLU gelu_tanh -> {out}")


if __name__ == "__main__":
    main()
