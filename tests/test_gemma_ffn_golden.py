# SPDX-License-Identifier: Apache-2.0
"""Host gate: the numpy Gemma FFN sub-block golden matches the HF oracle.

Validates the reference MATH (RMSNorm f32 ssq *(1+gamma), gelu_tanh gated GeGLU, sandwich post-norm +
residual) before any NPU kernel work. Runs against whatever oracle dir GEMMA_FFN_ORACLE points at
(default the cached Gemma-3-270m capture); the same gate re-runs on the Gemma-4-E2B oracle once it lands.

Run: GEMMA_FFN_ORACLE=artifacts/gemma3-270m/ffn_oracle ~/gemma4-ref-venv/bin/python -m pytest \
        tests/test_gemma_ffn_golden.py -v
"""
import json
import os
import pathlib

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO))
from route_b_kernels.gemma_ffn.gen_golden import ffn_forward, rel_l2, corr

ORACLE = pathlib.Path(os.environ.get("GEMMA_FFN_ORACLE", REPO / "artifacts/gemma3-270m/ffn_oracle"))


def _load():
    meta = json.load(open(ORACLE / "meta.json"))
    x = np.load(ORACLE / "ffn_in.npy")
    ref = np.load(ORACLE / "ffn_out.npy")
    return meta, x, ref


@pytest.mark.skipif(not (ORACLE / "meta.json").exists(), reason=f"no oracle at {ORACLE}")
def test_fp32_golden_matches_oracle_formula():
    """fp32 golden must reproduce the HF sub-block to ~float precision -> the formula is exact."""
    meta, x, ref = _load()
    got = ffn_forward(x, str(ORACLE / "weights"), meta["rms_norm_eps"], compute_dtype="float32")
    r = rel_l2(got, ref)
    assert r <= 1e-4, f"fp32 formula diverges: rel_L2={r:.2e} (check RMSNorm (1+gamma)/f32-ssq or gelu_tanh)"


@pytest.mark.skipif(not (ORACLE / "meta.json").exists(), reason=f"no oracle at {ORACLE}")
def test_bf16_golden_within_kernel_gate():
    """bf16 golden (what the NPU kernel targets) must sit inside the codebase bar: rel_L2<=0.08, corr>=0.99."""
    meta, x, ref = _load()
    got = ffn_forward(x, str(ORACLE / "weights"), meta["rms_norm_eps"], compute_dtype="bf16")
    r, c = rel_l2(got, ref), corr(got, ref)
    assert r <= 0.08 and c >= 0.99, f"bf16 golden out of gate: rel_L2={r:.2e} corr={c:.6f}"
