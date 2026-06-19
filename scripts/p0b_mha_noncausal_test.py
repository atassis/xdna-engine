#!/usr/bin/env python3
"""P0b stage-2: validate the non-causal MHA patch (MHA_NONCAUSAL) on-device vs a NON-causal golden.

Builds MHA(causal=False) at a small Whisper-ish shape (heads=1, S=448, d=64) and checks its output
against torch SDPA with is_causal=False (every query sees the whole KV). If this passes, the tethered
patch (patches/iron-mha-noncausal.patch) correctly disables causality and the operator is ready to wire
into the fused decode. Run from the IRON repo root with ironenv: `ironenv/bin/python <this>` (single-tenant).
"""
import numpy as np
import torch
from ml_dtypes import bfloat16

from iron.common import AIEContext
from iron.operators.mha.op import MHA
from iron.common.test_utils import run_test

HEADS, S, D = 1, 448, 64  # 448 = 7*64 (already a multiple of B_q=64)
PIPES, KVH = 1, 0          # num_KV_heads=0 -> standard MHA

torch.manual_seed(42)
Q = torch.rand(HEADS, S, D, dtype=torch.bfloat16) * 4
K = torch.rand(HEADS, S, D, dtype=torch.bfloat16) * 4
V = torch.rand(HEADS, S, D, dtype=torch.bfloat16) * 4
inv_scale = 1.0 / np.sqrt(D)
# NON-causal golden: every query attends the whole KV.
O = torch.nn.functional.scaled_dot_product_attention(
    Q.unsqueeze(0), K.unsqueeze(0), V.unsqueeze(0), dropout_p=0.0, is_causal=False, scale=inv_scale
).squeeze(0)

ctx = AIEContext()
op = MHA(num_heads=HEADS, seq_len=S, d=D, num_KV_heads=KVH, num_of_pipelines=PIPES, causal=False, context=ctx)

# run_test accepts torch tensors directly (cf. operators/mha/test.py); keep them as torch bf16.
inb = {"Q": Q.flatten(), "K": K.flatten(), "V": V.flatten()}
outb = {"O": O.to(torch.bfloat16).flatten()}

errors, latency_us, bw = run_test(op, inb, outb, rel_tol=4.0e-2, abs_tol=1.5e-1)
n_err = len(errors["O"])
max_ok = int(S * D * HEADS * 0.005)
print(f"\n[P0b-noncausal] latency={latency_us:.1f}us  errors={n_err}/{max_ok}  "
      f"-> {'PASS' if n_err <= max_ok else 'FAIL'}")
