#!/usr/bin/env python3
"""Isolated weighted-RMSNorm device probe: build ONE RMSNorm(D, weighted) op, feed the exact gemma
layer-0 input (emb[BOS]*sqrt(D)) + weight (1+w), run on NPU, compare to host. No fused aliasing."""
import os, sys, numpy as np, ml_dtypes
import newstack_compat  # noqa
from iron.common import AIEContext
from iron.common.sequence import OperatorSequence
from iron.operators.rms_norm.op import RMSNorm
bf16 = ml_dtypes.bfloat16
D = 640; W = sys.argv[1]
ctx = AIEContext()
op = RMSNorm(size=D, num_aie_columns=1, num_channels=1, tile_size=D, weighted=True, context=ctx)
seq = OperatorSequence("rmsprobe", [(op, "x", "w", "out")],
                       input_args=["x"], output_args=["out"],   # w is a SCRATCH weight (like gemma L0_n_in)
                       buffer_sizes={"x": D*2, "w": D*2, "out": D*2}, dispatch="fused", context=ctx)
seq.compile()
c = seq.get_callable()
emb = np.load(f"{W}/model.embed_tokens.weight.npy").astype(np.float32)
w = np.load(f"{W}/model.layers.0.input_layernorm.weight.npy").astype(np.float32)
x0 = np.asarray(emb[2] * np.sqrt(D), bf16)              # device input (bf16), tok=BOS=2
np.copyto(c.get_buffer("w").data, np.asarray(1.0 + w, bf16))   # scratch weight
c.scratch_buffer.to("npu")                                     # sync scratch (weight) once
np.copyto(c.get_buffer("x").data, x0)
c()
out = c.get_buffer("out").data[:D].astype(np.float32)
x0f = x0.astype(np.float32)
hn_ideal = (x0f / np.sqrt(np.mean(x0f*x0f) + 1e-6)) * (1.0 + w)
b2 = lambda a: np.asarray(a, bf16).astype(np.float32)
inv = 1.0/np.sqrt(np.sum(x0f*x0f)/D + 1e-5); repro = b2(b2(x0f*b2(inv)) * b2(1.0+w))
rel = lambda a, g: float(np.linalg.norm(a-g)/np.linalg.norm(g))
print(f"[rmsprobe] |out|={np.linalg.norm(out):.3f} |ideal|={np.linalg.norm(hn_ideal):.3f} |repro|={np.linalg.norm(repro):.3f}")
print(f"[rmsprobe] device vs ideal(eps1e-6): rel-L2 = {rel(out, hn_ideal):.4f}")
print(f"[rmsprobe] device vs faithful-repro: rel-L2 = {rel(out, repro):.4f}")
print(f"[rmsprobe] out[:6]  = {np.round(out[:6],3)}")
print(f"[rmsprobe] ideal[:6]= {np.round(hn_ideal[:6],3)}")
