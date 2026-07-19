#!/usr/bin/env python3
"""Oracle for the ESPCN arch converter + the npu-sr whole-net gate.

Dumps per-conv weights/biases as .npy under artifacts/espcn/ (flat names matching
rust/npu-weights/src/arch/espcn.rs: conv{1..4}_w / conv{1..4}_b), plus a small fixed LR input tile and
its CPU ESPCN SR output (gate_lr.npy / gate_sr.npy) for the npu-sr frontier parity gate.

Mirrors scripts/export_resnet.py. Run with any venv that has onnx + onnxruntime + numpy, cwd = repo root:
    <venv>/bin/python scripts/export_espcn.py
Requires artifacts/espcn/espcn_x3_dyn.onnx (the pretrained ESPCN, copied from the M1 asset)."""
import numpy as np, onnxruntime as ort, onnx
from pathlib import Path

OUT = Path("artifacts/espcn"); OUT.mkdir(parents=True, exist_ok=True)
MODEL = OUT / "espcn_x3_dyn.onnx"
assert MODEL.exists(), f"missing {MODEL} -- copy espcn_x3_dyn.onnx (M1 asset) into artifacts/espcn/ first"

# --- weights: pull conv initializers (4-D weight + 1-D bias), map conv1..conv4 in graph order ---
m = onnx.load(str(MODEL))
inits = [(i.name, onnx.numpy_helper.to_array(i)) for i in m.graph.initializer]
weights = [(n, a) for n, a in inits if a.ndim == 4]
biases  = [(n, a) for n, a in inits if a.ndim == 1]
assert len(weights) == 4 and len(biases) == 4, f"expected 4 conv+bias, got {len(weights)}/{len(biases)}"
for idx, ((wn, w), (bn, b)) in enumerate(zip(weights, biases), start=1):
    np.save(OUT / f"conv{idx}_w.npy", w.astype(np.float32))   # [Cout,Cin,kh,kw]
    np.save(OUT / f"conv{idx}_b.npy", b.astype(np.float32))   # [Cout]
    print(f"conv{idx}: {wn} w{tuple(w.shape)}  {bn} b{tuple(b.shape)}")

# --- whole-net CPU reference on a fixed LR tile (deterministic seed) for the engine gate ---
sess = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
rng = np.random.default_rng(0)
lr = rng.random((1, 1, 8, 8), dtype=np.float32)      # [1,1,H,W] luma in [0,1]
in_name = sess.get_inputs()[0].name
sr = sess.run(None, {in_name: lr})[0]                # [1,1,3H,3W]
np.save(OUT / "gate_lr.npy", lr)
np.save(OUT / "gate_sr.npy", sr)
print(f"gate: LR {tuple(lr.shape)} -> SR {tuple(sr.shape)}")
