#!/usr/bin/env python3
"""Regenerate the ESPCN image-gate fixtures that verify_upscaler_espcn_image.py needs.

WHY THIS EXISTS. That gate loads patch_y.npy and patch_cpu_sr_y.npy from
artifacts/upscaler/wholenet/. Nothing in the tree ever wrote them -- they were prepped in a session
scratchpad whose temp dir is gone -- so the gate has been unrunnable while still being named like a
gate. A gitignored input has to be recreatable from the tree; this is that recreation.

The patch is a REAL image patch (a 32x32 luma crop of artifacts/edsr/demo/lr.png), not synthetic
noise, because the point of that gate is a tangible whole-net-on-silicon result. The crop offset is
fixed so the fixture is deterministic.

The CPU reference is a plain float32 numpy forward pass of the SAME committed ESPCN weights
(artifacts/espcn/conv{1..4}_{w,b}.npy) plus the CRD pixel shuffle -- i.e. the same arithmetic the
gate's own conv/pshuf helpers do, minus the bf16 rounding. It deliberately does NOT need onnxruntime,
which is not installed in .venv-iron. Cross-checked against the committed onnxruntime output: running
this same forward on gate_lr.npy reproduces gate_sr.npy, so the reference is the real thing rather
than a self-consistent tautology.

    .venv-iron/bin/python scripts/make_espcn_gate_fixtures.py
"""
import numpy as np
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ESPCN = ROOT / "artifacts/espcn"
OUT = ROOT / "artifacts/upscaler/wholenet"
OUT.mkdir(parents=True, exist_ok=True)

W = {n: np.load(ESPCN / f"{n}.npy").astype(np.float32) for n in
     ("conv1_w", "conv1_b", "conv2_w", "conv2_b", "conv3_w", "conv3_b", "conv4_w", "conv4_b")}


def conv(x, w, b, k, pad, relu):
    """[Cin,H,W] -> [Cout,H,W], same-pad, float32."""
    cin, h, wd = x.shape
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    cout = w.shape[0]
    out = np.empty((cout, h, wd), np.float32)
    for oy in range(h):
        for ox in range(wd):
            patch = xp[:, oy:oy + k, ox:ox + k].reshape(-1)
            out[:, oy, ox] = w.reshape(cout, -1) @ patch + b
    return np.maximum(out, 0.0) if relu else out


def pshuf(x, r):
    """[C*r^2,H,W] -> [C,H*r,W*r], CRD order (matches the gates' own helper)."""
    cr2, h, wd = x.shape
    c = cr2 // (r * r)
    return x.reshape(c, r, r, h, wd).transpose(0, 3, 1, 4, 2).reshape(c, h * r, wd * r)


def espcn(y):
    x = conv(y, W["conv1_w"], W["conv1_b"], 5, 2, True)
    x = conv(x, W["conv2_w"], W["conv2_b"], 3, 1, True)
    x = conv(x, W["conv3_w"], W["conv3_b"], 3, 1, True)
    x = conv(x, W["conv4_w"], W["conv4_b"], 3, 1, False)
    return pshuf(x, 3)


# --- prove the numpy forward IS the onnxruntime reference before trusting it as one ---
lr = np.load(ESPCN / "gate_lr.npy").astype(np.float32)      # [1,1,8,8]
sr_ort = np.load(ESPCN / "gate_sr.npy").astype(np.float32)  # [1,1,24,24]
sr_np = espcn(lr[0])[None]
err = np.linalg.norm((sr_np - sr_ort).ravel()) / np.linalg.norm(sr_ort.ravel())
print(f"numpy forward vs committed onnxruntime reference: rel-L2 {err:.3e}")
assert err < 1e-5, f"numpy ESPCN does not reproduce the ORT reference (rel-L2 {err:.3e})"

# --- the real 32x32 luma patch ---
SRC = ROOT / "artifacts/edsr/demo/lr.png"
assert SRC.exists(), f"missing {SRC}"
img = Image.open(SRC).convert("YCbCr")
y = np.asarray(img)[:, :, 0].astype(np.float32) / 255.0
oy, ox = 16, 16                       # fixed crop -> deterministic fixture
assert y.shape[0] >= oy + 32 and y.shape[1] >= ox + 32, f"{SRC} too small: {y.shape}"
patch = y[oy:oy + 32, ox:ox + 32][None]                     # [1,32,32]

np.save(OUT / "patch_y.npy", patch[None][0][None])          # [1,1,32,32] -> gate does [0]
np.save(OUT / "patch_cpu_sr_y.npy", espcn(patch)[None])     # [1,1,96,96]
for n in ("conv1_w", "conv1_b", "conv2_w", "conv2_b", "conv3_w", "conv3_b", "conv4_w", "conv4_b"):
    np.save(OUT / f"{n}.npy", W[n])
print(f"wrote {OUT}/patch_y.npy {patch.shape} and patch_cpu_sr_y.npy, plus conv1..4 weights")
