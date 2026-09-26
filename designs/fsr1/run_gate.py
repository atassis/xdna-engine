#!/usr/bin/env python3
"""Regenerate test frames, run the GPU ground-truth harness, gate cpu_ref.py against it.
Run `gpu_harness/build.sh` once first (needs glslangValidator + a Vulkan ICD)."""
import subprocess
import sys
import numpy as np

sys.path.insert(0, ".")
from cpu_ref import fsr1_x3, fsr_easu

H, W = 360, 640


def make_frame(kind, rng):
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.zeros((H, W, 3), dtype=np.float32)
    if kind == "flat":
        img[:] = 128
    elif kind == "smooth":
        img[..., 0] = xx / W * 255
        img[..., 1] = yy / H * 255
        img[..., 2] = 128 + 64 * np.sin(xx / 40.0)
    elif kind == "noisy":
        img[..., 0] = xx / W * 255
        img[..., 1] = yy / H * 255
        img[..., 2] = 128 + 64 * np.sin(xx / 20.0) + 64 * np.cos(yy / 15.0)
        img += rng.normal(0, 8, size=img.shape)
        img[H // 3 : H // 3 + 20, :, :] = 250
        img[:, W // 4 : W // 4 + 3, :] = 0
    return np.clip(img, 0, 255).astype(np.uint8)


def gate(kind, rng):
    img = make_frame(kind, rng)
    rgba = np.dstack([img, np.full((H, W), 255, np.uint8)])
    rgba.tofile(f"gpu_harness/in_{kind}.rgba8")
    subprocess.run(
        ["./harness", str(W), str(H), f"in_{kind}.rgba8", f"out_{kind}.rgba8"],
        cwd="gpu_harness", check=True,
    )
    gpu_out = np.fromfile(f"gpu_harness/out_{kind}.rgba8", dtype=np.uint8).reshape(H * 3, W * 3, 4)[..., :3]
    cpu_out = fsr1_x3(img)
    d = np.abs(cpu_out.astype(np.int32) - gpu_out.astype(np.int32))
    print(f"{kind:8s} max_abs={d.max():4d} mean_abs={d.mean():.4f} pct_gt2={100*(d>2).mean():.3f}")


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    for kind in ["flat", "smooth", "noisy"]:
        gate(kind, rng)
