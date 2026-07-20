#!/usr/bin/env python3
"""EDSR ship-net before/after demo on a real image (kodim23): HR -> bicubic /r -> LR -> {bicubic, EDSR}
-> Y-PSNR/SSIM vs HR + saved before/after PNGs. The NPU frontier reproduces EDSR at rel-L2 ~3e-3, so this
is faithful to what the xdna-sr product outputs on the NPU. Run with the super_image venv."""
import argparse, numpy as np, torch
from pathlib import Path
from PIL import Image
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from super_image import EdsrModel

ap = argparse.ArgumentParser(); ap.add_argument("--scale", type=int, default=3)
ap.add_argument("--hr", default="artifacts/upscaler/test_hr.png")
A = ap.parse_args(); r = A.scale
OUT = Path("artifacts/edsr/demo"); OUT.mkdir(parents=True, exist_ok=True)
m = EdsrModel.from_pretrained("eugenesiow/edsr-base", scale=r).eval()

hr = Image.open(A.hr).convert("RGB"); W, H = hr.size; W, H = (W // r) * r, (H // r) * r
hr = hr.crop((0, 0, W, H)); lr = hr.resize((W // r, H // r), Image.BICUBIC)
with torch.no_grad():
    lr_t = torch.from_numpy(np.asarray(lr, np.float32).transpose(2, 0, 1)[None] / 255.0)
    edsr = (m(lr_t).clamp(0, 1)[0].numpy().transpose(1, 2, 0) * 255).round().clip(0, 255).astype(np.uint8)
bic = np.asarray(lr.resize((W, H), Image.BICUBIC))
hr_np = np.asarray(hr)

def y(im): return np.asarray(Image.fromarray(im).convert("YCbCr").split()[0], np.float64)
ep, bp = psnr(y(hr_np), y(edsr), data_range=255), psnr(y(hr_np), y(bic), data_range=255)
es, bs = ssim(y(hr_np), y(edsr), data_range=255), ssim(y(hr_np), y(bic), data_range=255)
Image.fromarray(hr_np).save(OUT / "hr.png"); Image.fromarray(bic).save(OUT / "bicubic.png")
Image.fromarray(edsr).save(OUT / "edsr.png"); lr.save(OUT / "lr.png")
print(f"kodim23 x{r} {W}x{H}: EDSR PSNR {ep:.2f} SSIM {es:.4f} | bicubic {bp:.2f} {bs:.4f} | "
      f"gain +{ep - bp:.2f} dB. Saved before/after to {OUT}")
