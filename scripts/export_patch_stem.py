#!/usr/bin/env python3
"""One-time fixture generator for the conv2d-kit patch-embed verifier.

For each ViT config: extract the REAL pretrained patch_embed Conv2d (weight+bias) via `transformers`,
build a tiny single-Conv2d ONNX (the INDEPENDENT oracle the Rust verifier runs via onnxruntime), and
dump a fixed-seed input + the weight/bias as .npy so the Rust side never needs torch/network at run
time.

Configs: vit_b16 (K=768,N=768 clean), vit_l16 (N=1024 -> N-pad), dinov2_b14 (K=588 -> K-pad).

Falls back to a fixed-seed random Conv2d if the checkpoint is unavailable (no network / model gated).
The lowering math is identical either way; only the 'real weights' provenance is lost — the script
prints real_weights=False so the run log can note the fallback.

Run from the worktree root with the export venv that has torch+transformers:
  $REPO/.venv/bin/python scripts/export_patch_stem.py
"""
import os
import numpy as np
import torch
import torch.nn as nn

OUT = "artifacts/patch_embed"
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(0)
np.random.seed(0)

# (name, hf_model, cin, img, patch, embed)
CONFIGS = [
    ("vit_b16",    "google/vit-base-patch16-224",  3, 224, 16, 768),
    ("vit_l16",    "google/vit-large-patch16-224", 3, 224, 16, 1024),
    ("dinov2_b14", "facebook/dinov2-base",         3, 224, 14, 768),
]


def get_proj(hf_model, cin, patch, embed):
    """Return (weight[embed,cin,patch,patch], bias[embed], real?) for the patch-embed Conv2d."""
    try:
        from transformers import AutoModel
        m = AutoModel.from_pretrained(hf_model)
        # ViT and DINOv2 both expose embeddings.patch_embeddings.projection (nn.Conv2d)
        proj = m.embeddings.patch_embeddings.projection
        assert isinstance(proj, nn.Conv2d), type(proj)
        assert proj.weight.shape == (embed, cin, patch, patch), proj.weight.shape
        return proj.weight.detach().float(), proj.bias.detach().float(), True
    except Exception as e:  # noqa: BLE001
        print(f"  [fallback] checkpoint unavailable ({e}); using seeded random weights")
        w = torch.randn(embed, cin, patch, patch) * 0.02
        b = torch.randn(embed) * 0.02
        return w, b, False


for name, hf_model, cin, img, patch, embed in CONFIGS:
    w, b, real = get_proj(hf_model, cin, patch, embed)
    conv = nn.Conv2d(cin, embed, kernel_size=patch, stride=patch, bias=True)
    with torch.no_grad():
        conv.weight.copy_(w)
        conv.bias.copy_(b)
    x = torch.randn(1, cin, img, img)
    onnx_path = f"{OUT}/{name}.onnx"
    torch.onnx.export(
        conv, x, onnx_path,
        input_names=["input"], output_names=["patch_conv"],
        opset_version=17, dynamo=False,
    )
    np.save(f"{OUT}/input_{name}.npy", x[0].numpy().astype(np.float32))   # [Cin,H,W]
    np.save(f"{OUT}/weight_{name}.npy", w.numpy().astype(np.float32))     # [embed,Cin,ph,pw]
    np.save(f"{OUT}/bias_{name}.npy", b.numpy().astype(np.float32))       # [embed]
    print(f"{name}: real_weights={real} embed={embed} patch={patch} -> {onnx_path}")
