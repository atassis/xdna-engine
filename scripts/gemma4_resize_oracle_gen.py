#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Oracle-side stage capture for Gemma-4-12B's vision tower WITH a real (non-no-op) resize.

`gemma4_towers_oracle_gen.py` deliberately picks an image size where
`get_aspect_ratio_preserving_size` is an exact no-op, to keep the torchvision resize kernel out of
its traced path (see that script's docstring). This script is the resize-focused sibling:
same weight-wiring and tower math, but a LARGER image that the aspect-ratio-preserving resize must
actually shrink, so `tvF.resize(..., antialias=True)`'s output is a genuine parity target for
`gemma4_media.rs::resize_bicubic_antialias`.

MUST run under the gemma4-oracle venv (transformers 5.17.0 + torch + torchvision).

  python scripts/gemma4_resize_oracle_gen.py \\
      --checkpoint-dir $XDNA_ARTIFACTS/gemma4-12b-qat/checkpoint \\
      --image-size 1200x1600 --out <dir>
"""
import argparse
import json
import os

import numpy as np
import torch
from PIL import Image


def save(out, name, t):
    np.save(os.path.join(out, f"{name}.npy"), t.detach().to(torch.float32).numpy())
    print(f"  {name}: {tuple(t.shape)} {t.dtype}")


def make_test_image(h, w, seed=20260918):
    """Same deterministic structured-image generator as gemma4_towers_oracle_gen.py, parameterized
    to any size (that script's version is only ever called at its one no-op size)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.zeros((h, w, 3), dtype=np.float32)
    img[..., 0] = 255 * (xx / w)
    img[..., 1] = 255 * (yy / h)
    img[..., 2] = 255 * (((xx // 48) + (yy // 48)) % 2)
    blocks = rng.integers(0, 256, size=(h // 48 + 1, w // 48 + 1, 3)).astype(np.float32)
    img += np.kron(blocks, np.ones((48, 48, 1), dtype=np.float32))[:h, :w] * 0.25
    diag = 80 * np.exp(-((xx - yy * (w / h)) ** 2) / (2 * 60.0 ** 2))
    img += diag[..., None]
    img = np.clip(img, 0, 255).astype(np.uint8)
    return Image.fromarray(img, mode="RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--image-size", default="1200x1600", help="HxW; must NOT be a fixed point of "
                     "get_aspect_ratio_preserving_size (the script asserts this and fails loud).")
    ap.add_argument("--real-image", default=None, help="path to a real photo instead of the "
                     "synthetic pattern (PNG/JPEG); still resized to whatever the budget picks.")
    ap.add_argument("--audio-wav", default=None, help="16kHz mono wav; if given, also runs the "
                     "audio tower and saves its stages for the Rust gate.")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from transformers import AutoConfig
    from transformers.models.gemma4_unified.modeling_gemma4_unified import Gemma4UnifiedVisionEmbedder
    from transformers.models.gemma4_unified.image_processing_gemma4_unified import (
        Gemma4UnifiedImageProcessor, convert_image_to_patches, patches_merge, pad_along_first_dim,
        get_aspect_ratio_preserving_size)
    from safetensors import safe_open

    cfg = AutoConfig.from_pretrained(a.checkpoint_dir)
    vc, tc = cfg.vision_config, cfg.text_config

    vis = Gemma4UnifiedVisionEmbedder(vc, tc).float().eval()
    path = os.path.join(a.checkpoint_dir, "model.safetensors")
    with safe_open(path, framework="pt") as f:
        def get(k):
            return f.get_tensor(k).float()
        vis.patch_ln1.weight.data = get("model.vision_embedder.patch_ln1.weight")
        vis.patch_ln1.bias.data = get("model.vision_embedder.patch_ln1.bias")
        vis.patch_dense.weight.data = get("model.vision_embedder.patch_dense.weight")
        vis.patch_dense.bias.data = get("model.vision_embedder.patch_dense.bias")
        vis.patch_ln2.weight.data = get("model.vision_embedder.patch_ln2.weight")
        vis.patch_ln2.bias.data = get("model.vision_embedder.patch_ln2.bias")
        vis.pos_embedding.data = get("model.vision_embedder.pos_embedding")
        vis.pos_norm.weight.data = get("model.vision_embedder.pos_norm.weight")
        vis.pos_norm.bias.data = get("model.vision_embedder.pos_norm.bias")
        vis.multimodal_embedder.embedding_projection.weight.data = get(
            "model.embed_vision.embedding_projection.weight")

    processor = Gemma4UnifiedImageProcessor()
    if a.real_image:
        image = Image.open(a.real_image)
    else:
        img_h, img_w = (int(v) for v in a.image_size.split("x"))
        image = make_test_image(img_h, img_w)
    tensor_img = processor.process_image(image, do_convert_rgb=True)  # uint8 (3,H,W)
    img_h, img_w = tensor_img.shape[-2], tensor_img.shape[-1]
    save(a.out, "prep_image_input", tensor_img)
    # Real encoded bytes too, so the Rust gate exercises PNG decode -> resize end to end instead
    # of starting from an already-decoded tensor.
    image.convert("RGB").save(os.path.join(a.out, "input_image.png"))

    num_soft_tokens = processor.max_soft_tokens
    max_patches = num_soft_tokens * vc.pooling_kernel_size ** 2
    th, tw = get_aspect_ratio_preserving_size(img_h, img_w, vc.patch_size, max_patches,
                                              vc.pooling_kernel_size)
    is_noop = (th, tw) == (img_h, img_w)
    print(f"  input {img_h}x{img_w} -> resize target {th}x{tw} (no-op: {is_noop})")

    resized = processor.aspect_ratio_preserving_resize(
        tensor_img, patch_size=vc.patch_size, max_patches=max_patches,
        pooling_kernel_size=vc.pooling_kernel_size, resample=processor.resample)
    save(a.out, "prep_image_resized", resized)  # the ONE tensor this gate exists to check

    rescaled = processor.rescale_and_normalize(resized, do_rescale=True, rescale_factor=1 / 255,
                                                do_normalize=False, image_mean=None, image_std=None)
    save(a.out, "prep_image_rescaled", rescaled)

    patch_h, patch_w = rescaled.shape[-2] // vc.patch_size, rescaled.shape[-1] // vc.patch_size
    teacher_patches = convert_image_to_patches(rescaled, vc.patch_size)
    grid = torch.meshgrid(torch.arange(patch_w), torch.arange(patch_h), indexing="xy")
    teacher_positions = torch.stack(grid, dim=-1).reshape(teacher_patches.shape[0], 2)

    num_model_patches = teacher_patches.shape[0] // (vc.pooling_kernel_size ** 2)
    merged_patches, merged_positions = patches_merge(
        teacher_patches.unsqueeze(0), teacher_positions.unsqueeze(0), num_model_patches)
    merged_patches, merged_positions = merged_patches.squeeze(0), merged_positions.squeeze(0)
    n_real = merged_patches.shape[0]
    merged_patches, merged_positions = pad_along_first_dim(
        merged_patches, merged_positions, num_soft_tokens)
    print(f"  soft tokens: {n_real} real + {num_soft_tokens - n_real} padded = {merged_patches.shape[0]}")
    save(a.out, "prep_padded_patches", merged_patches)
    save(a.out, "prep_padded_positions", merged_positions.float())

    pixel_values = merged_patches.unsqueeze(0)
    image_position_ids = merged_positions.unsqueeze(0)

    official = processor.preprocess(images=[image], return_tensors="pt")
    off_err = (official["pixel_values"] - pixel_values).abs().max().item()
    print(f"  manual-pipeline vs processor.preprocess(): pixel max-abs-diff={off_err:.3e} (expect 0.0)")

    full = vis(pixel_values, image_position_ids)
    save(a.out, "vis_s7_projection", full.pooler_output)

    audio_tokens = None
    if a.audio_wav:
        from transformers.models.gemma4_unified.modeling_gemma4_unified import Gemma4UnifiedMultimodalEmbedder
        import wave
        ac = cfg.audio_config
        aud = Gemma4UnifiedMultimodalEmbedder(ac, tc).float().eval()
        with safe_open(path, framework="pt") as f:
            aud.embedding_projection.weight.data = f.get_tensor(
                "model.embed_audio.embedding_projection.weight").float()
        with wave.open(a.audio_wav) as wf:
            assert wf.getframerate() == 16000 and wf.getnchannels() == 1, wf.getparams()
            raw = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
        from transformers.models.gemma4_unified.feature_extraction_gemma4_unified import (
            Gemma4UnifiedAudioFeatureExtractor)
        feats = Gemma4UnifiedAudioFeatureExtractor()(raw, return_tensors="pt")
        input_features = feats["input_features"]
        audio_out = aud(inputs_embeds=input_features)
        save(a.out, "aud_projection", audio_out)
        audio_tokens = int(input_features.shape[1])
        print(f"  audio: {raw.shape[0]} samples -> {audio_tokens} tokens")

    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump({"image_hw_in": [img_h, img_w], "image_hw_resized": [th, tw],
                    "num_soft_tokens": num_soft_tokens, "n_real_soft_tokens": n_real,
                    "audio_tokens": audio_tokens}, f, indent=2)


if __name__ == "__main__":
    main()
