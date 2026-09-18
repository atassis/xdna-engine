#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Oracle-side stage capture for Gemma-4-12B's vision + audio towers.

MUST run under the gemma4-oracle venv (transformers 5.17.0 + torch + torchvision --
scripts/setup_gemma4_oracle_venv.sh; torchvision was added to that venv this session, see the
report). It instantiates ONLY the vision/audio submodules (not the 22GB text stack) and wires
their weights DIRECTLY from the checkpoint's real tensor names, because
`Gemma4UnifiedForConditionalGeneration.from_pretrained`'s own key naming for `embed_vision`
(`Gemma4UnifiedVisionEmbedder`, so `embed_vision.patch_ln1.*` etc.) does NOT match this
checkpoint: the checkpoint has `vision_embedder.*` as a SIBLING of `embed_vision`, and
`embed_vision` itself holds only `embedding_projection.weight`. Loading this checkpoint through
`from_pretrained` at transformers 5.17.0 would silently report all 11 tower tensors as
"unexpected keys" and leave `embed_vision`/`embed_audio`/`vision_embedder`'s own params at random
init -- verified by `--check-from-pretrained-keys` below, which does the load and prints exactly
that missing/unexpected split without materializing the 22GB text stack.

Picks a 672x960 synthetic test image sized so `get_aspect_ratio_preserving_size` is an EXACT no-op
(720*896 would not be; 672x960 gives max_patches=2520=42*60 exactly, factor==1.0) -- this keeps the
generic torchvision resize kernel out of the traced path entirely, so the patchify/merge parity
check that gemma4_towers_host_ref.py does is not confounded by an interpolation kernel neither
script tries to reproduce. See the report for why resize is out of scope for this rung.

Writes every stage of both forward passes (the real nn.Module ops, called individually rather than
via .forward() so intermediates are observable) plus the real image/audio preprocessing tensors to
<out>/*.npy.
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


def make_test_image(h=672, w=960, seed=20260918):
    """Deterministic synthetic image with real spatial structure (gradients + blocks + a
    diagonal), not i.i.d. noise -- a transposed patchify would misplace visibly-structured
    content, unlike with a uniform-random image. 672x960 makes the aspect-preserving resize an
    exact no-op (see module docstring), so this is a genuine parity target for patchify/merge with
    no interpolation kernel involved."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.zeros((h, w, 3), dtype=np.float32)
    img[..., 0] = 255 * (xx / w)
    img[..., 1] = 255 * (yy / h)
    img[..., 2] = 255 * (((xx // 48) + (yy // 48)) % 2)
    blocks = rng.integers(0, 256, size=(h // 48, w // 48, 3)).astype(np.float32)
    img += np.kron(blocks, np.ones((48, 48, 1), dtype=np.float32)) * 0.25
    diag = 80 * np.exp(-((xx - yy * (w / h)) ** 2) / (2 * 60.0 ** 2))
    img += diag[..., None]
    img = np.clip(img, 0, 255).astype(np.uint8)
    return Image.fromarray(img, mode="RGB")


def check_from_pretrained_keys(ckpt_dir):
    """Load JUST the two tower submodules the way Gemma4UnifiedModel.__init__ wires them
    (`self.embed_vision = Gemma4UnifiedVisionEmbedder(...)`, `self.embed_audio =
    Gemma4UnifiedMultimodalEmbedder(...)`) -- WITHOUT the 22GB text stack a real
    Gemma4UnifiedModel(cfg) would also build (AutoModel.from_config alone materializes a
    262144x3840 embed_tokens) -- and print the missing/unexpected key split against the
    checkpoint's real tower keys, to make the naming-mismatch claim above a measurement."""
    import torch.nn as nn
    from transformers import AutoConfig
    from transformers.models.gemma4_unified.modeling_gemma4_unified import (
        Gemma4UnifiedVisionEmbedder, Gemma4UnifiedMultimodalEmbedder)

    cfg = AutoConfig.from_pretrained(ckpt_dir)
    shim = nn.Module()
    shim.embed_vision = Gemma4UnifiedVisionEmbedder(cfg.vision_config, cfg.text_config)
    shim.embed_audio = Gemma4UnifiedMultimodalEmbedder(cfg.audio_config, cfg.text_config)

    from safetensors import safe_open
    tower_sd = {}
    with safe_open(os.path.join(ckpt_dir, "model.safetensors"), framework="pt") as f:
        for k in f.keys():
            if "vision_embedder" in k or "embed_vision" in k or "embed_audio" in k:
                tower_sd[k[len("model."):]] = f.get_tensor(k)
    missing, unexpected = shim.load_state_dict(tower_sd, strict=False)
    print(f"from_pretrained-style load of embed_vision/embed_audio as the LIBRARY nests them: "
          f"{len(missing)} of the model's OWN tower params never matched a checkpoint key, "
          f"{len(unexpected)} checkpoint tower keys were never consumed (of {len(tower_sd)} given).")
    print("  missing (would silently stay random-init):", missing)
    print("  unexpected (checkpoint keys with nowhere to land):", unexpected)
    return len(missing), len(unexpected)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--audio-wav", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--check-from-pretrained-keys", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    if a.check_from_pretrained_keys:
        check_from_pretrained_keys(a.checkpoint_dir)

    from transformers import AutoConfig
    from transformers.models.gemma4_unified.modeling_gemma4_unified import (
        Gemma4UnifiedVisionEmbedder, Gemma4UnifiedMultimodalEmbedder)
    from transformers.models.gemma4_unified.image_processing_gemma4_unified import (
        Gemma4UnifiedImageProcessor, convert_image_to_patches, patches_merge, pad_along_first_dim)
    from transformers.models.gemma4_unified.feature_extraction_gemma4_unified import (
        Gemma4UnifiedAudioFeatureExtractor)
    from safetensors import safe_open

    cfg = AutoConfig.from_pretrained(a.checkpoint_dir)
    vc, ac, tc = cfg.vision_config, cfg.audio_config, cfg.text_config

    # ---- wire the towers from the checkpoint's REAL key names (not from_pretrained's) ----
    vis = Gemma4UnifiedVisionEmbedder(vc, tc).float().eval()
    aud = Gemma4UnifiedMultimodalEmbedder(ac, tc).float().eval()
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
        aud.embedding_projection.weight.data = get("model.embed_audio.embedding_projection.weight")

    # ================= VISION =================
    print("== vision preprocessing ==")
    image = make_test_image()
    processor = Gemma4UnifiedImageProcessor()
    tensor_img = processor.process_image(image, do_convert_rgb=True)  # uint8 (3,H,W)
    assert tuple(tensor_img.shape) == (3, 672, 960), tensor_img.shape
    max_patches = vc.num_soft_tokens * vc.pooling_kernel_size ** 2
    from transformers.models.gemma4_unified.image_processing_gemma4_unified import (
        get_aspect_ratio_preserving_size)
    th, tw = get_aspect_ratio_preserving_size(672, 960, vc.patch_size, max_patches, vc.pooling_kernel_size)
    assert (th, tw) == (672, 960), f"resize would NOT be a no-op: {(th, tw)}"  # confirms the docstring's claim

    rescaled = processor.rescale_and_normalize(tensor_img, do_rescale=True, rescale_factor=1 / 255,
                                                do_normalize=False, image_mean=None, image_std=None)
    save(a.out, "prep_image_rescaled", rescaled)

    patch_h, patch_w = rescaled.shape[-2] // vc.patch_size, rescaled.shape[-1] // vc.patch_size
    teacher_patches = convert_image_to_patches(rescaled, vc.patch_size)
    grid = torch.meshgrid(torch.arange(patch_w), torch.arange(patch_h), indexing="xy")
    teacher_positions = torch.stack(grid, dim=-1).reshape(teacher_patches.shape[0], 2)
    save(a.out, "prep_teacher_patches", teacher_patches)
    save(a.out, "prep_teacher_positions", teacher_positions.float())

    num_model_patches = teacher_patches.shape[0] // (vc.pooling_kernel_size ** 2)
    merged_patches, merged_positions = patches_merge(
        teacher_patches.unsqueeze(0), teacher_positions.unsqueeze(0), num_model_patches)
    merged_patches, merged_positions = merged_patches.squeeze(0), merged_positions.squeeze(0)
    assert merged_patches.shape[0] == vc.num_soft_tokens, \
        f"expected exactly {vc.num_soft_tokens} soft tokens with no padding, got {merged_patches.shape[0]}"
    save(a.out, "prep_merged_patches", merged_patches)
    save(a.out, "prep_merged_positions", merged_positions.float())

    pixel_values = merged_patches.unsqueeze(0)          # (1, 280, 6912)
    image_position_ids = merged_positions.unsqueeze(0)  # (1, 280, 2)

    # Cross-check against the processor's own black-box preprocess() entry point.
    official = processor.preprocess(images=[image], return_tensors="pt")
    off_err = (official["pixel_values"] - pixel_values).abs().max().item()
    off_pos_err = (official["image_position_ids"] - image_position_ids).abs().max().item()
    print(f"  manual-pipeline vs processor.preprocess(): pixel max-abs-diff={off_err:.3e}, "
          f"position_ids max-abs-diff={off_pos_err:.3e} (expect 0.0, same code path either way)")

    save(a.out, "in_pixel_values", pixel_values)
    save(a.out, "in_image_position_ids", image_position_ids.float())

    print("== vision tower forward (staged) ==")
    if (target_dtype := vis.patch_dense.weight.dtype).is_floating_point:
        pv = pixel_values.to(target_dtype)
    s1 = vis.patch_ln1(pv)
    save(a.out, "vis_s1_patch_ln1", s1)
    s2 = vis.patch_dense(s1)
    save(a.out, "vis_s2_patch_dense", s2)
    s3 = vis.patch_ln2(s2)
    save(a.out, "vis_s3_patch_ln2", s3)

    clamped = image_position_ids.clamp(min=0).long()
    valid = (image_position_ids != -1).to(vis.pos_embedding.dtype).unsqueeze(-1)
    axes = torch.arange(2)
    pos_embs = (vis.pos_embedding[clamped, axes] * valid).sum(-2)
    save(a.out, "vis_s4_pos_embs", pos_embs)
    s4 = s3 + pos_embs
    save(a.out, "vis_s4_hidden_plus_pos", s4)
    s5 = vis.pos_norm(s4)
    save(a.out, "vis_s5_pos_norm", s5)  # == vision_outputs.last_hidden_state

    s6 = vis.multimodal_embedder.embedding_pre_projection_norm(s5)
    save(a.out, "vis_s6_rmsnorm", s6)
    s7 = vis.multimodal_embedder.embedding_projection(s6)
    save(a.out, "vis_s7_projection", s7)  # == vision_outputs.pooler_output, the final feature

    # Full-module forward() as one more cross-check the staging above didn't drop a step.
    full = vis(pixel_values, image_position_ids)
    fe = (full.pooler_output - s7).abs().max().item()
    print(f"  staged-vs-forward() max-abs-diff: {fe:.3e} (expect 0.0)")

    # ================= AUDIO =================
    print("== audio preprocessing + tower forward (staged) ==")
    import wave
    with wave.open(a.audio_wav) as wf:
        assert wf.getframerate() == 16000 and wf.getnchannels() == 1, wf.getparams()
        raw = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0

    fe_ = Gemma4UnifiedAudioFeatureExtractor()
    feats = fe_(raw, return_tensors="pt")
    input_features = feats["input_features"]
    input_features_mask = feats["input_features_mask"]
    print(f"  waveform {raw.shape[0]} samples -> {input_features.shape[1]} audio tokens "
          f"(exact multiple of 640: {raw.shape[0] % 640 == 0})")
    save(a.out, "in_input_features", input_features)
    save(a.out, "in_input_features_mask", input_features_mask.float())

    a1 = aud.embedding_pre_projection_norm(input_features)
    save(a.out, "aud_s1_rmsnorm", a1)
    a2 = aud.embedding_projection(a1)
    save(a.out, "aud_s2_projection", a2)

    full_a = aud(inputs_embeds=input_features)
    fae = (full_a - a2).abs().max().item()
    print(f"  staged-vs-forward() max-abs-diff: {fae:.3e} (expect 0.0)")

    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump({"image_hw": [672, 960], "num_soft_tokens": vc.num_soft_tokens,
                    "audio_tokens": int(input_features.shape[1]),
                    "audio_samples": int(raw.shape[0])}, f, indent=2)
    print(f"oracle stage dump written to {a.out}")


if __name__ == "__main__":
    main()
