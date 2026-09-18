#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Host-only (numpy, no torch, no device) forward pass for Gemma-4-12B's vision + audio towers.

Reads the flat .npy dump `dump_gemma4_towers.py` writes and independently recomputes every stage
of `Gemma4UnifiedVisionEmbedder.forward` / `Gemma4UnifiedMultimodalEmbedder.forward` in plain
numpy -- this is the math a host-side NPU dataflow glue would run, not a call into HF's own
nn.Module code. It also reimplements `convert_image_to_patches`/`patches_merge` from
image_processing_gemma4_unified.py in numpy, to test the ONE thing source-reading cannot settle:
whether this patchify matches theirs.

Compares every stage against gemma4_towers_oracle_gen.py's saved reference tensors (real
transformers 5.17.0 nn.Module ops) and reports rel-L2 = ||host-oracle||_2 / ||oracle||_2 per stage
-- localizing any mismatch to a stage instead of reporting "the tower is wrong".

  python scripts/gemma4_towers_host_ref.py \\
      --weights-dir artifacts/gemma4-12b/towers --oracle-dir <oracle out dir>
"""
import argparse
import math
import os
import wave

import numpy as np

GATE = 1e-4  # both sides compute in float32 from the SAME bf16-rounded checkpoint values (see
# report) -- no precision gap is expected, so this gates on WIRING/MATH correctness, not on a
# bf16 rounding floor. float32 matmul accumulation order differs (numpy BLAS vs torch ATen), so
# exact 0.0 is not expected either; 1e-4 is ~1000x the fp32 epsilon-scale noise this loop actually
# produces (see the printed numbers) and would not survive an actual transposed axis or a dropped
# term.


def rel_l2(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def layer_norm(x, weight, bias, eps=1e-5):
    """nn.LayerNorm's own math: biased (ddof=0) variance, eps default 1e-5 -- none of
    patch_ln1/patch_ln2/pos_norm pass a custom eps in the modeling code."""
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * weight + bias


def rms_norm_no_scale(x, eps=1e-6):
    """Gemma4UnifiedRMSNorm with with_scale=False (embedding_pre_projection_norm for both
    towers) -- rsqrt(mean(x^2)+eps)*x, no learnable gain at all."""
    ms = (x ** 2).mean(axis=-1, keepdims=True) + eps
    return x * ms ** -0.5


def linear(x, weight, bias=None):
    """HF nn.Linear convention: weight is [out, in]."""
    y = x @ weight.T
    return y + bias if bias is not None else y


def convert_image_to_patches(image, patch_size):
    """Port of image_processing_gemma4_unified.convert_image_to_patches. image: (C,H,W)."""
    c, h, w = image.shape
    nph, npw = h // patch_size, w // patch_size
    patched = image.reshape(c, nph, patch_size, npw, patch_size)
    patched = patched.transpose(1, 3, 2, 4, 0)  # (nph, npw, ps, ps, c) -- torch permute(1,3,2,4,0)
    return patched.reshape(nph * npw, -1)


def patches_merge(patches, positions_xy, length):
    """Port of image_processing_gemma4_unified.patches_merge. patches: (B,L,D) float,
    positions_xy: (B,L,2) int/float holding integers, no padding (-1) entries in this rung's test
    input -- the padding-preserving branch is untested here, noted as a gap in the report."""
    b = patches.shape[:-2]
    d = patches.shape[-1]
    patch_size = math.isqrt(d // 3)
    assert d == patch_size * patch_size * 3, d
    length_L = patches.shape[-2]
    k = math.isqrt(length_L // length)
    assert k * k * length == length_L

    pos = positions_xy.astype(np.int64)
    max_x = pos[..., 0].max(axis=-1, keepdims=True) + 1
    kernel_idxs = pos // k
    num_from_top_left = k * k * kernel_idxs[..., 0] + k * max_x * kernel_idxs[..., 1]
    within = pos % k
    num_from_top_left_of_kernel = within[..., 0] + within[..., 1] * k
    target_ordering = num_from_top_left_of_kernel + num_from_top_left  # (B, L)

    perm = np.argsort(target_ordering, axis=-1)  # (B, L); unique keys -> stability doesn't matter
    perm_patches = np.broadcast_to(perm[..., None], patches.shape)
    kop = np.take_along_axis(patches, perm_patches, axis=-2)

    kop = kop.reshape(*b, length, k, k, patch_size, patch_size, 3)
    kop = np.transpose(kop, tuple(range(len(b))) + (len(b), len(b) + 1, len(b) + 3, len(b) + 2,
                                                      len(b) + 4, len(b) + 5))
    merged_patches = kop.reshape(*b, length, k * patch_size * k * patch_size * 3)

    perm_pos = np.broadcast_to(perm[..., None], positions_xy.shape)
    kop_pos = np.take_along_axis(pos.astype(np.float64), perm_pos, axis=-2)
    padding = (pos == -1).all(axis=-1, keepdims=True)
    kop_pos = kop_pos * (~padding) + pos.astype(np.float64) * padding
    kop_pos = kop_pos.reshape(*b, length, k * k, 2)
    new_positions = np.floor(kop_pos / k)
    new_positions = new_positions.min(axis=-2).astype(np.int64)
    return merged_patches, new_positions


def extract_waveform_features(waveform, samples_per_token=640):
    """Port of Gemma4UnifiedAudioFeatureExtractor._extract_waveform_features: zero-pad to a
    multiple of samples_per_token, reshape to (num_tokens, samples_per_token). No windowing, no
    normalization -- a pure reshape, per the source."""
    pad_len = (-len(waveform)) % samples_per_token
    if pad_len:
        waveform = np.pad(waveform, (0, pad_len))
    num_tokens = len(waveform) // samples_per_token
    features = waveform.reshape(num_tokens, samples_per_token).astype(np.float32)
    mask = np.ones(num_tokens, dtype=bool)
    return features, mask


def load(d, name):
    return np.load(os.path.join(d, f"{name}.npy"))


def report(host_dir, stage, host_val, oracle_dir, oracle_name):
    oracle_val = load(oracle_dir, oracle_name)
    np.save(os.path.join(host_dir, f"{stage}.npy"), host_val)
    err = rel_l2(host_val, oracle_val)
    status = "PASS" if err <= GATE else "FAIL"
    print(f"  {status}  {stage:28s} vs {oracle_name:24s} rel-L2={err:.3e}  shape={host_val.shape}")
    return err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights-dir", required=True)
    ap.add_argument("--oracle-dir", required=True)
    ap.add_argument("--audio-wav", required=True,
                     help="the SAME wav gemma4_towers_oracle_gen.py was pointed at, to test the "
                          "waveform-framing numpy port end to end (not just the tower math).")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    W, O = a.weights_dir, a.oracle_dir

    patch_ln1_w = load(W, "model.vision_embedder.patch_ln1.weight")
    patch_ln1_b = load(W, "model.vision_embedder.patch_ln1.bias")
    patch_dense_w = load(W, "model.vision_embedder.patch_dense.weight")
    patch_dense_b = load(W, "model.vision_embedder.patch_dense.bias")
    patch_ln2_w = load(W, "model.vision_embedder.patch_ln2.weight")
    patch_ln2_b = load(W, "model.vision_embedder.patch_ln2.bias")
    pos_embedding = load(W, "model.vision_embedder.pos_embedding")
    pos_norm_w = load(W, "model.vision_embedder.pos_norm.weight")
    pos_norm_b = load(W, "model.vision_embedder.pos_norm.bias")
    embed_vision_proj = load(W, "model.embed_vision.embedding_projection.weight")
    embed_audio_proj = load(W, "model.embed_audio.embedding_projection.weight")

    errs = {}

    print("== preprocessing parity: numpy patchify/merge vs torch's, on the SAME rescaled image ==")
    rescaled = load(O, "prep_image_rescaled")  # (3,672,960), identical input both sides
    teacher_patches = convert_image_to_patches(rescaled, 16)
    errs["patchify"] = report(a.out, "prep_teacher_patches", teacher_patches, O, "prep_teacher_patches")

    patch_h, patch_w = rescaled.shape[-2] // 16, rescaled.shape[-1] // 16
    xx, yy = np.meshgrid(np.arange(patch_w), np.arange(patch_h))  # indexing="xy" to match torch
    teacher_positions = np.stack([xx, yy], axis=-1).reshape(teacher_patches.shape[0], 2)
    errs["patchify_pos"] = report(a.out, "prep_teacher_positions", teacher_positions.astype(np.float32),
                                   O, "prep_teacher_positions")

    num_model_patches = teacher_patches.shape[0] // 9
    merged_patches, merged_positions = patches_merge(
        teacher_patches[None], teacher_positions[None], num_model_patches)
    merged_patches, merged_positions = merged_patches[0], merged_positions[0]
    errs["merge"] = report(a.out, "prep_merged_patches", merged_patches, O, "prep_merged_patches")
    errs["merge_pos"] = report(a.out, "prep_merged_positions", merged_positions.astype(np.float32),
                                O, "prep_merged_positions")

    print("== vision tower, staged ==")
    pixel_values = load(O, "in_pixel_values")             # (1,280,6912) -- HF's own preprocessing
    image_position_ids = load(O, "in_image_position_ids")  # output, used as input here so tower
    # parity is isolated from patchify parity (checked separately above).

    s1 = layer_norm(pixel_values, patch_ln1_w, patch_ln1_b)
    errs["s1_patch_ln1"] = report(a.out, "vis_s1_patch_ln1", s1, O, "vis_s1_patch_ln1")
    s2 = linear(s1, patch_dense_w, patch_dense_b)
    errs["s2_patch_dense"] = report(a.out, "vis_s2_patch_dense", s2, O, "vis_s2_patch_dense")
    s3 = layer_norm(s2, patch_ln2_w, patch_ln2_b)
    errs["s3_patch_ln2"] = report(a.out, "vis_s3_patch_ln2", s3, O, "vis_s3_patch_ln2")

    clamped = np.clip(image_position_ids, 0, None).astype(np.int64)
    valid = (image_position_ids != -1).astype(pos_embedding.dtype)[..., None]
    axes = np.arange(2)
    gathered = pos_embedding[clamped, axes]  # (1,280,2,3840), numpy advanced indexing == torch's
    pos_embs = (gathered * valid).sum(axis=-2)
    errs["s4_pos_embs"] = report(a.out, "vis_s4_pos_embs", pos_embs, O, "vis_s4_pos_embs")
    s4 = s3 + pos_embs
    errs["s4_hidden_plus_pos"] = report(a.out, "vis_s4_hidden_plus_pos", s4, O, "vis_s4_hidden_plus_pos")
    s5 = layer_norm(s4, pos_norm_w, pos_norm_b)
    errs["s5_pos_norm"] = report(a.out, "vis_s5_pos_norm", s5, O, "vis_s5_pos_norm")

    s6 = rms_norm_no_scale(s5)
    errs["s6_rmsnorm"] = report(a.out, "vis_s6_rmsnorm", s6, O, "vis_s6_rmsnorm")
    s7 = linear(s6, embed_vision_proj)  # bias=False
    errs["s7_projection"] = report(a.out, "vis_s7_projection", s7, O, "vis_s7_projection")

    print("== audio preprocessing parity: numpy waveform framing vs the feature extractor's ==")
    with wave.open(a.audio_wav) as wf:
        assert wf.getframerate() == 16000 and wf.getnchannels() == 1, wf.getparams()
        raw = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    feats, mask = extract_waveform_features(raw)
    errs["audio_frames"] = report(a.out, "prep_audio_frames", feats[None], O, "in_input_features")
    oracle_mask = load(O, "in_input_features_mask")[0].astype(bool)
    mask_match = np.array_equal(mask, oracle_mask)
    print(f"  {'PASS' if mask_match else 'FAIL'}  audio_mask                   "
          f"numpy all-{mask.all()} len={len(mask)} vs oracle all-{oracle_mask.all()} len={len(oracle_mask)}")
    errs["audio_mask"] = 0.0 if mask_match else 1.0

    print("== audio tower, staged ==")
    input_features = load(O, "in_input_features")  # (1,233,640)
    a1 = rms_norm_no_scale(input_features)
    errs["a1_rmsnorm"] = report(a.out, "aud_s1_rmsnorm", a1, O, "aud_s1_rmsnorm")
    a2 = linear(a1, embed_audio_proj)
    errs["a2_projection"] = report(a.out, "aud_s2_projection", a2, O, "aud_s2_projection")

    print(f"\ngate: rel-L2 <= {GATE:.0e} (float32 both sides, same bf16-rounded weight values --")
    print("no precision gap expected; this gates wiring/op correctness, not a bf16 rounding floor)")
    worst = max(errs.items(), key=lambda kv: kv[1])
    n_fail = sum(1 for e in errs.values() if e > GATE)
    print(f"worst stage: {worst[0]} rel-L2={worst[1]:.3e}")
    print(f"{len(errs) - n_fail}/{len(errs)} stages PASS" if n_fail == 0 else
          f"{n_fail}/{len(errs)} stages FAIL")


if __name__ == "__main__":
    main()
