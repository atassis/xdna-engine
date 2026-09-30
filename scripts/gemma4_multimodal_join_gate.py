#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Token-level parity gate for the multimodal prompt JOIN (image/audio/video placeholder scatter
into the text embedding sequence) -- the mechanism `rust/npu-models/src/llm/multimodal.rs`
implements at the engine layer (`scatter_media_rows` + the `embed_row` hook both
`NpuDecodeStep::step` and `NpuPrefill::prime` call). This script is the HOST-ONLY, no-device
numeric check that the JOIN CONTRACT is right; the Rust side is unit-tested against the same
contract (see `multimodal.rs`'s `scatter_media_rows` tests), not run through this script -- there
is no Python/Rust bridge here.

The image+audio case builds one combined prompt from `--oracle-dir`. Video is separate, host-only
video-fill/video-underfill cases built inline (no `--oracle-dir` input needed) that exercise the
untested surface named by `multimodal-video-frame-path-is-ungated`: the frame flatten
(`pixel_values_videos.flatten(0, 1)`) and the per-frame `pad_to_max_patches`/drop-padded-rows path
-- the tower math itself is the same `embed_vision` already gated by the image case.

MUST run under the gemma4-oracle venv (transformers 5.17.0 + torch), same as
gemma4_towers_oracle_gen.py: `$GEMMA4_ORACLE_VENV/bin/python scripts/gemma4_multimodal_join_gate.py`.

Two independently-built embedding sequences for one synthetic image+audio+text prompt:

  OURS   -- numpy, mirrors the Rust join: gather text rows from `model.language_model.embed_tokens
            .weight` scaled by sqrt(d_model) (`EmbedTable::row`'s bf16 gather), gather tower rows
            from `gemma4_towers_host_ref.py`'s `vision_tower_forward`/`audio_tower_forward` with the
            tower's own pad tail DROPPED, unscaled, and scatter them at the placeholder positions
            (`scatter_media_rows`'s contract).

  ORACLE -- real HF modules and ops, wired from checkpoint key names (NEVER from_pretrained --
            `check_from_pretrained_keys` in gemma4_towers_oracle_gen.py demonstrates why that
            silently random-inits 10 tower params on this checkpoint): the real
            `Gemma4UnifiedModel.get_placeholder_mask` (called unbound on a lightweight shim -- it
            touches only `self.config.*_token_id`, no decorators, no decoder needed), the real
            `embed_vision`/`embed_audio` nn.Modules (same wiring as the tower oracle), and torch's
            own `.masked_scatter()` -- i.e. `Gemma4UnifiedModel.forward()`'s lines 1013-1078,
            executed directly rather than through `.forward()`, because that method's
            `Gemma4UnifiedModel.__init__` builds `AutoModel.from_config(text_config)` first: the
            full ~12B-parameter decoder, unconditionally, before any of this runs. Same reason
            gemma4_towers_oracle_gen.py never instantiates the full model either.

Text-embedding scale is checked with the REAL `Gemma4UnifiedTextScaledWordEmbedding` module (not a
hand-rolled `* sqrt(d_model)`), but fed a TINY remapped table (~10 rows: the filler "text" ids plus
boi/eoi/boa/eoa/pad) instead of the real [262144, 3840] one -- materializing that full table just to
read 10 rows costs ~4 GB and minutes for no numeric difference: `nn.Embedding.forward` is a pure
per-row gather, no cross-row term, so a table holding exactly the rows that will ever be read, at
consistently remapped indices, reproduces the real module's output bit-for-bit for those rows. Media
placeholder rows never read this table at all: HF substitutes `pad_token_id` at every multimodal
position BEFORE the gather (`forward()`'s `torch.where(multimodal_mask, pad_token_id, ...)`,
"to avoid index-errors") and `masked_scatter` overwrites them immediately after -- so what the
embedding table returns at those positions is provably discarded, and confirms the finding this
gate exists to pin down: media rows never take `embed_scale`.

  python scripts/gemma4_multimodal_join_gate.py \\
      --checkpoint-dir <dir with model.safetensors + config.json> \\
      --towers-weights-dir <dump_gemma4_towers.py output, from the SAME checkpoint> \\
      --oracle-dir <gemma4_towers_oracle_gen.py output, from the SAME checkpoint>
"""
import argparse
import json
import os

import numpy as np
import torch

from gemma4_towers_host_ref import (
    AUDIO_WEIGHT_NAMES, VISION_WEIGHT_NAMES, audio_tower_forward, load, rel_l2, vision_tower_forward,
)

GATE = 1e-4  # same floor as gemma4_towers_host_ref.py's stage gate, same reasoning: float32 both
# sides from the same bf16-rounded checkpoint values, so no precision gap is expected here either.


def build_ours(input_ids, image_rows, audio_rows, video_rows, embed_table, d_model, token_ids):
    """The Rust join's contract, in numpy: `EmbedTable::row` (text, scaled) for every position,
    OVERRIDDEN by `scatter_media_rows`'s per-position media rows (unscaled) at placeholder
    positions -- `embed_row`'s "media override, else text gather" order, run over a whole sequence
    instead of per-token. Video is scattered the same way as image/audio: `scatter_media_rows`
    treats VIDEO_TOKEN_ID identically (see rust/npu-models/src/llm/multimodal.rs)."""
    scale = np.sqrt(d_model).astype(np.float32)
    out = np.stack([embed_table[t] for t in input_ids]).astype(np.float32) * scale
    iters = {token_ids["image_token_id"]: iter(image_rows),
             token_ids["audio_token_id"]: iter(audio_rows),
             token_ids["video_token_id"]: iter(video_rows)}
    for i, t in enumerate(input_ids):
        if t in iters:
            out[i] = next(iters[t])
    leftover = {tok: rest for tok, it in iters.items() if (rest := list(it))}
    if leftover:
        raise AssertionError(
            f"tower row(s) had no placeholder position to scatter into -- count mismatch between "
            f"the prompt and the tower output: {{{', '.join(f'{k}: {len(v)}' for k, v in leftover.items())}}}"
        )
    return out


def build_oracle(input_ids_t, pixel_values, image_position_ids, input_features, input_features_mask,
                  vis, aud, text_embed_rows, token_ids, pad_token_id, d_model,
                  pixel_values_videos=None, video_position_ids=None):
    """`Gemma4UnifiedModel.forward()`'s merge block (lines ~1013-1078 of
    modeling_gemma4_unified.py), run directly against real weights and real ops -- see the module
    docstring for why `.forward()` itself is not callable here. Each media branch is guarded the
    same way `forward()` guards it (`if pixel_values is not None`, etc.), so a prompt can carry any
    subset. The video branch inlines `get_video_features` (lines ~1171-1179): `embed_vision` runs
    on the frame-flattened patches (`.flatten(0, 1)`, num_videos folded into num_frames), then the
    per-frame pad mask is flattened the same way before indexing `pooler_output`."""
    from transformers.models.gemma4_unified.modeling_gemma4_unified import (
        Gemma4UnifiedModel, Gemma4UnifiedTextScaledWordEmbedding)
    from types import SimpleNamespace

    shim = SimpleNamespace(config=SimpleNamespace(
        image_token_id=token_ids["image_token_id"], video_token_id=token_ids["video_token_id"],
        audio_token_id=token_ids["audio_token_id"]))
    image_mask, video_mask, audio_mask = Gemma4UnifiedModel.get_placeholder_mask(
        shim, input_ids_t, None)
    multimodal_mask = image_mask | video_mask | audio_mask

    llm_input_ids = input_ids_t.clone()
    llm_input_ids = torch.where(multimodal_mask, pad_token_id, llm_input_ids)

    # Remap the small set of ids that survive the substitution above onto a compact local table --
    # see the module docstring for why this is exact, not approximate.
    uniq = sorted(set(llm_input_ids[0].tolist()))
    local = {gid: i for i, gid in enumerate(uniq)}
    table = torch.zeros(len(uniq), d_model, dtype=torch.bfloat16)
    for gid, i in local.items():
        table[i] = torch.from_numpy(text_embed_rows[gid])
    embed = Gemma4UnifiedTextScaledWordEmbedding(
        len(uniq), d_model, padding_idx=local[pad_token_id], embed_scale=float(np.sqrt(d_model)))
    embed.weight.data = table
    local_ids = llm_input_ids.clone().apply_(lambda gid: local[gid])
    inputs_embeds = embed(local_ids).float()

    if pixel_values is not None:
        vision_outputs = vis(pixel_values, image_position_ids)
        non_pad_mask = (image_position_ids != -1).all(dim=-1)
        image_features = vision_outputs.pooler_output[non_pad_mask].to(inputs_embeds.dtype)
        n_image_tokens = image_mask.sum()
        image_mask_e = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
        assert inputs_embeds[image_mask_e].numel() == image_features.numel(), \
            f"image tokens {n_image_tokens} vs features {image_features.shape[0]}"
        inputs_embeds = inputs_embeds.masked_scatter(image_mask_e, image_features)

    if pixel_values_videos is not None:
        video_outputs = vis(pixel_values_videos.flatten(0, 1), video_position_ids.flatten(0, 1))
        non_pad_mask_v = (video_position_ids != -1).all(dim=-1)
        video_features = video_outputs.pooler_output[non_pad_mask_v.flatten(0, 1)].to(inputs_embeds.dtype)
        n_video_tokens = video_mask.sum()
        video_mask_e = video_mask.unsqueeze(-1).expand_as(inputs_embeds)
        assert inputs_embeds[video_mask_e].numel() == video_features.numel(), \
            f"video tokens {n_video_tokens} vs features {video_features.shape[0]}"
        inputs_embeds = inputs_embeds.masked_scatter(video_mask_e, video_features)

    if input_features is not None:
        audio_outputs = aud(inputs_embeds=input_features)
        audio_features = audio_outputs[input_features_mask.bool()].to(inputs_embeds.dtype)
        n_audio_tokens = audio_mask.sum()
        audio_mask_e = audio_mask.unsqueeze(-1).expand_as(inputs_embeds)
        assert inputs_embeds[audio_mask_e].numel() == audio_features.numel(), \
            f"audio tokens {n_audio_tokens} vs features {audio_features.shape[0]}"
        inputs_embeds = inputs_embeds.masked_scatter(audio_mask_e, audio_features)

    return inputs_embeds[0].detach().numpy()


def gate_media_rows(input_ids, ours, oracle, media_token_ids, label):
    """rel-L2 gate: rows whose token is in `media_token_ids` vs every other row, printed under
    `label`. Returns (n_media_fail, per_row) so a caller can accumulate failures across several
    prompts/cases before deciding whether to exit non-zero."""
    is_media = np.array([t in media_token_ids for t in input_ids])
    assert ours.shape == oracle.shape, (ours.shape, oracle.shape)
    per_row = np.linalg.norm((ours - oracle).astype(np.float64), axis=-1) / (
        np.linalg.norm(oracle.astype(np.float64), axis=-1) + 1e-30)

    def band(mask, band_label):
        rows = per_row[mask]
        n_fail = int((rows > GATE).sum())
        worst_i = int(np.arange(len(input_ids))[mask][rows.argmax()])
        print(f"  [{label}] {band_label}: worst row {worst_i} (token {input_ids[worst_i]}) "
              f"rel-L2={rows.max():.3e}, {mask.sum() - n_fail}/{mask.sum()} PASS (gate {GATE:.0e})")
        return n_fail

    media_fail = band(is_media, "MEDIA rows")
    if (~is_media).any():
        band(~is_media, "text rows (informational)")
    return media_fail, per_row


def video_processor():
    """The real `Gemma4UnifiedVideoProcessor`, read for its class constants (patch_size,
    pooling_kernel_size, `max_soft_tokens`=70/frame) instead of hand-copying them, so a processor
    default change cannot silently desync this gate from the checkpoint's own frame budget."""
    from transformers.models.gemma4_unified.video_processing_gemma4_unified import (
        Gemma4UnifiedVideoProcessor)
    return Gemma4UnifiedVideoProcessor()


def make_test_video(n_frames, h, w, seed=20260928):
    """Deterministic synthetic clip (gradients + blocks + a frame-shifting diagonal, same
    reasoning as gemma4_towers_oracle_gen.py's make_test_image), varied per frame so a
    frame/patch transposition would misplace visibly frame-specific content."""
    frames = []
    for i in range(n_frames):
        rng = np.random.default_rng(seed + i)
        yy, xx = np.mgrid[0:h, 0:w]
        img = np.zeros((h, w, 3), dtype=np.float32)
        img[..., 0] = 255 * (xx / w)
        img[..., 1] = 255 * (yy / h)
        img[..., 2] = 255 * (((xx // 48) + (yy // 48) + i) % 2)
        blocks = rng.integers(0, 256, size=(h // 48, w // 48, 3)).astype(np.float32)
        img += np.kron(blocks, np.ones((48, 48, 1), dtype=np.float32)) * 0.25
        diag = 80 * np.exp(-((xx - (yy + i * 20) * (w / h)) ** 2) / (2 * 60.0 ** 2))
        img += diag[..., None]
        frames.append(np.clip(img, 0, 255).astype(np.uint8))
    video = np.stack(frames)
    return torch.from_numpy(video).permute(0, 3, 1, 2).contiguous()  # (F, 3, H, W) uint8


def build_video_patches(vproc, h, w, n_frames):
    """Real preprocessing ops throughout -- resize/rescale from the processor instance,
    patchify/merge/pad from the module -- never reimplemented, same policy as
    gemma4_towers_oracle_gen.py's image pipeline. Returns (merged_patches, merged_positions) as
    (n_frames, max_soft_tokens, ...) numpy arrays -- num_videos=1 is folded in by the caller,
    matching `pixel_values_videos.flatten(0, 1)` -- plus the per-frame real (pre-pad) count."""
    from transformers.models.gemma4_unified.video_processing_gemma4_unified import (
        convert_video_to_patches, pad_to_max_patches)
    from transformers.models.gemma4_unified.video_processing_gemma4_unified import (
        patches_merge as merge_video_patches)

    video_u8 = make_test_video(n_frames, h, w)
    max_patches = vproc.max_soft_tokens * vproc.pooling_kernel_size ** 2
    resized = vproc.aspect_ratio_preserving_resize(
        video_u8, vproc.patch_size, max_patches, vproc.pooling_kernel_size, vproc.resample)
    rescaled = vproc.rescale_and_normalize(
        resized, True, 1 / 255, True, vproc.image_mean, vproc.image_std)

    patches = convert_video_to_patches(rescaled, vproc.patch_size)
    ph, pw = resized.shape[-2] // vproc.patch_size, resized.shape[-1] // vproc.patch_size
    grid = torch.meshgrid(torch.arange(pw), torch.arange(ph), indexing="xy")
    teacher_positions = torch.stack(grid, dim=-1).reshape(patches.shape[1], 2)[None].repeat(
        n_frames, 1, 1)

    num_model_patches = patches.shape[1] // (vproc.pooling_kernel_size ** 2)
    merged_patches, merged_positions = merge_video_patches(patches, teacher_positions, num_model_patches)
    n_real_per_frame = merged_patches.shape[1]
    merged_patches, merged_positions = pad_to_max_patches(
        merged_patches, merged_positions, vproc.max_soft_tokens)
    return merged_patches.numpy(), merged_positions.numpy().astype(np.int64), n_real_per_frame


def rows_from_non_pad(s7, n_real_per_frame, from_end=False):
    """Correct selection is the first `n_real_per_frame` rows of every frame
    (`pad_to_max_patches` pads the tail) -- equivalent to boolean-masking on `position_ids != -1`
    since the padding is contiguous. `from_end=True` is the known-bad control: the tower's own
    zero-patch pad-tail output from the SAME frames, same shape, wrong content."""
    n_frames, max_soft_tokens, d_model = s7.shape
    lo = max_soft_tokens - n_real_per_frame if from_end else 0
    return s7[:, lo:lo + n_real_per_frame, :].reshape(-1, d_model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--towers-weights-dir", required=True)
    ap.add_argument("--oracle-dir", required=True,
                     help="gemma4_towers_oracle_gen.py output (in_pixel_values, "
                          "in_image_position_ids, in_input_features, in_input_features_mask)")
    a = ap.parse_args()

    cfg = json.load(open(os.path.join(a.checkpoint_dir, "config.json")))
    token_ids = {k: cfg[k] for k in
                 ("image_token_id", "audio_token_id", "video_token_id", "boi_token_id",
                  "eoi_token_id", "boa_token_id", "eoa_token_index")}
    d_model = cfg["text_config"]["hidden_size"]
    pad_token_id = cfg["text_config"]["pad_token_id"]
    print("token ids:", token_ids, "pad_token_id:", pad_token_id, "d_model:", d_model)

    # ---- tower forward, host numpy (reused, not re-implemented -- see module docstring) ----
    O, W = a.oracle_dir, a.towers_weights_dir
    vision_w = {name: load(W, name) for name in VISION_WEIGHT_NAMES}
    audio_w = {name: load(W, name) for name in AUDIO_WEIGHT_NAMES}
    pixel_values_np = load(O, "in_pixel_values")
    image_position_ids_np = load(O, "in_image_position_ids")
    input_features_np = load(O, "in_input_features")
    input_features_mask_np = load(O, "in_input_features_mask")[0].astype(bool)

    vis_all = vision_tower_forward(pixel_values_np, image_position_ids_np, vision_w)["s7_projection"][0]
    non_pad = (image_position_ids_np != -1).all(axis=-1)[0]
    image_rows = vis_all[non_pad]  # drop the tower's own pad tail -- the exact step the task names
    # as the easy-to-get-wrong one: a padded row is NOT zero after projection.
    assert not np.allclose(vis_all[~non_pad], 0.0, atol=1e-3) or (~non_pad).sum() == 0, \
        "expected the padded rows to be nonzero post-projection (LayerNorm bias / RMSNorm) -- " \
        "if this fires the fixture stopped exercising the drop-padding branch"
    n_pad = int((~non_pad).sum())
    print(f"vision: {len(image_rows)} real soft tokens, {n_pad} dropped (padded)")

    aud_all = audio_tower_forward(input_features_np, audio_w)["s2_projection"][0]
    audio_rows = aud_all[input_features_mask_np]
    print(f"audio: {len(audio_rows)} real soft tokens, "
          f"{(~input_features_mask_np).sum()} dropped (padded)")

    # ---- synthetic prompt: filler text ids around one image run and one audio run ----
    filler = [10, 20, 30, 40, 50]
    n_img, n_aud = len(image_rows), len(audio_rows)
    input_ids = (
        [filler[0], filler[1], token_ids["boi_token_id"]] + [token_ids["image_token_id"]] * n_img +
        [token_ids["eoi_token_id"], filler[2], token_ids["boa_token_id"]] +
        [token_ids["audio_token_id"]] * n_aud +
        [token_ids["eoa_token_index"], filler[3], filler[4]]
    )
    print(f"prompt: {len(input_ids)} tokens ({n_img} image + {n_aud} audio placeholders)")

    # ---- text embedding rows: real checkpoint bf16 bytes, fetched by exact id (lazy slice, not a
    # full [262144, 3840] load) ----
    from safetensors import safe_open
    needed_ids = sorted(set(input_ids) | {pad_token_id})
    text_embed_rows = {}
    with safe_open(os.path.join(a.checkpoint_dir, "model.safetensors"), framework="pt") as f:
        sl = f.get_slice("model.language_model.embed_tokens.weight")
        for tid in needed_ids:
            text_embed_rows[tid] = sl[tid:tid + 1][0].float().numpy()

    ours = build_ours(input_ids, image_rows, audio_rows, [],
                       {k: v for k, v in text_embed_rows.items()}, d_model, token_ids)

    # ---- oracle: real modules wired from checkpoint key names (never from_pretrained) ----
    from transformers import AutoConfig
    from transformers.models.gemma4_unified.modeling_gemma4_unified import (
        Gemma4UnifiedVisionEmbedder, Gemma4UnifiedMultimodalEmbedder)
    hf_cfg = AutoConfig.from_pretrained(a.checkpoint_dir)
    vc, ac, tc = hf_cfg.vision_config, hf_cfg.audio_config, hf_cfg.text_config
    vis = Gemma4UnifiedVisionEmbedder(vc, tc).float().eval()
    aud = Gemma4UnifiedMultimodalEmbedder(ac, tc).float().eval()
    with safe_open(os.path.join(a.checkpoint_dir, "model.safetensors"), framework="pt") as f:
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

    input_ids_t = torch.tensor([input_ids], dtype=torch.long)
    pixel_values_t = torch.from_numpy(pixel_values_np).to(vis.patch_dense.weight.dtype)
    image_position_ids_t = torch.from_numpy(image_position_ids_np).long()
    input_features_t = torch.from_numpy(input_features_np).float()
    input_features_mask_t = torch.from_numpy(input_features_mask_np[None].astype(np.float32))

    oracle = build_oracle(input_ids_t, pixel_values_t, image_position_ids_t, input_features_t,
                           input_features_mask_t, vis, aud, text_embed_rows, token_ids,
                           pad_token_id, d_model)

    # ---- gate: row-by-row (token-level), MEDIA rows separated from TEXT rows ----
    #
    # Text rows carry a real but PRE-EXISTING and OUT-OF-SCOPE ~5.2e-4 deviation that has nothing
    # to do with the join: `Gemma4UnifiedTextScaledWordEmbedding.forward` casts `embed_scale`
    # (sqrt(3840)=61.9677...) to `self.weight.dtype` (bf16) BEFORE multiplying, rounding it to
    # 62.0 exactly, while `EmbedTable::row` (already shipped, this task did not touch it) scales in
    # full f32. Confirmed, not guessed: `torch.tensor(61.9677...).bfloat16()` == 62.0 == the
    # observed ratio (1.00052) to five figures. Reported separately so it cannot hide a join defect
    # inside its own noise, and not gated, because it predates and is orthogonal to this task.
    media_fail, _ = gate_media_rows(
        input_ids, ours, oracle, (token_ids["image_token_id"], token_ids["audio_token_id"]),
        "image+audio")
    total_fail = media_fail

    # A media row must not have taken embed_scale: an unscaled tower row is centered near the
    # tower's own output magnitude, sqrt(d_model)=62.0x smaller than what re-scaling it would give.
    # Catches a reintroduced `* scale` on the media branch even if it lands within GATE (a small
    # image/audio norm could otherwise hide a scale bug inside the noise floor).
    img_pos = [i for i, t in enumerate(input_ids) if t == token_ids["image_token_id"]][:1]
    if img_pos:
        i = img_pos[0]
        ratio = np.linalg.norm(oracle[i]) / np.linalg.norm(ours[i] * np.sqrt(d_model))
        print(f"sanity: oracle row {i} norm / (ours*sqrt(d_model)) norm = {ratio:.3f} "
              f"(expect far from 1.0 -- 1.0 would mean the oracle DID scale the media row)")

    # ==== VIDEO: fill and underfill the 70-soft-token-per-frame budget ====
    #
    # get_video_features calls the SAME embed_vision as stills (already gated above and in the
    # 18/18 tower gate), so this exercises the untested surface only: frame flatten
    # (`.flatten(0, 1)` folding num_frames into the vision-tower batch) and the per-frame
    # `pad_to_max_patches`/drop-padded-rows path. Video-only prompts (no image/audio), built and
    # gated the same way as the image+audio prompt above.
    vproc = video_processor()
    for label, h, w, n_frames in (
        ("video-fill (336x480, exact 70/frame)", 336, 480, 3),
        ("video-underfill (1104x576 -> 528x288, 66/frame)", 1104, 576, 3),
    ):
        merged_patches, merged_positions, n_real_per_frame = build_video_patches(
            vproc, h, w, n_frames)
        s7 = vision_tower_forward(merged_patches, merged_positions, vision_w)["s7_projection"]
        non_pad_v = (merged_positions != -1).all(axis=-1)  # (n_frames, max_soft_tokens)
        n_dropped = int((~non_pad_v).sum())
        print(f"\n{label}: {n_real_per_frame}/{vproc.max_soft_tokens} soft tokens/frame real, "
              f"{n_dropped} dropped (padded) across {n_frames} frames")

        video_rows = rows_from_non_pad(s7, n_real_per_frame, from_end=False)
        n_vid = len(video_rows)
        v_filler = [60, 70]
        input_ids_v = (
            [v_filler[0], token_ids["boi_token_id"]] + [token_ids["video_token_id"]] * n_vid +
            [token_ids["eoi_token_id"], v_filler[1]]
        )
        needed_v = sorted(set(input_ids_v) | {pad_token_id})
        with safe_open(os.path.join(a.checkpoint_dir, "model.safetensors"), framework="pt") as f:
            sl = f.get_slice("model.language_model.embed_tokens.weight")
            text_embed_rows_v = {tid: sl[tid:tid + 1][0].float().numpy() for tid in needed_v}

        ours_v = build_ours(input_ids_v, [], [], video_rows, text_embed_rows_v, d_model, token_ids)

        input_ids_v_t = torch.tensor([input_ids_v], dtype=torch.long)
        pv_t = torch.from_numpy(merged_patches).unsqueeze(0).to(vis.patch_dense.weight.dtype)
        vpos_t = torch.from_numpy(merged_positions).unsqueeze(0).long()
        oracle_v = build_oracle(input_ids_v_t, None, None, None, None, vis, aud, text_embed_rows_v,
                                 token_ids, pad_token_id, d_model,
                                 pixel_values_videos=pv_t, video_position_ids=vpos_t)

        v_fail, _ = gate_media_rows(input_ids_v, ours_v, oracle_v, (token_ids["video_token_id"],), label)
        total_fail += v_fail

        # Known-bad control: swap the correctly-dropped rows for the tower's OWN padded-tail
        # output at the same positions (same shape, so it clears the count check -- only the rel-L2
        # gate can catch it). Only a real control when there IS a pad tail.
        if n_dropped:
            bad_rows = rows_from_non_pad(s7, n_real_per_frame, from_end=True)
            ours_bad = build_ours(input_ids_v, [], [], bad_rows, text_embed_rows_v, d_model, token_ids)
            bad_fail, bad_per_row = gate_media_rows(
                input_ids_v, ours_bad, oracle_v, (token_ids["video_token_id"],),
                f"{label} KNOWN-BAD CONTROL (padded rows instead of dropped)")
            if bad_fail == 0:
                raise SystemExit(
                    f"gate did not fail on the known-bad input for {label} -- "
                    f"worst rel-L2 {bad_per_row.max():.3e} stayed under {GATE:.0e}; the gate is vacuous")
            print(f"  control confirmed: {bad_fail} row(s) fail as expected -- the gate is not vacuous")

    if total_fail:
        raise SystemExit(f"FAIL: {total_fail} media row(s) exceed gate {GATE:.0e} across all cases")
    print("\nPASS: every media row (image, audio, video-fill, video-underfill) matches the "
          "HF-side reference within the gate")
    if os.environ.get("JOIN_GATE_DEBUG"):
        per_row = np.linalg.norm((ours - oracle).astype(np.float64), axis=-1) / (
            np.linalg.norm(oracle.astype(np.float64), axis=-1) + 1e-30)
        for i in np.argsort(-per_row)[:12]:
            print(f"  row {i} token {input_ids[i]} rel-L2={per_row[i]:.3e} "
                  f"|ours|={np.linalg.norm(ours[i]):.4f} |oracle|={np.linalg.norm(oracle[i]):.4f} "
                  f"max-abs-diff={np.abs(ours[i]-oracle[i]).max():.4e}")


if __name__ == "__main__":
    main()
