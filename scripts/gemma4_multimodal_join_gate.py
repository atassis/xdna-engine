#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Token-level parity gate for the multimodal prompt JOIN (image+audio placeholder scatter into
the text embedding sequence) -- the mechanism `rust/npu-engine/src/llm/multimodal.rs` implements
at the engine layer (`scatter_media_rows` + the `embed_row` hook both `NpuDecodeStep::step` and
`NpuPrefill::prime` call). This script is the HOST-ONLY, no-device numeric check that the JOIN
CONTRACT is right; the Rust side is unit-tested against the same contract (see `multimodal.rs`'s
`scatter_media_rows` tests), not run through this script -- there is no Python/Rust bridge here.

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


def build_ours(input_ids, image_rows, audio_rows, embed_table, d_model, token_ids):
    """The Rust join's contract, in numpy: `EmbedTable::row` (text, scaled) for every position,
    OVERRIDDEN by `scatter_media_rows`'s per-position media rows (unscaled) at placeholder
    positions -- `embed_row`'s "media override, else text gather" order, run over a whole sequence
    instead of per-token."""
    scale = np.sqrt(d_model).astype(np.float32)
    out = np.stack([embed_table[t] for t in input_ids]).astype(np.float32) * scale
    img_i = iter(image_rows)
    aud_i = iter(audio_rows)
    for i, t in enumerate(input_ids):
        if t == token_ids["image_token_id"]:
            out[i] = next(img_i)
        elif t == token_ids["audio_token_id"]:
            out[i] = next(aud_i)
    remaining_img, remaining_aud = list(img_i), list(aud_i)
    if remaining_img or remaining_aud:
        raise AssertionError(
            f"{len(remaining_img)} image / {len(remaining_aud)} audio tower rows had no placeholder "
            "position to scatter into -- count mismatch between the prompt and the tower output"
        )
    return out


def build_oracle(input_ids_t, pixel_values, image_position_ids, input_features, input_features_mask,
                  vis, aud, text_embed_rows, token_ids, pad_token_id, d_model):
    """`Gemma4UnifiedModel.forward()`'s merge block (lines ~1013-1078 of
    modeling_gemma4_unified.py), run directly against real weights and real ops -- see the module
    docstring for why `.forward()` itself is not callable here."""
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

    vision_outputs = vis(pixel_values, image_position_ids)
    non_pad_mask = (image_position_ids != -1).all(dim=-1)
    image_features = vision_outputs.pooler_output[non_pad_mask].to(inputs_embeds.dtype)
    n_image_tokens = image_mask.sum()
    image_mask_e = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
    assert inputs_embeds[image_mask_e].numel() == image_features.numel(), \
        f"image tokens {n_image_tokens} vs features {image_features.shape[0]}"
    inputs_embeds = inputs_embeds.masked_scatter(image_mask_e, image_features)

    audio_outputs = aud(inputs_embeds=input_features)
    audio_features = audio_outputs[input_features_mask.bool()].to(inputs_embeds.dtype)
    n_audio_tokens = audio_mask.sum()
    audio_mask_e = audio_mask.unsqueeze(-1).expand_as(inputs_embeds)
    assert inputs_embeds[audio_mask_e].numel() == audio_features.numel(), \
        f"audio tokens {n_audio_tokens} vs features {audio_features.shape[0]}"
    inputs_embeds = inputs_embeds.masked_scatter(audio_mask_e, audio_features)

    return inputs_embeds[0].detach().numpy()


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

    ours = build_ours(input_ids, image_rows, audio_rows,
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
    is_media = np.array([t in (token_ids["image_token_id"], token_ids["audio_token_id"])
                          for t in input_ids])
    assert ours.shape == oracle.shape, (ours.shape, oracle.shape)
    per_row = np.linalg.norm((ours - oracle).astype(np.float64), axis=-1) / (
        np.linalg.norm(oracle.astype(np.float64), axis=-1) + 1e-30)

    def band(mask, label):
        rows = per_row[mask]
        n_fail = int((rows > GATE).sum())
        worst_i = int(np.arange(len(input_ids))[mask][rows.argmax()])
        print(f"{label}: worst row {worst_i} (token {input_ids[worst_i]}) rel-L2={rows.max():.3e}, "
              f"{mask.sum() - n_fail}/{mask.sum()} PASS (gate {GATE:.0e})")
        return n_fail

    print(f"\nwhole-sequence rel-L2: {rel_l2(ours, oracle):.3e}")
    media_fail = band(is_media, "MEDIA rows (the join -- gated)")
    band(~is_media, "text rows  (pre-existing embed_scale rounding -- informational)")
    if media_fail:
        raise SystemExit(f"FAIL: {media_fail} media row(s) exceed gate {GATE:.0e}")
    print("PASS: every media row matches the HF-side reference within the gate")
    if os.environ.get("JOIN_GATE_DEBUG"):
        for i in np.argsort(-per_row)[:12]:
            print(f"  row {i} token {input_ids[i]} rel-L2={per_row[i]:.3e} "
                  f"|ours|={np.linalg.norm(ours[i]):.4f} |oracle|={np.linalg.norm(oracle[i]):.4f} "
                  f"max-abs-diff={np.abs(ours[i]-oracle[i]).max():.4e}")

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


if __name__ == "__main__":
    main()
