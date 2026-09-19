//! The multimodal prompt JOIN for Gemma-4-12B: fill a media placeholder token's embedding row
//! from a vision/audio tower's output instead of the text `W_head` gather. Gemma-4 is
//! encoder-free -- the towers themselves run on HOST (`scripts/gemma4_towers_host_ref.py`, numpy;
//! no Rust port of them exists yet) -- so this module owns only the JOIN: drop the tower's own
//! padding tail, scatter the remaining rows at the prompt's placeholder positions, and hand
//! `embed_row` to both dispatch paths as the ONE gather that decides, per position, whether to
//! read a media row or the text embedding table.
//!
//! Token ids are the checkpoint's own (`config.json`, NOT `meta.json` -- the decode artifact's
//! generator has no notion of media). Contract established by reading
//! `modeling_gemma4_unified.py` (transformers 5.17.0), not guessed, and pinned by
//! `scripts/gemma4_multimodal_join_gate.py`'s HF-side gate (509/509 media rows, worst rel-L2
//! 4.127e-05, against a reference wired from checkpoint key names -- never `from_pretrained`,
//! which silently random-inits 10 of these towers' own params on this checkpoint):
//!
//!   - a media row is the tower's OWN output verbatim -- it never takes `embed_scale`. HF scatters
//!     `get_image_features`'s output directly over the (scaled, then discarded) placeholder
//!     embedding via `masked_scatter`, so the scale the text gather would have applied at that
//!     position is never applied to what replaces it.
//!   - the tower runs on every row it is given, INCLUDING its own padding tail, and the CALLER
//!     drops the padding before scatter (`(position_ids != -1).all(-1)`) -- a padded row is NOT
//!     zero after projection (LayerNorm bias / RMSNorm), so treating it as a sentinel silently
//!     corrupts the prompt.
//!
//! **Known limitation, stated loudly, not routed around here:** a row inside a media block must
//! attend FORWARD to the block's end, but batched prefill dispatches `M` positions per chunk
//! (`NpuPrefill::prime`) and a ~280-token image spans 5 chunks at M=64 -- chunking would compute
//! silently wrong attention across the boundary. `NpuDecodeStep::prefill` gates the WHOLE batched
//! path off whenever a generation carries any media rather than reason about which chunks are
//! "safe"; see its doc comment.

use std::borrow::Cow;
use std::collections::HashMap;

use crate::api::EngineError;
use crate::llm::npu_decode::{pack_bf16_bytes, EmbedTable};

/// `config.json`'s `image_token_id` -- every soft token of an image placeholder run.
pub const IMAGE_TOKEN_ID: u32 = 258_880;
/// `config.json`'s `audio_token_id` -- every soft token of an audio placeholder run.
pub const AUDIO_TOKEN_ID: u32 = 258_881;
/// `config.json`'s `video_token_id` -- every soft token of a video placeholder run. The vision
/// tower re-run per frame (same weights, same op sequence as an image); the routing limitation
/// above applies to it too and is not exercised by the join gate.
pub const VIDEO_TOKEN_ID: u32 = 258_884;
/// `config.json`'s `boi_token_id`, wrapping an image/video run. A literal TEXT token: never
/// matched by [`scatter_media_rows`], gathered and scaled like any other.
pub const BOI_TOKEN_ID: u32 = 255_999;
/// `config.json`'s `eoi_token_id`, closing an image/video run. Literal text, as [`BOI_TOKEN_ID`].
pub const EOI_TOKEN_ID: u32 = 258_882;
/// `config.json`'s `boa_token_id`, wrapping an audio run. Literal text, as [`BOI_TOKEN_ID`].
pub const BOA_TOKEN_ID: u32 = 256_000;

/// One media attachment's tower output, ready to scatter: `rows[i]` is soft-token `i`'s
/// `[d_model]` f32 vector, in prompt order, with the tower's own padding tail ALREADY dropped
/// (see the module doc) and NOT scaled by `embed_scale`.
pub struct MediaAttachment {
    /// Which placeholder run this fills -- [`IMAGE_TOKEN_ID`], [`AUDIO_TOKEN_ID`] or
    /// [`VIDEO_TOKEN_ID`].
    pub token_id: u32,
    pub rows: Vec<Vec<f32>>,
}

/// Precomputed media embedding rows, keyed by ABSOLUTE prompt position -- never by token id, since
/// one placeholder id repeats once per soft token and only the position disambiguates which row a
/// given occurrence gets. Bf16-packed `d_model * 2` bytes each, the same wire shape
/// `EmbedTable::row` returns, so `embed_row` hands either straight to its caller with no
/// branch on the caller's side. Empty (`Default`) for a text-only generation -- the common case --
/// which costs `embed_row` exactly one hash lookup that always misses.
#[derive(Default, Clone, Debug)]
pub struct MediaEmbeds(HashMap<usize, Vec<u8>>);

impl MediaEmbeds {
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// Place every attachment's rows at its token id's placeholder positions in `tokens`, in prompt
/// order -- the host-side equivalent of `Gemma4UnifiedModel.forward`'s per-type `masked_scatter`.
///
/// Positions for a given `token_id` are taken ascending; attachments sharing a `token_id`
/// concatenate in the order given (two images scatter first-attachment-first, matching
/// `torch.cat(image_features, dim=0)` over `get_image_features`'s per-image split). Refuses,
/// rather than silently truncating or repeating, when an id's placeholder count and its
/// attachments' total row count disagree -- the same count HF itself asserts
/// (`torch_compilable_check(inputs_embeds[image_mask].numel() == image_features.numel(), ...)`)
/// immediately before the scatter it would otherwise corrupt.
pub fn scatter_media_rows(
    tokens: &[u32], attachments: &[MediaAttachment], d_model: usize,
) -> Result<MediaEmbeds, EngineError> {
    let mut out = HashMap::new();
    for tok in [IMAGE_TOKEN_ID, VIDEO_TOKEN_ID, AUDIO_TOKEN_ID] {
        let positions: Vec<usize> =
            tokens.iter().enumerate().filter(|&(_, &t)| t == tok).map(|(i, _)| i).collect();
        let rows: Vec<&Vec<f32>> =
            attachments.iter().filter(|a| a.token_id == tok).flat_map(|a| a.rows.iter()).collect();
        if positions.len() != rows.len() {
            return Err(EngineError::Unsupported(format!(
                "token {tok}: {} placeholder position(s) in the prompt but {} tower row(s) supplied",
                positions.len(),
                rows.len()
            )));
        }
        for (pos, row) in positions.into_iter().zip(rows) {
            if row.len() != d_model {
                return Err(EngineError::Unsupported(format!(
                    "token {tok} row for prompt position {pos} is {} wide, want d_model={d_model}",
                    row.len()
                )));
            }
            out.insert(pos, pack_bf16_bytes(row));
        }
    }
    Ok(MediaEmbeds(out))
}

/// The ONE gather both `NpuDecodeStep::step` and `NpuPrefill::prime` call: a media row at `pos`
/// when one was scattered there, else the ordinary text gather. Sharing this function (not just
/// the convention) is what keeps the per-token and batched paths from drifting on the media
/// branch, the same argument [`EmbedTable`]'s own doc makes for the text one.
pub(crate) fn embed_row<'a>(
    embed: &'a EmbedTable, media: &'a MediaEmbeds, token: u32, pos: usize,
) -> Result<Cow<'a, [u8]>, EngineError> {
    if let Some(row) = media.0.get(&pos) {
        return Ok(Cow::Borrowed(row));
    }
    embed.row(token)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::npu_decode::unpack_bf16_bytes;

    fn attach(token_id: u32, rows: &[[f32; 2]]) -> MediaAttachment {
        MediaAttachment { token_id, rows: rows.iter().map(|r| r.to_vec()).collect() }
    }

    #[test]
    fn scatters_rows_at_ascending_placeholder_positions() {
        let tokens = [10, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 20, IMAGE_TOKEN_ID];
        let media =
            scatter_media_rows(&tokens, &[attach(IMAGE_TOKEN_ID, &[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])], 2)
                .unwrap();
        assert_eq!(unpack_bf16_bytes(&media.0[&1]), vec![1.0, 1.0]);
        assert_eq!(unpack_bf16_bytes(&media.0[&2]), vec![2.0, 2.0]);
        assert_eq!(unpack_bf16_bytes(&media.0[&4]), vec![3.0, 3.0]);
        assert!(!media.0.contains_key(&0) && !media.0.contains_key(&3), "text positions untouched");
    }

    #[test]
    fn two_attachments_of_the_same_token_concatenate_in_submission_order() {
        let tokens = [IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID];
        let media = scatter_media_rows(
            &tokens,
            &[attach(IMAGE_TOKEN_ID, &[[1.0, 1.0]]), attach(IMAGE_TOKEN_ID, &[[2.0, 2.0], [3.0, 3.0]])],
            2,
        )
        .unwrap();
        assert_eq!(unpack_bf16_bytes(&media.0[&0]), vec![1.0, 1.0], "first attachment fills first");
        assert_eq!(unpack_bf16_bytes(&media.0[&1]), vec![2.0, 2.0]);
        assert_eq!(unpack_bf16_bytes(&media.0[&2]), vec![3.0, 3.0]);
    }

    #[test]
    fn image_audio_and_video_positions_are_independent() {
        let tokens = [AUDIO_TOKEN_ID, IMAGE_TOKEN_ID, VIDEO_TOKEN_ID];
        let media = scatter_media_rows(
            &tokens,
            &[
                attach(IMAGE_TOKEN_ID, &[[9.0, 9.0]]),
                attach(AUDIO_TOKEN_ID, &[[8.0, 8.0]]),
                attach(VIDEO_TOKEN_ID, &[[7.0, 7.0]]),
            ],
            2,
        )
        .unwrap();
        assert_eq!(unpack_bf16_bytes(&media.0[&0]), vec![8.0, 8.0], "audio at its own position");
        assert_eq!(unpack_bf16_bytes(&media.0[&1]), vec![9.0, 9.0], "image at its own position");
        assert_eq!(unpack_bf16_bytes(&media.0[&2]), vec![7.0, 7.0], "video at its own position");
    }

    #[test]
    fn a_text_only_prompt_scatters_nothing() {
        let media = scatter_media_rows(&[1, 2, 3], &[], 2).unwrap();
        assert!(media.is_empty());
    }

    #[test]
    fn refuses_a_placeholder_count_mismatch_rather_than_truncating_or_repeating() {
        let tokens = [IMAGE_TOKEN_ID, IMAGE_TOKEN_ID];
        let err = scatter_media_rows(&tokens, &[attach(IMAGE_TOKEN_ID, &[[1.0, 1.0]])], 2).unwrap_err();
        assert!(format!("{err}").contains("2 placeholder"), "{err}");
    }

    #[test]
    fn refuses_a_row_whose_width_disagrees_with_d_model() {
        let tokens = [IMAGE_TOKEN_ID];
        let err = scatter_media_rows(&tokens, &[attach(IMAGE_TOKEN_ID, &[[1.0, 1.0]])], 3).unwrap_err();
        assert!(format!("{err}").contains("d_model=3"), "{err}");
    }

    /// The hook end to end, with a REAL (mmapped) `EmbedTable` -- not just the map `scatter_media_rows`
    /// builds. A media position returns its scattered row VERBATIM (no `embed_scale`); any other
    /// position falls through to the ordinary scaled text gather.
    #[test]
    fn embed_row_prefers_the_media_override_and_falls_back_to_the_text_gather() {
        let dir = tempfile::tempdir().unwrap();
        let text_rows = vec![vec![1.0f32, 2.0], vec![3.0, 4.0]];
        let embed = EmbedTable::for_test(dir.path(), &text_rows, /* scale */ 10.0);
        let media =
            scatter_media_rows(&[0, IMAGE_TOKEN_ID], &[attach(IMAGE_TOKEN_ID, &[[9.0, 9.0]])], 2).unwrap();

        let media_row = embed_row(&embed, &media, IMAGE_TOKEN_ID, 1).unwrap();
        assert_eq!(unpack_bf16_bytes(&media_row), vec![9.0, 9.0], "unscaled, exactly the tower row");

        let text_row = embed_row(&embed, &media, 0, 0).unwrap();
        assert_eq!(unpack_bf16_bytes(&text_row), vec![10.0, 20.0], "row 0 = [1,2] * scale 10.0");

        // Same token id, a position `scatter_media_rows` never touched -- e.g. a second, shorter
        // generation reusing this table with an empty `MediaEmbeds`: must still read row 1 of the
        // TEXT table (token id 1 == IMAGE_TOKEN_ID would be out of this table's tiny vocab, so
        // exercise the fallback with an in-range id instead).
        let empty = MediaEmbeds::default();
        let still_text = embed_row(&embed, &empty, 1, 1).unwrap();
        assert_eq!(unpack_bf16_bytes(&still_text), vec![30.0, 40.0], "row 1 = [3,4] * scale 10.0");
    }
}
