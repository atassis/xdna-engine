//! Batched prefill: prime the KV cache for a prompt `M` positions per dispatch instead of one.
//!
//! The design is `xdna-engine-private/journal/docs/reference/batched-prefill-architecture.md`. Two
//! of its decisions are what this file implements, and both are load-bearing:
//!
//! **One arena, two ELFs.** The weights live in the SCRATCH arena, so a second arena would mean a
//! second copy of them plus a host round-trip of the KV cache between the halves. `FusedArena`
//! allocates its BOs against the DEVICE, not a hardware context, and `bind_resident` takes `&self`
//! -- so one arena serves N residents. Cleared on device by `npu-probes/src/bin/arena_share_probe.rs`
//! (two hardware contexts, one arena, 0/303872 logits bytes differ), which is why this is wired as a
//! shared arena rather than a copy. The layouts must AGREE for that to be safe, and agreement is a
//! property of how the two graphs were generated, never a guarantee --
//! [`LlmArtifact::check_shared_layout_agrees`](crate::llm::LlmArtifact::check_shared_layout_agrees)
//! is where that is checked.
//!
//! **Fixed `M`, pad the tail, never fall back to M=1 for the remainder.** `P = qM + r`: running the
//! `r` leftover positions through the per-token path costs up to `M-1` times the per-prompt-token
//! decode cost, where padding the final chunk costs at most one chunk whatever `r` is. Padding is
//! safe with no new mechanism, by the invariant the rail already relies on: pad rows sit at the END
//! of the chunk, so no real token attends them (causal), and their KV rows land past `n_past` where
//! the existing mask already kills them -- the same property that makes cross-request cache reuse
//! safe in [`NpuDecodeStep::reset`](crate::llm::NpuDecodeStep).
//!
//! Per-chunk protocol, which is the decode protocol with `M` substituted for 1 -- deliberately, so
//! that the decode graph stays the `M=1` instance of the prefill graph rather than a separate thing:
//!   1. `x`                     <- `M` embedding rows, token-major `[M, d_model]` bf16
//!   2. each declared RoPE table <- `M` angle rows for absolute positions `[start, start+M)`,
//!      token-major (`rope` on a prefill artifact, `rope_global`/`rope_local` on a decode one --
//!      resolved from `inputs`, never hardcoded)
//!   3. scratchpad `kv_param`   = `start * head_dim`  (element-unit BD offset; `pos * head_dim` at M=1)
//!   4. scratchpad `mask_param` = `start + M`         (causal width; `pos + 1` at M=1)
//!   5. one dispatch. Nothing is read back -- prefill's whole product is the KV it left in scratch.
//!
//! Step 5 does no `sync_from_device`, and that is not an omission. The KV writes are device-side and
//! the arena probe showed them visible to the other context without a host round-trip; what the
//! architecture doc warns about is the opposite direction (a HOST write to scratch that skips
//! `sync_to_device`), and this path makes none. A host READ of `kc`/`vc` -- the teacher-forced KV
//! gate, not this path -- is a separate question and does need one.

use std::rc::Rc;

use npu_xrt::{Device, ElfResident, FusedArena};

use crate::api::EngineError;
use crate::llm::artifact::{BufLoc, LlmArtifact};
use crate::llm::npu_decode::{pack_bf16_bytes, rope_row, EmbedTable};

/// `NPU_LLM_PREFILL_BATCHED` -- the one accessor (E003 of the env-flag contract).
///
/// Default ON with `not_zero` semantics (E001), matching `NPU_LLM_REUSE_KV` next door: `=0` takes
/// the per-token priming path, which is the A/B control for every prefill measurement and the
/// bisect handle if a batched prompt ever disagrees with `P` sequential steps. Both arms are
/// device-only -- this is a step WITHIN the tier ladder, not a silent fall to host.
fn batched_prefill_enabled() -> bool {
    std::env::var("NPU_LLM_PREFILL_BATCHED").ok().as_deref() != Some("0")
}

/// One padded dispatch of the prefill ELF: absolute positions `[start, start + M)`, of which the
/// first `real` carry prompt tokens and the rest are padding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrefillChunk {
    /// Absolute KV position of the chunk's first row.
    pub start: usize,
    /// Prompt tokens in this chunk, `1..=M`. `M - real` rows are padding.
    pub real: usize,
}

impl PrefillChunk {
    /// Padding rows in this chunk -- non-zero only for the last one, and only when `M` does not
    /// divide the prompt.
    pub fn pad(&self, batch: usize) -> usize {
        batch - self.real
    }
}

/// The `ceil(n / batch)` dispatches that cover positions `[0, n)`, each padded to `batch`.
///
/// Total device positions written is `ceil(n/batch) * batch`, which is why the pairing check
/// requires `S % M == 0`: at `n <= S-1` that product can then never exceed the KV window.
pub fn chunk_plan(n: usize, batch: usize) -> Vec<PrefillChunk> {
    assert!(batch > 0, "prefill batch must be non-zero (dims.M is validated at load)");
    (0..n.div_ceil(batch))
        .map(|c| {
            let start = c * batch;
            PrefillChunk { start, real: (n - start).min(batch) }
        })
        .collect()
}

/// The `[M, head_dim]` bf16 angle block for one chunk: `M` consecutive rows, token-major, row `i`
/// carrying the angles for ABSOLUTE position `start + i`.
///
/// Token-major is the device convention and it is a trap worth naming: the RoPE kernel
/// (`iron/operators/rope/design.py`) consumes `rows/angle_rows` CONSECUTIVE rows per angle row,
/// while the CPU reference (`rope/reference.py`, `cos.repeat`) implements the INTERLEAVED
/// convention. The two agree only at `angle_rows == 1` or `angle_rows == rows`; this rail is the
/// second case (one angle row per position), so a test written against `reference.py` would compute
/// a wrong expected answer. Each row is `rope_row`'s output verbatim -- the same function the M=1
/// path calls -- so the two paths cannot drift in the interleaved `[cos, sin, ...]` packing either.
pub fn rope_block(start: usize, batch: usize, head_dim: usize, theta: f64) -> Vec<u8> {
    let mut out = Vec::with_capacity(batch * head_dim * 2);
    for i in 0..batch {
        out.extend_from_slice(&pack_bf16_bytes(&rope_row(start + i, head_dim, theta)));
    }
    out
}

/// A resident device backend for one batched-prefill ELF, bound to a `FusedArena` it does not own.
///
/// Constructed only through [`NpuDecodeStep::with_prefill`](crate::llm::NpuDecodeStep::with_prefill),
/// because the arena must be sized and filled for BOTH halves before either resident binds to it.
pub struct NpuPrefill {
    artifact: LlmArtifact,
    res: ElfResident,
    batch: usize,
    /// Each declared RoPE input buffer with the base its rows are computed from, resolved at load
    /// against the DECODE artifact -- the authority for a model constant, which a prefill artifact
    /// may leave undeclared and which `check_prefill_pairing` refuses to let it contradict.
    rope_writes: Vec<(BufLoc, f64)>,
}

impl NpuPrefill {
    /// Open the prefill ELF on its own hardware context and bind it to the SHARED arena. `artifact`
    /// must already have been checked against `decode`.
    pub(crate) fn open(
        dev: &Rc<Device>,
        artifact: LlmArtifact,
        decode: &LlmArtifact,
        arena: &FusedArena,
    ) -> Result<Self, EngineError> {
        let rope_writes = artifact.rope_writes(decode)?;
        let elf = std::fs::read(artifact.elf_path())
            .map_err(|e| EngineError::Load(format!("read {}: {e}", artifact.elf_path().display())))?;
        let res = dev.open_elf_resident(&elf, Some(&artifact.kernel_name)).map_err(|e| {
            EngineError::Load(format!("open_elf_resident: prefill ELF lacks a ctrl scratchpad: {e}"))
        })?;
        arena
            .bind_resident(&res)
            .map_err(|e| EngineError::Load(format!("bind prefill resident to the shared arena: {e}")))?;
        let batch = artifact.batch;
        Ok(NpuPrefill { artifact, res, batch, rope_writes })
    }

    pub fn batch(&self) -> usize {
        self.batch
    }

    pub(crate) fn batched_enabled(&self) -> bool {
        batched_prefill_enabled()
    }

    /// Prime the KV cache for `tokens` at absolute positions `[0, tokens.len())`. Returns the number
    /// of positions primed, which is `tokens.len()` -- the caller resumes the per-token loop there.
    ///
    /// The caller must NOT include the prompt's last token: prefill produces no logits, so that one
    /// still goes through the decode ELF, which is also what leaves the KV in exactly the state a
    /// `P` sequential-step run would.
    pub(crate) fn prime(
        &self,
        arena: &FusedArena,
        embed: &EmbedTable,
        tokens: &[u32],
    ) -> Result<usize, EngineError> {
        if tokens.is_empty() {
            return Ok(0);
        }
        let d = embed.d_model();
        let hd = self.artifact.head_dim;
        let x_loc = *self.artifact.loc("x");
        let mut x = vec![0u8; self.batch * d * 2];

        for chunk in chunk_plan(tokens.len(), self.batch) {
            // Pad rows repeat the chunk's last real token rather than an arbitrary id: any token is
            // correct (the pad rows' KV lands past n_past and is masked), and repeating a real one
            // keeps the activations in-distribution, so a NaN in the padded tail is a genuine defect
            // rather than something the padding invented.
            let last = tokens[chunk.start + chunk.real - 1];
            for i in 0..self.batch {
                let tok = if i < chunk.real { tokens[chunk.start + i] } else { last };
                x[i * d * 2..(i + 1) * d * 2].copy_from_slice(&embed.row(tok)?);
            }
            arena
                .write_at(x_loc.arena, x_loc.off, &x)
                .map_err(|e| EngineError::Device(format!("write prefill x: {e}")))?;

            for (loc, theta) in &self.rope_writes {
                arena
                    .write_at(loc.arena, loc.off, &rope_block(chunk.start, self.batch, hd, *theta))
                    .map_err(|e| {
                        EngineError::Device(format!("write prefill rope table @{}: {e}", loc.off))
                    })?;
            }

            // `kv_param` is "addr"-kind (element-unit BD offset, no shift); `mask_param` is
            // "core"-kind and the firmware's UPDATE_REG convention requires the host to pre-shift
            // by 2 bits. Both values are the decode ones with `M` substituted for 1, so a prefill
            // ELF built at M=1 would be driven byte-identically to the decode ELF.
            let kv = (chunk.start * hd) as u32;
            self.res
                .write_scratchpad(self.artifact.kv_off.byte_offset, &kv.to_le_bytes())
                .map_err(|e| EngineError::Device(format!("write prefill kv_off scratchpad: {e}")))?;
            // Everything at or beyond `start + M` is masked. Within the chunk, row i must not see
            // row j > i, and a SCALAR width cannot say that -- the diagonal `[M, M]` triangle the
            // design adds is a constant buffer of the ELF's, not a host write. A prefill artifact
            // with no scalar width at all (a non-causal bring-up build) declares no `mask_param`,
            // and then there is nothing here to write.
            if let Some(mask) = self.artifact.sm_mask {
                let sm_raw = (chunk.start + self.batch) as u32;
                let sm = if mask.core { sm_raw << 2 } else { sm_raw };
                self.res
                    .write_scratchpad(mask.byte_offset, &sm.to_le_bytes())
                    .map_err(|e| EngineError::Device(format!("write prefill sm_mask scratchpad: {e}")))?;
            }

            arena.sync_input().map_err(|e| EngineError::Device(format!("sync prefill input: {e}")))?;
            self.res.dispatch().map_err(|e| EngineError::Device(format!("prefill dispatch: {e}")))?;
        }
        Ok(tokens.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan(n: usize, m: usize) -> Vec<(usize, usize, usize)> {
        chunk_plan(n, m).into_iter().map(|c| (c.start, c.real, c.pad(m))).collect()
    }

    #[test]
    fn a_prompt_shorter_than_the_batch_is_one_padded_chunk() {
        assert_eq!(plan(100, 256), vec![(0, 100, 156)]);
        assert_eq!(plan(1, 256), vec![(0, 1, 255)]);
    }

    #[test]
    fn a_prompt_exactly_the_batch_is_one_chunk_with_no_padding() {
        assert_eq!(plan(256, 256), vec![(0, 256, 0)]);
    }

    #[test]
    fn one_token_past_the_batch_costs_a_whole_second_chunk() {
        // The cost model's worst case, and the reason the design pads rather than running the
        // remainder at M=1: 255 of the second chunk's 256 rows are padding, and that is still
        // cheaper than 1 dispatch of the decode ELF, let alone 255.
        assert_eq!(plan(257, 256), vec![(0, 256, 0), (256, 1, 255)]);
    }

    #[test]
    fn a_prompt_that_does_not_divide_the_batch_pads_only_the_last_chunk() {
        assert_eq!(plan(700, 256), vec![(0, 256, 0), (256, 256, 0), (512, 188, 68)]);
    }

    #[test]
    fn an_exact_multiple_pads_nothing_anywhere() {
        assert_eq!(plan(1024, 256), vec![(0, 256, 0), (256, 256, 0), (512, 256, 0), (768, 256, 0)]);
    }

    #[test]
    fn an_empty_prompt_is_no_dispatches_at_all() {
        assert!(chunk_plan(0, 256).is_empty());
    }

    #[test]
    fn every_plan_covers_exactly_the_positions_asked_for() {
        // The property the individual cases sample: chunks are contiguous from 0, cover `n` real
        // positions in total, and no chunk is empty (an empty dispatch would write a full chunk of
        // pure padding over live KV).
        for m in [1usize, 2, 64, 256] {
            for n in 0..600usize {
                let cs = chunk_plan(n, m);
                assert_eq!(cs.len(), n.div_ceil(m), "n={n} m={m}");
                assert_eq!(cs.iter().map(|c| c.real).sum::<usize>(), n, "n={n} m={m}");
                for (i, c) in cs.iter().enumerate() {
                    assert_eq!(c.start, i * m, "n={n} m={m}");
                    assert!(c.real >= 1 && c.real <= m, "n={n} m={m} chunk {i}: {c:?}");
                }
            }
        }
    }

    #[test]
    fn the_padded_span_never_leaves_the_window_when_m_divides_s() {
        // The invariant `check_prefill_pairing`'s `S % M == 0` check buys, asserted rather than
        // argued: at any prompt the window admits, the padded span still fits.
        for (s, m) in [(2048usize, 256usize), (512, 256), (2048, 64), (1024, 1024)] {
            for n in 1..s {
                let cs = chunk_plan(n, m);
                assert!(cs.len() * m <= s, "S={s} M={m} n={n} would write {} positions", cs.len() * m);
            }
        }
    }

    // ---------------------------------------------------------------------------------------
    // RoPE. The block must be M consecutive `rope_row` outputs at the chunk's ABSOLUTE positions
    // -- an off-by-one here rotates every prompt token by one position and produces plausible
    // wrong text, so it is checked against the M=1 function rather than against a re-derivation.
    // ---------------------------------------------------------------------------------------

    const THETA: f64 = 1_000_000.0;

    #[test]
    fn a_rope_block_is_the_m1_rows_for_the_chunks_absolute_positions() {
        let (hd, batch, start) = (8usize, 4usize, 12usize);
        let block = rope_block(start, batch, hd, THETA);
        assert_eq!(block.len(), batch * hd * 2, "[M, head_dim] bf16, token-major");
        for i in 0..batch {
            let want = pack_bf16_bytes(&rope_row(start + i, hd, THETA));
            assert_eq!(&block[i * hd * 2..(i + 1) * hd * 2], &want[..], "row {i} != rope_row({})", start + i);
        }
    }

    #[test]
    fn the_first_chunks_first_row_is_position_zero() {
        let hd = 128;
        let block = rope_block(0, 2, hd, THETA);
        // pos 0 -> every angle is 0 -> cos=1, sin=0, whatever theta and head_dim are.
        let want = pack_bf16_bytes(&rope_row(0, hd, THETA));
        assert_eq!(&block[..hd * 2], &want[..]);
        assert_ne!(&block[hd * 2..], &want[..], "row 1 must be position 1, not a repeat of row 0");
    }

    #[test]
    fn chunk_two_starts_where_chunk_one_ended() {
        // The seam the chunk loop has to get right: the last row of chunk c and the first row of
        // chunk c+1 are consecutive absolute positions, not a restart.
        let (hd, m) = (8usize, 4usize);
        let c0 = rope_block(0, m, hd, THETA);
        let c1 = rope_block(m, m, hd, THETA);
        assert_eq!(&c1[..hd * 2], &pack_bf16_bytes(&rope_row(m, hd, THETA))[..]);
        assert_ne!(&c0[..hd * 2], &c1[..hd * 2]);
    }

    #[test]
    fn a_batch_of_one_is_byte_identical_to_the_decode_path() {
        // The architecture's own claim, made checkable: the decode graph is the M=1 instance of the
        // prefill graph. If this ever stops holding, the two halves have diverged in the host
        // protocol and the token-identity gate will fail for a reason no device trace will show.
        for pos in [0usize, 1, 7, 2047] {
            assert_eq!(rope_block(pos, 1, 128, THETA), pack_bf16_bytes(&rope_row(pos, 128, THETA)));
        }
    }
}
