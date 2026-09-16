//! Batched prefill: prime the KV cache for a prompt `M` positions per dispatch instead of one.
//!
//! Two decisions from the batched-prefill design are what this file implements, and both are
//! load-bearing:
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
//!   3. the causal widths buffer <- `[q_heads*M]` int32, row `h*M + i` = `start + i + 1` clamped
//!      to the window. At M=1 that vector is one value per head, all equal to `pos + 1`, which is
//!      what the decode ELF writes as its scalar `sm_mask` -- the same mask, one dimension down.
//!   4. scratchpad `kv_param`   = `start * head_dim`  (element-unit BD offset; `pos * head_dim` at M=1)
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
use crate::llm::artifact::{BufLoc, LlmArtifact, MaskWidths, RopeWrite};
use crate::llm::npu_decode::{pack_bf16_bytes, rope_angles, rope_row, EmbedTable};

/// `NPU_LLM_PREFILL_BATCHED` -- the one accessor (E003 of the env-flag contract).
///
/// Default **ON** since 2026-09-09 (`=0` opts out), the `not_zero` shape `NPU_LLM_REUSE_KV` next
/// door uses.
///
/// It was opt-in until its gate existed, and the condition written here for flipping it was "the
/// gate passing, not before it". That gate now exists and passes. What was missing was not a
/// measurement but a SUBJECT: `--tier2` drives `verify_llm_decode.py`, which is decode-only by its
/// own header, so the end-to-end gate had never run this path at all.
///
/// `scripts/gate_llm.sh --tier2-prefill` does, at seven prompt geometries (64 under a chunk, 255
/// pad-1, 256 exact, 257 the first cross-chunk handoff, 512, 600 ragged, 768) against a float32
/// reference, both arms:
///
///   * 14/14 PASS on the adopted rule -- top-5 inclusion at the first divergence.
///   * TEACHER-FORCED, which is the stronger statement: put on the reference's own trajectory so
///     all 32 steps are independently comparable rather than only the first divergence, the
///     reference token is in the device's top-5 at **32/32 steps at every length in both arms**,
///     and is top-1 28-31/32 batched against 30-32/32 per-token.
///
/// The step-0 logit deltas that justified the old default are still there and are still not
/// identity -- they are the cross-implementation cascade
/// -- two independent implementations of one op cannot agree bit for bit, and greedy decode turns
/// any difference into a token flip wherever the top two logits are close. That is the standard this
/// rail retired and which nobody in this domain gates on. What changed is that the distribution is
/// now measured not to have moved.
///
/// Worth it: 35.8-37.1x faster priming, 707-742 tok/s against 49.9-51.5 ms/token, measured with an
/// alternated control. And decode is not perturbed by the pair being resident -- the control arm,
/// which loads both ELFs and primes per token, reproduces the standing per-token figure.
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

/// The dispatches covering positions `[start, n)`, each padded to `batch`.
///
/// Total device positions written is `ceil((n-start)/batch) * batch`, which is why the pairing
/// check requires `S % M == 0`: at `n <= S-1` that product can then never exceed the KV window.
///
/// `start` was pinned to 0 until the prefix ledger existed. Nothing ELSE on this path had to change
/// for it, because everything downstream already takes an absolute position: `PrefillChunk::start`
/// is documented as one, and `rope_block`, `mask_widths_block` and `kv_off` are each computed from
/// it. Only the plan's own origin was the assumption.
pub fn chunk_plan(start: usize, n: usize, batch: usize) -> Vec<PrefillChunk> {
    assert!(batch > 0, "prefill batch must be non-zero (dims.M is validated at load)");
    assert!(start <= n, "prefill plan start {start} is past its end {n}");
    (0..(n - start).div_ceil(batch))
        .map(|c| {
            let at = start + c * batch;
            PrefillChunk { start: at, real: (n - at).min(batch) }
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
pub fn rope_block(
    start: usize, batch: usize, head_dim: usize, theta: f64, partial: Option<f64>,
) -> Vec<u8> {
    let angles = rope_angles(head_dim, partial);
    let mut out = Vec::with_capacity(batch * head_dim * 2);
    for i in 0..batch {
        out.extend_from_slice(&pack_bf16_bytes(&rope_row(start + i, head_dim, theta, angles)));
    }
    out
}

/// The chunk's causal mask: `[heads*M]` little-endian int32, row `h*M + i` holding the number of
/// positions token `i` may attend.
///
/// Row `i` of a chunk at absolute position `start` attends positions `<= start + i`, so its width
/// is `start + i + 1` -- independent of the head, which changes only WHICH scores the row is over.
/// Clamped into `[1, max_seq]`: `mask_bf16` loops `for (j = width; j < cols; j++)` over the raw
/// i32, so a width past the window masks nothing and a width of 0 leaves the row all -inf, whose
/// softmax is NaN rather than a small number. The upper clamp is reachable -- the last chunk of a
/// full window ends at `start + M - 1 == S - 1`.
///
/// Pad rows carry a real width, not a sentinel. They are the chunk's LAST rows, so their widths
/// are the largest ones and they attend the real tokens before them; nothing reads their output,
/// and their KV lands past `n_past` where the decode mask already kills it.
pub fn mask_widths_block(start: usize, batch: usize, heads: usize, max_seq: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(heads * batch * 4);
    for _ in 0..heads {
        for i in 0..batch {
            out.extend_from_slice(&((start + i + 1).clamp(1, max_seq) as u32).to_le_bytes());
        }
    }
    out
}

/// The highest position a prefill ELF of window `max_seq` can prime in batches of `batch`.
///
/// Batch-aligned DOWN, because a chunk writes `batch` consecutive positions from a single `kv_off`
/// and so cannot be shortened to fit under `max_seq` -- it runs whole or not at all. A caller that
/// primed past this would get rows whose causal width `mask_widths_block` clamps, and a clamped
/// width masks nothing.
pub fn prefill_window(max_seq: usize, batch: usize) -> usize {
    if batch == 0 { 0 } else { (max_seq / batch) * batch }
}

/// How far batched prefill may run, given each declared geometry's own KV capacity.
///
/// A narrowed (circular) geometry holds only its last `capacity` positions and the mask this ELF
/// can express is a PREFIX -- `mask_bf16` masks `[width, cols)`. While `start + batch <= capacity`
/// the valid slots ARE a prefix and one width names them. Past it they are a circular interval:
/// after a chunk at `start >= capacity` the buffer holds `start-capacity+batch .. start+batch-1`,
/// and row `i` must attend all of them except slots `i+1 ..= batch-1`, a hole in the MIDDLE that
/// no suffix mask reaches. So batched prefill stops at the narrowest capacity and the caller
/// finishes stepwise, which is why this returns a position rather than refusing.
pub fn batchable_window(max_seq: usize, batch: usize, capacities: &[usize]) -> usize {
    prefill_window(capacities.iter().copied().chain([max_seq]).min().unwrap_or(max_seq), batch)
}

/// Whether a chunk `[start, start + batch)` crosses a capacity-`capacity` geometry's wrap point.
/// One `kv_off` write drives one dispatch of `batch` CONSECUTIVE positions; a chunk that straddles
/// the wrap would need a second, shorter dispatch into the buffer's start, which the compiled ELF
/// has no way to issue. Gemma-4's shipped shape (capacity=1024, batch=256) never crosses, because
/// `capacity % batch == 0` puts every chunk boundary on a multiple of the capacity -- this is the
/// general check for a build where that stops holding.
fn crosses_wrap_point(start: usize, batch: usize, capacity: usize) -> bool {
    start % capacity + batch > capacity
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
    rope_writes: Vec<RopeWrite>,
    /// Where the causal widths go and how many heads they repeat over, resolved once at open for
    /// the same reason `rope_writes` is. `None` on a non-causal bring-up build, which declares no
    /// widths buffer and masks nothing.
    mask_write: Option<(BufLoc, MaskWidths)>,
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
        let mask_write =
            artifact.mask_widths.clone().map(|mw| (*artifact.loc(&mw.buffer), mw));
        let elf = std::fs::read(artifact.elf_path())
            .map_err(|e| EngineError::Load(format!("read {}: {e}", artifact.elf_path().display())))?;
        let res = dev.open_elf_resident(&elf, Some(&artifact.kernel_name)).map_err(|e| {
            EngineError::Load(format!("open_elf_resident: prefill ELF lacks a ctrl scratchpad: {e}"))
        })?;
        arena
            .bind_resident(&res)
            .map_err(|e| EngineError::Load(format!("bind prefill resident to the shared arena: {e}")))?;
        let batch = artifact.batch;
        Ok(NpuPrefill { artifact, res, batch, rope_writes, mask_write })
    }

    pub fn batch(&self) -> usize {
        self.batch
    }

    /// `self.artifact`'s measured crossover, or `None` on a pre-2026-09-11 artifact -- see
    /// [`LlmArtifact::prefill_break_even_tokens`]'s doc comment.
    pub fn break_even_tokens(&self) -> Option<usize> {
        self.artifact.prefill_break_even_tokens
    }

    pub(crate) fn batched_enabled(&self) -> bool {
        batched_prefill_enabled()
    }

    /// Prime the KV cache for `tokens[from..]` at absolute positions `[from, tokens.len())`.
    /// Returns the number of positions now primed, which the caller resumes the per-token loop at.
    /// That is `tokens.len()` for a prompt inside this ELF's window and the batch-aligned window
    /// otherwise -- see the `window` binding below for why a long prompt is declined rather than
    /// truncated or clamped.
    ///
    /// `tokens` is the WHOLE prompt prefix, not the tail: `from` shifts the plan's origin only, so
    /// `tokens[chunk.start + i]` keeps indexing the prompt by absolute position and the KV position
    /// a row lands at is the same number as its index. Passing a pre-sliced tail would make those
    /// two disagree, which is exactly the bug the ledger could introduce.
    ///
    /// The caller must NOT include the prompt's last token: prefill produces no logits, so that one
    /// still goes through the decode ELF, which is also what leaves the KV in exactly the state a
    /// `P` sequential-step run would.
    pub(crate) fn prime(
        &self,
        arena: &FusedArena,
        embed: &EmbedTable,
        tokens: &[u32],
        from: usize,
    ) -> Result<usize, EngineError> {
        // `from` MUST be batch-aligned, and this is a correctness check, not a tidiness one.
        //
        // One `kv_off` is written per chunk and the device then writes `batch` CONSECUTIVE
        // positions from it. Blocked, a head's positions are contiguous only inside a block
        // (`kv_layout`: `[S/T blocks, Hkv heads, T positions, HD dims]`), so a chunk that straddles
        // a block boundary primes the right bytes at the wrong addresses -- the exact failure the
        // `kv_off` call below documents having already been fixed once.
        //
        // Before the prefix ledger, every chunk started at a multiple of `batch` and the pairing
        // check's `S % M == 0` made that sufficient. An arbitrary `from` reintroduces the straddle,
        // so the caller aligns and this refuses rather than trusting it: silent KV corruption reads
        // as a model that has got worse, not as a bug.
        if from % self.batch != 0 {
            return Err(EngineError::Unsupported(format!(
                "prefill resume point {from} is not a multiple of the prefill batch {}; a chunk \
                 would straddle a KV block and prime at the wrong addresses",
                self.batch
            )));
        }
        if tokens.len() <= from {
            return Ok(tokens.len());
        }
        // How far this ELF can prime, batch-aligned DOWN. Two separate reasons, and neither is a
        // tidiness bound:
        //
        // `mask_widths_block` clamps a row's causal width to `max_seq`, and a clamped width masks
        // NOTHING -- the row attends every position in the window instead of only those at or
        // before itself. It is silent, it is correct-looking, and it only appears for prompts long
        // enough to reach the clamp.
        //
        // And a chunk writes `batch` CONSECUTIVE positions from one `kv_off`, so a chunk crossing
        // `max_seq` cannot be shortened to fit: it runs whole or not at all.
        //
        // The bound is per GEOMETRY, not the build-wide `max_seq` -- see `batchable_window`. On
        // Gemma-4-12B that is 1024, not S=6912, and the decline it used to describe as unreachable
        // is now the normal path for a prompt over the sliding window.
        let caps: Vec<usize> = self.artifact.kv_windows.iter().map(|&(_, _, c, _)| c).collect();
        let window = batchable_window(self.artifact.max_seq, self.batch, &caps);
        let end = tokens.len().min(window);
        if end <= from {
            return Ok(from);
        }
        let d = embed.d_model();
        let hd = self.artifact.head_dim;
        let x_loc = *self.artifact.loc("x");
        let mut x = vec![0u8; self.batch * d * 2];

        for chunk in chunk_plan(from, end, self.batch) {
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

            // `w.width / batch` and `w.partial`, both per TABLE: Gemma-4-12B's global table is
            // 512 wide and partially rotated against a sliding 256 that is not, so the scalar
            // `dims.head_dim` short-writes every global row by half and `None` rotates pairs the
            // model leaves as identity. Same two numbers decode reads off `RopeWrite`.
            for w in &self.rope_writes {
                let block =
                    rope_block(chunk.start, self.batch, w.width / self.batch, w.theta, w.partial);
                arena
                    .write_at(w.loc.arena, w.loc.off, &block)
                    .map_err(|e| {
                        EngineError::Device(format!("write prefill rope table @{}: {e}", w.loc.off))
                    })?;
            }

            if let Some((loc, mw)) = &self.mask_write {
                let widths =
                    mask_widths_block(chunk.start, self.batch, mw.heads, self.artifact.max_seq);
                arena.write_at(loc.arena, loc.off, &widths).map_err(|e| {
                    EngineError::Device(format!("write prefill {}: {e}", mw.buffer))
                })?;
            }

            // `kv_param` is "addr"-kind (element-unit BD offset, no shift). Both values are the
            // decode ones with `M` substituted for 1, so a prefill ELF built at M=1 would be
            // driven byte-identically to the decode ELF.
            //
            // ONE write per DISTINCT geometry, to that geometry's OWN slot, via the same
            // `kv_off_circular` decode drives its per-geometry slots through -- see its module
            // doc. `kv_windows` is empty only for an artifact built before per-geometry capacity
            // existed (`check_prefill_pairing` would already have refused a narrowed decode
            // paired with one), which keeps the flat single-slot write below unchanged for it.
            if self.artifact.kv_windows.is_empty() {
                // The BLOCKED offset, via the same helper decode uses. This was `chunk.start *
                // hd`, the flat formula: correct while the cache was [Hkv, S, HD] and silently
                // wrong once decode blocked it, because prefill then primed the right bytes at
                // the wrong addresses. At kv_block == max_seq the helper returns exactly `pos *
                // head_dim`, so the flat path is unchanged.
                let kv = crate::llm::kv_layout::kv_off(
                    chunk.start, self.artifact.kv_block, hd, self.artifact.kv_heads) as u32;
                self.res
                    .write_scratchpad(self.artifact.kv_off.byte_offset, &kv.to_le_bytes())
                    .map_err(|e| EngineError::Device(format!("write prefill kv_off scratchpad: {e}")))?;
            } else {
                for &(slot, head_dim, capacity, _) in &self.artifact.kv_windows {
                    if crosses_wrap_point(chunk.start, self.batch, capacity) {
                        return Err(EngineError::Unsupported(format!(
                            "prefill chunk at {} spans the wrap point of a capacity-{capacity} \
                             geometry (head_dim={head_dim}): one dispatch cannot split across it",
                            chunk.start
                        )));
                    }
                    // `kv_heads` below is the artifact's one scalar and it reaches the address
                    // only through `block_stride`, so it is right exactly while this geometry is
                    // ONE block -- true of every build today, and not of a blocked cache on a
                    // model whose geometries differ in kv_heads (Gemma-4: 8 sliding, 1 global).
                    // The per-geometry count is in neither meta, so refuse rather than address
                    // with the base geometry's.
                    if self.artifact.kv_windows.len() > 1 && self.artifact.kv_block < capacity {
                        return Err(EngineError::Unsupported(format!(
                            "capacity-{capacity} geometry (head_dim={head_dim}) is blocked at \
                             kv_block={}, so its address needs that geometry's own kv_heads and \
                             the artifact declares only {}",
                            self.artifact.kv_block, self.artifact.kv_heads
                        )));
                    }
                    let kv = crate::llm::kv_layout::kv_off_circular(
                        chunk.start, capacity, self.artifact.kv_block, head_dim,
                        self.artifact.kv_heads,
                    ) as u32;
                    self.res
                        .write_scratchpad(slot.byte_offset, &kv.to_le_bytes())
                        .map_err(|e| EngineError::Device(format!("write prefill kv_off scratchpad: {e}")))?;
                }
            }
            // A SCALAR width cannot express causality within a chunk (row i must not see row
            // j > i), which is why the causal arm streams the per-row vector above instead and
            // declares no `mask_param`. This branch survives for the degenerate build that has a
            // scalar and no vector -- a prefill ELF at M=1, driven byte-identically to decode.
            // `artifact.rs` refuses an artifact carrying both.
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
        Ok(end)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan(n: usize, m: usize) -> Vec<(usize, usize, usize)> {
        chunk_plan(0, n, m).into_iter().map(|c| (c.start, c.real, c.pad(m))).collect()
    }

    #[test]
    fn the_window_is_the_batch_aligned_floor_of_max_seq() {
        // What the pairing check guarantees today: S % M == 0, so nothing is ever declined.
        assert_eq!(prefill_window(4096, 256), 4096);
        assert_eq!(prefill_window(2048, 256), 2048);
        // Gemma-4's largest legal sliding window is 2040, which is NOT a multiple of 256. The last
        // chunk would cover 1792..2048 and run 8 positions past the cache, so it does not run.
        assert_eq!(prefill_window(2040, 256), 1792);
        // A window shorter than one batch primes nothing: there is no whole chunk to run.
        assert_eq!(prefill_window(200, 256), 0);
    }

    #[test]
    fn a_narrowed_geometry_floors_the_batchable_window_below_max_seq() {
        // The served Gemma-4-12B pair: S=6912, M=256, sliding capacity 1024 against global 6912.
        // Batched prefill covers the first 1024 positions; the rest is the caller's stepwise loop.
        assert_eq!(batchable_window(6912, 256, &[1024, 6912]), 1024);
        // No narrowed geometry, and no declared geometry at all (a pre-kv_windows artifact), are
        // both the unfloored window.
        assert_eq!(batchable_window(6912, 256, &[6912, 6912]), 6912);
        assert_eq!(batchable_window(6912, 256, &[]), 6912);
        // The floor is batch-aligned like every other window: a 1000-position capacity admits
        // three whole chunks, not three and a fragment.
        assert_eq!(batchable_window(6912, 256, &[1000]), 768);
    }

    #[test]
    fn gemma4s_shipped_capacity_and_batch_never_cross_the_wrap_point() {
        // capacity=1024, batch=256: 1024 % 256 == 0, so every chunk boundary is a multiple of the
        // capacity and no chunk's span reaches past it.
        for start in [0, 256, 512, 768, 1024, 1280, 6656] {
            assert!(!crosses_wrap_point(start, 256, 1024), "start={start}");
        }
    }

    #[test]
    fn a_batch_that_does_not_divide_capacity_can_cross() {
        let capacity = 1024;
        // [900, 1200) straddles the 1024 boundary.
        assert!(crosses_wrap_point(900, 300, capacity));
        // [600, 900) stays inside it.
        assert!(!crosses_wrap_point(600, 300, capacity));
        // The chunk landing exactly on the boundary is the edge case: [724, 1024) ends AT
        // capacity, not past it.
        assert!(!crosses_wrap_point(724, 300, capacity));
        assert!(crosses_wrap_point(725, 300, capacity));
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
        assert!(chunk_plan(0, 0, 256).is_empty());
    }

    #[test]
    fn every_plan_covers_exactly_the_positions_asked_for() {
        // The property the individual cases sample: chunks are contiguous from 0, cover `n` real
        // positions in total, and no chunk is empty (an empty dispatch would write a full chunk of
        // pure padding over live KV).
        for m in [1usize, 2, 64, 256] {
            for n in 0..600usize {
                let cs = chunk_plan(0, n, m);
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
                let cs = chunk_plan(0, n, m);
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
        let block = rope_block(start, batch, hd, THETA, None);
        assert_eq!(block.len(), batch * hd * 2, "[M, head_dim] bf16, token-major");
        for i in 0..batch {
            let want = pack_bf16_bytes(&rope_row(start + i, hd, THETA, rope_angles(hd, None)));
            assert_eq!(&block[i * hd * 2..(i + 1) * hd * 2], &want[..], "row {i} != rope_row({})", start + i);
        }
    }

    #[test]
    fn the_first_chunks_first_row_is_position_zero() {
        let hd = 128;
        let block = rope_block(0, 2, hd, THETA, None);
        // pos 0 -> every angle is 0 -> cos=1, sin=0, whatever theta and head_dim are.
        let want = pack_bf16_bytes(&rope_row(0, hd, THETA, rope_angles(hd, None)));
        assert_eq!(&block[..hd * 2], &want[..]);
        assert_ne!(&block[hd * 2..], &want[..], "row 1 must be position 1, not a repeat of row 0");
    }

    #[test]
    fn chunk_two_starts_where_chunk_one_ended() {
        // The seam the chunk loop has to get right: the last row of chunk c and the first row of
        // chunk c+1 are consecutive absolute positions, not a restart.
        let (hd, m) = (8usize, 4usize);
        let c0 = rope_block(0, m, hd, THETA, None);
        let c1 = rope_block(m, m, hd, THETA, None);
        assert_eq!(&c1[..hd * 2], &pack_bf16_bytes(&rope_row(m, hd, THETA, rope_angles(hd, None)))[..]);
        assert_ne!(&c0[..hd * 2], &c1[..hd * 2]);
    }

    // ---------------------------------------------------------------------------------------
    // The causal mask. It is the whole correctness gap the batched path had, and it is data --
    // one int32 per softmax row -- so it is checkable here without a device.
    // ---------------------------------------------------------------------------------------

    fn widths(start: usize, batch: usize, heads: usize, s: usize) -> Vec<u32> {
        mask_widths_block(start, batch, heads, s)
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn the_first_chunks_row_i_attends_exactly_i_plus_one_positions() {
        let w = widths(0, 8, 1, 2048);
        assert_eq!(w, vec![1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn a_later_chunk_carries_its_absolute_base() {
        // The off-by-one that would look like a working mask: widths restarting at 1 per chunk
        // would hide every token before position 512 from the whole chunk.
        let w = widths(512, 4, 1, 2048);
        assert_eq!(w, vec![513, 514, 515, 516]);
    }

    #[test]
    fn the_widths_repeat_once_per_head_in_head_major_order() {
        // Row r = h*M + i, which is the flattening the [q_heads*M, S] scores buffer already uses.
        let (m, heads) = (4usize, 3usize);
        let w = widths(16, m, heads, 2048);
        assert_eq!(w.len(), heads * m);
        for h in 0..heads {
            assert_eq!(&w[h * m..(h + 1) * m], &[17, 18, 19, 20], "head {h}");
        }
    }

    #[test]
    fn the_block_is_four_bytes_per_softmax_row() {
        // AIERuntimeArgSpec defaults to bfloat16 and IRON sizes the device buffer off that dtype,
        // so a widths buffer allocated at 2 bytes/row takes half of this and the rest lands in
        // whatever follows it in the input arena.
        assert_eq!(mask_widths_block(0, 256, 16, 2048).len(), 16 * 256 * 4);
    }

    #[test]
    fn widths_never_run_past_the_window_or_down_to_zero() {
        // `mask_bf16` loops `for (j = width; j < cols; j++)` over the raw i32: past S it masks
        // nothing, and at 0 the row is all -inf and its softmax is NaN, not a small number.
        let (s, m) = (2048usize, 256usize);
        let last = widths(s - m, m, 1, s);
        assert_eq!(*last.last().unwrap(), s as u32, "the final row of a full window sees all of it");
        for start in [0usize, 256, 1024, s - m, s] {
            for w in widths(start, m, 2, s) {
                assert!(w >= 1 && w <= s as u32, "start={start} width={w} outside [1, {s}]");
            }
        }
    }

    #[test]
    fn every_chunk_of_a_plan_masks_the_positions_that_plan_covers() {
        // The property the individual cases sample, tied to the chunk plan that produces the
        // starts: across a whole prompt, chunk c's row i is width c*M + i + 1 and the widths are
        // strictly increasing from 1 -- never restarting, never skipping a position.
        let (s, m) = (2048usize, 64usize);
        let mut want = 1u32;
        for c in chunk_plan(0, 512, m) {
            for w in widths(c.start, m, 1, s) {
                assert_eq!(w, want, "chunk at {}", c.start);
                want += 1;
            }
        }
        assert_eq!(want, 513, "8 chunks of 64 cover positions 1..=512");
    }

    #[test]
    fn a_batch_of_one_masks_exactly_what_the_decode_step_masks() {
        // The architecture's claim again, now for the mask: decode writes the scalar `pos + 1`,
        // and the M=1 instance of this vector is that value, once per head.
        for pos in [0usize, 1, 7, 2047] {
            assert_eq!(widths(pos, 1, 4, 2048), vec![pos as u32 + 1; 4]);
        }
    }

    #[test]
    fn a_batch_of_one_is_byte_identical_to_the_decode_path() {
        // The architecture's own claim, made checkable: the decode graph is the M=1 instance of the
        // prefill graph. If this ever stops holding, the two halves have diverged in the host
        // protocol and the token-identity gate will fail for a reason no device trace will show.
        for pos in [0usize, 1, 7, 2047] {
            assert_eq!(rope_block(pos, 1, 128, THETA, None), pack_bf16_bytes(&rope_row(pos, 128, THETA, rope_angles(128, None))));
        }
    }
}
