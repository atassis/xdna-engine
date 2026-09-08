//! The real device backend for [`DecodeStep`](crate::llm::generator::DecodeStep): drives a built
//! decoder-LLM fused-decode ELF (deep-C `aiex.scratchpad_parameter` / "Option C", the same mechanism
//! `asr::whisper_decoder`'s resident path uses) through `npu-xrt`'s `ElfResident`/`FusedArena`.
//!
//! Per-token protocol, from the authoritative Python driver
//! (`designs/decode_fused/verify_llm_decode.py:99-112`):
//!   1. host gathers `embed[token] * scale` -> write to `x`
//!   2. host computes the RoPE angle row for `pos` -> write to `rope_global`
//!   3. host writes ctrl-scratchpad `kv_off = pos*head_dim` and `sm_mask = pos+1`
//!   4. one dispatch (the whole layer stack)
//!   5. read back `logits`
//!
//! **Load-bearing property, checked by construction, not by convention**: every call to
//! [`NpuDecodeStep::step`] does write -> `sync_input()` -> `dispatch()` -> `sync_from_device()`
//! unconditionally. On the IRON Python rail the first dispatch after a host input write computes on
//! the PREVIOUS input, because `tensor_class.to()` early-returns on a stale coherence map. `npu-xrt`
//! has no such map -- `Bo::sync_to_device()` is an unconditional FFI call -- so this rail does not
//! inherit the defect PROVIDED every write is followed by a real sync. Never special-case that away
//! (e.g. "skip the sync when nothing changed"): that is exactly the shortcut that reintroduces it.

use std::path::Path;
use std::rc::Rc;

use npu_xrt::{Device, ElfResident, FusedArena};

use crate::api::EngineError;
use crate::llm::artifact::{EmbedScale, LlmArtifact};
use crate::llm::generator::DecodeStep;

fn pack_bf16_bytes(f: &[f32]) -> Vec<u8> {
    let mut bits = vec![0u16; f.len()];
    npu_xrt::pack_f32_to_bf16(f, &mut bits);
    let mut out = vec![0u8; bits.len() * 2];
    for (i, &b) in bits.iter().enumerate() {
        out[2 * i..2 * i + 2].copy_from_slice(&b.to_le_bytes());
    }
    out
}

fn unpack_bf16_bytes(bytes: &[u8]) -> Vec<f32> {
    let u16s: Vec<u16> = bytes.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
    let mut out = vec![0f32; u16s.len()];
    npu_xrt::unpack_bf16_to_f32(&u16s, &mut out);
    out
}

/// One position's RoPE angle row, INTERLEAVED `[cos, sin, cos, sin, ...]` -- the convention
/// `iron/operators/rope/reference.py` documents and `verify_llm_decode.py:34`'s `rope_row` implements.
/// This is NOT mlir-air's half-split `[cos..., sin...]` packing; porting that convention here would
/// compile, dispatch, and produce plausible-looking wrong logits.
fn rope_row(pos: usize, head_dim: usize, theta: f64, rope_angles: usize) -> Vec<f32> {
    let half = head_dim / 2;
    let mut row = vec![0f32; head_dim];
    for i in 0..half {
        // Past `rope_angles` the inverse frequency is ZERO, not absent. transformers'
        // `_compute_proportional_rope_parameters` concatenates `zeros(head_dim/2 - rope_angles)`
        // onto the rotated frequencies, and a zero frequency gives ang = 0 -> cos 1, sin 0, which
        // is the identity rotation. So the row keeps its full head_dim width and the un-rotated
        // dimensions pass through -- no buffer-size change, nothing downstream to teach.
        //
        // The exponent's denominator is head_dim even when only a fraction is rotated. That is
        // what makes this rope_type "proportional" rather than the usual partial rotary, which
        // divides by the rotated width instead and so spaces its frequencies differently. Read off
        // modeling_rope_utils.py, not inferred from the `default` path.
        let inv = if i < rope_angles { 1.0 / theta.powf((2 * i) as f64 / head_dim as f64) } else { 0.0 };
        let ang = pos as f64 * inv;
        row[2 * i] = ang.cos() as f32;
        row[2 * i + 1] = ang.sin() as f32;
    }
    row
}

/// How many of a row's `head_dim/2` frequency pairs are actually rotated.
///
/// `int(partial_rotary_factor * head_dim // 2)`, the arithmetic
/// `_compute_proportional_rope_parameters` does -- 64 of 256 at Gemma-4-12B's global head_dim 512
/// and 0.25. `None` (every model shipped today) rotates all of them, which makes the partial path
/// bit-identical to the full one rather than a second arm to keep in step.
fn rope_angles(head_dim: usize, partial: Option<f64>) -> usize {
    match partial {
        Some(f) => (f * head_dim as f64 / 2.0).floor() as usize,
        None => head_dim / 2,
    }
}

/// `logits = tanh(logits/c) * c`, the checkpoint's `final_logit_softcapping`
/// (`Gemma4UnifiedForCausalLM.forward`). An LM-HEAD axis, not the attention one -- that is a
/// separate config key, taken by `eager_attention_forward`, which this decoder layer leaves unset.
///
/// Host-side rather than a kernel epilogue because it is elementwise on the one vector that has
/// already crossed back, so it costs no dispatch on the axis that dominates the step.
fn apply_logit_softcap(logits: &mut [f32], cap: Option<f64>) {
    let Some(c) = cap else { return };
    let c = c as f32;
    for l in logits.iter_mut() {
        *l = (*l / c).tanh() * c;
    }
}

/// A resident device backend for one decoder-LLM fused decode ELF. Construction registers the
/// constant ELF and loads every weight buffer ONCE; [`step`](DecodeStep::step) then costs exactly one
/// dispatch. Holds the KV cache: a single instance decodes ONE generation (`pos` only ever
/// increases). Call [`reset`](NpuDecodeStep::reset) before starting another generation on the same
/// instance -- a fresh [`NpuDecodeStep::new`] is just as correct and costs the weight reload.
pub struct NpuDecodeStep {
    artifact: LlmArtifact,
    arena: FusedArena,
    res: ElfResident,
    /// The tied `W_head` blob, mmapped: `[vocab, d_model]` bf16 row-major, which IS the embedding
    /// table -- `gen_llm_decode.py` builds `W_head` from `model.embed_tokens.weight`. Gathered one
    /// row per step, so a generation never materialises the table.
    ///
    /// bf16 here is the artifact's real precision rather than a narrowing: the checkpoint ships
    /// bf16 and `dump_llm_weights.py` widens it with `.float()`, so the f32 `.npy` this reads
    /// instead of carries no information a bf16 does not.
    embed: memmap2::Mmap,
    embed_scale: f32,
}

impl NpuDecodeStep {
    /// `decode_dir` holds `meta.json` + the ELF + `buffers/<name>.bin` (see [`LlmArtifact`]) and is
    /// the ONLY input: the host embedding gather reads the tied `W_head` blob that is already there,
    /// so the checkpoint's dumped `.npy` weights are a build input and no longer a runtime one.
    pub fn new(dev: &Rc<Device>, decode_dir: &Path) -> Result<Self, EngineError> {
        let artifact = LlmArtifact::load(decode_dir)?;
        // Mirrors this exact loop's writes below (`x_loc`, `rope_loc`) -- an artifact declaring a
        // third per-token input buffer would otherwise leave it unwritten every token, silently.
        // The write list is DERIVED from the artifact, not a literal, because it is model-shaped:
        // a global-only model (Qwen3) declares two inputs, and one with interleaved local/global
        // attention (Gemma-3) declares three. The literal `["x", "rope_global"]` was correct for
        // every model on this rail until Gemma-3, and then reported the missing `rope_local` write
        // as an artifact defect -- which is exactly what the check is for, but the fix belongs
        // here.
        let mut writes: Vec<&str> = vec!["x", "rope_global"];
        if artifact.rope_theta_local.is_some() { writes.push("rope_local"); }
        artifact.check_per_token_writes(&writes)?;

        let arena = FusedArena::new(dev, artifact.input_size, artifact.output_size, artifact.scratch_size)
            .map_err(|e| EngineError::Load(format!("alloc fused arenas: {e}")))?;

        for name in &artifact.weights {
            let bytes = std::fs::read(artifact.weight_blob_path(name))
                .map_err(|e| EngineError::Load(format!("read weight buffer {name}.bin: {e}")))?;
            let loc = artifact.loc(name);
            if bytes.len() != loc.len {
                return Err(EngineError::Load(format!(
                    "weight buffer {name}.bin is {} bytes, layout declares {}",
                    bytes.len(),
                    loc.len
                )));
            }
            arena
                .write_at(loc.arena, loc.off, &bytes)
                .map_err(|e| EngineError::Load(format!("write weight buffer {name}: {e}")))?;
        }
        // Zero every KV-cache buffer EXPLICITLY, rather than relying on its `buffers/<name>.bin`
        // happening to be an all-zero blob (today all 56 are, but nothing enforces that). This is
        // what makes cross-request reuse safe: stale entries past `n_past` are masked to -inf and
        // contribute nothing ONLY while they are finite, and a cache that starts at zero can never
        // hold anything else -- every value in it afterwards was written by the model. Load-time,
        // so it costs one memset per load rather than one per request.
        for name in &artifact.cache_buffers {
            let loc = artifact.loc(name);
            arena
                .write_at(loc.arena, loc.off, &vec![0u8; loc.len])
                .map_err(|e| EngineError::Load(format!("zero cache buffer {name}: {e}")))?;
        }

        // One bulk sync covers every buffer written above (scratch is never re-synced per token --
        // only `sync_input()` is, mirroring `asr::whisper_decoder::FusedDecoder`'s resident path).
        arena.sync_to_device().map_err(|e| EngineError::Load(format!("sync weights to device: {e}")))?;

        let elf = std::fs::read(artifact.elf_path())
            .map_err(|e| EngineError::Load(format!("read {}: {e}", artifact.elf_path().display())))?;
        let res = dev
            .open_elf_resident(&elf, Some(&artifact.kernel_name))
            .map_err(|e| EngineError::Load(format!("open_elf_resident: decode ELF lacks a ctrl scratchpad: {e}")))?;
        arena.bind_resident(&res).map_err(|e| EngineError::Load(format!("bind resident arena BOs: {e}")))?;

        // Gate on the BYTE LENGTH the layout declares, not on the file merely existing: a W_head
        // built for another vocab is the failure that would otherwise gather a wrong row quietly.
        // `meta.json`'s `embed_blob` names the bf16 table the gather reads. It is "W_head" unless
        // the lm-head was quantised, in which case W_head.bin is packed [scale|payload] rows and
        // the generator emits a bf16 sidecar for this read. Defaults to "W_head" so every artifact
        // built before that field keeps working.
        let embed_path = artifact.weight_blob_path(artifact.embed_blob());
        let want = artifact.vocab * artifact.d_model * 2;
        let f = std::fs::File::open(&embed_path)
            .map_err(|e| EngineError::Load(format!("open {}: {e}", embed_path.display())))?;
        // SAFETY: the artifact directory is owned by the engine and read-only for its lifetime; a
        // concurrent truncation would be a corrupted install, which every other blob read shares.
        let embed = unsafe { memmap2::Mmap::map(&f) }
            .map_err(|e| EngineError::Load(format!("mmap {}: {e}", embed_path.display())))?;
        if embed.len() != want {
            return Err(EngineError::Load(format!(
                "{} is {} bytes, artifact declares vocab={} d_model={} (bf16 -> {} bytes)",
                embed_path.display(), embed.len(), artifact.vocab, artifact.d_model, want
            )));
        }
        let embed_scale = match artifact.embed_scale {
            EmbedScale::None => 1.0,
            EmbedScale::SqrtDModel => (artifact.d_model as f32).sqrt(),
        };

        Ok(NpuDecodeStep { artifact, arena, res, embed, embed_scale })
    }

    /// Re-zero every KV-cache scratch buffer (`meta.json`'s `cache_buffers`) and sync. Call before
    /// each new generation on a REUSED instance; a freshly-constructed instance is already zero (the
    /// artifact's own cache-buffer blobs are all-zero) and does not need this.
    pub fn reset(&mut self) -> Result<(), EngineError> {
        // The cache buffers are ALREADY zero when the model loads: every one of them is listed in
        // `meta.json`'s `weights` too, and its `buffers/<name>.bin` is an all-zero blob, so
        // `new()`'s weight loop zeroes them and syncs once. This per-request pass exists only to
        // stop request N+1 from seeing request N's history.
        //
        // Whether it is NEEDED is a question about the mask, not about the cache: `sm_mask` is
        // written as `n_past + 1` and the attention masks every position at or beyond it to -inf,
        // so stale entries past the current position should contribute nothing. If that holds, this
        // is 224 MiB of host memset plus an arena write per request at S=2048 (56 at S=512) that
        // buys nothing -- and it is a per-REQUEST cost, so it hurts short generations most.
        //
        // DEFAULT ON since 2026-09-08. The safety condition -- that a masked-out entry is finite,
        // because a stale inf or NaN would survive `0 * v` -- is now guaranteed by CONSTRUCTION
        // rather than by argument: `new()` zeroes these buffers explicitly at load, so everything
        // they ever hold afterwards was written by the model itself.
        //
        // Deliberately NOT time-based. Nothing about a KV cache decays with age; what makes a
        // retained entry wrong is a different token at that position, and the mask already handles
        // every position past `n_past`. An idle timer would add a knob that expires correct state
        // on a clock. The resource side is already covered one level up, at the right granularity:
        // `idle_unload_s` drops the whole model and frees its 2 GB arena, cache included.
        //
        // NPU_LLM_REUSE_KV=0 restores the per-request pass, for bisecting a suspected KV bug.
        if std::env::var("NPU_LLM_REUSE_KV").ok().as_deref() != Some("0") {
            return Ok(());
        }
        for name in &self.artifact.cache_buffers {
            let loc = self.artifact.loc(name);
            self.arena
                .write_at(loc.arena, loc.off, &vec![0u8; loc.len])
                .map_err(|e| EngineError::Device(format!("zero cache buffer {name}: {e}")))?;
        }
        self.arena.sync_to_device().map_err(|e| EngineError::Device(format!("sync reset KV to device: {e}")))
    }
}

impl DecodeStep for NpuDecodeStep {
    /// The artifact's own `dims.S`. This is what makes the generator's bound real: without it the
    /// trait default is `None` and the decode loop walks `pos` past the end of the KV cache.
    fn max_context(&self) -> Option<usize> {
        Some(self.artifact.max_seq)
    }

    /// Zero every KV cache buffer. The inherent `reset` already did this; wiring it through the
    /// trait is what makes it actually run, since the generator only ever sees `dyn DecodeStep`.
    fn reset(&mut self) -> Result<(), EngineError> {
        // Zero the dispatch accounting alongside the KV cache, so a report covers exactly the
        // generation that follows and not everything the process has ever dispatched. Both are
        // no-ops unless NPU_DISPATCH_LOG is set; the log is thread-local and the engine actor is
        // one thread, which is what makes per-generation scoping meaningful at all.
        if npu_xrt::dispatch_log::enabled() {
            npu_xrt::dispatch_log::reset();
        }
        NpuDecodeStep::reset(self)
    }

    /// One decode step is one dispatch of the fused ELF, so the count here is the claim
    /// "one dispatch per token for all 28 layers" measured rather than asserted. `switch_ms` is
    /// 0.0 deliberately: this rail holds ONE xclbin, so a predicted switch tax priced off a
    /// cross-program constant would be an invented number, and transitions should read 0.
    fn dispatch_report(&self) -> Option<String> {
        npu_xrt::dispatch_log::enabled().then(|| {
            format!("{}\n{}", npu_xrt::dispatch_log::report(0.0), npu_xrt::context_report())
        })
    }

    fn step(&mut self, token: u32, pos: usize) -> Result<Vec<f32>, EngineError> {
        let tok = token as usize;
        if tok >= self.artifact.vocab {
            return Err(EngineError::Unsupported(format!("token {tok} >= vocab {}", self.artifact.vocab)));
        }

        // The row is already bf16 in exactly the layout `x` wants, so an unscaled model writes the
        // mmapped bytes straight through: no unpack, no repack. A scaled model pays a conversion on
        // one row, and the f32 it converts through is the same value the old whole-table path held,
        // so both arms stay bit-identical to it.
        let d = self.artifact.d_model;
        let row = &self.embed[tok * d * 2..(tok + 1) * d * 2];
        let scaled;
        let x_bytes: &[u8] = if self.embed_scale == 1.0 {
            row
        } else {
            let v: Vec<f32> = unpack_bf16_bytes(row).iter().map(|&e| e * self.embed_scale).collect();
            scaled = pack_bf16_bytes(&v);
            &scaled
        };
        let x_loc = self.artifact.loc("x");
        self.arena
            .write_at(x_loc.arena, x_loc.off, x_bytes)
            .map_err(|e| EngineError::Device(format!("write x: {e}")))?;

        // The row WIDTH comes from the buffer the artifact declares, not from a scalar head_dim.
        // That is the number the ELF actually reads, so the two cannot drift apart -- and it makes
        // the per-layer case free: Gemma-4-12B's global layers use head_dim 512 where its sliding
        // layers use 256, so `rope_global` and `rope_local` must differ in WIDTH and not only in
        // theta, which a single artifact.head_dim cannot express.
        let rope_loc = self.artifact.loc("rope_global");
        let rope_hd = rope_loc.len / 2;
        // Partial rotary is a GLOBAL-layer axis: Gemma-4-12B's config gives full_attention
        // rope_type "proportional" with partial_rotary_factor 0.25 and sliding_attention plain
        // "default", so only this row narrows.
        let rope = rope_row(
            pos,
            rope_hd,
            self.artifact.rope_theta_global,
            rope_angles(rope_hd, self.artifact.rope_partial_rotary),
        );
        // The width/head_dim cross-check moved to `LlmArtifact::load`, which sees BOTH angle rows
        // and every declared geometry at once. Here it could only ever compare one row against one
        // scalar, which is why it was gated on `kv_offs.len() == 1` and did nothing in the
        // per-layer case it was written for.
        let rope_bytes = pack_bf16_bytes(&rope);
        self.arena
            .write_at(rope_loc.arena, rope_loc.off, &rope_bytes)
            .map_err(|e| EngineError::Device(format!("write rope_global: {e}")))?;

        // Same row, different base. Gemma-3 interleaves local and global attention layers and the
        // ELF reads a separate angle table for each; a model without local layers has no such
        // buffer and this is skipped.
        if let Some(theta_local) = self.artifact.rope_theta_local {
            let loc = self.artifact.loc("rope_local");
            let rope_l = rope_row(pos, loc.len / 2, theta_local, rope_angles(loc.len / 2, None));
            let rope_l_bytes = pack_bf16_bytes(&rope_l);
            self.arena
                .write_at(loc.arena, loc.off, &rope_l_bytes)
                .map_err(|e| EngineError::Device(format!("write rope_local: {e}")))?;
        }

        // `kv_off` is "addr"-kind (element-unit BD offset, no shift); `sm_mask` is "core"-kind and
        // the firmware's UPDATE_REG convention requires the host to pre-shift it left by 2 bits
        // (matches `asr::whisper_decoder::FusedDecoder::dispatch_resident`).
        // ONE WRITE PER DISTINCT head_dim. `kv_offs` has a single entry on every model shipped
        // today, so this is the same single write it has always been. It is a loop because
        // Gemma-4-12B's geometry is per-layer -- sliding head_dim 256, global 512 -- and
        // `pos * head_dim` is then two different byte offsets for the same logical position, which
        // one slot cannot carry. A spec with non-uniform geometry is still refused at build time
        // (LlmSpec.check); this is the host half of lifting that refusal.
        for (slot, head_dim) in &self.artifact.kv_offs {
            let kv_val = (pos * head_dim) as u32;
            self.res
                .write_scratchpad(slot.byte_offset, &kv_val.to_le_bytes())
                .map_err(|e| EngineError::Device(format!("write kv_off scratchpad: {e}")))?;
        }
        let sm_raw = (pos + 1) as u32;
        let sm_val = if self.artifact.sm_mask.core { sm_raw << 2 } else { sm_raw };
        self.res
            .write_scratchpad(self.artifact.sm_mask.byte_offset, &sm_val.to_le_bytes())
            .map_err(|e| EngineError::Device(format!("write sm_mask scratchpad: {e}")))?;

        // Unconditional every token -- see the module doc. No arm here may skip a step "because
        // nothing changed"; that branch is exactly the defect this mirrors away from.
        self.arena.sync_input().map_err(|e| EngineError::Device(format!("sync input: {e}")))?;
        // dispatch()'s own error already names "resident dispatch"; don't prefix it twice.
        self.res.dispatch().map_err(EngineError::Device)?;
        self.arena.sync_from_device().map_err(|e| EngineError::Device(format!("sync output: {e}")))?;

        let out_loc = self.artifact.loc(&self.artifact.output);
        let mut bytes = vec![0u8; out_loc.len];
        self.arena
            .read_at(out_loc.arena, out_loc.off, &mut bytes)
            .map_err(|e| EngineError::Device(format!("read {}: {e}", self.artifact.output)))?;
        let mut logits = unpack_bf16_bytes(&bytes);
        logits.truncate(self.artifact.vocab);
        apply_logit_softcap(&mut logits, self.artifact.logit_softcap);
        Ok(logits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    // `scripts/... rope_row` cross-check (designs/decode_fused/verify_llm_decode.py:34-48),
    // theta=1e6, head_dim=8 (half=4): computed independently in Python and pasted as a literal, the
    // same cross-language-oracle convention `llm::sampling`'s tests use.
    //   >>> import numpy as np
    //   >>> half=4; theta=1e6; hd=8
    //   >>> inv = 1.0/(theta**(np.arange(0,hd,2,dtype=np.float64)[:half]/hd))
    //   >>> [float(v) for v in inv]
    #[test]
    fn rope_row_matches_python_oracle_at_pos_zero() {
        // pos=0 -> ang=0 for every pair -> cos=1, sin=0 regardless of theta/head_dim.
        let row = rope_row(0, 8, 1_000_000.0, 4);
        assert_eq!(row, vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn rope_row_matches_python_oracle_at_pos_one() {
        // inv = [1.0, 0.03162277660168379, 0.001, 3.1622776601683795e-05] (theta=1e6, head_dim=8)
        let row = rope_row(1, 8, 1_000_000.0, 4);
        let inv = [1.0f64, 0.03162277660168379, 0.001, 3.1622776601683795e-05];
        let want: Vec<f32> = inv.iter().flat_map(|&a| [a.cos() as f32, a.sin() as f32]).collect();
        assert_eq!(row, want);
    }

    /// Cross-language oracle for the "proportional" rope_type, produced by CALLING transformers
    /// rather than reimplementing its arithmetic in the test:
    ///   >>> cfg = AutoConfig.from_pretrained("unsloth/gemma-4-12b-it").get_text_config()
    ///   >>> rot = Gemma4UnifiedTextRotaryEmbedding(cfg)
    ///   >>> ang = 7 * rot.full_attention_inv_freq.double()
    ///   >>> torch.stack([ang.cos(), ang.sin()], 1).flatten()
    /// Tolerance rather than equality because that reference builds inv_freq in f32 and this builds
    /// it in f64; the gap is ~1e-8, four orders under bf16's resolution at these magnitudes.
    ///
    /// This is the test that pins the DENOMINATOR. "proportional" divides the exponent by the full
    /// head_dim while ordinary partial rotary divides by the rotated width -- at pair 1 that is
    /// 0.9475 against 0.8058, so a row built the usual way misses these literals by 0.14.
    #[test]
    fn rope_row_matches_the_hf_proportional_oracle_on_a_global_row() {
        const HD: usize = 512; // Gemma-4-12B global_head_dim
        let row = rope_row(7, HD, 1_000_000.0, rope_angles(HD, Some(0.25)));
        assert_eq!(row.len(), HD);
        let want_head = [
            0.753902254343, 0.656986598719, 0.939694868055, 0.342013968942,
            0.999999804905, 0.000624652668, 0.946202850550, -0.323574049656,
        ];
        for (i, &w) in want_head.iter().enumerate() {
            assert!((row[i] as f64 - w).abs() < 1e-6, "row[{i}] = {} want {w}", row[i]);
        }
        // Pairs 62..65 -- the last two rotated and the first two that are not.
        let want_tail = [
            0.969750772636, 0.244097191651, 0.972831560659, 0.231514048355,
            1.0, 0.0, 1.0, 0.0,
        ];
        for (i, &w) in want_tail.iter().enumerate() {
            let j = 124 + i;
            assert!((row[j] as f64 - w).abs() < 1e-6, "row[{j}] = {} want {w}", row[j]);
        }
    }

    /// Same oracle, the SLIDING row: rope_type "default", theta 1e4, head_dim 256, nothing narrowed.
    /// Included because the two rows differ in width, theta AND rotated fraction at once, and a
    /// change that fixed the global row by breaking the local one would otherwise pass.
    #[test]
    fn rope_row_matches_the_hf_oracle_on_a_sliding_row() {
        const HD: usize = 256;
        let row = rope_row(7, HD, 10_000.0, rope_angles(HD, None));
        assert_eq!(row.len(), HD);
        let want = [
            0.753902254343, 0.656986598719, 0.973479372428, 0.228774805118,
            0.975583321243, -0.219629650350, 0.800726221135, -0.599030482351,
        ];
        for (i, &w) in want.iter().enumerate() {
            assert!((row[i] as f64 - w).abs() < 1e-6, "row[{i}] = {} want {w}", row[i]);
        }
    }

    /// The un-rotated tail is the IDENTITY rotation, not zeros: transformers pads inv_freq with
    /// zeros to the full head_dim/2, and a zero frequency lands on cos 1 / sin 0. Zeros there would
    /// annihilate the un-rotated dimensions instead of passing them through.
    #[test]
    fn rope_row_past_the_rotated_angles_is_the_identity_rotation() {
        const HD: usize = 512;
        let row = rope_row(1234, HD, 1_000_000.0, rope_angles(HD, Some(0.25)));
        for i in 64..HD / 2 {
            assert_eq!((row[2 * i], row[2 * i + 1]), (1.0, 0.0), "pair {i} must be identity");
        }
        assert_ne!((row[126], row[127]), (1.0, 0.0), "pair 63 is the last ROTATED one");
    }

    /// `int(f * head_dim // 2)`, the arithmetic `_compute_proportional_rope_parameters` does.
    #[test]
    fn rope_angles_matches_the_python_floor_arithmetic() {
        assert_eq!(rope_angles(512, Some(0.25)), 64, "0.25 * 512 // 2");
        assert_eq!(rope_angles(256, Some(0.25)), 32);
        assert_eq!(rope_angles(512, Some(1.0)), 256, "a full fraction rotates everything");
        assert_eq!(rope_angles(512, None), 256);
        assert_eq!(rope_angles(128, None), 64);
    }

    /// The partial path must be BIT-identical to the full one when nothing is narrowed -- that is
    /// what makes this a generalisation rather than a second arm that can drift. Every model
    /// shipped today takes the None branch.
    #[test]
    fn a_full_fraction_is_bit_identical_to_no_fraction() {
        let hd = 128;
        let full = rope_row(97, hd, 1_000_000.0, rope_angles(hd, None));
        let frac = rope_row(97, hd, 1_000_000.0, rope_angles(hd, Some(1.0)));
        assert_eq!(full, frac);
    }

    /// tanh(x/30)*30, from torch rather than from arithmetic done here:
    ///   >>> ((torch.tensor([0.,1.,-1.,30.,500.,-500.])/30).tanh()*30).tolist()
    ///
    /// The interesting entry is 30.0 -> 22.848, not 30: the cap is the ASYMPTOTE, so a logit AT the
    /// cap is already well inside it. Reading `logit_softcap` as a clamp gets that one wrong by 7.
    #[test]
    fn logit_softcap_matches_the_python_oracle_and_is_a_no_op_when_absent() {
        let raw = [0.0f32, 1.0, -1.0, 30.0, 500.0, -500.0];
        let mut got = raw.to_vec();
        apply_logit_softcap(&mut got, Some(30.0));
        let want = [0.0f64, 0.9996297955513, -0.9996297955513, 22.84782600402832, 30.0, -30.0];
        for (i, &w) in want.iter().enumerate() {
            assert!((got[i] as f64 - w).abs() < 1e-5, "logit[{i}] = {} want {w}", got[i]);
        }
        // SATURATION IS REAL, and it is why this belongs at the head rather than at sampling time:
        // at |x/c| ~ 16 the f32 tanh returns exactly 1, so two far-apart logits both land on the
        // cap and their ORDER is gone. Applying the cap later would change what argmax picks.
        assert_eq!(got[4], 30.0, "500/30 saturates f32 tanh");

        let mut untouched = raw.to_vec();
        apply_logit_softcap(&mut untouched, None);
        assert_eq!(untouched, raw.to_vec(), "no cap declared must not touch the logits");
    }

    #[test]
    fn rope_row_length_matches_head_dim() {
        assert_eq!(rope_row(5, 128, 1_000_000.0, 64).len(), 128);
    }

    /// Diagnostic, NOT a device test: dump the exact `x`/`rope_global` BYTES this rail would write
    /// for the failing teacher-forced step (device position 9, fed token 315 -- see
    /// `device_teacher_forced_matches_oracle`'s doc comment) for an out-of-band byte-for-byte
    /// compare against `verify_llm_decode.py`'s host computation. No device access -- `x`/
    /// `rope_global` are pure functions of (token, pos, embed table, scale, theta), so this needs
    /// only the embed `.npy` on disk. SKIPs if it is absent.
    #[test]
    fn dump_x_and_rope_bytes_for_the_failing_step() {
        let (decode_dir, weights_dir, _) = gate_paths();
        let embed_path = weights_dir.join("model.embed_tokens.weight.npy");
        if !embed_path.exists() {
            eprintln!("SKIP: {} not found", embed_path.display());
            return;
        }
        // Under the repo's own target/, not a session scratchpad: the first version of this wrote
        // into one agent session's /tmp dir, which is gone the moment that session ends.
        let out_dir = repo_root().join("rust/target/llm-byte-compare");
        std::fs::create_dir_all(&out_dir).unwrap();

        let embed: Array2<f32> = ndarray_npy::read_npy(&embed_path).expect("read embed npy");
        const TOKEN: usize = 315;
        const POS: usize = 9;
        const HEAD_DIM: usize = 128;
        const THETA: f64 = 1_000_000.0;
        let embed_scale = 1.0f32; // host_protocol.embed_scale == "none" for qwen3-0.6b

        let raw_row: Vec<f32> = embed.row(TOKEN).iter().copied().collect();
        let x: Vec<f32> = raw_row.iter().map(|&v| v * embed_scale).collect();
        let x_bytes = pack_bf16_bytes(&x);
        let rope = rope_row(POS, HEAD_DIM, THETA, rope_angles(HEAD_DIM, None));
        let rope_bytes = pack_bf16_bytes(&rope);

        // The f32 `.npy` above is the ORIGINAL source; `W_head.bin` is what `step` now gathers from.
        // Asserting they agree byte-for-byte is what licenses reading the blob instead of the table:
        // the checkpoint is bf16 and the dump widens it, so the narrowing here recovers exactly the
        // bits that were there. A mismatch means the tied-W_head assumption broke for this artifact.
        let wh = decode_dir.join("buffers/W_head.bin");
        if wh.exists() {
            let blob = std::fs::read(&wh).expect("read W_head.bin");
            let d = raw_row.len();
            let row = &blob[TOKEN * d * 2..(TOKEN + 1) * d * 2];
            assert_eq!(row, &x_bytes[..], "W_head row != bf16(embed row) for token {TOKEN}");
        } else {
            eprintln!("NOTE: {} absent, skipped the W_head equivalence assert", wh.display());
        }

        std::fs::write(out_dir.join("rust_embed_row_raw.bin"), bytemuck_f32_to_le_bytes(&raw_row)).unwrap();
        std::fs::write(out_dir.join("rust_x_f32.bin"), bytemuck_f32_to_le_bytes(&x)).unwrap();
        std::fs::write(out_dir.join("rust_x_bf16.bin"), &x_bytes).unwrap();
        std::fs::write(out_dir.join("rust_rope_bf16.bin"), &rope_bytes).unwrap();

        eprintln!("rust x_bytes len={}, rope_bytes len={}", x_bytes.len(), rope_bytes.len());
        eprintln!("rust raw_row[:8]={:?}", &raw_row[..8]);
        eprintln!("wrote rust_{{embed_row_raw,x_f32,x_bf16,rope_bf16}}.bin to {}", out_dir.display());
    }

    fn bytemuck_f32_to_le_bytes(v: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(v.len() * 4);
        for &f in v {
            out.extend_from_slice(&f.to_le_bytes());
        }
        out
    }

    // ---------------------------------------------------------------------------------------
    // Device gates. Plain `cargo test` (workspace or `-p npu-engine`) must NEVER touch
    // `/dev/accel/accel0` -- both tests below SKIP unless `NPU_LLM_DEVICE_GATE=1` is set, and the
    // caller is responsible for wrapping that explicit invocation in `npu_lock.sh queue --`.
    // ---------------------------------------------------------------------------------------

    fn argmax(logits: &[f32]) -> u32 {
        let mut best = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v > best_v {
                best_v = v;
                best = i;
            }
        }
        best as u32
    }

    fn env_or(key: &str, default: &str) -> String {
        std::env::var(key).unwrap_or_else(|_| default.to_string())
    }

    /// Repo root, from this crate's manifest dir -- so the gate resolves in a worktree, a clone or
    /// a CI checkout, none of which is the machine this test was written on.
    fn repo_root() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf()
    }

    /// `(decode_dir, weights_dir, oracle_path)`, defaulting to the same `artifacts/qwen3-0.6b/`
    /// tree `scenarios/generate-qwen3-0.6b.toml` names, overridable per checkout.
    ///
    /// The default USED to name a sibling `artifacts-qwen3-0.6b/decode` outside the repo. That copy
    /// predates the 2026-09-04 re-pin and reads 7/8 teacher-forced, so the gate's own default
    /// disagreed with the scenario the service loads, and the 8/8 result only appeared when the env
    /// override happened to be set.
    fn gate_paths() -> (std::path::PathBuf, std::path::PathBuf, std::path::PathBuf) {
        let root = repo_root();
        let d = |s: &str| root.join(s).to_string_lossy().into_owned();
        (
            env_or("NPU_LLM_DECODE_DIR", &d("artifacts/qwen3-0.6b/decode")).into(),
            env_or("NPU_LLM_WEIGHTS_DIR", &d("artifacts/qwen3-0.6b/weights")).into(),
            env_or("NPU_LLM_ORACLE", &d("tests/refs/qwen3-0.6b/bf16_oracle.json")).into(),
        )
    }

    /// True (and prints why not, otherwise) iff the device gate should actually run.
    fn device_gate_enabled(decode_dir: &Path, weights_dir: &Path, oracle_path: &Path) -> bool {
        if std::env::var("NPU_LLM_DEVICE_GATE").is_err() {
            eprintln!("SKIP: set NPU_LLM_DEVICE_GATE=1 to run (opens the NPU device -- wrap with npu_lock.sh queue --)");
            return false;
        }
        for (label, p) in [("decode dir", decode_dir), ("weights dir", weights_dir), ("oracle", oracle_path)] {
            if !p.exists() {
                eprintln!("SKIP: {label} not found at {}", p.display());
                return false;
            }
        }
        true
    }

    fn load_oracle(path: &Path) -> (Vec<u32>, Vec<u32>) {
        let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let ids = |k: &str| -> Vec<u32> {
            v[k].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect()
        };
        (ids("prompt_ids"), ids("gen_ids"))
    }

    /// Mirrors `verify_llm_decode.py --teacher-force` exactly: teacher-force through the prompt,
    /// then at each generated position feed the ORACLE's token regardless of the device's own
    /// argmax, so every step is graded independently of any earlier miss.
    fn teacher_forced_run(step: &mut NpuDecodeStep, prompt_ids: &[u32], gen_ids: &[u32]) -> Vec<u32> {
        let fed = prompt_ids;
        let n_steps = gen_ids.len();
        let mut produced = Vec::with_capacity(n_steps);
        let mut tok = fed[0];
        for pos in 0..(fed.len() + n_steps - 1) {
            let logits = step.step(tok, pos).expect("device step");
            let nxt = argmax(&logits);
            if pos + 1 < fed.len() {
                tok = fed[pos + 1];
            } else {
                let i = produced.len();
                produced.push(nxt);
                tok = gen_ids[i];
            }
            if produced.len() >= n_steps {
                break;
            }
        }
        produced
    }

    /// Free-running greedy decode from `prompt_ids`: prime the KV cache over the whole prompt, then
    /// argmax-and-feed `n_gen` tokens (no oracle influence at all -- this is what a real `npu chat`
    /// request would do at `temperature: 0.0`).
    fn free_run_greedy(step: &mut NpuDecodeStep, prompt_ids: &[u32], n_gen: usize) -> Vec<u32> {
        let mut pos = 0usize;
        let mut logits = Vec::new();
        for &t in prompt_ids {
            logits = step.step(t, pos).expect("device step");
            pos += 1;
        }
        let mut produced = Vec::with_capacity(n_gen);
        let mut nxt = argmax(&logits);
        for _ in 0..n_gen {
            produced.push(nxt);
            logits = step.step(nxt, pos).expect("device step");
            pos += 1;
            nxt = argmax(&logits);
        }
        produced
    }

    /// Gate 2: teacher-forced greedy parity vs the bf16 oracle. Expect 8/8 -- the Python rail needs
    /// a second dispatch to get there (a stale coherence map on ITS rail, see the module doc); the
    /// Rust rail syncs unconditionally every token and should not need that workaround.
    ///
    /// CORRECTED 2026-09-05 -- an initial reading of this gate was WRONG and is left here so the
    /// mistake stays legible. Against the DEFAULT `NPU_LLM_DECODE_DIR`
    /// (`artifacts-qwen3-0.6b/decode/`, built 2026-09-03 18:55) this gate scores **7/8**,
    /// deterministic across 3 repeats, missing at produced-index 5 (device position 9): oracle
    /// 15344 vs NPU 279, margin 0.0203. That was first read as "a genuine bf16 knife-edge tie this
    /// rail's numerics resolve the other way from the CPU oracle" -- plausible (margin < 0.25 in
    /// `verify_llm_decode.py`'s own classification) and WRONG. The control that decided it:
    /// `designs/decode_fused/gen_llm_decode.py` regenerated FRESH against the CURRENTLY
    /// PINNED toolchain reaches **8/8** on this exact rail (no code changed here at all) --
    /// `decode.elf` shrinks 22279888 -> 20952656 bytes, a real recompile, not container-metadata
    /// noise. Root cause: `toolchain.lock`'s `MLIR_AIE_FORK_COMMIT` advanced 035528f71cf1
    /// (upstream tip 2026-08-27) -> acecda2fc53e (2026-09-02) on 2026-09-04 10:38 -- AFTER the
    /// artifact was frozen -- and that pin's own history notes this class of bump changes emitted
    /// DMA/scheduling bytes even when it isn't a correctness fix. A ~6-day upstream advance
    /// changing instruction scheduling is fully sufficient to flip a genuine 0.02-margin bf16 tie.
    /// Host-side bytes were independently verified byte-identical against `verify_llm_decode.py`
    /// (embed row, `x`, `rope_global`), so this rail's host code is not the gap.
    ///
    /// CORRECTED 2026-09-08: "a freshly generated decode dir must read 8/8" does NOT hold. A fresh
    /// 28-layer bf16 build on the current pin reads 7/8, missing the same step with a THIRD token,
    /// 9625. Frozen 279, that fresh build 15344, this one 9625 -- which is exactly the three-way
    /// tie `probe_step5_topk.py` measured at 16.7500. So step 5 does not report artifact vintage:
    /// which tied token wins moves with any change to the numerics, in either direction. Judge this
    /// gate on the other seven steps, and judge a FORMAT change on perplexity, never here.
    #[test]
    fn device_teacher_forced_matches_oracle() {
        let (decode_dir, weights_dir, oracle_path) = gate_paths();
        if !device_gate_enabled(&decode_dir, &weights_dir, &oracle_path) {
            return;
        }
        let (prompt_ids, gen_ids) = load_oracle(&oracle_path);
        let dev = Rc::new(Device::open(0).expect("open NPU device (stop other services first)"));
        let mut step = NpuDecodeStep::new(&dev, &decode_dir).expect("build NpuDecodeStep");

        let produced = teacher_forced_run(&mut step, &prompt_ids, &gen_ids);
        let matches = produced.iter().zip(&gen_ids).filter(|(a, b)| a == b).count();
        eprintln!("[gate] oracle : {gen_ids:?}");
        eprintln!("[gate] NPU    : {produced:?}");
        eprintln!("[gate] teacher-forced parity: {matches}/{}", gen_ids.len());
        assert_eq!(produced, gen_ids, "teacher-forced greedy parity {matches}/{} -- see stderr for the sequences", gen_ids.len());
    }

    /// Gate 3: the SAME prompt, decoded free-running >=5 times on one resident instance (`reset()`
    /// between runs), must produce BIT-IDENTICAL token sequences -- an accuracy metric (rel-L2) or a
    /// single run cannot see a stale readback; only repeated identical runs can.
    #[test]
    fn device_free_running_is_deterministic_across_five_runs() {
        let (decode_dir, weights_dir, oracle_path) = gate_paths();
        if !device_gate_enabled(&decode_dir, &weights_dir, &oracle_path) {
            return;
        }
        let (prompt_ids, gen_ids) = load_oracle(&oracle_path);
        let dev = Rc::new(Device::open(0).expect("open NPU device (stop other services first)"));
        let mut step = NpuDecodeStep::new(&dev, &decode_dir).expect("build NpuDecodeStep");

        const N_RUNS: usize = 5;
        let mut seqs = Vec::with_capacity(N_RUNS);
        for i in 0..N_RUNS {
            step.reset().expect("reset KV cache");
            let seq = free_run_greedy(&mut step, &prompt_ids, gen_ids.len());
            eprintln!("[gate] run {i}: {seq:?}");
            seqs.push(seq);
        }
        let identical = seqs.iter().filter(|s| *s == &seqs[0]).count();
        eprintln!("[gate] determinism: {identical}/{N_RUNS} runs bit-identical to run 0");
        assert_eq!(identical, N_RUNS, "runs diverged: {seqs:?}");
    }
}
