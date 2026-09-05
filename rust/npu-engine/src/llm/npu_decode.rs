//! The real device backend for [`DecodeStep`](crate::llm::generator::DecodeStep): drives a built
//! decoder-LLM fused-decode ELF (deep-C `aiex.scratchpad_parameter` / "Option C", the same mechanism
//! `asr::whisper_decoder`'s resident path uses) through `npu-xrt`'s `ElfResident`/`FusedArena`.
//!
//! Per-token protocol, from the authoritative Python driver
//! (`route_b_kernels/decode_fused/verify_llm_decode.py:99-112`):
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

use ndarray::Array2;
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
fn rope_row(pos: usize, head_dim: usize, theta: f64) -> Vec<f32> {
    let half = head_dim / 2;
    let mut row = vec![0f32; head_dim];
    for i in 0..half {
        let inv = 1.0 / theta.powf((2 * i) as f64 / head_dim as f64);
        let ang = pos as f64 * inv;
        row[2 * i] = ang.cos() as f32;
        row[2 * i + 1] = ang.sin() as f32;
    }
    row
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
    /// `[vocab, d_model]` f32, for the host embedding gather (`step`'s `embed[token] * scale`). Kept
    /// as f32 (not the on-device bf16 `W_head` blob) because the host does this lookup in the
    /// artifact's own declared precision path -- HF weights are f32, exactly like
    /// `verify_llm_decode.py`'s `embed_tokens.weight.npy` load.
    embed: Array2<f32>,
    embed_scale: f32,
}

impl NpuDecodeStep {
    /// `decode_dir` holds `meta.json` + the ELF + `buffers/<name>.bin` (see [`LlmArtifact`]);
    /// `weights_dir` holds the checkpoint's dumped `.npy` weights, for `model.embed_tokens.weight.npy`
    /// (the host-side embedding gather -- everything else the device needs is already in
    /// `decode_dir/buffers`).
    pub fn new(dev: &Rc<Device>, decode_dir: &Path, weights_dir: &Path) -> Result<Self, EngineError> {
        let artifact = LlmArtifact::load(decode_dir)?;
        // Mirrors this exact loop's writes below (`x_loc`, `rope_loc`) -- an artifact declaring a
        // third per-token input buffer would otherwise leave it unwritten every token, silently.
        artifact.check_per_token_writes(&["x", "rope_global"])?;

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
        // One bulk sync covers every buffer written above (scratch is never re-synced per token --
        // only `sync_input()` is, mirroring `asr::whisper_decoder::FusedDecoder`'s resident path).
        arena.sync_to_device().map_err(|e| EngineError::Load(format!("sync weights to device: {e}")))?;

        let elf = std::fs::read(artifact.elf_path())
            .map_err(|e| EngineError::Load(format!("read {}: {e}", artifact.elf_path().display())))?;
        let res = dev
            .open_elf_resident(&elf, Some(&artifact.kernel_name))
            .map_err(|e| EngineError::Load(format!("open_elf_resident: decode ELF lacks a ctrl scratchpad: {e}")))?;
        arena.bind_resident(&res).map_err(|e| EngineError::Load(format!("bind resident arena BOs: {e}")))?;

        let embed_path = weights_dir.join("model.embed_tokens.weight.npy");
        let embed: Array2<f32> = ndarray_npy::read_npy(&embed_path)
            .map_err(|e| EngineError::Load(format!("read {}: {e}", embed_path.display())))?;
        if embed.shape() != [artifact.vocab, artifact.d_model] {
            return Err(EngineError::Load(format!(
                "{} is {:?}, artifact declares vocab={} d_model={}",
                embed_path.display(),
                embed.shape(),
                artifact.vocab,
                artifact.d_model
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

        let x: Vec<f32> = self.embed.row(tok).iter().map(|&v| v * self.embed_scale).collect();
        let x_loc = self.artifact.loc("x");
        self.arena
            .write_at(x_loc.arena, x_loc.off, &pack_bf16_bytes(&x))
            .map_err(|e| EngineError::Device(format!("write x: {e}")))?;

        let rope = rope_row(pos, self.artifact.head_dim, self.artifact.rope_theta_global);
        let rope_loc = self.artifact.loc("rope_global");
        self.arena
            .write_at(rope_loc.arena, rope_loc.off, &pack_bf16_bytes(&rope))
            .map_err(|e| EngineError::Device(format!("write rope_global: {e}")))?;

        // `kv_off` is "addr"-kind (element-unit BD offset, no shift); `sm_mask` is "core"-kind and
        // the firmware's UPDATE_REG convention requires the host to pre-shift it left by 2 bits
        // (matches `asr::whisper_decoder::FusedDecoder::dispatch_resident`).
        let kv_val = (pos * self.artifact.head_dim) as u32;
        self.res
            .write_scratchpad(self.artifact.kv_off.byte_offset, &kv_val.to_le_bytes())
            .map_err(|e| EngineError::Device(format!("write kv_off scratchpad: {e}")))?;
        let sm_raw = (pos + 1) as u32;
        let sm_val = if self.artifact.sm_mask.core { sm_raw << 2 } else { sm_raw };
        self.res
            .write_scratchpad(self.artifact.sm_mask.byte_offset, &sm_val.to_le_bytes())
            .map_err(|e| EngineError::Device(format!("write sm_mask scratchpad: {e}")))?;

        // Unconditional every token -- see the module doc. No arm here may skip a step "because
        // nothing changed"; that branch is exactly the defect this mirrors away from.
        self.arena.sync_input().map_err(|e| EngineError::Device(format!("sync input: {e}")))?;
        self.res.dispatch().map_err(|e| EngineError::Device(format!("resident dispatch: {e}")))?;
        self.arena.sync_from_device().map_err(|e| EngineError::Device(format!("sync output: {e}")))?;

        let out_loc = self.artifact.loc(&self.artifact.output);
        let mut bytes = vec![0u8; out_loc.len];
        self.arena
            .read_at(out_loc.arena, out_loc.off, &mut bytes)
            .map_err(|e| EngineError::Device(format!("read {}: {e}", self.artifact.output)))?;
        let mut logits = unpack_bf16_bytes(&bytes);
        logits.truncate(self.artifact.vocab);
        Ok(logits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // `scripts/... rope_row` cross-check (route_b_kernels/decode_fused/verify_llm_decode.py:34-48),
    // theta=1e6, head_dim=8 (half=4): computed independently in Python and pasted as a literal, the
    // same cross-language-oracle convention `llm::sampling`'s tests use.
    //   >>> import numpy as np
    //   >>> half=4; theta=1e6; hd=8
    //   >>> inv = 1.0/(theta**(np.arange(0,hd,2,dtype=np.float64)[:half]/hd))
    //   >>> [float(v) for v in inv]
    #[test]
    fn rope_row_matches_python_oracle_at_pos_zero() {
        // pos=0 -> ang=0 for every pair -> cos=1, sin=0 regardless of theta/head_dim.
        let row = rope_row(0, 8, 1_000_000.0);
        assert_eq!(row, vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn rope_row_matches_python_oracle_at_pos_one() {
        // inv = [1.0, 0.03162277660168379, 0.001, 3.1622776601683795e-05] (theta=1e6, head_dim=8)
        let row = rope_row(1, 8, 1_000_000.0);
        let inv = [1.0f64, 0.03162277660168379, 0.001, 3.1622776601683795e-05];
        let want: Vec<f32> = inv.iter().flat_map(|&a| [a.cos() as f32, a.sin() as f32]).collect();
        assert_eq!(row, want);
    }

    #[test]
    fn rope_row_length_matches_head_dim() {
        assert_eq!(rope_row(5, 128, 1_000_000.0).len(), 128);
    }

    /// Diagnostic, NOT a device test: dump the exact `x`/`rope_global` BYTES this rail would write
    /// for the failing teacher-forced step (device position 9, fed token 315 -- see
    /// `device_teacher_forced_matches_oracle`'s doc comment) for an out-of-band byte-for-byte
    /// compare against `verify_llm_decode.py`'s host computation. No device access -- `x`/
    /// `rope_global` are pure functions of (token, pos, embed table, scale, theta), so this needs
    /// only the embed `.npy` on disk. SKIPs if it is absent.
    #[test]
    fn dump_x_and_rope_bytes_for_the_failing_step() {
        let (_, weights_dir, _) = gate_paths();
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
        let rope = rope_row(POS, HEAD_DIM, THETA);
        let rope_bytes = pack_bf16_bytes(&rope);

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
    /// `route_b_kernels/decode_fused/gen_llm_decode.py` regenerated FRESH against the CURRENTLY
    /// PINNED toolchain reaches **8/8** on this exact rail (no code changed here at all) --
    /// `decode.elf` shrinks 22279888 -> 20952656 bytes, a real recompile, not container-metadata
    /// noise. Root cause: `toolchain.lock`'s `MLIR_AIE_FORK_COMMIT` advanced 035528f71cf1
    /// (upstream tip 2026-08-27) -> acecda2fc53e (2026-09-02) on 2026-09-04 10:38 -- AFTER the
    /// artifact was frozen -- and that pin's own history notes this class of bump changes emitted
    /// DMA/scheduling bytes even when it isn't a correctness fix. A ~6-day upstream advance
    /// changing instruction scheduling is fully sufficient to flip a genuine 0.02-margin bf16 tie.
    /// Host-side bytes were independently verified byte-identical against `verify_llm_decode.py`
    /// (embed row, `x`, `rope_global`) before this was found, so the artifact vintage -- not this
    /// rail's host code -- was the whole gap. Regenerate the default artifact to close it; until
    /// then this gate is EXPECTED to read 7/8 against the stale default, and a run pointed at a
    /// freshly generated decode dir is the one that must read 8/8.
    #[test]
    fn device_teacher_forced_matches_oracle() {
        let (decode_dir, weights_dir, oracle_path) = gate_paths();
        if !device_gate_enabled(&decode_dir, &weights_dir, &oracle_path) {
            return;
        }
        let (prompt_ids, gen_ids) = load_oracle(&oracle_path);
        let dev = Rc::new(Device::open(0).expect("open NPU device (stop other services first)"));
        let mut step = NpuDecodeStep::new(&dev, &decode_dir, &weights_dir).expect("build NpuDecodeStep");

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
        let mut step = NpuDecodeStep::new(&dev, &decode_dir, &weights_dir).expect("build NpuDecodeStep");

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
