//! On-NPU decode routing (staged behind the `npu` feature).
//!
//! Phase-0 status: this module wires the ROUTING STRUCTURE + per-token host protocol against the fused
//! decode ELF produced by `route_b_kernels/decode_fused/gen_gemma_decode.py`. The device backend (XRT BO
//! upload, per-token dispatch of the constant ELF, scratchpad writes) is NOT linked in this crate yet --
//! [`GemmaNpuDecoder::step`] returns [`NpuError::DeviceBackendUnlinked`] so the crate `cargo check`s (and
//! this feature compiles) WITHOUT XRT. The execution agent swaps the backend in once the ELF builds.
//!
//! The per-token protocol below is the contract `gen_gemma_decode.py`'s `meta.json` documents; it mirrors
//! the shipped Whisper deep-C decode (`gen_decode.py`): the ELF is CONSTANT across tokens; per token the
//! host writes three scratchpad/inputs and dispatches once.

use crate::config::GemmaConfig;
use crate::schedule::{decode_schedule, Brick};

/// The generator script the execution agent builds the ELF from (name pinned so both sides agree).
pub const GENERATOR: &str = "route_b_kernels/decode_fused/gen_gemma_decode.py";

/// Errors from the (not-yet-linked) device path.
#[derive(Debug, Clone, PartialEq)]
pub enum NpuError {
    /// The XRT device backend is not compiled into this crate yet (phase-0 scaffold).
    DeviceBackendUnlinked,
    /// The fused decode ELF / meta.json has not been loaded.
    ElfNotLoaded,
    /// `n_past` exceeded the ELF's compiled KV-cache capacity `S`.
    ContextOverflow { n_past: usize, capacity: usize },
}

impl std::fmt::Display for NpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NpuError::DeviceBackendUnlinked => {
                write!(f, "NPU device backend not linked (phase-0 scaffold); build the ELF via {GENERATOR}")
            }
            NpuError::ElfNotLoaded => write!(f, "fused decode ELF not loaded"),
            NpuError::ContextOverflow { n_past, capacity } => {
                write!(f, "context {n_past} exceeds compiled KV capacity {capacity}")
            }
        }
    }
}

impl std::error::Error for NpuError {}

/// The per-token host inputs the constant ELF consumes (documented in `meta.json`).
/// Every field is computed on host from the current token + position, then written before ONE dispatch.
#[derive(Debug, Clone)]
pub struct TokenInputs {
    /// `embed[token] * sqrt(d_model)`, bf16-cast -- the residual-stream head (host embedding gather).
    pub x: Vec<f32>,
    /// Absolute position of this token (drives `kv_off` and the RoPE angle row).
    pub n_past: usize,
}

/// Scratchpad values the host writes per token (deep-C constant-ELF mechanism, from `gen_decode.py`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Scratchpad {
    /// `addr`-kind param, element units, written raw: the KV-cache append position `= n_past * head_dim`.
    pub kv_off: u32,
    /// `core`-kind param, causal softmax width `= n_past + 1`, written `<<2` (firmware UPDATE_REG).
    pub sm_mask: u32,
}

/// Compute the per-token scratchpad values from the position + head_dim.
pub fn scratchpad_for(n_past: usize, head_dim: usize) -> Scratchpad {
    Scratchpad {
        kv_off: (n_past * head_dim) as u32,
        sm_mask: (n_past + 1) as u32,
    }
}

/// Host-side routing driver for the fused Gemma decode. Holds the planned schedule + dims; the device
/// handle is added by the execution agent (feature-gated behind the backend).
pub struct GemmaNpuDecoder {
    cfg: GemmaConfig,
    seq_capacity: usize,
    schedule: Vec<Brick>,
    elf_loaded: bool,
}

impl GemmaNpuDecoder {
    /// Plan the decoder for `cfg` with a KV-cache capacity of `seq_capacity` tokens. No device touched.
    pub fn plan(cfg: GemmaConfig, seq_capacity: usize) -> Self {
        let schedule = decode_schedule(&cfg, seq_capacity);
        Self { cfg, seq_capacity, schedule, elf_loaded: false }
    }

    /// The planned per-token op schedule (source of truth shared with the IRON generator).
    pub fn schedule(&self) -> &[Brick] {
        &self.schedule
    }

    pub fn config(&self) -> &GemmaConfig {
        &self.cfg
    }

    pub fn seq_capacity(&self) -> usize {
        self.seq_capacity
    }

    /// Load the fused decode ELF + `meta.json` produced by [`GENERATOR`]. STUB: records intent; the XRT
    /// upload is added with the backend. Returns the number of scheduled bricks for a sanity check.
    pub fn load_elf(&mut self, _elf_dir: &std::path::Path) -> Result<usize, NpuError> {
        // TODO(execution-agent): parse meta.json, upload weight BOs, register the constant ELF.
        self.elf_loaded = true;
        Ok(self.schedule.len())
    }

    /// Run ONE decode step: write `x` + scratchpad, dispatch the constant ELF, return the argmax token id.
    /// STUB: validates the host-side contract, then reports the backend is unlinked.
    pub fn step(&self, inp: &TokenInputs) -> Result<i64, NpuError> {
        if !self.elf_loaded {
            return Err(NpuError::ElfNotLoaded);
        }
        if inp.n_past >= self.seq_capacity {
            return Err(NpuError::ContextOverflow { n_past: inp.n_past, capacity: self.seq_capacity });
        }
        let _sp = scratchpad_for(inp.n_past, self.cfg.head_dim);
        assert_eq!(inp.x.len(), self.cfg.d_model, "residual head must be d_model wide");
        // TODO(execution-agent): write x + _sp to device, dispatch, on-NPU argmax over VOCAB, read id.
        Err(NpuError::DeviceBackendUnlinked)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::GEMMA3_270M;

    #[test]
    fn scratchpad_math() {
        // position 0: first token, kv_off 0, width 1.
        assert_eq!(scratchpad_for(0, 256), Scratchpad { kv_off: 0, sm_mask: 1 });
        // position 7: append at row 7 (7*256), attend 8 positions.
        assert_eq!(scratchpad_for(7, 256), Scratchpad { kv_off: 7 * 256, sm_mask: 8 });
    }

    #[test]
    fn planning_is_deviceless() {
        let d = GemmaNpuDecoder::plan(GEMMA3_270M, 2048);
        assert_eq!(d.schedule().len(), 26 * 18 + 2);
        assert_eq!(d.config().head_dim, 256);
    }

    #[test]
    fn step_reports_unlinked_after_load() {
        let mut d = GemmaNpuDecoder::plan(GEMMA3_270M, 2048);
        let n = d.load_elf(std::path::Path::new("/does/not/matter")).unwrap();
        assert_eq!(n, 26 * 18 + 2);
        let inp = TokenInputs { x: vec![0.0; 640], n_past: 0 };
        assert_eq!(d.step(&inp), Err(NpuError::DeviceBackendUnlinked));
    }

    #[test]
    fn context_overflow_guard() {
        let mut d = GemmaNpuDecoder::plan(GEMMA3_270M, 16);
        d.load_elf(std::path::Path::new("/x")).unwrap();
        let inp = TokenInputs { x: vec![0.0; 640], n_past: 16 };
        assert_eq!(d.step(&inp), Err(NpuError::ContextOverflow { n_past: 16, capacity: 16 }));
    }
}
