//! # npu-gemma -- Gemma 3 small-LLM decoder on the XDNA2 NPU engine (Phase 0 scaffold)
//!
//! The "run-any-model" proof: a Gemma 3 decoder is the SAME transformer-decoder shape as the Whisper
//! decoder we already run on-NPU ([[decode-microop-fusion-map]]), so it reuses our resident-FFN +
//! fused-decode + KV primitives directly. Honest framing: LLM *decode* is LPDDR-bandwidth-bound (weights
//! stream per token), so the NPU win is **energy / CPU-offload**, not raw tok/s -- the milestone is CORRECT
//! e2e generation through the engine.
//!
//! ## Phase 0 status (this crate)
//! - Host-CPU REFERENCE oracle = `scripts/gemma_ref_generate.py` (transformers, CPU; the ground truth).
//!   WORKS: gemma-3-270m-it "The capital of France is" -> " Paris." (first token id 9079).
//! - This crate = the config presets ([`config`]) + the NPU PORT MAP below + host reference primitives that
//!   the on-NPU path is validated against. The `npu` feature (off by default) will route matmuls/attention
//!   through our XDNA2 primitives in a later phase.
//!
//! ## NPU port map -- Gemma 3 decoder op -> our primitive (REUSE / WIRE / HOST)
//! Regime: single-stream M=1 decode = inter-op-OVERHEAD-bound (our exact decode-dispatch problem), so the
//! levers are MOVEMENT bricks + op-count, NOT compute. (Prefill of the prompt is M>=8 = compute-bound ->
//! mmul/bfp16, like the encoder.)
//!
//! KEY FINDING (verified 2026-06-29): our IRON fork (`amd/IRON/iron/operators/`) ALREADY ships the
//! Gemma-specific operators Whisper lacked -- `rms_norm` (with a `weighted` variant), `rope`,
//! `swiglu_{decode,prefill}`, GQA-aware `mha`, `repeat`, `dequant` (int4) -- plus a
//! `iron/applications/llama_3.2_1b/llama_npu.py` reference that assembles a full LLM decode as ONE fused ELF.
//! So Gemma is mostly **WIRING existing bricks**, not authoring new kernels.
//!
//! legend: REUSE = we already run this on-NPU (Whisper decode) · WIRE = operator EXISTS in IRON, wire it ·
//! HOST = host-CPU fallback for phase 0.
//!
//! | Gemma 3 op | maps to | status |
//! |---|---|---|
//! | Q/K/V/O projections (GQA, bias-free) | fused-decode **GEMV** (`gen_decode.py:186`) | REUSE (dims only) |
//! | GQA KV-head broadcast (1 KV head -> 4 q) | address the 1 KV head from all q heads (0 ops) / `repeat` op | WIRE (prefer free-by-layout) |
//! | RMSNorm input/post/pre/final (+ q_norm,k_norm) | `rms_norm` op (`weighted`); reduction + **invsqrt SFU** | WIRE (eps 1e-5->1e-6 const; fold (1+w) where possible; 2 un-foldable sandwich norms/layer) |
//! | RoPE (dual theta: local 1e4 / global 1e6) | `rope` op (head_dim 256 OK); per-type host sincos table | WIRE |
//! | FFN GeGLU (gate,up,down; gelu_tanh(gate)*up) | `swiglu_decode` composite + our fused-GELU epilogue | WIRE (swap SwiGLU's SiLU -> GELU-tanh) |
//! | sliding-window vs global mask (5:1, win 512) | the `sm_mask` per-token scratchpad (`gen_decode.py:208`) | REUSE mechanism, NEW windowed (lo,hi) shape |
//! | KV cache write | **StridedCopy** + `kv_off` scratchpad (`gen_decode.py:195`) | REUSE |
//! | softmax / attn QK,AV GEMVs / residual adds | `Softmax` / head-batched `GEMV` / `ElementwiseAdd` | REUSE |
//! | lm_head = embedding^T, vocab 262144 | **GEMV proj_out** (`--npu-logits`, `gen_decode.py:427`) | REUSE (tied weights; 262144 % 8 = 0, NO pad needed unlike Whisper) |
//! | on-NPU argmax over 262144 | shipped on-NPU argmax (returns a token id) | REUSE |
//! | input embed gather + sqrt(d) scale; sampling | host (one row/token; cheap) | HOST (phase 0) -> `parallel_lookup` later |
//!
//! The ONLY real kernel authoring = head_dim=256 in the PREFILL `mha` flash kernel (hard-codes d=64) -- and
//! that is NOT on the M=1 decode path (decode attention is the head-batched GEMV we already run), so it does
//! NOT block a phase-0 decode-only bring-up. int4/int8 (`dequant`) is the ENERGY lever, gated AFTER correctness.
//!
//! See `internal notes` for the full op-coverage audit + resume plan.

pub mod config;
pub use config::{GemmaConfig, GEMMA3_1B, GEMMA3_270M};

/// Host RMSNorm reference (the Gemma normalize: x / rms(x) * (1 + weight), NO mean-subtract).
/// The correctness oracle for the future on-NPU RMSNorm kernel (reduction + invsqrt SFU). f32 host.
pub fn rmsnorm_ref(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    assert_eq!(x.len(), weight.len());
    let n = x.len() as f32;
    let ms = x.iter().map(|v| v * v).sum::<f32>() / n;
    let inv = 1.0 / (ms + eps).sqrt();
    x.iter()
        .zip(weight)
        .map(|(v, w)| v * inv * (1.0 + w))
        .collect()
}

/// Host GeGLU reference (Gemma FFN gate: down( gelu_tanh(gate(x)) * up(x) )). Activation = gelu_pytorch_tanh.
/// The oracle for the on-NPU GeGLU fused epilogue (a GLU-mul over the gate/up GEMV outputs).
pub fn geglu_ref(gate: &[f32], up: &[f32]) -> Vec<f32> {
    assert_eq!(gate.len(), up.len());
    gate.iter().zip(up).map(|(&g, &u)| gelu_tanh(g) * u).collect()
}

/// gelu_pytorch_tanh: 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3))).
fn gelu_tanh(x: f32) -> f32 {
    const C: f32 = 0.797_884_56; // sqrt(2/pi)
    0.5 * x * (1.0 + (C * (x + 0.044_715 * x * x * x)).tanh())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rmsnorm_unit() {
        // weight 0 -> pure RMS normalize (scale by 1+0=1); unit-variance input stays ~unit.
        let x = [3.0, -3.0, 3.0, -3.0];
        let w = [0.0; 4];
        let y = rmsnorm_ref(&x, &w, 1e-6);
        for v in &y {
            assert!((v.abs() - 1.0).abs() < 1e-3, "got {v}");
        }
    }
    #[test]
    fn geglu_shapes() {
        let g = [1.0, 0.0, -1.0];
        let u = [2.0, 2.0, 2.0];
        let y = geglu_ref(&g, &u);
        assert_eq!(y.len(), 3);
        assert!((y[1]).abs() < 1e-6); // gelu(0)=0 -> 0*2=0
    }
}
