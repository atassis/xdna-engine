//! Gemma 3 (text) decoder config + the two phase-0 model presets.
//!
//! Dims are from the HF `config.json` of the ungated mirrors `unsloth/gemma-3-{270m,1b}-it`
//! (same weights as the HF-gated `google/gemma-3-*`). 270M is the de-risk first pass; 1B is the
//! ~1B-class target ([[run-small-llms-goal]]).

/// One Gemma 3 text-decoder configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GemmaConfig {
    pub d_model: usize,      // hidden_size
    pub n_layers: usize,     // num_hidden_layers
    pub n_q_heads: usize,    // num_attention_heads
    pub n_kv_heads: usize,   // num_key_value_heads (GQA: n_q_heads / n_kv_heads = group size)
    pub head_dim: usize,     // head_dim (note: n_q_heads*head_dim may exceed d_model in Gemma 3)
    pub ffn_dim: usize,      // intermediate_size (GeGLU: gate_proj + up_proj are d_model->ffn_dim)
    pub vocab: usize,        // vocab_size (tied input/output embedding)
    pub sliding_window: usize,        // local-attention window
    pub sliding_window_pattern: usize, // every Nth layer is GLOBAL attention (else local sliding-window)
    pub rope_theta_global: f32,       // rope_theta (global layers)
    pub rope_theta_local: f32,        // rope_local_base_freq (local layers)
    pub query_pre_attn_scalar: f32,   // attention score scale denominator (sqrt of this)
    pub rms_norm_eps: f32,
}

impl GemmaConfig {
    /// GQA group size: how many query heads share one KV head.
    pub fn gqa_group(&self) -> usize { self.n_q_heads / self.n_kv_heads }

    /// True if layer `i` (0-based) uses GLOBAL attention; else local sliding-window.
    /// Gemma 3 pattern: every `sliding_window_pattern`-th layer is global.
    pub fn is_global_layer(&self, i: usize) -> bool {
        (i + 1) % self.sliding_window_pattern == 0
    }
}

/// Gemma 3 270M (gemma3_text): d=640, 18 layers, GQA 4:1, head_dim 256, ffn 2048, vocab 262144.
/// The smallest real Gemma 3 — the phase-0 correctness target (host oracle: scripts/gemma_ref_generate.py).
pub const GEMMA3_270M: GemmaConfig = GemmaConfig {
    d_model: 640,
    n_layers: 18,
    n_q_heads: 4,
    n_kv_heads: 1,
    head_dim: 256,
    ffn_dim: 2048,
    vocab: 262144,
    sliding_window: 512,
    sliding_window_pattern: 6, // 1 global every 6 layers
    rope_theta_global: 1_000_000.0,
    rope_theta_local: 10_000.0,
    query_pre_attn_scalar: 256.0,
    rms_norm_eps: 1e-6,
};

/// Gemma 3 1B (gemma3_text): the ~1B-class scale-up target. Dims confirmed from
/// `unsloth/gemma-3-1b-it/config.json` before wiring tiles (the spec's "read the ONNX/config before sizing").
pub const GEMMA3_1B: GemmaConfig = GemmaConfig {
    d_model: 1152,
    n_layers: 26,
    n_q_heads: 4,
    n_kv_heads: 1,
    head_dim: 256,
    ffn_dim: 6912,
    vocab: 262144,
    sliding_window: 512,
    sliding_window_pattern: 6,
    rope_theta_global: 1_000_000.0,
    rope_theta_local: 10_000.0,
    query_pre_attn_scalar: 256.0,
    rms_norm_eps: 1e-6,
};

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn gqa_and_layer_pattern() {
        assert_eq!(GEMMA3_270M.gqa_group(), 4);
        // every 6th layer global: layers 5, 11, 17 (0-based) are global in an 18-layer stack.
        assert!(GEMMA3_270M.is_global_layer(5));
        assert!(!GEMMA3_270M.is_global_layer(0));
        assert!(GEMMA3_270M.is_global_layer(17));
    }
}
