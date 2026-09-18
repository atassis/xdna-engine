//! The S2 AR (autoregressive) half: text/semantic tokens -> codec codes, as opposed to the
//! crate-root `weights.rs` (codec DECODER, codes -> audio). Same GGUF checkpoint, disjoint tensor
//! namespace (`layers.*`/`fast_layers.*`/`embeddings.weight`/... vs `c.decoder.*`).
//!
//! `scripts/s2_ar_ref.py` is the graph reference (op order, RoPE convention, KV-cache equivalence
//! argument). [`weights`] resolves WEIGHTS by name -- see [`weights::ArWeights`] for why lazily
//! and what that costs/saves; [`sample`] is the token-selection path on top of the LM head's
//! logits (masking, top_k/top_p/temperature, repetition-aware resampling) -- see its module doc
//! for the one deliberate divergence from stock (a seeded, injectable RNG).

pub mod sample;
pub mod weights;

pub use sample::{
    greedy_token, ras_resample_params, sample_token, select_main_token, RasWindow, SampleRng,
    SamplerParams, SemanticMask, SplitMix64,
};
pub use weights::{ArLayerWeights, ArWeights};
