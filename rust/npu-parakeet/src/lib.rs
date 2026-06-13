//! Parakeet-tdt-0.6b-v3 FastConformer encoder — Phase 2 host CPU reference.
//!
//! A self-contained, host-only (ndarray f32) reference of the Parakeet encoder, verified
//! block-by-block against the ONNX reference activations (artifacts/parakeet/encoder/refs/,
//! produced by scripts/extract_parakeet_encoder.py) and matching the NumPy reference
//! scripts/parakeet_ref_encoder.py exactly. The new pieces vs GigaAM are rel-pos
//! (Transformer-XL) attention replacing RoPE, depthwise conv k=9, and a conv2D ÷8 subsample.
//!
//! Built to the `feat/general-engine` Encoder contract: `FastConformerEncoder` exposes
//! `forward_last(&self, x: &Array2<f32>, valid_len: usize) -> Array2<f32>`. The thin
//! `impl npu_engine::pipeline::Encoder` adapter + registry wiring are merge-time work
//! (reconcile in that branch's registry.rs); this crate stays decoupled so it builds alone.
//! Phase 3 swaps the host matmuls for `npu_asr::ctx2` NPU dispatches.

pub mod config;
pub mod encoder;
#[cfg(feature = "npu")]
pub mod npu;
pub mod ops;
pub mod pos;
pub mod prof;
pub mod weights;

pub use config::ModelCfg;
pub use encoder::FastConformerEncoder;
pub use weights::ParakeetWeights;
