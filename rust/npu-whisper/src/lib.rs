//! Whisper-small encoder — host CPU f32 reference.
//!
//! A self-contained, host-only (ndarray f32) reference of the Whisper encoder, gated
//! block-by-block against the ONNX golden activations (artifacts/whisper-small/refs/). Uses the
//! shared host ops in `npu-asr-host` (im2col conv1d, exact GELU, MHA, LayerNorm). The `npu`
//! feature is declared for a later task (route matmuls through XDNA2) but unused in this crate.

pub mod config;
pub mod encoder;
#[cfg(feature = "npu")]
pub mod mha_npu;
#[cfg(feature = "npu")]
pub mod npu;
pub mod weights;

pub use config::WhisperCfg;
pub use encoder::WhisperEncoder;
pub use weights::WhisperWeights;
