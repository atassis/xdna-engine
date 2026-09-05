//! Decoder-LLM serving: tokenizer + chat template + sampling + the generation loop, with the
//! device call behind [`generator::DecodeStep`] so the whole rail is testable with no NPU.
//! `llm-serve-openai-surface` milestone A -- see
//! `docs/superpowers/specs/2026-09-05-llm-serving-and-residency-design.md`.

pub mod artifact;
pub mod chat_template;
pub mod config;
pub mod detokenize;
pub mod generator;
pub mod npu_decode;
pub mod sampling;

pub use artifact::{EmbedScale, LlmArtifact};
pub use chat_template::ChatTemplate;
pub use config::{ModelConfig, StopTokens};
pub use detokenize::{IncrementalDetokenizer, StopFeed, StopMatcher};
pub use generator::{DecodeStep, LlmGenerator, ScriptedDecodeStep};
pub use npu_decode::NpuDecodeStep;
pub use sampling::{LogitView, SampleOutcome, SamplingConfig};
