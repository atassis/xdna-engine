//! Decoder-LLM serving: tokenizer + chat template + sampling + the generation loop, with the
//! device call behind [`generator::DecodeStep`] so the whole rail is testable with no NPU.

pub mod artifact;
pub mod chat_template;
pub mod config;
pub mod detokenize;
pub mod generator;
pub mod kv_layout;
pub mod npu_decode;
pub mod npu_prefill;
pub mod sampling;
pub mod tool_parse;
pub mod tool_syntax;

pub use artifact::{ArtifactRole, EmbedScale, LlmArtifact};
pub use chat_template::ChatTemplate;
pub use config::{ModelConfig, StopTokens};
pub use detokenize::{IncrementalDetokenizer, StopFeed, StopMatcher};
pub use generator::{CacheState, DecodeStep, LlmGenerator, ScriptedDecodeStep};
pub use kv_layout::kv_off;
pub use npu_decode::NpuDecodeStep;
pub use npu_prefill::{chunk_plan, NpuPrefill, PrefillChunk};
pub use sampling::{LogitView, SampleOutcome, SamplingConfig, SampleTimings};
pub use tool_parse::{parse_completion, ParseOut, ParsedCompletion, StreamingToolParser};
pub use tool_syntax::ToolSyntax;
