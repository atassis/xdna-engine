//! Decoder-LLM serving: tokenizer + chat template + sampling + the generation loop, with the
//! device call behind [`generator::DecodeStep`] so the whole rail is testable with no NPU.

pub mod artifact;
pub mod chat_template;
pub mod config;
pub mod detokenize;
pub mod generator;
pub mod gemma4_media;
pub mod kv_layout;
pub mod multimodal;
pub mod npu_decode;
pub mod npu_prefill;
pub mod resident;
pub mod resident_artifact;
pub mod resident_ladder;
pub mod resident_onecmd;
pub mod resident_raw;
pub mod sampling;
pub mod tool_parse;
pub mod tool_syntax;

pub use artifact::{ArtifactRole, EmbedScale, LlmArtifact};
pub use chat_template::ChatTemplate;
pub use config::{ModelConfig, StopTokens};
pub use detokenize::{IncrementalDetokenizer, StopFeed, StopMatcher};
pub use generator::{CacheState, DecodeStep, LlmGenerator, ScriptedDecodeStep};
pub use kv_layout::kv_off;
pub use multimodal::{scatter_media_rows, MediaAttachment, MediaEmbeds};
pub use npu_decode::NpuDecodeStep;
pub use npu_prefill::{chunk_plan, NpuPrefill, PrefillChunk};
pub use resident::ResidentForward;
pub use resident_artifact::{ResidentArtifact, StoreManifest};
pub use resident_ladder::{LadderMeta, LadderResidentForward};
pub use resident_onecmd::{OneCmdMeta, OneCommandResidentForward};
pub use resident_raw::{RawResidentForward, RawResidentMeta};
pub use sampling::{LogitView, SampleOutcome, SamplingConfig, SampleTimings};
pub use tool_parse::{parse_completion, ParseOut, ParsedCompletion, StreamingToolParser};
pub use tool_syntax::{PayloadFormat, ProbeReason, ToolProbe, ToolSyntax};

/// The context a resident-forward build serves, dispatched on its `meta.json` kind the same way
/// the registry picks its driver; `None` for anything that is not one.
pub fn resident_max_context(dir: &std::path::Path) -> Option<usize> {
    let v: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("meta.json")).ok()?).ok()?;
    match v.get("kind")?.as_str()? {
        "resident_forward_ladder" => resident_ladder::LadderMeta::load(dir).ok().map(|m| m.largest_keys(1)),
        "resident_forward_onecmd" => resident_onecmd::OneCmdMeta::load(dir)
            .ok()
            .map(|m| resident_onecmd::max_context_bound(m.s_cap, m.pmax, m.nbw, m.sliding_window)),
        _ => None,
    }
}
