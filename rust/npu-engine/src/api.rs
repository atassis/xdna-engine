//! The public, stable engine API. This module is the ONLY supported surface; everything else in the
//! crate is `#[doc(hidden)]` implementation detail and may change without notice.

use std::path::Path;

use crate::pipeline::Scenario;

/// What a loaded model does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelKind { Asr, Embed, Diarize }

impl std::fmt::Display for ModelKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            ModelKind::Asr => "asr",
            ModelKind::Embed => "embed",
            ModelKind::Diarize => "diarize",
        })
    }
}

impl ModelKind {
    /// The capability a scenario's `[scenario] kind` declares -- known from the TOML alone, with no
    /// device, no weights and no build. `registry::try_build` dispatches on this same answer, so the
    /// two cannot drift; the control plane uses it to route a request to the right model without
    /// having to load a wrong one first to find out what it was.
    pub fn from_scenario_kind(s: &str) -> Option<ModelKind> {
        match s {
            "asr" => Some(ModelKind::Asr),
            "embeddings" => Some(ModelKind::Embed),
            "diarize" => Some(ModelKind::Diarize),
            _ => None,
        }
    }

    /// This kind as a routing capability. `ModelKind` stays the engine's own scenario-dispatch enum
    /// (and the C ABI's `npu_model_kind`); `Capability` is what the control plane routes on, and it
    /// is open where this is closed.
    pub fn capability(self) -> crate::capability::Capability {
        match self {
            ModelKind::Asr => crate::capability::Capability::ASR,
            ModelKind::Embed => crate::capability::Capability::EMBED,
            ModelKind::Diarize => crate::capability::Capability::DIARIZE,
        }
    }
}

/// Engine error surface. Internal errors are flattened into these variants with a message.
#[derive(thiserror::Error, Debug)]
pub enum EngineError {
    #[error("no XDNA2 NPU device available")]
    NotAvailable,
    #[error("load failed: {0}")]
    Load(String),
    /// Carries `Capability`, not `ModelKind`: this is raised by capability routing, which must be
    /// able to name a capability no `ModelKind` variant exists for (`tts`, `generate`, `image-sr`).
    #[error("wrong model kind: wanted {wanted}, got {got}")]
    WrongKind { wanted: crate::capability::Capability, got: crate::capability::Capability },
    /// No configured model declares this capability. Distinct from `Unsupported` because it is a
    /// SERVER-configuration fact, not a bad request -- the HTTP surface answers it 503, not 400.
    #[error("no {0} model configured")]
    NoModel(crate::capability::Capability),
    #[error("unsupported: {0}")]
    Unsupported(String),
    #[error("device error: {0}")]
    Device(String),
}

/// Process-level engine facts.
pub struct Engine;
impl Engine {
    /// True if an XDNA2 NPU device node is present. Cheap: checks the device file, does not open it.
    pub fn available() -> bool {
        Path::new("/dev/accel/accel0").exists()
    }
}

/// A loaded model. Wraps an internal pipeline; holds device resources, so it is NOT Send/Sync and a
/// single instance must not be driven concurrently (the service serializes; the NPU is single-tenant).
pub struct Model {
    scen: Scenario,
    /// Configured hidden size, when the scenario has a `[model]` block at all. `None` for a
    /// non-transformer model, which is exactly when `embed_dim` has nothing to report.
    hidden: Option<usize>,
}

impl Model {
    /// Load a model from a scenario TOML, using the current working directory as the repo root
    /// (where artifacts/ live). Bakes the npu-weights checkpoint on miss (A4 declarative path).
    pub fn load(scenario: impl AsRef<Path>) -> Result<Model, EngineError> {
        let root = std::env::current_dir()
            .map_err(|e| EngineError::Load(format!("cwd: {e}")))?;
        Model::load_in(scenario, root)
    }

    /// Like `load`, with an explicit repo root.
    pub fn load_in(scenario: impl AsRef<Path>, root: impl AsRef<Path>) -> Result<Model, EngineError> {
        let cfg = crate::config::ScenarioConfig::load(scenario.as_ref())
            .map_err(|e| EngineError::Load(format!("scenario {}: {e}", scenario.as_ref().display())))?;
        let hidden = cfg.model.as_ref().map(|m| m.hidden);
        let scen = crate::registry::try_build(scenario.as_ref(), root.as_ref())?;
        Ok(Model { scen, hidden })
    }

    pub fn kind(&self) -> ModelKind {
        kind_of(&self.scen)
    }

    /// Embedding output dimension for an embed model (= configured hidden size); None for ASR.
    pub fn embed_dim(&self) -> Option<usize> {
        match self.scen { Scenario::Embed(_) => self.hidden, _ => None }
    }

    /// ASR: 16 kHz mono i16 PCM -> text.
    pub fn transcribe(&self, pcm: &[i16], sample_rate: u32) -> Result<String, EngineError> {
        if sample_rate != 16_000 {
            return Err(EngineError::Unsupported(format!("sample_rate {sample_rate} (need 16000)")));
        }
        match &self.scen {
            Scenario::Asr(m) => m.transcribe(pcm),
            other => Err(EngineError::WrongKind {
                wanted: ModelKind::Asr.capability(), got: kind_of(other).capability() }),
        }
    }

    /// Embedding: text -> vector.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>, EngineError> {
        match &self.scen {
            Scenario::Embed(m) => m.embed_one(text.to_string()),
            other => Err(EngineError::WrongKind {
                wanted: ModelKind::Embed.capability(), got: kind_of(other).capability() }),
        }
    }

    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EngineError> {
        texts.iter().map(|t| self.embed(t)).collect()
    }

    /// Diarization: 16 kHz mono i16 PCM -> speaker-attributed spans.
    pub fn diarize(&self, pcm: &[i16], sample_rate: u32)
        -> Result<Vec<crate::capability::Segment>, EngineError> {
        if sample_rate != 16_000 {
            return Err(EngineError::Unsupported(format!("sample_rate {sample_rate} (need 16000)")));
        }
        match &self.scen {
            Scenario::Diarize(m) => m.diarize(pcm),
            other => Err(EngineError::WrongKind {
                wanted: ModelKind::Diarize.capability(), got: kind_of(other).capability() }),
        }
    }
}

/// The kind of a scenario without needing a `Model`. Keeps every mismatch arm above from having to
/// re-match all three variants.
fn kind_of(s: &Scenario) -> ModelKind {
    match s {
        Scenario::Asr(_) => ModelKind::Asr,
        Scenario::Embed(_) => ModelKind::Embed,
        Scenario::Diarize(_) => ModelKind::Diarize,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn error_and_kind_display() {
        assert_eq!(ModelKind::Asr.to_string(), "asr");
        assert_eq!(ModelKind::Embed.to_string(), "embed");
        let e = EngineError::WrongKind {
            wanted: ModelKind::Asr.capability(), got: ModelKind::Embed.capability() };
        assert_eq!(e.to_string(), "wrong model kind: wanted asr, got embed");
        assert_eq!(EngineError::NotAvailable.to_string(), "no XDNA2 NPU device available");
    }

    #[test]
    fn load_missing_scenario_is_load_error() {
        // Model is not Debug (holds device resources), so match the Result directly.
        match Model::load("/nonexistent/scenario.toml") {
            Err(EngineError::Load(_)) => {}
            Err(other) => panic!("expected Load error, got {other:?}"),
            Ok(_) => panic!("expected an error loading a missing scenario"),
        }
    }

    #[test]
    fn diarize_is_a_model_kind_that_routes_and_refuses_the_other_two() {
        assert_eq!(ModelKind::Diarize.to_string(), "diarize");
        assert_eq!(ModelKind::from_scenario_kind("diarize"), Some(ModelKind::Diarize));
        assert_eq!(ModelKind::Diarize.capability(), crate::capability::Capability::DIARIZE);
        // The existing two are untouched -- this is the regression that matters.
        assert_eq!(ModelKind::from_scenario_kind("asr"), Some(ModelKind::Asr));
        assert_eq!(ModelKind::from_scenario_kind("embeddings"), Some(ModelKind::Embed));
        assert_eq!(ModelKind::from_scenario_kind("nope"), None);
    }
}
