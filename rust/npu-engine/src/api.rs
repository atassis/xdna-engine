//! The public, stable engine API. This module is the ONLY supported surface; everything else in the
//! crate is `#[doc(hidden)]` implementation detail and may change without notice.

use std::path::Path;

use crate::pipeline::Scenario;

/// What a loaded model does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelKind { Asr, Embed }

impl std::fmt::Display for ModelKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self { ModelKind::Asr => "asr", ModelKind::Embed => "embed" })
    }
}

/// Engine error surface. Internal errors are flattened into these variants with a message.
#[derive(thiserror::Error, Debug)]
pub enum EngineError {
    #[error("no XDNA2 NPU device available")]
    NotAvailable,
    #[error("load failed: {0}")]
    Load(String),
    #[error("wrong model kind: wanted {wanted}, got {got}")]
    WrongKind { wanted: ModelKind, got: ModelKind },
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
    hidden: usize,
}

impl Model {
    /// Load a model from a scenario TOML, using the current working directory as the repo root
    /// (where artifacts/ live). Bakes the npu-weights arena on miss (A4 declarative path).
    pub fn load(scenario: impl AsRef<Path>) -> Result<Model, EngineError> {
        let root = std::env::current_dir()
            .map_err(|e| EngineError::Load(format!("cwd: {e}")))?;
        Model::load_in(scenario, root)
    }

    /// Like `load`, with an explicit repo root.
    pub fn load_in(scenario: impl AsRef<Path>, root: impl AsRef<Path>) -> Result<Model, EngineError> {
        let cfg = crate::config::ScenarioConfig::load(scenario.as_ref())
            .map_err(|e| EngineError::Load(format!("scenario {}: {e}", scenario.as_ref().display())))?;
        let hidden = cfg.model.hidden;
        let scen = crate::registry::try_build(scenario.as_ref(), root.as_ref())?;
        Ok(Model { scen, hidden })
    }

    pub fn kind(&self) -> ModelKind {
        match self.scen { Scenario::Asr(_) => ModelKind::Asr, Scenario::Embed(_) => ModelKind::Embed }
    }

    /// Embedding output dimension for an embed model (= configured hidden size); None for ASR.
    pub fn embed_dim(&self) -> Option<usize> {
        match self.scen { Scenario::Embed(_) => Some(self.hidden), Scenario::Asr(_) => None }
    }

    /// ASR: 16 kHz mono i16 PCM -> text.
    pub fn transcribe(&self, pcm: &[i16], sample_rate: u32) -> Result<String, EngineError> {
        if sample_rate != 16_000 {
            return Err(EngineError::Unsupported(format!("sample_rate {sample_rate} (need 16000)")));
        }
        match &self.scen {
            Scenario::Asr(m) => Ok(m.transcribe(pcm)),
            Scenario::Embed(_) =>
                Err(EngineError::WrongKind { wanted: ModelKind::Asr, got: ModelKind::Embed }),
        }
    }

    /// Embedding: text -> vector.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>, EngineError> {
        match &self.scen {
            Scenario::Embed(m) => Ok(m.embed_one(text.to_string())),
            Scenario::Asr(_) =>
                Err(EngineError::WrongKind { wanted: ModelKind::Embed, got: ModelKind::Asr }),
        }
    }

    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EngineError> {
        texts.iter().map(|t| self.embed(t)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn error_and_kind_display() {
        assert_eq!(ModelKind::Asr.to_string(), "asr");
        assert_eq!(ModelKind::Embed.to_string(), "embed");
        let e = EngineError::WrongKind { wanted: ModelKind::Asr, got: ModelKind::Embed };
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
}
