//! How the registry turns a ModelCfg into something servable. The real impl wraps npu_engine::Model
//! (Phase 2); the mock makes the whole control plane host-testable with no device.
use crate::config::ModelCfg;
use npu_engine::{EngineError, ModelKind};

/// A loaded, servable model. Object-safe so the registry can hold `Box<dyn Inference>` (the real one
/// is !Send and lives only in the device-actor thread).
pub trait Inference {
    fn kind(&self) -> ModelKind;
    fn bo_bytes(&self) -> u64;
    fn transcribe(&self, pcm: &[i16], sample_rate: u32) -> Result<String, EngineError>;
    fn embed(&self, text: &str) -> Result<Vec<f32>, EngineError>;
}

pub trait ModelLoader {
    fn load(&self, cfg: &ModelCfg) -> Result<Box<dyn Inference>, EngineError>;
}

/// Real loader: turns a ModelCfg's scenario TOML into a live npu_engine::Model. `Capability` is
/// `npu_engine::ModelKind`, so `kind()` is a direct passthrough.
pub struct EngineLoader { pub root: std::path::PathBuf }
struct EngineModel { model: npu_engine::Model }
impl Inference for EngineModel {
    fn kind(&self) -> ModelKind { self.model.kind() }
    // Best-effort BO footprint: not yet measured per-model (backlog R11(f)); report 0 until then so
    // the accountant is a no-op rather than wrong. A later task wires real BO byte totals.
    fn bo_bytes(&self) -> u64 { 0 }
    fn transcribe(&self, pcm: &[i16], sr: u32) -> Result<String, EngineError> { self.model.transcribe(pcm, sr) }
    fn embed(&self, text: &str) -> Result<Vec<f32>, EngineError> { self.model.embed(text) }
}
impl ModelLoader for EngineLoader {
    fn load(&self, cfg: &ModelCfg) -> Result<Box<dyn Inference>, EngineError> {
        let model = npu_engine::Model::load_in(&cfg.scenario, &self.root)?;
        Ok(Box::new(EngineModel { model }))
    }
}

/// A scripted loader for tests + the actor integration test (gated behind `testkit`).
#[cfg(any(test, feature = "testkit"))]
pub mod mock {
    use super::*;
    use std::collections::BTreeMap;
    /// name -> Ok((kind, bo_bytes)) | Err(reason).
    pub struct MockLoader { pub table: BTreeMap<String, Result<(ModelKind, u64), String>> }
    pub struct MockModel { kind: ModelKind, bo: u64 }
    impl Inference for MockModel {
        fn kind(&self) -> ModelKind { self.kind }
        fn bo_bytes(&self) -> u64 { self.bo }
        fn transcribe(&self, _: &[i16], _: u32) -> Result<String, EngineError> {
            if self.kind == ModelKind::Asr { Ok("mock-text".into()) }
            else { Err(EngineError::WrongKind { wanted: ModelKind::Asr, got: ModelKind::Embed }) }
        }
        fn embed(&self, _: &str) -> Result<Vec<f32>, EngineError> {
            if self.kind == ModelKind::Embed { Ok(vec![0.0; 8]) }
            else { Err(EngineError::WrongKind { wanted: ModelKind::Embed, got: ModelKind::Asr }) }
        }
    }
    impl ModelLoader for MockLoader {
        fn load(&self, cfg: &ModelCfg) -> Result<Box<dyn Inference>, EngineError> {
            match self.table.get(&cfg.name) {
                Some(Ok((k, bo))) => Ok(Box::new(MockModel { kind: *k, bo: *bo })),
                Some(Err(e)) => Err(EngineError::Load(e.clone())),
                None => Err(EngineError::Load(format!("no mock entry for {}", cfg.name))),
            }
        }
    }
}
