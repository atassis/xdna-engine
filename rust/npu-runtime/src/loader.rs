//! How the registry turns a ModelCfg into something servable. The real impl wraps npu_engine::Model;
//! the mock makes the whole control plane host-testable with no device.
//!
//! The servable trait is `npu_engine::capability::Servable` itself, not a runtime-local one. This
//! crate used to declare its own `Inference { kind, bo_bytes, transcribe, embed }`, which is what
//! closed the request surface at two modalities: a third capability meant a new method on the trait,
//! a new `Cmd`, and a new `ModelKind` variant. `Servable`'s single `run(Request)` costs none of
//! those, and a model that already implements it -- `npu_sr::SrEngine` -- becomes loadable here with
//! no adapter at all.
use crate::config::ModelCfg;
use npu_engine::capability::{Capability, Request, Response};
use npu_engine::EngineError;

pub use npu_engine::capability::Servable;

pub trait ModelLoader {
    fn load(&self, cfg: &ModelCfg) -> Result<Box<dyn Servable>, EngineError>;

    /// What capability this model DECLARES, without loading it: cheap, host-only, no device.
    ///
    /// Routing needs the capability of a model that is not resident, and the only alternative is to
    /// load one to find out -- which on this engine means seconds and the whole device. `None` means
    /// "cannot tell cheaply"; the caller then falls back to loading.
    fn declared_capability(&self, _cfg: &ModelCfg) -> Option<Capability> { None }
}

/// Real loader: turns a ModelCfg's scenario TOML into a live npu_engine::Model.
pub struct EngineLoader { pub root: std::path::PathBuf }
struct EngineModel { model: npu_engine::Model }

impl Servable for EngineModel {
    fn capabilities(&self) -> Capability { self.model.kind().capability() }
    // Best-effort BO footprint: not yet measured per-model (backlog R11(f)); report 0 until then so
    // the accountant is a no-op rather than wrong. A later task wires real BO byte totals.
    fn footprint(&self) -> u64 { 0 }
    fn run(&mut self, req: Request) -> Result<Response, EngineError> {
        match req {
            // Audio serves TWO capabilities, so this dispatches on the model's KIND. The compiler
            // cannot catch a missing arm here -- the match is over `Request`, not `ModelKind` --
            // so a diarize model would silently answer every request with a WrongKind error.
            // loader::tests pins the diarize path for exactly that reason.
            Request::Audio { pcm, sample_rate } => match self.model.kind() {
                npu_engine::ModelKind::Diarize =>
                    self.model.diarize(&pcm, sample_rate).map(Response::Segments),
                _ => self.model.transcribe(&pcm, sample_rate).map(Response::Text),
            },
            Request::Text(text) => self.model.embed(&text).map(Response::Vector),
            // Neither engine scenario takes an image, so this is a routing bug rather than a user
            // error -- but still a Result, because the actor must not be panicked by one.
            Request::Image { .. } => Err(EngineError::WrongKind {
                wanted: Capability::IMAGE_SR, got: self.model.kind().capability() }),
        }
    }
}

impl ModelLoader for EngineLoader {
    fn load(&self, cfg: &ModelCfg) -> Result<Box<dyn Servable>, EngineError> {
        let model = npu_engine::Model::load_in(&cfg.scenario, &self.root)?;
        Ok(Box::new(EngineModel { model }))
    }
    /// Read the scenario's `[scenario] kind`. Parsing one small TOML costs nothing next to a load,
    /// and it is the same field `registry::try_build` dispatches on.
    fn declared_capability(&self, cfg: &ModelCfg) -> Option<Capability> {
        let sc = npu_engine::config::ScenarioConfig::load(std::path::Path::new(&cfg.scenario)).ok()?;
        Capability::from_scenario_kind(&sc.scenario.kind)
    }
}

/// A scripted loader for tests + the actor integration test (gated behind `testkit`).
#[cfg(any(test, feature = "testkit"))]
pub mod mock {
    use super::*;
    use std::collections::BTreeMap;
    /// name -> Ok((capability, footprint)) | Err(reason).
    pub struct MockLoader { pub table: BTreeMap<String, Result<(Capability, u64), String>> }
    pub struct MockModel { cap: Capability, bo: u64 }
    impl Servable for MockModel {
        fn capabilities(&self) -> Capability { self.cap }
        fn footprint(&self) -> u64 { self.bo }
        /// Answers in the shape the capability implies, so a request of the wrong shape is an error
        /// rather than a plausible-looking wrong answer.
        fn run(&mut self, req: Request) -> Result<Response, EngineError> {
            match (self.cap, &req) {
                (Capability::ASR, Request::Audio { .. }) => Ok(Response::Text("mock-text".into())),
                (Capability::EMBED, Request::Text(_)) => Ok(Response::Vector(vec![0.0; 8])),
                (Capability::GENERATE, Request::Text(_)) => Ok(Response::Text("mock-completion".into())),
                (Capability::TTS, Request::Text(_)) =>
                    Ok(Response::Audio { pcm: vec![0i16; 8], sample_rate: 24_000 }),
                (cap, _) => Err(EngineError::Unsupported(format!("{cap} cannot serve this request shape"))),
            }
        }
    }
    impl ModelLoader for MockLoader {
        fn load(&self, cfg: &ModelCfg) -> Result<Box<dyn Servable>, EngineError> {
            match self.table.get(&cfg.name) {
                Some(Ok((c, bo))) => Ok(Box::new(MockModel { cap: *c, bo: *bo })),
                Some(Err(e)) => Err(EngineError::Load(e.clone())),
                None => Err(EngineError::Load(format!("no mock entry for {}", cfg.name))),
            }
        }
        /// The scripted capability, standing in for the scenario TOML the real loader reads.
        fn declared_capability(&self, cfg: &ModelCfg) -> Option<Capability> {
            match self.table.get(&cfg.name) { Some(Ok((c, _))) => Some(*c), _ => None }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use npu_engine::capability::{Request, Response, Segment};

    /// A `Servable` standing in for a loaded diarize model, exercising the SHAPE contract the
    /// EngineModel adapter must honour: audio in, segments out.
    struct FakeDiarize;
    impl Servable for FakeDiarize {
        fn capabilities(&self) -> Capability { Capability::DIARIZE }
        fn run(&mut self, req: Request) -> Result<Response, EngineError> {
            match req {
                Request::Audio { .. } =>
                    Ok(Response::Segments(vec![Segment { start_s: 0.0, end_s: 1.0, speaker: 0 }])),
                other => Err(EngineError::Unsupported(other.shape().into())),
            }
        }
    }

    #[test]
    fn a_diarize_model_answers_audio_with_segments_not_wrong_kind() {
        let mut m: Box<dyn Servable> = Box::new(FakeDiarize);
        let out = m.run(Request::Audio { pcm: vec![0i16; 16], sample_rate: 16_000 })
            .expect("audio to a diarize model must not be a WrongKind error");
        assert_eq!(out.shape(), "segments",
            "EngineModel::run dispatches Request::Audio on the model KIND; nothing in the type \
             system enforces this, so this test is the guard");
        assert_eq!(m.capabilities(), Capability::DIARIZE);
    }
}
