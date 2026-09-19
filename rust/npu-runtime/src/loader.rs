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

/// Extends `Servable` with the streaming text-generation surface. Local to npu-runtime, not
/// `npu_engine::capability`: `Servable::run` is single-shot over `Request`/`Response`, which have no
/// shape for a token stream, while `TextGenerator::generate` takes a sink and a `&mut self` decoder.
/// Every `Servable` this crate constructs implements this too (the registry stores
/// `Box<dyn StreamServable>`); the default answers `Unsupported` for a model that cannot generate.
pub trait StreamServable: Servable {
    fn generate_stream(
        &mut self,
        _prompt: &npu_engine::Prompt,
        _params: &npu_engine::GenerateParams,
        _sink: &mut dyn FnMut(npu_engine::Chunk<'_>) -> bool,
    ) -> Result<(), EngineError> {
        Err(EngineError::Unsupported(format!("{} cannot stream text generation", self.capabilities())))
    }

    /// `Servable::run`, but interruptible: `cancel` is published for the call's duration the same
    /// way `GenerateParams::cancel` is for `generate_stream`, so a caller elsewhere can stop it.
    /// Default forwards to `run` unchanged -- embed/asr/diarize finish inside one dispatch and have
    /// nothing to interrupt. TTS is the one capability whose `run` is a long AR loop; `EngineModel`
    /// overrides this for it.
    fn run_cancellable(&mut self, req: Request, _cancel: npu_engine::Cancel) -> Result<Response, EngineError> {
        self.run(req)
    }
}

pub trait ModelLoader {
    fn load(&self, cfg: &ModelCfg) -> Result<Box<dyn StreamServable>, EngineError>;

    /// What capability this model DECLARES, without loading it: cheap, host-only, no device.
    ///
    /// Routing needs the capability of a model that is not resident, and the only alternative is to
    /// load one to find out -- which on this engine means seconds and the whole device. `None` means
    /// "cannot tell cheaply"; the caller then falls back to loading.
    fn declared_capability(&self, _cfg: &ModelCfg) -> Option<Capability> { None }

    /// Best-effort pre-load size estimate in bytes, without loading or touching a device.
    ///
    /// The pin invariant (sum of pinned bytes <= memory_ceiling_mb) has to be checkable for a model
    /// that has never been loaded -- there is no live device counter yet, and `npu config show`/`npu
    /// models` already work with the service down, so this can't wait for one. `None` means "cannot
    /// tell cheaply"; a caller weighs that as 0, the same honest-unmeasured convention `footprint()`
    /// already uses, not "zero bytes".
    fn declared_footprint(&self, _cfg: &ModelCfg) -> Option<u64> { None }

    /// Bake this model's declarative `{source, arch, checkpoint}` spec into a checkpoint on disk,
    /// host-only, no device. `Ok(None)` means the scenario has no such spec (legacy `weights =`
    /// npy path) -- nothing to bake, not a failure.
    fn bake(&self, _cfg: &ModelCfg, _force: bool) -> Result<Option<std::path::PathBuf>, EngineError> {
        Ok(None)
    }
}

/// Real loader: turns a ModelCfg's scenario TOML into a live npu_engine::Model.
pub struct EngineLoader { pub root: std::path::PathBuf }
struct EngineModel { model: npu_engine::Model }

impl Servable for EngineModel {
    fn capabilities(&self) -> Capability { self.model.kind().capability() }
    /// Real pinned BO bytes where the underlying pipeline exposes its device handle (Parakeet
    /// today); `0` for every other scenario kind until they are wired the same way, which
    /// `Registry::unweighed_residents` reports rather than silently trusting.
    fn footprint(&self) -> u64 { self.model.bo_bytes() }
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
            // Text serves two kinds here (Embed and Tts), same reasoning as Audio above. This is
            // the entry point for a caller that has no `Cancel` to hand in (npu-capi, a direct
            // `Servable::run`); `run_cancellable` is the one that carries a real one into synthesis.
            Request::Text(text) => match self.model.kind() {
                npu_engine::ModelKind::Tts => self.model.synthesize(&text, &npu_engine::Cancel::new())
                    .map(|(pcm, sample_rate)| Response::Audio { pcm, sample_rate }),
                _ => self.model.embed(&text).map(Response::Vector),
            },
            // Neither engine scenario takes an image, so this is a routing bug rather than a user
            // error -- but still a Result, because the actor must not be panicked by one.
            Request::Image { .. } => Err(EngineError::WrongKind {
                wanted: Capability::IMAGE_SR, got: self.model.kind().capability() }),
        }
    }
}

impl StreamServable for EngineModel {
    /// `Model::generate` already returns `WrongKind` for a non-Generate scenario, so this is a plain
    /// delegation -- no capability check duplicated here.
    fn generate_stream(&mut self, prompt: &npu_engine::Prompt, params: &npu_engine::GenerateParams,
        sink: &mut dyn FnMut(npu_engine::Chunk<'_>) -> bool) -> Result<(), EngineError> {
        self.model.generate(prompt, params, sink)
    }

    /// The one capability that needs the actor's published `Cancel`: a synthesis is a long AR loop
    /// with no sink to poll instead, the same reasoning `generate_stream` above needs none of --
    /// every other kind's `run` already returns inside one dispatch.
    fn run_cancellable(&mut self, req: Request, cancel: npu_engine::Cancel) -> Result<Response, EngineError> {
        match (self.model.kind(), req) {
            (npu_engine::ModelKind::Tts, Request::Text(text)) =>
                self.model.synthesize(&text, &cancel)
                    .map(|(pcm, sample_rate)| Response::Audio { pcm, sample_rate }),
            (_, req) => self.run(req),
        }
    }
}

impl EngineLoader {
    /// A scenario path resolved against the engine ROOT when it is relative.
    ///
    /// `root` used to reach only the artifacts, so a relative `scenario = "scenarios/x.toml"` was
    /// joined against the process's working directory instead. That works wherever cwd happens to
    /// equal the root -- which the systemd unit sets, masking it -- and fails everywhere else, so
    /// the CLI could not run from any other directory.
    fn scenario_path(&self, cfg: &ModelCfg) -> std::path::PathBuf {
        let p = std::path::Path::new(&cfg.scenario);
        if p.is_absolute() { p.to_path_buf() } else { self.root.join(p) }
    }
}

impl ModelLoader for EngineLoader {
    fn load(&self, cfg: &ModelCfg) -> Result<Box<dyn StreamServable>, EngineError> {
        let model = npu_engine::Model::load_in(self.scenario_path(cfg), &self.root)?;
        Ok(Box::new(EngineModel { model }))
    }
    /// Read the scenario's `[scenario] kind`. Parsing one small TOML costs nothing next to a load,
    /// and it is the same field `registry::try_build` dispatches on.
    fn declared_capability(&self, cfg: &ModelCfg) -> Option<Capability> {
        let sc = npu_engine::config::ScenarioConfig::load(&self.scenario_path(cfg)).ok()?;
        Capability::from_scenario_kind(&sc.scenario.kind)
    }

    /// Reads the same scenario TOML `declared_capability` does, then stats the artifact path(s) --
    /// a file's own size, or the recursive sum of a directory's. Approximate (host bytes, not device
    /// BO bytes -- padding/quantization can disagree with either), but it's the only number that
    /// exists before anything is loaded.
    ///
    /// `tts` is not one `artifacts.weights` dir: it is two autoregressive weight sets plus a codec
    /// (`[tts]`, `config::TtsCfg`), and this is read with the service DOWN to check the pin
    /// invariant before anything is loaded -- an undercount here corrupts residency accounting for
    /// every OTHER model, not just this one. Summing a field that is empty by default would silently
    /// answer `Some(0)`; worse, `self.root.join("")` resolves to `self.root` itself, so it would
    /// walk and sum the entire root. `dir_or_file_size` is never asked to do either: an empty part
    /// makes the whole footprint `None`, the same honest-unmeasured answer as a missing directory.
    fn declared_footprint(&self, cfg: &ModelCfg) -> Option<u64> {
        let sc = npu_engine::config::ScenarioConfig::load(&self.scenario_path(cfg)).ok()?;
        if npu_engine::ModelKind::from_scenario_kind(&sc.scenario.kind) == Some(npu_engine::ModelKind::Tts) {
            let mut total = 0u64;
            for part in [&sc.tts.slow_ar, &sc.tts.fast_ar, &sc.tts.codec] {
                if part.is_empty() { return None; }
                total += dir_or_file_size(&self.root.join(part))?;
            }
            return Some(total);
        }
        dir_or_file_size(&self.root.join(&sc.artifacts.weights))
    }

    /// Mirrors the CLI's own `bake()`: load the scenario, resolve its declarative spec if it has
    /// one, and bake against THIS loader's root -- the same root `load()` resolves scenarios
    /// against, so a bake and the load that follows it agree on where the checkpoint lives.
    fn bake(&self, cfg: &ModelCfg, force: bool) -> Result<Option<std::path::PathBuf>, EngineError> {
        let sc = npu_engine::config::ScenarioConfig::load(&self.scenario_path(cfg))
            .map_err(|e| EngineError::Load(e.to_string()))?;
        match sc.artifacts.model_spec().map_err(|e| EngineError::Load(e.to_string()))? {
            Some(spec) => spec.ensure_checkpoint(&self.root, force)
                .map(Some)
                .map_err(|e| EngineError::Load(e.to_string())),
            None => Ok(None),
        }
    }
}

/// A file's own size, or the recursive sum of a directory's. `None` if `path` does not exist or a
/// read fails partway -- an estimate that silently under-counts a partial failure is worse than one
/// that says it does not know.
fn dir_or_file_size(path: &std::path::Path) -> Option<u64> {
    let meta = std::fs::metadata(path).ok()?;
    if !meta.is_dir() {
        return Some(meta.len());
    }
    let mut total = 0u64;
    for entry in std::fs::read_dir(path).ok()? {
        let entry = entry.ok()?;
        total += dir_or_file_size(&entry.path())?;
    }
    Some(total)
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
    /// No generation support: models that need to stream text build a purpose-made loader (see
    /// `http.rs`'s test module) rather than growing this shared fixture's shape for every caller.
    impl StreamServable for MockModel {}

    impl ModelLoader for MockLoader {
        fn load(&self, cfg: &ModelCfg) -> Result<Box<dyn StreamServable>, EngineError> {
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
        /// The scripted footprint, standing in for a stat of the weight artifact. Tests do not
        /// distinguish pre-load estimate from post-load live bytes -- both come from the same table
        /// entry, which is fine for exercising admission logic that only cares about the number.
        fn declared_footprint(&self, cfg: &ModelCfg) -> Option<u64> {
            match self.table.get(&cfg.name) { Some(Ok((_, bo))) => Some(*bo), _ => None }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use npu_engine::capability::{Request, Response, Segment};

    #[test]
    fn a_relative_scenario_resolves_against_the_root_not_the_cwd() {
        // The regression the CLI hit: engine.toml carries `scenarios/x.toml`, and joining that
        // against the process cwd works only where cwd happens to equal the root -- which the
        // systemd unit sets, so the service masked it and only the CLI failed.
        let l = EngineLoader { root: std::path::PathBuf::from("/opt/engine") };
        let rel = ModelCfg { name: "m".into(), scenario: "scenarios/x.toml".into(), resident: false };
        assert_eq!(l.scenario_path(&rel), std::path::PathBuf::from("/opt/engine/scenarios/x.toml"));
        // An absolute path is already an answer and must be left alone.
        let abs = ModelCfg { name: "m".into(), scenario: "/etc/npu/y.toml".into(), resident: false };
        assert_eq!(l.scenario_path(&abs), std::path::PathBuf::from("/etc/npu/y.toml"));
    }

    /// A `Servable` standing in for a loaded diarize model, exercising the SHAPE contract the
    /// EngineModel adapter must honour: audio in, segments out.
    struct FakeDiarize;
    impl StreamServable for FakeDiarize {}
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

    fn write_scenario(root: &std::path::Path, weights: &str) {
        std::fs::write(root.join("scenario.toml"), format!(
            "[scenario]\nkind = \"embeddings\"\nname = \"m\"\n[artifacts]\nweights = \"{weights}\"\n"
        )).unwrap();
    }
    fn cfg() -> ModelCfg {
        ModelCfg { name: "m".into(), scenario: "scenario.toml".into(), resident: false }
    }

    /// The pre-load size estimate this needs to exist for at all: a pin-time budget check on a model
    /// that has never been loaded has no live device counter to read, only the weight artifact's own
    /// size on disk.
    #[test]
    fn declared_footprint_sums_every_file_under_the_weight_directory() {
        let dir = tempfile::tempdir().unwrap();
        write_scenario(dir.path(), "w");
        let wdir = dir.path().join("w");
        std::fs::create_dir(&wdir).unwrap();
        std::fs::write(wdir.join("a.bin"), vec![0u8; 100]).unwrap();
        std::fs::write(wdir.join("b.bin"), vec![0u8; 250]).unwrap();
        let l = EngineLoader { root: dir.path().to_path_buf() };
        assert_eq!(l.declared_footprint(&cfg()), Some(350));
    }

    #[test]
    fn declared_footprint_works_when_weights_names_a_single_file() {
        let dir = tempfile::tempdir().unwrap();
        write_scenario(dir.path(), "w.bin");
        std::fs::write(dir.path().join("w.bin"), vec![0u8; 42]).unwrap();
        let l = EngineLoader { root: dir.path().to_path_buf() };
        assert_eq!(l.declared_footprint(&cfg()), Some(42));
    }

    #[test]
    fn declared_footprint_is_none_when_the_scenario_or_weights_do_not_exist() {
        let dir = tempfile::tempdir().unwrap();
        let l = EngineLoader { root: dir.path().to_path_buf() };
        assert_eq!(l.declared_footprint(&cfg()), None, "no scenario.toml at all");
        write_scenario(dir.path(), "nowhere");
        assert_eq!(l.declared_footprint(&cfg()), None, "scenario parses but the weight path is missing");
    }

    fn write_tts_scenario(root: &std::path::Path, slow_ar: &str, fast_ar: &str, codec: &str) {
        std::fs::write(root.join("scenario.toml"), format!(
            "[scenario]\nkind = \"tts\"\nname = \"m\"\n[artifacts]\n\
             [tts]\nslow_ar = \"{slow_ar}\"\nfast_ar = \"{fast_ar}\"\ncodec = \"{codec}\"\n"
        )).unwrap();
    }

    /// A tts model is two weight sets plus a codec, not one `artifacts.weights` dir -- this sums
    /// all three real directories rather than reading (or, worse, mis-reading) a single field.
    #[test]
    fn declared_footprint_sums_all_three_tts_artifact_dirs() {
        let dir = tempfile::tempdir().unwrap();
        for (name, n) in [("slow", 100u8), ("fast", 20), ("codec", 3)] {
            std::fs::write(dir.path().join(name), vec![0u8; n as usize]).unwrap();
        }
        write_tts_scenario(dir.path(), "slow", "fast", "codec");
        let l = EngineLoader { root: dir.path().to_path_buf() };
        assert_eq!(l.declared_footprint(&cfg()), Some(123));
    }

    /// The honesty requirement: an unconfigured tts scenario (every `[tts]` field left at its
    /// empty default) must answer `None`, never `Some(0)` and never the size of the whole root --
    /// `self.root.join("")` resolves to `self.root` itself, so summing an empty field the naive
    /// way would silently walk and total the entire engine root.
    #[test]
    fn declared_footprint_is_none_for_an_unconfigured_tts_scenario_not_zero_or_the_whole_root() {
        let dir = tempfile::tempdir().unwrap();
        // A file that would be wrongly swept in if an empty artifact path resolved to `root`.
        std::fs::write(dir.path().join("unrelated"), vec![0u8; 999]).unwrap();
        write_tts_scenario(dir.path(), "", "", "");
        let l = EngineLoader { root: dir.path().to_path_buf() };
        assert_eq!(l.declared_footprint(&cfg()), None);
    }

    /// Two of three configured is still `None`: a partial footprint would undercount the pin
    /// invariant, which is worse than refusing to answer.
    #[test]
    fn declared_footprint_is_none_when_only_some_tts_artifacts_are_configured() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("slow"), vec![0u8; 100]).unwrap();
        std::fs::write(dir.path().join("fast"), vec![0u8; 20]).unwrap();
        write_tts_scenario(dir.path(), "slow", "fast", "");
        let l = EngineLoader { root: dir.path().to_path_buf() };
        assert_eq!(l.declared_footprint(&cfg()), None);
    }

    /// A `tts` scenario loads with no device at all (nothing composes onto one yet), and `run`/
    /// `run_cancellable` both dispatch `Request::Text` to synthesis rather than falling through to
    /// `embed` -- the bug this whole route existed to fix would otherwise resurface here as a
    /// `WrongKind` (embed) error instead of TTS's own honest `Unsupported`.
    #[test]
    fn a_tts_scenario_loads_with_no_device_and_dispatches_text_to_synthesis() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("scenario.toml"),
            "[scenario]\nkind = \"tts\"\nname = \"m\"\n[artifacts]\n").unwrap();
        let l = EngineLoader { root: dir.path().to_path_buf() };
        let mut m = l.load(&cfg()).expect("a tts scenario must build with no NPU device present");
        assert_eq!(m.capabilities(), Capability::TTS);
        match m.run(Request::Text("hi".into())) {
            Err(EngineError::Unsupported(_)) => {}
            other => panic!("expected TTS's own Unsupported, not embed's WrongKind: {other:?}"),
        }
        match m.run_cancellable(Request::Text("hi".into()), npu_engine::Cancel::new()) {
            Err(EngineError::Unsupported(_)) => {}
            other => panic!("expected TTS's own Unsupported, not embed's WrongKind: {other:?}"),
        }
    }
}
