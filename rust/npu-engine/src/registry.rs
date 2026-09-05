//! Scenario TOML -> assembled pipeline.

use std::path::Path;
use std::rc::Rc;

use npu_xrt::Device;

use crate::api::EngineError;
use crate::config::ScenarioConfig;
use crate::pipeline::Scenario;

/// Build a scenario from a TOML path. Returns an error instead of panicking. `root` is the repo
/// root (where artifacts/ and mlir-aie/ live). Note: the Parakeet/Whisper arms open their own NPU
/// device internally; the device is NOT opened here for those arms to avoid double-opening.
pub fn try_build(cfg_path: &Path, root: &Path) -> Result<Scenario, EngineError> {
    let cfg = ScenarioConfig::load(cfg_path)
        .map_err(|e| EngineError::Load(format!("scenario {}: {e}", cfg_path.display())))?;
    let open_dev = || -> Result<Rc<Device>, EngineError> {
        Device::open(0)
            .map(Rc::new)
            .map_err(|e| EngineError::Device(format!("open NPU (stop other ASR/embeddings service first): {e}")))
    };
    // Dispatch on the SAME answer `ModelKind::from_scenario_kind` gives the control plane before any
    // load, so a declared capability and the built one cannot disagree.
    let scen = match crate::ModelKind::from_scenario_kind(&cfg.scenario.kind) {
        Some(crate::ModelKind::Embed) => {
            let dev = open_dev()?;
            if cfg.scenario.name.to_lowercase().starts_with("esm") {
                Scenario::Embed(Box::new(crate::esm::EsmEmbedPipeline::build(&cfg, root, dev)?))
            } else {
                Scenario::Embed(Box::new(crate::bert::EmbedPipeline::build(&cfg, root, dev)?))
            }
        }
        Some(crate::ModelKind::Asr) => {
            if cfg.scenario.name.to_lowercase().contains("parakeet") {
                Scenario::Asr(Box::new(crate::asr::parakeet::ParakeetAsr::build(&cfg, root)?))
            } else if cfg.scenario.name.to_lowercase().contains("whisper") {
                Scenario::Asr(Box::new(crate::asr::whisper::WhisperAsr::build(&cfg, root)?))
            } else {
                let dev = open_dev()?;
                Scenario::Asr(Box::new(crate::asr::AsrPipeline::build(&cfg, root, dev)?))
            }
        }
        // NO open_dev(): v1 diarization is host-only, so a resident diarize model must not take a
        // hardware context away from the ASR model it coexists with.
        Some(crate::ModelKind::Diarize) => {
            let mpath = std::path::Path::new(&cfg.diarization.manifest);
            let mpath = if mpath.is_absolute() { mpath.to_path_buf() } else { root.join(mpath) };
            let txt = std::fs::read_to_string(&mpath).map_err(|e| EngineError::Load(
                format!("diarize manifest {}: {e}", mpath.display())))?;
            let manifest: crate::diarize::Manifest = serde_json::from_str(&txt).map_err(|e|
                EngineError::Load(format!("diarize manifest {}: {e}", mpath.display())))?;
            let dir = mpath.parent().unwrap_or(root).to_path_buf();
            let seg = crate::diarize::onnx::OnnxSegmenter::build(&manifest, &dir)?;
            let emb = crate::diarize::onnx::OnnxEmbedder::build(&manifest, &dir)?;
            Scenario::Diarize(Box::new(crate::diarize::DiarizePipeline::new(
                manifest, Box::new(seg), Box::new(emb), &dir)?))
        }
        // Placeholder arm, landed with the `TextGenerator` contract so the tree compiles while the
        // decoder is built (`llm-serve-openai-surface`). It FAILS LOUD rather than falling back to a
        // host implementation: a `kind = "generate"` scenario that silently served something else
        // would be indistinguishable from a working one until someone measured it.
        Some(crate::ModelKind::Generate) => return Err(EngineError::Load(format!(
            "scenario {:?} declares kind=generate, but no LLM decoder is wired yet \
             (llm-serve-openai-surface)", cfg.scenario.name))),
        None => return Err(EngineError::Load(format!("unknown scenario kind {:?}", cfg.scenario.kind))),
    };
    Ok(scen)
}

/// Panicking convenience wrapper (used by internal bins). Prefer `try_build` / the public `Model` API.
pub fn build(cfg_path: &Path, root: &Path) -> Scenario {
    try_build(cfg_path, root).expect("registry::build")
}
