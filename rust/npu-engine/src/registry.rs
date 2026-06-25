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
    let scen = match cfg.scenario.kind.as_str() {
        "embeddings" => {
            let dev = open_dev()?;
            if cfg.scenario.name.to_lowercase().starts_with("esm") {
                Scenario::Embed(Box::new(crate::esm::EsmEmbedPipeline::build(&cfg, root, dev)))
            } else {
                Scenario::Embed(Box::new(crate::bert::EmbedPipeline::build(&cfg, root, dev)))
            }
        }
        "asr" => {
            if cfg.scenario.name.to_lowercase().contains("parakeet") {
                Scenario::Asr(Box::new(crate::asr::parakeet::ParakeetAsr::build(&cfg, root)))
            } else if cfg.scenario.name.to_lowercase().contains("whisper") {
                Scenario::Asr(Box::new(crate::asr::whisper::WhisperAsr::build(&cfg, root)))
            } else {
                let dev = open_dev()?;
                Scenario::Asr(Box::new(crate::asr::AsrPipeline::build(&cfg, root, dev)))
            }
        }
        other => return Err(EngineError::Load(format!("unknown scenario kind {other:?}"))),
    };
    Ok(scen)
}

/// Panicking convenience wrapper (used by internal bins). Prefer `try_build` / the public `Model` API.
pub fn build(cfg_path: &Path, root: &Path) -> Scenario {
    try_build(cfg_path, root).expect("registry::build")
}
