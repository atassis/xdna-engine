//! Scenario TOML -> assembled pipeline.

use std::path::Path;
use std::rc::Rc;

use npu_xrt::Device;

use crate::config::ScenarioConfig;
use crate::pipeline::Scenario;

/// Build a scenario from a TOML path. `root` is the repo root (where artifacts/ and mlir-aie/ live).
/// Note: the Parakeet arm opens its own NPU device inside `new_npu`; the device is NOT opened here
/// for that arm to avoid double-opening.
pub fn build(cfg_path: &Path, root: &Path) -> Scenario {
    let cfg = ScenarioConfig::load(cfg_path).expect("load scenario");
    match cfg.scenario.kind.as_str() {
        "embeddings" => {
            let dev = Rc::new(Device::open(0).expect("open NPU (stop other ASR/embeddings service first)"));
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
                // Whisper opens its own NPU device inside new_npu (like parakeet); don't open here.
                Scenario::Asr(Box::new(crate::asr::whisper::WhisperAsr::build(&cfg, root)))
            } else {
                let dev = Rc::new(Device::open(0).expect("open NPU (stop other ASR service first)"));
                Scenario::Asr(Box::new(crate::asr::AsrPipeline::build(&cfg, root, dev)))
            }
        }
        other => panic!("unknown scenario kind {other:?}"),
    }
}
