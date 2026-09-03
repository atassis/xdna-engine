//! Declarative scenario manifest: everything that varies between models.

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct ScenarioConfig {
    pub scenario: Scenario,
    /// OPTIONAL because it is transformer-shaped and not every model is a transformer. Parakeet,
    /// Whisper and GigaAM already read none of it; PyanNet and ResNet34 have none of these fields.
    /// Builders that need it `ok_or` on it, so a missing block is a loud error rather than six
    /// fabricated numbers reaching a model that cannot use them.
    #[serde(default)]
    pub model: Option<ModelCfg>,
    pub artifacts: Artifacts,
    #[serde(default)]
    pub embeddings: EmbeddingsCfg,
    #[serde(default)]
    pub diarization: DiarizationCfg,
}

/// Per-kind block for `kind = "diarize"`, same shape as `embeddings`. One field on purpose: every
/// hyperparameter lives in the manifest the export script writes, WITH its upstream source, so no
/// pyannote constant is retyped here.
#[derive(Debug, Clone, Default, Deserialize, PartialEq)]
pub struct DiarizationCfg {
    #[serde(default)]
    pub manifest: String,
}

impl ScenarioConfig {
    pub fn from_str(s: &str) -> Result<ScenarioConfig, toml::de::Error> { toml::from_str(s) }

    /// The `[model]` block, or a loud error naming the scenario that lacks it.
    pub fn model_or_err(&self) -> Result<&ModelCfg, String> {
        self.model.as_ref().ok_or_else(|| format!(
            "scenario {:?} (kind {:?}) needs a [model] block",
            self.scenario.name, self.scenario.kind))
    }
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct Scenario {
    pub kind: String, // "asr" | "embeddings"
    pub name: String,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct ModelCfg {
    pub hidden: usize,
    pub ff: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub n_layers: usize,
    pub max_seq: usize,
    /// Log-mel filterbank channels the frontend produces. 80 for every Whisper before large-v3,
    /// 128 from large-v3 on; also the encoder conv stem's input channel count.
    #[serde(default = "default_n_mels")]
    pub n_mels: usize,
    /// Decoder depth, when the model has a decoder and it differs from `n_layers` (which is the
    /// ENCODER depth). whisper-small is 12/12; large-v3-turbo is 32 encoder / 4 decoder. Defaults to
    /// `n_layers` via `ModelCfg::decoder_layers`, so every existing scenario is unchanged.
    #[serde(default)]
    pub n_decoder_layers: Option<usize>,
    #[serde(default = "default_precision")]
    pub precision: String, // native | bf16 | int8
    #[serde(default = "default_kernel")]
    pub kernel: String, // zeropad | native (ESM matmul-shape strategy)
}
fn default_precision() -> String { "bf16".into() }
fn default_kernel() -> String { "zeropad".into() }
fn default_n_mels() -> usize { 80 }

impl ModelCfg {
    /// Decoder depth: the explicit `n_decoder_layers` when set, else `n_layers` (encoder-only models
    /// and every model whose two stacks are the same depth).
    pub fn decoder_layers(&self) -> usize { self.n_decoder_layers.unwrap_or(self.n_layers) }
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct Artifacts {
    /// Legacy npy/f32 weights directory. Still the default path when no declarative `source` is set,
    /// so existing scenarios parse + load byte-identically.
    #[serde(default)]
    pub weights: String,
    #[serde(default)]
    pub tokenizer: String,
    #[serde(default)]
    pub onnx_ref: String,
    /// Declarative weight source: `"hf:<repo>[@rev]"` or `"path:/abs"`. When set, the engine
    /// resolves + bakes (on missing) a `npu-weights` checkpoint via this spec instead of reading the
    /// legacy npy `weights` dir. Optional and additive: omit it and the npy path is unchanged.
    #[serde(default)]
    pub source: String,
    /// `npu-weights` arch name driving the bake transform: `bert|esm|vit|opt|whisper|fastconformer|gigaam`.
    /// Required when `source` is set; ignored otherwise.
    #[serde(default)]
    pub arch: String,
    /// Optional explicit checkpoint `.safetensors` path. When empty the checkpoint path is derived
    /// (`${XDNA_CHECKPOINT_DIR:-<root>/artifacts/checkpoints}/<arch>__<src>__<fp>.safetensors`).
    #[serde(default)]
    #[serde(alias = "arena")]
    pub checkpoint: String,
}

impl Artifacts {
    /// Build a declarative `npu_weights::spec::ModelSpec` from the `source`/`arch`/`checkpoint` fields,
    /// or `None` when no `source` is configured (legacy npy path). Errors on a malformed source or
    /// a `source` without an `arch`.
    pub fn model_spec(&self) -> anyhow::Result<Option<npu_weights::spec::ModelSpec>> {
        if self.source.is_empty() {
            return Ok(None);
        }
        anyhow::ensure!(!self.arch.is_empty(),
            "artifacts.source is set but artifacts.arch is empty (need bert|esm|vit|opt|whisper|fastconformer|gigaam)");
        let source = npu_weights::spec::Source::parse(&self.source)?;
        let checkpoint = if self.checkpoint.is_empty() {
            None
        } else {
            Some(std::path::PathBuf::from(&self.checkpoint))
        };
        Ok(Some(npu_weights::spec::ModelSpec { source, arch: self.arch.clone(), checkpoint }))
    }
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct EmbeddingsCfg {
    #[serde(default = "default_pooling")]
    pub pooling: String, // mean | cls
    #[serde(default = "default_true")]
    pub normalize: bool,
}
impl Default for EmbeddingsCfg {
    fn default() -> Self { EmbeddingsCfg { pooling: default_pooling(), normalize: true } }
}
fn default_pooling() -> String { "mean".into() }
fn default_true() -> bool { true }

impl ScenarioConfig {
    pub fn from_toml_str(s: &str) -> Result<Self, toml::de::Error> {
        toml::from_str(s)
    }
    pub fn load(path: &std::path::Path) -> std::io::Result<Self> {
        let s = std::fs::read_to_string(path)?;
        Self::from_toml_str(&s)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_embeddings_scenario() {
        let toml = r#"
[scenario]
kind = "embeddings"
name = "bge-base-en-v1.5"
[model]
hidden = 768
ff = 3072
n_heads = 12
head_dim = 64
n_layers = 12
max_seq = 512
[artifacts]
weights = "artifacts/bge-base/encoder"
tokenizer = "artifacts/bge-base/tokenizer.json"
onnx_ref = "artifacts/bge-base/model.onnx"
[embeddings]
pooling = "mean"
normalize = true
"#;
        let c = ScenarioConfig::from_toml_str(toml).expect("parse");
        assert_eq!(c.scenario.kind, "embeddings");
        assert_eq!(c.model.as_ref().unwrap().hidden, 768);
        assert_eq!(c.model.as_ref().unwrap().precision, "bf16"); // default applied
        assert!(c.embeddings.normalize);
    }

    #[test]
    fn a_scenario_without_a_model_block_parses_and_carries_its_diarization_manifest() {
        let toml = r#"
[scenario]
kind = "diarize"
name = "pyannote-speaker-diarization-3.1"
[artifacts]
weights = "artifacts/pyannote"
[diarization]
manifest = "artifacts/pyannote/diarize.json"
"#;
        let c = ScenarioConfig::from_str(toml).expect("a non-transformer scenario must parse");
        assert!(c.model.is_none(), "PyanNet has none of the six transformer fields");
        assert_eq!(c.diarization.manifest, "artifacts/pyannote/diarize.json");
        // ...and every shipped scenario still parses WITH its block. cwd is the crate root.
        let bge = std::fs::read_to_string("../../scenarios/bge-base.toml").unwrap();
        let b = ScenarioConfig::from_str(&bge).unwrap();
        assert_eq!(b.model.as_ref().unwrap().hidden, 768);
        assert!(b.diarization.manifest.is_empty(), "an absent block defaults, never errors");
    }

    /// The two fields a second Whisper size needs, and the guarantee that the first one does not
    /// have to name them: whisper-small's shipped scenario declares neither.
    #[test]
    fn whisper_shape_fields_default_to_small_and_parse_for_turbo() {
        let small = ScenarioConfig::from_str(
            &std::fs::read_to_string("../../scenarios/asr-whisper-small.toml").unwrap()).unwrap();
        let m = small.model.as_ref().unwrap();
        assert_eq!(m.n_mels, 80, "pre-large-v3 Whisper is 80-mel");
        assert_eq!(m.decoder_layers(), m.n_layers, "small is 12 encoder / 12 decoder");

        let turbo = ScenarioConfig::from_str(
            &std::fs::read_to_string("../../scenarios/asr-whisper-turbo.toml").unwrap()).unwrap();
        let m = turbo.model.as_ref().unwrap();
        assert_eq!((m.hidden, m.ff, m.n_heads, m.n_layers), (1280, 5120, 20, 32));
        assert_eq!(m.n_mels, 128, "large-v3 and later are 128-mel");
        assert_eq!(m.decoder_layers(), 4, "turbo's decoder is 4 layers, not its 32 encoder layers");
    }
}
