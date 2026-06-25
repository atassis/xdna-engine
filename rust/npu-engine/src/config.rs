//! Declarative scenario manifest: everything that varies between models.

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct ScenarioConfig {
    pub scenario: Scenario,
    pub model: ModelCfg,
    pub artifacts: Artifacts,
    #[serde(default)]
    pub embeddings: EmbeddingsCfg,
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
    #[serde(default = "default_precision")]
    pub precision: String, // native | bf16 | int8
    #[serde(default = "default_kernel")]
    pub kernel: String, // zeropad | native (ESM matmul-shape strategy)
}
fn default_precision() -> String { "bf16".into() }
fn default_kernel() -> String { "zeropad".into() }

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
    /// resolves + bakes (on missing) a `npu-weights` arena via this spec instead of reading the
    /// legacy npy `weights` dir. Optional and additive: omit it and the npy path is unchanged.
    #[serde(default)]
    pub source: String,
    /// `npu-weights` arch name driving the bake transform: `bert|esm|vit|opt|whisper|fastconformer|gigaam`.
    /// Required when `source` is set; ignored otherwise.
    #[serde(default)]
    pub arch: String,
    /// Optional explicit arena `.safetensors` path. When empty the arena path is derived
    /// (`${XDNA_ARENA_DIR:-<root>/artifacts/arenas}/<arch>__<src>__<fp>.safetensors`).
    #[serde(default)]
    pub arena: String,
}

impl Artifacts {
    /// Build a declarative `npu_weights::spec::ModelSpec` from the `source`/`arch`/`arena` fields,
    /// or `None` when no `source` is configured (legacy npy path). Errors on a malformed source or
    /// a `source` without an `arch`.
    pub fn model_spec(&self) -> anyhow::Result<Option<npu_weights::spec::ModelSpec>> {
        if self.source.is_empty() {
            return Ok(None);
        }
        anyhow::ensure!(!self.arch.is_empty(),
            "artifacts.source is set but artifacts.arch is empty (need bert|esm|vit|opt|whisper|fastconformer|gigaam)");
        let source = npu_weights::spec::Source::parse(&self.source)?;
        let arena = if self.arena.is_empty() {
            None
        } else {
            Some(std::path::PathBuf::from(&self.arena))
        };
        Ok(Some(npu_weights::spec::ModelSpec { source, arch: self.arch.clone(), arena }))
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
        assert_eq!(c.model.hidden, 768);
        assert_eq!(c.model.precision, "bf16"); // default applied
        assert!(c.embeddings.normalize);
    }
}
