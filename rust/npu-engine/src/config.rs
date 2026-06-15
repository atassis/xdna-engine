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
    pub weights: String,
    #[serde(default)]
    pub tokenizer: String,
    #[serde(default)]
    pub onnx_ref: String,
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
