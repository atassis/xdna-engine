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
    /// `kind = "generate"` only: per-model generation defaults, applied where a request leaves the
    /// field out. A request that names the field always wins -- this sets the default, never a cap.
    #[serde(default)]
    pub generation: GenerationCfg,
    /// Decode-backend tier default for autoregressive decode models (Whisper today). Optional and
    /// additive: absent means "not set", which resolves the same way an absent field always has --
    /// see `resolve_decode_backend`.
    #[serde(default)]
    pub decode: DecodeCfg,
}

/// Per-model generation defaults. Empty block = the engine's own defaults, which is what every
/// scenario got before this existed.
#[derive(Debug, Clone, Default, Deserialize, PartialEq)]
pub struct GenerationCfg {
    /// Default completion budget for this model. Unset falls through to the checkpoint's
    /// `max_new_tokens` and then to the engine's 256 -- OpenAI's number, and not a property of any
    /// model here, which is wrong for a reasoning model whose `<think>` block outgrows it.
    #[serde(default)]
    pub max_tokens: Option<u32>,
    /// Sampling overrides. Each is optional and each sits ABOVE the checkpoint's own
    /// `generation_config.json` and below an explicit request -- set one only to disagree with what
    /// the model ships, which is a deliberate act and should look like one in the config.
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<u32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
}

impl GenerationCfg {
    /// The engine-side tier this block represents.
    pub fn to_defaults(&self) -> crate::pipeline::GenerationDefaults {
        crate::pipeline::GenerationDefaults {
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            max_tokens: self.max_tokens,
            presence_penalty: self.presence_penalty,
            frequency_penalty: self.frequency_penalty,
            repetition_penalty: self.repetition_penalty,
        }
    }
}

/// Per-kind block for `kind = "diarize"`, same shape as `embeddings`. One field on purpose: every
/// hyperparameter lives in the manifest the export script writes, WITH its upstream source, so no
/// pyannote constant is retyped here.
#[derive(Debug, Clone, Default, Deserialize, PartialEq)]
pub struct DiarizationCfg {
    #[serde(default)]
    pub manifest: String,
}

/// Decode-backend tier: a rung on the device ladder, not an
/// implementation name, so a kernel rename or a new artifact dir under an existing tier never
/// touches a scenario file. Named after the three backends `WhisperAsr::build` already has:
/// `FusedDecoder` (whole-decoder ELF, one dispatch/token), the per-op `NPU_DECODE` NPU path
/// (~72 dispatches/token), and the host ONNX decoder graphs -- the one arm BELOW the ladder.
#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum DecodeTier {
    Fused,
    Dispatched,
    Host,
}

impl DecodeTier {
    pub fn as_str(self) -> &'static str {
        match self {
            DecodeTier::Fused => "fused",
            DecodeTier::Dispatched => "dispatched",
            DecodeTier::Host => "host",
        }
    }
}

impl std::fmt::Display for DecodeTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str(self.as_str()) }
}

/// `[decode]` scenario block. One field today, mirroring `DiarizationCfg`'s shape: a scenario that
/// doesn't declare it parses exactly as before this block existed.
#[derive(Debug, Clone, Default, Deserialize, PartialEq)]
pub struct DecodeCfg {
    #[serde(default)]
    pub backend: Option<DecodeTier>,
}

/// Which of the three sources produced a resolved `DecodeTier` -- reportable so a measurement can
/// name what selected the backend, mirroring `npu-cli`'s
/// `config_path_and_source`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeSource {
    /// `NPU_DECODE_FUSED` or `NPU_DECODE`, read directly. Overrides the scenario unconditionally --
    /// unchanged from the pre-existing env-only behavior, so the 13+ scripts that export them keep
    /// working.
    Env(&'static str),
    /// The scenario's own `[decode] backend` field, no env override present.
    Scenario,
    /// Neither an env var nor a scenario field is set: the byte-identical behavior every existing
    /// scenario had before this field existed.
    Default,
}

impl std::fmt::Display for DecodeSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DecodeSource::Env(name) => write!(f, "env ${name}"),
            DecodeSource::Scenario => write!(f, "scenario [decode] backend"),
            DecodeSource::Default => write!(f, "default (no field, no env)"),
        }
    }
}

/// Resolve the decode tier for `cfg`: `NPU_DECODE_FUSED` beats `NPU_DECODE` beats the scenario's
/// `[decode] backend` beats `Host` -- the exact fallback chain `WhisperAsr::build` already had
/// before this field existed, so a scenario with no `[decode]` block and no env vars set resolves
/// to `(Host, Default)`, unchanged. Pure and device-free: reads only `cfg` and process env, no
/// device or artifact I/O, so it is testable without a device (`WhisperAsr::build` is not).
pub fn resolve_decode_backend(cfg: &ScenarioConfig) -> (DecodeTier, DecodeSource) {
    if std::env::var("NPU_DECODE_FUSED").is_ok() {
        return (DecodeTier::Fused, DecodeSource::Env("NPU_DECODE_FUSED"));
    }
    if std::env::var("NPU_DECODE").is_ok() {
        return (DecodeTier::Dispatched, DecodeSource::Env("NPU_DECODE"));
    }
    match cfg.decode.backend {
        Some(tier) => (tier, DecodeSource::Scenario),
        None => (DecodeTier::Host, DecodeSource::Default),
    }
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
    /// `kind = "generate"` only: directory holding a fused-decode ELF (`meta.json` + `decode.elf` +
    /// `buffers/`), read through `llm::LlmArtifact`. Every other scenario kind leaves this empty.
    #[serde(default)]
    pub decode: String,
    /// `kind = "generate"`, optional: directory holding a batched-prefill ELF for the SAME model,
    /// generated to share `decode`'s arena layout. Set it and prompt priming runs `dims.M` positions
    /// per dispatch instead of one; leave it empty (the default) and the rail is exactly the
    /// per-token path. The two artifacts' shared arena offsets are checked at load, so a mismatched
    /// pair fails loud rather than corrupting the weights.
    #[serde(default)]
    pub prefill: String,
    /// `kind = "generate"` only: the checkpoint's directory (`tokenizer.json`,
    /// `tokenizer_config.json`, `generation_config.json`), read through `llm::ModelConfig::load`.
    /// Separate from `tokenizer` above, which every other scenario points at a single
    /// `tokenizer.json` file rather than its containing directory.
    #[serde(default)]
    pub tokenizer_dir: String,
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

    /// `kind = "generate"` needs no `[model]` block at all (the fused decode ELF's own `meta.json`
    /// carries every dimension) -- only the two new `[artifacts]` fields the registry's Generate
    /// arm reads.
    #[test]
    fn generate_scenario_parses_with_no_model_block_and_carries_decode_paths() {
        let toml = std::fs::read_to_string("../../scenarios/generate-qwen3-0.6b.toml").unwrap();
        let c = ScenarioConfig::from_str(&toml).expect("generate scenario must parse");
        assert_eq!(c.scenario.kind, "generate");
        assert!(c.model.is_none());
        // Root-relative, like every other shipped scenario. Absolute artifact paths parse fine and
        // then pin the config to one machine: this scenario shipped first with a /tmp scratchpad
        // and then with a developer's home dir, and neither survives being installed elsewhere.
        for p in [&c.artifacts.decode, &c.artifacts.weights, &c.artifacts.tokenizer_dir] {
            assert!(!p.starts_with('/'), "artifact path must be root-relative, got {p:?}");
            assert!(p.starts_with("artifacts/qwen3-0.6b/"), "unexpected artifact path {p:?}");
        }
    }

    #[test]
    fn an_absent_generation_block_invents_no_budget() {
        // Asserted on an inline scenario, not on a shipped one: a test that reads a real file to
        // check a field is ABSENT pins that file's current content, and fails the moment someone
        // legitimately sets it. This one is about the parse, so it owns its input.
        let c = ScenarioConfig::from_str(
            "[scenario]\nkind = \"generate\"\nname = \"m\"\n             [artifacts]\ndecode = \"d\"\nweights = \"w\"\ntokenizer_dir = \"t\"\n",
        )
        .expect("a scenario with no [generation] block must parse");
        assert_eq!(c.generation.max_tokens, None);
    }

    #[test]
    fn the_shipped_qwen3_scenario_declares_its_own_budget() {
        let toml = std::fs::read_to_string("../../scenarios/generate-qwen3-0.6b.toml").unwrap();
        let c = ScenarioConfig::from_str(&toml).expect("generate scenario must parse");
        assert_eq!(c.generation.max_tokens, Some(1024),
            "the reasoning model ships a budget bigger than the engine's 256");
    }

    #[test]
    fn a_generation_block_sets_the_models_default_completion_budget() {
        let c = ScenarioConfig::from_str(
            "[scenario]\nkind = \"generate\"\nname = \"m\"\n             [artifacts]\ndecode = \"d\"\nweights = \"w\"\ntokenizer_dir = \"t\"\n             [generation]\nmax_tokens = 1024\n",
        )
        .expect("a [generation] block must parse");
        assert_eq!(c.generation.max_tokens, Some(1024));
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

    #[test]
    fn decode_backend_field_parses_each_tier_and_rejects_unknown_values() {
        for (word, want) in
            [("fused", DecodeTier::Fused), ("dispatched", DecodeTier::Dispatched), ("host", DecodeTier::Host)]
        {
            let toml = format!(
                "[scenario]\nkind = \"asr\"\nname = \"m\"\n[artifacts]\nweights = \"w\"\n[decode]\nbackend = \"{word}\"\n"
            );
            let c = ScenarioConfig::from_str(&toml).expect("a valid tier must parse");
            assert_eq!(c.decode.backend, Some(want));
        }
        // E004: a bad value fails loud at config load, naming the value AND the valid set -- not a
        // silent fallback discovered later at resolution time.
        let bad = "[scenario]\nkind = \"asr\"\nname = \"m\"\n[artifacts]\nweights = \"w\"\n[decode]\nbackend = \"fuzed\"\n";
        let err = ScenarioConfig::from_str(bad).expect_err("an unrecognised tier must fail to parse");
        let msg = err.to_string();
        assert!(msg.contains("fuzed"), "error must name the bad value: {msg}");
        assert!(msg.contains("fused") && msg.contains("dispatched") && msg.contains("host"),
            "error must name the valid set: {msg}");
    }

    #[test]
    fn a_scenario_with_no_decode_field_parses_as_none_and_the_shipped_whisper_scenarios_declare_none() {
        let c = ScenarioConfig::from_str(
            "[scenario]\nkind = \"asr\"\nname = \"m\"\n[artifacts]\nweights = \"w\"\n",
        )
        .expect("a scenario with no [decode] block must parse");
        assert_eq!(c.decode.backend, None);

        // Backwards compatibility is a hard requirement: neither shipped Whisper scenario needs
        // touching for this field to exist.
        for f in ["../../scenarios/asr-whisper-small.toml", "../../scenarios/asr-whisper-turbo.toml"] {
            let s = ScenarioConfig::from_str(&std::fs::read_to_string(f).unwrap()).unwrap();
            assert_eq!(s.decode.backend, None, "{f} must not need updating for this field to exist");
        }
    }

    /// Env-var mutation is process-global; every case touching `NPU_DECODE_FUSED`/`NPU_DECODE` lives
    /// in this ONE test (matching `diarize::onnx`'s `NPU_DIARIZE_THREADS` precedent) so `cargo
    /// test`'s parallel test threads cannot interleave two cases and read each other's value.
    #[test]
    fn resolve_decode_backend_order_is_env_over_scenario_over_default() {
        std::env::remove_var("NPU_DECODE_FUSED");
        std::env::remove_var("NPU_DECODE");

        // No field, no env: today's behavior, byte-for-byte -- both flags read `.is_ok()` false, so
        // `WhisperAsr::build` took the ONNX (host) arm before this field existed.
        let no_field = ScenarioConfig::from_str(
            "[scenario]\nkind = \"asr\"\nname = \"m\"\n[artifacts]\nweights = \"w\"\n").unwrap();
        assert_eq!(resolve_decode_backend(&no_field), (DecodeTier::Host, DecodeSource::Default));

        // Scenario field alone -- the new default path -- picks the tier.
        let fused_field = ScenarioConfig::from_str(
            "[scenario]\nkind = \"asr\"\nname = \"m\"\n[artifacts]\nweights = \"w\"\n[decode]\nbackend = \"fused\"\n").unwrap();
        assert_eq!(resolve_decode_backend(&fused_field), (DecodeTier::Fused, DecodeSource::Scenario));
        let dispatched_field = ScenarioConfig::from_str(
            "[scenario]\nkind = \"asr\"\nname = \"m\"\n[artifacts]\nweights = \"w\"\n[decode]\nbackend = \"dispatched\"\n").unwrap();
        assert_eq!(resolve_decode_backend(&dispatched_field), (DecodeTier::Dispatched, DecodeSource::Scenario));
        let host_field = ScenarioConfig::from_str(
            "[scenario]\nkind = \"asr\"\nname = \"m\"\n[artifacts]\nweights = \"w\"\n[decode]\nbackend = \"host\"\n").unwrap();
        assert_eq!(resolve_decode_backend(&host_field), (DecodeTier::Host, DecodeSource::Scenario));

        // The 13+ benchmark scripts that export NPU_DECODE/NPU_DECODE_FUSED must keep working
        // unchanged: env overrides a scenario field that disagrees with it.
        std::env::set_var("NPU_DECODE", "1");
        assert_eq!(resolve_decode_backend(&host_field), (DecodeTier::Dispatched, DecodeSource::Env("NPU_DECODE")));

        // NPU_DECODE_FUSED beats NPU_DECODE when both are set -- existing precedence, unchanged --
        // and beats even a scenario with no [decode] block at all.
        std::env::set_var("NPU_DECODE_FUSED", "1");
        assert_eq!(resolve_decode_backend(&host_field), (DecodeTier::Fused, DecodeSource::Env("NPU_DECODE_FUSED")));
        assert_eq!(resolve_decode_backend(&no_field), (DecodeTier::Fused, DecodeSource::Env("NPU_DECODE_FUSED")));

        std::env::remove_var("NPU_DECODE_FUSED");
        std::env::remove_var("NPU_DECODE");
    }
}
