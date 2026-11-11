//! Causal embedding adapter over the existing final-normed decode hidden state.

use std::cell::RefCell;
use tokenizers::Tokenizer;
use crate::api::EngineError;
use crate::llm::DecodeStep;
use crate::pipeline::Embedder;

pub const MAX_TOKENS: usize = 512;
pub const QUERY_PREFIX: &str = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:";

pub struct QwenEmbedder<D> {
    tokenizer: Tokenizer,
    decode: RefCell<D>,
    dimension: usize,
    metadata: Option<crate::api::EmbeddingMetadata>,
}

impl<D: DecodeStep> QwenEmbedder<D> {
    pub(crate) fn new(mut tokenizer: Tokenizer, decode: D, dimension: usize) -> Result<Self, EngineError> {
        if !matches!(dimension, 1024 | 2560) {
            return Err(EngineError::Load("Qwen embedding requires full dimension 1024 or 2560".into()));
        }
        tokenizer.with_padding(None);
        tokenizer.with_truncation(None).map_err(|e| EngineError::Load(e.to_string()))?;
        Ok(Self { tokenizer, decode: RefCell::new(decode), dimension, metadata: None })
    }

    pub(crate) fn with_metadata(mut self, metadata: crate::api::EmbeddingMetadata) -> Self {
        self.metadata = Some(metadata);
        self
    }

    fn prepare(&self, texts: &[&str], query: bool) -> Result<Vec<Vec<u32>>, EngineError> {
        if texts.is_empty() {
            return Err(EngineError::Unsupported("Qwen embedding input is empty".into()));
        }
        let capacity = self.decode.borrow().max_context().unwrap_or(MAX_TOKENS).min(MAX_TOKENS);
        // Validate the complete batch before mutating device state. Never inherit truncation
        // or padding from tokenizer.json: serial documents have no padding or shared cache prefix.
        texts.iter().enumerate().map(|(i, text)| {
            if text.is_empty() {
                return Err(EngineError::Unsupported(format!("Qwen embedding item {i} is empty")));
            }
            let prepared = if query { format!("{QUERY_PREFIX}{text}") } else { (*text).to_string() };
            let encoded = self.tokenizer.encode(prepared, true)
                .map_err(|e| EngineError::Unsupported(format!("tokenize item {i}: {e}")))?;
            let ids = encoded.get_ids().to_vec();
            if ids.is_empty() || ids.len() > capacity {
                return Err(EngineError::Unsupported(format!(
                    "Qwen embedding item {i} has {} tokens; expected 1..={capacity}, including instruction", ids.len())));
            }
            Ok(ids)
        }).collect()
    }

    fn execute(&self, tokens: &[Vec<u32>]) -> Result<Vec<Vec<f32>>, EngineError> {
        let mut decode = self.decode.borrow_mut();
        let mut vectors = Vec::with_capacity(tokens.len());
        for ids in tokens {
            decode.reset_independent()?;
            // Each document owns fresh state; no prefix ledger is shared across batch items.
            let last = ids.len() - 1;
            let primed = if decode.prefill_batch().is_some() && last > 0 {
                decode.prefill(&ids[..last], 0)?
            } else { 0 };
            if primed > last {
                return Err(EngineError::Device("embedding prefill exceeded its token prefix".into()));
            }
            for pos in primed..ids.len() { decode.step(ids[pos], pos)?; }
            let mut hidden = decode.final_hidden()?.ok_or_else(||
                EngineError::Unsupported("embedding backend does not publish final hidden".into()))?;
            if hidden.len() != self.dimension || hidden.iter().any(|v| !v.is_finite()) {
                return Err(EngineError::Device("invalid embedding hidden dimension or nonfinite value".into()));
            }
            let norm = hidden.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
            if norm == 0. {
                return Err(EngineError::Device("embedding hidden has zero norm".into()));
            }
            for v in &mut hidden { *v = (*v as f64 / norm) as f32; }
            vectors.push(hidden);
        }
        Ok(vectors)
    }

    fn options_query(&self, options: &crate::api::EmbeddingOptions) -> Result<bool, EngineError> {
        if options.dimensions.is_some_and(|d| d != self.dimension) {
            return Err(EngineError::Unsupported(format!("Qwen embedding supports only full dimension {}", self.dimension)));
        }
        Ok(options.input_type == crate::api::EmbeddingInputType::Query)
    }

    pub fn encode(&self, texts: &[&str], query: bool) -> Result<Vec<Vec<f32>>, EngineError> {
        let tokens = self.prepare(texts, query)?;
        self.execute(&tokens)
    }
}

impl<D: DecodeStep> Embedder for QwenEmbedder<D> {
    fn embed_one(&self, text: String) -> Result<Vec<f32>, EngineError> {
        Ok(self.encode(&[&text], false)?.remove(0))
    }
    fn embed_many(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EngineError> {
        self.encode(texts, false)
    }
    fn embed_with_options(&self, texts: &[&str], options: &crate::api::EmbeddingOptions)
        -> Result<Vec<Vec<f32>>, EngineError> {
        let tokens = self.prepare(texts, self.options_query(options)?)?;
        self.execute(&tokens)
    }
    fn embed_with_capture(&self, texts: &[&str], options: &crate::api::EmbeddingOptions)
        -> Result<crate::api::EmbeddingCapture, EngineError> {
        let tokens = self.prepare(texts, self.options_query(options)?)?;
        // Missing loaded identity is an admission failure, never a reason to construct one after
        // decoder state has changed.
        let metadata = self.metadata.clone().ok_or_else(||
            EngineError::Unsupported("Qwen embedding capture requires loaded metadata".into()))?;
        let vectors = self.execute(&tokens)?;
        let attention_mask = tokens.iter().map(|ids| vec![1; ids.len()]).collect();
        Ok(crate::api::EmbeddingCapture { vectors, input_ids: tokens, attention_mask, metadata })
    }
    fn embedding_metadata(&self) -> Option<crate::api::EmbeddingMetadata> { self.metadata.clone() }
    fn bo_bytes(&self) -> u64 { self.decode.borrow().bo_bytes() }
}

pub(crate) fn loaded_metadata(source: &serde_json::Value, dimension: usize, capacity: usize,
    provenance: &crate::telemetry::ArmProvenance, quantization: Option<&serde_json::Value>) -> crate::api::EmbeddingMetadata {
    let profile = serde_json::json!({
        "model":source["model"], "revision":source["revision"], "files":source["files"],
        "query_prefix":QUERY_PREFIX, "document_prefix":"",
        "special_tokens":"tokenizer-add-special-tokens-no-chat-template", "padding":"left",
        "truncation":"reject", "max_tokens":capacity.min(MAX_TOKENS),
        "pooling":"final-backbone-last-valid-token", "normalize":"l2-float32",
        "dimension":dimension, "format":1,
    });
    // Keep execution precision and artifact identity separate from semantic index identity.
    // Only allowlisted runtime facts cross discovery; provenance also contains local paths.
    let runtime = serde_json::json!({
        "backend":"npu-fused-decode", "qualification":"unqualified",
        "index_compatibility":"not-qualified", "batching":"serial-unpadded",
        "hidden_dtype":"bfloat16", "output_dtype":"float32", "normalization_accumulator":"float64",
        "decode_artifact_hash_prefix":provenance.artifact_hash,
        "prefill_artifact_hash_prefix":provenance.prefill_artifact_hash,
        "artifact_hash_format":"sha256-prefix-12-hex", "toolchain_pin_hash":provenance.toolchain_pin_hash,
        "prefill_toolchain_pin_hash":provenance.prefill_toolchain_pin_hash,
        "weight_quantization":quantization.map(|q| serde_json::json!({
            "plan":q.get("plan"), "clip_search":q.get("clip_search"), "full_range":q.get("full_range"),
            "mlp_group_size":q.get("mlp_group_size"), "attn_group_size":q.get("attn_group_size"),
            "qkv_group_size":q.get("qkv_group_size"), "head_group_size":q.get("head_group_size"),
        })),
    });
    crate::api::EmbeddingMetadata { profile, runtime }
}

fn validate_source(meta: &serde_json::Value, model_id: &str) -> Result<serde_json::Value, EngineError> {
    let pins: serde_json::Value = serde_json::from_str(include_str!("../../../../scripts/qwen_embedding_pins.json"))
        .map_err(|e| EngineError::Load(format!("embedding pins: {e}")))?;
    let pin = pins.get(model_id).ok_or_else(|| EngineError::Load("unknown Qwen embedding backbone".into()))?;
    let source = &meta["embedding_source"];
    if meta["spec"] != model_id || source["model"] != pin["repo"] || source["revision"] != pin["revision"]
        || meta["dims"]["d_model"] != pin["dimension"] {
        return Err(EngineError::Load("Qwen embedding artifact identity or dimension mismatch".into()));
    }
    for (name, expected) in pin["files"].as_object().unwrap() {
        let hash = source["files"][name].as_str().unwrap_or_default();
        if hash.len() != 64 || !hash.bytes().all(|b| b.is_ascii_hexdigit())
            || expected.get("sha256").is_some_and(|v| v.as_str() != Some(hash)) {
            return Err(EngineError::Load(format!("Qwen embedding source hash mismatch: {name}")));
        }
    }
    Ok(pin.clone())
}

pub(crate) fn preflight(cfg: &crate::config::ScenarioConfig, root: &std::path::Path)
    -> Result<(Tokenizer, usize, serde_json::Value), EngineError> {
    use sha2::{Digest, Sha256};
    let model_id = cfg.embeddings.backbone.as_deref().unwrap_or_default();
    if cfg.embeddings.pooling != "last" || !cfg.embeddings.normalize || cfg.embeddings.batch_rows.is_some()
        || cfg.artifacts.decode.is_empty() || cfg.artifacts.tokenizer_dir.is_empty() || !cfg.artifacts.resident.is_empty() {
        return Err(EngineError::Load("Qwen embedding requires last pooling, normalization, decode/tokenizer_dir and serial documents".into()));
    }
    let read_json = |path: &std::path::Path| -> Result<serde_json::Value, EngineError> {
        let bytes = std::fs::read(path).map_err(|e| EngineError::Load(format!("{}: {e}", path.display())))?;
        serde_json::from_slice(&bytes).map_err(|e| EngineError::Load(format!("{}: {e}", path.display())))
    };
    let meta = read_json(&root.join(&cfg.artifacts.decode).join("meta.json"))?;
    let pin = validate_source(&meta, model_id)?;
    let dimension = pin["dimension"].as_u64().unwrap() as usize;
    if cfg.model.as_ref().map(|m| m.hidden) != Some(dimension) {
        return Err(EngineError::Load("scenario hidden dimension disagrees with Qwen embedding pin".into()));
    }
    let tokenizer_dir = root.join(&cfg.artifacts.tokenizer_dir);
    let mut verified = std::collections::BTreeMap::new();
    for name in ["config.json", "tokenizer_config.json", "tokenizer.json"] {
        let bytes = std::fs::read(tokenizer_dir.join(name))
            .map_err(|e| EngineError::Load(format!("read embedding {name}: {e}")))?;
        let expected = pin["files"][name]["sha256"].as_str().ok_or_else(||
            EngineError::Load(format!("runtime SHA256 pin for {model_id}/{name} not yet admitted")))?;
        if bytes.len() as u64 != pin["files"][name]["size"].as_u64().unwrap()
            || format!("{:x}", Sha256::digest(&bytes)) != expected {
            return Err(EngineError::Load(format!("pinned embedding file mismatch: {name}")));
        }
        verified.insert(name, bytes);
    }
    let config: serde_json::Value = serde_json::from_slice(&verified["config.json"])
        .map_err(|e| EngineError::Load(format!("embedding config: {e}")))?;
    for (artifact_key, config_key) in [("layers", "num_hidden_layers"), ("d_model", "hidden_size"),
        ("q_heads", "num_attention_heads"), ("kv_heads", "num_key_value_heads"),
        ("head_dim", "head_dim"), ("ffn", "intermediate_size"), ("vocab", "vocab_size")] {
        if meta["dims"][artifact_key].as_u64().is_none() || meta["dims"][artifact_key] != config[config_key] {
            return Err(EngineError::Load(format!("embedding artifact/config geometry mismatch: {artifact_key}")));
        }
    }
    if !cfg.artifacts.prefill.is_empty() {
        let prefill = read_json(&root.join(&cfg.artifacts.prefill).join("meta.json"))?;
        if prefill["embedding_source"] != meta["embedding_source"] {
            return Err(EngineError::Load("embedding prefill/decode source mismatch".into()));
        }
    }
    let tokenizer = Tokenizer::from_bytes(&verified["tokenizer.json"])
        .map_err(|e| EngineError::Load(format!("embedding tokenizer: {e}")))?;
    Ok((tokenizer, dimension, meta["embedding_source"].clone()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::rc::Rc;
    use std::cell::Cell;
    use crate::llm::CacheState;
    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::pre_tokenizers::whitespace::Whitespace;

    struct Decoder {
        calls: Rc<RefCell<Vec<(u32, usize)>>>,
        resets: Rc<Cell<usize>>,
        hidden: Option<Vec<f32>>,
        capacity: usize,
        primed: bool,
    }
    impl DecodeStep for Decoder {
        fn step(&mut self, token: u32, pos: usize) -> Result<Vec<f32>, EngineError> {
            self.calls.borrow_mut().push((token, pos));
            Ok(vec![])
        }
        fn reset(&mut self) -> Result<CacheState, EngineError> {
            self.resets.set(self.resets.get() + 1);
            Ok(CacheState::Cleared)
        }
        fn max_context(&self) -> Option<usize> { Some(self.capacity) }
        fn prefill_batch(&self) -> Option<usize> { Some(2) }
        fn prefill(&mut self, tokens: &[u32], from: usize) -> Result<usize, EngineError> {
            assert_eq!(from, 0);
            self.primed = true;
            Ok(tokens.len())
        }
        fn final_hidden(&mut self) -> Result<Option<Vec<f32>>, EngineError> { Ok(self.hidden.clone()) }
    }
    fn fixture(hidden: Option<Vec<f32>>, capacity: usize) -> QwenEmbedder<Decoder> {
        let vocab = [("[UNK]".to_string(), 0), ("a".to_string(), 1), ("b".to_string(), 2)];
        let model = WordLevel::builder().vocab(vocab.into_iter().collect()).unk_token("[UNK]".into()).build().unwrap();
        let mut tok = Tokenizer::new(model);
        tok.with_pre_tokenizer(Some(Whitespace));
        QwenEmbedder::new(tok, Decoder { calls: Rc::default(), resets: Rc::default(), hidden,
            capacity, primed: false }, 1024).unwrap()
    }
    fn hidden() -> Vec<f32> { let mut v = vec![0.; 1024]; v[0] = 3.; v[1] = 4.; v }
    fn metadata() -> crate::api::EmbeddingMetadata {
        crate::api::EmbeddingMetadata {
            profile: serde_json::json!({"loaded": "profile"}),
            runtime: serde_json::json!({"loaded": "runtime"}),
        }
    }

    #[test]
    #[ignore = "requires explicitly selected local checkpoint and offline artifact; no device"]
    fn pinned_tokenizer_matches_cpu_reference_and_preflight() {
        let scenario = std::env::var("QWEN_EMBEDDING_SCENARIO").expect("QWEN_EMBEDDING_SCENARIO");
        let cfg = crate::config::ScenarioConfig::load(std::path::Path::new(&scenario)).unwrap();
        let (mut tokenizer, dimension, source) = preflight(&cfg, std::path::Path::new(".")).unwrap();
        let model_cfg = cfg.model.as_ref().expect("embedding model geometry");
        assert_eq!(dimension, model_cfg.hidden);
        let decode_artifact = crate::llm::LlmArtifact::load(&std::path::Path::new(".").join(&cfg.artifacts.decode)).unwrap();
        assert_eq!(decode_artifact.embedding_source.as_ref(), Some(&source));
        if !cfg.artifacts.prefill.is_empty() {
            let prefill = crate::llm::LlmArtifact::load_prefill(&std::path::Path::new(".").join(&cfg.artifacts.prefill)).unwrap();
            decode_artifact.check_prefill_pairing(&prefill).unwrap();
        }
        tokenizer.with_padding(None);
        tokenizer.with_truncation(None).unwrap();
        for (kind, query) in [("document", false), ("query", true)] {
            let root = std::path::PathBuf::from(std::env::var("QWEN_EMBEDDING_REFERENCE_DIR").unwrap());
            let texts: Vec<String> = serde_json::from_slice(&std::fs::read(root.join(format!("{kind}-input.json"))).unwrap()).unwrap();
            let reference: serde_json::Value = serde_json::from_slice(&std::fs::read(root.join(format!("{kind}-reference.json"))).unwrap()).unwrap();
            let metadata = loaded_metadata(&source, dimension, model_cfg.max_seq, &Default::default(), None);
            assert_eq!(metadata.profile, reference["profile"], "semantic profile matches reference; runtime is not qualified");
            for (i, text) in texts.iter().enumerate() {
                let prepared = if query { format!("{QUERY_PREFIX}{text}") } else { text.clone() };
                let encoded = tokenizer.encode(prepared, true).unwrap();
                let expected: Vec<u32> = reference["input_ids"][i].as_array().unwrap().iter()
                    .zip(reference["attention_mask"][i].as_array().unwrap()).filter(|(_, mask)| mask.as_u64() == Some(1))
                    .map(|(token, _)| token.as_u64().unwrap() as u32).collect();
                assert_eq!(encoded.get_ids(), expected, "{kind} item {i}");
            }
        }
        let mut invalid = cfg.clone(); invalid.embeddings.pooling = "mean".into();
        assert!(preflight(&invalid, std::path::Path::new(".")).is_err());
        let mut invalid = cfg.clone(); invalid.embeddings.backbone = Some("qwen3-0.6b".into());
        assert!(preflight(&invalid, std::path::Path::new(".")).is_err());
        let mut invalid = cfg.clone(); invalid.model.as_mut().unwrap().hidden += 1;
        assert!(preflight(&invalid, std::path::Path::new(".")).is_err());

        let temp = tempfile::tempdir().unwrap();
        let mut invalid = cfg.clone();
        invalid.artifacts.tokenizer_dir = temp.path().to_string_lossy().into_owned();
        for name in ["config.json", "tokenizer_config.json", "tokenizer.json"] {
            std::fs::copy(std::path::Path::new(&cfg.artifacts.tokenizer_dir).join(name), temp.path().join(name)).unwrap();
        }
        for name in ["config.json", "tokenizer_config.json", "tokenizer.json"] {
            let path = temp.path().join(name);
            let original = std::fs::read(&path).unwrap();
            let mut changed = original.clone(); changed[0] ^= 1;
            std::fs::write(&path, changed).unwrap();
            assert!(preflight(&invalid, std::path::Path::new(".")).is_err(), "changed {name}");
            std::fs::remove_file(&path).unwrap();
            assert!(preflight(&invalid, std::path::Path::new(".")).is_err(), "missing {name}");
            std::fs::write(&path, original).unwrap();
        }
        assert!(preflight(&invalid, std::path::Path::new(".")).is_ok());

        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("meta.json");
        let mut changed: serde_json::Value = serde_json::from_slice(&std::fs::read(
            std::path::Path::new(&cfg.artifacts.decode).join("meta.json")).unwrap()).unwrap();
        changed["embedding_source"]["revision"] = "0".repeat(40).into();
        std::fs::write(&path, serde_json::to_vec(&changed).unwrap()).unwrap();
        let mut invalid = cfg.clone(); invalid.artifacts.decode = temp.path().to_string_lossy().into_owned();
        assert!(preflight(&invalid, std::path::Path::new(".")).is_err());
    }

    fn source_fixture() -> serde_json::Value {
        let pins: serde_json::Value = serde_json::from_str(include_str!("../../../../scripts/qwen_embedding_pins.json")).unwrap();
        let pin = &pins["qwen3-embedding-0.6b"];
        let files: serde_json::Map<String, serde_json::Value> = pin["files"].as_object().unwrap().iter()
            .map(|(name, value)| (name.clone(), value.get("sha256").cloned().unwrap_or_else(|| "0".repeat(64).into()))).collect();
        serde_json::json!({"spec":"qwen3-embedding-0.6b", "dims":{"d_model":1024},
            "embedding_source":{"model":pin["repo"],"revision":pin["revision"],"files":files}})
    }
    #[test]
    fn profile_identity_is_semantic_runtime_is_separate_and_no_paths_escape() {
        use crate::telemetry::ArmProvenance;
        let source = source_fixture()["embedding_source"].clone();
        let provenance = ArmProvenance { artifact_hash: Some("0123456789ab".into()),
            artifact_path: Some("/do-not-expose/decode".into()), prefill_artifact_path: Some("/do-not-expose/prefill".into()),
            prefill_artifact_hash: Some("abcdef012345".into()), ..Default::default() };
        let first = loaded_metadata(&source, 1024, 2048, &provenance, Some(&serde_json::json!({"plan":{"qkv":"bf16"}})));
        assert_eq!(first.profile["dimension"], 1024);
        assert_eq!(first.profile["max_tokens"], 512);
        assert_eq!(first.profile["query_prefix"], QUERY_PREFIX);
        assert_eq!(first.profile["files"], source["files"]);
        assert_eq!(first.runtime["qualification"], "unqualified");
        assert_eq!(first.runtime["decode_artifact_hash_prefix"], "0123456789ab");
        assert!(!serde_json::to_string(&first).unwrap().contains("do-not-expose"));
        let mut changed = provenance.clone(); changed.artifact_hash = Some("changed".into());
        let second = loaded_metadata(&source, 1024, 2048, &changed, None);
        assert_eq!(first.profile, second.profile);
        assert_ne!(first.runtime, second.runtime);
        let mut changed_source = source.clone(); changed_source["revision"] = "changed".into();
        assert_ne!(first.profile, loaded_metadata(&changed_source, 1024, 2048, &provenance, None).profile);
        assert_ne!(first.profile, loaded_metadata(&source, 1024, 256, &provenance, None).profile);
        assert_ne!(first.profile, loaded_metadata(&source, 2560, 2048, &provenance, None).profile);
    }

    #[test]
    fn rejects_generation_identity_changed_revision_and_missing_source_files() {
        let good = source_fixture();
        assert!(validate_source(&good, "qwen3-embedding-0.6b").is_ok());
        let mut bad = good.clone(); bad["embedding_source"]["revision"] = "other".into();
        assert!(validate_source(&bad, "qwen3-embedding-0.6b").is_err());
        let mut bad = good.clone(); bad["spec"] = "qwen3-0.6b".into();
        assert!(validate_source(&bad, "qwen3-embedding-0.6b").is_err());
        let mut bad = good.clone(); bad["embedding_source"]["files"].as_object_mut().unwrap().remove("tokenizer.json");
        assert!(validate_source(&bad, "qwen3-embedding-0.6b").is_err());
        let mut bad = good.clone(); bad["embedding_source"]["files"]["model.safetensors"] = "0".repeat(64).into();
        assert!(validate_source(&bad, "qwen3-embedding-0.6b").is_err());
        assert!(validate_source(&good, "qwen3-embedding-4b").is_err());
    }

    #[test]
    fn last_token_normalized_and_each_item_restarts_at_zero() {
        let model = fixture(Some(hidden()), MAX_TOKENS);
        let vectors = model.embed_many(&["a b a", "b", "a b a"]).unwrap();
        assert_eq!(vectors.len(), 3);
        assert!((vectors[0][0] - 0.6).abs() < 1e-6);
        assert!((vectors[0][1] - 0.8).abs() < 1e-6);
        assert_eq!(vectors[0], vectors[2]);
        let decoder = model.decode.borrow();
        assert_eq!(decoder.resets.get(), 3);
        // Prefill primes all but the last token; the last position must go through step.
        assert_eq!(*decoder.calls.borrow(), vec![(1, 2), (2, 0), (1, 2)]);
        assert!(decoder.primed);
    }
    #[test]
    fn retained_state_without_an_independent_reset_is_not_admitted() {
        struct Retained;
        impl DecodeStep for Retained {
            fn step(&mut self, _: u32, _: usize) -> Result<Vec<f32>, EngineError> { Ok(vec![]) }
            fn final_hidden(&mut self) -> Result<Option<Vec<f32>>, EngineError> { Ok(Some(hidden())) }
        }
        let model = fixture(Some(hidden()), MAX_TOKENS);
        let retained = QwenEmbedder::new(model.tokenizer.clone(), Retained, 1024).unwrap();
        assert!(retained.embed_one("a".into()).is_err());
    }

    #[test]
    fn rejects_entire_invalid_batch_before_any_decode() {
        for texts in [vec![], vec!["a", ""], vec!["a", "a b a"]] {
            let model = fixture(Some(hidden()), 2);
            assert!(model.embed_many(&texts).is_err());
            assert_eq!(model.decode.borrow().resets.get(), 0);
        }
    }
    #[test]
    fn policy_boundary_counts_tokens_without_truncation() {
        let model = fixture(Some(hidden()), 2048);
        assert!(model.embed_one(vec!["a"; 512].join(" ")).is_ok());
        let before = model.decode.borrow().resets.get();
        assert!(model.embed_one(vec!["a"; 513].join(" ")).is_err());
        assert_eq!(model.decode.borrow().resets.get(), before);
        assert!(model.encode(&[&vec!["a"; 512].join(" ")], true).is_err());
    }
    #[test]
    fn rejects_missing_wrong_sized_zero_and_nonfinite_hidden() {
        for hidden in [None, Some(vec![1.; 8]), Some(vec![0.; 1024]),
                       Some(vec![f32::NAN; 1024]), Some(vec![f32::INFINITY; 1024])] {
            assert!(fixture(hidden, MAX_TOKENS).embed_one("a".into()).is_err());
        }
    }
    #[test]
    fn options_require_full_dimensions_and_count_query_instructions() {
        use crate::api::{EmbeddingInputType, EmbeddingOptions};
        let model = fixture(Some(hidden()), 1);
        assert!(model.embed_with_options(&["a"], &EmbeddingOptions { input_type: EmbeddingInputType::Document, dimensions: Some(1024) }).is_ok());
        let before = model.decode.borrow().resets.get();
        for dimensions in [Some(0), Some(512), Some(2560)] {
            assert!(model.embed_with_options(&["a"], &EmbeddingOptions { input_type: EmbeddingInputType::Document, dimensions }).is_err());
        }
        assert!(model.embed_with_options(&["a"], &EmbeddingOptions { input_type: EmbeddingInputType::Query, dimensions: None }).is_err());
        assert_eq!(model.decode.borrow().resets.get(), before);
        let mut wide = fixture(Some(vec![1.; 2560]), MAX_TOKENS);
        wide.dimension = 2560;
        assert_eq!(wide.embed_one("a".into()).unwrap().len(), 2560);
    }

    #[test]
    fn capture_uses_normal_preparation_and_execution_with_loaded_metadata() {
        use crate::api::{EmbeddingInputType, EmbeddingOptions};
        let normal = fixture(Some(hidden()), MAX_TOKENS);
        let captured = fixture(Some(hidden()), MAX_TOKENS).with_metadata(metadata());
        let texts = ["a b", "a", "a b"];
        let options = EmbeddingOptions { input_type: EmbeddingInputType::Query, dimensions: Some(1024) };
        let vectors = normal.embed_with_options(&texts, &options).unwrap();
        let capture = captured.embed_with_capture(&texts, &options).unwrap();
        assert_eq!(capture.vectors, vectors);
        assert_eq!(capture.metadata, metadata());
        assert_eq!(capture.input_ids[0], captured.tokenizer.encode(format!("{QUERY_PREFIX}a b"), true).unwrap().get_ids());
        assert_eq!(capture.input_ids[1], captured.tokenizer.encode(format!("{QUERY_PREFIX}a"), true).unwrap().get_ids());
        assert_eq!(capture.input_ids[0], capture.input_ids[2]);
        assert_eq!(capture.attention_mask, capture.input_ids.iter().map(|ids| vec![1; ids.len()]).collect::<Vec<_>>());
        let decoder = captured.decode.borrow();
        assert_eq!(decoder.resets.get(), 3);
        assert!(decoder.primed);
    }

    #[test]
    fn capture_requires_loaded_metadata_and_validates_before_decode() {
        use crate::api::{EmbeddingInputType, EmbeddingOptions};
        let model = fixture(Some(hidden()), MAX_TOKENS);
        let options = EmbeddingOptions::default();
        assert!(model.embed_with_capture(&["a"], &options).is_err());
        assert_eq!(model.decode.borrow().resets.get(), 0);
        let with_metadata = fixture(Some(hidden()), MAX_TOKENS).with_metadata(metadata());
        let invalid = EmbeddingOptions { input_type: EmbeddingInputType::Document, dimensions: Some(1) };
        assert!(with_metadata.embed_with_capture(&["a", ""], &invalid).is_err());
        assert_eq!(with_metadata.decode.borrow().resets.get(), 0);
    }

    #[test]
    fn normalization_does_not_overflow_for_finite_f32_hidden() {
        let vectors = fixture(Some(vec![f32::MAX; 1024]), MAX_TOKENS).embed_many(&["a"]).unwrap();
        assert!((vectors[0].iter().map(|v| (*v as f64).powi(2)).sum::<f64>() - 1.).abs() < 1e-6);
    }
}
