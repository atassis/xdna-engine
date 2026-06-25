//! Host-only (no NPU) test for A4: declarative weight loading wired into the scenario config.
//!
//! Proves the uniform entry point: a scenario TOML that declares `source/arch/arena` parses,
//! resolves a `npu_weights::spec::ModelSpec`, bakes the arena on-missing, and loads it into the
//! engine's `BertWeights` -- all without touching XRT/the NPU. Also pins backward-compat: a legacy
//! `weights = "dir"` scenario still parses with NO declarative source (None spec).

use std::path::Path;

use npu_engine::bert::weights::BertWeights;
use npu_engine::config::ScenarioConfig;
use safetensors::tensor::{Dtype, TensorView};

/// Write a tiny but COMPLETE 12-layer BERT `model.safetensors` (the `bert` arch transform
/// hard-requires all 12 layers). All tensors are 2x2 / len-2 -- enough to exercise the wiring.
fn write_tiny_bert_source(dir: &Path) {
    std::fs::create_dir_all(dir).unwrap();
    // owned byte buffers so TensorViews can borrow them
    let mut bufs: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
    let push2x2 = |name: &str, bufs: &mut Vec<(String, Vec<usize>, Vec<u8>)>| {
        let data: Vec<u8> = [1f32, 2., 3., 4.].iter().flat_map(|x| x.to_le_bytes()).collect();
        bufs.push((name.to_string(), vec![2, 2], data));
    };
    let push2 = |name: &str, bufs: &mut Vec<(String, Vec<usize>, Vec<u8>)>| {
        let data: Vec<u8> = [5f32, 6.].iter().flat_map(|x| x.to_le_bytes()).collect();
        bufs.push((name.to_string(), vec![2], data));
    };
    // embeddings
    push2x2("embeddings.word_embeddings.weight", &mut bufs);
    push2x2("embeddings.position_embeddings.weight", &mut bufs);
    push2x2("embeddings.token_type_embeddings.weight", &mut bufs);
    push2("embeddings.LayerNorm.weight", &mut bufs);
    push2("embeddings.LayerNorm.bias", &mut bufs);
    // 12 layers
    for i in 0..12 {
        for s in [
            "attention.self.query", "attention.self.key", "attention.self.value",
            "attention.output.dense", "intermediate.dense", "output.dense",
        ] {
            push2x2(&format!("encoder.layer.{i}.{s}.weight"), &mut bufs);
            push2(&format!("encoder.layer.{i}.{s}.bias"), &mut bufs);
        }
        for s in ["attention.output.LayerNorm", "output.LayerNorm"] {
            push2(&format!("encoder.layer.{i}.{s}.weight"), &mut bufs);
            push2(&format!("encoder.layer.{i}.{s}.bias"), &mut bufs);
        }
    }
    let views: Vec<(String, TensorView)> = bufs
        .iter()
        .map(|(n, sh, by)| (n.clone(), TensorView::new(Dtype::F32, sh.clone(), by).unwrap()))
        .collect();
    let bytes = safetensors::serialize(views.iter().map(|(n, v)| (n.as_str(), v)), &None).unwrap();
    std::fs::write(dir.join("model.safetensors"), bytes).unwrap();
}

#[test]
fn declarative_source_parses_bakes_and_loads_bert_weights() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let src_dir = root.join("src");
    write_tiny_bert_source(&src_dir);
    let arena = root.join("arena.safetensors");

    let toml = format!(
        r#"
[scenario]
kind = "embeddings"
name = "tiny-bert"
[model]
hidden = 2
ff = 2
n_heads = 1
head_dim = 2
n_layers = 12
max_seq = 4
[artifacts]
source = "path:{src}"
arch = "bert"
arena = "{arena}"
"#,
        src = src_dir.display(),
        arena = arena.display(),
    );

    let cfg = ScenarioConfig::from_toml_str(&toml).expect("parse declarative scenario");
    // The declarative spec is recognized.
    let spec = cfg.artifacts.model_spec().expect("model_spec ok").expect("Some(spec) when source set");
    assert_eq!(spec.arch, "bert");

    // Uniform entry point: ensure (bake-on-missing) + load -- no NPU.
    let w = BertWeights::load_for(&cfg.artifacts, root, cfg.model.n_layers)
        .expect("declarative load_for must bake + load");
    assert!(arena.exists(), "arena was baked to the configured path");
    assert_eq!(w.n_layers(), 12);
    // word_emb came through (bf16-baked -> upcast on read; 1,2,3,4 are exactly representable).
    let we = w.word_emb();
    assert_eq!(we.shape(), &[2, 2]);
    assert_eq!(we.iter().copied().collect::<Vec<f32>>(), vec![1., 2., 3., 4.]);

    // Second load finds the arena FRESH (no re-bake) and returns the same shapes.
    let w2 = BertWeights::load_for(&cfg.artifacts, root, cfg.model.n_layers).expect("fresh reload");
    assert_eq!(w2.n_layers(), 12);
}

#[test]
fn legacy_weights_dir_scenario_has_no_declarative_source() {
    // Backward-compat: an old scenario with only `weights = "dir"` parses and yields NO ModelSpec,
    // so the engine takes the unchanged npy path.
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
"#;
    let cfg = ScenarioConfig::from_toml_str(toml).expect("legacy parse");
    assert_eq!(cfg.artifacts.weights, "artifacts/bge-base/encoder");
    assert!(cfg.artifacts.model_spec().unwrap().is_none(), "no source -> no spec (legacy npy path)");
}

#[test]
fn source_without_arch_is_an_error() {
    let toml = r#"
[scenario]
kind = "embeddings"
name = "x"
[model]
hidden = 2
ff = 2
n_heads = 1
head_dim = 2
n_layers = 1
max_seq = 4
[artifacts]
source = "hf:BAAI/bge-base-en-v1.5"
"#;
    let cfg = ScenarioConfig::from_toml_str(toml).expect("parse");
    assert!(cfg.artifacts.model_spec().is_err(), "source set but arch empty must error");
}
