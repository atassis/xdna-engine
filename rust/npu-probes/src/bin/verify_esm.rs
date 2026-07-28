//! ESM accuracy gate: our NPU embedding vs the exported golden (mean-pooled + L2). Idle NPU.
//! Usage: verify_esm [scenario.toml]
//! The golden.json (written by scripts/export_esm.py) carries {seq, ids, mean_emb}. We also assert our
//! Rust residue tokenizer reproduces HF's ids — validating the hand-port end-to-end.
use std::path::Path;
use std::rc::Rc;

use npu_engine::config::ScenarioConfig;
use npu_engine::esm::frontend::tokenize;
use npu_engine::esm::EsmEmbedPipeline;
use npu_xrt::Device;

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let d: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    d / (na * nb).max(1e-12)
}

fn main() {
    let scenario = std::env::args().nth(1).unwrap_or_else(|| "scenarios/esm2-8m.toml".into());
    let root = Path::new(".");
    let cfg = ScenarioConfig::load(Path::new(&scenario)).expect("scenario");

    let golden: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(root.join(&cfg.artifacts.tokenizer)).expect("open golden.json"))
            .expect("parse golden.json");
    let seq = golden["seq"].as_str().unwrap().to_string();
    let oracle: Vec<f32> = golden["mean_emb"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();
    let hf_ids: Vec<u32> = golden["ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();

    // Tokenizer cross-check (hand-port vs HF).
    let ours_ids = tokenize(&seq);
    assert_eq!(ours_ids, hf_ids, "residue tokenizer mismatch vs HF golden ids");
    println!("tokenizer OK ({} ids match HF)", ours_ids.len());

    let dev = Rc::new(Device::open(0).expect("open NPU (stop other NPU services first)"));
    let pipe = EsmEmbedPipeline::build(&cfg, root, dev);
    let ours = pipe.embed(seq);

    let cos = cosine(&ours, &oracle);
    println!("cos={cos:.5}  (need >= 0.920)");
    assert!(cos >= 0.92, "ESM accuracy gate FAILED: cos {cos} < 0.92");
    println!("ACCURACY GATE PASS");
}
