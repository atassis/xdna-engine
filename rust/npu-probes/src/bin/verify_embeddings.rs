//! Accuracy gate: our NPU embedding pipeline vs the bge ONNX oracle (CPU). Run from repo root with
//! the NPU idle. Usage: verify_embeddings [scenario.toml]
use std::path::Path;
use std::rc::Rc;

use ndarray::Array2;
use npu_engine::bert::EmbedPipeline;
use npu_engine::config::ScenarioConfig;
use npu_engine::bert::frontend::EmbedFrontend;
use npu_engine::bert::weights::BertWeights;
use npu_onnx::{Env, Session, Tensor};
use npu_xrt::Device;

const SENTENCES: &[&str] = &[
    "The quick brown fox jumps over the lazy dog.",
    "Local semantic search runs on the NPU.",
    "Embeddings power retrieval augmented generation.",
];
const TOL: f32 = 0.08;

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb).max(1e-12)
}

/// Mean-pool + L2 the ONNX last_hidden_state [seq,768] the same way EmbedHead does, for an apples-
/// to-apples oracle vector.
fn pool_l2(h: &[f32], seq: usize, d: usize) -> Vec<f32> {
    let mut v = vec![0f32; d];
    for t in 0..seq { for j in 0..d { v[j] += h[t * d + j]; } }
    let inv = 1.0 / seq as f32;
    for x in v.iter_mut() { *x *= inv; }
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    for x in v.iter_mut() { *x /= n; }
    v
}

fn main() {
    let scenario = std::env::args().nth(1).unwrap_or_else(|| "scenarios/bge-base.toml".into());
    let root = Path::new(".");
    let cfg = ScenarioConfig::load(Path::new(&scenario)).expect("scenario");

    // our pipeline
    let dev = Rc::new(Device::open(0).expect("open NPU (stop flm-asr/voxd first)"));
    let pipe = EmbedPipeline::build(&cfg, root, dev);

    // oracle: onnx session + a frontend reused only to get matching token ids
    let env = Env::new().expect("onnx env");
    let sess = Session::load(&env, root.join(&cfg.artifacts.onnx_ref).to_str().unwrap()).expect("onnx");
    let weights = Rc::new(BertWeights::load(&root.join(&cfg.artifacts.weights), cfg.model.n_layers).unwrap());
    let fe = EmbedFrontend::new(&root.join(&cfg.artifacts.tokenizer), weights, cfg.model.max_seq);

    let mut worst = 1.0f32;
    for s in SENTENCES {
        let ours = pipe.embed((*s).to_string());

        let ids: Vec<i64> = fe.token_ids(s).iter().map(|&i| i as i64).collect();
        let seq = ids.len();
        let mask = vec![1i64; seq];
        let types = vec![0i64; seq];
        let out = sess.run(
            &[
                ("input_ids", Tensor::I64(&ids, vec![1, seq as i64])),
                ("attention_mask", Tensor::I64(&mask, vec![1, seq as i64])),
                ("token_type_ids", Tensor::I64(&types, vec![1, seq as i64])),
            ],
            &["last_hidden_state"],
        ).expect("onnx run");
        let d = cfg.model.hidden;
        let oracle = pool_l2(out.f32(0), seq, d);

        let cos = cosine(&ours, &oracle);
        worst = worst.min(cos);
        println!("cos={cos:.5}  ({s})");
        let _ = Array2::<f32>::zeros((0, 0));
    }
    println!("worst cosine = {worst:.5}  (need >= {:.3})", 1.0 - TOL);
    assert!(worst >= 1.0 - TOL, "embedding accuracy gate FAILED: worst cosine {worst} < {}", 1.0 - TOL);
    println!("ACCURACY GATE PASS");
}
