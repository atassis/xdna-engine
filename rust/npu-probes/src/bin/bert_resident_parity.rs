//! Gate: BERT's K=768 resident GELU FFN rail (`BERT_RESIDENT_FFN=1`) against the host FFN, on ONE
//! pipeline so the only difference is which FFN path each block took. Run from repo root with the
//! NPU idle. Usage: bert_resident_parity [scenario.toml]
//!
//! Why A/B and not the ONNX oracle: `verify_embeddings` already scores the whole pipeline against
//! bge, and answers a question this step does not ask. What is unproven here is narrower -- that
//! the rail dispatches at all, and that swapping the FFN under a 12-layer encoder does not move the
//! embedding. So the flag is toggled between two `embed` calls on the same built pipeline: same
//! weights, same attention, same LNs, same device handles.
//!
//! Two things make the pass falsifiable. The rail's fallbacks are all SILENT -- an unbuilt width, a
//! wrong `kres`, a sequence over PAD_M each return None and run the host FFN -- so a perfect match
//! is exactly what a rail that never ran would print; the dispatch counters are asserted to have
//! moved by 4 per layer per sentence. And cosine near 1.0 means nothing until cosine can be far
//! from it, so every rail vector is also scored against the OTHER sentences' host vectors.
use std::path::Path;
use std::rc::Rc;

use npu_engine::bert::EmbedPipeline;
use npu_engine::config::ScenarioConfig;
use npu_xrt::Device;

const SENTENCES: &[&str] = &[
    "The quick brown fox jumps over the lazy dog.",
    "Local semantic search runs on the NPU.",
    "Embeddings power retrieval augmented generation.",
];

/// Cosine floor for rail-vs-host on the SAME sentence. Both paths are bf16 NPU matmuls differing
/// only in tile and in where the biases land, so this is a "did the FFN change meaning" bar, not a
/// numerics bar -- the per-block residual is scored by `fused_seam_parity k768ffn`.
const COS_MIN: f32 = 0.999;
/// Ceiling for the cross-sentence control. Unrelated sentences in a bge space are not orthogonal,
/// so this only has to sit far below COS_MIN to show cosine still discriminates at this precision.
const CONTROL_MAX: f32 = 0.99;

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb).max(1e-12)
}

fn rel_l2(a: &[f32], b: &[f32]) -> f32 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum();
    let den: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum();
    (num.sqrt() / den.sqrt().max(1e-12)) as f32
}

fn main() {
    let scenario = std::env::args().nth(1).unwrap_or_else(|| "scenarios/bge-base.toml".into());
    let root = Path::new(".");
    let cfg = ScenarioConfig::load(Path::new(&scenario)).expect("scenario");
    let n_layers = cfg.model.n_layers;

    // Set BEFORE build: the rail is constructed in BertEncoder::new, while the per-call check in
    // try_resident_ffn re-reads the flag, which is what lets one pipeline serve both arms.
    std::env::set_var("BERT_RESIDENT_FFN", "1");
    let dev = Rc::new(Device::open(0).expect("open NPU (stop xdna-engine/npu-vox first)"));
    let pipe = EmbedPipeline::build(&cfg, root, dev).expect("build embed pipeline");

    let stats = || pipe.encoder().resident_ffn_stats();
    let base = stats().unwrap_or_else(|| {
        panic!("[bert_resident_parity] no K=768 rail wired -- build it at the scenario's max_seq \
                width with scripts/build_k768_gelu_rail.sh, and check BERT_RESIDENT_FFN=1")
    });

    let mut rail: Vec<Vec<f32>> = Vec::new();
    let mut host: Vec<Vec<f32>> = Vec::new();
    let mut prev = base;
    for s in SENTENCES {
        std::env::set_var("BERT_RESIDENT_FFN", "1");
        rail.push(pipe.embed((*s).to_string()));
        let now = stats().unwrap();
        let (dc, dd) = (now.0 - prev.0, now.1 - prev.1);
        assert_eq!(dc, n_layers, "rail ran {dc} of {n_layers} blocks -- a block fell back to host");
        assert_eq!(dd, 4 * n_layers, "rail dispatched {dd}, expected 4 bricks x {n_layers} blocks");
        prev = now;

        // Same pipeline, flag off: every block takes the host FFN branch instead.
        std::env::set_var("BERT_RESIDENT_FFN", "0");
        host.push(pipe.embed((*s).to_string()));
        assert_eq!(stats().unwrap(), prev, "host arm still dispatched the rail");
    }

    let mut worst_cos = 1.0f32;
    let mut worst_ctl = 0.0f32;
    for (i, s) in SENTENCES.iter().enumerate() {
        let cos = cosine(&rail[i], &host[i]);
        let rl2 = rel_l2(&host[i], &rail[i]);
        worst_cos = worst_cos.min(cos);
        let ctl = (0..SENTENCES.len())
            .filter(|&j| j != i)
            .map(|j| cosine(&rail[i], &host[j]))
            .fold(0.0f32, f32::max);
        worst_ctl = worst_ctl.max(ctl);
        println!(
            "[bert_resident_parity] cos={cos:.6} rel-L2={rl2:.3e} control={ctl:.4}  \"{s}\""
        );
    }
    println!(
        "[bert_resident_parity] layers={n_layers} worst_cos={worst_cos:.6} worst_control={worst_ctl:.4}"
    );
    assert!(worst_cos >= COS_MIN, "rail vs host cosine {worst_cos:.6} < {COS_MIN}");
    assert!(worst_ctl <= CONTROL_MAX, "control {worst_ctl:.4} > {CONTROL_MAX} -- cosine does not \
             discriminate here, so the pass above is not evidence");
    println!("[bert_resident_parity] PASS (cos >= {COS_MIN}, control <= {CONTROL_MAX})");
}
