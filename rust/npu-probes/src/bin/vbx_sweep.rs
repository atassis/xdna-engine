//! Can VBx be parameterised out of its speaker collapse, or is the collapse in its INIT?
//!
//! VBx refines an existing partition: it agglomerates first (Euclidean centroid linkage cut at
//! `threshold`), one-hots that into `gamma`, and iterates. So `k` is decided BEFORE the VB loop --
//! and if the init returns k = 1, `gamma` is `[n, 1]` and no value of `fa`, `fb`, `init_smoothing`
//! or `max_iters` can ever produce a second speaker. That makes "which parameter" the wrong first
//! question; the first question is which STAGE decided the count.
//!
//! So this reports, per clip: the init's k, the shipped parameters' k, and then a grid. A clip
//! whose init is already 1 is an init bug wearing a VB costume.
//!
//! Scoring uses a regression set, not the failing clips alone: a threshold that splits the two
//! tail clips and merges nothing else is a fix, and one that also splits single-speaker audio is a
//! looser threshold pretending to be a fix.
//!
//!   rust/target/release/vbx_sweep <clip:expected> [clip:expected ...]
use std::path::Path;

use ndarray::Array2;
use npu_engine::diarize::onnx::{OnnxEmbedder, OnnxSegmenter};
use npu_engine::diarize::types::{Clusterer, Manifest, Segmenter, SpeakerEmbedder};
use npu_engine::diarize::{build_crops, cluster, powerset, vbx};

fn read_wav_i16(p: &Path) -> Vec<i16> {
    let b = std::fs::read(p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
    let pos = b.windows(4).position(|w| w == b"data").expect("no data chunk");
    b[pos + 8..].chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect()
}

fn n_groups(v: &[u32]) -> usize { v.iter().collect::<std::collections::BTreeSet<_>>().len() }

struct Case { name: String, expect: usize, emb: Array2<f32> }

fn main() {
    let model = "speaker-diarization-community-1";
    let dir = Path::new("artifacts/pyannote").join(model);
    let m: Manifest = serde_json::from_str(
        &std::fs::read_to_string(dir.join("diarize.json")).expect("diarize.json")).expect("manifest");
    let seg = OnnxSegmenter::build(&m, &dir).expect("segmenter");
    let emb = OnnxEmbedder::build(&m, &dir).expect("embedder");
    let plda = vbx::Plda::load(m.clustering.plda.as_ref().expect("plda"), &dir).expect("plda");

    // Embed every clip ONCE. The grid below re-clusters these same matrices thousands of times, so
    // the embedder must not be inside the loop -- it is 99.9% of the cost and none of the question.
    let mut cases = Vec::new();
    for arg in std::env::args().skip(1) {
        let (path, expect) = arg.rsplit_once(':').expect("usage: <clip.wav>:<expected speakers>");
        let expect: usize = expect.parse().expect("expected speakers must be a number");
        let pcm = read_wav_i16(Path::new(path));
        let (logits, _) = seg.segment(&pcm).expect("segment");
        let act = powerset::decode_powerset(&logits, m.segmentation.max_speakers_per_chunk);
        let (crops, _) = build_crops(&m, &act, pcm.len());
        if crops.is_empty() { continue }
        let e = emb.embed(&pcm, &crops).expect("embed");
        let name = Path::new(path).file_stem().unwrap().to_string_lossy().to_string();
        cases.push(Case { name, expect, emb: e });
    }
    eprintln!("embedded {} clips", cases.len());

    // ---- stage 1: is the count decided by the init? ------------------------------------------
    let shipped = vbx::VbxClusterer {
        plda: plda.clone(), threshold: m.clustering.threshold as f32, fa: m.clustering.fa as f32,
        fb: m.clustering.fb as f32, max_iters: m.clustering.max_iters.max(1),
        init_smoothing: m.clustering.init_smoothing as f32, lda_dim: m.clustering.lda_dim.max(1),
    };
    println!("{:>18} {:>6} {:>8} {:>8} {:>8}", "clip", "expect", "init_k", "vbx_k", "verdict");
    for c in &cases {
        let init = cluster::cluster_with(&c.emb, m.clustering.threshold as f32, 1,
                                         cluster::Metric::Euclidean);
        let ik = cluster::n_clusters(&init).max(1);
        let vk = shipped.cluster(&c.emb).map(|v| n_groups(&v)).unwrap_or(0);
        let verdict = if vk == c.expect { "ok" }
                      else if ik <= 1 { "INIT already collapsed" }
                      else { "VB collapsed a split init" };
        println!("{:>18} {:>6} {ik:>8} {vk:>8} {verdict:>8}", c.name, c.expect);
    }

    // ---- stage 2: grid ------------------------------------------------------------------------
    // threshold first, because stage 1 decides whether the rest can matter at all.
    let thresholds = [0.40f32, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90, 1.00];
    let fas = [0.02f32, 0.07, 0.15, 0.30];
    let fbs = [0.40f32, 0.80, 1.20];
    let smooths = [3.0f32, 7.0, 12.0];
    println!("\ngrid: {} settings x {} clips", thresholds.len()*fas.len()*fbs.len()*smooths.len(),
             cases.len());
    let mut best: Vec<(usize, f32, f32, f32, f32, Vec<usize>)> = Vec::new();
    for &t in &thresholds { for &fa in &fas { for &fb in &fbs { for &sm in &smooths {
        let c = vbx::VbxClusterer { plda: plda.clone(), threshold: t, fa, fb,
            max_iters: m.clustering.max_iters.max(1), init_smoothing: sm,
            lda_dim: m.clustering.lda_dim.max(1) };
        let got: Vec<usize> = cases.iter()
            .map(|x| c.cluster(&x.emb).map(|v| n_groups(&v)).unwrap_or(0)).collect();
        let score = got.iter().zip(&cases).filter(|(g, x)| **g == x.expect).count();
        best.push((score, t, fa, fb, sm, got));
    }}}}
    best.sort_by(|a, b| b.0.cmp(&a.0));
    // Parity check, which the score cannot express. `overlap_2spk` is the ONLY clip here with a
    // pyannote reference dump, so it is the only one where "we are wrong" is provable rather than
    // inferred from a peer's full-file answer. If no grid point reaches it, retuning is not finding
    // the defect -- it is trading which subset of clips is wrong.
    if let Some(i) = cases.iter().position(|c| c.name == "overlap_2spk") {
        let hits: Vec<_> = best.iter().filter(|(_, _, _, _, _, g)| g[i] == cases[i].expect).collect();
        println!("\nparity clip overlap_2spk (pyannote says {}): {} of {} grid settings reach it",
                 cases[i].expect, hits.len(), best.len());
        for (score, t, fa, fb, sm, g) in hits.iter().take(5) {
            println!("   score {score:>2}  thr {t:.2} fa {fa:.2} fb {fb:.2} smooth {sm:.1}  {g:?}");
        }
    }
    let shipped_score = best.iter()
        .find(|(_, t, fa, fb, sm, _)| (*t - m.clustering.threshold as f32).abs() < 1e-6
              && (*fa - m.clustering.fa as f32).abs() < 1e-6
              && (*fb - m.clustering.fb as f32).abs() < 1e-6
              && (*sm - m.clustering.init_smoothing as f32).abs() < 1e-6)
        .map(|x| x.0);
    println!("shipped (thr {:.2} fa {:.2} fb {:.2} smooth {:.1}) scores {:?} of {}",
             m.clustering.threshold, m.clustering.fa, m.clustering.fb, m.clustering.init_smoothing,
             shipped_score, cases.len());
    println!("\n{:>6} {:>6} {:>5} {:>5} {:>7}  per-clip counts ({})",
             "score", "thr", "fa", "fb", "smooth",
             cases.iter().map(|c| format!("{}={}", c.name, c.expect)).collect::<Vec<_>>().join(" "));
    for (score, t, fa, fb, sm, got) in best.iter().take(12) {
        println!("{score:>6} {t:>6.2} {fa:>5.2} {fb:>5.2} {sm:>7.1}  {got:?}");
    }
}
