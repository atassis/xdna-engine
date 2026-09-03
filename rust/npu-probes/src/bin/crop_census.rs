//! Why clustering collapses: count the crops each speaker actually gets.
//!
//! Every diarization failure measured so far has turned on the same quantity. Crops are per
//! (window, speaker), so a speaker's evidence is its SHARE of them, and both known collapses are
//! that share running out: community-1 returns one speaker for two on a 9.61 s clip, and a peer
//! session measured the same collapse on FOUR of eight 60 s windows of ordinary conversation --
//! where duration alone does not predict it (2.3 s of scattered minority speech survived, 10.7 s
//! contiguous did not).
//!
//! Their hypothesis, which this exists to test: what matters is the number of crops DOMINATED by
//! the minority speaker, and speech truncated by the window EDGE yields far fewer of those per
//! second than the same speech in the interior, because a crop overlapping the edge also carries
//! the majority speaker.
//!
//! So this reports, per clip: how many crops the pipeline built, how the minority share falls out
//! when the embeddings are split into exactly two groups, and what the real clusterer returned.
//! A collapse with a healthy minority share would refute the hypothesis; a collapse that always
//! coincides with a small one supports it.
//!
//!   rust/target/release/crop_census <clip.wav> [model] [more.wav ...]
use std::path::Path;

use ndarray::Array2;
use npu_engine::diarize::onnx::{OnnxEmbedder, OnnxSegmenter};
use npu_engine::diarize::types::{Clusterer, Manifest, Segmenter, SpeakerEmbedder};
use npu_engine::diarize::{cluster, vbx};
use npu_engine::diarize::{build_crops, powerset};

/// The manifest's clusterer, built directly so the probe can run it on embeddings it chose --
/// notably a REORDERED copy. Mirrors `clusterer_for`, which is private to the engine.
fn clusterer_of(m: &Manifest, dir: &Path) -> Box<dyn Clusterer> {
    match m.clustering.method.as_str() {
        "vbx" => Box::new(vbx::VbxClusterer {
            plda: vbx::Plda::load(m.clustering.plda.as_ref().expect("plda"), dir).expect("plda load"),
            threshold: m.clustering.threshold as f32,
            fa: m.clustering.fa as f32,
            fb: m.clustering.fb as f32,
            max_iters: m.clustering.max_iters.max(1),
            init_smoothing: m.clustering.init_smoothing as f32,
            lda_dim: m.clustering.lda_dim.max(1),
        }),
        _ => Box::new(cluster::AgglomerativeClusterer {
            threshold: m.clustering.threshold as f32,
            min_cluster_size: m.clustering.min_cluster_size,
        }),
    }
}

/// Rows of `e` in the given order.
fn reorder(e: &Array2<f32>, order: &[usize]) -> Array2<f32> {
    let mut out = Array2::<f32>::zeros((order.len(), e.ncols()));
    for (r, &i) in order.iter().enumerate() {
        for d in 0..e.ncols() { out[[r, d]] = e[[i, d]]; }
    }
    out
}

fn n_groups(v: &[u32]) -> usize { v.iter().collect::<std::collections::BTreeSet<_>>().len() }

fn read_wav_i16(p: &Path) -> Vec<i16> {
    let b = std::fs::read(p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
    let pos = b.windows(4).position(|w| w == b"data").expect("no data chunk");
    b[pos + 8..].chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect()
}

/// Split rows into exactly TWO groups and return the smaller group's size.
///
/// Deliberately not the shipped clusterer: the question is how much evidence the minority speaker
/// HAS, which has to be measured independently of the decision that is failing. Seeded from the two
/// most distant rows and assigned by cosine, so it is deterministic -- no random init to make the
/// same clip answer differently on a second run.
fn minority_of_two(e: &Array2<f32>) -> (Vec<usize>, f32) {
    let n = e.nrows();
    if n < 2 { return (Vec::new(), 0.0) }
    let row = |i: usize| (0..e.ncols()).map(|d| e[[i, d]]).collect::<Vec<f32>>();
    let cos = |a: &[f32], b: &[f32]| {
        let (mut d, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
        for i in 0..a.len().min(b.len()) { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
        if na <= 0.0 || nb <= 0.0 { 0.0 } else { d / (na.sqrt() * nb.sqrt()) }
    };
    let (mut bi, mut bj, mut worst) = (0usize, 1usize, f32::INFINITY);
    for i in 0..n {
        for j in (i + 1)..n {
            let c = cos(&row(i), &row(j));
            if c < worst { worst = c; bi = i; bj = j; }
        }
    }
    let (a, b) = (row(bi), row(bj));
    let mut in_a = Vec::new();
    let mut in_b = Vec::new();
    for i in 0..n {
        if cos(&row(i), &a) >= cos(&row(i), &b) { in_a.push(i) } else { in_b.push(i) }
    }
    let minority = if in_a.len() <= in_b.len() { in_a } else { in_b };
    // Separation between the two seeds, as a cosine DISTANCE: a clip whose two "speakers" are not
    // actually far apart tells us the split itself is meaningless, not that the minority is small.
    (minority, 1.0 - worst)
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    assert!(!args.is_empty(), "usage: crop_census <clip.wav> [model] [more.wav ...]");
    let model = if args.len() > 1 && !args[1].ends_with(".wav") { args.remove(1) }
                else { "speaker-diarization-community-1".to_string() };
    let dir = Path::new("artifacts/pyannote").join(&model);
    let m: Manifest = serde_json::from_str(
        &std::fs::read_to_string(dir.join("diarize.json")).expect("diarize.json")).expect("manifest");
    let seg = OnnxSegmenter::build(&m, &dir).expect("segmenter");
    let emb = OnnxEmbedder::build(&m, &dir).expect("embedder");
    let clu = npu_engine::diarize::DiarizePipeline::new(
        m.clone(), Box::new(OnnxSegmenter::build(&m, &dir).expect("s")),
        Box::new(OnnxEmbedder::build(&m, &dir).expect("e")), &dir).expect("pipeline");

    println!("model {model}  clustering {}", m.clustering.method);
    println!("{:>12} {:>6} {:>6} {:>8} {:>8} {:>6} {:>4}  {}",
             "clip", "wins", "crops", "minority", "min_frac", "sep", "spk", "k: fwd/rev/sorted");
    for path in &args {
        let p = Path::new(path);
        let pcm = read_wav_i16(p);
        let (logits, _) = seg.segment(&pcm).expect("segment");
        let activity = powerset::decode_powerset(&logits, m.segmentation.max_speakers_per_chunk);
        let (n_win, n_frames, _) = activity.dim();
        if n_win == 0 || n_frames == 0 { println!("{path}: no windows"); continue }
        let (crops, _keys) = build_crops(&m, &activity, pcm.len());
        if crops.is_empty() { println!("{path}: no crops"); continue }
        let e = emb.embed(&pcm, &crops).expect("embed");
        let (minority, sep) = minority_of_two(&e);
        let spk = clu.run(&pcm).expect("diarize").iter()
            .map(|s| s.speaker).collect::<std::collections::BTreeSet<_>>().len();
        // Does the ORDER of the crops change the answer? VBx is a Bayesian HMM over the crop
        // SEQUENCE, so a minority block sitting at the end of that sequence is not obviously the
        // same problem as one at the start -- and crops come out of build_crops in window order,
        // so where a speaker talks IS where their crops sit in the sequence. If reversing flips a
        // collapse, the mechanism is the sequence, not the amount of evidence.
        let cl = clusterer_of(&m, &dir);
        let fwd: Vec<usize> = (0..e.nrows()).collect();
        let rev: Vec<usize> = (0..e.nrows()).rev().collect();
        // A third order with no temporal meaning at all: minority rows first, then the rest.
        let mut sorted: Vec<usize> = minority.clone();
        sorted.extend((0..e.nrows()).filter(|i| !minority.contains(i)));
        let k = |o: &[usize]| cl.cluster(&reorder(&e, o)).map(|v| n_groups(&v)).unwrap_or(0);
        let (kf, kr, ks) = (k(&fwd), k(&rev), k(&sorted));
        let name = p.file_name().unwrap().to_string_lossy();
        // WHERE the minority crops sit is the second half of the question: a crop is a whole
        // window, so speech truncated by the clip edge yields crops that also carry the other
        // speaker, and the same seconds of speech buy fewer clean crops at an edge than inside.
        let mut pos: Vec<f64> = minority.iter().map(|&i| crops[i].start_s).collect();
        pos.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let span = match (pos.first(), pos.last()) {
            (Some(a), Some(b)) => format!("{a:.0}-{b:.0}s"),
            _ => "-".into(),
        };
        println!("{name:>12} {n_win:>6} {:>6} {:>8} {:>7.1}% {sep:>6.2} {spk:>4}  {kf}/{kr}/{ks}  {span:>10}",
                 crops.len(), minority.len(),
                 100.0 * minority.len() as f32 / crops.len() as f32);
        let _ = pos;
    }
}
