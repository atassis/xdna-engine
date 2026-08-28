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
use npu_engine::diarize::{build_crops, powerset};

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
             "clip", "wins", "crops", "minority", "min_frac", "sep", "spk", "minority crop start_s");
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
        println!("{name:>12} {n_win:>6} {:>6} {:>8} {:>7.1}% {sep:>6.2} {spk:>4}  {span:>10}  {}",
                 crops.len(), minority.len(),
                 100.0 * minority.len() as f32 / crops.len() as f32,
                 pos.iter().map(|x| format!("{x:.0}")).collect::<Vec<_>>().join(","));
    }
}
