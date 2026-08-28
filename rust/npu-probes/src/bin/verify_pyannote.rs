//! Stage-wise diarization parity against a pyannote reference dump.
//!
//! Order matters. Segmentation is compared BEFORE the final segments, because a clustering
//! mismatch and an embedding mismatch look identical in the final answer -- comparing only the end
//! result tells you that something is wrong, never which stage.
//!
//! Speaker labels are compared under the best PERMUTATION: cluster indices are arbitrary, so
//! "our 0 == their 1" is a correct answer, not a defect. Anything that compares labels literally
//! reports a false failure on every second run.
//!
//! Run from the repo root, after scripts/verify_pyannote_reference.py has produced the dump:
//!   rust/target/release/verify_pyannote artifacts/pyannote/fixtures/overlap_2spk.wav
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use ndarray::Array3;
use ndarray_npy::read_npy;
use npu_engine::capability::Segment;
use npu_engine::diarize::{onnx::OnnxSegmenter, types::Segmenter, DiarizePipeline, Manifest};

/// One reference span parsed out of the pipeline's summary.json.
#[derive(serde::Deserialize)]
struct RefSeg { start: f32, end: f32, speaker: String }
#[derive(serde::Deserialize)]
struct RefSummary { n_speakers: usize, overlap_s: f32, segments: Vec<RefSeg> }

fn read_wav_i16(p: &Path) -> Vec<i16> {
    let b = std::fs::read(p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
    // Minimal RIFF walk: find `data`, take the rest as i16 LE. The clips are all 16k mono 16-bit.
    let pos = b.windows(4).position(|w| w == b"data").expect("no data chunk");
    b[pos + 8..].chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect()
}

/// Total seconds where a and b disagree about who is speaking, minimised over label permutations.
/// A crude cousin of DER -- reported as a NOTE, never a gate.
fn disagreement_s(ours: &[Segment], theirs: &[RefSeg], step: f32, end_s: f32) -> (f32, f32) {
    let their_ids: Vec<&String> = {
        let mut v: Vec<&String> = theirs.iter().map(|s| &s.speaker).collect();
        v.sort(); v.dedup(); v
    };
    let our_ids: Vec<u32> = {
        let mut v: Vec<u32> = ours.iter().map(|s| s.speaker).collect();
        v.sort(); v.dedup(); v
    };
    // Brute-force permutations; speaker counts here are tiny (<= 4).
    let mut best = f32::INFINITY;
    let mut perms: Vec<Vec<usize>> = Vec::new();
    permute(&mut (0..their_ids.len()).collect::<Vec<_>>(), 0, &mut perms);
    for p in &perms {
        let mut bad = 0.0f32;
        let mut t = 0.0f32;
        while t < end_s {
            let o = ours.iter().find(|s| t >= s.start_s && t < s.end_s).map(|s| s.speaker);
            let r = theirs.iter().find(|s| t >= s.start && t < s.end).map(|s| &s.speaker);
            let mapped = o.and_then(|oi| {
                let idx = our_ids.iter().position(|&x| x == oi)?;
                p.get(idx).and_then(|&pi| their_ids.get(pi)).copied()
            });
            if mapped != r { bad += step; }
            t += step;
        }
        if bad < best { best = bad; }
    }
    (best, end_s)
}

fn permute(v: &mut Vec<usize>, k: usize, out: &mut Vec<Vec<usize>>) {
    if k == v.len() { out.push(v.clone()); return }
    for i in k..v.len() { v.swap(k, i); permute(v, k + 1, out); v.swap(k, i); }
}

fn main() {
    let wav = std::env::args().nth(1).expect("usage: verify_pyannote <clip.wav>");
    let wav = PathBuf::from(wav);
    let stem = wav.file_stem().unwrap().to_string_lossy().to_string();
    // Which model's artifacts to verify. Defaults to 3.1 for compatibility with existing dumps;
    // pass a second argument to check community-1 (or any other exported model).
    let model = std::env::args().nth(2)
        .unwrap_or_else(|| "speaker-diarization-3.1".to_string());
    let dir_owned = Path::new("artifacts/pyannote").join(&model);
    let dir = dir_owned.as_path();
    let refdir = Path::new("artifacts/pyannote/ref").join(&stem);
    assert!(refdir.is_dir(), "no reference dump at {} -- run scripts/verify_pyannote_reference.py first",
            refdir.display());

    let manifest: Manifest = serde_json::from_str(
        &std::fs::read_to_string(dir.join("diarize.json")).expect("diarize.json")).expect("manifest");
    let pcm = read_wav_i16(&wav);
    println!("clip {} ({:.2}s), model {model}, clustering {}",
             wav.display(), pcm.len() as f32 / manifest.sample_rate as f32,
             manifest.clustering.method);

    // ---- stage 1: segmentation logits -----------------------------------------------------
    let seg = OnnxSegmenter::build(&manifest, dir).expect("segmenter");
    let (ours, valid) = seg.segment(&pcm).expect("segment");
    let refr: Array3<f32> = read_npy(refdir.join("segmentation.npy")).expect("segmentation.npy");
    println!("segmentation ours={:?} ref={:?} valid_frames_last_window={valid}", ours.dim(), refr.dim());
    assert_eq!(ours.dim().2, refr.dim().2, "class count must match");
    if ours.dim().0 != refr.dim().0 {
        // Not silently min()'d away: we zero-pad and keep a final PARTIAL window (matching what
        // pyannote's Inference does with the clip tail), while the reference dumper only emits
        // windows that fit whole. The overlapping windows are still compared below; this line
        // exists so the difference is visible rather than absorbed.
        println!("  NOTE: window counts differ (ours keeps the padded tail window); \
                  comparing the {} both produced", ours.dim().0.min(refr.dim().0));
    }

    let nw = ours.dim().0.min(refr.dim().0);
    let nf = ours.dim().1.min(refr.dim().1);
    let (mut num, mut den) = (0.0f64, 0.0f64);
    let (mut agree, mut total) = (0usize, 0usize);
    for w in 0..nw {
        for f in 0..nf {
            let (mut ba, mut bb, mut va, mut vb) = (0usize, 0usize, f32::MIN, f32::MIN);
            for c in 0..ours.dim().2 {
                let (a, b) = (ours[[w, f, c]], refr[[w, f, c]]);
                num += ((a - b) as f64).powi(2);
                den += (b as f64).powi(2);
                if a > va { va = a; ba = c; }
                if b > vb { vb = b; bb = c; }
            }
            total += 1;
            if ba == bb { agree += 1; }
        }
    }
    let rel = (num / den.max(1e-30)).sqrt();
    let acc = agree as f64 / total.max(1) as f64;
    println!("  rel-L2 = {rel:.3e}  (note)");
    println!("  argmax agreement = {:.4}%  (GATE >= 99%)", acc * 100.0);

    // ---- stage 2: end-to-end segments, under the best label permutation --------------------
    let dp = DiarizePipeline::new(
        manifest.clone(),
        Box::new(OnnxSegmenter::build(&manifest, dir).expect("segmenter")),
        Box::new(npu_engine::diarize::onnx::OnnxEmbedder::build(&manifest, dir).expect("embedder")),
        dir,
    ).expect("build diarize pipeline");
    let segs = dp.run(&pcm).expect("diarize");
    let summary: RefSummary = serde_json::from_str(
        &std::fs::read_to_string(refdir.join("summary.json")).expect("summary.json")).expect("summary");

    let our_n = segs.iter().map(|s| s.speaker).collect::<std::collections::BTreeSet<_>>().len();
    println!("speakers: ours={our_n} ref={}", summary.n_speakers);
    let end = pcm.len() as f32 / manifest.sample_rate as f32;
    let (bad, span) = disagreement_s(&segs, &summary.segments, 0.01, end);
    println!("  frame disagreement = {:.3}s of {:.2}s = {:.2}%  (note, permutation-minimised)",
             bad, span, 100.0 * bad / span.max(1e-6));

    let mut by: BTreeMap<u32, f32> = BTreeMap::new();
    for s in &segs { *by.entry(s.speaker).or_insert(0.0) += s.end_s - s.start_s; }
    println!("  ours:  {:?}", segs.iter()
        .map(|s| format!("{:.2}-{:.2}#{}", s.start_s, s.end_s, s.speaker)).collect::<Vec<_>>());
    println!("  ref:   {:?}", summary.segments.iter()
        .map(|s| format!("{:.2}-{:.2}#{}", s.start, s.end, s.speaker)).collect::<Vec<_>>());
    if summary.overlap_s == 0.0 {
        println!("  NOTE: the reference itself found NO overlapping speech on this clip, so the \
                  masked-pooling path is exercised but NOT discriminated by it.");
    }

    assert!(acc > 0.99, "segmentation argmax agreement {acc:.4} below 0.99 -- the exported graph \
                         disagrees with the reference model");
    assert_eq!(our_n, summary.n_speakers, "speaker count must match the reference");
    println!("PASS");
}
