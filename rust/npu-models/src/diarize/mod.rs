//! Speaker diarization: who spoke when. Host-only in v1; the NPU embedder swaps in behind
//! `SpeakerEmbedder` without touching this file.

pub mod cluster;
pub mod onnx;
pub mod powerset;
pub mod stitch;
pub mod timeline;
pub mod types;
pub mod vbx;

pub use types::{Clusterer, Crop, Manifest, Segmenter, SpeakerEmbedder};

use std::collections::HashMap;

use crate::api::EngineError;
use crate::capability::Segment;
use crate::pipeline::Diarizer;

/// The assembled pipeline. Holds the manifest and the two swappable stages; every other stage is a
/// pure function in a sibling module, which is what lets the whole thing be gated with mocks.
pub struct DiarizePipeline {
    manifest: Manifest,
    segmenter: Box<dyn Segmenter>,
    embedder: Box<dyn SpeakerEmbedder>,
    clusterer: Box<dyn Clusterer>,
}

impl DiarizePipeline {
    pub fn new(manifest: Manifest, segmenter: Box<dyn Segmenter>, embedder: Box<dyn SpeakerEmbedder>,
               dir: &std::path::Path) -> Result<DiarizePipeline, EngineError> {
        let clusterer = clusterer_for(&manifest, dir)?;
        Ok(DiarizePipeline { manifest, segmenter, embedder, clusterer })
    }

    /// Explicit clusterer, for tests and for a caller assembling a model the manifest cannot name.
    pub fn with_clusterer(manifest: Manifest, segmenter: Box<dyn Segmenter>,
                          embedder: Box<dyn SpeakerEmbedder>, clusterer: Box<dyn Clusterer>)
        -> DiarizePipeline {
        DiarizePipeline { manifest, segmenter, embedder, clusterer }
    }

    pub fn run(&self, pcm: &[i16]) -> Result<Vec<Segment>, EngineError> {
        let m = &self.manifest;
        // Per-stage wall clock behind NPU_DIARIZE_TIME. Which stage dominates decides whether
        // moving the embedder to the NPU is worth anything at all, and that is not guessable:
        // segmentation is a sequential BiLSTM, the embedder is dense conv, and they are within an
        // order of magnitude of each other in FLOPs.
        let timing = std::env::var_os("NPU_DIARIZE_TIME").is_some();
        let t_all = std::time::Instant::now();
        let t = std::time::Instant::now();
        let (logits, _valid_last) = self.segmenter.segment(pcm)?;
        let t_seg = t.elapsed();
        let n_spk = m.segmentation.max_speakers_per_chunk;
        let activity = powerset::decode_powerset(&logits, n_spk);
        let (n_windows, n_frames, _) = activity.dim();
        if n_windows == 0 || n_frames == 0 { return Ok(Vec::new()) }

        // Seconds per segmentation frame, from the window duration and the model's own frame count.
        let frame_s = m.segmentation.duration_s / n_frames as f32;
        let hop_frames = ((m.segmentation.step_s / frame_s).round() as usize).max(1);
        let (crops, keys) = build_crops(m, &activity, pcm.len());
        if crops.is_empty() { return Ok(Vec::new()) }

        let t = std::time::Instant::now();
        let embeddings = self.embedder.embed(pcm, &crops)?;
        let t_emb = t.elapsed();
        let t = std::time::Instant::now();
        let labels_vec = self.clusterer.cluster(&embeddings)?;
        let t_clu = t.elapsed();
        let n_clusters = cluster::n_clusters(&labels_vec).max(1);
        let labels: HashMap<(usize, usize), u32> =
            keys.iter().copied().zip(labels_vec.iter().copied()).collect();

        let n_global = (n_windows - 1) * hop_frames + n_frames;
        let global = stitch::aggregate_windows(&activity, &labels, hop_frames, n_global, n_clusters);
        let cov = stitch::coverage(n_windows, n_frames, hop_frames, n_global);
        let out = timeline::to_segments(&global, &cov, 0.5, frame_s, m.min_duration_off_s);
        if timing {
            let total = t_all.elapsed();
            let audio_s = pcm.len() as f64 / m.sample_rate as f64;
            let pct = |d: std::time::Duration| 100.0 * d.as_secs_f64() / total.as_secs_f64();
            eprintln!("[diarize-time] audio {audio_s:.2}s | total {:.3}s ({:.1}x realtime)",
                      total.as_secs_f64(), audio_s / total.as_secs_f64());
            eprintln!("[diarize-time]   segment  {:.3}s  {:5.1}%  ({} windows)",
                      t_seg.as_secs_f64(), pct(t_seg), n_windows);
            eprintln!("[diarize-time]   embed    {:.3}s  {:5.1}%  ({} crops)",
                      t_emb.as_secs_f64(), pct(t_emb), crops.len());
            eprintln!("[diarize-time]   cluster  {:.3}s  {:5.1}%", t_clu.as_secs_f64(), pct(t_clu));
        }
        Ok(out)
    }
}

/// The `(window, speaker)` crops a segmentation implies, and the keys that name them.
///
/// One crop per (window, speaker) that has EXCLUSIVE speech. A speaker who is only ever overlapped
/// has no clean statistic to pool, so it is dropped rather than embedded badly.
///
/// Public and separate from `run` because the CROP COUNT is the quantity every clustering failure
/// so far has turned on -- crops are per (window, speaker), so a speaker's evidence is its share of
/// them, and both the sub-10 s collapse and the step_s collapse are that share running out. A probe
/// that wants to count them must count the ones the pipeline actually builds, not a reimplementation
/// that can drift from this one.
pub fn build_crops(m: &Manifest, activity: &ndarray::Array3<bool>, pcm_len: usize)
    -> (Vec<Crop>, Vec<(usize, usize)>) {
    let n_spk = m.segmentation.max_speakers_per_chunk;
    let (n_windows, _, _) = activity.dim();
    // Embedder frames per window, taken from the manifest rather than computed. The obvious
    // `duration / frame_shift` is WRONG: kaldi frames with snip_edges, giving
    // floor((samples - frame_length)/frame_shift) + 1 = 998, not 1000, for 10 s at 25/10 ms.
    // onnxruntime rejects a weights mask of the wrong length outright, so this must match the
    // graph exactly.
    let n_emb_frames = m.embedding.n_frames.max(1);
    let win_samples = (m.segmentation.duration_s * m.sample_rate as f32).round() as usize;
    let hop_samples = (m.segmentation.step_s * m.sample_rate as f32).round() as usize;
    let mut crops = Vec::new();
    let mut keys = Vec::new();
    for w in 0..n_windows {
        for s in 0..n_spk {
            let weights = stitch::exclusive_weights(activity, w, s, n_emb_frames);
            if !stitch::has_speech(&weights) { continue }
            let start = w * hop_samples;
            if start >= pcm_len { continue }
            crops.push(Crop {
                offset: start,
                len: win_samples,
                start_s: start as f64 / m.sample_rate as f64,
                weights,
            });
            keys.push((w, s));
        }
    }
    (crops, keys)
}

/// Pick the clustering stage the manifest names. Unknown methods fail LOUDLY at load rather than
/// silently falling back to centroid: a manifest asking for vbx and getting agglomerative would
/// produce plausible-looking labels that are simply a different model's answer.
fn clusterer_for(m: &Manifest, dir: &std::path::Path) -> Result<Box<dyn Clusterer>, EngineError> {
    match m.clustering.method.as_str() {
        "centroid" => Ok(Box::new(cluster::AgglomerativeClusterer {
            threshold: m.clustering.threshold as f32,
            min_cluster_size: m.clustering.min_cluster_size,
        })),
        "vbx" => {
            let p = m.clustering.plda.as_ref().ok_or_else(|| EngineError::Load(
                "clustering.method = vbx but the manifest carries no [plda] artifacts".into()))?;
            Ok(Box::new(vbx::VbxClusterer {
                plda: vbx::Plda::load(p, dir)?,
                threshold: m.clustering.threshold as f32,
                fa: m.clustering.fa as f32,
                fb: m.clustering.fb as f32,
                max_iters: m.clustering.max_iters.max(1),
                init_smoothing: m.clustering.init_smoothing as f32,
                lda_dim: m.clustering.lda_dim.max(1),
            }))
        }
        other => Err(EngineError::Load(format!(
            "unsupported clustering method {other:?} (this build has: centroid, vbx)"))),
    }
}

impl Diarizer for DiarizePipeline {
    fn diarize(&self, pcm: &[i16]) -> Result<Vec<Segment>, EngineError> { self.run(pcm) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diarize::types::{Crop, Segmenter, SpeakerEmbedder};
    use ndarray::{Array2, Array3};

    /// Two windows, both with local speaker 0 talking throughout.
    struct MockSeg;
    impl Segmenter for MockSeg {
        fn segment(&self, _pcm: &[i16]) -> Result<(Array3<f32>, usize), EngineError> {
            let mut a = Array3::zeros((2, 4, 7));
            for w in 0..2 { for f in 0..4 { a[[w, f, 1]] = 1.0; } }   // class 1 = {spk0}
            Ok((a, 4))
        }
    }
    /// Window 0's voice points +x, window 1's points +y -> two clusters.
    struct MockEmb;
    impl SpeakerEmbedder for MockEmb {
        fn embed(&self, _pcm: &[i16], crops: &[Crop]) -> Result<Array2<f32>, EngineError> {
            let mut e = Array2::zeros((crops.len(), 2));
            for (i, c) in crops.iter().enumerate() {
                if c.start_s < 0.5 { e[[i, 0]] = 1.0; } else { e[[i, 1]] = 1.0; }
            }
            Ok(e)
        }
    }

    fn manifest() -> Manifest {
        serde_json::from_str(r#"{
          "pyannote_audio_rev":"3.1.1","sample_rate":16000,
          "segmentation":{"onnx":"s","duration_s":1.0,"step_s":1.0,
                          "max_speakers_per_chunk":3,"powerset_classes":7,"source":"t"},
          "embedding":{"onnx":"e","dim":2,"num_mel_bins":80,"n_frames":98,
                       "frame_length_ms":25.0,"frame_shift_ms":10.0,"source":"t"},
          "clustering":{"method":"centroid","threshold":0.5,"min_cluster_size":1,
                        "exclude_overlap":true,"source":"t"},
          "min_duration_off_s":0.0}"#).unwrap()
    }

    #[test]
    fn two_windows_of_different_voices_become_two_speakers() {
        let p = DiarizePipeline::new(manifest(), Box::new(MockSeg), Box::new(MockEmb), std::path::Path::new(".")).unwrap();
        let segs = p.run(&vec![0i16; 32_000]).expect("diarize");
        assert!(!segs.is_empty(), "a fully-voiced clip must yield segments");
        let speakers: std::collections::BTreeSet<u32> = segs.iter().map(|s| s.speaker).collect();
        assert_eq!(speakers.len(), 2, "different voices must not collapse: {segs:?}");
        assert!(segs.iter().all(|s| s.end_s > s.start_s), "no empty spans: {segs:?}");
    }

    #[test]
    fn an_unknown_clustering_method_fails_at_load_not_silently() {
        let mut m = manifest();
        m.clustering.method = "spectral".into();
        let Err(e) = DiarizePipeline::new(m, Box::new(MockSeg), Box::new(MockEmb), std::path::Path::new(".")) else {
            panic!("a manifest naming an unimplemented clusterer must not build a pipeline");
        };
        assert!(e.to_string().contains("spectral"), "must name the method it cannot serve: {e}");
        assert!(e.to_string().contains("centroid"), "and say what it can: {e}");
        assert!(e.to_string().contains("vbx"), "vbx is implemented, so it must be listed: {e}");
    }

    #[test]
    fn crops_reference_the_source_instead_of_copying_it() {
        // The regression that took the service down: crops used to own a Vec<i16> each, so a
        // 4-minute clip held 462 x 320 KB of audio copies before a single embedding ran. A crop
        // must name a slice, so memory follows the batch and not the clip length.
        use std::sync::{Arc, Mutex};
        struct Spy(Arc<Mutex<Vec<(usize, usize)>>>);
        impl SpeakerEmbedder for Spy {
            fn embed(&self, pcm: &[i16], crops: &[Crop]) -> Result<Array2<f32>, EngineError> {
                for c in crops {
                    assert!(c.offset < pcm.len(), "crop offset must index the source clip");
                    self.0.lock().unwrap().push((c.offset, c.len));
                }
                Ok(Array2::zeros((crops.len(), 2)))
            }
        }
        let seen = Arc::new(Mutex::new(Vec::new()));
        let p = DiarizePipeline::new(manifest(), Box::new(MockSeg), Box::new(Spy(seen.clone())),
                                     std::path::Path::new(".")).unwrap();
        let _ = p.run(&vec![0i16; 32_000]);
        let got = seen.lock().unwrap().clone();
        assert!(!got.is_empty(), "the embedder must be called");
        assert!(got.iter().all(|&(_, len)| len > 0), "every crop names a real span: {got:?}");
    }

    #[test]
    fn a_silent_clip_yields_no_segments_and_no_panic() {
        struct Silent;
        impl Segmenter for Silent {
            fn segment(&self, _p: &[i16]) -> Result<(Array3<f32>, usize), EngineError> {
                let mut a = Array3::zeros((2, 4, 7));
                for w in 0..2 { for f in 0..4 { a[[w, f, 0]] = 1.0; } }   // class 0 = non-speech
                Ok((a, 4))
            }
        }
        let p = DiarizePipeline::new(manifest(), Box::new(Silent), Box::new(MockEmb), std::path::Path::new(".")).unwrap();
        assert!(p.run(&vec![0i16; 32_000]).unwrap().is_empty());
    }

    #[test]
    fn the_embedder_receives_weights_not_bare_audio() {
        use std::sync::{Arc, Mutex};
        struct Spy(Arc<Mutex<Vec<usize>>>);
        impl SpeakerEmbedder for Spy {
            fn embed(&self, _pcm: &[i16], crops: &[Crop]) -> Result<Array2<f32>, EngineError> {
                self.0.lock().unwrap().extend(crops.iter().map(|c| c.weights.len()));
                Ok(Array2::zeros((crops.len(), 2)))
            }
        }
        let seen = Arc::new(Mutex::new(Vec::new()));
        let p = DiarizePipeline::new(manifest(), Box::new(MockSeg), Box::new(Spy(seen.clone())), std::path::Path::new(".")).unwrap();
        let _ = p.run(&vec![0i16; 32_000]);
        let lens = seen.lock().unwrap().clone();
        assert!(!lens.is_empty(), "the embedder must be called");
        assert!(lens.iter().all(|&n| n > 0),
            "every crop must carry a non-empty weights mask -- an unweighted pool is the wrong statistic");
    }
}
