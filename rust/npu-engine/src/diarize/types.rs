//! Types and the two swappable stage traits. No logic lives here -- this file exists so the seam
//! between "what the pipeline needs" and "who computes it" is one small readable unit, and so the
//! later NPU embedder implements a trait rather than editing the pipeline.

use ndarray::{Array2, Array3};
use serde::Deserialize;

use crate::api::EngineError;

/// Everything the pipeline needs to know about the models, read from the export manifest rather
/// than typed here. Each block records the upstream `source` it came from, because the values live
/// in THREE different places: the pipeline config.yaml, the embedder's own config.yaml, and the
/// pyannote-audio library source (`embed_dim`, pooling, and the 0.1 step are in no config at all).
#[derive(Debug, Clone, Deserialize)]
pub struct Manifest {
    /// The pinned pyannote-audio revision that is the provenance for the library-sourced values.
    pub pyannote_audio_rev: String,
    pub sample_rate: u32,
    pub segmentation: SegCfg,
    pub embedding: EmbCfg,
    pub clustering: ClusterCfg,
    pub min_duration_off_s: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SegCfg {
    pub onnx: String,
    pub duration_s: f32,
    /// Window hop. Sourced from pyannote-audio's `SpeakerDiarization.__init__` default (0.1 x
    /// duration), NOT from any config.yaml.
    pub step_s: f32,
    pub max_speakers_per_chunk: usize,
    pub powerset_classes: usize,
    pub source: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct EmbCfg {
    pub onnx: String,
    pub dim: usize,
    pub num_mel_bins: usize,
    pub frame_length_ms: f32,
    pub frame_shift_ms: f32,
    /// Fbank frames the exported graph produces for ONE window. NOT computable as
    /// `duration / frame_shift`: kaldi frames with `snip_edges`, so the count is
    /// `floor((samples - frame_length) / frame_shift) + 1` -- 998, not 1000, for a 10 s window at
    /// 25 ms/10 ms. The export records what the graph actually emits, and the weights mask must be
    /// exactly this long or onnxruntime rejects the input.
    pub n_frames: usize,
    pub source: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ClusterCfg {
    pub method: String,
    pub threshold: f64,
    pub min_cluster_size: usize,
    /// `embedding_exclude_overlap` upstream. True in the shipped pipeline; the embedder pools
    /// WEIGHTED over a speaker's active non-overlapping frames, so this is a correctness switch.
    pub exclude_overlap: bool,
    pub source: String,
}

/// One speaker's crop of one window: the audio, where it sits in the clip, and the per-frame
/// activation weights for THAT speaker.
///
/// `weights` is not an optimisation. The shipped pipeline runs `embedding_exclude_overlap: true`
/// and WeSpeaker pools weighted statistics over only that speaker's active, non-overlapping frames
/// (`forward(waveforms, weights)` -> `resnet(fbank, weights=weights)`). Pooling unweighted computes
/// a different statistic that still looks right on single-speaker audio and diverges exactly where
/// diarization matters.
#[derive(Debug, Clone)]
pub struct Crop {
    pub pcm: Vec<i16>,
    pub start_s: f64,
    /// One weight per EMBEDDER fbank frame, 0.0..=1.0.
    pub weights: Vec<f32>,
}

/// Powerset segmentation over a sliding window.
pub trait Segmenter {
    /// Returns `([n_windows, n_frames, n_classes]` logits, valid frames in the LAST window`)`.
    ///
    /// The valid-frame count is part of the contract: the final window is zero-padded to the
    /// model's fixed input length, and a stitcher that consumes those padding frames drifts the
    /// timeline only at clip boundaries -- a defect that passes every mid-clip test.
    fn segment(&self, pcm: &[i16]) -> Result<(Array3<f32>, usize), EngineError>;
}

/// Group embeddings into speakers.
///
/// A trait, not a function, because the clustering STAGE is where diarization models actually
/// differ once segmentation and embedding are behind their own traits. pyannote 3.1 uses
/// centroid-linkage agglomerative clustering; community-1 uses Bayesian HMM (VBx) with a learned
/// PLDA. Same pipeline, same `[tile, D]` embeddings in, same labels out -- so the model is DATA
/// (a manifest naming its method) and not a second pipeline.
pub trait Clusterer {
    /// `[n_crops, dim]` L2-normalised embeddings -> one label per row, compacted to 0..k.
    fn cluster(&self, embeddings: &Array2<f32>) -> Result<Vec<u32>, EngineError>;
}

/// Fixed-dimension speaker embeddings, weighted-pooled per crop.
pub trait SpeakerEmbedder {
    /// Returns `[n_crops, dim]`, L2-normalised.
    fn embed(&self, crops: &[Crop]) -> Result<Array2<f32>, EngineError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_round_trips_and_records_where_each_value_came_from() {
        let json = r#"{
          "pyannote_audio_rev": "3.1.1",
          "sample_rate": 16000,
          "segmentation": {"onnx": "seg.onnx", "duration_s": 10.0, "step_s": 1.0,
                           "max_speakers_per_chunk": 3, "powerset_classes": 7,
                           "source": "pyannote/segmentation-3.0@main"},
          "embedding": {"onnx": "emb.onnx", "dim": 256, "num_mel_bins": 80,
                        "frame_length_ms": 25.0, "frame_shift_ms": 10.0, "n_frames": 998,
                        "source": "pyannote/wespeaker-voxceleb-resnet34-LM@main"},
          "clustering": {"method": "centroid", "threshold": 0.7045654963945799,
                         "min_cluster_size": 12, "exclude_overlap": true,
                         "source": "pyannote/hf-speaker-diarization-3.1@main"},
          "min_duration_off_s": 0.0
        }"#;
        let m: Manifest = serde_json::from_str(json).expect("manifest must parse");
        assert_eq!(m.clustering.method, "centroid");
        assert_eq!(m.clustering.min_cluster_size, 12);
        assert!(m.clustering.exclude_overlap, "the mask is the contract, not an option");
        assert_eq!(m.embedding.dim, 256);
        assert_eq!(m.embedding.n_frames, 998, "snip_edges framing, not duration/shift");
        assert_eq!(m.segmentation.powerset_classes, 7);
        // The library-sourced values must be present -- they are in NO upstream config.yaml.
        assert_eq!(m.pyannote_audio_rev, "3.1.1");
        assert_eq!(m.segmentation.step_s, 1.0);
    }

    #[test]
    fn a_crop_carries_its_own_activation_weights() {
        let c = Crop { pcm: vec![0i16; 16], start_s: 2.0, weights: vec![0.0, 1.0, 1.0, 0.0] };
        assert_eq!(c.weights.len(), 4);
        assert_eq!(c.start_s, 2.0);
    }
}
