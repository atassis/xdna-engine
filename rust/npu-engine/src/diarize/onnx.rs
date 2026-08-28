//! ONNX-backed stages. The ONLY artifact-dependent file in the module -- everything else is pure,
//! which is what lets the pipeline be gated without a model. The NPU embedder replaces
//! `OnnxEmbedder` behind `SpeakerEmbedder` and touches nothing here.

use std::path::Path;
use std::rc::Rc;

use ndarray::{Array2, Array3};
use npu_onnx::{Env, Session, Tensor};

use crate::api::EngineError;
use crate::diarize::types::{Crop, Manifest, Segmenter, SpeakerEmbedder};

fn load(dir: &Path, name: &str) -> Result<(Rc<Env>, Session), EngineError> {
    let p = dir.join(name);
    let env = Env::new().map_err(|e| EngineError::Load(format!("onnx env: {e}")))?;
    let s = Session::load(&env, p.to_str().unwrap_or(name))
        .map_err(|e| EngineError::Load(format!("load {}: {e}", p.display())))?;
    Ok((env, s))
}

pub struct OnnxSegmenter {
    _env: Rc<Env>,
    sess: Session,
    win_samples: usize,
    hop_samples: usize,
    n_classes: usize,
}

impl OnnxSegmenter {
    pub fn build(m: &Manifest, dir: &Path) -> Result<OnnxSegmenter, EngineError> {
        let (env, sess) = load(dir, &m.segmentation.onnx)?;
        Ok(OnnxSegmenter {
            _env: env,
            sess,
            win_samples: (m.segmentation.duration_s * m.sample_rate as f32).round() as usize,
            hop_samples: (m.segmentation.step_s * m.sample_rate as f32).round() as usize,
            n_classes: m.segmentation.powerset_classes,
        })
    }
}

impl Segmenter for OnnxSegmenter {
    fn segment(&self, pcm: &[i16]) -> Result<(Array3<f32>, usize), EngineError> {
        let win = self.win_samples.max(1);
        let hop = self.hop_samples.max(1);
        // At least one window even for a clip shorter than the model's input: it is zero-padded,
        // and `valid` reports how much of the last window was real.
        let n_windows = if pcm.len() <= win { 1 } else { (pcm.len() - win).div_ceil(hop) + 1 };
        let mut all: Vec<f32> = Vec::new();
        let mut n_frames = 0usize;
        let mut valid = 0usize;
        for w in 0..n_windows {
            let start = w * hop;
            let end = (start + win).min(pcm.len());
            let mut buf: Vec<f32> = pcm.get(start..end).unwrap_or(&[])
                .iter().map(|&s| s as f32 / 32768.0).collect();
            let real = buf.len();
            buf.resize(win, 0.0);   // the padding contract: pad, and report what was real
            let t = Tensor::F32(&buf, vec![1, 1, win as i64]);
            let out = self.sess.run(&[("waveform", t)], &["logits"])
                .map_err(|e| EngineError::Device(format!("segmentation onnx: {e}")))?;
            let d = out.f32(0);
            n_frames = d.len() / self.n_classes.max(1);
            if w + 1 == n_windows {
                valid = ((real as f64 / win as f64) * n_frames as f64).round() as usize;
            }
            all.extend_from_slice(d);
        }
        let a = Array3::from_shape_vec((n_windows, n_frames, self.n_classes), all)
            .map_err(|e| EngineError::Device(format!("segmentation shape: {e}")))?;
        Ok((a, valid.min(n_frames)))
    }
}

pub struct OnnxEmbedder { _env: Rc<Env>, sess: Session, dim: usize, batch: usize }

impl OnnxEmbedder {
    pub fn build(m: &Manifest, dir: &Path) -> Result<OnnxEmbedder, EngineError> {
        let (env, sess) = load(dir, &m.embedding.onnx)?;
        Ok(OnnxEmbedder { _env: env, sess, dim: m.embedding.dim, batch: m.embedding.batch_size.max(1) })
    }
}

impl SpeakerEmbedder for OnnxEmbedder {
    /// Embed in BATCHES, not one crop per call.
    ///
    /// Measured: crop-at-a-time was 95.6% of total diarization runtime (24.3 s of 25.5 s on a 46 s
    /// clip, ~329 ms per crop), because each call re-pays the session's fixed per-run cost on a
    /// single 10 s window. The graph is exported with a dynamic batch axis and upstream's own
    /// config uses `embedding_batch_size: 32`, so this is the shape the model expects.
    ///
    /// Crops within a batch must be the same length; they are, since every crop is one padded
    /// window. The assert states that rather than trusting it, because a ragged batch would
    /// silently reshape into nonsense instead of failing.
    fn embed(&self, crops: &[Crop]) -> Result<Array2<f32>, EngineError> {
        let mut out = Array2::<f32>::zeros((crops.len(), self.dim));
        if crops.is_empty() { return Ok(out) }
        let n_samp = crops[0].pcm.len();
        let n_w = crops[0].weights.len();

        for (b, chunk) in crops.chunks(self.batch).enumerate() {
            let ok = chunk.iter().all(|c| c.pcm.len() == n_samp && c.weights.len() == n_w);
            if !ok {
                return Err(EngineError::Device(
                    "embedder batch has ragged crops; every crop must be one padded window".into()));
            }
            let mut wav: Vec<f32> = Vec::with_capacity(chunk.len() * n_samp);
            let mut msk: Vec<f32> = Vec::with_capacity(chunk.len() * n_w);
            for c in chunk {
                wav.extend(c.pcm.iter().map(|&s| s as f32 / 32768.0));
                msk.extend_from_slice(&c.weights);
            }
            let bs = chunk.len() as i64;
            let wt = Tensor::F32(&wav, vec![bs, n_samp as i64]);
            // `weights` is not optional: without it the graph pools over the whole crop, which is a
            // different statistic and diverges exactly on overlapping speech.
            let mt = Tensor::F32(&msk, vec![bs, n_w as i64]);
            let r = self.sess.run(&[("waveform", wt), ("weights", mt)], &["embedding"])
                .map_err(|e| EngineError::Device(format!("embedding onnx: {e}")))?;
            let v = r.f32(0);
            for (j, _) in chunk.iter().enumerate() {
                let row = &v[j * self.dim..((j + 1) * self.dim).min(v.len())];
                // L2-normalise here so the clusterer's cosine distance is over unit vectors.
                let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
                let i = b * self.batch + j;
                for d in 0..self.dim.min(row.len()) { out[[i, d]] = row[d] / norm; }
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest() -> Manifest {
        serde_json::from_str(r#"{
          "pyannote_audio_rev":"3.1.1","sample_rate":16000,
          "segmentation":{"onnx":"nope.onnx","duration_s":10.0,"step_s":1.0,
                          "max_speakers_per_chunk":3,"powerset_classes":7,"source":"t"},
          "embedding":{"onnx":"nope.onnx","dim":256,"num_mel_bins":80,"n_frames":998,
                       "frame_length_ms":25.0,"frame_shift_ms":10.0,"source":"t"},
          "clustering":{"method":"centroid","threshold":0.7,"min_cluster_size":12,
                        "exclude_overlap":true,"source":"t"},
          "min_duration_off_s":0.0}"#).unwrap()
    }

    #[test]
    fn a_missing_graph_is_a_load_error_naming_the_path_not_a_panic() {
        let Err(e) = OnnxSegmenter::build(&manifest(), Path::new("/nonexistent")) else {
            panic!("a missing segmentation graph must not load");
        };
        assert!(e.to_string().contains("nope.onnx"), "the error must name the artifact: {e}");
        let Err(e) = OnnxEmbedder::build(&manifest(), Path::new("/nonexistent")) else {
            panic!("a missing embedding graph must not load");
        };
        assert!(e.to_string().contains("nope.onnx"), "{e}");
    }
}
