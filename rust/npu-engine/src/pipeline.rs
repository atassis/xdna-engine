//! The 3-stage model abstraction. `Encoder` is the shared NPU stage (object-safe); `Frontend`
//! and `Head` are per-domain host/ONNX glue. A `Scenario` is one assembled pipeline.

use ndarray::Array2;

/// The genuinely-shared, genuinely-hard NPU stage. INTERFACE CONTRACT for sibling models
/// (GigaAM Conformer, Parakeet FastConformer, BERT): implement this and the registry can host it.
pub trait Encoder {
    /// `x` is [M, D_in] (bf16-valued f32); `valid_len` = non-padded rows. Returns [M, D].
    fn forward_last(&self, x: &Array2<f32>, valid_len: usize) -> Array2<f32>;
}

/// A full ASR backend: raw PCM samples -> text. Different ASR models (GigaAM RNNT, Parakeet TDT)
/// have different preproc + decode, so the whole transcription path is the trait; the encoder stage
/// inside each still implements `Encoder`.
pub trait AsrModel {
    fn transcribe(&self, samples: &[i16]) -> String;
}

/// Raw input -> encoder input activations + valid_len.
pub trait Frontend {
    type Input;
    fn run(&self, input: Self::Input) -> (Array2<f32>, usize);
}

/// Encoder output -> final result.
pub trait Head {
    type Output;
    fn run(&self, encoded: &Array2<f32>, valid_len: usize) -> Self::Output;
}

/// One assembled, ready-to-serve pipeline. The registry returns this; `engine_serve` matches on it.
pub enum Scenario {
    Asr(Box<dyn AsrModel>),
    Embed(crate::bert::EmbedPipeline),
}
