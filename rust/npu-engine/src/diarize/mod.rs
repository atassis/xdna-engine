//! Speaker diarization: who spoke when. Host-only in v1; the NPU embedder swaps in behind
//! `SpeakerEmbedder` without touching the pipeline.

pub mod powerset;
pub mod stitch;
pub mod types;

pub use types::{Crop, Manifest, Segmenter, SpeakerEmbedder};
