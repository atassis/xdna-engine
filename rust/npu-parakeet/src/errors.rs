//! Typed load-path errors for [`crate::FastConformerEncoder::new`]/`new_npu` and, with the `npu`
//! feature, [`crate::npu::NpuMatmul::open`]. Each variant preserves the ONE failure mode these
//! constructors used to collapse into an opaque `String` -- a caller can now match "the artifact
//! directory is missing" apart from "the NPU is busy" apart from "the artifact directory belongs
//! to the wrong model" instead of matching prose.

use std::fmt;
use std::path::PathBuf;

#[derive(Debug)]
pub enum LoadError {
    /// `ParakeetWeights::load` failed reading `path` (missing directory, missing `.npy` file, bad
    /// permissions -- `source` carries which).
    Weights { path: PathBuf, source: std::io::Error },
    /// `path`'s block count does not match the requested `ModelCfg::n_layers` -- most often an
    /// artifacts directory built for a different model variant than the one requested.
    BlockCountMismatch { path: PathBuf, found: usize, expected: usize },
    /// Opening the NPU device failed (commonly: another process holds it -- single-tenant).
    Device(String),
    /// Loading the resident kernel xclbin at `path` failed (missing/malformed file, XRT
    /// rejection).
    Kernel { path: PathBuf, source: String },
    /// A kernel argument's group-id lookup for arg `arg` failed -- an xclbin/kernel signature
    /// mismatch.
    GroupId { arg: i32, source: String },
    /// Allocating the `what` device buffer object failed (commonly: out of device memory).
    Alloc { what: &'static str, source: String },
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LoadError::Weights { path, source } => {
                write!(f, "load parakeet weights from {}: {source}", path.display())
            }
            LoadError::BlockCountMismatch { path, found, expected } => write!(
                f,
                "{}: block count {found} != expected {expected} (wrong model artifacts?)",
                path.display()
            ),
            LoadError::Device(e) => write!(f, "open NPU (single-tenant: stop npu-asr/voxd): {e}"),
            LoadError::Kernel { path, source } => {
                write!(f, "load resident {}: {source}", path.display())
            }
            LoadError::GroupId { arg, source } => write!(f, "group_id({arg}): {source}"),
            LoadError::Alloc { what, source } => write!(f, "alloc {what}: {source}"),
        }
    }
}

impl std::error::Error for LoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            LoadError::Weights { source, .. } => Some(source),
            _ => None,
        }
    }
}
