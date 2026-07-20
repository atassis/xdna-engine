//! A super-resolution net as DATA: an ordered list of ops over the brick vocabulary + a weights arena
//! ref + the integer scale. `espcn.json` is one instance; `abpn.json` (M3) is another over the schema.
use crate::SrError;
use serde::Deserialize;
use std::path::Path;

/// One op in the schedule. `weights` names the arena tensors (e.g. "conv1" -> conv1_w / conv1_b).
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "op")]
pub enum Op {
    /// 2-D convolution: kh=kw=k, symmetric pad, optional ReLU epilogue.
    #[serde(rename = "conv2d")]
    Conv2d {
        weights: String,
        k: usize,
        pad: usize,
        relu: bool,
        cin: usize,
        cout: usize,
    },
    /// Depth-to-space upsample by factor r (CRD order), zero compute.
    #[serde(rename = "pixel_shuffle")]
    PixelShuffle { r: usize },
    /// Save the current stream into a named skip register (residual connections).
    #[serde(rename = "save")]
    Save { name: String },
    /// Add a previously-saved stream (by name) to the current stream, elementwise (same shape).
    #[serde(rename = "add")]
    Add { name: String },
}

/// How the engine feeds pixels to the net.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum InputMode {
    /// Y-only: net upscales luma; chroma is bicubic (ESPCN). The default.
    #[default]
    Y,
    /// RGB: net upscales all three planar channels (EDSR).
    Rgb,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Schedule {
    pub name: String,
    pub scale: usize,
    pub arena: String, // path to the baked safetensors arena, relative to repo root
    #[serde(default)]
    pub input: InputMode,
    pub ops: Vec<Op>,
}

impl Schedule {
    pub fn load(path: &Path) -> Result<Schedule, SrError> {
        let s = std::fs::read_to_string(path)
            .map_err(|e| SrError::Load(format!("read {}: {e}", path.display())))?;
        serde_json::from_str(&s).map_err(|e| SrError::Load(format!("parse {}: {e}", path.display())))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn loads_espcn_schedule() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap();
        let p = root.join("artifacts/espcn/espcn.json");
        if !p.exists() {
            eprintln!("SKIP: espcn.json missing");
            return;
        }
        let s = Schedule::load(&p).unwrap();
        assert_eq!(s.scale, 3);
        assert_eq!(s.ops.len(), 5); // 4 conv2d + 1 pixel_shuffle
        match &s.ops[0] {
            Op::Conv2d { k, cout, .. } => {
                assert_eq!(*k, 5);
                assert_eq!(*cout, 64);
            }
            _ => panic!("op0 not conv2d"),
        }
        match s.ops.last().unwrap() {
            Op::PixelShuffle { r } => assert_eq!(*r, 3),
            _ => panic!("last op not pixel_shuffle"),
        }
    }
}
