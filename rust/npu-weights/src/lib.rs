// rust/npu-weights/src/lib.rs
//! Rust-native weight loader. See internal notes
pub mod arch;
pub mod checkpoint;
pub mod fingerprint;
pub mod onnx;
pub mod source;
pub mod spec;

/// Bump to invalidate every baked checkpoint on a layout-affecting change.
pub const FORMAT_VERSION: u32 = 1;
