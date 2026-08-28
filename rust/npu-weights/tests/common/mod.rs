//! Shared helper for the checkpoint parity tests.
//!
//! These used to each shell out to the `npu-weights` executable. That binary is now a deprecation
//! shim (the tooling moved to `npu weights`), and more to the point the subprocess was never what
//! the tests were checking -- they check that a baked checkpoint matches a Python oracle, which is
//! library behaviour. Going through the library also makes a failure a Rust panic with a message
//! instead of a non-zero exit code and a scraped stdout string.
#![allow(dead_code)]

use std::path::{Path, PathBuf};

use npu_weights::checkpoint;
use npu_weights::spec::{ModelSpec, Source};

/// Repo root, from this crate's manifest dir.
pub fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf()
}

/// Bake `source` under `arch` and compare every tensor to `refs/<name>.npy`.
///
/// Panics with the tensor count and worst rel-err on failure. The 5e-2 bound is the same one the
/// `npu weights verify` subcommand enforces, so the test and the tool agree by construction.
pub fn bake_and_verify(label: &str, source: &str, arch: &str, refs: &Path) {
    let root = repo_root();
    let checkpoint = root.join(format!("target/test-checkpoints/{label}.safetensors"));
    let spec = ModelSpec {
        source: Source::parse(source).unwrap_or_else(|e| panic!("{label}: bad source: {e}")),
        arch: arch.to_string(),
        checkpoint: Some(checkpoint.clone()),
    };
    spec.ensure_checkpoint(&root, true)
        .unwrap_or_else(|e| panic!("{label}: bake failed: {e}"));
    let loaded = checkpoint::load(&checkpoint, arch)
        .unwrap_or_else(|e| panic!("{label}: load failed: {e}"));
    let (n, max) = checkpoint::verify_against_npy(&loaded, refs)
        .unwrap_or_else(|e| panic!("{label}: verify failed: {e}"));
    assert!(n > 0, "{label}: no reference tensors compared -- refs dir matched nothing");
    assert!(max < 5e-2, "{label}: parity FAILED over {n} tensors, max rel-err {max:.4e} >= 5e-2");
}
