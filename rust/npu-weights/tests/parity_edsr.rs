mod common;

// rust/npu-weights/tests/parity_edsr.rs
// Bakes EDSR-base via the `edsr` arch and checks every baked tensor against the Python oracle npy
// (scripts/export_edsr.py). Refs dir = artifacts/edsr. Gated on oracle presence.
use std::path::Path;

#[test]
fn edsr_checkpoint_matches_python_oracle() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap();
    let refs = root.join("artifacts/edsr");
    if !refs.join("head_w.npy").exists() {
        eprintln!("SKIP edsr: oracle missing - run <venv>/bin/python scripts/export_edsr.py");
        return;
    }
    let src = format!("path:{}", refs.join("edsr_base.safetensors").to_str().unwrap());
    common::bake_and_verify("edsr", &src, "edsr", &refs);
}
