mod common;

// rust/npu-weights/tests/parity_espcn.rs
// Bakes the pretrained ESPCN via the `espcn` arch and checks every baked tensor against the Python
// oracle npy (scripts/export_espcn.py). Refs dir = artifacts/espcn. Gated on oracle presence.
use std::path::Path;

#[test]
fn espcn_checkpoint_matches_python_oracle() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap();
    let refs = root.join("artifacts/espcn");
    if !refs.join("conv1_w.npy").exists() {
        eprintln!("SKIP espcn: oracle missing - run <venv>/bin/python scripts/export_espcn.py");
        return;
    }
    let src = format!("path:{}", refs.join("espcn_x3_dyn.onnx").to_str().unwrap());
    common::bake_and_verify("espcn", &src, "espcn", &refs);
}
