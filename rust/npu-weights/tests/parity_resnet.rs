mod common;

// rust/npu-weights/tests/parity_resnet.rs
// Bakes microsoft/resnet-18 via the `resnet` arch and checks every baked tensor against the Python
// oracle npy (scripts/export_resnet.py). Refs dir is the model root `artifacts/resnet18`; flat checkpoint
// names (stem_w/_b, s{S}l{L}c{0,1}_w/_b, s{S}l{L}sc_w/_b, fc_w/_b) map directly to the oracle's npy
// paths. The arch folds BatchNorm into the conv exactly as the oracle does. Gated on oracle presence.
use std::path::Path;

#[test]
fn resnet18_checkpoint_matches_python_oracle() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap();
    let refs = root.join("artifacts/resnet18");
    if !refs.join("stem_w.npy").exists() {
        eprintln!("SKIP resnet: oracle missing - run .venv/bin/python scripts/export_resnet.py");
        return;
    }
    common::bake_and_verify("resnet", "hf:microsoft/resnet-18", "resnet", &refs);
}
