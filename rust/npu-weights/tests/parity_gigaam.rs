mod common;

// rust/npu-weights/tests/parity_gigaam.rs
// Bakes the GigaAM-v3 Conformer ENCODER weights from the local ONNX
// (models/gigaam_v3_encoder_static.onnx, weights inline) via the `gigaam` arch and checks every
// baked tensor against the Python oracle npy (scripts/extract_encoder.py). The refs dir is
// artifacts/encoder, so checkpoint names (L{i}/..., pre_encode/pre_encode.conv.*) map directly to the
// oracle npy paths. Gated on both the ONNX model AND the oracle npy being present (skips otherwise).
use std::path::Path;

#[test]
fn gigaam_conformer_encoder_checkpoint_matches_python_oracle() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap();
    let onnx = root.join("models/gigaam_v3_encoder_static.onnx");
    let refs = root.join("artifacts/encoder");
    if !onnx.exists() {
        eprintln!("SKIP gigaam: models/gigaam_v3_encoder_static.onnx missing");
        return;
    }
    if !refs.join("L0/self_attn.linear_q.weight.npy").exists() {
        eprintln!("SKIP gigaam: oracle missing - run .venv/bin/python scripts/extract_encoder.py");
        return;
    }
    common::bake_and_verify("gigaam", "path:models/gigaam_v3_encoder_static.onnx", "gigaam", &refs);
}
