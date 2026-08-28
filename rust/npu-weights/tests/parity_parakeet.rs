mod common;

// rust/npu-weights/tests/parity_parakeet.rs
// Bakes the Parakeet-tdt-0.6b-v3 FastConformer ENCODER weights from the local ONNX
// (models/parakeet/encoder-model.onnx + .data sidecar) via the `fastconformer` arch and checks
// every baked tensor against the Python oracle npy (scripts/extract_parakeet_encoder.py). The refs
// dir is artifacts/parakeet/encoder, so checkpoint names (L{i}/..., pre_encode/...) map directly to the
// oracle npy paths. Gated on both the ONNX model AND the oracle npy being present (skips otherwise).
use std::path::Path;

#[test]
fn parakeet_fastconformer_encoder_checkpoint_matches_python_oracle() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap();
    let onnx = root.join("models/parakeet/encoder-model.onnx");
    let refs = root.join("artifacts/parakeet/encoder");
    if !onnx.exists() {
        eprintln!("SKIP parakeet: models/parakeet/encoder-model.onnx missing");
        return;
    }
    if !refs.join("L0/self_attn.linear_q.weight.npy").exists() {
        eprintln!("SKIP parakeet: oracle missing - run .venv/bin/python scripts/extract_parakeet_encoder.py");
        return;
    }
    common::bake_and_verify("parakeet", "path:models/parakeet/encoder-model.onnx", "fastconformer", &refs);
}
