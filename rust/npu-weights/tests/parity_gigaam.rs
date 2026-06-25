// rust/npu-weights/tests/parity_gigaam.rs
// Bakes the GigaAM-v3 Conformer ENCODER weights from the local ONNX
// (models/gigaam_v3_encoder_static.onnx, weights inline) via the `gigaam` arch and checks every
// baked tensor against the Python oracle npy (scripts/extract_encoder.py). The refs dir is
// artifacts/encoder, so arena names (L{i}/..., pre_encode/pre_encode.conv.*) map directly to the
// oracle npy paths. Gated on both the ONNX model AND the oracle npy being present (skips otherwise).
use std::path::Path;
use std::process::Command;

#[test]
fn gigaam_conformer_encoder_arena_matches_python_oracle() {
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
    let arena = root.join("target/test-arenas/gigaam-encoder.safetensors");
    let bin = env!("CARGO_BIN_EXE_npu-weights");
    let st = Command::new(bin)
        .current_dir(root)
        .args(["bake", "--source", "path:models/gigaam_v3_encoder_static.onnx", "--arch", "gigaam",
               "--arena", arena.to_str().unwrap(), "--force"])
        .status().unwrap();
    assert!(st.success(), "bake failed for gigaam");
    let out = Command::new(bin)
        .current_dir(root)
        .args(["verify", "--arena", arena.to_str().unwrap(), "--arch", "gigaam",
               "--refs", refs.to_str().unwrap()])
        .output().unwrap();
    let s = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "verify failed for gigaam:\n{s}\n{}", String::from_utf8_lossy(&out.stderr));
    assert!(s.contains("PARITY PASS"), "no parity pass for gigaam:\n{s}");
}
