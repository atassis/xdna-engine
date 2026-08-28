mod common;

// rust/npu-weights/tests/parity_whisper.rs
// Bakes the openai/whisper-small ENCODER weights via the `whisper` arch and checks every baked
// tensor against the Python oracle npy (scripts/extract_whisper_encoder.py). Refs dir is the model
// root `artifacts/whisper-small` so checkpoint names (conv/..., L{i}/..., refs/ln_post.*) map directly
// to the npy paths the oracle wrote. Gated on oracle presence (skips with a hint if absent).
use std::path::Path;

#[test]
fn whisper_small_encoder_checkpoint_matches_python_oracle() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap();
    let refs = root.join("artifacts/whisper-small");
    if !refs.join("conv/conv1.weight.npy").exists() {
        eprintln!("SKIP whisper: oracle missing - run .venv/bin/python scripts/extract_whisper_encoder.py");
        return;
    }
    common::bake_and_verify("whisper", "hf:openai/whisper-small", "whisper", &refs);
}
