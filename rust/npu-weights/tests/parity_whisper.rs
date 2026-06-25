// rust/npu-weights/tests/parity_whisper.rs
// Bakes the openai/whisper-small ENCODER weights via the `whisper` arch and checks every baked
// tensor against the Python oracle npy (scripts/extract_whisper_encoder.py). Refs dir is the model
// root `artifacts/whisper-small` so arena names (conv/..., L{i}/..., refs/ln_post.*) map directly
// to the npy paths the oracle wrote. Gated on oracle presence (skips with a hint if absent).
use std::path::Path;
use std::process::Command;

#[test]
fn whisper_small_encoder_arena_matches_python_oracle() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap();
    let refs = root.join("artifacts/whisper-small");
    if !refs.join("conv/conv1.weight.npy").exists() {
        eprintln!("SKIP whisper: oracle missing - run .venv/bin/python scripts/extract_whisper_encoder.py");
        return;
    }
    let arena = root.join("target/test-arenas/whisper-small.safetensors");
    let bin = env!("CARGO_BIN_EXE_npu-weights");
    let st = Command::new(bin)
        .current_dir(root)
        .args(["bake", "--source", "hf:openai/whisper-small", "--arch", "whisper",
               "--arena", arena.to_str().unwrap(), "--force"])
        .status().unwrap();
    assert!(st.success(), "bake failed for whisper-small");
    let out = Command::new(bin)
        .current_dir(root)
        .args(["verify", "--arena", arena.to_str().unwrap(), "--arch", "whisper",
               "--refs", refs.to_str().unwrap()])
        .output().unwrap();
    let s = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "verify failed for whisper:\n{s}\n{}", String::from_utf8_lossy(&out.stderr));
    assert!(s.contains("PARITY PASS"), "no parity pass for whisper:\n{s}");
}
