// rust/npu-weights/tests/parity_bge.rs
use std::path::Path;
use std::process::Command;

#[test]
fn bge_arena_matches_python_oracle() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap();
    let refs = root.join("artifacts/bge-base/encoder");
    if !refs.join("emb/word_emb.npy").exists() {
        eprintln!("SKIP: oracle missing - run .venv/bin/python scripts/export_bge.py");
        return;
    }
    let arena = root.join("target/test-arenas/bge.safetensors");
    let bin = env!("CARGO_BIN_EXE_npu-weights");
    // bake from the local HF cache dir (offline-safe path: HF cache already populated)
    let st = Command::new(bin)
        .current_dir(root)
        .args(["bake", "--source", "hf:BAAI/bge-base-en-v1.5", "--arch", "bert",
               "--arena", arena.to_str().unwrap(), "--force"])
        .status().unwrap();
    assert!(st.success(), "bake failed");
    let out = Command::new(bin)
        .current_dir(root)
        .args(["verify", "--arena", arena.to_str().unwrap(), "--arch", "bert",
               "--refs", refs.to_str().unwrap()])
        .output().unwrap();
    let s = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "verify failed:\n{s}\n{}", String::from_utf8_lossy(&out.stderr));
    assert!(s.contains("PARITY PASS"), "no parity pass:\n{s}");
}
