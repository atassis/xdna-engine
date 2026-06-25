// rust/npu-weights/tests/parity_modernbert.rs
// Bakes answerdotai/ModernBERT-base via the `modernbert` arch and checks every baked tensor against
// the Python oracle npy (scripts/convert_modernbert.py). Refs dir is the model root
// `artifacts/modernbert-base`; arena names (emb/*, final_norm_w, L{i}/*) map directly to the oracle's
// npy paths. ModernBERT is bias-free with fused QKV + GeGLU + RoPE; layer 0's attn_norm is Identity
// (absent). Gated on oracle presence.
use std::path::Path;
use std::process::Command;

#[test]
fn modernbert_base_arena_matches_python_oracle() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap();
    let refs = root.join("artifacts/modernbert-base");
    if !refs.join("emb/tok_emb.npy").exists() {
        eprintln!("SKIP modernbert: oracle missing - run .venv/bin/python scripts/convert_modernbert.py");
        return;
    }
    let arena = root.join("target/test-arenas/modernbert-base.safetensors");
    let bin = env!("CARGO_BIN_EXE_npu-weights");
    let st = Command::new(bin)
        .current_dir(root)
        .args(["bake", "--source", "hf:answerdotai/ModernBERT-base", "--arch", "modernbert",
               "--arena", arena.to_str().unwrap(), "--force"])
        .status().unwrap();
    assert!(st.success(), "bake failed for modernbert-base");
    let out = Command::new(bin)
        .current_dir(root)
        .args(["verify", "--arena", arena.to_str().unwrap(), "--arch", "modernbert",
               "--refs", refs.to_str().unwrap()])
        .output().unwrap();
    let s = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "verify failed for modernbert:\n{s}\n{}", String::from_utf8_lossy(&out.stderr));
    assert!(s.contains("PARITY PASS"), "no parity pass for modernbert:\n{s}");
}
