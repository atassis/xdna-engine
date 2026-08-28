mod common;

// rust/npu-weights/tests/parity_modernbert.rs
// Bakes answerdotai/ModernBERT-base via the `modernbert` arch and checks every baked tensor against
// the Python oracle npy (scripts/convert_modernbert.py). Refs dir is the model root
// `artifacts/modernbert-base`; checkpoint names (emb/*, final_norm_w, L{i}/*) map directly to the oracle's
// npy paths. ModernBERT is bias-free with fused QKV + GeGLU + RoPE; layer 0's attn_norm is Identity
// (absent). Gated on oracle presence.
use std::path::Path;

#[test]
fn modernbert_base_checkpoint_matches_python_oracle() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap();
    let refs = root.join("artifacts/modernbert-base");
    if !refs.join("emb/tok_emb.npy").exists() {
        eprintln!("SKIP modernbert: oracle missing - run .venv/bin/python scripts/convert_modernbert.py");
        return;
    }
    common::bake_and_verify("modernbert", "hf:answerdotai/ModernBERT-base", "modernbert", &refs);
}
