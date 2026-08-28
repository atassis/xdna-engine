mod common;

// rust/npu-weights/tests/parity_esm.rs
use std::path::Path;

fn check_one(sub: &str, hf: &str) {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap();
    let refs = root.join(format!("artifacts/{sub}/encoder"));
    if !refs.join("emb/word_emb.npy").exists() {
        eprintln!("SKIP {sub}: oracle missing - run .venv/bin/python scripts/export_esm.py {hf} {sub}");
        return;
    }
    common::bake_and_verify(sub, &format!("hf:{hf}"), "esm", &refs);
}

#[test]
fn esm2_8m_checkpoint_matches_python_oracle() {
    check_one("esm2-8m", "facebook/esm2_t6_8M_UR50D");
}

#[test]
fn esm2_35m_checkpoint_matches_python_oracle() {
    check_one("esm2-35m", "facebook/esm2_t12_35M_UR50D");
}
