// rust/npu-weights/tests/parity_esm.rs
use std::path::Path;
use std::process::Command;

fn check_one(sub: &str, hf: &str) {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap();
    let refs = root.join(format!("artifacts/{sub}/encoder"));
    if !refs.join("emb/word_emb.npy").exists() {
        eprintln!("SKIP {sub}: oracle missing - run .venv/bin/python scripts/export_esm.py {hf} {sub}");
        return;
    }
    let arena = root.join(format!("target/test-arenas/{sub}.safetensors"));
    let bin = env!("CARGO_BIN_EXE_npu-weights");
    let st = Command::new(bin)
        .current_dir(root)
        .args(["bake", "--source", &format!("hf:{hf}"), "--arch", "esm",
               "--arena", arena.to_str().unwrap(), "--force"])
        .status().unwrap();
    assert!(st.success(), "bake failed for {sub}");
    let out = Command::new(bin)
        .current_dir(root)
        .args(["verify", "--arena", arena.to_str().unwrap(), "--arch", "esm",
               "--refs", refs.to_str().unwrap()])
        .output().unwrap();
    let s = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "verify failed for {sub}:\n{s}\n{}", String::from_utf8_lossy(&out.stderr));
    assert!(s.contains("PARITY PASS"), "no parity pass for {sub}:\n{s}");
}

#[test]
fn esm2_8m_arena_matches_python_oracle() {
    check_one("esm2-8m", "facebook/esm2_t6_8M_UR50D");
}

#[test]
fn esm2_35m_arena_matches_python_oracle() {
    check_one("esm2-35m", "facebook/esm2_t12_35M_UR50D");
}
