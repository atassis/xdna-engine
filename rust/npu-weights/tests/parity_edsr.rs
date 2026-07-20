// rust/npu-weights/tests/parity_edsr.rs
// Bakes EDSR-base via the `edsr` arch and checks every baked tensor against the Python oracle npy
// (scripts/export_edsr.py). Refs dir = artifacts/edsr. Gated on oracle presence.
use std::path::Path;
use std::process::Command;

#[test]
fn edsr_arena_matches_python_oracle() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap();
    let refs = root.join("artifacts/edsr");
    if !refs.join("head_w.npy").exists() {
        eprintln!("SKIP edsr: oracle missing - run <venv>/bin/python scripts/export_edsr.py");
        return;
    }
    let arena = root.join("target/test-arenas/edsr.safetensors");
    let bin = env!("CARGO_BIN_EXE_npu-weights");
    let src = format!("path:{}", refs.join("edsr_base.safetensors").to_str().unwrap());
    let st = Command::new(bin)
        .current_dir(root)
        .args([
            "bake", "--source", &src, "--arch", "edsr",
            "--arena", arena.to_str().unwrap(), "--force",
        ])
        .status()
        .unwrap();
    assert!(st.success(), "bake failed for edsr");
    let out = Command::new(bin)
        .current_dir(root)
        .args([
            "verify", "--arena", arena.to_str().unwrap(), "--arch", "edsr",
            "--refs", refs.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    let s = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "verify failed for edsr:\n{s}\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(s.contains("PARITY PASS"), "no parity pass for edsr:\n{s}");
}
