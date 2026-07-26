// rust/npu-weights/tests/parity_espcn.rs
// Bakes the pretrained ESPCN via the `espcn` arch and checks every baked tensor against the Python
// oracle npy (scripts/export_espcn.py). Refs dir = artifacts/espcn. Gated on oracle presence.
use std::path::Path;
use std::process::Command;

#[test]
fn espcn_checkpoint_matches_python_oracle() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap();
    let refs = root.join("artifacts/espcn");
    if !refs.join("conv1_w.npy").exists() {
        eprintln!("SKIP espcn: oracle missing - run <venv>/bin/python scripts/export_espcn.py");
        return;
    }
    let checkpoint = root.join("target/test-checkpoints/espcn.safetensors");
    let bin = env!("CARGO_BIN_EXE_npu-weights");
    let src = format!("path:{}", refs.join("espcn_x3_dyn.onnx").to_str().unwrap());
    let st = Command::new(bin)
        .current_dir(root)
        .args([
            "bake", "--source", &src, "--arch", "espcn",
            "--checkpoint", checkpoint.to_str().unwrap(), "--force",
        ])
        .status()
        .unwrap();
    assert!(st.success(), "bake failed for espcn");
    let out = Command::new(bin)
        .current_dir(root)
        .args([
            "verify", "--checkpoint", checkpoint.to_str().unwrap(), "--arch", "espcn",
            "--refs", refs.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    let s = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "verify failed for espcn:\n{s}\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(s.contains("PARITY PASS"), "no parity pass for espcn:\n{s}");
}
