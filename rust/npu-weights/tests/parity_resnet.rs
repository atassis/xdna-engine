// rust/npu-weights/tests/parity_resnet.rs
// Bakes microsoft/resnet-18 via the `resnet` arch and checks every baked tensor against the Python
// oracle npy (scripts/export_resnet.py). Refs dir is the model root `artifacts/resnet18`; flat arena
// names (stem_w/_b, s{S}l{L}c{0,1}_w/_b, s{S}l{L}sc_w/_b, fc_w/_b) map directly to the oracle's npy
// paths. The arch folds BatchNorm into the conv exactly as the oracle does. Gated on oracle presence.
use std::path::Path;
use std::process::Command;

#[test]
fn resnet18_arena_matches_python_oracle() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap();
    let refs = root.join("artifacts/resnet18");
    if !refs.join("stem_w.npy").exists() {
        eprintln!("SKIP resnet: oracle missing - run .venv/bin/python scripts/export_resnet.py");
        return;
    }
    let arena = root.join("target/test-arenas/resnet18.safetensors");
    let bin = env!("CARGO_BIN_EXE_npu-weights");
    let st = Command::new(bin)
        .current_dir(root)
        .args(["bake", "--source", "hf:microsoft/resnet-18", "--arch", "resnet",
               "--arena", arena.to_str().unwrap(), "--force"])
        .status().unwrap();
    assert!(st.success(), "bake failed for resnet-18");
    let out = Command::new(bin)
        .current_dir(root)
        .args(["verify", "--arena", arena.to_str().unwrap(), "--arch", "resnet",
               "--refs", refs.to_str().unwrap()])
        .output().unwrap();
    let s = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "verify failed for resnet:\n{s}\n{}", String::from_utf8_lossy(&out.stderr));
    assert!(s.contains("PARITY PASS"), "no parity pass for resnet:\n{s}");
}
