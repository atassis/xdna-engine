// rust/npu-weights/tests/parity_vit.rs
// Bakes google/vit-base-patch16-224 via the `vit` arch and checks every baked tensor against the
// Python oracle npy (scripts/convert_vit.py). Refs dir is the model root `artifacts/vit-base`; arena
// names (patch_proj.*, cls_token, pos_emb, ln_final.*, classifier.*, L{i}/...) map directly to the
// oracle's npy paths. The patch-embed conv2d is im2col-flattened + transposed exactly as the oracle
// does. Gated on oracle presence.
use std::path::Path;
use std::process::Command;

#[test]
fn vit_base_arena_matches_python_oracle() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap();
    let refs = root.join("artifacts/vit-base");
    if !refs.join("patch_proj.weight.npy").exists() {
        eprintln!("SKIP vit: oracle missing - run .venv/bin/python scripts/convert_vit.py");
        return;
    }
    let arena = root.join("target/test-arenas/vit-base.safetensors");
    let bin = env!("CARGO_BIN_EXE_npu-weights");
    let st = Command::new(bin)
        .current_dir(root)
        .args(["bake", "--source", "hf:google/vit-base-patch16-224", "--arch", "vit",
               "--arena", arena.to_str().unwrap(), "--force"])
        .status().unwrap();
    assert!(st.success(), "bake failed for vit-base");
    let out = Command::new(bin)
        .current_dir(root)
        .args(["verify", "--arena", arena.to_str().unwrap(), "--arch", "vit",
               "--refs", refs.to_str().unwrap()])
        .output().unwrap();
    let s = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "verify failed for vit:\n{s}\n{}", String::from_utf8_lossy(&out.stderr));
    assert!(s.contains("PARITY PASS"), "no parity pass for vit:\n{s}");
}
