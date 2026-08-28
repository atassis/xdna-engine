mod common;

// rust/npu-weights/tests/parity_vit.rs
// Bakes google/vit-base-patch16-224 via the `vit` arch and checks every baked tensor against the
// Python oracle npy (scripts/convert_vit.py). Refs dir is the model root `artifacts/vit-base`; checkpoint
// names (patch_proj.*, cls_token, pos_emb, ln_final.*, classifier.*, L{i}/...) map directly to the
// oracle's npy paths. The patch-embed conv2d is im2col-flattened + transposed exactly as the oracle
// does. Gated on oracle presence.
use std::path::Path;

#[test]
fn vit_base_checkpoint_matches_python_oracle() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap();
    let refs = root.join("artifacts/vit-base");
    if !refs.join("patch_proj.weight.npy").exists() {
        eprintln!("SKIP vit: oracle missing - run .venv/bin/python scripts/convert_vit.py");
        return;
    }
    common::bake_and_verify("vit", "hf:google/vit-base-patch16-224", "vit", &refs);
}
