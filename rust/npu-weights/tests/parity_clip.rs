// rust/npu-weights/tests/parity_clip.rs
// Bakes laion/CLIP-ViT-B-32-laion2B-s34B-b79K (transformers CLIPModel, openai architecture) via the
// `clip` arch and checks every baked tensor against the Python oracle npy (scripts/convert_clip.py).
// Refs dir is the model root `artifacts/clip-vit-b32`; checkpoint names (text/*, vision/*, text_projection,
// visual_projection, logit_scale) map directly to the oracle's npy paths. Both towers share the
// CLIPEncoderLayer; the vision patch-embed conv2d is im2col-flattened + transposed like vit. Gated on
// oracle presence. (laion is used because openai/clip-vit-base-patch32 ships only pytorch_model.bin.)
use std::path::Path;
use std::process::Command;

#[test]
fn clip_vit_b32_checkpoint_matches_python_oracle() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap();
    let refs = root.join("artifacts/clip-vit-b32");
    if !refs.join("text/tok_emb.npy").exists() {
        eprintln!("SKIP clip: oracle missing - run .venv/bin/python scripts/convert_clip.py");
        return;
    }
    let checkpoint = root.join("target/test-checkpoints/clip-vit-b32.safetensors");
    let bin = env!("CARGO_BIN_EXE_npu-weights");
    let st = Command::new(bin)
        .current_dir(root)
        .args(["bake", "--source", "hf:laion/CLIP-ViT-B-32-laion2B-s34B-b79K", "--arch", "clip",
               "--checkpoint", checkpoint.to_str().unwrap(), "--force"])
        .status().unwrap();
    assert!(st.success(), "bake failed for clip-vit-b32");
    let out = Command::new(bin)
        .current_dir(root)
        .args(["verify", "--checkpoint", checkpoint.to_str().unwrap(), "--arch", "clip",
               "--refs", refs.to_str().unwrap()])
        .output().unwrap();
    let s = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "verify failed for clip:\n{s}\n{}", String::from_utf8_lossy(&out.stderr));
    assert!(s.contains("PARITY PASS"), "no parity pass for clip:\n{s}");
}
