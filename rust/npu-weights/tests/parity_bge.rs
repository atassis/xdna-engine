// rust/npu-weights/tests/parity_bge.rs
//
// Calls the LIBRARY, not a binary. It used to shell out to the `npu-weights` executable, which
// broke the moment that binary became a deprecation shim -- and the shell-out was never what the
// test was about: this checks checkpoint parity, not CLI plumbing.
use std::path::Path;

use npu_weights::checkpoint;
use npu_weights::spec::{ModelSpec, Source};

#[test]
fn bge_checkpoint_matches_python_oracle() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap();
    let refs = root.join("artifacts/bge-base/encoder");
    if !refs.join("emb/word_emb.npy").exists() {
        eprintln!("SKIP: oracle missing - run .venv/bin/python scripts/export_bge.py");
        return;
    }
    let checkpoint = root.join("target/test-checkpoints/bge.safetensors");
    let spec = ModelSpec {
        source: Source::parse("hf:BAAI/bge-base-en-v1.5").expect("source"),
        arch: "bert".into(),
        checkpoint: Some(checkpoint.clone()),
    };
    // Bake from the local HF cache (offline-safe: the cache is already populated).
    spec.ensure_checkpoint(root, true).expect("bake");

    let loaded = checkpoint::load(&checkpoint, "bert").expect("load");
    let (n, max) = checkpoint::verify_against_npy(&loaded, &refs).expect("verify");
    assert!(n > 0, "no reference tensors compared -- the oracle dir matched nothing");
    assert!(max < 5e-2, "parity FAILED: {n} tensors, max rel-err {max:.4e} >= 5e-2");
}
