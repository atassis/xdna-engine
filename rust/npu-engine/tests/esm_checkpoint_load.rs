//! Host-only equivalence check: EsmWeights loaded from a baked bge checkpoint must match the npy-dir
//! load within bf16 tolerance. Gated on artifacts being present; prints SKIP and passes if the
//! bge checkpoint or the npy encoder dir is absent (regenerate-free - assumes the caller already ran
//! the bge bake + export).
//!
//! Populate artifacts first (host-only, fast):
//!   ~/.local/bin/npu-weights bake --source hf:BAAI/bge-base-en-v1.5 --arch bert
//!   .venv/bin/python scripts/export_bge.py

use std::path::{Path, PathBuf};

use npu_engine::esm::weights::EsmWeights;

const N_LAYERS: usize = 12;
const REL_TOL: f32 = 5e-2; // bf16 tolerance

/// Repo root = two levels up from this crate's manifest dir (rust/npu-engine -> repo root).
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("manifest has a repo root")
        .to_path_buf()
}

/// First bge checkpoint under artifacts/checkpoints/ matching bert__*.safetensors, if any.
fn find_bge_checkpoint(root: &Path) -> Option<PathBuf> {
    let dir = root.join("artifacts/checkpoints");
    let mut hits: Vec<PathBuf> = std::fs::read_dir(&dir)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            let n = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
            n.starts_with("bert__") && n.ends_with(".safetensors")
        })
        .collect();
    hits.sort();
    hits.into_iter().next()
}

/// Max relative error between two flat slices (rel = |a-b| / max(|a|,|b|,eps)).
fn max_rel_err(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "tensor length mismatch");
    let mut m = 0.0f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let denom = x.abs().max(y.abs()).max(1e-6);
        m = m.max((x - y).abs() / denom);
    }
    m
}

#[test]
fn esm_checkpoint_matches_npy_load() {
    let root = repo_root();

    let checkpoint = match find_bge_checkpoint(&root) {
        Some(p) => p,
        None => {
            eprintln!("SKIP: no bge checkpoint under artifacts/checkpoints/bert__*.safetensors");
            return;
        }
    };
    let npy_dir = root.join("artifacts/bge-base/encoder");
    if !npy_dir.join("emb").is_dir() {
        eprintln!("SKIP: npy encoder dir absent at {}", npy_dir.display());
        return;
    }

    let from_checkpoint = EsmWeights::load_checkpoint(&checkpoint, "bert", N_LAYERS)
        .expect("load_checkpoint must succeed for a baked bge checkpoint");
    // The npy path is unconditional here (NPU_WEIGHTS_CHECKPOINT is not set in this test process).
    let from_npy = EsmWeights::load(&npy_dir, N_LAYERS).expect("npy load must succeed");

    assert_eq!(from_checkpoint.n_layers(), N_LAYERS);
    assert_eq!(from_npy.n_layers(), N_LAYERS);
    assert!(from_checkpoint.final_ln.is_none(), "bge has no final_ln");
    assert!(from_npy.final_ln.is_none(), "bge has no final_ln");

    let mut global_max = 0.0f32;
    let mut compared = 0usize;

    // emb map: every key present in BOTH.
    for (k, v_npy) in from_npy.emb.iter() {
        let v_checkpoint = from_checkpoint
            .emb
            .get(k)
            .unwrap_or_else(|| panic!("emb/{k} present in npy but missing from checkpoint"));
        let a: Vec<f32> = v_checkpoint.iter().copied().collect();
        let b: Vec<f32> = v_npy.iter().copied().collect();
        let e = max_rel_err(&a, &b);
        global_max = global_max.max(e);
        compared += 1;
    }

    // per-layer maps: compare via the EsmLayer accessor on every key the npy side has.
    for i in 0..N_LAYERS {
        let la = &from_checkpoint.layers[i];
        let ln = &from_npy.layers[i];
        for k in ln.keys() {
            let a = la.v(k); // panics if missing in checkpoint -> a real failure
            let b = ln.v(k);
            let e = max_rel_err(&a, &b);
            global_max = global_max.max(e);
            compared += 1;
        }
    }

    println!("checkpoint<->npy equivalence: compared {compared} tensors, max rel-err = {global_max:.6}");
    assert!(compared > 0, "no tensors compared - keys did not line up");
    assert!(
        global_max <= REL_TOL,
        "max rel-err {global_max} exceeds bf16 tolerance {REL_TOL}"
    );
}
