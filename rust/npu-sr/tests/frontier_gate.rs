//! The CPU frontier reproduces the ONNX ESPCN whole-net output within tolerance -- the M2 correctness
//! spine. Requires the checkpoint baked (`cargo test -p npu-weights parity_espcn`) + the oracle npy
//! (scripts/export_espcn.py). Runs from the repo root (the engine's production convention); a single
//! test per binary makes set_current_dir race-free.
use ndarray::ArrayD;
use ndarray_npy::read_npy;
use npu_sr::{Plane, SrEngine};

fn rel_l2(a: &[f32], b: &[f32]) -> f32 {
    let num: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| ((*x - *y) as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    let den: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum::<f64>().sqrt();
    (num / (den + 1e-12)) as f32
}

#[test]
fn cpu_frontier_matches_onnx_oracle() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    let refs = root.join("artifacts/espcn");
    if !refs.join("gate_sr.npy").exists() || !root.join("target/test-checkpoints/espcn.safetensors").exists() {
        eprintln!("SKIP: oracle/checkpoint missing - run export_espcn.py + parity_espcn first");
        return;
    }
    // The engine resolves the schedule's relative checkpoint path from cwd = repo root (production convention).
    std::env::set_current_dir(&root).unwrap();

    let lr: ArrayD<f32> = read_npy(refs.join("gate_lr.npy")).unwrap();
    let sr_ref: ArrayD<f32> = read_npy(refs.join("gate_sr.npy")).unwrap();
    let (ls, ss) = (lr.shape(), sr_ref.shape());
    let (lr_h, lr_w) = (ls[ls.len() - 2], ls[ls.len() - 1]);
    let (sr_h, sr_w) = (ss[ss.len() - 2], ss[ss.len() - 1]);

    let mut eng = SrEngine::load("artifacts/espcn/espcn.json", false).unwrap();
    let out = eng
        .upscale_plane(&Plane {
            w: lr_w,
            h: lr_h,
            data: lr.iter().cloned().collect(),
        })
        .unwrap();
    assert_eq!((out.h, out.w), (sr_h, sr_w));
    let sr_v: Vec<f32> = sr_ref.iter().cloned().collect();
    let rl2 = rel_l2(&out.data, &sr_v);
    eprintln!("CPU frontier rel-L2 vs ONNX oracle = {rl2:.3e}");
    assert!(rl2 < 1.0e-2, "CPU frontier rel-L2 vs ONNX oracle = {rl2:.3e} (want < 1e-2)");
}
