//! M3 CPU frontier gate: the EDSR-base whole net (skip registers + RGB + pixel_shuffle) reproduces the
//! torch EDSR oracle within tolerance. Requires the arena baked (`cargo test -p npu-weights parity_edsr`)
//! + the oracle npy (scripts/export_edsr.py). Runs from repo root. Gate on rel-L2, never WER.
use ndarray::ArrayD;
use ndarray_npy::read_npy;
use npu_sr::SrEngine;

fn rel_l2(a: &[f32], b: &[f32]) -> f32 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| ((*x - *y) as f64).powi(2)).sum::<f64>().sqrt();
    let den: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum::<f64>().sqrt();
    (num / (den + 1e-12)) as f32
}

#[test]
fn cpu_edsr_matches_torch_oracle() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap().parent().unwrap().to_path_buf();
    let refs = root.join("artifacts/edsr");
    if !refs.join("gate_sr.npy").exists() || !root.join("target/test-arenas/edsr.safetensors").exists() {
        eprintln!("SKIP: edsr oracle/arena missing - run export_edsr.py + parity_edsr");
        return;
    }
    std::env::set_current_dir(&root).unwrap();

    let lr: ArrayD<f32> = read_npy(refs.join("gate_lr.npy")).unwrap(); // [1,3,H,W]
    let sr_ref: ArrayD<f32> = read_npy(refs.join("gate_sr.npy")).unwrap(); // [1,3,3H,3W]
    let (ls, ss) = (lr.shape(), sr_ref.shape());
    let (lr_h, lr_w) = (ls[ls.len() - 2], ls[ls.len() - 1]);
    let (sr_h, sr_w) = (ss[ss.len() - 2], ss[ss.len() - 1]);

    let mut eng = SrEngine::load("artifacts/edsr/edsr.json", false).unwrap();
    // gate_lr is already [1,3,H,W] row-major = planar [3,H,W].
    let planar: Vec<f32> = lr.iter().cloned().collect();
    let (out, ow, oh) = eng.upscale_planar_rgb(&planar, lr_w, lr_h).unwrap();
    assert_eq!((oh, ow), (sr_h, sr_w));
    let sr_v: Vec<f32> = sr_ref.iter().cloned().collect();
    let rl2 = rel_l2(&out, &sr_v);
    eprintln!("CPU EDSR frontier rel-L2 vs torch oracle = {rl2:.3e}");
    assert!(rl2 < 1.5e-2, "CPU EDSR rel-L2 vs oracle = {rl2:.3e} (want < 1.5e-2)");
}
