//! M3 device gate: the whole EDSR-base net runs on the NPU (host im2col -> N-tiled whole-array bf16 GEMM
//! + host skips/pixel_shuffle) and matches the CPU frontier + the torch oracle within tolerance. Skips if
//! no NPU / xclbin / checkpoint. Runs from repo root. Serialize device access under scripts/npu_lock.sh.
use ndarray::ArrayD;
use ndarray_npy::read_npy;
use npu_sr::SrEngine;

fn rel_l2(a: &[f32], b: &[f32]) -> f32 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| ((*x - *y) as f64).powi(2)).sum::<f64>().sqrt();
    let den: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum::<f64>().sqrt();
    (num / (den + 1e-12)) as f32
}

#[test]
fn npu_edsr_matches_cpu_and_oracle() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap().parent().unwrap().to_path_buf();
    let refs = root.join("artifacts/edsr");
    let xclbin = root.join(
        "mlir-aie/programming_examples/basic/matrix_multiplication/whole_array/build/final_512x576x256_32x32x32_8c.xclbin",
    );
    if !npu_sr::npu_available() {
        eprintln!("SKIP: no NPU device");
        return;
    }
    if !refs.join("gate_sr.npy").exists() || !root.join("target/test-checkpoints/edsr.safetensors").exists() || !xclbin.exists() {
        eprintln!("SKIP: edsr oracle/checkpoint/xclbin missing");
        return;
    }
    std::env::set_current_dir(&root).unwrap();
    let lr: ArrayD<f32> = read_npy(refs.join("gate_lr.npy")).unwrap();
    let sr_ref: ArrayD<f32> = read_npy(refs.join("gate_sr.npy")).unwrap();
    let ls = lr.shape();
    let (lr_h, lr_w) = (ls[ls.len() - 2], ls[ls.len() - 1]);
    let planar: Vec<f32> = lr.iter().cloned().collect();

    let mut cpu = SrEngine::load("artifacts/edsr/edsr.json", false).unwrap();
    let (out_cpu, _, _) = cpu.upscale_planar_rgb(&planar, lr_w, lr_h).unwrap();

    let mut npu = SrEngine::load("artifacts/edsr/edsr.json", true).unwrap();
    let (out_npu, _, _) = npu.upscale_planar_rgb(&planar, lr_w, lr_h).unwrap();

    let sr_v: Vec<f32> = sr_ref.iter().cloned().collect();
    let rl2_vs_cpu = rel_l2(&out_npu, &out_cpu);
    let rl2_vs_oracle = rel_l2(&out_npu, &sr_v);
    eprintln!("NPU EDSR: rel-L2 vs CPU = {rl2_vs_cpu:.3e}, vs torch oracle = {rl2_vs_oracle:.3e}");
    assert!(rl2_vs_cpu < 1.5e-2, "NPU EDSR vs CPU rel-L2 = {rl2_vs_cpu:.3e} (want < 1.5e-2)");
    assert!(rl2_vs_oracle < 2.0e-2, "NPU EDSR vs oracle rel-L2 = {rl2_vs_oracle:.3e} (want < 2e-2)");
}
