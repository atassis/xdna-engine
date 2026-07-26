//! Device parity gate (Task 4b): the NPU frontier (host im2col -> whole-array bf16 GEMM) reproduces the
//! CPU frontier AND the ONNX oracle within the M1 regime (rel-L2 <= 1.1e-2). Gate on rel-L2, NEVER WER.
//! Skips cleanly if no NPU device or the whole_array xclbin is absent. Runs from repo root (xclbin +
//! checkpoint paths are repo-root-relative). Serialize device access under scripts/npu_lock.sh externally.
use ndarray::ArrayD;
use ndarray_npy::read_npy;
use npu_sr::{Plane, SrEngine};

fn rel_l2(a: &[f32], b: &[f32]) -> f32 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| ((*x - *y) as f64).powi(2)).sum::<f64>().sqrt();
    let den: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum::<f64>().sqrt();
    (num / (den + 1e-12)) as f32
}

#[test]
fn npu_frontier_matches_cpu_and_oracle() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap().parent().unwrap().to_path_buf();
    let refs = root.join("artifacts/espcn");
    let xclbin = root.join(
        "mlir-aie/programming_examples/basic/matrix_multiplication/whole_array/build/final_512x576x256_32x32x32_8c.xclbin",
    );
    if !npu_sr::npu_available() {
        eprintln!("SKIP: no NPU device");
        return;
    }
    if !refs.join("gate_sr.npy").exists() || !root.join("target/test-checkpoints/espcn.safetensors").exists() {
        eprintln!("SKIP: oracle/checkpoint missing");
        return;
    }
    if !xclbin.exists() {
        eprintln!("SKIP: whole_array xclbin missing - build it (make M=512 K=576 N=256 n_aie_cols=8 NPU2=1 dtype_in=bf16 dtype_out=f32) + copy insts .bin to .txt");
        return;
    }
    std::env::set_current_dir(&root).unwrap();

    let lr: ArrayD<f32> = read_npy(refs.join("gate_lr.npy")).unwrap();
    let sr_ref: ArrayD<f32> = read_npy(refs.join("gate_sr.npy")).unwrap();
    let (ls, ss) = (lr.shape(), sr_ref.shape());
    let (lr_h, lr_w) = (ls[ls.len() - 2], ls[ls.len() - 1]);
    let plane = Plane { w: lr_w, h: lr_h, data: lr.iter().cloned().collect() };

    let mut cpu = SrEngine::load("artifacts/espcn/espcn.json", false).unwrap();
    let out_cpu = cpu.upscale_plane(&plane).unwrap();

    let mut npu = SrEngine::load("artifacts/espcn/espcn.json", true).unwrap();
    let out_npu = npu.upscale_plane(&plane).unwrap();

    assert_eq!((out_npu.h, out_npu.w), (out_cpu.h, out_cpu.w));
    let sr_v: Vec<f32> = sr_ref.iter().cloned().collect();
    let rl2_vs_cpu = rel_l2(&out_npu.data, &out_cpu.data);
    let rl2_vs_oracle = rel_l2(&out_npu.data, &sr_v);
    eprintln!("NPU frontier: rel-L2 vs CPU = {rl2_vs_cpu:.3e}, vs ONNX oracle = {rl2_vs_oracle:.3e}");
    assert!(rl2_vs_cpu < 1.1e-2, "NPU vs CPU rel-L2 = {rl2_vs_cpu:.3e} (want < 1.1e-2)");
    assert!(rl2_vs_oracle < 1.5e-2, "NPU vs oracle rel-L2 = {rl2_vs_oracle:.3e} (want < 1.5e-2)");
}
