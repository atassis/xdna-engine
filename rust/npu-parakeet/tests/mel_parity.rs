//! Parity test: `npu_parakeet::mel::compute_features` vs real onnxruntime running
//! `preprocessor.onnx`, golden fixtures captured by `scripts/gen_cpu_glue_fixtures.py`.
//!
//! Tolerance: 1e-3 relative L2. The Rust port computes the STFT via an f64-accumulated naive
//! DFT while onnxruntime's own STFT kernel and downstream ops run entirely in f32 with a
//! different (and unspecified) summation order; that alone is enough to separate two otherwise
//! node-for-node-identical pipelines by more than f32 epsilon. Measured achieved rel-L2 on both
//! fixtures is printed by the test and is ~1e-5, so 1e-3 leaves >100x headroom -- loose enough to
//! not be a source of test flakiness, tight enough that a real math bug (wrong FFT bin, wrong
//! window, wrong normalization) would still fail it by orders of magnitude.
use ndarray::{Array1, Array2};
use ndarray_npy::read_npy;
use npu_parakeet::mel::compute_features;

fn fixture(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/cpu_glue/mel").join(name)
}

fn rel_l2(a: &Array2<f32>, b: &Array2<f32>) -> f64 {
    assert_eq!(a.shape(), b.shape());
    let mut num = 0f64;
    let mut den = 0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (*x as f64) - (*y as f64);
        num += d * d;
        den += (*y as f64) * (*y as f64);
    }
    if den == 0.0 {
        num.sqrt()
    } else {
        (num / den).sqrt()
    }
}

fn check(case: &str) {
    let wav: Array1<f32> = read_npy(fixture(&format!("{case}_wav.npy"))).unwrap();
    let expected: Array2<f32> = read_npy(fixture(&format!("{case}_features.npy"))).unwrap();
    let got = compute_features(wav.as_slice().unwrap());
    assert_eq!(got.shape(), expected.shape(), "{case}: shape mismatch");
    let err = rel_l2(&got, &expected);
    eprintln!("[mel_parity] {case}: rel-L2 = {err:.3e} (shape {:?})", got.shape());
    assert!(err < 1e-3, "{case}: rel-L2 {err:.3e} exceeds 1e-3 tolerance");
}

#[test]
fn synthetic_sweep_matches_onnxruntime() {
    check("synthetic_sweep");
}

#[test]
fn real_clip_matches_onnxruntime() {
    check("real_clip_1s");
}
