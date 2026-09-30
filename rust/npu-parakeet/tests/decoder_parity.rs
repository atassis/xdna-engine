//! Parity test: `TdtDecoder` (host Rust prednet + joint, weights read straight out of
//! `decoder_joint.onnx` via `onnx_init.rs`, no onnxruntime) vs real onnxruntime running the same
//! graph over a fixed synthetic trajectory, captured by `scripts/gen_cpu_glue_fixtures.py`.
//!
//! Requires the real model file at `artifacts/parakeet/decoder_joint.onnx` (or $XDNA_ARTIFACTS,
//! gitignored) -- skips with a message if absent instead of failing, since it is not a repo
//! artifact.
//!
//! Tolerance: 1e-4 relative L2 per step. This is the SAME arithmetic (embedding lookup, 2-layer
//! LSTM, two linear projections, ReLU, one more linear) in f32 on both sides, so the only
//! divergence source is floating-point summation order -- no algorithmic difference like the mel
//! test's independent DFT. `decoder.rs`'s own module doc records ~5e-7 measured previously
//! (`scripts/parakeet_tdt_decoder_ref.py`'s numpy reference vs this same oracle); 1e-4 keeps that
//! headroom while not being so tight that a legitimate summation-order difference on a new
//! machine/BLAS trips it.
use ndarray::{Array1, Array2};
use ndarray_npy::read_npy;
use npu_parakeet::decoder::{PredState, TdtDecoder, BLANK_IDX};

fn fixture(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/cpu_glue/decoder")
        .join(name)
}

#[test]
fn matches_onnxruntime_over_synthetic_trajectory() {
    let onnx_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../artifacts/parakeet/decoder_joint.onnx");
    if !onnx_path.exists() {
        eprintln!("[decoder_parity] SKIP: {} not present (gitignored model artifact)", onnx_path.display());
        return;
    }

    let decoder = TdtDecoder::load_from_onnx(&onnx_path).expect("load_from_onnx");
    let encodings: Array2<f32> = read_npy(fixture("encodings.npy")).unwrap();
    let targets: Array1<i64> = read_npy(fixture("targets.npy")).unwrap();
    let golden: Array2<f32> = read_npy(fixture("golden_logits.npy")).unwrap();
    let n_steps = encodings.nrows();
    assert_eq!(n_steps, targets.len());
    assert_eq!(n_steps, golden.nrows());

    // Fixture generation always primes step 0 with BLANK_IDX (see gen_cpu_glue_fixtures.py).
    assert_eq!(targets[0] as usize, BLANK_IDX);

    let mut state = PredState::zeros();
    let mut max_rel = 0f64;
    let mut sum_rel = 0f64;
    for t in 0..n_steps {
        let (pred_u, new_state) = decoder.prednet_step(targets[t] as usize, &state);
        let logits = decoder.joint(encodings.row(t), &pred_u);
        state = new_state;

        let mut num = 0f64;
        let mut den = 0f64;
        for (a, b) in logits.iter().zip(golden.row(t).iter()) {
            let d = (*a as f64) - (*b as f64);
            num += d * d;
            den += (*b as f64) * (*b as f64);
        }
        let rel = if den == 0.0 { num.sqrt() } else { (num / den).sqrt() };
        max_rel = max_rel.max(rel);
        sum_rel += rel;
    }
    let mean_rel = sum_rel / n_steps as f64;
    eprintln!("[decoder_parity] {n_steps} steps: max rel-L2 = {max_rel:.3e}, mean rel-L2 = {mean_rel:.3e}");
    assert!(max_rel < 1e-4, "max rel-L2 {max_rel:.3e} exceeds 1e-4 tolerance");
}
