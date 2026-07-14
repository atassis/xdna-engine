//! Device probe for the resident LN->fc1 seam (resident-rails Task 3, first frontier advance).
//!
//! Verifies `NpuMatmul::resident_ff1_fc1` (on-chip normalize-only LN + f32->bf16 cast chained
//! device-side -> modal fc1 with the gamma-folded weight) reproduces the host affine-LN then fc1:
//!   host:  (norm(x)*gamma + beta) @ W1
//!   dev:   resident_ff1_fc1(x, gamma.W1) + (beta @ W1)          [affine folded into W1/bias, exact]
//! Gate: bf16-class rel-err <= 1e-2. Single-tenant NPU.
//!
//! Usage:  resident_ff1_probe [--t 32] [--blk 0]   (needs --features npu; run from repo root)
#![cfg(feature = "npu")]
use std::path::Path;

use ndarray::prelude::*;
use npu_parakeet::config::ModelCfg;
use npu_parakeet::npu::NpuMatmul;
use npu_parakeet::weights::ParakeetWeights;

fn rand_mat(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    Array2::from_shape_fn((rows, cols), |_| {
        s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        ((z >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    })
}

fn norm_only(x: &Array2<f32>) -> Array2<f32> {
    // f32 two-pass centered normalize-only (ln_2pass.cc math, eps=1e-5)
    let (t, d) = x.dim();
    let mut out = Array2::<f32>::zeros((t, d));
    for i in 0..t {
        let row = x.row(i);
        let mean = row.sum() / d as f32;
        let var = row.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
        let inv = 1.0 / (var + 1e-5).sqrt();
        for j in 0..d {
            out[[i, j]] = (row[j] - mean) * inv;
        }
    }
    out
}

fn rel_err(a: &Array2<f32>, b: &Array2<f32>) -> f32 {
    let mut mx = 0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let r = (x - y).abs() / x.abs().max(1e-3);
        if r > mx { mx = r; }
    }
    mx
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let av = |f: &str, d: &str| args.iter().position(|a| a == f).and_then(|i| args.get(i + 1)).cloned().unwrap_or_else(|| d.into());
    let t: usize = av("--t", "32").parse().unwrap();
    let blk: usize = av("--blk", "0").parse().unwrap();
    let cfg = ModelCfg::PARAKEET_V3;
    let (d, f) = (cfg.hidden, cfg.ff); // 1024, 4096

    let artifacts = Path::new("artifacts/parakeet/encoder");
    let w = ParakeetWeights::load(artifacts).expect("weights");
    let b = w.block(blk);
    let gamma = b.v("norm_feed_forward1.weight"); // [D]
    let beta = b.v("norm_feed_forward1.bias");     // [D]
    let w1 = b.m("feed_forward1.linear1.weight");  // [K=D, N=F] (mm layout)
    assert_eq!(w1.dim(), (d, f));

    let x = rand_mat(t, d, 1);

    // host reference: affine LN then fc1
    let xn = norm_only(&x);
    let mut aff = xn.clone();
    for i in 0..t { for j in 0..d { aff[[i, j]] = aff[[i, j]] * gamma[j] + beta[j]; } }
    let host = aff.dot(&w1); // [t, F]

    // gamma-folded weight + beta bias
    let mut w1f = w1.clone();
    for i in 0..d { for j in 0..f { w1f[[i, j]] *= gamma[i]; } } // scale K rows by gamma
    let beta_bias = beta.dot(&w1); // [F]

    let root = std::env::var("NPU_XCLBIN_ROOT").unwrap_or_else(|_| ".".into());
    let npu = NpuMatmul::open(Path::new(&root));
    let raw = npu.resident_ff1_fc1(&x, || w1f.clone(), &format!("{blk}.ff1.l1f"), f); // norm(x)@(gamma.W1)
    let mut dev = raw.clone();
    for i in 0..t { for j in 0..f { dev[[i, j]] += beta_bias[j]; } }

    let re = rel_err(&host, &dev);
    println!("[resident_ff1] t={t} blk={blk}  fc1 (LN->cast->modal, device-side) rel_err={re:.3e}");
    println!("[resident_ff1] host[0,:3]={:?}  dev[0,:3]={:?}", &host.row(0).to_vec()[..3], &dev.row(0).to_vec()[..3]);
    assert!(re <= 1e-2, "resident FF1 fc1 FAILED: rel_err {re:.3e} > 1e-2 (bf16 gate)");
    println!("[resident_ff1] PASS (<= 1e-2) -- LN->fc1 seam runs device-side, WER-class accurate");
}
