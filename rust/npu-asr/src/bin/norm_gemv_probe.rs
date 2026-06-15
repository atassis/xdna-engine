//! Fused Norm+GEMV decode primitive probe (Phase: decode-norm-gemv).
//!
//! Semantics (one dispatch, decode M=1, M=64-padded): out[N] = norm(x[K]) @ W[K,N] + bias[N],
//! norm in {RMSNorm(γ,eps), LayerNorm(γ,β,eps)}. The kernel runs on the FOLDED weight:
//!   W'' = diag(γ)·W ;  bias' = β@W + bias
//!   RMS: out = inv_rms·(x @ W'') + bias       inv_rms = 1/sqrt(mean(x²)+eps)
//!   LN : out = inv_std·((x-mean) @ W'') + bias'  mean=Σx/K, inv_std=1/sqrt(mean((x-mean)²)+eps)
//! `register_fused` (host) precomputes W''/bias'; this probe does the same fold + a torch-equivalent
//! golden, and (Task 2+) drives the xclbin via run_matmul8 and verifies.
//!
//! Task 1 entry: `norm_gemv_probe selftest` — pure CPU, proves the fold identity in f64 (no NPU).
//! Device path (Task 2+): `norm_gemv_probe <xclbin> <insts> <M> <K> <N> <rms|ln>`.
use std::path::Path;

const EPS: f64 = 1e-5;

// ---- fold + goldens (generic over f64 for the selftest; the device path reuses the fold in f32) ----

/// W''[k][n] = γ[k]·W[k][n]
fn fold_weight(w: &[Vec<f64>], gamma: &[f64], k: usize, n: usize) -> Vec<Vec<f64>> {
    (0..k).map(|kk| (0..n).map(|nn| gamma[kk] * w[kk][nn]).collect()).collect()
}
/// bias'[n] = Σ_k β[k]·W[k][n] + bias[n]
fn fold_bias(w: &[Vec<f64>], beta: &[f64], bias: &[f64], k: usize, n: usize) -> Vec<f64> {
    (0..n).map(|nn| (0..k).map(|kk| beta[kk] * w[kk][nn]).sum::<f64>() + bias[nn]).collect()
}
fn matvec(x: &[f64], w: &[Vec<f64>], k: usize, n: usize) -> Vec<f64> {
    (0..n).map(|nn| (0..k).map(|kk| x[kk] * w[kk][nn]).sum::<f64>()).collect()
}

/// Direct RMS golden: out = (x·inv_rms·γ) @ W + bias
fn rms_golden(x: &[f64], gamma: &[f64], w: &[Vec<f64>], bias: &[f64], k: usize, n: usize) -> Vec<f64> {
    let ms = x.iter().map(|v| v * v).sum::<f64>() / k as f64;
    let inv = 1.0 / (ms + EPS).sqrt();
    let xn: Vec<f64> = (0..k).map(|kk| x[kk] * inv * gamma[kk]).collect();
    let mut o = matvec(&xn, w, k, n);
    for nn in 0..n { o[nn] += bias[nn]; }
    o
}
/// Folded RMS: out = inv_rms·(x @ W'') + bias
fn rms_folded(x: &[f64], wpp: &[Vec<f64>], bias: &[f64], k: usize, n: usize) -> Vec<f64> {
    let ms = x.iter().map(|v| v * v).sum::<f64>() / k as f64;
    let inv = 1.0 / (ms + EPS).sqrt();
    let xw = matvec(x, wpp, k, n);
    (0..n).map(|nn| inv * xw[nn] + bias[nn]).collect()
}

/// Direct LN golden: out = ((x-mean)·inv_std·γ + β) @ W + bias
fn ln_golden(x: &[f64], gamma: &[f64], beta: &[f64], w: &[Vec<f64>], bias: &[f64], k: usize, n: usize) -> Vec<f64> {
    let mean = x.iter().sum::<f64>() / k as f64;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / k as f64;
    let inv = 1.0 / (var + EPS).sqrt();
    let xn: Vec<f64> = (0..k).map(|kk| (x[kk] - mean) * inv * gamma[kk] + beta[kk]).collect();
    let mut o = matvec(&xn, w, k, n);
    for nn in 0..n { o[nn] += bias[nn]; }
    o
}
/// Folded LN: out = inv_std·((x-mean) @ W'') + bias'
fn ln_folded(x: &[f64], wpp: &[Vec<f64>], biasp: &[f64], k: usize, n: usize) -> Vec<f64> {
    let mean = x.iter().sum::<f64>() / k as f64;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / k as f64;
    let inv = 1.0 / (var + EPS).sqrt();
    let xc: Vec<f64> = (0..k).map(|kk| x[kk] - mean).collect();
    let xw = matvec(&xc, wpp, k, n);
    (0..n).map(|nn| inv * xw[nn] + biasp[nn]).collect()
}

fn maxabs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}

fn selftest() {
    let (k, n) = (8usize, 4usize);
    // deterministic small inputs
    let x: Vec<f64> = (0..k).map(|i| ((i * 7 % 11) as f64 - 5.0) * 0.3).collect();
    let gamma: Vec<f64> = (0..k).map(|i| 1.0 + 0.1 * (i as f64)).collect();
    let beta: Vec<f64> = (0..k).map(|i| 0.05 * (i as f64 - 3.0)).collect();
    let bias: Vec<f64> = (0..n).map(|i| 0.2 * (i as f64 - 1.0)).collect();
    let w: Vec<Vec<f64>> = (0..k).map(|kk| (0..n).map(|nn| (((kk * 5 + nn * 3) % 13) as f64 - 6.0) * 0.1).collect()).collect();

    let wpp = fold_weight(&w, &gamma, k, n);
    let biasp = fold_bias(&w, &beta, &bias, k, n);

    let d_rms = maxabs_diff(&rms_golden(&x, &gamma, &w, &bias, k, n), &rms_folded(&x, &wpp, &bias, k, n));
    let d_ln = maxabs_diff(&ln_golden(&x, &gamma, &beta, &w, &bias, k, n), &ln_folded(&x, &wpp, &biasp, k, n));
    println!("[norm_gemv_probe selftest] RMS fold max|Δ|={d_rms:.3e}   LN fold max|Δ|={d_ln:.3e}");
    assert!(d_rms < 1e-9, "RMS fold identity broken");
    assert!(d_ln < 1e-9, "LN fold identity broken");
    println!("[norm_gemv_probe selftest] fold OK");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() == 2 && args[1] == "selftest" {
        selftest();
        return;
    }
    if args.len() != 7 {
        eprintln!("usage: norm_gemv_probe selftest");
        eprintln!("   or: norm_gemv_probe <xclbin> <insts> <M> <K> <N> <rms|ln>   (device path: Task 2+)");
        std::process::exit(2);
    }
    // Device path implemented in Task 2+ (plain GEMV baseline → RMS → LN).
    let _ = Path::new(&args[1]);
    eprintln!("[norm_gemv_probe] device path lands in Task 2; run `selftest` for the CPU fold check.");
    std::process::exit(3);
}
