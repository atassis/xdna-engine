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
use std::time::Instant;

use npu_xrt::{pack_f32_to_bf16, Device, FLAG_CACHEABLE, FLAG_HOST_ONLY};

const EPS: f64 = 1e-5;
const M: usize = 64; // smallest legal M for the whole_array 8-col native-bf16 design

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

fn u16_bytes(v: &[u16]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

/// Device path: drive the resident GEMV xclbin on the FOLDED weight `W''` (host-precomputed) with the
/// host-normalized input row, then host-add the folded `bias'`, and compare to the direct
/// `norm(x)@W + bias` golden. This is the same separable-fold mechanism as `CtxDecode::fused_norm_gemv`,
/// driven directly here so the probe is self-contained. NPU is single-tenant: stop npu-asr/voxd first.
fn device_path(xclbin_p: &str, insts_p: &str, k: usize, n: usize, kind: &str) {
    let n_pad = n.div_ceil(32) * 32;
    println!("[norm_gemv_probe device] {kind}  M={M} K={k} N={n} (N_pad={n_pad})");
    println!("  xclbin: {xclbin_p}");

    // Deterministic inputs (LCG, matching decode_gemv_probe's generator).
    let mut s: u32 = 0x9E37_79B9;
    let mut lcg = |buf: &mut [f64]| {
        for v in buf.iter_mut() {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *v = ((s >> 8) as f64 / u32::MAX as f64) - 0.5;
        }
    };
    let mut x = vec![0f64; k];
    lcg(&mut x);
    let mut gamma = vec![0f64; k];
    lcg(&mut gamma);
    for g in gamma.iter_mut() {
        *g += 1.0;
    }
    let mut beta = vec![0f64; k];
    lcg(&mut beta);
    let mut bias = vec![0f64; n];
    lcg(&mut bias);
    let w: Vec<Vec<f64>> = {
        let mut flat = vec![0f64; k * n];
        lcg(&mut flat);
        (0..k).map(|kk| flat[kk * n..kk * n + n].to_vec()).collect()
    };

    // Host fold + golden + host input-normalize (the separable-fold math).
    let is_ln = kind == "ln";
    let wpp = fold_weight(&w, &gamma, k, n);
    let (golden, biasp, x_norm) = if is_ln {
        let biasp = fold_bias(&w, &beta, &bias, k, n);
        let mean = x.iter().sum::<f64>() / k as f64;
        let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / k as f64;
        let inv = 1.0 / (var + EPS).sqrt();
        let xn: Vec<f64> = (0..k).map(|kk| (x[kk] - mean) * inv).collect();
        (ln_golden(&x, &gamma, &beta, &w, &bias, k, n), biasp, xn)
    } else {
        let ms = x.iter().map(|v| v * v).sum::<f64>() / k as f64;
        let inv = 1.0 / (ms + EPS).sqrt();
        let xn: Vec<f64> = (0..k).map(|kk| x[kk] * inv).collect();
        (rms_golden(&x, &gamma, &w, &bias, k, n), bias.clone(), xn)
    };

    // --- device dispatch: x_norm (row 0, M=64-padded) @ W''[K,N_pad] -> out[64,N_pad] f32 ---
    let dev = Device::open(0).expect("open NPU (stop npu-asr.service/voxd.service first)");
    let kern = dev
        .load_kernel(xclbin_p, None)
        .unwrap_or_else(|e| panic!("load {xclbin_p}: {e}"));
    let ibytes = std::fs::read(insts_p).unwrap_or_else(|e| panic!("read insts {insts_p}: {e}"));
    let n_instr = ibytes.len() / 4;
    let g = |i| kern.group_id(i).unwrap();

    let instr = dev.alloc_bo(&kern, ibytes.len(), FLAG_CACHEABLE, g(1)).unwrap();
    instr.write_bytes(&ibytes).unwrap();
    instr.sync_to_device().unwrap();

    let bo_a = dev.alloc_bo(&kern, M * k * 2, FLAG_HOST_ONLY, g(3)).unwrap();
    let bo_b = dev.alloc_bo(&kern, k * n_pad * 2, FLAG_HOST_ONLY, g(4)).unwrap();
    let bo_c = dev.alloc_bo(&kern, M * n_pad * 4, FLAG_HOST_ONLY, g(5)).unwrap();
    let bo_tmp = dev.alloc_bo(&kern, 1, FLAG_HOST_ONLY, g(6)).unwrap();
    let bo_tr = dev.alloc_bo(&kern, 4, FLAG_HOST_ONLY, g(7)).unwrap();

    // A: x_norm in row 0, rows 1..M zero.
    let mut a_f32 = vec![0f32; M * k];
    for (i, &v) in x_norm.iter().enumerate() {
        a_f32[i] = v as f32;
    }
    let mut a_bf16 = vec![0u16; M * k];
    pack_f32_to_bf16(&a_f32, &mut a_bf16);
    bo_a.write_bytes(u16_bytes(&a_bf16)).unwrap();
    bo_a.sync_to_device().unwrap();

    // B: W''[K, N_pad] bf16 (real columns from wpp, padding zero).
    let mut b_f32 = vec![0f32; k * n_pad];
    for kk in 0..k {
        for nn in 0..n {
            b_f32[kk * n_pad + nn] = wpp[kk][nn] as f32;
        }
    }
    let mut b_bf16 = vec![0u16; k * n_pad];
    pack_f32_to_bf16(&b_f32, &mut b_bf16);
    bo_b.write_bytes(u16_bytes(&b_bf16)).unwrap();
    bo_b.sync_to_device().unwrap();

    let run = || {
        kern.run_matmul8(3, &instr, n_instr, &bo_a, &bo_b, &bo_c, &bo_tmp, &bo_tr)
            .expect("fused GEMV dispatch failed");
    };
    for _ in 0..20 {
        run();
    }
    let iters = 200;
    let t0 = Instant::now();
    for _ in 0..iters {
        run();
    }
    let fused_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;

    bo_c.sync_from_device().unwrap();
    let mut cbuf = vec![0u8; M * n_pad * 4];
    bo_c.read_bytes(&mut cbuf).unwrap();
    let c0: &[f32] = unsafe { std::slice::from_raw_parts(cbuf.as_ptr() as *const f32, n) };

    // Host-add bias' and compare to the direct golden.
    let out: Vec<f64> = (0..n).map(|nn| c0[nn] as f64 + biasp[nn]).collect();
    let mut num = 0f64;
    let mut den = 0f64;
    let mut nan = 0usize;
    for nn in 0..n {
        if !out[nn].is_finite() {
            nan += 1;
        }
        let d = out[nn] - golden[nn];
        num += d * d;
        den += golden[nn] * golden[nn];
    }
    let rel_l2 = (num / den).sqrt();

    println!("\n=== fused {kind} norm+GEMV (device, 1 dispatch) ===");
    println!("  rel-L2 vs norm(x)@W+bias golden : {rel_l2:.4e}  (gate <= 0.08)");
    println!("  NaN/Inf in output               : {nan}/{n}");
    println!("  fused dispatch latency          : {fused_us:.1} us/op ({iters} iters)");
    println!(
        "  out[0..4]={:?}  golden[0..4]={:?}",
        &out[..4.min(n)],
        &golden[..4.min(n)]
    );
    if rel_l2 > 0.08 || nan > 0 {
        eprintln!("[norm_gemv_probe] FAIL: rel-L2 {rel_l2:.4e} or {nan} NaNs");
        std::process::exit(1);
    }
    println!("[norm_gemv_probe] PASS");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() == 2 && args[1] == "selftest" {
        selftest();
        return;
    }
    if args.len() != 7 {
        eprintln!("usage: norm_gemv_probe selftest");
        eprintln!("   or: norm_gemv_probe <xclbin> <insts> <M> <K> <N> <rms|ln>");
        std::process::exit(2);
    }
    let xclbin = &args[1];
    let insts = &args[2];
    let _m: usize = args[3].parse().expect("M");
    let k: usize = args[4].parse().expect("K");
    let n: usize = args[5].parse().expect("N");
    let kind = args[6].as_str();
    if kind != "rms" && kind != "ln" {
        eprintln!("norm kind must be 'rms' or 'ln'");
        std::process::exit(2);
    }
    assert!(Path::new(xclbin).exists(), "xclbin not found: {xclbin}");
    assert!(Path::new(insts).exists(), "insts not found: {insts}");
    device_path(xclbin, insts, k, n, kind);
}
