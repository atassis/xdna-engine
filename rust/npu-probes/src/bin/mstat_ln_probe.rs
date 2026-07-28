//! M-STATIONARY + FUSED LAYERNORM probe (Phase 1.2 spike). Verifies a fused
//! GEMM->LayerNorm xclbin (one dispatch) against a host GEMM->LayerNorm golden,
//! and measures dispatch latency. Single-block (n == N) variant.
//!
//! The kernel computes, per output row: C = A@B (f32 acc), then NORMALIZE-ONLY
//! two-pass LayerNorm over the full row of N (gamma=1/beta=0), stored bf16:
//!   mean = Σx/N;  var = Σ(x-mean)²/N;  out = (x-mean)/sqrt(var+1e-5)
//! Golden uses the SAME bf16-truncated inputs in f32; tolerance is bf16-output
//! level (rel ~1e-2, gate <= 0.08).
//!
//! NPU single-tenant — stop npu-asr/voxd first. Run from repo root:
//!   mstat_ln_probe <xclbin> <insts.txt> <M> <K> <N>
//!
use std::path::Path;
use std::time::Instant;

use npu_xrt::{Device, FLAG_HOST_ONLY, FLAG_CACHEABLE};

fn f32_to_bf16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let rounding = 0x7fff + ((b >> 16) & 1);
    ((b.wrapping_add(rounding)) >> 16) as u16
}
fn bf16_bits_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 6 {
        eprintln!("usage: mstat_ln_probe <xclbin> <insts.txt> <M> <K> <N>");
        std::process::exit(2);
    }
    let xclbin = &args[1];
    let insts_path = &args[2];
    let m: usize = args[3].parse().unwrap();
    let k: usize = args[4].parse().unwrap();
    let n: usize = args[5].parse().unwrap();

    let dev = Device::open(0).expect("open NPU (stop npu-asr/voxd first)");
    let kern = dev.load_kernel(xclbin, None).expect("load xclbin");
    println!("[mstat_ln_probe] loaded {}  shape [{m},{k}]x[{k},{n}] -> LN", Path::new(xclbin).file_name().unwrap().to_string_lossy());

    let instr_bytes = std::fs::read(insts_path).unwrap_or_else(|e| panic!("read insts {insts_path}: {e}"));
    let n_instr = instr_bytes.len() / 4;
    let g = |i| kern.group_id(i).unwrap();
    let instr = dev.alloc_bo(&kern, instr_bytes.len(), FLAG_CACHEABLE, g(1)).unwrap();
    instr.write_bytes(&instr_bytes).unwrap();
    instr.sync_to_device().unwrap();

    // --- inputs: row-major A[m,k], B[k,n] bf16 from a deterministic generator ---
    let mut a_bits = vec![0u16; m * k];
    let mut b_bits = vec![0u16; k * n];
    for i in 0..m {
        for j in 0..k {
            let v = (((i * 7 + j * 3) % 17) as f32 - 8.0) * 0.05;
            a_bits[i * k + j] = f32_to_bf16_bits(v);
        }
    }
    for i in 0..k {
        for j in 0..n {
            let v = (((i * 5 + j * 11) % 13) as f32 - 6.0) * 0.04;
            b_bits[i * n + j] = f32_to_bf16_bits(v);
        }
    }
    let a_bytes: Vec<u8> = a_bits.iter().flat_map(|h| h.to_le_bytes()).collect();
    let b_bytes: Vec<u8> = b_bits.iter().flat_map(|h| h.to_le_bytes()).collect();

    let bo_a = dev.alloc_bo(&kern, m * k * 2, FLAG_HOST_ONLY, g(3)).unwrap();
    let bo_b = dev.alloc_bo(&kern, k * n * 2, FLAG_HOST_ONLY, g(4)).unwrap();
    let bo_c = dev.alloc_bo(&kern, m * n * 2, FLAG_HOST_ONLY, g(5)).unwrap();  // bf16 output
    let bo_tmp = dev.alloc_bo(&kern, 1, FLAG_HOST_ONLY, g(6)).unwrap();
    let bo_tr = dev.alloc_bo(&kern, 4, FLAG_HOST_ONLY, g(7)).unwrap();
    bo_a.write_bytes(&a_bytes).unwrap(); bo_a.sync_to_device().unwrap();
    bo_b.write_bytes(&b_bytes).unwrap(); bo_b.sync_to_device().unwrap();

    // --- run once + verify ---
    kern.run_matmul8(3, &instr, n_instr, &bo_a, &bo_b, &bo_c, &bo_tmp, &bo_tr)
        .expect("dispatch failed");
    bo_c.sync_from_device().unwrap();
    let mut c_bytes = vec![0u8; m * n * 2];
    bo_c.read_bytes(&mut c_bytes).unwrap();
    let c: Vec<f32> = c_bytes.chunks_exact(2).map(|b| bf16_bits_to_f32(u16::from_le_bytes([b[0], b[1]]))).collect();

    // golden: GEMM (f32, from bf16-truncated inputs) then two-pass LayerNorm over the FULL row.
    let af: Vec<f32> = a_bits.iter().map(|&h| bf16_bits_to_f32(h)).collect();
    let bf: Vec<f32> = b_bits.iter().map(|&h| bf16_bits_to_f32(h)).collect();
    let (mut max_abs, mut max_rel, mut ref_max) = (0f32, 0f32, 0f32);
    let mut nan = 0usize;
    let row_step = (m / 64).max(1);  // spot-check rows across all 8 M-bands; full row each (LN needs it)
    let mut nchk = 0usize;
    for i in (0..m).step_by(row_step) {
        // full GEMM row in f32
        let mut row = vec![0f32; n];
        for j in 0..n {
            let mut acc = 0f32;
            for kk in 0..k { acc += af[i * k + kk] * bf[kk * n + j]; }
            row[j] = acc;
        }
        let mean: f32 = row.iter().sum::<f32>() / n as f32;
        let var: f32 = row.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n as f32;
        let inv = 1.0f32 / (var + 1e-5f32).sqrt();
        for j in (0..n).step_by(3) {
            let gold = (row[j] - mean) * inv;
            let got = c[i * n + j];
            if got.is_nan() { nan += 1; continue; }
            let d = (got - gold).abs();
            max_abs = max_abs.max(d);
            ref_max = ref_max.max(gold.abs());
            max_rel = max_rel.max(d / (gold.abs() + 1e-3));
            nchk += 1;
        }
    }
    println!("[mstat_ln_probe] verify ({nchk} elems): max|Δ|={max_abs:.4e}  max_rel={max_rel:.3e}  NaN={nan}  (ref_max={ref_max:.3})");

    // --- warm + time ---
    for _ in 0..10 {
        kern.run_matmul8(3, &instr, n_instr, &bo_a, &bo_b, &bo_c, &bo_tmp, &bo_tr).unwrap();
    }
    let iters = 50;
    let t = Instant::now();
    for _ in 0..iters {
        kern.run_matmul8(3, &instr, n_instr, &bo_a, &bo_b, &bo_c, &bo_tmp, &bo_tr).unwrap();
    }
    let ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;
    let gflop = 2.0 * m as f64 * k as f64 * n as f64 / 1e9;  // GEMM flops (LN negligible)
    let gflops = gflop / (ms / 1e3);
    println!("[mstat_ln_probe] dispatch {ms:.3} ms/op (GEMM+LN, 1 dispatch)   {gflop:.3} GFLOP -> {gflops:.1} GFLOP/s");
}
