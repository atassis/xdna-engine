//! M-STATIONARY probe (internal notes CLUE 1 gate): verify a whole-array GEMM xclbin is correct
//! and measure its throughput (GFLOP/s). Generic over xclbin/insts/shape so it drives BOTH the
//! M-stationary kernel and the shipped N-stationary baselines on the same FFN-mm1 shape, for the
//! "M-stationary util within ~2-3x of N-stationary" gate.
//!
//! Host gives ROW-MAJOR A[M,K], B[K,N] bf16; the design's shim DMAs tile them; C comes back
//! row-major f32. Reference is computed from the SAME bf16-truncated inputs in f32, so the rel
//! error reflects only bf16-accumulate (expect ~1e-2), proving the dataflow is numerically correct.
//!
//! NPU single-tenant — stop npu-asr/voxd first. Run from repo root:
//!   mstat_probe <xclbin> <insts.txt> <M> <K> <N>
//!
use std::path::Path;
use std::time::Instant;

use npu_xrt::{Device, FLAG_HOST_ONLY, FLAG_CACHEABLE};

fn f32_to_bf16_bits(x: f32) -> u16 {
    // round-to-nearest-even truncation to bf16
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
        eprintln!("usage: mstat_probe <xclbin> <insts.txt> <M> <K> <N>");
        std::process::exit(2);
    }
    let xclbin = &args[1];
    let insts_path = &args[2];
    let m: usize = args[3].parse().unwrap();
    let k: usize = args[4].parse().unwrap();
    let n: usize = args[5].parse().unwrap();

    let dev = Device::open(0).expect("open NPU (stop npu-asr/voxd first)");
    let kern = dev.load_kernel(xclbin, None).expect("load xclbin");
    println!("[mstat_probe] loaded {}  shape [{m},{k}]x[{k},{n}]", Path::new(xclbin).file_name().unwrap().to_string_lossy());

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
    let bo_c = dev.alloc_bo(&kern, m * n * 4, FLAG_HOST_ONLY, g(5)).unwrap();
    let bo_tmp = dev.alloc_bo(&kern, 1, FLAG_HOST_ONLY, g(6)).unwrap();
    let bo_tr = dev.alloc_bo(&kern, 4, FLAG_HOST_ONLY, g(7)).unwrap();
    bo_a.write_bytes(&a_bytes).unwrap(); bo_a.sync_to_device().unwrap();
    bo_b.write_bytes(&b_bytes).unwrap(); bo_b.sync_to_device().unwrap();

    // --- run once + verify ---
    kern.run_matmul8(3, &instr, n_instr, &bo_a, &bo_b, &bo_c, &bo_tmp, &bo_tr)
        .expect("dispatch failed");
    bo_c.sync_from_device().unwrap();
    let mut c_bytes = vec![0u8; m * n * 4];
    bo_c.read_bytes(&mut c_bytes).unwrap();
    let c: Vec<f32> = c_bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();

    // reference from bf16-truncated inputs (f32 accumulate)
    let af: Vec<f32> = a_bits.iter().map(|&h| bf16_bits_to_f32(h)).collect();
    let bf: Vec<f32> = b_bits.iter().map(|&h| bf16_bits_to_f32(h)).collect();
    let (mut max_abs, mut max_rel, mut ref_max) = (0f32, 0f32, 0f32);
    // spot-check a stride of rows to keep it quick but cover the whole M (all 8 columns' bands)
    let row_step = (m / 256).max(1);
    let mut nchk = 0usize;
    for i in (0..m).step_by(row_step) {
        for j in (0..n).step_by(7) {
            let mut acc = 0f32;
            for kk in 0..k { acc += af[i * k + kk] * bf[kk * n + j]; }
            let got = c[i * n + j];
            let d = (got - acc).abs();
            max_abs = max_abs.max(d);
            ref_max = ref_max.max(acc.abs());
            max_rel = max_rel.max(d / (acc.abs() + 1e-6));
            nchk += 1;
        }
    }
    println!("[mstat_probe] verify ({nchk} elems): max|Δ|={max_abs:.4e}  rel={:.3e}  (ref_max={ref_max:.3})",
             max_abs / (ref_max + 1e-9));

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
    let gflop = 2.0 * m as f64 * k as f64 * n as f64 / 1e9;
    let gflops = gflop / (ms / 1e3);
    println!("[mstat_probe] dispatch {ms:.3} ms/op   {gflop:.3} GFLOP -> {gflops:.1} GFLOP/s");
}
