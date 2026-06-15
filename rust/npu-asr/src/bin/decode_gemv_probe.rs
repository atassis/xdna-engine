//! Decode-GEMV de-risk probe (Task 0): measure the cost of ONE thin-M GEMV dispatch on THIS NPU.
//!
//! Decode is M=1 (a single token). The resident encoder GEMM is tuned for M=512. This probe loads
//! the smallest-legal-M whole_array xclbin (M=64, tile 8x32x32, 8 cols, native bf16->f32; built by
//! `scripts/build_decode_kernels.sh`), pads the single query row up to M=64, and times the dispatch
//! over warmup=20 / iters=200 to report the avg microseconds per GEMV.
//!
//! The matmul host ABI (see ctx_ln.rs / dispatch_spike.rs / ctx2.rs:660): kernel args are
//!   1=instr (CACHEABLE), 3=A=activation bf16, 4=B=weight bf16, 5=C=output f32, 6=tmp, 7=trace.
//!   run_matmul8(opcode=3, instr, n_instr, A, B, C, tmp, trace).
//!
//! NPU is single-tenant — stop npu-asr.service / voxd.service BEFORE running, restart AFTER.
//! Run from the repo root (paths are relative to ".").
//!
//! Usage:  decode_gemv_probe [K] [N]   (defaults K=768 N=768)

use std::path::Path;
use std::time::Instant;

use npu_xrt::{pack_f32_to_bf16, Device, FLAG_CACHEABLE, FLAG_HOST_ONLY};

const WA: &str = "mlir-aie/programming_examples/basic/matrix_multiplication/whole_array/build";

/// Smallest legal M for the native-bf16 8-col whole_array design (see build_decode_kernels.sh).
const M: usize = 64;
const TILE: &str = "8x32x32"; // m x k x n
const COLS: usize = 8;

const WARMUP: usize = 20;
const ITERS: usize = 200;

fn u16_bytes(v: &[u16]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let k: usize = args.next().map(|s| s.parse().expect("K must be a usize")).unwrap_or(768);
    let n: usize = args.next().map(|s| s.parse().expect("N must be a usize")).unwrap_or(768);

    let root = Path::new(".");
    let wa = root.join(WA);
    let stem = format!("{M}x{k}x{n}_{TILE}_{COLS}c");
    let xclbin = wa.join(format!("final_{stem}.xclbin"));
    let insts = wa.join(format!("insts_{stem}.txt"));

    println!("[decode_gemv_probe] M={M} K={k} N={n} (tile {TILE}, {COLS} cols, native bf16->f32)");
    println!("  xclbin: {}", xclbin.display());

    let dev = Device::open(0).expect("open NPU (stop npu-asr.service/voxd.service first)");
    let kern = dev
        .load_kernel(xclbin.to_str().unwrap(), None)
        .unwrap_or_else(|e| panic!("load {}: {e}", xclbin.display()));

    let ibytes = std::fs::read(&insts).unwrap_or_else(|e| panic!("read insts {}: {e}", insts.display()));
    let n_instr = ibytes.len() / 4;
    let g = |i| kern.group_id(i).unwrap();

    // instr BO (cacheable). data BOs (host-only): A=[M,K] bf16, B=[K,N] bf16, C=[M,N] f32.
    let instr = dev.alloc_bo(&kern, ibytes.len(), FLAG_CACHEABLE, g(1)).unwrap();
    instr.write_bytes(&ibytes).unwrap();
    instr.sync_to_device().unwrap();

    let bo_a = dev.alloc_bo(&kern, M * k * 2, FLAG_HOST_ONLY, g(3)).unwrap();
    let bo_b = dev.alloc_bo(&kern, k * n * 2, FLAG_HOST_ONLY, g(4)).unwrap();
    let bo_c = dev.alloc_bo(&kern, M * n * 4, FLAG_HOST_ONLY, g(5)).unwrap();
    let bo_tmp = dev.alloc_bo(&kern, 1, FLAG_HOST_ONLY, g(6)).unwrap();
    let bo_tr = dev.alloc_bo(&kern, 4, FLAG_HOST_ONLY, g(7)).unwrap();

    // Activation A: the single decode query in row 0 (pseudo-random), rows 1..M zero-padded.
    let mut a_f32 = vec![0f32; M * k];
    let mut s: u32 = 0x9E37_79B9;
    for v in a_f32[..k].iter_mut() {
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *v = ((s >> 8) as f32 / u32::MAX as f32) - 0.5; // ~[-0.5, 0.5)
    }
    let mut a_bf16 = vec![0u16; M * k];
    pack_f32_to_bf16(&a_f32, &mut a_bf16);
    bo_a.write_bytes(u16_bytes(&a_bf16)).unwrap();
    bo_a.sync_to_device().unwrap();

    // Weight B: pseudo-random bf16 weights.
    let mut b_f32 = vec![0f32; k * n];
    for v in b_f32.iter_mut() {
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *v = ((s >> 8) as f32 / u32::MAX as f32) - 0.5;
    }
    let mut b_bf16 = vec![0u16; k * n];
    pack_f32_to_bf16(&b_f32, &mut b_bf16);
    bo_b.write_bytes(u16_bytes(&b_bf16)).unwrap();
    bo_b.sync_to_device().unwrap();

    let run = || {
        kern.run_matmul8(3, &instr, n_instr, &bo_a, &bo_b, &bo_c, &bo_tmp, &bo_tr)
            .expect("GEMV dispatch failed");
    };

    // warmup (first dispatch loads the array program; XRT/JIT settles)
    for _ in 0..WARMUP {
        run();
    }

    // timed: pure dispatch latency (run_matmul8 = submit+exec+wait), inputs resident on device.
    let t0 = Instant::now();
    for _ in 0..ITERS {
        run();
    }
    let total = t0.elapsed();
    let avg_us = total.as_secs_f64() * 1e6 / ITERS as f64;

    // sanity: read back row 0, confirm the GEMV produced finite, non-zero output.
    bo_c.sync_from_device().unwrap();
    let mut cbuf = vec![0u8; M * n * 4];
    bo_c.read_bytes(&mut cbuf).unwrap();
    let c0: &[f32] = unsafe { std::slice::from_raw_parts(cbuf.as_ptr() as *const f32, n) };
    let nz = c0.iter().filter(|&&x| x != 0.0).count();
    let all_finite = c0.iter().all(|x| x.is_finite());

    println!("\n=== decode GEMV dispatch latency (warmup={WARMUP}, iters={ITERS}) ===");
    println!("  avg per GEMV dispatch : {avg_us:.1} us  (run_matmul8 submit+exec+wait, inputs resident)");
    println!("  total {ITERS} dispatches: {:.2} ms", total.as_secs_f64() * 1e3);
    println!(
        "  output row0 sanity    : {nz}/{n} non-zero, all_finite={all_finite}, c0[0..4]={:?}",
        &c0[..4.min(n)]
    );
}
