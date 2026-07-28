//! WS-3 Phase-0 sanity: does AVX-512 have FP throughput headroom over AVX2 on this Zen5-mobile
//! part, or is it double-pumped (same FLOP/s, fewer instructions)? Pure register-resident FMA
//! throughput, no memory. If AVX-512 ~= AVX2 GFLOP/s, a custom AVX-512 mha kernel cannot beat
//! `matrixmultiply` (already ~80% of AVX2 peak) and WS-3 walls here.
//!
//!   cargo run --release -p npu-asr-host --bin fma_peak
#![allow(unused)]
use std::hint::black_box;
use std::time::Instant;

#[cfg(target_arch = "x86_64")]
mod bench {
    use super::*;
    use std::arch::x86_64::*;

    // 12 independent accumulator chains to hide FMA latency (~4-5 cyc) across 2 FMA pipes.
    const NACC: usize = 12;
    const ITERS: u64 = 200_000_000;

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn avx2() -> f32 {
        let a = _mm256_set1_ps(1.0000001);
        let b = _mm256_set1_ps(0.9999999);
        let mut acc = [_mm256_set1_ps(0.1); NACC];
        for _ in 0..ITERS {
            for j in 0..NACC {
                acc[j] = _mm256_fmadd_ps(acc[j], a, b);
            }
        }
        let mut s = _mm256_setzero_ps();
        for j in 0..NACC { s = _mm256_add_ps(s, acc[j]); }
        let mut out = [0f32; 8];
        _mm256_storeu_ps(out.as_mut_ptr(), s);
        out.iter().sum()
    }

    #[target_feature(enable = "avx512f")]
    pub unsafe fn avx512() -> f32 {
        let a = _mm512_set1_ps(1.0000001);
        let b = _mm512_set1_ps(0.9999999);
        let mut acc = [_mm512_set1_ps(0.1); NACC];
        for _ in 0..ITERS {
            for j in 0..NACC {
                acc[j] = _mm512_fmadd_ps(acc[j], a, b);
            }
        }
        let mut s = _mm512_setzero_ps();
        for j in 0..NACC { s = _mm512_add_ps(s, acc[j]); }
        let mut out = [0f32; 16];
        _mm512_storeu_ps(out.as_mut_ptr(), s);
        out.iter().sum()
    }

    pub fn run() {
        // flops per call: ITERS * NACC * lanes * 2 (fma = mul+add)
        let warm = unsafe { avx2() }; black_box(warm);
        let mut best2 = f64::INFINITY;
        for _ in 0..5 {
            let t = Instant::now();
            black_box(unsafe { avx2() });
            best2 = best2.min(t.elapsed().as_secs_f64());
        }
        let g2 = (ITERS * NACC as u64 * 8 * 2) as f64 / best2 / 1e9;

        if is_x86_feature_detected!("avx512f") {
            let warm = unsafe { avx512() }; black_box(warm);
            let mut best5 = f64::INFINITY;
            for _ in 0..5 {
                let t = Instant::now();
                black_box(unsafe { avx512() });
                best5 = best5.min(t.elapsed().as_secs_f64());
            }
            let g5 = (ITERS * NACC as u64 * 16 * 2) as f64 / best5 / 1e9;
            println!("single-core peak FMA throughput:");
            println!("  AVX2   (256b):  {g2:7.1} GFLOP/s");
            println!("  AVX512 (512b):  {g5:7.1} GFLOP/s   ({:.2}x AVX2)", g5 / g2);
            println!();
            if g5 / g2 < 1.25 {
                println!("=> ~equal: AVX-512 is double-pumped (no FP headroom). A custom AVX-512 mha");
                println!("   kernel cannot beat matrixmultiply's ~80% of AVX2 peak. WS-3 WALLS.");
            } else {
                println!("=> AVX-512 has real FP headroom ({:.2}x). A custom kernel could clear the", g5/g2);
                println!("   65 GFLOP/s baseline -> WS-3 worth a real microkernel bench at the mha shapes.");
            }
        } else {
            println!("AVX2 peak: {g2:.1} GFLOP/s (no AVX-512 on this part)");
        }
    }
}

fn main() {
    #[cfg(target_arch = "x86_64")]
    bench::run();
    #[cfg(not(target_arch = "x86_64"))]
    println!("x86_64 only");
}
