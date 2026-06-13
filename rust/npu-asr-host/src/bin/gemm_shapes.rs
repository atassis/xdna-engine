//! WS-3 Phase-0: matrixmultiply (ndarray `.dot`) efficiency at the EXACT mha matmul shapes,
//! single-core (taskset cpu0 -> ~5 GHz, no all-core throttle). Context: single-core FMA peak is
//! ~161 GFLOP/s and AVX-512==AVX2 (double-pumped, see fma_peak). If `.dot` already runs these
//! shapes near peak, there is no efficiency room for a hand kernel and WS-3 walls.
//!
//!   taskset -c 0 cargo run --release -p npu-asr-host --bin gemm_shapes
use std::hint::black_box;
use std::time::Instant;
use ndarray::prelude::*;

const T: usize = 400;
const HD: usize = 48;
const TILE: usize = 64; // mha row-tile

fn fill(r: usize, c: usize, s: f32) -> Array2<f32> {
    Array2::from_shape_fn((r, c), |(i, j)| ((i as f32 * 0.013 + j as f32 * 0.007 + s).sin()))
}

fn bench(name: &str, flops: f64, iters: usize, mut f: impl FnMut()) {
    for _ in 0..5 { f(); }
    let mut best = f64::INFINITY;
    for _ in 0..iters {
        let t = Instant::now();
        f();
        best = best.min(t.elapsed().as_secs_f64());
    }
    let g = flops / best / 1e9;
    println!("  {name:<28} {:8.4} ms   {g:7.1} GFLOP/s   ({:4.0}% of 161 peak)",
        best * 1e3, 100.0 * g / 161.0);
}

fn main() {
    println!("mha matmul shapes, single-core, matrixmultiply via ndarray .dot:");
    // scores: qh[TILE,HD] . kh[T,HD]^T -> [TILE,T]    M=64 N=400 K=48
    let qh = fill(TILE, HD, 0.0);
    let kh = fill(T, HD, 1.0);
    let f_scores = 2.0 * TILE as f64 * T as f64 * HD as f64;
    bench("scores qh@kh^T (K=48)", f_scores, 2000, || {
        black_box(qh.dot(&kh.t()));
    });
    // ctxV: sc[TILE,T] . vh[T,HD] -> [TILE,HD]         M=64 N=48 K=400
    let sc = fill(TILE, T, 2.0);
    let vh = fill(T, HD, 3.0);
    let f_ctxv = 2.0 * TILE as f64 * HD as f64 * T as f64;
    bench("ctxV  sc@vh   (K=400)", f_ctxv, 2000, || {
        black_box(sc.dot(&vh));
    });
    // reference: a big square GEMM where matrixmultiply runs near peak (sanity for the % scale)
    let a = fill(512, 512, 0.0);
    let b = fill(512, 512, 1.0);
    let f_sq = 2.0 * 512.0 * 512.0 * 512.0;
    bench("ref 512^3 (high-reuse)", f_sq, 200, || {
        black_box(a.dot(&b));
    });
}
