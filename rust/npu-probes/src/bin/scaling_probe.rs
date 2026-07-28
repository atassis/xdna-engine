//! WS-2 Phase-0: is the poor in-pipeline scaling of `residual`/`layer_norm` caused by
//! all-core frequency throttle, rayon granularity, or memory bandwidth?
//!
//! `host_bench` reuses ONE [T,D] array across 200 iters → it stays L3-resident, so it measures
//! cache bandwidth, not the cold-DRAM regime the real encoder hits (x is freshly produced by the
//! NPU/prev op each block). This probe contrasts WARM (one reused pair, cache-resident) vs COLD
//! (a ring of K pairs sized far beyond L3 → every input load misses to DRAM). If COLD scaling
//! collapses toward the in-pipeline 1.5x/2.7x while WARM scales 4-6x, the limiter is memory
//! bandwidth (=> "coarsen granularity" won't help; only cutting bytes-moved will).
//!
//!   RAYON_NUM_THREADS=N cargo run --release -p npu-asr-host --bin scaling_probe

use std::time::Instant;

use ndarray::prelude::*;
use npu_asr_host::*;

const T: usize = 400;
const D: usize = 768;
// ring big enough that revisiting ring[0] has evicted it from L3 (Strix Point L3 ~16 MB).
// K pairs * (x+y) = 48 * 2.36 MB ~= 113 MB of inputs >> L3.
const K: usize = 48;

fn fill(seed: f32) -> Array2<f32> {
    Array2::from_shape_fn((T, D), |(i, j)| {
        ((i as f32 * 0.013 + j as f32 * 0.007 + seed).sin()) * 1.3
    })
}

/// min-of-iters ms.
fn bench(iters: usize, mut f: impl FnMut(usize)) -> f64 {
    for w in 0..3 {
        f(w);
    }
    let mut best = f64::INFINITY;
    for it in 0..iters {
        let t0 = Instant::now();
        f(it);
        best = best.min(t0.elapsed().as_secs_f64() * 1e3);
    }
    best
}

fn main() {
    let n = std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "default".into());
    println!("RAYON_NUM_THREADS={n}  (T={T} D={D}, cold-ring K={K} = {} MB inputs)",
        K * T * D * 4 * 2 / (1 << 20));
    let gamma: Vec<f32> = (0..D).map(|i| 1.0 + 0.01 * i as f32).collect();
    let beta: Vec<f32> = (0..D).map(|i| 0.001 * i as f32).collect();
    let iters = 200;

    // WARM: one reused pair, stays in cache.
    let xw = fill(0.0);
    let yw = fill(1.0);
    let res_warm = bench(iters, |_| {
        let _ = residual_add_round(&xw, &yw, 0.5);
    });
    let ln_warm = bench(iters, |_| {
        let _ = layer_norm(&xw, &gamma, &beta, 1e-5);
    });

    // COLD: rotate through K distinct pairs so each input load misses to DRAM.
    let xs: Vec<Array2<f32>> = (0..K).map(|k| fill(k as f32 * 0.31)).collect();
    let ys: Vec<Array2<f32>> = (0..K).map(|k| fill(k as f32 * 0.31 + 7.0)).collect();
    let res_cold = bench(iters, |it| {
        let k = it % K;
        let _ = residual_add_round(&xs[k], &ys[k], 0.5);
    });
    let ln_cold = bench(iters, |it| {
        let k = it % K;
        let _ = layer_norm(&xs[k], &gamma, &beta, 1e-5);
    });

    println!("  residual    warm={res_warm:.3} ms   cold={res_cold:.3} ms   (cold/warm {:.2}x)",
        res_cold / res_warm);
    println!("  layer_norm  warm={ln_warm:.3} ms   cold={ln_cold:.3} ms   (cold/warm {:.2}x)",
        ln_cold / ln_warm);
}
