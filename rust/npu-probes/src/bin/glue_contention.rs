//! Phase-1a of the task-graph scheduler gate (internal notes).
//! CPU-ONLY (no NPU). Answers the decisive question the throughput projection rests on:
//!
//!   Is the host glue memory-BANDWIDTH-bound, so that running N streams concurrently contends
//!   instead of scaling? (If yes, the throughput ceiling is host-bound and well below the NPU's ~5.4x.)
//!
//! Three measurements on a representative one-block glue bundle (T=512, D=768, 16 heads):
//!   (1) SELF-SCALING: the bundle under rayon pools of size {1,2,4,6,8,12,20}. If it plateaus early,
//!       the glue is bandwidth-bound (can't use all cores).
//!   (2) SUSTAINED MULTI-STREAM: C streams each on a DISJOINT ~20/C-core pool (the scheduler model),
//!       each running `reps` bundles back-to-back after a barrier so they overlap for the whole run.
//!       host-throughput = C * solo / mean-per-bundle. ~C => headroom; ~1 => bandwidth-saturated.
//!   (3) PER-OP contention: which op (LN / residual / bf16_round / mha) slows most under concurrency.
//!
//!   cargo run --release -p npu-asr-host --bin glue_contention      (set REPS=60 for a longer clean run)
//!
//! RUN ON AN IDLE MACHINE for clean numbers (music/other sessions inflate the contention floor).

use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Instant;

use ndarray::prelude::*;
use npu_asr_host::*;
use rayon::ThreadPoolBuilder;

const T: usize = 512;
const D: usize = 768;
const N_HEADS: usize = 16;
const HEAD_DIM: usize = 48;

fn fill(t: usize, d: usize) -> Array2<f32> {
    Array2::from_shape_fn((t, d), |(i, j)| ((i as f32 * 0.013 + j as f32 * 0.007).sin()) * 1.3)
}

#[derive(Clone)]
struct In {
    x: Array2<f32>,
    f1: Array2<f32>,
    q: Array2<f32>,
    k: Array2<f32>,
    v: Array2<f32>,
    gamma: Vec<f32>,
    beta: Vec<f32>,
    cos: Array2<f32>,
    sin: Array2<f32>,
}
impl In {
    fn new() -> Self {
        In {
            x: fill(T, D),
            f1: fill(T, D),
            q: fill(T, D),
            k: fill(T, D),
            v: fill(T, D),
            gamma: (0..D).map(|i| 1.0 + 0.01 * i as f32).collect(),
            beta: (0..D).map(|i| 0.001 * i as f32).collect(),
            cos: Array2::from_shape_fn((T, HEAD_DIM), |(i, j)| (0.01 * (i + j) as f32).cos()),
            sin: Array2::from_shape_fn((T, HEAD_DIM), |(i, j)| (0.01 * (i + j) as f32).sin()),
        }
    }
}

/// One Conformer block's worth of host glue: 6x layer_norm, 1x rope, 1x mha, 4x residual+bf16, 1x round.
fn block_glue(inp: &In) {
    let mut acc = inp.x.mapv(bf16_round);
    for _ in 0..6 {
        acc = layer_norm(&acc, &inp.gamma, &inp.beta, 1e-5);
    }
    let _r = rope(&acc, &inp.cos, &inp.sin, N_HEADS, HEAD_DIM);
    let _ctx = mha(&inp.q, &inp.k, &inp.v, N_HEADS, HEAD_DIM, true, T);
    for _ in 0..4 {
        acc = residual_add_round(&acc, &inp.f1, 0.5);
    }
    std::hint::black_box(&acc);
}

fn min_ms(iters: usize, mut f: impl FnMut()) -> f64 {
    for _ in 0..3 {
        f();
    }
    let mut best = f64::INFINITY;
    for _ in 0..iters {
        let t0 = Instant::now();
        f();
        best = best.min(t0.elapsed().as_secs_f64() * 1e3);
    }
    best
}

/// Sustained concurrent throughput: `c` workers, each on its own `cores`-thread pool, each runs
/// `reps` units of `work` back-to-back after a shared barrier (so all overlap for ~the whole run).
/// Returns the MEAN per-unit ms across workers (representative of sustained, contended throughput).
fn sustained_mean_ms<F>(c: usize, cores: usize, reps: usize, make: F) -> f64
where
    F: Fn() -> Box<dyn Fn()> + Send + Sync + 'static + Clone,
{
    let barrier = Arc::new(Barrier::new(c));
    let mut handles = Vec::new();
    for _ in 0..c {
        let b = barrier.clone();
        let make = make.clone();
        handles.push(thread::spawn(move || {
            let pool = ThreadPoolBuilder::new().num_threads(cores).build().unwrap();
            pool.install(|| {
                let work = make();
                for _ in 0..3 {
                    work(); // warmup (not timed)
                }
                b.wait();
                let t0 = Instant::now();
                for _ in 0..reps {
                    work();
                }
                (t0.elapsed().as_secs_f64() * 1e3) / reps as f64
            })
        }));
    }
    let per: Vec<f64> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    per.iter().sum::<f64>() / per.len() as f64
}

fn main() {
    let ncpu = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0);
    let reps: usize = std::env::var("REPS").ok().and_then(|s| s.parse().ok()).unwrap_or(30);
    println!("glue contention probe — T={T} D={D} heads={N_HEADS}, {ncpu} CPUs, REPS={reps}\n");
    let inp = Arc::new(In::new());
    let iters = 15;

    // (1) self-scaling
    println!("(1) one stream's glue under N cores:");
    println!("    {:>5}  {:>9}  {:>8}", "cores", "ms", "speedup");
    let mut t1 = 0.0f64;
    for &p in &[1usize, 2, 4, 6, 8, 12, 20] {
        if p > ncpu {
            continue;
        }
        let pool = ThreadPoolBuilder::new().num_threads(p).build().unwrap();
        let ms = pool.install(|| min_ms(iters, || block_glue(&inp)));
        if p == 1 {
            t1 = ms;
        }
        println!("    {:>5}  {:>9.3}  {:>7.2}x", p, ms, t1 / ms);
    }
    println!("    plateau => glue is bandwidth-bound (can't use more cores).\n");

    // reference: one stream, full machine
    let solo_full = min_ms(iters, || block_glue(&inp));

    // (2) sustained bounded-pool multi-stream. Baseline = ONE stream measured the SAME (sustained) way,
    // so host-thruput is apples-to-apples (sustained vs sustained), not min-vs-mean.
    let inp_b = inp.clone();
    let base = sustained_mean_ms(1, ncpu.min(20), reps, move || {
        let i = inp_b.clone();
        Box::new(move || block_glue(&i))
    });
    let _ = solo_full;
    println!("(2) sustained multi-stream (each on disjoint ~20/C cores); 1-stream sustained baseline={base:.3} ms:");
    println!(
        "    {:>3} {:>7} {:>12} {:>13}",
        "C", "cores/s", "per-bundle", "host-thruput"
    );
    for &(c, cores) in &[(2usize, 10usize), (3, 6), (4, 5), (5, 4)] {
        if c * cores > ncpu {
            continue;
        }
        let inp2 = inp.clone();
        let mean = sustained_mean_ms(c, cores, reps, move || {
            let inp3 = inp2.clone();
            Box::new(move || block_glue(&inp3))
        });
        let thruput = (c as f64) * base / mean; // C streams' rate vs 1 stream's rate, both sustained
        println!("    {:>3} {:>7} {:>11.3} {:>11.2}x", c, cores, mean, thruput);
    }
    println!("    host-thruput ~C => headroom; flat/low => bandwidth-saturated (caps the NPU's ~5.4x).\n");

    // (3) per-op contention: solo (full) vs 4 streams x 5 cores, sustained
    println!("(3) per-op: solo (full machine, min ms) vs 4 streams x 5 cores (sustained mean):");
    println!("    {:>16} {:>10} {:>12} {:>10}", "op", "solo", "x4@5core", "slowdown");
    let ops: Vec<(&str, Arc<dyn Fn(&In) + Send + Sync>)> = vec![
        ("layer_norm", Arc::new(|i: &In| { let _ = layer_norm(&i.x, &i.gamma, &i.beta, 1e-5); })),
        ("residual+bf16", Arc::new(|i: &In| { let _ = residual_add_round(&i.x, &i.f1, 0.5); })),
        ("bf16_round", Arc::new(|i: &In| { let _ = i.x.mapv(bf16_round); })),
        ("mha", Arc::new(|i: &In| { let _ = mha(&i.q, &i.k, &i.v, N_HEADS, HEAD_DIM, true, T); })),
    ];
    for (name, op) in ops {
        let solo = {
            let o = op.clone();
            let i = inp.clone();
            min_ms(iters, || o(&i))
        };
        let i2 = inp.clone();
        let o2 = op.clone();
        let mean = sustained_mean_ms(4, 5, reps, move || {
            let i3 = i2.clone();
            let o3 = o2.clone();
            Box::new(move || o3(&i3))
        });
        println!("    {:>16} {:>8.3}ms {:>10.3}ms {:>9.2}x", name, solo, mean, mean / solo);
    }
    println!("    high slowdown = bandwidth-bound op (contends across streams). LN/residual expected worst.");
}
