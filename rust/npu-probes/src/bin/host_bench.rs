//! Standalone CPU-only microbenchmarks for the GigaAM-v3 host glue ops, at realistic encoder
//! shapes (T=400, D=768, 16 heads x 48). NO NPU. Reports per-call ms and the per-block cost
//! (call x per-block-count) so you can see where the ~260 ms host budget over 16 blocks goes.
//!
//!   cargo run --release -p npu-asr-host --bin host_bench
//!
//! Pin the thread pool with RAYON_NUM_THREADS to mirror the encoder host pool if desired.

use std::time::Instant;

use ndarray::prelude::*;
use npu_asr_host::*;

const T: usize = 400;
const D: usize = 768;
const N_HEADS: usize = 16;
const HEAD_DIM: usize = 48; // 16*48 = 768

fn fill(t: usize, d: usize) -> Array2<f32> {
    Array2::from_shape_fn((t, d), |(i, j)| {
        ((i as f32 * 0.013 + j as f32 * 0.007).sin()) * 1.3
    })
}

/// run `f` `iters` times, return best (min) ms to reduce noise.
fn bench(name: &str, per_block: f64, iters: usize, mut f: impl FnMut()) {
    // warmup
    for _ in 0..3 {
        f();
    }
    let mut best = f64::INFINITY;
    for _ in 0..iters {
        let t0 = Instant::now();
        f();
        best = best.min(t0.elapsed().as_secs_f64() * 1e3);
    }
    println!(
        "  {:<20} {:>9.3} ms/call   x{:>4.1}/blk = {:>7.3} ms/blk   x16 = {:>7.2} ms",
        name,
        best,
        per_block,
        best * per_block,
        best * per_block * 16.0,
    );
}

fn main() {
    println!(
        "RAYON_NUM_THREADS={}",
        std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "default".into())
    );
    println!("host-op microbench, T={T} D={D} heads={N_HEADS} hd={HEAD_DIM}\n");
    let iters = 200;

    // --- shared inputs ---
    let x = fill(T, D);
    let gamma: Vec<f32> = (0..D).map(|i| 1.0 + 0.01 * i as f32).collect();
    let beta: Vec<f32> = (0..D).map(|i| 0.001 * i as f32).collect();
    let cos = Array2::from_shape_fn((T, HEAD_DIM), |(i, j)| (0.01 * (i + j) as f32).cos());
    let sin = Array2::from_shape_fn((T, HEAD_DIM), |(i, j)| (0.01 * (i + j) as f32).sin());
    let q = fill(T, D);
    let k = fill(T, D);
    let v = fill(T, D);
    let pw = fill(2 * D, T); // GLU input [2C, T] channel-major
    let conv_x = fill(D, T); // [C, T]
    let taps = Array2::from_shape_fn((D, 5), |(c, ki)| 0.1 * ((c + ki) as f32).sin());
    let f1 = fill(T, D); // an FFN/proj output for residual benches

    println!("op                   per-call           per-block            x16 (all blocks)");
    println!("------------------------------------------------------------------------------");

    // bf16_round of a full [T,D] array: ~5 full-array sweeps/block (initial + 4 residuals fuse it)
    bench("bf16_round[T,D]", 1.0, iters, || {
        let _ = x.mapv(bf16_round);
    });

    // residual add + bf16 round (the fused form used after each sublayer): 4x/block
    bench("residual+bf16(old)", 4.0, iters, || {
        let _ = (&x + &f1.mapv(|v| 0.5 * v)).mapv(bf16_round);
    });
    bench("residual+bf16(fused)", 4.0, iters, || {
        let _ = residual_add_round(&x, &f1, 0.5);
    });

    // layer_norm [T,D] affine: ~6/block (satt, conv, bn, out, ffn1-norm, ffn2-norm)
    bench("layer_norm[T,D]", 6.0, iters, || {
        let _ = layer_norm(&x, &gamma, &beta, 1e-5);
    });

    // rope: 1/block
    bench("rope[T,D]", 1.0, iters, || {
        let _ = rope(&x, &cos, &sin, N_HEADS, HEAD_DIM);
    });

    // mha (scores QK^T [16,400,400] + softmax + context AV): 1/block — the big one
    bench("mha (full)", 1.0, iters, || {
        let _ = mha(&q, &k, &v, N_HEADS, HEAD_DIM, true, T);
    });

    // --- mha decomposition (single head, serial, to isolate cost centers) ---
    {
        use rayon::prelude::*;
        // per-head matmuls only (QK^T + AV), all heads parallel, no softmax
        bench("mha:matmuls-only", 1.0, iters, || {
            let _ctxs: Vec<Array2<f32>> = (0..N_HEADS)
                .into_par_iter()
                .map(|h| {
                    let base = h * HEAD_DIM;
                    let qh = q.slice(s![.., base..base + HEAD_DIM]);
                    let kh = k.slice(s![.., base..base + HEAD_DIM]);
                    let vh = v.slice(s![.., base..base + HEAD_DIM]);
                    let sc = qh.dot(&kh.t());
                    sc.dot(&vh)
                })
                .collect();
        });
        // softmax-only over [16,400,400] (parallel over heads), no matmul: OLD libm-exp form
        let mut scores: Vec<Array2<f32>> =
            (0..N_HEADS).map(|_| fill(T, T)).collect();
        bench("mha:softmax(libm)", 1.0, iters, || {
            scores.par_iter_mut().for_each(|sc| {
                for mut row in sc.rows_mut() {
                    let mut maxv = f32::NEG_INFINITY;
                    for &x in row.iter() {
                        if x > maxv {
                            maxv = x;
                        }
                    }
                    let mut sum = 0f32;
                    for x in row.iter_mut() {
                        *x = (*x - maxv).exp();
                        sum += *x;
                    }
                    let inv = 1.0 / sum;
                    for x in row.iter_mut() {
                        *x *= inv;
                        *x = bf16_round(*x);
                    }
                }
            });
        });
        // NEW softmax: fast-exp + scale-fused, no bf16_round (isolates the exp win)
        let mut scores_fe: Vec<Array2<f32>> = (0..N_HEADS).map(|_| fill(T, T)).collect();
        let scale = 1.0 / (HEAD_DIM as f32).sqrt();
        bench("mha:softmax(fastexp)", 1.0, iters, || {
            scores_fe.par_iter_mut().for_each(|sc| {
                for mut row in sc.rows_mut() {
                    let r = row.as_slice_mut().unwrap();
                    let mut maxv = f32::NEG_INFINITY;
                    for &x in r.iter() {
                        if x > maxv {
                            maxv = x;
                        }
                    }
                    let off = maxv * scale;
                    let mut sum = 0f32;
                    for x in r.iter_mut() {
                        let e = npu_asr_host::fast_exp_nonpos(*x * scale - off);
                        *x = e;
                        sum += e;
                    }
                    let inv = 1.0 / sum;
                    for x in r.iter_mut() {
                        *x *= inv;
                    }
                }
            });
        });
        // softmax-only WITHOUT bf16_round (libm exp) to isolate the rounding cost
        let mut scores2: Vec<Array2<f32>> = (0..N_HEADS).map(|_| fill(T, T)).collect();
        bench("mha:softmax(no-round)", 1.0, iters, || {
            scores2.par_iter_mut().for_each(|sc| {
                for mut row in sc.rows_mut() {
                    let mut maxv = f32::NEG_INFINITY;
                    for &x in row.iter() {
                        if x > maxv {
                            maxv = x;
                        }
                    }
                    let mut sum = 0f32;
                    for x in row.iter_mut() {
                        *x = (*x - maxv).exp();
                        sum += *x;
                    }
                    let inv = 1.0 / sum;
                    for x in row.iter_mut() {
                        *x *= inv;
                    }
                }
            });
        });
    }

    // glu [2C,T] -> [C,T]: 1/block
    bench("glu[2C,T]", 1.0, iters, || {
        let _ = glu(&pw);
    });

    // dwconv k=5 [C,T]: 1/block
    bench("dwconv_k5[C,T]", 1.0, iters, || {
        let _ = dwconv_k5(&conv_x, &taps);
    });

    // silu [C,T]: 1/block
    bench("silu[C,T]", 1.0, iters, || {
        let _ = silu(&conv_x);
    });
}
