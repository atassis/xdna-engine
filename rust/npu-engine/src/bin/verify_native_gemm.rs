//! Validate the standalone NATIVE whole-array GEMM dispatcher against a CPU reference.
//! Proves the parallel native path (load arbitrary-shape xclbin + insts, dispatch, read) works
//! end-to-end before building the full native ESM encoder. Idle NPU.
//! Usage: verify_native_gemm [K] [N] [tile]   (default 320 1280 32x32x32)
use std::path::Path;
use std::rc::Rc;

use ndarray::Array2;
use npu_engine::esm::native::NativeGemm;
use npu_xrt::{bf16_bits_to_f32, f32_to_bf16_bits, Device};

fn bf16(x: f32) -> f32 {
    bf16_bits_to_f32(f32_to_bf16_bits(x))
}

fn main() {
    let k: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(320);
    let n: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(1280);
    let tile = std::env::args().nth(3).unwrap_or_else(|| "32x32x32".into());
    let m = 64usize;
    let wa = Path::new("mlir-aie/programming_examples/basic/matrix_multiplication/whole_array/build");

    // deterministic pseudo-random A, B in [-1,1) (no Math.random; index-derived).
    let r = |i: usize| ((i.wrapping_mul(2654435761) >> 8) & 0xffff) as f32 / 32768.0 - 1.0;
    let a = Array2::from_shape_fn((m, k), |(i, j)| 0.5 * r(i * k + j + 1));
    let b = Array2::from_shape_fn((k, n), |(i, j)| 0.5 * r(7 + i * n + j));

    // CPU reference at bf16 input precision (kernel rounds A,B to bf16; accumulate in f32).
    let mut cpu = Array2::<f32>::zeros((m, n));
    for i in 0..m {
        for kk in 0..k {
            let av = bf16(a[[i, kk]]);
            for j in 0..n {
                cpu[[i, j]] += av * bf16(b[[kk, j]]);
            }
        }
    }

    let dev = Rc::new(Device::open(0).expect("open NPU"));
    let mut g = NativeGemm::load(&dev, wa, k, n, &tile);
    g.set_weight(&b);
    let npu = g.matmul(&a, None);

    let mut max_rel = 0f32;
    let mut max_abs = 0f32;
    for i in 0..m {
        for j in 0..n {
            let c = cpu[[i, j]];
            let o = npu[[i, j]];
            max_abs = max_abs.max((c - o).abs());
            let denom = c.abs().max(1.0);
            max_rel = max_rel.max((c - o).abs() / denom);
        }
    }
    println!("native GEMM K={k} N={n} tile={tile}: max_abs={max_abs:.4} max_rel={max_rel:.4}");
    assert!(max_rel < 0.08, "native GEMM rel error {max_rel} too high (>0.08) — dispatch/layout bug");
    println!("NATIVE GEMM OK (rel < 0.08)");
}
