//! Goal-4 feasibility probe: can per-matmul dual-precision work on this whole-array arch?
//!
//! Each whole-array xclbin occupies all 8 NPU columns, so a bf16 (64x32x96) and an int8 (64x64x96)
//! kernel are DIFFERENT 8-column programs that cannot be co-resident. This probe (a) checks whether
//! two SharedCtxA of different precision can even be built + dispatched in one process, and (b)
//! measures the per-op cost of ALTERNATING between them (= a precision switch every op) vs staying
//! on one context. That switch cost is what a naive per-matmul-mixed encoder would pay at every
//! precision boundary. NPU single-tenant — stop the services first.
//!
//! Run from repo root: rust/target/release/dual_precision_probe

use std::path::Path;
use std::rc::Rc;
use std::time::Instant;

use ndarray::prelude::*;
use npu_asr::ctx2::{CtxAOp, Epi, Precision, SharedCtxA};
use npu_xrt::Device;

fn bench<F: FnMut()>(label: &str, n: usize, mut f: F) -> f64 {
    let t = Instant::now();
    for _ in 0..n {
        f();
    }
    let ms = t.elapsed().as_secs_f64() * 1e3 / n as f64;
    println!("  {label:<26} {ms:.3} ms/op  ({n} ops)");
    ms
}

fn main() {
    let root = Path::new(".");
    let dev = Rc::new(Device::open(0).expect("open NPU (stop npu-asr/voxd first)"));

    println!("[probe] building bf16 (FastBf16 64x32x96) context...");
    let ctx_b = SharedCtxA::with_precision(&dev, root, Precision::FastBf16);
    println!("[probe] building int8 (64x64x96) context in the SAME process...");
    let ctx_i = SharedCtxA::with_precision(&dev, root, Precision::Int8);
    println!("[probe] BOTH contexts built — two 8-column hw-contexts coexist in one process.\n");

    // a real-sized projection: [768,768] weight, N=768, plus a [400,768] activation.
    let w = Array2::<f32>::from_shape_fn((768, 768), |(k, n)| ((k + n) as f32 * 0.001).sin() * 0.03);
    let a = Array2::<f32>::from_shape_fn((400, 768), |(t, c)| ((t + c) as f32 * 0.002).cos() * 0.1);
    let bias = vec![0.0f32; 768];
    let op_b = CtxAOp::new(ctx_b.clone(), &w, 768, Epi::Bias, &bias);
    let op_i = CtxAOp::new(ctx_i.clone(), &w, 768, Epi::Bias, &bias);

    // correctness sanity: both finite, and int8 ~ bf16 (same weight/act).
    let yb = op_b.forward(&a);
    let yi = op_i.forward(&a);
    let (mut maxd, mut maxr) = (0f32, 0f32);
    for (b, i) in yb.iter().zip(yi.iter()) {
        maxd = maxd.max((b - i).abs());
        maxr = maxr.max(b.abs());
    }
    println!("[probe] sanity: bf16 vs int8 same op  max|Δ|={maxd:.3e}  rel={:.2e}\n", maxd / (maxr + 1e-9));

    // warm both contexts.
    for _ in 0..5 {
        op_b.forward(&a);
        op_i.forward(&a);
    }

    println!("[probe] per-op latency:");
    let n = 60;
    let same_b = bench("stay on bf16", n, || {
        op_b.forward(&a);
    });
    let same_i = bench("stay on int8", n, || {
        op_i.forward(&a);
    });
    // alternating = a precision switch (different 8-col xclbin) every dispatch.
    let alt = bench("ALTERNATE bf16<->int8", n, || {
        op_b.forward(&a);
        op_i.forward(&a);
    }) / 2.0; // two ops per iter; report per-op
    println!("  (alternate is per-op = total/2)\n");

    let base = (same_b + same_i) / 2.0;
    let switch = alt - base;
    println!("[probe] VERDICT:");
    println!("  same-context avg  : {base:.3} ms/op");
    println!("  alternating avg   : {alt:.3} ms/op");
    println!("  => switch overhead : ~{switch:.3} ms per precision boundary");
    if switch > 0.5 {
        println!("  Per-matmul mixing pays this at EVERY precision boundary — confirms the");
        println!("  whole-array 8-column reload wall. Group same-precision ops or stay single-precision.");
    } else {
        println!("  Switch is cheap here — per-matmul mixing is viable without a reload penalty.");
    }
}
