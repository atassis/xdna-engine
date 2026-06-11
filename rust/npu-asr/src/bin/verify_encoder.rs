//! Verify the Rust fused GigaAM-v3 encoder vs the static-ONNX reference tensors, and benchmark
//! warm latency with an NPU-vs-host split. Mirrors `scripts/verify_fused_encoder.py`.
//!
//! NPU is single-tenant — stop flm-asr.service/voxd.service first.
//! Run from the repo root:  rust/target/release/verify_encoder [n_blocks] [iters]

use std::path::Path;
use std::rc::Rc;
use std::time::Instant;

use ndarray::prelude::*;
use npu_asr::encoder::{subsample, Encoder};
use npu_asr::engines::{prof, reset_prof};
use npu_asr::weights::WeightStore;
use npu_xrt::Device;

const TARGET_WHISPER_MS: f64 = 3300.0;
const TARGET_CPU_MS: f64 = 890.0;

fn rel(got: &Array2<f32>, refr: &Array2<f32>) -> (f32, f32) {
    let mut maxd = 0f32;
    let mut maxr = 0f32;
    for (g, r) in got.iter().zip(refr.iter()) {
        maxd = maxd.max((g - r).abs());
        maxr = maxr.max(r.abs());
    }
    (maxd, maxd / (maxr + 1e-9))
}

fn squeeze0(a: ArrayD<f32>) -> Array2<f32> {
    a.index_axis(Axis(0), 0)
        .to_owned()
        .into_dimensionality::<Ix2>()
        .unwrap()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n_blocks: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(16);
    let iters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(5);
    let tol = 0.08f32;

    let root = Path::new(".");
    let artifacts = root.join("artifacts/encoder");
    let ws = WeightStore::load(&artifacts).expect("load weights");
    let nb = n_blocks.min(ws.nblocks());

    let dev = Rc::new(Device::open(0).expect("open NPU (stop flm-asr/voxd first)"));
    let tb = Instant::now();
    let enc = Encoder::new(dev, root, &ws, nb);
    println!(
        "fused encoder: {nb} blocks; matmul-heavy ops on NPU, glue on host\n  build (weights pre-folded/synced once) = {:.2} s\n",
        tb.elapsed().as_secs_f64()
    );

    let audio = squeeze0(ws.ref_tensor("audio_signal")); // [64,1600]
    let x0 = subsample(&ws, &audio);

    // correctness pass (also warms the device)
    let outs = enc.forward_blocks(&x0, 400);
    let mut worst = 0f32;
    for i in 0..nb {
        let refr = squeeze0(ws.ref_tensor(&format!("out_L{i}")));
        let (_d, r) = rel(&outs[i], &refr);
        worst = worst.max(r);
        if i < 2 || i >= nb.saturating_sub(2) || r > tol {
            let flag = if r > tol { "  **OFF**" } else { "" };
            println!("  block {i:<2}    rel={r:.2e}{flag}");
        }
    }
    if nb == 16 {
        let encoded = outs[nb - 1].t().to_owned(); // [768,400]
        let refr = squeeze0(ws.ref_tensor("encoded")); // [768,400]
        let (_d, r) = rel(&encoded, &refr);
        let verdict = if r < tol { "PASS" } else { "FAIL" };
        println!("\n  {:12} rel={r:.2e} vs static ONNX  ({verdict})", "ENCODED");
    }
    println!("  worst per-block rel = {worst:.2e}");

    // steady-state (warm) latency + NPU-vs-host split
    let _ = enc.forward_blocks(&x0, 400); // extra warmup
    reset_prof();
    npu_asr::engines::marsh::reset();
    npu_asr_host::prof::reset();
    let t0 = Instant::now();
    for _ in 0..iters {
        let xs = subsample(&ws, &audio);
        let _ = enc.forward_blocks(&xs, 400);
    }
    let warm_ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;
    let (npu_s, ndisp) = prof();
    let npu_ms = npu_s * 1e3 / iters as f64;
    println!(
        "  STEADY-STATE inference ({nb} blocks, warm) = {warm_ms:.0} ms/run (vs Whisper {TARGET_WHISPER_MS:.0} ms, CPU {TARGET_CPU_MS:.0} ms)"
    );
    println!(
        "  split: NPU dispatch {npu_ms:.0} ms ({} dispatches) | host (glue+numpy+dwconv) {:.0} ms",
        ndisp / iters as u64,
        warm_ms - npu_ms
    );
    npu_asr_host::prof::dump(iters); // per-op host breakdown when NPU_HOST_PROF is set
    npu_asr::engines::marsh::dump(iters); // per-dispatch marshaling breakdown when NPU_MARSH_PROF is set
}
