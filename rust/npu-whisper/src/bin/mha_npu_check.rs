//! Validate MhaNpu against the host `mha` it replaces: random Q/K/V [1500,768], compare contexts.
//! Run under device serialization (stop npu-asr/voxd, fuser check, restart).
//!
//!   cargo run -p npu-whisper --features npu --release --bin mha_npu_check -- \
//!       artifacts/encoder_mha/StaticMHA_h12_s1500_d64_kv0_causal0_npu2.xclbin \
//!       artifacts/encoder_mha/StaticMHA_h12_s1500_d64_kv0_causal0_npu2.bin

use std::path::Path;
use std::rc::Rc;

use ndarray::prelude::*;
use npu_asr_host::mha;
use npu_whisper::mha_npu::MhaNpu;
use npu_xrt::Device;

const SEQ: usize = 1500;
const DMODEL: usize = 768;
const HEADS: usize = 12;
const HEAD_DIM: usize = 64;

fn lcg(state: &mut u64) -> f32 {
    // simple deterministic [-1,1) rng (no extra deps)
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*state >> 33) as f32 / (1u64 << 31) as f32) - 1.0
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let xclbin = Path::new(&args[1]);
    let insts = Path::new(&args[2]);

    // optional 3rd arg = input scale (default 1.0); larger -> more peaked softmax (tests bf16 error
    // sensitivity to attention concentration, mimicking real post-LayerNorm projected activations).
    let scale: f32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1.0);
    let mut s = 0x1234_5678_9abc_def0u64;
    let mk = |s: &mut u64| Array2::<f32>::from_shape_fn((SEQ, DMODEL), |_| lcg(s) * scale);
    let q = mk(&mut s);
    let k = mk(&mut s);
    let v = mk(&mut s);
    println!("[mha_npu_check] input scale={scale}");

    let host = mha(&q, &k, &v, HEADS, HEAD_DIM, false, SEQ); // [1500,768]

    let dev = Rc::new(Device::open(0).expect("open NPU (stop npu-asr/voxd first)"));
    let op = MhaNpu::open(&dev, xclbin, insts).expect("MhaNpu::open");
    let got = op.forward(&q, &k, &v); // [1500,768]

    let diff = &got - &host;
    let l2 = diff.iter().map(|x| x * x).sum::<f32>().sqrt();
    let den = host.iter().map(|x| x * x).sum::<f32>().sqrt() + 1e-12;
    let rel_l2 = l2 / den;
    let max_abs = diff.iter().fold(0.0f32, |m, x| m.max(x.abs()));
    let mean_abs = diff.iter().map(|x| x.abs()).sum::<f32>() / diff.len() as f32;
    let n_big = diff.iter().filter(|x| x.abs() > 0.5).count();
    println!("[mha_npu_check] rel-L2={rel_l2:.5} max_abs={max_abs:.4} mean_abs={mean_abs:.5} #abs>0.5={n_big}/{}", diff.len());
    println!("[mha_npu_check] RESULT: {}", if rel_l2 < 0.05 { "PASS (rel-L2<0.05)" } else { "FAIL" });
    std::process::exit(if rel_l2 < 0.05 { 0 } else { 2 });
}
