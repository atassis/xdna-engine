//! Verify the Rust Parakeet host reference encoder vs the ONNX reference activations
//! (artifacts/parakeet/encoder/refs/). Mirrors scripts/parakeet_ref_encoder.py gates.
//! Run from repo root:  rust/target/release/verify_parakeet

use std::path::Path;

use ndarray::prelude::*;
use npu_parakeet::config::ModelCfg;
use npu_parakeet::encoder::FastConformerEncoder;

fn rel(got: &Array2<f32>, refr: &Array2<f32>) -> f32 {
    let mut num = 0f64;
    let mut den = 0f64;
    for (g, r) in got.iter().zip(refr.iter()) {
        let d = (*g as f64) - (*r as f64);
        num += d * d;
        den += (*r as f64) * (*r as f64);
    }
    (num.sqrt() / (den.sqrt() + 1e-12)) as f32
}

fn squeeze0(a: ArrayD<f32>) -> Array2<f32> {
    a.index_axis(Axis(0), 0).to_owned().into_dimensionality::<Ix2>().unwrap()
}

fn main() {
    // bf16 NPU matmuls diverge more than the f32 host ref, so relax tol on the NPU path
    // (GigaAM's bf16 encoder lands ~1.8e-2 vs ONNX; 0.08 is the verify_encoder bar).
    let npu_mode = std::env::args().any(|a| a == "--npu");
    let tol = 0.08f32;
    let artifacts = Path::new("artifacts/parakeet/encoder");

    #[cfg(feature = "npu")]
    let enc = if npu_mode {
        let root = std::env::var("NPU_XCLBIN_ROOT")
            .unwrap_or_else(|_| "$REPO".into());
        println!("[npu] matmuls on NPU; xclbin root = {root}");
        FastConformerEncoder::new_npu(artifacts, ModelCfg::PARAKEET_V3, Path::new(&root))
    } else {
        FastConformerEncoder::new(artifacts, ModelCfg::PARAKEET_V3)
    };
    #[cfg(not(feature = "npu"))]
    let enc = {
        if npu_mode {
            eprintln!("built without --features npu; running host reference");
        }
        FastConformerEncoder::new(artifacts, ModelCfg::PARAKEET_V3)
    };

    let w = enc.weights();

    let mut fails: Vec<String> = Vec::new();

    // ---- gate 2: subsample (audio_signal [1,128,T] -> block_in [1,T',D]) ----
    let audio = squeeze0(w.ref_tensor("audio_signal")); // [128, T]
    let block_in = squeeze0(w.ref_tensor("block_in")); // [T', D]
    let sub = enc.subsample(&audio);
    let r2 = rel(&sub, &block_in);
    println!("[gate2] subsample vs block_in:   rel={r2:.2e}  {}", if r2 <= tol { "OK" } else { "FAIL" });
    if r2 > tol {
        fails.push("subsample".into());
    }

    // ---- gates 3+4: block stack from block_in ----
    let outs = enc.forward_collect(&block_in);
    let r3 = rel(&outs[0], &squeeze0(w.ref_tensor("out_L0")));
    println!("[gate3] block-0 vs out_L0:        rel={r3:.2e}  {}  <- rel-pos gate", if r3 <= tol { "OK" } else { "FAIL" });
    if r3 > tol {
        fails.push("block0".into());
    }

    let mut worst = 0f32;
    for (b, out) in outs.iter().enumerate() {
        let rb = rel(out, &squeeze0(w.ref_tensor(&format!("out_L{b}"))));
        worst = worst.max(rb);
        if rb > tol {
            println!("  [block {b}] rel={rb:.2e} FAIL");
            fails.push(format!("block{b}"));
        }
    }
    // encoded ref is [1, D, T'] -> transpose to [T', D]
    let enc_ref = squeeze0(w.ref_tensor("encoded")).reversed_axes().to_owned();
    let r_enc = rel(outs.last().unwrap(), &enc_ref);
    let g4 = worst <= tol && r_enc <= tol;
    println!("[gate4] full {}-block: worst per-block rel={worst:.2e}; final vs encoded rel={r_enc:.2e}  {}",
             enc.cfg.n_layers, if g4 { "OK" } else { "FAIL" });
    if r_enc > tol {
        fails.push("encoded".into());
    }

    if fails.is_empty() {
        println!("\nALL GATES PASS");
    } else {
        fails.sort();
        fails.dedup();
        println!("\nFAILED: {fails:?}");
        std::process::exit(1);
    }
}
