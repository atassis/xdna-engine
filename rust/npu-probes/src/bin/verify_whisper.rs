//! Gate the Rust Whisper-small host reference encoder vs the ONNX golden activations
//! (artifacts/whisper-small/refs/). TDD: the rel gates ARE the test. Run from the worktree ROOT
//! so the `artifacts/` symlink resolves:  rust/target/release/verify_whisper [artifacts_dir]

use std::path::Path;

use ndarray::prelude::*;
use npu_whisper::config::WhisperCfg;
use npu_whisper::encoder::WhisperEncoder;

const TOL_HOST: f32 = 5e-3;
/// P2 make-or-break gate: NPU (bf16/int8 over 12 layers) vs ONNX golden.
#[cfg(feature = "npu")]
const TOL_NPU: f32 = 0.08;

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

fn as2(a: ArrayD<f32>) -> Array2<f32> {
    a.into_dimensionality::<Ix2>().unwrap()
}

fn main() {
    // Parse args: an optional positional artifacts dir + an optional `--npu` flag (any order).
    let mut npu = false;
    let mut artifacts: Option<String> = None;
    for a in std::env::args().skip(1) {
        if a == "--npu" {
            npu = true;
        } else if !a.starts_with("--") {
            artifacts = Some(a);
        }
    }
    let artifacts = artifacts.unwrap_or_else(|| "artifacts/whisper-small".into());

    let (enc, tol, backend) = if npu {
        #[cfg(feature = "npu")]
        {
            // root = worktree root (cwd), where mlir-aie/.../whole_array/build resolves.
            let enc = WhisperEncoder::new_npu(Path::new(&artifacts), WhisperCfg::SMALL, Path::new("."));
            (enc, TOL_NPU, "npu")
        }
        #[cfg(not(feature = "npu"))]
        {
            eprintln!("--npu requested but binary built without the `npu` feature; rebuild with --features npu");
            std::process::exit(2);
        }
    } else {
        (WhisperEncoder::new(Path::new(&artifacts), WhisperCfg::SMALL), TOL_HOST, "host")
    };
    let w = enc.weights();

    // mel input_features [1, 80, 3000] -> [80, 3000]
    let mel = w.ref_tensor("input_features").index_axis(Axis(0), 0).to_owned().into_dimensionality::<Ix2>().unwrap();

    let mut fails: Vec<String> = Vec::new();

    // ---- gate 1: conv stem + positional embedding vs after_conv ----
    let mut conv = enc.conv_stem(&mel);
    enc.add_pos(&mut conv);
    let after_conv = as2(w.ref_tensor("after_conv"));
    let r_conv = rel(&conv, &after_conv);
    println!("[conv_stem] add_pos(conv_stem(mel)) vs after_conv:  rel={r_conv:.3e}  {}", if r_conv <= tol { "OK" } else { "FAIL" });
    if r_conv > tol {
        fails.push("conv_stem".into());
    }

    // ---- gate 2: each encoder block i vs block_i ----
    // On the NPU path the per-block rel is REPORT-ONLY: mid-stack blocks (2..5) have a tiny RMS
    // (~0.6) hiding a single outlier feature that only explodes at block 6 (golden max 4.8 -> 753),
    // so the relative-error denominator there is pathologically small and bf16 noise looks large.
    // Pre-norm LayerNorm renormalizes this away — the make-or-break quantity is the post-LN
    // `encoded` (gate 3). On the host f32 path the per-block gate stays strict (it's a real test).
    let block_gates = backend != "npu";
    #[cfg(feature = "npu")]
    let rail_before = enc.resident_ffn_stats();
    let outs = enc.forward_collect(&after_conv);
    // WHISPER_RESIDENT_FFN: the rail falls back to the ctx2 FFN silently, so matching activations
    // are also what a rail that never dispatched would produce. Count the bricks instead.
    #[cfg(feature = "npu")]
    if let (Some((c0, d0)), Some((c1, d1))) = (rail_before, enc.resident_ffn_stats()) {
        let (dc, dd) = (c1 - c0, d1 - d0);
        let want = enc.cfg.n_layers;
        println!("[k768_rail] {dc}/{want} blocks dispatched, {dd} bricks ({} per block)",
                 if dc > 0 { dd / dc } else { 0 });
        if dc != want || dd != 4 * want {
            fails.push(format!("k768_rail dispatched {dc}/{want} blocks, {dd} bricks (want {})", 4 * want));
        }
    }
    let mut worst = 0f32;
    for (i, out) in outs.iter().enumerate() {
        let rb = rel(out, &as2(w.ref_tensor(&format!("block_{i}"))));
        worst = worst.max(rb);
        let pass = rb <= tol;
        if i == 0 || i == enc.cfg.n_layers - 1 {
            println!("[block_{i}] rel={rb:.3e}  {}", if pass { "OK" } else if block_gates { "FAIL" } else { "(info)" });
        } else if !pass {
            println!("[block_{i}] rel={rb:.3e}  {}", if block_gates { "FAIL" } else { "(info)" });
        }
        if !pass && block_gates {
            fails.push(format!("block_{i}"));
        }
    }
    println!("[blocks] worst per-block rel={worst:.3e}{}", if block_gates { "" } else { "  (report-only on npu; gate is `encoded`)" });

    // ---- gate 3: full encoder (last block THEN ln_post) vs encoded ----
    let encoded = enc.forward_last(&mel);
    let r_enc = rel(&encoded, &as2(w.ref_tensor("encoded")));
    println!("[encoded] forward_last(mel) vs encoded:  rel={r_enc:.3e}  {}", if r_enc <= tol { "OK" } else { "FAIL" });
    if r_enc > tol {
        fails.push("encoded".into());
    }

    if fails.is_empty() {
        println!("OK ({backend})");
    } else {
        fails.sort();
        fails.dedup();
        eprintln!("FAILED: {fails:?}");
        std::process::exit(1);
    }
}
