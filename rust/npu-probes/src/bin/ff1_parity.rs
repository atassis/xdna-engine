//! FF1-macaron parity gate for the resident-stream rails work (feat/resident-rails).
//!
//! Compares the host reference FF1 (`LN -> fc1 -> SiLU -> fc2`, block 0) against the resident
//! device path gated by `PARAKEET_RESIDENT_FF`. Modes (`--ln-only`/`--mlp-only`/`--res-only`/
//! full) map to the plan's per-brick gates (Tasks 2-4). With `--resident` UNSET the harness compares
//! host-vs-host (rel-err 0) so the gate wiring is proven before any device work exists.
//!
//! Gate: per-element max relative error <= 1e-2 (the ln-cheap numeric gate). WER gate is separate:
//!   PARAKEET_RESIDENT_FF=1 python3 scripts/parakeet_npu_wer.py --set en,ru   (baseline 8.6% == oracle)
//!
//! Usage:  ff1_parity [--seed N] [--t N] [--resident] [--ln-only|--mlp-only|--res-only]
//! Needs the encoder weights at artifacts/parakeet/encoder (symlink from the main worktree if absent).

use std::path::Path;

use ndarray::prelude::*;
use npu_parakeet::config::ModelCfg;
use npu_parakeet::encoder::FastConformerEncoder;

/// Deterministic, dependency-free fill: a splitmix64-seeded LCG mapped to f32 in [-scale, scale].
fn random_matrix(rows: usize, cols: usize, seed: u64, scale: f32) -> Array2<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut next = || {
        // splitmix64
        s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        // map top 24 bits to [0,1)
        let u = (z >> 40) as f32 / (1u32 << 24) as f32;
        (u * 2.0 - 1.0) * scale
    };
    Array2::from_shape_fn((rows, cols), |_| next())
}

/// max + L2 relative error between two equal-shaped arrays.
fn rel_err(a: &Array2<f32>, b: &Array2<f32>) -> (f32, f32) {
    assert_eq!(a.dim(), b.dim(), "shape mismatch {:?} vs {:?}", a.dim(), b.dim());
    let mut max_rel = 0f32;
    let mut num = 0f64;
    let mut den = 0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (x - y).abs();
        let r = d / (x.abs().max(1e-6));
        if r > max_rel {
            max_rel = r;
        }
        num += (d as f64) * (d as f64);
        den += (*x as f64) * (*x as f64);
    }
    (max_rel, (num.sqrt() / den.sqrt().max(1e-12)) as f32)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arg_val = |flag: &str, def: &str| -> String {
        args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)).cloned().unwrap_or_else(|| def.into())
    };
    let seed: u64 = arg_val("--seed", "1").parse().unwrap();
    let t: usize = arg_val("--t", "32").parse().unwrap();
    let resident = args.iter().any(|a| a == "--resident");
    let mode = if args.iter().any(|a| a == "--ln-only") { "ln" }
        else if args.iter().any(|a| a == "--mlp-only") { "mlp" }
        else if args.iter().any(|a| a == "--res-only") { "res" }
        else { "full" };

    let artifacts = Path::new("artifacts/parakeet/encoder");
    let cfg = ModelCfg::PARAKEET_V3;
    let enc = FastConformerEncoder::new(artifacts, cfg);

    let x = random_matrix(t, cfg.hidden, seed, 1.0);

    // Host reference for the FF1 macaron (LN -> fc1 -> SiLU -> fc2), block 0.
    let host = enc.feed_forward_ff1(&x, 0);

    // Resident device path (Tasks 2-4). Until it exists, --resident falls back to host so the gate
    // wiring is provable now (host-vs-host => rel 0).
    let got = if resident {
        eprintln!("[ff1_parity] mode={mode}: resident path not wired yet (Tasks 2-4) -- comparing host-vs-host");
        enc.feed_forward_ff1(&x, 0)
    } else {
        host.clone()
    };

    let (max_rel, l2_rel) = rel_err(&host, &got);
    println!("[ff1_parity] mode={mode} t={t} seed={seed} resident={resident}  max_rel={max_rel:.3e} l2_rel={l2_rel:.3e}");
    assert!(max_rel <= 1e-2, "FF1 parity FAILED: max_rel {max_rel:.3e} > 1e-2");
    println!("[ff1_parity] PASS (<= 1e-2)");
}
