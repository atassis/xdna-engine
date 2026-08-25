//! Per-seam device parity gate for the whole-block-resident fusion (feat/whole-block-fusion).
//!
//! Each subcommand gates one fusion primitive on device: does the `*_dev` (device-resident) variant
//! match its host-assembled reference within the seam's rel-L2 tolerance? Fast (synthetic input),
//! run BEFORE the full 17-clip WER regression.
//!
//!   ffn      -- Task 1: on-device fc2 accumulation. resident_ffn (host-accum) vs resident_ffn_dev
//!               (acc_add on-chip). Accumulation is the ONLY change -> rel-L2 must be ~0 (<= 1e-4).
//!   residual -- Task 2: on-chip scaled residual add. host `a + 0.5*b` vs residual_add_dev.
//!               f32 mul+add near-exact -> rel-L2 must be ~0 (<= 1e-4).
//!   ln       -- Task 3: device-in LN. host ops::layernorm(x,g,b) vs ln_affine_cast_dev (device-in
//!               ctxLN+affine). bf16 output -> rel-L2 <= 5e-3.
//!   k768ffn  -- the LN-less K=768 GELU rail as a CHAIN: fc1 modalgelu -> cast@3072 -> fc2 modalid
//!               -> resadd s100 vs a host oracle fed the same bf16 operands. bfp16 GEMMs over
//!               K=800 and K=3072 -> rel-L2 <= 5e-2, the bar the per-brick gate script uses.
//!               `--pad-m` picks the built width (512 = BERT short seq, 1536 = Whisper-small).
//!   k768resid -- the same chain through the PRE-norm entry point, fc1's input and the residual
//!               operand drawn independently (Whisper's `ln2` and `x`). Controlled by the swapped
//!               residual, which is what a rail collapsing the two operands computes.
//!
//! Run (NPU quiesced, from the repo root):
//!   NPU_XCLBIN_ROOT=$PWD cargo run --features npu --release --bin fused_seam_parity -- ffn
//!   NPU_XCLBIN_ROOT=$PWD cargo run --features npu --release --bin fused_seam_parity -- residual
//! Needs the resident modal xclbin + artifacts/parakeet/ln/{ctxln,affcast,deint,accadd,resadd} (built
//! by scripts/build_parakeet_modal_kernels.sh).

use ndarray::Array2;
use npu_parakeet::npu::NpuMatmul;
use std::path::Path;

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
    let seam = args.iter().skip(1).find(|a| !a.starts_with("--")).cloned().unwrap_or_else(|| "ffn".into());
    let arg_val = |flag: &str, def: &str| -> String {
        args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)).cloned().unwrap_or_else(|| def.into())
    };
    let t: usize = arg_val("--t", "64").parse().unwrap();
    let seed: u64 = arg_val("--seed", "1").parse().unwrap();

    let root = std::env::var("NPU_XCLBIN_ROOT").unwrap_or_else(|_| ".".into());
    // The K=768 rail is a different NpuMatmul CONFIGURATION, not another call on Parakeet's. Every
    // k768* seam needs it; matching one seam name by hand is how `k768resid` silently opened a
    // KRES=1024 Parakeet and reported the rail missing.
    let npu = if seam.starts_with("k768") {
        let pad_m: usize = arg_val("--pad-m", "512").parse().unwrap();
        NpuMatmul::open_with_rail(Path::new(&root), 768, pad_m, 3072)
    } else {
        NpuMatmul::open(Path::new(&root))
    };

    match seam.as_str() {
        "ffn" => {
            let (host, dev) = npu.ffn_devacc_selftest(t, seed).unwrap_or_else(|| {
                panic!("[fused_seam_parity] ffn: modal/resident/acc_add xclbins absent -- build \
                        scripts/build_parakeet_modal_kernels.sh (needs final_accadd_512x1024)");
            });
            let (max_rel, l2_rel) = rel_err(&host, &dev);
            println!("[fused_seam_parity] seam=ffn t={t} seed={seed}  max_rel={max_rel:.3e} rel-L2={l2_rel:.3e}");
            assert!(l2_rel <= 1e-4, "FFN device-accum parity FAILED: rel-L2 {l2_rel:.3e} > 1e-4");
            println!("[fused_seam_parity] PASS (rel-L2 <= 1e-4)");
        }
        "residual" => {
            let scale: f32 = arg_val("--scale", "0.5").parse().unwrap();
            let (host, dev) = npu.residual_add_selftest(t, seed, scale).unwrap_or_else(|| {
                panic!("[fused_seam_parity] residual: resadd xclbin absent for scale={scale} -- build \
                        scripts/build_parakeet_modal_kernels.sh (needs final_resadd_512x1024_s050/s100)");
            });
            let (max_rel, l2_rel) = rel_err(&host, &dev);
            println!("[fused_seam_parity] seam=residual scale={scale} t={t} seed={seed}  max_rel={max_rel:.3e} rel-L2={l2_rel:.3e}");
            assert!(l2_rel <= 1e-4, "residual_add parity FAILED: rel-L2 {l2_rel:.3e} > 1e-4");
            println!("[fused_seam_parity] PASS (rel-L2 <= 1e-4)");
        }
        "ln" => {
            let (host, dev) = npu.ln_affine_cast_dev_selftest(t, seed).unwrap_or_else(|| {
                panic!("[fused_seam_parity] ln: ctxln/affcast xclbins absent -- build \
                        scripts/build_parakeet_modal_kernels.sh (needs final_ctxln/affcast_512x1024)");
            });
            let (max_rel, l2_rel) = rel_err(&host, &dev);
            println!("[fused_seam_parity] seam=ln t={t} seed={seed}  max_rel={max_rel:.3e} rel-L2={l2_rel:.3e}");
            assert!(l2_rel <= 5e-3, "ln device-in parity FAILED: rel-L2 {l2_rel:.3e} > 5e-3");
            println!("[fused_seam_parity] PASS (rel-L2 <= 5e-3)");
        }
        "linout" => {
            let (host, dev) = npu.linout_selftest(t, seed).expect("linout_selftest: modal absent");
            let (max_rel, l2_rel) = rel_err(&host, &dev);
            println!("[fused_seam_parity] seam=linout t={t} seed={seed}  max_rel={max_rel:.3e} rel-L2={l2_rel:.3e}");
            assert!(l2_rel <= 1e-4, "linout parity FAILED: rel-L2 {l2_rel:.3e} > 1e-4");
            println!("[fused_seam_parity] PASS (rel-L2 <= 1e-4)");
        }
        "convfront" => {
            let (host, dev) = npu.conv_front_selftest(t, seed).expect("conv_front_selftest: xclbins absent");
            let (max_rel, l2_rel) = rel_err(&host, &dev);
            println!("[fused_seam_parity] seam=convfront t={t} seed={seed}  max_rel={max_rel:.3e} rel-L2={l2_rel:.3e}");
            assert!(l2_rel <= 1e-4, "convfront parity FAILED: rel-L2 {l2_rel:.3e} > 1e-4");
            println!("[fused_seam_parity] PASS (rel-L2 <= 1e-4)");
        }
        "k768ffn" => {
            let (host, dev, controls) = npu.k768_ffn_selftest(t, seed).unwrap_or_else(|| {
                panic!("[fused_seam_parity] k768ffn: K=768 rail xclbins absent -- build \
                        PAD_M=<width> scripts/build_k768_gelu_rail.sh into artifacts/k768_gelu_rail");
            });
            let (max_rel, l2_rel) = rel_err(&host, &dev);
            println!("[fused_seam_parity] seam=k768ffn t={t} seed={seed}  max_rel={max_rel:.3e} rel-L2={l2_rel:.3e}");
            // The epilogue rides in the instruction stream, so gelu must be shown to BEAT the modes
            // the rail could have dispatched instead -- one residual alone does not say which ran.
            let mut worst = f32::INFINITY;
            for (name, ctl) in &controls {
                let (_, r) = rel_err(ctl, &dev);
                println!("[fused_seam_parity]   control {name}: rel-L2={r:.3e} ({:.1}x gelu)", r / l2_rel.max(1e-30));
                worst = worst.min(r);
            }
            assert!(l2_rel <= 5e-2, "K=768 GELU rail chain parity FAILED: rel-L2 {l2_rel:.3e} > 5e-2");
            assert!(worst > 4.0 * l2_rel, "K=768 rail control did NOT fire: nearest wrong epilogue is only {:.1}x off (rel-L2 {worst:.3e} vs {l2_rel:.3e}) -- the run does not show gelu was the mode dispatched", worst / l2_rel.max(1e-30));
            println!("[fused_seam_parity] PASS (rel-L2 <= 5e-2, gelu {:.1}x clear of the nearest control)", worst / l2_rel.max(1e-30));
        }
        "k768resid" => {
            let (host, dev, controls) = npu.k768_ffn_resid_selftest(t, seed).unwrap_or_else(|| {
                panic!("[fused_seam_parity] k768resid: K=768 rail xclbins absent -- build \
                        PAD_M=<width> scripts/build_k768_gelu_rail.sh into artifacts/k768_gelu_rail");
            });
            let (max_rel, l2_rel) = rel_err(&host, &dev);
            println!("[fused_seam_parity] seam=k768resid t={t} seed={seed}  max_rel={max_rel:.3e} rel-L2={l2_rel:.3e}");
            // The control here is the PRE-norm bug itself: a rail that residuals fc1's input instead
            // of the separate operand reproduces `swapped-residual` exactly, and no residual against
            // the correct oracle alone would distinguish the two.
            let mut worst = f32::INFINITY;
            for (name, ctl) in &controls {
                let (_, r) = rel_err(ctl, &dev);
                println!("[fused_seam_parity]   control {name}: rel-L2={r:.3e} ({:.1}x correct)", r / l2_rel.max(1e-30));
                worst = worst.min(r);
            }
            assert!(l2_rel <= 5e-2, "K=768 rail pre-norm parity FAILED: rel-L2 {l2_rel:.3e} > 5e-2");
            assert!(worst > 4.0 * l2_rel, "K=768 rail residual control did NOT fire: residualling fc1's input is only {:.1}x off -- the run does not show the separate operand was used", worst / l2_rel.max(1e-30));
            println!("[fused_seam_parity] PASS (rel-L2 <= 5e-2, correct operand {:.1}x clear of the swapped one)", worst / l2_rel.max(1e-30));
        }
        other => {
            eprintln!("[fused_seam_parity] unknown seam '{other}' (known: ffn, residual, ln, linout, convfront, k768ffn, k768resid)");
            std::process::exit(2);
        }
    }
}
