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
    let npu = NpuMatmul::open(Path::new(&root));

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
        other => {
            eprintln!("[fused_seam_parity] unknown seam '{other}' (known: ffn, residual, ln, linout, convfront)");
            std::process::exit(2);
        }
    }
}
