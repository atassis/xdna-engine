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
/// Calibrated 2026-08-31 (device, k768-gelu-rail item (iv)): the two currently-shipped
/// computation paths for `block_{n-1}` measure |alpha| = 2.069e-3 (NPU_ENC_GELU_FUSED on-chip
/// GELU epilogue) and 3.426e-3 (host GELU, unfused) against the ONNX golden -- both believed
/// correct, both pass `encoded`. Set at ~2x the larger of those two, so neither shipped path
/// false-positives while still catching a regression an order of magnitude worse. A prior scale
/// bug measured 8.564e-3 on the ISOLATED epilogue kernel alone -- a narrower, dirtier-signal
/// scope than this whole-block measurement, so that figure informed the bound rather than
/// setting it. CONTROLLED 2026-08-31 with `SCALE_INJECT`, which multiplies the measured block by
/// (1+f) post-hoc so the response has a PREDICTED value: +0.01 -> predicted +1.207e-2, measured
/// +1.209e-2; +0.003 (straddling this threshold) -> predicted +5.069e-3, measured +5.076e-3;
/// -0.01 tracks in sign. So this is a demonstrated detector at its own threshold's magnitude,
/// not an inferred bound. (An earlier truncate-vs-nearest-even rebuild appeared not to diverge;
/// that was traced to a build-path artifact -- the dispatched xclbin was a different tile than
/// the one rebuilt -- and is not evidence about rounding.)
const TOL_SCALE: f32 = 5e-3;

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

/// FNV-1a 64 over an array's raw f32 bytes (LE). Dependency-free bit fingerprint for artifact-
/// identity checks: two runs producing the SAME hash ran bit-identical data, full stop, no
/// precision loss from printing rel/alpha to 3 significant figures.
fn fingerprint(a: &Array2<f32>) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for x in a.iter() {
        for b in x.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

/// Split `got - refr` into the part proportional to `refr` and the part orthogonal to it:
/// `got = (1 + alpha) * refr + r`, `<r, refr> = 0`. Returns `(alpha, ||r|| / ||refr||)`.
///
/// A LayerNorm divides by the per-token std, so it erases `alpha` and passes `r` through. A rel-L2
/// sampled downstream of one therefore cannot rank two kernels whose errors differ in `alpha` --
/// it reports the smaller `r`, which can belong to the less accurate kernel.
fn scale_split(got: &Array2<f32>, refr: &Array2<f32>) -> (f32, f32) {
    let (mut er, mut rr) = (0f64, 0f64);
    for (g, r) in got.iter().zip(refr.iter()) {
        er += ((*g as f64) - (*r as f64)) * (*r as f64);
        rr += (*r as f64) * (*r as f64);
    }
    let alpha = er / (rr + 1e-30);
    let mut num = 0f64;
    for (g, r) in got.iter().zip(refr.iter()) {
        let d = (*g as f64) - (1.0 + alpha) * (*r as f64);
        num += d * d;
    }
    (alpha as f32, (num.sqrt() / (rr.sqrt() + 1e-12)) as f32)
}

/// Fraction of the squared error that `scale_split` attributes to `alpha`.
fn scale_explained(rel: f32, rel_sf: f32) -> f32 {
    if rel <= 0.0 { 0.0 } else { (1.0 - (rel_sf / rel).powi(2)).max(0.0) }
}

fn main() {
    // Parse args: an optional positional artifacts dir + an optional `--npu` flag + an optional
    // `--turbo` flag (any order). `--turbo` selects WhisperCfg::TURBO (d_model 1280, 32 layers)
    // instead of the default SMALL -- the encoder/gate code itself is shape-agnostic (WhisperCfg
    // is data), so this is the only cfg-selection change either backend needed for the new shape.
    let mut npu = false;
    let mut turbo = false;
    let mut artifacts: Option<String> = None;
    for a in std::env::args().skip(1) {
        if a == "--npu" {
            npu = true;
        } else if a == "--turbo" {
            turbo = true;
        } else if !a.starts_with("--") {
            artifacts = Some(a);
        }
    }
    let default_dir = if turbo { "artifacts/whisper-turbo" } else { "artifacts/whisper-small" };
    let artifacts = artifacts.unwrap_or_else(|| default_dir.into());
    let cfg = if turbo { WhisperCfg::TURBO } else { WhisperCfg::SMALL };

    let (enc, tol, backend) = if npu {
        #[cfg(feature = "npu")]
        {
            // root = worktree root (cwd), where mlir-aie/.../whole_array/build resolves.
            let enc = WhisperEncoder::new_npu(Path::new(&artifacts), cfg, Path::new("."));
            (enc, TOL_NPU, "npu")
        }
        #[cfg(not(feature = "npu"))]
        {
            eprintln!("--npu requested but binary built without the `npu` feature; rebuild with --features npu");
            std::process::exit(2);
        }
    } else {
        (WhisperEncoder::new(Path::new(&artifacts), cfg), TOL_HOST, "host")
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
    let outs = enc.forward_collect(&after_conv);
    let mut worst = 0f32;
    let mut last_alpha = 0f32;
    for (i, out) in outs.iter().enumerate() {
        let rb = rel(out, &as2(w.ref_tensor(&format!("block_{i}"))));
        let (alpha, rb_sf) = scale_split(out, &as2(w.ref_tensor(&format!("block_{i}"))));
        worst = worst.max(rb);
        if i == enc.cfg.n_layers - 1 {
            last_alpha = alpha;
        }
        let pass = rb <= tol;
        let note = format!(
            "rel={rb:.3e}  alpha={alpha:+.3e}  rel_sf={rb_sf:.3e}  scale={:.1}%",
            100.0 * scale_explained(rb, rb_sf)
        );
        if i == 0 || i == enc.cfg.n_layers - 1 {
            println!("[block_{i}] {note}  {}", if pass { "OK" } else if block_gates { "FAIL" } else { "(info)" });
        } else if !pass {
            println!("[block_{i}] {note}  {}", if block_gates { "FAIL" } else { "(info)" });
        }
        if !pass && block_gates {
            fails.push(format!("block_{i}"));
        }
    }
    println!("[blocks] worst per-block rel={worst:.3e}{}", if block_gates { "" } else { "  (report-only on npu; gate is `encoded`)" });

    // ---- gate 3: full encoder (last block THEN ln_post) vs encoded ----
    let encoded = enc.forward_last(&mel);
    let r_enc = rel(&encoded, &as2(w.ref_tensor("encoded")));
    let (a_enc, r_enc_sf) = scale_split(&encoded, &as2(w.ref_tensor("encoded")));
    println!(
        "[encoded] forward_last(mel) vs encoded:  rel={r_enc:.3e}  alpha={a_enc:+.3e}  rel_sf={r_enc_sf:.3e}  {}",
        if r_enc <= tol { "OK" } else { "FAIL" }
    );
    if r_enc > tol {
        fails.push("encoded".into());
    }

    // ---- gate 4: scale companion for gate 3 ----
    // `encoded` is sampled after `ln_post`, which divides out any error proportional to the signal,
    // so gate 3 is blind to a systematic shrink and will rank a biased kernel above an unbiased one
    // with twice the accuracy. `block_{n-1}` is the raw residual stream feeding `ln_post` -- the last
    // point where that error class is still visible -- so gate its `alpha` directly.
    println!(
        "[scale] block_{} alpha={last_alpha:+.3e}  |alpha| <= {TOL_SCALE:.1e}  {}",
        enc.cfg.n_layers - 1,
        if last_alpha.abs() <= TOL_SCALE { "OK" } else { "FAIL" }
    );
    if last_alpha.abs() > TOL_SCALE {
        fails.push("scale".into());
    }

    // Artifact-identity fingerprint for round-trip rounding/kernel experiments: prints ALWAYS
    // (not gated), so a rebuild-and-redispatch control can diff two runs' printed lines directly
    // rather than needing a separate file dump. FNV-1a over block_{n-1}'s raw f32 bytes.
    println!(
        "[fingerprint] block_{} = {:016x}",
        enc.cfg.n_layers - 1,
        fingerprint(&outs[enc.cfg.n_layers - 1])
    );

    // ---- control: does gate 4 actually see a scale error? ----
    // TOL_SCALE was calibrated (2026-08-31) from two real but unverified-as-representative alpha
    // readings, because a rounding-mode device control came back bit-identical to the arm it was
    // meant to differ from (open observation, not resolved: see this task's worklog). This is the
    // control that replaces it: perturb REAL data by a KNOWN multiplicative factor -- not the
    // kernel's ambient state, so device-free, no recompile, nothing to restore -- and check that
    // `scale_split` recovers the injected factor on top of whatever real alpha was already there.
    // `SCALE_INJECT=<f32>` (e.g. 0.01) multiplies the already-computed `block_{n-1}` by (1+f)
    // before re-running scale_split against the SAME golden; to first order (f small, base alpha
    // near zero) the predicted new alpha is base_alpha + f, since scaling `got` by (1+f) adds
    // f*refr to the numerator `got-refr` and only a second-order f*(got-refr) cross term is
    // dropped. This is a stronger check than recovering an injected shrink on synthetic i.i.d.
    // data (already covered by `scale_split_recovers_an_injected_shrink`): it tests the SAME
    // real, correlated, non-Gaussian data class the gate actually runs against.
    if let Ok(s) = std::env::var("SCALE_INJECT") {
        let f: f32 = s.parse().expect("SCALE_INJECT must parse as f32, e.g. 0.01");
        let last = &outs[enc.cfg.n_layers - 1];
        let golden_last = as2(w.ref_tensor(&format!("block_{}", enc.cfg.n_layers - 1)));
        let perturbed = last.mapv(|x| x * (1.0 + f));
        let (alpha_inj, _) = scale_split(&perturbed, &golden_last);
        let predicted = last_alpha + f;
        let err = (alpha_inj - predicted).abs();
        // 5% relative to the injected factor itself (not to the tiny base alpha), since that is
        // the quantity a real scale bug of this size would move the gate by.
        let ok = err <= 0.05 * f.abs();
        println!(
            "[scale-control] SCALE_INJECT={f:+.3e}  alpha(injected)={alpha_inj:+.3e}  predicted={predicted:+.3e}  |err|={err:.3e}  {}",
            if ok { "OK -- alpha tracks the injected factor" } else { "FAIL -- gate did not see the injected scale error" }
        );
        if !ok {
            fails.push("scale-control".into());
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A residual alone cannot tell a working `scale_split` from one that always returns zero, so
    /// drive it with a known shrink plus orthogonal noise and check it recovers both terms.
    #[test]
    fn scale_split_recovers_an_injected_shrink() {
        let n = 4096;
        let refr = Array2::from_shape_fn((n, 1), |(i, _)| ((i * 2654435761) % 1000) as f32 / 500.0 - 1.0);
        for &inj in &[-1e-2f32, -1e-4, 0.0, 3e-3] {
            let got = refr.mapv(|r| r * (1.0 + inj));
            let (alpha, sf) = scale_split(&got, &refr);
            assert!((alpha - inj).abs() < 1e-6, "alpha {alpha} != injected {inj}");
            assert!(sf < 1e-6, "pure shrink must leave no orthogonal residual, got {sf}");
            assert!(scale_explained(rel(&got, &refr), sf) > 0.999 || inj == 0.0);
        }
    }

    /// Noise orthogonal to the signal must land entirely in `rel_sf`, not in `alpha`.
    #[test]
    fn orthogonal_noise_does_not_move_alpha() {
        let n = 4096;
        let refr = Array2::from_shape_fn((n, 1), |(i, _)| (i as f32 * 0.001).sin());
        let noise = Array2::from_shape_fn((n, 1), |(i, _)| (i as f32 * 1.7).cos() * 0.01);
        let got = &refr + &noise;
        let (alpha, sf) = scale_split(&got, &refr);
        assert!(alpha.abs() < 1e-3, "alpha {alpha} should be ~0 for orthogonal noise");
        assert!((sf - rel(&got, &refr)).abs() / sf < 1e-2, "rel_sf should track rel");
    }
}
