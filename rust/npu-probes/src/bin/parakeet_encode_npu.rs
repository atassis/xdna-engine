//! Encode mel spectrograms through the Parakeet NPU encoder (Phase 4 bridge for WER/latency).
//! Reads mel `.npy` ([128,T] or [1,128,T]) from <mel_dir>, runs the NPU encoder, writes
//! encoded `.npy` ([T',1024]) to <out_dir>, and prints per-clip encode wall time.
//!
//! Single-tenant NPU (stop npu-asr/voxd, hold the flock). NPU_XCLBIN_ROOT = repo root with
//! the Parakeet xclbins (defaults to the main worktree).
//!
//! Usage:  parakeet_encode_npu <mel_dir> <out_dir> [--cpu]

use std::path::Path;
use std::time::Instant;

use ndarray::prelude::*;
use ndarray_npy::{read_npy, write_npy};
use npu_parakeet::config::ModelCfg;
use npu_parakeet::encoder::FastConformerEncoder;

fn load_mel(p: &Path) -> Array2<f32> {
    let a: ArrayD<f32> = read_npy(p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
    match a.ndim() {
        2 => a.into_dimensionality::<Ix2>().unwrap(),
        3 => a.index_axis(Axis(0), 0).to_owned().into_dimensionality::<Ix2>().unwrap(),
        n => panic!("mel ndim {n} unexpected"),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mel_dir = Path::new(&args[1]);
    let out_dir = Path::new(&args[2]);
    let cpu = args.iter().any(|a| a == "--cpu");
    std::fs::create_dir_all(out_dir).unwrap();

    let artifacts = Path::new("artifacts/parakeet/encoder");
    let cfg = ModelCfg::PARAKEET_V3;
    let enc = if cpu {
        println!("[cpu] host f32 encoder");
        FastConformerEncoder::new(artifacts, cfg).expect("build FastConformerEncoder")
    } else {
        let root = std::env::var("NPU_XCLBIN_ROOT")
            .unwrap_or_else(|_| "$REPO".into());
        println!("[npu] xclbin root = {root}");
        #[cfg(feature = "npu")]
        { FastConformerEncoder::new_npu(artifacts, cfg, Path::new(&root)).expect("build FastConformerEncoder (npu)") }
        #[cfg(not(feature = "npu"))]
        { panic!("built without --features npu") }
    };

    let mut names: Vec<_> = std::fs::read_dir(mel_dir).unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|x| x.to_str()) == Some("npy"))
        .collect();
    names.sort();

    let mut total = 0f64;
    let mut n = 0;
    let mut last_phase: Option<npu_parakeet::prof::phase::PhaseReport> = None;
    for p in &names {
        let mel = load_mel(p);
        // Reset per clip so the report below describes ONE warm clip, not the whole set.
        #[cfg(feature = "npu")]
        npu_xrt::dispatch_log::reset();
        npu_parakeet::prof::phase::reset(); // same window as dispatch_log, so the two reports join
        let t0 = Instant::now();
        let enc_out = enc.encode(&mel); // [T', 1024]
        let dt = t0.elapsed().as_secs_f64();
        last_phase = Some(npu_parakeet::prof::phase::report(t0.elapsed()));
        total += dt;
        n += 1;
        let stem = p.file_stem().unwrap().to_string_lossy();
        write_npy(out_dir.join(format!("{stem}.npy")), &enc_out).unwrap();
        println!("[enc] {stem}  T'={}  {:.3}s", enc_out.nrows(), dt);
    }
    // warm mean: exclude the first clip (cold weight-BO load) if >1 clip
    println!("\nmean encode {:.3}s/clip over {n} clips", total / n as f64);
    #[cfg(feature = "npu")]
    if let Some(s) = enc.npu_stats_string() {
        println!("{s}");
    }
    println!("host profile (desc by time):\n{}", npu_parakeet::prof::report());
    // PARAKEET_PHASE_TIMING=1: the non-overlapping bucket split for the LAST clip, on the same
    // window as the dispatch-transition report below so the two join. `residual` is wall time no
    // scope claimed; `overlap` is buckets summing past the wall clock, i.e. real concurrency.
    if let Some(r) = last_phase.filter(|r| r.npu_ms + r.host_ms + r.marshal_ms > 0.0) {
        // `ffn_*` scopes sit INSIDE ff_resident / fx_ff1_resident: they decompose the FFN rather
        // than partitioning the clip. Counting them in the totals is the same double-count that made
        // the old `fused_ff1_mhsa` wrapper read 135.7%, so split them out and subtract.
        let is_detail = |s: &str| ["ffn_", "mh_", "ss_", "cf_"].iter().any(|p| s.starts_with(p));
        // Subtract the nested detail from ITS OWN bucket -- `ss_*`/`mh_pack` are Host, `ffn_*` are Npu.
        // Taking it all off `npu` inflated host by the Host-bucketed detail and deflated npu by it.
        let det = |bk: npu_parakeet::prof::phase::Bucket| -> f64 {
            r.rows.iter()
                .filter(|(s, b, _, _)| is_detail(s) && std::mem::discriminant(b) == std::mem::discriminant(&bk))
                .map(|(_, _, ms, _)| ms).sum()
        };
        let (npu, host, marshal) = (
            r.npu_ms - det(npu_parakeet::prof::phase::Bucket::Npu),
            r.host_ms - det(npu_parakeet::prof::phase::Bucket::Host),
            r.marshal_ms - det(npu_parakeet::prof::phase::Bucket::Marshal),
        );
        let e2e = r.e2e_ms;
        println!(
            "\nphase buckets (last clip): e2e {e2e:.1} ms = npu {npu:.1} ({:.1}%) + host {host:.1} ({:.1}%) + marshal {marshal:.1} ({:.1}%) | residual {:.1} overlap {:.1}",
            100.0 * npu / e2e, 100.0 * host / e2e, 100.0 * marshal / e2e,
            (e2e - npu - host - marshal).max(0.0),
            (npu + host + marshal - e2e).max(0.0),
        );
        for (stage, bucket, ms, calls) in r.rows.iter().filter(|(s, _, _, _)| !is_detail(s)).take(24) {
            println!("  {stage:<22} {:<8} {ms:8.1} ms  x{calls}", format!("{bucket:?}"));
        }
        if r.rows.iter().any(|(s, _, _, _)| is_detail(s)) {
            println!("  -- detail, NESTED inside the rows above (not in the totals) --");
            for (stage, _, ms, calls) in r.rows.iter().filter(|(s, _, _, _)| is_detail(s)) {
                println!("  {stage:<22} {:<8} {ms:8.1} ms  x{calls}  ({:.2} ms/disp)", "nested", ms / *calls as f64);
            }
        }
    }
    // NPU_DISPATCH_LOG=1: xclbin-transition accounting for the LAST clip. 0.99 ms/switch is the
    // measured modal<->relpos hw-context-switch cost (`modal-relpos-per-switch-cost`).
    #[cfg(feature = "npu")]
    if npu_xrt::dispatch_log::enabled() {
        println!("\ndispatch transitions (last clip):\n{}", npu_xrt::dispatch_log::report(0.99));
        if let Ok(p) = std::env::var("NPU_DISPATCH_SEQ") {
            npu_xrt::dispatch_log::write_sequence(&p).unwrap();
            println!("wrote dispatch sequence -> {p}");
        }
    }
}
