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
        FastConformerEncoder::new(artifacts, cfg)
    } else {
        let root = std::env::var("NPU_XCLBIN_ROOT")
            .unwrap_or_else(|_| "$REPO".into());
        println!("[npu] xclbin root = {root}");
        #[cfg(feature = "npu")]
        { FastConformerEncoder::new_npu(artifacts, cfg, Path::new(&root)) }
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
    for p in &names {
        let mel = load_mel(p);
        let t0 = Instant::now();
        let enc_out = enc.encode(&mel); // [T', 1024]
        let dt = t0.elapsed().as_secs_f64();
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
}
