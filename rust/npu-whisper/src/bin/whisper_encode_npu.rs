//! Part 2 (P3): encode whisper-small mel features through the XDNA2 NPU encoder.
//! Reads mel `.npy` ([1,80,3000] or [80,3000]) from --mels <dir>, runs the NPU encoder
//! (`forward_last` = conv stem + 12 pre-norm blocks + ln_post), writes encoded hidden
//! states `[1500,768]` to --out <dir>/<name>.npy, prints per-clip encode wall time.
//!
//! The encoder is built ONCE outside the loop and reused — the NPU op cache is per-encoder.
//!
//! Single-tenant NPU: stop voxd.service first; hold /dev/accel/accel0. Run from the worktree
//! ROOT so the `mlir-aie/.../whole_array/build` and `artifacts/` paths resolve.
//!
//! Usage:  whisper_encode_npu --mels <dir> --out <dir> [--cpu]

use std::path::Path;
use std::time::Instant;

use ndarray::prelude::*;
use ndarray_npy::{read_npy, write_npy};
use npu_whisper::config::WhisperCfg;
use npu_whisper::encoder::WhisperEncoder;

fn arg_val(args: &[String], flag: &str) -> Option<String> {
    args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1).cloned())
}

/// Load a mel `[1,80,3000]` (squeeze) or `[80,3000]` -> `[80,3000]`.
fn load_mel(p: &Path) -> Array2<f32> {
    let a: ArrayD<f32> = read_npy(p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
    match a.ndim() {
        2 => a.into_dimensionality::<Ix2>().unwrap(),
        3 => a.index_axis(Axis(0), 0).to_owned().into_dimensionality::<Ix2>().unwrap(),
        n => panic!("mel ndim {n} unexpected (want [80,3000] or [1,80,3000])"),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mel_dir = arg_val(&args, "--mels").expect("--mels <dir> required");
    let out_dir = arg_val(&args, "--out").expect("--out <dir> required");
    let cpu = args.iter().any(|a| a == "--cpu");
    let mel_dir = Path::new(&mel_dir);
    let out_dir = Path::new(&out_dir);
    std::fs::create_dir_all(out_dir).unwrap();

    let artifacts = Path::new("artifacts/whisper-small");

    // Build the encoder ONCE and reuse across all clips (per-encoder NPU op cache).
    let enc = if cpu {
        println!("[cpu] host f32 encoder");
        WhisperEncoder::new(artifacts, WhisperCfg::SMALL)
    } else {
        #[cfg(feature = "npu")]
        {
            println!("[npu] root = . (cwd)");
            WhisperEncoder::new_npu(artifacts, WhisperCfg::SMALL, Path::new("."))
        }
        #[cfg(not(feature = "npu"))]
        {
            panic!("built without --features npu (pass --cpu, or rebuild with --features npu)");
        }
    };

    let mut names: Vec<_> = std::fs::read_dir(mel_dir).unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|x| x.to_str()) == Some("npy"))
        .collect();
    names.sort();

    // Step 0 (resident full-NPU spec): NPU_MARSH_PROF=1 splits the encoder's NPU dispatch stream into
    // NPU-compute (stall) vs host marshaling (round-trip). Reset the cumulative profilers before the
    // timed loop; dump the per-pass average after. No-op (cheap recording only) when the var is unset.
    #[cfg(feature = "npu")]
    if !cpu {
        npu_asr::engines::reset_prof();
        npu_asr::engines::marsh::reset();
    }

    let mut total = 0f64;
    let mut n = 0;
    for p in &names {
        let mel = load_mel(p); // [80,3000]
        let t0 = Instant::now();
        let enc_out = enc.forward_last(&mel); // [1500,768]
        let dt = t0.elapsed().as_secs_f64();
        total += dt;
        n += 1;
        let stem = p.file_stem().unwrap().to_string_lossy();
        write_npy(out_dir.join(format!("{stem}.npy")), &enc_out).unwrap();
        println!("[enc] {stem}  [{}, {}]  {:.3}s", enc_out.nrows(), enc_out.ncols(), dt);
    }
    println!("\nmean encode {:.3}s/clip over {n} clips", total / n.max(1) as f64);

    #[cfg(feature = "npu")]
    if !cpu {
        npu_asr::engines::dump_dispatch_prof(n); // NPU_MARSH_PROF=1: round-trip vs stall split
    }
}
