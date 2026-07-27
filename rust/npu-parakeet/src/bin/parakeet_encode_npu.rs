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

    // INTERLEAVED A/B (`--ab VAR`). Runs BOTH arms on every clip, alternating which goes first, and
    // attributes each tag to the arm that produced it.
    //
    // This exists because sequential same-session A/B does not survive this device's drift:
    // conveyor-dispatch-session-drift records the IDENTICAL xclbin measuring 4.305 ms/dispatch early
    // in a session and 6.917 ms ~40 minutes later (+61%) with nothing changed. Comparing arm A now
    // against arm B in ten minutes cannot distinguish a real 10% win from drift, and "normalise
    // against the untouched kernels" assumes drift is a smooth scale factor -- which a 61% swing
    // says it need not be. Alternating arm order per clip cancels any drift that is monotone over
    // the run, and cancels first-vs-second-position bias with it.
    #[cfg(feature = "npu")]
    let ab_var: Option<String> = args.iter().position(|a| a == "--ab")
        .and_then(|i| args.get(i + 1)).cloned();
    #[cfg(feature = "npu")]
    if let Some(var) = ab_var {
        run_interleaved_ab(&enc, &names, out_dir, &var);
        return;
    }

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
    // PHASE SPLIT (PARAKEET_PHASE_TIMING=1). encoder.rs already brackets every stage with a
    // PhaseScope, but nothing printed the accumulator, so the Npu/Host/Marshal attribution the
    // instrumentation exists to produce was invisible -- and `prof::report()` above only covers the
    // two labels wrapped in prof::time (subsample, dwconv), which leaves most of the wall
    // unattributed. Same failure mode as the untimed dispatch sites: the instrument was there and
    // not being read.
    if npu_parakeet::prof::phase::timing_on() {
        let r = npu_parakeet::prof::phase::report(std::time::Duration::from_secs_f64(total));
        println!(
            "\nphase split over {n} clips: e2e={:.0}ms npu={:.0}ms host={:.0}ms marshal={:.0}ms residual={:.0}ms overlap={:.0}ms",
            r.e2e_ms, r.npu_ms, r.host_ms, r.marshal_ms, r.residual_ms, r.overlap_ms
        );
        for (stage, bucket, ms, calls) in r.rows.iter().take(24) {
            println!("  {stage:22} {:8} {ms:8.1}ms  x{calls}", format!("{bucket:?}"));
        }
    }
}

/// Interleaved A/B over the clip set: every clip runs BOTH arms, alternating which goes first.
///
/// arm OFF = `var` unset, arm ON = `var=1`. The flags this targets (PARAKEET_RESADD_LN,
/// PARAKEET_FUSED_BLOCK, ...) are read via `std::env::var` at each use rather than cached, so
/// flipping the process env between encodes actually switches the arm.
///
/// Clip 0 runs both arms but is EXCLUDED from the aggregates -- first-touch pays weight-BO upload
/// and hw-context load, which would otherwise land entirely on whichever arm happened to go first.
#[cfg(feature = "npu")]
fn run_interleaved_ab(
    enc: &npu_parakeet::encoder::FastConformerEncoder,
    names: &[std::path::PathBuf],
    out_dir: &Path,
    var: &str,
) {
    use std::collections::BTreeMap;
    type Tags = BTreeMap<&'static str, (usize, f64)>;
    let mut wall = [0f64; 2];
    let mut acc: [Tags; 2] = [Tags::new(), Tags::new()];
    let mut counted = 0usize;

    let mut one = |enc: &npu_parakeet::encoder::FastConformerEncoder, mel: &ndarray::Array2<f32>, on: bool| {
        if on { std::env::set_var(var, "1"); } else { std::env::remove_var(var); }
        let before = enc.npu_tag_snapshot();
        let t0 = Instant::now();
        let out = enc.encode(mel);
        let dt = t0.elapsed().as_secs_f64();
        let after = enc.npu_tag_snapshot();
        let mut delta: Tags = Tags::new();
        for (k, (n1, t1)) in after {
            let (n0, t0b) = before.get(k).copied().unwrap_or((0, 0.0));
            if n1 > n0 { delta.insert(k, (n1 - n0, t1 - t0b)); }
        }
        (dt, delta, out)
    };

    for (i, p) in names.iter().enumerate() {
        let mel = load_mel(p);
        // alternate which arm runs first so position bias cancels along with drift
        let order = if i % 2 == 0 { [false, true] } else { [true, false] };
        let mut saved = None;
        for &on in &order {
            let (dt, delta, out) = one(enc, &mel, on);
            if i > 0 {
                let a = on as usize;
                wall[a] += dt;
                for (k, (n, t)) in delta {
                    let e = acc[a].entry(k).or_insert((0, 0.0));
                    e.0 += n; e.1 += t;
                }
            }
            if on { saved = Some(out); }
        }
        if i > 0 { counted += 1; }
        let stem = p.file_stem().unwrap().to_string_lossy();
        write_npy(out_dir.join(format!("{stem}.npy")), &saved.unwrap()).unwrap();
    }
    std::env::remove_var(var);

    println!("\n=== INTERLEAVED A/B on {var} -- {counted} clips, both arms per clip, order alternated");
    println!("clip 0 excluded (first-touch weight upload + context load)");
    for (a, label) in [(0usize, "OFF"), (1usize, "ON ")] {
        println!("  arm {label}  mean wall {:.3} s/clip", wall[a] / counted as f64);
    }
    let d = (wall[1] - wall[0]) / counted as f64;
    println!("  wall delta ON-OFF: {d:+.4} s/clip ({:+.1}%)", d / (wall[0] / counted as f64) * 100.0);

    println!("\n  {:<20}{:>12}{:>12}{:>10}{:>12}", "tag", "OFF ms/cmd", "ON ms/cmd", "delta%", "cmds OFF/ON");
    let mut keys: Vec<&'static str> = acc[0].keys().chain(acc[1].keys()).copied().collect();
    keys.sort(); keys.dedup();
    for k in keys {
        let (n0, t0) = acc[0].get(k).copied().unwrap_or((0, 0.0));
        let (n1, t1) = acc[1].get(k).copied().unwrap_or((0, 0.0));
        let p0 = if n0 > 0 { t0 / n0 as f64 * 1000.0 } else { f64::NAN };
        let p1 = if n1 > 0 { t1 / n1 as f64 * 1000.0 } else { f64::NAN };
        let pct = if p0.is_finite() && p1.is_finite() { (p1 - p0) / p0 * 100.0 } else { f64::NAN };
        println!("  {k:<20}{p0:>12.3}{p1:>12.3}{pct:>9.1}%{:>12}", format!("{n0}/{n1}"));
    }
}
