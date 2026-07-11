//! Parakeet phase-timing bench driver (r1 Phase-0 encoder timing-breakout, Task 5).
//!
//! Builds the shipped Parakeet ASR once (warm weights), then for each clip runs one discarded
//! warmup pass + N=3 measured passes. Each measured pass resets the thread-local phase profiler
//! (`npu_parakeet::prof::phase`), times the whole `transcribe()` with an outer `Instant`, and
//! folds the resulting `PhaseReport`. Also samples RAPL package energy (best-effort) across the
//! measured passes. Prints a ranked per-stage table, one greppable `[PARAKEET_PHASE]` line per
//! clip, and a 100%-attribution check.
//!
//! Requires `PARAKEET_PHASE_TIMING=1` to be set for the buckets to be non-zero (Task 6 sets it).
//!
//! Usage (single-tenant NPU -- stop other ASR/embeddings services first):
//!   PARAKEET_PHASE_TIMING=1 LD_LIBRARY_PATH=... \
//!     bench_parakeet_phase [clip1.wav clip2.wav ...]
//! Defaults to artifacts/wer_clips/en_01.wav + artifacts/wer_clips/ru_01.wav. Run from repo root
//! (artifacts/ + resident xclbins are resolved relative to CWD unless NPU_XCLBIN_ROOT is set).

use std::path::Path;
use std::time::Instant;

use npu_engine::asr::parakeet::ParakeetAsr;
use npu_engine::config::ScenarioConfig;
use npu_engine::pipeline::AsrModel;
use npu_parakeet::prof::phase::{self, Bucket, PhaseReport};

const SCENARIO: &str = "scenarios/asr.toml";
const N_PASSES: usize = 3;
const DEFAULT_CLIPS: &[&str] = &["artifacts/wer_clips/en_01.wav", "artifacts/wer_clips/ru_01.wav"];

fn main() {
    if !phase::timing_on() {
        eprintln!(
            "HINT: PARAKEET_PHASE_TIMING is not set -- phase buckets will all read 0.0ms. \
             Re-run with `PARAKEET_PHASE_TIMING=1` to get the breakdown."
        );
    }

    let mut clips: Vec<String> = std::env::args().skip(1).collect();
    if clips.is_empty() {
        clips = DEFAULT_CLIPS.iter().map(|s| s.to_string()).collect();
    }

    let cfg = ScenarioConfig::load(Path::new(SCENARIO)).expect("load scenario scenarios/asr.toml");
    // Build once so the weights/resident xclbins are warm for every clip. `ParakeetAsr::build`
    // opens its own NPU device and reads artifacts relative to `root` (CWD = repo root) unless
    // NPU_XCLBIN_ROOT overrides the xclbin location -- same contract as the other NPU bins.
    let asr = ParakeetAsr::build(&cfg, Path::new("."));
    eprintln!("[bench] model built; {} clip(s), {N_PASSES} measured pass(es) each", clips.len());

    for clip in &clips {
        let name = Path::new(clip)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or(clip.as_str())
            .to_string();
        let samples = parse_wav_i16(
            &std::fs::read(clip).unwrap_or_else(|e| panic!("read {clip}: {e}")),
        )
        .expect("parse 16k/mono/16-bit WAV");

        // Warmup pass (discarded for timing): pages in kernels/caches so pass-1 is not an
        // outlier. We also print its transcript so Task 6 can run the WER-neutral gate (confirm
        // instrumentation does not alter the text) without a separate binary.
        let warm_txt = asr.transcribe(&samples);
        println!("[PARAKEET_TEXT] clip={name} :: {warm_txt}");

        let mut reports: Vec<PhaseReport> = Vec::with_capacity(N_PASSES);
        let pkg_before = rapl_energy_uj();
        for _ in 0..N_PASSES {
            phase::reset();
            let t0 = Instant::now();
            let _txt = asr.transcribe(&samples);
            let e2e = t0.elapsed();
            reports.push(phase::report(e2e));
        }
        let pkg_after = rapl_energy_uj();
        let pkg_j = pkg_delta_joules(pkg_before, pkg_after);

        print_clip(&name, &reports, pkg_j);
    }
}

/// Per-clip aggregation + printout.
fn print_clip(name: &str, reports: &[PhaseReport], pkg_j: Option<f64>) {
    let n = reports.len();
    let e2e = reports.iter().map(|r| r.e2e_ms).collect::<Vec<_>>();
    let npu = reports.iter().map(|r| r.npu_ms).collect::<Vec<_>>();
    let host = reports.iter().map(|r| r.host_ms).collect::<Vec<_>>();
    let marshal = reports.iter().map(|r| r.marshal_ms).collect::<Vec<_>>();
    let overlap = reports.iter().map(|r| r.overlap_ms).collect::<Vec<_>>();
    let residual = reports.iter().map(|r| r.residual_ms).collect::<Vec<_>>();

    let (e2e_m, e2e_s) = mean_std(&e2e);
    let (npu_m, npu_s) = mean_std(&npu);
    let (host_m, host_s) = mean_std(&host);
    let (marshal_m, marshal_s) = mean_std(&marshal);
    let (overlap_m, overlap_s) = mean_std(&overlap);
    let (residual_m, _residual_s) = mean_std(&residual);

    println!("\n==== clip={name}  (mean +/- stdev over {n} pass(es)) ====");
    println!(
        "  e2e={e2e_m:8.3} +/-{e2e_s:6.3}  npu={npu_m:8.3} +/-{npu_s:6.3}  \
         host={host_m:8.3} +/-{host_s:6.3}  marshal={marshal_m:8.3} +/-{marshal_s:6.3}  \
         overlap={overlap_m:8.3} +/-{overlap_s:6.3}   [ms]"
    );

    // Ranked per-stage table. We rank by MEAN ms across passes for each (stage,bucket) key so a
    // single noisy pass cannot reorder the table; `calls` is taken from the last pass (constant
    // across passes for a fixed clip).
    println!("  {:<20} {:<8} {:>10} {:>8}", "stage", "bucket", "mean_ms", "calls");
    for (stage, bucket, ms, calls) in ranked_rows(reports) {
        println!("  {:<20} {:<8} {ms:>10.3} {calls:>8}", stage, bucket_str(bucket));
    }

    let pkg_str = match pkg_j {
        Some(j) => format!("{j:.3}"),
        None => "NA".to_string(),
    };
    // Greppable line (means).
    println!(
        "[PARAKEET_PHASE] clip={name} e2e={e2e_m:.3} npu={npu_m:.3} host={host_m:.3} \
         marshal={marshal_m:.3} overlap={overlap_m:.3} residual={residual_m:.3} pkgJ={pkg_str}"
    );

    // 100%-attribution check: residual should be < 2% of e2e. Warn loudly but never panic
    // (Task 6 investigates any real breach on-device).
    if residual_m.abs() >= 0.02 * e2e_m {
        let pct = if e2e_m > 0.0 { 100.0 * residual_m / e2e_m } else { 0.0 };
        println!("WARN: unattributed {residual_m:.3}ms ({pct:.2}%) exceeds 2% of e2e for clip={name}");
    }
}

/// Mean-rank the `(stage,bucket)` rows across all passes; returns them sorted desc by mean ms.
/// `calls` comes from whichever pass last reported that key (constant per clip).
fn ranked_rows(reports: &[PhaseReport]) -> Vec<(String, Bucket, f64, u64)> {
    use std::collections::HashMap;
    // key -> (sum_ms, count_passes, last_calls)
    let mut agg: HashMap<(String, Bucket), (f64, usize, u64)> = HashMap::new();
    for r in reports {
        for (stage, bucket, ms, calls) in &r.rows {
            let e = agg.entry((stage.clone(), *bucket)).or_insert((0.0, 0, 0));
            e.0 += *ms;
            e.1 += 1;
            e.2 = *calls;
        }
    }
    let mut rows: Vec<(String, Bucket, f64, u64)> = agg
        .into_iter()
        .map(|((stage, bucket), (sum, cnt, calls))| {
            (stage, bucket, if cnt > 0 { sum / cnt as f64 } else { 0.0 }, calls)
        })
        .collect();
    rows.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    rows
}

fn bucket_str(b: Bucket) -> &'static str {
    match b {
        Bucket::Npu => "npu",
        Bucket::Host => "host",
        Bucket::Marshal => "marshal",
    }
}

/// Population mean + stdev of a small sample. Stdev is 0.0 for n<=1.
fn mean_std(xs: &[f64]) -> (f64, f64) {
    let n = xs.len();
    if n == 0 {
        return (0.0, 0.0);
    }
    let mean = xs.iter().sum::<f64>() / n as f64;
    if n == 1 {
        return (mean, 0.0);
    }
    let var = xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n as f64;
    (mean, var.sqrt())
}

// ---- RAPL package energy (best-effort) --------------------------------------------------------

/// Read total package energy in microjoules, or None if no readable RAPL/hwmon counter is found.
/// Tries the canonical intel-rapl domain first (works on many AMD parts too), then any
/// `intel-rapl*` powercap domain, then an `amd_energy` hwmon `energy*_input` (already microjoules).
fn rapl_energy_uj() -> Option<u64> {
    // 1. Canonical powercap domain.
    if let Some(v) = read_u64("/sys/class/powercap/intel-rapl:0/energy_uj") {
        return Some(v);
    }
    // 2. Any intel-rapl* powercap domain (glob without external crates).
    if let Ok(entries) = std::fs::read_dir("/sys/class/powercap") {
        let mut names: Vec<String> = entries
            .flatten()
            .filter_map(|e| e.file_name().into_string().ok())
            .filter(|n| n.starts_with("intel-rapl"))
            .collect();
        names.sort();
        for n in names {
            if let Some(v) = read_u64(&format!("/sys/class/powercap/{n}/energy_uj")) {
                return Some(v);
            }
        }
    }
    // 3. amd_energy hwmon: energy*_input is already in microjoules.
    if let Ok(hwmons) = std::fs::read_dir("/sys/class/hwmon") {
        for hw in hwmons.flatten() {
            let dir = hw.path();
            let is_amd = read_str(&dir.join("name")).map(|s| s.trim() == "amd_energy").unwrap_or(false);
            if !is_amd {
                continue;
            }
            // Prefer the package/socket accumulator (energy1_input) if present.
            if let Some(v) = read_u64(dir.join("energy1_input").to_str().unwrap_or("")) {
                return Some(v);
            }
        }
    }
    None
}

/// Delta in joules between two energy samples. Handles a single trivial 32/64-bit wraparound: if
/// `after < before`, assume one wrap of the counter's max_energy_range and fall back to None if we
/// cannot bound it safely (rare -- Task 6 runs are short).
fn pkg_delta_joules(before: Option<u64>, after: Option<u64>) -> Option<f64> {
    let (b, a) = (before?, after?);
    if a >= b {
        return Some((a - b) as f64 / 1_000_000.0);
    }
    // Wraparound: try the domain's declared range; if unavailable, we cannot trust the delta.
    let range = read_u64("/sys/class/powercap/intel-rapl:0/max_energy_range_uj");
    match range {
        Some(max) if max > b => Some(((max - b) + a) as f64 / 1_000_000.0),
        _ => None,
    }
}

fn read_u64(path: &str) -> Option<u64> {
    if path.is_empty() {
        return None;
    }
    std::fs::read_to_string(path).ok()?.trim().parse::<u64>().ok()
}

fn read_str(path: &Path) -> Option<String> {
    std::fs::read_to_string(path).ok()
}

// ---- WAV read (16 kHz / mono / 16-bit PCM) ----------------------------------------------------

/// 16 kHz / mono / 16-bit PCM WAV -> i16 (mirrors wer_m1_decode::parse_wav_i16 -- no `hound` dep in
/// this crate, so a minimal RIFF walk keeps the bench self-contained).
fn parse_wav_i16(wav: &[u8]) -> Option<Vec<i16>> {
    if wav.len() < 12 || &wav[0..4] != b"RIFF" || &wav[8..12] != b"WAVE" {
        return None;
    }
    let mut off = 12usize;
    let mut fmt_ok = false;
    let mut data: Option<&[u8]> = None;
    while off + 8 <= wav.len() {
        let id = &wav[off..off + 4];
        let sz = u32::from_le_bytes([wav[off + 4], wav[off + 5], wav[off + 6], wav[off + 7]]) as usize;
        let body_start = off + 8;
        let body_end = body_start.saturating_add(sz).min(wav.len());
        match id {
            b"fmt " if body_end - body_start >= 16 => {
                let b = &wav[body_start..body_end];
                let audio_fmt = u16::from_le_bytes([b[0], b[1]]);
                let channels = u16::from_le_bytes([b[2], b[3]]);
                let rate = u32::from_le_bytes([b[4], b[5], b[6], b[7]]);
                let bits = u16::from_le_bytes([b[14], b[15]]);
                fmt_ok = (audio_fmt == 1 || audio_fmt == 0xFFFE) && bits == 16 && channels == 1 && rate == 16_000;
            }
            b"data" => data = Some(&wav[body_start..body_end]),
            _ => {}
        }
        off = body_start.saturating_add(sz).saturating_add(sz & 1);
    }
    if !fmt_ok {
        return None;
    }
    let data = data?;
    let n = data.len() / 2;
    Some((0..n).map(|i| i16::from_le_bytes([data[i * 2], data[i * 2 + 1]])).collect())
}
