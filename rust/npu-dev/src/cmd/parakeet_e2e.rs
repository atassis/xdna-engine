//! Full Parakeet-tdt ASR (mel + NPU encoder + TDT decode/joint) latency bench with per-stage
//! breakdown -- the TDT-decode-cost measurement task, mirroring `whisper_e2e`'s shape.
//!
//! Usage: PARAKEET_PHASE_TIMING=1 [PASSES=10] parakeet_e2e <clip.wav> [scenario.toml]

use std::path::Path;
use std::time::Instant;

use npu_engine::pipeline::Scenario;
use npu_engine::registry;

const SCENARIO_DEFAULT: &str = "scenarios/asr-parakeet-tdt.toml";

pub fn run(argv: Vec<String>) {
    let wav_path = argv.get(1).cloned().expect("usage: parakeet_e2e <clip.wav> [scenario.toml]");
    let scenario = argv.get(2).cloned().unwrap_or_else(|| SCENARIO_DEFAULT.into());
    let passes: usize = std::env::var("PASSES").ok().and_then(|s| s.parse().ok()).unwrap_or(10);

    let bytes = std::fs::read(&wav_path).unwrap_or_else(|e| panic!("read {wav_path}: {e}"));
    let samples = parse_wav_i16(&bytes).expect("parse 16k/mono/16-bit WAV");
    let dur_s = samples.len() as f64 / 16_000.0;
    eprintln!(
        "[bench] clip={wav_path} scenario={scenario} samples={} duration_s={dur_s:.3} passes={passes}",
        samples.len()
    );

    let scen = registry::build(Path::new(&scenario), Path::new("."));
    let pipe = match scen {
        Scenario::Asr(p) => p,
        _ => panic!("scenario is not ASR"),
    };

    eprintln!("[bench] --- warmup pass (untimed) ---");
    let warm = pipe.transcribe(&samples).expect("transcribe");
    eprintln!("[bench] warmup text: {warm:?}");

    let mut e2e_ms = Vec::with_capacity(passes);
    let mut last_report: Option<npu_parakeet::prof::phase::PhaseReport> = None;
    let mut last_ttft: Option<f64> = None;
    for p in 0..passes {
        npu_parakeet::prof::phase::reset();
        let t0 = Instant::now();
        let text = pipe.transcribe(&samples).expect("transcribe");
        let dt = t0.elapsed();
        let r = npu_parakeet::prof::phase::report(dt);
        eprintln!(
            "[pass {}/{passes}] e2e {:.1} ms  text={text:?}",
            p + 1,
            dt.as_secs_f64() * 1000.0
        );
        e2e_ms.push(dt.as_secs_f64() * 1000.0);
        last_ttft = npu_parakeet::prof::phase::first_token_ms();
        last_report = Some(r);
    }

    let n = e2e_ms.len() as f64;
    let mean = e2e_ms.iter().sum::<f64>() / n;
    let min = e2e_ms.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = e2e_ms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    eprintln!(
        "\n[summary] n={} mean={mean:.1}ms min={min:.1}ms max={max:.1}ms  per-pass={:?}",
        e2e_ms.len(),
        e2e_ms.iter().map(|v| format!("{v:.0}")).collect::<Vec<_>>()
    );
    if let Some(t) = last_ttft {
        eprintln!("[summary] time-to-first-token (last pass) = {t:.1} ms");
    } else {
        eprintln!("[summary] time-to-first-token: no non-blank token recorded (unexpected)");
    }

    if let Some(r) = last_report {
        let e2e = r.e2e_ms;
        eprintln!(
            "\n[summary] last-pass phase buckets: e2e {e2e:.1} ms = npu {:.1} + host {:.1} + marshal {:.1} \
             | residual {:.1} overlap {:.1}",
            r.npu_ms, r.host_ms, r.marshal_ms, r.residual_ms, r.overlap_ms
        );
        for (stage, bucket, ms, calls) in r.rows.iter().take(30) {
            eprintln!("  {stage:<22} {:<8} {ms:9.2} ms  x{calls}", format!("{bucket:?}"));
        }
        // Derived: tdt_run_dj is one call per encoder frame visited by the decode loop (blank or
        // not); tdt_token_emitted is a zero-duration counter, one per REAL (non-blank) token.
        let find = |name: &str| r.rows.iter().find(|(s, _, _, _)| s == name).cloned();
        if let Some((_, _, ms, calls)) = find("tdt_run_dj") {
            eprintln!(
                "\n[summary] joint calls = {calls}  tdt_run_dj total = {ms:.2} ms  ({:.4} ms/call)",
                ms / calls as f64
            );
        }
        if let Some((_, _, _, tok_calls)) = find("tdt_token_emitted") {
            if let Some((_, _, dec_ms, _)) = find("tdt_decode") {
                eprintln!(
                    "[summary] tokens emitted = {tok_calls}  tdt_decode wall = {dec_ms:.2} ms  \
                     ({:.3} ms/token)",
                    dec_ms / tok_calls as f64
                );
            }
        }
        // preproc/detok sit outside the top-30 stage dump once the encoder's own per-op scopes
        // fill it, so name them explicitly -- these ARE reported (part of r.host_ms) either way.
        for name in ["preproc", "detok"] {
            if let Some((_, bucket, ms, calls)) = find(name) {
                eprintln!("[summary] {name:<10} {:<8} {ms:9.3} ms  x{calls}", format!("{bucket:?}"));
            }
        }
    }
}

/// Parse a 16 kHz / mono / 16-bit PCM WAV into little-endian i16 samples. Mirrors
/// `whisper_e2e::parse_wav_i16` (kept in sync -- same front-end format contract).
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
                fmt_ok = (audio_fmt == 1 || audio_fmt == 0xFFFE)
                    && bits == 16
                    && channels == 1
                    && rate == 16_000;
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
