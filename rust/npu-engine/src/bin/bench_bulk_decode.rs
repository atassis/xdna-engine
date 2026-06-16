//! Subsystem-B O3 bench: length-bucketed bulk decode of N clips (N may exceed B). Compares
//! length-SORTED bucketing vs UNSORTED (input-order) bucketing over the same N clips, reporting
//! per-mode decode time, tok/s, utilisation (real_tokens / Σ steps×B), and RAPL package energy.
//! Asserts both modes produce identical per-clip token ids (correctness — bucketing only reorders).
//!
//!   NPU_DECODE_FUSED_BATCH=1 NPU_DECODE_FUSED_BATCH_DIR=<dir> bench_bulk_decode <clip1> <clip2> ...
//! Single-tenant NPU — stop npu-asr/voxd first. Pass N>B clips (e.g. the 16 wer_clips twice = 32).

use std::path::Path;

use npu_engine::asr::whisper::WhisperAsr;
use npu_engine::config::ScenarioConfig;

const SCENARIO: &str = "scenarios/asr-whisper-small.toml";
const RAPL_PKG: &str = "/sys/class/powercap/intel-rapl:0/energy_uj";
const RAPL_MAX: &str = "/sys/class/powercap/intel-rapl:0/max_energy_range_uj";

fn rapl_uj(p: &str) -> Option<u128> {
    std::fs::read_to_string(p).ok()?.trim().parse().ok()
}
fn uj_delta(b: u128, a: u128, max: u128) -> u128 {
    if a >= b { a - b } else { a + max - b }
}
fn ntok(ids: &[Vec<i64>]) -> usize {
    ids.iter().map(|v| v.len().saturating_sub(4)).sum()
}

fn main() {
    let paths: Vec<String> = std::env::args().skip(1).collect();
    assert!(paths.len() >= 2, "usage: bench_bulk_decode <clip1.wav> <clip2.wav> ... (N clips, N>B to show the win)");
    let samples: Vec<Vec<i16>> = paths
        .iter()
        .map(|p| parse_wav_i16(&std::fs::read(p).unwrap_or_else(|e| panic!("read {p}: {e}"))).expect("parse WAV"))
        .collect();
    let refs: Vec<&[i16]> = samples.iter().map(|v| v.as_slice()).collect();
    let sort_key: Vec<usize> = samples.iter().map(|s| s.len()).collect();

    let cfg = ScenarioConfig::load(Path::new(SCENARIO)).expect("load scenario");
    let asr = WhisperAsr::build(&cfg, Path::new("."));
    let n = refs.len();
    eprintln!("[bulk] encoding {n} clips (sequential NPU encoder)...");
    let (encs, _prep, _enc) = asr.encode_clips_timed(&refs);
    let rmax = rapl_uj(RAPL_MAX).unwrap_or(u128::MAX);
    let b = asr.batch_width().expect("batched decoder B");
    let nbuckets = n.div_ceil(b);

    // warmup
    let _ = asr.decode_bulk_ids(&encs, &sort_key, false);

    let run = |sort: bool| {
        let e0 = rapl_uj(RAPL_PKG);
        let t0 = std::time::Instant::now();
        let (ids, slots) = asr.decode_bulk_ids(&encs, &sort_key, sort);
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        let j = match (e0, rapl_uj(RAPL_PKG)) {
            (Some(a), Some(c)) => Some(uj_delta(a, c, rmax) as f64 / 1e6),
            _ => None,
        };
        (ids, slots, ms, j)
    };

    let (un_ids, un_slots, un_ms, un_j) = run(false);
    let (so_ids, so_slots, so_ms, so_j) = run(true);

    // correctness: bucketing only reorders — per-clip ids must be identical between modes.
    assert_eq!(un_ids.len(), so_ids.len());
    let identical = un_ids.iter().zip(&so_ids).all(|(a, c)| a == c);

    let real = ntok(&un_ids);
    let row = |label: &str, ms: f64, slots: usize, j: Option<f64>| {
        let util = 100.0 * real as f64 / (slots.max(1) as f64);
        let toks = real.max(1) as f64;
        let jpt = j.map(|x| format!("{:.4}", x / toks)).unwrap_or_else(|| "n/a".into());
        println!(
            "  {label:<10} decode_ms={ms:8.1}  real_tok={real:4}  slots={slots:5}  util={util:5.1}%  tok/s={:6.1}  J/tok={jpt}",
            toks / (ms / 1e3),
        );
    };
    println!("\n=== O3 length-bucketed BULK decode: N={n} clips, B={b}, {nbuckets} buckets ===");
    row("unsorted", un_ms, un_slots, un_j);
    row("sorted", so_ms, so_slots, so_j);
    let _ = (un_j, so_j);
    let spd = un_ms / so_ms;
    println!("\n  sorted vs unsorted: {spd:.2}x decode wall; util {:.1}% -> {:.1}%; per-clip ids identical: {identical}",
        100.0 * real as f64 / un_slots.max(1) as f64, 100.0 * real as f64 / so_slots.max(1) as f64);
    if !identical {
        eprintln!("[bulk] FATAL: sorted/unsorted produced different ids — bucketing reorder bug");
        std::process::exit(1);
    }
}

/// 16 kHz / mono / 16-bit PCM WAV -> i16.
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
                let bb = &wav[body_start..body_end];
                let audio_fmt = u16::from_le_bytes([bb[0], bb[1]]);
                let channels = u16::from_le_bytes([bb[2], bb[3]]);
                let rate = u32::from_le_bytes([bb[4], bb[5], bb[6], bb[7]]);
                let bits = u16::from_le_bytes([bb[14], bb[15]]);
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
    let nn = data.len() / 2;
    Some((0..nn).map(|i| i16::from_le_bytes([data[i * 2], data[i * 2 + 1]])).collect())
}
