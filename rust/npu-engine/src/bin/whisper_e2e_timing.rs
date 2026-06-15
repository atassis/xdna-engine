//! End-to-end Whisper-small ASR latency bench with per-stage breakdown.
//!
//! Builds the whisper-small pipeline (same `registry::build` + `scenarios/asr-whisper-small.toml`
//! as `engine_serve`), loads a WAV from argv, then runs `transcribe` once (warmup) + 3 timed passes.
//! Per-stage timing is printed by `WhisperAsr::transcribe` itself when `WHISPER_TIMING` is set
//! (e2e / preproc / encoder / decode ms, #tokens, ms/token, dispatches/token).
//!
//! Backend select: default = ONNX decoder; `NPU_DECODE=1` = on-NPU decoder. Both use the NPU encoder
//! (single-tenant — stop npu-asr.service / voxd.service first).
//!
//! Usage: WHISPER_TIMING=1 [NPU_DECODE=1] whisper_e2e_timing <clip.wav>

use std::path::Path;

use npu_engine::pipeline::Scenario;
use npu_engine::registry;

const SCENARIO: &str = "scenarios/asr-whisper-small.toml";
const PASSES: usize = 3;

fn main() {
    let wav_path = std::env::args().nth(1).expect("usage: whisper_e2e_timing <clip.wav>");

    let bytes = std::fs::read(&wav_path).unwrap_or_else(|e| panic!("read {wav_path}: {e}"));
    let samples = parse_wav_i16(&bytes).expect("parse 16k/mono/16-bit WAV");
    let dur_s = samples.len() as f64 / 16_000.0;
    let backend = if std::env::var("NPU_DECODE").is_ok() { "NPU" } else { "ONNX" };
    eprintln!(
        "[bench] clip={wav_path} samples={} duration_s={dur_s:.3} backend={backend} passes={PASSES}",
        samples.len()
    );

    let scen = registry::build(Path::new(SCENARIO), Path::new("."));
    let pipe = match scen {
        Scenario::Asr(p) => p,
        _ => panic!("scenario is not ASR"),
    };

    // Warmup pass (not counted): primes ONNX session arenas, NPU kernels, governor, page-ins.
    eprintln!("[bench] --- warmup pass (untimed) ---");
    let warm = pipe.transcribe(&samples);
    eprintln!("[bench] warmup text: {warm:?}");

    eprintln!("[bench] --- {PASSES} timed passes ---");
    for p in 0..PASSES {
        eprintln!("[bench] timed pass {}/{PASSES}", p + 1);
        let _ = pipe.transcribe(&samples);
    }
    eprintln!("[bench] done (duration_s={dur_s:.3})");
}

/// Parse a 16 kHz / mono / 16-bit PCM WAV into little-endian i16 samples. Walks the RIFF chunk list
/// (id[4] + LE u32 size + word-aligned body) and validates `fmt ` is PCM/extensible, 16-bit, mono,
/// 16 kHz. Mirrors `engine_serve::parse_wav_i16` (kept in sync — same front-end format contract).
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
