//! M=1 reference WER driver: transcribe clips one-at-a-time through the shipped single-stream
//! fused decode path (`WhisperAsr::transcribe`, NPU_DECODE_FUSED) and print `path<TAB>text`. The
//! batched decoder's per-stream output is identical to this by construction (+ the replicate test),
//! so this is the authoritative apples-to-apples baseline for the batched WER gate.
//!
//! Usage: NPU_DECODE_FUSED=1 NPU_DECODE_FUSED_DIR=artifacts/fused_decode12 \
//!        LD_LIBRARY_PATH=... wer_m1_decode <clip1.wav> ...   (single-tenant NPU — stop services first)

use std::path::Path;

use npu_engine::asr::whisper::WhisperAsr;
use npu_engine::config::ScenarioConfig;
use npu_engine::pipeline::AsrModel;

const SCENARIO: &str = "scenarios/asr-whisper-small.toml";

fn main() {
    let paths: Vec<String> = std::env::args().skip(1).collect();
    assert!(!paths.is_empty(), "usage: wer_m1_decode <clip1.wav> ...");
    let cfg = ScenarioConfig::load(Path::new(SCENARIO)).expect("load scenario");
    let asr = WhisperAsr::build(&cfg, Path::new("."));
    for p in &paths {
        let samples = parse_wav_i16(&std::fs::read(p).unwrap_or_else(|e| panic!("read {p}: {e}")))
            .expect("parse 16k/mono/16-bit WAV");
        let txt = asr.transcribe(&samples);
        println!("{p}\t{txt}");
    }
}

/// 16 kHz / mono / 16-bit PCM WAV -> i16 (mirrors verify_batched_decode::parse_wav_i16).
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
