//! Subsystem-B offline batched decode driver + check (lever-3 vector-b).
//!
//! Builds WhisperAsr with the batched decoder (NPU_DECODE_FUSED_BATCH + NPU_DECODE_FUSED_BATCH_DIR),
//! then transcribes B clips at once via `transcribe_batch` (one batched dispatch/step over all B
//! streams). Single-tenant NPU — stop npu-asr.service / voxd.service first.
//!
//! Usage:
//!   verify_batched_decode <clip1.wav> <clip2.wav> ...        # B clips (must == decoder B)
//!   verify_batched_decode --replicate <B> <clip.wav>         # B copies of one clip (all outputs
//!                                                              #   should be IDENTICAL == the M=1 text)
//! Env: NPU_DECODE_FUSED_BATCH=1 NPU_DECODE_FUSED_BATCH_DIR=artifacts/decode_batched_B16_L12_sp

use std::path::Path;

use npu_engine::asr::whisper::WhisperAsr;
use npu_engine::config::ScenarioConfig;

const SCENARIO: &str = "scenarios/asr-whisper-small.toml";

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (paths, samples): (Vec<String>, Vec<Vec<i16>>) = if args.first().map(|s| s.as_str()) == Some("--replicate") {
        let b: usize = args[1].parse().expect("--replicate <B>");
        let path = args[2].clone();
        let s = parse_wav_i16(&std::fs::read(&path).unwrap_or_else(|e| panic!("read {path}: {e}")))
            .expect("parse 16k/mono/16-bit WAV");
        let paths = (0..b).map(|i| format!("{path}#{i}")).collect();
        let samples = (0..b).map(|_| s.clone()).collect();
        (paths, samples)
    } else {
        let samples = args
            .iter()
            .map(|p| parse_wav_i16(&std::fs::read(p).unwrap_or_else(|e| panic!("read {p}: {e}"))).expect("parse WAV"))
            .collect();
        (args.clone(), samples)
    };

    eprintln!("[verify_batched] {} clips", paths.len());
    let cfg = ScenarioConfig::load(Path::new(SCENARIO)).expect("load scenario");
    let asr = WhisperAsr::build(&cfg, Path::new("."));

    let refs: Vec<&[i16]> = samples.iter().map(|v| v.as_slice()).collect();
    let t = std::time::Instant::now();
    let texts = asr.transcribe_batch(&refs);
    let ms = t.elapsed().as_secs_f64() * 1e3;
    eprintln!("[verify_batched] batched transcribe of B={} done in {ms:.0} ms", texts.len());

    for (p, txt) in paths.iter().zip(&texts) {
        println!("{p}\t{txt}");
    }
    // --replicate sanity: all outputs identical?
    if paths.first().map(|p| p.contains('#')).unwrap_or(false) {
        let all_same = texts.iter().all(|t| t == &texts[0]);
        eprintln!("[verify_batched] replicate identical across streams: {all_same}");
        if !all_same {
            std::process::exit(1);
        }
    }
}

/// 16 kHz / mono / 16-bit PCM WAV -> i16 (mirrors engine_serve::parse_wav_i16).
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
