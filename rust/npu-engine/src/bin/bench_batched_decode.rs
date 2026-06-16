//! Subsystem-B perf bench: DECODE-ONLY J/token + tok/s, batched (B streams) vs M=1, over the SAME
//! pre-encoded clips in one session (the e2e number is encoder-dominated, so we isolate decode here).
//!
//! Requires BOTH backends built in WhisperAsr:
//!   NPU_DECODE_FUSED=1       NPU_DECODE_FUSED_DIR=<M=1 ELF dir, e.g. artifacts/fused_decode12>
//!   NPU_DECODE_FUSED_BATCH=1 NPU_DECODE_FUSED_BATCH_DIR=<batched ELF dir, decode_batched_B16_L12_sp>
//! Pass exactly B clips (the batched decoder's B). Single-tenant NPU — stop services first.
//!
//! Method: encode all clips ONCE (untimed). Then time + RAPL-package-energy (a) the batched decode of
//! all B at once, (b) the M=1 decode looped over the same B. Both include the per-utterance cross-K/V
//! fold + the greedy token loop. Reports decode_ms, tokens, ms/token, tok/s, J/token + the batched/M=1
//! throughput and energy ratios.

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
    ids.iter().map(|v| v.len().saturating_sub(4)).sum() // minus the 4-token prompt
}

fn main() {
    let paths: Vec<String> = std::env::args().skip(1).collect();
    assert!(!paths.is_empty(), "usage: bench_batched_decode <clip1.wav> ... (exactly B clips)");
    let samples: Vec<Vec<i16>> = paths
        .iter()
        .map(|p| parse_wav_i16(&std::fs::read(p).unwrap_or_else(|e| panic!("read {p}: {e}"))).expect("parse WAV"))
        .collect();
    let refs: Vec<&[i16]> = samples.iter().map(|v| v.as_slice()).collect();

    let cfg = ScenarioConfig::load(Path::new(SCENARIO)).expect("load scenario");
    let asr = WhisperAsr::build(&cfg, Path::new("."));

    let nclips = refs.len();
    eprintln!("[bench] encoding {nclips} clips (sequential NPU encoder)...");
    let (encs, prep_ms, enc_ms) = asr.encode_clips_timed(&refs);
    eprintln!(
        "[bench] encode stage: preproc {prep_ms:.1} ms + encoder {enc_ms:.1} ms = {:.1} ms for {nclips} clips ({:.1} ms/clip encoder)",
        prep_ms + enc_ms,
        enc_ms / nclips as f64
    );
    let rmax = rapl_uj(RAPL_MAX).unwrap_or(u128::MAX);

    // warmup each backend once (prime kernels, page-ins) — untimed.
    let _ = asr.decode_batch_ids(&encs);
    let _ = asr.decode_m1_ids(&encs[0]);

    // (a) BATCHED decode of all B at once.
    let e0 = rapl_uj(RAPL_PKG);
    let t0 = std::time::Instant::now();
    let bids = asr.decode_batch_ids(&encs);
    let b_ms = t0.elapsed().as_secs_f64() * 1e3;
    let b_j = match (e0, rapl_uj(RAPL_PKG)) {
        (Some(a), Some(c)) => Some(uj_delta(a, c, rmax) as f64 / 1e6),
        _ => None,
    };
    let bt = ntok(&bids);

    // (b) M=1 decode looped over the same clips.
    let e1 = rapl_uj(RAPL_PKG);
    let t1 = std::time::Instant::now();
    let mids: Vec<Vec<i64>> = encs.iter().map(|e| asr.decode_m1_ids(e)).collect();
    let m_ms = t1.elapsed().as_secs_f64() * 1e3;
    let m_j = match (e1, rapl_uj(RAPL_PKG)) {
        (Some(a), Some(c)) => Some(uj_delta(a, c, rmax) as f64 / 1e6),
        _ => None,
    };
    let mt = ntok(&mids);

    let row = |label: &str, ms: f64, tok: usize, j: Option<f64>| {
        let toks = tok.max(1) as f64;
        let jpt = j.map(|x| format!("{:.4}", x / toks)).unwrap_or_else(|| "n/a".into());
        let jtot = j.map(|x| format!("{x:.2}")).unwrap_or_else(|| "n/a".into());
        println!(
            "  {label:<9} decode_ms={ms:9.1}  tokens={tok:5}  ms/tok={:7.3}  tok/s={:8.1}  J_total={jtot:>7}  J/tok={jpt}",
            ms / toks,
            toks / (ms / 1e3),
        );
    };
    println!("\n=== DECODE-ONLY bench (same clips, one session; incl. per-utterance cross-fold) ===");
    row("batched", b_ms, bt, b_j);
    row("m1", m_ms, mt, m_j);
    let bts = bt.max(1) as f64 / (b_ms / 1e3);
    let mts = mt.max(1) as f64 / (m_ms / 1e3);
    println!("\n  throughput: batched {:.1} tok/s vs M=1 {:.1} tok/s  =>  {:.2}x", bts, mts, bts / mts);
    if let (Some(bj), Some(mj)) = (b_j, m_j) {
        let bjt = bj / bt.max(1) as f64;
        let mjt = mj / mt.max(1) as f64;
        println!("  energy:     batched {:.4} J/tok vs M=1 {:.4} J/tok  =>  {:.2}x lower", bjt, mjt, mjt / bjt);
    } else {
        println!("  energy: RAPL unreadable ({RAPL_PKG}); run `sudo chmod -R a+r /sys/class/powercap/intel-rapl*/`");
    }

    // ---- full e2e picture (encode stage is shared + sequential per clip) ----
    let enc_stage = prep_ms + enc_ms;
    println!("\n=== FULL E2E for {nclips} clips (preproc+encoder shared; decode differs) ===");
    println!(
        "  encode stage (preproc {prep_ms:.1} + encoder {enc_ms:.1}) = {enc_stage:.1} ms  [per-clip preproc/encoder split: see whisper_e2e_timing]"
    );
    println!("  batched e2e = encode {enc_stage:.1} + decode {b_ms:.1} = {:.1} ms", enc_stage + b_ms);
    println!("  M=1     e2e = encode {enc_stage:.1} + decode {m_ms:.1} = {:.1} ms", enc_stage + m_ms);
    println!(
        "  NOTE: encoder runs {nclips}x sequentially ({:.0}% of batched e2e) — batching the encoder is the next e2e lever.",
        100.0 * enc_stage / (enc_stage + b_ms)
    );
    println!("  (per-phase decode breakdown above: [BATCHED_PHASE] and [FUSED_PHASE] lines on stderr)");
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
