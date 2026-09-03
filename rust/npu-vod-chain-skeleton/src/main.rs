//! VOD -> STT -> LLM -> tool-call wiring skeleton.
//!
//! Design audit: `journal/docs/log/2026-08/2026-08-28-vod-stt-llm-tools-design.md`. That design
//! recommends NOT building the product now (LLM device backend unlinked, no tool-calling anywhere
//! in the tree). This binary is the smallest thing that proves the four fronts compose into one
//! flow using the real trait contracts, not a product. It is device-free by construction (hard
//! gate: no `/dev/accel/accel0`, no xclbin dispatch) -- every stage that a real run would put on
//! the NPU is either genuinely run on host (audio extraction) or an explicit stub that says so.
//!
//! Stage map:
//!   1. ffmpeg subprocess:      video file -> 16 kHz mono PCM.                REAL, host, ctx 0.
//!   2. `HostStubAsr`:          PCM -> transcript text.                       STUB (device-free lane).
//!      Real path: `npu_engine::pipeline::{Frontend,Encoder,Head}` via `FastConformerEncoder`
//!      (rust/npu-parakeet/src/npu.rs) -- exists on NPU, out of reach here (hard gate #2, not a gap).
//!   3. `npu_gemma::npu::GemmaNpuDecoder`: REAL crate code, REAL call, REAL error. The device
//!      backend is unlinked in-tree (rust/npu-gemma/src/npu.rs) -- this is a genuine gap, not a
//!      lane restriction, so calling it is honest evidence rather than a fabricated stub.
//!   4. `stub_tool_classifier`:  transcript -> tool call.                     STUB (no LLM reasoning
//!      exists anywhere in the tree yet; this is a keyword substitute for wiring, not the
//!      vocab-pruned closed-set decode the design doc names as the real MVP mechanism).
//!   5. Tool call emitted as JSON on stdout.

use ndarray::Array2;
use npu_engine::pipeline::{AsrModel, Encoder};
use npu_gemma::config::GEMMA3_270M;
use npu_gemma::npu::{GemmaNpuDecoder, NpuError, TokenInputs};
use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};

/// Marks a stage that would touch NPU hardware in a real run but does not here. Every call site
/// names the real path and the reason it is not taken, so a grep for this string is the complete
/// list of what is fake in this binary.
macro_rules! stub_marker {
    ($stage:expr, $real_path:expr, $reason:expr) => {
        eprintln!("[STUB stage={} real_path={} reason={}]", $stage, $real_path, $reason);
    };
}

/// Stage 1, real: decode the input file's audio track to 16 kHz mono PCM16 via a system `ffmpeg`
/// subprocess (same pattern as `npu-sr`'s decoder pipe, `rust/npu-sr/src/pipeline.rs`). Host-only,
/// crosses zero NPU hardware contexts.
fn extract_pcm16(video: &Path) -> std::io::Result<Vec<i16>> {
    let mut ff = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(video)
        .args(["-vn", "-ar", "16000", "-ac", "1", "-f", "s16le", "-"])
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()?;
    let mut raw = Vec::new();
    ff.stdout.take().unwrap().read_to_end(&mut raw)?;
    let st = ff.wait()?;
    if !st.success() {
        return Err(std::io::Error::other(format!("ffmpeg exited {st}")));
    }
    Ok(raw
        .chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]))
        .collect())
}

/// Stage 2, stub: implements the real `Encoder` trait (`rust/npu-engine/src/pipeline.rs`) so the
/// type contract is genuinely exercised, but never touches a device. Any transcript content is
/// fabricated -- this proves the seam compiles and dispatches through a `dyn Encoder`, nothing
/// about the audio.
struct HostStubEncoder;
impl Encoder for HostStubEncoder {
    fn forward_last(&self, x: &Array2<f32>, valid_len: usize) -> Array2<f32> {
        stub_marker!(
            "stt-encoder",
            "npu_parakeet::npu::FastConformerEncoder (rust/npu-parakeet/src/npu.rs), dispatched \
             through npu_engine::pipeline::Encoder",
            "hard gate #2 (no device) -- this exists and runs on NPU today, it is just out of \
             reach in this lane, not a missing rail"
        );
        // No compute: keep the [M, D] shape contract (`M = valid_len`), zero-filled.
        Array2::zeros((valid_len.max(1), x.ncols()))
    }
}

/// Stage 2, stub: implements the real `AsrModel` trait. Ignores the PCM content entirely and
/// returns a fixed transcript built to give the tool classifier something to select on.
struct HostStubAsr;
impl AsrModel for HostStubAsr {
    fn transcribe(&self, samples: &[i16]) -> Result<String, npu_engine::EngineError> {
        let enc = HostStubEncoder;
        let dummy = Array2::<f32>::zeros((1, 8));
        let _ = enc.forward_last(&dummy, 1); // exercise the Encoder trait object, discard output
        stub_marker!(
            "stt-decode",
            "WhisperAsr's fused/per-op decode or Parakeet TDT decode (rust/npu-engine/src/asr/*.rs)",
            "hard gate #2 (no device); also default-host for Whisper per the design doc audit"
        );
        eprintln!("[STUB] {} PCM samples ignored; returning a fixed transcript", samples.len());
        Ok("please schedule a meeting with alex tomorrow at 3pm".to_string())
    }
}

/// Stage 4, stub: NOT the vocab-pruned closed-set LM-head decode the design doc names as the
/// honest MVP mechanism (§3 item 5) -- that needs real logits from a real forward pass, which does
/// not exist. This is a keyword substitute whose only job is to prove a transcript can drive a
/// structured tool-call decision downstream of wherever LLM reasoning eventually lands.
fn stub_tool_classifier(transcript: &str) -> serde_json::Value {
    stub_marker!(
        "llm-tool-decision",
        "vocab-pruned closed-set LM-head argmax over npu_gemma's fused decode ELF logits \
         (design doc 2026-08-28-vod-stt-llm-tools-design.md §3 item 5)",
        "no LLM forward pass exists on this stack in any form (device or host); \
         GemmaNpuDecoder::step is a hard stub, see the real call above"
    );
    let t = transcript.to_lowercase();
    let (tool, args) = if t.contains("schedule") || t.contains("meeting") {
        ("schedule_meeting", serde_json::json!({ "raw_transcript": transcript }))
    } else if t.contains("email") || t.contains("send") {
        ("send_email", serde_json::json!({ "raw_transcript": transcript }))
    } else if t.contains("remind") {
        ("set_reminder", serde_json::json!({ "raw_transcript": transcript }))
    } else {
        ("unknown", serde_json::json!({ "raw_transcript": transcript }))
    };
    serde_json::json!({
        "tool": tool,
        "arguments": args,
        "decision_source": "stub-keyword-classifier",
    })
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let video_path: std::path::PathBuf = if args.len() > 1 {
        args[1].clone().into()
    } else {
        // No input given: synthesize a tiny clip (silent video + a sine tone) via ffmpeg lavfi so
        // the binary is runnable standalone. Content is irrelevant -- the ASR stage is stubbed.
        let dir = std::env::temp_dir().join("vod-chain-skeleton");
        std::fs::create_dir_all(&dir).expect("create scratch dir");
        let p = dir.join("synth.mp4");
        let st = Command::new("ffmpeg")
            .args([
                "-y", "-v", "error",
                "-f", "lavfi", "-i", "color=c=black:s=32x32:d=2",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
                "-c:v", "libx264", "-c:a", "aac", "-shortest",
            ])
            .arg(&p)
            .status()
            .expect("spawn ffmpeg synth");
        assert!(st.success(), "ffmpeg synth-clip generation failed");
        eprintln!("[INFO] no video path given; synthesized {}", p.display());
        p
    };

    eprintln!("== stage 1: ffmpeg audio extraction (real, host, 0 hardware contexts) ==");
    let pcm = extract_pcm16(&video_path).expect("audio extraction failed");
    eprintln!("[INFO] extracted {} PCM16 samples ({:.2}s @16kHz)", pcm.len(), pcm.len() as f64 / 16_000.0);

    eprintln!("== stage 2: STT (host stub, device-free lane) ==");
    let asr = HostStubAsr;
    let transcript = asr.transcribe(&pcm).expect("stub transcribe");
    eprintln!("[INFO] transcript = {transcript:?}");

    eprintln!("== stage 3: LLM decode (real npu-gemma crate call, real stub error) ==");
    let mut dec = GemmaNpuDecoder::plan(GEMMA3_270M, 2048);
    let n_bricks = dec.load_elf(Path::new("/does/not/exist")).expect("load_elf stub");
    eprintln!("[INFO] planned schedule: {n_bricks} bricks (26/layer * 18 layers + 2)");
    let inp = TokenInputs { x: vec![0.0; GEMMA3_270M.d_model], n_past: 0 };
    match dec.step(&inp) {
        Err(e @ NpuError::DeviceBackendUnlinked) => {
            eprintln!("[REAL ERROR, not fabricated] GemmaNpuDecoder::step -> {e}");
        }
        other => panic!("expected DeviceBackendUnlinked, got {other:?} -- npu-gemma's stub changed shape"),
    }

    eprintln!("== stage 4: tool decision (host stub keyword classifier) ==");
    let tool_call = stub_tool_classifier(&transcript);

    println!("{}", serde_json::to_string_pretty(&tool_call).unwrap());

    eprintln!("\n== hardware-context report ==");
    eprintln!("THIS RUN crossed 0 hardware contexts: ffmpeg is host-only, both NPU-shaped stages");
    eprintln!("(STT encoder/decode, LLM decode) are stubs that never opened /dev/accel/accel0.");
    eprintln!("A REAL (unstubbed) run, per the design doc's audit, would cross approximately:");
    eprintln!("  ctx 0: none (ffmpeg, host)");
    eprintln!("  ctx 1: STT encoder xclbin (FastConformerEncoder / whole_array GEMM)");
    eprintln!("  ctx 2: STT decode ELF (Whisper fused decode or Parakeet TDT decode)  <- 1 transition");
    eprintln!("  ctx 3: LLM decode ELF (Gemma fused decode, once built)                <- 1 transition");
    eprintln!("  = 3 hardware-context loads, 2 transitions, reused across many dispatches each");
    eprintln!("  (many audio frames in ctx1, many decode steps in ctx2, many tokens in ctx3).");
    eprintln!("  Per [[transition-cost-is-a-property-of-the-destination-program-not-a-constant]]");
    eprintln!("  each transition costs ~1-5ms; a batch VOD clip pays this a HANDFUL of times total,");
    eprintln!("  not per-token -- see design doc sec 2 for why that makes the tenancy tax small here.");
}
