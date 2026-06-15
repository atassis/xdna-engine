//! Parity test: the host-f32 Whisper-small decoder (`asr::whisper_decoder::HostDecoder`) vs. the
//! ONNX decoder graphs (`asr::whisper::WhisperOnnxDecoder`), on the SAME fixed-random encoder hidden
//! states. Device-FREE: no NPU, no preprocessor, no real audio — semantics are irrelevant, we only
//! need numerical agreement on identical inputs.
//!
//! Both paths run a fixed 5-token greedy decode (start from `<|startoftranscript|>`, argmax each
//! step, feed back). We assert per-step logits rel-L2 <= 1e-3 AND identical argmax tokens.
//!
//! Run:  cd rust && cargo run -p npu-engine --bin verify_whisper_decode --release -- --host
//!
//! Paths resolve under `$WHISPER_ROOT` (default `..`, i.e. the worktree root when run from `rust/`):
//!   $WHISPER_ROOT/artifacts/whisper-small/onnx/decoder_model.onnx (+ decoder_with_past_model.onnx)
//!   $WHISPER_ROOT/artifacts/whisper-small/whisper_decoder/        (extracted host weights)

use std::path::PathBuf;
use std::rc::Rc;

use ndarray::Array2;
use npu_engine::asr::whisper::WhisperOnnxDecoder;
use npu_engine::asr::whisper_decoder::{HostDecoder, WhisperDecoderWeights};
use npu_xrt::Device;

const D: usize = 768;
const T_ENC: usize = 50; // small synthetic encoder length (cross-attn is length-invariant)
const N_STEPS: usize = 5;
const SOT: i64 = 50258; // <|startoftranscript|>
const REL_TOL: f32 = 1e-3;
/// Looser tolerance for the NPU path: per-token matmuls run in bf16 on the device, so the logits
/// accumulate bf16 rounding. Gate is rel-L2 <= 0.08 AND identical argmax (the token that matters).
const NPU_REL_TOL: f32 = 0.08;

/// Deterministic SplitMix64 -> uniform f32 in [-1, 1). No external rng dependency.
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// f32 in [-scale, scale)
    fn unif(&mut self, scale: f32) -> f32 {
        let u = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32; // [0,1)
        (u * 2.0 - 1.0) * scale
    }
}

fn argmax(v: &[f32]) -> i64 {
    let mut best = 0usize;
    for i in 1..v.len() {
        if v[i] > v[best] {
            best = i;
        }
    }
    best as i64
}

/// Relative L2: ||a-b||_2 / ||b||_2.
fn rel_l2(a: &[f32], b: &[f32]) -> f32 {
    let mut num = 0f64;
    let mut den = 0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let d = (x - y) as f64;
        num += d * d;
        den += (y as f64) * (y as f64);
    }
    (num.sqrt() / den.sqrt().max(1e-12)) as f32
}

fn main() {
    let host = std::env::args().any(|a| a == "--host");
    let npu = std::env::args().any(|a| a == "--npu");
    if !host && !npu {
        eprintln!("note: pass --host (host-vs-onnx) or --npu (npu-vs-onnx) to select the parity check.");
        return;
    }

    let root = PathBuf::from(std::env::var("WHISPER_ROOT").unwrap_or_else(|_| "..".into()));
    let ws = root.join("artifacts/whisper-small");
    let onnx_dir = ws.join("onnx");
    let weights_dir = ws.join("whisper_decoder");
    let tol = if npu { NPU_REL_TOL } else { REL_TOL };

    // --- fixed-random encoder hidden states [T_enc, 768] (seeded) ---
    let mut rng = Rng(0xD1CE_F00D_1234_5678);
    let mut enc = Array2::<f32>::zeros((T_ENC, D));
    for v in enc.iter_mut() {
        *v = rng.unif(1.0);
    }
    let enc_flat: Vec<f32> = enc.as_standard_layout().iter().copied().collect();
    let enc_shape = vec![1i64, T_ENC as i64, D as i64];

    // --- ONNX reference path ---
    println!("loading ONNX decoder graphs from {} ...", onnx_dir.display());
    let onnx = WhisperOnnxDecoder::load(&onnx_dir);

    // --- candidate decoder path (host f32 or NPU per-token matmuls) ---
    println!("loading host decoder weights from {} ...", weights_dir.display());
    let weights =
        Rc::new(WhisperDecoderWeights::load(&weights_dir).expect("load host decoder weights"));
    // Keep the NPU device alive for the whole run (decoder borrows it).
    let _dev;
    let mut hostdec = if npu {
        println!("NPU path: opening device (single-tenant — stop npu-asr/voxd first) ...");
        let dev = Rc::new(
            Device::open(0).expect("open NPU (stop npu-asr.service/voxd.service first)"),
        );
        _dev = Rc::clone(&dev);
        // root for CtxDecode = WHISPER_ROOT (worktree root; holds the `mlir-aie` symlink + xclbins).
        HostDecoder::new_npu(Rc::clone(&weights), &dev, &root)
    } else {
        HostDecoder::new(Rc::clone(&weights))
    };
    hostdec.precompute_cross(&enc);

    let label = if npu { "npu" } else { "host" };

    // Greedy decode N_STEPS on BOTH, starting from SOT, comparing per-step logits + argmax.
    // ONNX step 0 runs the no-past graph over the single SOT prompt; candidate runs step(SOT, pos=0).
    let mut max_rel = 0f32;
    let mut all_match = true;

    // step 0
    let (onnx_logits0, mut kv) = onnx.step0(&[SOT], &enc_shape, &enc_flat);
    let host_logits0 = hostdec.step(SOT, 0);
    let r0 = rel_l2(&host_logits0, &onnx_logits0);
    let a_onnx = argmax(&onnx_logits0);
    let a_host = argmax(&host_logits0);
    max_rel = max_rel.max(r0);
    let m0 = a_onnx == a_host;
    all_match &= m0;
    println!(
        "step 0: rel_l2={r0:.3e} argmax onnx={a_onnx} {label}={a_host} {}",
        if m0 { "OK" } else { "MISMATCH" }
    );
    let mut onnx_tok = a_onnx;
    let mut host_tok = a_host;

    // steps 1..N_STEPS
    for step in 1..N_STEPS {
        let (onnx_logits, new_kv) = onnx.step_cached(onnx_tok, &kv);
        kv = new_kv;
        let host_logits = hostdec.step(host_tok, step);
        let r = rel_l2(&host_logits, &onnx_logits);
        let a_onnx = argmax(&onnx_logits);
        let a_host = argmax(&host_logits);
        max_rel = max_rel.max(r);
        let m = a_onnx == a_host;
        all_match &= m;
        println!(
            "step {step}: rel_l2={r:.3e} argmax onnx={a_onnx} {label}={a_host} {}",
            if m { "OK" } else { "MISMATCH" }
        );
        onnx_tok = a_onnx;
        host_tok = a_host;
    }

    println!("\nmax logits rel_l2 over {N_STEPS} steps = {max_rel:.3e} (tol {tol:.0e})");
    let rel_ok = max_rel <= tol;
    if all_match && rel_ok {
        println!("PARITY PASS: {label} vs onnx {N_STEPS}/{N_STEPS} argmax match, max rel {max_rel:.3e}");
    } else {
        eprintln!(
            "PARITY FAIL: argmax_all_match={all_match} rel_ok={rel_ok} (max rel {max_rel:.3e})"
        );
        std::process::exit(1);
    }
}
