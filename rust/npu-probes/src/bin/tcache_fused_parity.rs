//! Arm-vs-arm token parity for the M0.5 transposed self-V cache, driven through the ENGINE's fused
//! decoder (`FusedDecoder`, the `NPU_DECODE_FUSED` path) so the RUST HOST's handling of the 4-param
//! contract is what is under test. `verify_tcache_parity.py` already proved the arm itself by
//! driving `decode.elf` directly; this proves the host makes that arm reachable.
//!
//! Input is the SAVED REAL encoder output (`artifacts/whisper-small/refs/encoded.npy`), not audio:
//! whisper-small has no mel `preprocessor.onnx` here. Real states matter — fixed-seed random ones
//! decode to `<|endoftext|>` on step 2 and stay there, which makes the whole run a two-token
//! constant that almost any wrong cache still reproduces. `--synthetic` keeps that weaker mode for
//! when the ref is unavailable.
//!
//! Each step emits `argmax:hash`, where hash is FNV-1a over the raw logit BYTES. Bit-exact is the
//! right bar, not a tolerance: the transpose the tr arm deletes is pure data movement, so the two
//! arms must agree exactly — and the hash catches a divergence the argmax would round away.
//!
//! Usage: tcache_fused_parity <fused_dir> [steps] [--synthetic]  (single-tenant NPU — stop services)

use std::path::{Path, PathBuf};
use std::rc::Rc;

use ndarray::Array2;
use npu_engine::asr::whisper_decoder::{FusedDecoder, WhisperDecoderWeights};
use npu_xrt::Device;

const D: usize = 768;
const T_ENC: usize = 1500;
const SOT: i64 = 50258; // <|startoftranscript|>

/// Deterministic SplitMix64 -> uniform f32 in [-1, 1). The seed is fixed so both arms are fed the
/// same encoder states without having to materialise them to disk.
struct Rng(u64);
impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        ((z >> 40) as f32 / 8_388_608.0) - 1.0
    }
}

/// FNV-1a over the logit bytes: a bit-exact fingerprint of the whole vector, so a divergence that
/// does not move the argmax still shows up.
fn fnv1a(logits: &[f32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for v in logits {
        for b in v.to_bits().to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x1000_0000_01b3);
        }
    }
    h
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let synthetic = args.iter().any(|a| a == "--synthetic");
    let mut pos_args = args.iter().filter(|a| !a.starts_with("--"));
    let dir = PathBuf::from(
        pos_args.next().expect("usage: tcache_fused_parity <fused_dir> [steps] [--synthetic]"),
    );
    let steps: usize = pos_args.next().map_or(64, |s| s.parse().expect("steps must be an integer"));
    // Positions index embed_positions [448, D], which is also the cache width S.
    assert!(steps <= 447, "steps {steps} exceeds the 448-column cache / position table");

    let wdir = Path::new("artifacts/whisper-small/whisper_decoder");
    let w = Rc::new(WhisperDecoderWeights::load(wdir).expect("load decoder weights"));
    let dev = Rc::new(Device::open(0).expect("open NPU (stop xdna-engine + voxd first)"));
    // `shared: None` -> the cross-K/V fold runs on host f32; identical for both arms.
    let mut fd = FusedDecoder::new(w, &dev, &dir, None);

    let enc: Array2<f32> = if synthetic {
        let mut rng = Rng(0x5EED_1234_ABCD_0001);
        eprintln!("[enc] SYNTHETIC fixed-seed states (weak signal: decodes to a constant token)");
        Array2::from_shape_fn((T_ENC, D), |_| rng.next_f32() * 0.5)
    } else {
        let p = Path::new("artifacts/whisper-small/refs/encoded.npy");
        let a: Array2<f32> = ndarray_npy::read_npy(p)
            .unwrap_or_else(|e| panic!("read {}: {e} (or pass --synthetic)", p.display()));
        assert_eq!(a.dim(), (T_ENC, D), "{}: unexpected shape", p.display());
        eprintln!("[enc] real encoder output {}", p.display());
        a
    };
    fd.precompute_cross(&enc);

    let mut token = SOT;
    for pos in 0..steps {
        let logits = fd.step(token, pos);
        // total_cmp, not partial_cmp: a NaN logit must fail as a token mismatch, not a panic that
        // looks like a harness bug.
        token = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i as i64)
            .expect("empty logits");
        println!("{token}:{:016x}", fnv1a(&logits));
    }
}
