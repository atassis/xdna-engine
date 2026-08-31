//! Parakeet-tdt ASR hosted in the engine: nemo128 mel preproc (ONNX) + FastConformer encoder (NPU)
//! + TDT greedy decode. Reuses the validated npu_parakeet encoder + decode; not a rewrite.

use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;

use ndarray::Array2;
use npu_onnx::{Env, Session, Tensor};
use npu_parakeet::config::ModelCfg;
use npu_parakeet::encoder::FastConformerEncoder;
use npu_parakeet::prof::phase::{Bucket, PhaseScope};

use crate::api::EngineError;
use crate::config::ScenarioConfig;
use crate::pipeline::{AsrModel, Encoder};

const MEL: usize = 128;
const D: usize = 1024;
const VOCAB: usize = 8193;
const BLANK: i64 = 8192;
const N_DUR: usize = 5;
const STATE_DIM: usize = 640;
const STATE_LAYERS: usize = 2;
const MAX_TOK: usize = 10;
const WIN_MEL: usize = 2040;
// Step 2 windowing (parakeet-window-truncates-silently). ~1.5s of skirt on each side of a window
// boundary, mel-domain: enough for the encoder's own conv/attention receptive field to see real
// audio either side of the region actually decoded from that window, and enough slop to absorb the
// subsample stack's own stride/padding rounding when converting a mel offset to an encoder-frame
// offset (see `transcribe`'s window loop) -- this is a margin, not a precise boundary, deliberately:
// the task's own step-2 scoping says a windowed decode is not expected to reproduce a hypothetical
// single-pass decode at the seam. Must stay well under WIN_MEL/2 so STRIDE (below) stays positive
// and every window still yields useful new audio.
const OVERLAP_MEL: usize = 150;

/// `run_dj`'s (out, next_state_1, next_state_2) triple.
type DjOut = (Vec<f32>, Vec<f32>, Vec<f32>);

/// Everything `tdt_decode` must carry across a windowed decode boundary. THREE pieces, not the two
/// (`st1`/`st2`) the task's step-2 scoping named first: `last_token` -- the last emitted token id,
/// which seeds `run_dj`'s target embedding for the NEXT frame decoded -- has to move with them.
/// Dropping it back to BLANK at every window boundary (which `fresh()` correctly does ONLY for the
/// very first window) would feed the decoder a wrong target embedding at the first decoded frame of
/// every window after the first: `st1`/`st2` alone is not enough to reproduce a continuous decode.
struct DecodeCarry {
    st1: Vec<f32>,
    st2: Vec<f32>,
    last_token: i64,
}

impl DecodeCarry {
    /// The state a decode starts from at t=0 with no history -- byte-identical to what `tdt_decode`
    /// used to hardcode at its own top before windowing existed.
    fn fresh() -> Self {
        DecodeCarry {
            st1: vec![0f32; STATE_LAYERS * STATE_DIM],
            st2: vec![0f32; STATE_LAYERS * STATE_DIM],
            last_token: BLANK,
        }
    }
}

/// Engine `Encoder`-trait seam for the Parakeet FastConformerEncoder (the contract the parakeet
/// crate was built to fit). `ParakeetAsr` uses the proven `encode()` path internally; this adapter
/// exists so the encoder is also usable generically behind the engine's `Encoder` trait.
/// Orphan rule OK: `Encoder` is local to this crate.
impl Encoder for FastConformerEncoder {
    fn forward_last(&self, x: &Array2<f32>, valid_len: usize) -> Array2<f32> {
        FastConformerEncoder::forward_last(self, x, valid_len)
    }
}

pub struct ParakeetAsr {
    prep: Session,
    dj: Session,
    enc: FastConformerEncoder,
    vocab: HashMap<i64, String>,
    _env: Rc<Env>,
}

impl ParakeetAsr {
    /// `cfg.artifacts.weights` points at the parakeet artifact dir (contains preprocessor.onnx,
    /// decoder_joint.onnx, vocab.txt, encoder/). Opens its own NPU device via `new_npu`.
    pub fn build(cfg: &ScenarioConfig, root: &Path) -> Result<Self, EngineError> {
        let env = Env::new().map_err(|e| EngineError::Load(format!("onnx env: {e}")))?; // Env::new() returns Rc<Env>
        let pk = root.join(&cfg.artifacts.weights);
        let load = |f: &str| -> Result<Session, EngineError> {
            Session::load(&env, pk.join(f).to_str().unwrap())
                .map_err(|e| EngineError::Load(format!("load {f}: {e}")))
        };
        let prep = load("preprocessor.onnx")?;
        let dj = load("decoder_joint.onnx")?;
        let xroot = std::env::var("NPU_XCLBIN_ROOT")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| root.to_path_buf());
        let enc = FastConformerEncoder::new_npu(&pk.join("encoder"), ModelCfg::PARAKEET_V3, &xroot)
            .map_err(|e| EngineError::Load(e.to_string()))?;
        let vocab = load_vocab(&pk.join("vocab.txt"))?;
        Ok(ParakeetAsr { prep, dj, enc, vocab, _env: env })
    }

    /// TDT duration-split greedy decode (mirrors onnx-asr _AsrWithTransducerDecoding +
    /// NemoConformerTdt). Originally lifted verbatim from parakeet_serve.rs; now decodes only
    /// encoder frames `[t_start, t_stop)` and threads `carry` in/out, so a windowed caller can skip
    /// a window's own left/right context skirt (frames the encoder attended to for context but that
    /// a NEIGHBOURING window decodes with full context) while still continuing the SAME logical
    /// decode across the boundary. `t_start=0, t_stop=valid, carry=DecodeCarry::fresh()` reproduces
    /// the original unwindowed behaviour exactly -- the single-window (t <= WIN_MEL) path in
    /// `transcribe` always calls it this way.
    fn tdt_decode(
        &self,
        encoded: &Array2<f32>,
        t_start: usize,
        t_stop: usize,
        carry: DecodeCarry,
    ) -> Result<(Vec<i64>, DecodeCarry), EngineError> {
        let DecodeCarry { mut st1, mut st2, mut last_token } = carry;
        let mut tokens: Vec<i64> = Vec::new();
        let (mut t, mut emitted) = (t_start, 0usize);
        while t < t_stop {
            let frame = encoded.row(t).to_vec(); // [1024]
            let last = last_token as i32;
            let (out, nst1, nst2) = self.run_dj(&frame, last, &st1, &st2)?;
            let token = argmax(&out[..VOCAB]); // 8193 token logits
            let step = argmax(&out[VOCAB..VOCAB + N_DUR]) as usize; // duration 0..4
            if token != BLANK {
                st1 = nst1; // commit predictor state on emission
                st2 = nst2;
                last_token = token;
                tokens.push(token);
                emitted += 1;
            }
            if step > 0 {
                t += step;
                emitted = 0;
            } else if token == BLANK || emitted == MAX_TOK {
                t += 1;
                emitted = 0;
            }
        }
        Ok((tokens, DecodeCarry { st1, st2, last_token }))
    }

    fn run_dj(
        &self,
        frame: &[f32],
        last_tok: i32,
        st1: &[f32],
        st2: &[f32],
    ) -> Result<DjOut, EngineError> {
        let targets = [last_tok];
        let tlen = [1i32];
        let sd = vec![STATE_LAYERS as i64, 1, STATE_DIM as i64];
        let out = self
            .dj
            .run(
                &[
                    ("encoder_outputs", Tensor::F32(frame, vec![1, D as i64, 1])),
                    ("targets", Tensor::I32(&targets, vec![1, 1])),
                    ("target_length", Tensor::I32(&tlen, vec![1])),
                    ("input_states_1", Tensor::F32(st1, sd.clone())),
                    ("input_states_2", Tensor::F32(st2, sd)),
                ],
                &["outputs", "output_states_1", "output_states_2"],
            )
            .map_err(|e| EngineError::Device(format!("decoder_joint: {e}")))?;
        Ok((out.f32(0).to_vec(), out.f32(1).to_vec(), out.f32(2).to_vec()))
    }

    fn detokenize(&self, ids: &[i64]) -> String {
        let s: String = ids
            .iter()
            .map(|id| self.vocab.get(id).map(|x| x.as_str()).unwrap_or(""))
            .collect();
        s.trim().to_string()
    }
}

impl AsrModel for ParakeetAsr {
    fn transcribe(&self, samples: &[i16]) -> Result<String, EngineError> {
        // NO-DOUBLE-COUNT RULE: `report()` SUMS every recorded (stage,bucket). The
        // FastConformerEncoder self-attributes 100% of `encode()` internally (ff/mhsa/conv/ln/...),
        // so we deliberately do NOT wrap `self.enc.encode(...)` in a PhaseScope -- a wrapping
        // "encode" scope would be summed ON TOP of the encoder's own scopes and inflate the total.
        // The scopes below are LEAF Host stages OUTSIDE the encoder: each contains no encoder scope
        // and no mm() call. Still holds with windowing (below): `encode()` stays unwrapped on
        // every window's call, and `preproc`/`tdt_decode` still contain neither an encoder scope
        // nor an `mm()` call.
        let (t, feats) = {
            // preproc: mel-frontend (preprocessor.onnx run) + host marshal into an owned flat
            // channel-major buffer. Run ONCE for the whole input regardless of length -- the ONNX
            // mel frontend is O(T), unlike the encoder's O(T^2) self-attention, so there is no
            // windowing to do here; windowing (below) slices this buffer per window instead of
            // re-running the preprocessor per window.
            let _p = PhaseScope::new("preproc", Bucket::Host);
            let wav: Vec<f32> = samples.iter().map(|&s| s as f32 / 32768.0).collect();
            let n = wav.len() as i64;
            let lens = [n];
            let feat = self
                .prep
                .run(
                    &[
                        ("waveforms", Tensor::F32(&wav, vec![1, n])),
                        ("waveforms_lens", Tensor::I64(&lens, vec![1])),
                    ],
                    &["features", "features_lens"],
                )
                .map_err(|e| EngineError::Device(format!("preprocessor: {e}")))?;
            let t = feat.shape(0)[2] as usize; // [1,128,T]
            let feats = feat.f32(0).to_vec(); // [128*T] channel-major, owned past this scope
            (t, feats)
        };

        // Step 2 (parakeet-window-truncates-silently): window the mel at <=WIN_MEL with
        // OVERLAP_MEL of skirt on each internal boundary, carrying TDT predictor state
        // (`DecodeCarry`) across windows so long audio transcribes as one continuous decode
        // instead of erroring (step 1) or silently truncating (the original `.min()` bug).
        // `t <= WIN_MEL` takes exactly one iteration with `t_start=0, t_stop=valid,
        // carry=DecodeCarry::fresh()` -- byte-for-byte the pre-windowing call shape, so every
        // request this engine served before today (everything under 20.4s) is unaffected.
        let stride = WIN_MEL - 2 * OVERLAP_MEL;
        let mut window_start = 0usize;
        let mut carry = DecodeCarry::fresh();
        let mut all_ids: Vec<i64> = Vec::new();
        loop {
            let window_end = (window_start + WIN_MEL).min(t);
            let window_len = window_end - window_start;
            let is_first = window_start == 0;
            let is_last = window_end == t;

            let mut mel = Array2::<f32>::zeros((MEL, window_len));
            for c in 0..MEL {
                for ti in 0..window_len {
                    mel[[c, ti]] = feats[c * t + window_start + ti];
                }
            }
            // encode() is OUTSIDE every PhaseScope here (see NO-DOUBLE-COUNT above); the encoder
            // attributes its own time via its internal scopes, on every window.
            let encoded = self.enc.encode(&mel); // [T', 1024] on the NPU
            let valid = encoded.nrows();

            // Keep boundaries in LOCAL encoder-frame space. Skip OVERLAP_MEL/8 frames of left
            // skirt (already decoded by the PREVIOUS window with real left context) except on the
            // first window, which has no earlier audio to skip. Stop OVERLAP_MEL/8 frames before
            // the end (this window's own right skirt, which lacks real right context within this
            // window -- the NEXT window decodes it instead, with full context) except on the last
            // window, which has no more audio to hand off. The /8 is the subsample stack's own
            // downsample ratio; a few frames of slop against its exact conv/stride boundary is
            // exactly what OVERLAP_MEL's margin exists to absorb -- see this file's OVERLAP_MEL
            // comment and the task's own step-2 scoping (do not chase exactness at this seam).
            let t_start = if is_first { 0 } else { (OVERLAP_MEL / 8).min(valid) };
            let t_stop = if is_last { valid } else { valid.saturating_sub(OVERLAP_MEL / 8) };
            let t_stop = t_stop.max(t_start); // guard: a degenerate short tail window never inverts

            let (ids, next_carry) = {
                // tdt_decode: TDT greedy decode loop (per-frame decoder_joint.onnx calls), this
                // window's slice of it. Leaf Host.
                let _d = PhaseScope::new("tdt_decode", Bucket::Host);
                self.tdt_decode(&encoded, t_start, t_stop, carry)?
            };
            if std::env::var_os("PARAKEET_WINDOW_DEBUG").is_some() {
                eprintln!(
                    "[window-debug] mel[{window_start},{window_end}) len={window_len} \
first={is_first} last={is_last} valid={valid} t_start={t_start} t_stop={t_stop} ntok={} \
text={:?}",
                    ids.len(),
                    self.detokenize(&ids)
                );
            }
            all_ids.extend(ids);
            carry = next_carry;

            if is_last {
                break;
            }
            window_start += stride;
        }

        // detok: token -> text assembly. Leaf Host.
        let _t = PhaseScope::new("detok", Bucket::Host);
        Ok(self.detokenize(&all_ids))
    }
}

fn load_vocab(path: &Path) -> Result<HashMap<i64, String>, EngineError> {
    let txt = std::fs::read_to_string(path)
        .map_err(|e| EngineError::Load(format!("vocab {}: {e}", path.display())))?;
    let mut m = HashMap::new();
    for line in txt.lines() {
        if let Some((tok, id)) = line.rsplit_once(' ') {
            if let Ok(id) = id.trim().parse::<i64>() {
                m.insert(id, tok.replace('\u{2581}', " "));
            }
        }
    }
    Ok(m)
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
