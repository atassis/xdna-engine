//! Whisper-small ASR hosted in the engine: log-mel preproc (ONNX) + Whisper encoder (NPU) +
//! KV-cached greedy ONNX decoder loop with GPT-2 byte-level BPE detokenization.
//!
//! Mirrors `asr::parakeet::ParakeetAsr` (the encoder opens its OWN NPU device via `new_npu`, so the
//! registry's "asr" arm must NOT open a device for whisper).
//!
//! Decode is KV-cached (decoder_with_past) for fair, fast benchmarking:
//! - **Step 0** runs `decoder_model.onnx` over the full prompt + `encoder_hidden_states[1,1500,768]`.
//!   It emits `logits` AND all 48 `present.{0..11}.{decoder,encoder}.{key,value}` — the encoder KV
//!   (`...encoder.*`, shape `[1,12,1500,64]`) are computed here ONCE and stay fixed.
//! - **Steps ≥1** run `decoder_with_past_model.onnx` over `input_ids=[[last]]` (length 1) + all 48
//!   `past_key_values.*` from the previous step. It emits `logits` + only the 24 *decoder* present
//!   KV (which grow by 1 row/step); the encoder past is consumed but not re-emitted, so we carry the
//!   step-0 encoder KV forward unchanged. No bool `use_cache_branch` input → fits the F32/I64 shim.

use std::cell::RefCell;
use std::path::Path;
use std::rc::Rc;

use ndarray::Array2;
use npu_onnx::{Env, Session, Tensor};
use npu_whisper::config::WhisperCfg;
use npu_whisper::encoder::WhisperEncoder;
use tokenizers::Tokenizer;

use crate::asr::whisper_decoder::{FusedDecoder, HostDecoder, WhisperDecoderWeights};
use crate::config::ScenarioConfig;
use crate::pipeline::AsrModel;

const N_SAMPLES: usize = 480_000; // 30 s @ 16 kHz (preprocessor.onnx is fixed-shape)
const N_MELS: usize = 80;
const N_FRAMES: usize = 3000;
const D: usize = 768;
const T_ENC: usize = 1500; // encoder output rows
const VOCAB: usize = 51865;
const MAX_DECODE: usize = 200;
const N_LAYERS: usize = 12;

// Whisper special-token ids (from WhisperProcessor.tokenizer).
const SOT: i64 = 50258; // <|startoftranscript|>
const EOT: i64 = 50257; // <|endoftext|>
const TRANSCRIBE: i64 = 50359; // <|transcribe|>
const NOTIMESTAMPS: i64 = 50363; // <|notimestamps|>
// Language tags are a contiguous block: <|en|>=50259 .. <|su|>=50357 (99 languages). We auto-detect
// the language with a 1-step decode (argmax restricted to this block) so RU stays RU and EN stays EN
// — without this, forcing one language makes Whisper TRANSLATE the other.
const LANG_LO: i64 = 50259; // <|en|>
const LANG_HI: i64 = 50357; // last language token

/// A held key/value cache tensor: the past-input name it feeds, its flat f32 data, and its shape.
/// We own the data so it survives across `Session::run` boundaries (the ONNX `Outputs` borrow does
/// not), then re-feed it as a `Tensor::F32` on the next step.
pub struct Kv {
    name: String, // the `past_key_values.*` input name this entry feeds on the next step
    data: Vec<f32>,
    shape: Vec<i64>,
}

/// Device-FREE ONNX decoder reference: loads ONLY the two decoder ONNX graphs (no NPU encoder, no
/// preprocessor, no tokenizer) and runs the KV-cached greedy loop against a CALLER-SUPPLIED
/// `encoder_hidden_states`. Used by `verify_whisper_decode` to get an ONNX ground truth for the host
/// reimplementation without touching the device. The step logic mirrors `WhisperAsr` exactly.
pub struct WhisperOnnxDecoder {
    decoder: Session,      // decoder_model.onnx — step 0
    decoder_past: Session, // decoder_with_past_model.onnx — steps >=1
    _env: Rc<Env>,
}

impl WhisperOnnxDecoder {
    /// `onnx_dir` points at `artifacts/whisper-small/onnx` (holding `decoder_model.onnx` and
    /// `decoder_with_past_model.onnx`). Opens NO device.
    pub fn load(onnx_dir: &Path) -> Self {
        let env = Env::new().expect("onnx env");
        let load = |p: std::path::PathBuf| {
            Session::load(&env, p.to_str().unwrap())
                .unwrap_or_else(|e| panic!("load {}: {e}", p.display()))
        };
        let decoder = load(onnx_dir.join("decoder_model.onnx"));
        let decoder_past = load(onnx_dir.join("decoder_with_past_model.onnx"));
        WhisperOnnxDecoder { decoder, decoder_past, _env: env }
    }

    /// Step 0 over the full prompt + encoder hidden states; returns last-position logits + 48-entry KV.
    pub fn step0(&self, ids: &[i64], enc_shape: &[i64], encoder_hidden: &[f32]) -> (Vec<f32>, Vec<Kv>) {
        decode_step0(&self.decoder, ids, enc_shape, encoder_hidden)
    }

    /// Cached step over one new token + prior KV; returns logits + the next 48-entry KV.
    pub fn step_cached(&self, tok: i64, prev: &[Kv]) -> (Vec<f32>, Vec<Kv>) {
        decode_step_cached(&self.decoder_past, tok, prev)
    }
}

/// The 48 `present.*` output names emitted by `decoder_model.onnx` (step 0), in graph order.
fn step0_present_names() -> Vec<String> {
    let mut v = Vec::with_capacity(4 * N_LAYERS);
    for l in 0..N_LAYERS {
        for kind in ["decoder", "encoder"] {
            for kv in ["key", "value"] {
                v.push(format!("present.{l}.{kind}.{kv}"));
            }
        }
    }
    v
}

/// The 24 *decoder* `present.*` output names emitted by `decoder_with_past_model.onnx` (steps>=1).
fn past_present_names() -> Vec<String> {
    let mut v = Vec::with_capacity(2 * N_LAYERS);
    for l in 0..N_LAYERS {
        for kv in ["key", "value"] {
            v.push(format!("present.{l}.decoder.{kv}"));
        }
    }
    v
}

/// Free-standing step-0 decode (shared by `WhisperAsr` and `WhisperOnnxDecoder`).
fn decode_step0(
    decoder: &Session,
    ids: &[i64],
    enc_shape: &[i64],
    encoder_hidden: &[f32],
) -> (Vec<f32>, Vec<Kv>) {
    let l = ids.len() as i64;
    let present_names = step0_present_names();
    let out_names: Vec<&str> = std::iter::once("logits")
        .chain(present_names.iter().map(|s| s.as_str()))
        .collect();
    let out = decoder
        .run(
            &[
                ("input_ids", Tensor::I64(ids, vec![1, l])),
                ("encoder_hidden_states", Tensor::F32(encoder_hidden, enc_shape.to_vec())),
            ],
            &out_names,
        )
        .expect("whisper decoder (step 0)");
    let logits_all = out.f32(0);
    let last = (ids.len() - 1) * VOCAB;
    let logits = logits_all[last..last + VOCAB].to_vec();
    let kv: Vec<Kv> = present_names
        .iter()
        .enumerate()
        .map(|(i, pname)| Kv {
            name: pname.replacen("present", "past_key_values", 1),
            data: out.f32(i + 1).to_vec(),
            shape: out.shape(i + 1),
        })
        .collect();
    (logits, kv)
}

/// Free-standing cached-step decode (shared by `WhisperAsr` and `WhisperOnnxDecoder`).
fn decode_step_cached(decoder_past: &Session, tok: i64, prev: &[Kv]) -> (Vec<f32>, Vec<Kv>) {
    let ids = [tok];
    let mut inputs: Vec<(&str, Tensor)> = Vec::with_capacity(1 + prev.len());
    inputs.push(("input_ids", Tensor::I64(&ids, vec![1, 1])));
    for kv in prev {
        inputs.push((kv.name.as_str(), Tensor::F32(&kv.data, kv.shape.clone())));
    }
    let present_names = past_present_names();
    let out_names: Vec<&str> = std::iter::once("logits")
        .chain(present_names.iter().map(|s| s.as_str()))
        .collect();
    let out = decoder_past.run(&inputs, &out_names).expect("whisper decoder (cached)");
    let logits = out.f32(0).to_vec();
    let new_decoder: Vec<Kv> = present_names
        .iter()
        .enumerate()
        .map(|(i, pname)| Kv {
            name: pname.replacen("present", "past_key_values", 1),
            data: out.f32(i + 1).to_vec(),
            shape: out.shape(i + 1),
        })
        .collect();
    let next: Vec<Kv> = prev
        .iter()
        .map(|old| {
            if old.name.contains(".decoder.") {
                let pos = new_decoder
                    .iter()
                    .position(|n| n.name == old.name)
                    .expect("matching refreshed decoder KV");
                Kv {
                    name: new_decoder[pos].name.clone(),
                    data: new_decoder[pos].data.clone(),
                    shape: new_decoder[pos].shape.clone(),
                }
            } else {
                Kv { name: old.name.clone(), data: old.data.clone(), shape: old.shape.clone() }
            }
        })
        .collect();
    (logits, next)
}

pub struct WhisperAsr {
    prep: Session,
    decoder: Session,      // decoder_model.onnx — step 0, no past, emits encoder+decoder present
    decoder_past: Session, // decoder_with_past_model.onnx — steps >=1, cached
    enc: WhisperEncoder,
    tok: Tokenizer,
    /// On-NPU per-token decoder, constructed ONCE when `NPU_DECODE` is set (weights + `CtxDecode`
    /// registered up front, sharing the encoder's single-tenant device). `None` => ONNX decode path.
    /// `RefCell` because `transcribe(&self)` mutates the decoder's self-KV cache (`step`/`reset`).
    npu_decoder: Option<RefCell<HostDecoder>>,
    /// Whole-decode fused-ELF backend (env `NPU_DECODE_FUSED`): the ENTIRE 12-layer decoder in one
    /// fused-ELF dispatch/token (vs `npu_decoder`'s ~72). Takes precedence over `npu_decoder`.
    npu_fused: Option<RefCell<FusedDecoder>>,
    _env: Rc<Env>,
}

impl WhisperAsr {
    /// `cfg.artifacts.weights` points at `artifacts/whisper-small` (weights + `onnx/` + the exported
    /// `preprocessor.onnx` + `tokenizer.json`). Opens its own NPU device inside `new_npu`.
    pub fn build(cfg: &ScenarioConfig, root: &Path) -> Self {
        let env = Env::new().expect("onnx env");
        let ws = root.join(&cfg.artifacts.weights); // artifacts/whisper-small
        let load = |p: std::path::PathBuf| {
            Session::load(&env, p.to_str().unwrap())
                .unwrap_or_else(|e| panic!("load {}: {e}", p.display()))
        };
        let prep = load(ws.join("preprocessor.onnx"));
        let decoder = load(ws.join("onnx/decoder_model.onnx"));
        let decoder_past = load(ws.join("onnx/decoder_with_past_model.onnx"));
        let xroot = std::env::var("NPU_XCLBIN_ROOT")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| root.to_path_buf());
        let enc = WhisperEncoder::new_npu(&ws, WhisperCfg::SMALL, &xroot);
        let tok = Tokenizer::from_file(ws.join("tokenizer.json"))
            .unwrap_or_else(|e| panic!("load tokenizer.json: {e}"));

        // NPU_DECODE: route the per-token decoder matmuls to the NPU (HostDecoder::new_npu) instead
        // of the ONNX decoder graphs. Built ONCE here (weights + resident CtxDecode kernels), sharing
        // the encoder's already-open single-tenant device. When unset, the decoder is None and the
        // transcribe path is byte-identical to the ONNX baseline.
        // Decode backend: NPU_DECODE_FUSED (whole 12-layer fused ELF, 1 dispatch/token) takes
        // precedence over NPU_DECODE (per-op, ~72 dispatches/token); else ONNX. All share the
        // encoder's single-tenant device + the same host weights.
        let fused_on = std::env::var("NPU_DECODE_FUSED").is_ok();
        let npu_on = std::env::var("NPU_DECODE").is_ok();
        let (npu_decoder, npu_fused) = if fused_on || npu_on {
            let dev = enc
                .device()
                .expect("NPU decode: encoder must hold an open NPU device (built via new_npu)");
            let weights = Rc::new(
                WhisperDecoderWeights::load(&ws.join("whisper_decoder"))
                    .expect("NPU decode: load whisper_decoder host weights"),
            );
            if fused_on {
                let fdir = xroot.join("artifacts/fused_decode12");
                // Share the encoder's resident ctx2 kernel so the cross-K/V fold runs on the NPU.
                let fd = FusedDecoder::new(weights, &dev, &fdir, enc.shared());
                eprintln!("[whisper] NPU_DECODE_FUSED=1: whole 12-layer decode in ONE fused-ELF dispatch/token");
                (None, Some(RefCell::new(fd)))
            } else {
                let dec = HostDecoder::new_npu(weights, &dev, &xroot);
                eprintln!("[whisper] NPU_DECODE=1: per-token decoder matmuls on the NPU");
                (Some(RefCell::new(dec)), None)
            }
        } else {
            (None, None)
        };

        WhisperAsr { prep, decoder, decoder_past, enc, tok, npu_decoder, npu_fused, _env: env }
    }

    /// Step 0: run the no-past graph over the full prompt + encoder hidden states. Delegates to the
    /// free-standing `decode_step0` (shared with `WhisperOnnxDecoder`).
    fn decode_step0(
        &self,
        ids: &[i64],
        enc_shape: &[i64],
        encoder_hidden: &[f32],
    ) -> (Vec<f32>, Vec<Kv>) {
        decode_step0(&self.decoder, ids, enc_shape, encoder_hidden)
    }

    /// Steps >=1: run the cached graph. Delegates to the free-standing `decode_step_cached`.
    fn decode_step_cached(&self, tok: i64, prev: &[Kv]) -> (Vec<f32>, Vec<Kv>) {
        decode_step_cached(&self.decoder_past, tok, prev)
    }

    /// Argmax over the contiguous language-tag block `[LANG_LO, LANG_HI]` — the shared
    /// language-detection rule used by both decode backends.
    fn pick_lang(logits: &[f32]) -> i64 {
        let lo = LANG_LO as usize;
        let hi = LANG_HI as usize;
        let mut best = lo;
        for i in lo..=hi {
            if logits[i] > logits[best] {
                best = i;
            }
        }
        best as i64
    }

    /// KV-cached greedy autoregressive decode against the cached encoder hidden states.
    /// `encoder_hidden` is the flat row-major `[1500*768]` slice from `forward_last`.
    ///
    /// Dispatches to the NPU per-token decoder when `NPU_DECODE` is set (the decoder was built in
    /// `build`), else the ONNX KV-cached path. BOTH paths share the EXACT same control logic:
    /// 1-step language detection (argmax over the language block), the prompt
    /// `[SOT, lang, TRANSCRIBE, NOTIMESTAMPS]`, full-vocab argmax, EOT stop, and `MAX_DECODE` cap.
    /// The ONLY difference is the source of per-step logits.
    fn greedy_decode(&self, encoder_hidden: &[f32]) -> Vec<i64> {
        if let Some(fd) = &self.npu_fused {
            self.greedy_decode_fused(&mut fd.borrow_mut(), encoder_hidden)
        } else if let Some(dec) = &self.npu_decoder {
            self.greedy_decode_npu(&mut dec.borrow_mut(), encoder_hidden)
        } else {
            self.greedy_decode_onnx(encoder_hidden)
        }
    }

    /// Whole-decode fused-ELF greedy decode. IDENTICAL control logic to `greedy_decode_npu` (lang
    /// detect, prompt, argmax, EOT, MAX_DECODE) — only the backend is `FusedDecoder` (1 dispatch/token).
    fn greedy_decode_fused(&self, dec: &mut FusedDecoder, encoder_hidden: &[f32]) -> Vec<i64> {
        let enc2 = Array2::from_shape_vec((T_ENC, D), encoder_hidden.to_vec())
            .expect("encoder_hidden is [T_ENC*D]");
        dec.precompute_cross(&enc2);
        let lang_logits = dec.step(SOT, 0);
        let lang = Self::pick_lang(&lang_logits);
        let prompt: Vec<i64> = vec![SOT, lang, TRANSCRIBE, NOTIMESTAMPS];
        let mut ids = prompt.clone();
        dec.reset();
        let mut logits = Vec::new();
        for (pos, &tok) in prompt.iter().enumerate() {
            logits = dec.step(tok, pos);
        }
        let mut next = argmax(&logits);
        if next != EOT {
            ids.push(next);
        }
        for step in 0..(MAX_DECODE - 1) {
            if next == EOT {
                break;
            }
            let pos = prompt.len() + step;
            let logits = dec.step(next, pos);
            next = argmax(&logits);
            if next == EOT {
                break;
            }
            ids.push(next);
        }
        // P0: per-phase breakdown for this utterance (no-op unless FUSED_PHASE_TIMING set).
        dec.dump_phase_timing();
        ids
    }

    /// ONNX KV-cached greedy decode (the baseline; unchanged behavior).
    fn greedy_decode_onnx(&self, encoder_hidden: &[f32]) -> Vec<i64> {
        let enc_shape = vec![1, T_ENC as i64, D as i64];
        // language detection: 1-step `[SOT]` decode via the no-past graph (KV discarded).
        let (lang_logits, _kv) = self.decode_step0(&[SOT], &enc_shape, encoder_hidden);
        let lang = Self::pick_lang(&lang_logits);
        let prompt: Vec<i64> = vec![SOT, lang, TRANSCRIBE, NOTIMESTAMPS];
        let mut ids = prompt.clone();

        // Step 0: full prompt through the no-past graph; seeds the KV cache.
        let (logits, mut kv) = self.decode_step0(&prompt, &enc_shape, encoder_hidden);
        let mut next = argmax(&logits);
        if next != EOT {
            ids.push(next);
        }
        // Steps >=1: feed one token at a time through the cached graph.
        for _ in 1..MAX_DECODE {
            if next == EOT {
                break;
            }
            let (logits, new_kv) = self.decode_step_cached(next, &kv);
            kv = new_kv;
            next = argmax(&logits);
            if next == EOT {
                break;
            }
            ids.push(next);
        }
        ids
    }

    /// On-NPU per-token greedy decode (`HostDecoder`). MIRRORS `greedy_decode_onnx` exactly — same
    /// language detection, prompt, argmax, EOT stop and `MAX_DECODE` — only the logits come from the
    /// NPU decoder. The host decoder advances one token at a time with an explicit position; the
    /// 4-token prompt is fed sequentially (positions 0..3) and the next token argmaxed after the last
    /// prompt token, exactly matching the ONNX step-0-over-full-prompt semantics.
    fn greedy_decode_npu(&self, dec: &mut HostDecoder, encoder_hidden: &[f32]) -> Vec<i64> {
        // Cross-KV from the encoder hidden states (also resets the self-KV cache for this utterance).
        let enc2 = Array2::from_shape_vec((T_ENC, D), encoder_hidden.to_vec())
            .expect("encoder_hidden is [T_ENC*D]");

        // language detection: precompute cross-KV, decode `[SOT]` at pos 0, argmax over the lang
        // block, then reset (drop the SOT self-KV so the real prompt starts clean — mirrors the ONNX
        // path which discards the detection KV).
        dec.precompute_cross(&enc2);
        let lang_logits = dec.step(SOT, 0);
        let lang = Self::pick_lang(&lang_logits);

        let prompt: Vec<i64> = vec![SOT, lang, TRANSCRIBE, NOTIMESTAMPS];
        let mut ids = prompt.clone();

        // Re-seed: fresh self-KV for the actual prompt (cross-KV is unchanged for this utterance).
        dec.reset();
        // Feed the whole prompt; argmax only after the final prompt token (== ONNX step-0 last pos).
        let mut logits = Vec::new();
        for (pos, &tok) in prompt.iter().enumerate() {
            logits = dec.step(tok, pos);
        }
        let mut next = argmax(&logits);
        if next != EOT {
            ids.push(next);
        }
        // Steps >=1: feed the last emitted token at the next position.
        for step in 0..(MAX_DECODE - 1) {
            if next == EOT {
                break;
            }
            let pos = prompt.len() + step; // position of the token we are about to feed
            let logits = dec.step(next, pos);
            next = argmax(&logits);
            if next == EOT {
                break;
            }
            ids.push(next);
        }
        ids
    }

    fn detokenize(&self, ids: &[i64]) -> String {
        let u: Vec<u32> = ids.iter().map(|&i| i as u32).collect();
        self.tok
            .decode(&u, true) // skip_special_tokens = true
            .unwrap_or_default()
            .trim()
            .to_string()
    }
}

impl AsrModel for WhisperAsr {
    fn transcribe(&self, samples: &[i16]) -> String {
        let timing = std::env::var("WHISPER_TIMING").is_ok();
        let t_e2e = std::time::Instant::now();

        // i16 -> f32 in [-1,1], pad/truncate to exactly N_SAMPLES (preprocessor.onnx is fixed-shape).
        let t_prep = std::time::Instant::now();
        let mut wav = vec![0f32; N_SAMPLES];
        let m = samples.len().min(N_SAMPLES);
        for i in 0..m {
            wav[i] = samples[i] as f32 / 32768.0;
        }
        let feat = self
            .prep
            .run(
                &[("waveform", Tensor::F32(&wav, vec![1, N_SAMPLES as i64]))],
                &["input_features"],
            )
            .expect("preprocessor");
        // input_features: [1, 80, 3000] flat channel-major -> Array2 [80, 3000] for the encoder.
        let feats = feat.f32(0);
        let mut mel = Array2::<f32>::zeros((N_MELS, N_FRAMES));
        for c in 0..N_MELS {
            for t in 0..N_FRAMES {
                mel[[c, t]] = feats[c * N_FRAMES + t];
            }
        }
        let prep_ms = t_prep.elapsed().as_secs_f64() * 1e3;

        let t_enc = std::time::Instant::now();
        let encoded = self.enc.forward_last(&mel); // [1500, 768] on the NPU
        // row-major [1500*768] for the decoder's encoder_hidden_states[1,1500,768]
        let std = encoded.as_standard_layout();
        let flat: Vec<f32> = std.iter().copied().collect();
        let enc_ms = t_enc.elapsed().as_secs_f64() * 1e3;

        // Reset the NPU dispatch counter (no-op on the ONNX path) so dispatches/token is per-utterance.
        if let Some(dec) = &self.npu_decoder {
            dec.borrow().reset_npu_dispatches();
        }
        let t_dec = std::time::Instant::now();
        let ids = self.greedy_decode(&flat);
        let dec_ms = t_dec.elapsed().as_secs_f64() * 1e3;

        let text = self.detokenize(&ids);
        let e2e_ms = t_e2e.elapsed().as_secs_f64() * 1e3;

        if timing {
            // #tokens = emitted ids minus the 4-token prompt [SOT, lang, TRANSCRIBE, NOTIMESTAMPS].
            let n_tok = ids.len().saturating_sub(4).max(1);
            let ms_per_tok = dec_ms / n_tok as f64;
            let (backend, disp_per_tok) = if self.npu_fused.is_some() {
                ("FUSED", 1.0) // whole 12-layer decode = ONE dispatch/token by construction
            } else if let Some(dec) = &self.npu_decoder {
                ("NPU", dec.borrow().npu_dispatches() as f64 / n_tok as f64)
            } else {
                ("ONNX", 0.0)
            };
            eprintln!(
                "[WHISPER_TIMING] backend={backend} e2e_ms={e2e_ms:.2} preproc_ms={prep_ms:.2} \
                 encoder_ms={enc_ms:.2} decode_ms={dec_ms:.2} tokens={n_tok} \
                 ms_per_tok={ms_per_tok:.3} disp_per_tok={disp_per_tok:.2}"
            );
        }
        text
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
