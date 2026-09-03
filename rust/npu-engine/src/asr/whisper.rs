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

use crate::api::EngineError;
use crate::asr::whisper_decoder::{BatchedFusedDecoder, FusedDecoder, HostDecoder, WhisperDecoderWeights};
use crate::config::ScenarioConfig;
use crate::pipeline::AsrModel;

const N_SAMPLES: usize = 480_000; // 30 s @ 16 kHz (preprocessor.onnx is fixed-shape)
const N_FRAMES: usize = 3000;
const T_ENC: usize = 1500; // encoder output rows (max_source_positions, 1500 for every Whisper)
const MAX_DECODE: usize = 200;

/// Whisper's special-token ids, RESOLVED FROM THE TOKENIZER rather than written down.
///
/// They are not constant across Whisper sizes: large-v3 inserted `<|yue|>` into the language block,
/// which shifts every id above it by one — `<|transcribe|>` is 50359 on whisper-small and 50360 on
/// large-v3-turbo, `<|notimestamps|>` 50363 vs 50364. Hardcoding the small-model values decodes
/// turbo as a different task with a language tag one off, silently: no crash, just wrong text. So
/// each id is looked up by its literal token string.
///
/// `lang_hi` is DERIVED, not looked up: the language tags are a contiguous block that ends where
/// `<|translate|>` begins, which is exactly what makes the block scannable for language detection.
/// Checking against the last tag by name would need to know whether this model has `<|yue|>`.
struct WhisperTokens {
    sot: i64,          // <|startoftranscript|>
    eot: i64,          // <|endoftext|>
    transcribe: i64,   // <|transcribe|>
    notimestamps: i64, // <|notimestamps|>
    lang_lo: i64,      // <|en|>, first language tag
    lang_hi: i64,      // last language tag = <|translate|> - 1
}

impl WhisperTokens {
    fn resolve(tok: &Tokenizer) -> Result<WhisperTokens, EngineError> {
        let id = |s: &str| -> Result<i64, EngineError> {
            tok.token_to_id(s)
                .map(|v| v as i64)
                .ok_or_else(|| EngineError::Load(format!("tokenizer has no {s:?} token")))
        };
        let translate = id("<|translate|>")?;
        let lang_lo = id("<|en|>")?;
        let t = WhisperTokens {
            sot: id("<|startoftranscript|>")?,
            eot: id("<|endoftext|>")?,
            transcribe: id("<|transcribe|>")?,
            notimestamps: id("<|notimestamps|>")?,
            lang_lo,
            lang_hi: translate - 1,
        };
        if t.lang_hi < t.lang_lo {
            return Err(EngineError::Load(format!(
                "whisper tokenizer: language block is empty (<|en|>={lang_lo}, <|translate|>={translate})"
            )));
        }
        Ok(t)
    }

    /// The decode prompt: transcribe `lang`, no timestamps.
    fn prompt(&self, lang: i64) -> Vec<i64> { vec![self.sot, lang, self.transcribe, self.notimestamps] }

    /// Argmax over the contiguous language-tag block — the shared language-detection rule used by
    /// every decode backend. Auto-detecting keeps RU as RU and EN as EN; forcing one language makes
    /// Whisper TRANSLATE the other.
    fn pick_lang(&self, logits: &[f32]) -> i64 {
        let (lo, hi) = (self.lang_lo as usize, self.lang_hi as usize);
        let mut best = lo;
        for i in lo..=hi {
            if logits[i] > logits[best] {
                best = i;
            }
        }
        best as i64
    }
}

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
    n_layers: usize,       // DECODER depth: names the `present.*`/`past_key_values.*` tensors
    _env: Rc<Env>,
}

impl WhisperOnnxDecoder {
    /// `onnx_dir` points at `artifacts/whisper-small/onnx` (holding `decoder_model.onnx` and
    /// `decoder_with_past_model.onnx`). Opens NO device.
    pub fn load(onnx_dir: &Path, n_layers: usize) -> Self {
        let env = Env::new().expect("onnx env");
        let load = |p: std::path::PathBuf| {
            Session::load(&env, p.to_str().unwrap())
                .unwrap_or_else(|e| panic!("load {}: {e}", p.display()))
        };
        let decoder = load(onnx_dir.join("decoder_model.onnx"));
        let decoder_past = load(onnx_dir.join("decoder_with_past_model.onnx"));
        WhisperOnnxDecoder { decoder, decoder_past, n_layers, _env: env }
    }

    /// Step 0 over the full prompt + encoder hidden states; returns last-position logits + the
    /// `4 * n_layers` KV entries.
    pub fn step0(&self, ids: &[i64], enc_shape: &[i64], encoder_hidden: &[f32]) -> (Vec<f32>, Vec<Kv>) {
        decode_step0(&self.decoder, self.n_layers, ids, enc_shape, encoder_hidden)
    }

    /// Cached step over one new token + prior KV; returns logits + the next KV set.
    pub fn step_cached(&self, tok: i64, prev: &[Kv]) -> (Vec<f32>, Vec<Kv>) {
        decode_step_cached(&self.decoder_past, self.n_layers, tok, prev)
    }
}

/// The `4 * n_layers` `present.*` output names emitted by `decoder_model.onnx` (step 0), in graph
/// order. `n_layers` is the DECODER depth (12 for whisper-small, 4 for large-v3-turbo).
fn step0_present_names(n_layers: usize) -> Vec<String> {
    let mut v = Vec::with_capacity(4 * n_layers);
    for l in 0..n_layers {
        for kind in ["decoder", "encoder"] {
            for kv in ["key", "value"] {
                v.push(format!("present.{l}.{kind}.{kv}"));
            }
        }
    }
    v
}

/// The `2 * n_layers` *decoder* `present.*` names emitted by `decoder_with_past_model.onnx` (steps>=1).
fn past_present_names(n_layers: usize) -> Vec<String> {
    let mut v = Vec::with_capacity(2 * n_layers);
    for l in 0..n_layers {
        for kv in ["key", "value"] {
            v.push(format!("present.{l}.decoder.{kv}"));
        }
    }
    v
}

/// Free-standing step-0 decode (shared by `WhisperAsr` and `WhisperOnnxDecoder`).
fn decode_step0(
    decoder: &Session,
    n_layers: usize,
    ids: &[i64],
    enc_shape: &[i64],
    encoder_hidden: &[f32],
) -> (Vec<f32>, Vec<Kv>) {
    let l = ids.len() as i64;
    let present_names = step0_present_names(n_layers);
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
    // Vocabulary width comes from the tensor the graph just returned, not a constant: it is 51865 on
    // whisper-small and 51866 on large-v3-turbo, and reading it here cannot drift from the model.
    let vocab = logits_all.len() / ids.len();
    let last = (ids.len() - 1) * vocab;
    let logits = logits_all[last..last + vocab].to_vec();
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
fn decode_step_cached(decoder_past: &Session, n_layers: usize, tok: i64, prev: &[Kv]) -> (Vec<f32>, Vec<Kv>) {
    let ids = [tok];
    let mut inputs: Vec<(&str, Tensor)> = Vec::with_capacity(1 + prev.len());
    inputs.push(("input_ids", Tensor::I64(&ids, vec![1, 1])));
    for kv in prev {
        inputs.push((kv.name.as_str(), Tensor::F32(&kv.data, kv.shape.clone())));
    }
    let present_names = past_present_names(n_layers);
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
    /// Encoder shape (d_model / layers / heads / ffn / mels) from the scenario manifest — the same
    /// value handed to `WhisperEncoder`, kept so the host side agrees with the encoder on `d_model`
    /// and `n_mels` instead of re-deriving them.
    cfg: WhisperCfg,
    /// DECODER depth. Not `cfg.n_layers`: whisper-small is 12 encoder / 12 decoder, but
    /// large-v3-turbo is 32 / 4, and this is the one that names the ONNX KV tensors.
    dec_layers: usize,
    /// Special-token ids read out of this model's own tokenizer (they shift between Whisper sizes).
    tokens: WhisperTokens,
    /// On-NPU per-token decoder, constructed ONCE when `NPU_DECODE` is set (weights + `CtxDecode`
    /// registered up front, sharing the encoder's single-tenant device). `None` => ONNX decode path.
    /// `RefCell` because `transcribe(&self)` mutates the decoder's self-KV cache (`step`/`reset`).
    npu_decoder: Option<RefCell<HostDecoder>>,
    /// Whole-decode fused-ELF backend (env `NPU_DECODE_FUSED`): the ENTIRE 12-layer decoder in one
    /// fused-ELF dispatch/token (vs `npu_decoder`'s ~72). Takes precedence over `npu_decoder`.
    npu_fused: Option<RefCell<FusedDecoder>>,
    /// Subsystem B (env `NPU_DECODE_FUSED_BATCH` + `NPU_DECODE_FUSED_BATCH_DIR`): batched decode over B
    /// streams in one dispatch/step. Driven by `transcribe_batch` (offline bulk), not the serve path.
    npu_fused_batch: Option<RefCell<BatchedFusedDecoder>>,
    _env: Rc<Env>,
}

impl WhisperAsr {
    /// `cfg.artifacts.weights` points at `artifacts/whisper-small` (weights + `onnx/` + the exported
    /// `preprocessor.onnx` + `tokenizer.json`). Opens its own NPU device inside `new_npu`.
    pub fn build(cfg: &ScenarioConfig, root: &Path) -> Result<Self, EngineError> {
        let env = Env::new().map_err(|e| EngineError::Load(format!("onnx env: {e}")))?;
        let ws = root.join(&cfg.artifacts.weights); // artifacts/whisper-small
        let load = |p: std::path::PathBuf| -> Result<Session, EngineError> {
            Session::load(&env, p.to_str().unwrap())
                .map_err(|e| EngineError::Load(format!("load {}: {e}", p.display())))
        };
        let prep = load(ws.join("preprocessor.onnx"))?;
        let decoder = load(ws.join("onnx/decoder_model.onnx"))?;
        let decoder_past = load(ws.join("onnx/decoder_with_past_model.onnx"))?;
        let xroot = std::env::var("NPU_XCLBIN_ROOT")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| root.to_path_buf());
        // Shape comes from the scenario's `[model]` block, so a second Whisper size is a manifest,
        // not a code change. `n_mels` and `n_decoder_layers` default to 80 / `n_layers`, which is
        // exactly whisper-small, so its scenario keeps parsing unchanged.
        let m = cfg.model_or_err().map_err(EngineError::Load)?;
        let wcfg = WhisperCfg {
            d_model: m.hidden,
            n_layers: m.n_layers,
            n_heads: m.n_heads,
            head_dim: m.head_dim,
            ffn: m.ff,
            n_mels: m.n_mels,
        };
        let dec_layers = m.decoder_layers();
        // WHISPER_ENC_HOST=1 runs the encoder on the host instead of the NPU. Opt-in and loud, never
        // a silent fallback: a model that quietly moved off the device is the one bug this engine
        // must not have. It exists so a newly baked size is usable end to end before its xclbins are
        // built, and so a box with no NPU can still run the reference path.
        // WhisperEncoder::new_npu still panics internally on failure (npu-whisper crate, out of
        // scope for this pass -- see engine-errors-are-real worklog).
        let enc = if std::env::var("WHISPER_ENC_HOST").is_ok() {
            eprintln!("[whisper] WHISPER_ENC_HOST=1: encoder runs on the HOST, not the NPU");
            WhisperEncoder::new(&ws, wcfg)
        } else {
            WhisperEncoder::new_npu(&ws, wcfg, &xroot)
        };
        let tok = Tokenizer::from_file(ws.join("tokenizer.json"))
            .map_err(|e| EngineError::Load(format!("load tokenizer.json: {e}")))?;
        let tokens = WhisperTokens::resolve(&tok)?;

        // NPU_DECODE: route the per-token decoder matmuls to the NPU (HostDecoder::new_npu) instead
        // of the ONNX decoder graphs. Built ONCE here (weights + resident CtxDecode kernels), sharing
        // the encoder's already-open single-tenant device. When unset, the decoder is None and the
        // transcribe path is byte-identical to the ONNX baseline.
        // Decode backend: NPU_DECODE_FUSED (whole 12-layer fused ELF, 1 dispatch/token) takes
        // precedence over NPU_DECODE (per-op, ~72 dispatches/token); else ONNX. All share the
        // encoder's single-tenant device + the same host weights.
        let fused_on = std::env::var("NPU_DECODE_FUSED").is_ok();
        let npu_on = std::env::var("NPU_DECODE").is_ok();
        let batch_on = std::env::var("NPU_DECODE_FUSED_BATCH").is_ok();
        let (npu_decoder, npu_fused, npu_fused_batch) = if fused_on || npu_on || batch_on {
            let dev = enc.device().ok_or_else(|| EngineError::Load(
                "NPU decode: encoder must hold an open NPU device (built via new_npu)".into()))?;
            let weights = Rc::new(
                WhisperDecoderWeights::load(&ws.join("whisper_decoder"))
                    .map_err(|e| EngineError::Load(format!("load whisper_decoder host weights: {e}")))?,
            );
            // Subsystem B: batched decoder (offline-bulk), independent of the single-stream backend.
            // BatchedFusedDecoder::new still panics internally on failure (whisper_decoder.rs, out of
            // scope for this pass).
            let nfb = if batch_on {
                let bdir = std::path::PathBuf::from(
                    std::env::var("NPU_DECODE_FUSED_BATCH_DIR").map_err(|_| EngineError::Load(
                        "NPU_DECODE_FUSED_BATCH requires NPU_DECODE_FUSED_BATCH_DIR".into()))?,
                );
                eprintln!("[whisper] batched fused decode dir: {}", bdir.display());
                // O1: share the encoder's resident ctx2 kernel so the batched cross-K/V fold runs on
                // the NPU (like M=1), not the naive host f32 loop.
                Some(RefCell::new(BatchedFusedDecoder::new(Rc::clone(&weights), &dev, &bdir, enc.shared())?))
            } else {
                None
            };
            if !fused_on && !npu_on {
                (None, None, nfb)
            } else if fused_on {
                // NPU_DECODE_FUSED_DIR overrides the fused-ELF artifact dir (for A/B of alternate
                // builds, e.g. the lever-3 coalesced ELF). Default: artifacts/fused_decode12.
                let fdir = match std::env::var("NPU_DECODE_FUSED_DIR") {
                    Ok(d) => std::path::PathBuf::from(d),
                    Err(_) => xroot.join("artifacts/fused_decode12"),
                };
                eprintln!("[whisper] fused decode ELF dir: {}", fdir.display());
                // Share the encoder's resident ctx2 kernel so the cross-K/V fold runs on the NPU.
                let fd = FusedDecoder::new(weights, &dev, &fdir, enc.shared())?;
                eprintln!("[whisper] NPU_DECODE_FUSED=1: whole 12-layer decode in ONE fused-ELF dispatch/token");
                (None, Some(RefCell::new(fd)), nfb)
            } else {
                let dec = HostDecoder::new_npu(weights, &dev, &xroot)?;
                eprintln!("[whisper] NPU_DECODE=1: per-token decoder matmuls on the NPU");
                (Some(RefCell::new(dec)), None, nfb)
            }
        } else {
            (None, None, None)
        };

        Ok(WhisperAsr {
            prep, decoder, decoder_past, enc, tok, cfg: wcfg, dec_layers, tokens,
            npu_decoder, npu_fused, npu_fused_batch, _env: env,
        })
    }

    /// Step 0: run the no-past graph over the full prompt + encoder hidden states. Delegates to the
    /// free-standing `decode_step0` (shared with `WhisperOnnxDecoder`).
    fn decode_step0(
        &self,
        ids: &[i64],
        enc_shape: &[i64],
        encoder_hidden: &[f32],
    ) -> (Vec<f32>, Vec<Kv>) {
        decode_step0(&self.decoder, self.dec_layers, ids, enc_shape, encoder_hidden)
    }

    /// Steps >=1: run the cached graph. Delegates to the free-standing `decode_step_cached`.
    fn decode_step_cached(&self, tok: i64, prev: &[Kv]) -> (Vec<f32>, Vec<Kv>) {
        decode_step_cached(&self.decoder_past, self.dec_layers, tok, prev)
    }

    /// KV-cached greedy autoregressive decode against the cached encoder hidden states.
    /// `encoder_hidden` is the flat row-major `[1500*768]` slice from `forward_last`.
    ///
    /// Dispatches to the NPU per-token decoder when `NPU_DECODE` is set (the decoder was built in
    /// `build`), else the ONNX KV-cached path. BOTH paths share the EXACT same control logic:
    /// 1-step language detection (argmax over the language block), the prompt
    /// `[SOT, lang, TRANSCRIBE, NOTIMESTAMPS]`, full-vocab argmax, EOT stop, and `MAX_DECODE` cap.
    /// The ONLY difference is the source of per-step logits. The ONNX arm (`greedy_decode_onnx`)
    /// is not converted: it holds no resident-context/KV-cache state, so it has no device error to
    /// report; it is wrapped in `Ok` rather than given a narrower signature than its siblings.
    ///
    /// `lang` short-circuits detection, for the caller that wants one tag pinned across windows.
    /// Returns the tag actually used.
    fn greedy_decode(&self, encoder_hidden: &[f32], lang: Option<i64>)
        -> Result<(Vec<i64>, i64), EngineError> {
        if let Some(fd) = &self.npu_fused {
            self.greedy_decode_fused(&mut fd.borrow_mut(), encoder_hidden, lang)
        } else if let Some(dec) = &self.npu_decoder {
            self.greedy_decode_npu(&mut dec.borrow_mut(), encoder_hidden, lang)
        } else {
            Ok(self.greedy_decode_onnx(encoder_hidden, lang))
        }
    }

    /// Whole-decode fused-ELF greedy decode. IDENTICAL control logic to `greedy_decode_npu` (lang
    /// detect, prompt, argmax, EOT, MAX_DECODE) — only the backend is `FusedDecoder` (1 dispatch/token).
    fn greedy_decode_fused(&self, dec: &mut FusedDecoder, encoder_hidden: &[f32], lang: Option<i64>)
        -> Result<(Vec<i64>, i64), EngineError> {
        let enc2 = Array2::from_shape_vec((T_ENC, self.cfg.d_model), encoder_hidden.to_vec())
            .expect("encoder_hidden is [T_ENC*D]");
        dec.precompute_cross(&enc2)?;
        // Step-0 language detection needs the FULL logits (restricted argmax over the language block) → it
        // stays on host. The steady-state loop uses the on-NPU argmax (`step_token`) when the argmax-fused
        // proj_out ELF is loaded: the ELF returns a token id and the 104 KB logits readback is dropped.
        let lang = match lang {
            Some(l) => l,
            None => self.tokens.pick_lang(&dec.step(self.tokens.sot, 0)?),
        };
        let prompt: Vec<i64> = self.tokens.prompt(lang);
        let mut ids = prompt.clone();
        dec.reset()?;
        let npu_argmax = dec.has_npu_argmax();
        // next token after feeding token `tok` at `pos`: on-NPU argmax (id) when available, else host argmax.
        let next_tok = |dec: &mut FusedDecoder, tok: i64, pos: usize| -> Result<i64, EngineError> {
            if npu_argmax {
                dec.step_token(tok, pos)
            } else {
                Ok(argmax(&dec.step(tok, pos)?))
            }
        };
        // Feed the prompt; the next token comes from the LAST prompt position. Earlier positions just
        // advance the KV cache (dispatch); their result is discarded (use the cheap id path when available).
        let mut next = 0i64;
        for (pos, &tok) in prompt.iter().enumerate() {
            next = next_tok(dec, tok, pos)?;
        }
        if next != self.tokens.eot {
            ids.push(next);
        }
        for step in 0..(MAX_DECODE - 1) {
            if next == self.tokens.eot {
                break;
            }
            let pos = prompt.len() + step;
            next = next_tok(dec, next, pos)?;
            if next == self.tokens.eot {
                break;
            }
            ids.push(next);
        }
        // P0: per-phase breakdown for this utterance (no-op unless FUSED_PHASE_TIMING set).
        dec.dump_phase_timing();
        Ok((ids, lang))
    }

    /// ONNX KV-cached greedy decode (the baseline; unchanged behavior).
    fn greedy_decode_onnx(&self, encoder_hidden: &[f32], lang: Option<i64>) -> (Vec<i64>, i64) {
        let enc_shape = vec![1, T_ENC as i64, self.cfg.d_model as i64];
        // language detection: 1-step `[SOT]` decode via the no-past graph (KV discarded).
        let lang = lang.unwrap_or_else(|| {
            let (l, _kv) = self.decode_step0(&[self.tokens.sot], &enc_shape, encoder_hidden);
            self.tokens.pick_lang(&l)
        });
        let prompt: Vec<i64> = self.tokens.prompt(lang);
        let mut ids = prompt.clone();

        // Step 0: full prompt through the no-past graph; seeds the KV cache.
        let (logits, mut kv) = self.decode_step0(&prompt, &enc_shape, encoder_hidden);
        let mut next = argmax(&logits);
        if next != self.tokens.eot {
            ids.push(next);
        }
        // Steps >=1: feed one token at a time through the cached graph.
        for _ in 1..MAX_DECODE {
            if next == self.tokens.eot {
                break;
            }
            let (logits, new_kv) = self.decode_step_cached(next, &kv);
            kv = new_kv;
            next = argmax(&logits);
            if next == self.tokens.eot {
                break;
            }
            ids.push(next);
        }
        (ids, lang)
    }

    /// On-NPU per-token greedy decode (`HostDecoder`). MIRRORS `greedy_decode_onnx` exactly — same
    /// language detection, prompt, argmax, EOT stop and `MAX_DECODE` — only the logits come from the
    /// NPU decoder. The host decoder advances one token at a time with an explicit position; the
    /// 4-token prompt is fed sequentially (positions 0..3) and the next token argmaxed after the last
    /// prompt token, exactly matching the ONNX step-0-over-full-prompt semantics.
    fn greedy_decode_npu(&self, dec: &mut HostDecoder, encoder_hidden: &[f32], lang: Option<i64>)
        -> Result<(Vec<i64>, i64), EngineError> {
        // Cross-KV from the encoder hidden states (also resets the self-KV cache for this utterance).
        let enc2 = Array2::from_shape_vec((T_ENC, self.cfg.d_model), encoder_hidden.to_vec())
            .expect("encoder_hidden is [T_ENC*D]");

        // language detection: precompute cross-KV (pure host, infallible), decode `[SOT]` at pos 0,
        // argmax over the lang block, then reset (drop the SOT self-KV so the real prompt starts
        // clean — mirrors the ONNX path which discards the detection KV).
        dec.precompute_cross(&enc2);
        let lang = match lang {
            Some(l) => l,
            None => self.tokens.pick_lang(&dec.step(self.tokens.sot, 0)?),
        };

        let prompt: Vec<i64> = self.tokens.prompt(lang);
        let mut ids = prompt.clone();

        // Re-seed: fresh self-KV for the actual prompt (cross-KV is unchanged for this utterance).
        dec.reset();
        // Feed the whole prompt; argmax only after the final prompt token (== ONNX step-0 last pos).
        let mut logits = Vec::new();
        for (pos, &tok) in prompt.iter().enumerate() {
            logits = dec.step(tok, pos)?;
        }
        let mut next = argmax(&logits);
        if next != self.tokens.eot {
            ids.push(next);
        }
        // Steps >=1: feed the last emitted token at the next position.
        for step in 0..(MAX_DECODE - 1) {
            if next == self.tokens.eot {
                break;
            }
            let pos = prompt.len() + step; // position of the token we are about to feed
            let logits = dec.step(next, pos)?;
            next = argmax(&logits);
            if next == self.tokens.eot {
                break;
            }
            ids.push(next);
        }
        Ok((ids, lang))
    }

    /// Preprocess + NPU-encode one clip → (flat encoder hidden `[T_ENC*D]`, preproc_ms, encoder_ms).
    fn encode_clip_timed(&self, samples: &[i16]) -> (Vec<f32>, f64, f64) {
        let mut wav = vec![0f32; N_SAMPLES];
        let m = samples.len().min(N_SAMPLES);
        for i in 0..m {
            wav[i] = samples[i] as f32 / 32768.0;
        }
        let tp = std::time::Instant::now();
        let feat = self
            .prep
            .run(&[("waveform", Tensor::F32(&wav, vec![1, N_SAMPLES as i64]))], &["input_features"])
            .expect("preprocessor");
        let feats = feat.f32(0);
        let mut mel = Array2::<f32>::zeros((self.cfg.n_mels, N_FRAMES));
        for c in 0..self.cfg.n_mels {
            for t in 0..N_FRAMES {
                mel[[c, t]] = feats[c * N_FRAMES + t];
            }
        }
        let prep_ms = tp.elapsed().as_secs_f64() * 1e3;
        let te = std::time::Instant::now();
        let encoded = self.enc.forward_last(&mel);
        let flat: Vec<f32> = encoded.as_standard_layout().iter().copied().collect();
        let enc_ms = te.elapsed().as_secs_f64() * 1e3;
        (flat, prep_ms, enc_ms)
    }

    fn encode_clip(&self, samples: &[i16]) -> Vec<f32> {
        self.encode_clip_timed(samples).0
    }

    /// Encode B clips, returning the hiddens + total preproc_ms + total encoder_ms (the encoder stage
    /// is per-clip sequential — NOT batched — so this is the shared front-end cost for the bench).
    pub fn encode_clips_timed(&self, clips: &[&[i16]]) -> (Vec<Vec<f32>>, f64, f64) {
        let (mut prep, mut enc) = (0.0, 0.0);
        let mut out = Vec::with_capacity(clips.len());
        for s in clips {
            let (f, p, e) = self.encode_clip_timed(s);
            prep += p;
            enc += e;
            out.push(f);
        }
        (out, prep, enc)
    }

    /// Subsystem B: transcribe exactly B clips at once (offline-bulk lockstep). Encodes each clip
    /// (sequential, single-tenant NPU), then runs ONE batched greedy decode over all B streams.
    pub fn transcribe_batch(&self, clips: &[&[i16]]) -> Result<Vec<String>, EngineError> {
        let cell = self.npu_fused_batch.as_ref().expect("NPU_DECODE_FUSED_BATCH not enabled");
        let mut dec = cell.borrow_mut();
        let b = dec.batch();
        assert_eq!(clips.len(), b, "transcribe_batch needs exactly B={b} clips");
        let encs: Vec<Array2<f32>> = clips
            .iter()
            .map(|s| Array2::from_shape_vec((T_ENC, self.cfg.d_model), self.encode_clip(s)).expect("enc shape"))
            .collect();
        let ids = self.greedy_decode_fused_batch(&mut dec, &encs)?;
        Ok(ids.iter().map(|id| self.detokenize(id)).collect())
    }

    /// Subsystem B — O3: transcribe N clips (any N) via length-bucketed offline-bulk. Sorts clips by
    /// sample-count (the decode-length proxy — the encoder pads all clips to T_ENC, so encoder length
    /// is useless), chunks into ⌈N/B⌉ B-sized buckets (last padded by repeating its longest clip),
    /// decodes each bucket lockstep, and reassembles transcripts in the ORIGINAL order. Similar-length
    /// clips per bucket cut lockstep waste vs length-mixed batches (each bucket runs to ITS longest,
    /// not the global longest).
    pub fn transcribe_bulk(&self, clips: &[&[i16]]) -> Result<Vec<String>, EngineError> {
        let b = self.npu_fused_batch.as_ref().expect("NPU_DECODE_FUSED_BATCH not enabled").borrow().batch();
        let n = clips.len();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by_key(|&i| clips[i].len()); // ascending by samples; bucket = similar lengths
        let mut out = vec![String::new(); n];
        for chunk in order.chunks(b) {
            let mut idxs: Vec<usize> = chunk.to_vec();
            while idxs.len() < b {
                idxs.push(*chunk.last().unwrap()); // pad short last bucket by repeating its longest
            }
            let bucket: Vec<&[i16]> = idxs.iter().map(|&i| clips[i]).collect();
            let texts = self.transcribe_batch(&bucket)?;
            for (k, &orig) in chunk.iter().enumerate() {
                out[orig] = texts[k].clone();
            }
        }
        Ok(out)
    }

    // ---- decode-only bench hooks (subsystem-B perf): encode once (untimed), then time each decode
    // backend over the SAME encoder outputs. Token ids include the 4-token prompt (caller subtracts).

    /// Preprocess + NPU-encode each clip → flat encoder hiddens `[T_ENC*D]` (sequential; untimed by caller).
    pub fn encode_clips(&self, clips: &[&[i16]]) -> Vec<Vec<f32>> {
        clips.iter().map(|s| self.encode_clip(s)).collect()
    }

    /// Batched decoder width B, if the batched backend is enabled (for the O3 bulk scheduler/bench).
    pub fn batch_width(&self) -> Option<usize> {
        self.npu_fused_batch.as_ref().map(|c| c.borrow().batch())
    }

    /// Single-stream (M=1) fused greedy decode for one pre-encoded clip → token ids (incl. prompt).
    pub fn decode_m1_ids(&self, enc_flat: &[f32]) -> Result<Vec<i64>, EngineError> {
        let cell = self.npu_fused.as_ref().expect("decode_m1_ids needs NPU_DECODE_FUSED");
        Ok(self.greedy_decode_fused(&mut cell.borrow_mut(), enc_flat, None)?.0)
    }

    /// Batched (B-stream) fused greedy decode over B pre-encoded clips → per-stream token ids.
    pub fn decode_batch_ids(&self, encs_flat: &[Vec<f32>]) -> Result<Vec<Vec<i64>>, EngineError> {
        let cell = self.npu_fused_batch.as_ref().expect("decode_batch_ids needs NPU_DECODE_FUSED_BATCH");
        let mut dec = cell.borrow_mut();
        let encs: Vec<Array2<f32>> = encs_flat
            .iter()
            .map(|f| Array2::from_shape_vec((T_ENC, self.cfg.d_model), f.clone()).expect("enc shape"))
            .collect();
        self.greedy_decode_fused_batch(&mut dec, &encs)
    }

    /// O3 bench hook: decode N pre-encoded clips bucketed by `sort_key` (sample-count). Returns
    /// (ids in ORIGINAL order, total_computed_slots = Σ_bucket steps×B). `sort=true` length-sorts
    /// before bucketing; `sort=false` keeps input order (length-mixed buckets) for the A/B. The pad
    /// slots that fill a short last bucket are counted in `slots` (they are real dispatch work).
    pub fn decode_bulk_ids(&self, encs_flat: &[Vec<f32>], sort_key: &[usize], sort: bool) -> Result<(Vec<Vec<i64>>, usize), EngineError> {
        let cell = self.npu_fused_batch.as_ref().expect("decode_bulk_ids needs NPU_DECODE_FUSED_BATCH");
        let mut dec = cell.borrow_mut();
        let b = dec.batch();
        let n = encs_flat.len();
        let mut order: Vec<usize> = (0..n).collect();
        if sort {
            order.sort_by_key(|&i| sort_key[i]);
        }
        let mut out: Vec<Vec<i64>> = vec![Vec::new(); n];
        let mut slots = 0usize;
        for chunk in order.chunks(b) {
            let mut idxs: Vec<usize> = chunk.to_vec();
            while idxs.len() < b {
                idxs.push(*chunk.last().unwrap());
            }
            let encs: Vec<Array2<f32>> = idxs
                .iter()
                .map(|&i| Array2::from_shape_vec((T_ENC, self.cfg.d_model), encs_flat[i].clone()).expect("enc shape"))
                .collect();
            let ids = self.greedy_decode_fused_batch(&mut dec, &encs)?;
            slots += dec.last_steps() * b;
            for (k, &orig) in chunk.iter().enumerate() {
                out[orig] = ids[k].clone();
            }
        }
        Ok((out, slots))
    }

    /// Batched greedy decode (lockstep) → per-stream token ids. IDENTICAL control logic to
    /// `greedy_decode_fused` (lang detect, prompt, argmax, EOT, MAX_DECODE), widened to B streams: all
    /// advance one token/step; finished streams feed EOT (ignored) until every stream hits EOT.
    fn greedy_decode_fused_batch(
        &self,
        dec: &mut BatchedFusedDecoder,
        encs: &[Array2<f32>],
    ) -> Result<Vec<Vec<i64>>, EngineError> {
        let b = encs.len();
        dec.precompute_cross_batch(encs)?;
        let lang_logits = dec.step_batch(&vec![self.tokens.sot; b], 0)?;
        let langs: Vec<i64> = lang_logits.iter().map(|l| self.tokens.pick_lang(l)).collect();
        dec.reset()?;
        let prompts: Vec<Vec<i64>> = langs.iter().map(|&l| self.tokens.prompt(l)).collect();
        let mut ids: Vec<Vec<i64>> = prompts.clone();
        let plen = prompts[0].len();
        let mut logits: Vec<Vec<f32>> = Vec::new();
        for pos in 0..plen {
            let toks: Vec<i64> = (0..b).map(|bi| prompts[bi][pos]).collect();
            logits = dec.step_batch(&toks, pos)?;
        }
        let mut next: Vec<i64> = logits.iter().map(|l| argmax(l)).collect();
        let mut active = vec![true; b];
        for bi in 0..b {
            if next[bi] == self.tokens.eot {
                active[bi] = false;
            } else {
                ids[bi].push(next[bi]);
            }
        }
        for step in 0..(MAX_DECODE - 1) {
            if active.iter().all(|&a| !a) {
                break;
            }
            let pos = plen + step;
            let toks: Vec<i64> = (0..b).map(|bi| if active[bi] { next[bi] } else { self.tokens.eot }).collect();
            let logits = dec.step_batch(&toks, pos)?;
            for bi in 0..b {
                if !active[bi] {
                    continue;
                }
                next[bi] = argmax(&logits[bi]);
                if next[bi] == self.tokens.eot {
                    active[bi] = false;
                } else {
                    ids[bi].push(next[bi]);
                }
            }
        }
        dec.dump_phase_timing(); // FUSED_PHASE_TIMING: per-dispatch batched breakdown
        Ok(ids)
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

/// How far back from the 30 s mark to look for a quiet cut.
const SEEK_BACK: usize = 48_000; // 3 s @ 16 kHz
/// Energy window for that search.
const CUT_FRAME: usize = 320; // 20 ms @ 16 kHz

/// Cut points for audio longer than one Whisper window.
///
/// The frontend is fixed at 30 s and the old path simply truncated to it: a 90-minute recording
/// returned its first 30 s with a 200 OK and nothing on the wire to say the rest was dropped -- the
/// same defect already fixed on the parakeet side (`asr::parakeet`, `WIN_MEL`/`DecodeCarry`).
///
/// Each boundary is placed at the QUIETEST 20 ms frame within `SEEK_BACK` of the hard limit, so a
/// cut is far more likely to land in a pause than mid-word. The windows do not overlap: the decode
/// prompt ends in `<|notimestamps|>`, so there are no timestamps to align two overlapping decodes
/// with, and overlap would duplicate text rather than stitch it.
fn window_bounds(samples: &[i16]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut pos = 0usize;
    while pos < samples.len() {
        let hard = (pos + N_SAMPLES).min(samples.len());
        if hard == samples.len() {
            out.push((pos, hard));
            break;
        }
        // SEEK_BACK < N_SAMPLES, so `lo` is always well inside this window and the cut advances.
        let lo = hard - SEEK_BACK;
        let (mut cut, mut quietest) = (hard, u64::MAX);
        let mut f = lo;
        while f + CUT_FRAME <= hard {
            let e: u64 = samples[f..f + CUT_FRAME].iter().map(|&v| { let v = v as i64; (v * v) as u64 }).sum();
            if e < quietest {
                quietest = e;
                cut = f + CUT_FRAME / 2;
            }
            f += CUT_FRAME;
        }
        out.push((pos, cut));
        pos = cut;
    }
    out
}

impl AsrModel for WhisperAsr {
    /// Transcribe audio of ANY length, window by window, each window detecting its own language.
    fn transcribe(&self, samples: &[i16]) -> Result<String, EngineError> {
        // WHISPER_LANG_PIN=1 detects once and reuses that tag for every window. OFF by default,
        // measured 2026-09-03 on a 90 s clip that is English for its first 38 s and Russian after:
        // pinning translated both Russian windows into English ("In Japan, there are about 7000
        // islands...") and dropped content, while per-window detection kept each window in its own
        // language. Pinning is right only for a recording known to be single-language, where a quiet
        // window could otherwise flip the tag.
        let pin_lang = std::env::var("WHISPER_LANG_PIN").is_ok();
        let mut text = String::new();
        let mut lang = None;
        for (a, b) in window_bounds(samples) {
            let (t, l) = self.transcribe_window(&samples[a..b], lang)?;
            if pin_lang {
                lang = Some(l);
            }
            if !t.is_empty() {
                if !text.is_empty() {
                    text.push(' ');
                }
                text.push_str(&t);
            }
        }
        Ok(text)
    }
}

impl WhisperAsr {
    /// One <= 30 s window: preprocess, encode, decode. Returns the text and the language tag used,
    /// so the caller can pin it across the remaining windows.
    fn transcribe_window(&self, samples: &[i16], lang: Option<i64>) -> Result<(String, i64), EngineError> {
        let timing = std::env::var("WHISPER_TIMING").is_ok();
        let t_e2e = std::time::Instant::now();

        // i16 -> f32 in [-1,1], zero-pad to exactly N_SAMPLES (preprocessor.onnx is fixed-shape).
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
            .map_err(|e| EngineError::Device(format!("preprocessor: {e}")))?;
        // input_features: [1, 80, 3000] flat channel-major -> Array2 [80, 3000] for the encoder.
        let feats = feat.f32(0);
        let mut mel = Array2::<f32>::zeros((self.cfg.n_mels, N_FRAMES));
        for c in 0..self.cfg.n_mels {
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
        // greedy_decode and the NPU_DECODE*/fused decode machinery it dispatches into
        // (whisper_decoder.rs: FusedDecoder/BatchedFusedDecoder/HostDecoder's resident-context
        // step/precompute_cross/dispatch_* chain) propagate real device errors instead of
        // panicking. greedy_decode_onnx (the no-NPU-decoder fallback) is unconverted -- see the
        // note on `greedy_decode` above.
        let (ids, lang) = self.greedy_decode(&flat, lang)?;
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
        Ok((text, lang))
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The regression this windowing exists for: audio past 30 s used to be dropped silently.
    #[test]
    fn long_audio_is_covered_gaplessly_instead_of_truncated() {
        let samples = vec![1000i16; 75 * 16_000]; // 75 s
        let w = window_bounds(&samples);
        assert!(w.len() >= 3, "75 s needs at least 3 windows, got {}", w.len());
        assert_eq!(w[0].0, 0);
        assert_eq!(w.last().unwrap().1, samples.len(), "the tail must be transcribed, not dropped");
        for pair in w.windows(2) {
            assert_eq!(pair[0].1, pair[1].0, "windows must be contiguous, no gap and no overlap");
        }
        for &(a, b) in &w {
            assert!(b > a, "no empty window");
            assert!(b - a <= N_SAMPLES, "a window must fit the fixed 30 s frontend");
        }
    }

    /// Short audio keeps the exact old shape: one window, no cut search.
    #[test]
    fn audio_under_one_window_stays_one_window() {
        let samples = vec![1000i16; 5 * 16_000];
        assert_eq!(window_bounds(&samples), vec![(0, samples.len())]);
        assert!(window_bounds(&[]).is_empty(), "no audio, no work");
    }

    /// The point of searching backwards: a boundary lands in a pause, not mid-word.
    #[test]
    fn a_cut_prefers_a_silent_gap_to_the_hard_limit() {
        let mut samples = vec![8000i16; 45 * 16_000];
        // 200 ms of silence starting at 28.0 s -- inside SEEK_BACK of the 30 s hard limit.
        let gap = 28 * 16_000;
        for v in &mut samples[gap..gap + 3_200] {
            *v = 0;
        }
        let cut = window_bounds(&samples)[0].1;
        assert!((gap..gap + 3_200).contains(&cut), "cut {cut} should be inside the silence at {gap}");
    }
}
