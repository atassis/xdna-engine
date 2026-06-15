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

use std::path::Path;
use std::rc::Rc;

use ndarray::Array2;
use npu_onnx::{Env, Session, Tensor};
use npu_whisper::config::WhisperCfg;
use npu_whisper::encoder::WhisperEncoder;
use tokenizers::Tokenizer;

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
struct Kv {
    name: String, // the `past_key_values.*` input name this entry feeds on the next step
    data: Vec<f32>,
    shape: Vec<i64>,
}

pub struct WhisperAsr {
    prep: Session,
    decoder: Session,      // decoder_model.onnx — step 0, no past, emits encoder+decoder present
    decoder_past: Session, // decoder_with_past_model.onnx — steps >=1, cached
    enc: WhisperEncoder,
    tok: Tokenizer,
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
        WhisperAsr { prep, decoder, decoder_past, enc, tok, _env: env }
    }

    /// The 48 `present.*` output names emitted by `decoder_model.onnx` (step 0), in graph order:
    /// `present.{0..11}.{decoder,encoder}.{key,value}`.
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

    /// The 24 *decoder* `present.*` output names emitted by `decoder_with_past_model.onnx` (steps>=1),
    /// in graph order: `present.{0..11}.decoder.{key,value}`. (Encoder KV are not re-emitted.)
    fn past_present_names() -> Vec<String> {
        let mut v = Vec::with_capacity(2 * N_LAYERS);
        for l in 0..N_LAYERS {
            for kv in ["key", "value"] {
                v.push(format!("present.{l}.decoder.{kv}"));
            }
        }
        v
    }

    /// Step 0: run the no-past graph over the full prompt + encoder hidden states. Returns the
    /// last-position logits and the full 48-entry KV cache (decoder + encoder present), each keyed
    /// by the `past_key_values.*` input name it feeds on subsequent cached steps.
    fn decode_step0(
        &self,
        ids: &[i64],
        enc_shape: &[i64],
        encoder_hidden: &[f32],
    ) -> (Vec<f32>, Vec<Kv>) {
        let l = ids.len() as i64;
        let present_names = Self::step0_present_names();
        let out_names: Vec<&str> = std::iter::once("logits")
            .chain(present_names.iter().map(|s| s.as_str()))
            .collect();
        let out = self
            .decoder
            .run(
                &[
                    ("input_ids", Tensor::I64(ids, vec![1, l])),
                    ("encoder_hidden_states", Tensor::F32(encoder_hidden, enc_shape.to_vec())),
                ],
                &out_names,
            )
            .expect("whisper decoder (step 0)");
        let logits_all = out.f32(0); // [1, L, VOCAB] row-major
        let last = (ids.len() - 1) * VOCAB;
        let logits = logits_all[last..last + VOCAB].to_vec();
        // present.* outputs are at indices 1..=48; map name `present.X` -> input `past_key_values.X`.
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

    /// Steps >=1: run the cached graph over a single new token + the prior KV. Returns the logits
    /// and the NEW KV cache: refreshed decoder KV (grown by 1) + the encoder KV carried unchanged.
    fn decode_step_cached(&self, tok: i64, prev: &[Kv]) -> (Vec<f32>, Vec<Kv>) {
        let ids = [tok];
        // Inputs: input_ids + all 48 past_key_values.* (borrow prev's data).
        let mut inputs: Vec<(&str, Tensor)> = Vec::with_capacity(1 + prev.len());
        inputs.push(("input_ids", Tensor::I64(&ids, vec![1, 1])));
        for kv in prev {
            inputs.push((kv.name.as_str(), Tensor::F32(&kv.data, kv.shape.clone())));
        }
        let present_names = Self::past_present_names(); // 24 decoder present
        let out_names: Vec<&str> = std::iter::once("logits")
            .chain(present_names.iter().map(|s| s.as_str()))
            .collect();
        let out = self
            .decoder_past
            .run(&inputs, &out_names)
            .expect("whisper decoder (cached)");
        // length-1 step -> logits is [1,1,VOCAB]; take the whole row.
        let logits = out.f32(0).to_vec();
        // Refreshed decoder KV (outputs 1..=24).
        let new_decoder: Vec<Kv> = present_names
            .iter()
            .enumerate()
            .map(|(i, pname)| Kv {
                name: pname.replacen("present", "past_key_values", 1),
                data: out.f32(i + 1).to_vec(),
                shape: out.shape(i + 1),
            })
            .collect();
        // Reassemble the full 48-entry cache: new decoder KV where available, encoder KV carried
        // over unchanged from `prev` (matched by input name).
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

    /// Detect the language token with a 1-step `[SOT]` decode (argmax restricted to the language
    /// block). Returns the chosen language token id. (Uses the no-past graph; KV discarded.)
    fn detect_lang(&self, enc_shape: &[i64], encoder_hidden: &[f32]) -> i64 {
        let (logits, _kv) = self.decode_step0(&[SOT], enc_shape, encoder_hidden);
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
    fn greedy_decode(&self, encoder_hidden: &[f32]) -> Vec<i64> {
        let enc_shape = vec![1, T_ENC as i64, D as i64];
        let lang = self.detect_lang(&enc_shape, encoder_hidden);
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
        // i16 -> f32 in [-1,1], pad/truncate to exactly N_SAMPLES (preprocessor.onnx is fixed-shape).
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
        let encoded = self.enc.forward_last(&mel); // [1500, 768] on the NPU
        // row-major [1500*768] for the decoder's encoder_hidden_states[1,1500,768]
        let std = encoded.as_standard_layout();
        let flat: Vec<f32> = std.iter().copied().collect();
        let ids = self.greedy_decode(&flat);
        self.detokenize(&ids)
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
