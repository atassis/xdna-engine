//! Parakeet-tdt ASR hosted in the engine: nemo128 mel preproc (ONNX) + FastConformer encoder (NPU)
//! + TDT greedy decode. Reuses the validated npu_parakeet encoder + decode; not a rewrite.

use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;

use ndarray::Array2;
use npu_onnx::{Env, Session, Tensor};
use npu_parakeet::config::ModelCfg;
use npu_parakeet::encoder::FastConformerEncoder;

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
    pub fn build(cfg: &ScenarioConfig, root: &Path) -> Self {
        let env = Env::new().expect("onnx env"); // Env::new() returns Rc<Env>
        let pk = root.join(&cfg.artifacts.weights);
        let load = |f: &str| {
            Session::load(&env, pk.join(f).to_str().unwrap())
                .unwrap_or_else(|e| panic!("load {f}: {e}"))
        };
        let prep = load("preprocessor.onnx");
        let dj = load("decoder_joint.onnx");
        let xroot = std::env::var("NPU_XCLBIN_ROOT")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| root.to_path_buf());
        let enc = FastConformerEncoder::new_npu(&pk.join("encoder"), ModelCfg::PARAKEET_V3, &xroot);
        let vocab = load_vocab(&pk.join("vocab.txt"));
        ParakeetAsr { prep, dj, enc, vocab, _env: env }
    }

    /// TDT duration-split greedy decode (mirrors onnx-asr _AsrWithTransducerDecoding +
    /// NemoConformerTdt). Lifted verbatim from parakeet_serve.rs.
    fn tdt_decode(&self, encoded: &Array2<f32>, valid: usize) -> Vec<i64> {
        let mut st1 = vec![0f32; STATE_LAYERS * STATE_DIM];
        let mut st2 = vec![0f32; STATE_LAYERS * STATE_DIM];
        let mut tokens: Vec<i64> = Vec::new();
        let (mut t, mut emitted) = (0usize, 0usize);
        while t < valid {
            let frame = encoded.row(t).to_vec(); // [1024]
            let last = *tokens.last().unwrap_or(&BLANK) as i32;
            let (out, nst1, nst2) = self.run_dj(&frame, last, &st1, &st2);
            let token = argmax(&out[..VOCAB]); // 8193 token logits
            let step = argmax(&out[VOCAB..VOCAB + N_DUR]) as usize; // duration 0..4
            if token != BLANK {
                st1 = nst1; // commit predictor state on emission
                st2 = nst2;
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
        tokens
    }

    fn run_dj(
        &self,
        frame: &[f32],
        last_tok: i32,
        st1: &[f32],
        st2: &[f32],
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
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
            .expect("decoder_joint");
        (out.f32(0).to_vec(), out.f32(1).to_vec(), out.f32(2).to_vec())
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
    fn transcribe(&self, samples: &[i16]) -> String {
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
            .expect("preprocessor");
        let t = feat.shape(0)[2] as usize; // [1,128,T]
        let feats = feat.f32(0); // [128*T] channel-major
        let teff = t.min(WIN_MEL);
        let mut mel = Array2::<f32>::zeros((MEL, teff));
        for c in 0..MEL {
            for ti in 0..teff {
                mel[[c, ti]] = feats[c * t + ti];
            }
        }
        let encoded = self.enc.encode(&mel); // [T', 1024] on the NPU
        let valid = encoded.nrows();
        let ids = self.tdt_decode(&encoded, valid);
        self.detokenize(&ids)
    }
}

fn load_vocab(path: &Path) -> HashMap<i64, String> {
    let txt = std::fs::read_to_string(path).expect("vocab");
    let mut m = HashMap::new();
    for line in txt.lines() {
        if let Some((tok, id)) = line.rsplit_once(' ') {
            if let Ok(id) = id.trim().parse::<i64>() {
                m.insert(id, tok.replace('\u{2581}', " "));
            }
        }
    }
    m
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
