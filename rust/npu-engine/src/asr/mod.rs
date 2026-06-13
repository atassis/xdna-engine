//! ASR pipeline: GigaAM-v3 encoder (wrapped behind Encoder) + ONNX mel preproc/RNNT decode.

pub mod parakeet;

use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;

use ndarray::Array2;
use npu_asr::encoder::Encoder as GigaEncoder;
use npu_asr::weights::WeightStore;
use npu_onnx::{Env, Session, Tensor};
use npu_xrt::Device;

use crate::config::ScenarioConfig;
use crate::pipeline::{AsrModel, Encoder};

const MEL: usize = 64;
const WIN: usize = 1600;
const D: usize = 768;
const PRED: usize = 320;
const BLANK: i64 = 33;
const MAX_TOK: usize = 3;

/// Newtype adapting the GigaAM encoder to the engine's Encoder trait.
pub struct ConformerEncoder { inner: GigaEncoder, ws: WeightStore }
impl ConformerEncoder {
    pub fn new(dev: Rc<Device>, root: &Path, weights_dir: &Path) -> Self {
        let ws = WeightStore::load(weights_dir).expect("encoder weights");
        let n = ws.nblocks();
        let inner = GigaEncoder::new(dev, root, &ws, n);
        ConformerEncoder { inner, ws }
    }
    pub fn subsample(&self, audio: &Array2<f32>) -> Array2<f32> { self.inner.subsample(&self.ws, audio) }
}
impl Encoder for ConformerEncoder {
    fn forward_last(&self, x: &Array2<f32>, valid_len: usize) -> Array2<f32> {
        self.inner.forward_last(x, valid_len)
    }
}

pub struct AsrPipeline {
    prep: Session, decoder: Session, joint: Session,
    enc: ConformerEncoder,
    vocab: HashMap<i64, String>,
    _env: Rc<Env>,
}

impl AsrPipeline {
    pub fn build(cfg: &ScenarioConfig, root: &Path, dev: Rc<Device>) -> Self {
        let env = Env::new().expect("onnx env");
        let base = root.join(&cfg.artifacts.weights); // e.g. "artifacts" (parent of asr/ and encoder/)
        let asr = base.join("asr");
        let load = |f: &str| Session::load(&env, asr.join(f).to_str().unwrap()).expect("onnx load");
        let prep = load("preprocessor.onnx");
        let decoder = load("decoder.onnx");
        let joint = load("joint.onnx");
        let enc = ConformerEncoder::new(dev, root, &base.join("encoder"));
        let vocab = load_vocab(&asr.join("vocab.txt"));
        AsrPipeline { prep, decoder, joint, enc, vocab, _env: env }
    }

    fn decode(&self, encoded: &Array2<f32>, valid: usize) -> Vec<i64> {
        let mut h = vec![0f32; PRED]; let mut c = vec![0f32; PRED];
        let mut dec = self.run_decoder(BLANK, &mut h, &mut c);
        let mut tokens = Vec::new();
        let (mut t, mut emitted) = (0usize, 0usize);
        while t < valid {
            let frame = encoded.row(t).to_vec();
            let logits = self.run_joint(&frame, &dec);
            let tok = argmax(&logits);
            if tok != BLANK {
                tokens.push(tok); emitted += 1;
                dec = self.run_decoder(tok, &mut h, &mut c);
                if emitted == MAX_TOK { t += 1; emitted = 0; }
            } else { t += 1; emitted = 0; }
        }
        tokens
    }
    fn run_decoder(&self, x: i64, h: &mut Vec<f32>, c: &mut Vec<f32>) -> Vec<f32> {
        let xv = [x];
        let out = self.decoder.run(
            &[("x", Tensor::I64(&xv, vec![1, 1])),
              ("h.1", Tensor::F32(h, vec![1, 1, PRED as i64])),
              ("c.1", Tensor::F32(c, vec![1, 1, PRED as i64]))],
            &["dec", "h", "c"]).expect("decoder");
        let (d, nh, nc) = (out.f32(0).to_vec(), out.f32(1).to_vec(), out.f32(2).to_vec());
        *h = nh; *c = nc; d
    }
    fn run_joint(&self, enc: &[f32], dec: &[f32]) -> Vec<f32> {
        let out = self.joint.run(
            &[("enc", Tensor::F32(enc, vec![1, D as i64, 1])),
              ("dec", Tensor::F32(dec, vec![1, PRED as i64, 1]))],
            &["joint"]).expect("joint");
        out.f32(0).to_vec()
    }
    fn detokenize(&self, ids: &[i64]) -> String {
        ids.iter().map(|id| self.vocab.get(id).map(|x| x.as_str()).unwrap_or("")).collect::<String>().trim().to_string()
    }
}

impl AsrModel for AsrPipeline {
    fn transcribe(&self, samples: &[i16]) -> String {
        let wav: Vec<f32> = samples.iter().map(|&s| s as f32 / 32768.0).collect();
        let n = wav.len() as i64;
        let lens = [n];
        let feat = self.prep.run(
            &[("waveforms", Tensor::F32(&wav, vec![1, n])),
              ("waveforms_lens", Tensor::I64(&lens, vec![1]))],
            &["features", "features_lens"]).expect("preproc");
        let t = feat.shape(0)[2] as usize;
        let feats = feat.f32(0);
        let teff = t.min(WIN);
        let mut audio = Array2::<f32>::zeros((MEL, WIN));
        for c in 0..MEL { for ti in 0..teff { audio[[c, ti]] = feats[c * t + ti]; } }
        let valid = (teff.max(1) - 1) / 4 + 1;
        let x0 = self.enc.subsample(&audio);
        let encoded = self.enc.forward_last(&x0, valid);
        let ids = self.decode(&encoded, valid);
        self.detokenize(&ids)
    }
}

fn argmax(v: &[f32]) -> i64 {
    let mut b = 0usize; for i in 1..v.len() { if v[i] > v[b] { b = i; } } b as i64
}
fn load_vocab(path: &Path) -> HashMap<i64, String> {
    let txt = std::fs::read_to_string(path).expect("vocab");
    let mut m = HashMap::new();
    for line in txt.lines() {
        if let Some((tok, id)) = line.rsplit_once(' ') {
            if let Ok(id) = id.trim().parse::<i64>() { m.insert(id, tok.replace('\u{2581}', " ")); }
        }
    }
    m
}
