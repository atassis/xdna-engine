//! Parakeet-TDT decoder -- host (ndarray f32) reference.
//!
//! Mirrors `scripts/parakeet_tdt_decoder_ref.py` (the NumPy golden, validated vs the
//! onnx_asr / decoder_joint.onnx oracle at rel-L2 ~5e-7 and exact greedy token parity).
//! Two parts + a loop, all CPU (the TDT decoder is tiny, M=1 GEMV-shaped):
//!   * prediction network: token embedding lookup + 2-layer LSTM (ONNX gate order i,o,f,c)
//!   * joint network: enc_t + pred_u -> [vocab(8193) + durations(5)] logits
//!   * greedy token/duration emit loop with frame-skipping (mirrors onnx_asr NeMo TDT)
//!
//! NPU port map (per aie2p-architecture-and-roofline): embedding -> parallel_lookup; the LSTM/joint
//! matmuls -> M=1 GEMV (mac/accumulate, NOT mmul -- overhead-bound); argmax -> max_cmp.
//!
//! Weights are loaded from the .npy files produced by
//! `scripts/parakeet_tdt_decoder_ref.py --dump-weights`
//! (artifacts/parakeet/decoder/weights/).

use std::path::Path;

use ndarray::prelude::*;
use ndarray_npy::read_npy;

use crate::onnx_init::{self, Tensor as OnnxTensor};

pub const HIDDEN: usize = 640;
pub const VOCAB_SIZE: usize = 8193; // token logits incl <blk>
pub const NUM_DURATIONS: usize = 5; // output 8198 = 8193 + 5
pub const BLANK_IDX: usize = 8192;
pub const MAX_TOKENS_PER_STEP: usize = 10;

fn load_2d(p: &Path) -> Array2<f32> {
    read_npy(p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()))
}

fn load_1d(p: &Path) -> Array1<f32> {
    read_npy(p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()))
}

struct LstmLayer {
    w: Array2<f32>, // [4*hidden, input]
    r: Array2<f32>, // [4*hidden, hidden]
    b: Array1<f32>, // [8*hidden] = Wb(iofc) ++ Rb(iofc)
}

pub struct TdtDecoder {
    embed: Array2<f32>, // [vocab, hidden]
    lstm: [LstmLayer; 2],
    enc_w: Array2<f32>, // [1024, hidden]
    enc_b: Array1<f32>,
    pred_w: Array2<f32>, // [hidden, hidden]
    pred_b: Array1<f32>,
    joint_w: Array2<f32>, // [hidden, vocab+dur]
    joint_b: Array1<f32>,
}

/// Per-layer LSTM hidden/cell state.
#[derive(Clone)]
pub struct PredState {
    h: [Array1<f32>; 2],
    c: [Array1<f32>; 2],
}

impl PredState {
    pub fn zeros() -> Self {
        Self {
            h: [Array1::zeros(HIDDEN), Array1::zeros(HIDDEN)],
            c: [Array1::zeros(HIDDEN), Array1::zeros(HIDDEN)],
        }
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

impl TdtDecoder {
    pub fn load(dir: &Path) -> Self {
        let lstm = [
            LstmLayer {
                w: load_2d(&dir.join("lstm0_W.npy")),
                r: load_2d(&dir.join("lstm0_R.npy")),
                b: load_1d(&dir.join("lstm0_B.npy")),
            },
            LstmLayer {
                w: load_2d(&dir.join("lstm1_W.npy")),
                r: load_2d(&dir.join("lstm1_R.npy")),
                b: load_1d(&dir.join("lstm1_B.npy")),
            },
        ];
        Self {
            embed: load_2d(&dir.join("embed.npy")),
            lstm,
            enc_w: load_2d(&dir.join("enc_W.npy")),
            enc_b: load_1d(&dir.join("enc_b.npy")),
            pred_w: load_2d(&dir.join("pred_W.npy")),
            pred_b: load_1d(&dir.join("pred_b.npy")),
            joint_w: load_2d(&dir.join("joint_W.npy")),
            joint_b: load_1d(&dir.join("joint_b.npy")),
        }
    }

    /// Load weights straight out of `decoder_joint.onnx`'s initializers (`onnx_init.rs`), with
    /// no onnxruntime and no `--dump-weights` pre-processing step. Mirrors
    /// `scripts/parakeet_tdt_decoder_ref.py::load_weights` field-for-field: fixed names for the
    /// embedding/bias tensors, LSTM node input names for the two LSTM weight triples (ONNX gives
    /// LSTM weights compiler-generated names, so they're found via the node, not a string), and
    /// exact-shape lookup for the three `onnx::MatMul_*` joint projections (unique per shape --
    /// verified against the real decoder_joint.onnx, see the crate's tests).
    pub fn load_from_onnx(onnx_path: &Path) -> onnx_init::Result<Self> {
        let g = onnx_init::load(onnx_path)?;
        let get = |name: &str| -> onnx_init::Result<&OnnxTensor> {
            g.initializers.get(name).ok_or_else(|| format!("missing initializer {name}"))
        };
        let to_2d = |t: &OnnxTensor| -> Array2<f32> {
            // LSTM W/R come as [num_directions=1, rows, cols]; drop the leading 1.
            let (rows, cols) = match t.dims.as_slice() {
                [r, c] => (*r as usize, *c as usize),
                [1, r, c] => (*r as usize, *c as usize),
                d => panic!("to_2d: unexpected dims {d:?}"),
            };
            Array2::from_shape_vec((rows, cols), t.data.clone()).unwrap()
        };
        let to_1d = |t: &OnnxTensor| -> Array1<f32> { Array1::from_vec(t.data.clone()) };

        let lstm_nodes: Vec<&(String, Vec<String>)> =
            g.nodes.iter().filter(|(op, _)| op == "LSTM").collect();
        if lstm_nodes.len() != 2 {
            return Err(format!("expected 2 LSTM nodes, found {}", lstm_nodes.len()));
        }
        let mut lstm_iter = lstm_nodes.into_iter();
        let mut load_layer = || -> onnx_init::Result<LstmLayer> {
            let (_, inputs) = lstm_iter.next().unwrap();
            Ok(LstmLayer {
                w: to_2d(get(&inputs[1])?),
                r: to_2d(get(&inputs[2])?),
                b: to_1d(get(&inputs[3])?),
            })
        };
        let lstm = [load_layer()?, load_layer()?];

        let find_by_shape = |dims: [i64; 2]| -> onnx_init::Result<&OnnxTensor> {
            g.initializers
                .values()
                .find(|t| t.dims.as_slice() == dims)
                .ok_or_else(|| format!("no initializer with shape {dims:?}"))
        };

        Ok(TdtDecoder {
            embed: to_2d(get("decoder.prediction.embed.weight")?),
            lstm,
            enc_w: to_2d(find_by_shape([1024, HIDDEN as i64])?),
            enc_b: to_1d(get("joint.enc.bias")?),
            pred_w: to_2d(find_by_shape([HIDDEN as i64, HIDDEN as i64])?),
            pred_b: to_1d(get("joint.pred.bias")?),
            joint_w: to_2d(find_by_shape([HIDDEN as i64, (VOCAB_SIZE + NUM_DURATIONS) as i64])?),
            joint_b: to_1d(get("joint.joint_net.2.bias")?),
        })
    }

    /// One ONNX-LSTM cell step. Returns (h_new, c_new).
    fn lstm_step(
        layer: &LstmLayer,
        x: &Array1<f32>,
        h: &Array1<f32>,
        c: &Array1<f32>,
    ) -> (Array1<f32>, Array1<f32>) {
        let hh = HIDDEN;
        // z = W*x + R*h + Wb + Rb ; gate order i,o,f,c
        let mut z = layer.w.dot(x) + layer.r.dot(h);
        let wb = layer.b.slice(s![0..4 * hh]);
        let rb = layer.b.slice(s![4 * hh..8 * hh]);
        z += &wb;
        z += &rb;
        let mut h_new = Array1::<f32>::zeros(hh);
        let mut c_new = Array1::<f32>::zeros(hh);
        for j in 0..hh {
            let i = sigmoid(z[j]);
            let o = sigmoid(z[hh + j]);
            let f = sigmoid(z[2 * hh + j]);
            let g = z[3 * hh + j].tanh();
            let cj = f * c[j] + i * g;
            c_new[j] = cj;
            h_new[j] = o * cj.tanh();
        }
        (h_new, c_new)
    }

    /// Embedding lookup + 2-layer LSTM. Returns (pred_u[hidden], new_state).
    pub fn prednet_step(&self, token: usize, st: &PredState) -> (Array1<f32>, PredState) {
        let x = self.embed.row(token).to_owned(); // parallel_lookup on NPU
        let (h0, c0) = Self::lstm_step(&self.lstm[0], &x, &st.h[0], &st.c[0]);
        let (h1, c1) = Self::lstm_step(&self.lstm[1], &h0, &st.h[1], &st.c[1]);
        let new = PredState {
            h: [h0, h1.clone()],
            c: [c0, c1],
        };
        (h1, new)
    }

    /// Joint network: enc_t + pred_u -> [vocab+durations] logits.
    pub fn joint(&self, enc_t: ArrayView1<f32>, pred_u: &Array1<f32>) -> Array1<f32> {
        let enc_proj = self.enc_w.t().dot(&enc_t) + &self.enc_b; // [hidden] GEMV
        let pred_proj = self.pred_w.t().dot(pred_u) + &self.pred_b; // [hidden] GEMV
        let mut act = enc_proj + pred_proj;
        act.mapv_inplace(|v| v.max(0.0)); // ReLU
        self.joint_w.t().dot(&act) + &self.joint_b // [vocab+dur] GEMV
    }

    /// Greedy TDT decode with frame-skipping. `enc` is [T,1024]. Returns emitted token ids.
    pub fn greedy_decode(&self, enc: &Array2<f32>, enc_len: usize) -> Vec<usize> {
        let st0 = PredState::zeros();
        let (mut cur_pred, mut cur_state) = self.prednet_step(BLANK_IDX, &st0);
        let mut tokens: Vec<usize> = Vec::new();
        let mut t = 0usize;
        let mut emitted = 0usize;
        while t < enc_len {
            let logits = self.joint(enc.row(t), &cur_pred);
            let token = argmax(logits.slice(s![0..VOCAB_SIZE]));
            let step = argmax(logits.slice(s![VOCAB_SIZE..VOCAB_SIZE + NUM_DURATIONS]));
            if token != BLANK_IDX {
                let (p, s) = self.prednet_step(token, &cur_state);
                cur_pred = p;
                cur_state = s;
                tokens.push(token);
                emitted += 1;
            }
            if step > 0 {
                t += step;
                emitted = 0;
            } else if token == BLANK_IDX || emitted == MAX_TOKENS_PER_STEP {
                t += 1;
                emitted = 0;
            }
        }
        tokens
    }
}

fn argmax(v: ArrayView1<f32>) -> usize {
    let mut best = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > bv {
            bv = x;
            best = i;
        }
    }
    best
}
