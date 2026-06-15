//! Whisper-small DECODER reimplemented on the host in f32 — the correctness foundation for later
//! offloading the decoder matmuls to the NPU. The decoder presently runs only as a monolithic ONNX
//! graph (`decoder_model.onnx` / `decoder_with_past_model.onnx`); to be able to route its matmuls to
//! the device we must own the forward pass op-by-op. This module is the all-host reference; its
//! numerical parity vs. the ONNX graph is proven by `bin/verify_whisper_decode` (rel-L2 <= 1e-3 +
//! identical argmax over a fixed greedy sequence).
//!
//! Architecture (pre-norm transformer decoder, whisper-small: d=768, 12 layers, 12 heads, hd=64,
//! ffn=3072, vocab=51865). Input to step t: `embed_tokens[token] + embed_positions[pos]`. Per layer:
//!   1. h = x + self_out( self_attn( ln_self(x) ) )   — CAUSAL self-attn over the growing self-KV.
//!   2. h = h + cross_out( cross_attn( ln_cross(h) ) ) — cross-attn: K/V from the encoder (cached).
//!   3. h = h + fc2( gelu( fc1( ln_final(h) ) ) ).
//! After all layers: logits = proj_out( ln_post(h) ).
//!
//! Linear weights are stored `[K_in, N_out]` (already transposed for `x @ W`), matching the encoder
//! extractor; k_proj / cross_k biases are zeros. LayerNorm eps = 1e-5 (population variance), matching
//! `npu_asr_host::layer_norm` and the encoder.

use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;

use ndarray::prelude::*;
use ndarray_npy::read_npy;
use npu_asr::ctx_decode::{CtxDecode, DecodeEpi, DecodeWeight};
use npu_asr_host::gelu;
use npu_xrt::Device;

const D: usize = 768;
const N_LAYERS: usize = 12;
const N_HEADS: usize = 12;
const HEAD_DIM: usize = 64; // D / N_HEADS
const FFN: usize = 3072;
const VOCAB: usize = 51865;
const LN_EPS: f32 = 1e-5;

/// A keyed bag of fp32 tensors (one directory's worth of `.npy`, keyed by file stem).
/// Mirrors `npu_whisper::weights::TensorMap`.
struct TensorMap {
    map: HashMap<String, ArrayD<f32>>,
}

impl TensorMap {
    fn v(&self, key: &str) -> Array1<f32> {
        self.get(key).clone().into_dimensionality::<Ix1>().unwrap_or_else(|_| panic!("`{key}` not 1-D"))
    }
    fn m(&self, key: &str) -> Array2<f32> {
        self.get(key).clone().into_dimensionality::<Ix2>().unwrap_or_else(|_| panic!("`{key}` not 2-D"))
    }
    fn get(&self, key: &str) -> &ArrayD<f32> {
        self.map.get(key).unwrap_or_else(|| panic!("missing weight `{key}`"))
    }
}

fn load_dir(dir: &Path) -> std::io::Result<HashMap<String, ArrayD<f32>>> {
    let mut map = HashMap::new();
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.extension().and_then(|e| e.to_str()) == Some("npy") {
            let stem = path.file_stem().unwrap().to_string_lossy().into_owned();
            let arr: ArrayD<f32> =
                read_npy(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
            map.insert(stem, arr);
        }
    }
    Ok(map)
}

/// Per-layer host-f32 weights (linear weights stored `[K_in, N_out]`).
struct LayerWeights {
    ln_self_w: Array1<f32>,
    ln_self_b: Array1<f32>,
    q_w: Array2<f32>,
    q_b: Array1<f32>,
    k_w: Array2<f32>,
    k_b: Array1<f32>, // zeros
    v_w: Array2<f32>,
    v_b: Array1<f32>,
    out_w: Array2<f32>,
    out_b: Array1<f32>,

    ln_cross_w: Array1<f32>,
    ln_cross_b: Array1<f32>,
    cross_q_w: Array2<f32>,
    cross_q_b: Array1<f32>,
    cross_k_w: Array2<f32>,
    cross_k_b: Array1<f32>, // zeros
    cross_v_w: Array2<f32>,
    cross_v_b: Array1<f32>,
    cross_out_w: Array2<f32>,
    cross_out_b: Array1<f32>,

    ln_final_w: Array1<f32>,
    ln_final_b: Array1<f32>,
    fc1_w: Array2<f32>,
    fc1_b: Array1<f32>,
    fc2_w: Array2<f32>,
    fc2_b: Array1<f32>,
}

/// All decoder weights loaded from `artifacts/whisper-small/whisper_decoder/`.
pub struct WhisperDecoderWeights {
    layers: Vec<LayerWeights>,
    embed_tokens: Array2<f32>,    // [vocab, 768]
    embed_positions: Array2<f32>, // [448, 768]
    proj_out_w: Array2<f32>,      // [768, vocab]
    ln_post_w: Array1<f32>,
    ln_post_b: Array1<f32>,
}

impl WhisperDecoderWeights {
    /// `dir` points at `artifacts/whisper-small/whisper_decoder` (root globals + L0..L11/).
    pub fn load(dir: &Path) -> std::io::Result<Self> {
        let root = TensorMap { map: load_dir(dir)? };
        let layers = (0..N_LAYERS)
            .map(|i| {
                let b = TensorMap {
                    map: load_dir(&dir.join(format!("L{i}"))).expect("load decoder layer dir"),
                };
                LayerWeights {
                    ln_self_w: b.v("ln_self.weight"),
                    ln_self_b: b.v("ln_self.bias"),
                    q_w: b.m("q.weight"),
                    q_b: b.v("q.bias"),
                    k_w: b.m("k.weight"),
                    k_b: b.v("k.bias"),
                    v_w: b.m("v.weight"),
                    v_b: b.v("v.bias"),
                    out_w: b.m("out.weight"),
                    out_b: b.v("out.bias"),
                    ln_cross_w: b.v("ln_cross.weight"),
                    ln_cross_b: b.v("ln_cross.bias"),
                    cross_q_w: b.m("cross_q.weight"),
                    cross_q_b: b.v("cross_q.bias"),
                    cross_k_w: b.m("cross_k.weight"),
                    cross_k_b: b.v("cross_k.bias"),
                    cross_v_w: b.m("cross_v.weight"),
                    cross_v_b: b.v("cross_v.bias"),
                    cross_out_w: b.m("cross_out.weight"),
                    cross_out_b: b.v("cross_out.bias"),
                    ln_final_w: b.v("ln_final.weight"),
                    ln_final_b: b.v("ln_final.bias"),
                    fc1_w: b.m("fc1.weight"),
                    fc1_b: b.v("fc1.bias"),
                    fc2_w: b.m("fc2.weight"),
                    fc2_b: b.v("fc2.bias"),
                }
            })
            .collect();
        Ok(WhisperDecoderWeights {
            layers,
            embed_tokens: root.m("embed_tokens"),
            embed_positions: root.m("embed_positions"),
            proj_out_w: root.m("proj_out.weight"),
            ln_post_w: root.v("ln_post.weight"),
            ln_post_b: root.v("ln_post.bias"),
        })
    }
}

/// Per-layer mutable decode state: the growing self-attention KV cache (one row appended per token)
/// and the encoder cross-attention K/V (computed once per utterance via `precompute_cross`).
#[derive(Default)]
struct LayerState {
    self_k: Vec<f32>, // flat [S, 768] row-major, grows by one row per step
    self_v: Vec<f32>,
    n_self: usize, // number of cached self positions (rows in self_k/self_v)
    cross_k: Array2<f32>, // [T_enc, 768]
    cross_v: Array2<f32>, // [T_enc, 768]
}

/// Per-layer NPU-resident decoder weights (registered once at init). Each is a single resident bf16
/// `[K, N_pad]` matrix on the device; `gemv` runs `x · W` (M=1) against it. The fused self-QKV packs
/// q/k/v into one `[768, 2304]` weight so self-attn projections cost ONE dispatch (its output is then
/// sliced into q,k,v on the host). All bias adds / GELU stay on the host after readback.
struct NpuLayer {
    qkv: DecodeWeight,       // fused self q|k|v: [768, 2304], DecodeEpi::Bias (concat q/k/v biases)
    self_out: DecodeWeight,  // [768, 768], DecodeEpi::Bias
    cross_q: DecodeWeight,   // [768, 768], DecodeEpi::Bias
    cross_out: DecodeWeight, // [768, 768], DecodeEpi::Bias
    fc1: DecodeWeight,       // [768, 3072], DecodeEpi::Bias (GELU on host after)
    fc2: DecodeWeight,       // [3072, 768], DecodeEpi::Bias
}

/// On-NPU per-token matmul backend for the decoder: a resident [`CtxDecode`] primitive plus the
/// per-layer registered weights. Construction registers every per-token weight ONCE (and loads the
/// needed xclbins). `proj_out` (logits, [768, 51865]) is intentionally NOT here: its N pads to 51872
/// which violates the whole_array `N % (n*cols)=256 == 0` tiling constraint, so logits stay on host
/// f32 (the decode argmax gate is unaffected — see `verify_whisper_decode --npu`).
struct NpuCtx {
    decode: CtxDecode,
    layers: Vec<NpuLayer>,
}

/// Host-f32 Whisper decoder: holds the weights + per-layer self-KV cache + cached encoder cross-KV.
/// When `npu` is `Some`, the per-token decoder matmuls (self QKV / out, cross q / out, fc1 / fc2) run
/// on the NPU via [`CtxDecode`]; everything else (LayerNorm, attention, GELU, residuals, cross K/V
/// precompute, and the final logits projection) stays on the host.
pub struct HostDecoder {
    w: Rc<WhisperDecoderWeights>,
    state: Vec<LayerState>,
    npu: Option<NpuCtx>,
}

/// `x[1,K] @ W[K,N] + b[N]` for a single row vector. Returns a length-N row.
fn linear_row(x: &[f32], w: &Array2<f32>, b: &Array1<f32>) -> Vec<f32> {
    let (k, n) = w.dim();
    debug_assert_eq!(x.len(), k);
    let mut out = b.to_vec();
    // out[j] += sum_i x[i] * W[i,j]
    let ws = w.as_standard_layout();
    let wslice = ws.as_slice().unwrap();
    for i in 0..k {
        let xi = x[i];
        let row = &wslice[i * n..i * n + n];
        for j in 0..n {
            out[j] += xi * row[j];
        }
    }
    out
}

/// `X[M,K] @ W[K,N] + b[N]` for a matrix; returns [M,N].
fn linear_mat(x: &Array2<f32>, w: &Array2<f32>, b: &Array1<f32>) -> Array2<f32> {
    let mut y = x.dot(w);
    y += &b.view().insert_axis(Axis(0));
    y
}

/// Single-query multi-head attention: q is one row [768]; keys/values are [S,768] row-major flat.
/// Returns the context row [768]. Softmax over all S cached positions (causality is enforced by the
/// cache only ever containing positions <= the current one). Matches HF Whisper scaling 1/sqrt(hd).
fn attend_one(q: &[f32], k_flat: &[f32], v_flat: &[f32], s: usize) -> Vec<f32> {
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let mut ctx = vec![0f32; D];
    for h in 0..N_HEADS {
        let base = h * HEAD_DIM;
        // scores[j] = (q_h . k_j_h) * scale
        let mut scores = vec![0f32; s];
        for j in 0..s {
            let krow = &k_flat[j * D + base..j * D + base + HEAD_DIM];
            let mut dot = 0f32;
            for d in 0..HEAD_DIM {
                dot += q[base + d] * krow[d];
            }
            scores[j] = dot * scale;
        }
        // softmax (numerically stable)
        let mut maxv = f32::NEG_INFINITY;
        for &v in &scores {
            maxv = maxv.max(v);
        }
        let mut sum = 0f32;
        for v in scores.iter_mut() {
            *v = (*v - maxv).exp();
            sum += *v;
        }
        let inv = 1.0 / sum;
        // ctx_h = sum_j softmax_j * v_j_h
        for j in 0..s {
            let p = scores[j] * inv;
            let vrow = &v_flat[j * D + base..j * D + base + HEAD_DIM];
            for d in 0..HEAD_DIM {
                ctx[base + d] += p * vrow[d];
            }
        }
    }
    ctx
}

impl HostDecoder {
    pub fn new(w: Rc<WhisperDecoderWeights>) -> Self {
        let state = (0..N_LAYERS).map(|_| LayerState::default()).collect();
        HostDecoder { w, state, npu: None }
    }

    /// Total NPU GEMV dispatches issued so far (0 on the host-only path). Used by the timing harness.
    pub fn npu_dispatches(&self) -> u64 {
        self.npu.as_ref().map(|c| c.decode.dispatches()).unwrap_or(0)
    }

    /// Reset the NPU dispatch counter (no-op on the host-only path). Called before each transcription.
    pub fn reset_npu_dispatches(&self) {
        if let Some(c) = &self.npu {
            c.decode.reset_dispatches();
        }
    }

    /// Build a decoder whose per-token matmuls run on the NPU. `dev` is an open device (single-tenant
    /// — stop npu-asr/voxd first); `root` is the worktree root (where the `mlir-aie` symlink and
    /// the `whole_array/build` xclbins live). Registers every per-token weight ONCE (fused self-QKV,
    /// self out, cross q/out, fc1, fc2) — this loads the needed resident xclbins and panics with a
    /// clear message if a shape's xclbin is missing (build via scripts/build_decode_kernels.sh).
    pub fn new_npu(w: Rc<WhisperDecoderWeights>, dev: &Rc<Device>, root: &Path) -> Self {
        let mut decode = CtxDecode::new(dev, root);
        let layers = w
            .layers
            .iter()
            .map(|lw| {
                // Fused self-QKV: concat q|k|v weights [768,768] each -> [768,2304]; biases likewise.
                let qkv_w = ndarray::concatenate(
                    Axis(1),
                    &[lw.q_w.view(), lw.k_w.view(), lw.v_w.view()],
                )
                .expect("concat self qkv weights");
                let mut qkv_b = Vec::with_capacity(3 * D);
                qkv_b.extend_from_slice(lw.q_b.as_slice().unwrap());
                qkv_b.extend_from_slice(lw.k_b.as_slice().unwrap());
                qkv_b.extend_from_slice(lw.v_b.as_slice().unwrap());

                NpuLayer {
                    qkv: decode.register_weight(&qkv_w, DecodeEpi::Bias, &qkv_b),
                    self_out: decode.register_weight(
                        &lw.out_w,
                        DecodeEpi::Bias,
                        lw.out_b.as_slice().unwrap(),
                    ),
                    cross_q: decode.register_weight(
                        &lw.cross_q_w,
                        DecodeEpi::Bias,
                        lw.cross_q_b.as_slice().unwrap(),
                    ),
                    cross_out: decode.register_weight(
                        &lw.cross_out_w,
                        DecodeEpi::Bias,
                        lw.cross_out_b.as_slice().unwrap(),
                    ),
                    fc1: decode.register_weight(
                        &lw.fc1_w,
                        DecodeEpi::Bias,
                        lw.fc1_b.as_slice().unwrap(),
                    ),
                    fc2: decode.register_weight(
                        &lw.fc2_w,
                        DecodeEpi::Bias,
                        lw.fc2_b.as_slice().unwrap(),
                    ),
                }
            })
            .collect();
        let state = (0..N_LAYERS).map(|_| LayerState::default()).collect();
        HostDecoder { w, state, npu: Some(NpuCtx { decode, layers }) }
    }

    /// Clear the self-KV caches for a new utterance. (Cross-KV must be re-set via `precompute_cross`.)
    pub fn reset(&mut self) {
        for st in &mut self.state {
            st.self_k.clear();
            st.self_v.clear();
            st.n_self = 0;
        }
    }

    /// Precompute and cache the encoder cross-attention K/V per layer:
    ///   K_enc = enc_hidden · cross_k  (no bias — cross_k.bias is zeros)
    ///   V_enc = enc_hidden · cross_v  (+ cross_v.bias)
    /// `enc_hidden` is [T_enc, 768]. Also clears the self-KV caches (start of a new utterance).
    pub fn precompute_cross(&mut self, enc_hidden: &Array2<f32>) {
        self.reset();
        let w = Rc::clone(&self.w);
        for (li, lw) in w.layers.iter().enumerate() {
            let kenc = linear_mat(enc_hidden, &lw.cross_k_w, &lw.cross_k_b);
            let venc = linear_mat(enc_hidden, &lw.cross_v_w, &lw.cross_v_b);
            self.state[li].cross_k = kenc;
            self.state[li].cross_v = venc;
        }
    }

    /// One decode step: token `token` at position `pos`. Updates the self-KV cache and returns the
    /// vocab logits `[51865]`.
    pub fn step(&mut self, token: i64, pos: usize) -> Vec<f32> {
        // input embedding: embed_tokens[token] + embed_positions[pos]
        let tok = token as usize;
        // MAX_DECODE (200, in whisper.rs) must stay < embed_positions rows (448).
        debug_assert!(
            pos < self.w.embed_positions.nrows(),
            "decode pos {} exceeds embed_positions rows",
            pos
        );
        let mut x: Vec<f32> = (0..D)
            .map(|d| self.w.embed_tokens[[tok, d]] + self.w.embed_positions[[pos, d]])
            .collect();

        // Hold an owned Rc handle to the weights and a disjoint borrow of the NPU backend for the
        // whole step, so mutating `self.state` below doesn't conflict with the borrow checker.
        let w = Rc::clone(&self.w);
        let w = w.as_ref();
        let npu = self.npu.as_ref();

        for li in 0..N_LAYERS {
            let lw = &w.layers[li];
            let npu_layer = npu.map(|n| &n.layers[li]);

            // --- 1. self-attention (pre-norm, causal) ---
            let ln = ln_row(&x, &lw.ln_self_w, &lw.ln_self_b);
            // Self q/k/v: NPU runs ONE fused [768,2304] gemv (bias-fused), host slices into q/k/v;
            // the host fallback does three [768,768] matmuls.
            let (q, k, v) = match (npu, npu_layer) {
                (Some(ctx), Some(nl)) => {
                    let qkv = ctx.decode.gemv(&nl.qkv, &ln).expect("npu self-qkv gemv");
                    (qkv[0..D].to_vec(), qkv[D..2 * D].to_vec(), qkv[2 * D..3 * D].to_vec())
                }
                _ => (
                    linear_row(&ln, &lw.q_w, &lw.q_b),
                    linear_row(&ln, &lw.k_w, &lw.k_b),
                    linear_row(&ln, &lw.v_w, &lw.v_b),
                ),
            };
            {
                let st = &mut self.state[li];
                st.self_k.extend_from_slice(&k);
                st.self_v.extend_from_slice(&v);
                st.n_self += 1;
            }
            let st = &self.state[li];
            let ctx_vec = attend_one(&q, &st.self_k, &st.self_v, st.n_self);
            let attn = match (npu, npu_layer) {
                (Some(ctx), Some(nl)) => ctx.decode.gemv(&nl.self_out, &ctx_vec).expect("npu self-out gemv"),
                _ => linear_row(&ctx_vec, &lw.out_w, &lw.out_b),
            };
            for d in 0..D {
                x[d] += attn[d];
            }

            // --- 2. cross-attention (pre-norm) ---
            let ln = ln_row(&x, &lw.ln_cross_w, &lw.ln_cross_b);
            let q = match (npu, npu_layer) {
                (Some(ctx), Some(nl)) => ctx.decode.gemv(&nl.cross_q, &ln).expect("npu cross-q gemv"),
                _ => linear_row(&ln, &lw.cross_q_w, &lw.cross_q_b),
            };
            let st = &self.state[li];
            let t_enc = st.cross_k.nrows();
            let ck = st.cross_k.as_standard_layout();
            let cv = st.cross_v.as_standard_layout();
            let ctx_vec = attend_one(&q, ck.as_slice().unwrap(), cv.as_slice().unwrap(), t_enc);
            let attn = match (npu, npu_layer) {
                (Some(ctx), Some(nl)) => ctx.decode.gemv(&nl.cross_out, &ctx_vec).expect("npu cross-out gemv"),
                _ => linear_row(&ctx_vec, &lw.cross_out_w, &lw.cross_out_b),
            };
            for d in 0..D {
                x[d] += attn[d];
            }

            // --- 3. feed-forward (pre-norm) ---
            let ln = ln_row(&x, &lw.ln_final_w, &lw.ln_final_b);
            let h1 = match (npu, npu_layer) {
                (Some(ctx), Some(nl)) => ctx.decode.gemv(&nl.fc1, &ln).expect("npu fc1 gemv"),
                _ => linear_row(&ln, &lw.fc1_w, &lw.fc1_b),
            }; // [3072]
            debug_assert_eq!(h1.len(), FFN);
            let h1 = gelu_row(&h1); // GELU stays on host
            let h2 = match (npu, npu_layer) {
                (Some(ctx), Some(nl)) => ctx.decode.gemv(&nl.fc2, &h1).expect("npu fc2 gemv"),
                _ => linear_row(&h1, &lw.fc2_w, &lw.fc2_b),
            }; // [768]
            for d in 0..D {
                x[d] += h2[d];
            }
        }

        // final LN + proj_out -> logits [vocab]. proj_out STAYS on host f32: its [768, 51865] weight
        // pads to N=51872 which violates the whole_array `N % (n*cols)=256 == 0` tiling constraint
        // (51872 % 256 = 160), so the logits kernel cannot be built. Logits on host f32 is also the
        // safest choice for argmax fidelity (no bf16 rounding on the near-ties that pick the token).
        let ln = ln_row(&x, &w.ln_post_w, &w.ln_post_b);
        let mut logits = vec![0f32; VOCAB];
        let ws = w.proj_out_w.as_standard_layout(); // [768, vocab]
        let wslice = ws.as_slice().unwrap();
        for i in 0..D {
            let xi = ln[i];
            let row = &wslice[i * VOCAB..i * VOCAB + VOCAB];
            for j in 0..VOCAB {
                logits[j] += xi * row[j];
            }
        }
        logits
    }
}

/// LayerNorm of a single row (last-axis affine, population variance, eps 1e-5) — same math as
/// `npu_asr_host::layer_norm` on a 1xD matrix.
fn ln_row(x: &[f32], gamma: &Array1<f32>, beta: &Array1<f32>) -> Vec<f32> {
    let d = x.len();
    let mean: f32 = x.iter().sum::<f32>() / d as f32;
    let var: f32 = x.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
    let inv = 1.0 / (var + LN_EPS).sqrt();
    (0..d).map(|j| (x[j] - mean) * inv * gamma[j] + beta[j]).collect()
}

/// Exact GELU of a single row (reuses `npu_asr_host::gelu` for bit-identical math vs. the encoder).
fn gelu_row(x: &[f32]) -> Vec<f32> {
    let a = Array2::from_shape_vec((1, x.len()), x.to_vec()).unwrap();
    gelu(&a).into_raw_vec_and_offset().0
}
