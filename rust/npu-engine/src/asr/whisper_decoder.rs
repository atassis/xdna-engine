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
use npu_asr::ctx2::{CtxAOp, Epi, SharedCtxA};
use npu_asr::ctx_decode::{CtxDecode, DecodeEpi, DecodeWeight, FusedWeight, Norm};
use npu_asr::engines::PAD_M;
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
    /// Fused LN_self + self-QKV: `W'' = diag(γ)·[q|k|v]`, `bias' = β@W + concat(q/k/v bias)` —
    /// ONE dispatch does LN_self→QKV when `NPU_DECODE_ATTN` is set (replaces host-LN + qkv gemv).
    qkv_fused: FusedWeight,  // [768, 2304], LN folded
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
    /// When true (env `NPU_DECODE_ATTN` set, NPU active), the SELF-attention sublayer runs on-NPU:
    /// fused LN_self+QKV (`fused_norm_gemv`) and on-chip attention (`CtxDecode::attn`) replace the
    /// step-1 host-LN + qkv gemv + host `attend_one`. Cross-attn + FFN stay exactly as step-1 (M3).
    npu_attn: bool,
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
        HostDecoder { w, state, npu: None, npu_attn: false }
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

                // Fused LN_self + QKV: fold ln_self (γ,β) into the concat [768,2304] weight + bias.
                let qkv_fused = decode.register_fused(
                    &qkv_w,
                    Norm::Ln {
                        gamma: lw.ln_self_w.as_slice().unwrap().to_vec(),
                        beta: lw.ln_self_b.as_slice().unwrap().to_vec(),
                        eps: LN_EPS,
                    },
                    &qkv_b,
                );

                NpuLayer {
                    qkv: decode.register_weight(&qkv_w, DecodeEpi::Bias, &qkv_b),
                    qkv_fused,
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
        // On-NPU self-attention is opt-in via NPU_DECODE_ATTN (only meaningful with the NPU active).
        let npu_attn = std::env::var("NPU_DECODE_ATTN").is_ok();
        if npu_attn {
            // Preload the MHA xclbin so a missing artifact fails loudly at construction, not mid-decode.
            decode
                .ensure_attn_loaded()
                .unwrap_or_else(|e| panic!("NPU_DECODE_ATTN set but MHA kernel unavailable: {e}"));
            eprintln!("[whisper_decoder] NPU_DECODE_ATTN: on-chip SELF-attention enabled (cross+FFN stay host/step-1)");
        }
        HostDecoder { w, state, npu: Some(NpuCtx { decode, layers }), npu_attn }
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
            // Self q/k/v projection. THREE paths:
            //  * NPU + NPU_DECODE_ATTN: the COLLAPSED self-attn sublayer (`self_attn_chained`) runs
            //    LN+QKV → on-chip MHA → O as one host call, threading q/ctx buffer-to-buffer (no
            //    caller-visible intermediate Vec hops). Byte-identical to the M1.a 3-call sequence.
            //  * NPU only (step-1): host LN_self, then ONE fused [768,2304] qkv gemv (bias-fused).
            //  * host: LN_self, then three [768,768] matmuls.
            let attn = if let (Some(ctx), Some(nl)) = (npu, npu_layer) {
                if self.npu_attn {
                    let st = &mut self.state[li];
                    let n_before = st.n_self;
                    let out = ctx
                        .decode
                        .self_attn_chained(
                            &nl.qkv_fused,
                            &nl.self_out,
                            &x,
                            &mut st.self_k,
                            &mut st.self_v,
                            n_before,
                        )
                        .expect("npu collapsed self-attn (QKV→MHA→O)");
                    st.n_self += 1;
                    out
                } else {
                    // NPU step-1: host LN_self + fused qkv gemv, host attend_one, npu self-out gemv.
                    let ln = ln_row(&x, &lw.ln_self_w, &lw.ln_self_b);
                    let qkv = ctx.decode.gemv(&nl.qkv, &ln).expect("npu self-qkv gemv");
                    let (q, k, v) = (
                        &qkv[0..D],
                        &qkv[D..2 * D],
                        &qkv[2 * D..3 * D],
                    );
                    let st = &mut self.state[li];
                    st.self_k.extend_from_slice(k);
                    st.self_v.extend_from_slice(v);
                    st.n_self += 1;
                    let st = &self.state[li];
                    let ctx_vec = attend_one(q, &st.self_k, &st.self_v, st.n_self);
                    ctx.decode.gemv(&nl.self_out, &ctx_vec).expect("npu self-out gemv")
                }
            } else {
                // Pure host path.
                let ln = ln_row(&x, &lw.ln_self_w, &lw.ln_self_b);
                let q = linear_row(&ln, &lw.q_w, &lw.q_b);
                let k = linear_row(&ln, &lw.k_w, &lw.k_b);
                let v = linear_row(&ln, &lw.v_w, &lw.v_b);
                let st = &mut self.state[li];
                st.self_k.extend_from_slice(&k);
                st.self_v.extend_from_slice(&v);
                st.n_self += 1;
                let st = &self.state[li];
                let ctx_vec = attend_one(&q, &st.self_k, &st.self_v, st.n_self);
                linear_row(&ctx_vec, &lw.out_w, &lw.out_b)
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

// =============================================================================================
// Fused whole-decode backend (NPU_DECODE_FUSED): the ENTIRE 12-layer decoder runs as ONE fused-ELF
// dispatch per token (vs the per-op `CtxDecode` path's ~72 dispatches). Loads the prebuilt fused ELF
// + resident weight arena from `artifacts/fused_decode12/` (gen_decode.py). Per utterance: compute
// encoder cross-K/V into the resident scratch arena. Per token: embed→write x→patch KV/mask→reload
// ELF→dispatch→read x12→host ln_post+proj_out logits (the lm-head stays host, like every other path).
// Numerically validated: verify_fused_decode.py = 21/21 argmax vs f32 reference on the real encoder.
// =============================================================================================
use npu_xrt::{
    pack_f32_to_bf16, unpack_bf16_to_f32, Arena, ElfCtx, ElfKernel, ElfKernel2, ElfResident,
    FusedArena, FusedElfPatcher, Run,
};

/// Deep-C resident-scratchpad state: the decode ELF is CONSTANT (registered ONCE via
/// [`ElfResident`]); per token the host writes two scratchpad words — `kv_off` (addr-kind, raw,
/// element-units = `n_self*head_dim`) and `sm_mask` (core-kind, written `<<2` per the firmware
/// UPDATE_REG requirement, value = `n_self+1`) — then dispatches. Replaces the per-token
/// `patch_elf` + `load_elf` re-registration entirely (energy + latency win).
struct ResidentState {
    res: ElfResident,
    kv_off_byte: usize,
    sm_off_byte: usize,
    sm_core: bool,
    head_dim: u32,
    /// M0.5 (--coalesce-self-tr): byte offset of the 2nd addr scratchpad `vcache_off` (= n_self, the
    /// transposed-vcache write column). `None` = deep-C layout ([H,S,HD], no transposed write).
    vcache_off_byte: Option<usize>,
}

/// Per-token dispatch handle: either a freshly-registered ELF (own hw_context, the original
/// ~20 ms/token path) or one rebound onto a persistent [`ElfCtx`] (partition config hoisted out).
enum FusedKern {
    Fresh(ElfKernel),
    Reuse(ElfKernel2),
}

const T_PAD: usize = 1536; // encoder positions padded to a %64,%16 multiple (matches gen_decode.py)

/// (arena, byte-offset, byte-len) of a named buffer in the fused arenas (from meta.json layout).
#[derive(Clone, Copy)]
struct BufLoc {
    arena: Arena,
    off: usize,
    len: usize,
}

/// e2e/NPU wide-dispatch lm-head: a standalone `proj_out` GEMV ELF run as ONE dispatch/token, replacing
/// both the host f32 ~40M-MAC matmul and the latency-negative 17-chunk ctx2 path. The ELF is CONSTANT (no
/// per-token patch / scratchpad), so it registers once and dispatches with the bound arena each token.
/// The GEMV computes `logits[VOCAB_PAD] = (γ⊙proj_out_w).Tᵀ · norm` (vocab-as-M → contiguous vector output,
/// no whole_array DMA-stride wall); the LN affine-normalize and the `β·W` bias stay on host (cheap). Built
/// by `route_b_kernels/decode_fused/gen_projout.py` / `scripts/build_projout_elf.sh`.
struct ProjOutElf {
    arena: FusedArena,
    kern: ElfKernel,
    x_loc: BufLoc,
    logits_loc: BufLoc,
    vocab: usize,
    // step-2 (on-NPU argmax): when the ELF has the fused per-column partial argmax, `amax_loc` is the 64-B
    // output (cols × [val:f32 | idx:i32]); the host does the trivial cols-way reduce → token id. While
    // validating, we compute BOTH the NPU id and the host argmax(logits) and count mismatches.
    amax_loc: Option<BufLoc>,
    cols: usize,
    vocab_pad: usize,
    amax_mismatch: std::cell::Cell<usize>,
    amax_checked: std::cell::Cell<usize>,
}

impl ProjOutElf {
    fn load(dev: &Rc<Device>, dir: &Path, _w: &WhisperDecoderWeights) -> Self {
        let elf = std::fs::read(dir.join("projout.elf"))
            .unwrap_or_else(|e| panic!("read projout.elf: {e} (run scripts/build_projout_elf.sh)"));
        let meta: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("meta.json")).expect("read projout meta.json"))
                .expect("parse projout meta.json");
        let usz = |k: &str| meta[k].as_u64().expect(k) as usize;
        let (in_sz, out_sz, scr_sz) = (usz("input_size"), usz("output_size"), usz("scratch_size"));
        let vocab = usz("vocab");
        let mut layout = HashMap::new();
        for (name, e) in meta["layout"].as_object().expect("projout layout") {
            let arena = match e["type"].as_str().unwrap() {
                "input" => Arena::Input,
                "output" => Arena::Output,
                "scratch" => Arena::Scratch,
                o => panic!("bad arena type {o}"),
            };
            layout.insert(
                name.clone(),
                BufLoc { arena, off: e["offset"].as_u64().unwrap() as usize, len: e["len"].as_u64().unwrap() as usize },
            );
        }
        let arena = FusedArena::new(dev, in_sz, out_sz, scr_sz).expect("alloc projout arena");
        for name in meta["weights"].as_array().expect("projout weights") {
            let name = name.as_str().unwrap();
            let bytes = std::fs::read(dir.join("buffers").join(format!("{name}.bin")))
                .unwrap_or_else(|e| panic!("read projout buffer {name}.bin: {e}"));
            let loc = &layout[name];
            assert_eq!(bytes.len(), loc.len, "{name}: blob {} != layout {}", bytes.len(), loc.len);
            arena.write_at(loc.arena, loc.off, &bytes).unwrap();
        }
        // K-aug bias: write the input tail [1, 0…0] (vs elems) at element offset D ONCE. The β·W bias is
        // folded into the GEMV weight's column D, so GEMV(mat, [nrm, 1, 0…]) = norm·(γ⊙W) + β·W = complete
        // logits on-device. Per token we only overwrite [0:D], so this tail persists.
        let vs = usz("vs");
        let d = usz("D");
        let x_loc = layout["x"];
        let mut tail = vec![0f32; vs];
        tail[0] = 1.0;
        arena.write_at(x_loc.arena, x_loc.off + d * 2, &pack_bf16_bytes(&tail)).unwrap();
        arena.sync_input().unwrap();
        let kern = dev.load_elf_kernel(&elf, Some("main:sequence")).expect("register projout ELF");
        let has_argmax = meta.get("argmax").and_then(|v| v.as_bool()).unwrap_or(false);
        let cols = meta.get("cols").and_then(|v| v.as_u64()).unwrap_or(8) as usize;
        let vocab_pad = usz("vocab_pad");
        let amax_loc = if has_argmax { Some(layout["amax"]) } else { None };
        eprintln!("[whisper_decoder] NPU_DECODE_PROJOUT_ELF: proj_out GEMV ELF, 1 dispatch/token (vocab_pad={}, K-aug bias{}, weight {:.0} MB)",
            vocab_pad, if has_argmax { ", +on-NPU argmax (validating)" } else { "" }, scr_sz as f64 / 1e6);
        ProjOutElf {
            arena, kern, x_loc, logits_loc: layout["logits"], vocab, amax_loc, cols, vocab_pad,
            amax_mismatch: std::cell::Cell::new(0), amax_checked: std::cell::Cell::new(0),
        }
    }

    /// Host cols-way reduce of the NPU per-column partials: `amax` = cols × [val:f32 | idx:i32] (8 B each).
    /// Column c covers logits[c·slice : (c+1)·slice]; global token = c·slice + local_idx; pick the column
    /// with the largest value (strict `>`, first wins — matches host argmax). Returns the token id.
    fn argmax_from_amax(&self, amax: &[u8]) -> i64 {
        let slice = self.vocab_pad / self.cols;
        let (mut best_val, mut best_tok) = (f32::NEG_INFINITY, 0usize);
        for c in 0..self.cols {
            let o = c * 8;
            let val = f32::from_le_bytes([amax[o], amax[o + 1], amax[o + 2], amax[o + 3]]);
            let idx = i32::from_le_bytes([amax[o + 4], amax[o + 5], amax[o + 6], amax[o + 7]]) as usize;
            if val > best_val {
                best_val = val;
                best_tok = c * slice + idx;
            }
        }
        best_tok as i64
    }

    /// `nrm` = affine-free normalized hidden [D]. One NPU dispatch → logits[0:vocab] (β·W bias is folded
    /// into the GEMV via K-aug — the device emits complete logits, no host bias-add).
    fn logits(&self, nrm: &[f32]) -> Vec<f32> {
        let xb = pack_bf16_bytes(nrm);
        self.arena.write_at(self.x_loc.arena, self.x_loc.off, &xb).unwrap();
        self.arena.sync_input().unwrap();
        self.arena.dispatch(&self.kern).expect("projout dispatch");
        self.arena.sync_from_device().unwrap();
        let mut ob = vec![0u8; self.logits_loc.len];
        self.arena.read_at(self.logits_loc.arena, self.logits_loc.off, &mut ob).unwrap();
        let mut logits = unpack_bf16_bytes(&ob); // [vocab_pad]
        logits.truncate(self.vocab);
        // step-2 validation: compare the on-NPU argmax (host cols-way reduce of `amax`) vs host argmax(logits).
        if let Some(al) = self.amax_loc {
            let mut ab = vec![0u8; al.len];
            self.arena.read_at(al.arena, al.off, &mut ab).unwrap();
            let id_npu = self.argmax_from_amax(&ab);
            let mut id_host = 0usize;
            for i in 1..self.vocab {
                if logits[i] > logits[id_host] {
                    id_host = i;
                }
            }
            self.amax_checked.set(self.amax_checked.get() + 1);
            if id_npu != id_host as i64 {
                let n = self.amax_mismatch.get() + 1;
                self.amax_mismatch.set(n);
                if n <= 8 {
                    eprintln!("[projout argmax] MISMATCH #{n}: npu={id_npu} host={id_host} (logit npu={:.4} host={:.4})",
                        logits.get(id_npu as usize).copied().unwrap_or(f32::NAN), logits[id_host]);
                }
            }
        }
        logits
    }

    /// e2e/NPU steady-state: dispatch + read ONLY the 64-B `amax` partials → token id (host cols-way
    /// reduce). Drops the 104 KB logits readback + host argmax. Requires the argmax-fused ELF.
    fn token_id(&self, nrm: &[f32]) -> i64 {
        let al = self.amax_loc.expect("token_id needs the argmax-fused proj_out ELF");
        let xb = pack_bf16_bytes(nrm);
        self.arena.write_at(self.x_loc.arena, self.x_loc.off, &xb).unwrap();
        self.arena.sync_input().unwrap();
        self.arena.dispatch(&self.kern).expect("projout dispatch");
        self.arena.sync_from_device().unwrap();
        let mut ab = vec![0u8; al.len];
        self.arena.read_at(al.arena, al.off, &mut ab).unwrap();
        self.argmax_from_amax(&ab)
    }

    fn has_argmax(&self) -> bool {
        self.amax_loc.is_some()
    }

    fn dump_amax_stats(&self) {
        if self.amax_loc.is_some() {
            eprintln!("[projout argmax] validation: {} mismatches / {} tokens (host cols-way vs host argmax)",
                self.amax_mismatch.get(), self.amax_checked.get());
        }
    }
}

/// P0 per-phase timing accumulator (env `FUSED_PHASE_TIMING`). All values are nanoseconds, summed
/// across every `step` of one utterance (reset in `precompute_cross`). `cross_fold` is the single
/// per-utterance encoder cross-K/V compute+write; the rest are per-token (one dispatch each).
#[derive(Default)]
struct PhaseAcc {
    embed: u128,       // token+pos embedding lookup → host x[768]
    write_x: u128,     // pack x to bf16 + write into the input arena
    patch: u128,       // FusedElfPatcher::patch (27 MB to_vec + KV-offset/mask scan)
    load_elf: u128,    // load_elf_kernel — per-token ELF re-registration with XRT
    prefetch: u128,    // host patch+load_elf of the NEXT position, OVERLAPPED under `dispatch` (the
                       //   default PIPE path; informational, NOT in step_sum — it hides under the run)
    sync_in: u128,     // arena.sync_input (host→device DMA of the input arena)
    dispatch: u128,    // arena.dispatch — the actual 12-layer NPU run (incl. the overlapped prefetch)
    sync_out: u128,    // arena.sync_from_device (device→host DMA of the output arena)
    read_unpack: u128, // read_at + unpack bf16 output → x12[768]
    lm_head: u128,     // host ln_post + proj_out (768×51865 naive loop) → logits
    steps: u64,        // number of dispatches (== tokens incl. lang-detect + prompt)
    cross_fold: u128,  // per-utterance encoder cross-K/V fold (host matmuls + arena writes + sync)
    utterances: u64,
}

/// A running lap timer: `lap(&mut acc)` adds the time since the previous lap (or `start`) to `acc`
/// and re-marks. A no-op when constructed `off` (timing disabled) — zero Instant calls on the hot
/// path, so the breakdown can ship in-tree without perturbing the un-instrumented latency.
struct Lap {
    mark: Option<std::time::Instant>,
}
impl Lap {
    fn start(on: bool) -> Self {
        Lap { mark: on.then(std::time::Instant::now) }
    }
    fn lap(&mut self, acc: &mut u128) {
        if let Some(m) = self.mark {
            let now = std::time::Instant::now();
            *acc += now.duration_since(m).as_nanos();
            self.mark = Some(now);
        }
    }
}

/// Whole-decode fused-ELF backend. Mirrors `HostDecoder`'s decode contract (`precompute_cross`,
/// per-token logits) but collapses all 12 layers into one dispatch.
pub struct FusedDecoder {
    dev: Rc<Device>,
    w: Rc<WhisperDecoderWeights>,
    arena: FusedArena,
    base_elf: Vec<u8>,
    patcher: FusedElfPatcher,
    layout: HashMap<String, BufLoc>,
    output: String, // e.g. "x12"
    t_enc: usize,
    /// lever #3 (i): meta `coalesce_cross` — when true the ELF reads the encoder Venc PRE-TRANSPOSED
    /// [H,HD,TP] (dropping the per-token op_tr_c cross-V transpose, the #1 inter-op round-trip), so
    /// `precompute_cross` must write each L*_Venc in that layout. Default false = deep-C [H,TP,HD].
    coalesce_cross: bool,
    /// int8 cross-K (meta `int8_cross_k`): quantize each per-utterance Kenc to int8 with the fixed
    /// per-layer `cross_k_scales` (s_k folded into mat_cq at build, so it cancels) + write int8 bytes into
    /// the bf16-typed Kenc buffer (halves its LPDDR re-read). Default false.
    int8_cross_k: bool,
    /// int8 cross-V (meta `int8_cross_v`, implies `coalesce_cross`): quantize each per-utterance Venc
    /// (pre-transposed [H,HD,TP]) to int8 with a per-(head,HD) scale `s_cv` the host computes from the real
    /// Venc + writes for op_mul_cv to apply to the context output. Halves the Venc LPDDR re-read. Default false.
    int8_cross_v: bool,
    /// e2e/NPU: the ELF outputs logits[VOCAB_PAD] (ln_post+proj_out on-NPU) -> step() returns them
    /// directly (drops the host proj_out matmul). Default false = ELF outputs the 768-hidden.
    npu_logits: bool,
    n_self: usize, // self-KV positions already written this utterance (== KV write slot for next token)
    timing: bool,  // env FUSED_PHASE_TIMING: accumulate + dump the per-phase breakdown
    ph: PhaseAcc,
    /// env `NPU_DECODE_FUSED_REUSECTX`: P1.1 spike — persistent hw_context built once from the base
    /// ELF, per-token patched ELFs rebound onto it. **BLOCKED on this XRT**: `ext::kernel(ctx, module,
    /// name)` rejects an ELF-built hw_context ("not created using XCLBIN") — an ELF context binds only
    /// its own module. The plumbing is kept (works for an xclbin-backed ctx); the flag panics if set.
    /// See `log/2026-06/fused-decode-reusectx-wall.md`. The viable levers are runtime-offset kernel
    /// regen or async-prefetch of the position-only registration.
    reuse_ctx: Option<ElfCtx>,
    /// Lever #2 (energy + latency): per-layer (cross_k, cross_v) GEMM ops on the NPU's resident ctx2
    /// kernel. When present, `precompute_cross` computes the encoder cross-K/V fold (12×2 [1500,768]@
    /// [768,768] GEMMs, ~370 ms host) on the NPU instead of the CPU. `None` = host f32 fold (fallback).
    cross_ops: Option<Vec<(CtxAOp, CtxAOp)>>,
    /// e2e/NPU step-1 (env `NPU_DECODE_PROJOUT_CTX2=1`): run the per-token `ln_post`+`proj_out` logits
    /// projection on the NPU's resident ctx2 kernel instead of the host f32 ~40M-MAC matmul. The vocab
    /// (51865) is computed in `ceil(VOCAB/NA)=17` chunks of N=`NA`=3072 (the resident kernel's max served
    /// stream width — a single 52224-wide stream would need a new 17×-larger xclbin), each a `CtxAOp` whose
    /// weight folds the LN affine: `W'_chunk = γ[:,None]·proj_out_w[:,chunk]`, `bias'_chunk = β·proj_out_w[:,chunk]`.
    /// `step()` normalizes the hidden WITHOUT affine, runs the 17 chunk GEMVs, concatenates → logits[0:VOCAB].
    /// Host argmax stays (step-2 moves it on-NPU). bf16 ctx2 (default precision) → gate WER. `None` = host f32.
    proj_out_ops: Option<Vec<CtxAOp>>,
    /// e2e/NPU wide-dispatch lm-head (env `NPU_DECODE_PROJOUT_ELF=1`): the standalone proj_out GEMV ELF —
    /// ONE dispatch/token (vs the 17 ctx2 chunks), the latency-positive path. Takes precedence over
    /// `proj_out_ops` when both are set. `None` = use `proj_out_ops` or the host f32 matmul.
    proj_out_elf: Option<ProjOutElf>,
    /// PIPE 1-deep lookahead slot (the DEFAULT fused decode path): `(n_self_it_was_built_for, kernel)`.
    /// The patched ELF for the next token depends ONLY on `n_self` (KV-write offset + softmax mask), not
    /// the argmax'd token, so it is registered on the host during the PREVIOUS token's dispatch (see
    /// `dispatch_pipe`), hiding the ~14 ms `load_elf` off the critical path. Consumed by the next `step`
    /// iff its `n_self` matches; invalidated (`None`) by `reset`/`precompute_cross` where `n_self`
    /// rewinds and a stale prefetch would mispatch. Unused on the `reuse_ctx` (REUSECTX) diagnostic path.
    next_kern: Option<(usize, ElfKernel)>,
    /// Deep-C (DEFAULT when the decode ELF carries scratchpad params): register the constant ELF once
    /// and drive per-token KV-offset + softmax-mask via the ctrl scratchpad. When `Some`, the
    /// patch/PIPE/REUSECTX paths are bypassed. Force the legacy patch path with NPU_DECODE_FUSED_PATCH.
    resident: Option<ResidentState>,
}

fn pack_bf16_bytes(f: &[f32]) -> Vec<u8> {
    let mut bits = vec![0u16; f.len()];
    pack_f32_to_bf16(f, &mut bits);
    let mut out = vec![0u8; bits.len() * 2];
    for (i, &b) in bits.iter().enumerate() {
        out[2 * i..2 * i + 2].copy_from_slice(&b.to_le_bytes());
    }
    out
}

fn unpack_bf16_bytes(bytes: &[u8]) -> Vec<f32> {
    let u16s: Vec<u16> = bytes.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
    let mut out = vec![0f32; u16s.len()];
    unpack_bf16_to_f32(&u16s, &mut out);
    out
}

impl FusedDecoder {
    /// Load the prebuilt fused decode ELF + resident weight arena. `fused_dir` =
    /// `artifacts/fused_decode12` (decode.elf, meta.json, buffers/<name>.bin). Static weights are
    /// written into the scratch arena once here; encoder cross-K/V + self-KV caches are populated
    /// per utterance in `precompute_cross`.
    pub fn new(
        w: Rc<WhisperDecoderWeights>,
        dev: &Rc<Device>,
        fused_dir: &Path,
        shared: Option<Rc<SharedCtxA>>,
    ) -> Self {
        let base_elf = std::fs::read(fused_dir.join("decode.elf"))
            .unwrap_or_else(|e| panic!("read decode.elf: {e} (run gen_decode.py --layers 12)"));
        let meta: serde_json::Value =
            serde_json::from_slice(&std::fs::read(fused_dir.join("meta.json")).expect("read meta.json"))
                .expect("parse meta.json");
        let usz = |k: &str| meta[k].as_u64().expect(k) as usize;
        let (in_sz, out_sz, scr_sz) = (usz("input_size"), usz("output_size"), usz("scratch_size"));
        let output = meta["output"].as_str().expect("output").to_string();

        let mut layout = HashMap::new();
        for (name, e) in meta["layout"].as_object().expect("layout") {
            let arena = match e["type"].as_str().unwrap() {
                "input" => Arena::Input,
                "output" => Arena::Output,
                "scratch" => Arena::Scratch,
                o => panic!("bad arena type {o}"),
            };
            layout.insert(
                name.clone(),
                BufLoc { arena, off: e["offset"].as_u64().unwrap() as usize, len: e["len"].as_u64().unwrap() as usize },
            );
        }

        let arena = FusedArena::new(dev, in_sz, out_sz, scr_sz).expect("alloc fused arenas");

        // Write static weight buffers (everything except the per-utterance encoder-K/V and self-KV
        // caches, which we populate in precompute_cross).
        for name in meta["weights"].as_array().expect("weights") {
            let name = name.as_str().unwrap();
            // NOTE: `_s_cq` (leading underscore) is REQUIRED — the host-computed int8 scale buffers are
            // named `L{li}_s_cq`. A bare `s_cq` suffix would also (wrongly) match the real `L{li}_bias_cq`
            // cross-attention query-bias weights ("bias_cq" ends in "s_cq"), skipping them and breaking
            // the deep-C baseline (1-token "You" garbage). Keep the underscore.
            if name.ends_with("Kenc") || name.ends_with("Venc") || name.ends_with("kcache") || name.ends_with("vcache") || name.ends_with("_s_cq") || name.ends_with("_s_cv") {
                continue;
            }
            let bytes = std::fs::read(fused_dir.join("buffers").join(format!("{name}.bin")))
                .unwrap_or_else(|e| panic!("read buffer {name}.bin: {e}"));
            let loc = &layout[name];
            assert_eq!(bytes.len(), loc.len, "{name}: blob {} != layout {}", bytes.len(), loc.len);
            arena.write_at(loc.arena, loc.off, &bytes).unwrap();
        }

        // BIAS FUSION (K-aug): write the augmentation tail [1, 0..0] (VS elems) ONCE at element offset k of
        // each L*_<suffix> GEMV-input buffer. The producing op writes only [0:k], so this constant persists
        // for every token → GEMV(W_aug, [x,1,0..]) = W·x + bias, with the separate bias-add op eliminated.
        if let Some(aug) = meta.get("fuse_bias_aug").and_then(|v| v.as_object()) {
            let vs = meta.get("fuse_bias_vs").and_then(|v| v.as_u64()).unwrap_or(64) as usize;
            let nl = meta["dims"]["layers"].as_u64().expect("dims.layers") as usize;
            let mut tail = vec![0f32; vs];
            tail[0] = 1.0;
            let tail_bytes = pack_bf16_bytes(&tail);
            for (suffix, kval) in aug {
                let k = kval.as_u64().expect("fuse_bias_aug k") as usize;
                for li in 0..nl {
                    let loc = &layout[&format!("L{li}_{suffix}")];
                    arena.write_at(loc.arena, loc.off + k * 2, &tail_bytes).unwrap();
                }
            }
            if !aug.is_empty() {
                eprintln!("[whisper_decoder] fuse_bias: K-aug tails written ({} input(s) × {} layers)", aug.len(), nl);
            }
        }

        // KV cache offsets (bytes) for the patcher: every per-layer kcache/vcache.
        let mut kv_offsets: Vec<u32> = Vec::new();
        for (name, loc) in &layout {
            if name.ends_with("kcache") || name.ends_with("vcache") {
                kv_offsets.push(loc.off as u32);
            }
        }
        let head_dim = meta["patch"]["head_dim"].as_u64().unwrap_or(HEAD_DIM as u64) as u32;
        let patcher = FusedElfPatcher::build(&base_elf, &kv_offsets, head_dim);
        let t_enc = meta["dims"]["T_enc"].as_u64().expect("dims.T_enc") as usize;
        let coalesce_cross = meta.get("coalesce_cross").and_then(|v| v.as_bool()).unwrap_or(false);
        let int8_cross_k = meta.get("int8_cross_k").and_then(|v| v.as_bool()).unwrap_or(false);
        if int8_cross_k {
            eprintln!("[whisper_decoder] int8_cross_k: per-utterance per-channel Kenc int8 (s_cq -> op_mul_cq on qc)");
        }
        let int8_cross_v = meta.get("int8_cross_v").and_then(|v| v.as_bool()).unwrap_or(false);
        let npu_logits = meta.get("npu_logits").and_then(|v| v.as_bool()).unwrap_or(false);
        if npu_logits {
            eprintln!("[whisper_decoder] npu_logits: ln_post+proj_out on the NPU (ELF outputs logits)");
        }
        if int8_cross_v {
            assert!(coalesce_cross, "int8_cross_v requires coalesce_cross (Venc must be resident pre-transposed)");
            eprintln!("[whisper_decoder] int8_cross_v: per-utterance per-channel Venc int8 (s_cv -> op_mul_cv on ctc)");
        }
        if coalesce_cross {
            eprintln!("[whisper_decoder] coalesce_cross: writing encoder Venc pre-transposed [H,HD,TP] (cross-V transpose eliminated)");
        }

        // Deep-C: if the ELF carries scratchpad params (gen_decode emits the `scratchpad` meta block),
        // register it ONCE as a resident kernel and bind the arena BOs once. Per token we then only
        // write 2 scratchpad words + dispatch — no per-token ELF patch/reload. Opt out (legacy patch
        // path) with NPU_DECODE_FUSED_PATCH=1 for A/B comparison.
        let resident = if meta.get("scratchpad").is_some()
            && std::env::var("NPU_DECODE_FUSED_PATCH").is_err()
        {
            let sp = &meta["scratchpad"];
            let kv_name = sp["kv_param"].as_str().expect("scratchpad.kv_param");
            let sm_name = sp["mask_param"].as_str().expect("scratchpad.mask_param");
            let pget = |n: &str, f: &str| sp["params"][n][f].clone();
            let kv_off_byte = pget(kv_name, "byte_offset").as_u64().expect("kv byte_offset") as usize;
            let sm_off_byte = pget(sm_name, "byte_offset").as_u64().expect("sm byte_offset") as usize;
            let sm_core = pget(sm_name, "kind").as_str() == Some("core");
            // M0.5: optional 2nd addr scratchpad for the transposed self-vcache write column (= n_self).
            let vcache_off_byte = meta.get("vcache_param").and_then(|v| v.as_str())
                .map(|name| pget(name, "byte_offset").as_u64().expect("vcache byte_offset") as usize);
            let sp_head_dim = sp["head_dim"].as_u64().unwrap_or(HEAD_DIM as u64) as u32;
            let res = dev
                .open_elf_resident(&base_elf, Some("main:sequence"))
                .expect("open_elf_resident: decode ELF lacks a ctrl scratchpad (rebuild gen_decode.py)");
            arena.bind_resident(&res).expect("bind resident arena BOs");
            eprintln!(
                "[whisper_decoder] DEEP-C resident scratchpad decode: register-once + per-token \
                 kv_off(byte {kv_off_byte})/sm_mask(byte {sm_off_byte}, core={sm_core}) — no patch/reload"
            );
            Some(ResidentState { res, kv_off_byte, sm_off_byte, sm_core, head_dim: sp_head_dim, vcache_off_byte })
        } else {
            None
        };

        let timing = std::env::var("FUSED_PHASE_TIMING").is_ok();
        if timing {
            eprintln!("[whisper_decoder] FUSED_PHASE_TIMING: per-phase decode breakdown enabled");
        }
        let reuse_ctx = if std::env::var("NPU_DECODE_FUSED_REUSECTX").is_ok() {
            let c = dev.open_elf_ctx(&base_elf).expect("open persistent fused-ELF hw_context");
            eprintln!("[whisper_decoder] NPU_DECODE_FUSED_REUSECTX: persistent hw_context — per-token rebind, no re-registration");
            Some(c)
        } else {
            None
        };
        // Lever #2: register per-layer cross-K/V GEMM ops on the shared ctx2 kernel (NPU fold). Opt
        // out with NPU_DECODE_FUSED_HOSTCROSS=1 (keeps the host f32 fold for A/B + WER comparison).
        let host_cross = std::env::var("NPU_DECODE_FUSED_HOSTCROSS").is_ok();
        // e2e/NPU step-1: build the proj_out ctx2 ops (same shared kernel) when opted-in. Keep an Rc
        // clone since the cross-ops match below consumes `shared`.
        let proj_out_ops = match (shared.as_ref(), std::env::var("NPU_DECODE_PROJOUT_CTX2").is_ok()) {
            (Some(sh), true) => {
                let ops = build_proj_out_ctx2(sh, &w);
                eprintln!(
                    "[whisper_decoder] NPU_DECODE_PROJOUT_CTX2: ln_post+proj_out on NPU ctx2 ({} chunks of N={})",
                    ops.len(),
                    npu_asr::ctx2::NA
                );
                Some(ops)
            }
            _ => None,
        };
        // e2e/NPU wide-dispatch lm-head: the standalone proj_out GEMV ELF (1 dispatch/token) runs
        // ln_post + proj_out + argmax on the NPU, so the host does zero lm-head math (WER-exact 0.1172,
        // ~4.3% faster decode / lm_head -48%, CPU-offload). DEFAULT-ON. Opt out with
        // NPU_DECODE_PROJOUT_ELF=0 (host lm-head). Auto-falls back to the host lm-head when the prebuilt
        // ELF is absent (fresh checkout / CI without the ~84 MB artifact) or when the ctx2 A/B path is
        // explicitly requested. Default dir artifacts/projout_elf; override with NPU_DECODE_PROJOUT_ELF_DIR.
        let projout_optout = matches!(
            std::env::var("NPU_DECODE_PROJOUT_ELF").as_deref(),
            Ok("0") | Ok("false") | Ok("no")
        ) || std::env::var("NPU_DECODE_PROJOUT_CTX2").is_ok();
        let proj_out_elf = if projout_optout {
            None
        } else {
            let dir = std::env::var("NPU_DECODE_PROJOUT_ELF_DIR")
                .map(std::path::PathBuf::from)
                .unwrap_or_else(|_| fused_dir.parent().unwrap_or(fused_dir).join("projout_elf"));
            if dir.join("projout.elf").exists() {
                Some(ProjOutElf::load(dev, &dir, &w))
            } else {
                eprintln!(
                    "[whisper_decoder] proj_out ELF absent at {} -- host lm-head fallback (build scripts/build_projout_elf.sh to enable the on-NPU lm-head; set NPU_DECODE_PROJOUT_ELF=0 to silence)",
                    dir.display()
                );
                None
            }
        };
        let cross_ops = match (shared, host_cross) {
            (Some(sh), false) => {
                let ops: Vec<(CtxAOp, CtxAOp)> = w
                    .layers
                    .iter()
                    .map(|lw| {
                        // cross_k bias is zeros (Epi::None); cross_v has a bias (applied NPU-side).
                        let ck = CtxAOp::new(sh.clone(), &lw.cross_k_w, D, Epi::None, &[]);
                        let cv = CtxAOp::new(
                            sh.clone(),
                            &lw.cross_v_w,
                            D,
                            Epi::Bias,
                            lw.cross_v_b.as_slice().unwrap(),
                        );
                        (ck, cv)
                    })
                    .collect();
                eprintln!("[whisper_decoder] cross-K/V fold on NPU (ctx2 GEMM, lever #2)");
                Some(ops)
            }
            _ => None,
        };
        FusedDecoder {
            dev: Rc::clone(dev),
            w,
            arena,
            base_elf,
            patcher,
            layout,
            output,
            t_enc,
            coalesce_cross,
            int8_cross_k,
            int8_cross_v,
            npu_logits,
            n_self: 0,
            timing,
            ph: PhaseAcc::default(),
            reuse_ctx,
            cross_ops,
            proj_out_ops,
            proj_out_elf,
            next_kern: None,
            resident,
        }
    }

    fn write_buf(&self, name: &str, f: &[f32]) {
        let loc = &self.layout[name];
        let bytes = pack_bf16_bytes(f);
        assert_eq!(bytes.len(), loc.len, "{name}: {} != {}", bytes.len(), loc.len);
        self.arena.write_at(loc.arena, loc.off, &bytes).unwrap();
    }

    /// int8 cross-K: quantize f32 -> int8 with PER-CHANNEL (h,d) scales and write the int8 BYTES into the
    /// bf16-typed buffer (the GEMV kernel reinterprets them as int8). 1 byte/elem -> buffer is half-size.
    /// `scales` is [H*HD] in (h,d) order; the padded `f` is [N_HEADS, T_PAD, HEAD_DIM] head-major, so
    /// element idx -> channel (h = idx/(T_PAD*HEAD_DIM), d = idx%HEAD_DIM).
    fn write_buf_i8(&self, name: &str, f: &[f32], scales: &[f32]) {
        let loc = &self.layout[name];
        let tphd = T_PAD * HEAD_DIM;
        let bytes: Vec<u8> = f.iter().enumerate().map(|(idx, &v)| {
            let s = scales[(idx / tphd) * HEAD_DIM + (idx % HEAD_DIM)];
            let inv = if s != 0.0 { 1.0 / s } else { 0.0 };
            ((v * inv).round().clamp(-127.0, 127.0) as i8) as u8
        }).collect();
        assert_eq!(bytes.len(), loc.len, "{name} (i8): {} != {}", bytes.len(), loc.len);
        self.arena.write_at(loc.arena, loc.off, &bytes).unwrap();
    }

    /// int8 cross-V: like `write_buf_i8` but for the PRE-TRANSPOSED Venc layout [H,HD,TP]. The contiguous
    /// inner dim is T_PAD, so element idx -> channel (h,d) = idx / T_PAD (and `scales` is [H*HD] in (h,d)
    /// order, matching the L*_s_cv buffer op_mul_cv reads).
    fn write_buf_i8_venc(&self, name: &str, f: &[f32], scales: &[f32]) {
        let loc = &self.layout[name];
        let bytes: Vec<u8> = f.iter().enumerate().map(|(idx, &v)| {
            let s = scales[idx / T_PAD];
            let inv = if s != 0.0 { 1.0 / s } else { 0.0 };
            ((v * inv).round().clamp(-127.0, 127.0) as i8) as u8
        }).collect();
        assert_eq!(bytes.len(), loc.len, "{name} (i8v): {} != {}", bytes.len(), loc.len);
        self.arena.write_at(loc.arena, loc.off, &bytes).unwrap();
    }

    fn zero_buf(&self, name: &str) {
        let loc = &self.layout[name];
        self.arena.write_at(loc.arena, loc.off, &vec![0u8; loc.len]).unwrap();
    }

    /// Encoder cross-K/V → per-layer resident scratch (head-major, padded T_enc→T_PAD); also clears
    /// the self-KV caches and the position counter. Mirrors gen_decode.py's heads_pad layout exactly.
    pub fn precompute_cross(&mut self, enc_hidden: &Array2<f32>) {
        // New utterance: start a fresh per-phase breakdown (so each dumped line is one utterance).
        if self.timing {
            self.ph = PhaseAcc::default();
        }
        let mut tmr = Lap::start(self.timing);
        let t = enc_hidden.nrows();
        assert_eq!(t, self.t_enc, "encoder T_enc {} != ELF T_enc {}", t, self.t_enc);
        let w = Rc::clone(&self.w);
        for (li, lw) in w.layers.iter().enumerate() {
            // cross-K/V fold: NPU ctx2 GEMM when registered (lever #2), else host f32.
            let (kenc, venc) = match &self.cross_ops {
                Some(ops) => {
                    let (ck, cv) = &ops[li];
                    // cross_v bias applied NPU-side via Epi::Bias; cross_k has none.
                    (apply_tiled_ctxa(ck, enc_hidden), apply_tiled_ctxa(cv, enc_hidden))
                }
                None => {
                    let kenc = enc_hidden.dot(&lw.cross_k_w); // [T,768], cross_k_b is zeros
                    let mut venc = enc_hidden.dot(&lw.cross_v_w);
                    venc += &lw.cross_v_b.view().insert_axis(Axis(0));
                    (kenc, venc)
                }
            };
            for (name, src) in [(format!("L{li}_Kenc"), &kenc), (format!("L{li}_Venc"), &venc)] {
                // head-major padded: out[h, t, d] = src[t, h*HEAD_DIM + d]; rows t>=T_enc are zero.
                // lever #3 (i): when coalesce_cross, write Venc PRE-TRANSPOSED [H,HD,TP] so the ELF's
                // op_ct_c reads it directly (eliminates the per-token cross-V transpose, the #1 inter-op
                // round-trip). Kenc always stays [H,TP,HD].
                let venc_coalesced = self.coalesce_cross && name.ends_with("Venc");
                let mut padded = vec![0f32; N_HEADS * T_PAD * HEAD_DIM];
                for tt in 0..t {
                    let row = src.row(tt);
                    for h in 0..N_HEADS {
                        for d in 0..HEAD_DIM {
                            let dst = if venc_coalesced {
                                (h * HEAD_DIM + d) * T_PAD + tt // [H,HD,TP] pre-transposed
                            } else {
                                (h * T_PAD + tt) * HEAD_DIM + d // [H,TP,HD] deep-C
                            };
                            padded[dst] = row[h * HEAD_DIM + d];
                        }
                    }
                }
                if self.int8_cross_k && name.ends_with("Kenc") {
                    // PER-UTTERANCE per-channel: s[h,d] = headroom*max_t|Kenc[h,t,d]|/127 from THIS real
                    // Kenc; quantize Kenc -> int8 + write the s_cq buffer that op_mul_cq applies to qc.
                    // headroom (env INT8_CK_HEADROOM, default 1.0): >1 leaves int8 range above max (no
                    // clipping but coarser); =1 maps max->127 (full range, finest). Per-utterance max has
                    // no outliers beyond itself, so 1.0 is safe and uses the whole int8 range.
                    let headroom: f32 = std::env::var("INT8_CK_HEADROOM").ok()
                        .and_then(|v| v.parse().ok()).unwrap_or(1.0);
                    let tphd = T_PAD * HEAD_DIM;
                    let mut s_hd = vec![0f32; N_HEADS * HEAD_DIM];
                    for (idx, &v) in padded.iter().enumerate() {
                        let ch = (idx / tphd) * HEAD_DIM + (idx % HEAD_DIM);
                        let a = v.abs();
                        if a > s_hd[ch] { s_hd[ch] = a; }
                    }
                    for s in s_hd.iter_mut() { *s = if *s > 0.0 { *s * headroom / 127.0 } else { 1.0 }; }
                    self.write_buf_i8(&name, &padded, &s_hd);
                    self.write_buf(&format!("L{li}_s_cq"), &s_hd);
                } else if self.int8_cross_v && name.ends_with("Venc") {
                    // `padded` here is pre-transposed [H,HD,TP] (coalesce_cross is enforced for int8_cross_v),
                    // so the contiguous inner dim is T_PAD and channel (h,d) = idx / T_PAD. Per-channel scale
                    // s_cv[h,d] = headroom*max_t|Venc[h,d,t]|/127; op_mul_cv applies it to the ctc output.
                    let headroom: f32 = std::env::var("INT8_CV_HEADROOM").ok()
                        .and_then(|v| v.parse().ok()).unwrap_or(1.0);
                    let mut s_cv = vec![0f32; N_HEADS * HEAD_DIM];
                    for (idx, &v) in padded.iter().enumerate() {
                        let ch = idx / T_PAD;
                        let a = v.abs();
                        if a > s_cv[ch] { s_cv[ch] = a; }
                    }
                    for s in s_cv.iter_mut() { *s = if *s > 0.0 { *s * headroom / 127.0 } else { 1.0 }; }
                    self.write_buf_i8_venc(&name, &padded, &s_cv);
                    self.write_buf(&format!("L{li}_s_cv"), &s_cv);
                } else {
                    self.write_buf(&name, &padded);
                }
            }
            self.zero_buf(&format!("L{li}_kcache"));
            self.zero_buf(&format!("L{li}_vcache"));
        }
        self.arena.sync_to_device().unwrap();
        self.n_self = 0;
        self.next_kern = None; // PIPE: n_self rewound — any prefetched kernel is now mispatched.
        tmr.lap(&mut self.ph.cross_fold);
        self.ph.utterances += 1;
    }

    /// Fresh self-KV for a new prompt (cross-K/V unchanged for this utterance).
    pub fn reset(&mut self) {
        for li in 0..N_LAYERS {
            self.zero_buf(&format!("L{li}_kcache"));
            self.zero_buf(&format!("L{li}_vcache"));
        }
        self.arena.sync_to_device().unwrap();
        self.n_self = 0;
        self.next_kern = None; // PIPE: n_self rewound — any prefetched kernel is now mispatched.
    }

    /// One decode step → vocab logits `[51865]`. Embeds token+pos, dispatches the whole 12-layer ELF
    /// (KV write slot + softmax mask patched for this position), then host ln_post + proj_out.
    /// Shared decode preamble: embed the token, dispatch the fused decode ELF, return the 768-hidden `x12`.
    fn run_hidden(&mut self, token: i64, pos: usize, tmr: &mut Lap) -> Vec<f32> {
        let tok = token as usize;
        let x: Vec<f32> = (0..D)
            .map(|d| self.w.embed_tokens[[tok, d]] + self.w.embed_positions[[pos, d]])
            .collect();
        tmr.lap(&mut self.ph.embed);
        self.write_buf("x", &x);
        tmr.lap(&mut self.ph.write_x);

        // Deep-C (default when the ELF carries scratchpad params): register-once + per-token scratchpad
        // writes, no patch/reload. Else PIPE (async-prefetch the next position's ELF registration under
        // this dispatch) is the default; the REUSECTX diagnostic opts out to the synchronous rebind path.
        if self.resident.is_some() {
            self.dispatch_resident(tmr);
        } else if self.reuse_ctx.is_some() {
            self.dispatch_sync(tmr);
        } else {
            self.dispatch_pipe(tmr);
        }

        let oloc = &self.layout[&self.output];
        let mut out_bytes = vec![0u8; oloc.len];
        self.arena.read_at(oloc.arena, oloc.off, &mut out_bytes).unwrap();
        let x12 = unpack_bf16_bytes(&out_bytes);
        tmr.lap(&mut self.ph.read_unpack);
        x12
    }

    /// e2e/NPU steady-state: hidden → proj_out + argmax ON THE NPU → token id (only the 64-B partials are
    /// read back, not the 104 KB logits). Requires the argmax-fused proj_out ELF (`gen_projout --argmax`).
    pub fn step_token(&mut self, token: i64, pos: usize) -> i64 {
        let mut tmr = Lap::start(self.timing);
        let x12 = self.run_hidden(token, pos, &mut tmr);
        let nrm = ln_norm_only(&x12[0..D]);
        let id = self.proj_out_elf.as_ref().expect("step_token needs proj_out_elf").token_id(&nrm);
        tmr.lap(&mut self.ph.lm_head);
        self.ph.steps += 1;
        id
    }

    /// Whether the steady-state token-id path (`step_token`) is available (argmax-fused ELF loaded).
    pub fn has_npu_argmax(&self) -> bool {
        self.proj_out_elf.as_ref().is_some_and(|pe| pe.has_argmax())
    }

    pub fn step(&mut self, token: i64, pos: usize) -> Vec<f32> {
        let mut tmr = Lap::start(self.timing);
        let x12 = self.run_hidden(token, pos, &mut tmr);

        // e2e/NPU: the ELF already computed ln_post + proj_out → logits[VOCAB_PAD]. Return logits[0:VOCAB]
        // directly (host argmax over them); the ~40M-MAC host proj_out matmul is gone.
        if self.npu_logits {
            self.ph.steps += 1;
            return x12[0..VOCAB].to_vec();
        }

        // e2e/NPU wide-dispatch lm-head: the standalone proj_out GEMV ELF — ONE dispatch/token (vs the 17
        // ctx2 chunks). Highest precedence. The LN affine + β·W bias are handled inside `logits()`.
        if let Some(ref pe) = self.proj_out_elf {
            let nrm = ln_norm_only(&x12[0..D]);
            let logits = pe.logits(&nrm);
            tmr.lap(&mut self.ph.lm_head);
            self.ph.steps += 1;
            return logits;
        }

        // e2e/NPU step-1: ln_post + proj_out on the NPU ctx2 kernel (17 chunks of N=NA). The LN affine
        // (γ,β) is folded into each chunk's weight/bias, so here we normalize the hidden WITHOUT affine,
        // then run the chunk GEMVs and concatenate into logits[0:VOCAB]. Host argmax follows (step-2 moves it).
        if let Some(ref pops) = self.proj_out_ops {
            let nrm = ln_norm_only(&x12[0..D]);
            let nrm2d = Array2::from_shape_vec((1, D), nrm).expect("nrm [1,D]");
            let na = npu_asr::ctx2::NA;
            let mut logits = vec![0f32; VOCAB];
            for (c, op) in pops.iter().enumerate() {
                let out = op.forward(&nrm2d); // [1, NA] f32
                let row = out.row(0);
                let j0 = c * na;
                let width = (VOCAB - j0).min(na); // last chunk's pad cols (0-weight → 0) are dropped
                logits[j0..j0 + width].copy_from_slice(&row.as_slice().unwrap()[0..width]);
            }
            tmr.lap(&mut self.ph.lm_head);
            self.ph.steps += 1;
            return logits;
        }

        // final LN + proj_out → logits (host f32, like every other backend).
        let ln = ln_row(&x12[0..D], &self.w.ln_post_w, &self.w.ln_post_b);
        let mut logits = vec![0f32; VOCAB];
        let ws = self.w.proj_out_w.as_standard_layout();
        let wslice = ws.as_slice().unwrap();
        for i in 0..D {
            let xi = ln[i];
            let row = &wslice[i * VOCAB..i * VOCAB + VOCAB];
            for j in 0..VOCAB {
                logits[j] += xi * row[j];
            }
        }
        tmr.lap(&mut self.ph.lm_head);
        self.ph.steps += 1;
        logits
    }

    /// Deep-C dispatch: the constant ELF is already registered (resident) and the arena BOs bound, so
    /// one token = write the 2 scratchpad words for this position + dispatch. No patch, no reload, no
    /// re-registration (removes the ~14 ms load_elf + ~1.2 ms patch the patch path pays per token).
    ///   kv_off (addr-kind): raw element-units BD offset = n_self*head_dim
    ///   sm_mask (core-kind): context length n_self+1, shifted <<2 (firmware UPDATE_REG; core >>2)
    fn dispatch_resident(&mut self, tmr: &mut Lap) {
        let n = self.n_self as u32;
        {
            let r = self.resident.as_ref().unwrap();
            let kv_val = n.wrapping_mul(r.head_dim);
            r.res
                .write_scratchpad(r.kv_off_byte, &kv_val.to_le_bytes())
                .expect("write kv_off scratchpad");
            let sm_raw = n + 1;
            let sm_val = if r.sm_core { sm_raw << 2 } else { sm_raw };
            r.res
                .write_scratchpad(r.sm_off_byte, &sm_val.to_le_bytes())
                .expect("write sm_mask scratchpad");
            // M0.5: transposed self-vcache write column = n_self (raw addr, no *head_dim).
            if let Some(vb) = r.vcache_off_byte {
                r.res.write_scratchpad(vb, &n.to_le_bytes()).expect("write vcache_off scratchpad");
            }
        }
        tmr.lap(&mut self.ph.patch); // scratchpad writes (~µs) occupy the old per-token "patch" slot
        self.arena.sync_input().unwrap();
        tmr.lap(&mut self.ph.sync_in);
        self.resident.as_ref().unwrap().res.dispatch().expect("resident decode dispatch");
        tmr.lap(&mut self.ph.dispatch);
        self.arena.sync_from_device().unwrap();
        tmr.lap(&mut self.ph.sync_out);
        self.n_self += 1;
    }

    /// Synchronous registration + dispatch for one token (the REUSECTX diagnostic path only — the
    /// default decode path is `dispatch_pipe`): patch the position-only KV-write offset + softmax mask,
    /// rebind onto the persistent ctx (or, defensively, freshly register), sync the input arena,
    /// dispatch (start+wait), sync the output back, advance `n_self`. `x` is already in the input arena.
    fn dispatch_sync(&mut self, tmr: &mut Lap) {
        let patched = self.patcher.patch(&self.base_elf, self.n_self as u32);
        tmr.lap(&mut self.ph.patch);
        let kern = match &self.reuse_ctx {
            Some(ctx) => FusedKern::Reuse(ctx.rebind(&patched, Some("main:sequence")).expect("rebind fused ELF")),
            None => FusedKern::Fresh(self.dev.load_elf_kernel(&patched, Some("main:sequence")).expect("load fused ELF")),
        };
        tmr.lap(&mut self.ph.load_elf);
        self.arena.sync_input().unwrap();
        tmr.lap(&mut self.ph.sync_in);
        match &kern {
            FusedKern::Reuse(k) => self.arena.dispatch2(k),
            FusedKern::Fresh(k) => self.arena.dispatch(k),
        }
        .expect("fused decode dispatch");
        tmr.lap(&mut self.ph.dispatch);
        self.arena.sync_from_device().unwrap();
        tmr.lap(&mut self.ph.sync_out);
        self.n_self += 1;
    }

    /// PIPE registration + dispatch for one token: the patched ELF for position `n_self` was already
    /// registered during the PREVIOUS token's dispatch (the 1-deep `next_kern` slot) — so this token
    /// skips straight to dispatch. We start the NPU run ASYNC, then register the NEXT position's ELF on
    /// the host WHILE the 56.9 ms run is in flight (it depends only on position, so it is knowable
    /// before this token's logits return), and finally wait. The first token of an utterance and the
    /// first after a `reset` have no predecessor to hide under, so they register synchronously here
    /// (counted into `load_elf`); every steady-state token registers under the dispatch (counted into
    /// `prefetch`, which is OFF the critical path → `load_elf` leaves the per-token `step_sum`).
    fn dispatch_pipe(&mut self, tmr: &mut Lap) {
        // Current kernel: the prefetch from the previous step iff it was built for this exact n_self;
        // otherwise (first token / post-reset) register it now, on the critical path.
        let cur = match self.next_kern.take() {
            Some((p, k)) if p == self.n_self => k,
            _ => {
                let patched = self.patcher.patch(&self.base_elf, self.n_self as u32);
                tmr.lap(&mut self.ph.patch);
                let k = self
                    .dev
                    .load_elf_kernel(&patched, Some("main:sequence"))
                    .expect("load fused ELF");
                tmr.lap(&mut self.ph.load_elf);
                k
            }
        };
        self.arena.sync_input().unwrap();
        tmr.lap(&mut self.ph.sync_in);
        // Start the NPU run; it returns immediately. `cur` + the arena must outlive `run` (held below).
        let run: Run = self.arena.dispatch_start(&cur).expect("fused decode dispatch start");
        // Overlap window: build + register the NEXT position's patched ELF while the NPU computes.
        let pf_mark = self.timing.then(std::time::Instant::now);
        let next_pos = self.n_self + 1;
        let patched_next = self.patcher.patch(&self.base_elf, next_pos as u32);
        let next_k = self
            .dev
            .load_elf_kernel(&patched_next, Some("main:sequence"))
            .expect("load next fused ELF");
        self.next_kern = Some((next_pos, next_k));
        if let Some(m) = pf_mark {
            self.ph.prefetch += std::time::Instant::now().duration_since(m).as_nanos();
        }
        // Block for the NPU run. `dispatch` thus times start + overlapped prefetch + wait = the true
        // per-token critical path (if XRT serialized the host build behind the run, it shows up here).
        run.wait().expect("fused decode dispatch wait");
        tmr.lap(&mut self.ph.dispatch);
        self.arena.sync_from_device().unwrap();
        tmr.lap(&mut self.ph.sync_out);
        self.n_self += 1;
        // `cur` owns the in-flight run's hw_context; it MUST NOT drop before run.wait() above. Owned
        // values drop at lexical scope end (not last-use), so this explicit drop only documents intent.
        drop(cur);
    }

    /// Dump the per-phase breakdown accumulated since the last `precompute_cross` (one utterance).
    /// Per-token phases are reported as mean ms/step (= ms/token) and as a share of the per-step sum;
    /// `cross_fold` is the once-per-utterance encoder cross-K/V cost. The per-step sum reconciles to
    /// the `WHISPER_TIMING` `ms_per_tok` (modulo the few extra lang-detect+prompt dispatches). No-op
    /// unless `FUSED_PHASE_TIMING` is set.
    pub fn dump_phase_timing(&self) {
        if let Some(pe) = &self.proj_out_elf {
            pe.dump_amax_stats();
        }
        if !self.timing || self.ph.steps == 0 {
            return;
        }
        let p = &self.ph;
        let n = p.steps as f64;
        let ms = |x: u128| x as f64 / 1e6;
        let per = |x: u128| ms(x) / n; // mean ms per dispatch (per token)
        let step_sum =
            p.embed + p.write_x + p.patch + p.load_elf + p.sync_in + p.dispatch + p.sync_out + p.read_unpack + p.lm_head;
        let pct = |x: u128| if step_sum > 0 { 100.0 * x as f64 / step_sum as f64 } else { 0.0 };
        // Single greppable line (means in ms/token) for the result note. `prefetch_ms` is the PIPE
        // overlap (next-token registration hidden under the NPU dispatch); it is NOT in step_sum.
        eprintln!(
            "[FUSED_PHASE] steps={} cross_fold_ms={:.2} embed_ms={:.3} write_x_ms={:.3} \
             patch_ms={:.3} load_elf_ms={:.3} prefetch_ms={:.3} sync_in_ms={:.3} dispatch_ms={:.3} \
             sync_out_ms={:.3} read_unpack_ms={:.3} lm_head_ms={:.3} step_sum_ms={:.3}",
            p.steps,
            ms(p.cross_fold),
            per(p.embed),
            per(p.write_x),
            per(p.patch),
            per(p.load_elf),
            per(p.prefetch),
            per(p.sync_in),
            per(p.dispatch),
            per(p.sync_out),
            per(p.read_unpack),
            per(p.lm_head),
            ms(step_sum) / n,
        );
        // Human-readable ranked table (descending share of the per-token sum).
        let mut rows = [
            ("embed", p.embed),
            ("write_x", p.write_x),
            ("patch", p.patch),
            ("load_elf", p.load_elf),
            ("sync_in", p.sync_in),
            ("dispatch", p.dispatch),
            ("sync_out", p.sync_out),
            ("read_unpack", p.read_unpack),
            ("lm_head", p.lm_head),
        ];
        rows.sort_by(|a, b| b.1.cmp(&a.1));
        eprintln!("[FUSED_PHASE] per-token breakdown ({} dispatches, mean ms/token, ranked):", p.steps);
        for (name, v) in rows {
            eprintln!("  {name:<12} {:>8.3} ms  {:>5.1}%", per(v), pct(v));
        }
        eprintln!(
            "  {:<12} {:>8.3} ms/token   (+ {:.2} ms/utterance cross_fold)",
            "step_sum",
            ms(step_sum) / n,
            ms(p.cross_fold),
        );
        if p.prefetch > 0 {
            eprintln!(
                "  {:<12} {:>8.3} ms/token   (PIPE: next-token ELF registration, OVERLAPPED under dispatch — off critical path)",
                "prefetch",
                per(p.prefetch),
            );
        }
    }
}

// =============================================================================================
// Subsystem-B: BatchedFusedDecoder — drive the batched decode ELF (gen_decode_batched.py
// --scratchpad) for B streams at once, OFFLINE-BULK LOCKSTEP. All B streams advance one token per
// step, so deep-C's scalar scratchpad params (kv_off = n_self*head_dim, sm_mask = (n_self+1)<<2) are
// SHARED across the batch. Per step: write B embed rows into the B-wide `x` input -> write the 2
// scalar params -> ONE dispatch -> read B output rows -> B host lm-heads. Resident-only.
// =============================================================================================
/// Batched whole-decode fused-ELF backend (subsystem B). See module banner.
pub struct BatchedFusedDecoder {
    w: Rc<WhisperDecoderWeights>,
    arena: FusedArena,
    layout: HashMap<String, BufLoc>,
    output: String, // e.g. "x12"
    b: usize,       // batch width (streams)
    nl: usize,      // layer count from meta.dims.layers (12 for prod; <12 for build-cheap perf-probe ELFs)
    t_enc: usize,
    t_pad: usize,
    n_self: usize,
    res: ElfResident,
    kv_off_byte: usize,
    sm_off_byte: usize,
    sm_core: bool,
    head_dim: u32,
    timing: bool, // env FUSED_PHASE_TIMING: per-phase decode breakdown
    ph: PhaseAcc,
    /// O1 (lever #2 for the batch): per-layer (cross_k, cross_v) GEMM ops on the encoder's shared ctx2
    /// kernel. When `Some`, `precompute_cross_batch` folds each stream's encoder cross-K/V on the NPU
    /// (~0.078 s/utt, like M=1) instead of the naive host f32 `enc.dot()` (6.36 s/batch). `None` = host
    /// fold (fallback / A-B). Opt out with NPU_DECODE_FUSED_HOSTCROSS=1.
    cross_ops: Option<Vec<(CtxAOp, CtxAOp)>>,
}

impl BatchedFusedDecoder {
    /// Load the prebuilt batched decode ELF + resident weight arena. `dir` holds decode_b.elf,
    /// meta.json (with a `scratchpad` block + dims.B/T), buffers/<name>.bin. `shared` = the encoder's
    /// resident ctx2 kernel (Some → O1 NPU cross-K/V fold; None → host f32 fold).
    pub fn new(w: Rc<WhisperDecoderWeights>, dev: &Rc<Device>, dir: &Path, shared: Option<Rc<SharedCtxA>>) -> Self {
        let base_elf = std::fs::read(dir.join("decode_b.elf"))
            .unwrap_or_else(|e| panic!("read decode_b.elf: {e} (gen_decode_batched.py --scratchpad)"));
        let meta: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("meta.json")).expect("read meta.json"))
                .expect("parse meta.json");
        let usz = |k: &str| meta[k].as_u64().expect(k) as usize;
        let (in_sz, out_sz, scr_sz) = (usz("input_size"), usz("output_size"), usz("scratch_size"));
        let output = meta["output"].as_str().expect("output").to_string();
        let b = meta["dims"]["B"].as_u64().expect("dims.B") as usize;
        let nl = meta["dims"]["layers"].as_u64().unwrap_or(N_LAYERS as u64) as usize;
        let t_enc = meta["dims"]["T"].as_u64().expect("dims.T") as usize;
        let t_pad = ((t_enc + 63) / 64) * 64;

        let mut layout = HashMap::new();
        for (name, e) in meta["layout"].as_object().expect("layout") {
            let arena = match e["type"].as_str().unwrap() {
                "input" => Arena::Input,
                "output" => Arena::Output,
                "scratch" => Arena::Scratch,
                o => panic!("bad arena type {o}"),
            };
            layout.insert(
                name.clone(),
                BufLoc { arena, off: e["offset"].as_u64().unwrap() as usize, len: e["len"].as_u64().unwrap() as usize },
            );
        }
        let arena = FusedArena::new(dev, in_sz, out_sz, scr_sz).expect("alloc batched fused arenas");
        // static weights (skip per-utterance encoder-K/V + self-KV caches).
        for name in meta["weights"].as_array().expect("weights") {
            let name = name.as_str().unwrap();
            if name.ends_with("Kenc") || name.ends_with("Venc") || name.ends_with("kcache") || name.ends_with("vcache") {
                continue;
            }
            let bytes = std::fs::read(dir.join("buffers").join(format!("{name}.bin")))
                .unwrap_or_else(|e| panic!("read buffer {name}.bin: {e}"));
            let loc = &layout[name];
            assert_eq!(bytes.len(), loc.len, "{name}: blob {} != layout {}", bytes.len(), loc.len);
            arena.write_at(loc.arena, loc.off, &bytes).unwrap();
        }
        let sp = &meta["scratchpad"];
        let kvn = sp["kv_param"].as_str().expect("scratchpad.kv_param");
        let smn = sp["mask_param"].as_str().expect("scratchpad.mask_param");
        let kv_off_byte = sp["params"][kvn]["byte_offset"].as_u64().expect("kv byte_offset") as usize;
        let sm_off_byte = sp["params"][smn]["byte_offset"].as_u64().expect("sm byte_offset") as usize;
        let sm_core = sp["params"][smn]["kind"].as_str() == Some("core");
        let head_dim = sp["head_dim"].as_u64().unwrap_or(HEAD_DIM as u64) as u32;
        let res = dev
            .open_elf_resident(&base_elf, Some("main:sequence"))
            .expect("open_elf_resident (batched decode ELF lacks scratchpad?)");
        arena.bind_resident(&res).expect("bind resident arena BOs");
        eprintln!("[batched] B={b} t_enc={t_enc} resident scratchpad decode (scratch {:.0} MB)", scr_sz as f64 / 1e6);
        let timing = std::env::var("FUSED_PHASE_TIMING").is_ok();
        // O1: register per-layer cross-K/V GEMM ops on the encoder's shared ctx2 kernel (NPU fold),
        // mirroring FusedDecoder's lever #2. Opt out (host f32 fold) with NPU_DECODE_FUSED_HOSTCROSS=1.
        let host_cross = std::env::var("NPU_DECODE_FUSED_HOSTCROSS").is_ok();
        let cross_ops = match (shared, host_cross) {
            (Some(sh), false) => {
                let ops: Vec<(CtxAOp, CtxAOp)> = w
                    .layers
                    .iter()
                    .take(nl) // only the nl layers this ELF uses (avoids wasted weight uploads on perf-probe ELFs)
                    .map(|lw| {
                        let ck = CtxAOp::new(sh.clone(), &lw.cross_k_w, D, Epi::None, &[]);
                        let cv = CtxAOp::new(
                            sh.clone(),
                            &lw.cross_v_w,
                            D,
                            Epi::Bias,
                            lw.cross_v_b.as_slice().unwrap(),
                        );
                        (ck, cv)
                    })
                    .collect();
                eprintln!("[batched] cross-K/V fold on NPU (ctx2 GEMM, O1) for all B streams");
                Some(ops)
            }
            _ => {
                eprintln!("[batched] cross-K/V fold on HOST (naive f32) — no shared ctx2 / HOSTCROSS set");
                None
            }
        };
        BatchedFusedDecoder {
            w, arena, layout, output, b, nl, t_enc, t_pad, n_self: 0, res, kv_off_byte, sm_off_byte, sm_core, head_dim,
            timing, ph: PhaseAcc::default(), cross_ops,
        }
    }

    pub fn batch(&self) -> usize {
        self.b
    }

    /// Dispatches accumulated since the last `precompute_cross_batch` (O3: per-bucket step count for
    /// the utilisation metric real_tokens / (steps × B)).
    pub fn last_steps(&self) -> usize {
        self.ph.steps as usize
    }

    fn zero_buf(&self, name: &str) {
        let loc = &self.layout[name];
        self.arena.write_at(loc.arena, loc.off, &vec![0u8; loc.len]).unwrap();
    }

    /// Write one stream's row into a B-wide buffer (`x` input, [B, D] stream-major).
    fn write_row(&self, name: &str, bi: usize, f: &[f32]) {
        let loc = &self.layout[name];
        let bytes = pack_bf16_bytes(f);
        let row = bytes.len();
        assert_eq!(loc.len, self.b * row, "{name} not B-wide ({} != {}*{})", loc.len, self.b, row);
        self.arena.write_at(loc.arena, loc.off + bi * row, &bytes).unwrap();
    }

    /// Fold B encoders' cross-K/V into the B-wide per-layer resident scratch (head-major, padded
    /// T_enc->T_PAD per stream); clear self-KV; reset position. Host f32 fold (per-stream, parallel).
    pub fn precompute_cross_batch(&mut self, encs: &[Array2<f32>]) {
        assert_eq!(encs.len(), self.b, "need exactly B={} encoder outputs", self.b);
        // Fresh per-phase counters for this batch/bucket (so last_steps() == this bucket's dispatches,
        // O3; and each FUSED_PHASE_TIMING dump is one batch).
        self.ph = PhaseAcc::default();
        let mut tmr = Lap::start(self.timing);
        let stream_elems = N_HEADS * self.t_pad * HEAD_DIM;
        let w = Rc::clone(&self.w);
        for (li, lw) in w.layers.iter().take(self.nl).enumerate() {
            let mut kbuf = vec![0f32; self.b * stream_elems];
            let mut vbuf = vec![0f32; self.b * stream_elems];
            for (bi, enc) in encs.iter().enumerate() {
                let t = enc.nrows();
                assert_eq!(t, self.t_enc, "encoder T_enc {} != ELF {}", t, self.t_enc);
                // O1: NPU ctx2 GEMM fold when registered (cross_v bias applied NPU-side via Epi::Bias),
                // else naive host f32 (the 6.36 s/batch fallback).
                let (kenc, venc) = match &self.cross_ops {
                    Some(ops) => {
                        let (ck, cv) = &ops[li];
                        (apply_tiled_ctxa(ck, enc), apply_tiled_ctxa(cv, enc))
                    }
                    None => {
                        let kenc = enc.dot(&lw.cross_k_w); // [T,768], cross_k bias is zeros
                        let mut venc = enc.dot(&lw.cross_v_w);
                        venc += &lw.cross_v_b.view().insert_axis(Axis(0));
                        (kenc, venc)
                    }
                };
                let base = bi * stream_elems;
                for tt in 0..t {
                    let kr = kenc.row(tt);
                    let vr = venc.row(tt);
                    for h in 0..N_HEADS {
                        let dst = base + (h * self.t_pad + tt) * HEAD_DIM;
                        for d in 0..HEAD_DIM {
                            kbuf[dst + d] = kr[h * HEAD_DIM + d];
                            vbuf[dst + d] = vr[h * HEAD_DIM + d];
                        }
                    }
                }
            }
            for (name, src) in [(format!("L{li}_Kenc"), &kbuf), (format!("L{li}_Venc"), &vbuf)] {
                let loc = &self.layout[&name];
                let bytes = pack_bf16_bytes(src);
                assert_eq!(bytes.len(), loc.len, "{name}: {} != {}", bytes.len(), loc.len);
                self.arena.write_at(loc.arena, loc.off, &bytes).unwrap();
            }
            self.zero_buf(&format!("L{li}_kcache"));
            self.zero_buf(&format!("L{li}_vcache"));
        }
        self.arena.sync_to_device().unwrap();
        self.n_self = 0;
        tmr.lap(&mut self.ph.cross_fold);
        self.ph.utterances += 1;
    }

    /// Fresh self-KV for a new prompt (cross-K/V unchanged for this utterance batch).
    pub fn reset(&mut self) {
        for li in 0..self.nl {
            self.zero_buf(&format!("L{li}_kcache"));
            self.zero_buf(&format!("L{li}_vcache"));
        }
        self.arena.sync_to_device().unwrap();
        self.n_self = 0;
    }

    /// One lockstep decode step for all B streams. `tokens[bi]` is stream bi's current token; `pos` is
    /// the shared position (lockstep). Returns B logit vectors [VOCAB].
    pub fn step_batch(&mut self, tokens: &[i64], pos: usize) -> Vec<Vec<f32>> {
        assert_eq!(tokens.len(), self.b, "need B={} tokens", self.b);
        let mut tmr = Lap::start(self.timing);
        for (bi, &tok) in tokens.iter().enumerate() {
            let t = tok as usize;
            let x: Vec<f32> = (0..D)
                .map(|d| self.w.embed_tokens[[t, d]] + self.w.embed_positions[[pos, d]])
                .collect();
            self.write_row("x", bi, &x);
        }
        tmr.lap(&mut self.ph.write_x); // B embed lookups + pack + write
        let n = self.n_self as u32;
        let kv_val = n.wrapping_mul(self.head_dim);
        self.res.write_scratchpad(self.kv_off_byte, &kv_val.to_le_bytes()).expect("kv_off");
        let sm_raw = n + 1;
        let sm_val = if self.sm_core { sm_raw << 2 } else { sm_raw };
        self.res.write_scratchpad(self.sm_off_byte, &sm_val.to_le_bytes()).expect("sm_mask");
        tmr.lap(&mut self.ph.patch); // scratchpad writes
        self.arena.sync_input().unwrap();
        tmr.lap(&mut self.ph.sync_in);
        self.res.dispatch().expect("batched decode dispatch");
        tmr.lap(&mut self.ph.dispatch);
        self.arena.sync_from_device().unwrap();
        tmr.lap(&mut self.ph.sync_out);
        self.n_self += 1;

        let oloc = &self.layout[&self.output];
        let row = D * 2;
        let mut out_bytes = vec![0u8; oloc.len];
        self.arena.read_at(oloc.arena, oloc.off, &mut out_bytes).unwrap();
        tmr.lap(&mut self.ph.read_unpack);
        // O2: batched lm-head. Build LN_all [B,D] (per-stream ln_post), then ONE GEMM
        // LN_all @ proj_out_w[D,VOCAB] -> logits [B,VOCAB] (ndarray .dot, cache-blocked/SIMD),
        // replacing the B naive D×VOCAB triple loops. Same f32 math; argmax-identical.
        let mut ln_all = Array2::<f32>::zeros((self.b, D));
        for bi in 0..self.b {
            let x12 = unpack_bf16_bytes(&out_bytes[bi * row..(bi + 1) * row]);
            let ln = ln_row(&x12[0..D], &self.w.ln_post_w, &self.w.ln_post_b);
            ln_all.row_mut(bi).assign(&Array1::from_vec(ln));
        }
        let logits_mat = ln_all.dot(&self.w.proj_out_w); // [B, VOCAB]
        let all: Vec<Vec<f32>> = (0..self.b).map(|bi| logits_mat.row(bi).to_vec()).collect();
        tmr.lap(&mut self.ph.lm_head); // batched ln_post + one proj_out GEMM
        self.ph.steps += 1;
        all
    }

    /// Per-phase batched-decode breakdown (env FUSED_PHASE_TIMING). `steps` = dispatches; each dispatch
    /// produces B tokens, so per-token figures divide the per-step mean by B.
    pub fn dump_phase_timing(&self) {
        if !self.timing || self.ph.steps == 0 {
            return;
        }
        let p = &self.ph;
        let n = p.steps as f64;
        let ms = |x: u128| x as f64 / 1e6;
        let per = |x: u128| ms(x) / n; // mean ms per dispatch (per step = B tokens)
        let step_sum = p.write_x + p.patch + p.sync_in + p.dispatch + p.sync_out + p.read_unpack + p.lm_head;
        eprintln!(
            "[BATCHED_PHASE] B={} steps={} cross_fold_ms={:.2} write_x_ms={:.3} patch_ms={:.3} \
             sync_in_ms={:.3} dispatch_ms={:.3} sync_out_ms={:.3} read_unpack_ms={:.3} lm_head_ms={:.3} \
             step_sum_ms_per_dispatch={:.3}  (per-token = /{}; cross_fold once/batch)",
            self.b, p.steps, ms(p.cross_fold), per(p.write_x), per(p.patch), per(p.sync_in),
            per(p.dispatch), per(p.sync_out), per(p.read_unpack), per(p.lm_head), ms(step_sum) / n, self.b,
        );
        let mut rows = [
            ("write_x", p.write_x), ("patch", p.patch), ("sync_in", p.sync_in), ("dispatch", p.dispatch),
            ("sync_out", p.sync_out), ("read_unpack", p.read_unpack), ("lm_head", p.lm_head),
        ];
        rows.sort_by(|a, b| b.1.cmp(&a.1));
        let pct = |x: u128| if step_sum > 0 { 100.0 * x as f64 / step_sum as f64 } else { 0.0 };
        eprintln!("[BATCHED_PHASE] per-dispatch breakdown (B={} streams/dispatch, mean ms, ranked):", self.b);
        for (name, v) in rows {
            eprintln!("  {name:<12} {:>8.3} ms/dispatch  {:>8.4} ms/token  {:>5.1}%", per(v), per(v) / self.b as f64, pct(v));
        }
        eprintln!(
            "  {:<12} {:>8.3} ms/dispatch  {:>8.4} ms/token   (+ {:.1} ms cross_fold/batch over {} streams)",
            "step_sum", ms(step_sum) / n, (ms(step_sum) / n) / self.b as f64, ms(p.cross_fold), self.b,
        );
    }
}

/// Apply a K=768→768 ctx2 GEMM op to `x` `[M, 768]` (M may exceed PAD_M), row-tiling into chunks of
/// ≤PAD_M and stacking the `[M, 768]` result in row order. Mirrors `npu_whisper::npu::apply_tiled`
/// (inlined here to avoid the cfg-gated dependency). Used for the on-NPU cross-K/V fold (lever #2).
fn apply_tiled_ctxa(op: &CtxAOp, x: &Array2<f32>) -> Array2<f32> {
    let m = x.nrows();
    let mut out = Array2::<f32>::zeros((m, D));
    let mut r = 0;
    while r < m {
        let end = (r + PAD_M).min(m);
        let chunk = x.slice(s![r..end, ..]).to_owned();
        out.slice_mut(s![r..end, ..]).assign(&op.forward(&chunk));
        r = end;
    }
    out
}

/// e2e/NPU step-1: build the per-chunk `proj_out` ctx2 ops. The resident ctx2 kernel serves output
/// widths ≤ `NA`=3072, so the 51865-wide vocab is computed in `ceil(VOCAB/NA)=17` chunks, each a
/// `CtxAOp(N=NA, Epi::Bias)`. The LN affine folds into the weight/bias so the GEMV input is the
/// affine-free normalized hidden: `logits = (norm·γ+β)·W = norm·(γ⊙W) + (β·W)`. Per chunk:
///   `W'[i, jj] = γ[i] · proj_out_w[i, j0+jj]`,  `bias'[jj] = Σ_i β[i]·proj_out_w[i, j0+jj]`
/// (the last chunk's `jj ≥ width` cols get 0 weight + 0 bias — harmless, dropped in `step()`).
fn build_proj_out_ctx2(sh: &Rc<SharedCtxA>, w: &WhisperDecoderWeights) -> Vec<CtxAOp> {
    let na = npu_asr::ctx2::NA;
    let gamma = &w.ln_post_w; // [D]
    let beta = &w.ln_post_b;  // [D]
    let pw = &w.proj_out_w;   // [D, VOCAB]
    assert_eq!(pw.dim(), (D, VOCAB), "proj_out_w shape");
    let n_chunks = VOCAB.div_ceil(na);
    let mut ops = Vec::with_capacity(n_chunks);
    for c in 0..n_chunks {
        let j0 = c * na;
        let width = (VOCAB - j0).min(na);
        let mut wp = Array2::<f32>::zeros((D, na));
        let mut bias = vec![0f32; na];
        for i in 0..D {
            let (gi, bi) = (gamma[i], beta[i]);
            for jj in 0..width {
                let wv = pw[[i, j0 + jj]];
                wp[[i, jj]] = gi * wv;
                bias[jj] += bi * wv;
            }
        }
        ops.push(CtxAOp::new(sh.clone(), &wp, na, Epi::Bias, &bias));
    }
    ops
}

/// Affine-free LayerNorm normalize of a single row: `(x - μ)/σ` (population variance, eps 1e-5), WITHOUT
/// the γ/β affine — used by the e2e/NPU proj_out paths, where the ln_post affine is folded into the
/// proj_out weight/bias instead (`logits = (norm·γ+β)·W = norm·(γ⊙W) + β·W`).
fn ln_norm_only(x: &[f32]) -> Vec<f32> {
    let d = x.len();
    let mean: f32 = x.iter().sum::<f32>() / d as f32;
    let var: f32 = x.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
    let inv = 1.0 / (var + LN_EPS).sqrt();
    x.iter().map(|&v| (v - mean) * inv).collect()
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
