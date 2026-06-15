//! ESM-2 pre-norm encoder on the NPU (C1 zero-pad path). The resident matmul kernel is hardwired to
//! K=KA(768), N∈{768,1536,3072}, and FfnMm2 to [3072,768]. ESM's real widths (hidden 320/480, ff
//! 1280/1920) are zero-padded onto those shapes as the matmul B-operand; LayerNorm/residual/RoPE run
//! on the REAL hidden H (never on the zero padding, else the norm statistics are wrong). Reuses the
//! npu_asr ctx2 matmul engines untouched.
use std::path::Path;
use std::rc::Rc;

use ndarray::{s, Array2};
use npu_asr::ctx2::{CtxAOp, Epi, FfnMm2, SharedCtxA, KA, MM2_OUT, NA};
use npu_asr_host::{gelu, layer_norm, mha};
use npu_xrt::Device;

use crate::esm::rope;
use crate::esm::weights::{EsmLayer, EsmWeights};
use crate::pipeline::Encoder;

const LN_EPS: f32 = 1e-5;

/// Place `w` (rows×cols ≤ target) in the top-left of a zeros(target) — zero-pads K (rows) and N (cols).
fn pad2(w: &Array2<f32>, rows: usize, cols: usize) -> Array2<f32> {
    let (r, c) = w.dim();
    if r == rows && c == cols {
        return w.clone();
    }
    let mut p = Array2::<f32>::zeros((rows, cols));
    p.slice_mut(s![..r, ..c]).assign(w);
    p
}
/// Pad an activation [seq, h] to [seq, cols] with zero columns.
fn pad_cols(x: &Array2<f32>, cols: usize) -> Array2<f32> {
    let (seq, h) = x.dim();
    if h == cols {
        return x.clone();
    }
    let mut p = Array2::<f32>::zeros((seq, cols));
    p.slice_mut(s![.., ..h]).assign(x);
    p
}
/// Slice an activation [seq, *] down to its first `h` columns.
fn slice_cols(y: &Array2<f32>, h: usize) -> Array2<f32> {
    y.slice(s![.., ..h]).to_owned()
}
/// Smallest served stream width >= n.
fn round_stream(n: usize) -> usize {
    [768usize, 1536, 3072].into_iter().find(|&s| s >= n).expect("N must be <= 3072")
}
fn vpad(v: &[f32], n: usize) -> Vec<f32> {
    let mut o = v.to_vec();
    o.resize(n, 0.0);
    o
}
fn add(a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
    let mut o = a.clone();
    o.zip_mut_with(b, |x, &y| *x += y);
    o
}
/// Apply RoPE per head over x[seq, n_heads*head_dim].
fn rope_heads(x: &mut Array2<f32>, n_heads: usize, head_dim: usize) {
    let seq = x.nrows();
    let (cos, sin) = rope::tables(seq, head_dim);
    for hh in 0..n_heads {
        let mut head = x.slice(s![.., hh * head_dim..(hh + 1) * head_dim]).to_owned();
        rope::apply(&mut head, &cos, &sin);
        x.slice_mut(s![.., hh * head_dim..(hh + 1) * head_dim]).assign(&head);
    }
}

struct EsmBlock {
    ln_attn_w: Vec<f32>,
    ln_attn_b: Vec<f32>,
    ln_ffn_w: Vec<f32>,
    ln_ffn_b: Vec<f32>,
    q: CtxAOp,
    k: CtxAOp,
    v: CtxAOp,
    o: CtxAOp,
    ffn1: CtxAOp,
    ffn2: FfnMm2,
    hidden: usize,
    ff: usize,
    n_heads: usize,
    head_dim: usize,
}
impl EsmBlock {
    fn new(shared: Rc<SharedCtxA>, l: &EsmLayer, hidden: usize, ff: usize, n_heads: usize, head_dim: usize) -> Self {
        let proj_n = round_stream(hidden); // 320/480 -> 768
        let ff_n = round_stream(ff); // 1280 -> 1536, 1920 -> 3072
        // q/k/v/o: weight [hidden,hidden] -> pad to [KA, proj_n]; bias -> proj_n.
        let mk = |wk: &str, bk: &str, n: usize| {
            CtxAOp::new(shared.clone(), &pad2(&l.m(wk), KA, n), n, Epi::Bias, &vpad(&l.v(bk), n))
        };
        let ffn1 = CtxAOp::new(shared.clone(), &pad2(&l.m("ffn1_w"), KA, ff_n), ff_n, Epi::Bias, &vpad(&l.v("ffn1_b"), ff_n));
        // ffn2: weight [ff, hidden] -> pad to [NA=3072, MM2_OUT=768]; bias -> 768.
        let ffn2 = FfnMm2::new(shared.clone(), &pad2(&l.m("ffn2_w"), NA, MM2_OUT), &vpad(&l.v("ffn2_b"), MM2_OUT));
        EsmBlock {
            ln_attn_w: l.v("ln_attn_w"),
            ln_attn_b: l.v("ln_attn_b"),
            ln_ffn_w: l.v("ln_ffn_w"),
            ln_ffn_b: l.v("ln_ffn_b"),
            q: mk("q_w", "q_b", proj_n),
            k: mk("k_w", "k_b", proj_n),
            v: mk("v_w", "v_b", proj_n),
            o: mk("attn_out_w", "attn_out_b", proj_n),
            ffn1,
            ffn2,
            hidden,
            ff,
            n_heads,
            head_dim,
        }
    }
    fn forward(&self, x: &Array2<f32>, valid: usize) -> Array2<f32> {
        let h = self.hidden;
        // --- pre-norm self-attention ---
        let xn = layer_norm(x, &self.ln_attn_w, &self.ln_attn_b, LN_EPS); // [seq, h]
        let xn_p = pad_cols(&xn, KA);
        let mut q = slice_cols(&self.q.forward(&xn_p), h);
        let mut k = slice_cols(&self.k.forward(&xn_p), h);
        let vv = slice_cols(&self.v.forward(&xn_p), h);
        rope_heads(&mut q, self.n_heads, self.head_dim);
        rope_heads(&mut k, self.n_heads, self.head_dim);
        let ctx = mha(&q, &k, &vv, self.n_heads, self.head_dim, false, valid); // [seq, h]
        let attn = slice_cols(&self.o.forward(&pad_cols(&ctx, KA)), h);
        let x = add(x, &attn); // residual, NO LayerNorm (pre-norm)
        // --- pre-norm FFN ---
        let fnn = layer_norm(&x, &self.ln_ffn_w, &self.ln_ffn_b, LN_EPS); // [seq, h]
        let ff_out = slice_cols(&self.ffn1.forward(&pad_cols(&fnn, KA)), self.ff); // [seq, ff]
        let hmid = gelu(&ff_out); // gelu(0)=0 so padding stays 0
        let y = slice_cols(&self.ffn2.forward(&pad_cols(&hmid, NA)), h); // [seq, h]
        add(&x, &y)
    }
}

pub struct EsmEncoder {
    blocks: Vec<EsmBlock>,
    final_ln_w: Vec<f32>,
    final_ln_b: Vec<f32>,
}
impl EsmEncoder {
    pub fn new(dev: Rc<Device>, root: &Path, w: &EsmWeights, hidden: usize, ff: usize, n_heads: usize, head_dim: usize) -> Self {
        let cfg = crate::tuning_profile::resolve(root, npu_asr::ctx2::Precision::from_env());
        let shared = SharedCtxA::with_tuning(&dev, root, &cfg);
        let blocks = (0..w.n_layers())
            .map(|i| EsmBlock::new(shared.clone(), &w.layers[i], hidden, ff, n_heads, head_dim))
            .collect();
        EsmEncoder { blocks, final_ln_w: w.final_ln_w.clone(), final_ln_b: w.final_ln_b.clone() }
    }
}
impl Encoder for EsmEncoder {
    fn forward_last(&self, x: &Array2<f32>, valid: usize) -> Array2<f32> {
        let mut x = x.clone();
        for b in &self.blocks {
            x = b.forward(&x, valid);
        }
        layer_norm(&x, &self.final_ln_w, &self.final_ln_b, LN_EPS) // final LN after all layers
    }
}

// ===================== NATIVE path (research/comparison) =====================
use crate::esm::native::{NativeKernel, NativeWeight};

/// Round N up to a multiple of 256 (= tile_n(32) * n_aie_cols(8), the whole-array tiling constraint).
fn round256(n: usize) -> usize { n.div_ceil(256) * 256 }

struct EsmBlockNative {
    ln_attn_w: Vec<f32>, ln_attn_b: Vec<f32>,
    ln_ffn_w: Vec<f32>, ln_ffn_b: Vec<f32>,
    wq: NativeWeight, wk: NativeWeight, wv: NativeWeight, wo: NativeWeight,
    bq: Vec<f32>, bk: Vec<f32>, bv: Vec<f32>, bo: Vec<f32>,
    wf1: NativeWeight, bf1: Vec<f32>,
    wf2: NativeWeight, bf2: Vec<f32>,
    hidden: usize, ff: usize, n_heads: usize, head_dim: usize,
}

/// Native ESM encoder: REAL K per matmul (no zero-pad to 768), N padded only to the 256 tiling
/// multiple. 3 shared NativeKernels (proj/ffn1/ffn2) => 3 hw-contexts; switching between them per
/// layer costs context switches (recorded as native overhead). LN/residual/RoPE on the real hidden.
pub struct EsmEncoderNative {
    kp: Rc<NativeKernel>,  // projections: K=hidden, N=round256(hidden)
    kf1: Rc<NativeKernel>, // ffn1: K=hidden, N=round256(ff)
    kf2: Rc<NativeKernel>, // ffn2: K=ff,    N=round256(hidden)
    blocks: Vec<EsmBlockNative>,
    final_ln_w: Vec<f32>, final_ln_b: Vec<f32>,
}
impl EsmEncoderNative {
    pub fn new(dev: Rc<Device>, root: &Path, w: &EsmWeights, hidden: usize, ff: usize, n_heads: usize, head_dim: usize) -> Self {
        let wa = root.join("mlir-aie/programming_examples/basic/matrix_multiplication/whole_array/build");
        let tile = "32x32x32";
        let kp = NativeKernel::load(&dev, &wa, hidden, round256(hidden), tile);
        let kf1 = NativeKernel::load(&dev, &wa, hidden, round256(ff), tile);
        let kf2 = NativeKernel::load(&dev, &wa, ff, round256(hidden), tile);
        let blocks = (0..w.n_layers()).map(|i| {
            let l = &w.layers[i];
            EsmBlockNative {
                ln_attn_w: l.v("ln_attn_w"), ln_attn_b: l.v("ln_attn_b"),
                ln_ffn_w: l.v("ln_ffn_w"), ln_ffn_b: l.v("ln_ffn_b"),
                wq: kp.weight(&l.m("q_w")), wk: kp.weight(&l.m("k_w")), wv: kp.weight(&l.m("v_w")), wo: kp.weight(&l.m("attn_out_w")),
                bq: l.v("q_b"), bk: l.v("k_b"), bv: l.v("v_b"), bo: l.v("attn_out_b"),
                wf1: kf1.weight(&l.m("ffn1_w")), bf1: l.v("ffn1_b"),
                wf2: kf2.weight(&l.m("ffn2_w")), bf2: l.v("ffn2_b"),
                hidden, ff, n_heads, head_dim,
            }
        }).collect();
        EsmEncoderNative { kp, kf1, kf2, blocks, final_ln_w: w.final_ln_w.clone(), final_ln_b: w.final_ln_b.clone() }
    }
    fn block_forward(&self, b: &EsmBlockNative, x: &Array2<f32>, valid: usize) -> Array2<f32> {
        let h = b.hidden;
        let xn = layer_norm(x, &b.ln_attn_w, &b.ln_attn_b, LN_EPS);
        let mut q = self.kp.matmul(&b.wq, &xn, h, Some(&b.bq));
        let mut k = self.kp.matmul(&b.wk, &xn, h, Some(&b.bk));
        let vv = self.kp.matmul(&b.wv, &xn, h, Some(&b.bv));
        rope_heads(&mut q, b.n_heads, b.head_dim);
        rope_heads(&mut k, b.n_heads, b.head_dim);
        let ctx = mha(&q, &k, &vv, b.n_heads, b.head_dim, false, valid);
        let attn = self.kp.matmul(&b.wo, &ctx, h, Some(&b.bo));
        let x = add(x, &attn);
        let fnn = layer_norm(&x, &b.ln_ffn_w, &b.ln_ffn_b, LN_EPS);
        let ff_out = self.kf1.matmul(&b.wf1, &fnn, b.ff, Some(&b.bf1)); // [seq, ff]
        let hmid = gelu(&ff_out);
        let y = self.kf2.matmul(&b.wf2, &hmid, h, Some(&b.bf2)); // [seq, h]
        add(&x, &y)
    }
}
impl Encoder for EsmEncoderNative {
    fn forward_last(&self, x: &Array2<f32>, valid: usize) -> Array2<f32> {
        let mut x = x.clone();
        for b in &self.blocks {
            x = self.block_forward(b, &x, valid);
        }
        layer_norm(&x, &self.final_ln_w, &self.final_ln_b, LN_EPS)
    }
}
