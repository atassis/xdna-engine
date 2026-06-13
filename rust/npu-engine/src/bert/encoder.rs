//! BERT encoder on the NPU: post-norm Transformer layers reusing the npu_asr matmul engines.

use std::path::Path;
use std::rc::Rc;

use ndarray::{s, Array2};
use npu_asr::ctx2::{CtxAOp, Epi, FfnMm2, SharedCtxA};
use npu_asr_host::{gelu, layer_norm, mha};
use npu_xrt::Device;

use crate::bert::weights::BertWeights;
use crate::pipeline::Encoder;

const LN_EPS: f32 = 1e-12;

struct BertBlock {
    q: CtxAOp, k: CtxAOp, v: CtxAOp, o: CtxAOp,
    ffn1: CtxAOp,     // [768->3072] + bias; gelu applied on host
    ffn2: FfnMm2,     // [3072->768] + bias2
    attn_ln_w: Vec<f32>, attn_ln_b: Vec<f32>,
    out_ln_w: Vec<f32>,  out_ln_b: Vec<f32>,
    n_heads: usize, head_dim: usize,
}

impl BertBlock {
    fn new(shared: Rc<SharedCtxA>, l: &crate::bert::weights::BertLayer, n_heads: usize, head_dim: usize) -> Self {
        let mk = |wk: &str, bk: &str, n: usize| {
            CtxAOp::new(shared.clone(), &l.m(wk), n, Epi::Bias, &l.v(bk))
        };
        BertBlock {
            q: mk("q_w", "q_b", 768),
            k: mk("k_w", "k_b", 768),
            v: mk("v_w", "v_b", 768),
            o: mk("attn_out_w", "attn_out_b", 768),
            ffn1: mk("ffn1_w", "ffn1_b", 3072),
            ffn2: FfnMm2::new(shared, &l.m("ffn2_w"), &l.v("ffn2_b")),
            attn_ln_w: l.v("attn_ln_w"), attn_ln_b: l.v("attn_ln_b"),
            out_ln_w: l.v("out_ln_w"),  out_ln_b: l.v("out_ln_b"),
            n_heads, head_dim,
        }
    }

    fn forward(&self, x: &Array2<f32>, valid_len: usize) -> Array2<f32> {
        // --- self-attention (post-norm) ---
        let q = self.q.forward(x);
        let k = self.k.forward(x);
        let v = self.v.forward(x);
        let ctx = mha(&q, &k, &v, self.n_heads, self.head_dim, false, valid_len);
        let attn = self.o.forward(&ctx);
        let x = add(x, &attn);
        let x = layer_norm(&x, &self.attn_ln_w, &self.attn_ln_b, LN_EPS);
        // --- FFN (post-norm), GELU on host ---
        let h = gelu(&self.ffn1.forward(&x)); // [seq,3072]
        let y = self.ffn2.forward(&h);        // [seq,768]
        let x = add(&x, &y);
        layer_norm(&x, &self.out_ln_w, &self.out_ln_b, LN_EPS)
    }
}

fn add(a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
    let mut o = a.clone();
    o.zip_mut_with(b, |x, &y| *x += y);
    o
}

pub struct BertEncoder {
    blocks: Vec<BertBlock>,
}

impl BertEncoder {
    pub fn new(dev: Rc<Device>, root: &Path, weights: &BertWeights, n_heads: usize, head_dim: usize) -> Self {
        let shared = SharedCtxA::new(&dev, root);
        let blocks = (0..weights.n_layers())
            .map(|i| BertBlock::new(shared.clone(), &weights.layers[i], n_heads, head_dim))
            .collect();
        BertEncoder { blocks }
    }
}

impl Encoder for BertEncoder {
    fn forward_last(&self, x: &Array2<f32>, valid_len: usize) -> Array2<f32> {
        let mut x = x.clone();
        for b in &self.blocks {
            x = b.forward(&x, valid_len);
        }
        let _ = s![..]; // keep ndarray::s import used if slicing added later
        x
    }
}
