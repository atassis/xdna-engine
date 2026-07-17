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

// --- K=768 GELU resident-rail seam (prep, not yet active) --------------------------------------
// BERT-base (hidden=768, intermediate=3072, GELU) shares its FFN shape with Whisper-small and the
// ESM-2 ctx2 zero-pad convention, so ONE (K=768, DFF<=3072, GELU, residual-scale=1.0) resident rail
// serves three of the four shipped encoders (scout: generalization.md). The rail ENGINE lives in
// npu-parakeet (npu.rs), but is baked to Parakeet's KRES=1024/PAD_M=512 -- every brick asserts KRES,
// so it panics on BERT's K=768 until those consts become rail parameters. The K=768 GELU/identity
// xclbins are also not built yet (recipe: scripts/build_k768_gelu_rail.sh; GELU epilogue is TANH-approx
// bf16, so BERT's exact-GELU parity must be gated on rel-L2 vs host truth, NOT the chaotic WER).
//
// Schedule wiring for the device session (attention stays HOST-routed this step -- a legitimate
// front-to-back partial advance, FFN frontier first, exactly like Parakeet's FFN->conv->MHSA order):
//   post-norm FFN sublayer  x -> cast(x)->bf16 -> fc1_gelu(K_aug=800,N=3072) -> cast->bf16
//                             -> fc2_collapse(K=3072,N=768) + host b2 -> resadd_s100(x, y) -> LN
//   (BERT is POST-norm: the trailing LN moves AFTER the residual add, vs Parakeet's pre-norm order.)
// The device build + rel-L2 gate for this rail is a staged task (build_k768_gelu_rail.sh).
impl BertEncoder {
    /// Opt-in flag (reserved) for the K=768 GELU resident FFN rail. Returns false unless
    /// `BERT_RESIDENT_FFN=1`. The rail itself is not yet built/genericized (see the seam note
    /// above), so a set flag currently only emits a one-time notice and keeps the host FFN path.
    fn resident_ffn_requested() -> bool {
        std::env::var("BERT_RESIDENT_FFN").map(|v| v == "1").unwrap_or(false)
    }
}

impl BertEncoder {
    pub fn new(dev: Rc<Device>, root: &Path, weights: &BertWeights, n_heads: usize, head_dim: usize) -> Self {
        let cfg = crate::tuning_profile::resolve(root, npu_asr::ctx2::Precision::from_env());
        let shared = SharedCtxA::with_tuning(&dev, root, &cfg);
        let blocks = (0..weights.n_layers())
            .map(|i| BertBlock::new(shared.clone(), &weights.layers[i], n_heads, head_dim))
            .collect();
        BertEncoder { blocks }
    }
}

impl Encoder for BertEncoder {
    fn forward_last(&self, x: &Array2<f32>, valid_len: usize) -> Array2<f32> {
        if Self::resident_ffn_requested() {
            // K=768 GELU resident FFN rail is prep-staged (build recipe + turnkey doc) but not yet
            // wired: npu-parakeet's rail is KRES=1024-locked and the K=768 xclbins are unbuilt.
            // Fall through to the host path (gelu on host) until the device session lands it.
            eprintln!(
                "[bert] BERT_RESIDENT_FFN=1 set, but the K=768 GELU resident rail is not built/genericized yet; \
                 using host FFN (see handoffs/active/2026-07-17-k768-gelu-rail-device.md)"
            );
        }
        let mut x = x.clone();
        for b in &self.blocks {
            x = b.forward(&x, valid_len);
        }
        let _ = s![..]; // keep ndarray::s import used if slicing added later
        x
    }
}
