//! BERT encoder on the NPU: post-norm Transformer layers reusing the npu_asr matmul engines.

use std::path::Path;
use std::rc::Rc;

use ndarray::{s, Array2};
use npu_asr::ctx2::{CtxAOp, Epi, FfnMm2, SharedCtxA};
use npu_asr_host::{gelu, layer_norm, mha};
use npu_parakeet::npu::{Act, NpuMatmul};
use npu_xrt::Device;

use crate::bert::weights::BertWeights;
use crate::pipeline::Encoder;

const LN_EPS: f32 = 1e-12;

/// K=768 resident-FFN rail handle injected into a BertBlock (BERT_RESIDENT_FFN=1). Holds a
/// K=768-configured `NpuMatmul` (fc1 `modalgelu` -> cast -> fc2 `modalid` -> resadd_s100) plus the
/// raw fc1/fc2 weights + biases the rail dispatches. None on the shipped host path; a DEVICE session
/// builds the K=768 xclbins + a K=768 `NpuMatmul` and sets this to `Some(..)` to light the rail up
/// (see `NpuMatmul::resident_ffn_nonorm` DEVICE-GATED notes). Never constructed on the CPU-only base.
#[allow(dead_code)]
struct ResidentFfn {
    npu: Rc<NpuMatmul>,
    w1: Array2<f32>, b1: Vec<f32>, // fc1 [768,3072] + bias
    w2: Array2<f32>, b2: Vec<f32>, // fc2 [3072,768] + bias
}

struct BertBlock {
    q: CtxAOp, k: CtxAOp, v: CtxAOp, o: CtxAOp,
    ffn1: CtxAOp,     // [768->3072] + bias; gelu applied on host
    ffn2: FfnMm2,     // [3072->768] + bias2
    attn_ln_w: Vec<f32>, attn_ln_b: Vec<f32>,
    out_ln_w: Vec<f32>,  out_ln_b: Vec<f32>,
    n_heads: usize, head_dim: usize,
    // K=768 resident FFN rail (BERT_RESIDENT_FFN=1). None => host FFN (the shipped default).
    resident: Option<ResidentFfn>,
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
            resident: None, // host FFN by default; a DEVICE session injects the K=768 rail
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
        // --- FFN (post-norm). Default = host GELU. BERT_RESIDENT_FFN=1 opts into the K=768 resident
        // GELU rail (cast -> fc1 modalgelu -> cast -> fc2 modalid -> resadd_s100), which returns the
        // residual result on-device so only the trailing post-norm LN stays host. ---
        let x = match self.try_resident_ffn(&x) {
            // resident path added the residual on-device: post = x + fc2(gelu(fc1(x))).
            Some(post) => post,
            None => {
                // HOST fallback (default, and whenever the rail is unavailable / gates out).
                let h = gelu(&self.ffn1.forward(&x)); // [seq,3072]
                let y = self.ffn2.forward(&h);        // [seq,768]
                add(&x, &y)                            // scale=1.0 full residual
            }
        };
        layer_norm(&x, &self.out_ln_w, &self.out_ln_b, LN_EPS) // POST-norm LN after the residual
    }

    /// K=768 resident FFN sublayer, gated behind `BERT_RESIDENT_FFN=1` AND a wired `resident` rail.
    /// Returns `x + fc2(gelu(fc1(x)))` (residual added on-device); None -> the caller runs the host
    /// FFN (the shipped default). None whenever the env flag is unset, no rail is injected, or the
    /// rail's capability dim isn't 768.
    ///
    /// DEVICE-GATED: `self.resident` is None on the CPU-only base, so this returns None today. A
    /// device session (a) builds the K=768 xclbins + a K=768-configured `NpuMatmul`, (b) sets
    /// `resident = Some(ResidentFfn{..})` per block, then this dispatches the rail. The `upload_stream`
    /// / `readback_stream` below are KRES-shaped on the current NpuMatmul (D=1024); a K=768 rail needs
    /// their [T,768] variants (same const->field parameterization as `resident_ffn_nonorm`).
    fn try_resident_ffn(&self, x: &Array2<f32>) -> Option<Array2<f32>> {
        if std::env::var("BERT_RESIDENT_FFN").as_deref() != Ok("1") {
            return None; // flag unset -> host FFN (default)
        }
        let r = self.resident.as_ref()?; // no rail wired (CPU-only base) -> host fallback
        if r.npu.resident_kres() != 768 {
            return None; // capability gate: wrong hidden dim -> host fallback
        }
        // DEVICE-GATED: KRES-shaped upload/readback (D=1024); K=768 rail needs [T,768] variants.
        let x_bo = r.npu.upload_stream(x);
        let (w1, w2) = (r.w1.clone(), r.w2.clone()); // DEVICE-GATED: prefer lazy/packed BOs over a per-call clone
        let y_bo = r.npu.resident_ffn_nonorm(
            &x_bo, x.nrows(),
            move || w1, &r.b1, "bert.ffn1.w1",
            move || w2, &r.b2, "bert.ffn2.w2",
            Act::Gelu,
        )?;
        Some(r.npu.readback_stream(&y_bo, x.nrows()))
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
        let mut x = x.clone();
        for b in &self.blocks {
            x = b.forward(&x, valid_len);
        }
        let _ = s![..]; // keep ndarray::s import used if slicing added later
        x
    }
}
