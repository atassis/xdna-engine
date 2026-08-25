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

/// K=768 resident-FFN rail handle injected into a BertBlock (BERT_RESIDENT_FFN=1). Holds the
/// K=768-configured `NpuMatmul` (fc1 `modalgelu` -> cast -> fc2 `modalid` -> resadd_s100) shared by
/// every block, plus this block's own fc1/fc2 weights + biases. None on the shipped host path.
struct ResidentFfn {
    npu: Rc<NpuMatmul>,
    w1: Array2<f32>, b1: Vec<f32>, // fc1 [768,3072] + bias
    w2: Array2<f32>, b2: Vec<f32>, // fc2 [3072,768] + bias
    // Weight-cache keys. PER LAYER: the rail caches packed weight BOs by id across the whole
    // NpuMatmul, so a key shared between blocks would serve layer 0's W1/W2 to all 12.
    id1: String, id2: String,
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
    fn new(
        shared: Rc<SharedCtxA>, l: &crate::bert::weights::BertLayer,
        n_heads: usize, head_dim: usize, resident: Option<ResidentFfn>,
    ) -> Self {
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
            resident, // None => host FFN (the shipped default)
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
    /// Goes through the HOST-IN entry, not `resident_ffn_nonorm(x_bo, ..)`: `x` is already a host
    /// array here (the post-attention LN runs on host), and the rail packs fc1's K-augmented A and
    /// the residual operand on the host anyway, so uploading `x` first would only buy a readback.
    /// The weight closures clone lazily -- they fire once per id, on the rail's cache miss.
    ///
    /// `self.resident` is None on the CPU-only base, so this returns None there. Wiring a rail means
    /// building a K=768-configured `NpuMatmul` and setting `resident = Some(ResidentFfn{..})`.
    fn try_resident_ffn(&self, x: &Array2<f32>) -> Option<Array2<f32>> {
        if std::env::var("BERT_RESIDENT_FFN").as_deref() != Ok("1") {
            return None; // flag unset -> host FFN (default)
        }
        let r = self.resident.as_ref()?; // no rail wired (CPU-only base) -> host fallback
        if r.npu.resident_kres() != 768 {
            return None; // capability gate: wrong hidden dim -> host fallback
        }
        if x.nrows() > r.npu.resident_pad_m() {
            return None; // sequence longer than the built width -> host, not the rail's assert
        }
        let y_bo = r.npu.resident_ffn_nonorm_hostx(
            x,
            || r.w1.clone(), &r.b1, &r.id1,
            || r.w2.clone(), &r.b2, &r.id2,
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

/// Rail widths the K=768 GELU bricks are built at (`scripts/build_k768_gelu_rail.sh` PAD_M). The
/// rail pads every dispatch to one fixed width, so the encoder takes the narrowest that still covers
/// `max_seq` -- a shorter sequence costs padding rows, a longer one would trip the rail's assert.
const K768_BUILT_WIDTHS: [usize; 4] = [256, 512, 1024, 1536];

impl BertEncoder {
    pub fn new(
        dev: Rc<Device>, root: &Path, weights: &BertWeights,
        n_heads: usize, head_dim: usize, max_seq: usize,
    ) -> Self {
        let cfg = crate::tuning_profile::resolve(root, npu_asr::ctx2::Precision::from_env());
        let shared = SharedCtxA::with_tuning(&dev, root, &cfg);
        let rail = Self::open_rail(root, max_seq);
        let blocks = (0..weights.n_layers())
            .map(|i| {
                let l = &weights.layers[i];
                let resident = rail.as_ref().map(|npu| ResidentFfn {
                    npu: npu.clone(),
                    w1: l.m("ffn1_w"), b1: l.v("ffn1_b"),
                    w2: l.m("ffn2_w"), b2: l.v("ffn2_b"),
                    id1: format!("bert.l{i}.ffn1.w1"),
                    id2: format!("bert.l{i}.ffn2.w2"),
                });
                BertBlock::new(shared.clone(), l, n_heads, head_dim, resident)
            })
            .collect();
        BertEncoder { blocks }
    }

    /// `(calls, dispatches)` the shared K=768 rail has recorded so far, or None when no rail is
    /// wired. Every fallback in `try_resident_ffn` is SILENT, so a parity gate needs this to tell
    /// "the embeddings match because the rail is correct" from "they match because it never ran".
    pub fn resident_ffn_stats(&self) -> Option<(usize, usize)> {
        let r = self.blocks.first()?.resident.as_ref()?;
        let s = r.npu.stats.borrow();
        Some((s.calls, s.dispatches))
    }

    /// The K=768 GELU rail shared by every block, or None to leave the whole encoder on the host
    /// FFN. Opt-in via `BERT_RESIDENT_FFN=1`, so the default path opens no second device handle and
    /// loads no bricks. Absence of the artifacts is a fallback, not an error: `open_with_rail`
    /// panics on a missing resident xclbin, so the built-check has to come first.
    fn open_rail(root: &Path, max_seq: usize) -> Option<Rc<NpuMatmul>> {
        if std::env::var("BERT_RESIDENT_FFN").as_deref() != Ok("1") {
            return None;
        }
        let pad_m = *K768_BUILT_WIDTHS.iter().find(|&&w| w >= max_seq)?;
        if !NpuMatmul::k768_rail_built(root, 768, pad_m, 3072) {
            eprintln!(
                "[bert] BERT_RESIDENT_FFN=1 but the K=768 rail is not built at PAD_M={pad_m} \
                 -- staying on the host FFN (build it with scripts/build_k768_gelu_rail.sh)"
            );
            return None;
        }
        Some(Rc::new(NpuMatmul::open_with_rail(root, 768, pad_m, 3072)))
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
