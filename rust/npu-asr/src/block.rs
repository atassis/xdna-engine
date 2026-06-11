//! One GigaAM-v3 Conformer block, matmul-heavy ops fused on the NPU (FFN×2, q/k/v/out,
//! pointwise1/2 weight-bound; dwconv on NPU); LayerNorm/RoPE/GLU/softmax/residual on host.
//! Faithful port of `npu_asr/fused.py` (FusedFFN + FusedBlock).

use std::path::Path;
use std::rc::Rc;

use ndarray::prelude::*;
use npu_asr_host::prof;
use npu_asr_host::{
    bf16_round, dwconv_k5, glu, layer_norm, layer_norm_normalize, mha, residual_add_round, rope,
    silu,
};
use npu_xrt::Device;

#[cfg(all(feature = "chained_ffn", not(feature = "two_ctx")))]
use crate::engines::ChainedFFN;
#[cfg(not(feature = "two_ctx"))]
use crate::engines::WAEpilogue;
#[cfg(feature = "two_ctx")]
use crate::ctx2::{CtxAOp, Epi, FfnMm2, SharedCtxA};
use crate::weights::BlockWeights;

const D_MODEL: usize = 768;
const D_FF: usize = 3072;
const N_HEADS: usize = 16;
const HEAD_DIM: usize = 48;
const LN_EPS: f32 = 1e-5;

/// Fold a LayerNorm affine (scale g, shift beta) into the following matmul (so the normalize is
/// affine-free): (norm*g+beta)@W1 + b1 == norm@W1' + b1'.
fn fold_ln_into_mm1(g: &[f32], beta: &[f32], w1: &Array2<f32>, b1: &[f32]) -> (Array2<f32>, Vec<f32>) {
    let (k, n) = w1.dim();
    let mut w1p = w1.clone();
    for kk in 0..k {
        let gk = g[kk];
        for nn in 0..n {
            w1p[[kk, nn]] *= gk;
        }
    }
    let mut b1p = b1.to_vec();
    for nn in 0..n {
        let mut s = 0f32;
        for kk in 0..k {
            s += beta[kk] * w1[[kk, nn]];
        }
        b1p[nn] += s;
    }
    (w1p, b1p)
}

/// Macaron-half FFN. By default (`chained_ffn` feature) the 3072-wide intermediate H is kept in a
/// device BO across the two dispatches (no host round-trip). Without the feature it falls back to
/// the original two independent weight-bound dispatches with a host hop between them. Either way:
/// LN affine folded into mm1, bias1 K-augmented, SiLU on-chip; bias2 added after mm2.
#[cfg(feature = "two_ctx")]
struct FusedFFN {
    mm1: CtxAOp, // ctxA plain matmul N=3072; host epilogue = SiLU(z + b1) (Epi::SiluBias)
    mm2: FfnMm2, // SAME ctxA, K=3072 split into 4× N=768 partials; host-accumulate + bias2
}
#[cfg(all(not(feature = "two_ctx"), feature = "chained_ffn"))]
struct FusedFFN {
    ffn: ChainedFFN,
}
#[cfg(all(not(feature = "two_ctx"), not(feature = "chained_ffn")))]
struct FusedFFN {
    mm1: WAEpilogue,
    mm2: WAEpilogue,
}

impl FusedFFN {
    #[cfg(not(feature = "two_ctx"))]
    fn new(dev: Rc<Device>, root: &Path, w: &BlockWeights, pfx: &str, norm: &str) -> Self {
        let g = w.v(&format!("{norm}.weight"));
        let beta = w.v(&format!("{norm}.bias"));
        let w1 = w.m(&format!("{pfx}.linear1.weight")); // [768,3072]
        let b1 = w.v(&format!("{pfx}.linear1.bias")); // [3072]
        let (w1p, b1p) = fold_ln_into_mm1(&g, &beta, &w1, &b1);
        let w2 = w.m(&format!("{pfx}.linear2.weight")); // [3072,768]
        let b2 = w.v(&format!("{pfx}.linear2.bias"));
        #[cfg(feature = "chained_ffn")]
        {
            let ffn = ChainedFFN::new(dev, root, D_MODEL, D_FF, D_MODEL, &w1p, &b1p, &w2, &b2);
            FusedFFN { ffn }
        }
        #[cfg(not(feature = "chained_ffn"))]
        {
            let mm1 = WAEpilogue::new(dev.clone(), root, "silu", D_MODEL, D_FF, &w1p, &b1p);
            let mm2 = WAEpilogue::new(dev, root, "bias", D_FF, D_MODEL, &w2, &b2);
            FusedFFN { mm1, mm2 }
        }
    }

    /// two_ctx FFN: mm1 on shared ctxA (plain, N=3072), mm2 on shared ctxB (N=768). The LN affine is
    /// folded into W1 exactly as before; bias1 is added on host then SiLU; bias2 added on host.
    #[cfg(feature = "two_ctx")]
    fn new(w: &BlockWeights, pfx: &str, norm: &str, ctx_a: Rc<SharedCtxA>) -> Self {
        let g = w.v(&format!("{norm}.weight"));
        let beta = w.v(&format!("{norm}.bias"));
        let w1 = w.m(&format!("{pfx}.linear1.weight")); // [768,3072]
        let b1 = w.v(&format!("{pfx}.linear1.bias")); // [3072]
        let (w1p, b1p) = fold_ln_into_mm1(&g, &beta, &w1, &b1);
        let w2 = w.m(&format!("{pfx}.linear2.weight")); // [3072,768]
        let b2 = w.v(&format!("{pfx}.linear2.bias"));
        // mm1 epilogue is SiLU(z + b1): the SiluBias epilogue adds b1 then applies SiLU on host,
        // matching the old K-augmented `_silu` xclbin (bias rode an extra k-block, added pre-SiLU).
        let mm1 = CtxAOp::new(ctx_a.clone(), &w1p, D_FF, Epi::SiluBias, &b1p);
        // mm2 on the SAME ctxA (K=3072 split into 4× N=768) -> one resident xclbin, zero switches.
        let mm2 = FfnMm2::new(ctx_a, &w2, &b2);
        FusedFFN { mm1, mm2 }
    }

    #[cfg(not(feature = "two_ctx"))]
    fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let norm = prof::time("layer_norm", || layer_norm_normalize(x, LN_EPS)); // affine folded into mm1
        #[cfg(feature = "chained_ffn")]
        {
            self.ffn.forward(&norm)
        }
        #[cfg(not(feature = "chained_ffn"))]
        {
            self.mm2.forward(&self.mm1.forward(&norm))
        }
    }

    #[cfg(feature = "two_ctx")]
    fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let norm = prof::time("layer_norm", || layer_norm_normalize(x, LN_EPS)); // affine folded into mm1
        let h = self.mm1.forward(&norm); // [Mp,3072] = SiLU(norm@W1' + b1) (host epilogue)
        self.mm2.forward(&h) // [Mp,768] = h@W2 + b2 (host bias2)
    }
}

/// The per-block K=768 projection op type. Default: `WAEpilogue` (its own `_bias` xclbin per shape).
/// With `two_ctx`: `CtxAOp`, all sharing one `SharedCtxA` kernel (no per-op hw-context switch).
#[cfg(not(feature = "two_ctx"))]
type ProjOp = WAEpilogue;
#[cfg(feature = "two_ctx")]
type ProjOp = CtxAOp;

pub struct FusedBlock {
    ffn1: FusedFFN,
    ffn2: FusedFFN,
    qk: ProjOp, // q and k share the `rope` input -> one [768,1536] dispatch, split after
    v: ProjOp,
    o: ProjOp,
    pw1: ProjOp,
    pw2: ProjOp,
    // host LayerNorm affines
    ln_satt_w: Vec<f32>,
    ln_satt_b: Vec<f32>,
    ln_conv_w: Vec<f32>,
    ln_conv_b: Vec<f32>,
    bn_w: Vec<f32>,
    bn_b: Vec<f32>,
    ln_out_w: Vec<f32>,
    ln_out_b: Vec<f32>,
    dw_taps: Array2<f32>, // [768,5]
    dw_bias: Vec<f32>,
    cos: Array2<f32>,
    sin: Array2<f32>,
}

impl FusedBlock {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dev: Rc<Device>,
        root: &Path,
        w: &BlockWeights,
        cos: &Array2<f32>,
        sin: &Array2<f32>,
        #[cfg(feature = "two_ctx")] ctx_a: Rc<SharedCtxA>,
    ) -> Self {
        #[cfg(not(feature = "two_ctx"))]
        let ffn1 = FusedFFN::new(dev.clone(), root, w, "feed_forward1", "norm_feed_forward1");
        #[cfg(not(feature = "two_ctx"))]
        let ffn2 = FusedFFN::new(dev.clone(), root, w, "feed_forward2", "norm_feed_forward2");
        #[cfg(feature = "two_ctx")]
        let ffn1 = FusedFFN::new(w, "feed_forward1", "norm_feed_forward1", ctx_a.clone());
        #[cfg(feature = "two_ctx")]
        let ffn2 = FusedFFN::new(w, "feed_forward2", "norm_feed_forward2", ctx_a.clone());

        // Build a K=768 `_bias` projection op. Default -> its own `_bias` WAEpilogue (per-shape
        // xclbin). two_ctx -> a CtxAOp on the shared ctxA kernel (bias added on host).
        #[cfg(not(feature = "two_ctx"))]
        let mk_bias = |wmat: &Array2<f32>, n: usize, bias: &[f32]| {
            WAEpilogue::new(dev.clone(), root, "bias", 768, n, wmat, bias)
        };
        #[cfg(feature = "two_ctx")]
        let mk_bias = |wmat: &Array2<f32>, n: usize, bias: &[f32]| {
            CtxAOp::new(ctx_a.clone(), wmat, n, Epi::Bias, bias)
        };

        let proj = |key: &str| {
            mk_bias(
                &w.m(&format!("self_attn.{key}.weight")),
                768,
                &w.v(&format!("self_attn.{key}.bias")),
            )
        };
        // q and k both project the RoPE'd input -> stack their weights [768, 1536] and do one
        // dispatch, then split the output.
        let wq = w.m("self_attn.linear_q.weight");
        let wk = w.m("self_attn.linear_k.weight");
        let wqk = ndarray::concatenate(Axis(1), &[wq.view(), wk.view()]).unwrap(); // [768,1536]
        let mut bqk = w.v("self_attn.linear_q.bias");
        bqk.extend(w.v("self_attn.linear_k.bias"));
        let qk = mk_bias(&wqk, 1536, &bqk);
        let v = proj("linear_v");
        let o = proj("linear_out");

        // pointwise convs: [out,in,1] -> [in,out] (squeeze k, transpose)
        let pw1w = w.m3("conv.pointwise_conv1.weight"); // [1536,768,1]
        let pw1_bt = squeeze_t(&pw1w); // [768,1536]
        let pw1 = mk_bias(&pw1_bt, 1536, &w.v("conv.pointwise_conv1.bias"));
        let pw2w = w.m3("conv.pointwise_conv2.weight"); // [768,768,1]
        let pw2_bt = squeeze_t(&pw2w);
        let pw2 = mk_bias(&pw2_bt, 768, &w.v("conv.pointwise_conv2.bias"));

        let _ = &dev; // dwconv now runs on host (cheap 5-tap FIR, parallelized)
        #[cfg(feature = "two_ctx")]
        let _ = root; // two_ctx ops live on the pre-loaded shared contexts, no per-block xclbin load
        // depthwise taps [768,1,5] -> [768,5]
        let dww = w.m3("conv.depthwise_conv.weight");
        let (ch, _one, kk) = dww.dim();
        let mut dw_taps = Array2::<f32>::zeros((ch, kk));
        for c in 0..ch {
            for ki in 0..kk {
                dw_taps[[c, ki]] = dww[[c, 0, ki]];
            }
        }

        FusedBlock {
            ffn1,
            ffn2,
            qk,
            v,
            o,
            pw1,
            pw2,
            ln_satt_w: w.v("norm_self_att.weight"),
            ln_satt_b: w.v("norm_self_att.bias"),
            ln_conv_w: w.v("norm_conv.weight"),
            ln_conv_b: w.v("norm_conv.bias"),
            bn_w: w.v("conv.batch_norm.weight"),
            bn_b: w.v("conv.batch_norm.bias"),
            ln_out_w: w.v("norm_out.weight"),
            ln_out_b: w.v("norm_out.bias"),
            dw_taps,
            dw_bias: w.v("conv.depthwise_conv.bias"),
            cos: cos.clone(),
            sin: sin.clone(),
        }
    }

    /// x is [T, 768] (bf16-valued f32). `valid_len` is the number of non-padded time frames;
    /// the two time-mixing ops (attention, depthwise conv) mask positions >= valid_len so padding
    /// doesn't leak into valid frames (pass valid_len >= T for no masking). Returns [T, 768].
    pub fn forward(&self, x: &Array2<f32>, valid_len: usize) -> Array2<f32> {
        let mut x = prof::time("bf16_round", || x.mapv(bf16_round));

        // --- FFN1 (macaron half), residual ×0.5 ---
        let f1 = self.ffn1.forward(&x);
        x = prof::time("residual+bf16", || residual_add_round(&x, &f1, 0.5));

        // --- MHSA ---
        let ln = prof::time("layer_norm", || {
            layer_norm(&x, &self.ln_satt_w, &self.ln_satt_b, LN_EPS)
        });
        let r = prof::time("rope", || {
            rope(&ln, &self.cos, &self.sin, N_HEADS, HEAD_DIM)
        });
        let qk = self.qk.forward(&r); // [T,1536]
        let qp = qk.slice(s![.., ..D_MODEL]).to_owned();
        let kp = qk.slice(s![.., D_MODEL..]).to_owned();
        let vp = self.v.forward(&ln);
        let ctx = prof::time("mha", || {
            mha(&qp, &kp, &vp, N_HEADS, HEAD_DIM, true, valid_len)
        });
        let op = self.o.forward(&ctx);
        x = prof::time("residual+bf16", || residual_add_round(&x, &op, 1.0));

        // --- ConvModule ---
        let ln = prof::time("layer_norm", || {
            layer_norm(&x, &self.ln_conv_w, &self.ln_conv_b, LN_EPS)
        });
        let pw1 = self.pw1.forward(&ln); // [T,1536]
        let mut g = prof::time("glu", || {
            let pw1t = pw1.t().to_owned(); // [1536,T]
            glu(&pw1t) // [768,T]
        });
        // zero padded time columns so the depthwise FIR (k=5) doesn't pull padding into valid frames
        let tt_g = g.ncols();
        if valid_len < tt_g {
            for c in 0..g.nrows() {
                for ti in valid_len..tt_g {
                    g[[c, ti]] = 0.0;
                }
            }
        }
        let dwout = prof::time("dwconv", || {
            let mut dwout = dwconv_k5(&g, &self.dw_taps); // [768,T] on host (parallel 5-tap FIR)
            // + depthwise bias (broadcast over T)
            let (ch, tt) = dwout.dim();
            for c in 0..ch {
                let bc = self.dw_bias[c];
                for ti in 0..tt {
                    dwout[[c, ti]] += bc;
                }
            }
            dwout
        });
        let bn = prof::time("layer_norm", || {
            layer_norm(&dwout.t().to_owned(), &self.bn_w, &self.bn_b, LN_EPS) // [T,768]
        });
        let sw = prof::time("silu", || silu(&bn.t().to_owned())); // [768,T]
        let pw2 = self.pw2.forward(&sw.t().to_owned()); // [T,768]
        x = prof::time("residual+bf16", || residual_add_round(&x, &pw2, 1.0));

        // --- FFN2 (macaron half), residual ×0.5 ---
        let f2 = self.ffn2.forward(&x);
        x = prof::time("residual+bf16", || residual_add_round(&x, &f2, 0.5));

        // --- final LayerNorm ---
        prof::time("layer_norm", || {
            layer_norm(&x, &self.ln_out_w, &self.ln_out_b, LN_EPS).mapv(bf16_round)
        })
    }
}

/// [out, in, 1] -> [in, out] (squeeze the trailing k=1, transpose to B-operand x@W form).
fn squeeze_t(w: &Array3<f32>) -> Array2<f32> {
    let (out, inp, _one) = w.dim();
    let mut m = Array2::<f32>::zeros((inp, out));
    for o in 0..out {
        for i in 0..inp {
            m[[i, o]] = w[[o, i, 0]];
        }
    }
    m
}
