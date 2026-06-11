//! One GigaAM-v3 Conformer block, matmul-heavy ops fused on the NPU (FFN×2, q/k/v/out,
//! pointwise1/2 weight-bound; dwconv on NPU); LayerNorm/RoPE/GLU/softmax/residual on host.
//! Faithful port of `npu_asr/fused.py` (FusedFFN + FusedBlock).

use std::path::Path;
use std::rc::Rc;

use ndarray::prelude::*;
use npu_asr_host::{bf16_round, dwconv_k5, glu, layer_norm, layer_norm_normalize, mha, rope, silu};
use npu_xrt::Device;

use crate::engines::WAEpilogue;
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

/// Macaron-half FFN as 2 weight-bound fused dispatches (LN affine folded into mm1, biases
/// K-augmented, SiLU on-chip).
struct FusedFFN {
    mm1: WAEpilogue,
    mm2: WAEpilogue,
}

impl FusedFFN {
    fn new(dev: Rc<Device>, root: &Path, w: &BlockWeights, pfx: &str, norm: &str) -> Self {
        let g = w.v(&format!("{norm}.weight"));
        let beta = w.v(&format!("{norm}.bias"));
        let w1 = w.m(&format!("{pfx}.linear1.weight")); // [768,3072]
        let b1 = w.v(&format!("{pfx}.linear1.bias")); // [3072]
        let (w1p, b1p) = fold_ln_into_mm1(&g, &beta, &w1, &b1);
        let mm1 = WAEpilogue::new(dev.clone(), root, "silu", D_MODEL, D_FF, &w1p, &b1p);
        let w2 = w.m(&format!("{pfx}.linear2.weight")); // [3072,768]
        let b2 = w.v(&format!("{pfx}.linear2.bias"));
        let mm2 = WAEpilogue::new(dev, root, "bias", D_FF, D_MODEL, &w2, &b2);
        FusedFFN { mm1, mm2 }
    }

    fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let norm = layer_norm_normalize(x, LN_EPS); // affine folded into mm1
        self.mm2.forward(&self.mm1.forward(&norm))
    }
}

pub struct FusedBlock {
    ffn1: FusedFFN,
    ffn2: FusedFFN,
    q: WAEpilogue,
    k: WAEpilogue,
    v: WAEpilogue,
    o: WAEpilogue,
    pw1: WAEpilogue,
    pw2: WAEpilogue,
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
    pub fn new(
        dev: Rc<Device>,
        root: &Path,
        w: &BlockWeights,
        cos: &Array2<f32>,
        sin: &Array2<f32>,
    ) -> Self {
        let ffn1 = FusedFFN::new(dev.clone(), root, w, "feed_forward1", "norm_feed_forward1");
        let ffn2 = FusedFFN::new(dev.clone(), root, w, "feed_forward2", "norm_feed_forward2");
        let proj = |key: &str| {
            WAEpilogue::new(
                dev.clone(),
                root,
                "bias",
                768,
                768,
                &w.m(&format!("self_attn.{key}.weight")),
                &w.v(&format!("self_attn.{key}.bias")),
            )
        };
        let q = proj("linear_q");
        let k = proj("linear_k");
        let v = proj("linear_v");
        let o = proj("linear_out");

        // pointwise convs: [out,in,1] -> [in,out] (squeeze k, transpose)
        let pw1w = w.m3("conv.pointwise_conv1.weight"); // [1536,768,1]
        let pw1_bt = squeeze_t(&pw1w); // [768,1536]
        let pw1 = WAEpilogue::new(
            dev.clone(),
            root,
            "bias",
            768,
            1536,
            &pw1_bt,
            &w.v("conv.pointwise_conv1.bias"),
        );
        let pw2w = w.m3("conv.pointwise_conv2.weight"); // [768,768,1]
        let pw2_bt = squeeze_t(&pw2w);
        let pw2 = WAEpilogue::new(
            dev.clone(),
            root,
            "bias",
            768,
            768,
            &pw2_bt,
            &w.v("conv.pointwise_conv2.bias"),
        );

        let _ = &dev; // dwconv now runs on host (cheap 5-tap FIR, parallelized)
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
            q,
            k,
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
        let mut x = x.mapv(bf16_round);

        // --- FFN1 (macaron half), residual ×0.5 ---
        let f1 = self.ffn1.forward(&x);
        x = (&x + &f1.mapv(|v| 0.5 * v)).mapv(bf16_round);

        // --- MHSA ---
        let ln = layer_norm(&x, &self.ln_satt_w, &self.ln_satt_b, LN_EPS);
        let r = rope(&ln, &self.cos, &self.sin, N_HEADS, HEAD_DIM);
        let qp = self.q.forward(&r);
        let kp = self.k.forward(&r);
        let vp = self.v.forward(&ln);
        let ctx = mha(&qp, &kp, &vp, N_HEADS, HEAD_DIM, true, valid_len);
        let op = self.o.forward(&ctx);
        x = (&x + &op).mapv(bf16_round);

        // --- ConvModule ---
        let ln = layer_norm(&x, &self.ln_conv_w, &self.ln_conv_b, LN_EPS);
        let pw1 = self.pw1.forward(&ln); // [T,1536]
        let pw1t = pw1.t().to_owned(); // [1536,T]
        let mut g = glu(&pw1t); // [768,T]
        // zero padded time columns so the depthwise FIR (k=5) doesn't pull padding into valid frames
        let tt_g = g.ncols();
        if valid_len < tt_g {
            for c in 0..g.nrows() {
                for ti in valid_len..tt_g {
                    g[[c, ti]] = 0.0;
                }
            }
        }
        let mut dwout = dwconv_k5(&g, &self.dw_taps); // [768,T] on host (parallel 5-tap FIR)
        // + depthwise bias (broadcast over T)
        let (ch, tt) = dwout.dim();
        for c in 0..ch {
            let bc = self.dw_bias[c];
            for ti in 0..tt {
                dwout[[c, ti]] += bc;
            }
        }
        let bn = layer_norm(&dwout.t().to_owned(), &self.bn_w, &self.bn_b, LN_EPS); // [T,768]
        let sw = silu(&bn.t().to_owned()); // [768,T]
        let pw2 = self.pw2.forward(&sw.t().to_owned()); // [T,768]
        x = (&x + &pw2).mapv(bf16_round);

        // --- FFN2 (macaron half), residual ×0.5 ---
        let f2 = self.ffn2.forward(&x);
        x = (&x + &f2.mapv(|v| 0.5 * v)).mapv(bf16_round);

        // --- final LayerNorm ---
        layer_norm(&x, &self.ln_out_w, &self.ln_out_b, LN_EPS).mapv(bf16_round)
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
