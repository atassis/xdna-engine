//! One GigaAM-v3 Conformer block, matmul-heavy ops fused on the NPU (FFN×2, q/k/v/out,
//! pointwise1/2 weight-bound; dwconv on NPU); LayerNorm/RoPE/GLU/softmax/residual on host.
//! Faithful port of `npu_asr/fused.py` (FusedFFN + FusedBlock).

use std::path::Path;
use std::rc::Rc;

use ndarray::prelude::*;
use npu_asr_host::prof;
use npu_asr_host::{
    bf16_round, dwconv_k5, glu, glu_fused, layer_norm, layer_norm_normalize, mha,
    residual_add_round, rope, silu,
};
use npu_xrt::Device;

#[cfg(all(feature = "chained_ffn", not(feature = "two_ctx")))]
use crate::engines::ChainedFFN;
#[cfg(not(feature = "two_ctx"))]
use crate::engines::WAEpilogue;
#[cfg(feature = "two_ctx")]
use crate::ctx2::{CtxAOp, Epi, FfnMm2, SharedCtxA};
#[cfg(feature = "two_ctx")]
use crate::ctx_ln::CtxLn;
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

/// Apply the LN affine on the host: out[r,j] = x[r,j]*gamma[j] + beta[j]. Used after the on-NPU
/// normalize-only ctxLN for the 4 affine LN sites (the normalize — mean/var/invsqrt, the numerically
/// hard transcendental — is on the NPU; this cheap linear affine stays exact on the host).
#[cfg(feature = "two_ctx")]
fn apply_affine(x: &Array2<f32>, gamma: &[f32], beta: &[f32]) -> Array2<f32> {
    Array2::from_shape_fn(x.dim(), |(r, j)| x[[r, j]] * gamma[j] + beta[j])
}

/// Macaron-half FFN. By default (`chained_ffn` feature) the 3072-wide intermediate H is kept in a
/// device BO across the two dispatches (no host round-trip). Without the feature it falls back to
/// the original two independent weight-bound dispatches with a host hop between them. Either way:
/// LN affine folded into mm1, bias1 K-augmented, SiLU on-chip; bias2 added after mm2.
#[cfg(feature = "two_ctx")]
struct FusedFFN {
    mm1: CtxAOp, // ctxA plain matmul N=3072; host epilogue = SiLU(z + b1) (Epi::SiluBias)
    mm2: FfnMm2, // SAME ctxA, K=3072 split into 4× N=768 partials; host-accumulate + bias2
    // ctxLN: the (folded, normalize-only) LN runs on the NPU when NPU_LN_NPU=1 (Step D). None = host.
    ctx_ln: Option<Rc<CtxLn>>,
    // NPU_ENC_FFN_RESIDENT (draft, default OFF): keep the [Mp,3072] mm1->mm2 intermediate in one bf16
    // buffer across the seam (no host f32 materialize / per-partial re-conversion). Set from the
    // resolved TuningConfig in `new_with_flags`; defaults from env in `new`.
    resident_on: bool,
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
    fn new(w: &BlockWeights, pfx: &str, norm: &str, ctx_a: Rc<SharedCtxA>, ctx_ln: Option<Rc<CtxLn>>) -> Self {
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
        FusedFFN {
            mm1,
            mm2,
            ctx_ln,
            resident_on: std::env::var("NPU_ENC_FFN_RESIDENT").as_deref() == Ok("1"),
        }
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
        // affine folded into mm1 -> normalize-only LN; on the NPU (ctxLN) when NPU_LN_NPU=1, else host.
        let norm = prof::time("layer_norm", || match &self.ctx_ln {
            Some(cl) => cl.normalize(x),
            None => layer_norm_normalize(x, LN_EPS),
        });
        // RESIDENT-INTERMEDIATE FFN (NPU_ENC_FFN_RESIDENT, draft): keep the [Mp,3072] mm1->mm2
        // intermediate in one bf16 buffer (no host f32 materialize / per-partial re-conversion).
        // forward_resident falls back internally for int8; for bf16 it is numerically identical to
        // the non-resident path. Default OFF -> the shipped two-step path below.
        if self.resident_on {
            return self.mm2.forward_resident(&self.mm1, &norm);
        }
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
    /// pw1's bias [1536] (two_ctx: applied in `glu_fused`; non-two_ctx: unused, rides the WAEpilogue).
    pw1_bias: Vec<f32>,
    /// two_ctx Goal-2 toggle: fused bias+GLU+transpose (default) vs the original bias→transpose→glu.
    /// Runtime-selected (`NPU_GLU_FUSED=0` disables) so the cut can be A/B'd cleanly in one binary.
    glu_fused_on: bool,
    /// two_ctx Goal-1 toggle: overlap qk ∥ v (independent attention projections) on the 2 pipe slots.
    /// MEASURED opt-in (`NPU_QKV_OVERLAP=1`): byte-identical, but NEUTRAL on the fast-bf16 default
    /// (per-dispatch NPU compute ~0.3ms — only 2 ops/block to hide), ~6ms on native. Default-OFF so the
    /// shipped default is unchanged; kept as a lever for the native path. No-op if the mm2 pipeline is off.
    #[cfg(feature = "two_ctx")]
    qkv_overlap_on: bool,
    // host LayerNorm affines
    ln_satt_w: Vec<f32>,
    ln_satt_b: Vec<f32>,
    ln_conv_w: Vec<f32>,
    ln_conv_b: Vec<f32>,
    bn_w: Vec<f32>,
    bn_b: Vec<f32>,
    ln_out_w: Vec<f32>,
    ln_out_b: Vec<f32>,
    // ctxLN: the 4 affine LNs run normalize-on-NPU + host γ,β when NPU_LN_NPU=1 (Step D). None = host.
    #[cfg(feature = "two_ctx")]
    ctx_ln: Option<Rc<CtxLn>>,
    dw_taps: Array2<f32>, // [768,5]
    dw_bias: Vec<f32>,
    cos: Array2<f32>,
    sin: Array2<f32>,
}

impl FusedBlock {
    /// As `new`, but glu_fused / qkv_overlap come from resolved config instead of env. Behavior is
    /// identical to `new` when the flags equal the env-derived defaults.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_flags(
        dev: Rc<Device>,
        root: &Path,
        w: &BlockWeights,
        cos: &Array2<f32>,
        sin: &Array2<f32>,
        #[cfg(feature = "two_ctx")] ctx_a: Rc<SharedCtxA>,
        #[cfg(feature = "two_ctx")] ctx_ln: Option<Rc<CtxLn>>,
        glu_fused: bool,
        qkv_overlap: bool,
        ffn_resident: bool,
    ) -> Self {
        let mut blk = Self::new(
            dev, root, w, cos, sin,
            #[cfg(feature = "two_ctx")] ctx_a,
            #[cfg(feature = "two_ctx")] ctx_ln,
        );
        blk.glu_fused_on = glu_fused;
        #[cfg(feature = "two_ctx")]
        {
            blk.qkv_overlap_on = qkv_overlap;
            blk.ffn1.resident_on = ffn_resident;
            blk.ffn2.resident_on = ffn_resident;
        }
        #[cfg(not(feature = "two_ctx"))]
        { let _ = (qkv_overlap, ffn_resident); }
        blk
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dev: Rc<Device>,
        root: &Path,
        w: &BlockWeights,
        cos: &Array2<f32>,
        sin: &Array2<f32>,
        #[cfg(feature = "two_ctx")] ctx_a: Rc<SharedCtxA>,
        #[cfg(feature = "two_ctx")] ctx_ln: Option<Rc<CtxLn>>,
    ) -> Self {
        #[cfg(not(feature = "two_ctx"))]
        let ffn1 = FusedFFN::new(dev.clone(), root, w, "feed_forward1", "norm_feed_forward1");
        #[cfg(not(feature = "two_ctx"))]
        let ffn2 = FusedFFN::new(dev.clone(), root, w, "feed_forward2", "norm_feed_forward2");
        #[cfg(feature = "two_ctx")]
        let ffn1 = FusedFFN::new(w, "feed_forward1", "norm_feed_forward1", ctx_a.clone(), ctx_ln.clone());
        #[cfg(feature = "two_ctx")]
        let ffn2 = FusedFFN::new(w, "feed_forward2", "norm_feed_forward2", ctx_a.clone(), ctx_ln.clone());

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
        let pw1_bias = w.v("conv.pointwise_conv1.bias"); // [1536]
        // two_ctx: pw1 emits the RAW matmul (Epi::None) and the bias folds into glu_fused (Goal-2
        // host cut: one fused bias+GLU+transpose pass replaces the bias epilogue + the [T,1536]->
        // [1536,T] transpose copy + the standalone glu). Fallback path keeps the biased pw1 + glu.
        #[cfg(feature = "two_ctx")]
        let pw1 = CtxAOp::new(ctx_a.clone(), &pw1_bt, 1536, Epi::None, &[]);
        #[cfg(not(feature = "two_ctx"))]
        let pw1 = mk_bias(&pw1_bt, 1536, &pw1_bias);
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
            pw1_bias,
            glu_fused_on: std::env::var("NPU_GLU_FUSED").as_deref() != Ok("0"),
            #[cfg(feature = "two_ctx")]
            qkv_overlap_on: std::env::var("NPU_QKV_OVERLAP").as_deref() == Ok("1"),
            ln_satt_w: w.v("norm_self_att.weight"),
            ln_satt_b: w.v("norm_self_att.bias"),
            ln_conv_w: w.v("norm_conv.weight"),
            ln_conv_b: w.v("norm_conv.bias"),
            bn_w: w.v("conv.batch_norm.weight"),
            bn_b: w.v("conv.batch_norm.bias"),
            ln_out_w: w.v("norm_out.weight"),
            ln_out_b: w.v("norm_out.bias"),
            #[cfg(feature = "two_ctx")]
            ctx_ln,
            dw_taps,
            dw_bias: w.v("conv.depthwise_conv.bias"),
            cos: cos.clone(),
            sin: sin.clone(),
        }
    }

    /// Affine LayerNorm dispatch: normalize on the NPU (ctxLN) then host γ,β when NPU_LN_NPU=1,
    /// else the all-host `layer_norm`. Both match the host reference exactly.
    #[cfg(feature = "two_ctx")]
    fn ln_affine(&self, x: &Array2<f32>, gamma: &[f32], beta: &[f32]) -> Array2<f32> {
        match &self.ctx_ln {
            Some(cl) => apply_affine(&cl.normalize(x), gamma, beta),
            None => layer_norm(x, gamma, beta, LN_EPS),
        }
    }
    #[cfg(not(feature = "two_ctx"))]
    fn ln_affine(&self, x: &Array2<f32>, gamma: &[f32], beta: &[f32]) -> Array2<f32> {
        layer_norm(x, gamma, beta, LN_EPS)
    }

    /// x is [T, 768] (bf16-valued f32). `valid_len` is the number of non-padded time frames;
    /// the two time-mixing ops (attention, depthwise conv) mask positions >= valid_len so padding
    /// doesn't leak into valid frames (pass valid_len >= T for no masking). Returns [T, 768].
    pub fn forward(&self, x: &Array2<f32>, valid_len: usize, is_first: bool) -> Array2<f32> {
        // R1: the block stack maintains a bf16-valued invariant — enc_prep rounds block 0's input, and
        // every block returns a bf16-rounded output (final LN + residual_add_round). So this per-block entry
        // round is IDEMPOTENT for blocks >=1; skip it (byte-identical). Keep block 0's as a defensive
        // boundary even though enc_prep already rounds it.
        let mut x = if is_first {
            prof::time("bf16_round", || x.mapv(bf16_round))
        } else {
            x.to_owned()
        };

        // --- FFN1 (macaron half), residual ×0.5 ---
        let f1 = self.ffn1.forward(&x);
        x = prof::time("residual+bf16", || residual_add_round(&x, &f1, 0.5));

        // --- MHSA ---
        let ln = prof::time("layer_norm", || {
            self.ln_affine(&x, &self.ln_satt_w, &self.ln_satt_b)
        });
        let r = prof::time("rope", || {
            rope(&ln, &self.cos, &self.sin, N_HEADS, HEAD_DIM)
        });
        // qk (reads rope'd `r`) ∥ v (reads `ln`) are independent and both feed mha → overlap their
        // NPU compute with each other's host marshaling on the 2 pipe slots (Goal-1, two_ctx default).
        #[cfg(feature = "two_ctx")]
        let (qk, vp) = if self.qkv_overlap_on {
            self.qk.forward2_overlapped(r.view(), &self.v, ln.view())
        } else {
            (self.qk.forward(&r), self.v.forward(&ln))
        };
        #[cfg(not(feature = "two_ctx"))]
        let qk = self.qk.forward(&r); // [T,1536]
        #[cfg(not(feature = "two_ctx"))]
        let vp = self.v.forward(&ln);
        let (qp, kp) = prof::time("qk_split", || {
            (
                qk.slice(s![.., ..D_MODEL]).to_owned(),
                qk.slice(s![.., D_MODEL..]).to_owned(),
            )
        });
        let ctx = prof::time("mha", || {
            mha(&qp, &kp, &vp, N_HEADS, HEAD_DIM, true, valid_len)
        });
        let op = self.o.forward(&ctx);
        x = prof::time("residual+bf16", || residual_add_round(&x, &op, 1.0));

        // --- ConvModule ---
        let ln = prof::time("layer_norm", || {
            self.ln_affine(&x, &self.ln_conv_w, &self.ln_conv_b)
        });
        let pw1 = self.pw1.forward(&ln); // two_ctx: [T,1536] RAW; fallback: [T,1536] biased
        #[cfg(feature = "two_ctx")]
        let mut g = prof::time("glu", || {
            if self.glu_fused_on {
                glu_fused(&pw1, &self.pw1_bias) // [768,T] — bias+GLU+transpose in one pass
            } else {
                // original path (for A/B): apply bias, transpose, then glu — from the raw pw1.
                let two_c = pw1.ncols();
                let mut biased = pw1.clone();
                biased.axis_iter_mut(Axis(0)).for_each(|mut row| {
                    for j in 0..two_c {
                        row[j] += self.pw1_bias[j];
                    }
                });
                glu(&biased.t().to_owned()) // [768,T]
            }
        });
        #[cfg(not(feature = "two_ctx"))]
        let mut g = prof::time("glu", || {
            let pw1t = pw1.t().to_owned(); // [1536,T]
            glu(&pw1t) // [768,T]
        });
        // zero padded time columns so the depthwise FIR (k=5) doesn't pull padding into valid frames
        prof::time("pad_mask", || {
            let tt_g = g.ncols();
            if valid_len < tt_g {
                for c in 0..g.nrows() {
                    for ti in valid_len..tt_g {
                        g[[c, ti]] = 0.0;
                    }
                }
            }
        });
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
            self.ln_affine(&dwout.t().to_owned(), &self.bn_w, &self.bn_b) // [T,768]
        });
        // silu is elementwise, so silu(bn) == silu(bn.t()).t(), and pw2 wants [T,768] = silu(bn)
        // directly — so apply silu in the [T,768] layout and skip the bn.t() and sw.t() copies (the
        // conv tail had 3 transposes of [768,T]; this removes 2). Byte-identical (same f32 values).
        let sw = prof::time("silu", || silu(&bn)); // [T,768]
        let pw2 = self.pw2.forward(&sw); // [T,768]
        x = prof::time("residual+bf16", || residual_add_round(&x, &pw2, 1.0));

        // --- FFN2 (macaron half), residual ×0.5 ---
        let f2 = self.ffn2.forward(&x);
        x = prof::time("residual+bf16", || residual_add_round(&x, &f2, 0.5));

        // --- final LayerNorm ---
        prof::time("layer_norm", || {
            self.ln_affine(&x, &self.ln_out_w, &self.ln_out_b).mapv(bf16_round)
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
