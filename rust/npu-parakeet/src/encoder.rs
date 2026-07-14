//! Parakeet FastConformer encoder (host f32 reference). Ports scripts/parakeet_ref_encoder.py.
//! `forward_last` is the general-engine Encoder-contract entry point.

use std::path::Path;

use ndarray::prelude::*;

use crate::config::ModelCfg;
use crate::ops::{conv2d, dwconv1d, layernorm, rel_shift, sigmoid, silu_inplace};
use crate::prof;
use crate::prof::phase::{Bucket, PhaseScope};
use crate::pos::rel_pos_encoding;
use crate::weights::{BlockWeights, ParakeetWeights};

pub struct FastConformerEncoder {
    pub cfg: ModelCfg,
    w: ParakeetWeights,
    #[cfg(feature = "npu")]
    npu: Option<crate::npu::NpuMatmul>,
}

impl FastConformerEncoder {
    pub fn new(artifacts: &Path, cfg: ModelCfg) -> Self {
        let w = ParakeetWeights::load(artifacts).expect("load parakeet weights");
        assert_eq!(w.nblocks(), cfg.n_layers, "block count mismatch");
        FastConformerEncoder {
            cfg,
            w,
            #[cfg(feature = "npu")]
            npu: None,
        }
    }

    /// Construct with the NPU matmul path enabled. `root` = repo root holding the mlir-aie build
    /// dir with the Parakeet xclbins. Single-tenant NPU only.
    #[cfg(feature = "npu")]
    pub fn new_npu(artifacts: &Path, cfg: ModelCfg, root: &Path) -> Self {
        let mut e = Self::new(artifacts, cfg);
        e.npu = Some(crate::npu::NpuMatmul::open(root));
        e
    }

    /// Weight matmul C[m,n] = A[m,k] @ B[k,n] — NPU if enabled, else host ndarray. `id` keys the
    /// NPU weight-BO cache (unique per fixed weight, e.g. "3.ff1.l1"). Eager (non-lazy) sibling of
    /// [`Self::mm_lazy`]; retained as the general entry point (all encoder call sites use the lazy
    /// variant, which skips the constant-weight reclone on warm passes).
    #[allow(dead_code)]
    fn mm(&self, a: &Array2<f32>, b: &Array2<f32>, id: &str) -> Array2<f32> {
        #[cfg(feature = "npu")]
        {
            if let Some(npu) = &self.npu {
                return npu.matmul_id(a, b, id);
            }
        }
        let _ = id;
        a.dot(b)
    }

    /// Lazy weight matmul: mirrors [`Self::mm`] but the weight matrix is built by `make_b` ONLY on
    /// a NPU weight-BO cache miss. On warm (cache-hit) passes the closure never runs, so the whole
    /// constant-weight host reclone/transpose inside it is skipped -- this is the point. The host
    /// (no-NPU) fallback always runs `make_b` (identical to the eager path).
    fn mm_lazy<F: FnOnce() -> Array2<f32>>(&self, a: &Array2<f32>, make_b: F, id: &str) -> Array2<f32> {
        #[cfg(feature = "npu")]
        {
            if let Some(npu) = &self.npu {
                return npu.matmul_id_lazy(a, make_b, id);
            }
        }
        let _ = id;
        let b = make_b();
        a.dot(&b)
    }

    /// Lazy matmul that fuses the FFN SiLU as the on-chip GEMM epilogue when the modal resident is
    /// loaded (A1 / `ff_act` on-chip). On that path the result is already `silu(a @ b)`; on the
    /// plain NPU resident or the host fallback it returns the raw `a @ b` and the caller applies
    /// host silu (gated on [`Self::ff_act_on_chip`]).
    fn mm_lazy_silu<F: FnOnce() -> Array2<f32>>(&self, a: &Array2<f32>, make_b: F, id: &str) -> Array2<f32> {
        #[cfg(feature = "npu")]
        {
            if let Some(npu) = &self.npu {
                return npu.matmul_id_lazy_silu(a, make_b, id);
            }
        }
        let _ = id;
        let b = make_b();
        a.dot(&b)
    }

    /// True when the FFN SiLU is applied on chip (modal resident), so the host must skip it.
    fn ff_act_on_chip(&self) -> bool {
        #[cfg(feature = "npu")]
        {
            if let Some(npu) = &self.npu {
                return npu.modal();
            }
        }
        false
    }

    fn feed_forward(&self, x: &Array2<f32>, b: &BlockWeights, blk: usize, tag: &str, norm_w: &str, norm_b: &str, l1: &str, l2: &str) -> Array2<f32> {
        let stage: &'static str = if tag == "ff1" { "ff1" } else { "ff2" };
        // RESIDENT FF1 seam (DEFAULT on the modal resident; opt out with PARAKEET_RESIDENT_FF=0):
        // LN + fc1 + SiLU run FULLY on-NPU device-side (ctxLN -> affine_cast(gamma,beta) -> modal fc1
        // on-chip silu), the activation stream never touching host across LN->fc1. WER-neutral (==
        // baseline). Returns silu(affine_LN(x)@W1); fc2 stays host-fed for now (next frontier step).
        // Falls back to the host LN path when the resident xclbins aren't built (resident_ff_available).
        #[cfg(feature = "npu")]
        if std::env::var("PARAKEET_RESIDENT_FF").map(|v| v != "0").unwrap_or(true) {
            if let Some(npu) = &self.npu {
                if npu.resident_ff_available() {
                    let gamma = b.v(norm_w);
                    let beta = b.v(norm_b);
                    prof::phase::set_stage(stage);
                    let h = {
                        let _h = PhaseScope::new("ff_resident", Bucket::Npu);
                        npu.resident_ff1_fc1(x, gamma.as_slice().unwrap(), beta.as_slice().unwrap(),
                            || b.m(l1), &format!("{blk}.{tag}.l1"), self.cfg.ff)
                    };
                    prof::phase::set_stage(stage);
                    return self.mm_lazy(&h, || { let _wp = PhaseScope::new("ff_wprep", Bucket::Marshal); b.m(l2) }, &format!("{blk}.{tag}.l2"));
                }
            }
        }
        let n = {
            let _h = PhaseScope::new("ln", Bucket::Host);
            layernorm(x, &b.v(norm_w), &b.v(norm_b))
        };
        // ff_wprep: materialize the (T'-independent) FFN weight matrix for mm(). `b.m()` clones the
        // whole [D,DFF]/[DFF,D] array out of the weight map + reifies its layout -- pure host data
        // movement (no math, no device). Now materialized LAZILY inside mm_lazy's closure: on a warm
        // (weight-BO cache hit) pass the clone never runs, eliminating the per-pass reclone of the
        // constant weight (the #0 perf lever). The ff_wprep scope stays inside the closure so a miss
        // still attributes; a hit skips the closure (and thus the scope) entirely.
        let mut h = {
            prof::phase::set_stage(stage);
            // A1: fc1 fuses SiLU into the GEMM epilogue on the modal resident (result already
            // activated). ff_wprep stays inside the lazy closure so a weight-BO miss still attributes.
            self.mm_lazy_silu(&n, || { let _wp = PhaseScope::new("ff_wprep", Bucket::Marshal); b.m(l1) }, &format!("{blk}.{tag}.l1")) // [T, DFF]
        };
        // Host SiLU only when the NPU epilogue did not apply it (plain resident / host fallback).
        if !self.ff_act_on_chip() {
            let _h = PhaseScope::new("ff_act", Bucket::Host);
            silu_inplace(&mut h);
        }
        {
            prof::phase::set_stage(stage);
            self.mm_lazy(&h, || { let _wp = PhaseScope::new("ff_wprep", Bucket::Marshal); b.m(l2) }, &format!("{blk}.{tag}.l2")) // [T, D]
        }
    }

    pub fn weights(&self) -> &ParakeetWeights {
        &self.w
    }

    /// Public shim over the private [`Self::feed_forward`] for the FF1 macaron, used by the
    /// `ff1_parity` gate harness (resident-rails work). Runs `LN -> fc1 -> SiLU -> fc2` for block
    /// `blk` with the ff1 weight keys; the caller applies the `0.5*` residual (as `block()` does).
    pub fn feed_forward_ff1(&self, x: &Array2<f32>, blk: usize) -> Array2<f32> {
        let b = self.w.block(blk);
        self.feed_forward(
            x, b, blk, "ff1",
            "norm_feed_forward1.weight", "norm_feed_forward1.bias",
            "feed_forward1.linear1.weight", "feed_forward1.linear2.weight",
        )
    }

    /// NPU timing breakdown (feature `npu`, NPU path only).
    #[cfg(feature = "npu")]
    pub fn npu_stats_string(&self) -> Option<String> {
        self.npu.as_ref().map(|n| {
            let s = n.stats.borrow();
            format!(
                "npu breakdown: calls={} dispatches={} weight_load={:.2}s pack_a={:.2}s dispatch={:.2}s read={:.2}s accum={:.2}s",
                s.calls, s.dispatches, s.weight_load_s, s.pack_a_s, s.dispatch_s, s.read_s, s.accum_s
            )
        })
    }

    /// conv2D ÷8 dw-striding subsample: mel [128, T] -> [T/8, hidden].
    /// ONNX feeds conv as [b,1,time,freq]; flatten is [time, C*freq] (Transpose [0,2,1,3]).
    pub fn subsample(&self, mel: &Array2<f32>) -> Array2<f32> {
        // Whole subsample stem is host math (conv2d x5 + relu + flatten + final gemm);
        // no self.mm()/device call lives here, so one Host leaf scope cannot double-count.
        let _h = PhaseScope::new("subsample", Bucket::Host);
        let pe4 = |k: &str| self.w.pre(k).clone().into_dimensionality::<Ix4>().unwrap();
        let pe1 = |k: &str| self.w.pre(k).clone().into_dimensionality::<Ix1>().unwrap();
        // [1, time, freq]
        let (f, t) = mel.dim();
        let mut x = Array3::<f32>::zeros((1, t, f));
        for i in 0..t {
            for j in 0..f {
                x[[0, i, j]] = mel[[j, i]];
            }
        }
        let relu = |a: &mut Array3<f32>| a.mapv_inplace(|v| v.max(0.0));
        let mut x = conv2d(&x, &pe4("conv.0.weight"), &pe1("conv.0.bias"), 2, 1, 1);
        relu(&mut x);
        let x = conv2d(&x, &pe4("conv.2.weight"), &pe1("conv.2.bias"), 2, 1, 256);
        let mut x = conv2d(&x, &pe4("conv.3.weight"), &pe1("conv.3.bias"), 1, 0, 1);
        relu(&mut x);
        let x = conv2d(&x, &pe4("conv.5.weight"), &pe1("conv.5.bias"), 2, 1, 256);
        let mut x = conv2d(&x, &pe4("conv.6.weight"), &pe1("conv.6.bias"), 1, 0, 1);
        relu(&mut x);
        // x: [C=256, H=time, W=freq]; flatten -> [time, C*freq]
        let (c, ht, wf) = x.dim();
        let mut flat = Array2::<f32>::zeros((ht, c * wf));
        for ti in 0..ht {
            for ci in 0..c {
                for fi in 0..wf {
                    flat[[ti, ci * wf + fi]] = x[[ci, ti, fi]];
                }
            }
        }
        let wout = self.w.pre("out.weight").clone().into_dimensionality::<Ix2>().unwrap(); // [4096, hidden]
        let bout = self.w.pre("out.bias").clone().into_dimensionality::<Ix1>().unwrap();
        prof::phase::set_stage("subsample"); // final gemm (host .dot here; labels device path if ever routed via mm)
        flat.dot(&wout) + &bout
    }

    fn mhsa(&self, x: &Array2<f32>, blk: usize, pos_enc: &Array2<f32>) -> Array2<f32> {
        let b = self.w.block(blk);
        let (h, dk, d) = (self.cfg.n_heads, self.cfg.head_dim, self.cfg.hidden);
        let t = x.nrows();
        let p = pos_enc.nrows(); // 2T-1
        // mhsa_wprep: materialize each (T'-independent) attention projection weight for its mm().
        // Each `b.m()` clones the whole [D,D]/[P,D] matrix out of the weight map -- pure host data
        // movement (no math, no device). Now materialized LAZILY inside mm_lazy's closure: on a warm
        // (weight-BO cache hit) pass the clone never runs, eliminating the per-pass reclone of the
        // constant qkv/pos/out weights. Only the tiny pos_bias_u/v (consumed later in the score
        // loop, NOT a cached BO) stay eager, exactly as the original.
        prof::phase::set_stage("mhsa_qkv");
        let q = self.mm_lazy(x, || { let _wp = PhaseScope::new("mhsa_wprep", Bucket::Marshal); b.m("self_attn.linear_q.weight") }, &format!("{blk}.q")); // [T, D]
        prof::phase::set_stage("mhsa_qkv");
        let k = self.mm_lazy(x, || { let _wp = PhaseScope::new("mhsa_wprep", Bucket::Marshal); b.m("self_attn.linear_k.weight") }, &format!("{blk}.k"));
        prof::phase::set_stage("mhsa_qkv");
        let v = self.mm_lazy(x, || { let _wp = PhaseScope::new("mhsa_wprep", Bucket::Marshal); b.m("self_attn.linear_v.weight") }, &format!("{blk}.v"));
        prof::phase::set_stage("mhsa_pos");
        let pm = self.mm_lazy(pos_enc, || { let _wp = PhaseScope::new("mhsa_wprep", Bucket::Marshal); b.m("self_attn.linear_pos.weight") }, &format!("{blk}.pos")); // [P, D]
        let (ubias, vbias) = {
            let _wp = PhaseScope::new("mhsa_wprep", Bucket::Marshal);
            (b.m("self_attn.pos_bias_u"), b.m("self_attn.pos_bias_v")) // [H, DK] each
        };
        let scale = (dk as f32).sqrt();

        // RESIDENT MHA (opt-in PARAKEET_RESIDENT_MHA=1): replace the host per-head
        // scores/rel_shift/softmax/context with the on-chip STEP=8 block, one dispatch per head.
        // The kernel bakes inv_scale=1/sqrt(128), so pass qu=qh+u / qv=qh+v / k / p / v directly.
        #[cfg(feature = "npu")]
        if std::env::var("PARAKEET_RESIDENT_MHA").is_ok() {
            if let Some(npu) = &self.npu {
                let _h = PhaseScope::new("mhsa_resident", Bucket::Npu);
                let mut ctx = Array2::<f32>::zeros((t, d));
                for hh in 0..h {
                    let col = hh * dk;
                    let qh = q.slice(s![.., col..col + dk]);
                    let kh = k.slice(s![.., col..col + dk]).to_owned();
                    let ph = pm.slice(s![.., col..col + dk]).to_owned();
                    let vh = v.slice(s![.., col..col + dk]).to_owned();
                    let mut qu = qh.to_owned();
                    let mut qv = qh.to_owned();
                    for i in 0..t {
                        for c in 0..dk {
                            qu[[i, c]] += ubias[[hh, c]];
                            qv[[i, c]] += vbias[[hh, c]];
                        }
                    }
                    let ch = npu.relpos_mha(&qu, &qv, &kh, &ph, &vh);
                    ctx.slice_mut(s![.., col..col + dk]).assign(&ch);
                }
                prof::phase::set_stage("mhsa_qkv");
                return self.mm_lazy(&ctx, || { let _wp = PhaseScope::new("mhsa_wprep", Bucket::Marshal); b.m("self_attn.linear_out.weight") }, &format!("{blk}.out"));
            }
        }

        // assemble bd_all [H, T, P] then rel_shift -> [H, T, T]
        let (mut bd_all, mut ac_all);
        {
            // Per-head QK^T (ac) + QV.pos (bd) score matrices are host ndarray dots (not self.mm):
            // charged to mhsa_scores. Not one of the plan's named labels; see task report. The
            // score-buffer zeros allocations are folded into this scope (were previously an
            // un-scoped span leaking to the report-level residual).
            let _h = PhaseScope::new("mhsa_scores", Bucket::Host);
            bd_all = Array3::<f32>::zeros((h, t, p));
            ac_all = Array3::<f32>::zeros((h, t, t));
            for hh in 0..h {
                let col = hh * dk;
                // per-head slices
                let qh = q.slice(s![.., col..col + dk]); // [T, DK]
                let kh = k.slice(s![.., col..col + dk]);
                let ph = pm.slice(s![.., col..col + dk]); // [P, DK]
                // qu = qh + u[h]; qv = qh + v[h]
                let mut qu = qh.to_owned();
                let mut qv = qh.to_owned();
                for i in 0..t {
                    for c in 0..dk {
                        qu[[i, c]] += ubias[[hh, c]];
                        qv[[i, c]] += vbias[[hh, c]];
                    }
                }
                ac_all.slice_mut(s![hh, .., ..]).assign(&qu.dot(&kh.t())); // [T, T]
                bd_all.slice_mut(s![hh, .., ..]).assign(&qv.dot(&ph.t())); // [T, P]
            }
        }
        let bd = prof::time("rel_shift", || {
            let _h = PhaseScope::new("mhsa_relshift", Bucket::Host);
            rel_shift(&bd_all, t)
        }); // [H, T, T]

        // scores -> softmax -> context -> merge -> linear_out
        let ctx = prof::time("mha_softmax", || {
        let mut ctx = Array2::<f32>::zeros((t, d));
        for hh in 0..h {
            let col = hh * dk;
            let vh = v.slice(s![.., col..col + dk]); // [T, DK]
            let mut scores = Array2::<f32>::zeros((t, t));
            {
                let _h = PhaseScope::new("mhsa_softmax", Bucket::Host);
                for i in 0..t {
                    let mut mx = f32::NEG_INFINITY;
                    for j in 0..t {
                        let sc = (ac_all[[hh, i, j]] + bd[[hh, i, j]]) / scale;
                        scores[[i, j]] = sc;
                        mx = mx.max(sc);
                    }
                    let mut sum = 0.0;
                    for j in 0..t {
                        let e = (scores[[i, j]] - mx).exp();
                        scores[[i, j]] = e;
                        sum += e;
                    }
                    for j in 0..t {
                        scores[[i, j]] /= sum;
                    }
                }
            }
            {
                let _h = PhaseScope::new("mhsa_context", Bucket::Host);
                let ch = scores.dot(&vh); // [T, DK]
                ctx.slice_mut(s![.., col..col + dk]).assign(&ch);
            }
        }
        ctx
        });
        prof::phase::set_stage("mhsa_qkv");
        self.mm_lazy(&ctx, || { let _wp = PhaseScope::new("mhsa_wprep", Bucket::Marshal); b.m("self_attn.linear_out.weight") }, &format!("{blk}.out"))
    }

    fn conv_module(&self, x: &Array2<f32>, blk: usize) -> Array2<f32> {
        let b = self.w.block(blk);
        let d = self.cfg.hidden;
        let t = x.nrows();
        // conv_wprep: materialize + reshape the (T'-independent) conv weights for mm(). The pointwise
        // conv1/conv2 weights (pw1/pw2) feed a cached NPU weight BO, so they are now materialized
        // LAZILY inside mm_lazy's closure (whole `b.m3(..).index_axis(..).to_owned().t().to_owned()`
        // chain) -- skipped on warm cache-hit passes (the #0 lever). The depthwise taps + bias feed
        // the HOST dwconv1d (no cached BO), so they are genuinely needed every pass and stay eager.
        let (taps, dwb) = {
            let _wp = PhaseScope::new("conv_wprep", Bucket::Marshal);
            let dw3 = b.m3("conv.depthwise_conv.weight"); // [D, 1, 9]
            let taps = dw3.index_axis(Axis(1), 0).to_owned(); // [D, 9]
            let dwb = b.v("conv.depthwise_conv.bias");
            (taps, dwb)
        };

        prof::phase::set_stage("conv_pw");
        // pw1 chain -> [D, 2D] (mm's [K,N] layout): materialized lazily; skipped on warm passes.
        let h = self.mm_lazy(x, || {
            let _wp = PhaseScope::new("conv_wprep", Bucket::Marshal);
            b.m3("conv.pointwise_conv1.weight").index_axis(Axis(2), 0).to_owned().t().to_owned() // [2D,D,1]->[2D,D]->[D,2D]
        }, &format!("{blk}.pw1")); // [T, 2D]
        // GLU: a * sigmoid(g)
        let glu = prof::time("glu", || {
            let _h = PhaseScope::new("conv_glu", Bucket::Host);
            let mut glu = Array2::<f32>::zeros((t, d));
            for i in 0..t {
                for c in 0..d {
                    glu[[i, c]] = h[[i, c]] * sigmoid(h[[i, d + c]]);
                }
            }
            glu
        });
        // depthwise along time: [D, T]. Bracketing transposes + trailing SiLU are host math
        // with no mm() inside, so they fold into the conv_dwconv Host leaf scope.
        let back = prof::time("dwconv", || {
            let _h = PhaseScope::new("conv_dwconv", Bucket::Host);
            let glu_t = glu.t().to_owned();
            let mut dwc = dwconv1d(&glu_t, &taps, &dwb, 9); // [D, T]
            silu_inplace(&mut dwc);
            dwc.t().to_owned() // [T, D]
        });
        prof::phase::set_stage("conv_pw");
        // pw2 chain -> [D, D]: materialized lazily; skipped on warm passes.
        self.mm_lazy(&back, || {
            let _wp = PhaseScope::new("conv_wprep", Bucket::Marshal);
            b.m3("conv.pointwise_conv2.weight").index_axis(Axis(2), 0).to_owned().t().to_owned() // [D,D,1]->[D,D]->[D,D]
        }, &format!("{blk}.pw2"))
    }

    fn block(&self, x: &Array2<f32>, blk: usize, pos_enc: &Array2<f32>) -> Array2<f32> {
        let b = self.w.block(blk);
        // block_io: the [T', D] residual-stream clone at block entry (a working copy the residual
        // adds mutate). T'-dependent host data movement; scoped LEAF so it doesn't leak to residual.
        let mut x = {
            let _h = PhaseScope::new("block_io", Bucket::Marshal);
            x.clone()
        };
        let ff1 = prof::time("ff", || self.feed_forward(&x, b, blk, "ff1", "norm_feed_forward1.weight", "norm_feed_forward1.bias",
                                    "feed_forward1.linear1.weight", "feed_forward1.linear2.weight"));
        {
            let _h = PhaseScope::new("residual", Bucket::Host);
            x = x + ff1.mapv(|v| 0.5 * v); // macaron 0.5 scaling + residual add
        }
        let satt_in = prof::time("ln", || {
            let _h = PhaseScope::new("ln", Bucket::Host);
            layernorm(&x, &b.v("norm_self_att.weight"), &b.v("norm_self_att.bias"))
        });
        let mhsa_out = prof::time("mhsa", || self.mhsa(&satt_in, blk, pos_enc));
        {
            let _h = PhaseScope::new("residual", Bucket::Host);
            x = &x + &mhsa_out;
        }
        let conv_in = prof::time("ln", || {
            let _h = PhaseScope::new("ln", Bucket::Host);
            layernorm(&x, &b.v("norm_conv.weight"), &b.v("norm_conv.bias"))
        });
        let conv_out = prof::time("conv_mod", || self.conv_module(&conv_in, blk));
        {
            let _h = PhaseScope::new("residual", Bucket::Host);
            x = &x + &conv_out;
        }
        let ff2 = prof::time("ff", || self.feed_forward(&x, b, blk, "ff2", "norm_feed_forward2.weight", "norm_feed_forward2.bias",
                                    "feed_forward2.linear1.weight", "feed_forward2.linear2.weight"));
        {
            let _h = PhaseScope::new("residual", Bucket::Host);
            x = x + ff2.mapv(|v| 0.5 * v); // macaron 0.5 scaling + residual add
        }
        {
            let _h = PhaseScope::new("ln", Bucket::Host);
            layernorm(&x, &b.v("norm_out.weight"), &b.v("norm_out.bias"))
        }
    }

    /// Encoder block stack: x [T, hidden] -> [T, hidden]. (Contract entry point;
    /// valid_len is the unpadded length — masking is a no-op for full-length inputs.)
    pub fn forward_last(&self, x: &Array2<f32>, _valid_len: usize) -> Array2<f32> {
        // enc_setup: once-per-transcribe relative-position-encoding table build + input clone,
        // outside the 24-block loop. Scoped LEAF (host math + data movement, no mm()) so it lands
        // in a named stage rather than the report-level residual.
        let (pos_enc, mut x) = {
            let _h = PhaseScope::new("enc_setup", Bucket::Host);
            (rel_pos_encoding(x.nrows(), self.cfg.hidden), x.clone())
        };
        for blk in 0..self.cfg.n_layers {
            x = self.block(&x, blk, &pos_enc);
        }
        x
    }

    /// Run the block stack, returning every block's output (verification helper).
    pub fn forward_collect(&self, x0: &Array2<f32>) -> Vec<Array2<f32>> {
        let pos_enc = rel_pos_encoding(x0.nrows(), self.cfg.hidden);
        let mut x = x0.clone();
        let mut outs = Vec::with_capacity(self.cfg.n_layers);
        for blk in 0..self.cfg.n_layers {
            x = self.block(&x, blk, &pos_enc);
            outs.push(x.clone());
        }
        outs
    }

    /// Full encode from a mel spectrogram [128, T]: subsample then block stack.
    pub fn encode(&self, mel: &Array2<f32>) -> Array2<f32> {
        let x = prof::time("subsample", || self.subsample(mel));
        let t = x.nrows();
        self.forward_last(&x, t)
    }
}
