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
        // RESIDENT FFN (DEFAULT on the modal resident; opt out with PARAKEET_RESIDENT_FF=0):
        // LN + fc1 + SiLU run FULLY on-NPU (ctxLN -> affine_cast(gamma,beta) -> modal fc1 on-chip silu),
        // the activation stream never touching host across LN->fc1. Falls back to the host LN path when
        // the resident xclbins aren't built (resident_ff_available).
        #[cfg(feature = "npu")]
        if std::env::var("PARAKEET_RESIDENT_FF").map(|v| v != "0").unwrap_or(true) {
            if let Some(npu) = &self.npu {
                if npu.resident_ff_available() {
                    let gamma = b.v(norm_w);
                    let beta = b.v(norm_b);
                    prof::phase::set_stage(stage);
                    let _h = PhaseScope::new("ff_resident", Bucket::Npu);
                    // FULL FFN device-side (LN->fc1->fc2, Variant B, DEFAULT; opt out PARAKEET_RESIDENT_FFN=0):
                    // fc2's K-split partials stay on-device (deinterleave -> sub-BO chunks + host-sum),
                    // bit-identical to the host 4xK-split -> WER-NEUTRAL. resident_ff_available() requires the
                    // deint xclbin, so this falls back to the host-fed fc2 (below) when that xclbin is absent.
                    if std::env::var("PARAKEET_RESIDENT_FFN").map(|v| v != "0").unwrap_or(true) {
                        return npu.resident_ffn(x, gamma.as_slice().unwrap(), beta.as_slice().unwrap(),
                            || b.m(l1), &format!("{blk}.{tag}.l1"),
                            || b.m(l2), &format!("{blk}.{tag}.l2"));
                    }
                    let h = npu.resident_ff1_fc1(x, gamma.as_slice().unwrap(), beta.as_slice().unwrap(),
                        || b.m(l1), &format!("{blk}.{tag}.l1"), self.cfg.ff, true);
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
        // x is pre-LN. RESIDENT LN->QKV seam (opt-in PARAKEET_RESIDENT_MHA): norm_self_att LN runs
        // on-NPU (ctxLN -> affine_cast) and feeds the q/k/v modal GEMMs device-side off one resident
        // bf16 A -- the host LN is off the MHSA frontier. Falls back to host layernorm + mm_lazy when
        // the seam is off or the resident xclbins are absent (WER-identical to the old block()-level LN).
        #[cfg(feature = "npu")]
        let resident_mha = std::env::var("PARAKEET_RESIDENT_MHA").is_ok();
        #[cfg(not(feature = "npu"))]
        let resident_mha = false;
        // DIAGNOSTIC (PARAKEET_MHA_HOSTQKV=1): keep the resident attention block but feed it
        // HOST f32-LN + mm_lazy q/k/v (the DEFAULT path's qkv), decoupling the LN->QKV seam from
        // the resident attention to isolate which owns any WER gap. No effect unless RESIDENT_MHA.
        #[cfg(feature = "npu")]
        let resident_mha_qkv = resident_mha && std::env::var("PARAKEET_MHA_HOSTQKV").is_err();
        #[cfg(feature = "npu")]
        let resident_qkv = if resident_mha_qkv {
            self.npu.as_ref().filter(|n| n.resident_ff_available()).map(|npu| {
                let gamma = b.v("norm_self_att.weight");
                let beta = b.v("norm_self_att.bias");
                let _hh = PhaseScope::new("mha_resident_qkv", Bucket::Npu);
                npu.resident_mha_ln_qkv(x, gamma.as_slice().unwrap(), beta.as_slice().unwrap(),
                    || { let _wp = PhaseScope::new("mhsa_wprep", Bucket::Marshal); b.m("self_attn.linear_q.weight") }, &format!("{blk}.q"),
                    || { let _wp = PhaseScope::new("mhsa_wprep", Bucket::Marshal); b.m("self_attn.linear_k.weight") }, &format!("{blk}.k"),
                    || { let _wp = PhaseScope::new("mhsa_wprep", Bucket::Marshal); b.m("self_attn.linear_v.weight") }, &format!("{blk}.v"))
            })
        } else { None };
        #[cfg(not(feature = "npu"))]
        let resident_qkv: Option<(Array2<f32>, Array2<f32>, Array2<f32>)> = None;
        let (q, k, v) = if let Some((q, k, v)) = resident_qkv {
            (q, k, v)
        } else {
            // Host LN (or no-npu) + mm_lazy projections. x is pre-LN, so do the norm_self_att LN here
            // -- identical to the old block()-level LN, so this path (incl. the host-MHA DEFAULT) is
            // WER-neutral. The resident seam replaces exactly this LN + these three GEMMs.
            let ln_x = {
                let _h = PhaseScope::new("ln", Bucket::Host);
                layernorm(x, &b.v("norm_self_att.weight"), &b.v("norm_self_att.bias"))
            };
            let q = self.mm_lazy(&ln_x, || { let _wp = PhaseScope::new("mhsa_wprep", Bucket::Marshal); b.m("self_attn.linear_q.weight") }, &format!("{blk}.q")); // [T, D]
            let k = self.mm_lazy(&ln_x, || { let _wp = PhaseScope::new("mhsa_wprep", Bucket::Marshal); b.m("self_attn.linear_k.weight") }, &format!("{blk}.k"));
            let v = self.mm_lazy(&ln_x, || { let _wp = PhaseScope::new("mhsa_wprep", Bucket::Marshal); b.m("self_attn.linear_v.weight") }, &format!("{blk}.v"));
            (q, k, v)
        };
        // A/B (PARAKEET_MHA_QKV_AB=1): resident LN->QKV vs host layernorm(x)@W, rel-L2 per projection.
        #[cfg(feature = "npu")]
        if resident_mha && std::env::var("PARAKEET_MHA_QKV_AB").is_ok()
            && self.npu.as_ref().map(|n| n.resident_ff_available()).unwrap_or(false) {
            let ln_x = layernorm(x, &b.v("norm_self_att.weight"), &b.v("norm_self_att.bias"));
            let rel = |dev: &Array2<f32>, wname: &str| {
                let host = ln_x.dot(&b.m(wname));
                let mut num = 0f64; let mut den = 0f64;
                for i in 0..dev.nrows() { for j in 0..dev.ncols() {
                    let e = (dev[[i, j]] - host[[i, j]]) as f64; let g = host[[i, j]] as f64;
                    num += e * e; den += g * g;
                } }
                if den > 0.0 { (num / den).sqrt() } else { 0.0 }
            };
            eprintln!("[MHA_QKV_AB] blk={blk} T={t} q_relL2={:.4e} k_relL2={:.4e} v_relL2={:.4e}",
                rel(&q, "self_attn.linear_q.weight"), rel(&k, "self_attn.linear_k.weight"), rel(&v, "self_attn.linear_v.weight"));
        }
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
        // The resident relpos block is baked at RELPOS_BUILT_T (=172); it cannot serve longer clips.
        // Gate on t <= relpos_max_t() PER-CLIP: a T>BUILT_T clip skips the resident per-head loop and
        // falls through to the host attention path below (whole-block golden), so no crash/corruption.
        #[cfg(feature = "npu")]
        if std::env::var("PARAKEET_RESIDENT_MHA").is_ok()
            && self.npu.as_ref().map(|n| t <= n.relpos_max_t()).unwrap_or(false) {
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
                    // A/B localizer (PARAKEET_MHA_AB=1): compare head-0 resident ctx vs f32 host golden.
                    if hh == 0 && std::env::var("PARAKEET_MHA_AB").is_ok() {
                        let pp = ph.nrows();
                        let ac = qu.dot(&kh.t()); // [T,T]
                        let mut bd_all1 = Array3::<f32>::zeros((1, t, pp));
                        bd_all1.slice_mut(s![0, .., ..]).assign(&qv.dot(&ph.t()));
                        let bd = rel_shift(&bd_all1, t); // [1,T,T]
                        let mut scores = Array2::<f32>::zeros((t, t));
                        for i in 0..t {
                            let mut mx = f32::NEG_INFINITY;
                            for j in 0..t { let sc = (ac[[i, j]] + bd[[0, i, j]]) / scale; scores[[i, j]] = sc; mx = mx.max(sc); }
                            let mut sum = 0.0;
                            for j in 0..t { let e = (scores[[i, j]] - mx).exp(); scores[[i, j]] = e; sum += e; }
                            for j in 0..t { scores[[i, j]] /= sum; }
                        }
                        let ch_host = scores.dot(&vh); // [T,DK]
                        let mut num = 0.0f64; let mut den = 0.0f64; let mut maxrow = (0usize, 0.0f64);
                        for i in 0..t {
                            let mut rn = 0.0f64; let mut rd = 0.0f64;
                            for c in 0..dk {
                                let d = (ch[[i, c]] - ch_host[[i, c]]) as f64; let g = ch_host[[i, c]] as f64;
                                rn += d * d; rd += g * g;
                            }
                            num += rn; den += rd;
                            let rrel = if rd > 0.0 { (rn / rd).sqrt() } else { 0.0 };
                            if rrel > maxrow.1 { maxrow = (i, rrel); }
                        }
                        eprintln!("[MHA_AB] blk={blk} h0 T={t} ctx_relL2={:.4e} worst_row={} row_relL2={:.4e}",
                            (num / den).sqrt(), maxrow.0, maxrow.1);

                        // ---- PROBE (Ladder step 1): decompose the ~1% bf16 I/O quantization. Feed
                        // bf16-rounded operands into the SAME f32 host golden and measure ctx rel-L2
                        // vs pure-f32 (ch_host). Pure host math, no device -- isolates which rounding
                        // hop (AC inputs / BD inputs / probs narrow / V narrow / ctx-out narrow) owns
                        // the gap, and whether the full emulation reproduces the resident ~1.05e-2.
                        {
                            let rb = |x: f32| npu_xrt::bf16_bits_to_f32(npu_xrt::f32_to_bf16_bits(x));
                            let rl2 = |a: &Array2<f32>| -> f64 {
                                let mut n = 0f64; let mut dd = 0f64;
                                for i in 0..t { for c in 0..dk {
                                    let e = (a[[i, c]] - ch_host[[i, c]]) as f64; let g = ch_host[[i, c]] as f64;
                                    n += e * e; dd += g * g;
                                } }
                                if dd > 0.0 { (n / dd).sqrt() } else { 0.0 }
                            };
                            // f32 attention over (possibly bf16-rounded) operands; rprobs/rout narrow.
                            let fwd = |qu_: &Array2<f32>, qv_: &Array2<f32>, kh_: &Array2<f32>,
                                       ph_: &Array2<f32>, vh_: &Array2<f32>, rprobs: bool, rout: bool| -> Array2<f32> {
                                let ac = qu_.dot(&kh_.t());
                                let mut bd3 = Array3::<f32>::zeros((1, t, ph_.nrows()));
                                bd3.slice_mut(s![0, .., ..]).assign(&qv_.dot(&ph_.t()));
                                let bd = rel_shift(&bd3, t);
                                let mut probs = Array2::<f32>::zeros((t, t));
                                for i in 0..t {
                                    let mut mx = f32::NEG_INFINITY;
                                    for j in 0..t { let sc = (ac[[i, j]] + bd[[0, i, j]]) / scale; probs[[i, j]] = sc; mx = mx.max(sc); }
                                    let mut sum = 0.0;
                                    for j in 0..t { let e = (probs[[i, j]] - mx).exp(); probs[[i, j]] = e; sum += e; }
                                    let inv = 1.0 / sum;
                                    for j in 0..t { let mut pv = probs[[i, j]] * inv; if rprobs { pv = rb(pv); } probs[[i, j]] = pv; }
                                }
                                let mut out = probs.dot(vh_);
                                if rout { out.mapv_inplace(|x| rb(x)); }
                                out
                            };
                            let qu_b = qu.mapv(|x| rb(x)); let qv_b = qv.mapv(|x| rb(x));
                            let kh_b = kh.mapv(|x| rb(x)); let ph_b = ph.mapv(|x| rb(x));
                            let vh_b = vh.mapv(|x| rb(x));
                            let bd_in  = rl2(&fwd(&qu, &qv_b, &kh, &ph_b, &vh, false, false));
                            let bd_qv  = rl2(&fwd(&qu, &qv_b, &kh, &ph, &vh, false, false));
                            let bd_p   = rl2(&fwd(&qu, &qv, &kh, &ph_b, &vh, false, false));
                            let emul   = rl2(&fwd(&qu_b, &qv_b, &kh_b, &ph_b, &vh_b, true, true));
                            // Split-bf16 emulation of the BD (qv.p^T) matmul: hi=bf16(x), lo=bf16(x-hi).
                            // A@B = Ahi.Bhi + Ahi.Blo + Alo.Bhi (+Alo.Blo), each an exact bf16-input dot.
                            // The rest of the pipeline stays at emul precision (AC bf16-in, probs/V/ctx bf16).
                            let lo = |x: &Array2<f32>, hi: &Array2<f32>| -> Array2<f32> {
                                let mut r = x - hi; r.mapv_inplace(|z| rb(z)); r
                            };
                            let qv_lo = lo(&qv, &qv_b); let ph_lo = lo(&ph, &ph_b);
                            // full-pipeline fwd but with a caller-supplied precomputed BD [t, P] (pre-shift).
                            let fwd_bd = |bd_full: &Array2<f32>| -> Array2<f32> {
                                let ac = qu_b.dot(&kh_b.t());
                                let mut bd3 = Array3::<f32>::zeros((1, t, bd_full.ncols()));
                                bd3.slice_mut(s![0, .., ..]).assign(bd_full);
                                let bd = rel_shift(&bd3, t);
                                let mut probs = Array2::<f32>::zeros((t, t));
                                for i in 0..t {
                                    let mut mx = f32::NEG_INFINITY;
                                    for j in 0..t { let sc = (ac[[i, j]] + bd[[0, i, j]]) / scale; probs[[i, j]] = sc; mx = mx.max(sc); }
                                    let mut sum = 0.0;
                                    for j in 0..t { let e = (probs[[i, j]] - mx).exp(); probs[[i, j]] = e; sum += e; }
                                    let inv = 1.0 / sum;
                                    for j in 0..t { probs[[i, j]] = rb(probs[[i, j]] * inv); }
                                }
                                let mut out = probs.dot(&vh_b); out.mapv_inplace(|x| rb(x)); out
                            };
                            // bd_x2p: split p only (qv single bf16)  -> qv_b.(ph_hi+ph_lo)
                            let bd_x2p = &qv_b.dot(&ph_b.t()) + &qv_b.dot(&ph_lo.t());
                            // bd_x3: split both, drop lo.lo -> qv_b.ph_hi + qv_b.ph_lo + qv_lo.ph_hi
                            let bd_x3 = &(&qv_b.dot(&ph_b.t()) + &qv_b.dot(&ph_lo.t())) + &qv_lo.dot(&ph_b.t());
                            let split_p  = rl2(&fwd_bd(&bd_x2p));
                            let split_x3 = rl2(&fwd_bd(&bd_x3));
                            eprintln!("[MHA_PROBE] blk={blk} h0 T={t} bd_in={bd_in:.4e} bd_qv={bd_qv:.4e} bd_p={bd_p:.4e} emul_full={emul:.4e} FIX_split_p={split_p:.4e} FIX_split_x3={split_x3:.4e}");
                        }
                    }
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
        // RESIDENT conv module is DEFAULT-ON (opt out PARAKEET_RESIDENT_CONV=0 -> full host conv). The whole
        // module runs resident: LN -> pw1 (modal GEMM) -> GLU -> dwconv -> silu (time-major [T,D], transposes
        // dissolved) -> pw2 (modal GEMM), the activation stream never touching host across the frontier.
        // The on-NPU SiLU is also default-on; opt out PARAKEET_RESIDENT_SILU=0 to fall back to HOST silu (the
        // clean WER-8.2 reference path, kept for the future WER-refinement pass). ~8.5 is the accepted resident
        // baseline; silu precision is WER-irrelevant (the 8.2 vs 8.5 delta is a host-silu 17-clip decoder-chaos
        // artifact, not a brick defect). The stack_size root-cause that once blocked the exact on-NPU silu is
        // fixed (see silu-stack-overflow-root-cause-and-wer-reframe).
        #[cfg(feature = "npu")]
        let resident_conv = std::env::var("PARAKEET_RESIDENT_CONV").map(|v| v != "0").unwrap_or(true);
        #[cfg(feature = "npu")]
        let resident_silu = std::env::var("PARAKEET_RESIDENT_SILU").map(|v| v != "0").unwrap_or(true);
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
        // RESIDENT conv front (DEFAULT-ON, opt out PARAKEET_RESIDENT_CONV=0). norm_conv LN + pw1 + GLU run
        // FULLY on-NPU (ctxLN -> affine_cast -> modal pw1 N=2D identity -> GLU brick a*sigmoid(g)),
        // producing the gated [T, D] directly -- the activation never touches host across LN->pw1->GLU.
        // If the glu xclbin is absent, fall back to resident LN->pw1 [T,2D] + host GLU; if the resident seam
        // is off entirely (=0), full host LN + pw1 + host GLU.
        #[cfg(feature = "npu")]
        let resident_glu = if resident_conv {
            self.npu.as_ref().filter(|n| n.resident_ff_available()).and_then(|npu| {
                let gamma = b.v("norm_conv.weight");
                let beta = b.v("norm_conv.bias");
                let _hh = PhaseScope::new("conv_resident_glu", Bucket::Npu);
                npu.resident_conv_pw1_glu(x, gamma.as_slice().unwrap(), beta.as_slice().unwrap(),
                    || b.m3("conv.pointwise_conv1.weight").index_axis(Axis(2), 0).to_owned().t().to_owned(),
                    &format!("{blk}.pw1"))
            })
        } else { None };
        #[cfg(not(feature = "npu"))]
        let resident_glu: Option<Array2<f32>> = None;

        let glu = if let Some(g) = resident_glu {
            g // [T, D] -- GLU already applied on-NPU (step 2)
        } else {
            // step-1 resident LN->pw1 if available, else host LN + pw1 GEMM -> h [T, 2D]
            #[cfg(feature = "npu")]
            let resident_h = if resident_conv {
                self.npu.as_ref().filter(|n| n.resident_ff_available()).map(|npu| {
                    let gamma = b.v("norm_conv.weight");
                    let beta = b.v("norm_conv.bias");
                    let _hh = PhaseScope::new("conv_resident_pw1", Bucket::Npu);
                    npu.resident_ff1_fc1(x, gamma.as_slice().unwrap(), beta.as_slice().unwrap(),
                        || b.m3("conv.pointwise_conv1.weight").index_axis(Axis(2), 0).to_owned().t().to_owned(),
                        &format!("{blk}.pw1"), 2 * d, false)
                })
            } else { None };
            #[cfg(not(feature = "npu"))]
            let resident_h: Option<Array2<f32>> = None;
            let h = resident_h.unwrap_or_else(|| {
                let conv_in = prof::time("ln", || {
                    let _hh = PhaseScope::new("ln", Bucket::Host);
                    layernorm(x, &b.v("norm_conv.weight"), &b.v("norm_conv.bias"))
                });
                self.mm_lazy(&conv_in, || {
                    let _wp = PhaseScope::new("conv_wprep", Bucket::Marshal);
                    b.m3("conv.pointwise_conv1.weight").index_axis(Axis(2), 0).to_owned().t().to_owned() // [2D,D,1]->[2D,D]->[D,2D]
                }, &format!("{blk}.pw1"))
            }); // [T, 2D]
            // GLU host: a * sigmoid(g)
            prof::time("glu", || {
                let _h = PhaseScope::new("conv_glu", Bucket::Host);
                let mut glu = Array2::<f32>::zeros((t, d));
                for i in 0..t {
                    for c in 0..d {
                        glu[[i, c]] = h[[i, c]] * sigmoid(h[[i, d + c]]);
                    }
                }
                glu
            })
        };
        // depthwise along time: [D, T]. Bracketing transposes + trailing SiLU are host math
        // with no mm() inside, so they fold into the conv_dwconv Host leaf scope.
        let back = prof::time("dwconv", || {
            let _h = PhaseScope::new("conv_dwconv", Bucket::Host);
            // TIME-MAJOR fused dwconv+silu (step 3b): consumes GLU [T,D] DIRECTLY and emits [T,D]
            // DIRECTLY -- BOTH host transposes DISSOLVED (no glu.t() in, no dwc.t() out). Gated like the
            // channel-major fused path (CONV+SILU). Falls back to the channel-major path below (which
            // keeps the two transposes) when the time-major xclbin is absent or t > DW_T.
            #[cfg(feature = "npu")]
            let tmajor = if resident_conv && resident_silu {
                self.npu.as_ref().and_then(|npu| npu.npu_dwconv_silu_tmajor(&glu, &taps, &dwb))
            } else { None };
            #[cfg(not(feature = "npu"))]
            let tmajor: Option<Array2<f32>> = None;
            if let Some(f) = tmajor {
                return f; // [T,D] -- dwconv+silu applied on-NPU, transposes dissolved (step 3b)
            }
            // ---- fallback: channel-major fused / separate bricks / host (transposes stay host) ----
            let glu_t = glu.t().to_owned(); // [T,D] -> [D,T]  (transpose 1, killed on the time-major path)
            // FUSED dwconv->SiLU (steps 3+4 in ONE xclbin, roadmap 5-A) when CONV+SILU are on + the fused
            // brick is present: one hw-context, the post-dwconv SiLU runs device-to-device (no second
            // switch, no host bridge). Returns silu(dwconv(glu_t)) [D,T] directly. Falls back to the
            // separate dwconv + silu path below (then host) if the fused xclbin is absent.
            #[cfg(feature = "npu")]
            let fused = if resident_conv && resident_silu {
                self.npu.as_ref().and_then(|npu| npu.npu_dwconv_silu(&glu_t, &taps, &dwb))
            } else { None };
            #[cfg(not(feature = "npu"))]
            let fused: Option<Array2<f32>> = None;
            let dwc = if let Some(f) = fused {
                f // [D,T] -- dwconv+silu already applied on-NPU, one hw-context
            } else {
                // dwconv on NPU (step 3a, host-fed [D,T]) when the resident conv path is on + the brick is
                // present + T<=400; else the host FIR. Transposes stay host here (cut in 3b).
                #[cfg(feature = "npu")]
                let dw_npu = if resident_conv {
                    self.npu.as_ref().and_then(|npu| npu.npu_dwconv1d(&glu_t, &taps, &dwb))
                } else { None };
                #[cfg(not(feature = "npu"))]
                let dw_npu: Option<Array2<f32>> = None;
                let mut dwc = dw_npu.unwrap_or_else(|| dwconv1d(&glu_t, &taps, &dwb, 9)); // [D, T]
                // SiLU on NPU (step 4) as a SEPARATE brick, DEFAULT-ON with the resident conv path (opt out
                // PARAKEET_RESIDENT_SILU=0 -> host silu). dwc is [D=C, T] channel-major == the silu brick's
                // [C,T] shape. (Separate brick, NOT a dwconv epilogue -- the fused epilogue miscompiles
                // alternate channels on this toolchain; see dwconv-fused-epilogue-alt-channel-miscompile.)
                // The on-NPU silu is bf16-tanh precision; that precision is WER-IRRELEVANT (~8.5 accepted as
                // the resident baseline, the 8.2 host-silu delta is a 17-clip decoder-chaos artifact). The
                // separate opt-out preserves the clean host-silu path for the future WER-refinement pass.
                #[cfg(feature = "npu")]
                let silu_npu = if resident_conv && resident_silu {
                    self.npu.as_ref().and_then(|npu| npu.npu_silu(&dwc))
                } else { None };
                #[cfg(not(feature = "npu"))]
                let silu_npu: Option<Array2<f32>> = None;
                silu_npu.unwrap_or_else(|| { silu_inplace(&mut dwc); dwc })
            };
            dwc.t().to_owned() // [D,T] -> [T,D]  (transpose 2, killed on the time-major path)
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
        // mhsa now does its own norm_self_att LN (resident LN->QKV seam or host fallback), so pass
        // pre-LN x -- mirroring conv_module. The residual below still adds mhsa_out to pre-LN x.
        let mhsa_out = prof::time("mhsa", || self.mhsa(&x, blk, pos_enc));
        {
            let _h = PhaseScope::new("residual", Bucket::Host);
            x = &x + &mhsa_out;
        }
        // conv_module now does its own norm_conv LN (resident seam or host fallback), so pass pre-LN x.
        let conv_out = prof::time("conv_mod", || self.conv_module(&x, blk));
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
