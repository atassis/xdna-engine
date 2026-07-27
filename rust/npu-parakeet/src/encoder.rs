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

    /// bf16-checkpoint fast path for a plain (non-K-split, K=KRES) weight matmul: when `b.bf16_m(key)`
    /// yields pre-packed bf16 bits (NPU_WEIGHTS_CHECKPOINT loaded), dispatch straight from those bits,
    /// skipping the host f32->bf16 pack entirely on a cache miss. Returns `None` on the npy path
    /// (bf16_m always None there) or without the NPU feature/instance, so the caller falls back to
    /// the existing `mm_lazy(.., || b.m(key), id)` path unchanged.
    fn mm_checkpoint(&self, a: &Array2<f32>, b: &BlockWeights, key: &str, id: &str) -> Option<Array2<f32>> {
        #[cfg(feature = "npu")]
        {
            if let Some(npu) = &self.npu {
                if let Some((k, n, bits)) = b.bf16_m(key) {
                    return Some(npu.matmul_id_bf16(a, id, k, n, bits));
                }
            }
        }
        #[cfg(not(feature = "npu"))]
        {
            let _ = (a, b, key, id);
        }
        None
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
                        // GATE (PARAKEET_FFN_DEVACC): accumulate fc2 ON-DEVICE (acc_add brick) then read
                        // back at the FFN boundary -- ONLY the accumulation moved on-device (block
                        // dataflow unchanged). A WER-neutral result proves the device-accumulate is
                        // bit-identical to the host K-split. Falls through to resident_ffn if acc_add absent.
                        let id1 = format!("{blk}.{tag}.l1");
                        let id2 = format!("{blk}.{tag}.l2");
                        if std::env::var("PARAKEET_FFN_DEVACC").map(|v| v != "0").unwrap_or(false) {
                            if let Some(out) = npu.resident_ffn_devacc_readback(x, gamma.as_slice().unwrap(), beta.as_slice().unwrap(),
                                || b.m(l1), &id1,
                                || b.m(l2), &id2) {
                                return out;
                            }
                        }
                        // bf16-checkpoint fast path (NPU_WEIGHTS_CHECKPOINT): fc1/fc2 are both baked bf16
                        // verbatim [K,N], so when both are available skip the host f32->bf16 pack
                        // entirely on a cache miss. Falls back to the f32 path below (byte-identical
                        // to before) when the checkpoint isn't loaded (bf16_m always None on npy).
                        if let (Some((k1, n1, bits1)), Some((k2, n2, bits2))) =
                            (b.bf16_m(l1), b.bf16_m(l2))
                        {
                            return npu.resident_ffn_bf16(x, gamma.as_slice().unwrap(), beta.as_slice().unwrap(),
                                &id1, k1, n1, bits1, &id2, k2, n2, bits2);
                        }
                        return npu.resident_ffn(x, gamma.as_slice().unwrap(), beta.as_slice().unwrap(),
                            || b.m(l1), &id1,
                            || b.m(l2), &id2);
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

    /// Snapshot of the per-kernel dispatch accounting, for differencing across a single encode.
    /// The interleaved A/B harness diffs this before/after each clip so a tag's cost is attributed
    /// to the ARM that ran it, which is what makes drift-cancelled comparison possible at all.
    #[cfg(feature = "npu")]
    pub fn npu_tag_snapshot(&self) -> std::collections::BTreeMap<&'static str, (usize, f64)> {
        self.npu.as_ref().map(|n| n.stats.borrow().by_tag.clone()).unwrap_or_default()
    }

    /// NPU timing breakdown (feature `npu`, NPU path only).
    #[cfg(feature = "npu")]
    pub fn npu_stats_string(&self) -> Option<String> {
        self.npu.as_ref().map(|n| {
            let s = n.stats.borrow();
            // Per-kernel dispatch split, costliest first. The aggregate alone says the fused path is
            // slower without saying WHICH dispatch is slower -- and every cost model built on the
            // aggregate has been falsified so far, so print the split by default.
            let mut tags: Vec<_> = s.by_tag.iter().map(|(k, (n, t))| (*k, *n, *t)).collect();
            tags.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
            let split: String = tags
                .iter()
                .map(|(k, n, t)| format!(" {k}={t:.2}s/{n}"))
                .collect();
            let sw = if s.submit_s > 0.0 || s.wait_s > 0.0 {
                format!("\nnpu submit/wait split: submit={:.2}s wait={:.2}s", s.submit_s, s.wait_s)
            } else { String::new() };
            format!(
                "npu breakdown: calls={} dispatches={} weight_load={:.2}s pack_a={:.2}s dispatch={:.2}s read={:.2}s accum={:.2}s\nnpu dispatch split:{}{}",
                s.calls, s.dispatches, s.weight_load_s, s.pack_a_s, s.dispatch_s, s.read_s, s.accum_s, split, sw
            )
        })
    }

    /// conv2D ÷8 dw-striding subsample: mel [128, T] -> [T/8, hidden].
    /// ONNX feeds conv as [b,1,time,freq]; flatten is [time, C*freq] (Transpose [0,2,1,3]).
    pub fn subsample(&self, mel: &Array2<f32>) -> Array2<f32> {
        // Whole subsample stem is host math (conv2d x5 + relu + flatten + final gemm);
        // no self.mm()/device call lives here, so one Host leaf scope cannot double-count.
        let _h = PhaseScope::new("subsample", Bucket::Host);
        let pe4 = |k: &str| self.w.pre(k).into_dimensionality::<Ix4>().unwrap();
        let pe1 = |k: &str| self.w.pre(k).into_dimensionality::<Ix1>().unwrap();
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
        let wout = self.w.pre("out.weight").into_dimensionality::<Ix2>().unwrap(); // [4096, hidden]
        let bout = self.w.pre("out.bias").into_dimensionality::<Ix1>().unwrap();
        prof::phase::set_stage("subsample"); // final gemm (host .dot here; labels device path if ever routed via mm)
        flat.dot(&wout) + &bout
    }

    fn mhsa(&self, x: &Array2<f32>, blk: usize, pos_enc: &Array2<f32>) -> Array2<f32> {
        let b = self.w.block(blk);
        let t = x.nrows();
        // mhsa_wprep: materialize each (T'-independent) attention projection weight for its mm().
        // Each `b.m()` clones the whole [D,D]/[P,D] matrix out of the weight map -- pure host data
        // movement (no math, no device). Materialized LAZILY inside mm_lazy's closure: on a warm
        // (weight-BO cache hit) pass the clone never runs, eliminating the per-pass reclone.
        // Each projection prefers the bf16-checkpoint fast path (NPU_WEIGHTS_CHECKPOINT); `mm_checkpoint` returns
        // None on the npy path, so the `unwrap_or_else` fallback is the pre-existing f32 mm_lazy
        // call, byte-identical to before.
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
        let resident_qkv: Option<(Array2<f32>, Array2<f32>, Array2<f32>)> = if resident_mha_qkv
            && self.npu.as_ref().map(|n| n.resident_ff_available()).unwrap_or(false) {
            let npu = self.npu.as_ref().unwrap();
            let gamma = b.v("norm_self_att.weight");
            let beta = b.v("norm_self_att.bias");
            // bf16x2 device-A is ON by default for the resident path (opt-out PARAKEET_MHA_SPLITA=0 ->
            // old single-bf16-A path, WER 8.9). It makes the opt-in resident MHA WER-NEUTRAL (8.5),
            // unblocking the (owner-deferred) DEFAULT flip. The shipped host-MHA default is untouched.
            let split_a = std::env::var("PARAKEET_MHA_SPLITA").map(|v| v != "0").unwrap_or(true);
            if split_a {
                // Split the DEVICE ctxLN affine_LN(x) into A_hi + A_lo (near-f32) and feed the QKV modal
                // GEMM twice (q = A_hi@W + A_lo@W), summing on host. The +0.4pp seam was the bf16-round
                // coin-flip of the device A (host golden HOSTQKV=8.5 vs full-resident bf16-A 8.9); near-f32
                // A recovers 8.5. Reuses mm_lazy unchanged -> no shared default-infra touch. NOTE: reads the
                // device A back to host to split it (a residency step-back, compute stays on-NPU); the clean
                // follow-on is a device-side affine_cast_split (A_hi+A_lo on-device, no readback).
                let _hh = PhaseScope::new("mha_resident_qkv", Bucket::Npu);
                let a = npu.resident_mha_affine_ln_f32(x, gamma.as_slice().unwrap(), beta.as_slice().unwrap());
                let a_hi = a.mapv(|v| npu_xrt::bf16_bits_to_f32(npu_xrt::f32_to_bf16_bits(v)));
                let a_lo = &a - &a_hi;
                let q = { let hi = self.mm_lazy(&a_hi, || b.m("self_attn.linear_q.weight"), &format!("{blk}.q"));
                          let lo = self.mm_lazy(&a_lo, || b.m("self_attn.linear_q.weight"), &format!("{blk}.q")); hi + &lo };
                let k = { let hi = self.mm_lazy(&a_hi, || b.m("self_attn.linear_k.weight"), &format!("{blk}.k"));
                          let lo = self.mm_lazy(&a_lo, || b.m("self_attn.linear_k.weight"), &format!("{blk}.k")); hi + &lo };
                let v = { let hi = self.mm_lazy(&a_hi, || b.m("self_attn.linear_v.weight"), &format!("{blk}.v"));
                          let lo = self.mm_lazy(&a_lo, || b.m("self_attn.linear_v.weight"), &format!("{blk}.v")); hi + &lo };
                Some((q, k, v))
            } else {
                let _hh = PhaseScope::new("mha_resident_qkv", Bucket::Npu);
                Some(npu.resident_mha_ln_qkv(x, gamma.as_slice().unwrap(), beta.as_slice().unwrap(),
                    || { let _wp = PhaseScope::new("mhsa_wprep", Bucket::Marshal); b.m("self_attn.linear_q.weight") }, &format!("{blk}.q"),
                    || { let _wp = PhaseScope::new("mhsa_wprep", Bucket::Marshal); b.m("self_attn.linear_k.weight") }, &format!("{blk}.k"),
                    || { let _wp = PhaseScope::new("mhsa_wprep", Bucket::Marshal); b.m("self_attn.linear_v.weight") }, &format!("{blk}.v")))
            }
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
            // Each projection prefers the bf16-checkpoint fast path (NPU_WEIGHTS_CHECKPOINT); `mm_checkpoint`
            // returns None on the npy path, so the fallback is the pre-existing f32 mm_lazy call.
            let id_q = format!("{blk}.q");
            let q = self.mm_checkpoint(&ln_x, b, "self_attn.linear_q.weight", &id_q)
                .unwrap_or_else(|| self.mm_lazy(&ln_x, || { let _wp = PhaseScope::new("mhsa_wprep", Bucket::Marshal); b.m("self_attn.linear_q.weight") }, &id_q)); // [T, D]
            let id_k = format!("{blk}.k");
            let k = self.mm_checkpoint(&ln_x, b, "self_attn.linear_k.weight", &id_k)
                .unwrap_or_else(|| self.mm_lazy(&ln_x, || { let _wp = PhaseScope::new("mhsa_wprep", Bucket::Marshal); b.m("self_attn.linear_k.weight") }, &id_k));
            let id_v = format!("{blk}.v");
            let v = self.mm_checkpoint(&ln_x, b, "self_attn.linear_v.weight", &id_v)
                .unwrap_or_else(|| self.mm_lazy(&ln_x, || { let _wp = PhaseScope::new("mhsa_wprep", Bucket::Marshal); b.m("self_attn.linear_v.weight") }, &id_v));
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
        let id_pos = format!("{blk}.pos");
        let pm = self.mm_checkpoint(pos_enc, b, "self_attn.linear_pos.weight", &id_pos)
            .unwrap_or_else(|| self.mm_lazy(pos_enc, || { let _wp = PhaseScope::new("mhsa_wprep", Bucket::Marshal); b.m("self_attn.linear_pos.weight") }, &id_pos)); // [P, D]
        let ctx = self.attention_core(&q, &k, &v, &pm, blk, t);
        // linear_out moved to this caller in main's mhsa/attention_core split, so the checkpoint fast
        // path attaches here rather than inside the attention body as on the original branch.
        prof::phase::set_stage("mhsa_qkv");
        let id_out = format!("{blk}.out");
        self.mm_checkpoint(&ctx, b, "self_attn.linear_out.weight", &id_out)
            .unwrap_or_else(|| self.mm_lazy(&ctx, || { let _wp = PhaseScope::new("mhsa_wprep", Bucket::Marshal); b.m("self_attn.linear_out.weight") }, &id_out))
    }

    /// Device-in MHSA for the fused seam: the attention input is the resident-stream LN output
    /// `satt_bo` (device bf16 [PAD_M,KRES]); q/k/v are projected DEVICE-IN (no host re-upload of satt).
    /// pm stays host-in (pos_enc != satt). attention_core runs host; the OUTPUT linear_out lands in a
    /// DEVICE BO ([`NpuMatmul::matmul_id_to_bo`]) so the MHSA output stays resident for the conv seam.
    #[cfg(feature = "npu")]
    fn mhsa_dev(&self, satt_bo: &npu_xrt::Bo, m: usize, blk: usize, pos_enc: &Array2<f32>) -> std::rc::Rc<npu_xrt::Bo> {
        let b = self.w.block(blk);
        let npu = self.npu.as_ref().expect("mhsa_dev without npu");
        let d = self.cfg.hidden;
        // FULL DEVICE-IN/OUT interior (opt-in PARAKEET_MHSA_DEV_IO=1): q/k/v never come back to host
        // and ctx never goes back up -- the BD-onchip conveyor's --stream-io taps read the GEMM
        // outputs in place and drain ctx straight into linear_out's A. This deletes the last host
        // excursion in the fused block's residual stream. Falls through to the read-back path below
        // when the artifacts/conveyor_bd_io xclbin is absent, so an unbuilt tree still runs.
        // PART OF THE DEFAULT FULL-NPU RAIL since 2026-07-27 (opt out with PARAKEET_MHSA_DEV_IO=0).
        // Costs +144 command submissions/clip and ~+11% wall -- and is the most ACCURATE arm measured
        // (rel-L2 0.0866 vs the hybrid's 0.0891 and the pre-MHSA rail's 0.0886). Kept on because the
        // rail's purpose is a single-hardware graph; the wall is bought back by deleting command
        // submissions (one-xclbin-per-block), not by leaving ops on the host.
        if std::env::var("PARAKEET_MHSA_DEV_IO").map(|v| v != "0")
            .unwrap_or_else(|_| npu.full_npu_rail()) && npu.conveyor_bd_io_available() {
            prof::phase::set_stage("mhsa_pos");
            let id_pos = format!("{blk}.pos");
            let pm = self.mm_checkpoint(pos_enc, b, "self_attn.linear_pos.weight", &id_pos)
                .unwrap_or_else(|| self.mm_lazy(pos_enc, || b.m("self_attn.linear_pos.weight"), &id_pos));
            let (ubias, vbias) = (b.m("self_attn.pos_bias_u"), b.m("self_attn.pos_bias_v"));
            prof::phase::set_stage("mhsa_devio");
            let ctx_bo = npu.relpos_mha_conveyor_bdonchip_dev(
                satt_bo, m,
                || b.m("self_attn.linear_q.weight"), &format!("{blk}.q"),
                || b.m("self_attn.linear_k.weight"), &format!("{blk}.k"),
                || b.m("self_attn.linear_v.weight"), &format!("{blk}.v"),
                &pm, &ubias, &vbias, self.cfg.n_heads,
            );
            if let Some(ctx_bo) = ctx_bo {
                // linear_out device-in (A = the bf16 ctx BO) and device-out, same f32 [PAD_M,D]
                // contract the host-in path's matmul_id_to_bo returns, so the MHSA residual is unchanged.
                prof::phase::set_stage("mhsa_qkv");
                return npu.proj_from_bf16_to_bo(&ctx_bo, m, || b.m("self_attn.linear_out.weight"), &format!("{blk}.out"), d);
            }
        }
        prof::phase::set_stage("mhsa_qkv");
        let q = npu.proj_from_bf16(satt_bo, m, || b.m("self_attn.linear_q.weight"), &format!("{blk}.q"), d);
        prof::phase::set_stage("mhsa_qkv");
        let k = npu.proj_from_bf16(satt_bo, m, || b.m("self_attn.linear_k.weight"), &format!("{blk}.k"), d);
        prof::phase::set_stage("mhsa_qkv");
        let v = npu.proj_from_bf16(satt_bo, m, || b.m("self_attn.linear_v.weight"), &format!("{blk}.v"), d);
        prof::phase::set_stage("mhsa_pos");
        // pos stays host-in on the device-in path (pos_enc != satt), so it takes the checkpoint fast path
        // too. q/k/v go through `proj_from_bf16` and linear_out through `matmul_id_to_bo` -- both are
        // device-in/device-out and have no bf16-checkpoint sibling, so they are deliberately untouched.
        let id_pos = format!("{blk}.pos");
        let pm = self.mm_checkpoint(pos_enc, b, "self_attn.linear_pos.weight", &id_pos)
            .unwrap_or_else(|| self.mm_lazy(pos_enc, || b.m("self_attn.linear_pos.weight"), &id_pos));
        let ctx = self.attention_core(&q, &k, &v, &pm, blk, m);
        prof::phase::set_stage("mhsa_qkv");
        npu.matmul_id_to_bo(&ctx, || b.m("self_attn.linear_out.weight"), &format!("{blk}.out"), d)
    }

    /// The attention core shared by the host-in [`Self::mhsa`] and the device-in [`Self::mhsa_dev`]:
    /// given the projected q/k/v [T,D] and pos-projection pm [P,D], compute rel-pos scores -> softmax
    /// -> context -> merge -> linear_out. Identical to the pre-refactor mhsa tail (the 3 attention
    /// variants: resident / conveyor / host score loop).
    fn attention_core(&self, q: &Array2<f32>, k: &Array2<f32>, v: &Array2<f32>, pm: &Array2<f32>, blk: usize, t: usize) -> Array2<f32> {
        let b = self.w.block(blk);
        let (h, dk, d) = (self.cfg.n_heads, self.cfg.head_dim, self.cfg.hidden);
        let p = pm.nrows(); // 2T-1
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
                // Phase-2: ALL h heads in ONE dispatch (h parallel cores) -> ctx [t, d].
                let ctx = npu.relpos_mha_batched(&q, &k, &pm, &v, &ubias, &vbias);
                // A/B localizer (PARAKEET_MHA_AB=1): compare resident ctx vs f32 host golden for
                // head 0 AND a non-zero head (parallelism must not change per-head numerics).
                if std::env::var("PARAKEET_MHA_AB").is_ok() {
                    for &hh in &[0usize, (h / 2).min(h - 1)] {
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
                        let ch = ctx.slice(s![.., col..col + dk]).to_owned();
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
                        eprintln!("[MHA_AB] blk={blk} h{hh} T={t} ctx_relL2={:.4e} worst_row={} row_relL2={:.4e}",
                            (num / den).sqrt(), maxrow.0, maxrow.1);

                      // ---- PROBE (Ladder step 1, head-0 only): decompose the ~1% bf16 I/O quantization. Feed
                        // bf16-rounded operands into the SAME f32 host golden and measure ctx rel-L2
                        // vs pure-f32 (ch_host). Pure host math, no device -- isolates which rounding
                        // hop (AC inputs / BD inputs / probs narrow / V narrow / ctx-out narrow) owns
                        // the gap, and whether the full emulation reproduces the resident ~1.05e-2.
                        if hh == 0 {
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
                }
                prof::phase::set_stage("mhsa_qkv");
                return ctx; // merged [T,D]; linear_out applied by the caller (mhsa / mhsa_dev)
            }
        }

        // BD-ONCHIP CONVEYOR MHA (opt-in PARAKEET_CONVEYOR_MHA_BDONCHIP=1): the WIN variant -- same
        // 1-dispatch/block conveyor but BD = rel_shift((q+v[h])@p^T) is computed ON-CHIP (4th stage),
        // deleting the host BD precompute that made PARAKEET_CONVEYOR_MHA +19%. Needs the BD-onchip
        // xclbin (scripts/conveyor_bd_prebuild.sh --tactive-mask); falls back to the host score path
        // when the artifact is absent (relpos_mha_conveyor_bdonchip panics on load -> keep unset until
        // built). Checked BEFORE the host-BD conveyor so it wins when both flags are set.
        #[cfg(feature = "npu")]
        if std::env::var("PARAKEET_CONVEYOR_MHA_BDONCHIP").is_ok() {
            if let Some(npu) = &self.npu {
                let _h = PhaseScope::new("mhsa_conveyor_bd", Bucket::Npu);
                let ctx = npu.relpos_mha_conveyor_bdonchip(q, k, v, pm, &ubias, &vbias, h);
                prof::phase::set_stage("mhsa_qkv");
                return ctx; // merged [T,D]; linear_out applied by the caller (mhsa / mhsa_dev)
            }
        }

        // CONVEYOR MHA (opt-in PARAKEET_CONVEYOR_MHA=1): replace the per-head relpos_mha LOOP (8
        // dispatches) with ONE 8-head conveyor dispatch. The host packs the query belt (qu = q+u[h];
        // BD_shifted = rel_shift((q+v[h]) @ p^T), carriage per PARAKEET_CONVEYOR_BD -- default plain,
        // see scripts/conveyor_bd_precision_check.py). npu.relpos_mha_conveyor returns merged ctx
        // [T, D]; the 8-head xclbin dispatch inside it is a TODO stub until the artifact is built
        // (see CONVEYOR_INTEGRATION_RUNBOOK.md). Falls back to the host score path when unset.
        #[cfg(feature = "npu")]
        if std::env::var("PARAKEET_CONVEYOR_MHA").is_ok() {
            if let Some(npu) = &self.npu {
                let _h = PhaseScope::new("mhsa_conveyor", Bucket::Npu);
                let ctx = npu.relpos_mha_conveyor(q, k, v, pm, &ubias, &vbias, h);
                prof::phase::set_stage("mhsa_qkv");
                return ctx; // merged [T,D]; linear_out applied by the caller (mhsa / mhsa_dev)
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
        ctx // merged [T,D]; linear_out applied by the caller (mhsa / mhsa_dev)
    }

    /// `precomputed_glu`: Some when the caller (the fused seam) already ran the conv front DEVICE-IN
    /// (LN->pw1->GLU from a device BO) -- then the resident/host front here is skipped and the rest of
    /// the module (dwconv->silu->pw2) continues from it. None = the normal self-contained conv module.
    /// The conv module's pw2 tail, split out so a caller that already has the dwconv+SiLU output
    /// (the device-in conv seam) can finish the module without re-entering the LN/pw1/GLU/dwconv
    /// chain. Byte-identical to `conv_module`'s own tail.
    fn conv_pw2(&self, back: &Array2<f32>, blk: usize) -> Array2<f32> {
        let b = self.w.block(blk);
        prof::phase::set_stage("conv_pw");
        self.mm_lazy(back, || {
            let _wp = PhaseScope::new("conv_wprep", Bucket::Marshal);
            b.m3("conv.pointwise_conv2.weight").index_axis(Axis(2), 0).to_owned().t().to_owned()
        }, &format!("{blk}.pw2"))
    }

    fn conv_module(&self, x: &Array2<f32>, blk: usize, precomputed_glu: Option<Array2<f32>>) -> Array2<f32> {
        let b = self.w.block(blk);
        let d = self.cfg.hidden;
        let t = x.nrows();
        // RESIDENT conv module: the whole module can run resident -- LN -> pw1 (modal GEMM) -> GLU ->
        // dwconv -> silu (time-major [T,D], transposes dissolved) -> pw2 (modal GEMM), the activation
        // stream never touching host across the frontier. On-NPU SiLU is part of it.
        //
        // OPT-IN (PARAKEET_RESIDENT_CONV=1 / PARAKEET_RESIDENT_SILU=1). The originating branch had
        // these DEFAULT-ON, flipped on a 17-clip WER read (RU 8.5 / EN 8.6 / ALL 8.5). That flip is
        // held back deliberately and is NOT part of this merge, for two reasons: a shipped-default
        // flip is an owner decision, and the 17-clip greedy WER gate is chaotic at ~1e-5 -- it cannot
        // validate a device change of this kind, so it is the wrong instrument to flip on. The
        // capability lands here in full; flipping the default is a two-line change once a rel-L2
        // -vs-shipped number or a larger eval backs it.
        #[cfg(feature = "npu")]
        let resident_conv = std::env::var("PARAKEET_RESIDENT_CONV").map(|v| v != "0").unwrap_or(false);
        // (PARAKEET_RESIDENT_SILU is read inside conv_dwconv_silu, which owns the silu decision now.)
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
        let resident_glu = if resident_conv && precomputed_glu.is_none() {
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

        let glu = if let Some(g) = precomputed_glu {
            g // [T, D] -- conv front already ran DEVICE-IN in the fused seam
        } else if let Some(g) = resident_glu {
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
        let back = self.conv_dwconv_silu(&glu, &taps, &dwb);
        prof::phase::set_stage("conv_pw");
        // pw2 chain -> [D, D]: materialized lazily; skipped on warm passes.
        self.mm_lazy(&back, || {
            let _wp = PhaseScope::new("conv_wprep", Bucket::Marshal);
            b.m3("conv.pointwise_conv2.weight").index_axis(Axis(2), 0).to_owned().t().to_owned() // [D,D,1]->[D,D]->[D,D]
        }, &format!("{blk}.pw2"))
    }

    /// dwconv (+SiLU) over the GLU output: [T,D] -> [T,D]. Extracted so the host `conv_module` tail and
    /// the fused `conv_module_to_bo` tail (fusion T6) run the SAME conv middle and cannot drift apart.
    fn conv_dwconv_silu(&self, glu: &Array2<f32>, taps: &Array2<f32>, dwb: &Array1<f32>) -> Array2<f32> {
        #[cfg(feature = "npu")]
        let resident_conv = std::env::var("PARAKEET_RESIDENT_CONV").map(|v| v != "0").unwrap_or(false);
        #[cfg(feature = "npu")]
        let resident_silu = std::env::var("PARAKEET_RESIDENT_SILU").map(|v| v != "0").unwrap_or(false);
        // depthwise along time: [D, T]. Bracketing transposes + trailing SiLU are host math
        // with no mm() inside, so they fold into the conv_dwconv Host leaf scope.
        prof::time("dwconv", || {
            let _h = PhaseScope::new("conv_dwconv", Bucket::Host);
            // TIME-MAJOR fused dwconv+silu (step 3b): consumes GLU [T,D] DIRECTLY and emits [T,D]
            // DIRECTLY -- BOTH host transposes DISSOLVED (no glu.t() in, no dwc.t() out). Gated like the
            // channel-major fused path (CONV+SILU). Falls back to the channel-major path below (which
            // keeps the two transposes) when the time-major xclbin is absent or t > DW_T.
            #[cfg(feature = "npu")]
            let tmajor = if resident_conv && resident_silu {
                self.npu.as_ref().and_then(|npu| npu.npu_dwconv_silu_tmajor(glu, taps, dwb))
            } else { None };
            #[cfg(not(feature = "npu"))]
            let tmajor: Option<Array2<f32>> = None;
            if let Some(f) = tmajor {
                return f; // [T,D] -- dwconv+silu applied on-NPU, transposes dissolved (step 3b)
            }
            // ---- fallback: channel-major fused / separate bricks / host (transposes stay host) ----
            let glu_t = glu.t().to_owned(); // [T,D] -> [D,T]  (transpose 1, killed on the time-major path)
            // FUSED dwconv->SiLU (steps 3+4 in ONE xclbin) when CONV+SILU are on + the fused
            // brick is present: one hw-context, the post-dwconv SiLU runs device-to-device (no second
            // switch, no host bridge). Returns silu(dwconv(glu_t)) [D,T] directly. Falls back to the
            // separate dwconv + silu path below (then host) if the fused xclbin is absent.
            #[cfg(feature = "npu")]
            let fused = if resident_conv && resident_silu {
                self.npu.as_ref().and_then(|npu| npu.npu_dwconv_silu(&glu_t, taps, dwb))
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
                    self.npu.as_ref().and_then(|npu| npu.npu_dwconv1d(&glu_t, taps, dwb))
                } else { None };
                #[cfg(not(feature = "npu"))]
                let dw_npu: Option<Array2<f32>> = None;
                let mut dwc = dw_npu.unwrap_or_else(|| dwconv1d(&glu_t, taps, dwb, 9)); // [D, T]
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
        })
    }

    /// Fused-block conv tail (fusion T6): the conv middle + pw2 landing DIRECTLY in a device BO via
    /// `matmul_id_to_bo`, so the conv output never round-trips to host on its way to the conv residual.
    /// The host twin is `conv_module`'s `mm_lazy` tail; both share `conv_dwconv_silu` (so the conv middle
    /// cannot drift) and both key the same `{blk}.pw2` weight-BO cache, so warm passes hit either way.
    ///
    /// The dwconv middle itself is still HOST-bracketed (the GLU comes back to host from
    /// `resident_conv_pw1_glu_dev`), so this closes the conv->FFN2 seam, not the conv interior.
    #[cfg(feature = "npu")]
    fn conv_module_to_bo(&self, glu: &Array2<f32>, blk: usize) -> Option<std::rc::Rc<npu_xrt::Bo>> {
        let npu = self.npu.as_ref()?;
        let b = self.w.block(blk);
        let (taps, dwb) = {
            let _wp = PhaseScope::new("conv_wprep", Bucket::Marshal);
            let dw3 = b.m3("conv.depthwise_conv.weight"); // [D, 1, 9]
            (dw3.index_axis(Axis(1), 0).to_owned(), b.v("conv.depthwise_conv.bias"))
        };
        // The conv middle is the fused rail's surviving HOST excursion: pw1+GLU returns an ndarray,
        // dwconv+SiLU run on x86, then pw2 re-uploads. Labelled so it shows in the host profile --
        // previously only the inner `dwconv` was, which understated the excursion.
        let back = prof::time("conv_dw_silu", || self.conv_dwconv_silu(glu, &taps, &dwb));
        prof::phase::set_stage("conv_pw");
        Some(npu.matmul_id_to_bo(&back, || {
            let _wp = PhaseScope::new("conv_wprep", Bucket::Marshal);
            b.m3("conv.pointwise_conv2.weight").index_axis(Axis(2), 0).to_owned().t().to_owned() // [D,D,1]->[D,D]->[D,D]
        }, &format!("{blk}.pw2"), self.cfg.hidden))
    }

    /// Device-in / device-out conformer block (whole-encoder residency): consumes the previous
    /// block's f32 [PAD_M,KRES] output BO and returns this block's, so the activation stream never
    /// touches host between blocks. The chain is the same one `block()`'s fused branch runs; the only
    /// difference is the ends -- no `upload_stream` at entry, and the exit LN stores f32 to a BO
    /// (`ln_affine_f32_dev`) instead of narrowing to bf16 and reading back.
    #[cfg(feature = "npu")]
    fn block_dev(&self, a_bo: &npu_xrt::Bo, blk: usize, pos_enc: &Array2<f32>, m: usize) -> std::rc::Rc<npu_xrt::Bo> {
        let b = self.w.block(blk);
        let npu = self.npu.as_ref().expect("block_dev without npu");
        let _h = PhaseScope::new("fused_block_dev", Bucket::Npu);
        let ff1n_g = b.v("norm_feed_forward1.weight");
        let ff1n_b = b.v("norm_feed_forward1.bias");
        let satt_g = b.v("norm_self_att.weight");
        let satt_b = b.v("norm_self_att.bias");
        let ff1_bo = npu.resident_ffn_dev_bo(a_bo, ff1n_g.as_slice().unwrap(), ff1n_b.as_slice().unwrap(),
            || b.m("feed_forward1.linear1.weight"), &format!("{blk}.ff1.l1"),
            || b.m("feed_forward1.linear2.weight"), &format!("{blk}.ff1.l2")).expect("resident_ffn_dev_bo");
        // COLLAPSE 1 of the one-xclbin-per-block work: `resadd(0.5) -> satt LN` is ONE command, with
        // the intermediate residual handed core-to-core on-chip instead of through DDR.
        //
        // OPT-IN (`PARAKEET_RESADD_LN=1`). The 4-column shim-budget wash is FIXED: `a`(8) + `b`(8) +
        // `gb`(1) = 17 shim MM2S against the 16 the array has (AIE2 gives each ShimNOC 2 in/2 out and
        // NPU2 has 8 columns -- silicon, checked in AIETargetModel, not a placer limit), so the first
        // build fell to 4 columns = half the cores. resadd_ln_iron.py now stages ONE core pair's `a`
        // through a MemTile split(), which costs 7 channels instead of 8 and fits at 16 exactly; each
        // pair is also pinned to its own column so all 8 columns carry compute.
        //
        // NOT the RTP fix this comment used to prescribe. An RTP is indeed an L1 address rather than a
        // DMA channel, but aie/dialects/aie.py emits ONE npu_write_rtp per INDEX and rejects slicing,
        // so a [2*KRES] gamma|beta costs 2048 write32 per LN core -- ~200 KB of instruction stream per
        // command on a brick whose whole point is deleting bytes. mm_silu_epilogue.cc's RTP carries one
        // i32 mode word; that precedent does not extend to 8 KB.
        //
        // Measured, same session, drift-normalised against the untouched kernels: 1.899 ms/command vs
        // 2.328 for the resadd+ln_affcast pair it replaces (was a wash at 4 columns). Still OPT-IN
        // because at ~0.43 ms x 1 site x 24 blocks it is ~9 ms/clip, inside the wall-clock noise --
        // it pays properly only once the other sites fold too.
        let fused_rl = std::env::var("PARAKEET_RESADD_LN").map(|v| v == "1").unwrap_or(false);
        let (x_bo, satt_bo) = match fused_rl
            .then(|| npu.resadd_ln_dev(a_bo, &ff1_bo, 0.5, satt_g.as_slice().unwrap(), satt_b.as_slice().unwrap()))
            .flatten()
        {
            Some(pair) => pair,
            None => {
                let x_bo = npu.residual_add_dev(a_bo, &ff1_bo, 0.5, m).expect("residual_add_dev(0.5)");
                let satt_bo = npu.ln_affine_cast_dev_bf16(&x_bo, satt_g.as_slice().unwrap(), satt_b.as_slice().unwrap()).expect("ln_affine_cast_dev");
                (x_bo, satt_bo)
            }
        };
        let mhsa_out_bo = self.mhsa_dev(&satt_bo, m, blk, pos_enc);
        let conv_in_bo = npu.residual_add_dev(&x_bo, &mhsa_out_bo, 1.0, m).expect("residual_add_dev(1.0)");
        let conv_g = b.v("norm_conv.weight");
        let conv_b = b.v("norm_conv.bias");
        let glu = npu.resident_conv_pw1_glu_dev(&conv_in_bo, m, conv_g.as_slice().unwrap(), conv_b.as_slice().unwrap(),
            || b.m3("conv.pointwise_conv1.weight").index_axis(Axis(2), 0).to_owned().t().to_owned(), &format!("{blk}.pw1"))
            .expect("resident_conv_pw1_glu_dev (block_dev requires the glu xclbin)");
        let conv_out_bo = self.conv_module_to_bo(&glu, blk).expect("conv_module_to_bo");
        let x_bo = npu.residual_add_dev(&conv_in_bo, &conv_out_bo, 1.0, m).expect("residual_add_dev(conv 1.0)");
        let ff2n_g = b.v("norm_feed_forward2.weight");
        let ff2n_b = b.v("norm_feed_forward2.bias");
        let ff2_bo = npu.resident_ffn_dev_bo(&x_bo, ff2n_g.as_slice().unwrap(), ff2n_b.as_slice().unwrap(),
            || b.m("feed_forward2.linear1.weight"), &format!("{blk}.ff2.l1"),
            || b.m("feed_forward2.linear2.weight"), &format!("{blk}.ff2.l2")).expect("resident_ffn_dev_bo(ff2)");
        let x_bo = npu.residual_add_dev(&x_bo, &ff2_bo, 0.5, m).expect("residual_add_dev(ff2 0.5)");
        let out_g = b.v("norm_out.weight");
        let out_b = b.v("norm_out.bias");
        npu.ln_affine_f32_dev(&x_bo, out_g.as_slice().unwrap(), out_b.as_slice().unwrap())
            .expect("ln_affine_f32_dev (block_dev requires the f32-out block-exit LN)")
    }

    fn block(&self, x: &Array2<f32>, blk: usize, pos_enc: &Array2<f32>) -> Array2<f32> {
        let b = self.w.block(blk);
        // block_io: the [T', D] residual-stream clone at block entry (a working copy the residual
        // adds mutate). T'-dependent host data movement; scoped LEAF so it doesn't leak to residual.
        let mut x = {
            let _h = PhaseScope::new("block_io", Bucket::Marshal);
            x.clone()
        };
        // FUSED-BLOCK seam (opt-in PARAKEET_FUSED_BLOCK): keep the [T,D] activation RESIDENT across
        // FFN1 -> Macaron residual -> satt-LN -> MHSA q/k/v projections (no host round-trip inside that
        // frontier). For THIS seam, after MHSA read back to host and rejoin the default conv/FFN2 path.
        // Falls through to the default path when the fused bricks (acc_add/resadd/resident-ln) are absent.
        #[cfg(feature = "npu")]
        if self.npu.as_ref().map(|n| n.full_npu_rail()).unwrap_or(false) {
            if let Some(npu) = &self.npu {
                // Whole-encoder residency: when the f32-out block-exit LN exists, forward_last drives
                // `block_dev` in a loop and this per-block upload/readback pair never runs at all.
                if npu.resident_encoder_available() {
                    let m = x.nrows();
                    let a_bo = npu.upload_stream(&x);
                    let out_bo = self.block_dev(&a_bo, blk, pos_enc, m);
                    return npu.readback_stream(&out_bo, m);
                }
                if npu.resident_fused_available() {
                    let _h = PhaseScope::new("fused_ff1_mhsa", Bucket::Npu);
                    let m = x.nrows();
                    let ff1n_g = b.v("norm_feed_forward1.weight");
                    let ff1n_b = b.v("norm_feed_forward1.bias");
                    let satt_g = b.v("norm_self_att.weight");
                    let satt_b = b.v("norm_self_att.bias");
                    let x_bo = npu.upload_stream(&x);
                    let ff1_bo = npu.resident_ffn_dev_bo(&x_bo, ff1n_g.as_slice().unwrap(), ff1n_b.as_slice().unwrap(),
                        || b.m("feed_forward1.linear1.weight"), &format!("{blk}.ff1.l1"),
                        || b.m("feed_forward1.linear2.weight"), &format!("{blk}.ff1.l2")).expect("resident_ffn_dev_bo");
                    let x_bo = npu.residual_add_dev(&x_bo, &ff1_bo, 0.5, m).expect("residual_add_dev(0.5)");
                    let satt_bo = npu.ln_affine_cast_dev_bf16(&x_bo, satt_g.as_slice().unwrap(), satt_b.as_slice().unwrap()).expect("ln_affine_cast_dev");
                    // MHSA output stays on device (linear_out -> device BO).
                    let mhsa_out_bo = self.mhsa_dev(&satt_bo, m, blk, pos_enc);
                    // MHSA residual ON-DEVICE: conv_in = (x + 0.5*ff1) + mhsa_out (scale 1.0), no host round-trip.
                    let conv_in_bo = npu.residual_add_dev(&x_bo, &mhsa_out_bo, 1.0, m).expect("residual_add_dev(1.0)");
                    // Conv FRONT device-in: norm_conv LN + pw1 + GLU consume conv_in_bo directly (no host
                    // round-trip between MHSA and conv); returns the host GLU output for the conv rest.
                    let conv_g = b.v("norm_conv.weight");
                    let conv_b = b.v("norm_conv.bias");
                    // CONV-MIDDLE device-in (opt-in PARAKEET_CONV_DEV_IO=1): carry the frontier past GLU
                    // to the dwconv+SiLU output, so the GLU readback AND the dwconv re-upload both
                    // dissolve (the cast that the bf16 dtype change needs anyway also does the 'same'
                    // top pad). pw2 onward stays host. Unset / bricks absent -> the GLU hand-off below.
                    let dw_dev = if std::env::var("PARAKEET_CONV_DEV_IO").is_ok() {
                        let dw3 = b.m3("conv.depthwise_conv.weight");
                        let taps = dw3.index_axis(Axis(1), 0).to_owned();
                        let dwb = b.v("conv.depthwise_conv.bias");
                        npu.resident_conv_pw1_glu_dw_dev(&conv_in_bo, m, conv_g.as_slice().unwrap(), conv_b.as_slice().unwrap(),
                            || b.m3("conv.pointwise_conv1.weight").index_axis(Axis(2), 0).to_owned().t().to_owned(),
                            &format!("{blk}.pw1"), &taps, &dwb)
                    } else { None };
                    let glu = if dw_dev.is_some() { None } else {
                        npu.resident_conv_pw1_glu_dev(&conv_in_bo, m, conv_g.as_slice().unwrap(), conv_b.as_slice().unwrap(),
                            || b.m3("conv.pointwise_conv1.weight").index_axis(Axis(2), 0).to_owned().t().to_owned(), &format!("{blk}.pw1"))
                    };
                    // T6 BLOCK CLOSE (device all the way to the block exit). Taken only when the conv
                    // middle did NOT go device-side, because `conv_module_to_bo` continues from the HOST
                    // `glu` array. Running BOTH -- a device dwconv+SiLU output feeding a device pw2 --
                    // is the remaining seam (task 7c), not something this merge can assume.
                    let glu = npu.resident_conv_pw1_glu_dev(&conv_in_bo, m, conv_g.as_slice().unwrap(), conv_b.as_slice().unwrap(),
                        || b.m3("conv.pointwise_conv1.weight").index_axis(Axis(2), 0).to_owned().t().to_owned(), &format!("{blk}.pw1"));
                    // ---- T6: conv rest -> conv residual -> FFN2 -> Macaron residual -> block-exit LN ----
                    // The whole tail stays in device BOs; the block's ONE readback is the block-exit LN.
                    // Needs the device-out conv tail, which needs the device-in conv front's GLU: when the
                    // GLU xclbin is absent `glu` is None and we take the T5 host rejoin instead.
                    if let Some(glu) = glu.as_ref() {
                        if let Some(conv_out_bo) = self.conv_module_to_bo(glu, blk) {
                            // conv residual ON-DEVICE: x = conv_in + conv_out (scale 1.0).
                            let x_bo = npu.residual_add_dev(&conv_in_bo, &conv_out_bo, 1.0, m).expect("residual_add_dev(conv 1.0)");
                            // FFN2 device-in/device-out -- its norm_feed_forward2 LN runs device-in too.
                            let ff2n_g = b.v("norm_feed_forward2.weight");
                            let ff2n_b = b.v("norm_feed_forward2.bias");
                            let ff2_bo = npu.resident_ffn_dev_bo(&x_bo, ff2n_g.as_slice().unwrap(), ff2n_b.as_slice().unwrap(),
                                || b.m("feed_forward2.linear1.weight"), &format!("{blk}.ff2.l1"),
                                || b.m("feed_forward2.linear2.weight"), &format!("{blk}.ff2.l2")).expect("resident_ffn_dev_bo(ff2)");
                            // Macaron 0.5 residual ON-DEVICE.
                            let x_bo = npu.residual_add_dev(&x_bo, &ff2_bo, 0.5, m).expect("residual_add_dev(ff2 0.5)");
                            let out_g = b.v("norm_out.weight");
                            let out_b = b.v("norm_out.bias");
                            // Block-exit LN. Device by default (the frontier runs to the block boundary);
                            // PARAKEET_FUSED_OUTLN_HOST=1 keeps the f32 host LN so the bf16 block-boundary
                            // rounding can be A/B'd against it on rel-L2 rather than assumed harmless.
                            if !std::env::var("PARAKEET_FUSED_OUTLN_HOST").map(|v| v != "0").unwrap_or(false) {
                                if let Some(out) = npu.ln_affine_cast_dev_readback(&x_bo, out_g.as_slice().unwrap(), out_b.as_slice().unwrap(), m) {
                                    return out;
                                }
                            }
                            let x = npu.readback_stream(&x_bo, m);
                            let _h = PhaseScope::new("ln", Bucket::Host);
                            return layernorm(&x, &out_g, &out_b);
                        }
                    }
                    // rejoin host: x = (x + 0.5*ff1) + mhsa_out (the conv residual base), then conv rest / FFN2 / out.
                    let mut x = npu.readback_stream(&conv_in_bo, m);
                    let conv_out = match dw_dev {
                        // dwconv+SiLU already done on device -> only the pw2 chain remains.
                        Some(back) => prof::time("conv_mod", || self.conv_pw2(&back, blk)),
                        None => prof::time("conv_mod", || self.conv_module(&x, blk, glu)),
                    };
                    x = &x + &conv_out;
                    let ff2 = prof::time("ff", || self.feed_forward(&x, b, blk, "ff2", "norm_feed_forward2.weight", "norm_feed_forward2.bias",
                                                "feed_forward2.linear1.weight", "feed_forward2.linear2.weight"));
                    x = x + ff2.mapv(|v| 0.5 * v);
                    return layernorm(&x, &b.v("norm_out.weight"), &b.v("norm_out.bias"));
                }
            }
        }
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
        let conv_out = prof::time("conv_mod", || self.conv_module(&x, blk, None));
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
        // WHOLE-ENCODER RESIDENCY: upload the stream ONCE, run all 24 blocks device-to-device, read
        // back ONCE. The per-block upload/readback pair disappears entirely (24 of each per clip).
        #[cfg(feature = "npu")]
        if self.npu.as_ref().map(|n| n.full_npu_rail()).unwrap_or(false) {
            if let Some(npu) = &self.npu {
                if npu.resident_encoder_available() {
                    let m = x.nrows();
                    let _h = PhaseScope::new("resident_encoder", Bucket::Npu);
                    let mut bo = npu.upload_stream(&x);
                    for blk in 0..self.cfg.n_layers {
                        bo = self.block_dev(&bo, blk, &pos_enc, m);
                    }
                    return npu.readback_stream(&bo, m);
                }
            }
        }
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
