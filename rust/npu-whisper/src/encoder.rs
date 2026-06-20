//! Whisper-small encoder (host f32 reference).
//!
//! Conv stem: mel [n_mels, 3000] -> conv1(k3,s1,p1)+GELU -> [d_model, 3000]
//!            -> conv2(k3,s2,p1)+GELU -> [d_model, 1500] -> transpose -> [1500, d_model]
//!            -> + embed_positions[:1500].
//! Each block is PRE-NORM:
//!   ln1 = LN(x); attn = mha(ln1·Wq+bq, ln1·Wk+bk, ln1·Wv+bv)·Wout+bout; x = x + attn
//!   ln2 = LN(x); f = gelu(ln2·Wfc1+bfc1)·Wfc2+bfc2; x = x + f
//! `forward_last` applies the final ln_post LayerNorm after the last block.

use std::path::Path;

use ndarray::prelude::*;
use npu_asr_host::{gelu, im2col_conv1d, layer_norm, mha};

use crate::config::WhisperCfg;
use crate::weights::{TensorMap, WhisperWeights};

const LN_EPS: f32 = 1e-5;

// --- B3: env-gated per-op encoder timing (ENC_PEROP_TIMING=1). Off by default -> production untouched.
// Accumulates host-visible wall ms per stage across all 12 layers; printed by forward_last.
thread_local! {
    static ENC_PEROP: std::cell::RefCell<Option<Vec<(&'static str, f64)>>> =
        const { std::cell::RefCell::new(None) };
}
#[inline]
fn perop_add(stage: &'static str, dt_ms: f64) {
    ENC_PEROP.with(|p| {
        if let Some(v) = p.borrow_mut().as_mut() {
            if let Some(e) = v.iter_mut().find(|(s, _)| *s == stage) { e.1 += dt_ms; }
            else { v.push((stage, dt_ms)); }
        }
    });
}
/// Time an expression into `stage` (no-op unless ENC_PEROP is initialized).
macro_rules! timed {
    ($stage:expr, $e:expr) => {{
        let __t = std::time::Instant::now();
        let __r = $e;
        perop_add($stage, __t.elapsed().as_secs_f64() * 1e3);
        __r
    }};
}

pub struct WhisperEncoder {
    pub cfg: WhisperCfg,
    w: WhisperWeights,
    #[cfg(feature = "npu")]
    npu: Option<crate::npu::WhisperNpu>,
    #[cfg(feature = "npu")]
    block_ops: Vec<crate::npu::BlockOps>,
    // NPU_ENC_MHA_NPU=1: full attention on the NPU (static-shape MHA xclbin) instead of host `mha`.
    // Default None -> host path (production untouched). Gated + WER-validated separately.
    #[cfg(feature = "npu")]
    mha_npu: Option<crate::mha_npu::MhaNpu>,
    // NPU_ENC_CONV_NPU=1: conv stem (conv1/conv2) as M-stationary GEMM on the NPU (reuses the prebuilt
    // 512x768x768 band) instead of host im2col_conv1d. GELU/transpose stay host. Gated; default None.
    #[cfg(feature = "npu")]
    conv_npu: Option<npu_asr::conv_npu::ConvNpu>,
}

impl WhisperEncoder {
    pub fn new(artifacts: &Path, cfg: WhisperCfg) -> Self {
        let w = WhisperWeights::load(artifacts).expect("load whisper weights");
        assert_eq!(w.nblocks(), cfg.n_layers, "block count mismatch");
        WhisperEncoder {
            cfg,
            w,
            #[cfg(feature = "npu")]
            npu: None,
            #[cfg(feature = "npu")]
            block_ops: Vec::new(),
            #[cfg(feature = "npu")]
            mha_npu: None,
            #[cfg(feature = "npu")]
            conv_npu: None,
        }
    }

    /// NPU-backed encoder: loads the same weights as `new`, opens the NPU + resident ctx2 kernel,
    /// and precomputes the per-block matmul ops. `root` is the worktree root (where
    /// `mlir-aie/.../whole_array/build` resolves); `artifacts` points at `whisper-small/`.
    #[cfg(feature = "npu")]
    pub fn new_npu(artifacts: &Path, cfg: WhisperCfg, root: &Path) -> Self {
        use npu_asr::ctx2::{CtxAOp, Epi, FfnMm2};

        let w = WhisperWeights::load(artifacts).expect("load whisper weights");
        assert_eq!(w.nblocks(), cfg.n_layers, "block count mismatch");

        let npu = crate::npu::WhisperNpu::open(root);
        let shared = npu.shared.clone();

        let block_ops = (0..cfg.n_layers)
            .map(|i| {
                let bw: &TensorMap = w.block(i);
                // q/k/v/out: K=768 -> n=768, bias on the NPU side (Epi::Bias). k.bias is zeros.
                let mk = |wk: &str, bk: &str, n: usize| {
                    CtxAOp::new(
                        shared.clone(),
                        &bw.m(wk),
                        n,
                        Epi::Bias,
                        bw.v(bk).as_slice().unwrap(),
                    )
                };
                crate::npu::BlockOps {
                    q: mk("q.weight", "q.bias", 768),
                    k: mk("k.weight", "k.bias", 768),
                    v: mk("v.weight", "v.bias", 768),
                    out: mk("out.weight", "out.bias", 768),
                    // FFN mm1: K=768 -> 3072 + bias. NPU_ENC_GELU_FUSED=1 folds GELU into the GEMM epilogue
                    // (Epi::GeluBias, modal rtp[0]=2) — drops the ~260 ms/utt host GELU; else GELU on host.
                    fc1: if std::env::var("NPU_ENC_GELU_FUSED").is_ok() {
                        CtxAOp::new(shared.clone(), &bw.m("fc1.weight"), 3072, Epi::GeluBias,
                                    bw.v("fc1.bias").as_slice().unwrap())
                    } else {
                        mk("fc1.weight", "fc1.bias", 3072)
                    },
                    // FFN mm2: K=3072 -> 768 + bias2 (host-accumulated K-split, bias2 added once).
                    fc2: FfnMm2::new(shared.clone(), &bw.m("fc2.weight"), bw.v("fc2.bias").as_slice().unwrap()),
                }
            })
            .collect();

        // NPU_ENC_MHA_NPU=1: load the static-shape encoder-MHA xclbin onto the SAME device (single-tenant).
        let mha_npu = if std::env::var("NPU_ENC_MHA_NPU").is_ok() {
            let base = root.join("artifacts/encoder_mha");
            let xclbin = base.join("StaticMHA_h12_s1500_d64_kv0_causal0_npu2.xclbin");
            let insts = base.join("StaticMHA_h12_s1500_d64_kv0_causal0_npu2.bin");
            Some(crate::mha_npu::MhaNpu::open(&npu.device(), &xclbin, &insts)
                .expect("NPU_ENC_MHA_NPU: load encoder-MHA xclbin (build via gen_encoder_mha.py)"))
        } else {
            None
        };

        // NPU_ENC_CONV_NPU=1: route the conv stem through the M-stationary GEMM conv (prebuilt 768 band).
        let conv_npu = if std::env::var("NPU_ENC_CONV_NPU").is_ok() {
            let wa = root.join(npu_asr::engines::WA_SUBDIR);
            Some(npu_asr::conv_npu::ConvNpu::new(npu.device(), wa))
        } else {
            None
        };

        WhisperEncoder {
            cfg,
            w,
            npu: Some(npu),
            block_ops,
            mha_npu,
            conv_npu,
        }
    }

    pub fn weights(&self) -> &WhisperWeights {
        &self.w
    }

    /// The NPU device this encoder opened (when built via `new_npu`), so a co-resident decoder can
    /// share the SAME single-tenant device instead of double-opening `/dev/accel/accel0`.
    #[cfg(feature = "npu")]
    pub fn device(&self) -> Option<std::rc::Rc<npu_xrt::Device>> {
        self.npu.as_ref().map(|n| n.device())
    }

    /// The resident ctx2 shared kernel this encoder loaded (when built via `new_npu`), so a
    /// co-resident decoder can register its OWN K=768 GEMM ops on it (e.g. the per-utterance
    /// cross-K/V fold) and run them on the NPU instead of the host.
    #[cfg(feature = "npu")]
    pub fn shared(&self) -> Option<std::rc::Rc<npu_asr::ctx2::SharedCtxA>> {
        self.npu.as_ref().map(|n| n.shared.clone())
    }

    /// Linear with weights stored [K_in, N_out]: `x[M,K]·W[K,N] + b[N]` (bias broadcast over rows).
    fn linear(&self, x: &Array2<f32>, w: &Array2<f32>, b: &Array1<f32>, _id: &str) -> Array2<f32> {
        // host path: id unused (the NPU path in A9 will use it to key the weight-BO cache)
        let mut y = x.dot(w);
        y += &b.view().insert_axis(Axis(0));
        y
    }

    /// Conv stem: mel [n_mels, 3000] -> [1500, d_model] (BEFORE positional embedding).
    pub fn conv_stem(&self, mel: &Array2<f32>) -> Array2<f32> {
        let c = self.w.conv();
        // conv1: k3 s1 p1, Cin=n_mels -> Cout=d_model ; [d_model, 3000]
        // conv2: k3 s2 p1, Cin=d_model -> Cout=d_model ; [d_model, 1500]
        #[cfg(feature = "npu")]
        if let Some(cv) = &self.conv_npu {
            let h = self.conv1d_npu(cv, mel, &c.m3("conv1.weight"), &c.v("conv1.bias"), 1, 1);
            let h = gelu(&h);
            let h = self.conv1d_npu(cv, &h, &c.m3("conv2.weight"), &c.v("conv2.bias"), 2, 1);
            let h = gelu(&h);
            return h.t().to_owned(); // [1500, d_model]
        }
        let h = im2col_conv1d(mel, &c.m3("conv1.weight"), c.v("conv1.bias").as_slice().unwrap(), 1, 1);
        let h = gelu(&h);
        let h = im2col_conv1d(&h, &c.m3("conv2.weight"), c.v("conv2.bias").as_slice().unwrap(), 2, 1);
        let h = gelu(&h);
        h.t().to_owned() // [1500, d_model]
    }

    /// conv1d-as-GEMM on the NPU: `x[Cin,W]` * `w[Cout,Cin,k]` (+bias) -> `[Cout,Wout]`. Wraps the 2D
    /// ConvNpu (treats the 1D conv as kh=1). bf16 on-chip — gated + WER-validated (NPU_ENC_CONV_NPU).
    #[cfg(feature = "npu")]
    fn conv1d_npu(
        &self,
        cv: &npu_asr::conv_npu::ConvNpu,
        x: &Array2<f32>,
        w3: &Array3<f32>,
        b: &Array1<f32>,
        stride: usize,
        pad: usize,
    ) -> Array2<f32> {
        let (cin, wd) = x.dim();
        let (cout, _cin, k) = w3.dim();
        let x3 = x.view().insert_axis(Axis(1)).to_owned(); // [Cin, 1, W]
        let w4 = w3.view().insert_axis(Axis(2)).to_owned(); // [Cout, Cin, 1, k]
        // 1D conv: H=1 with kh=1, NO H-padding (ph=0, sh=1); W carries the sequence (kw=k, pad=pw).
        let y3 = cv.conv_asym(&x3, &w4, b, 1, k, 1, stride, 0, pad); // [Cout, 1, Wout]
        let wout = y3.dim().2;
        debug_assert_eq!((y3.dim().0, y3.dim().1), (cout, 1));
        let _ = (cin, wd);
        y3.into_shape_with_order((cout, wout)).unwrap()
    }

    /// Add the learned positional embedding `embed_positions[:T]` in place.
    pub fn add_pos(&self, x: &mut Array2<f32>) {
        let t = x.nrows();
        let pos = self.w.conv().m("embed_positions"); // [max_src, d_model]
        *x += &pos.slice(s![..t, ..]);
    }

    /// One pre-norm encoder block i. When the NPU backend is active (`new_npu`), the four K=768
    /// projections + the two FFN matmuls run on the NPU (row-tiled to PAD_M=512); attention, LN,
    /// GELU and residuals stay on host. Otherwise the all-host f32 path runs.
    pub fn block(&self, i: usize, x: &Array2<f32>) -> Array2<f32> {
        let b: &TensorMap = self.w.block(i);
        let m = x.nrows();

        // --- self-attention sublayer ---
        let ln1 = timed!("ln", layer_norm(x, b.v("ln1.weight").as_slice().unwrap(), b.v("ln1.bias").as_slice().unwrap(), LN_EPS));

        #[cfg(feature = "npu")]
        let use_npu = self.npu.is_some();
        #[cfg(not(feature = "npu"))]
        let use_npu = false;

        let (q, k, v);
        if use_npu {
            #[cfg(feature = "npu")]
            {
                use crate::npu::apply_tiled;
                let ops = &self.block_ops[i];
                q = timed!("qkv_proj", apply_tiled(&ops.q, &ln1, 768));
                k = timed!("qkv_proj", apply_tiled(&ops.k, &ln1, 768)); // k.bias is zeros (applied on NPU)
                v = timed!("qkv_proj", apply_tiled(&ops.v, &ln1, 768));
            }
            #[cfg(not(feature = "npu"))]
            unreachable!();
        } else {
            q = timed!("qkv_proj", self.linear(&ln1, &b.m("q.weight"), &b.v("q.bias"), &format!("{i}.q")));
            k = timed!("qkv_proj", self.linear(&ln1, &b.m("k.weight"), &b.v("k.bias"), &format!("{i}.k")));
            v = timed!("qkv_proj", self.linear(&ln1, &b.m("v.weight"), &b.v("v.bias"), &format!("{i}.v")));
        }
        // full attention: NPU static-MHA op when gated (NPU_ENC_MHA_NPU), else host f32 mha.
        // NPU_ENC_MHA_MAXLAYER=N: run NPU MHA only for the first N blocks (i<N). The bf16 attention
        // error compounds over layers, so the first few may stay WER-acceptable — a PARTIAL offload.
        #[cfg(feature = "npu")]
        let mha_npu_here = self.mha_npu.as_ref().filter(|_| {
            let maxl = std::env::var("NPU_ENC_MHA_MAXLAYER").ok().and_then(|s| s.parse::<usize>().ok()).unwrap_or(usize::MAX);
            i < maxl
        });
        #[cfg(feature = "npu")]
        let ctx = match mha_npu_here {
            Some(op) if m == 1500 => timed!("mha", op.forward(&q, &k, &v)),
            _ => timed!("mha", mha(&q, &k, &v, self.cfg.n_heads, self.cfg.head_dim, false, m)),
        };
        #[cfg(not(feature = "npu"))]
        let ctx = timed!("mha", mha(&q, &k, &v, self.cfg.n_heads, self.cfg.head_dim, false, m)); // full attention
        let attn;
        if use_npu {
            #[cfg(feature = "npu")]
            {
                attn = timed!("out_proj", crate::npu::apply_tiled(&self.block_ops[i].out, &ctx, 768));
            }
            #[cfg(not(feature = "npu"))]
            unreachable!();
        } else {
            attn = timed!("out_proj", self.linear(&ctx, &b.m("out.weight"), &b.v("out.bias"), &format!("{i}.out")));
        }
        let x = timed!("residual", x + &attn); // residual

        // --- feed-forward sublayer ---
        let ln2 = timed!("ln", layer_norm(&x, b.v("ln2.weight").as_slice().unwrap(), b.v("ln2.bias").as_slice().unwrap(), LN_EPS));
        let f_out;
        if use_npu {
            #[cfg(feature = "npu")]
            {
                use crate::npu::{apply_tiled, apply_tiled_mm2};
                let ops = &self.block_ops[i];
                // NPU_ENC_GELU_FUSED: fc1 is built with Epi::GeluBias → its output is already gelu(W·x+b),
                // so the host GELU is skipped. Else GELU on host (default).
                let h1 = timed!("fc1", apply_tiled(&ops.fc1, &ln2, 3072));
                let f = if std::env::var("NPU_ENC_GELU_FUSED").is_ok() { h1 } else { timed!("gelu", gelu(&h1)) };
                f_out = timed!("fc2", apply_tiled_mm2(&ops.fc2, &f));
            }
            #[cfg(not(feature = "npu"))]
            unreachable!();
        } else {
            let f = timed!("gelu", gelu(&timed!("fc1", self.linear(&ln2, &b.m("fc1.weight"), &b.v("fc1.bias"), &format!("{i}.fc1")))));
            f_out = timed!("fc2", self.linear(&f, &b.m("fc2.weight"), &b.v("fc2.bias"), &format!("{i}.fc2")));
        }
        timed!("residual", x + &f_out) // residual
    }

    /// Run all blocks from a conv-stem-output (with pos already added). Returns each block's output
    /// (BEFORE the final post-LN), so `out[i]` == golden `block_i`.
    pub fn forward_collect(&self, after_conv: &Array2<f32>) -> Vec<Array2<f32>> {
        let mut x = after_conv.clone();
        let mut outs = Vec::with_capacity(self.cfg.n_layers);
        for i in 0..self.cfg.n_layers {
            x = self.block(i, &x);
            outs.push(x.clone());
        }
        outs
    }

    /// Full encoder from a mel [n_mels, 3000]: conv stem + pos + all blocks + final ln_post.
    /// Returns `encoded` == post-LN(block_{n-1}).
    pub fn forward_last(&self, mel: &Array2<f32>) -> Array2<f32> {
        let perop = std::env::var("ENC_PEROP_TIMING").is_ok();
        if perop { ENC_PEROP.with(|p| *p.borrow_mut() = Some(Vec::new())); }
        let mut x = timed!("conv_stem", self.conv_stem(mel));
        timed!("residual", self.add_pos(&mut x));
        for i in 0..self.cfg.n_layers {
            x = self.block(i, &x);
        }
        let g = self.w.ref_tensor("ln_post.weight").into_dimensionality::<Ix1>().unwrap();
        let be = self.w.ref_tensor("ln_post.bias").into_dimensionality::<Ix1>().unwrap();
        let out = timed!("ln", layer_norm(&x, g.as_slice().unwrap(), be.as_slice().unwrap(), LN_EPS));
        if perop {
            ENC_PEROP.with(|p| {
                if let Some(v) = p.borrow_mut().take() {
                    let total: f64 = v.iter().map(|(_, t)| t).sum();
                    let mut sorted = v.clone();
                    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                    eprint!("[ENC_PEROP] sum_ms={total:.2}");
                    for (s, t) in &sorted { eprint!(" {s}={t:.2}({:.0}%)", 100.0 * t / total); }
                    eprintln!();
                }
            });
        }
        out
    }
}
