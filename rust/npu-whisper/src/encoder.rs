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

pub struct WhisperEncoder {
    pub cfg: WhisperCfg,
    w: WhisperWeights,
    #[cfg(feature = "npu")]
    npu: Option<crate::npu::WhisperNpu>,
    #[cfg(feature = "npu")]
    block_ops: Vec<crate::npu::BlockOps>,
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
                    // FFN mm1: K=768 -> 3072 + bias; GELU applied on host (NOT SiluBias).
                    fc1: mk("fc1.weight", "fc1.bias", 3072),
                    // FFN mm2: K=3072 -> 768 + bias2 (host-accumulated K-split, bias2 added once).
                    fc2: FfnMm2::new(shared.clone(), &bw.m("fc2.weight"), bw.v("fc2.bias").as_slice().unwrap()),
                }
            })
            .collect();

        WhisperEncoder {
            cfg,
            w,
            npu: Some(npu),
            block_ops,
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
        let h = im2col_conv1d(mel, &c.m3("conv1.weight"), c.v("conv1.bias").as_slice().unwrap(), 1, 1);
        let h = gelu(&h);
        // conv2: k3 s2 p1, Cin=d_model -> Cout=d_model ; [d_model, 1500]
        let h = im2col_conv1d(&h, &c.m3("conv2.weight"), c.v("conv2.bias").as_slice().unwrap(), 2, 1);
        let h = gelu(&h);
        h.t().to_owned() // [1500, d_model]
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
        let ln1 = layer_norm(x, b.v("ln1.weight").as_slice().unwrap(), b.v("ln1.bias").as_slice().unwrap(), LN_EPS);

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
                q = apply_tiled(&ops.q, &ln1, 768);
                k = apply_tiled(&ops.k, &ln1, 768); // k.bias is zeros (applied on NPU)
                v = apply_tiled(&ops.v, &ln1, 768);
            }
            #[cfg(not(feature = "npu"))]
            unreachable!();
        } else {
            q = self.linear(&ln1, &b.m("q.weight"), &b.v("q.bias"), &format!("{i}.q"));
            k = self.linear(&ln1, &b.m("k.weight"), &b.v("k.bias"), &format!("{i}.k"));
            v = self.linear(&ln1, &b.m("v.weight"), &b.v("v.bias"), &format!("{i}.v"));
        }
        let ctx = mha(&q, &k, &v, self.cfg.n_heads, self.cfg.head_dim, false, m); // full attention
        let attn;
        if use_npu {
            #[cfg(feature = "npu")]
            {
                attn = crate::npu::apply_tiled(&self.block_ops[i].out, &ctx, 768);
            }
            #[cfg(not(feature = "npu"))]
            unreachable!();
        } else {
            attn = self.linear(&ctx, &b.m("out.weight"), &b.v("out.bias"), &format!("{i}.out"));
        }
        let x = x + &attn; // residual

        // --- feed-forward sublayer ---
        let ln2 = layer_norm(&x, b.v("ln2.weight").as_slice().unwrap(), b.v("ln2.bias").as_slice().unwrap(), LN_EPS);
        let f_out;
        if use_npu {
            #[cfg(feature = "npu")]
            {
                use crate::npu::{apply_tiled, apply_tiled_mm2};
                let ops = &self.block_ops[i];
                let f = gelu(&apply_tiled(&ops.fc1, &ln2, 3072)); // bias on NPU, GELU on host
                f_out = apply_tiled_mm2(&ops.fc2, &f);
            }
            #[cfg(not(feature = "npu"))]
            unreachable!();
        } else {
            let f = gelu(&self.linear(&ln2, &b.m("fc1.weight"), &b.v("fc1.bias"), &format!("{i}.fc1")));
            f_out = self.linear(&f, &b.m("fc2.weight"), &b.v("fc2.bias"), &format!("{i}.fc2"));
        }
        x + &f_out // residual
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
        let mut x = self.conv_stem(mel);
        self.add_pos(&mut x);
        for i in 0..self.cfg.n_layers {
            x = self.block(i, &x);
        }
        let g = self.w.ref_tensor("ln_post.weight").into_dimensionality::<Ix1>().unwrap();
        let be = self.w.ref_tensor("ln_post.bias").into_dimensionality::<Ix1>().unwrap();
        layer_norm(&x, g.as_slice().unwrap(), be.as_slice().unwrap(), LN_EPS)
    }
}
