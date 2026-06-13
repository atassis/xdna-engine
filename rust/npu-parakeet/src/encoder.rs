//! Parakeet FastConformer encoder (host f32 reference). Ports scripts/parakeet_ref_encoder.py.
//! `forward_last` is the general-engine Encoder-contract entry point.

use std::path::Path;

use ndarray::prelude::*;

use crate::config::ModelCfg;
use crate::ops::{conv2d, dwconv1d, layernorm, rel_shift, sigmoid, silu_inplace};
use crate::prof;
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
    /// NPU weight-BO cache (unique per fixed weight, e.g. "3.ff1.l1").
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

    fn feed_forward(&self, x: &Array2<f32>, b: &BlockWeights, blk: usize, tag: &str, norm_w: &str, norm_b: &str, l1: &str, l2: &str) -> Array2<f32> {
        let n = layernorm(x, &b.v(norm_w), &b.v(norm_b));
        let mut h = self.mm(&n, &b.m(l1), &format!("{blk}.{tag}.l1")); // [T, DFF]
        silu_inplace(&mut h);
        self.mm(&h, &b.m(l2), &format!("{blk}.{tag}.l2")) // [T, D]
    }

    pub fn weights(&self) -> &ParakeetWeights {
        &self.w
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
        flat.dot(&wout) + &bout
    }

    fn mhsa(&self, x: &Array2<f32>, blk: usize, pos_enc: &Array2<f32>) -> Array2<f32> {
        let b = self.w.block(blk);
        let (h, dk, d) = (self.cfg.n_heads, self.cfg.head_dim, self.cfg.hidden);
        let t = x.nrows();
        let p = pos_enc.nrows(); // 2T-1
        let q = self.mm(x, &b.m("self_attn.linear_q.weight"), &format!("{blk}.q")); // [T, D]
        let k = self.mm(x, &b.m("self_attn.linear_k.weight"), &format!("{blk}.k"));
        let v = self.mm(x, &b.m("self_attn.linear_v.weight"), &format!("{blk}.v"));
        let pm = self.mm(pos_enc, &b.m("self_attn.linear_pos.weight"), &format!("{blk}.pos")); // [P, D]
        let ubias = b.m("self_attn.pos_bias_u"); // [H, DK]
        let vbias = b.m("self_attn.pos_bias_v");
        let scale = (dk as f32).sqrt();

        // assemble bd_all [H, T, P] then rel_shift -> [H, T, T]
        let mut bd_all = Array3::<f32>::zeros((h, t, p));
        let mut ac_all = Array3::<f32>::zeros((h, t, t));
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
        let bd = prof::time("rel_shift", || rel_shift(&bd_all, t)); // [H, T, T]

        // scores -> softmax -> context -> merge -> linear_out
        let _sm = prof::time("mha_softmax", || {
        let mut ctx = Array2::<f32>::zeros((t, d));
        for hh in 0..h {
            let col = hh * dk;
            let vh = v.slice(s![.., col..col + dk]); // [T, DK]
            let mut scores = Array2::<f32>::zeros((t, t));
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
            let ch = scores.dot(&vh); // [T, DK]
            ctx.slice_mut(s![.., col..col + dk]).assign(&ch);
        }
        ctx
        });
        self.mm(&_sm, &b.m("self_attn.linear_out.weight"), &format!("{blk}.out"))
    }

    fn conv_module(&self, x: &Array2<f32>, blk: usize) -> Array2<f32> {
        let b = self.w.block(blk);
        let d = self.cfg.hidden;
        let t = x.nrows();
        let pw1 = b.m3("conv.pointwise_conv1.weight"); // [2D, D, 1]
        let pw1 = pw1.index_axis(Axis(2), 0).to_owned(); // [2D, D]
        let pw2 = b.m3("conv.pointwise_conv2.weight").index_axis(Axis(2), 0).to_owned(); // [D, D]
        let dw3 = b.m3("conv.depthwise_conv.weight"); // [D, 1, 9]
        let taps = dw3.index_axis(Axis(1), 0).to_owned(); // [D, 9]
        let dwb = b.v("conv.depthwise_conv.bias");

        let h = self.mm(x, &pw1.t().to_owned(), &format!("{blk}.pw1")); // [T, 2D] (pw1.T = [D, 2D])
        // GLU: a * sigmoid(g)
        let glu = prof::time("glu", || {
            let mut glu = Array2::<f32>::zeros((t, d));
            for i in 0..t {
                for c in 0..d {
                    glu[[i, c]] = h[[i, c]] * sigmoid(h[[i, d + c]]);
                }
            }
            glu
        });
        // depthwise along time: [D, T]
        let glu_t = glu.t().to_owned();
        let mut dwc = prof::time("dwconv", || dwconv1d(&glu_t, &taps, &dwb, 9)); // [D, T]
        silu_inplace(&mut dwc);
        let back = dwc.t().to_owned(); // [T, D]
        self.mm(&back, &pw2.t().to_owned(), &format!("{blk}.pw2"))
    }

    fn block(&self, x: &Array2<f32>, blk: usize, pos_enc: &Array2<f32>) -> Array2<f32> {
        let b = self.w.block(blk);
        let mut x = x.clone();
        let ff1 = prof::time("ff", || self.feed_forward(&x, b, blk, "ff1", "norm_feed_forward1.weight", "norm_feed_forward1.bias",
                                    "feed_forward1.linear1.weight", "feed_forward1.linear2.weight"));
        x = x + ff1.mapv(|v| 0.5 * v);
        let satt_in = prof::time("ln", || layernorm(&x, &b.v("norm_self_att.weight"), &b.v("norm_self_att.bias")));
        x = &x + &prof::time("mhsa", || self.mhsa(&satt_in, blk, pos_enc));
        let conv_in = prof::time("ln", || layernorm(&x, &b.v("norm_conv.weight"), &b.v("norm_conv.bias")));
        x = &x + &prof::time("conv_mod", || self.conv_module(&conv_in, blk));
        let ff2 = prof::time("ff", || self.feed_forward(&x, b, blk, "ff2", "norm_feed_forward2.weight", "norm_feed_forward2.bias",
                                    "feed_forward2.linear1.weight", "feed_forward2.linear2.weight"));
        x = x + ff2.mapv(|v| 0.5 * v);
        layernorm(&x, &b.v("norm_out.weight"), &b.v("norm_out.bias"))
    }

    /// Encoder block stack: x [T, hidden] -> [T, hidden]. (Contract entry point;
    /// valid_len is the unpadded length — masking is a no-op for full-length inputs.)
    pub fn forward_last(&self, x: &Array2<f32>, _valid_len: usize) -> Array2<f32> {
        let pos_enc = rel_pos_encoding(x.nrows(), self.cfg.hidden);
        let mut x = x.clone();
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
