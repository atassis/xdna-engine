//! Full GigaAM-v3 encoder: subsampling front-end + N stacked Conformer blocks.
//! Mirrors `npu_asr/encoder.py`.

use std::path::Path;
use std::rc::Rc;

use ndarray::prelude::*;
use npu_asr_host::{bf16_round, prof, subsample as host_subsample};
#[cfg(feature = "two_ctx")]
use npu_asr_host::{im2col, im2col_conv1d, relu_inplace};
use npu_xrt::Device;

use crate::block::FusedBlock;
use crate::weights::WeightStore;

/// Subsampling front-end (host im2col conv1d + ReLU): audio [64,1600] -> [400,768].
pub fn subsample(ws: &WeightStore, audio: &Array2<f32>) -> Array2<f32> {
    let (w0, b0, w2, b2) = prof::time("ss_wprep", || {
        let w0 = ws
            .pre("pre_encode.conv.0.weight")
            .clone()
            .into_dimensionality::<Ix3>()
            .unwrap();
        let b0: Vec<f32> = ws.pre("pre_encode.conv.0.bias").iter().copied().collect();
        let w2 = ws
            .pre("pre_encode.conv.2.weight")
            .clone()
            .into_dimensionality::<Ix3>()
            .unwrap();
        let b2: Vec<f32> = ws.pre("pre_encode.conv.2.bias").iter().copied().collect();
        (w0, b0, w2, b2)
    });
    prof::time("ss_conv", || host_subsample(audio, &w0, &b0, &w2, &b2))
}

pub struct Encoder {
    blocks: Vec<FusedBlock>,
    /// PROTOTYPE (`NPU_SS_NPU=1`): conv2 of the subsample front-end on the resident ctxA (5-way
    /// K-split). `None` = the all-host front-end (default). See [`crate::ctx2::Conv2Mm`].
    #[cfg(feature = "two_ctx")]
    conv2mm: Option<crate::ctx2::Conv2Mm>,
}

impl Encoder {
    pub fn new(dev: Rc<Device>, root: &Path, ws: &WeightStore, n_blocks: usize) -> Self {
        // With `two_ctx`, build the ONE shared hw-context (resident ctxA xclbin) ONCE for the whole
        // encoder and hand every block a clone of the `Rc` -> ALL matmul ops (the 7 K=768 ops AND
        // the FFN mm2, K-split into 4× N=768) dispatch on the same resident kernel: zero context
        // switches across the whole encoder.
        #[cfg(feature = "two_ctx")]
        let ctx_a = crate::ctx2::SharedCtxA::new(&dev, root);
        // Step D: on-NPU LayerNorm (ctxLN), opt-in via NPU_LN_NPU=1 (default OFF -> host LN). Built
        // ONCE and shared across all blocks (the encoder runs sequentially, so one BO set suffices).
        #[cfg(feature = "two_ctx")]
        let ctx_ln = if std::env::var("NPU_LN_NPU").as_deref() == Ok("1") {
            Some(crate::ctx_ln::CtxLn::new(&dev, root))
        } else {
            None
        };
        // conv2 of the subsample on the resident ctxA (built once). DEFAULT-ON; NPU_SS_NPU=0 reverts to
        // the all-host front-end. MEASURED net-positive (e2e −20ms bf16) + WER-safe at every precision
        // (bf16 9.6→9.2%, int8 9.2→8.7%, native 9.2% unchanged).
        #[cfg(feature = "two_ctx")]
        let conv2mm = if std::env::var("NPU_SS_NPU").as_deref() != Ok("0") {
            let w2 = ws
                .pre("pre_encode.conv.2.weight")
                .clone()
                .into_dimensionality::<Ix3>()
                .unwrap();
            let b2: Vec<f32> = ws.pre("pre_encode.conv.2.bias").iter().copied().collect();
            eprintln!("[ctx2] subsample conv2 ON NPU (default; ctxA K-split×5; NPU_SS_NPU=0 disables)");
            Some(crate::ctx2::Conv2Mm::new(ctx_a.clone(), &w2, &b2))
        } else {
            None
        };
        let blocks = (0..n_blocks)
            .map(|i| {
                FusedBlock::new(
                    dev.clone(),
                    root,
                    ws.block(i),
                    &ws.cos,
                    &ws.sin,
                    #[cfg(feature = "two_ctx")]
                    ctx_a.clone(),
                    #[cfg(feature = "two_ctx")]
                    ctx_ln.clone(),
                )
            })
            .collect();
        Encoder {
            blocks,
            #[cfg(feature = "two_ctx")]
            conv2mm,
        }
    }

    /// Front-end subsampling, encoder-owned so the conv2 matmul can route to the NPU. Default: conv0
    /// on host (M=800 > PAD_M), conv2's K=3840 matmul on the resident ctxA (5-way K-split), ReLU on
    /// host. `NPU_SS_NPU=0` reverts to the all-host front-end.
    pub fn subsample(&self, ws: &WeightStore, audio: &Array2<f32>) -> Array2<f32> {
        #[cfg(feature = "two_ctx")]
        if let Some(c2) = &self.conv2mm {
            return prof::time("ss_conv", || {
                let w0 = ws
                    .pre("pre_encode.conv.0.weight")
                    .clone()
                    .into_dimensionality::<Ix3>()
                    .unwrap();
                let b0: Vec<f32> = ws.pre("pre_encode.conv.0.bias").iter().copied().collect();
                let mut h0 = im2col_conv1d(audio, &w0, &b0, 2, 2); // conv0 host [768, 800]
                relu_inplace(&mut h0);
                let cols = im2col(&h0, 5, 2, 2); // [400, 3840]
                let mut out = c2.forward(&cols); // [400, 768] = conv2 pre-activation (+b2) on NPU
                relu_inplace(&mut out);
                out
            });
        }
        subsample(ws, audio) // all-host front-end
    }

    /// Run the block stack from a [T,768] input, returning every block's output ([T,768]).
    /// `valid_len` is the number of non-padded time frames; padded frames are zeroed and masked
    /// through the time-mixing ops (attention, dwconv). Pass valid_len >= T for no masking.
    pub fn forward_blocks(&self, x0: &Array2<f32>, valid_len: usize) -> Vec<Array2<f32>> {
        let mut x = prof::time("enc_prep", || {
            let mut x = x0.mapv(bf16_round);
            // zero padded frames at the block-stack input so block 0 starts clean
            let (t, d) = x.dim();
            if valid_len < t {
                for ti in valid_len..t {
                    for c in 0..d {
                        x[[ti, c]] = 0.0;
                    }
                }
            }
            x
        });
        let mut outs = Vec::with_capacity(self.blocks.len());
        for (i, blk) in self.blocks.iter().enumerate() {
            x = blk.forward(&x, valid_len, i == 0);
            prof::time("out_clone", || outs.push(x.clone()));
        }
        outs
    }

    /// As [`forward_blocks`] but returns ONLY the final block output — no per-block `Vec<clone>`
    /// (the production path: the service uses `outs.last()` only, so cloning + retaining all 16
    /// intermediate `[T,768]` tensors was ~20 MB of pure waste per inference). Numerically identical
    /// to `forward_blocks().pop()`.
    pub fn forward_last(&self, x0: &Array2<f32>, valid_len: usize) -> Array2<f32> {
        let mut x = prof::time("enc_prep", || {
            let mut x = x0.mapv(bf16_round);
            let (t, d) = x.dim();
            if valid_len < t {
                for ti in valid_len..t {
                    for c in 0..d {
                        x[[ti, c]] = 0.0;
                    }
                }
            }
            x
        });
        for (i, blk) in self.blocks.iter().enumerate() {
            x = blk.forward(&x, valid_len, i == 0);
        }
        x
    }
}
