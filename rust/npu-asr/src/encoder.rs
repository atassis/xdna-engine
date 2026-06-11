//! Full GigaAM-v3 encoder: subsampling front-end + N stacked Conformer blocks.
//! Mirrors `npu_asr/encoder.py`.

use std::path::Path;
use std::rc::Rc;

use ndarray::prelude::*;
use npu_asr_host::{bf16_round, subsample as host_subsample};
use npu_xrt::Device;

use crate::block::FusedBlock;
use crate::weights::WeightStore;

/// Subsampling front-end (host im2col conv1d + ReLU): audio [64,1600] -> [400,768].
pub fn subsample(ws: &WeightStore, audio: &Array2<f32>) -> Array2<f32> {
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
    host_subsample(audio, &w0, &b0, &w2, &b2)
}

pub struct Encoder {
    blocks: Vec<FusedBlock>,
}

impl Encoder {
    pub fn new(dev: Rc<Device>, root: &Path, ws: &WeightStore, n_blocks: usize) -> Self {
        // With `two_ctx`, build the ONE shared hw-context (resident ctxA xclbin) ONCE for the whole
        // encoder and hand every block a clone of the `Rc` -> ALL matmul ops (the 7 K=768 ops AND
        // the FFN mm2, K-split into 4× N=768) dispatch on the same resident kernel: zero context
        // switches across the whole encoder.
        #[cfg(feature = "two_ctx")]
        let ctx_a = crate::ctx2::SharedCtxA::new(&dev, root);
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
                )
            })
            .collect();
        Encoder { blocks }
    }

    /// Run the block stack from a [T,768] input, returning every block's output ([T,768]).
    /// `valid_len` is the number of non-padded time frames; padded frames are zeroed and masked
    /// through the time-mixing ops (attention, dwconv). Pass valid_len >= T for no masking.
    pub fn forward_blocks(&self, x0: &Array2<f32>, valid_len: usize) -> Vec<Array2<f32>> {
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
        let mut outs = Vec::with_capacity(self.blocks.len());
        for blk in &self.blocks {
            x = blk.forward(&x, valid_len);
            outs.push(x.clone());
        }
        outs
    }
}
