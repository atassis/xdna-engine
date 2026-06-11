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
        let blocks = (0..n_blocks)
            .map(|i| FusedBlock::new(dev.clone(), root, ws.block(i), &ws.cos, &ws.sin))
            .collect();
        Encoder { blocks }
    }

    /// Run the block stack from a [400,768] input, returning every block's output ([T,768]).
    pub fn forward_blocks(&self, x0: &Array2<f32>) -> Vec<Array2<f32>> {
        let mut x = x0.mapv(bf16_round);
        let mut outs = Vec::with_capacity(self.blocks.len());
        for blk in &self.blocks {
            x = blk.forward(&x);
            outs.push(x.clone());
        }
        outs
    }
}
