//! S2 decoder weights, read from the real GGUF checkpoint by the exact tensor names
//! `designs/codec_block/decoder_chain.py` reads (`PREFIX = "c.decoder"`, `model.0` = head,
//! `model.1..4` = the four upsample stages, `model.5`/`model.6` = the tail snake/conv). Layouts
//! mirror `stage_shapes.py`'s documented ggml->numpy conventions: a plain conv weight is
//! `[c_out, c_in, k]`, a conv_transpose weight is `[c_in, c_out, k]` (note the swap).

use ndarray::{Array1, Array3};

use crate::gguf::GgufFile;
use crate::S2Error;

pub const PREFIX: &str = "c.decoder";

pub struct S2Weights {
    gguf: GgufFile,
}

/// One residual unit's six tensors, in the order `window_driver.residual_unit`'s `wts` tuple
/// expects: `(a0, w1, b1, a2, w3, b3)` = snake -> dilated conv -> snake -> 1x1 conv(+residual).
pub struct ResidualUnitWeights {
    pub alpha0: Array1<f32>,
    pub w1: Array3<f32>,
    pub b1: Array1<f32>,
    pub alpha2: Array1<f32>,
    pub w3: Array3<f32>,
    pub b3: Array1<f32>,
}

impl S2Weights {
    pub fn open(path: &std::path::Path) -> crate::Result<Self> {
        Ok(S2Weights { gguf: GgufFile::open(path)? })
    }

    /// The codec's output sample rate as the model declares it. `s2_codec.cpp:826` reads this same
    /// key and `:701` falls back to 44100 when it is absent; callers that need a number should
    /// make that fallback explicit rather than assume one here.
    pub fn sample_rate(&self) -> Option<u32> {
        self.gguf.meta_u32("fish_speech.codec.sample_rate")
    }

    /// Flat vector, regardless of the on-disk rank -- decoder_chain.py's `g(name).reshape(-1)`
    /// convention for every alpha/bias tensor (some are stored `[1, C]`, some `[C]`).
    fn v(&self, name: &str) -> crate::Result<Array1<f32>> {
        Ok(Array1::from_vec(self.gguf.tensor_f32(name)?))
    }

    /// Requires an on-disk 3-D shape; errors (rather than guessing a reshape) otherwise -- the one
    /// exception, the tail conv weight, is handled by its own accessor below.
    fn m3(&self, name: &str) -> crate::Result<Array3<f32>> {
        let shape = self.gguf.shape(name)?;
        let [d0, d1, d2]: [usize; 3] = shape.clone().try_into().map_err(|_| {
            S2Error::Shape(format!("{name}: expected a 3-D tensor, got shape {shape:?}"))
        })?;
        let data = self.gguf.tensor_f32(name)?;
        Array3::from_shape_vec((d0, d1, d2), data)
            .map_err(|e| S2Error::Shape(format!("{name}: {e}")))
    }

    /// `(weight [c_out, c_in, k], bias [c_out])`. model.0.conv, k=7 dilation=1.
    pub fn head_conv(&self) -> crate::Result<(Array3<f32>, Array1<f32>)> {
        Ok((self.m3(&format!("{PREFIX}.model.0.conv.weight"))?, self.v(&format!("{PREFIX}.model.0.conv.bias"))?))
    }

    /// The upsample block's snake alpha, `[c_in]`. `stage` in 1..=4.
    pub fn stage_upsample_alpha(&self, stage: u32) -> crate::Result<Array1<f32>> {
        self.v(&format!("{PREFIX}.model.{stage}.block.0.alpha"))
    }

    /// `(weight [c_in, c_out, k], bias [c_out])` -- conv_transpose layout, NOT `[c_out, c_in, k]`.
    pub fn stage_conv_transpose(&self, stage: u32) -> crate::Result<(Array3<f32>, Array1<f32>)> {
        Ok((
            self.m3(&format!("{PREFIX}.model.{stage}.block.1.conv.weight"))?,
            self.v(&format!("{PREFIX}.model.{stage}.block.1.conv.bias"))?,
        ))
    }

    /// One residual unit's six tensors. `stage` in 1..=4, `unit` in 0..3 (block.2/3/4).
    pub fn residual_unit(&self, stage: u32, unit: u32) -> crate::Result<ResidualUnitWeights> {
        let sub = format!("block.{}", 2 + unit);
        let p = format!("{PREFIX}.model.{stage}.{sub}");
        Ok(ResidualUnitWeights {
            alpha0: self.v(&format!("{p}.block.0.alpha"))?,
            w1: self.m3(&format!("{p}.block.1.conv.weight"))?,
            b1: self.v(&format!("{p}.block.1.conv.bias"))?,
            alpha2: self.v(&format!("{p}.block.2.alpha"))?,
            w3: self.m3(&format!("{p}.block.3.conv.weight"))?,
            b3: self.v(&format!("{p}.block.3.conv.bias"))?,
        })
    }

    /// The tail's snake alpha, `model.5.alpha` (`last = len(DECODER_RATES) + 1` in decoder_chain.py,
    /// pinned to `5` here since `DECODER_RATES` has exactly 4 entries for this codec).
    pub fn tail_snake_alpha(&self) -> crate::Result<Array1<f32>> {
        self.v(&format!("{PREFIX}.model.5.alpha"))
    }

    /// `(weight [1, c_in, k], bias [1])`. `model.6.conv.weight` is stored 2-D (`[c_in, k]`, the
    /// single output channel folded out of the ne entirely) -- reshaped to `[1, c_in, k]` here,
    /// mirroring `decoder_chain.run_tail`'s `w_tail.reshape(1, *w_tail.shape)`.
    pub fn tail_conv(&self) -> crate::Result<(Array3<f32>, Array1<f32>)> {
        let name = format!("{PREFIX}.model.6.conv.weight");
        let shape = self.gguf.shape(&name)?;
        let data = self.gguf.tensor_f32(&name)?;
        let w = match shape.as_slice() {
            [c_in, k] => Array3::from_shape_vec((1, *c_in, *k), data),
            [c_out, c_in, k] => Array3::from_shape_vec((*c_out, *c_in, *k), data),
            _ => return Err(S2Error::Shape(format!("{name}: expected 2-D or 3-D, got {shape:?}"))),
        }
        .map_err(|e| S2Error::Shape(format!("{name}: {e}")))?;
        Ok((w, self.v(&format!("{PREFIX}.model.6.conv.bias"))?))
    }
}
