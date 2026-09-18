//! The decoder's device chain: head -> stage1..4 -> tail, assembled from the window/chunk bricks
//! in [`crate::window`]. Ports `designs/codec_block/decoder_chain.py`: same op sequence,
//! same GGUF weight names (`crate::weights::S2Weights`), same forward-offset arithmetic
//! ([`chain_offset`]). [`S2DecoderChain::open`] validates every design but opens no hw_context:
//! the chain has 68 designs against a 16-context driver budget, so they share a [`DesignPool`]
//! that loads on first dispatch and evicts least-recently-used. The chain runs op-major -- an op
//! dispatches all of its chunks before the next op is touched -- so each design loads once per
//! pass and the LRU never thrashes.

use std::rc::Rc;

use ndarray::{s, Array1, Array2};
use npu_xrt::Device;

use crate::weights::{ResidualUnitWeights, S2Weights};
use crate::window::{ConvOp, ConvTransposeOp, SnakeOp};
use crate::{DesignPool, S2Artifacts, S2Design, S2Error, S2ManifestDesign, S2OpParams};

const N_STAGES: u32 = 4;
const N_UNITS: u32 = 3;

fn open_one(
    dev: &Rc<Device>,
    artifacts: &S2Artifacts,
    op: &str,
    pool: &Rc<DesignPool>,
) -> crate::Result<S2Design> {
    let d = artifacts
        .by_op(op)
        .ok_or_else(|| S2Error::Shape(format!("no design with op '{op}' in this artifact set")))?;
    S2Design::open_pooled(dev, &artifacts.design_dir(d), pool)
}

fn design_has_add(d: &S2Design) -> bool {
    matches!(d.meta.op_params, Some(S2OpParams::Conv { has_add: true, .. }))
}

/// A conv op has ONE design unless it is a chunked op that carries a residual add, in which case
/// the exporter produced two (see `S2Artifacts::designs_by_op`'s doc). Classify by `has_add`, not
/// by name, so a future naming-convention change can't silently swap chunk 0 with the rest.
fn open_conv_op(
    dev: &Rc<Device>,
    artifacts: &S2Artifacts,
    op: &str,
    pool: &Rc<DesignPool>,
) -> crate::Result<ConvOp> {
    let matches: Vec<&S2ManifestDesign> = artifacts.designs_by_op(op);
    match matches.len() {
        0 => Err(S2Error::Shape(format!("no design with op '{op}' in this artifact set"))),
        1 => {
            let d = S2Design::open_pooled(dev, &artifacts.design_dir(matches[0]), pool)?;
            ConvOp::new(d, None)
        }
        2 => {
            let a = S2Design::open_pooled(dev, &artifacts.design_dir(matches[0]), pool)?;
            let b = S2Design::open_pooled(dev, &artifacts.design_dir(matches[1]), pool)?;
            match (design_has_add(&a), design_has_add(&b)) {
                (true, false) => ConvOp::new(a, Some(b)),
                (false, true) => ConvOp::new(b, Some(a)),
                (aa, bb) => Err(S2Error::Shape(format!(
                    "op '{op}': two designs but has_add is ({aa},{bb}), can't tell chunk 0 from the rest"
                ))),
            }
        }
        n => Err(S2Error::Shape(format!("op '{op}': expected 1 or 2 designs, found {n}"))),
    }
}

/// `[C, L] -> [C, L - 6*dilation]`: snake -> dilated conv -> snake -> 1x1 conv + residual. Ports
/// `window_driver.residual_unit`.
pub struct ResidualUnit {
    snake_a: SnakeOp,
    conv_dil: ConvOp,
    snake_b: SnakeOp,
    conv_1x1: ConvOp,
}

impl ResidualUnit {
    fn open(
        dev: &Rc<Device>,
        artifacts: &S2Artifacts,
        stage: u32,
        unit: u32,
        pool: &Rc<DesignPool>,
    ) -> crate::Result<Self> {
        Ok(ResidualUnit {
            snake_a: SnakeOp::new(open_one(dev, artifacts, &format!("stage{stage}_res{unit}_snake_a"), pool)?)?,
            conv_dil: open_conv_op(dev, artifacts, &format!("stage{stage}_res{unit}_dil"), pool)?,
            snake_b: SnakeOp::new(open_one(dev, artifacts, &format!("stage{stage}_res{unit}_snake_b"), pool)?)?,
            conv_1x1: open_conv_op(dev, artifacts, &format!("stage{stage}_res{unit}_1x1"), pool)?,
        })
    }

    pub fn run(&self, x: &Array2<f32>, w: &ResidualUnitWeights) -> crate::Result<Array2<f32>> {
        let s1 = self.snake_a.run(x, &w.alpha0)?;
        let h = self.conv_dil.run(&s1, &w.w1, &w.b1, None)?;
        let s2 = self.snake_b.run(&h, &w.alpha2)?;
        // The residual reads x at the OUTPUT's positions, i.e. shifted by this unit's own context.
        let ctx = self.conv_dil.ctx();
        let add = x.slice(s![.., ctx..]).to_owned();
        self.conv_1x1.run(&s2, &w.w3, &w.b3, Some(&add))
    }

    /// This unit's context consumption, `6 * dilation` -- read off the dilated conv's own
    /// op_params rather than hardcoded, since `ctx = (k-1)*dilation` and `k` is fixed at 7.
    pub fn ctx(&self) -> usize {
        self.conv_dil.ctx()
    }
}

/// One upsample stage: snake -> conv_transpose (the "unfused" pair `decoder_chain.run_stage`
/// uses, not the fused `upsample_stage.cc` kernel -- see `window_driver.upsample_unfused`'s
/// docstring on why: the fused kernel's static L1 scratch made it fragile).
pub struct Stage {
    up_snake: SnakeOp,
    up_ct: ConvTransposeOp,
    units: Vec<ResidualUnit>,
}

pub struct StageWeights {
    pub up_alpha: Array1<f32>,
    pub up_w: ndarray::Array3<f32>,
    pub up_b: Array1<f32>,
    pub units: Vec<ResidualUnitWeights>,
}

impl StageWeights {
    pub fn load(w: &S2Weights, stage: u32) -> crate::Result<Self> {
        let up_alpha = w.stage_upsample_alpha(stage)?;
        let (up_w, up_b) = w.stage_conv_transpose(stage)?;
        let units = (0..N_UNITS).map(|u| w.residual_unit(stage, u)).collect::<crate::Result<Vec<_>>>()?;
        Ok(StageWeights { up_alpha, up_w, up_b, units })
    }
}

impl Stage {
    fn open(
        dev: &Rc<Device>,
        artifacts: &S2Artifacts,
        stage: u32,
        pool: &Rc<DesignPool>,
    ) -> crate::Result<Self> {
        let up_snake = SnakeOp::new(open_one(dev, artifacts, &format!("stage{stage}_upsample_snake"), pool)?)?;
        let up_ct = ConvTransposeOp::new(open_one(dev, artifacts, &format!("stage{stage}_upsample_convtranspose"), pool)?)?;
        let units = (0..N_UNITS)
            .map(|u| ResidualUnit::open(dev, artifacts, stage, u, pool))
            .collect::<crate::Result<Vec<_>>>()?;
        Ok(Stage { up_snake, up_ct, units })
    }

    pub fn run(&self, x: &Array2<f32>, w: &StageWeights) -> crate::Result<Array2<f32>> {
        let sn = self.up_snake.run(x, &w.up_alpha)?;
        let mut cur = self.up_ct.run(&sn, &w.up_w, &w.up_b)?;
        for (unit, uw) in self.units.iter().zip(&w.units) {
            cur = unit.run(&cur, uw)?;
        }
        Ok(cur)
    }
}

/// The whole decoder chain, every design opened once. `open()` touches the device (loads every
/// design's xclbin); `run_*` only dispatch.
pub struct S2DecoderChain {
    head: ConvOp,
    stages: Vec<Stage>,
    tail_snake: SnakeOp,
    tail_conv: ConvOp,
}

impl S2DecoderChain {
    pub fn open(dev: &Rc<Device>, artifacts: &S2Artifacts) -> crate::Result<Self> {
        Self::open_with_pool(dev, artifacts, &DesignPool::with_default_limit())
    }

    /// [`open`](Self::open) against a caller-supplied pool, so a process running two chains (or a
    /// chain alongside another engine) can bound their combined context use as one budget.
    pub fn open_with_pool(
        dev: &Rc<Device>,
        artifacts: &S2Artifacts,
        pool: &Rc<DesignPool>,
    ) -> crate::Result<Self> {
        let head = open_conv_op(dev, artifacts, "head_conv", pool)?;
        let stages = (1..=N_STAGES)
            .map(|s| Stage::open(dev, artifacts, s, pool))
            .collect::<crate::Result<Vec<_>>>()?;
        let tail_snake = SnakeOp::new(open_one(dev, artifacts, "tail_snake", pool)?)?;
        let tail_conv = open_conv_op(dev, artifacts, "tail_conv", pool)?;
        Ok(S2DecoderChain { head, stages, tail_snake, tail_conv })
    }

    /// `[1024, L] -> [1536, L - CTX_HEAD]`. model.0.conv, k=7 dilation=1.
    pub fn run_head(&self, z: &Array2<f32>, w: &S2Weights) -> crate::Result<Array2<f32>> {
        let (hw, hb) = w.head_conv()?;
        self.head.run(z, &hw, &hb, None)
    }

    pub fn run_stage(&self, x: &Array2<f32>, stage: u32, w: &StageWeights) -> crate::Result<Array2<f32>> {
        self.stages[(stage - 1) as usize].run(x, w)
    }

    /// `[96, L] -> tanh'd audio [L - CTX_TAIL]`. Tail snake -> tail conv (both on device) -> tanh
    /// (host, f64 accumulate matching `decoder_chain.run_tail`'s `.astype(np.float64)`).
    pub fn run_tail(&self, x: &Array2<f32>, w: &S2Weights) -> crate::Result<Array1<f32>> {
        let alpha = w.tail_snake_alpha()?;
        let (tw, tb) = w.tail_conv()?;
        let sn = self.tail_snake.run(x, &alpha)?;
        let raw = self.tail_conv.run(&sn, &tw, &tb, None)?;
        Ok(raw.row(0).mapv(|v| (v as f64).tanh() as f32))
    }

    /// `[1024, L]` latent -> tanh'd audio, head -> stage1..4 -> tail, run once end to end.
    pub fn run_chain(&self, z: &Array2<f32>, w: &S2Weights) -> crate::Result<Array1<f32>> {
        let mut cur = self.run_head(z, w)?;
        for stage in 1..=N_STAGES {
            let sw = StageWeights::load(w, stage)?;
            cur = self.run_stage(&cur, stage, &sw)?;
        }
        self.run_tail(&cur, w)
    }

    /// Forward per-stage position tracking (see [`chain_offset`]), reading CTX_HEAD/CTX_TAIL/
    /// UP_CTX/stride/RES_CTX off this chain's own opened ops rather than a hardcoded table.
    pub fn chain_offset(&self, latent_start: i64, latent_len: i64) -> crate::Result<(i64, i64)> {
        let stages: Vec<StageOffset> = self
            .stages
            .iter()
            .map(|s| StageOffset {
                up_ctx: s.up_ct.ctx() as i64,
                stride: s.up_ct.stride() as i64,
                res_ctx: s.units.iter().map(|u| u.ctx() as i64).sum(),
            })
            .collect();
        chain_offset(self.head.ctx() as i64, self.tail_conv.ctx() as i64, &stages, latent_start, latent_len)
    }
}

/// One stage's contribution to [`chain_offset`]'s forward walk: `up_ctx` = the upsample
/// conv_transpose's context (`ceil((k-1)/stride)`, `2` for every codec rate), `stride`, and
/// `res_ctx` = the sum of its 3 residual units' own context (`78` for this codec, `6*(1+3+9)`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StageOffset {
    pub up_ctx: i64,
    pub stride: i64,
    pub res_ctx: i64,
}

/// Pure forward position/length arithmetic for a chain fed `latent_len` samples starting at
/// `latent_start` in some true (unwindowed) latent stream. Returns `(audio_start, audio_len)`:
/// where the chain's first output sample lands in the true (unwindowed) audio stream, and how
/// long the chain's output actually is. Ports `decoder_chain.chain_offset` exactly (same formula,
/// walked forward instead of solved backward) -- factored out from [`S2DecoderChain`] so the
/// arithmetic is testable without a device: see `tests/chain_offset_matches_python_rail.rs`, which
/// gates this against `decoder_chain.chain_offset`'s own real output.
pub fn chain_offset(
    ctx_head: i64, ctx_tail: i64, stages: &[StageOffset], latent_start: i64, latent_len: i64,
) -> crate::Result<(i64, i64)> {
    let mut p = latent_start + ctx_head;
    let mut length = latent_len - ctx_head;
    if length <= 0 {
        return Err(S2Error::Shape(format!("head: {latent_len} latent samples <= CTX_HEAD={ctx_head}")));
    }
    for (i, s) in stages.iter().enumerate() {
        p = (p + s.up_ctx) * s.stride + s.res_ctx;
        length = (length - s.up_ctx) * s.stride - s.res_ctx;
        if length <= 0 {
            return Err(S2Error::Shape(format!(
                "stage {}: window exhausted (output length {length} <= 0) -- widen the input",
                i + 1
            )));
        }
    }
    let audio_start = p + ctx_tail;
    let audio_len = length - ctx_tail;
    if audio_len <= 0 {
        return Err(S2Error::Shape(format!("tail: output length {audio_len} <= 0")));
    }
    Ok((audio_start, audio_len))
}
