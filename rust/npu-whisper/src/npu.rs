//! NPU backend for the Whisper-small encoder.
//!
//! Routes the per-token linear projections (q/k/v/out and the two FFN matmuls) of each pre-norm
//! transformer block through the shared ctx2 K=768 whole-array kernel. Attention (MHA), LayerNorm,
//! GELU and the residual adds stay on host f32 (numerically equal-or-better than the on-chip
//! bf16 approximations) — see `bert::encoder` for the structurally identical post-norm template.
//!
//! Whisper's encoder sequence length is T'=1500, which exceeds the kernel's PAD_M=512 tile, so the
//! per-projection apply ROW-TILES the activation into chunks of <=512 rows (`apply_tiled`).

use std::path::Path;
use std::rc::Rc;

use ndarray::prelude::*;
use npu_asr::ctx2::{CtxAOp, FfnMm2, Precision, SharedCtxA};
use npu_xrt::Device;

/// Opened NPU device + the single resident ctx2 shared kernel (loaded once).
pub struct WhisperNpu {
    #[allow(dead_code)] // kept alive so the Rc<Device> outlives the SharedCtxA / op BOs.
    dev: Rc<Device>,
    pub shared: Rc<SharedCtxA>,
    pub precision: Precision,
}

impl WhisperNpu {
    /// Open the NPU and load the resident ctx2 kernel from `root` (the worktree root, where
    /// `mlir-aie/.../whole_array/build` resolves). Precision comes from `NPU_PRECISION` (default
    /// fast bf16).
    pub fn open(root: &Path) -> Self {
        let dev = Rc::new(Device::open(0).expect("open NPU (stop voxd.service / other NPU services first)"));
        let precision = Precision::from_env();
        let shared = SharedCtxA::with_precision(&dev, root, precision);
        WhisperNpu { dev, shared, precision }
    }

    /// Share the opened device (single-tenant) so a co-resident decoder can reuse it instead of
    /// double-opening `/dev/accel/accel0`.
    pub fn device(&self) -> Rc<Device> {
        Rc::clone(&self.dev)
    }
}

/// The NPU matmul ops for one pre-norm block: the four K=768 projections (q/k/v/out, n=768), the
/// FFN mm1 (n=3072, GELU applied on host), and the FFN mm2 (K=3072 -> 768 via the K-split FfnMm2).
/// Biases are applied on the NPU side via `Epi::Bias` — do NOT re-add them on host.
pub struct BlockOps {
    pub q: CtxAOp,
    pub k: CtxAOp,
    pub v: CtxAOp,
    pub out: CtxAOp,
    pub fc1: CtxAOp,
    pub fc2: FfnMm2,
}

/// Apply a K=768 -> n projection to `x` `[M, 768]` (M may exceed PAD_M=512), row-tiling into
/// chunks of <=512 and vstacking the `[M, n]` result in row order.
pub fn apply_tiled(op: &CtxAOp, x: &Array2<f32>, n: usize) -> Array2<f32> {
    let m = x.nrows();
    let mut out = Array2::<f32>::zeros((m, n));
    let mut r = 0;
    while r < m {
        let end = (r + npu_asr::engines::PAD_M).min(m);
        let chunk = x.slice(s![r..end, ..]).to_owned();
        let y = op.forward(&chunk); // [end-r, n]
        out.slice_mut(s![r..end, ..]).assign(&y);
        r = end;
    }
    out
}

/// RESIDENT-INTERMEDIATE FFN (flag-gated, `NPU_ENC_FFN_RESIDENT` + `NPU_ENC_GELU_FUSED`): fuse fc1 and
/// fc2 per row-tile so the `[mp, 3072]` fc1->fc2 intermediate stays in ONE bf16 buffer on the device
/// side (no host readback materialize / re-conversion of the largest data object -- the dominant
/// encoder marshaling seam). `ln2` is `[M, 768]` (M may exceed PAD_M=512); returns `[M, 768]`.
/// Numerically identical to `gelu(apply_tiled(fc1, ln2)) -> apply_tiled_mm2(fc2, .)` with the GELU
/// fused on-chip (the intermediate fc2 consumes is the same activated-f32 -> bf16 value).
pub fn apply_tiled_ffn_resident(fc1: &CtxAOp, fc2: &FfnMm2, ln2: &Array2<f32>) -> Array2<f32> {
    let m = ln2.nrows();
    let mut out = Array2::<f32>::zeros((m, 768));
    let mut r = 0;
    while r < m {
        let end = (r + npu_asr::engines::PAD_M).min(m);
        let chunk = ln2.slice(s![r..end, ..]).to_owned(); // [<=512, 768]
        let y = fc2.forward_resident(fc1, &chunk); // [end-r, 768], intermediate held resident
        out.slice_mut(s![r..end, ..]).assign(&y);
        r = end;
    }
    out
}

/// As `apply_tiled` but for the FFN mm2 (`FfnMm2`): `h` `[M, 3072]` -> `[M, 768]`, row-tiled.
pub fn apply_tiled_mm2(op: &FfnMm2, h: &Array2<f32>) -> Array2<f32> {
    let m = h.nrows();
    let mut out = Array2::<f32>::zeros((m, 768));
    let mut r = 0;
    while r < m {
        let end = (r + npu_asr::engines::PAD_M).min(m);
        let chunk = h.slice(s![r..end, ..]).to_owned();
        let y = op.forward(&chunk); // [end-r, 768]
        out.slice_mut(s![r..end, ..]).assign(&y);
        r = end;
    }
    out
}
