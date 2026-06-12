//! GigaAM-v3 Conformer encoder driven from Rust on the XDNA2 NPU (Rung 6).
//!
//! Matmul-heavy ops (FFN×2, q/k/v/out projections, pointwise1/2) run on the NPU as
//! whole-array + bias(+SiLU) epilogue dispatches with reused buffers; depthwise-conv on
//! the NPU; the LayerNorm/RoPE/attention/GLU/softmax/residual glue runs on the host via
//! `npu_asr_host`. Mirrors `npu_asr/fused.py` (the Python correctness oracle).

pub mod weights;
pub mod engines;
#[cfg(feature = "two_ctx")]
pub mod ctx2;
#[cfg(feature = "two_ctx")]
pub mod ctx_ln;
pub mod block;
pub mod encoder;
