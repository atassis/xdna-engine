//! GigaAM-v3 Conformer encoder driven from Rust on the XDNA2 NPU (Rung 6).
//!
//! Matmul-heavy ops (FFN×2, q/k/v/out projections, pointwise1/2) run on the NPU as
//! whole-array + bias(+SiLU) epilogue dispatches with reused buffers; depthwise-conv on
//! the NPU; the LayerNorm/RoPE/attention/GLU/softmax/residual glue runs on the host via
//! `npu_asr_host`. Ported from `npu_asr/fused.py` (RETIRED 2026-06-11); structure deliberately
//! diverges -- see npu_asr/RETIRED.md. Do not restore parity: that reintroduces per-op dispatch.

pub mod weights;
pub mod kernel_registry;
pub mod engines;
pub mod conv_npu;
pub mod ctx_decode;
#[cfg(feature = "two_ctx")]
pub mod ctx2;
#[cfg(feature = "two_ctx")]
pub mod ctx_ln;
#[cfg(feature = "two_ctx")]
pub mod tuning;
pub mod block;
pub mod encoder;
