//! General multi-model engine over the XDNA2 NPU kernel kit. A model is a 3-stage pipeline
//! (Frontend -> Encoder -> Head) selected by a TOML scenario. The Encoder stage reuses the
//! existing `npu-asr` whole-array matmul engines unchanged.

pub mod config;
pub mod pipeline;
pub mod registry;
pub mod bert;
pub mod asr;
pub mod tuning_profile;
