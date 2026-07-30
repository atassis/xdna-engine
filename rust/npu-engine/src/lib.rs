//! General multi-model engine over the XDNA2 NPU kernel kit.
//!
//! Public API: [`Engine`], [`Model`], [`ModelKind`], [`EngineError`]. Load a model from a scenario
//! TOML and call [`Model::transcribe`] or [`Model::embed`]. Everything else in this crate is
//! implementation detail (`#[doc(hidden)]`) and may change without notice.

pub mod api;
pub use api::{Engine, EngineError, Model, ModelKind};

#[doc(hidden)] pub mod config;
#[doc(hidden)] pub mod pipeline;
#[doc(hidden)] pub mod registry;
#[doc(hidden)] pub mod bert;
#[doc(hidden)] pub mod esm;
#[doc(hidden)] pub mod asr;
#[doc(hidden)] pub mod tuning_profile;
// PROBE (engine-open-capability-contract): the open request-surface trait. `#[doc(hidden)]` for now
// like the other internals above -- it is a candidate contract validated against two instances, not
// yet the wired-in replacement for `pipeline::Scenario` / `npu-runtime`'s closed routing.
#[doc(hidden)] pub mod capability;
