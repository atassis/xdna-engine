//! Control plane over npu-engine: a persistent desired-state config reconciled into loaded models,
//! behind a single device actor. Public: `Config`, `Handle`, `start`, `ModelStatus`.
pub mod config;
pub mod loader;
pub mod registry;
pub mod select;
pub mod reconcile;
pub mod actor;
pub mod http;

pub use config::Config;
pub use actor::{start, Handle, Served};
pub use reconcile::ReconcileReport;
pub use registry::{Capability, LoadState, ModelStatus};
pub use npu_engine::EngineError;
