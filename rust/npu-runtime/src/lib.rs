//! Control plane over npu-engine: a persistent desired-state config reconciled into loaded models,
//! behind a single device actor. Public: `Config`, `Handle`, `start`, `ModelStatus`.
pub mod config;
pub mod config_doc;
pub mod loader;
pub mod registry;
pub mod status_file;
pub mod select;
pub mod reconcile;
pub mod actor;
pub mod http;
pub mod media;
pub mod stream;

pub use config::Config;
pub use config_doc::ConfigDoc;
pub use actor::{start, start_lazy, Handle, LoadReport, Served};
pub use reconcile::ReconcileReport;
pub use registry::{Capability, LoadState, ModelStatus};
pub use npu_engine::EngineError;
