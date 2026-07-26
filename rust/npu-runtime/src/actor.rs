//! The single device owner. One thread holds the Registry (and the !Send models) and serves a
//! cloneable Send Handle over an mpsc channel - total serialization of the single-tenant NPU.
use std::sync::mpsc::{channel, Sender};
use std::thread::JoinHandle;

use crate::config::Config;
use crate::loader::ModelLoader;
use crate::reconcile::{reconcile, ReconcileReport};
use crate::registry::{Capability, ModelStatus, Registry};
use crate::select::resolve;
use npu_engine::EngineError;

/// Result carrying which model served (the echo).
pub struct Served<T> { pub model: String, pub value: T }

/// Run `f`, converting a panic into `Err(message)` instead of unwinding out of the actor thread.
///
/// The engine's model constructors still `.expect()` on missing artifacts (a moved weights dir, a
/// stale scenario), so a panic here is reachable from ordinary misconfiguration. Before this, such
/// a panic killed the actor and every later request returned the useless "actor stopped" while the
/// real message went only to stderr. `AssertUnwindSafe` is required because the registry holds
/// `Box<dyn Inference>`; the actor owns that state exclusively and does not observe it again after
/// a caught panic beyond reporting, so no torn state escapes.
fn guard<T>(f: impl FnOnce() -> T) -> Result<T, String> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).map_err(|p| {
        if let Some(s) = p.downcast_ref::<&str>() { (*s).to_string() }
        else if let Some(s) = p.downcast_ref::<String>() { s.clone() }
        else { "panic in engine (no message)".to_string() }
    })
}

enum Cmd {
    Transcribe { model: Option<String>, pcm: Vec<i16>, sr: u32, reply: Sender<Result<Served<String>, EngineError>> },
    Embed { model: Option<String>, text: String, reply: Sender<Result<Served<Vec<f32>>, EngineError>> },
    Reconcile { cfg: Box<Config>, reply: Sender<ReconcileReport> },
    Status { reply: Sender<Vec<ModelStatus>> },
    Shutdown,
}

#[derive(Clone)]
pub struct Handle { tx: Sender<Cmd> }

/// Spawn the actor with an initial config + a loader; performs the initial reconcile before returning.
pub fn start(cfg: Config, loader: Box<dyn ModelLoader + Send>) -> (Handle, JoinHandle<()>) {
    let (tx, rx) = channel::<Cmd>();
    let (ready_tx, ready_rx) = channel::<()>();
    let join = std::thread::spawn(move || {
        let mut reg = Registry::default();
        let mut cfg = cfg;
        // A panic anywhere below used to kill this thread, after which every request failed with
        // "actor stopped" and the real cause was gone. Model constructors still `.expect()` on
        // missing artifacts, so a bad config or a moved weights dir landed here. Catch it: the
        // actor survives, and the panic message is returned as the error the caller sees.
        let _ = guard(|| reconcile(&cfg, &mut reg, loader.as_ref()));
        let _ = ready_tx.send(());
        while let Ok(cmd) = rx.recv() {
            match cmd {
                Cmd::Transcribe { model, pcm, sr, reply } => {
                    let r = guard(|| run_transcribe(&cfg, &reg, model.as_deref(), &pcm, sr))
                        .unwrap_or_else(|msg| Err(EngineError::Device(msg)));
                    let _ = reply.send(r);
                }
                Cmd::Embed { model, text, reply } => {
                    let r = guard(|| run_embed(&cfg, &reg, model.as_deref(), &text))
                        .unwrap_or_else(|msg| Err(EngineError::Device(msg)));
                    let _ = reply.send(r);
                }
                Cmd::Reconcile { cfg: newcfg, reply } => {
                    cfg = *newcfg;
                    let rep = guard(|| reconcile(&cfg, &mut reg, loader.as_ref()))
                        .unwrap_or_else(|msg| ReconcileReport { failed: vec![msg], ..Default::default() });
                    let _ = reply.send(rep);
                }
                Cmd::Status { reply } => { let _ = reply.send(reg.status()); }
                Cmd::Shutdown => break,
            }
        }
    });
    let _ = ready_rx.recv();
    (Handle { tx }, join)
}

fn run_transcribe(cfg: &Config, reg: &Registry, want: Option<&str>, pcm: &[i16], sr: u32)
    -> Result<Served<String>, EngineError> {
    let name = resolve(cfg, reg, Capability::Asr, want)?;
    let m = reg.get_loaded(&name).ok_or_else(|| EngineError::Load(format!("{name} not loaded")))?;
    Ok(Served { model: name.clone(), value: m.transcribe(pcm, sr)? })
}
fn run_embed(cfg: &Config, reg: &Registry, want: Option<&str>, text: &str)
    -> Result<Served<Vec<f32>>, EngineError> {
    let name = resolve(cfg, reg, Capability::Embed, want)?;
    let m = reg.get_loaded(&name).ok_or_else(|| EngineError::Load(format!("{name} not loaded")))?;
    Ok(Served { model: name.clone(), value: m.embed(text)? })
}

impl Handle {
    pub fn transcribe(&self, model: Option<&str>, pcm: Vec<i16>, sr: u32) -> Result<Served<String>, EngineError> {
        let (r, rx) = channel();
        self.tx.send(Cmd::Transcribe { model: model.map(String::from), pcm, sr, reply: r })
            .map_err(|_| EngineError::Device("actor stopped".into()))?;
        rx.recv().map_err(|_| EngineError::Device("actor dropped reply".into()))?
    }
    pub fn embed(&self, model: Option<&str>, text: &str) -> Result<Served<Vec<f32>>, EngineError> {
        let (r, rx) = channel();
        self.tx.send(Cmd::Embed { model: model.map(String::from), text: text.into(), reply: r })
            .map_err(|_| EngineError::Device("actor stopped".into()))?;
        rx.recv().map_err(|_| EngineError::Device("actor dropped reply".into()))?
    }
    pub fn reconcile(&self, cfg: Config) -> Result<ReconcileReport, EngineError> {
        let (r, rx) = channel();
        self.tx.send(Cmd::Reconcile { cfg: Box::new(cfg), reply: r })
            .map_err(|_| EngineError::Device("actor stopped".into()))?;
        rx.recv().map_err(|_| EngineError::Device("actor dropped reply".into()))
    }
    pub fn status(&self) -> Vec<ModelStatus> {
        let (r, rx) = channel();
        if self.tx.send(Cmd::Status { reply: r }).is_err() { return vec![]; }
        rx.recv().unwrap_or_default()
    }
    pub fn shutdown(&self) { let _ = self.tx.send(Cmd::Shutdown); }
}
