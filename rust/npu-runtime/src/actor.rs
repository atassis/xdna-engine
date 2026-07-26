//! The single device owner. One thread holds the Registry (and the !Send models) and serves a
//! cloneable Send Handle over an mpsc channel - total serialization of the single-tenant NPU.
use std::sync::mpsc::{channel, RecvTimeoutError, Sender};
use std::thread::JoinHandle;
use std::time::Instant;

use crate::config::Config;
use crate::loader::ModelLoader;
use crate::reconcile::{reconcile, ReconcileReport};
use crate::registry::{deep_release_due, release_free_memory, Capability, ModelStatus, Registry};
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
/// This is the SERVICE start: a server should come up warm and answer `/v1/models` with what is
/// really resident. For a one-shot invocation use [`start_lazy`].
pub fn start(cfg: Config, loader: Box<dyn ModelLoader + Send>) -> (Handle, JoinHandle<()>) {
    spawn(cfg, loader, true)
}

/// Spawn the actor WITHOUT loading anything: models are declared from their scenarios (host-only,
/// no device) and load when a request routes to one.
///
/// For a one-shot `npu embed` / `npu transcribe`, the eager reconcile is pure waste and worse than
/// waste: with `max_resident = 1` it loads the first configured model, and the request then evicts it
/// to load the one it actually wanted -- two full device loads to serve one request. Worse, `npu
/// embed` against an ASR-only config paid a complete parakeet load before it could report that no
/// embed model was configured at all. Declaring is enough to route correctly.
pub fn start_lazy(cfg: Config, loader: Box<dyn ModelLoader + Send>) -> (Handle, JoinHandle<()>) {
    spawn(cfg, loader, false)
}

fn spawn(cfg: Config, loader: Box<dyn ModelLoader + Send>, eager: bool) -> (Handle, JoinHandle<()>) {
    let (tx, rx) = channel::<Cmd>();
    let (ready_tx, ready_rx) = channel::<()>();
    let join = std::thread::spawn(move || {
        let mut reg = Registry::default();
        let mut cfg = cfg;
        // A panic anywhere below used to kill this thread, after which every request failed with
        // "actor stopped" and the real cause was gone. Model constructors still `.expect()` on
        // missing artifacts, so a bad config or a moved weights dir landed here. Catch it: the
        // actor survives, and the panic message is returned as the error the caller sees.
        if eager {
            let _ = guard(|| reconcile(&cfg, &mut reg, loader.as_ref()));
        } else {
            for m in &cfg.models {
                let kind = guard(|| loader.declared_kind(m)).unwrap_or(None);
                reg.declare(m, kind);
            }
        }
        let _ = ready_tx.send(());
        // `recv_timeout` rather than `recv`: the actor is the ONLY owner of the single-tenant NPU,
        // so idle unload has to happen on this thread or not at all. Waking on a timeout gives the
        // timer for free and puts the sweep strictly BETWEEN commands -- an eviction can never race
        // a request in flight, and no second thread or lock enters the design.
        //
        // The wait is computed from a DEADLINE, not passed as a fixed interval. With a fixed
        // interval the timeout only fires after a fully quiet window, so any client polling
        // /healthz or /v1/models more often than sweep_interval_s would reset the wait forever and
        // idle models would never be released.
        let mut next_sweep = Instant::now() + cfg.server.sweep_interval();
        // Second level of idleness. `last_request` deliberately tracks REQUESTS, not commands: a
        // /healthz or /v1/models poll must not be able to hold the process at its working-set size
        // forever (the same starvation the sweep deadline above avoids). `released` latches so the
        // trim runs once per idle stretch, and is re-armed by a request or by an unload freeing more.
        let mut last_request = Instant::now();
        let mut released = false;
        loop {
            match rx.recv_timeout(next_sweep.saturating_duration_since(Instant::now())) {
                Ok(Cmd::Transcribe { model, pcm, sr, reply }) => {
                    last_request = Instant::now(); released = false;
                    let r = guard(|| run_transcribe(&cfg, &mut reg, loader.as_ref(), model.as_deref(), &pcm, sr))
                        .unwrap_or_else(|msg| Err(EngineError::Device(msg)));
                    let _ = reply.send(r);
                }
                Ok(Cmd::Embed { model, text, reply }) => {
                    last_request = Instant::now(); released = false;
                    let r = guard(|| run_embed(&cfg, &mut reg, loader.as_ref(), model.as_deref(), &text))
                        .unwrap_or_else(|msg| Err(EngineError::Device(msg)));
                    let _ = reply.send(r);
                }
                Ok(Cmd::Reconcile { cfg: newcfg, reply }) => {
                    last_request = Instant::now(); released = false;
                    cfg = *newcfg;
                    let rep = guard(|| reconcile(&cfg, &mut reg, loader.as_ref()))
                        .unwrap_or_else(|msg| ReconcileReport { failed: vec![msg], ..Default::default() });
                    let _ = reply.send(rep);
                }
                Ok(Cmd::Status { reply }) => { let _ = reply.send(reg.status()); }
                Ok(Cmd::Shutdown) => break,
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => break,
            }
            let now = Instant::now();
            if now >= next_sweep {
                if let Some(idle) = cfg.server.idle_unload() {
                    let freed = guard(|| reg.sweep_idle(now, idle)).unwrap_or_default();
                    for name in &freed {
                        eprintln!("[npu-runtime] unloaded {name}: idle >= {}s", idle.as_secs());
                    }
                    // An unload just freed a working set; there is something new worth trimming.
                    if !freed.is_empty() { released = false; }
                }
                // Level 2: the models are gone but the allocator is still holding their pages.
                if deep_release_due(cfg.server.idle_release(), now.saturating_duration_since(last_request), released) {
                    released = true;
                    if release_free_memory() {
                        eprintln!("[npu-runtime] released free memory to the OS: idle >= {}s",
                            cfg.server.idle_release_s);
                    }
                }
                next_sweep = now + cfg.server.sweep_interval();
            }
        }
    });
    let _ = ready_rx.recv();
    (Handle { tx }, join)
}

/// Route the request to a model, make that model resident, and verify it has the capability asked
/// for. Loading here is what turns "switch models" into a per-request choice instead of a config
/// edit + reload. Safe to load/evict at this point: the actor serves one command at a time.
fn serve_ready(cfg: &Config, reg: &mut Registry, loader: &dyn ModelLoader, cap: Capability,
               want: Option<&str>) -> Result<String, EngineError> {
    let name = resolve(cfg, reg, cap, want)?;
    let now = Instant::now();
    if reg.get_loaded(&name).is_none() {
        let mcfg = cfg.find(&name).cloned()
            .ok_or_else(|| EngineError::Load(format!("model {name:?} is not in the config")))?;
        reg.ensure_resident(&mcfg, loader, &cfg.server, now)?;
    }
    // The kind is only a promise until the model is actually loaded (a name resolved on demand has
    // never reported its capability), so this is the one authoritative check.
    match reg.get_loaded(&name).map(|m| m.kind()) {
        Some(k) if k == cap => {
            // Stamp on selection, not on success: a request that then fails inside the model still
            // counted as use, and must not be the next thing evicted or swept.
            reg.touch(&name, now);
            Ok(name)
        }
        Some(got) => Err(EngineError::WrongKind { wanted: cap, got }),
        None => Err(EngineError::Load(format!("{name} not loaded"))),
    }
}

fn run_transcribe(cfg: &Config, reg: &mut Registry, loader: &dyn ModelLoader, want: Option<&str>,
                  pcm: &[i16], sr: u32) -> Result<Served<String>, EngineError> {
    let name = serve_ready(cfg, reg, loader, Capability::Asr, want)?;
    let m = reg.get_loaded(&name).ok_or_else(|| EngineError::Load(format!("{name} not loaded")))?;
    Ok(Served { model: name.clone(), value: m.transcribe(pcm, sr)? })
}
fn run_embed(cfg: &Config, reg: &mut Registry, loader: &dyn ModelLoader, want: Option<&str>, text: &str)
    -> Result<Served<Vec<f32>>, EngineError> {
    let name = serve_ready(cfg, reg, loader, Capability::Embed, want)?;
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

// These live in the crate (not tests/actor.rs, which is behind the `testkit` feature) so the plain
// `cargo test --workspace` in scripts/ci_gate.sh actually runs them.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Defaults, ModelCfg, ServerCfg};
    use crate::loader::mock::MockLoader;
    use crate::registry::LoadState;
    use npu_engine::ModelKind;
    use std::collections::BTreeMap;
    use std::time::Duration;

    /// asr + embed configured, but only ONE slot: serving both means swapping.
    fn swap_setup(srv: ServerCfg) -> (Handle, JoinHandle<()>) {
        let mut t = BTreeMap::new();
        t.insert("asr".to_string(), Ok((ModelKind::Asr, 1)));
        t.insert("bge".to_string(), Ok((ModelKind::Embed, 1)));
        let cfg = Config {
            server: srv,
            defaults: Defaults { asr: Some("asr".into()), embed: Some("bge".into()) },
            models: vec![
                ModelCfg { name: "asr".into(), scenario: "x".into() },
                ModelCfg { name: "bge".into(), scenario: "y".into() },
            ],
        };
        start(cfg, Box::new(MockLoader { table: t }))
    }
    fn state_of(h: &Handle, name: &str) -> LoadState {
        h.status().into_iter().find(|s| s.name == name).expect("entry").state
    }

    #[test]
    fn one_slot_serves_both_models_by_swapping() {
        // idle_unload off: this test is about max_resident as an evict trigger, nothing else.
        let (h, j) = swap_setup(ServerCfg { max_resident: 1, idle_unload_s: 0, ..Default::default() });
        // Boot loaded the first configured model and deferred the second.
        assert_eq!(state_of(&h, "asr"), LoadState::Loaded);
        assert_eq!(state_of(&h, "bge"), LoadState::Unloaded);
        // An embed request pulls bge in on demand, evicting asr...
        assert_eq!(h.embed(None, "hi").unwrap().model, "bge");
        assert_eq!(state_of(&h, "bge"), LoadState::Loaded);
        assert_eq!(state_of(&h, "asr"), LoadState::Unloaded);
        // ...and asr swaps back for a transcription. Before this, the second model just Failed.
        let tr = h.transcribe(None, vec![0i16; 4], 16_000).unwrap();
        assert_eq!((tr.model.as_str(), tr.value.as_str()), ("asr", "mock-text"));
        assert_eq!(state_of(&h, "asr"), LoadState::Loaded);
        h.shutdown(); j.join().unwrap();
    }

    #[test]
    fn explicit_model_wins_over_the_default() {
        let (h, j) = swap_setup(ServerCfg { max_resident: 2, idle_unload_s: 0, ..Default::default() });
        assert_eq!(h.embed(Some("bge"), "hi").unwrap().model, "bge");
        // Naming an ASR model on the embed route is still a WrongKind error, not a silent swap.
        assert!(h.embed(Some("asr"), "hi").is_err());
        h.shutdown(); j.join().unwrap();
    }

    #[test]
    fn idle_sweep_releases_the_device_then_reloads_on_demand() {
        let (h, j) = swap_setup(ServerCfg {
            max_resident: 2, idle_unload_s: 1, sweep_interval_s: 1, ..Default::default()
        });
        assert_eq!(h.embed(None, "hi").unwrap().model, "bge");
        // Poll until the actor's own sweep releases it. Polling this fast is deliberate: it is the
        // regression test for the deadline (a fixed recv_timeout interval would be reset by every
        // poll and never fire).
        let deadline = Instant::now() + Duration::from_secs(20);
        while state_of(&h, "bge") == LoadState::Loaded && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(50));
        }
        let s = h.status().into_iter().find(|s| s.name == "bge").unwrap();
        assert_eq!(s.state, LoadState::Unloaded, "idle model was never swept: {}", s.detail);
        assert!(s.detail.contains("idle"), "{}", s.detail);
        assert_eq!(s.idle_s, None);
        // The whole point: the next request just works, no /admin/reload.
        assert_eq!(h.embed(None, "again").unwrap().model, "bge");
        assert_eq!(state_of(&h, "bge"), LoadState::Loaded);
        h.shutdown(); j.join().unwrap();
    }

    #[test]
    fn lazy_start_declares_without_loading_then_loads_only_what_it_serves() {
        let mut t = BTreeMap::new();
        t.insert("asr".to_string(), Ok((ModelKind::Asr, 1)));
        t.insert("bge".to_string(), Ok((ModelKind::Embed, 1)));
        let cfg = Config {
            server: ServerCfg { max_resident: 1, idle_unload_s: 0, ..Default::default() },
            defaults: Defaults { asr: Some("asr".into()), embed: Some("bge".into()) },
            models: vec![
                ModelCfg { name: "asr".into(), scenario: "x".into() },
                ModelCfg { name: "bge".into(), scenario: "y".into() },
            ],
        };
        let (h, j) = start_lazy(cfg, Box::new(MockLoader { table: t }));
        // Nothing loaded, but both are known -- including their capability, read from the scenario.
        let s = h.status();
        assert!(s.iter().all(|x| x.state == LoadState::Unloaded), "lazy start must not load: {s:?}");
        assert_eq!(s.iter().find(|x| x.name == "bge").unwrap().kind, Some(ModelKind::Embed));
        // The embed request loads bge and ONLY bge -- eagerly, this would have loaded asr first and
        // then evicted it at max_resident = 1.
        assert_eq!(h.embed(None, "hi").unwrap().model, "bge");
        assert_eq!(state_of(&h, "bge"), LoadState::Loaded);
        assert_eq!(state_of(&h, "asr"), LoadState::Unloaded);
        h.shutdown(); j.join().unwrap();
    }

    #[test]
    fn lazy_start_reports_a_missing_capability_without_touching_the_device() {
        // The shipped shape: one ASR model, no embed model anywhere. `npu embed` must say so, not
        // load an ASR model to find out.
        let mut t = BTreeMap::new();
        t.insert("asr".to_string(), Ok((ModelKind::Asr, 1)));
        let cfg = Config {
            server: ServerCfg { max_resident: 1, idle_unload_s: 0, ..Default::default() },
            defaults: Defaults { asr: Some("asr".into()), embed: None },
            models: vec![ModelCfg { name: "asr".into(), scenario: "x".into() }],
        };
        let (h, j) = start_lazy(cfg, Box::new(MockLoader { table: t }));
        let e = match h.embed(None, "hi") { Err(e) => e.to_string(), Ok(s) => panic!("served {}", s.model) };
        assert!(e.contains("no embed model"), "{e}");
        assert_eq!(state_of(&h, "asr"), LoadState::Unloaded, "nothing may load to answer this");
        // ...and ASR still works on the same actor.
        assert_eq!(h.transcribe(None, vec![0i16; 4], 16_000).unwrap().model, "asr");
        h.shutdown(); j.join().unwrap();
    }

    #[test]
    fn idle_unload_zero_keeps_models_resident() {
        let (h, j) = swap_setup(ServerCfg {
            max_resident: 2, idle_unload_s: 0, sweep_interval_s: 1, ..Default::default()
        });
        assert_eq!(h.embed(None, "hi").unwrap().model, "bge");
        std::thread::sleep(Duration::from_millis(2500));
        assert_eq!(state_of(&h, "bge"), LoadState::Loaded, "idle_unload_s = 0 must disable the sweep");
        h.shutdown(); j.join().unwrap();
    }
}
