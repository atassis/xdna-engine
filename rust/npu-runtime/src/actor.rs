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
use npu_engine::capability::{Request, Response, Segment};
use npu_engine::EngineError;

/// Result carrying which model served (the echo).
pub struct Served<T> { pub model: String, pub value: T }

/// A model answered a capability with a payload shape that capability never returns. Reachable only
/// from a buggy `Servable` impl, not from any request -- routing has already checked the capability
/// by this point -- so it names the model rather than blaming the caller.
fn wrong_shape(cap: Capability, model: &str, got: &Response) -> EngineError {
    EngineError::Device(format!("{model} served {cap} but returned a {} response", got.shape()))
}

/// Run `f`, converting a panic into `Err(message)` instead of unwinding out of the actor thread.
///
/// The engine's model constructors still `.expect()` on missing artifacts (a moved weights dir, a
/// stale scenario), so a panic here is reachable from ordinary misconfiguration. Before this, such
/// a panic killed the actor and every later request returned the useless "actor stopped" while the
/// real message went only to stderr. `AssertUnwindSafe` is required because the registry holds
/// `Box<dyn Inference>`; the actor owns that state exclusively and does not observe it again after
/// a caught panic beyond reporting, so no torn state escapes.
pub(crate) fn guard<T>(f: impl FnOnce() -> T) -> Result<T, String> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).map_err(|p| {
        if let Some(s) = p.downcast_ref::<&str>() { (*s).to_string() }
        else if let Some(s) = p.downcast_ref::<String>() { s.clone() }
        else { "panic in engine (no message)".to_string() }
    })
}

enum Cmd {
    /// One command for every capability. It used to be one variant per modality, which is why
    /// adding a third meant editing this enum, its match arm, `Handle`, and the trait it calls.
    Serve {
        cap: Capability,
        model: Option<String>,
        req: Request,
        reply: Sender<Result<Served<Response>, EngineError>>,
    },
    Reconcile { cfg: Box<Config>, reply: Sender<ReconcileReport> },
    Status { reply: Sender<Vec<ModelStatus>> },
    Shutdown,
}

#[derive(Clone)]
pub struct Handle { tx: Sender<Cmd> }

/// Spawn the actor with an initial config + a loader; performs the initial reconcile before returning.
/// This is the SERVICE start: a server should come up warm and answer `/v1/models` with what is
/// really resident. For a one-shot invocation use [`start_lazy`].
///
/// `Err` means the initial reconcile PANICKED (e.g. a bug outside the normal per-model load-failure
/// path, which `reconcile` already reports via `ModelStatus` without panicking) -- the actor thread
/// is shut down before returning, so a caller never holds a `Handle` to a half-dead actor that was
/// never actually reconciled.
pub fn start(cfg: Config, loader: Box<dyn ModelLoader + Send>) -> Result<(Handle, JoinHandle<()>), EngineError> {
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
pub fn start_lazy(cfg: Config, loader: Box<dyn ModelLoader + Send>) -> Result<(Handle, JoinHandle<()>), EngineError> {
    spawn(cfg, loader, false)
}

fn spawn(cfg: Config, loader: Box<dyn ModelLoader + Send>, eager: bool) -> Result<(Handle, JoinHandle<()>), EngineError> {
    let (tx, rx) = channel::<Cmd>();
    let (ready_tx, ready_rx) = channel::<Result<(), String>>();
    let join = std::thread::spawn(move || {
        let mut reg = Registry::default();
        let mut cfg = cfg;
        // A panic anywhere below used to kill this thread, after which every request failed with
        // "actor stopped" and the real cause was gone. Model constructors still `.expect()` on
        // missing artifacts, so a bad config or a moved weights dir landed here. Catch it: the
        // actor survives, and the panic message is sent back to `spawn` instead of being dropped on
        // the floor (`let _ = ready_rx.recv()` used to discard it, handing the caller a `Handle` to
        // an actor whose initial reconcile silently never ran).
        let init: Result<(), String> = if eager {
            guard(|| reconcile(&cfg, &mut reg, loader.as_ref())).map(|_report| ())
        } else {
            for m in &cfg.models {
                let cap = guard(|| loader.declared_capability(m)).unwrap_or(None);
                reg.declare(m, cap);
            }
            Ok(())
        };
        let init_failed = init.is_err();
        let _ = ready_tx.send(init);
        // The caller already got (and will act on) the Err above; if the init panicked, don't run
        // the serve loop on a registry that was never actually reconciled -- exit so the thread this
        // function spawned does not idle forever un-owned (spawn() sends Shutdown, but only after
        // learning the send above failed; exiting here makes that race harmless either way).
        if init_failed {
            return;
        }
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
                Ok(Cmd::Serve { cap, model, req, reply }) => {
                    last_request = Instant::now(); released = false;
                    // Two guarded steps rather than one, so the model NAME is known when the second
                    // fails. The shipped failure is a PANIC inside the dispatch (a missing insts
                    // file panics in npu-asr), which unwinds past any Result handling inside the
                    // call -- so condemning the model has to happen out here, after catch_unwind.
                    let ready = guard(|| serve_ready(&cfg, &mut reg, loader.as_ref(), cap, model.as_deref()))
                        .unwrap_or_else(|msg| Err(EngineError::Device(msg)));
                    let r = match ready {
                        Err(e) => Err(e),
                        Ok(name) => {
                            let out = guard(|| run_named(&mut reg, &name, req))
                                .unwrap_or_else(|msg| Err(EngineError::Device(msg)));
                            match out {
                                Ok(value) => Ok(Served { model: name, value }),
                                Err(e) => {
                                    if condemns_model(&e) {
                                        reg.mark_failed(&name, &e.to_string());
                                        eprintln!("[npu-runtime] {name} FAILED serving {cap}: {e}");
                                    }
                                    Err(e)
                                }
                            }
                        }
                    };
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
    match ready_rx.recv() {
        Ok(Ok(())) => Ok((Handle { tx }, join)),
        Ok(Err(msg)) => {
            // The thread already exited on its own (init_failed branch above); Shutdown is a no-op if
            // it beat us here, harmless either way. join() cannot hang: the thread returns right after
            // sending on ready_tx.
            let _ = tx.send(Cmd::Shutdown);
            let _ = join.join();
            Err(EngineError::Load(format!("initial reconcile: {msg}")))
        }
        // The sender was dropped without sending -- the thread panicked before reaching guard()
        // itself (e.g. inside Registry::default()). No Handle to hand back; nothing to shut down.
        Err(_) => Err(EngineError::Device("actor thread died before completing its initial reconcile".into())),
    }
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
    // The capability is only a promise until the model is actually loaded (a name resolved on demand
    // has never reported it), so this is the one authoritative check.
    match reg.get_loaded(&name).map(|m| m.capabilities()) {
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

fn run_named(reg: &mut Registry, name: &str, req: Request) -> Result<Response, EngineError> {
    let m = reg.get_loaded_mut(name).ok_or_else(|| EngineError::Load(format!("{name} not loaded")))?;
    m.run(req)
}

/// Whether a failure condemns the MODEL or just this request.
///
/// A device or load error is a property of the model -- a missing instruction stream fails
/// identically for every caller, forever. WrongKind/Unsupported/NoModel are the caller's problem and
/// must never condemn a working model.
fn condemns_model(e: &EngineError) -> bool {
    matches!(e, EngineError::Device(_) | EngineError::Load(_))
}

impl Handle {
    /// Serve any capability. The typed helpers below are conveniences over this; a caller with a
    /// capability that has no helper (tts, generate) uses it directly.
    pub fn serve(&self, cap: Capability, model: Option<&str>, req: Request)
        -> Result<Served<Response>, EngineError> {
        let (r, rx) = channel();
        self.tx.send(Cmd::Serve { cap, model: model.map(String::from), req, reply: r })
            .map_err(|_| EngineError::Device("actor stopped".into()))?;
        rx.recv().map_err(|_| EngineError::Device("actor dropped reply".into()))?
    }
    pub fn transcribe(&self, model: Option<&str>, pcm: Vec<i16>, sr: u32) -> Result<Served<String>, EngineError> {
        let s = self.serve(Capability::ASR, model, Request::Audio { pcm, sample_rate: sr })?;
        match s.value {
            Response::Text(t) => Ok(Served { model: s.model, value: t }),
            other => Err(wrong_shape(Capability::ASR, &s.model, &other)),
        }
    }
    pub fn embed(&self, model: Option<&str>, text: &str) -> Result<Served<Vec<f32>>, EngineError> {
        let s = self.serve(Capability::EMBED, model, Request::Text(text.to_string()))?;
        match s.value {
            Response::Vector(v) => Ok(Served { model: s.model, value: v }),
            other => Err(wrong_shape(Capability::EMBED, &s.model, &other)),
        }
    }
    pub fn diarize(&self, model: Option<&str>, pcm: Vec<i16>, sr: u32)
        -> Result<Served<Vec<Segment>>, EngineError> {
        let s = self.serve(Capability::DIARIZE, model, Request::Audio { pcm, sample_rate: sr })?;
        match s.value {
            Response::Segments(v) => Ok(Served { model: s.model, value: v }),
            other => Err(wrong_shape(Capability::DIARIZE, &s.model, &other)),
        }
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
    use std::collections::BTreeMap;
    use std::time::Duration;

    /// asr + embed configured, but only ONE slot: serving both means swapping. `.unwrap()`: a mock
    /// loader's initial reconcile does not panic, so a start() failure here is a real regression.
    fn swap_setup(srv: ServerCfg) -> (Handle, JoinHandle<()>) {
        let mut t = BTreeMap::new();
        t.insert("asr".to_string(), Ok((Capability::ASR, 1)));
        t.insert("bge".to_string(), Ok((Capability::EMBED, 1)));
        let cfg = Config {
            server: srv,
            defaults: Defaults::from_pairs([
                (Capability::ASR, "asr".to_string()), (Capability::EMBED, "bge".to_string())]),
            models: vec![
                ModelCfg { name: "asr".into(), scenario: "x".into() },
                ModelCfg { name: "bge".into(), scenario: "y".into() },
            ],
        };
        start(cfg, Box::new(MockLoader { table: t })).unwrap()
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
        t.insert("asr".to_string(), Ok((Capability::ASR, 1)));
        t.insert("bge".to_string(), Ok((Capability::EMBED, 1)));
        let cfg = Config {
            server: ServerCfg { max_resident: 1, idle_unload_s: 0, ..Default::default() },
            defaults: Defaults::from_pairs([
                (Capability::ASR, "asr".to_string()), (Capability::EMBED, "bge".to_string())]),
            models: vec![
                ModelCfg { name: "asr".into(), scenario: "x".into() },
                ModelCfg { name: "bge".into(), scenario: "y".into() },
            ],
        };
        let (h, j) = start_lazy(cfg, Box::new(MockLoader { table: t })).unwrap();
        // Nothing loaded, but both are known -- including their capability, read from the scenario.
        let s = h.status();
        assert!(s.iter().all(|x| x.state == LoadState::Unloaded), "lazy start must not load: {s:?}");
        assert_eq!(s.iter().find(|x| x.name == "bge").unwrap().capability, Some(Capability::EMBED));
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
        t.insert("asr".to_string(), Ok((Capability::ASR, 1)));
        let cfg = Config {
            server: ServerCfg { max_resident: 1, idle_unload_s: 0, ..Default::default() },
            defaults: Defaults::from_pairs([(Capability::ASR, "asr".to_string())]),
            models: vec![ModelCfg { name: "asr".into(), scenario: "x".into() }],
        };
        let (h, j) = start_lazy(cfg, Box::new(MockLoader { table: t })).unwrap();
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

    /// A loader whose `load()` panics instead of returning `Err` -- simulates a bug outside the
    /// normal per-model load-failure path (which `reconcile` already records as `Failed`, no panic
    /// involved). Before this fix `start()` swallowed this via `let _ = ready_rx.recv()` and handed
    /// back a `Handle` to an actor whose initial reconcile silently never completed.
    struct PanicLoader;
    impl ModelLoader for PanicLoader {
        fn load(&self, _cfg: &ModelCfg) -> Result<Box<dyn crate::loader::Servable>, EngineError> {
            panic!("boom: simulated load-time bug");
        }
    }

    /// A model that loads fine and then PANICS on every dispatch -- the shipped failure when an
    /// instruction stream is missing. It must be marked Failed, not stay Loaded while every request
    /// errors, which is the "reports healthy while broken" defect.
    struct PanicOnRun;
    impl crate::loader::Servable for PanicOnRun {
        fn capabilities(&self) -> Capability { Capability::EMBED }
        fn run(&mut self, _req: Request) -> Result<Response, EngineError> {
            panic!("read instr insts_512x800x768.txt: No such file or directory");
        }
    }
    struct PanicOnRunLoader;
    impl ModelLoader for PanicOnRunLoader {
        fn load(&self, _cfg: &ModelCfg) -> Result<Box<dyn crate::loader::Servable>, EngineError> {
            Ok(Box::new(PanicOnRun))
        }
        fn declared_capability(&self, _cfg: &ModelCfg) -> Option<Capability> { Some(Capability::EMBED) }
    }

    #[test]
    fn a_model_that_panics_while_serving_is_marked_failed() {
        let cfg = Config {
            server: ServerCfg { max_resident: 1, idle_unload_s: 0, ..Default::default() },
            defaults: Defaults::from_pairs([(Capability::EMBED, "bge".to_string())]),
            models: vec![ModelCfg { name: "bge".into(), scenario: "x".into() }],
        };
        let (h, j) = start(cfg, Box::new(PanicOnRunLoader)).unwrap();
        assert_eq!(state_of(&h, "bge"), LoadState::Loaded, "it loads fine; the panic is at dispatch");
        let e = match h.embed(None, "hi") { Err(e) => e.to_string(), Ok(s) => panic!("served {}", s.model) };
        assert!(e.contains("No such file"), "the real cause must reach the caller: {e}");
        // The point: the failure STICKS to the model instead of vanishing with the request.
        let s = h.status().into_iter().find(|s| s.name == "bge").unwrap();
        assert_eq!(s.state, LoadState::Failed, "a dispatch panic must condemn the model");
        assert!(s.detail.contains("No such file"), "{}", s.detail);
        h.shutdown(); j.join().unwrap();
    }

    /// ...but a request-shaped error must NOT condemn a working model.
    #[test]
    fn a_wrong_capability_request_does_not_condemn_the_model() {
        let (h, j) = swap_setup(ServerCfg { max_resident: 2, idle_unload_s: 0, ..Default::default() });
        assert_eq!(h.embed(None, "hi").unwrap().model, "bge");
        assert!(h.embed(Some("asr"), "hi").is_err(), "asr cannot embed");
        assert_ne!(state_of(&h, "asr"), LoadState::Failed, "a routing error is the caller's fault");
        assert_eq!(h.embed(None, "hi").unwrap().model, "bge", "and bge still serves");
        h.shutdown(); j.join().unwrap();
    }

    /// A panicking loader is recorded as Failed rather than unwinding through whoever triggered it.
    #[test]
    fn a_panicking_load_on_the_request_path_is_recorded_not_propagated() {
        let cfg = Config {
            server: ServerCfg { max_resident: 1, idle_unload_s: 0, ..Default::default() },
            defaults: Defaults::from_pairs([(Capability::EMBED, "bge".to_string())]),
            models: vec![ModelCfg { name: "bge".into(), scenario: "x".into() }],
        };
        // start_lazy: nothing loads until the request, so the panic happens on the REQUEST path.
        let (h, j) = start_lazy(cfg, Box::new(PanicLoader)).unwrap();
        let e = match h.embed(None, "hi") { Err(e) => e.to_string(), Ok(s) => panic!("served {}", s.model) };
        assert!(e.contains("boom"), "{e}");
        assert_eq!(state_of(&h, "bge"), LoadState::Failed);
        h.shutdown(); j.join().unwrap();
    }

    /// A panicking LOAD during the initial reconcile is now recorded as `Failed`, not propagated as
    /// a `start()` error.
    ///
    /// This changed deliberately when `try_load` started catching loader panics. The old contract
    /// ("start() returns Err") collapsed every model into one panic string; the new one names each
    /// model and its cause, which is what `npu serve` prints before refusing to bind. The protective
    /// intent is unchanged and asserted here: the caller must never be left believing the model
    /// loaded.
    #[test]
    fn a_panicking_load_during_reconcile_is_recorded_as_failed() {
        let cfg = Config {
            server: ServerCfg { max_resident: 1, idle_unload_s: 0, ..Default::default() },
            defaults: Defaults::from_pairs([(Capability::ASR, "asr".to_string())]),
            models: vec![ModelCfg { name: "asr".into(), scenario: "x".into() }],
        };
        let (h, j) = match start(cfg, Box::new(PanicLoader)) {
            Ok(v) => v,
            Err(e) => panic!("a load panic is a per-model failure, not a start() error: {e}"),
        };
        let s = h.status().into_iter().find(|s| s.name == "asr").expect("entry");
        assert_eq!(s.state, LoadState::Failed, "the model must not look healthy");
        assert!(s.detail.contains("boom"), "the panic message is the cause: {}", s.detail);
        h.shutdown(); j.join().unwrap();
    }
}
