//! The single device owner. One thread holds the Registry (and the !Send models) and serves a
//! cloneable Send Handle over an mpsc channel - total serialization of the single-tenant NPU.
use std::sync::mpsc::{channel, sync_channel, RecvTimeoutError, Sender, SyncSender};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use std::thread::JoinHandle;
use std::time::Instant;

use crate::config::{Config, EvictPolicy};
use crate::control_socket::LiveStatus;
use crate::loader::ModelLoader;
use crate::reconcile::{reconcile, ReconcileReport};
use crate::registry::{deep_release_due, is_hwctx_exhaustion, release_free_memory, Capability,
                      ModelStatus, Registry, UnloadReason};
use crate::select::resolve;
use crate::stream::StreamItem;
use npu_engine::capability::{Request, Response, Segment};
use npu_engine::{Chunk, EngineError, GenerateParams, Prompt};

/// Bound on the actor -> socket channel for a streaming generation. Bounded so a slow client cannot
/// make the generator race ahead and allocate without limit; small because a token is one item, not
/// a batch of work.
const GENERATE_CHANNEL_CAP: usize = 8;

/// Result carrying which model served (the echo).
pub struct Served<T> {
    pub model: String,
    pub value: T,
    /// `queue_us`/`load_us`: how long the request waited for the actor and for its model to become
    /// resident; 0 on `Handle::generate`, whose report carries its own.
    pub queue_us: u64,
    pub load_us: u64,
}

/// What an explicit load did, so the caller can report it without a second round trip.
#[derive(Debug, Clone, PartialEq)]
pub struct LoadReport {
    /// False when the model was ALREADY resident. The load is idempotent, and saying which of the
    /// two happened is the difference between "I warmed the device" and "nothing to do".
    pub loaded: bool,
    pub resident_mb: u64,
    pub ceiling_mb: u64,
    /// Resident models the memory accountant cannot weigh. Non-empty means `memory_ceiling_mb` is
    /// not bounding anything, which an operator who just set it needs told.
    pub unweighed: Vec<String>,
}

/// A model answered a capability with a payload shape that capability never returns. Reachable only
/// from a buggy `Servable` impl, not from any request -- routing has already checked the capability
/// by this point -- so it names the model rather than blaming the caller.
fn wrong_shape(cap: Capability, model: &str, got: &Response) -> EngineError {
    EngineError::Device(format!("{model} served {cap} but returned a {} response", got.shape()))
}

/// Say when the config pinned a model that admission then declined.
///
/// Pins are admitted BEFORE any on-demand model, so landing here means the pin does not fit even
/// walked first -- either alone against `memory_ceiling_mb`, or against an earlier pin in config
/// order. Until this existed the only trace was a `/v1/models` detail line nobody reads until
/// something is already wrong -- the config stated an intent and the runtime declined it in silence.
fn warn_declined_pins(rep: &ReconcileReport) {
    for n in &rep.pinned_deferred {
        eprintln!("[npu] WARNING: {n} is pinned (resident = true) but was not made resident: it \
                   does not fit under memory_ceiling_mb even admitted first. Raise \
                   memory_ceiling_mb, or move it earlier among the other pins.");
    }
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
        /// Published to `InFlight` for the call's duration, the same way `Cmd::Generate`'s
        /// `params.cancel` is -- most capabilities finish inside one dispatch and nothing ever reads
        /// it, but TTS's synthesis is a long AR loop with no other way to reach `npu cancel`/
        /// `npu model stop` while the actor thread is blocked inside it.
        cancel: npu_engine::Cancel,
        reply: Sender<Result<Served<Response>, EngineError>>,
        /// Stamped by the caller, read by the actor: the gap is the request's queue wait. See
        /// `Cmd::Generate`'s `enqueued`.
        enqueued: Instant,
    },
    /// Text generation, split from `Serve` because it does not answer with one `Response`: the
    /// result is a STREAM of chunks, produced on this thread and drained on the caller's. `ack`
    /// carries the routing/load outcome synchronously (so a caller can answer 503/400 before ever
    /// opening an SSE body); `tx` then carries the chunks as they are produced.
    Generate {
        model: Option<String>,
        prompt: Prompt,
        params: GenerateParams,
        tx: SyncSender<StreamItem>,
        ack: Sender<Result<String, EngineError>>,
        /// Stamped by the caller, read by the actor: the gap is the request's queue wait. One
        /// thread owns the device, so a second request waits out the first one's whole generation
        /// -- a real cost, and until now an invisible one that showed up inside TTFT with no name.
        enqueued: Instant,
    },
    Reconcile { cfg: Box<Config>, reply: Sender<ReconcileReport> },
    /// Make a model resident because an operator asked. NEVER evicts: over `memory_ceiling_mb` this
    /// fails and names what holds the budget. The request path (`Cmd::Serve`) still evicts, because
    /// a request asks for a capability while this asks for capacity.
    Load { name: String, reply: Sender<Result<LoadReport, EngineError>> },
    /// Give a model's device memory back now, keeping its config entry so routing still knows what
    /// it is and the next request reloads it. The same call the idle sweep makes, fired by hand.
    Unload { name: String, reply: Sender<Result<bool, EngineError>> },
    /// Host-only, no device: bake a model's declarative weight spec into a checkpoint. Goes through
    /// the actor (not run inline in the HTTP handler) so it serializes against every other command
    /// touching this model's config entry, same as `Load`/`Unload`.
    Bake { name: String, force: bool, reply: Sender<Result<Option<std::path::PathBuf>, EngineError>> },
    Status { reply: Sender<Vec<ModelStatus>> },
    Shutdown,
}

/// The in-flight generation's cancel handle, published so a thread that is NOT the actor can stop
/// the work the actor is inside.
///
/// This is the whole of stage 2a, and it is deliberately not a scheduler. An operator whose device
/// is held by a generation cannot be helped by ASKING the actor -- the actor is precisely what is
/// busy -- so the actor publishes the one thing that ends the work, and anybody can pull it. The
/// generation then stops within one dispatch and the queued command is serviced normally.
///
/// What this is NOT: preemption of the actor's LOOP. Commands still queue; they just stop queueing
/// behind work nobody wants. Interleaving two generations needs per-sequence KV (`kv_off` addresses
/// the cache by global position) and is a different project.
#[derive(Clone, Default)]
pub struct InFlight(Arc<Mutex<Option<(String, npu_engine::Cancel)>>>);

impl InFlight {
    fn set(&self, model: &str, cancel: npu_engine::Cancel) {
        *self.0.lock().unwrap() = Some((model.to_string(), cancel));
    }

    fn clear(&self) {
        *self.0.lock().unwrap() = None;
    }

    /// Cancel the running generation only if it belongs to `model`.
    ///
    /// Stopping one model must never abort a generation belonging to another: the device is
    /// single-flight, so a request for a DIFFERENT model is simply queued behind that generation
    /// and waits for it in the ordinary way.
    fn cancel_if(&self, model: &str) -> bool {
        let held = self.0.lock().unwrap().clone();
        match held {
            Some((n, c)) if n == model => {
                c.cancel(npu_engine::CancelReason::Operator);
                true
            }
            _ => false,
        }
    }

    /// Cancel whatever is running, naming it. `None` when nothing is.
    fn cancel(&self) -> Option<String> {
        let held = self.0.lock().unwrap().clone();
        held.map(|(name, c)| {
            c.cancel(npu_engine::CancelReason::Operator);
            name
        })
    }
}

#[derive(Clone)]
pub struct Handle { tx: Sender<Cmd>, live: LiveStatus, inflight: InFlight }

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
/// waste: at a tight ceiling it loads the first configured model that fits, and the request then
/// evicts it to load the one it actually wanted -- two full device loads to serve one request.
/// Worse, `npu embed` against an ASR-only config paid a complete parakeet load before it could
/// report that no embed model was configured at all. Declaring is enough to route correctly.
pub fn start_lazy(cfg: Config, loader: Box<dyn ModelLoader + Send>) -> Result<(Handle, JoinHandle<()>), EngineError> {
    spawn(cfg, loader, false)
}

fn spawn(cfg: Config, loader: Box<dyn ModelLoader + Send>, eager: bool) -> Result<(Handle, JoinHandle<()>), EngineError> {
    let (tx, rx) = channel::<Cmd>();
    let (ready_tx, ready_rx) = channel::<Result<(), String>>();
    // Off the request path on purpose: it is a subprocess, and the first request must not pay for
    // it. See `conditions::spawn_probe`.
    crate::conditions::spawn_probe();
    // Fixed once at spawn, not read fresh per publish: it is the denominator `npu top` divides
    // cumulative busy time by, and needs to name when THIS process started serving.
    let started_unix = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let live = LiveStatus::new(cfg.server.port, started_unix);
    let live_actor = live.clone();
    let inflight = InFlight::default();
    let inflight_actor = inflight.clone();
    let join = std::thread::spawn(move || {
        let live = live_actor;
        let inflight = inflight_actor;
        let mut reg = Registry::default();
        let mut cfg = cfg;
        // A panic anywhere below used to kill this thread, after which every request failed with
        // "actor stopped" and the real cause was gone. Model constructors still `.expect()` on
        // missing artifacts, so a bad config or a moved weights dir landed here. Catch it: the
        // actor survives, and the panic message is sent back to `spawn` instead of being dropped on
        // the floor (`let _ = ready_rx.recv()` used to discard it, handing the caller a `Handle` to
        // an actor whose initial reconcile silently never ran).
        let init: Result<(), String> = if eager {
            // Declared, not live: nothing is loaded yet at this point, so the only number available
            // is the loader's pre-load estimate for each pinned model.
            if let Some(w) = cfg.pin_overcommit(|m| loader.declared_footprint(m).unwrap_or(0)) {
                eprintln!("[npu] WARNING: {w}");
            }
            guard(|| reconcile(&cfg, &mut reg, loader.as_ref())).map(|report| warn_declined_pins(&report))
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
        // Before the first command or sweep, so a reader at boot sees what reconcile made resident
        // rather than an empty list it cannot distinguish from a server with no models.
        live.set(reg.status_at(Instant::now()));
        loop {
            match rx.recv_timeout(next_sweep.saturating_duration_since(Instant::now())) {
                Ok(Cmd::Serve { cap, model, req, cancel, reply, enqueued }) => {
                    last_request = Instant::now(); released = false;
                    let queue_us = enqueued.elapsed().as_micros() as u64;
                    // Two guarded steps rather than one, so the model NAME is known when the second
                    // fails. The shipped failure is a PANIC inside the dispatch (a missing insts
                    // file panics in npu-asr), which unwinds past any Result handling inside the
                    // call -- so condemning the model has to happen out here, after catch_unwind.
                    live.set_doing(reg.status_at(Instant::now()),
                        Some(format!("loading for {cap}{}", named(model.as_deref()))));
                    let t_load = Instant::now();
                    let ready = guard(|| serve_ready(&cfg, &mut reg, loader.as_ref(), cap, model.as_deref()))
                        .unwrap_or_else(|msg| Err(EngineError::Device(msg)));
                    let load_us = t_load.elapsed().as_micros() as u64;
                    let r = match ready {
                        Err(e) => Err(e),
                        Ok(name) => {
                            // Publish BUSY before the work, not after: this loop does not come back
                            // round until the request finishes, so the end-of-iteration publish can
                            // never observe a model that is serving.
                            live.set_doing(reg.status_serving(Instant::now(), Some(&name)), Some(format!("serving {name}")));
                            // Published BEFORE the work and cleared after it, panic included -- see
                            // the identical pattern around `run_generate` below. Harmless for the
                            // capabilities that finish in one dispatch: nothing ever reads it before
                            // `clear()` removes it again. `run_named_evicting` re-publishes it on
                            // every retry attempt.
                            let t_serve = Instant::now();
                            let (name, out) = run_named_evicting(&cfg, &mut reg, loader.as_ref(), cap,
                                model.as_deref(), name, req, cancel, &inflight);
                            inflight.clear();
                            // Charged whether it succeeded or failed: a request that held the
                            // device and then errored still held it, and occupancy that only
                            // counted successes would understate exactly the runs worth noticing.
                            reg.charge(&name, t_serve.elapsed().as_micros() as u64);
                            match out {
                                Ok(value) => Ok(Served { model: name, value, queue_us, load_us }),
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
                    // Publish BEFORE replying. The end-of-iteration publish happens AFTER
                    // this send, so a caller that mutates and then reads status could see
                    // state older than the command it just completed -- `npu model start`
                    // returning, then `/v1/models` not showing it. The reply must not
                    // outrun the snapshot it changed.
                    live.set(reg.status_at(Instant::now()));
                    let _ = reply.send(r);
                }
                Ok(Cmd::Generate { model, prompt, params, tx, ack, enqueued }) => {
                    last_request = Instant::now(); released = false;
                    let queue_us = enqueued.elapsed().as_micros() as u64;
                    // Snapshot residency BEFORE resolving, so a cold first token can be told from a
                    // warm one afterwards. Reading it back from the elapsed time would be an
                    // inference wearing a measurement's clothes.
                    let resident_before: Vec<String> = reg.entries.iter()
                        .filter(|e| e.model.is_some()).map(|e| e.cfg.name.clone()).collect();
                    let t_load = Instant::now();
                    live.set_doing(reg.status_at(Instant::now()),
                        Some(format!("loading for generate{}", named(model.as_deref()))));
                    let ready = guard(|| serve_ready(&cfg, &mut reg, loader.as_ref(),
                            Capability::GENERATE, model.as_deref()))
                        .unwrap_or_else(|msg| Err(EngineError::Device(msg)));
                    let load_us = t_load.elapsed().as_micros() as u64;
                    match ready {
                        Err(e) => { let _ = ack.send(Err(e)); }
                        Ok(name) => {
                            // The ack reaches the caller before any chunk does, which is what lets
                            // `Handle::generate` answer routing errors before an SSE body ever opens.
                            if ack.send(Ok(name.clone())).is_ok() {
                                let was_resident = resident_before.contains(&name);
                                let conditions = npu_engine::RunConditions {
                                    engine_version: env!("CARGO_PKG_VERSION").to_string(),
                                    model: name.clone(),
                                    power_mode: crate::conditions::power_mode(),
                                    resident: Some(was_resident),
                                    kernel: crate::conditions::kernel_release(),
                                    started_unix: std::time::SystemTime::now()
                                        .duration_since(std::time::UNIX_EPOCH)
                                        .map(|d| d.as_secs() as i64).unwrap_or(0),
                                };
                                let power_start_uw = crate::conditions::npu_power_uw();
                                let mut sink = |c: Chunk<'_>| -> bool {
                                    let item = match c {
                                        Chunk::Text(t) => StreamItem::Text(t.to_string()),
                                        Chunk::Step(r) => StreamItem::Step(r.clone()),
                                        Chunk::ToolCall(c) => StreamItem::ToolCall(c.clone()),
                                        Chunk::Progress { prefilled, total } =>
                                            StreamItem::Progress { prefilled, total },
                                        Chunk::Done { reason, usage, report } => {
                                            // The generator measured the generation; only this
                                            // thread saw the queue, the load and the machine.
                                            let mut report = report.clone();
                                            report.conditions = conditions.clone();
                                            report.queue_us = queue_us;
                                            report.load_us = load_us;
                                            report.npu_power_start_uw = power_start_uw;
                                            report.npu_power_end_uw = crate::conditions::npu_power_uw();
                                            StreamItem::Done { reason, usage, report: Box::new(report) }
                                        }
                                    };
                                    // `Err` here means the receiver (the socket thread) is gone --
                                    // the client hung up. Returning `false` is the sink's documented
                                    // abort signal.
                                    tx.send(item).is_ok()
                                };
                                live.set_doing(reg.status_serving(Instant::now(), Some(&name)), Some(format!("serving {name}")));
                                // Published BEFORE the work and cleared after it, panic included --
                                // `guard` catches the unwind, so a lost clear would leave a stale
                                // handle that cancels the NEXT generation instead of this one.
                                inflight.set(&name, params.cancel.clone());
                                let t_serve = Instant::now();
                                let out = guard(|| run_generate(&mut reg, &name, &prompt, &params, &mut sink))
                                    .unwrap_or_else(|msg| Err(EngineError::Device(msg)));
                                inflight.clear();
                                reg.charge(&name, t_serve.elapsed().as_micros() as u64);
                                if let Err(e) = out {
                                    if condemns_model(&e) {
                                        reg.mark_failed(&name, &e.to_string());
                                        eprintln!("[npu-runtime] {name} FAILED generating: {e}");
                                    }
                                    let _ = tx.send(StreamItem::Error(e));
                                }
                            }
                        }
                    }
                }
                Ok(Cmd::Reconcile { cfg: newcfg, reply }) => {
                    last_request = Instant::now(); released = false;
                    cfg = *newcfg;
                    live.set_doing(reg.status_at(Instant::now()), Some("reconciling".into()));
                    let rep = guard(|| reconcile(&cfg, &mut reg, loader.as_ref()))
                        .unwrap_or_else(|msg| ReconcileReport { failed: vec![msg], ..Default::default() });
                    warn_declined_pins(&rep);
                    // Publish BEFORE replying. The end-of-iteration publish happens AFTER
                    // this send, so a caller that mutates and then reads status could see
                    // state older than the command it just completed -- `npu model start`
                    // returning, then `/v1/models` not showing it. The reply must not
                    // outrun the snapshot it changed.
                    live.set(reg.status_at(Instant::now()));
                    let _ = reply.send(rep);
                }
                Ok(Cmd::Load { name, reply }) => {
                    // Counts as activity: an operator warming the device must not have it swept out
                    // from under them by an idle window that started before they asked.
                    last_request = Instant::now(); released = false;
                    // A cold 15 GB load is the longest thing this loop does -- measured ~12 s on
                    // gemma4-12b -- so anyone waiting on it is entitled to be told that is what
                    // they are waiting for.
                    live.set_doing(reg.status_at(Instant::now()), Some(format!("loading {name}")));
                    let now = Instant::now();
                    let r = match cfg.find(&name).cloned() {
                        None => Err(EngineError::Load(format!(
                            "unknown model {name:?} (not in the config)"))),
                        Some(m) => {
                            let already = reg.get_loaded(&name).is_some();
                            guard(|| reg.load_explicit(&m, loader.as_ref(), &cfg.server, now))
                                .unwrap_or_else(|msg| Err(EngineError::Load(msg)))
                                .map(|()| {
                                    // Stamp it, or the model an operator just loaded is the LRU
                                    // victim of the very next request that needs a slot.
                                    reg.touch(&name, now);
                                    LoadReport {
                                        loaded: !already,
                                        resident_mb: reg.resident_bytes() / (1024 * 1024),
                                        ceiling_mb: cfg.server.memory_ceiling_mb,
                                        unweighed: reg.unweighed_residents(),
                                    }
                                })
                        }
                    };
                    // Publish BEFORE replying. The end-of-iteration publish happens AFTER
                    // this send, so a caller that mutates and then reads status could see
                    // state older than the command it just completed -- `npu model start`
                    // returning, then `/v1/models` not showing it. The reply must not
                    // outrun the snapshot it changed.
                    live.set(reg.status_at(Instant::now()));
                    let _ = reply.send(r);
                }
                Ok(Cmd::Bake { name, force, reply }) => {
                    live.set_doing(reg.status_at(Instant::now()), Some(format!("baking {name}")));
                    let r = match cfg.find(&name).cloned() {
                        None => Err(EngineError::Load(format!(
                            "unknown model {name:?} (not in the config)"))),
                        Some(m) => guard(|| loader.bake(&m, force))
                            .unwrap_or_else(|msg| Err(EngineError::Load(msg))),
                    };
                    let _ = reply.send(r);
                }
                Ok(Cmd::Unload { name, reply }) => {
                    live.set_doing(reg.status_at(Instant::now()), Some(format!("unloading {name}")));
                    let r = match cfg.find(&name) {
                        None => Err(EngineError::Load(format!(
                            "unknown model {name:?} (not in the config)"))),
                        Some(_) => {
                            let was = reg.get_loaded(&name).is_some();
                            // Behind `guard`, like the idle sweep's release -- dropping a model runs
                            // native teardown (XRT, onnxruntime) and a panic there must not take the
                            // actor with it. NOTE this does NOT make teardown safe: the observed
                            // failure is a SIGSEGV, which catch_unwind cannot see. See task
                            // `model-teardown-segfaults-the-service`.
                            match was {
                                false => Ok(false),
                                true => match guard(|| reg.release(&name, "unloaded: asked for", UnloadReason::Operator)) {
                                    Ok(()) => {
                                        // An unload frees a working set, so the deep release has
                                        // something new to trim -- as after the idle sweep.
                                        released = false;
                                        Ok(true)
                                    }
                                    Err(m) => Err(EngineError::Device(format!("unload {name}: {m}"))),
                                },
                            }
                        }
                    };
                    // Publish BEFORE replying. The end-of-iteration publish happens AFTER
                    // this send, so a caller that mutates and then reads status could see
                    // state older than the command it just completed -- `npu model start`
                    // returning, then `/v1/models` not showing it. The reply must not
                    // outrun the snapshot it changed.
                    live.set(reg.status_at(Instant::now()));
                    let _ = reply.send(r);
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
            // Publish after every command and every sweep -- the two things that can change what is
            // resident. This thread is the only owner of the registry, so the snapshot is written
            // from the same place the state lives and cannot disagree with it. Clears BUSY
            // implicitly: `status_at` never sets it, so returning to the top of the loop is exactly
            // the moment nothing is being served.
            live.set(reg.status_at(Instant::now()));
        }
    });
    match ready_rx.recv() {
        Ok(Ok(())) => Ok((Handle { tx, live, inflight }, join)),
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

fn run_named(reg: &mut Registry, name: &str, req: Request, cancel: npu_engine::Cancel)
    -> Result<Response, EngineError> {
    let m = reg.get_loaded_mut(name).ok_or_else(|| EngineError::Load(format!("{name} not loaded")))?;
    m.run_cancellable(req, cancel)
}

/// `run_named`, retrying once per hardware-context exhaustion by evicting the LRU OTHER resident
/// model and reloading `name`.
///
/// Covers the case `Registry::ensure_resident`'s own hwctx retry cannot see: a model like parakeet
/// LOADS successfully and opens some kernels lazily on its first REQUEST, so the exhaustion only
/// shows up here, inside `run`. Stops -- returning the failing attempt -- at the first
/// non-exhaustion error, at `EvictPolicy::None`, or once no evictable victim remains; the caller
/// condemns `name` on that final error exactly as it always has.
fn run_named_evicting(cfg: &Config, reg: &mut Registry, loader: &dyn ModelLoader, cap: Capability,
                      want: Option<&str>, mut name: String, req: Request, cancel: npu_engine::Cancel,
                      inflight: &InFlight) -> (String, Result<Response, EngineError>) {
    let victim = |reg: &Registry, name: &str| -> Option<String> {
        if cfg.server.evict_policy == EvictPolicy::None { return None; }
        reg.lru_victim_except(name)
    };
    // `run` consumes the request, so a retry needs a copy taken up front: one clone per request
    // whenever another model could be evicted, none when this is the only resident model.
    let mut spare = victim(reg, &name).is_some().then(|| req.clone());
    let mut req = Some(req);
    loop {
        let this_req = req.take().expect("a request is queued for every loop iteration");
        inflight.set(&name, cancel.clone());
        let out = guard(|| run_named(reg, &name, this_req, cancel.clone()))
            .unwrap_or_else(|msg| Err(EngineError::Device(msg)));
        let Err(e) = &out else { return (name, out) };
        if !(condemns_model(e) && is_hwctx_exhaustion(&e.to_string())) { return (name, out); }
        let (Some(v), Some(next_req)) = (victim(reg, &name), spare.take()) else { return (name, out); };
        reg.mark_failed(&name, &e.to_string());
        eprintln!("[npu-runtime] {name} FAILED serving {cap}: {e}");
        reg.release(&v, &format!("evicted for {name}: hardware contexts"), UnloadReason::Evicted);
        match guard(|| serve_ready(cfg, reg, loader, cap, want))
            .unwrap_or_else(|msg| Err(EngineError::Device(msg))) {
            Err(e2) => return (name, Err(e2)),
            Ok(new_name) => {
                name = new_name;
                spare = victim(reg, &name).is_some().then(|| next_req.clone());
                req = Some(next_req);
            }
        }
    }
}

fn run_generate(reg: &mut Registry, name: &str, prompt: &Prompt, params: &GenerateParams,
                sink: &mut dyn FnMut(Chunk<'_>) -> bool) -> Result<(), EngineError> {
    let m = reg.get_loaded_mut(name).ok_or_else(|| EngineError::Load(format!("{name} not loaded")))?;
    m.generate_stream(prompt, params, sink)
}

/// Whether a failure condemns the MODEL or just this request.
///
/// A device or load error is a property of the model -- a missing instruction stream fails
/// identically for every caller, forever. WrongKind/Unsupported/NoModel are the caller's problem and
/// must never condemn a working model.
fn condemns_model(e: &EngineError) -> bool {
    matches!(e, EngineError::Device(_) | EngineError::Load(_))
}


/// How long a caller waits for the actor before calling it busy.
///
/// Generous on purpose: a cold load of a 15 GB model behind a running generation is legitimate work
/// and turning it into an error would be worse than the hang. The point is a BOUND, not a tight one.
/// `" (model-name)"`, or empty when the request named none and routing will pick the default.
fn named(model: Option<&str>) -> String {
    model.map(|m| format!(" ({m})")).unwrap_or_default()
}

fn actor_timeout() -> Duration {
    std::env::var("NPU_ACTOR_TIMEOUT_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .map(Duration::from_millis)
        .unwrap_or(Duration::from_secs(120))
}

impl Handle {
    /// Wait for the actor's reply, BOUNDED.
    ///
    /// Every one of these was a bare `recv()`. One long command then made every other caller look
    /// broken in the same way, and none could say why: `npu model stop`, a request naming a model
    /// that does not exist, and `/v1/models` all simply never returned. Measured 2026-09-14 during
    /// a ~60-minute prefill.
    fn await_reply<T>(&self, rx: std::sync::mpsc::Receiver<T>) -> Result<T, EngineError> {
        self.await_reply_within(rx, actor_timeout())
    }

    /// The bound, taken as an argument so a test can choose it. `NPU_ACTOR_TIMEOUT_MS` is
    /// process-global and these tests run in parallel threads of one process, so a test that set it
    /// would be setting it for every other test at the same time.
    fn await_reply_within<T>(
        &self,
        rx: std::sync::mpsc::Receiver<T>,
        within: Duration,
    ) -> Result<T, EngineError> {
        match rx.recv_timeout(within) {
            Ok(v) => Ok(v),
            Err(RecvTimeoutError::Disconnected) => {
                Err(EngineError::Device("actor dropped reply".into()))
            }
            Err(RecvTimeoutError::Timeout) => Err(EngineError::Busy(self.busy_note())),
        }
    }


    /// What is holding the actor -- readable precisely because the snapshot does NOT go through it.
    fn busy_note(&self) -> String {
        let snap = self.live.get();
        if let Some(d) = snap.doing {
            // A generation can be stopped; a load cannot, so only the first gets the suggestion.
            let fix = match d.starts_with("serving") {
                true => "; stop it with `npu cancel`",
                false => "",
            };
            return format!(
                "the device actor is {d} (snapshot {}s old){fix}",
                snap.at.elapsed().as_secs()
            );
        }
        match snap.models.iter().find(|m| m.busy).map(|m| m.name.clone()) {
            Some(n) => format!(
                "the device is serving {n} (snapshot {}s old); retry, or stop it with `npu cancel`",
                snap.at.elapsed().as_secs()
            ),
            None => format!(
                "the device actor did not answer in {}s and reports nothing serving",
                actor_timeout().as_secs()
            ),
        }
    }

    /// The actor's last published status, without asking the actor. See [`Handle::await_reply`] for
    /// why asking is not an option on a status path.
    pub fn snapshot(&self) -> crate::control_socket::Snapshot {
        self.live.get()
    }

    /// Stop the running generation, naming the model it was serving. `None` when none is running.
    ///
    /// Does NOT go through the actor, for the same reason status does not: the actor is what is
    /// busy. The generation ends within one dispatch and whatever was queued behind it is then
    /// serviced in the ordinary way.
    pub fn cancel_current(&self) -> Option<String> {
        self.inflight.cancel()
    }

    /// Stop the running generation if it is `model`'s. See [`InFlight::cancel_if`].
    pub fn cancel_model(&self, model: &str) -> bool {
        self.inflight.cancel_if(model)
    }
}

impl Handle {
    /// Serve any capability. The typed helpers below are conveniences over this; a caller with a
    /// capability that has no helper (tts, generate) uses it directly.
    pub fn serve(&self, cap: Capability, model: Option<&str>, req: Request)
        -> Result<Served<Response>, EngineError> {
        let (r, rx) = channel();
        self.tx.send(Cmd::Serve { cap, model: model.map(String::from), req,
                                  cancel: npu_engine::Cancel::new(), reply: r, enqueued: Instant::now() })
            .map_err(|_| EngineError::Device("actor stopped".into()))?;
        self.await_reply(rx)?
    }
    /// Text generation. Unlike `serve`, this does not wait for a `Response`: it returns as soon as
    /// routing/loading is decided, handing back a receiver the caller drains at its own pace (an SSE
    /// socket loop, or a buffered accumulator). The bounded channel is what keeps a slow drain from
    /// letting the generator run unbounded ahead of it.
    pub fn generate(&self, model: Option<&str>, prompt: Prompt, params: GenerateParams)
        -> Result<Served<std::sync::mpsc::Receiver<StreamItem>>, EngineError> {
        let (tx, rx) = sync_channel(GENERATE_CHANNEL_CAP);
        let (ack_tx, ack_rx) = channel();
        self.tx.send(Cmd::Generate { model: model.map(String::from), prompt, params, tx,
                                     ack: ack_tx, enqueued: Instant::now() })
            .map_err(|_| EngineError::Device("actor stopped".into()))?;
        let name = self.await_reply(ack_rx)??;
        Ok(Served { model: name, value: rx, queue_us: 0, load_us: 0 })
    }
    pub fn transcribe(&self, model: Option<&str>, pcm: Vec<i16>, sr: u32) -> Result<Served<String>, EngineError> {
        let s = self.serve(Capability::ASR, model, Request::Audio { pcm, sample_rate: sr })?;
        match s.value {
            Response::Text(t) => Ok(Served { model: s.model, value: t, queue_us: s.queue_us, load_us: s.load_us }),
            other => Err(wrong_shape(Capability::ASR, &s.model, &other)),
        }
    }
    pub fn embed(&self, model: Option<&str>, text: &str) -> Result<Served<Vec<f32>>, EngineError> {
        let s = self.serve(Capability::EMBED, model, Request::Text(text.to_string()))?;
        match s.value {
            Response::Vector(v) => Ok(Served { model: s.model, value: v, queue_us: s.queue_us, load_us: s.load_us }),
            other => Err(wrong_shape(Capability::EMBED, &s.model, &other)),
        }
    }
    pub fn diarize(&self, model: Option<&str>, pcm: Vec<i16>, sr: u32)
        -> Result<Served<Vec<Segment>>, EngineError> {
        let s = self.serve(Capability::DIARIZE, model, Request::Audio { pcm, sample_rate: sr })?;
        match s.value {
            Response::Segments(v) => Ok(Served { model: s.model, value: v, queue_us: s.queue_us, load_us: s.load_us }),
            other => Err(wrong_shape(Capability::DIARIZE, &s.model, &other)),
        }
    }
    pub fn reconcile(&self, cfg: Config) -> Result<ReconcileReport, EngineError> {
        let (r, rx) = channel();
        self.tx.send(Cmd::Reconcile { cfg: Box::new(cfg), reply: r })
            .map_err(|_| EngineError::Device("actor stopped".into()))?;
        self.await_reply(rx)
    }
    /// Make a model resident now. `Err` over `memory_ceiling_mb` -- this never evicts; see
    /// `Registry::load_explicit` for why the request path and this one differ.
    pub fn load(&self, name: &str) -> Result<LoadReport, EngineError> {
        let (r, rx) = channel();
        self.tx.send(Cmd::Load { name: name.to_string(), reply: r })
            .map_err(|_| EngineError::Device("actor stopped".into()))?;
        self.await_reply(rx)?
    }
    /// Release a model's device memory. `Ok(false)` when it was not resident to begin with.
    pub fn unload(&self, name: &str) -> Result<bool, EngineError> {
        let (r, rx) = channel();
        self.tx.send(Cmd::Unload { name: name.to_string(), reply: r })
            .map_err(|_| EngineError::Device("actor stopped".into()))?;
        self.await_reply(rx)?
    }
    /// Bake a model's declarative weight spec into a checkpoint, host-only. `Ok(None)` means the
    /// scenario has no such spec (legacy `weights =` npy path).
    pub fn bake(&self, name: &str, force: bool) -> Result<Option<std::path::PathBuf>, EngineError> {
        let (r, rx) = channel();
        self.tx.send(Cmd::Bake { name: name.to_string(), force, reply: r })
            .map_err(|_| EngineError::Device("actor stopped".into()))?;
        self.await_reply(rx)?
    }
    pub fn status(&self) -> Vec<ModelStatus> {
        let (r, rx) = channel();
        if self.tx.send(Cmd::Status { reply: r }).is_err() { return vec![]; }
        self.await_reply(rx).unwrap_or_default()
    }
    /// The out-of-band snapshot the control socket answers `GET /v1/models` from -- an `Arc` clone,
    /// cheap, and readable without ever touching the actor's channel.
    pub fn live_status(&self) -> LiveStatus { self.live.clone() }
    pub fn shutdown(&self) { let _ = self.tx.send(Cmd::Shutdown); }
}

// These live in the crate (not tests/actor.rs, which is behind the `testkit` feature) so the plain
// `cargo test --workspace` in scripts/ci_gate.sh actually runs them.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Defaults, ModelCfg, ServerCfg};
    use crate::loader::mock::MockLoader;
    use crate::loader::{Servable, StreamServable};
    use crate::registry::LoadState;
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::time::Duration;

    const MB: u64 = 1024 * 1024;
    /// asr + embed configured, but only ONE MB of budget by default: serving both means swapping.
    /// `.unwrap()`: a mock loader's initial reconcile does not panic, so a start() failure here is a
    /// real regression.
    /// The ordering invariant behind moving status off the actor: a mutating command must not
    /// RETURN before the published snapshot reflects it.
    ///
    /// Broken for one commit. The reply was sent inside the command arm and the publish happened at
    /// the end of the loop iteration, so `npu model start` could return while `/v1/models` still
    /// showed the model unloaded. Two route tests caught it as an intermittent failure, which is
    /// exactly how a race presents and exactly why it is worth a test that names the rule.
    #[test]
    fn a_mutating_command_does_not_return_before_the_snapshot_reflects_it() {
        let (h, j) = swap_setup(ServerCfg {
            memory_ceiling_mb: 64,
            idle_unload_s: 0,
            ..Default::default()
        });
        h.load("asr").expect("load asr");
        let snap = h.snapshot();
        let asr = snap.models.iter().find(|m| m.name == "asr").expect("asr missing from snapshot");
        assert_eq!(asr.state, LoadState::Loaded, "load returned before the snapshot saw it");

        h.unload("asr").expect("unload asr");
        let snap = h.snapshot();
        let asr = snap.models.iter().find(|m| m.name == "asr").expect("asr missing from snapshot");
        assert_eq!(asr.state, LoadState::Unloaded, "unload returned before the snapshot saw it");
        h.shutdown();
        j.join().unwrap();
    }

    fn swap_setup(srv: ServerCfg) -> (Handle, JoinHandle<()>) {
        let mut t = BTreeMap::new();
        t.insert("asr".to_string(), Ok((Capability::ASR, MB)));
        t.insert("bge".to_string(), Ok((Capability::EMBED, MB)));
        let cfg = Config {
            server: srv,
            defaults: Defaults::from_pairs([
                (Capability::ASR, "asr".to_string()), (Capability::EMBED, "bge".to_string())]),
            models: vec![
                ModelCfg { name: "asr".into(), scenario: "x".into(), resident: false },
                ModelCfg { name: "bge".into(), scenario: "y".into(), resident: false },
            ],
        };
        start(cfg, Box::new(MockLoader { table: t })).unwrap()
    }
    fn state_of(h: &Handle, name: &str) -> LoadState {
        h.status().into_iter().find(|s| s.name == name).expect("entry").state
    }

    #[test]
    fn one_slot_serves_both_models_by_swapping() {
        // idle_unload off: this test is about the byte ceiling as an evict trigger, nothing else.
        let (h, j) = swap_setup(ServerCfg { memory_ceiling_mb: 1, idle_unload_s: 0, ..Default::default() });
        // Neither is pinned, so boot loads neither -- both cold until a request wants one.
        assert_eq!(state_of(&h, "asr"), LoadState::Unloaded);
        assert_eq!(state_of(&h, "bge"), LoadState::Unloaded);
        // An embed request pulls bge in on demand...
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
        let (h, j) = swap_setup(ServerCfg { memory_ceiling_mb: 2, idle_unload_s: 0, ..Default::default() });
        assert_eq!(h.embed(Some("bge"), "hi").unwrap().model, "bge");
        // Naming an ASR model on the embed route is still a WrongKind error, not a silent swap.
        assert!(h.embed(Some("asr"), "hi").is_err());
        h.shutdown(); j.join().unwrap();
    }

    #[test]
    fn idle_sweep_releases_the_device_then_reloads_on_demand() {
        let (h, j) = swap_setup(ServerCfg {
            memory_ceiling_mb: 2, idle_unload_s: 1, sweep_interval_s: 1, ..Default::default()
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
            server: ServerCfg { memory_ceiling_mb: 1, idle_unload_s: 0, ..Default::default() },
            defaults: Defaults::from_pairs([
                (Capability::ASR, "asr".to_string()), (Capability::EMBED, "bge".to_string())]),
            models: vec![
                ModelCfg { name: "asr".into(), scenario: "x".into(), resident: false },
                ModelCfg { name: "bge".into(), scenario: "y".into(), resident: false },
            ],
        };
        let (h, j) = start_lazy(cfg, Box::new(MockLoader { table: t })).unwrap();
        // Nothing loaded, but both are known -- including their capability, read from the scenario.
        let s = h.status();
        assert!(s.iter().all(|x| x.state == LoadState::Unloaded), "lazy start must not load: {s:?}");
        assert_eq!(s.iter().find(|x| x.name == "bge").unwrap().capability, Some(Capability::EMBED));
        // The embed request loads bge and ONLY bge -- eagerly, this would have loaded asr first and
        // then evicted it to fit the tight ceiling.
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
            server: ServerCfg { memory_ceiling_mb: 1, idle_unload_s: 0, ..Default::default() },
            defaults: Defaults::from_pairs([(Capability::ASR, "asr".to_string())]),
            models: vec![ModelCfg { name: "asr".into(), scenario: "x".into(), resident: false }],
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
            memory_ceiling_mb: 2, idle_unload_s: 0, sweep_interval_s: 1, ..Default::default()
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
        fn load(&self, _cfg: &ModelCfg) -> Result<Box<dyn crate::loader::StreamServable>, EngineError> {
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
    impl crate::loader::StreamServable for PanicOnRun {}
    struct PanicOnRunLoader;
    impl ModelLoader for PanicOnRunLoader {
        fn load(&self, _cfg: &ModelCfg) -> Result<Box<dyn crate::loader::StreamServable>, EngineError> {
            Ok(Box::new(PanicOnRun))
        }
        fn declared_capability(&self, _cfg: &ModelCfg) -> Option<Capability> { Some(Capability::EMBED) }
    }

    #[test]
    fn a_model_that_panics_while_serving_is_marked_failed() {
        let cfg = Config {
            server: ServerCfg { memory_ceiling_mb: 1, idle_unload_s: 0, ..Default::default() },
            defaults: Defaults::from_pairs([(Capability::EMBED, "bge".to_string())]),
            // Pinned: this test needs it loaded before the dispatch that panics, and only a pin is
            // eagerly loaded by reconcile now.
            models: vec![ModelCfg { name: "bge".into(), scenario: "x".into(), resident: true }],
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
        let (h, j) = swap_setup(ServerCfg { memory_ceiling_mb: 2, idle_unload_s: 0, ..Default::default() });
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
            server: ServerCfg { memory_ceiling_mb: 1, idle_unload_s: 0, ..Default::default() },
            defaults: Defaults::from_pairs([(Capability::EMBED, "bge".to_string())]),
            models: vec![ModelCfg { name: "bge".into(), scenario: "x".into(), resident: false }],
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
            server: ServerCfg { memory_ceiling_mb: 1, idle_unload_s: 0, ..Default::default() },
            defaults: Defaults::from_pairs([(Capability::ASR, "asr".to_string())]),
            // Pinned: reconcile only ever ATTEMPTS a load for a pin, which is the whole premise of
            // this test ("during reconcile").
            models: vec![ModelCfg { name: "asr".into(), scenario: "x".into(), resident: true }],
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

    const CREATE_HWCTX_EINVAL: &str =
        "DRM_IOCTL_AMDXDNA_CREATE_HWCTX IOCTL failed (err=-22): Invalid argument";

    /// Parakeet's shape: the LOAD always succeeds, but the first `fails` calls to `run` fail with
    /// the driver's context-exhaustion text, standing in for kernels opened lazily on first
    /// dispatch. The budget lives on the LOADER (shared via `Arc`, decremented on every `run`
    /// regardless of which model instance), not on a model instance -- `mark_failed` drops the
    /// instance and a retry loads a fresh one, so instance-local state would silently reset.
    struct HwctxOnRunModel { inner: Box<dyn StreamServable>, fails: Arc<AtomicU32> }
    impl Servable for HwctxOnRunModel {
        fn capabilities(&self) -> Capability { self.inner.capabilities() }
        fn footprint(&self) -> u64 { self.inner.footprint() }
        fn run(&mut self, req: Request) -> Result<Response, EngineError> {
            let left = self.fails.load(Ordering::SeqCst);
            if left > 0 {
                self.fails.store(left - 1, Ordering::SeqCst);
                return Err(EngineError::Device(CREATE_HWCTX_EINVAL.to_string()));
            }
            self.inner.run(req)
        }
    }
    impl StreamServable for HwctxOnRunModel {}
    struct HwctxOnRunLoader { inner: MockLoader, target: String, fails: Arc<AtomicU32> }
    impl ModelLoader for HwctxOnRunLoader {
        fn load(&self, cfg: &ModelCfg) -> Result<Box<dyn StreamServable>, EngineError> {
            let m = self.inner.load(cfg)?;
            if cfg.name == self.target {
                Ok(Box::new(HwctxOnRunModel { inner: m, fails: self.fails.clone() }))
            } else {
                Ok(m)
            }
        }
        fn declared_capability(&self, cfg: &ModelCfg) -> Option<Capability> { self.inner.declared_capability(cfg) }
        fn declared_footprint(&self, cfg: &ModelCfg) -> Option<u64> { self.inner.declared_footprint(cfg) }
    }

    #[test]
    fn a_one_shot_request_exhausting_hwctx_evicts_the_lru_other_model_and_retries() {
        let mut t = BTreeMap::new();
        t.insert("target".to_string(), Ok((Capability::ASR, MB)));
        t.insert("o1".to_string(), Ok((Capability::EMBED, MB)));
        t.insert("o2".to_string(), Ok((Capability::EMBED, MB)));
        let l = HwctxOnRunLoader { inner: MockLoader { table: t }, target: "target".into(),
                                   fails: Arc::new(AtomicU32::new(1)) };
        let cfg = Config {
            server: ServerCfg { memory_ceiling_mb: 100, idle_unload_s: 0, ..Default::default() },
            defaults: Defaults::from_pairs([(Capability::ASR, "target".to_string())]),
            models: vec![
                ModelCfg { name: "target".into(), scenario: "x".into(), resident: false },
                ModelCfg { name: "o1".into(), scenario: "x".into(), resident: false },
                ModelCfg { name: "o2".into(), scenario: "x".into(), resident: false },
            ],
        };
        let (h, j) = start_lazy(cfg, Box::new(l)).unwrap();
        // o1 before o2, so o1 is the colder of the two -- the one eviction must pick.
        h.load("o1").unwrap();
        h.load("o2").unwrap();
        let tr = h.transcribe(None, vec![0i16; 4], 16_000).unwrap();
        assert_eq!((tr.model.as_str(), tr.value.as_str()), ("target", "mock-text"),
            "the retried request must still land on the model that asked for it");
        assert_eq!(state_of(&h, "target"), LoadState::Loaded, "the reload after eviction must have succeeded");
        let o1 = h.status().into_iter().find(|s| s.name == "o1").unwrap();
        assert_eq!(o1.state, LoadState::Unloaded, "o1 is the LRU and must be the one evicted");
        assert!(o1.detail.contains("evicted for target: hardware contexts"), "{}", o1.detail);
        assert_eq!(state_of(&h, "o2"), LoadState::Loaded, "o2 is more recently used and must survive");
        h.shutdown(); j.join().unwrap();
    }

    #[test]
    fn hwctx_exhaustion_with_only_a_pinned_other_resident_fails_without_touching_the_pin() {
        let mut t = BTreeMap::new();
        t.insert("target".to_string(), Ok((Capability::ASR, MB)));
        t.insert("p".to_string(), Ok((Capability::EMBED, MB)));
        let l = HwctxOnRunLoader { inner: MockLoader { table: t }, target: "target".into(),
                                   fails: Arc::new(AtomicU32::new(1)) };
        let cfg = Config {
            server: ServerCfg { memory_ceiling_mb: 100, idle_unload_s: 0, ..Default::default() },
            defaults: Defaults::from_pairs([(Capability::ASR, "target".to_string())]),
            models: vec![
                ModelCfg { name: "target".into(), scenario: "x".into(), resident: false },
                ModelCfg { name: "p".into(), scenario: "x".into(), resident: true },
            ],
        };
        let (h, j) = start_lazy(cfg, Box::new(l)).unwrap();
        h.load("p").unwrap(); // pinned, so the load itself honours the pin
        let e = match h.transcribe(None, vec![0i16; 4], 16_000) {
            Err(e) => e.to_string(),
            Ok(s) => panic!("served {}", s.model),
        };
        assert!(e.contains("CREATE_HWCTX"), "{e}");
        assert_eq!(state_of(&h, "p"), LoadState::Loaded, "an honoured pin must never be the victim");
        assert_eq!(state_of(&h, "target"), LoadState::Failed, "with no victim, the failure condemns as before");
        h.shutdown(); j.join().unwrap();
    }

    /// A model whose `run` always fails with an ordinary device error -- not the driver's
    /// context-exhaustion text -- so the retry path must never engage.
    struct AlwaysDeviceErrorModel;
    impl Servable for AlwaysDeviceErrorModel {
        fn capabilities(&self) -> Capability { Capability::ASR }
        fn run(&mut self, _req: Request) -> Result<Response, EngineError> {
            Err(EngineError::Device("some other device failure".into()))
        }
    }
    impl StreamServable for AlwaysDeviceErrorModel {}
    struct AlwaysDeviceErrorLoader { target: String, inner: MockLoader }
    impl ModelLoader for AlwaysDeviceErrorLoader {
        fn load(&self, cfg: &ModelCfg) -> Result<Box<dyn StreamServable>, EngineError> {
            if cfg.name == self.target { Ok(Box::new(AlwaysDeviceErrorModel)) } else { self.inner.load(cfg) }
        }
        fn declared_capability(&self, cfg: &ModelCfg) -> Option<Capability> { self.inner.declared_capability(cfg) }
        fn declared_footprint(&self, cfg: &ModelCfg) -> Option<u64> { self.inner.declared_footprint(cfg) }
    }

    #[test]
    fn a_non_hwctx_device_error_does_not_evict_anything() {
        let mut t = BTreeMap::new();
        t.insert("o1".to_string(), Ok((Capability::EMBED, MB)));
        t.insert("o2".to_string(), Ok((Capability::EMBED, MB)));
        let l = AlwaysDeviceErrorLoader { target: "target".into(), inner: MockLoader { table: t } };
        let cfg = Config {
            server: ServerCfg { memory_ceiling_mb: 100, idle_unload_s: 0, ..Default::default() },
            defaults: Defaults::from_pairs([(Capability::ASR, "target".to_string())]),
            models: vec![
                ModelCfg { name: "target".into(), scenario: "x".into(), resident: false },
                ModelCfg { name: "o1".into(), scenario: "x".into(), resident: false },
                ModelCfg { name: "o2".into(), scenario: "x".into(), resident: false },
            ],
        };
        let (h, j) = start_lazy(cfg, Box::new(l)).unwrap();
        h.load("o1").unwrap();
        h.load("o2").unwrap();
        let e = match h.transcribe(None, vec![0i16; 4], 16_000) {
            Err(e) => e.to_string(),
            Ok(s) => panic!("served {}", s.model),
        };
        assert!(e.contains("some other device failure"), "{e}");
        assert_eq!(state_of(&h, "o1"), LoadState::Loaded, "a non-exhaustion failure must not evict anything");
        assert_eq!(state_of(&h, "o2"), LoadState::Loaded);
        assert_eq!(state_of(&h, "target"), LoadState::Failed);
        h.shutdown(); j.join().unwrap();
    }

    #[test]
    fn evict_policy_none_leaves_hwctx_exhaustion_unretried() {
        let mut t = BTreeMap::new();
        t.insert("target".to_string(), Ok((Capability::ASR, MB)));
        t.insert("o1".to_string(), Ok((Capability::EMBED, MB)));
        let l = HwctxOnRunLoader { inner: MockLoader { table: t }, target: "target".into(),
                                   fails: Arc::new(AtomicU32::new(1)) };
        let cfg = Config {
            server: ServerCfg { memory_ceiling_mb: 100, idle_unload_s: 0,
                                evict_policy: EvictPolicy::None, ..Default::default() },
            defaults: Defaults::from_pairs([(Capability::ASR, "target".to_string())]),
            models: vec![
                ModelCfg { name: "target".into(), scenario: "x".into(), resident: false },
                ModelCfg { name: "o1".into(), scenario: "x".into(), resident: false },
            ],
        };
        let (h, j) = start_lazy(cfg, Box::new(l)).unwrap();
        h.load("o1").unwrap();
        let e = match h.transcribe(None, vec![0i16; 4], 16_000) {
            Err(e) => e.to_string(),
            Ok(s) => panic!("served {}", s.model),
        };
        assert!(e.contains("CREATE_HWCTX"), "{e}");
        assert_eq!(state_of(&h, "o1"), LoadState::Loaded, "evict_policy = none must not evict o1 either");
        assert_eq!(state_of(&h, "target"), LoadState::Failed);
        h.shutdown(); j.join().unwrap();
    }

    /// Wraps a `MockLoader`, sleeping 30 ms inside `load()` -- stands in for a real model's load
    /// time so a one-shot request's `load_us` has something non-trivial to report.
    struct SleepyLoader { inner: MockLoader }
    impl ModelLoader for SleepyLoader {
        fn load(&self, cfg: &ModelCfg) -> Result<Box<dyn StreamServable>, EngineError> {
            std::thread::sleep(Duration::from_millis(30));
            self.inner.load(cfg)
        }
        fn declared_capability(&self, cfg: &ModelCfg) -> Option<Capability> { self.inner.declared_capability(cfg) }
        fn declared_footprint(&self, cfg: &ModelCfg) -> Option<u64> { self.inner.declared_footprint(cfg) }
    }

    #[test]
    fn a_one_shot_reports_its_load_and_queue() {
        let mut t = BTreeMap::new();
        t.insert("bge".to_string(), Ok((Capability::EMBED, MB)));
        let cfg = Config {
            server: ServerCfg { memory_ceiling_mb: 1, idle_unload_s: 0, ..Default::default() },
            defaults: Defaults::from_pairs([(Capability::EMBED, "bge".to_string())]),
            models: vec![ModelCfg { name: "bge".into(), scenario: "x".into(), resident: false }],
        };
        let (h, j) = start_lazy(cfg, Box::new(SleepyLoader { inner: MockLoader { table: t } })).unwrap();
        let s = h.serve(Capability::EMBED, None, Request::Text("x".into())).unwrap();
        assert!(s.load_us >= 30_000, "load_us {} missed the 30 ms load", s.load_us);
        let s = h.serve(Capability::EMBED, None, Request::Text("x".into())).unwrap();
        assert!(s.load_us < 30_000, "a resident model must not report the load again: {}", s.load_us);
        assert!(s.queue_us < 1_000_000);
        h.shutdown(); j.join().unwrap();
    }
}

#[cfg(test)]
mod bound_tests {
    use super::*;
    use crate::registry::LoadState;

    fn busy_model(name: &str, busy: bool) -> ModelStatus {
        ModelStatus {
            name: name.into(),
            state: LoadState::Loaded,
            detail: String::new(),
            capability: Capability::from_name("generate"),
            bo_bytes: 0,
            idle_s: None,
            served: 0,
            busy_us: 0,
            busy,
            pinned: false,
            pin_honored: false,
        }
    }

    fn handle_with(models: Vec<ModelStatus>) -> (Handle, Sender<Cmd>) {
        let (tx, _never_read) = channel();
        let live = LiveStatus::new(11434, 0);
        live.set(models);
        // `_never_read` is returned so the channel is not Disconnected -- an actor that is BUSY and
        // an actor that is GONE are different answers, and this exercises the first.
        (Handle { tx: tx.clone(), live, inflight: InFlight::default() }, tx)
    }

    /// The bound. A reply that never comes has to become an ANSWER: every one of these was a bare
    /// `recv()`, so one long command left every other caller waiting with nothing to look at.
    #[test]
    fn a_reply_that_never_comes_becomes_busy_naming_what_holds_the_device() {
        let (h, _keep) = handle_with(vec![busy_model("gemma4-12b", true)]);
        let (_reply_tx, rx) = channel::<u8>();
        let e = h.await_reply_within(rx, Duration::from_millis(50)).unwrap_err();
        match e {
            EngineError::Busy(m) => {
                assert!(m.contains("gemma4-12b"), "the wait must name what holds the device: {m}")
            }
            other => panic!("expected Busy, got {other:?}"),
        }
    }

    /// A busy actor and a dead one need different answers: one is worth retrying and the other is
    /// not, and before the bound they were indistinguishable because neither returned.
    #[test]
    fn a_dropped_actor_is_a_device_error_not_a_busy_one() {
        let (h, _keep) = handle_with(vec![busy_model("m", true)]);
        let (reply_tx, rx) = channel::<u8>();
        drop(reply_tx);
        match h.await_reply_within(rx, Duration::from_secs(5)).unwrap_err() {
            EngineError::Device(_) => {}
            other => panic!("expected Device, got {other:?}"),
        }
    }

    /// Nothing serving is a different sentence from something serving, because the operator's next
    /// move differs: cancel a request, or look at why the actor stopped coming round its loop.
    #[test]
    fn an_idle_but_unresponsive_actor_says_so() {
        let (h, _keep) = handle_with(vec![busy_model("m", false)]);
        let (_reply_tx, rx) = channel::<u8>();
        match h.await_reply_within(rx, Duration::from_millis(50)).unwrap_err() {
            EngineError::Busy(m) => assert!(m.contains("nothing serving"), "{m}"),
            other => panic!("expected Busy, got {other:?}"),
        }
    }

    /// Stage 2a's primitive: the actor publishes the running generation's cancel handle, so a
    /// thread that is NOT the actor can end the work the actor is inside. Asking the actor cannot
    /// work here -- the actor is what is busy.
    #[test]
    fn cancelling_the_current_generation_names_it_and_sets_the_flag() {
        let (h, _keep) = handle_with(vec![]);
        let c = npu_engine::Cancel::new();
        h.inflight.set("gemma4-12b", c.clone());
        assert_eq!(h.cancel_current().as_deref(), Some("gemma4-12b"));
        assert_eq!(c.reason(), Some(npu_engine::CancelReason::Operator));
    }

    /// Racing the end of a generation and winning is not a failure.
    #[test]
    fn cancelling_with_nothing_running_is_not_an_error() {
        let (h, _keep) = handle_with(vec![]);
        assert_eq!(h.cancel_current(), None);
    }

    /// `model stop` is hard by default, but it must only stop ITS model: the device is
    /// single-flight, so a request for another model queues behind that generation rather than
    /// being entitled to kill it.
    #[test]
    fn cancelling_by_model_never_touches_another_models_generation() {
        let (h, _keep) = handle_with(vec![]);
        let c = npu_engine::Cancel::new();
        h.inflight.set("gemma4-12b", c.clone());
        assert!(!h.cancel_model("qwen3-0.6b"), "stopping one model claimed another's run");
        assert!(!c.is_cancelled());
        assert!(h.cancel_model("gemma4-12b"));
        assert_eq!(c.reason(), Some(npu_engine::CancelReason::Operator));
    }

    /// A stale handle would cancel the NEXT generation instead of the one the operator meant --
    /// silently, and only sometimes. The clear runs after `guard` catches a panic for this reason.
    #[test]
    fn a_cleared_handle_cannot_cancel_the_next_run() {
        let (h, _keep) = handle_with(vec![]);
        let first = npu_engine::Cancel::new();
        h.inflight.set("m", first.clone());
        h.inflight.clear();
        assert_eq!(h.cancel_current(), None);
        assert!(!first.is_cancelled(), "clearing must not cancel what it cleared");
    }

    /// The status path must not touch the channel at all -- that is the whole point. Readable even
    /// with no actor on the other end.
    #[test]
    fn a_snapshot_is_readable_with_no_actor_at_all() {
        let (h, keep) = handle_with(vec![busy_model("gemma4-12b", true)]);
        drop(keep);
        let snap = h.snapshot();
        assert_eq!(snap.models.len(), 1);
        assert_eq!(snap.models[0].name, "gemma4-12b");
    }
}
