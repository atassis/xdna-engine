//! Actual state: which models are loaded, their status, the memory accountant, and residency --
//! `last_used` per entry, which is what LRU eviction and idle unload both order themselves by.
use crate::config::{EvictPolicy, ModelCfg, ServerCfg};
use crate::loader::{ModelLoader, StreamServable};
use npu_engine::EngineError;
use std::time::{Duration, Instant};

pub use npu_engine::capability::Capability;

// glibc's malloc_trim(3). Declared directly rather than pulling in the libc crate: no crate in this
// workspace depends on it, and this is one symbol with a stable signature.
#[cfg(target_env = "gnu")]
extern "C" {
    fn malloc_trim(pad: usize) -> i32;
}

/// Hand memory that an unload freed back to the OS -- the second, deeper level of idleness.
///
/// Unloading a model frees its allocations, but the allocator keeps the pages: measured on this
/// engine, a process holding NO model still sat on ~2.5 GB of anonymous private-dirty pages spread
/// over ~200 glibc arena mappings, none of it referenced by anything. Only `malloc_trim` walks those
/// arenas and returns what is free.
///
/// Milliseconds, and only ever called from the actor's idle sweep -- never on a request path.
/// Returns whether a trim was actually available on this target.
pub fn release_free_memory() -> bool {
    #[cfg(target_env = "gnu")]
    {
        // SAFETY: malloc_trim takes no pointer of ours and only releases pages the allocator already
        // considers free; live allocations are untouched.
        unsafe { malloc_trim(0) };
        true
    }
    #[cfg(not(target_env = "gnu"))]
    {
        false
    }
}

/// Whether the deep release is due: idle for `idle_release`, and not already done since the last
/// request or unload. Pure, so the policy is testable without a clock or an allocator.
pub fn deep_release_due(
    idle_release: Option<Duration>,
    since_activity: Duration,
    already_released: bool,
) -> bool {
    match idle_release {
        Some(window) => !already_released && since_activity >= window,
        None => false,
    }
}

/// Status detail for a model the memory accountant could not weigh. Named rather than inlined
/// because the test and the reader must agree on it, and because it should disappear the day
/// `footprint()` returns real bytes.
pub const UNWEIGHED: &str = "memory_ceiling not applied: footprint unmeasured";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadState { Loaded, Failed, Unloaded }

#[derive(Debug, Clone, PartialEq)]
pub struct ModelStatus {
    pub name: String,
    pub state: LoadState,
    pub detail: String,
    pub capability: Option<Capability>,
    pub bo_bytes: u64,
    /// Seconds since this model last served a request, for resident models only. `None` when the
    /// model is not resident (nothing is holding the device on its behalf).
    pub idle_s: Option<u64>,
    /// Requests this model has served since the process started, and the wall time it held the
    /// device for. Cumulative, not a rate: a rate needs a window, and every consumer wants a
    /// different one -- `npu top` divides by process uptime, a future scrape would difference two
    /// samples. Publishing the raw counters lets both be correct from the same field.
    ///
    /// The device is single-tenant and the actor is single-flight, so these sum to real occupancy
    /// rather than to something that can exceed the wall clock.
    pub served: u64,
    pub busy_us: u64,
    /// True while this model is the one the device actor is currently inside a request for.
    ///
    /// At most one model can be busy: the actor is single-flight, which is precisely why this is
    /// worth reporting -- it is the difference between "the engine is slow" and "something else has
    /// the device and you are queued behind it". It is published from around the serve rather than
    /// from the end of the actor loop, because the loop does not come back round until the request
    /// it is serving has finished, so a status written there could never observe a busy model.
    pub busy: bool,
    /// `resident = true` in the config: exempt from the idle sweep and never an eviction victim.
    ///
    /// Reported because a pin was otherwise invisible from outside -- `/v1/models` showed a pinned
    /// and an unpinned model identically, so the only way to know whether the intent had taken was
    /// to read the config file and trust that the running service agreed with it.
    pub pinned: bool,
    /// Whether the pin is CURRENTLY protecting this model, distinct from `pinned` (which only says
    /// the config declares it). A pin that would push `sum(pinned bytes)` over `memory_ceiling_mb`
    /// is refused rather than silently granted -- see `Entry::pin_honored` -- so `pinned &&
    /// !pin_honored` is a real, reportable state: "declared pinned, currently not protected, here's
    /// why" (`detail` carries the reason). Meaningless when `pinned` is false.
    pub pin_honored: bool,
}

pub struct Entry {
    pub cfg: ModelCfg,
    pub model: Option<Box<dyn StreamServable>>,
    pub status: ModelStatus,
    /// Last time this entry served a request; set to the load time when it becomes resident. Stale
    /// but harmless while the entry is not resident -- both readers filter on residency first.
    pub last_used: Instant,
    /// `Some(ceiling_mb)` when this entry was deferred because the byte budget was spent AT THAT
    /// MOMENT. The reason is a snapshot and the condition lapses: the idle sweep frees bytes without
    /// revisiting anyone's stored detail, so a boot-time "over budget" was still being reported after
    /// every resident model had gone idle. `status_at` re-reads it against live bytes instead of
    /// trusting it.
    pub deferred_capacity: Option<u64>,
    /// Whether this pin is currently honoured -- see `ModelStatus::pin_honored`, which mirrors this.
    /// Read by `lru_victim`/`sweep_idle` in place of `cfg.resident` directly, so an over-budget pin
    /// becomes evictable like any unpinned model instead of jamming eviction with a guarantee the
    /// invariant has already refused to keep. Recomputed every `reconcile` pass, never trusted stale.
    pub pin_honored: bool,
}

#[derive(Default)]
pub struct Registry {
    pub entries: Vec<Entry>,
}

impl Registry {
    pub fn get_loaded(&self, name: &str) -> Option<&(dyn StreamServable + 'static)> {
        self.entries.iter().find(|e| e.cfg.name == name).and_then(|e| e.model.as_deref())
    }
    /// `Servable::run` takes `&mut self` (every shipped model is mutable in truth -- some launder it
    /// through a `RefCell`), so serving needs this and not `get_loaded`.
    pub fn get_loaded_mut(&mut self, name: &str) -> Option<&mut (dyn StreamServable + 'static)> {
        self.entries.iter_mut().find(|e| e.cfg.name == name).and_then(|e| e.model.as_deref_mut())
    }
    pub fn resident_bytes(&self) -> u64 {
        self.entries.iter().filter_map(|e| e.model.as_ref().map(|m| m.footprint())).sum()
    }
    pub fn resident_count(&self) -> usize {
        self.entries.iter().filter(|e| e.model.is_some()).count()
    }
    pub fn status(&self) -> Vec<ModelStatus> { self.status_at(Instant::now()) }
    /// Charge a completed request to a model: one more served, and the device time it held.
    ///
    /// Called by the actor after the work, not before, so an in-flight request is `busy` but not
    /// yet counted -- a request that is still running has no duration to charge.
    pub fn charge(&mut self, name: &str, busy_us: u64) {
        if let Some(e) = self.entries.iter_mut().find(|e| e.cfg.name == name) {
            e.status.served += 1;
            e.status.busy_us += busy_us;
        }
    }

    /// `status_at`, with `serving` (if any) marked busy. The actor is single-flight, so at most one
    /// name is ever passed here.
    pub fn status_serving(&self, now: Instant, serving: Option<&str>) -> Vec<ModelStatus> {
        let mut v = self.status_at(now);
        if let Some(name) = serving {
            if let Some(s) = v.iter_mut().find(|s| s.name == name) { s.busy = true; }
        }
        v
    }
    /// `status()` with the clock passed in, so idle reporting is testable without sleeping.
    pub fn status_at(&self, now: Instant) -> Vec<ModelStatus> {
        let live = self.resident_bytes();
        self.entries.iter().map(|e| {
            let mut s = e.status.clone();
            s.idle_s = e.model.as_ref().map(|_| now.saturating_duration_since(e.last_used).as_secs());
            // Read off `cfg`/`pin_honored`, like the sweep and the evictor do, so the reported pin
            // cannot drift from the one those two act on.
            s.pinned = e.cfg.resident;
            s.pin_honored = e.pin_honored;
            // A capacity deferral is a snapshot. Report it only while it is still true, or the
            // config reads as refused long after the sweep freed every byte.
            if let Some(ceiling) = e.deferred_capacity {
                if e.model.is_none() && live < ceiling {
                    s.detail = format!("not resident: {} MB of {} MB in use; loads on demand",
                        live / (1024 * 1024), ceiling / (1024 * 1024));
                }
            }
            s
        }).collect()
    }
    /// The capability of an entry: from the live model, else the kind remembered from its last
    /// successful load. Survives an unload, which is what lets routing stay sweep-invariant.
    pub fn known_capability(&self, name: &str) -> Option<Capability> {
        let e = self.entries.iter().find(|e| e.cfg.name == name)?;
        e.model.as_ref().map(|m| m.capabilities()).or(e.status.capability)
    }
    /// Adopt a new `ModelCfg` for an entry whose loaded model is still the right one.
    ///
    /// `resident` is read off `e.cfg` by `lru_victim` and `sweep_idle`, and reconcile only replaced
    /// the entry when the SCENARIO changed -- so a pin or unpin was silently ignored for exactly as
    /// long as the model stayed loaded, which is the whole time it matters. Everything that would
    /// need a reload (the scenario) still goes through unload + load; this is for the fields that
    /// do not touch the device.
    pub fn update_cfg(&mut self, name: &str, cfg: &ModelCfg) {
        if let Some(e) = self.entries.iter_mut().find(|e| e.cfg.name == name) { e.cfg = cfg.clone(); }
    }

    /// Stamp a model as just used. Called once per served request.
    pub fn touch(&mut self, name: &str, now: Instant) {
        if let Some(e) = self.entries.iter_mut().find(|e| e.cfg.name == name) { e.last_used = now; }
    }

    /// Best-effort byte cost for a not-yet-loaded model: the loader's pre-load estimate (a stat of
    /// the weight artifact), or 0 if the loader cannot tell cheaply. For an ALREADY-loaded model,
    /// call `footprint()` on the live model instead -- this is specifically the "need a number
    /// before touching the device" case (admission, and a pin on a cold model).
    fn estimated_bytes(&self, cfg: &ModelCfg, loader: &dyn ModelLoader) -> u64 {
        loader.declared_footprint(cfg).unwrap_or(0)
    }

    /// Try to load one model under the byte budget, recording status. Never panics.
    ///
    /// This is the RECONCILE path, and it still refuses over `memory_ceiling_mb` -- booting a config
    /// with more models than fit must not thrash the device loading and evicting in a loop, and the
    /// models that do not fit are a capacity decision, not a failure (hence `Unloaded`, not
    /// `Failed`). The request path is `ensure_resident`, which evicts instead.
    pub fn try_load(&mut self, cfg: &ModelCfg, loader: &dyn ModelLoader, srv: &ServerCfg, now: Instant) {
        let ceiling = srv.memory_ceiling_mb * 1024 * 1024;
        let estimate = self.estimated_bytes(cfg, loader);
        if self.resident_bytes() + estimate > ceiling {
            // Record what the model IS even though it is not loaded. Without this a deferred model
            // reports capability None, which `/v1/models` renders as kind "unknown". The lazy path
            // already asks the loader here (`actor::run`); the eager reconcile did not, so the two
            // disagreed about the same model.
            let cap = self.declared(cfg, loader);
            self.set_deferred(cfg, format!(
                "not resident: over memory_ceiling_mb ({} MB); loads on demand",
                srv.memory_ceiling_mb), cap, ceiling);
            return;
        }
        // The loader can PANIC, not just Err: model constructors still `.expect()` on missing
        // artifacts. Catch it here, where a load failure is already the expected outcome, so it is
        // recorded as Failed instead of unwinding through whichever caller happened to trigger it.
        match crate::actor::guard(|| loader.load(cfg)).unwrap_or_else(|msg| Err(EngineError::Load(msg))) {
            Ok(m) => {
                let bo = m.footprint();
                // Real bytes can exceed the pre-load ESTIMATE even when the estimate fit -- the
                // estimate is a host file size, not device BO bytes. This is the safety net that
                // catches an underestimate; the check above is the fast path that avoids loading
                // something the estimate already ruled out.
                if self.resident_bytes() + bo > ceiling {
                    self.set_failed(cfg, "over memory_ceiling".into(), None);
                    return;
                }
                let status = ModelStatus {
                    name: cfg.name.clone(), state: LoadState::Loaded,
                    // Say when the accountant could not weigh this model. A model kind nobody has
                    // wired footprint() for reports 0, which would let the check above admit an
                    // unbounded number of them while reading as if the ceiling still applied.
                    detail: if bo == 0 { UNWEIGHED.into() } else { String::new() },
                    capability: Some(m.capabilities()), bo_bytes: bo, idle_s: Some(0),
                    served: 0,
                    busy_us: 0,
                    busy: false,
                    pinned: cfg.resident,
                    pin_honored: cfg.resident,
                };
                self.upsert(Entry { cfg: cfg.clone(), model: Some(m), status, last_used: now,
                                    deferred_capacity: None, pin_honored: cfg.resident });
            }
            Err(e) => { let cap = self.declared(cfg, loader); self.set_failed(cfg, e.to_string(), cap) }
        }
    }

    /// Make `name` resident, loading it on demand and evicting to make room if the byte budget is
    /// full. This is the hot-swap path: here the ceiling is the EVICT TRIGGER, not a refusal.
    ///
    /// Only ever called between commands (the actor is the single device owner and is not serving
    /// anything else while this runs), so an eviction can never pull a model out from under a
    /// request in flight.
    pub fn ensure_resident(&mut self, cfg: &ModelCfg, loader: &dyn ModelLoader, srv: &ServerCfg,
                           now: Instant) -> Result<(), EngineError> {
        if self.get_loaded(&cfg.name).is_some() { return Ok(()); }
        let ceiling = srv.memory_ceiling_mb * 1024 * 1024;
        let estimate = self.estimated_bytes(cfg, loader);
        while self.resident_bytes() + estimate > ceiling {
            if srv.evict_policy == EvictPolicy::None {
                return Err(EngineError::Unsupported(format!(
                    "{} is not resident and evict_policy = \"none\" at memory_ceiling_mb ({} MB)",
                    cfg.name, srv.memory_ceiling_mb)));
            }
            // Nothing resident left to evict: stop, and let try_load record the capacity refusal
            // rather than loop forever. `pin_honored` entries are excluded from `lru_victim`, so an
            // honoured pin is never sacrificed to make room for something else.
            match self.lru_victim() {
                Some(v) => self.release(&v, &format!("evicted for {}", cfg.name)),
                None => break,
            }
        }
        self.try_load(cfg, loader, srv, now);
        if self.get_loaded(&cfg.name).is_some() { return Ok(()); }
        let why = self.entries.iter().find(|e| e.cfg.name == cfg.name)
            .map(|e| e.status.detail.clone()).unwrap_or_else(|| "load failed".into());
        Err(EngineError::Load(format!("{}: {why}", cfg.name)))
    }

    /// Make `cfg.name` resident because an OPERATOR asked, or fail saying why.
    ///
    /// This never evicts, and that is the whole difference from `ensure_resident`. On the request
    /// path the byte ceiling is an evict trigger: a request names a capability, not a capacity, so
    /// swapping a model in to serve it is the right answer. An explicit `npu load` is the opposite
    /// -- it IS a statement about capacity -- and silently dropping a model someone else pinned or
    /// is about to use, in order to honour it, answers a question that was not asked. So it refuses,
    /// and the refusal names what is holding the budget, because "over budget" alone tells the
    /// operator nothing they can act on.
    ///
    /// Idempotent: loading an already-resident model succeeds without touching the device.
    pub fn load_explicit(&mut self, cfg: &ModelCfg, loader: &dyn ModelLoader, srv: &ServerCfg,
                         now: Instant) -> Result<(), EngineError> {
        if self.get_loaded(&cfg.name).is_some() { return Ok(()); }
        let ceiling = srv.memory_ceiling_mb * 1024 * 1024;
        let estimate = self.estimated_bytes(cfg, loader);
        if self.resident_bytes() + estimate > ceiling {
            let held: Vec<&str> = self.entries.iter().filter(|e| e.model.is_some())
                .map(|e| e.cfg.name.as_str()).collect();
            return Err(EngineError::Unsupported(format!(
                "{} cannot be made resident: {} MB of {} MB in use. Resident now: {}. \
                 Free one with `npu unload <model>`, or raise memory_ceiling_mb.",
                cfg.name, self.resident_bytes() / (1024 * 1024), srv.memory_ceiling_mb,
                held.join(" "))));
        }
        self.try_load(cfg, loader, srv, now);
        if self.get_loaded(&cfg.name).is_some() { return Ok(()); }
        let why = self.entries.iter().find(|e| e.cfg.name == cfg.name)
            .map(|e| e.status.detail.clone()).unwrap_or_else(|| "load failed".into());
        Err(EngineError::Load(format!("{}: {why}", cfg.name)))
    }

    /// Resident models whose footprint the accountant could not weigh.
    ///
    /// `Servable::footprint()` returns a hardcoded 0 for a model kind nobody has wired yet, which
    /// admits an unbounded number of that kind against `memory_ceiling_mb` without ever refusing. A
    /// caller that reports a successful load says so, rather than letting an operator who just set
    /// the ceiling believe it is now bounding something it cannot see.
    pub fn unweighed_residents(&self) -> Vec<String> {
        self.entries.iter()
            .filter(|e| e.model.as_ref().is_some_and(|m| m.footprint() == 0))
            .map(|e| e.cfg.name.clone()).collect()
    }

    /// Record a configured model WITHOUT loading it, keeping the capability its scenario declares so
    /// routing can pick it (and `/v1/models` can show it) before anything touches the device. A no-op
    /// on a model the registry already knows: a live entry always outranks a declaration.
    pub fn declare(&mut self, cfg: &ModelCfg, capability: Option<Capability>) {
        if self.entries.iter().any(|e| e.cfg.name == cfg.name) { return; }
        let detail = match capability {
            Some(_) => "declared: loads on demand".to_string(),
            // Nothing to route on: the request path will have to load it to find out what it is.
            None => "declared: capability unknown until loaded".to_string(),
        };
        self.entries.push(Entry {
            cfg: cfg.clone(),
            model: None,
            status: ModelStatus {
                name: cfg.name.clone(), state: LoadState::Unloaded, detail, capability, bo_bytes: 0,
                idle_s: None, served: 0, busy_us: 0, busy: false, pinned: cfg.resident,
                pin_honored: cfg.resident,
            },
            last_used: Instant::now(),
            deferred_capacity: None,
            pin_honored: cfg.resident,
        });
    }

    /// Least-recently-used RESIDENT model, if any. HONOURED pins are not candidates: a pin the
    /// invariant is currently protecting is not a pin if LRU can still evict it. An over-budget pin
    /// -- `pinned` true, `pin_honored` false -- IS a candidate: the invariant already refused to
    /// protect it, so ordinary LRU pressure is free to reclaim it like any unpinned model.
    pub fn lru_victim(&self) -> Option<String> {
        self.entries.iter().filter(|e| e.model.is_some() && !e.pin_honored)
            .min_by_key(|e| e.last_used).map(|e| e.cfg.name.clone())
    }

    /// Unload every resident, non-honoured-pin model idle for at least `idle`, returning what was
    /// released.
    ///
    /// The actor calls this from its `recv_timeout` idle branch, i.e. only between commands.
    pub fn sweep_idle(&mut self, now: Instant, idle: Duration) -> Vec<String> {
        let expired: Vec<String> = self.entries.iter()
            .filter(|e| e.model.is_some() && !e.pin_honored
                && now.saturating_duration_since(e.last_used) >= idle)
            .map(|e| e.cfg.name.clone()).collect();
        for n in &expired { self.release(n, &format!("unloaded: idle >= {}s", idle.as_secs())); }
        expired
    }

    /// Drop the model but KEEP the entry, so `/v1/models` can still show what happened to it (and
    /// so routing remembers its capability). Contrast `unload`, which forgets the entry entirely
    /// because the config no longer asks for it.
    pub fn release(&mut self, name: &str, reason: &str) {
        if let Some(e) = self.entries.iter_mut().find(|e| e.cfg.name == name) {
            e.model = None;
            e.status.state = LoadState::Unloaded;
            e.status.detail = reason.to_string();
            e.status.bo_bytes = 0;
            e.status.idle_s = None;
        }
    }
    pub fn unload(&mut self, name: &str) {
        self.entries.retain(|e| e.cfg.name != name);
    }

    /// Record that a RESIDENT model failed while serving, and drop it.
    ///
    /// Without this a model that loads and then dies on every dispatch -- the shipped failure mode
    /// when an instruction stream is missing -- stays `Loaded` forever while every request errors,
    /// which is precisely the "reports healthy while broken" defect. Dropping the model is also what
    /// makes recovery automatic: the next request re-runs `ensure_resident`, so a transient fault
    /// self-heals and only a persistent one keeps the entry `Failed`.
    pub fn mark_failed(&mut self, name: &str, detail: &str) {
        if let Some(e) = self.entries.iter_mut().find(|e| e.cfg.name == name) {
            e.model = None;
            e.status.state = LoadState::Failed;
            e.status.detail = detail.to_string();
            e.status.bo_bytes = 0;
            e.status.idle_s = None;
        }
    }

    /// Names of every configured model currently in `Failed`. This is the health signal: `Unloaded`
    /// is deliberate (deferred over the byte budget, or swept for being idle) and must never count.
    pub fn failed(&self) -> Vec<String> {
        self.entries.iter().filter(|e| e.status.state == LoadState::Failed)
            .map(|e| e.cfg.name.clone()).collect()
    }

    /// Recompute which pins the invariant currently protects, called once per `reconcile` pass with
    /// the set it has just decided fits. Everything else -- honoured pin, over-budget pin, or never
    /// pinned at all -- resolves to `false`, which is exactly what `lru_victim`/`sweep_idle` want:
    /// they only ever ask "is this NOT protected", never why.
    pub fn set_pin_honored(&mut self, honored: &std::collections::HashSet<String>) {
        for e in &mut self.entries {
            e.pin_honored = e.cfg.resident && honored.contains(&e.cfg.name);
            e.status.pin_honored = e.pin_honored;
        }
    }

    /// What the loader says this model is, without loading it. `None` when it cannot tell cheaply.
    ///
    /// Behind `guard` because `declared_capability` reads and parses a scenario file, and the real
    /// loader's callees still `.expect()` in places -- a malformed manifest must leave the model
    /// unlabelled, not unwind through a reconcile.
    pub(crate) fn declared(&self, cfg: &ModelCfg, loader: &dyn ModelLoader) -> Option<Capability> {
        crate::actor::guard(|| loader.declared_capability(cfg)).unwrap_or(None)
    }
    fn set_failed(&mut self, cfg: &ModelCfg, detail: String, declared: Option<Capability>) {
        self.set_state(cfg, LoadState::Failed, detail, declared);
    }
    /// Configured, wanted, and deliberately not resident -- distinct from a failure.
    fn set_deferred(&mut self, cfg: &ModelCfg, detail: String, declared: Option<Capability>,
                    ceiling_bytes: u64) {
        self.set_state_inner(cfg, LoadState::Unloaded, detail, declared, Some(ceiling_bytes));
    }
    fn set_state(&mut self, cfg: &ModelCfg, state: LoadState, detail: String,
                 declared: Option<Capability>) {
        self.set_state_inner(cfg, state, detail, declared, None);
    }
    fn set_state_inner(&mut self, cfg: &ModelCfg, state: LoadState, detail: String,
                       declared: Option<Capability>, ceiling_bytes: Option<u64>) {
        // Keep a capability learned from an earlier successful load: still true, and routing uses
        // it. Fall back to what the loader declares, so a model that has never loaded is still
        // labelled instead of reading as "unknown".
        let capability = self.known_capability(&cfg.name).or(declared);
        let status = ModelStatus {
            name: cfg.name.clone(), state, detail, capability, bo_bytes: 0, idle_s: None,
            served: 0,
            busy_us: 0,
            busy: false,
            pinned: cfg.resident,
            pin_honored: false,
        };
        self.upsert(Entry { cfg: cfg.clone(), model: None, status, last_used: Instant::now(),
                            deferred_capacity: ceiling_bytes, pin_honored: false });
    }
    fn upsert(&mut self, e: Entry) {
        if let Some(slot) = self.entries.iter_mut().find(|x| x.cfg.name == e.cfg.name) { *slot = e; }
        else { self.entries.push(e); }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loader::mock::MockLoader;
    use std::collections::BTreeMap;
    const MB: u64 = 1024 * 1024;
    fn cfg(name: &str) -> ModelCfg { ModelCfg { name: name.into(), scenario: "x".into(), resident: false } }
    fn pinned(name: &str) -> ModelCfg { ModelCfg { name: name.into(), scenario: "x".into(), resident: true } }
    fn ceiling_mb(mb: u64) -> ServerCfg { ServerCfg { memory_ceiling_mb: mb, ..Default::default() } }
    /// The point of the operator path: it REFUSES at capacity where the request path evicts.
    #[test]
    fn load_explicit_refuses_at_capacity_instead_of_evicting() {
        let mut t = BTreeMap::new();
        for n in ["a", "b"] { t.insert(n.to_string(), Ok((Capability::EMBED, MB))); }
        let l = MockLoader { table: t };
        let srv = ceiling_mb(1);
        let mut r = Registry::default();
        let now = Instant::now();

        r.load_explicit(&cfg("a"), &l, &srv, now).unwrap();
        let e = r.load_explicit(&cfg("b"), &l, &srv, now).unwrap_err().to_string();
        assert!(e.contains("1 MB of 1 MB in use"), "the refusal has to be quantified: {e}");
        assert!(e.contains("Resident now: a"), "and has to name what is in the way: {e}");
        assert!(r.get_loaded("a").is_some(), "a REFUSAL must not have evicted anything");
        assert!(r.get_loaded("b").is_none());

        // ...where the request path, asked the same question, swaps.
        r.ensure_resident(&cfg("b"), &l, &srv, now).unwrap();
        assert!(r.get_loaded("b").is_some(), "the request path still evicts");
        assert!(r.get_loaded("a").is_none());
    }

    #[test]
    fn load_explicit_is_idempotent_and_reports_a_real_load_failure() {
        let mut t = BTreeMap::new();
        t.insert("a".to_string(), Ok((Capability::EMBED, 1)));
        t.insert("broken".to_string(), Err("no instruction stream".to_string()));
        let l = MockLoader { table: t };
        let srv = ServerCfg::default();
        let mut r = Registry::default();
        let now = Instant::now();

        r.load_explicit(&cfg("a"), &l, &srv, now).unwrap();
        r.load_explicit(&cfg("a"), &l, &srv, now).expect("loading a resident model is a no-op, not an error");
        assert_eq!(r.resident_count(), 1);

        let e = r.load_explicit(&cfg("broken"), &l, &srv, now).unwrap_err().to_string();
        assert!(e.contains("no instruction stream"), "a load failure must surface its cause: {e}");
    }

    /// A pin must not be collateral damage: refusing to evict is what makes `npu load` safe to run
    /// against a server someone else is using.
    #[test]
    fn load_explicit_refuses_rather_than_touching_a_pinned_model() {
        let mut t = BTreeMap::new();
        for n in ["p", "b"] { t.insert(n.to_string(), Ok((Capability::EMBED, MB))); }
        let l = MockLoader { table: t };
        let srv = ceiling_mb(1);
        let mut r = Registry::default();
        let now = Instant::now();
        r.load_explicit(&pinned("p"), &l, &srv, now).unwrap();
        assert!(r.load_explicit(&cfg("b"), &l, &srv, now).is_err());
        assert!(r.get_loaded("p").is_some(), "the pinned model is untouched");
    }

    /// `memory_ceiling_mb` sums footprints; a model kind nobody has wired `footprint()` for reports
    /// 0 and is invisible to the sum. A caller has to be able to say so rather than reporting a load
    /// as if the limit had been checked.
    #[test]
    fn unweighed_residents_names_every_model_the_ceiling_cannot_bound() {
        let mut t = BTreeMap::new();
        // `unweighed` stands in for a model kind nobody has wired footprint() for -- reports 0 --
        // and `weighed` is what a measured footprint looks like. Both, so this pins a filter and not
        // a constant.
        t.insert("unweighed".to_string(), Ok((Capability::EMBED, 0)));
        t.insert("weighed".to_string(), Ok((Capability::EMBED, 4096)));
        let l = MockLoader { table: t };
        let srv = ServerCfg::default();
        let mut r = Registry::default();
        let now = Instant::now();
        assert!(r.unweighed_residents().is_empty(), "nothing resident, nothing to report");
        r.load_explicit(&cfg("weighed"), &l, &srv, now).unwrap();
        assert!(r.unweighed_residents().is_empty(), "a measured model is not reported");
        r.load_explicit(&cfg("unweighed"), &l, &srv, now).unwrap();
        assert_eq!(r.unweighed_residents(), vec!["unweighed".to_string()]);
    }

    /// The accountant is INERT, and the engine has to say so.
    ///
    /// Every shipped model returns `footprint() == 0`, so `resident_bytes() + 0 > ceiling` can
    /// never fire however small the ceiling. The bug is not the arithmetic, it is that a config
    /// advertising `memory_ceiling_mb = 4096` read as a bound while bounding nothing -- which is
    /// how a memory failure reached a service that claimed to prevent it.
    #[test]
    fn a_model_with_no_measured_footprint_loads_and_says_the_ceiling_did_not_apply() {
        let mut t = BTreeMap::new();
        t.insert("weightless".to_string(), Ok((Capability::EMBED, 0u64)));
        let l = MockLoader { table: t };
        let mut srv = ServerCfg::default();
        srv.memory_ceiling_mb = 0;          // the tightest ceiling expressible
        let mut r = Registry::default();
        r.try_load(&cfg("weightless"), &l, &srv, Instant::now());
        let s = r.status();
        let m = s.iter().find(|x| x.name == "weightless").expect("status entry");
        assert_eq!(m.state, LoadState::Loaded,
            "a 0-byte footprint cannot exceed even a 0 MB ceiling, so it must load: {}", m.detail);
        assert_eq!(m.detail, UNWEIGHED,
            "loading unweighed must be VISIBLE, not silent: {:?}", m.detail);
        assert_eq!(m.bo_bytes, 0);
    }

    /// ...and the arithmetic itself is correct, so the day footprints are measured the ceiling
    /// starts working with no other change. Without this the test above would equally pass over a
    /// check that had been deleted.
    ///
    /// Wraps `MockLoader` to report NO pre-load estimate, standing in for a model kind whose
    /// declared-footprint estimate undershoots or does not exist -- exactly the case the post-load
    /// safety net in `try_load` exists for, since a real load's own admission check would otherwise
    /// pre-empt this test's real target.
    struct UnderestimatingLoader(MockLoader);
    impl ModelLoader for UnderestimatingLoader {
        fn load(&self, cfg: &ModelCfg) -> Result<Box<dyn StreamServable>, EngineError> { self.0.load(cfg) }
        fn declared_capability(&self, cfg: &ModelCfg) -> Option<Capability> { self.0.declared_capability(cfg) }
    }
    #[test]
    fn a_model_that_reports_bytes_over_the_ceiling_is_refused() {
        let mut t = BTreeMap::new();
        t.insert("heavy".to_string(), Ok((Capability::EMBED, 3 * 1024 * 1024)));
        let l = UnderestimatingLoader(MockLoader { table: t });
        let mut srv = ServerCfg::default();
        srv.memory_ceiling_mb = 2;
        let mut r = Registry::default();
        r.try_load(&cfg("heavy"), &l, &srv, Instant::now());
        let s = r.status();
        let m = s.iter().find(|x| x.name == "heavy").expect("status entry");
        assert_eq!(m.state, LoadState::Failed, "3 MB must not fit under a 2 MB ceiling");
        assert!(m.detail.contains("memory_ceiling"), "the refusal must name the bound: {}", m.detail);
        assert_ne!(m.detail, UNWEIGHED, "a weighed model must not be reported as unweighed");
    }

    /// The ceiling is a SUM over resident models, not a per-model test: two models that each fit
    /// can still exceed it together. This is the case a per-model check would pass wrongly.
    #[test]
    fn the_ceiling_sums_across_resident_models() {
        let mut t = BTreeMap::new();
        t.insert("a".to_string(), Ok((Capability::EMBED, 3 * 1024 * 1024)));
        t.insert("b".to_string(), Ok((Capability::EMBED, 3 * 1024 * 1024)));
        let l = UnderestimatingLoader(MockLoader { table: t });
        let mut srv = ServerCfg::default();
        srv.memory_ceiling_mb = 4;          // each fits alone; together they do not
        let mut r = Registry::default();
        let now = Instant::now();
        r.try_load(&cfg("a"), &l, &srv, now);
        r.try_load(&cfg("b"), &l, &srv, now);
        let s = r.status();
        assert_eq!(s.iter().find(|x| x.name == "a").unwrap().state, LoadState::Loaded);
        let b = s.iter().find(|x| x.name == "b").unwrap();
        assert_eq!(b.state, LoadState::Failed, "3 + 3 MB must not fit under 4 MB: {}", b.detail);
    }

    /// A model that has never loaded must still report WHAT IT IS.
    ///
    /// At a tight ceiling every model past the first that does not fit is deferred by the eager
    /// reconcile, and a deferred entry used to carry `capability: None` -- which `/v1/models`
    /// renders as kind "unknown". The lazy path already asked the loader; only the eager path did
    /// not, so the same model was labelled or not depending on how the server started.
    #[test]
    fn a_deferred_model_reports_its_declared_kind_not_unknown() {
        let mut t = BTreeMap::new();
        t.insert("first".to_string(), Ok((Capability::ASR, 0u64)));
        t.insert("second".to_string(), Ok((Capability::DIARIZE, MB)));
        let l = MockLoader { table: t };
        let srv = ceiling_mb(0); // "first" is 0 bytes and fits; "second" is 1 MB and does not
        let mut r = Registry::default();
        let now = Instant::now();
        r.try_load(&cfg("first"), &l, &srv, now);
        r.try_load(&cfg("second"), &l, &srv, now);
        let s = r.status();
        let d = s.iter().find(|x| x.name == "second").expect("status entry");
        assert_eq!(d.state, LoadState::Unloaded, "second must be deferred: it does not fit");
        assert_eq!(d.capability, Some(Capability::DIARIZE),
            "a deferred model must carry its declared capability, not None: {:?}", d.capability);
    }

    /// A model whose LOAD failed is still a known kind: the scenario file said so, and failing to
    /// load does not unsay it. Routing and the model listing both read this field.
    #[test]
    fn a_failed_model_keeps_its_declared_kind() {
        let mut r = Registry::default();
        // A loader that cannot load this model but can still say what it is -- the real case, since
        // declared_capability reads the scenario TOML and the failure is in the artifacts.
        struct Declaring;
        impl ModelLoader for Declaring {
            fn load(&self, _c: &ModelCfg) -> Result<Box<dyn StreamServable>, EngineError> {
                Err(EngineError::Load("boom".into()))
            }
            fn declared_capability(&self, _c: &ModelCfg) -> Option<Capability> {
                Some(Capability::DIARIZE)
            }
        }
        r.try_load(&cfg("broken"), &Declaring, &ServerCfg::default(), Instant::now());
        let s = r.status();
        let f = s.iter().find(|x| x.name == "broken").expect("status entry");
        assert_eq!(f.state, LoadState::Failed);
        assert!(f.detail.contains("boom"), "{}", f.detail);
        assert_eq!(f.capability, Some(Capability::DIARIZE),
            "a failed load must not erase the declared kind: {:?}", f.capability);
    }

    /// Every mock model here costs one MB, so `ceiling_mb(n)` admits exactly `n` of them -- the same
    /// shape the old count-based `max_resident: n` tests relied on, now expressed in bytes.
    fn loader(names: &[&str]) -> MockLoader {
        let mut t = BTreeMap::new();
        for n in names { t.insert((*n).to_string(), Ok((Capability::EMBED, MB))); }
        MockLoader { table: t }
    }

    #[test]
    fn load_failure_is_recorded_not_fatal() {
        let mut t = BTreeMap::new();
        t.insert("good".to_string(), Ok((Capability::EMBED, 10)));
        t.insert("bad".to_string(), Err("boom".to_string()));
        let l = MockLoader { table: t };
        let srv = ServerCfg::default();
        let mut r = Registry::default();
        let now = Instant::now();
        r.try_load(&cfg("good"), &l, &srv, now);
        r.try_load(&cfg("bad"), &l, &srv, now);
        assert_eq!(r.resident_count(), 1);
        let s = r.status();
        assert!(s.iter().any(|x| x.name == "good" && x.state == LoadState::Loaded));
        assert!(s.iter().any(|x| x.name == "bad" && x.state == LoadState::Failed && x.detail.contains("boom")));
    }
    #[test]
    fn memory_ceiling_defers_the_overflow_instead_of_failing_it() {
        let l = loader(&["a", "b"]);
        let srv = ceiling_mb(1);
        let mut r = Registry::default();
        let now = Instant::now();
        r.try_load(&cfg("a"), &l, &srv, now);
        r.try_load(&cfg("b"), &l, &srv, now);
        assert_eq!(r.resident_count(), 1, "reconcile must not evict: booting would thrash");
        let b = r.status().into_iter().find(|x| x.name == "b").unwrap();
        assert_eq!(b.state, LoadState::Unloaded, "over capacity is a capacity decision, not a failure");
        assert!(b.detail.contains("memory_ceiling_mb"), "{}", b.detail);
    }
    /// The deferral reason is a snapshot, and it outlives the condition it describes.
    ///
    /// Observed on the running server 2026-09-08 under the old count-based cap: all seven models
    /// `unloaded`, five of them by the idle sweep, and the two pinned ones still reporting "not
    /// resident: at max_resident (5)" long after every slot was free -- the message was a fossil
    /// from boot, not a live verdict. Same failure shape is possible with bytes if the stored reason
    /// is trusted instead of re-derived, so this test carries over unchanged in spirit.
    #[test]
    fn a_deferral_reason_outlives_the_capacity_that_caused_it() {
        let l = loader(&["a", "b"]);
        let srv = ceiling_mb(1);
        let mut r = Registry::default();
        let t0 = Instant::now();
        r.try_load(&cfg("a"), &l, &srv, t0);
        r.try_load(&pinned("b"), &l, &srv, t0);   // deferred: "a" took the only MB
        assert_eq!(r.resident_count(), 1);

        // The idle sweep releases "a" (unpinned), so the byte it held is free again.
        let swept = r.sweep_idle(t0 + Duration::from_secs(1000), Duration::from_secs(900));
        assert_eq!(swept, vec!["a".to_string()]);
        assert_eq!(r.resident_count(), 0, "every byte is free once the sweep has run");

        let b = r.status().into_iter().find(|x| x.name == "b").unwrap();
        assert!(
            !b.detail.contains("memory_ceiling_mb"),
            "with {} bytes resident of a {} MB ceiling, the stored reason still claims capacity: {:?}",
            r.resident_bytes(), srv.memory_ceiling_mb, b.detail
        );
    }

    #[test]
    fn a_pinned_model_is_never_the_eviction_victim() {
        let l = loader(&["a", "b", "c"]);
        let srv = ceiling_mb(2);
        let mut r = Registry::default();
        let t0 = Instant::now();
        r.try_load(&pinned("a"), &l, &srv, t0);   // pinned AND the least recently used
        r.try_load(&cfg("b"), &l, &srv, t0);
        r.touch("b", t0 + Duration::from_secs(10));
        r.ensure_resident(&cfg("c"), &l, &srv, t0 + Duration::from_secs(20)).unwrap();
        assert!(r.get_loaded("a").is_some(), "the pin must outrank LRU order");
        assert!(r.get_loaded("c").is_some());
        assert_eq!(r.status().into_iter().find(|x| x.name == "b").unwrap().state, LoadState::Unloaded);
    }
    #[test]
    fn a_pinned_model_survives_the_idle_sweep_that_drops_its_neighbour() {
        let l = loader(&["a", "b"]);
        let srv = ceiling_mb(2);
        let mut r = Registry::default();
        let t0 = Instant::now();
        r.try_load(&pinned("a"), &l, &srv, t0);
        r.try_load(&cfg("b"), &l, &srv, t0);
        // Both are equally idle: only the pin distinguishes them, which is the point.
        let released = r.sweep_idle(t0 + Duration::from_secs(3600), Duration::from_secs(900));
        assert_eq!(released, vec!["b".to_string()]);
        assert!(r.get_loaded("a").is_some(), "a pinned model must not be swept for idleness");
    }
    #[test]
    fn lru_victim_is_none_when_every_resident_model_is_pinned() {
        let l = loader(&["a"]);
        let srv = ceiling_mb(1);
        let mut r = Registry::default();
        r.try_load(&pinned("a"), &l, &srv, Instant::now());
        assert_eq!(r.lru_victim(), None, "a pin-only registry offers no victim");
    }
    #[test]
    fn ensure_resident_evicts_the_lru_not_just_anyone() {
        let l = loader(&["a", "b", "c"]);
        let srv = ceiling_mb(2);
        let mut r = Registry::default();
        let t0 = Instant::now();
        r.try_load(&cfg("a"), &l, &srv, t0);
        r.try_load(&cfg("b"), &l, &srv, t0);
        // `a` is used after `b`, so `b` is the cold one.
        r.touch("a", t0 + Duration::from_secs(10));
        r.ensure_resident(&cfg("c"), &l, &srv, t0 + Duration::from_secs(20)).unwrap();
        assert_eq!(r.resident_count(), 2);
        assert!(r.get_loaded("a").is_some(), "the recently used model must survive");
        assert!(r.get_loaded("c").is_some());
        let b = r.status().into_iter().find(|x| x.name == "b").unwrap();
        assert_eq!(b.state, LoadState::Unloaded);
        assert!(b.detail.contains("evicted for c"), "{}", b.detail);
        assert_eq!(b.capability, Some(Capability::EMBED), "an evicted entry keeps its known capability");
    }
    #[test]
    fn ensure_resident_is_a_noop_when_already_loaded() {
        let l = loader(&["a"]);
        let srv = ceiling_mb(1);
        let mut r = Registry::default();
        let t0 = Instant::now();
        r.try_load(&cfg("a"), &l, &srv, t0);
        r.ensure_resident(&cfg("a"), &l, &srv, t0 + Duration::from_secs(5)).unwrap();
        assert_eq!(r.resident_count(), 1);
        assert_eq!(r.status().into_iter().find(|x| x.name == "a").unwrap().state, LoadState::Loaded);
    }
    #[test]
    fn evict_policy_none_refuses_and_keeps_the_incumbent() {
        let l = loader(&["a", "b"]);
        let srv = ServerCfg { memory_ceiling_mb: 1, evict_policy: EvictPolicy::None, ..Default::default() };
        let mut r = Registry::default();
        let t0 = Instant::now();
        r.try_load(&cfg("a"), &l, &srv, t0);
        let e = r.ensure_resident(&cfg("b"), &l, &srv, t0).unwrap_err();
        assert!(e.to_string().contains("evict_policy"), "{e}");
        assert!(r.get_loaded("a").is_some(), "a refusal must not disturb what is resident");
        assert!(r.get_loaded("b").is_none());
    }
    #[test]
    fn sweep_releases_only_what_is_actually_idle() {
        let l = loader(&["cold", "warm"]);
        let srv = ServerCfg::default();
        let mut r = Registry::default();
        let t0 = Instant::now();
        r.try_load(&cfg("cold"), &l, &srv, t0);
        r.try_load(&cfg("warm"), &l, &srv, t0);
        let now = t0 + Duration::from_secs(900);
        r.touch("warm", now - Duration::from_secs(10));
        let freed = r.sweep_idle(now, Duration::from_secs(900));
        assert_eq!(freed, vec!["cold".to_string()]);
        assert!(r.get_loaded("cold").is_none(), "expired model must release the device");
        assert!(r.get_loaded("warm").is_some(), "a model used 10s ago is not idle");
        let s = r.status_at(now);
        let cold = s.iter().find(|x| x.name == "cold").unwrap();
        assert_eq!(cold.state, LoadState::Unloaded);
        assert!(cold.detail.contains("idle"), "{}", cold.detail);
        assert_eq!(cold.idle_s, None, "a non-resident model reports no idle time");
        assert_eq!(s.iter().find(|x| x.name == "warm").unwrap().idle_s, Some(10));
    }
    #[test]
    fn released_model_reloads_on_demand() {
        let l = loader(&["a"]);
        let srv = ceiling_mb(1);
        let mut r = Registry::default();
        let t0 = Instant::now();
        r.try_load(&cfg("a"), &l, &srv, t0);
        assert_eq!(r.sweep_idle(t0 + Duration::from_secs(900), Duration::from_secs(900)), vec!["a"]);
        assert_eq!(r.known_capability("a"), Some(Capability::EMBED), "capability survives the unload");
        r.ensure_resident(&cfg("a"), &l, &srv, t0 + Duration::from_secs(901)).unwrap();
        assert!(r.get_loaded("a").is_some());
        assert_eq!(r.status().into_iter().find(|x| x.name == "a").unwrap().state, LoadState::Loaded);
    }
    /// The correction mechanism, at the registry level: `set_pin_honored` is how a pin the invariant
    /// has refused to protect loses its eviction immunity, without touching `cfg.resident` (the
    /// config's own declared intent stays exactly what it was).
    #[test]
    fn an_unhonored_pin_becomes_a_normal_eviction_candidate() {
        let l = loader(&["a"]);
        let srv = ceiling_mb(1);
        let mut r = Registry::default();
        r.try_load(&pinned("a"), &l, &srv, Instant::now());
        assert_eq!(r.lru_victim(), None, "freshly loaded and honoured, so not a candidate");
        assert!(r.status()[0].pinned && r.status()[0].pin_honored, "both true while it fits");

        r.set_pin_honored(&std::collections::HashSet::new()); // the invariant no longer protects it
        assert_eq!(r.lru_victim(), Some("a".to_string()),
            "an over-budget pin must become evictable like any unpinned model");
        assert!(r.status()[0].pinned, "the config's OWN declared intent is untouched");
        assert!(!r.status()[0].pin_honored, "but it is no longer what protects the model");
    }

    #[test]
    fn deep_release_fires_once_per_idle_stretch() {
        let w = Some(Duration::from_secs(1800));
        // not idle long enough yet
        assert!(!deep_release_due(w, Duration::from_secs(1799), false));
        // idle long enough, not yet released -> due
        assert!(deep_release_due(w, Duration::from_secs(1800), false));
        // ...but only once: the latch stops a trim on every sweep of a quiet night
        assert!(!deep_release_due(w, Duration::from_secs(9999), true));
        // switched off
        assert!(!deep_release_due(None, Duration::from_secs(9999), false));
    }
    #[test]
    fn release_free_memory_is_available_on_this_target() {
        // Not a memory assertion (RSS is not a unit-testable quantity) -- just that the platform has
        // the call, so a green test suite cannot hide a silently no-op second idle level.
        assert!(release_free_memory(), "glibc target should have malloc_trim");
    }
    #[test]
    fn ensure_resident_reports_why_a_load_failed() {
        let mut t = BTreeMap::new();
        t.insert("bad".to_string(), Err("no such xclbin".to_string()));
        let l = MockLoader { table: t };
        let srv = ServerCfg::default();
        let mut r = Registry::default();
        let e = r.ensure_resident(&cfg("bad"), &l, &srv, Instant::now()).unwrap_err();
        assert!(e.to_string().contains("no such xclbin"), "{e}");
    }
}
