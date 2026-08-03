//! Actual state: which models are loaded, their status, the memory accountant, and residency --
//! `last_used` per entry, which is what LRU eviction and idle unload both order themselves by.
use crate::config::{EvictPolicy, ModelCfg, ServerCfg};
use crate::loader::{ModelLoader, Servable};
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
}

pub struct Entry {
    pub cfg: ModelCfg,
    pub model: Option<Box<dyn Servable>>,
    pub status: ModelStatus,
    /// Last time this entry served a request; set to the load time when it becomes resident. Stale
    /// but harmless while the entry is not resident -- both readers filter on residency first.
    pub last_used: Instant,
}

#[derive(Default)]
pub struct Registry {
    pub entries: Vec<Entry>,
}

impl Registry {
    pub fn get_loaded(&self, name: &str) -> Option<&(dyn Servable + 'static)> {
        self.entries.iter().find(|e| e.cfg.name == name).and_then(|e| e.model.as_deref())
    }
    /// `Servable::run` takes `&mut self` (every shipped model is mutable in truth -- some launder it
    /// through a `RefCell`), so serving needs this and not `get_loaded`.
    pub fn get_loaded_mut(&mut self, name: &str) -> Option<&mut (dyn Servable + 'static)> {
        self.entries.iter_mut().find(|e| e.cfg.name == name).and_then(|e| e.model.as_deref_mut())
    }
    pub fn resident_bytes(&self) -> u64 {
        self.entries.iter().filter_map(|e| e.model.as_ref().map(|m| m.footprint())).sum()
    }
    pub fn resident_count(&self) -> usize {
        self.entries.iter().filter(|e| e.model.is_some()).count()
    }
    pub fn status(&self) -> Vec<ModelStatus> { self.status_at(Instant::now()) }
    /// `status()` with the clock passed in, so idle reporting is testable without sleeping.
    pub fn status_at(&self, now: Instant) -> Vec<ModelStatus> {
        self.entries.iter().map(|e| {
            let mut s = e.status.clone();
            s.idle_s = e.model.as_ref().map(|_| now.saturating_duration_since(e.last_used).as_secs());
            s
        }).collect()
    }
    /// The capability of an entry: from the live model, else the kind remembered from its last
    /// successful load. Survives an unload, which is what lets routing stay sweep-invariant.
    pub fn known_capability(&self, name: &str) -> Option<Capability> {
        let e = self.entries.iter().find(|e| e.cfg.name == name)?;
        e.model.as_ref().map(|m| m.capabilities()).or(e.status.capability)
    }
    /// Stamp a model as just used. Called once per served request.
    pub fn touch(&mut self, name: &str, now: Instant) {
        if let Some(e) = self.entries.iter_mut().find(|e| e.cfg.name == name) { e.last_used = now; }
    }

    /// Try to load one model under the budget, recording status. Never panics.
    ///
    /// This is the RECONCILE path, and it still refuses over `max_resident` -- booting a config with
    /// more models than slots must not thrash the device loading and evicting in a loop, and the
    /// models that do not fit are a capacity decision, not a failure (hence `Unloaded`, not
    /// `Failed`). The request path is `ensure_resident`, which evicts instead.
    pub fn try_load(&mut self, cfg: &ModelCfg, loader: &dyn ModelLoader, srv: &ServerCfg, now: Instant) {
        if self.resident_count() >= srv.max_resident {
            self.set_deferred(cfg, format!(
                "not resident: at max_resident ({}); loads on demand", srv.max_resident));
            return;
        }
        match loader.load(cfg) {
            Ok(m) => {
                let bo = m.footprint();
                if self.resident_bytes() + bo > srv.memory_ceiling_mb * 1024 * 1024 {
                    self.set_failed(cfg, "over memory_ceiling".into());
                    return;
                }
                let status = ModelStatus {
                    name: cfg.name.clone(), state: LoadState::Loaded, detail: String::new(),
                    capability: Some(m.capabilities()), bo_bytes: bo, idle_s: Some(0),
                };
                self.upsert(Entry { cfg: cfg.clone(), model: Some(m), status, last_used: now });
            }
            Err(e) => self.set_failed(cfg, e.to_string()),
        }
    }

    /// Make `name` resident, loading it on demand and freeing a slot first if `max_resident` is
    /// full. This is the hot-swap path: here `max_resident` is the EVICT TRIGGER, not a refusal.
    ///
    /// Only ever called between commands (the actor is the single device owner and is not serving
    /// anything else while this runs), so an eviction can never pull a model out from under a
    /// request in flight.
    pub fn ensure_resident(&mut self, cfg: &ModelCfg, loader: &dyn ModelLoader, srv: &ServerCfg,
                           now: Instant) -> Result<(), EngineError> {
        if self.get_loaded(&cfg.name).is_some() { return Ok(()); }
        while self.resident_count() >= srv.max_resident {
            if srv.evict_policy == EvictPolicy::None {
                return Err(EngineError::Unsupported(format!(
                    "{} is not resident and evict_policy = \"none\" at max_resident ({})",
                    cfg.name, srv.max_resident)));
            }
            // Nothing resident left to evict (max_resident = 0): stop, and let try_load record the
            // capacity refusal rather than loop forever.
            match self.lru_victim() {
                Some(v) => self.release(&v, &format!("evicted for {}", cfg.name)),
                None => break,
            }
        }
        // NOTE: the memory_ceiling is checked inside try_load and is NOT an evict trigger -- a
        // model's footprint is unknown until it is loaded, so there is no amount to free "enough"
        // of. bo_bytes() is 0 for every real model today (backlog R11(f)), so this is moot in
        // production; when it starts reporting, revisit.
        self.try_load(cfg, loader, srv, now);
        if self.get_loaded(&cfg.name).is_some() { return Ok(()); }
        let why = self.entries.iter().find(|e| e.cfg.name == cfg.name)
            .map(|e| e.status.detail.clone()).unwrap_or_else(|| "load failed".into());
        Err(EngineError::Load(format!("{}: {why}", cfg.name)))
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
                name: cfg.name.clone(), state: LoadState::Unloaded, detail, capability, bo_bytes: 0, idle_s: None,
            },
            last_used: Instant::now(),
        });
    }

    /// Least-recently-used RESIDENT model, if any.
    pub fn lru_victim(&self) -> Option<String> {
        self.entries.iter().filter(|e| e.model.is_some())
            .min_by_key(|e| e.last_used).map(|e| e.cfg.name.clone())
    }

    /// Unload every resident model idle for at least `idle`, returning what was released.
    ///
    /// The actor calls this from its `recv_timeout` idle branch, i.e. only between commands.
    pub fn sweep_idle(&mut self, now: Instant, idle: Duration) -> Vec<String> {
        let expired: Vec<String> = self.entries.iter()
            .filter(|e| e.model.is_some() && now.saturating_duration_since(e.last_used) >= idle)
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

    fn set_failed(&mut self, cfg: &ModelCfg, detail: String) {
        self.set_state(cfg, LoadState::Failed, detail);
    }
    /// Configured, wanted, and deliberately not resident -- distinct from a failure.
    fn set_deferred(&mut self, cfg: &ModelCfg, detail: String) {
        self.set_state(cfg, LoadState::Unloaded, detail);
    }
    fn set_state(&mut self, cfg: &ModelCfg, state: LoadState, detail: String) {
        // Keep a capability learned from an earlier successful load: still true, and routing uses it.
        let capability = self.known_capability(&cfg.name);
        let status = ModelStatus {
            name: cfg.name.clone(), state, detail, capability, bo_bytes: 0, idle_s: None,
        };
        self.upsert(Entry { cfg: cfg.clone(), model: None, status, last_used: Instant::now() });
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
    fn cfg(name: &str) -> ModelCfg { ModelCfg { name: name.into(), scenario: "x".into() } }
    fn loader(names: &[&str]) -> MockLoader {
        let mut t = BTreeMap::new();
        for n in names { t.insert((*n).to_string(), Ok((Capability::EMBED, 1))); }
        MockLoader { table: t }
    }

    #[test]
    fn load_failure_is_recorded_not_fatal() {
        let mut t = BTreeMap::new();
        t.insert("good".to_string(), Ok((Capability::EMBED, 10)));
        t.insert("bad".to_string(), Err("boom".to_string()));
        let l = MockLoader { table: t };
        let srv = ServerCfg { max_resident: 8, ..Default::default() };
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
    fn max_resident_defers_the_overflow_instead_of_failing_it() {
        let l = loader(&["a", "b"]);
        let srv = ServerCfg { max_resident: 1, ..Default::default() };
        let mut r = Registry::default();
        let now = Instant::now();
        r.try_load(&cfg("a"), &l, &srv, now);
        r.try_load(&cfg("b"), &l, &srv, now);
        assert_eq!(r.resident_count(), 1, "reconcile must not evict: booting would thrash");
        let b = r.status().into_iter().find(|x| x.name == "b").unwrap();
        assert_eq!(b.state, LoadState::Unloaded, "over capacity is a capacity decision, not a failure");
        assert!(b.detail.contains("max_resident"), "{}", b.detail);
    }
    #[test]
    fn ensure_resident_evicts_the_lru_not_just_anyone() {
        let l = loader(&["a", "b", "c"]);
        let srv = ServerCfg { max_resident: 2, ..Default::default() };
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
        let srv = ServerCfg { max_resident: 1, ..Default::default() };
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
        let srv = ServerCfg { max_resident: 1, evict_policy: EvictPolicy::None, ..Default::default() };
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
        let srv = ServerCfg { max_resident: 8, ..Default::default() };
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
        let srv = ServerCfg { max_resident: 1, ..Default::default() };
        let mut r = Registry::default();
        let t0 = Instant::now();
        r.try_load(&cfg("a"), &l, &srv, t0);
        assert_eq!(r.sweep_idle(t0 + Duration::from_secs(900), Duration::from_secs(900)), vec!["a"]);
        assert_eq!(r.known_capability("a"), Some(Capability::EMBED), "capability survives the unload");
        r.ensure_resident(&cfg("a"), &l, &srv, t0 + Duration::from_secs(901)).unwrap();
        assert!(r.get_loaded("a").is_some());
        assert_eq!(r.status().into_iter().find(|x| x.name == "a").unwrap().state, LoadState::Loaded);
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
        let srv = ServerCfg { max_resident: 1, ..Default::default() };
        let mut r = Registry::default();
        let e = r.ensure_resident(&cfg("bad"), &l, &srv, Instant::now()).unwrap_err();
        assert!(e.to_string().contains("no such xclbin"), "{e}");
    }
}
