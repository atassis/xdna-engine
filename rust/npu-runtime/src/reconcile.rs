//! Bring the registry (actual) in line with the config (desired): load missing, unload removed,
//! reload changed, evict over the cap. Each step independent; failures recorded as Failed, never
//! fatal.
use crate::config::Config;
use crate::loader::ModelLoader;
use crate::registry::{LoadState, Registry, UnloadReason};

#[derive(Debug, Default, PartialEq)]
pub struct ReconcileReport {
    /// A PIN this pass actually loaded. Unpinned models are declared, not loaded, here -- see the
    /// load loop below -- so this (and `failed`) only ever names pins.
    pub loaded: Vec<String>,
    pub unloaded: Vec<String>,
    /// A PIN whose load failed. An unpinned model's load failure is discovered lazily, on the
    /// request that actually wants it, not by this eager pass.
    pub failed: Vec<String>,
    /// A PIN, wanted resident, but left out because it does not fit under `memory_ceiling_mb`.
    /// Pins-only: an unpinned model is never attempted here at all (see the load loop below), so it
    /// can never be deferred by it either -- it loads on demand when a request actually asks for it
    /// (via `ensure_resident`, which evicts to make room; this pass never does).
    pub deferred: Vec<String>,
    /// Released because `memory_ceiling_mb` was lowered under what is already resident, or an
    /// already-resident pin's own live footprint grew past what the invariant can still protect.
    ///
    /// The cap used to be an ADMISSION limit only: `try_load` refused new loads over it and nothing
    /// looked at the models already in. So lowering it left everything resident, and `npu reload`
    /// -- whose entire job is making actual match desired -- reported `deferred` and changed
    /// nothing. The cap converged only through the idle sweep or through a later load evicting one
    /// victim at a time, which is to say through traffic rather than through the command that was
    /// asked to do it.
    pub evicted: Vec<String>,
    /// Pinned AND currently loaded, but the invariant (`sum(pinned bytes) <= memory_ceiling_mb`) no
    /// longer protects it -- demoted by `Registry::set_pin_honored`, in least-recently-touched-pin
    /// order, until the pinned sum fits again. `reload` will NOT fix this: the config still says
    /// `resident = true` (untouched, see `Entry::pin_honored`), and it will keep landing here every
    /// pass until the operator actually changes something -- unpins a different model, raises the
    /// ceiling, or repoints a scenario.
    pub pinned_over_cap: Vec<String>,
    /// Deferred models that the config PINS. Pins are admitted BEFORE any on-demand model (see the
    /// load loop below), so landing here means the pin does not fit even walked first -- either
    /// alone against the ceiling, or against earlier pins in config order. Order among competing
    /// pins is still a real, if narrower, thing to report: this is what
    /// `boot-admission-order-ignores-pins` resolved, not eliminated.
    pub pinned_deferred: Vec<String>,
}

pub fn reconcile(cfg: &Config, reg: &mut Registry, loader: &dyn ModelLoader) -> ReconcileReport {
    let mut rep = ReconcileReport::default();
    // One clock for the whole pass, so models loaded together share a last_used and LRU order falls
    // back to config order rather than to load duration.
    let now = std::time::Instant::now();
    // unload: entries whose name is gone from config
    let want: std::collections::BTreeSet<&str> = cfg.models.iter().map(|m| m.name.as_str()).collect();
    let to_unload: Vec<String> = reg.entries.iter().map(|e| e.cfg.name.clone())
        .filter(|n| !want.contains(n.as_str())).collect();
    for n in to_unload { reg.unload(&n); rep.unloaded.push(n); }

    // load / reload -- PINS ONLY. Reconcile is the eager pass ("a server should come up warm"),
    // and a pin is the sole way to declare "always on" -- so it is also the sole thing eager
    // admission reaches for. An unpinned model that isn't already resident is DECLARED (host-only,
    // no device touch, exactly what `start_lazy` already does for everything), never proactively
    // loaded; a real request drives its actual load via `ensure_resident`. This was not always
    // true: before the byte-based rewrite, EVERY configured model was walked through `try_load`
    // here, gated only by `max_resident`/the ceiling -- which meant "how much eager admission
    // reaches" was an accident of whatever cap happened to be set, not a declared intent. Measured
    // cost of that: a 13-model config with a 20 GB ceiling (sized for one large model) tried to
    // eagerly load most of them at boot, competing for host memory it had no real reason to spend
    // before anything asked for it.
    let (pinned_models, other_models): (Vec<&crate::config::ModelCfg>, Vec<&crate::config::ModelCfg>) =
        cfg.models.iter().partition(|m| m.resident);
    for m in pinned_models.into_iter().chain(other_models) {
        let existing = reg.entries.iter().find(|e| e.cfg.name == m.name);
        let needs = match existing {
            None => true,                                  // not present
            Some(e) if e.model.is_none() => true,          // present but failed/unloaded -> retry
            Some(e) => e.cfg.scenario != m.scenario,        // spec changed -> reload
        };
        if !needs {
            // The loaded model is still the right one, but the rest of the ModelCfg may have moved.
            // Without this a pin/unpin took effect only after something else unloaded the model.
            reg.update_cfg(&m.name, m);
            continue;
        }
        if !m.resident {
            // New, or its scenario changed: re-declare under the fresh manifest. `model.is_none()`
            // alone (an idle-swept or previously-declared entry, scenario unchanged) is left exactly
            // as it is -- `declare` would no-op on it anyway, and there is nothing to refresh.
            if existing.is_none() || existing.is_some_and(|e| e.cfg.scenario != m.scenario) {
                reg.unload(&m.name);
                let cap = reg.declared(m, loader);
                reg.declare(m, cap);
            }
            continue;
        }
        reg.unload(&m.name);
        reg.try_load(m, loader, &cfg.server, now);
        if reg.get_loaded(&m.name).is_some() { rep.loaded.push(m.name.clone()); }
        else if reg.entries.iter().any(|e| e.cfg.name == m.name && e.status.state == LoadState::Unloaded) {
            // Over the byte budget. Reporting this as `failed` was honest before hot-swap and is a
            // lie now: nothing went wrong and the model is one request away from being resident.
            rep.deferred.push(m.name.clone());
            if m.resident { rep.pinned_deferred.push(m.name.clone()); }
        }
        else { rep.failed.push(m.name.clone()); }
    }

    // Recompute which pins the invariant currently protects, using LIVE bytes (everything here is
    // already loaded, so there is a real number, not an estimate). Least-recently-touched pin is
    // dropped first if the sum does not fit -- e.g. the ceiling was just lowered under an
    // already-resident pinned set. `set_pin_honored` is what makes a demoted pin evictable below.
    let ceiling = cfg.server.memory_ceiling_mb * 1024 * 1024;
    let mut pinned_loaded: Vec<(String, u64, std::time::Instant)> = reg.entries.iter()
        .filter(|e| e.cfg.resident && e.model.is_some())
        .map(|e| (e.cfg.name.clone(), e.model.as_ref().unwrap().footprint(), e.last_used))
        .collect();
    // Newest-touched first, so it claims the budget first; the LEAST-recently-touched pin is what
    // runs out of room and gets demoted -- genuine LRU, not just insertion order.
    pinned_loaded.sort_by_key(|(_, _, last_used)| std::cmp::Reverse(*last_used));
    let mut honored = std::collections::HashSet::new();
    let mut spent = 0u64;
    for (name, bytes, _) in &pinned_loaded {
        if spent + bytes <= ceiling { spent += bytes; honored.insert(name.clone()); }
    }
    reg.set_pin_honored(&honored);
    rep.pinned_over_cap = pinned_loaded.into_iter()
        .map(|(n, _, _)| n).filter(|n| !honored.contains(n)).collect();

    // Evict down to the ceiling. LRU order; `lru_victim` excludes only HONOURED pins, so a pin just
    // demoted above is a normal candidate, same as any unpinned model. The loop terminates because
    // every iteration releases one resident model.
    while reg.resident_bytes() > ceiling {
        let Some(victim) = reg.lru_victim() else { break };
        reg.release(&victim, &format!("evicted: over memory_ceiling_mb ({} MB)",
            cfg.server.memory_ceiling_mb), UnloadReason::Evicted);
        rep.evicted.push(victim);
    }
    rep
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Config, ModelCfg, ServerCfg};
    use crate::loader::mock::MockLoader;
    use npu_engine::capability::Capability;
    use std::collections::BTreeMap;
    const MB: u64 = 1024 * 1024;
    /// Every mock model here costs one MB, so `ceiling_mb(n)` admits exactly `n` of them.
    fn loader(names: &[(&str, bool)]) -> MockLoader {
        let mut t = BTreeMap::new();
        for (n, ok) in names {
            t.insert((*n).to_string(), if *ok { Ok((Capability::EMBED, MB)) } else { Err("fail".into()) });
        }
        MockLoader { table: t }
    }
    fn model(name: &str) -> ModelCfg {
        ModelCfg { name: name.into(), scenario: "x".into(), resident: false }
    }
    fn cfg(names: &[&str]) -> Config {
        Config {
            server: ServerCfg { memory_ceiling_mb: 8, ..Default::default() },
            models: names.iter().map(|n| ModelCfg { name: (*n).into(), scenario: "x".into(), resident: false }).collect(),
            ..Default::default()
        }
    }
    /// The reported bug: `npu config set max_resident 2` then `npu reload` left five models
    /// resident and answered `deferred: 4`. The cap was an admission limit only. Unpinned models are
    /// no longer eagerly loaded by reconcile at all (see `nothing_is_eagerly_loaded_with_no_pins`),
    /// so this now warms them the way a real deployment would -- through requests -- via
    /// `load_explicit`, then checks reconcile still enforces a lowered cap on whatever is actually
    /// resident, regardless of how it got there.
    #[test]
    fn lowering_the_cap_evicts_down_to_it() {
        let l = loader(&[("a", true), ("b", true), ("c", true)]);
        let mut c = cfg(&["a", "b", "c"]);
        let mut reg = Registry::default();
        let now = std::time::Instant::now();
        for n in ["a", "b", "c"] { reg.load_explicit(&model(n), &l, &c.server, now).unwrap(); }
        assert_eq!(reg.resident_count(), 3, "all three fit under the default cap");

        c.server.memory_ceiling_mb = 1;
        let rep = reconcile(&c, &mut reg, &l);
        assert_eq!(reg.resident_count(), 1, "the cap is now enforced downward");
        assert_eq!(rep.evicted.len(), 2, "and says what it dropped: {:?}", rep.evicted);
        assert!(rep.failed.is_empty(), "eviction is not a failure: {:?}", rep.failed);
        assert!(rep.pinned_over_cap.is_empty());
    }

    /// The owner's exact question: with nothing pinned, `npu serve`'s eager reconcile must not touch
    /// the device at all -- not "load whatever fits the ceiling". A 20 GB ceiling sized for one large
    /// model must not turn into an invitation to warm every other configured model at boot.
    #[test]
    fn nothing_is_eagerly_loaded_with_no_pins() {
        let l = loader(&[("a", true), ("b", true), ("c", true)]);
        let c = cfg(&["a", "b", "c"]); // default ceiling is generous; all three would fit
        let mut reg = Registry::default();
        let rep = reconcile(&c, &mut reg, &l);
        assert_eq!(reg.resident_count(), 0, "nothing is pinned, so nothing is loaded");
        assert!(rep.loaded.is_empty() && rep.deferred.is_empty() && rep.failed.is_empty(),
            "unpinned models are declared, not attempted, so none of these can name one: {rep:?}");
        // Declared, not silently absent: routing and `npu model ls` still know what each one is.
        for n in ["a", "b", "c"] {
            assert_eq!(reg.known_capability(n), Some(Capability::EMBED), "{n} must still be declared");
        }
    }

    /// Two pins together exceed a lowered ceiling: the invariant is restored by demoting the
    /// LEAST-RECENTLY-TOUCHED one, which then becomes a normal eviction candidate -- not by leaving
    /// both resident and merely reporting the disagreement, which is what this used to do before the
    /// invariant was enforced rather than advisory.
    #[test]
    fn the_older_of_two_over_budget_pins_is_demoted_and_evicted() {
        let l = loader(&[("a", true), ("b", true)]);
        let mut c = cfg(&["a", "b"]);
        c.models[0].resident = true;
        c.models[1].resident = true;
        let mut reg = Registry::default();
        reconcile(&c, &mut reg, &l);
        assert_eq!(reg.resident_count(), 2);
        reg.touch("b", std::time::Instant::now() + std::time::Duration::from_secs(10)); // `b` is now the newer one

        c.server.memory_ceiling_mb = 1; // only one MB model fits
        let rep = reconcile(&c, &mut reg, &l);
        assert_eq!(rep.pinned_over_cap, vec!["a".to_string()], "the older pin loses protection first");
        assert_eq!(rep.evicted, vec!["a".to_string()], "and is then evicted like any unpinned model");
        assert!(reg.get_loaded("a").is_none());
        assert!(reg.get_loaded("b").is_some(), "the newer pin still fits and survives");
        assert!(reg.status().into_iter().find(|s| s.name == "a").unwrap().pinned,
            "the config's own declared intent is untouched by the demotion");
    }

    /// Unpinned first, and only as many as the cap requires -- the pin survives, the cap is met.
    /// `b`/`c` are warmed via `load_explicit` (real request traffic, not eager reconcile -- they are
    /// unpinned, so reconcile alone would only declare them); the pin comes from the first
    /// `reconcile`, matching how it would actually get there in a running service.
    #[test]
    fn eviction_takes_unpinned_models_and_stops_at_the_cap() {
        let l = loader(&[("a", true), ("b", true), ("c", true)]);
        let mut c = cfg(&["a", "b", "c"]);
        c.models[0].resident = true;               // `a` is pinned
        let mut reg = Registry::default();
        let now = std::time::Instant::now();
        reconcile(&c, &mut reg, &l);
        for n in ["b", "c"] { reg.load_explicit(&model(n), &l, &c.server, now).unwrap(); }

        c.server.memory_ceiling_mb = 2;
        let rep = reconcile(&c, &mut reg, &l);
        assert_eq!(reg.resident_count(), 2);
        assert_eq!(rep.evicted.len(), 1, "exactly one over the cap: {:?}", rep.evicted);
        assert!(!rep.evicted.contains(&"a".to_string()), "the pin is not the victim");
        assert!(reg.get_loaded("a").is_some(), "the pin is still resident");
        assert!(rep.pinned_over_cap.is_empty());
    }

    /// `loaded`/`failed` only ever name PINS now -- an unpinned model is declared, not attempted, so
    /// both are pinned here to actually exercise this path.
    #[test]
    fn loads_unloads_and_records_failures() {
        let l = loader(&[("a", true), ("b", false)]);
        let mut reg = Registry::default();
        let mut c = cfg(&["a", "b"]);
        c.models[0].resident = true;
        c.models[1].resident = true;
        let rep = reconcile(&c, &mut reg, &l);
        assert_eq!(rep.loaded, vec!["a"]);
        assert_eq!(rep.failed, vec!["b"]);
        let mut c2 = cfg(&["b"]);
        c2.models[0].resident = true;
        let rep2 = reconcile(&c2, &mut reg, &l);
        assert!(rep2.unloaded.contains(&"a".to_string()));
        assert!(reg.get_loaded("a").is_none());
    }
    /// Same reasoning: `deferred` only ever names a pin that does not fit. Both pinned here.
    #[test]
    fn over_capacity_reports_deferred_not_failed() {
        let l = loader(&[("a", true), ("b", true)]);
        let mut reg = Registry::default();
        let mut c = cfg(&["a", "b"]);
        c.models[0].resident = true;
        c.models[1].resident = true;
        c.server.memory_ceiling_mb = 1;
        let rep = reconcile(&c, &mut reg, &l);
        assert_eq!(rep.loaded, vec!["a"]);
        assert_eq!(rep.deferred, vec!["b"], "over the ceiling is a deferral, not a failure");
        assert!(rep.failed.is_empty(), "{:?}", rep.failed);
    }
    /// The bug `update_cfg` exists for: a pin only took effect once something else had unloaded the
    /// model, i.e. never while it was the thing you were trying to protect. "a" is unpinned, so it
    /// is warmed via `load_explicit` (real request traffic), not by reconcile's own eager pass.
    #[test]
    fn a_pin_flipped_on_a_loaded_model_takes_effect_without_a_reload() {
        let l = loader(&[("a", true)]);
        let mut reg = Registry::default();
        let mut c = cfg(&["a"]);
        reg.load_explicit(&model("a"), &l, &c.server, std::time::Instant::now()).unwrap();
        assert_eq!(reg.lru_victim().as_deref(), Some("a"), "unpinned, so it is an eviction candidate");

        c.models[0].resident = true;
        let rep = reconcile(&c, &mut reg, &l);
        assert!(rep.loaded.is_empty(), "a pin change must not churn the device");
        assert_eq!(reg.lru_victim(), None, "the pin has to reach lru_victim via pin_honored");
        assert!(reg.status()[0].pinned, "and it has to be visible from outside");

        c.models[0].resident = false;
        reconcile(&c, &mut reg, &l);
        assert_eq!(reg.lru_victim().as_deref(), Some("a"), "unpin has to take effect the same way");
        assert!(!reg.status()[0].pinned);
    }

    /// A pin outranks an unpinned model at boot trivially now: the unpinned one was never a
    /// candidate for eager admission to begin with (declared, not attempted), so "second in the
    /// file" cannot cost it a slot it was never competing for either. This is
    /// `boot-admission-order-ignores-pins` resolved by construction -- the contention it used to
    /// name (a pin losing to an EARLIER unpinned model) no longer exists as a category.
    #[test]
    fn a_pin_now_outranks_config_order_at_boot() {
        let l = loader(&[("a", true), ("b", true)]);
        let mut reg = Registry::default();
        let mut c = cfg(&["a", "b"]);
        c.server.memory_ceiling_mb = 1;
        c.models[1].resident = true;               // pinned, second in the file, wins anyway
        let rep = reconcile(&c, &mut reg, &l);
        assert_eq!(rep.loaded, vec!["b"], "the pin is admitted regardless of file order");
        assert!(rep.deferred.is_empty(), "the unpinned model was declared, never a candidate: {rep:?}");
        assert!(rep.pinned_deferred.is_empty(), "the pin itself was admitted -- nothing was declined");
        assert!(reg.get_loaded("a").is_none(), "unpinned, so still cold");
    }

    /// Two competing pins can still collide with EACH OTHER's order -- pins-first narrows the old
    /// defect, it does not remove config order as a tiebreaker among pins.
    #[test]
    fn a_pin_can_still_lose_to_an_earlier_pin() {
        let l = loader(&[("a", true), ("b", true)]);
        let mut reg = Registry::default();
        let mut c = cfg(&["a", "b"]);
        c.server.memory_ceiling_mb = 1;
        c.models[0].resident = true;
        c.models[1].resident = true;               // both pinned; only one MB to share
        let rep = reconcile(&c, &mut reg, &l);
        assert_eq!(rep.loaded, vec!["a"], "first pin in config order wins the shared budget");
        assert_eq!(rep.deferred, vec!["b"]);
        assert_eq!(rep.pinned_deferred, vec!["b"], "a declined pin must still be named");
    }

    /// An unpinned model too small to be the issue here: even against a ceiling that could not admit
    /// it if it TRIED, it is never deferred OR failed, because it is never attempted at all.
    #[test]
    fn an_unpinned_model_over_the_ceiling_is_declared_not_deferred() {
        let l = loader(&[("a", true), ("b", true)]);
        let mut reg = Registry::default();
        let mut c = cfg(&["a", "b"]);
        c.server.memory_ceiling_mb = 1;
        let rep = reconcile(&c, &mut reg, &l);
        assert!(rep.deferred.is_empty(), "neither is pinned, so neither is a candidate: {rep:?}");
        assert!(rep.pinned_deferred.is_empty(), "nothing was pinned, so nothing was declined");
        assert_eq!(reg.known_capability("b"), Some(Capability::EMBED), "declared anyway, not silently absent");
    }

    /// A PIN the idle sweep released comes back on the next reconcile -- that is the whole point of
    /// "always on". An UNPINNED model released the same way does not: reconcile only ever eagerly
    /// loads pins (see `nothing_is_eagerly_loaded_with_no_pins`), so bringing an unpinned model back
    /// is left to the request that actually wants it next.
    #[test]
    fn reconcile_reloads_a_released_pin_but_not_a_released_unpinned_model() {
        let l = loader(&[("a", true), ("b", true)]);
        let mut reg = Registry::default();
        let mut c = cfg(&["a", "b"]);
        c.models[0].resident = true; // "a" pinned, "b" not
        reconcile(&c, &mut reg, &l);
        reg.load_explicit(&model("b"), &l, &c.server, std::time::Instant::now()).unwrap();
        assert!(reg.get_loaded("a").is_some() && reg.get_loaded("b").is_some());

        reg.release("a", "idle", UnloadReason::Idle);
        reg.release("b", "idle", UnloadReason::Idle);
        let rep = reconcile(&c, &mut reg, &l);
        assert_eq!(rep.loaded, vec!["a"], "the pin must come back on its own");
        assert!(reg.get_loaded("b").is_none(), "the unpinned model stays cold until a request asks for it");
    }
}
