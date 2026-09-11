//! Bring the registry (actual) in line with the config (desired): load missing, unload removed,
//! reload changed, evict over the cap. Each step independent; failures recorded as Failed, never
//! fatal.
use crate::config::Config;
use crate::loader::ModelLoader;
use crate::registry::{LoadState, Registry};

#[derive(Debug, Default, PartialEq)]
pub struct ReconcileReport {
    pub loaded: Vec<String>,
    pub unloaded: Vec<String>,
    pub failed: Vec<String>,
    /// Configured and wanted, but left out of residency by `max_resident`. Not a failure: these load
    /// on demand when a request asks for them.
    pub deferred: Vec<String>,
    /// Released because `max_resident` was lowered under what is already resident.
    ///
    /// `max_resident` used to be an ADMISSION limit only: `try_load` refused new loads over it and
    /// nothing looked at the models already in. So lowering it from 5 to 2 left five resident, and
    /// `npu reload` -- whose entire job is making actual match desired -- reported `deferred: 4`
    /// and changed nothing. The cap converged only through the idle sweep or through a later load
    /// evicting one victim at a time, which is to say through traffic rather than through the
    /// command that was asked to do it.
    pub evicted: Vec<String>,
    /// Still resident over the cap after eviction, because a pin is not an eviction candidate.
    ///
    /// Reported rather than resolved: `resident = true` means "never an eviction victim", and a
    /// cap that silently overrode it would make a pin mean nothing. Both numbers being visible is
    /// what lets an operator see that the two settings disagree.
    pub pinned_over_cap: Vec<String>,
    /// Deferred models that the config PINS (`resident = true`). A pin is only exempt-from-eviction,
    /// not entitlement to a slot -- admission is still first-N-in-config-order -- so a pin listed
    /// after enough unpinned models never becomes resident at boot. That is a real outcome, but it
    /// is the opposite of what the config asked for, so it is reported rather than left for someone
    /// to notice in `/v1/models`.
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
    // load / reload
    for m in &cfg.models {
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
        reg.unload(&m.name);
        reg.try_load(m, loader, &cfg.server, now);
        if reg.get_loaded(&m.name).is_some() { rep.loaded.push(m.name.clone()); }
        else if reg.entries.iter().any(|e| e.cfg.name == m.name && e.status.state == LoadState::Unloaded) {
            // Over max_resident. Reporting this as `failed` was honest before hot-swap and is a lie
            // now: nothing went wrong and the model is one request away from being resident.
            rep.deferred.push(m.name.clone());
            if m.resident { rep.pinned_deferred.push(m.name.clone()); }
        }
        else { rep.failed.push(m.name.clone()); }
    }

    // Evict down to the cap. LRU order, and `lru_victim` already excludes pins, so this stops when
    // only pinned models remain over the cap rather than breaking the pin contract to satisfy the
    // number. The loop terminates because every iteration releases one resident model.
    while reg.resident_count() > cfg.server.max_resident {
        let Some(victim) = reg.lru_victim() else { break };
        reg.release(&victim, &format!("evicted: over max_resident ({})", cfg.server.max_resident));
        rep.evicted.push(victim);
    }
    if reg.resident_count() > cfg.server.max_resident {
        rep.pinned_over_cap = reg.entries.iter()
            .filter(|e| e.model.is_some() && e.cfg.resident)
            .map(|e| e.cfg.name.clone())
            .collect();
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
    fn loader(names: &[(&str, bool)]) -> MockLoader {
        let mut t = BTreeMap::new();
        for (n, ok) in names {
            t.insert((*n).to_string(), if *ok { Ok((Capability::EMBED, 1)) } else { Err("fail".into()) });
        }
        MockLoader { table: t }
    }
    fn cfg(names: &[&str]) -> Config {
        Config {
            server: ServerCfg { max_resident: 8, ..Default::default() },
            models: names.iter().map(|n| ModelCfg { name: (*n).into(), scenario: "x".into(), resident: false }).collect(),
            ..Default::default()
        }
    }
    /// The reported bug: `npu config set max_resident 2` then `npu reload` left five models
    /// resident and answered `deferred: 4`. The cap was an admission limit only.
    #[test]
    fn lowering_the_cap_evicts_down_to_it() {
        let l = loader(&[("a", true), ("b", true), ("c", true)]);
        let mut c = cfg(&["a", "b", "c"]);
        let mut reg = Registry::default();
        reconcile(&c, &mut reg, &l);
        assert_eq!(reg.resident_count(), 3, "all three fit under the default cap");

        c.server.max_resident = 1;
        let rep = reconcile(&c, &mut reg, &l);
        assert_eq!(reg.resident_count(), 1, "the cap is now enforced downward");
        assert_eq!(rep.evicted.len(), 2, "and says what it dropped: {:?}", rep.evicted);
        assert!(rep.failed.is_empty(), "eviction is not a failure: {:?}", rep.failed);
        assert!(rep.pinned_over_cap.is_empty());
    }

    /// A pin means "never an eviction victim". A cap that overrode it would make a pin mean
    /// nothing, so the disagreement is reported instead of resolved.
    #[test]
    fn a_pin_is_not_evicted_to_satisfy_the_cap() {
        let l = loader(&[("a", true), ("b", true)]);
        let mut c = cfg(&["a", "b"]);
        c.models[0].resident = true;
        c.models[1].resident = true;
        let mut reg = Registry::default();
        reconcile(&c, &mut reg, &l);
        assert_eq!(reg.resident_count(), 2);

        c.server.max_resident = 1;
        let rep = reconcile(&c, &mut reg, &l);
        assert_eq!(reg.resident_count(), 2, "both are pinned, so neither is a candidate");
        assert!(rep.evicted.is_empty());
        assert_eq!(rep.pinned_over_cap.len(), 2, "the operator is told the two settings disagree");
    }

    /// Unpinned first, and only as many as the cap requires -- the pin survives, the cap is met.
    #[test]
    fn eviction_takes_unpinned_models_and_stops_at_the_cap() {
        let l = loader(&[("a", true), ("b", true), ("c", true)]);
        let mut c = cfg(&["a", "b", "c"]);
        c.models[0].resident = true;               // `a` is pinned
        let mut reg = Registry::default();
        reconcile(&c, &mut reg, &l);

        c.server.max_resident = 2;
        let rep = reconcile(&c, &mut reg, &l);
        assert_eq!(reg.resident_count(), 2);
        assert_eq!(rep.evicted.len(), 1, "exactly one over the cap: {:?}", rep.evicted);
        assert!(!rep.evicted.contains(&"a".to_string()), "the pin is not the victim");
        assert!(reg.get_loaded("a").is_some(), "the pin is still resident");
        assert!(rep.pinned_over_cap.is_empty());
    }

    #[test]
    fn loads_unloads_and_records_failures() {
        let l = loader(&[("a", true), ("b", false)]);
        let mut reg = Registry::default();
        let rep = reconcile(&cfg(&["a", "b"]), &mut reg, &l);
        assert_eq!(rep.loaded, vec!["a"]);
        assert_eq!(rep.failed, vec!["b"]);
        let rep2 = reconcile(&cfg(&["b"]), &mut reg, &l);
        assert!(rep2.unloaded.contains(&"a".to_string()));
        assert!(reg.get_loaded("a").is_none());
    }
    #[test]
    fn over_capacity_reports_deferred_not_failed() {
        let l = loader(&[("a", true), ("b", true)]);
        let mut reg = Registry::default();
        let mut c = cfg(&["a", "b"]);
        c.server.max_resident = 1;
        let rep = reconcile(&c, &mut reg, &l);
        assert_eq!(rep.loaded, vec!["a"]);
        assert_eq!(rep.deferred, vec!["b"], "over max_resident is a deferral, not a failure");
        assert!(rep.failed.is_empty(), "{:?}", rep.failed);
    }
    /// The bug `update_cfg` exists for: a pin only took effect once something else had unloaded the
    /// model, i.e. never while it was the thing you were trying to protect.
    #[test]
    fn a_pin_flipped_on_a_loaded_model_takes_effect_without_a_reload() {
        let l = loader(&[("a", true)]);
        let mut reg = Registry::default();
        let mut c = cfg(&["a"]);
        reconcile(&c, &mut reg, &l);
        assert_eq!(reg.lru_victim().as_deref(), Some("a"), "unpinned, so it is an eviction candidate");

        c.models[0].resident = true;
        let rep = reconcile(&c, &mut reg, &l);
        assert!(rep.loaded.is_empty(), "a pin change must not churn the device");
        assert_eq!(reg.lru_victim(), None, "the pin has to reach lru_victim, which reads e.cfg");
        assert!(reg.status()[0].pinned, "and it has to be visible from outside");

        c.models[0].resident = false;
        reconcile(&c, &mut reg, &l);
        assert_eq!(reg.lru_victim().as_deref(), Some("a"), "unpin has to take effect the same way");
        assert!(!reg.status()[0].pinned);
    }

    /// A pin that admission declines is reported. `resident = true` means exempt-from-eviction, not
    /// entitled to a slot, so config order still decides -- but silence about it is what made the
    /// intent look honoured when it was not.
    #[test]
    fn a_pin_that_admission_declines_is_named_not_swallowed() {
        let l = loader(&[("a", true), ("b", true)]);
        let mut reg = Registry::default();
        let mut c = cfg(&["a", "b"]);
        c.server.max_resident = 1;
        c.models[1].resident = true;               // pinned, but second in config order
        let rep = reconcile(&c, &mut reg, &l);
        assert_eq!(rep.loaded, vec!["a"]);
        assert_eq!(rep.deferred, vec!["b"]);
        assert_eq!(rep.pinned_deferred, vec!["b"], "a declined pin must be named");
    }

    #[test]
    fn an_unpinned_deferral_is_not_reported_as_a_declined_pin() {
        let l = loader(&[("a", true), ("b", true)]);
        let mut reg = Registry::default();
        let mut c = cfg(&["a", "b"]);
        c.server.max_resident = 1;
        let rep = reconcile(&c, &mut reg, &l);
        assert_eq!(rep.deferred, vec!["b"]);
        assert!(rep.pinned_deferred.is_empty(), "nothing was pinned, so nothing was declined");
    }

    #[test]
    fn reconcile_reloads_a_model_the_sweep_released() {
        let l = loader(&[("a", true)]);
        let mut reg = Registry::default();
        let c = cfg(&["a"]);
        reconcile(&c, &mut reg, &l);
        reg.release("a", "idle");
        let rep = reconcile(&c, &mut reg, &l);
        assert_eq!(rep.loaded, vec!["a"], "an idle-unloaded model must come back on /admin/reload");
    }
}
