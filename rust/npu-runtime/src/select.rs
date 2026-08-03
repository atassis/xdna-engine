//! Capability routing: omit -> configured default; named -> that one; ambiguous/unknown -> error.
//!
//! The name this returns may NOT be resident yet. Residency is the caller's job (`ensure_resident`),
//! which is what makes switching models a per-request choice instead of a config edit. Routing
//! therefore reasons over CONFIGURED models and their last-known capability, not over what happens
//! to be loaded right now -- so the answer does not change just because an idle sweep ran.
use crate::config::Config;
use crate::registry::{Capability, Registry};
use npu_engine::EngineError;

/// Resolve the model NAME to serve a capability. Caller then makes it resident and fetches it.
pub fn resolve(cfg: &Config, reg: &Registry, cap: Capability, want: Option<&str>)
    -> Result<String, EngineError> {
    if let Some(n) = want {
        return match reg.get_loaded(n) {
            Some(m) if m.capabilities() == cap => Ok(n.to_string()),
            Some(m) => Err(EngineError::WrongKind { wanted: cap, got: m.capabilities() }),
            // Not resident. If it is configured, it loads on demand and its capability is checked
            // after the load; if it is not configured at all there is nothing to load.
            None if cfg.find(n).is_some() => match reg.known_capability(n) {
                Some(k) if k != cap => Err(EngineError::WrongKind { wanted: cap, got: k }),
                _ => Ok(n.to_string()),
            },
            None => Err(EngineError::Load(format!("unknown model {n:?} (not in the config)"))),
        };
    }
    // One map lookup where this used to be a match over every ModelKind variant -- the reason a
    // third capability could not have a configured default at all.
    if let Some(n) = cfg.defaults.get(cap) {
        match reg.known_capability(n) {
            Some(k) if k == cap => return Ok(n.clone()),
            // Never loaded, so its capability is unknown: the operator nominated it for this
            // capability, so try it and let the post-load check have the last word.
            None if cfg.find(n).is_some() => return Ok(n.clone()),
            _ => {}                                   // wrong kind, or not configured -> fall through
        }
    }
    // Candidates = every model KNOWN to have this capability, resident or not (an unloaded entry
    // still remembers its kind). Counting the non-resident ones keeps "which model did you mean?" a
    // property of the config rather than of sweep timing: two embed models stay ambiguous even after
    // one of them is unloaded.
    let mut candidates = reg.entries.iter()
        .filter(|e| e.model.as_ref().map(|m| m.capabilities()).or(e.status.capability) == Some(cap))
        .map(|e| e.cfg.name.clone());
    match (candidates.next(), candidates.next()) {
        (Some(only), None) => Ok(only),
        (None, _) => Err(EngineError::Unsupported(format!("no {cap} model available"))),
        (Some(_), Some(_)) => Err(EngineError::Unsupported(format!("model required for {cap} (multiple configured)"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Config, Defaults, ModelCfg, ServerCfg};
    use crate::loader::mock::MockLoader;
    use crate::registry::Registry;
    use std::collections::BTreeMap;
    /// A registry with every named model loaded, plus a Config that lists them all.
    fn reg_with(models: &[(&str, Capability)]) -> (Config, Registry) {
        let mut t = BTreeMap::new();
        for (n, c) in models { t.insert((*n).to_string(), Ok((*c, 1))); }
        let l = MockLoader { table: t };
        let srv = ServerCfg { max_resident: 8, ..Default::default() };
        let mut r = Registry::default();
        let now = std::time::Instant::now();
        let mut c = Config::default();
        for (n, _) in models {
            let m = ModelCfg { name: (*n).into(), scenario: "x".into() };
            r.try_load(&m, &l, &srv, now);
            c.models.push(m);
        }
        (c, r)
    }
    #[test]
    fn explicit_known_and_unknown() {
        let (c, r) = reg_with(&[("bge", Capability::EMBED)]);
        assert_eq!(resolve(&c, &r, Capability::EMBED, Some("bge")).unwrap(), "bge");
        assert!(resolve(&c, &r, Capability::EMBED, Some("nope")).is_err());
        // ...and the error says WHY, instead of the misleading "not loaded".
        let e = resolve(&c, &r, Capability::EMBED, Some("nope")).unwrap_err().to_string();
        assert!(e.contains("not in the config"), "{e}");
    }
    #[test]
    fn explicit_name_resolves_while_unloaded_so_it_can_load_on_demand() {
        let (c, mut r) = reg_with(&[("bge", Capability::EMBED)]);
        r.release("bge", "idle");
        assert_eq!(resolve(&c, &r, Capability::EMBED, Some("bge")).unwrap(), "bge",
            "a swept-out model must still be routable -- the caller reloads it");
        // The remembered kind still rejects a capability mismatch without paying a load.
        assert!(matches!(resolve(&c, &r, Capability::ASR, Some("bge")),
            Err(EngineError::WrongKind { .. })));
    }
    #[test]
    fn configured_but_never_loaded_is_routable_by_name() {
        let (mut c, r) = reg_with(&[("bge", Capability::EMBED)]);
        c.models.push(ModelCfg { name: "fresh".into(), scenario: "y".into() });
        assert_eq!(resolve(&c, &r, Capability::ASR, Some("fresh")).unwrap(), "fresh",
            "kind is unknown until it loads; the post-load check decides");
    }
    #[test]
    fn ambiguity_survives_a_sweep() {
        let (c, mut r) = reg_with(&[("bge", Capability::EMBED), ("e5", Capability::EMBED)]);
        assert!(resolve(&c, &r, Capability::EMBED, None).is_err());
        r.release("bge", "idle");
        let e = resolve(&c, &r, Capability::EMBED, None).unwrap_err().to_string();
        assert!(e.contains("model required"), "unloading one of two must not silently pick the other: {e}");
    }
    #[test]
    fn default_resolves_after_release() {
        let (mut c, mut r) = reg_with(&[("asr", Capability::ASR)]);
        c.defaults = Defaults::from_pairs([(Capability::ASR, "asr".to_string())]);
        r.release("asr", "idle");
        assert_eq!(resolve(&c, &r, Capability::ASR, None).unwrap(), "asr",
            "the shipped path (no model named, one default) must survive an idle unload");
    }
    #[test]
    fn omit_uses_default_then_single_then_errors() {
        let (mut c, r) = reg_with(&[("bge", Capability::EMBED), ("e5", Capability::EMBED)]);
        assert!(resolve(&c, &r, Capability::EMBED, None).is_err()); // two, no default -> ambiguous
        c.defaults = Defaults::from_pairs([(Capability::EMBED, "e5".to_string())]);
        assert_eq!(resolve(&c, &r, Capability::EMBED, None).unwrap(), "e5");
    }
    #[test]
    fn omit_single_loaded_ok_without_default() {
        let (c, r) = reg_with(&[("only", Capability::ASR)]);
        assert_eq!(resolve(&c, &r, Capability::ASR, None).unwrap(), "only");
    }
}
