//! Capability routing: omit -> configured default; named -> that one; ambiguous/unknown -> error.
use crate::config::Config;
use crate::registry::{Capability, Registry};
use npu_engine::{EngineError, ModelKind};

/// Resolve the model NAME to serve a capability. Caller then fetches it from the registry.
pub fn resolve(cfg: &Config, reg: &Registry, cap: Capability, want: Option<&str>)
    -> Result<String, EngineError> {
    if let Some(n) = want {
        return match reg.get_loaded(n) {
            Some(m) if m.kind() == cap => Ok(n.to_string()),
            Some(m) => Err(EngineError::WrongKind { wanted: cap, got: m.kind() }),
            None => Err(EngineError::Load(format!("model {n:?} not loaded"))),
        };
    }
    let default = match cap { ModelKind::Asr => &cfg.defaults.asr, ModelKind::Embed => &cfg.defaults.embed };
    if let Some(n) = default {
        if reg.get_loaded(n).map(|m| m.kind()) == Some(cap) { return Ok(n.clone()); }
    }
    let mut loaded_of_cap = reg.entries.iter()
        .filter(|e| e.model.as_ref().map(|m| m.kind()) == Some(cap))
        .map(|e| e.cfg.name.clone());
    match (loaded_of_cap.next(), loaded_of_cap.next()) {
        (Some(only), None) => Ok(only),
        (None, _) => Err(EngineError::Unsupported(format!("no {cap} model loaded"))),
        (Some(_), Some(_)) => Err(EngineError::Unsupported(format!("model required for {cap} (multiple loaded)"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Config, Defaults, ModelCfg, ServerCfg};
    use crate::loader::mock::MockLoader;
    use crate::registry::Registry;
    use std::collections::BTreeMap;
    fn reg_with(models: &[(&str, ModelKind)]) -> (Config, Registry) {
        let mut t = BTreeMap::new();
        for (n, k) in models { t.insert((*n).to_string(), Ok((*k, 1))); }
        let l = MockLoader { table: t };
        let srv = ServerCfg { max_resident: 8, ..Default::default() };
        let mut r = Registry::default();
        for (n, _) in models {
            r.try_load(&ModelCfg { name: (*n).into(), scenario: "x".into() }, &l, &srv);
        }
        (Config::default(), r)
    }
    #[test]
    fn explicit_known_and_unknown() {
        let (c, r) = reg_with(&[("bge", ModelKind::Embed)]);
        assert_eq!(resolve(&c, &r, ModelKind::Embed, Some("bge")).unwrap(), "bge");
        assert!(resolve(&c, &r, ModelKind::Embed, Some("nope")).is_err());
    }
    #[test]
    fn omit_uses_default_then_single_then_errors() {
        let (mut c, r) = reg_with(&[("bge", ModelKind::Embed), ("e5", ModelKind::Embed)]);
        assert!(resolve(&c, &r, ModelKind::Embed, None).is_err()); // two, no default -> ambiguous
        c.defaults = Defaults { asr: None, embed: Some("e5".into()) };
        assert_eq!(resolve(&c, &r, ModelKind::Embed, None).unwrap(), "e5");
    }
    #[test]
    fn omit_single_loaded_ok_without_default() {
        let (c, r) = reg_with(&[("only", ModelKind::Asr)]);
        assert_eq!(resolve(&c, &r, ModelKind::Asr, None).unwrap(), "only");
    }
}
