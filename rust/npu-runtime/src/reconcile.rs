//! Bring the registry (actual) in line with the config (desired): load missing, unload removed,
//! reload changed. Each step independent; failures recorded as Failed, never fatal.
use crate::config::Config;
use crate::loader::ModelLoader;
use crate::registry::Registry;

#[derive(Debug, Default, PartialEq)]
pub struct ReconcileReport {
    pub loaded: Vec<String>,
    pub unloaded: Vec<String>,
    pub failed: Vec<String>,
}

pub fn reconcile(cfg: &Config, reg: &mut Registry, loader: &dyn ModelLoader) -> ReconcileReport {
    let mut rep = ReconcileReport::default();
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
        if !needs { continue; }
        reg.unload(&m.name);
        reg.try_load(m, loader, &cfg.server);
        if reg.get_loaded(&m.name).is_some() { rep.loaded.push(m.name.clone()); }
        else { rep.failed.push(m.name.clone()); }
    }
    rep
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Config, ModelCfg, ServerCfg};
    use crate::loader::mock::MockLoader;
    use npu_engine::ModelKind;
    use std::collections::BTreeMap;
    fn loader(names: &[(&str, bool)]) -> MockLoader {
        let mut t = BTreeMap::new();
        for (n, ok) in names {
            t.insert((*n).to_string(), if *ok { Ok((ModelKind::Embed, 1)) } else { Err("fail".into()) });
        }
        MockLoader { table: t }
    }
    fn cfg(names: &[&str]) -> Config {
        Config {
            server: ServerCfg { max_resident: 8, ..Default::default() },
            models: names.iter().map(|n| ModelCfg { name: (*n).into(), scenario: "x".into() }).collect(),
            ..Default::default()
        }
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
}
