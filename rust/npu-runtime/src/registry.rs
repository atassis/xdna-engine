//! Actual state: which models are loaded, their status, and the memory accountant.
use crate::config::{ModelCfg, ServerCfg};
use crate::loader::{Inference, ModelLoader};
use npu_engine::ModelKind;

pub type Capability = ModelKind; // Asr -> transcription, Embed -> embeddings

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadState { Loaded, Failed, Unloaded }

#[derive(Debug, Clone, PartialEq)]
pub struct ModelStatus {
    pub name: String,
    pub state: LoadState,
    pub detail: String,
    pub kind: Option<ModelKind>,
    pub bo_bytes: u64,
}

pub struct Entry {
    pub cfg: ModelCfg,
    pub model: Option<Box<dyn Inference>>,
    pub status: ModelStatus,
}

#[derive(Default)]
pub struct Registry {
    pub entries: Vec<Entry>,
}

impl Registry {
    pub fn get_loaded(&self, name: &str) -> Option<&Box<dyn Inference>> {
        self.entries.iter().find(|e| e.cfg.name == name).and_then(|e| e.model.as_ref())
    }
    pub fn resident_bytes(&self) -> u64 {
        self.entries.iter().filter_map(|e| e.model.as_ref().map(|m| m.bo_bytes())).sum()
    }
    pub fn resident_count(&self) -> usize {
        self.entries.iter().filter(|e| e.model.is_some()).count()
    }
    pub fn status(&self) -> Vec<ModelStatus> {
        self.entries.iter().map(|e| e.status.clone()).collect()
    }

    /// Try to load one model under the budget, recording status. Never panics.
    pub fn try_load(&mut self, cfg: &ModelCfg, loader: &dyn ModelLoader, srv: &ServerCfg) {
        if self.resident_count() >= srv.max_resident {
            self.set_failed(cfg, format!("over max_resident ({})", srv.max_resident));
            return;
        }
        match loader.load(cfg) {
            Ok(m) => {
                let bo = m.bo_bytes();
                if self.resident_bytes() + bo > srv.memory_ceiling_mb * 1024 * 1024 {
                    self.set_failed(cfg, "over memory_ceiling".into());
                    return;
                }
                let status = ModelStatus {
                    name: cfg.name.clone(), state: LoadState::Loaded, detail: String::new(),
                    kind: Some(m.kind()), bo_bytes: bo,
                };
                self.upsert(Entry { cfg: cfg.clone(), model: Some(m), status });
            }
            Err(e) => self.set_failed(cfg, e.to_string()),
        }
    }
    pub fn unload(&mut self, name: &str) {
        self.entries.retain(|e| e.cfg.name != name);
    }

    fn set_failed(&mut self, cfg: &ModelCfg, detail: String) {
        let status = ModelStatus {
            name: cfg.name.clone(), state: LoadState::Failed, detail, kind: None, bo_bytes: 0,
        };
        self.upsert(Entry { cfg: cfg.clone(), model: None, status });
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

    #[test]
    fn load_failure_is_recorded_not_fatal() {
        let mut t = BTreeMap::new();
        t.insert("good".to_string(), Ok((ModelKind::Embed, 10)));
        t.insert("bad".to_string(), Err("boom".to_string()));
        let l = MockLoader { table: t };
        let srv = ServerCfg { max_resident: 8, ..Default::default() };
        let mut r = Registry::default();
        r.try_load(&cfg("good"), &l, &srv);
        r.try_load(&cfg("bad"), &l, &srv);
        assert_eq!(r.resident_count(), 1);
        let s = r.status();
        assert!(s.iter().any(|x| x.name == "good" && x.state == LoadState::Loaded));
        assert!(s.iter().any(|x| x.name == "bad" && x.state == LoadState::Failed && x.detail.contains("boom")));
    }
    #[test]
    fn max_resident_caps_loads() {
        let mut t = BTreeMap::new();
        t.insert("a".into(), Ok((ModelKind::Embed, 1)));
        t.insert("b".into(), Ok((ModelKind::Embed, 1)));
        let l = MockLoader { table: t };
        let srv = ServerCfg { max_resident: 1, ..Default::default() };
        let mut r = Registry::default();
        r.try_load(&cfg("a"), &l, &srv);
        r.try_load(&cfg("b"), &l, &srv);
        assert_eq!(r.resident_count(), 1);
        assert!(r.status().iter().any(|x| x.name == "b" && x.detail.contains("max_resident")));
    }
}
