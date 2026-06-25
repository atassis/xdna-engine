//! Desired state: the persisted engine config. The file IS the persistence (restart-survival is
//! automatic). Atomic save (temp + rename).
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Config {
    #[serde(default)] pub server: ServerCfg,
    #[serde(default)] pub defaults: Defaults,
    #[serde(default, rename = "model")] pub models: Vec<ModelCfg>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ServerCfg { pub port: u16, pub memory_ceiling_mb: u64, pub max_resident: usize }
impl Default for ServerCfg {
    fn default() -> Self { ServerCfg { port: 11434, memory_ceiling_mb: 4096, max_resident: 1 } }
}
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Defaults {
    #[serde(default)] pub asr: Option<String>,
    #[serde(default)] pub embed: Option<String>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelCfg { pub name: String, pub scenario: String }

impl Config {
    pub fn from_str(s: &str) -> Result<Config, toml::de::Error> { toml::from_str(s) }
    /// Load from path; a MISSING file yields the default empty config (resilient startup).
    pub fn load(path: &Path) -> Result<Config, String> {
        match std::fs::read_to_string(path) {
            Ok(s) => Config::from_str(&s).map_err(|e| format!("{}: {e}", path.display())),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Config::default()),
            Err(e) => Err(format!("{}: {e}", path.display())),
        }
    }
    /// Atomic save: write a temp file beside the target, then rename.
    pub fn save(&self, path: &Path) -> Result<(), String> {
        if let Some(dir) = path.parent() { let _ = std::fs::create_dir_all(dir); }
        let tmp = path.with_extension("toml.tmp");
        let s = toml::to_string_pretty(self).map_err(|e| e.to_string())?;
        std::fs::write(&tmp, s).map_err(|e| e.to_string())?;
        std::fs::rename(&tmp, path).map_err(|e| e.to_string())
    }
    pub fn find(&self, name: &str) -> Option<&ModelCfg> { self.models.iter().find(|m| m.name == name) }
}
impl Default for Config {
    fn default() -> Self {
        Config { server: ServerCfg::default(), defaults: Defaults::default(), models: vec![] }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn roundtrip_and_defaults() {
        let toml = r#"
[server]
port = 11434
memory_ceiling_mb = 4096
max_resident = 1
[defaults]
asr = "parakeet"
[[model]]
name = "parakeet"
scenario = "scenarios/asr.toml"
"#;
        let c = Config::from_str(toml).unwrap();
        assert_eq!(c.server.port, 11434);
        assert_eq!(c.defaults.asr.as_deref(), Some("parakeet"));
        assert_eq!(c.find("parakeet").unwrap().scenario, "scenarios/asr.toml");
        // missing file -> default empty
        let missing = Config::load(Path::new("/nope/x.toml")).unwrap();
        assert!(missing.models.is_empty());
        assert_eq!(missing.server.max_resident, 1);
    }
    #[test]
    fn save_then_load_is_identity() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("engine.toml");
        let c = Config {
            defaults: Defaults { asr: Some("a".into()), embed: None },
            models: vec![ModelCfg { name: "a".into(), scenario: "s.toml".into() }],
            ..Default::default()
        };
        c.save(&p).unwrap();
        assert_eq!(Config::load(&p).unwrap(), c);
    }
}
