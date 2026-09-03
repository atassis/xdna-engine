//! Desired state: the persisted engine config. The file IS the persistence (restart-survival is
//! automatic). Atomic save (temp + rename).
use npu_engine::capability::Capability;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::Duration;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Config {
    #[serde(default)] pub server: ServerCfg,
    #[serde(default)] pub defaults: Defaults,
    #[serde(default, rename = "model")] pub models: Vec<ModelCfg>,
}

/// Every field carries its own `#[serde(default)]`. Without them a `[server]` table that omits any
/// key failed the whole parse, and toml reported it as a span over `[server]` -- which reads like a
/// syntax error in the table rather than "you left out max_resident".
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ServerCfg {
    #[serde(default = "default_port")] pub port: u16,
    /// Ceiling on the summed DEVICE-BO bytes of resident models, from `Servable::footprint`.
    ///
    /// It does NOT bound host RSS, which is the memory that actually took this service down: that
    /// failure was the onnxruntime arena sizing itself for the diarization embedder's batch
    /// (measured 1519 MB at batch 32 against 568 MB at 8), and no device-BO accountant would ever
    /// have seen it. `NPU_DIARIZE_MEM_MB` is the knob for that one; `idle_release_s` is what gives
    /// host pages back.
    ///
    /// INERT TODAY: every shipped model reports `footprint() == 0`, so the sum is always 0 and the
    /// check never fires. A model that loads without a measured footprint says so in its status
    /// detail rather than passing silently -- an unenforceable bound that looks enforced is the
    /// failure this note exists to prevent.
    #[serde(default = "default_memory_ceiling_mb")] pub memory_ceiling_mb: u64,
    /// How many models may be resident at once. Not a refusal: at the cap, a request for another
    /// model evicts per `evict_policy` (see `Registry::ensure_resident`).
    #[serde(default = "default_max_resident")] pub max_resident: usize,
    /// Unload a model that has not served a request for this long, releasing the device. `0`
    /// disables idle unload entirely.
    #[serde(default = "default_idle_unload_s")] pub idle_unload_s: u64,
    /// How often the device actor looks for expired models. It sweeps only *between* commands, so
    /// this is also the worst-case delay before an idle model is released.
    #[serde(default = "default_sweep_interval_s")] pub sweep_interval_s: u64,
    /// The SECOND level of idleness, and a deeper one. `idle_unload_s` gives back the device; this
    /// gives back the memory the unload freed but the allocator kept (~2.5 GB after a parakeet
    /// unload, measured). Counts from the last REQUEST -- a `/healthz` or `/v1/models` poll must not
    /// be able to keep the process fat forever. `0` disables it.
    #[serde(default = "default_idle_release_s")] pub idle_release_s: u64,
    /// What to drop when a load needs a slot and `max_resident` is already full.
    #[serde(default)] pub evict_policy: EvictPolicy,
}
fn default_port() -> u16 { 11434 }
fn default_memory_ceiling_mb() -> u64 { 4096 }
fn default_max_resident() -> usize { 1 }
fn default_idle_unload_s() -> u64 { 900 }
fn default_sweep_interval_s() -> u64 { 30 }
fn default_idle_release_s() -> u64 { 1800 }
impl Default for ServerCfg {
    fn default() -> Self {
        ServerCfg {
            port: default_port(),
            memory_ceiling_mb: default_memory_ceiling_mb(),
            max_resident: default_max_resident(),
            idle_unload_s: default_idle_unload_s(),
            sweep_interval_s: default_sweep_interval_s(),
            idle_release_s: default_idle_release_s(),
            evict_policy: EvictPolicy::default(),
        }
    }
}
impl ServerCfg {
    /// The idle window, or `None` when idle unload is switched off.
    pub fn idle_unload(&self) -> Option<Duration> {
        if self.idle_unload_s == 0 { None } else { Some(Duration::from_secs(self.idle_unload_s)) }
    }
    /// How long the actor waits for a command before it sweeps. Clamped to >= 1s: a
    /// `sweep_interval_s = 0` config would otherwise spin the device thread.
    pub fn sweep_interval(&self) -> Duration { Duration::from_secs(self.sweep_interval_s.max(1)) }
    /// The deep-release window, or `None` when that second level is switched off.
    pub fn idle_release(&self) -> Option<Duration> {
        if self.idle_release_s == 0 { None } else { Some(Duration::from_secs(self.idle_release_s)) }
    }
}

/// Which resident model gives up its slot when another one has to load.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EvictPolicy {
    /// Drop the least-recently-used resident model.
    #[default] Lru,
    /// Never evict: a load that would exceed `max_resident` is refused. This is the behaviour from
    /// before hot-swap existed, kept as an opt-out for a box that must not pay reload latency.
    None,
}
/// Which model serves a capability when a request does not name one, keyed by capability name.
///
/// A map, not the `{ asr, embed }` struct it replaces: that struct was the config half of the closed
/// request surface -- `[defaults] tts = "kokoro"` was not a missing feature but an unrepresentable
/// one. `#[serde(transparent)]` keeps every shipped `engine.toml` parsing unchanged, because
/// `[defaults] asr = "parakeet"` is already exactly this map's TOML form.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Defaults(pub std::collections::BTreeMap<String, String>);

impl Defaults {
    pub fn get(&self, cap: Capability) -> Option<&String> { self.0.get(cap.0) }
    pub fn set(&mut self, cap: Capability, model: String) { self.0.insert(cap.0.to_string(), model); }
    /// Build from `(capability, model)` pairs -- the shape tests and `npu init` want.
    pub fn from_pairs<I: IntoIterator<Item = (Capability, String)>>(it: I) -> Defaults {
        Defaults(it.into_iter().map(|(c, m)| (c.0.to_string(), m)).collect())
    }
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
        assert_eq!(c.defaults.get(Capability::ASR).map(String::as_str), Some("parakeet"));
        assert_eq!(c.find("parakeet").unwrap().scenario, "scenarios/asr.toml");
        // missing file -> default empty
        let missing = Config::load(Path::new("/nope/x.toml")).unwrap();
        assert!(missing.models.is_empty());
        assert_eq!(missing.server.max_resident, 1);
    }
    #[test]
    fn partial_server_table_uses_field_defaults() {
        // A `[server]` table naming only one key must parse. Before the per-field serde defaults
        // this was a hard error whose span pointed at `[server]`, reading like a syntax error.
        let c = Config::from_str("[server]\nport = 9999\n").unwrap();
        assert_eq!(c.server.port, 9999);
        assert_eq!(c.server.max_resident, 1);
        assert_eq!(c.server.memory_ceiling_mb, 4096);
        assert_eq!(c.server.idle_unload_s, 900);
        assert_eq!(c.server.sweep_interval_s, 30);
        assert_eq!(c.server.evict_policy, EvictPolicy::Lru);
        // ...and an empty file is the full default config.
        assert_eq!(Config::from_str("").unwrap(), Config::default());
    }
    #[test]
    fn idle_and_sweep_knobs() {
        let c = Config::from_str("[server]\nidle_unload_s = 0\nsweep_interval_s = 0\nevict_policy = \"none\"\n").unwrap();
        assert_eq!(c.server.idle_unload(), None, "0 disables idle unload");
        assert_eq!(c.server.sweep_interval(), std::time::Duration::from_secs(1), "0 clamps to 1s, never spins");
        assert_eq!(c.server.evict_policy, EvictPolicy::None);
        let d = ServerCfg::default();
        assert_eq!(d.idle_unload(), Some(std::time::Duration::from_secs(900)));
        assert_eq!(d.sweep_interval(), std::time::Duration::from_secs(30));
    }
    #[test]
    fn save_then_load_is_identity() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("engine.toml");
        let c = Config {
            defaults: Defaults::from_pairs([(Capability::ASR, "a".to_string())]),
            models: vec![ModelCfg { name: "a".into(), scenario: "s.toml".into() }],
            ..Default::default()
        };
        c.save(&p).unwrap();
        assert_eq!(Config::load(&p).unwrap(), c);
    }

    /// A capability with no `ModelKind` variant must survive a config round-trip -- the whole point
    /// of the map. `[defaults] tts = ...` was previously dropped silently on save.
    #[test]
    fn defaults_carry_a_capability_the_old_struct_could_not_name() {
        let c = Config::from_str("[defaults]\nasr = \"parakeet\"\ntts = \"kokoro\"\n").unwrap();
        assert_eq!(c.defaults.get(Capability::TTS).map(String::as_str), Some("kokoro"));
        assert_eq!(c.defaults.get(Capability::EMBED), None);
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("engine.toml");
        c.save(&p).unwrap();
        assert_eq!(Config::load(&p).unwrap(), c);
    }
}
