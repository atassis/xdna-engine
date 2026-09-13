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
/// syntax error in the table rather than naming the missing key.
///
/// `deny_unknown_fields`: closed on purpose, same reason `config_doc::SERVER_KEYS` is a closed list
/// for `npu config set` -- an unrecognised key must fail loud, not parse fine and silently do
/// nothing. This is also the migration path for the retired `max_resident`: a config still carrying
/// it now fails to parse with a message naming the field, rather than being quietly ignored.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServerCfg {
    #[serde(default = "default_port")] pub port: u16,
    /// Ceiling on the summed DEVICE-BO bytes of resident models, from `Servable::footprint`, and the
    /// ONE capacity knob: it gates admission, drives eviction, and bounds the sum of pinned models'
    /// bytes (see `Registry`/`reconcile`). Real for Parakeet, Whisper, the embedders and Generate;
    /// `0` for a model kind nobody has wired yet, which `Registry::unweighed_residents` reports
    /// rather than letting the ceiling silently not apply to it.
    ///
    /// It does NOT bound host RSS, which is the memory that actually took this service down once:
    /// that failure was the onnxruntime arena sizing itself for the diarization embedder's batch
    /// (measured 1519 MB at batch 32 against 568 MB at 8), and no device-BO accountant would ever
    /// have seen it. `NPU_DIARIZE_MEM_MB` is the knob for that one; `idle_release_s` is what gives
    /// host pages back.
    #[serde(default = "default_memory_ceiling_mb")] pub memory_ceiling_mb: u64,
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
    /// What to drop when a load needs room and `memory_ceiling_mb` is already spent.
    #[serde(default)] pub evict_policy: EvictPolicy,
}
fn default_port() -> u16 { 11434 }
fn default_memory_ceiling_mb() -> u64 { 4096 }
fn default_idle_unload_s() -> u64 { 900 }
fn default_sweep_interval_s() -> u64 { 30 }
fn default_idle_release_s() -> u64 { 1800 }
impl Default for ServerCfg {
    fn default() -> Self {
        ServerCfg {
            port: default_port(),
            memory_ceiling_mb: default_memory_ceiling_mb(),
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
    /// Never evict: a load that would exceed `memory_ceiling_mb` is refused. This is the behaviour
    /// from before hot-swap existed, kept as an opt-out for a box that must not pay reload latency.
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
pub struct ModelCfg {
    pub name: String,
    pub scenario: String,
    /// Pin this model: always on. Eagerly admitted (ahead of any unpinned model, see `reconcile`),
    /// exempt from idle unload, never chosen as an eviction victim -- as long as the invariant in
    /// `Config::pin_overcommit` holds. A pin that does not fit is refused, not silently granted: the
    /// registry tracks whether a declared pin is currently HONOURED separately from this raw config
    /// value (`Entry::pin_honored`), because "declared pinned" and "actually protected right now"
    /// are not the same fact once the invariant can be violated by something other than this field
    /// changing (the budget being lowered, or a live model's own footprint growing).
    #[serde(default)] pub resident: bool,
}

impl Config {
    pub fn from_str(s: &str) -> Result<Config, toml::de::Error> { toml::from_str(s) }
    /// Load from path; a MISSING file yields the default empty config (resilient startup).
    pub fn load(path: &Path) -> Result<Config, String> {
        match std::fs::read_to_string(path) {
            Ok(s) => Config::from_str(&s).map_err(|e| {
                let e = format!("{}: {e}", path.display());
                // `deny_unknown_fields` already fails loud on a retired `max_resident` key -- this
                // just turns "unknown field `max_resident`" into an actionable next step, the same
                // spirit as SERVER_KEYS' closed list, rather than leaving the operator to guess one.
                if e.contains("max_resident") {
                    format!("{e}\nmax_resident was removed: memory is bounded by memory_ceiling_mb \
                             alone now. Delete the max_resident line from [server].")
                } else {
                    e
                }
            }),
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
    /// Pinned models, in config order.
    pub fn pinned(&self) -> impl Iterator<Item = &ModelCfg> { self.models.iter().filter(|m| m.resident) }
    /// `Some(message)` when the pinned set's total estimated bytes exceed `memory_ceiling_mb`.
    ///
    /// Takes an estimator rather than owning device state: the same check answers "does the config
    /// on disk make sense" (an offline, declared-footprint estimator, works with the service down)
    /// and "does the invariant hold right now" (a live, registry-backed one) without two
    /// implementations of the arithmetic. Advisory in the sense that this reports the violation --
    /// `reconcile` is what refuses to admit the model(s) that push the sum over, and demotes
    /// anything already loaded that no longer fits (see `Entry::pin_honored`).
    pub fn pin_overcommit(&self, footprint: impl Fn(&ModelCfg) -> u64) -> Option<String> {
        let pinned: Vec<&ModelCfg> = self.pinned().collect();
        let total: u64 = pinned.iter().map(|m| footprint(m)).sum();
        let ceiling = self.server.memory_ceiling_mb * 1024 * 1024;
        (total > ceiling).then(|| format!(
            "{} model(s) pinned resident total ~{} MB, over memory_ceiling_mb = {} MB",
            pinned.len(), total / (1024 * 1024), self.server.memory_ceiling_mb))
    }
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
    fn resident_defaults_false_and_parses_from_toml() {
        let c = Config::from_str(
            "[[model]]\nname = \"a\"\nscenario = \"x\"\n\n[[model]]\nname = \"b\"\nscenario = \"y\"\nresident = true\n"
        ).unwrap();
        assert!(!c.find("a").unwrap().resident, "absent key must stay false, so old configs are unchanged");
        assert!(c.find("b").unwrap().resident);
        assert_eq!(c.pinned().map(|m| m.name.as_str()).collect::<Vec<_>>(), vec!["b"]);
    }
    /// The estimator every `pin_overcommit` test uses: 100 MB per pinned model, so the arithmetic in
    /// each assertion is easy to check by hand.
    fn hundred_mb(_m: &ModelCfg) -> u64 { 100 * 1024 * 1024 }

    #[test]
    fn pin_overcommit_fires_only_when_the_pinned_sum_exceeds_the_ceiling() {
        let mk = |n: usize, ceiling_mb: u64| {
            let mut c = Config::default();
            c.server.memory_ceiling_mb = ceiling_mb;
            c.models = (0..n).map(|i| ModelCfg {
                name: format!("m{i}"), scenario: "x".into(), resident: true }).collect();
            c
        };
        assert!(mk(1, 500).pin_overcommit(hundred_mb).is_none(), "100 MB pinned under a 500 MB ceiling");
        assert!(mk(4, 500).pin_overcommit(hundred_mb).is_none(), "400 MB pinned still fits 500 MB");
        assert!(mk(5, 500).pin_overcommit(hundred_mb).is_none(), "exactly at the ceiling is not OVER it");
        let w = mk(6, 500).pin_overcommit(hundred_mb).expect("600 MB pinned over a 500 MB ceiling");
        assert!(w.contains("600 MB") && w.contains("memory_ceiling_mb = 500 MB"), "{w}");
    }
    #[test]
    fn roundtrip_and_defaults() {
        let toml = r#"
[server]
port = 11434
memory_ceiling_mb = 4096
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
        assert_eq!(missing.server.memory_ceiling_mb, 4096);
    }
    #[test]
    fn partial_server_table_uses_field_defaults() {
        // A `[server]` table naming only one key must parse. Before the per-field serde defaults
        // this was a hard error whose span pointed at `[server]`, reading like a syntax error.
        let c = Config::from_str("[server]\nport = 9999\n").unwrap();
        assert_eq!(c.server.port, 9999);
        assert_eq!(c.server.memory_ceiling_mb, 4096);
        assert_eq!(c.server.idle_unload_s, 900);
        assert_eq!(c.server.sweep_interval_s, 30);
        assert_eq!(c.server.evict_policy, EvictPolicy::Lru);
        // ...and an empty file is the full default config.
        assert_eq!(Config::from_str("").unwrap(), Config::default());
    }
    /// The migration path: `deny_unknown_fields` makes a retired key a parse error rather than a
    /// silent no-op, and `Config::load`'s wrapper turns that into an actionable message.
    #[test]
    fn a_config_still_carrying_max_resident_fails_loud_and_names_the_fix() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("engine.toml");
        std::fs::write(&p, "[server]\nmax_resident = 2\n").unwrap();
        let e = Config::load(&p).unwrap_err();
        assert!(e.contains("max_resident"), "{e}");
        assert!(e.contains("memory_ceiling_mb"), "the message must name the replacement: {e}");
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
            models: vec![ModelCfg { name: "a".into(), scenario: "s.toml".into(), resident: false }],
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
