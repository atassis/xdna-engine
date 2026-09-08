//! Format-preserving edits to `engine.toml`.
//!
//! [`Config`] is the READ model: deserialize, reason, done. It is the wrong thing to write back,
//! because `toml::to_string_pretty(&Config)` reconstructs the file from the struct and everything
//! the struct does not carry is destroyed -- every comment, the key order, the blank lines. The
//! engine's own generated config ships with explanatory comments, so the old writer was deleting
//! its own output; the values survived, which is exactly why nobody noticed.
//!
//! So mutations go through the DOCUMENT, not the struct: load `engine.toml` as a `DocumentMut`,
//! change the one key that was asked for, write it back. Everything not named is untouched by
//! construction rather than by care.
//!
//! Every edit is re-parsed into a `Config` before it is written ([`ConfigDoc::validate`]), so a
//! command cannot leave a file the service will refuse at boot.

use crate::config::{Config, EvictPolicy};
use npu_engine::capability::Capability;
use std::path::Path;
use toml_edit::{value, ArrayOfTables, DocumentMut, Item, Table};

/// What a `[server]` key accepts. The CLI's completion values, its help text and this validator all
/// read [`SERVER_KEYS`], so a knob cannot be settable without being documented, or documented
/// without being settable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyType {
    /// A non-negative integer. `0` is meaningful for the idle windows (it disables them).
    Int,
    /// One of a fixed set of words.
    Enum(&'static [&'static str]),
}

/// The `[server]` keys `npu config set` accepts, in the order `npu config show` prints them.
///
/// This list is deliberately CLOSED. Writing an unrecognised key would produce a file that parses
/// (nothing in `Config` is `deny_unknown_fields`) and silently does nothing -- a knob that looks
/// set and is not, which is the failure mode a typo in a config should never have.
pub const SERVER_KEYS: &[(&str, KeyType)] = &[
    ("port", KeyType::Int),
    ("memory_ceiling_mb", KeyType::Int),
    ("max_resident", KeyType::Int),
    ("idle_unload_s", KeyType::Int),
    ("sweep_interval_s", KeyType::Int),
    ("idle_release_s", KeyType::Int),
    ("evict_policy", KeyType::Enum(&["lru", "none"])),
];

/// One-line description per settable key, for `npu config set --help`. Separate from `SERVER_KEYS`
/// only so the type table stays readable; the test below pins the two to the same key set.
pub fn server_key_help() -> Vec<(&'static str, &'static str)> {
    vec![
        ("port", "TCP port the service binds"),
        ("memory_ceiling_mb", "ceiling on summed device-BO bytes (inert while footprints read 0)"),
        ("max_resident", "how many models may hold the device at once"),
        ("idle_unload_s", "unload a model idle this long; 0 disables idle unload"),
        ("sweep_interval_s", "how often the actor looks for expired models (clamped to >= 1)"),
        ("idle_release_s", "return the allocator's free pages this long after the last request; 0 disables"),
        ("evict_policy", "lru | none -- what gives up a slot when max_resident is full"),
    ]
}

/// [`server_key_help`] rendered for `npu config set --help`.
pub fn server_key_help_text() -> String {
    let mut out = String::from("Settable [server] keys:\n");
    for (k, h) in server_key_help() { out.push_str(&format!("  {k:<19}{h}\n")); }
    out.push_str("\nTakes effect on `npu reload` (or a service restart).");
    out
}

pub struct ConfigDoc {
    doc: DocumentMut,
}

impl ConfigDoc {
    /// Load the document. A MISSING file is an empty document, matching `Config::load`: the first
    /// `npu config` command on a fresh box must create the file, not fail on it.
    pub fn load(path: &Path) -> Result<ConfigDoc, String> {
        let text = match std::fs::read_to_string(path) {
            Ok(s) => s,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
            Err(e) => return Err(format!("{}: {e}", path.display())),
        };
        Ok(ConfigDoc { doc: text.parse::<DocumentMut>().map_err(|e| format!("{}: {e}", path.display()))? })
    }

    /// Parse the current document as a `Config`. Called before every write, so an edit that would
    /// produce a file the service refuses fails in the CLI instead of at the next boot.
    pub fn validate(&self) -> Result<Config, String> {
        Config::from_str(&self.doc.to_string()).map_err(|e| e.to_string())
    }

    /// Validate, then write atomically (temp beside the target, then rename).
    pub fn save(&self, path: &Path) -> Result<Config, String> {
        let cfg = self.validate()?;
        if let Some(dir) = path.parent() { let _ = std::fs::create_dir_all(dir); }
        let tmp = path.with_extension("toml.tmp");
        std::fs::write(&tmp, self.doc.to_string()).map_err(|e| e.to_string())?;
        std::fs::rename(&tmp, path).map_err(|e| e.to_string())?;
        Ok(cfg)
    }

    pub fn to_toml_string(&self) -> String { self.doc.to_string() }

    /// The `[[model]]` array, created empty if the file has none.
    ///
    /// `Err` when `model` exists as something other than an array of tables (`model = [...]`
    /// inline, or a plain `[model]` table). Refusing beats "fixing" it: rewriting a shape the
    /// operator chose is the same silent destruction this module exists to stop.
    fn models_mut(&mut self) -> Result<&mut ArrayOfTables, String> {
        if self.doc.get("model").is_none() {
            self.doc.insert("model", Item::ArrayOfTables(ArrayOfTables::new()));
        }
        self.doc["model"].as_array_of_tables_mut()
            .ok_or_else(|| "`model` is not an array of tables ([[model]]); edit the file by hand".to_string())
    }

    fn model_mut(&mut self, name: &str) -> Result<Option<&mut Table>, String> {
        Ok(self.models_mut()?.iter_mut()
            .find(|t| t.get("name").and_then(|v| v.as_str()) == Some(name)))
    }

    /// Add the model, or update the scenario of one already there.
    ///
    /// Updating IN PLACE is the point: the old writer dropped the entry and pushed a fresh one with
    /// `resident: false`, so re-running `add-model` to correct a scenario path silently unpinned
    /// the model. Here every key the operator set but did not name survives.
    pub fn add_model(&mut self, name: &str, scenario: &str) -> Result<(), String> {
        if let Some(t) = self.model_mut(name)? {
            t["scenario"] = value(scenario);
            return Ok(());
        }
        let mut t = Table::new();
        t["name"] = value(name);
        t["scenario"] = value(scenario);
        self.models_mut()?.push(t);
        Ok(())
    }

    /// Remove every `[[model]]` entry with this name. `false` when there was none.
    pub fn remove_model(&mut self, name: &str) -> Result<bool, String> {
        let models = self.models_mut()?;
        let before = models.len();
        models.retain(|t| t.get("name").and_then(|v| v.as_str()) != Some(name));
        Ok(models.len() != before)
    }

    /// Pin (`resident = true`) or unpin (`resident = false`) a model. `false` when the model is not
    /// in the config.
    ///
    /// Unpinning WRITES the default back rather than removing the key, which is not the tidier of
    /// the two and is the safer one. toml_edit attaches a key's preceding comment lines to the key,
    /// so `remove` takes the operator's note with it -- measured on a real config, where unpinning
    /// deleted `# Pinned: the default generate model, so a quiet spell does not cost the next
    /// request a reload.` A stale comment is visible and a human can fix it; a deleted one is gone.
    /// This module's whole premise is that the writer does not destroy prose it did not write, and
    /// "the key is being removed anyway" is not an exception to that -- the comment may say why the
    /// operator wanted the pin, which is exactly what they need when deciding to put it back.
    pub fn set_resident(&mut self, name: &str, on: bool) -> Result<bool, String> {
        match self.model_mut(name)? {
            Some(t) => { t["resident"] = value(on); Ok(true) }
            None => Ok(false),
        }
    }

    pub fn set_default(&mut self, cap: Capability, model: &str) {
        self.table_mut("defaults")[cap.0] = value(model);
    }

    /// Set one `[server]` key, validating the name against [`SERVER_KEYS`] and the value against
    /// that key's type.
    pub fn set_server(&mut self, key: &str, raw: &str) -> Result<(), String> {
        let ty = SERVER_KEYS.iter().find(|(k, _)| *k == key).map(|(_, t)| *t).ok_or_else(|| {
            format!("unknown server key {key:?} (one of: {})",
                SERVER_KEYS.iter().map(|(k, _)| *k).collect::<Vec<_>>().join(", "))
        })?;
        let v = match ty {
            KeyType::Int => {
                let n: i64 = raw.parse()
                    .map_err(|_| format!("{key} takes a non-negative integer, not {raw:?}"))?;
                if n < 0 { return Err(format!("{key} takes a non-negative integer, not {raw:?}")) }
                value(n)
            }
            KeyType::Enum(choices) => {
                if !choices.contains(&raw) {
                    return Err(format!("{key} takes one of: {}", choices.join(" | ")));
                }
                value(raw)
            }
        };
        self.table_mut("server")[key] = v;
        Ok(())
    }

    /// A top-level table, created (non-inline, so it renders as `[name]`) if absent.
    fn table_mut(&mut self, name: &str) -> &mut Item {
        if self.doc.get(name).is_none() {
            let mut t = Table::new();
            t.set_implicit(false);
            self.doc.insert(name, Item::Table(t));
        }
        &mut self.doc[name]
    }
}

/// `EvictPolicy` as the word `npu config set` and the TOML both use.
pub fn evict_policy_str(p: EvictPolicy) -> &'static str {
    match p { EvictPolicy::Lru => "lru", EvictPolicy::None => "none" }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The whole reason this module exists. A commented config must come back with its comments.
    const COMMENTED: &str = r#"# top of file
[server]
# 2, not 1: an alternating workload pays a full reload every request otherwise.
max_resident = 5
idle_unload_s = 900

[defaults]
asr = "parakeet"

# Qwen3-0.6B, fully resident: one fused xclbin, one hardware context.
[[model]]
name = "qwen3"
scenario = "scenarios/generate-qwen3.toml"
# Pinned: the default generate model.
resident = true

[[model]]
name = "parakeet"
scenario = "scenarios/asr.toml"
"#;

    fn doc(s: &str) -> ConfigDoc { ConfigDoc { doc: s.parse().unwrap() } }

    #[test]
    fn every_edit_preserves_every_comment() {
        for (label, edit) in [
            ("pin", Box::new(|d: &mut ConfigDoc| { d.set_resident("parakeet", true).unwrap(); })
                as Box<dyn Fn(&mut ConfigDoc)>),
            ("unpin", Box::new(|d: &mut ConfigDoc| { d.set_resident("qwen3", false).unwrap(); })),
            ("add-model", Box::new(|d: &mut ConfigDoc| { d.add_model("bge", "scenarios/bge.toml").unwrap(); })),
            ("remove-model", Box::new(|d: &mut ConfigDoc| { d.remove_model("parakeet").unwrap(); })),
            ("set-default", Box::new(|d: &mut ConfigDoc| { d.set_default(Capability::ASR, "whisper"); })),
            ("set", Box::new(|d: &mut ConfigDoc| { d.set_server("idle_unload_s", "0").unwrap(); })),
        ] {
            let mut d = doc(COMMENTED);
            edit(&mut d);
            let out = d.to_toml_string();
            for c in ["# top of file", "# 2, not 1:", "# Qwen3-0.6B, fully resident"] {
                assert!(out.contains(c), "{label} destroyed {c:?}:\n{out}");
            }
            d.validate().unwrap_or_else(|e| panic!("{label} produced an unparseable config: {e}"));
        }
    }

    /// The bug the in-place update exists to stop: re-running `add-model` on a pinned model.
    #[test]
    fn add_model_on_an_existing_name_keeps_the_pin_and_updates_the_scenario() {
        let mut d = doc(COMMENTED);
        d.add_model("qwen3", "scenarios/generate-qwen3-v2.toml").unwrap();
        let cfg = d.validate().unwrap();
        let m = cfg.find("qwen3").unwrap();
        assert_eq!(m.scenario, "scenarios/generate-qwen3-v2.toml");
        assert!(m.resident, "add-model must not silently unpin a model it is only re-pointing");
        assert_eq!(cfg.models.len(), 2, "re-adding must update in place, not duplicate");
        assert!(d.to_toml_string().contains("# Pinned: the default generate model."),
            "the comment on the key being changed must survive too");
    }

    #[test]
    fn pin_and_unpin_round_trip_through_the_parsed_config() {
        let mut d = doc(COMMENTED);
        assert!(d.set_resident("parakeet", true).unwrap());
        assert!(d.validate().unwrap().find("parakeet").unwrap().resident);
        assert!(d.set_resident("parakeet", false).unwrap());
        assert!(!d.validate().unwrap().find("parakeet").unwrap().resident);
        assert!(!d.set_resident("nope", true).unwrap(), "an unknown model reports, never invents");
    }

    /// Unpinning must not take the operator's note with the key. Measured on a real config: the
    /// `remove` this replaces deleted the comment explaining WHY the model was pinned, which is the
    /// one thing someone needs when deciding whether to pin it again.
    #[test]
    fn unpin_keeps_the_comment_written_above_the_key() {
        let mut d = doc(COMMENTED);
        d.set_resident("qwen3", false).unwrap();
        let out = d.to_toml_string();
        assert!(out.contains("# Pinned: the default generate model."),
            "unpin destroyed the operator's note:\n{out}");
        assert!(out.contains("resident = false"), "the default is written back explicitly:\n{out}");
        assert!(!d.validate().unwrap().find("qwen3").unwrap().resident);
    }

    /// A knob that looks set and is not is the failure a closed key list exists to prevent.
    #[test]
    fn set_server_rejects_an_unknown_key_and_a_wrong_value() {
        let mut d = doc(COMMENTED);
        assert!(d.set_server("max_residents", "3").is_err(), "a plural typo must not be written");
        assert!(d.set_server("evict_policy", "lru2").is_err());
        assert!(d.set_server("max_resident", "-1").is_err());
        assert!(d.set_server("max_resident", "abc").is_err());
        d.set_server("max_resident", "3").unwrap();
        d.set_server("evict_policy", "none").unwrap();
        let cfg = d.validate().unwrap();
        assert_eq!(cfg.server.max_resident, 3);
        assert_eq!(cfg.server.evict_policy, EvictPolicy::None);
    }

    /// `idle_unload_s = 0` is the documented way to switch idle unload off, so it must be settable.
    #[test]
    fn zero_is_a_legal_value_for_the_idle_windows() {
        let mut d = doc(COMMENTED);
        d.set_server("idle_unload_s", "0").unwrap();
        d.set_server("idle_release_s", "0").unwrap();
        let cfg = d.validate().unwrap();
        assert_eq!(cfg.server.idle_unload(), None);
        assert_eq!(cfg.server.idle_release(), None);
    }

    /// An empty file is the fresh-install path: the first command has to create the tables.
    #[test]
    fn an_empty_document_grows_the_tables_it_needs() {
        let mut d = doc("");
        d.add_model("a", "s.toml").unwrap();
        d.set_resident("a", true).unwrap();
        d.set_default(Capability::ASR, "a");
        d.set_server("max_resident", "2").unwrap();
        let cfg = d.validate().unwrap();
        assert!(cfg.find("a").unwrap().resident);
        assert_eq!(cfg.defaults.get(Capability::ASR).map(String::as_str), Some("a"));
        assert_eq!(cfg.server.max_resident, 2);
    }

    /// Refuse a shape we would otherwise silently rewrite.
    #[test]
    fn an_inline_model_array_is_refused_not_rewritten() {
        let mut d = doc("model = [{ name = \"a\", scenario = \"s\" }]\n");
        assert!(d.add_model("b", "t").is_err());
        assert!(d.set_resident("a", true).is_err());
    }

    #[test]
    fn the_help_table_and_the_type_table_name_the_same_keys() {
        let types: Vec<&str> = SERVER_KEYS.iter().map(|(k, _)| *k).collect();
        let help: Vec<&str> = server_key_help().iter().map(|(k, _)| *k).collect();
        assert_eq!(types, help, "a settable key with no help, or help for a key nobody can set");
    }
}
