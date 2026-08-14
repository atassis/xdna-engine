//! `npu` - the single engine entrypoint. Thin clap shell over npu-runtime (control plane) and
//! npu-engine. Subcommands: serve, transcribe, embed, models, config, reload, bake.
use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use clap::{Parser, Subcommand};
use npu_runtime::actor::{start, start_lazy};
use npu_engine::capability::Capability;
use npu_runtime::config::{Config, EvictPolicy};
use npu_runtime::http;
use npu_runtime::loader::EngineLoader;

#[derive(Parser)]
#[command(name = "npu", about = "XDNA2 NPU engine multitool")]
struct Cli {
    /// Config path (default: $NPU_CONFIG or ~/.config/npu/engine.toml)
    #[arg(long, global = true)]
    config: Option<PathBuf>,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run the HTTP service (single device owner).
    Serve {
        #[arg(long)] port: Option<u16>,
        /// Bind even when a configured model failed to load. `/healthz` still reports 503.
        #[arg(long)] allow_degraded: bool,
    },
    /// One-shot transcription of a 16 kHz mono 16-bit WAV.
    Transcribe { wav: PathBuf, #[arg(long)] model: Option<String> },
    /// One-shot embedding of a text string.
    /// `allow_hyphen_values`: the text to embed is prose, and prose begins with `-` all the time
    /// (every Markdown bullet). Without it clap read a bullet as an unknown flag and failed with a
    /// usage error, so the CLI rejected inputs the HTTP route accepted.
    Embed { #[arg(allow_hyphen_values = true)] text: String, #[arg(long)] model: Option<String> },
    /// List models on a running server.
    Models { #[arg(long)] port: Option<u16> },
    /// Ask a running server to re-read the config and reconcile.
    Reload { #[arg(long)] port: Option<u16> },
    /// Pre-bake a model's weight checkpoint (host-only, no device).
    Bake { name: String },
    /// Inspect / edit the desired-state config.
    Config { #[command(subcommand)] action: ConfigCmd },
}

#[derive(Subcommand)]
enum ConfigCmd {
    Show,
    AddModel { name: String, scenario: String },
    RemoveModel { name: String },
    SetDefault { capability: String, model: String },
}

fn config_path(cli: &Cli) -> PathBuf {
    if let Some(p) = &cli.config { return p.clone(); }
    if let Ok(p) = std::env::var("NPU_CONFIG") { return PathBuf::from(p); }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join(".config/npu/engine.toml")
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let path = config_path(&cli);
    match &cli.cmd {
        Cmd::Serve { port, allow_degraded } => serve(&path, *port, *allow_degraded),
        Cmd::Transcribe { wav, model } => transcribe(&path, wav, model.as_deref()),
        Cmd::Embed { text, model } => embed(&path, text, model.as_deref()),
        Cmd::Models { port } => models(&path, *port),
        Cmd::Reload { port } => reload(&path, *port),
        Cmd::Bake { name } => bake(&path, name),
        Cmd::Config { action } => config_cmd(&path, action),
    }
}

fn load_cfg(path: &Path) -> Result<Config> { Config::load(path).map_err(|e| anyhow!(e)) }

/// A one-shot invocation's output is its VALUE -- an embedding, a transcript -- so the engine's
/// load-time banners ("which precision", "which resident xclbin") are noise in front of it. The
/// service keeps them: there they are the record of what the device is actually running. Only a
/// default: `NPU_QUIET=0 npu embed ...` brings them back.
fn quiet_one_shot() {
    if std::env::var_os("NPU_QUIET").is_none() {
        std::env::set_var("NPU_QUIET", "1");
    }
}

/// Repo root that a scenario's relative `artifacts.weights` path resolves against.
///
/// This used to be plain `current_dir()`, which is only correct when you happen to be standing in
/// the repo. The service never noticed because its unit sets `WorkingDirectory=$REPO`; every other
/// invocation (`npu embed`, `npu transcribe`, `npu serve` from a shell) resolved
/// `artifacts/parakeet/...` against the caller's cwd and died on a missing file.
///
/// Order: explicit env override, then derive from an absolute scenario path
/// (`<repo>/scenarios/x.toml` -> `<repo>`), then cwd as the last resort.
fn root(cfg: &Config) -> Result<PathBuf> {
    if let Ok(p) = std::env::var("XDNA_ENGINE_ROOT") {
        return Ok(PathBuf::from(p));
    }
    for m in &cfg.models {
        let s = Path::new(&m.scenario);
        if !s.is_absolute() { continue; }
        let dir = match s.parent() { Some(d) => d, None => continue };
        if dir.file_name().map(|n| n == "scenarios").unwrap_or(false) {
            if let Some(r) = dir.parent() { return Ok(r.to_path_buf()); }
        }
    }
    std::env::current_dir().context("cwd")
}

/// Fail SOFTLY when the port is already taken, instead of loading models first and dying on an
/// opaque "Address already in use" (os error 98) after a panic.
///
/// 11434 is NOT ours exclusively -- ollama, FLM and others default to it too -- so we do not claim
/// to know who is there. Probe `/healthz` and only name xdna-engine when the reply is actually
/// ours; otherwise report an unidentified listener and let the operator decide.
fn preflight_serve(port: u16) -> Result<()> {
    use std::time::Duration;
    let addr = match format!("127.0.0.1:{port}").parse() { Ok(a) => a, Err(_) => return Ok(()) };
    if TcpStream::connect_timeout(&addr, Duration::from_millis(300)).is_err() {
        // Nothing listening; the port is ours to bind.
        return if npu_engine::Engine::available() {
            Ok(())
        } else {
            bail!("no XDNA2 NPU device at /dev/accel/accel0 (is the amdxdna driver loaded?)")
        };
    }
    // Something is listening. Ask it who it is rather than assuming.
    let mine = http_get(port, "/healthz").map(|b| b.contains("\"npu\"")).unwrap_or(false);
    if mine {
        bail!(
            "port {port} is already served by an xdna-engine instance.\n  \
             status : systemctl --user status xdna-engine\n  \
             stop   : systemctl --user stop xdna-engine\n  \
             or use another port: npu serve --port <other>"
        );
    }
    bail!(
        "port {port} is already in use by another process (it did not answer /healthz as an\n  \
         xdna-engine, so it is likely ollama, FLM or a different server -- {port} is a shared\n  \
         default). Identify it with:  ss -ltnp 'sport = :{port}'\n  \
         Then stop it, or use another port: npu serve --port <other>"
    );
}

/// Scenario name gating the whole_array resident-artifact freshness check below. String, not a
/// structural field: `ScenarioConfig` has no "which NPU backend does this need" marker (the
/// still-open half of `artifact-preflight-and-fail-loud` -- see its `next:`), so this is the
/// same kind of pragmatic name match the engine itself already uses at a few call sites (e.g.
/// `xclbin.file_name()...contains("krtp")` in npu.rs). Narrow on purpose: it catches exactly the
/// artifact behind the 5-day outage this task names, not every artifact every scenario can touch.
const PARAKEET_SCENARIO_NAME: &str = "parakeet-tdt-0.6b-v3";

/// Device-free: fails BEFORE `start()` ever calls `Device::open`, so a stale/missing resident
/// build is diagnosed without needing (or touching) the NPU at all. Scoped to configs that
/// actually load the Parakeet scenario -- see `PARAKEET_SCENARIO_NAME`.
fn preflight_artifacts(cfg: &Config, root: &Path) -> Result<()> {
    for m in &cfg.models {
        let p = Path::new(&m.scenario);
        let scenario_path = if p.is_absolute() { p.to_path_buf() } else { root.join(p) };
        let Ok(sc) = npu_engine::config::ScenarioConfig::load(&scenario_path) else { continue };
        if sc.scenario.name == PARAKEET_SCENARIO_NAME {
            npu_parakeet::npu::preflight(root)
                .map_err(|e| anyhow!("model {:?} ({}): {e}", m.name, sc.scenario.name))?;
        }
    }
    Ok(())
}

fn serve(path: &Path, port: Option<u16>, allow_degraded: bool) -> Result<()> {
    let cfg = load_cfg(path)?;
    let port = port.unwrap_or(cfg.server.port);
    preflight_serve(port)?;
    let root = root(&cfg)?;
    preflight_artifacts(&cfg, &root)?;
    let (handle, _join) = start(cfg, Box::new(EngineLoader { root }))?;
    // Do not bind a port the service cannot serve from. The initial reconcile records a load
    // failure as `Failed` rather than panicking, so before this the socket came up and every
    // request answered "actor dropped reply" while systemd showed active -- how a 5-day outage
    // went unnoticed. Refuse instead, naming each model and its cause.
    let failed: Vec<_> = handle.status().into_iter()
        .filter(|s| s.state == npu_runtime::registry::LoadState::Failed).collect();
    if !failed.is_empty() {
        for s in &failed {
            eprintln!("[npu-serve] FAILED {}: {}", s.name, s.detail);
        }
        if !allow_degraded {
            handle.shutdown();
            bail!("{} of the configured models failed to load; refusing to bind port {port} \
                   (use --allow-degraded to serve anyway)", failed.len());
        }
        eprintln!("[npu-serve] --allow-degraded: binding anyway, /healthz will report 503");
    }
    http::serve(handle, path.to_path_buf(), port).context("serve")
}

fn transcribe(path: &Path, wav: &Path, model: Option<&str>) -> Result<()> {
    quiet_one_shot();
    let cfg = load_cfg(path)?;
    let root = root(&cfg)?;
    // Lazy: a one-shot run should load the model it serves, and nothing else.
    let (handle, join) = start_lazy(cfg, Box::new(EngineLoader { root }))?;
    let bytes = std::fs::read(wav).with_context(|| format!("read {}", wav.display()))?;
    let samples = http::parse::parse_wav_i16(&bytes).ok_or_else(|| anyhow!("bad wav (need 16k mono 16-bit)"))?;
    let out = handle.transcribe(model, samples, 16_000).map_err(|e| anyhow!(e.to_string()));
    handle.shutdown(); let _ = join.join();
    println!("{}", out?.value);
    Ok(())
}

fn embed(path: &Path, text: &str, model: Option<&str>) -> Result<()> {
    quiet_one_shot();
    let cfg = load_cfg(path)?;
    let root = root(&cfg)?;
    // Lazy: `npu embed` against an ASR-only config used to pay a full parakeet load before it could
    // say there was no embed model at all.
    let (handle, join) = start_lazy(cfg, Box::new(EngineLoader { root }))?;
    let out = handle.embed(model, text).map_err(|e| anyhow!(e.to_string()));
    handle.shutdown(); let _ = join.join();
    let v = out?.value;
    let arr = v.iter().map(|x| format!("{x}")).collect::<Vec<_>>().join(",");
    println!("[{arr}]");
    Ok(())
}

fn models(path: &Path, port: Option<u16>) -> Result<()> {
    let port = resolve_port(path, port)?;
    match http_get(port, "/v1/models") { Ok(body) => { println!("{body}"); Ok(()) }
        Err(_) => { println!("no server on 127.0.0.1:{port}"); Ok(()) } }
}

fn reload(path: &Path, port: Option<u16>) -> Result<()> {
    let port = resolve_port(path, port)?;
    let body = http_post(port, "/admin/reload", "").context("reload (is the server running?)")?;
    println!("{body}");
    Ok(())
}

fn bake(path: &Path, name: &str) -> Result<()> {
    let cfg = load_cfg(path)?;
    let m = cfg.find(name).ok_or_else(|| anyhow!("unknown model {name:?} in config"))?;
    let sc = npu_engine::config::ScenarioConfig::load(Path::new(&m.scenario))
        .with_context(|| format!("scenario {}", m.scenario))?;
    match sc.artifacts.model_spec()? {
        Some(spec) => { let p = spec.ensure_checkpoint(&root(&cfg)?, false)?; println!("baked: {}", p.display()); }
        None => println!("nothing to bake ({} uses legacy npy weights)", name),
    }
    Ok(())
}

fn config_cmd(path: &Path, action: &ConfigCmd) -> Result<()> {
    let mut cfg = load_cfg(path)?;
    match action {
        ConfigCmd::Show => { print!("{}", render(&cfg)); return Ok(()); }
        ConfigCmd::AddModel { name, scenario } => {
            cfg.models.retain(|m| &m.name != name);
            cfg.models.push(npu_runtime::config::ModelCfg { name: name.clone(), scenario: scenario.clone() });
        }
        ConfigCmd::RemoveModel { name } => cfg.models.retain(|m| &m.name != name),
        ConfigCmd::SetDefault { capability, model } => match Capability::from_name(capability) {
            Some(cap) => cfg.defaults.set(cap, model.clone()),
            None => return Err(anyhow!("unknown capability {capability:?} (one of: {})",
                Capability::ALL.iter().map(|c| c.0).collect::<Vec<_>>().join("|"))),
        },
    }
    cfg.save(path).map_err(|e| anyhow!(e))?;
    println!("updated {}", path.display());
    Ok(())
}

/// Human-readable config summary (pure, testable).
fn render(cfg: &Config) -> String {
    let mut s = format!("port {}  max_resident {}  memory_ceiling_mb {}\n",
        cfg.server.port, cfg.server.max_resident, cfg.server.memory_ceiling_mb);
    s.push_str(&format!("residency: idle_unload_s {}  idle_release_s {}  sweep_interval_s {}  evict_policy {}\n",
        cfg.server.idle_unload_s, cfg.server.idle_release_s, cfg.server.sweep_interval_s,
        match cfg.server.evict_policy { EvictPolicy::Lru => "lru", EvictPolicy::None => "none" }));
    let defaults = cfg.defaults.0.iter().map(|(c, m)| format!("{c}={m}")).collect::<Vec<_>>();
    s.push_str(&format!("defaults: {}\n",
        if defaults.is_empty() { "(none)".to_string() } else { defaults.join(" ") }));
    if cfg.models.is_empty() { s.push_str("models: (none)\n"); }
    for m in &cfg.models { s.push_str(&format!("model {} -> {}\n", m.name, m.scenario)); }
    s
}

fn resolve_port(path: &Path, port: Option<u16>) -> Result<u16> {
    Ok(port.unwrap_or_else(|| Config::load(path).map(|c| c.server.port).unwrap_or(11434)))
}

// --- minimal HTTP/1.1 client (std only) ---
fn http_get(port: u16, path: &str) -> Result<String> { http_req(port, "GET", path, "") }
fn http_post(port: u16, path: &str, body: &str) -> Result<String> { http_req(port, "POST", path, body) }
fn http_req(port: u16, method: &str, path: &str, body: &str) -> Result<String> {
    let mut s = TcpStream::connect(("127.0.0.1", port))?;
    let req = format!("{method} {path} HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len());
    s.write_all(req.as_bytes())?;
    let mut resp = String::new();
    s.read_to_string(&mut resp)?;
    Ok(resp.split("\r\n\r\n").nth(1).unwrap_or("").to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use npu_runtime::config::{Defaults, ModelCfg, ServerCfg};
    #[test]
    fn render_empty_and_populated() {
        let empty = Config::default();
        let r = render(&empty);
        assert!(r.contains("models: (none)"));
        assert!(r.contains("port 11434"));
        assert!(r.contains("defaults: (none)"), "{r}");
        let c = Config {
            server: ServerCfg::default(),
            defaults: Defaults::from_pairs([
                (Capability::ASR, "parakeet".to_string()), (Capability::TTS, "kokoro".to_string())]),
            models: vec![ModelCfg { name: "parakeet".into(), scenario: "scenarios/asr.toml".into() }],
        };
        let r = render(&c);
        assert!(r.contains("model parakeet -> scenarios/asr.toml"));
        // Every configured default is rendered, including one no `ModelKind` variant can name.
        assert!(r.contains("asr=parakeet") && r.contains("tts=kokoro"), "{r}");
        assert!(r.contains("idle_unload_s 900") && r.contains("idle_release_s 1800")
            && r.contains("evict_policy lru"), "{r}");
    }

    /// Builds a fake `root` with a real `toolchain.lock`, a `scenarios/asr.toml` naming the
    /// Parakeet scenario, and (optionally) a resident build dir stamped for a DIFFERENT pin --
    /// the re-pin-wipe shape `artifact-preflight-and-fail-loud` exists to catch. Returns the root
    /// and the `Config` a real `serve()` call would have loaded.
    fn fake_parakeet_root(stamp_matches_current_pin: bool) -> (tempfile::TempDir, Config) {
        let td = tempfile::tempdir().unwrap();
        std::fs::write(td.path().join("toolchain.lock"), b"pin-current").unwrap();
        std::fs::create_dir_all(td.path().join("scenarios")).unwrap();
        std::fs::write(
            td.path().join("scenarios/asr.toml"),
            "[scenario]\nkind = \"asr\"\nname = \"parakeet-tdt-0.6b-v3\"\n\
             [model]\nhidden=1024\nff=4096\nn_heads=8\nhead_dim=128\nn_layers=24\nmax_seq=2040\n\
             [artifacts]\nweights = \"artifacts/parakeet\"\n",
        ).unwrap();
        let wa_dir = td.path().join(npu_parakeet::npu::WA_SUBDIR);
        std::fs::create_dir_all(&wa_dir).unwrap();
        std::fs::write(wa_dir.join("final_512x1024x4096_64x32x128_8c.xclbin"), b"fake").unwrap();
        let stamp = if stamp_matches_current_pin {
            npu_asr::kernel_registry::current_toolchain_hash(td.path()).unwrap()
        } else {
            "stale00000pin".to_string()
        };
        std::fs::write(wa_dir.join(".toolchain-stamp"), stamp).unwrap();
        let cfg = Config {
            server: ServerCfg::default(),
            defaults: Defaults::default(),
            models: vec![ModelCfg {
                name: "parakeet".into(),
                scenario: td.path().join("scenarios/asr.toml").to_str().unwrap().to_string(),
            }],
        };
        (td, cfg)
    }

    #[test]
    fn preflight_artifacts_passes_when_resident_build_matches_current_pin() {
        let (td, cfg) = fake_parakeet_root(true);
        preflight_artifacts(&cfg, td.path()).expect("fresh resident build must pass");
    }

    /// THE demonstration: construct the exact broken state (5-day-outage shape) and show the
    /// check refuses it, naming both the stale and current pin.
    #[test]
    fn preflight_artifacts_fails_loud_when_resident_build_predates_a_repin() {
        let (td, cfg) = fake_parakeet_root(false);
        let err = preflight_artifacts(&cfg, td.path())
            .expect_err("a resident build stamped for a DIFFERENT pin must not pass preflight");
        let msg = err.to_string();
        assert!(msg.contains("parakeet-tdt-0.6b-v3"), "{msg}");
        assert!(msg.contains("re-pinned"), "{msg}");
        assert!(msg.contains("build_parakeet_kernels.sh"), "{msg}");
    }

    /// A config with no Parakeet scenario at all must not pay (or fail) this check -- proves the
    /// scoping in `PARAKEET_SCENARIO_NAME` actually gates, not just names, the check.
    #[test]
    fn preflight_artifacts_is_a_noop_for_a_non_parakeet_config() {
        let td = tempfile::tempdir().unwrap();
        std::fs::write(td.path().join("toolchain.lock"), b"pin-current").unwrap();
        std::fs::create_dir_all(td.path().join("scenarios")).unwrap();
        std::fs::write(
            td.path().join("scenarios/bge.toml"),
            "[scenario]\nkind = \"embeddings\"\nname = \"bge-base-en-v1.5\"\n\
             [model]\nhidden=768\nff=3072\nn_heads=12\nhead_dim=64\nn_layers=12\nmax_seq=512\n\
             [artifacts]\nweights = \"artifacts/bge-base/encoder\"\n",
        ).unwrap();
        let cfg = Config {
            server: ServerCfg::default(),
            defaults: Defaults::default(),
            models: vec![ModelCfg {
                name: "bge".into(),
                scenario: td.path().join("scenarios/bge.toml").to_str().unwrap().to_string(),
            }],
        };
        // No mlir-aie/.../whole_array/build dir exists under td at all -- if the check were not
        // scoped, this would fail on MissingDir. It must pass because nothing here is Parakeet.
        preflight_artifacts(&cfg, td.path()).expect("non-parakeet config must skip the check");
    }
}
