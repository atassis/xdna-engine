//! `npu` - the single engine entrypoint. Thin clap shell over npu-runtime (control plane) and
//! npu-engine. Subcommands: serve, transcribe, embed, models, config, reload, bake.
use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use clap::{Parser, Subcommand};
use npu_runtime::actor::start;
use npu_runtime::config::Config;
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
    Serve { #[arg(long)] port: Option<u16> },
    /// One-shot transcription of a 16 kHz mono 16-bit WAV.
    Transcribe { wav: PathBuf, #[arg(long)] model: Option<String> },
    /// One-shot embedding of a text string.
    Embed { text: String, #[arg(long)] model: Option<String> },
    /// List models on a running server.
    Models { #[arg(long)] port: Option<u16> },
    /// Ask a running server to re-read the config and reconcile.
    Reload { #[arg(long)] port: Option<u16> },
    /// Pre-bake a model's weight arena (host-only, no device).
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
        Cmd::Serve { port } => serve(&path, *port),
        Cmd::Transcribe { wav, model } => transcribe(&path, wav, model.as_deref()),
        Cmd::Embed { text, model } => embed(&path, text, model.as_deref()),
        Cmd::Models { port } => models(&path, *port),
        Cmd::Reload { port } => reload(&path, *port),
        Cmd::Bake { name } => bake(&path, name),
        Cmd::Config { action } => config_cmd(&path, action),
    }
}

fn load_cfg(path: &Path) -> Result<Config> { Config::load(path).map_err(|e| anyhow!(e)) }
fn root() -> Result<PathBuf> { std::env::current_dir().context("cwd") }

fn serve(path: &Path, port: Option<u16>) -> Result<()> {
    let cfg = load_cfg(path)?;
    let port = port.unwrap_or(cfg.server.port);
    let (handle, _join) = start(cfg, Box::new(EngineLoader { root: root()? }));
    http::serve(handle, path.to_path_buf(), port).context("serve")
}

fn transcribe(path: &Path, wav: &Path, model: Option<&str>) -> Result<()> {
    let cfg = load_cfg(path)?;
    let (handle, join) = start(cfg, Box::new(EngineLoader { root: root()? }));
    let bytes = std::fs::read(wav).with_context(|| format!("read {}", wav.display()))?;
    let samples = http::parse::parse_wav_i16(&bytes).ok_or_else(|| anyhow!("bad wav (need 16k mono 16-bit)"))?;
    let out = handle.transcribe(model, samples, 16_000).map_err(|e| anyhow!(e.to_string()));
    handle.shutdown(); let _ = join.join();
    println!("{}", out?.value);
    Ok(())
}

fn embed(path: &Path, text: &str, model: Option<&str>) -> Result<()> {
    let cfg = load_cfg(path)?;
    let (handle, join) = start(cfg, Box::new(EngineLoader { root: root()? }));
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
        Some(spec) => { let p = spec.ensure_arena(&root()?, false)?; println!("baked: {}", p.display()); }
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
        ConfigCmd::SetDefault { capability, model } => match capability.as_str() {
            "asr" => cfg.defaults.asr = Some(model.clone()),
            "embed" => cfg.defaults.embed = Some(model.clone()),
            other => return Err(anyhow!("unknown capability {other:?} (asr|embed)")),
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
    s.push_str(&format!("defaults: asr={:?} embed={:?}\n", cfg.defaults.asr, cfg.defaults.embed));
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
        let c = Config {
            server: ServerCfg::default(),
            defaults: Defaults { asr: Some("parakeet".into()), embed: None },
            models: vec![ModelCfg { name: "parakeet".into(), scenario: "scenarios/asr.toml".into() }],
        };
        let r = render(&c);
        assert!(r.contains("model parakeet -> scenarios/asr.toml"));
        assert!(r.contains("asr=Some(\"parakeet\")"));
    }
}
