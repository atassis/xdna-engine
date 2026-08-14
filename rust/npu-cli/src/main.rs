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

fn serve(path: &Path, port: Option<u16>, allow_degraded: bool) -> Result<()> {
    let cfg = load_cfg(path)?;
    let port = port.unwrap_or(cfg.server.port);
    preflight_serve(port)?;
    let root = root(&cfg)?;
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
}

/// Differential fuzz: does `npu embed <text>` extract the same string that `/v1/embeddings`
/// extracts from `{"input":<text>}`? Both paths hand the extracted string to the identical
/// `Handle::embed`, so this boundary is the only place they CAN diverge -- and it is host-only
/// (no NPU), unlike the vector-equality gate this task also names, which needs the device.
#[cfg(test)]
mod embed_cli_http_fuzz {
    use super::*;

    /// The text `npu embed` hands to `Handle::embed`, going through the real clap parser (not a
    /// reimplementation of it).
    fn cli_extract(text: &str) -> Result<String, String> {
        match Cli::try_parse_from(["npu", "embed", text]) {
            Ok(cli) => match cli.cmd { Cmd::Embed { text, .. } => Ok(text), _ => unreachable!() },
            Err(e) => Err(e.to_string()),
        }
    }

    /// The text `/v1/embeddings` hands to `Handle::embed`. `serde_json` builds the request body
    /// so the reference encoding is independent of `parse_inputs`, the scanner under test.
    fn http_extract(text: &str) -> Result<String, String> {
        let body = format!("{{\"input\":{}}}", serde_json::to_string(text).unwrap());
        http::parse::parse_inputs(&body).map(|v| v[0].clone())
    }

    /// One divergence record: what a passing case must never produce.
    struct Divergence { input: String, cli: Result<String, String>, http: Result<String, String> }

    fn check(input: &str, out: &mut Vec<Divergence>) {
        let (cli, http) = (cli_extract(input), http_extract(input));
        if cli.as_deref().ok() != http.as_deref().ok() {
            out.push(Divergence { input: input.to_string(), cli, http });
        }
    }

    /// Splitmix64 -- self-contained so the fuzz corpus needs no new crate.
    struct Rng(u64);
    impl Rng {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        }
        fn u32(&mut self) -> u32 { (self.next_u64() >> 32) as u32 }
    }

    /// A random string biased toward CLI/JSON metacharacters (quotes, brackets, backslashes,
    /// leading `-`, control chars, RTL/zero-width marks) with a random-Unicode-scalar filler, so
    /// the fuzzer spends most of its budget near the two parsers' actual decision points.
    fn random_string(rng: &mut Rng, max_len: usize) -> String {
        const HOT: &[char] = &[
            '"', '\\', '[', ']', '{', '}', ':', ',', '-', '\'', '`',
            '\n', '\r', '\t', ' ', '\u{7f}', '\u{200b}', '\u{feff}', '\u{202e}',
            'п', 'р', 'и', '🌍', '🧵', '\u{1f469}', '\u{200d}',
        ];
        let len = rng.u32() as usize % (max_len + 1);
        let mut s = String::new();
        for _ in 0..len {
            if rng.u32().is_multiple_of(2) {
                s.push(HOT[rng.u32() as usize % HOT.len()]);
            } else {
                // Any scalar value except the surrogate range D800-DFFF, which `char` cannot
                // represent anyway (a lone surrogate is exactly what a client can never send as a
                // valid Rust/JSON string -- `scan_unicode_escape` rejects the wire form of that).
                let cp = 0x20 + rng.u32() % (0x2FFFF - 0x20);
                if let Some(c) = char::from_u32(cp) { s.push(c); }
            }
        }
        s
    }

    /// Gate 1's fixture list (this task's own corpus), run through the real CLI parser this time
    /// instead of asserting `parse_inputs` alone. `-`/`--`-leading text is the case
    /// `allow_hyphen_values` exists for.
    #[test]
    fn curated_adversarial_corpus_agrees() {
        let mut div = Vec::new();
        for s in [
            "hello", "", "   ", "- a markdown bullet", "-- a flag", "---",
            "see [[dispatch-cost]] here", "a [bracket] and a ] stray one",
            "he said \"hi\"", "back\\slash", "a lone { brace", "line\nnext\ttab",
            "привет 🌍", "🌍", "\u{200b}zero-width", "\u{202e}rtl-override",
            &"x".repeat(10_000),
        ] {
            check(s, &mut div);
        }
        assert!(div.is_empty(), "{} divergence(s) on the curated corpus: {:#?}", div.len(),
            div.iter().map(|d| (&d.input, &d.cli, &d.http)).collect::<Vec<_>>());
    }

    /// 4 seeds x 20k cases, lengths up to 96. A failure prints every divergence found, not just the
    /// first, since the point of this task is to enumerate them.
    #[test]
    fn random_corpus_agrees() {
        let mut div = Vec::new();
        let mut n = 0u32;
        for seed in [0x2545F4914F6CDD1D, 0x853C49E6748FEA9B, 0x1FFFFFFFFFFFFFF, 0xD1B54A32D192ED03] {
            let mut rng = Rng(seed);
            for _ in 0..20_000u32 {
                let s = random_string(&mut rng, 96);
                check(&s, &mut div);
                n += 1;
            }
        }
        assert!(div.is_empty(), "{} divergence(s) out of {n} random cases: {:#?}", div.len(),
            div.iter().take(20).map(|d| (&d.input, &d.cli, &d.http)).collect::<Vec<_>>());
    }

    /// `allow_hyphen_values` does not cover text that is an EXACT match for a flag `clap` already
    /// knows about at that parse position: `-h`/`--help` (built in), `-V`/`--version` is NOT
    /// defined at the subcommand level so it passes through, but `--model` (this subcommand's own
    /// flag) and `--config` (the top-level global flag) still swallow the positional and error
    /// asking for a value, and a bare `--` is consumed as the "rest is positional" separator and
    /// leaves nothing behind. HTTP has none of this: `{"input":"--model"}` embeds the four
    /// literal characters. Measured with `Cli::try_parse_from`, not asserted:
    /// `"-h"`/`"--help"` -> Err (clap prints help as the error body); `"--model"`/`"--config"` ->
    /// Err ("a value is required for ... but none was supplied"); `"--"` alone -> Err ("required
    /// arguments were not provided"). A `-- <text>` prefix is the workaround for a caller building
    /// argv programmatically; a real document is exceedingly unlikely to consist of exactly one of
    /// these four literals, which is why the corpora above never hit it. Documented here as a
    /// known divergence, not fixed: removing help/version at this subcommand is a behavior change
    /// this task was not scoped to make.
    #[test]
    fn known_divergence_exact_flag_literal_collision() {
        for text in ["-h", "--help", "--model", "--config", "--"] {
            assert!(cli_extract(text).is_err(), "expected {text:?} to still trip clap");
            assert_eq!(http_extract(text).as_deref(), Ok(text), "HTTP must embed it literally");
        }
        // Not a divergence: no flag named `-V`/`--version` is declared on the `Embed` subcommand
        // itself (only auto-generated at the top level), so this one round-trips.
        assert_eq!(cli_extract("-V").as_deref(), Ok("-V"));
        assert_eq!(http_extract("-V").as_deref(), Ok("-V"));
    }
}
