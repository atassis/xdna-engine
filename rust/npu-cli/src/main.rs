//! `npu` - the single engine entrypoint. Thin clap shell over npu-runtime (control plane) and
//! npu-engine. Subcommands: serve, transcribe, embed, models, config, reload, bake.
use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};

mod cli_def;
mod media;

use anyhow::{anyhow, bail, Context, Result};
use clap::{CommandFactory, Parser};

use cli_def::{Cli, Cmd, ConfigCmd, OutFormat};
use clap_complete::Shell;
use npu_runtime::actor::{start, start_lazy};
use npu_engine::capability::Capability;
use npu_runtime::config::{Config, EvictPolicy};
use npu_runtime::http;
use npu_runtime::loader::EngineLoader;

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
        Cmd::Diarize { wav, model, json } => diarize(&path, wav, model.as_deref(), *json),
        Cmd::TranscribeMedia { input, out, format, asr, diarize: diar, track, no_diarize } =>
            transcribe_media(&path, input, out.as_deref(), *format, asr.as_deref(),
                             diar.as_deref(), *track, *no_diarize),
        Cmd::Models { port } => models(&path, *port),
        Cmd::Reload { port } => reload(&path, *port),
        Cmd::Bake { name } => bake(&path, name),
        Cmd::Config { action } => config_cmd(&path, action),
        Cmd::Completions { shell } => {
            let mut cmd = Cli::command();
            let name = cmd.get_name().to_string();
            clap_complete::generate(*shell, &mut cmd, name, &mut std::io::stdout());
            Ok(())
        }
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

/// Shortest span worth sending to ASR. Below this a "segment" is a diarization edge artefact and
/// the transcript gains a line of noise, not a word.
const MIN_UTTERANCE_S: f32 = 0.30;

#[allow(clippy::too_many_arguments)]
fn transcribe_media(path: &Path, input: &Path, out: Option<&Path>, format: OutFormat,
                    asr: Option<&str>, diar: Option<&str>, only_track: Option<usize>,
                    no_diarize: bool) -> Result<()> {
    quiet_one_shot();
    let cfg = load_cfg(path)?;
    let root = root(&cfg)?;
    let tracks = media::probe_audio_tracks(input)?;
    if tracks.is_empty() { bail!("{} has no audio tracks", input.display()); }
    let wanted: Vec<&media::AudioTrack> = match only_track {
        Some(n) => tracks.iter().filter(|t| t.ord == n).collect(),
        None => tracks.iter().collect(),
    };
    if wanted.is_empty() {
        bail!("no audio track {} in {} (it has {})", only_track.unwrap(), input.display(), tracks.len());
    }
    eprintln!("[npu] {} audio track(s): {}", wanted.len(),
        wanted.iter().map(|t| t.label()).collect::<Vec<_>>().join(", "));

    // One actor for the whole run: the models stay resident across tracks and segments instead of
    // reloading per call. `max_resident` must be >= 2 for asr + diarize to coexist.
    let (handle, join) = start_lazy(cfg, Box::new(EngineLoader { root }))?;
    let tmp = std::env::temp_dir().join(format!("npu-media-{}", std::process::id()));
    std::fs::create_dir_all(&tmp).context("temp dir")?;

    let result = (|| -> Result<Vec<media::Utterance>> {
        let mut utts: Vec<media::Utterance> = Vec::new();
        for t in &wanted {
            let wav = tmp.join(format!("track{}.wav", t.ord));
            media::extract_track(input, t.ord, &wav)?;
            let bytes = std::fs::read(&wav).with_context(|| format!("read {}", wav.display()))?;
            let pcm = http::parse::parse_wav_i16(&bytes)
                .ok_or_else(|| anyhow!("track {} did not decode to 16k mono 16-bit", t.ord))?;
            let label = t.label();

            // Spans to transcribe: diarized turns, or the whole track when diarization is off.
            let spans: Vec<(f32, f32, u32)> = if no_diarize {
                vec![(0.0, pcm.len() as f32 / 16_000.0, 0)]
            } else {
                handle.diarize(diar, pcm.clone(), 16_000)
                    .map_err(|e| anyhow!("diarize track {}: {e}", t.ord))?
                    .value.iter().map(|s| (s.start_s, s.end_s, s.speaker)).collect()
            };
            let n_spk = spans.iter().map(|s| s.2).collect::<std::collections::BTreeSet<_>>().len();
            eprintln!("[npu] {label}: {} span(s), {n_spk} speaker(s)", spans.len());

            for (start_s, end_s, spk) in spans {
                if end_s - start_s < MIN_UTTERANCE_S { continue }
                // Slice the PCM directly rather than re-invoking ffmpeg per span: the samples are
                // already in memory and a subprocess per utterance would dominate the runtime.
                let (a, b) = ((start_s * 16_000.0) as usize, (end_s * 16_000.0) as usize);
                let slice = pcm[a.min(pcm.len())..b.min(pcm.len())].to_vec();
                if slice.is_empty() { continue }
                let text = handle.transcribe(asr, slice, 16_000)
                    .map_err(|e| anyhow!("transcribe {label} [{start_s:.2}-{end_s:.2}]: {e}"))?
                    .value.trim().to_string();
                if text.is_empty() { continue }
                utts.push(media::Utterance {
                    start_s, end_s,
                    speaker: media::speaker_label(&label, spk, n_spk),
                    text,
                });
            }
        }
        // All tracks share the recording's clock, so the merge is chronological, not per-track.
        utts.sort_by(|x, y| x.start_s.partial_cmp(&y.start_s).unwrap_or(std::cmp::Ordering::Equal));
        Ok(utts)
    })();

    handle.shutdown(); let _ = join.join();
    let _ = std::fs::remove_dir_all(&tmp);
    let utts = result?;

    let out_path = match out {
        Some(p) => p.to_path_buf(),
        None => input.with_extension(format.ext()),
    };
    let body = media::render(&utts, format.as_str(), &input.display().to_string())?;
    std::fs::write(&out_path, body).with_context(|| format!("write {}", out_path.display()))?;
    println!("{} ({} utterances)", out_path.display(), utts.len());
    Ok(())
}

fn diarize(path: &Path, wav: &Path, model: Option<&str>, json: bool) -> Result<()> {
    quiet_one_shot();
    let cfg = load_cfg(path)?;
    let root = root(&cfg)?;
    // Lazy, same reason as `transcribe`: a one-shot run loads the model it serves and nothing else.
    let (handle, join) = start_lazy(cfg, Box::new(EngineLoader { root }))?;
    let bytes = std::fs::read(wav).with_context(|| format!("read {}", wav.display()))?;
    let samples = http::parse::parse_wav_i16(&bytes)
        .ok_or_else(|| anyhow!("bad wav (need 16k mono 16-bit)"))?;
    let out = handle.diarize(model, samples, 16_000).map_err(|e| anyhow!(e.to_string()));
    handle.shutdown(); let _ = join.join();
    println!("{}", render_segments(&out?.value, json));
    Ok(())
}

/// Human lines by default, the HTTP JSON body under `--json`. Pure, so it is testable without a
/// device, a model or a server.
fn render_segments(segs: &[npu_engine::capability::Segment], json: bool) -> String {
    if json {
        let items: Vec<String> = segs.iter().map(|s| format!(
            "{{\"start\":{:.3},\"end\":{:.3},\"speaker\":\"SPEAKER_{:02}\"}}",
            s.start_s, s.end_s, s.speaker)).collect();
        return format!("{{\"segments\":[{}]}}", items.join(","));
    }
    segs.iter()
        .map(|s| format!("[{:.2} - {:.2}] SPEAKER_{:02}", s.start_s, s.end_s, s.speaker))
        .collect::<Vec<_>>()
        .join("\n")
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

    #[test]
    fn diarize_lines_are_human_readable_and_json_is_machine_readable() {
        let segs = vec![
            npu_engine::capability::Segment { start_s: 0.5, end_s: 3.25, speaker: 0 },
            npu_engine::capability::Segment { start_s: 3.25, end_s: 9.0, speaker: 1 },
        ];
        let lines = render_segments(&segs, false);
        assert_eq!(lines.lines().count(), 2, "{lines}");
        assert!(lines.starts_with("[0.50 - 3.25] SPEAKER_00"), "{lines}");
        assert!(lines.contains("[3.25 - 9.00] SPEAKER_01"), "{lines}");
        let json = render_segments(&segs, true);
        assert!(json.starts_with('{') && json.contains("\"segments\""), "{json}");
        assert!(json.contains("\"speaker\":\"SPEAKER_01\""), "{json}");
    }

    #[test]
    fn an_empty_diarization_renders_without_panicking() {
        assert_eq!(render_segments(&[], false), "");
        assert!(render_segments(&[], true).contains("\"segments\":[]"));
    }
}
