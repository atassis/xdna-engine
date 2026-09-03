//! `npu` - the single engine entrypoint. Thin clap shell over npu-runtime (control plane) and
//! npu-engine. Subcommands: serve, transcribe, embed, models, config, reload, bake.
use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};

mod cli_def;
mod media;

use anyhow::{anyhow, bail, Context, Result};
use clap::{CommandFactory, Parser};

use cli_def::{Cli, Cmd, ConfigCmd, OutFormat, WeightsCmd};
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
        Cmd::Transcribe { input, model } => transcribe(&path, input, model.as_deref()),
        Cmd::Embed { text, model } => embed(&path, text, model.as_deref()),
        Cmd::Diarize { wav, model, json } => diarize(&path, wav, model.as_deref(), *json),
        Cmd::TranscribeMedia { input, out, format, asr, diarize: diar, track, no_diarize } =>
            transcribe_media(&path, input, out.as_deref(), *format, asr.as_deref(),
                             diar.as_deref(), *track, *no_diarize),
        Cmd::Models { port } => models(&path, *port),
        Cmd::Reload { port } => reload(&path, *port),
        Cmd::Bake { name } => bake(&path, name),
        Cmd::Config { action } => config_cmd(&path, action),
        Cmd::Weights { action } => weights_cmd(&path, action),
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

fn transcribe(path: &Path, input: &Path, model: Option<&str>) -> Result<()> {
    quiet_one_shot();
    let cfg = load_cfg(path)?;
    let root = root(&cfg)?;
    // Decode BEFORE loading a model: a bad path or a file with no audio should fail in a second,
    // not after a multi-second model load.
    let samples = npu_runtime::media::decode_file(input).map_err(|e| anyhow!(e))?;
    // Lazy: a one-shot run should load the model it serves, and nothing else.
    let (handle, join) = start_lazy(cfg, Box::new(EngineLoader { root }))?;
    let out = handle.transcribe(model, samples, 16_000).map_err(|e| anyhow!(e.to_string()));
    handle.shutdown(); let _ = join.join();
    println!("{}", out?.value);
    Ok(())
}

/// Shortest span worth sending to ASR. Below this a "segment" is a diarization edge artefact and
/// the transcript gains a line of noise, not a word.
const MIN_UTTERANCE_S: f32 = 0.30;

/// Longest span sent to ASR in one call.
///
/// This was a correctness guard: parakeet used to truncate at WIN_MEL = 20.4 s and whisper at its
/// 30 s frontend window, both returning 200 OK with the tail missing, so 18 s kept every span under
/// the shorter cliff. Both backends now window internally and transcribe any length, so the cap is
/// only about span granularity. The number has not been re-measured against the windowed backends;
/// raising it gives whisper more context per call and should be swept before it is changed.
fn asr_window_s() -> f32 {
    std::env::var("NPU_ASR_MAX_SPAN_S").ok().and_then(|v| v.parse().ok()).unwrap_or(18.0)
}

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

            // Split turns the ASR cannot hold whole, then transcribe each piece. Without this a
            // long turn returns only its first ~20 s, with no error to notice.
            let max_span = asr_window_s();
            let spans: Vec<(f32, f32, u32)> = spans.iter()
                .flat_map(|&(a, b, spk)| media::split_turn(a, b, max_span).into_iter()
                    .map(move |(x, y)| (x, y, spk)))
                .collect();
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

/// Weight-checkpoint tooling.
///
/// Resolves the repo root the SAME way every other subcommand does (`root()`: XDNA_ENGINE_ROOT,
/// then an absolute scenario path, then cwd). The standalone binary used a bare `current_dir()`,
/// which is the cwd dependency the service install just removed -- folding it in drops that too.
fn weights_cmd(path: &Path, action: &WeightsCmd) -> Result<()> {
    use npu_weights::{checkpoint, spec::ModelSpec, spec::Source};
    let root = load_cfg(path).ok().and_then(|c| root(&c).ok())
        .map(Ok)
        .unwrap_or_else(std::env::current_dir)
        .context("repo root")?;
    match action {
        WeightsCmd::Bake { source, arch, checkpoint, force } => {
            let spec = ModelSpec {
                source: Source::parse(source)?, arch: arch.clone(), checkpoint: checkpoint.clone() };
            let p = spec.ensure_checkpoint(&root, *force)?;
            println!("checkpoint ready: {}", p.display());
        }
        WeightsCmd::Load { checkpoint, arch } => {
            let l = checkpoint::load(checkpoint, arch)?;
            println!("arch={} version={} tensors={}", l.arch, l.meta_version, l.names.len());
            for n in l.names.iter().take(5) {
                let (sh, _) = l.tensor_f32(n)?;
                println!("  {n} {sh:?}");
            }
        }
        WeightsCmd::Verify { checkpoint, arch, refs } => {
            let l = checkpoint::load(checkpoint, arch)?;
            let (n, max) = checkpoint::verify_against_npy(&l, refs)?;
            println!("verified {n} tensors; max abs rel-err {max:.4e}");
            anyhow::ensure!(max < 5e-2, "parity FAILED: max rel-err {max:.4e} >= 5e-2");
            println!("PARITY PASS");
        }
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
            "see [[Dispatch Cost]] here", "a [bracket] and a ] stray one",
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
