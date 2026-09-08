//! `npu` - the single engine entrypoint. Thin clap shell over npu-runtime (control plane) and
//! npu-engine. Subcommands: serve, transcribe, embed, models, config, reload, bake.
use std::io::{BufRead, Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

mod cli_def;
mod doctor;
mod exit;
mod media;

use anyhow::{anyhow, bail, Context, Result};
use clap::{CommandFactory, Parser};

use cli_def::{Cli, Cmd, ConfigCmd, OutFormat, SamplingArgs, WeightsCmd};
use clap_complete::Shell;
use exit::{engine_error, Code, Tagged};
use npu_runtime::actor::{start, start_lazy};
use npu_engine::capability::Capability;
use npu_runtime::config::{Config, EvictPolicy};
use npu_runtime::http;
use npu_runtime::loader::EngineLoader;
use npu_runtime::stream::StreamItem;

fn config_path(cli: &Cli) -> PathBuf { config_path_and_source(cli).0 }

/// [`config_path`] plus WHICH of the three sources won -- `--config` beats `$NPU_CONFIG` beats the
/// default path. `npu doctor` reports this directly; every other caller just wants the path.
fn config_path_and_source(cli: &Cli) -> (PathBuf, &'static str) {
    if let Some(p) = &cli.config { return (p.clone(), "--config flag"); }
    if let Ok(p) = std::env::var("NPU_CONFIG") { return (PathBuf::from(p), "$NPU_CONFIG"); }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    (PathBuf::from(home).join(".config/npu/engine.toml"), "default path (~/.config/npu/engine.toml)")
}

/// The one place an error becomes a process exit code (`exit::of`) -- see `exit.rs`. Printing
/// stays exactly what `Result<(), E: Debug>`'s stdlib `Termination` impl already did (`Error:
/// {e:?}`, the anyhow chain with "Caused by:"); only the exit status is new.
fn main() -> ExitCode {
    let cli = Cli::parse();
    let path = config_path(&cli);
    match run(&cli, &path) {
        Ok(()) => ExitCode::from(Code::Success as u8),
        Err(e) => {
            eprintln!("Error: {e:?}");
            ExitCode::from(exit::of(&e) as u8)
        }
    }
}

fn run(cli: &Cli, path: &Path) -> Result<()> {
    match &cli.cmd {
        Cmd::Serve { port, allow_degraded } => serve(path, *port, *allow_degraded),
        Cmd::Transcribe { input, model } => transcribe(path, input, model.as_deref()),
        Cmd::Generate { prompt, model, sampling, no_stream, raw } =>
            generate(path, prompt, model.as_deref(), sampling, *no_stream, *raw),
        Cmd::Chat { prompt, model, sampling, no_stream } =>
            chat(path, prompt.as_deref(), model.as_deref(), sampling, *no_stream),
        Cmd::Embed { text, model } => embed(path, text, model.as_deref()),
        Cmd::Diarize { wav, model, json } => diarize(path, wav, model.as_deref(), *json),
        Cmd::TranscribeMedia { input, out, format, asr, diarize: diar, track, no_diarize } =>
            transcribe_media(path, input, out.as_deref(), *format, asr.as_deref(),
                             diar.as_deref(), *track, *no_diarize),
        Cmd::Models { json, port } => models(&path, *json, *port),
        Cmd::Reload { port } => reload(&path, *port),
        Cmd::Load { model, port } => load_model(&path, model, *port),
        Cmd::Unload { model, port } => unload_model(&path, model, *port),
        Cmd::Bake { name } => bake(&path, name),
        Cmd::Config { action } => config_cmd(&path, action),
        Cmd::Flags { json } => flags_cmd(*json),
        Cmd::Weights { action } => weights_cmd(&path, action),
        Cmd::Doctor { json } => doctor::doctor(&cli, *json),
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
/// Where `artifacts/` and `scenarios/` live.
///
/// Ordered by how explicit each source is, and every candidate is CHECKED before it is returned:
/// a root is only a root if it actually holds `scenarios/`. That check is the point. The absolute-
/// scenario branch below shipped unverified and was dead the whole time -- `engine.toml` writes
/// scenario paths relative (`scenarios/asr.toml`), so `is_absolute()` skipped every model and the
/// function fell through to the working directory. Any invocation from outside a checkout then died
/// on `scenario <cwd>/scenarios/....toml: No such file`, which is not a diagnosis of anything.
fn root(cfg: &Config, config_path: &Path) -> Result<PathBuf> {
    if let Ok(p) = std::env::var("XDNA_ENGINE_ROOT") {
        return Ok(PathBuf::from(p));
    }
    let home = std::env::var("HOME").ok().map(PathBuf::from);
    let install = std::env::var("XDG_DATA_HOME").ok().map(PathBuf::from)
        .or_else(|| home.map(|h| h.join(".local/share")))
        // The prefix install.sh stages and bakes into the unit (`ENGINE_ROOT`, install.sh). If that
        // name changes there, it must change here: these are one constant in two files, and the
        // only reason it is not shared is that one of them is bash.
        .map(|d| d.join("xdna-engine"));
    let cwd = std::env::current_dir().ok();
    for cand in root_candidates(cfg, config_path, cwd, install) {
        if cand.join("scenarios").is_dir() { return Ok(cand) }
    }
    std::env::current_dir().context("cwd")
}

/// The candidate roots, most explicit first. Separate from `root` so the ordering is testable
/// without setting process-wide environment variables.
fn root_candidates(cfg: &Config, config_path: &Path, cwd: Option<PathBuf>,
                   install: Option<PathBuf>) -> Vec<PathBuf> {
    let mut out = Vec::new();
    // 1. An absolute `.../scenarios/x.toml` names its own root.
    for m in &cfg.models {
        let s = Path::new(&m.scenario);
        if !s.is_absolute() { continue }
        let Some(dir) = s.parent() else { continue };
        if dir.file_name().map(|n| n == "scenarios").unwrap_or(false) {
            if let Some(r) = dir.parent() { out.push(r.to_path_buf()) }
        }
    }
    // 2. The config's own directory, for a self-contained layout that keeps the two together.
    //    NOT the common case -- ~/.config/npu holds only engine.toml -- so it is checked for
    //    `scenarios/` by the caller like every other candidate rather than assumed.
    if let Some(d) = config_path.parent() { out.push(d.to_path_buf()) }
    // 3. A checkout the operator is standing in. Ranked ABOVE the install root, not below: the
    //    installed prefix symlinks `artifacts/` back to whichever checkout was installed from, so
    //    preferring it would hand a second worktree the first one's weights without saying so.
    //    Only taken when it really holds `scenarios/`, which is what makes it safe to try early.
    out.extend(cwd);
    // 4. Where install.sh put it. This is the one that makes `npu diarize foo.wav` work in a plain
    //    shell, which is the case that was broken.
    out.extend(install);
    out
}

/// Is the thing listening on `port` an xdna-engine, or somebody else's server?
///
/// 11434 is a shared default -- ollama and FLM take it too -- so every command that talks to it has
/// to ask, not assume. `preflight_serve` already did; `models` did not, and would print a foreign
/// server's model list as ours. One function so the next caller cannot forget.
fn listener_is_ours(port: u16) -> bool {
    http_get(port, "/healthz").map(|b| b.contains("\"npu\"")).unwrap_or(false)
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
            Err(Tagged(Code::Device,
                "no XDNA2 NPU device at /dev/accel/accel0 (is the amdxdna driver loaded?)".into())
                .into())
        };
    }
    // Something is listening. Ask it who it is rather than assuming.
    if listener_is_ours(port) {
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
    let root = root(&cfg, path)?;
    preflight_artifacts(&cfg, &root)?;
    let (handle, _join) = start(cfg, Box::new(EngineLoader { root }))
        .map_err(|e| Tagged(engine_error(&e), e.to_string()))?;
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
    let root = root(&cfg, path)?;
    // Decode BEFORE loading a model: a bad path or a file with no audio should fail in a second,
    // not after a multi-second model load.
    let samples = npu_runtime::media::decode_file(input).map_err(|e| anyhow!(e))?;
    // Lazy: a one-shot run should load the model it serves, and nothing else.
    let (handle, join) = start_lazy(cfg, Box::new(EngineLoader { root }))
        .map_err(|e| Tagged(engine_error(&e), e.to_string()))?;
    let out = handle.transcribe(model, samples, 16_000)
        .map_err(|e| Tagged(engine_error(&e), e.to_string()));
    handle.shutdown(); let _ = join.join();
    println!("{}", out?.value);
    Ok(())
}

/// `--flag <value>` overrides one `GenerateParams` field; an absent flag keeps the engine default
/// (`GenerateParams::default()`, OpenAI's own defaults) -- never a CLI-chosen substitute.
fn build_params(s: &SamplingArgs) -> Result<npu_engine::GenerateParams, String> {
    let mut p = npu_engine::GenerateParams::default();
    p.temperature = s.temperature;
    p.top_p = s.top_p;
    p.top_k = s.top_k;
    p.presence_penalty = s.presence_penalty;
    p.frequency_penalty = s.frequency_penalty;
    p.repetition_penalty = s.repetition_penalty;
    p.max_tokens = match (s.max_tokens, s.max_completion_tokens) {
        (Some(a), Some(b)) if a != b => return Err(format!(
            "--max-tokens ({a}) and --max-completion-tokens ({b}) disagree; pass one")),
        (Some(a), _) | (None, Some(a)) => Some(a),
        (None, None) => None,
    };
    if !s.stop.is_empty() { p.stop = s.stop.clone(); }
    p.seed = s.seed;
    // Neither flag leaves the template's own default -- `None`, not a defaulted `true`, because for
    // Qwen3 those are the same prompt and only an explicit `false` is an instruction.
    p.enable_thinking = match (s.think, s.no_think) {
        (true, false) => Some(true),
        (false, true) => Some(false),
        _ => None,
    };
    // The SAME check the HTTP surface runs, from the same function -- two surfaces validating
    // separately is how they drift on what they accept.
    p.validate()?;
    Ok(p)
}

/// `npu generate`/`npu chat` run IN-PROCESS (`start_lazy` + `EngineLoader`), the same pattern as
/// `transcribe`/`embed`/`diarize`, rather than talking to a running server over HTTP the way `npu
/// models`/`npu reload` do. Reasons: (1) those two already load their own model one-shot with no
/// server required, which is the point of a CLI generate command existing at all; (2) streaming
/// tokens to stdout is a direct callback from `Handle::generate`'s receiver, whereas an HTTP client
/// here would mean writing an incremental SSE parser in the CLI for no benefit, since the process
/// already IS the engine; (3) `Handle::generate`'s `Prompt::Chat` with real history is exactly what
/// a REPL wants and is not staged through JSON at all this way.
///
/// Drains `rx` to completion either way, so `Cmd::Generate` on the actor side always finishes even
/// under `--no-stream`.
fn drain_generation(rx: std::sync::mpsc::Receiver<StreamItem>, stream: bool) -> Result<String> {
    let mut text = String::new();
    loop {
        match rx.recv() {
            Ok(StreamItem::Text(t)) => {
                if stream { print!("{t}"); std::io::stdout().flush().ok(); }
                else { text.push_str(&t); }
            }
            Ok(StreamItem::Done { .. }) => return Ok(text),
            Ok(StreamItem::Error(e)) => bail!("{e}"),
            Err(_) => bail!("generation ended without a result"),
        }
    }
}

/// Chat-templated by default, raw only on request.
///
/// `Prompt::Raw` sends the bytes verbatim, which is the right semantics for `/v1/completions` and
/// the wrong DEFAULT for a CLI: an instruction-tuned model never sees a turn open, so it never
/// emits the token that closes one and runs to max_tokens producing drift. Observed on
/// `npu generate 'Привет!'` -- 256 tokens of invented statistics homework, in three languages,
/// with a YouTube link. The stop machinery was working; the prompt simply never gave it a stop to
/// find.
fn generate(path: &Path, prompt: &str, model: Option<&str>, sampling: &SamplingArgs,
            no_stream: bool, raw: bool) -> Result<()> {
    quiet_one_shot();
    let cfg = load_cfg(path)?;
    let root = root(&cfg, path)?;
    let (handle, join) = start_lazy(cfg, Box::new(EngineLoader { root }))
        .map_err(|e| Tagged(engine_error(&e), e.to_string()))?;
    // Code::Failure, the documented generic bucket: this closed set has no invalid-argument code,
    // and NoService (2) would tell a caller to start a server for what is a bad flag value.
    let params = build_params(sampling).map_err(|m| Tagged(Code::Failure, m))?;
    let prompt = if raw {
        npu_engine::Prompt::Raw(prompt.to_string())
    } else {
        npu_engine::Prompt::Chat(vec![npu_engine::ChatMessage {
            role: "user".to_string(),
            content: prompt.to_string(),
        }])
    };
    let result = handle.generate(model, prompt, params)
        .map_err(|e| {
            // A base LM with no chat template is a legitimate case; name the flag rather than
            // silently answering a different request than the one that was sent.
            let code = engine_error(&e);
            let msg = e.to_string();
            let tagged = if msg.contains("chat_template") {
                Tagged(code, format!("{msg}\n  this model has no chat template -- use `npu generate --raw`"))
            } else { Tagged(code, msg) };
            anyhow::Error::from(tagged)
        })
        .and_then(|served| drain_generation(served.value, !no_stream));
    handle.shutdown(); let _ = join.join();
    let text = result?;
    if no_stream { print!("{text}"); }
    println!();
    Ok(())
}

/// `opening` is the turn given on the command line. It is answered before stdin is read once, and
/// then the REPL continues from it -- a seeded session, not a one-shot. The one-shot spelling is
/// `npu generate`, which builds the identical single-message `Prompt::Chat`; duplicating it here
/// would add a second name for a command we have and drop the history that makes this one a REPL.
fn chat(path: &Path, opening: Option<&str>, model: Option<&str>, sampling: &SamplingArgs,
        no_stream: bool) -> Result<()> {
    quiet_one_shot();
    let cfg = load_cfg(path)?;
    let root = root(&cfg, path)?;
    let (handle, join) = start_lazy(cfg, Box::new(EngineLoader { root }))
        .map_err(|e| Tagged(engine_error(&e), e.to_string()))?;
    // Code::Failure, the documented generic bucket: this closed set has no invalid-argument code,
    // and NoService (2) would tell a caller to start a server for what is a bad flag value.
    let params = build_params(sampling).map_err(|m| Tagged(Code::Failure, m))?;
    let mut history: Vec<npu_engine::ChatMessage> = Vec::new();
    let stdin = std::io::stdin();
    // Whitespace-only counts as absent: `npu chat ""` must open the REPL, not send an empty turn.
    let mut opening = opening.map(str::trim).filter(|s| !s.is_empty()).map(str::to_string);
    let result = (|| -> Result<()> {
        loop {
            let line = match opening.take() {
                // Echoed at the prompt so the transcript reads the same whether the turn came from
                // argv or the keyboard.
                Some(turn) => { println!("> {turn}"); turn }
                None => {
                    print!("> "); std::io::stdout().flush().ok();
                    let mut line = String::new();
                    if stdin.lock().read_line(&mut line)? == 0 { println!(); return Ok(()); } // Ctrl-D
                    let line = line.trim_end().to_string();
                    if line.is_empty() { continue; }
                    line
                }
            };
            history.push(npu_engine::ChatMessage { role: "user".into(), content: line });
            let served = handle.generate(model, npu_engine::Prompt::Chat(history.clone()), params.clone())
                .map_err(|e| Tagged(engine_error(&e), e.to_string()))?;
            let reply = drain_generation(served.value, !no_stream)?;
            if no_stream { print!("{reply}"); }
            println!();
            history.push(npu_engine::ChatMessage { role: "assistant".into(), content: reply });
        }
    })();
    handle.shutdown(); let _ = join.join();
    result
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
///
/// A malformed value is an ERROR, not a silent 18.0 -- contract rule E004, whose worked example is
/// this flag: `NPU_ASR_MAX_SPAN_S=2O` (letter O) parsed as nothing and returned the default, so an
/// operator who asked for a different window got the old one with no diagnostic. Non-positive and
/// non-finite are rejected at the same place; neither names a span.
fn asr_window_s() -> Result<f32> {
    parse_asr_window(std::env::var_os("NPU_ASR_MAX_SPAN_S").as_deref())
}

/// Split from the read so it is testable without mutating the process environment, which is global
/// and races every other test in this binary.
fn parse_asr_window(raw: Option<&std::ffi::OsStr>) -> Result<f32> {
    let Some(raw) = raw else { return Ok(18.0) };
    match raw.to_str().and_then(|v| v.parse::<f32>().ok()).filter(|s| *s > 0.0 && s.is_finite()) {
        Some(s) => Ok(s),
        // Code::Failure: the closed set has no invalid-argument code, and this is a bad value, not
        // a missing service or model.
        None => Err(Tagged(Code::Failure, format!(
            "NPU_ASR_MAX_SPAN_S: expected a positive number of seconds, got {raw:?}")).into()),
    }
}

#[allow(clippy::too_many_arguments)]
fn transcribe_media(path: &Path, input: &Path, out: Option<&Path>, format: OutFormat,
                    asr: Option<&str>, diar: Option<&str>, only_track: Option<usize>,
                    no_diarize: bool) -> Result<()> {
    quiet_one_shot();
    // Before the device, ffmpeg or diarization: a bad NPU_ASR_MAX_SPAN_S is an operator typo, and
    // reporting it after a model load and a diarize pass is loud but far too late.
    let max_span = asr_window_s()?;
    let cfg = load_cfg(path)?;
    let root = root(&cfg, path)?;
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
    let (handle, join) = start_lazy(cfg, Box::new(EngineLoader { root }))
        .map_err(|e| Tagged(engine_error(&e), e.to_string()))?;
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
                    .map_err(|e| Tagged(engine_error(&e), format!("diarize track {}: {e}", t.ord)))?
                    .value.iter().map(|s| (s.start_s, s.end_s, s.speaker)).collect()
            };
            let n_spk = spans.iter().map(|s| s.2).collect::<std::collections::BTreeSet<_>>().len();
            eprintln!("[npu] {label}: {} span(s), {n_spk} speaker(s)", spans.len());

            // Split turns the ASR cannot hold whole, then transcribe each piece. Without this a
            // long turn returns only its first ~20 s, with no error to notice.
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
                    .map_err(|e| Tagged(engine_error(&e),
                        format!("transcribe {label} [{start_s:.2}-{end_s:.2}]: {e}")))?
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
    let root = root(&cfg, path)?;
    // Lazy, same reason as `transcribe`: a one-shot run loads the model it serves and nothing else.
    let (handle, join) = start_lazy(cfg, Box::new(EngineLoader { root }))
        .map_err(|e| Tagged(engine_error(&e), e.to_string()))?;
    let bytes = std::fs::read(wav).with_context(|| format!("read {}", wav.display()))?;
    let samples = http::parse::parse_wav_i16(&bytes)
        .ok_or_else(|| anyhow!("bad wav (need 16k mono 16-bit)"))?;
    let out = handle.diarize(model, samples, 16_000)
        .map_err(|e| Tagged(engine_error(&e), e.to_string()));
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
    let root = root(&cfg, path)?;
    // Lazy: `npu embed` against an ASR-only config used to pay a full parakeet load before it could
    // say there was no embed model at all.
    let (handle, join) = start_lazy(cfg, Box::new(EngineLoader { root }))
        .map_err(|e| Tagged(engine_error(&e), e.to_string()))?;
    let out = handle.embed(model, text).map_err(|e| Tagged(engine_error(&e), e.to_string()));
    handle.shutdown(); let _ = join.join();
    let v = out?.value;
    let arr = v.iter().map(|x| format!("{x}")).collect::<Vec<_>>().join(",");
    println!("[{arr}]");
    Ok(())
}

/// The configured models, and -- when the service is running -- what it currently has resident.
///
/// Never contacts the service. The CLI and the service must not depend on each other: every other
/// one-shot command drives the engine directly, and this one used to need a running HTTP server to
/// say anything at all.
///
/// Live state comes from a FILE the service publishes, not a socket or the port. `RuntimeDirectory=`
/// has systemd create that directory on start and remove it on stop, so its presence is the liveness
/// signal -- no probe, no handshake, no timeout, and no way to mistake ollama on the shared 11434
/// for us. A wedged service cannot hang this command, because reading bytes is not connecting; it
/// shows the last published state and how old it is, and lets the reader judge.
fn models(path: &Path, as_json: bool, port: Option<u16>) -> Result<()> {
    let cfg = load_cfg(path)?;
    let live = read_live_status(port.unwrap_or(cfg.server.port));

    if as_json {
        let rows: Vec<_> = cfg.models.iter().map(|m| {
            let l = live.as_ref().and_then(|(_, v)| find_live(v, &m.name));
            serde_json::json!({
                "id": m.name, "scenario": m.scenario,
                "state": l.and_then(|x| x.get("state").and_then(|s| s.as_str())).unwrap_or("unknown"),
                // Both, because they are allowed to differ: the config is desired state and the
                // service only adopts it on reload. That gap is the thing worth reporting.
                "pinned": m.resident,
                "live_pinned": l.and_then(|x| x.get("pinned").and_then(|s| s.as_bool())),
            })
        }).collect();
        let age = live.as_ref().map(|(a, _)| serde_json::json!(a));
        println!("{}", serde_json::json!({"source": path.display().to_string(),
                                          "live_age_s": age, "data": rows}));
        return Ok(());
    }

    println!("{:<22} {:<9} {:<8} {:<5}  {}", "NAME", "STATE", "KIND", "PIN", "SCENARIO");
    let mut drifted = false;
    for m in &cfg.models {
        let l = live.as_ref().and_then(|(_, v)| find_live(v, &m.name));
        let f = |k: &str| l.and_then(|x| x.get(k).and_then(|s| s.as_str())).unwrap_or("-").to_string();
        let pin = pin_cell(m.resident, l.and_then(|x| x.get("pinned")).and_then(|p| p.as_bool()));
        if pin.ends_with('*') { drifted = true; }
        println!("{:<22} {:<9} {:<8} {:<5}  {}", m.name, f("state"), f("kind"), pin, m.scenario);
    }
    match &live {
        Some((age, _)) => println!("\n(live state as of {age}s ago)"),
        None => println!("\n(service not running -- configured models only)"),
    }
    if drifted {
        println!("* the running server has a different pin than the config -- run `npu reload`");
    }
    Ok(())
}

/// The published status and its age in seconds, or `None` when nothing is serving.
///
/// Liveness comes from the PID recorded in the file, not from the directory existing. The first
/// version trusted the directory -- `RuntimeDirectory=` with `RuntimeDirectoryPreserve=no` is
/// documented to be removed on stop -- and that is FALSE as observed here: after a clean
/// `systemctl --user stop`, `/run/user/1000/xdna-engine` survived with its `status.json` intact, so
/// the command reported a dead service as live. Reproduced deliberately before changing this.
///
/// The pid check costs one `stat` of `/proc/<pid>`, cannot hang, and cannot be fooled by a leftover
/// directory -- which the directory test could not say the same of.
fn read_live_status(want_port: u16) -> Option<(u64, serde_json::Value)> {
    let p = npu_runtime::status_file::path()?;
    let body = std::fs::read_to_string(p).ok()?;
    let v: serde_json::Value = serde_json::from_str(&body).ok()?;
    let pid = v.get("pid")?.as_u64()?;
    if !std::path::Path::new(&format!("/proc/{pid}")).exists() {
        return None; // stale file from a process that is gone
    }
    // The path is per-USER, so another engine on another port publishes here too -- a test instance,
    // a parallel session. Without this the command reports someone else's models as ours, which it
    // did today. Same check `preflight_serve` makes about who holds a socket.
    //
    // ABSENT is not WRONG. A service running an older binary publishes no `port` at all, and the
    // first version of this check read that as a mismatch and called a live service down -- during
    // exactly the rolling upgrade where the two binaries differ. An unknown port falls back to the
    // pid, which is weaker identification but an honest one.
    match v.get("port").and_then(|p| p.as_u64()) {
        Some(p) if p != want_port as u64 => return None,
        _ => {}
    }
    let written = v.get("written_unix")?.as_u64()?;
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).ok()?.as_secs();
    Some((now.saturating_sub(written), v))
}

/// The PIN column. `*` marks a config pin the running server has not adopted yet -- which is the
/// normal state between `npu config pin` and `npu reload`, and the one thing a pin column has to be
/// able to say. A server too old to publish `pinned` reports `None`, and gets the config's answer
/// without a drift marker rather than a fabricated disagreement.
fn pin_cell(want: bool, live: Option<bool>) -> String {
    let w = if want { "yes" } else { "no" };
    match live {
        Some(l) if l != want => format!("{w}*"),
        _ => w.to_string(),
    }
}

fn find_live<'a>(doc: &'a serde_json::Value, name: &str) -> Option<&'a serde_json::Value> {
    doc.get("models")?.get("data")?.as_array()?
        .iter().find(|m| m.get("id").and_then(|i| i.as_str()) == Some(name))
}

fn reload(path: &Path, port: Option<u16>) -> Result<()> {
    let port = resolve_port(path, port)?;
    let body = http_post(port, "/admin/reload", "")
        .context(Tagged(Code::NoService, "reload (is the server running?)".into()))?;
    println!("{body}");
    Ok(())
}

/// `npu load` / `npu unload` talk to the SERVICE, not the device.
///
/// Every other one-shot command drives the engine in-process, but residency is a property of the
/// running server's registry -- the thing that owns `max_resident`, eviction and the idle sweep.
/// Loading a model into this process would take its own hardware context and change nothing the
/// server can see, which is the opposite of what was asked.
fn load_model(path: &Path, model: &str, port: Option<u16>) -> Result<()> {
    let port = resolve_port(path, port)?;
    let body = http_post(port, &format!("/admin/models/{model}/load"), "")
        .context(Tagged(Code::NoService, "load (is the server running?)".into()))?;
    let v: serde_json::Value = serde_json::from_str(&body)
        .with_context(|| format!("unexpected reply: {body}"))?;
    if let Some(e) = v.get("error").and_then(|e| e.as_str()) { return Err(admin_err(e, port)) }
    let n = |k: &str| v.get(k).and_then(|x| x.as_u64()).unwrap_or(0);
    println!("{model}: {}  ({}/{} resident)",
        if v.get("loaded").and_then(|x| x.as_bool()) == Some(true) { "loaded" } else { "already resident" },
        n("resident"), n("max_resident"));
    // Say when the ceiling the operator may have just set is not bounding anything. An unenforceable
    // limit that looks enforced is the failure `memory_ceiling_mb`'s own doc comment warns about.
    let unweighed: Vec<&str> = v.get("unweighed").and_then(|x| x.as_array())
        .map(|a| a.iter().filter_map(|s| s.as_str()).collect()).unwrap_or_default();
    if !unweighed.is_empty() {
        eprintln!("note: memory_ceiling_mb is not bounding these -- they report no measured \
                   footprint: {}", unweighed.join(" "));
    }
    Ok(())
}

fn unload_model(path: &Path, model: &str, port: Option<u16>) -> Result<()> {
    let port = resolve_port(path, port)?;
    let body = http_post(port, &format!("/admin/models/{model}/unload"), "")
        .context(Tagged(Code::NoService, "unload (is the server running?)".into()))?;
    let v: serde_json::Value = serde_json::from_str(&body)
        .with_context(|| format!("unexpected reply: {body}"))?;
    if let Some(e) = v.get("error").and_then(|e| e.as_str()) { return Err(admin_err(e, port)) }
    println!("{model}: {}", match v.get("released").and_then(|x| x.as_bool()) {
        Some(true) => "released",
        _ => "was not resident",
    });
    Ok(())
}

/// An admin route's error, with the one case that is not about the request spelled out.
///
/// `http_req` keeps only the body, so a 404 arrives as this server's generic `not found` and reads
/// as though the MODEL was not found -- which is the wrong thing entirely, and is what a CLI newer
/// than the service it is talking to hits every time.
fn admin_err(e: &str, port: u16) -> anyhow::Error {
    if e == "not found" {
        return anyhow!("the server on port {port} has no load/unload route -- it is older than \
                        this CLI.{}", restart_hint(port));
    }
    anyhow!("{e}")
}

/// How to restart whatever is serving `port` -- RESOLVED from the running process, not guessed.
///
/// The first version hardcoded `systemctl --user restart npu-asr`, and install.sh had just
/// superseded that unit, so the advice named a service the box does not have. The unit name is
/// readable: systemd puts it in the process's cgroup path. So is the more useful fact underneath,
/// which is why the server is stale at all -- `install` replaces the binary's inode, and a server
/// started before that keeps running the old one, which the kernel marks `(deleted)`.
///
/// Returns "" rather than a guess when the process cannot be identified. Silence beats wrong advice.
fn restart_hint(port: u16) -> String {
    let Some(pid) = serving_pid(port) else { return String::new() };
    let stale = std::fs::read_link(format!("/proc/{pid}/exe"))
        .map(|p| p.to_string_lossy().ends_with("(deleted)")).unwrap_or(false);
    let why = if stale {
        " It is running a binary that has already been replaced on disk (/proc/<pid>/exe is deleted),           so this is an install that has not been restarted into."
    } else { "" };
    match unit_of(pid) {
        Some(unit) => format!("{why} Restart it: systemctl --user restart {}",
                              unit.trim_end_matches(".service")),
        // Not under a unit: started by hand, so there is no restart command to offer.
        None => format!("{why} It was started outside systemd (pid {pid}); restart it the way it                          was started."),
    }
}

/// The pid the running server published, or `None` when nothing is serving this port.
fn serving_pid(port: u16) -> Option<u64> {
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(npu_runtime::status_file::path()?).ok()?).ok()?;
    match v.get("port").and_then(|p| p.as_u64()) {
        Some(p) if p != port as u64 => return None,
        _ => {}
    }
    let pid = v.get("pid")?.as_u64()?;
    std::path::Path::new(&format!("/proc/{pid}")).exists().then_some(pid)
}

/// The systemd unit owning `pid`, from its cgroup path -- the last `*.service` component of
/// `/user.slice/.../app.slice/xdna-engine.service`. `None` when the process is not under one.
fn unit_of(pid: u64) -> Option<String> {
    std::fs::read_to_string(format!("/proc/{pid}/cgroup")).ok()?
        .split(['/', '\n'])
        .filter(|c| c.ends_with(".service"))
        .last()
        .map(str::to_string)
}

fn bake(path: &Path, name: &str) -> Result<()> {
    let cfg = load_cfg(path)?;
    let m = cfg.find(name)
        .ok_or_else(|| Tagged(Code::NoModel, format!("unknown model {name:?} in config")))?;
    let sc = npu_engine::config::ScenarioConfig::load(Path::new(&m.scenario))
        .with_context(|| format!("scenario {}", m.scenario))?;
    match sc.artifacts.model_spec()? {
        Some(spec) => { let p = spec.ensure_checkpoint(&root(&cfg, path)?, false)?; println!("baked: {}", p.display()); }
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
    let root = load_cfg(path).ok().and_then(|c| root(&c, path).ok())
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

/// Every mutation goes through `ConfigDoc`, which edits the FILE rather than round-tripping a
/// deserialized `Config` back through the serializer. The struct does not carry comments, so the
/// old path silently deleted every one of them -- including the ones the engine's own generated
/// config ships with.
fn config_cmd(path: &Path, action: &ConfigCmd) -> Result<()> {
    if let ConfigCmd::Show = action {
        print!("{}", render(&load_cfg(path)?));
        return Ok(());
    }
    let mut doc = npu_runtime::ConfigDoc::load(path).map_err(|e| anyhow!(e))?;
    // What to print once the write lands. Held rather than printed inline so a command that then
    // fails validation says nothing, instead of reporting a change it did not make.
    let note = match action {
        ConfigCmd::Show => unreachable!("handled above"),
        ConfigCmd::AddModel { name, scenario } => {
            doc.add_model(name, scenario).map_err(|e| anyhow!(e))?;
            format!("model {name} -> {scenario}")
        }
        ConfigCmd::RemoveModel { name } => {
            if !doc.remove_model(name).map_err(|e| anyhow!(e))? {
                return Err(Tagged(Code::NoModel, format!("unknown model {name:?} (not in the config)")).into());
            }
            format!("removed model {name}")
        }
        ConfigCmd::Pin { model } | ConfigCmd::Unpin { model } => {
            let on = matches!(action, ConfigCmd::Pin { .. });
            // Refuse rather than write: a pin on a name the config does not have is a typo, and
            // there is nothing in the file for the key to attach to.
            if !doc.set_resident(model, on).map_err(|e| anyhow!(e))? {
                return Err(Tagged(Code::NoModel, format!("unknown model {model:?} (not in the config)")).into());
            }
            if on { format!("pinned {model} resident") } else { format!("unpinned {model}") }
        }
        ConfigCmd::Set { key, value } => {
            doc.set_server(key, value).map_err(|e| anyhow!(e))?;
            format!("server.{key} = {value}")
        }
        ConfigCmd::SetDefault { capability, model } => match Capability::from_name(capability) {
            Some(cap) => { doc.set_default(cap, model); format!("default {capability} = {model}") }
            None => return Err(anyhow!("unknown capability {capability:?} (one of: {})",
                Capability::ALL.iter().map(|c| c.0).collect::<Vec<_>>().join("|"))),
        },
    };
    let cfg = doc.save(path).map_err(|e| anyhow!(e))?;
    println!("{note}  [{}]", path.display());
    // Warn on the same conditions `npu config show` does, so an edit that creates one is caught
    // where it is made rather than at the next boot.
    if let Some(w) = cfg.pin_overcommit() { eprintln!("WARNING: {w}"); }
    if let Some(w) = pins_behind_admission(&cfg) { eprintln!("WARNING: {w}"); }
    // The file is desired state; the running service only picks it up when asked.
    if matches!(action, ConfigCmd::Pin { .. } | ConfigCmd::Unpin { .. } | ConfigCmd::Set { .. }) {
        println!("run `npu reload` to apply this to a running server");
    }
    Ok(())
}

/// Every registered `NPU_*`/related env var against the LIVE process environment: whether it is
/// currently set, its raw value if so, its truth semantics, and what it does.
///
/// `npu_runtime::env_flags::FLAGS` is the single source; this only renders it. Reads with
/// `var_os` (not `var`) so presence is detected independent of UTF-8 validity, matching the
/// `Presence`/`IsOk`/`NotZero` sites themselves.
fn flags_cmd(as_json: bool) -> Result<()> {
    if as_json {
        let rows: Vec<_> = npu_runtime::env_flags::FLAGS.iter().map(|f| {
            let raw = std::env::var_os(f.name);
            serde_json::json!({
                "name": f.name,
                "owner": f.owner,
                "site": f.site,
                "semantics": f.semantics.code(),
                "semantics_rule": f.semantics.describe(),
                "default": f.default,
                "set": raw.is_some(),
                "value": raw.map(|v| v.to_string_lossy().into_owned()),
                "doc": f.doc,
            })
        }).collect();
        println!("{}", serde_json::to_string_pretty(&rows)?);
        return Ok(());
    }
    for f in npu_runtime::env_flags::FLAGS {
        let raw = std::env::var_os(f.name);
        let (source, value) = match &raw {
            Some(v) => ("env", v.to_string_lossy().into_owned()),
            None => ("default", format!("(default: {})", f.default)),
        };
        println!("{:<32} {:<8} {:<28} {:<10} {}", f.name, source, value, f.semantics.code(), f.owner);
        println!("    {}", f.doc);
    }
    Ok(())
}

/// `Some(message)` when a pinned model sits behind enough unpinned ones that boot admission will
/// not reach it.
///
/// A pin means exempt-from-eviction, NOT entitled to a slot: `reconcile` walks the models in CONFIG
/// ORDER and stops admitting at `max_resident`, and `Config::pinned()` has no caller there. So the
/// pin is real but arrives late -- the model loads on demand and then stays. Worth saying, because
/// the config states an intent the runtime partly declines and used to do so in silence.
fn pins_behind_admission(cfg: &Config) -> Option<String> {
    let late: Vec<&str> = cfg.models.iter().enumerate()
        .filter(|(i, m)| m.resident && *i >= cfg.server.max_resident)
        .map(|(_, m)| m.name.as_str()).collect();
    (!late.is_empty()).then(|| format!(
        "pinned but not admitted at boot: {}. A pin exempts a model from eviction and the idle \
         sweep; it does not win a slot. Admission is the first {} models in config order, so these \
         load on demand (and then stay). Move them earlier, or raise max_resident.",
        late.join(" "), cfg.server.max_resident))
}

/// Human-readable config summary (pure, testable).
fn render(cfg: &Config) -> String {
    // The ceiling's scope is printed with it. It bounds summed DEVICE-BO bytes, and a model that
    // reports no footprint is exempt -- which today is every shipped model, so the number alone
    // reads as a guarantee it does not give. The wording stays true once footprints are measured,
    // rather than being an "inert" note that would go stale silently. `npu status` has the live
    // per-model answer.
    let mut s = format!("port {}  max_resident {}  memory_ceiling_mb {} (device BOs; \
                         models reporting no footprint are exempt)\n",
        cfg.server.port, cfg.server.max_resident, cfg.server.memory_ceiling_mb);
    s.push_str(&format!("residency: idle_unload_s {}  idle_release_s {}  sweep_interval_s {}  evict_policy {}\n",
        cfg.server.idle_unload_s, cfg.server.idle_release_s, cfg.server.sweep_interval_s,
        match cfg.server.evict_policy { EvictPolicy::Lru => "lru", EvictPolicy::None => "none" }));
    let pins = cfg.pinned().map(|m| m.name.as_str()).collect::<Vec<_>>();
    s.push_str(&format!("pinned resident: {}\n",
        if pins.is_empty() { "(none)".to_string() } else { pins.join(" ") }));
    // Surface the overcommit here rather than only at load time: the config summary is where an
    // operator looks BEFORE a refusal, not after one.
    if let Some(w) = cfg.pin_overcommit() { s.push_str(&format!("WARNING: {w}\n")); }
    if let Some(w) = pins_behind_admission(cfg) { s.push_str(&format!("WARNING: {w}\n")); }
    let defaults = cfg.defaults.0.iter().map(|(c, m)| format!("{c}={m}")).collect::<Vec<_>>();
    s.push_str(&format!("defaults: {}\n",
        if defaults.is_empty() { "(none)".to_string() } else { defaults.join(" ") }));
    if cfg.models.is_empty() { s.push_str("models: (none)\n"); }
    // The pin is marked on the model's own line as well as summarised above: the summary answers
    // "what is pinned", the marker answers "is THIS one pinned", and the second question is the one
    // asked while reading down a list of eight.
    for m in &cfg.models {
        s.push_str(&format!("model {} -> {}{}\n", m.name, m.scenario,
            if m.resident { "  [pinned]" } else { "" }));
    }
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

    /// The listing has to stay splittable: `npu models | awk '{print $1}'` is the obvious use, and a
    /// scenario path can contain no spaces while a model name never does -- so name first, path last.
    #[test]
    fn model_listing_is_splittable_by_column() {
        let cfg = npu_runtime::config::Config::from_str(
            "[[model]]\nname = \"whisper-turbo\"\nscenario = \"scenarios/asr-whisper-turbo.toml\"\n",
        )
        .unwrap();
        let m = &cfg.models[0];
        let line = format!("{:<22}  {}", m.name, m.scenario);
        let f: Vec<&str> = line.split_whitespace().collect();
        assert_eq!(f[0], "whisper-turbo");
        assert_eq!(f[1], "scenarios/asr-whisper-turbo.toml");
        assert_eq!(f.len(), 2, "a row must be exactly two fields: {line:?}");
    }

    /// The one thing a PIN column must be able to say: the config and the running server disagree,
    /// which is the normal state between `npu config pin` and `npu reload`.
    #[test]
    fn pin_cell_marks_config_and_server_disagreeing() {
        assert_eq!(pin_cell(true, Some(true)), "yes");
        assert_eq!(pin_cell(false, Some(false)), "no");
        assert_eq!(pin_cell(true, Some(false)), "yes*", "pinned in the config, not yet reloaded");
        assert_eq!(pin_cell(false, Some(true)), "no*", "unpinned in the config, not yet reloaded");
        // A server too old to publish `pinned`, or none running: report the config, invent nothing.
        assert_eq!(pin_cell(true, None), "yes");
        assert_eq!(pin_cell(false, None), "no");
    }

    fn pin_cfg(max_resident: usize, models: &[(&str, bool)]) -> npu_runtime::config::Config {
        npu_runtime::config::Config {
            server: ServerCfg { max_resident, ..Default::default() },
            models: models.iter().map(|(n, r)| ModelCfg {
                name: (*n).into(), scenario: "x".into(), resident: *r }).collect(),
            ..Default::default()
        }
    }

    /// A pin is exempt-from-eviction, not entitled to a slot. Saying so where the pin is SET beats
    /// leaving it to be discovered in `/v1/models` after something has already gone wrong.
    #[test]
    fn a_pin_boot_admission_cannot_reach_is_reported() {
        assert!(pins_behind_admission(&pin_cfg(1, &[("a", false), ("b", true)]))
            .is_some_and(|w| w.contains('b')), "b is pinned but second with one slot");
        assert!(pins_behind_admission(&pin_cfg(2, &[("a", false), ("b", true)])).is_none(),
            "two slots reach b, so there is nothing to warn about");
        assert!(pins_behind_admission(&pin_cfg(1, &[("b", true), ("a", false)])).is_none(),
            "a pin first in config order is admitted; order is what decides, not the pin");
        assert!(pins_behind_admission(&pin_cfg(1, &[("a", false), ("b", false)])).is_none(),
            "an unpinned deferral is ordinary capacity, not a declined intent");
    }

    #[test]
    fn render_marks_which_models_are_pinned() {
        let out = render(&pin_cfg(4, &[("a", false), ("b", true)]));
        assert!(out.contains("model b -> x  [pinned]"), "{out}");
        assert!(out.contains("model a -> x\n"), "an unpinned model gets no marker: {out}");
        assert!(out.contains("pinned resident: b"), "{out}");
    }

    /// `npu config` edits the file a human wrote. Every verb has to leave the rest of it alone.
    #[test]
    fn config_verbs_edit_in_place_without_destroying_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("engine.toml");
        std::fs::write(&p, "# keep me\n[server]\nmax_resident = 2\n\n[[model]]\nname = \"a\"\nscenario = \"s.toml\"\n").unwrap();

        config_cmd(&p, &ConfigCmd::Pin { model: "a".into() }).unwrap();
        assert!(npu_runtime::config::Config::load(&p).unwrap().find("a").unwrap().resident);

        config_cmd(&p, &ConfigCmd::Set { key: "idle_unload_s".into(), value: "0".into() }).unwrap();
        let cfg = npu_runtime::config::Config::load(&p).unwrap();
        assert_eq!(cfg.server.idle_unload(), None, "0 is how idle unload is switched off");
        assert_eq!(cfg.server.max_resident, 2, "an unnamed key must not move");

        // Re-pointing a scenario must not silently unpin.
        config_cmd(&p, &ConfigCmd::AddModel { name: "a".into(), scenario: "t.toml".into() }).unwrap();
        let cfg = npu_runtime::config::Config::load(&p).unwrap();
        assert_eq!(cfg.find("a").unwrap().scenario, "t.toml");
        assert!(cfg.find("a").unwrap().resident);

        config_cmd(&p, &ConfigCmd::Unpin { model: "a".into() }).unwrap();
        assert!(!npu_runtime::config::Config::load(&p).unwrap().find("a").unwrap().resident);

        assert!(std::fs::read_to_string(&p).unwrap().contains("# keep me"),
            "every verb has to preserve the comments");
    }

    #[test]
    fn config_verbs_refuse_a_name_or_key_they_cannot_honour() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("engine.toml");
        std::fs::write(&p, "[[model]]\nname = \"a\"\nscenario = \"s\"\n").unwrap();
        let before = std::fs::read_to_string(&p).unwrap();
        assert!(config_cmd(&p, &ConfigCmd::Pin { model: "nope".into() }).is_err());
        assert!(config_cmd(&p, &ConfigCmd::Unpin { model: "nope".into() }).is_err());
        assert!(config_cmd(&p, &ConfigCmd::RemoveModel { name: "nope".into() }).is_err());
        assert!(config_cmd(&p, &ConfigCmd::Set { key: "max_resident".into(), value: "-1".into() }).is_err());
        assert_eq!(std::fs::read_to_string(&p).unwrap(), before, "a refused command writes nothing");
    }

    /// The unit name must be READ, not guessed. The first version hardcoded `npu-asr`, which
    /// install.sh had superseded, so the advice named a service that does not exist.
    #[test]
    fn unit_of_reads_the_service_from_a_cgroup_path() {
        let parse = |s: &str| s.split(['/', '\n']).filter(|c| c.ends_with(".service"))
            .last().map(str::to_string);
        assert_eq!(parse("0::/user.slice/user-1000.slice/user@1000.service/app.slice/xdna-engine.service"),
                   Some("xdna-engine.service".to_string()),
                   "the LAST .service component is the unit; user@1000.service is the manager");
        assert_eq!(parse("0::/user.slice/user-1000.slice/user@1000.service/app.slice/npu-asr.service"),
                   Some("npu-asr.service".to_string()));
        // Started by hand: no unit, so there is no restart command to offer and none is invented.
        assert_eq!(parse("0::/user.slice/user-1000.slice/session-3.scope"), None);
        assert_eq!(parse(""), None);
    }

    #[test]
    fn build_params_with_no_flags_is_the_engine_default() {
        let s = cli_def::SamplingArgs {
            temperature: None, top_p: None, top_k: None, max_tokens: None,
            max_completion_tokens: None, presence_penalty: None, frequency_penalty: None,
            repetition_penalty: None, stop: vec![], seed: None, think: false, no_think: false,
        };
        let p = build_params(&s).unwrap();
        // Everything UNSET, exactly like an HTTP body with no sampling fields -- the CLI must not
        // pre-fill a value, or it would silently outrank the scenario and checkpoint tiers.
        assert_eq!(p.temperature, None);
        assert_eq!(p.top_p, None);
        assert_eq!(p.top_k, None);
        assert_eq!(p.max_tokens, None);
        assert_eq!(p.presence_penalty, None);
        assert_eq!(p.frequency_penalty, None);
        assert_eq!(p.repetition_penalty, None);
        assert!(p.stop.is_empty());
        assert_eq!(p.seed, None);
    }

    /// The CLI and the HTTP surface must accept the SAME sampling set. This is the parity guard:
    /// every field on `GenerateParams` that a caller can set has a flag here.
    #[test]
    fn every_sampling_flag_reaches_generate_params() {
        let s = cli_def::SamplingArgs {
            temperature: Some(0.4), top_p: Some(0.9), top_k: Some(50), max_tokens: None,
            max_completion_tokens: Some(64), presence_penalty: Some(0.5),
            frequency_penalty: Some(-0.5), repetition_penalty: Some(1.2),
            stop: vec!["END".into()], seed: Some(7), think: false, no_think: true,
        };
        let p = build_params(&s).unwrap();
        assert_eq!(p.presence_penalty, Some(0.5));
        assert_eq!(p.frequency_penalty, Some(-0.5));
        assert_eq!(p.repetition_penalty, Some(1.2));
        assert_eq!(p.max_tokens, Some(64), "--max-completion-tokens is an alias for --max-tokens");
    }

    #[test]
    fn the_two_max_token_spellings_must_agree_when_both_are_given() {
        let mk = |a, b| cli_def::SamplingArgs {
            temperature: None, top_p: None, top_k: None, max_tokens: a,
            max_completion_tokens: b, presence_penalty: None, frequency_penalty: None,
            repetition_penalty: None, stop: vec![], seed: None, think: false, no_think: false,
        };
        assert_eq!(build_params(&mk(Some(8), Some(8))).unwrap().max_tokens, Some(8));
        let err = build_params(&mk(Some(8), Some(9))).unwrap_err();
        assert!(err.contains("disagree"), "{err}");
    }

    /// Out-of-range values are rejected by the SAME `validate()` the HTTP surface calls, so the two
    /// cannot drift on what they accept. Silently degenerate is the behaviour being removed here:
    /// a negative temperature read as greedy and a top_p above 1 disabled the filter.
    #[test]
    fn out_of_range_sampling_values_are_rejected() {
        let mk = |t, tp| cli_def::SamplingArgs {
            temperature: t, top_p: tp, top_k: None, max_tokens: None,
            max_completion_tokens: None, presence_penalty: None, frequency_penalty: None,
            repetition_penalty: None, stop: vec![], seed: None, think: false, no_think: false,
        };
        assert!(build_params(&mk(Some(-1.0), None)).unwrap_err().contains("temperature"));
        assert!(build_params(&mk(Some(3.0), None)).unwrap_err().contains("temperature"));
        assert!(build_params(&mk(None, Some(1.5))).unwrap_err().contains("top_p"));
        assert!(build_params(&mk(Some(0.0), Some(1.0))).is_ok(), "the endpoints are legal");
        assert!(build_params(&mk(Some(2.0), Some(0.0))).is_ok());
    }

    #[test]
    fn build_params_applies_every_flag() {
        let s = cli_def::SamplingArgs {
            temperature: Some(0.4), top_p: Some(0.9), top_k: Some(50), max_tokens: Some(64),
            max_completion_tokens: None, presence_penalty: None, frequency_penalty: None,
            repetition_penalty: None,
            stop: vec!["END".into(), "STOP".into()], seed: Some(7), think: false, no_think: true,
        };
        let p = build_params(&s).unwrap();
        assert_eq!(p.temperature, Some(0.4));
        assert_eq!(p.top_p, Some(0.9));
        assert_eq!(p.top_k, Some(50));
        assert_eq!(p.max_tokens, Some(64));
        assert_eq!(p.stop, vec!["END".to_string(), "STOP".to_string()]);
        assert_eq!(p.seed, Some(7));
        assert_eq!(p.enable_thinking, Some(false));
    }

    /// Neither flag must leave `None`, not a substituted `true`. Qwen3's template branches on
    /// `enable_thinking is defined and ... is false`, so `None` and `Some(true)` render the same
    /// prompt -- which is exactly why a defaulted `true` would pass every prompt-level check and
    /// still be the wrong value to hand a model whose template default is not thinking-on.
    #[test]
    fn think_flags_map_to_a_tri_state_and_neither_flag_leaves_the_template_default() {
        let base = |think, no_think| cli_def::SamplingArgs {
            max_completion_tokens: None, presence_penalty: None, frequency_penalty: None,
            repetition_penalty: None,
            temperature: None, top_p: None, top_k: None, max_tokens: None,
            stop: vec![], seed: None, think, no_think,
        };
        assert_eq!(build_params(&base(false, false)).unwrap().enable_thinking, None);
        assert_eq!(build_params(&base(true, false)).unwrap().enable_thinking, Some(true));
        assert_eq!(build_params(&base(false, true)).unwrap().enable_thinking, Some(false));
    }

    /// `--think` and `--no-think` override each other rather than erroring, so the last one on the
    /// command line wins -- the shape a shell alias or a wrapper script needs.
    #[test]
    fn think_and_no_think_are_last_one_wins() {
        let cli = Cli::try_parse_from(["npu", "generate", "hi", "--think", "--no-think"]).unwrap();
        match &cli.cmd {
            Cmd::Generate { sampling, .. } => {
                assert!(sampling.no_think && !sampling.think);
                assert_eq!(build_params(sampling).unwrap().enable_thinking, Some(false));
            }
            _ => panic!("expected Cmd::Generate"),
        }
        let cli = Cli::try_parse_from(["npu", "chat", "--no-think", "--think"]).unwrap();
        match &cli.cmd {
            Cmd::Chat { sampling, .. } => {
                assert!(sampling.think && !sampling.no_think);
                assert_eq!(build_params(sampling).unwrap().enable_thinking, Some(true));
            }
            _ => panic!("expected Cmd::Chat"),
        }
    }

    /// `generate`'s clap definition: a free-text positional with hyphen values allowed (prose starts
    /// with `-` routinely), plus the shared sampling flags.
    #[test]
    fn generate_cli_parses_a_hyphen_leading_prompt_and_sampling_flags() {
        let cli = Cli::try_parse_from([
            "npu", "generate", "- a bullet point", "--temperature", "0.5", "--stop", "END",
            "--stop", "STOP", "--seed", "3", "--no-stream",
        ]).expect("must parse");
        match cli.cmd {
            Cmd::Generate { prompt, sampling, no_stream, model, raw } => {
                assert_eq!(prompt, "- a bullet point");
                assert_eq!(sampling.temperature, Some(0.5));
                assert_eq!(sampling.stop, vec!["END".to_string(), "STOP".to_string()]);
                assert_eq!(sampling.seed, Some(3));
                assert!(no_stream);
                assert_eq!(model, None);
                // Chat-templated unless asked otherwise. Raw is `/v1/completions` semantics and
                // the wrong CLI default: a chat-tuned model never sees a turn open, so it never
                // emits the token that closes one and runs to max_tokens.
                assert!(!raw, "generate must default to the chat template, not raw continuation");
            }
            _ => panic!("expected Cmd::Generate"),
        }
    }

    /// The opening turn is optional and positional: `npu chat "hi"` answers immediately, `npu chat`
    /// alone still opens an empty REPL. `allow_hyphen_values` for the same reason `generate` has it.
    #[test]
    fn chat_takes_an_optional_opening_turn() {
        let cli = Cli::try_parse_from(["npu", "chat"]).expect("a bare chat must still parse");
        match cli.cmd {
            Cmd::Chat { prompt, .. } => assert_eq!(prompt, None),
            _ => panic!("expected Cmd::Chat"),
        }
        let cli = Cli::try_parse_from(["npu", "chat", "- an opening turn", "--seed", "3"])
            .expect("an opening turn must parse");
        match cli.cmd {
            Cmd::Chat { prompt, sampling, .. } => {
                assert_eq!(prompt.as_deref(), Some("- an opening turn"));
                assert_eq!(sampling.seed, Some(3));
            }
            _ => panic!("expected Cmd::Chat"),
        }
    }

    /// E004: a `Value` flag must fail loudly. The worked example in the contract is this flag --
    /// `NPU_ASR_MAX_SPAN_S=2O` (letter O, not zero) parsed as nothing and silently became 18.0.
    #[test]
    fn a_malformed_asr_window_is_an_error_not_the_default() {
        use std::ffi::OsStr;
        assert_eq!(parse_asr_window(None).unwrap(), 18.0, "unset means the default");
        assert_eq!(parse_asr_window(Some(OsStr::new("24.5"))).unwrap(), 24.5);
        for bad in ["2O", "", "0", "-3", "inf", "nan", "18s"] {
            let e = parse_asr_window(Some(OsStr::new(bad)))
                .expect_err(&format!("{bad:?} must not silently become 18.0"));
            assert_eq!(exit::of(&e), Code::Failure);
            assert!(e.to_string().contains("NPU_ASR_MAX_SPAN_S"), "the message must name the flag");
        }
    }

    fn cfg_with(scenarios: &[&str]) -> Config {
        Config {
            server: ServerCfg::default(),
            defaults: Defaults::default(),
            models: scenarios.iter().enumerate()
                .map(|(i, s)| ModelCfg { name: format!("m{i}"), scenario: (*s).into(), resident: false })
                .collect(),
        }
    }

    /// The regression: a config with RELATIVE scenario paths -- which is what `npu config` writes
    /// and what every installed engine.toml holds -- must still produce a usable root.
    ///
    /// The old chain had exactly one branch between the env var and the working directory, and it
    /// required an ABSOLUTE scenario path. So it never fired for a real config, and `npu diarize`
    /// outside a checkout died on `<cwd>/scenarios/...: No such file`. Nothing tested it, which is
    /// how a dead branch stayed in a shipped fix.
    #[test]
    fn a_relative_config_still_finds_the_install_root() {
        let install = tempfile::tempdir().unwrap();
        std::fs::create_dir(install.path().join("scenarios")).unwrap();
        let cfgdir = tempfile::tempdir().unwrap();          // holds engine.toml and nothing else
        let cfg_path = cfgdir.path().join("engine.toml");

        let cands = root_candidates(&cfg_with(&["scenarios/asr.toml"]), &cfg_path, None,
                                    Some(install.path().to_path_buf()));
        assert!(!cands.is_empty(), "a relative config must still offer candidates");
        // The config's own directory is offered but has no scenarios/, so it must not be chosen;
        // the install root has one and must be.
        let chosen = cands.iter().find(|c| c.join("scenarios").is_dir());
        assert_eq!(chosen, Some(&install.path().to_path_buf()),
            "the checked candidate must be the install root, not the config dir: {cands:?}");
    }

    /// An absolute `.../scenarios/x.toml` still names its own root, and wins over the install
    /// location -- otherwise pointing the CLI at a second checkout would silently serve the
    /// installed one.
    #[test]
    fn an_absolute_scenario_path_outranks_the_install_root() {
        let repo = tempfile::tempdir().unwrap();
        std::fs::create_dir(repo.path().join("scenarios")).unwrap();
        let install = tempfile::tempdir().unwrap();
        std::fs::create_dir(install.path().join("scenarios")).unwrap();
        let abs = repo.path().join("scenarios/asr.toml");
        let cands = root_candidates(&cfg_with(&[abs.to_str().unwrap()]),
                                    Path::new("/nowhere/engine.toml"), None,
                                    Some(install.path().to_path_buf()));
        assert_eq!(cands.first(), Some(&repo.path().to_path_buf()),
            "an absolute scenario names its root and must rank first: {cands:?}");
    }

    /// A candidate without `scenarios/` is not a root. This is the check whose absence let the old
    /// chain fall through to a working directory that was never going to work.
    #[test]
    fn a_candidate_without_scenarios_is_not_a_root() {
        let empty = tempfile::tempdir().unwrap();
        let cands = root_candidates(&cfg_with(&["scenarios/asr.toml"]),
                                    &empty.path().join("engine.toml"), None,
                                    Some(empty.path().to_path_buf()));
        assert!(cands.iter().all(|c| !c.join("scenarios").is_dir()),
            "nothing here holds scenarios/, so no candidate may pass the check: {cands:?}");
    }

    /// A checkout you are standing in outranks the installed prefix.
    ///
    /// Not a preference: install.sh symlinks the installed `artifacts/` back to whichever checkout
    /// it was run from, so ranking the install root first would give a SECOND worktree the FIRST
    /// one's weights and report nothing. The bug being fixed here is a silent wrong root; trading
    /// it for a different silent wrong root is not a fix.
    #[test]
    fn a_checkout_you_are_standing_in_outranks_the_install_root() {
        let repo = tempfile::tempdir().unwrap();
        std::fs::create_dir(repo.path().join("scenarios")).unwrap();
        let install = tempfile::tempdir().unwrap();
        std::fs::create_dir(install.path().join("scenarios")).unwrap();
        let cands = root_candidates(&cfg_with(&["scenarios/asr.toml"]),
                                    Path::new("/nowhere/engine.toml"),
                                    Some(repo.path().to_path_buf()),
                                    Some(install.path().to_path_buf()));
        let chosen = cands.iter().find(|c| c.join("scenarios").is_dir());
        assert_eq!(chosen, Some(&repo.path().to_path_buf()),
            "the checkout must win over the install prefix: {cands:?}");
    }

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
            models: vec![ModelCfg { name: "parakeet".into(), scenario: "scenarios/asr.toml".into(), resident: false }],
        };
        let r = render(&c);
        assert!(r.contains("model parakeet -> scenarios/asr.toml"));
        // Every configured default is rendered, including one no `ModelKind` variant can name.
        assert!(r.contains("asr=parakeet") && r.contains("tts=kokoro"), "{r}");
        assert!(r.contains("idle_unload_s 900") && r.contains("idle_release_s 1800")
            && r.contains("evict_policy lru"), "{r}");
        // The ceiling must not be printed as a bare number: it bounds device BOs, exempts models
        // with no measured footprint, and today that is all of them. A reader who takes 4096 as a
        // guarantee is the person a memory failure surprised.
        assert!(r.contains("memory_ceiling_mb 4096"), "{r}");
        assert!(r.contains("device BOs") && r.contains("no footprint are exempt"),
            "the ceiling's scope must be printed with it: {r}");
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
                resident: false,
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
                resident: false,
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
