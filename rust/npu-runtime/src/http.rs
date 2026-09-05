//! Thin blocking HTTP surface over the device `Handle`. The NPU is single-tenant, so this is a
//! single-flight server (one request at a time). OpenAI-shaped inference routes + control/admin
//! routes. The request->response decision is the pure `route()` fn (host-testable with a mock
//! Handle); `serve()` is only the socket plumbing.
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::actor::Handle;
use crate::config::{Config, ModelCfg};
use crate::registry::{LoadState, ModelStatus};
use crate::stream::StreamItem;
use npu_engine::capability::{Capability, Request as EngineReq, Response as EngineResp};
use npu_engine::FinishReason;

const MAX_BODY: usize = 16 * 1024 * 1024;
const SOCKET_TIMEOUT: Duration = Duration::from_secs(60);

/// A parsed request, enough for routing.
pub struct Request {
    pub method: String,
    pub path: String,
    pub boundary: String,
    pub body: Vec<u8>,
}

/// A response body. Not always JSON: `/v1/audio/speech` returns audio bytes, the same way OpenAI's
/// does, so the body cannot be a `String`.
pub enum Body {
    Json(String),
    Wav(Vec<u8>),
    /// Server-Sent Events. The generator runs on the actor thread; this is the receiving end of the
    /// channel it feeds, plus what the socket loop needs to render each item into a `data:` frame.
    /// No `Debug`/`PartialEq`: a `Receiver` has neither, and nothing needs to compare a stream body.
    Stream(SseStream),
}

impl Body {
    /// The body as text -- the JSON for a JSON body, empty otherwise. For tests and logging.
    pub fn text(&self) -> &str {
        match self { Body::Json(s) => s, Body::Wav(_) | Body::Stream(_) => "" }
    }
    pub fn content_type(&self) -> &'static str {
        match self {
            Body::Json(_) => "application/json",
            Body::Wav(_) => "audio/wav",
            Body::Stream(_) => "text/event-stream",
        }
    }
    pub fn bytes(&self) -> &[u8] {
        match self { Body::Json(s) => s.as_bytes(), Body::Wav(v) => v, Body::Stream(_) => &[] }
    }
}
impl std::fmt::Display for Body {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Body::Json(s) => f.write_str(s),
            Body::Wav(v) => write!(f, "<{} bytes of audio/wav>", v.len()),
            Body::Stream(_) => f.write_str("<event-stream>"),
        }
    }
}
impl From<String> for Body { fn from(s: String) -> Body { Body::Json(s) } }
impl From<&str> for Body { fn from(s: &str) -> Body { Body::Json(s.to_string()) } }

/// Which OpenAI route an `SseStream` is rendering for -- the two shapes differ (`delta` vs `text`,
/// and chat alone has a role-announcement chunk).
pub enum SseKind { Chat, Completion }

/// A streaming generation in progress, plus what `respond()` needs to render OpenAI-shaped frames
/// from it without knowing anything about JSON itself living on the actor side.
pub struct SseStream {
    rx: std::sync::mpsc::Receiver<StreamItem>,
    id: String,
    created: i64,
    model: String,
    kind: SseKind,
}

impl SseStream {
    fn new(rx: std::sync::mpsc::Receiver<StreamItem>, model: String, kind: SseKind) -> SseStream {
        let prefix = match kind { SseKind::Chat => "chatcmpl", SseKind::Completion => "cmpl" };
        SseStream { rx, id: gen_id(prefix), created: unix_now(), model, kind }
    }
    /// The chat-only preamble: OpenAI announces the role before any content, in its own chunk.
    fn render_role(&self) -> String {
        format!(
            "{{\"id\":\"{}\",\"object\":\"chat.completion.chunk\",\"created\":{},\"model\":\"{}\",\
             \"choices\":[{{\"index\":0,\"delta\":{{\"role\":\"assistant\"}},\"finish_reason\":null}}]}}",
            self.id, self.created, parse::json_escape(&self.model))
    }
    fn render_text(&self, text: &str) -> String {
        match self.kind {
            SseKind::Chat => format!(
                "{{\"id\":\"{}\",\"object\":\"chat.completion.chunk\",\"created\":{},\"model\":\"{}\",\
                 \"choices\":[{{\"index\":0,\"delta\":{{\"content\":\"{}\"}},\"finish_reason\":null}}]}}",
                self.id, self.created, parse::json_escape(&self.model), parse::json_escape(text)),
            SseKind::Completion => format!(
                "{{\"id\":\"{}\",\"object\":\"text_completion\",\"created\":{},\"model\":\"{}\",\
                 \"choices\":[{{\"index\":0,\"text\":\"{}\",\"finish_reason\":null}}]}}",
                self.id, self.created, parse::json_escape(&self.model), parse::json_escape(text)),
        }
    }
    fn render_done(&self, reason: FinishReason) -> String {
        match self.kind {
            SseKind::Chat => format!(
                "{{\"id\":\"{}\",\"object\":\"chat.completion.chunk\",\"created\":{},\"model\":\"{}\",\
                 \"choices\":[{{\"index\":0,\"delta\":{{}},\"finish_reason\":\"{}\"}}]}}",
                self.id, self.created, parse::json_escape(&self.model), reason.as_str()),
            SseKind::Completion => format!(
                "{{\"id\":\"{}\",\"object\":\"text_completion\",\"created\":{},\"model\":\"{}\",\
                 \"choices\":[{{\"index\":0,\"text\":\"\",\"finish_reason\":\"{}\"}}]}}",
                self.id, self.created, parse::json_escape(&self.model), reason.as_str()),
        }
    }
    fn render_error(&self, msg: &str) -> String {
        format!("{{\"error\":{{\"message\":\"{}\"}}}}", parse::json_escape(msg))
    }
}

/// A process-unique id for a completion object (`chatcmpl-...` / `cmpl-...`). Not cryptographic,
/// just distinct: a nanosecond timestamp plus a monotonic counter, so two completions started in
/// the same nanosecond (the actor is single-flight, but the counter costs nothing) still differ.
fn gen_id(prefix: &str) -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos()).unwrap_or(0);
    format!("{prefix}-{nanos:x}{n:x}")
}
fn unix_now() -> i64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64).unwrap_or(0)
}

/// (status code, body).
pub type Response = (u16, Body);

/// Pure routing decision. Mutating admin routes load/edit/save the config at `cfg_path` then ask the
/// actor to reconcile. No socket here -> unit-testable with a mock-backed Handle.
pub fn route(req: &Request, handle: &Handle, cfg_path: &Path) -> Response {
    match (req.method.as_str(), req.path.as_str()) {
        ("GET", "/health") => (200, "{\"status\":\"ok\"}".into()),
        // `ok` is COMPUTED, not asserted. It used to be the literal `true`, so a service whose model
        // had failed to load reported healthy to systemd while answering every request with an error
        // -- the shape that hid a 5-day outage. A Failed model makes this 503; `Unloaded` never does,
        // because that is deliberate (deferred by max_resident, or swept for being idle).
        ("GET", "/healthz") => {
            let npu = npu_engine::Engine::available();
            let st = handle.status();
            let n = st.iter().filter(|s| s.state == LoadState::Loaded).count();
            let failed: Vec<&ModelStatus> = st.iter().filter(|s| s.state == LoadState::Failed).collect();
            let names = failed.iter()
                .map(|s| format!("\"{}\"", parse::json_escape(&s.name))).collect::<Vec<_>>().join(",");
            let ok = failed.is_empty();
            (if ok { 200 } else { 503 },
             format!("{{\"ok\":{ok},\"npu\":{npu},\"loaded\":{n},\"failed\":[{names}]}}").into())
        }
        ("GET", "/v1/models") => (200, models_json(&handle.status()).into()),
        ("POST", "/v1/chat/completions") => chat_completions(req, handle),
        ("POST", "/v1/completions") => completions(req, handle),
        ("POST", "/v1/embeddings") => embeddings(req, handle),
        ("POST", "/v1/audio/speech") => audio_speech(req, handle),
        ("POST", "/v1/audio/transcriptions") => transcriptions(req, handle),
        ("POST", "/v1/audio/diarizations") => diarizations(req, handle),
        ("POST", "/admin/reload") => admin_reload(handle, cfg_path),
        ("POST", "/admin/models") => admin_add_model(req, handle, cfg_path),
        ("POST", "/admin/defaults") => admin_set_default(req, handle, cfg_path),
        ("DELETE", p) if p.starts_with("/admin/models/") =>
            admin_remove_model(&p["/admin/models/".len()..].to_string(), handle, cfg_path),
        ("GET", _) => (404, "{\"error\":\"not found\"}".into()),
        _ => (404, "{\"error\":\"not found\"}".into()),
    }
}

/// Render model statuses as the `/v1/models` JSON list (reused by the C ABI control surface).
///
/// `state` + `idle_s` are what make a hot swap observable from outside: `idle_s` counts seconds since
/// the model last served a request and is `null` while it is not resident.
pub fn models_json(status: &[ModelStatus]) -> String {
    let mut data = String::new();
    for (i, s) in status.iter().enumerate() {
        if i > 0 { data.push(','); }
        let kind = s.capability.map(|c| c.0).unwrap_or("unknown");
        let state = match s.state { LoadState::Loaded => "loaded", LoadState::Failed => "failed", LoadState::Unloaded => "unloaded" };
        let idle = match s.idle_s { Some(n) => n.to_string(), None => "null".to_string() };
        data.push_str(&format!(
            "{{\"id\":\"{}\",\"object\":\"model\",\"kind\":\"{kind}\",\"state\":\"{state}\",\"detail\":\"{}\",\"bo_bytes\":{},\"idle_s\":{idle}}}",
            s.name, parse::json_escape(&s.detail), s.bo_bytes));
    }
    format!("{{\"object\":\"list\",\"data\":[{data}]}}")
}

/// Map an engine error onto a status code. `NoModel` is 503, not 400: nothing is wrong with the
/// request -- the server has no model for that capability, and a client cannot fix it by retrying
/// differently. This is what `/v1/chat/completions` and `/v1/audio/speech` answer today, since no
/// generate or tts model is configured yet; both routes are otherwise complete.
fn engine_err(e: &npu_engine::EngineError) -> Response {
    let code = match e {
        npu_engine::EngineError::NoModel(_) | npu_engine::EngineError::NotAvailable => 503,
        npu_engine::EngineError::WrongKind { .. } | npu_engine::EngineError::Unsupported(_) => 400,
        npu_engine::EngineError::Load(_) | npu_engine::EngineError::Device(_) => 500,
    };
    (code, format!("{{\"error\":\"{}\"}}", parse::json_escape(&e.to_string())).into())
}

/// OpenAI chat completions. Serves `Capability::GENERATE` with the FULL message array (system
/// prompt + history, not just the last turn) and the full sampling surface -- see `parse::
/// parse_chat_request`. Streams via SSE when the body asks for it, buffers otherwise.
fn chat_completions(req: &Request, handle: &Handle) -> Response {
    let body = String::from_utf8_lossy(&req.body).to_string();
    let parsed = match parse::parse_chat_request(&body) {
        Ok(p) => p,
        Err(e) => return (400, format!("{{\"error\":\"{}\"}}", parse::json_escape(&e)).into()),
    };
    let served = match handle.generate(parsed.model.as_deref(), parsed.prompt, parsed.params) {
        Ok(s) => s,
        Err(e) => return engine_err(&e),
    };
    if parsed.stream {
        (200, Body::Stream(SseStream::new(served.value, served.model, SseKind::Chat)))
    } else {
        render_buffered(served.model, served.value, SseKind::Chat)
    }
}

/// OpenAI text completions: same generation path as chat, over a raw (non-templated) prompt string.
fn completions(req: &Request, handle: &Handle) -> Response {
    let body = String::from_utf8_lossy(&req.body).to_string();
    let parsed = match parse::parse_completion_request(&body) {
        Ok(p) => p,
        Err(e) => return (400, format!("{{\"error\":\"{}\"}}", parse::json_escape(&e)).into()),
    };
    let served = match handle.generate(parsed.model.as_deref(), parsed.prompt, parsed.params) {
        Ok(s) => s,
        Err(e) => return engine_err(&e),
    };
    if parsed.stream {
        (200, Body::Stream(SseStream::new(served.value, served.model, SseKind::Completion)))
    } else {
        render_buffered(served.model, served.value, SseKind::Completion)
    }
}

/// Drain a generation to completion and render the OpenAI non-streaming shape. Draining fully
/// (rather than stopping at the first error) is deliberate: the actor side always sends exactly one
/// terminal item (`Done` or `Error`), so this loop always terminates.
fn render_buffered(model: String, rx: std::sync::mpsc::Receiver<StreamItem>, kind: SseKind) -> Response {
    let mut text = String::new();
    let (reason, usage) = loop {
        match rx.recv() {
            Ok(StreamItem::Text(t)) => text.push_str(&t),
            Ok(StreamItem::Done { reason, usage }) => break (reason, usage),
            Ok(StreamItem::Error(e)) =>
                return (500, format!("{{\"error\":\"{}\"}}", parse::json_escape(&e)).into()),
            Err(_) =>
                return (500, "{\"error\":\"generation ended without a result\"}".into()),
        }
    };
    let id = gen_id(match kind { SseKind::Chat => "chatcmpl", SseKind::Completion => "cmpl" });
    let created = unix_now();
    let total = usage.prompt_tokens + usage.completion_tokens;
    let usage_json = format!("\"usage\":{{\"prompt_tokens\":{},\"completion_tokens\":{},\"total_tokens\":{total}}}",
        usage.prompt_tokens, usage.completion_tokens);
    let body = match kind {
        SseKind::Chat => format!(
            "{{\"id\":\"{id}\",\"object\":\"chat.completion\",\"created\":{created},\"model\":\"{}\",\
             \"choices\":[{{\"index\":0,\"message\":{{\"role\":\"assistant\",\"content\":\"{}\"}},\
             \"finish_reason\":\"{}\"}}],{usage_json}}}",
            parse::json_escape(&model), parse::json_escape(&text), reason.as_str()),
        SseKind::Completion => format!(
            "{{\"id\":\"{id}\",\"object\":\"text_completion\",\"created\":{created},\"model\":\"{}\",\
             \"choices\":[{{\"index\":0,\"text\":\"{}\",\"finish_reason\":\"{}\"}}],{usage_json}}}",
            parse::json_escape(&model), parse::json_escape(&text), reason.as_str()),
    };
    (200, body.into())
}

/// OpenAI speech synthesis. Serves `Capability::TTS` and returns audio bytes, not JSON.
///
/// `voice` is accepted and currently ignored: no model resolves one yet, and silently accepting a
/// field that does nothing is better than rejecting requests a real client sends.
fn audio_speech(req: &Request, handle: &Handle) -> Response {
    let body = String::from_utf8_lossy(&req.body).to_string();
    let model = extract_str_field(&body, "model");
    let input = match parse::parse_inputs(&body) {
        Ok(v) => match v.into_iter().next() {
            Some(t) => t,
            None => return (400, "{\"error\":\"input is empty\"}".into()),
        },
        Err(e) => return (400, format!("{{\"error\":\"{}\"}}", parse::json_escape(&e)).into()),
    };
    let served = match handle.serve(Capability::TTS, model.as_deref(), EngineReq::Text(input)) {
        Ok(s) => s,
        Err(e) => return engine_err(&e),
    };
    match served.value {
        EngineResp::Audio { pcm, sample_rate } => (200, Body::Wav(parse::wav_from_i16(&pcm, sample_rate))),
        other => (500, format!("{{\"error\":\"{} returned a {} response\"}}",
            parse::json_escape(&served.model), other.shape()).into()),
    }
}

fn embeddings(req: &Request, handle: &Handle) -> Response {
    let body = String::from_utf8_lossy(&req.body).to_string();
    let model = extract_str_field(&body, "model");
    let inputs = match parse::parse_inputs(&body) {
        Ok(v) if v.is_empty() => return (400, "{\"error\":\"input is empty\"}".into()),
        Ok(v) => v,
        Err(e) => return (400, format!("{{\"error\":\"{}\"}}", parse::json_escape(&e)).into()),
    };
    let mut data = String::new();
    let mut served = String::new();
    for (i, text) in inputs.iter().enumerate() {
        match handle.embed(model.as_deref(), text) {
            Ok(s) => {
                served = s.model;
                let arr = s.value.iter().map(|x| format!("{x}")).collect::<Vec<_>>().join(",");
                if i > 0 { data.push(','); }
                data.push_str(&format!("{{\"object\":\"embedding\",\"index\":{i},\"embedding\":[{arr}]}}"));
            }
            Err(e) => return (400, format!("{{\"error\":\"{}\"}}", parse::json_escape(&e.to_string())).into()),
        }
    }
    (200, format!("{{\"object\":\"list\",\"data\":[{data}],\"model\":\"{}\"}}", parse::json_escape(&served)).into())
}

fn transcriptions(req: &Request, handle: &Handle) -> Response {
    let file = match parse::extract_file_part(&req.body, &req.boundary) {
        Some(w) => w, None => return (400, "{\"error\":\"no file part\"}".into()),
    };
    // Any container ffmpeg can read, not only an exact 16 kHz mono WAV. OpenAI-shaped clients upload
    // mp3/m4a/webm and video, and the strict parser answered every one of them with the same 400.
    let samples = match crate::media::decode_bytes(file) {
        Ok(s) if !s.is_empty() => s,
        Ok(_) => return (400, "{\"error\":\"file decoded to no audio\"}".into()),
        Err(e) => return (400, format!("{{\"error\":\"{}\"}}", parse::json_escape(&e)).into()),
    };
    // OpenAI's transcription request carries `model` as a form field. This route used to drop it and
    // always serve the default, which left ASR -- the capability this engine actually ships -- with
    // no way to pick a model per request even though the actor has always taken one.
    let model = parse::extract_form_field(&req.body, &req.boundary, "model");
    match handle.transcribe(model.as_deref(), samples, 16_000) {
        Ok(s) => (200, format!("{{\"text\":\"{}\",\"model\":\"{}\"}}",
            parse::json_escape(&s.value), parse::json_escape(&s.model)).into()),
        Err(e) => engine_err(&e),
    }
}

/// Speaker diarization. OpenAI has no diarization endpoint, so this mirrors the shape of our own
/// `/v1/audio/transcriptions`: multipart `file` part + a `model` form field.
fn diarizations(req: &Request, handle: &Handle) -> Response {
    let file = match parse::extract_file_part(&req.body, &req.boundary) {
        Some(w) => w, None => return (400, "{\"error\":\"no file part\"}".into()),
    };
    // Any container ffmpeg can read, not only an exact 16 kHz mono WAV. OpenAI-shaped clients upload
    // mp3/m4a/webm and video, and the strict parser answered every one of them with the same 400.
    let samples = match crate::media::decode_bytes(file) {
        Ok(s) if !s.is_empty() => s,
        Ok(_) => return (400, "{\"error\":\"file decoded to no audio\"}".into()),
        Err(e) => return (400, format!("{{\"error\":\"{}\"}}", parse::json_escape(&e)).into()),
    };
    let model = parse::extract_form_field(&req.body, &req.boundary, "model");
    match handle.diarize(model.as_deref(), samples, 16_000) {
        Ok(s) => (200, segments_json(&s.model, &s.value).into()),
        Err(e) => engine_err(&e),
    }
}

/// Render segments. `speaker` is a cluster INDEX internally; the `SPEAKER_NN` label is produced
/// here, at the edge, so nothing downstream has to parse a string back into a number.
fn segments_json(model: &str, segs: &[npu_engine::capability::Segment]) -> String {
    let items: Vec<String> = segs.iter().map(|s| format!(
        "{{\"start\":{:.3},\"end\":{:.3},\"speaker\":\"SPEAKER_{:02}\"}}",
        s.start_s, s.end_s, s.speaker)).collect();
    format!("{{\"model\":\"{}\",\"segments\":[{}]}}", parse::json_escape(model), items.join(","))
}

fn admin_reload(handle: &Handle, cfg_path: &Path) -> Response {
    match Config::load(cfg_path) {
        Ok(cfg) => match handle.reconcile(cfg) {
            Ok(rep) => (200, format!("{{\"loaded\":{},\"unloaded\":{},\"failed\":{},\"deferred\":{}}}",
                rep.loaded.len(), rep.unloaded.len(), rep.failed.len(), rep.deferred.len()).into()),
            Err(e) => engine_err(&e),
        },
        Err(e) => (400, format!("{{\"error\":\"{}\"}}", parse::json_escape(&e)).into()),
    }
}

fn admin_add_model(req: &Request, handle: &Handle, cfg_path: &Path) -> Response {
    let body = String::from_utf8_lossy(&req.body).to_string();
    let (name, scenario) = match (extract_str_field(&body, "name"), extract_str_field(&body, "scenario")) {
        (Some(n), Some(s)) => (n, s),
        _ => return (400, "{\"error\":\"need name + scenario\"}".into()),
    };
    mutate_and_reconcile(handle, cfg_path, |cfg| {
        cfg.models.retain(|m| m.name != name);
        cfg.models.push(ModelCfg { name: name.clone(), scenario: scenario.clone() });
    })
}

fn admin_remove_model(name: &str, handle: &Handle, cfg_path: &Path) -> Response {
    let name = name.to_string();
    mutate_and_reconcile(handle, cfg_path, |cfg| cfg.models.retain(|m| m.name != name))
}

fn admin_set_default(req: &Request, handle: &Handle, cfg_path: &Path) -> Response {
    let body = String::from_utf8_lossy(&req.body).to_string();
    let (cap, model) = match (extract_str_field(&body, "capability"), extract_str_field(&body, "model")) {
        (Some(c), Some(m)) => (c, m),
        _ => return (400, "{\"error\":\"need capability + model\"}".into()),
    };
    // A capability nothing implements is rejected rather than written: the old match silently
    // dropped anything that was not asr/embed, so `/admin/defaults` reported 200 and changed nothing.
    let cap = match Capability::from_name(&cap) {
        Some(c) => c,
        None => return (400, format!("{{\"error\":\"unknown capability {}\"}}", parse::json_escape(&cap)).into()),
    };
    mutate_and_reconcile(handle, cfg_path, |cfg| cfg.defaults.set(cap, model.clone()))
}

fn mutate_and_reconcile(handle: &Handle, cfg_path: &Path, f: impl FnOnce(&mut Config)) -> Response {
    let mut cfg = match Config::load(cfg_path) { Ok(c) => c, Err(e) => return (400, format!("{{\"error\":\"{}\"}}", parse::json_escape(&e)).into()) };
    f(&mut cfg);
    if let Err(e) = cfg.save(cfg_path) { return (500, format!("{{\"error\":\"{}\"}}", parse::json_escape(&e)).into()); }
    match handle.reconcile(cfg) {
        Ok(rep) => (200, format!("{{\"loaded\":{},\"unloaded\":{},\"failed\":{},\"deferred\":{}}}",
            rep.loaded.len(), rep.unloaded.len(), rep.failed.len(), rep.deferred.len()).into()),
        Err(e) => engine_err(&e),
    }
}

/// Minimal extraction of a JSON string field `"<key>":"<value>"`.
fn extract_str_field(body: &str, key: &str) -> Option<String> {
    let needle = format!("\"{key}\"");
    let idx = body.find(&needle)?;
    let rest = &body[idx + needle.len()..];
    let q1 = rest.find('"')?;
    let s = &rest[q1 + 1..];
    let q2 = s.find('"')?;
    Some(s[..q2].to_string())
}

/// Blocking single-flight server. Reads each request, routes it, writes the response.
pub fn serve(handle: Handle, cfg_path: PathBuf, port: u16) -> std::io::Result<()> {
    let listener = TcpListener::bind(format!("127.0.0.1:{port}"))?;
    serve_on(listener, handle, cfg_path)
}

/// Like `serve`, but on an already-bound listener. Lets a caller (a test, chiefly) claim an
/// OS-assigned ephemeral port via `TcpListener::bind("127.0.0.1:0")` and read it back with
/// `local_addr()` before handing the listener over here -- no bind-then-guess race.
pub fn serve_on(listener: TcpListener, handle: Handle, cfg_path: PathBuf) -> std::io::Result<()> {
    eprintln!("[npu-serve] ready on http://{}", listener.local_addr()?);
    for stream in listener.incoming() {
        match stream {
            Ok(s) => { if let Err(e) = handle_conn(s, &handle, &cfg_path) { eprintln!("[npu-serve] {e}"); } }
            Err(e) => eprintln!("[npu-serve] accept: {e}"),
        }
    }
    Ok(())
}

fn handle_conn(mut stream: TcpStream, handle: &Handle, cfg_path: &Path) -> std::io::Result<()> {
    let _ = stream.set_read_timeout(Some(SOCKET_TIMEOUT));
    let _ = stream.set_write_timeout(Some(SOCKET_TIMEOUT));
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut line = String::new();
    reader.read_line(&mut line)?;
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let path = parts.next().unwrap_or("").to_string();
    let mut content_len = 0usize;
    let mut boundary = String::new();
    loop {
        let mut h = String::new();
        if reader.read_line(&mut h)? == 0 { break; }
        let h = h.trim_end();
        if h.is_empty() { break; }
        let l = h.to_ascii_lowercase();
        if let Some(v) = l.strip_prefix("content-length:") { content_len = v.trim().parse().unwrap_or(0); }
        else if l.starts_with("content-type:") {
            if let Some(idx) = l.find("boundary=") { boundary = h[idx + "boundary=".len()..].trim().trim_matches('"').to_string(); }
        }
    }
    if content_len > MAX_BODY { return respond(&mut stream, 413, &"{\"error\":\"too large\"}".into()); }
    let mut body = vec![0u8; content_len];
    reader.read_exact(&mut body)?;
    let req = Request { method, path, boundary, body };
    let (code, body) = route(&req, handle, cfg_path);
    respond(&mut stream, code, &body)
}

fn respond(stream: &mut TcpStream, code: u16, body: &Body) -> std::io::Result<()> {
    if let Body::Stream(s) = body {
        return respond_stream(stream, code, s);
    }
    let data = body.bytes();
    // Header and body are written separately because the body is not always UTF-8 (audio/wav).
    let head = format!(
        "HTTP/1.1 {code} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        reason_phrase(code), body.content_type(), data.len());
    stream.write_all(head.as_bytes())?;
    stream.write_all(data)?;
    stream.flush()
}

fn reason_phrase(code: u16) -> &'static str {
    match code {
        200 => "OK", 400 => "Bad Request", 404 => "Not Found", 413 => "Payload Too Large",
        500 => "Internal Server Error", 501 => "Not Implemented", 503 => "Service Unavailable",
        _ => "Error",
    }
}

/// Stream Server-Sent Events as the actor produces them: one `data:` frame per item, `[DONE]`
/// terminates. No `Content-Length` -- the length is not known up front, which is the whole point.
///
/// A failed write (the client hung up) returns `Err` immediately instead of trying the rest of the
/// stream, which drops `s.rx` on the way out. The actor's next `tx.send` then fails and the
/// generator's sink returns `false` -- this is the entire disconnect-abort mechanism; nothing here
/// signals the actor directly.
fn respond_stream(stream: &mut TcpStream, code: u16, s: &SseStream) -> std::io::Result<()> {
    let head = format!(
        "HTTP/1.1 {code} {}\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n\
         Connection: close\r\n\r\n", reason_phrase(code));
    stream.write_all(head.as_bytes())?;
    if matches!(s.kind, SseKind::Chat) {
        stream.write_all(format!("data: {}\n\n", s.render_role()).as_bytes())?;
    }
    for item in s.rx.iter() {
        let frame = match item {
            StreamItem::Text(t) => s.render_text(&t),
            StreamItem::Done { reason, .. } => s.render_done(reason),
            StreamItem::Error(e) => s.render_error(&e),
        };
        stream.write_all(format!("data: {frame}\n\n").as_bytes())?;
    }
    stream.write_all(b"data: [DONE]\n\n")?;
    stream.flush()
}

/// HTTP/JSON/WAV parsing helpers (ported from the C3 npu-serve), pure + unit-tested.
pub mod parse {
    /// The `input` of an embeddings request: one string, or an array of them.
    ///
    /// A real scan over JSON string literals, not a search for `[` and `]`. The previous version
    /// looked for those characters anywhere in the body, so a `[` inside a string value started an
    /// "array", a `]` inside one ended it, and splitting on unescaped `"` cut an input in half at
    /// every `\"`. All three returned a wrong answer under HTTP 200; prose with links, quotes or
    /// brackets is the common case, not the corner case.
    ///
    /// `Err` rather than a best guess: the old fallback treated an unparseable body as the text to
    /// embed, which turned a client bug into a plausible-looking vector.
    pub fn parse_inputs(body: &str) -> Result<Vec<String>, String> {
        let b = body.as_bytes();
        let mut i = 0;
        while i < b.len() {
            if b[i] != b'"' { i += 1; continue; }
            let (key, after) = scan_json_string(body, i)?;
            // A string is a KEY only if a colon follows it; otherwise it is a value, and scanning
            // past it as a unit is exactly what stops `"input"` inside a value from matching.
            let mut j = after;
            while j < b.len() && b[j].is_ascii_whitespace() { j += 1; }
            if j < b.len() && b[j] == b':' {
                if key == "input" { return scan_input_value(body, j + 1); }
                i = j + 1;
            } else {
                i = after;
            }
        }
        Err("missing \"input\" field".into())
    }

    /// The value after `"input":` -- a string, or an array of strings.
    fn scan_input_value(s: &str, mut i: usize) -> Result<Vec<String>, String> {
        let b = s.as_bytes();
        while i < b.len() && b[i].is_ascii_whitespace() { i += 1; }
        match b.get(i) {
            Some(b'"') => Ok(vec![scan_json_string(s, i)?.0]),
            Some(b'[') => {
                let mut out = Vec::new();
                i += 1;
                loop {
                    while i < b.len() && b[i].is_ascii_whitespace() { i += 1; }
                    match b.get(i) {
                        Some(b']') => return Ok(out),
                        Some(b',') => i += 1,
                        Some(b'"') => { let (v, n) = scan_json_string(s, i)?; out.push(v); i = n; }
                        Some(c) => return Err(format!("input array: expected a string, got {:?}", *c as char)),
                        None => return Err("unterminated input array".into()),
                    }
                }
            }
            Some(c) => Err(format!("input must be a string or an array of strings, got {:?}", *c as char)),
            None => Err("input has no value".into()),
        }
    }

    /// Decode the JSON string literal starting at `start` (which must be its opening quote).
    /// Returns the decoded text and the index just past the closing quote.
    fn scan_json_string(s: &str, start: usize) -> Result<(String, usize), String> {
        let b = s.as_bytes();
        if b.get(start) != Some(&b'"') { return Err("expected a string".into()); }
        let mut out = String::new();
        let mut i = start + 1;
        while i < b.len() {
            match b[i] {
                b'"' => return Ok((out, i + 1)),
                b'\\' => {
                    i += 1;
                    match *b.get(i).ok_or("string ends inside an escape")? {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => { let (c, n) = scan_unicode_escape(b, i + 1)?; out.push(c); i = n - 1; }
                        c => return Err(format!("bad escape \\{:?}", c as char)),
                    }
                    i += 1;
                }
                // Not ASCII-indexable: step by whole chars so multi-byte UTF-8 is copied intact.
                // Slicing `s` is O(1), so this stays linear over the body.
                _ => {
                    let c = s[i..].chars().next().ok_or("invalid UTF-8 in string")?;
                    out.push(c);
                    i += c.len_utf8();
                }
            }
        }
        Err("unterminated string".into())
    }

    /// A `\uXXXX` escape, including the surrogate PAIR a non-BMP character needs. Returns the
    /// character and the index just past the escape. A lone surrogate is an error, not a
    /// replacement char: it means the client sent something it could not have meant.
    fn scan_unicode_escape(b: &[u8], i: usize) -> Result<(char, usize), String> {
        let hi = hex4(b, i)?;
        if !(0xD800..0xDC00).contains(&hi) {
            let c = char::from_u32(hi as u32).ok_or("invalid \\u escape")?;
            return Ok((c, i + 4));
        }
        if b.get(i + 4) != Some(&b'\\') || b.get(i + 5) != Some(&b'u') {
            return Err("high surrogate without a following \\u escape".into());
        }
        let lo = hex4(b, i + 6)?;
        if !(0xDC00..0xE000).contains(&lo) { return Err("high surrogate not followed by a low one".into()); }
        let cp = 0x10000 + (((hi - 0xD800) as u32) << 10) + (lo - 0xDC00) as u32;
        Ok((char::from_u32(cp).ok_or("invalid surrogate pair")?, i + 10))
    }

    /// The generation request extracted from a chat/completions body: routing (`model`), the prompt,
    /// the full sampling surface, and whether to stream.
    pub struct ParsedGenerate {
        pub model: Option<String>,
        pub prompt: npu_engine::Prompt,
        pub params: npu_engine::GenerateParams,
        pub stream: bool,
    }

    /// `/v1/chat/completions`: the full `messages` array (system prompt + history, not just the last
    /// turn -- that was the bug) plus the sampling surface. `serde_json`, not the hand-rolled scanner
    /// above: `messages` is genuinely nested (array of objects, content sometimes itself an array),
    /// and that shape is exactly where a hand-rolled parser accumulates bugs.
    pub fn parse_chat_request(body: &str) -> Result<ParsedGenerate, String> {
        let v: serde_json::Value = serde_json::from_str(body).map_err(|e| format!("invalid JSON: {e}"))?;
        let model = v.get("model").and_then(|m| m.as_str()).map(str::to_string);
        let messages = v.get("messages").and_then(|m| m.as_array())
            .ok_or_else(|| "missing \"messages\" array".to_string())?;
        if messages.is_empty() { return Err("\"messages\" must not be empty".into()); }
        let mut chat = Vec::with_capacity(messages.len());
        for (i, m) in messages.iter().enumerate() {
            let role = m.get("role").and_then(|r| r.as_str())
                .ok_or_else(|| format!("messages[{i}]: missing \"role\""))?.to_string();
            let content = parse_content(m.get("content")).map_err(|e| format!("messages[{i}]: {e}"))?;
            chat.push(npu_engine::ChatMessage { role, content });
        }
        let params = parse_generate_params(&v)?;
        let stream = v.get("stream").and_then(|s| s.as_bool()).unwrap_or(false);
        reject_unsupported(&v, false)?;
        Ok(ParsedGenerate { model, prompt: npu_engine::Prompt::Chat(chat), params, stream })
    }

    /// A message's `content`: a plain string, or OpenAI's multi-part array form when every part is
    /// `{"type":"text","text":...}`. Any other part type is REJECTED rather than silently dropped (the
    /// old behaviour) -- this surface has no vision/audio input, and dropping content changes the
    /// prompt's meaning with no trace, which is exactly what spec S6 bans.
    fn parse_content(v: Option<&serde_json::Value>) -> Result<String, String> {
        match v {
            Some(serde_json::Value::String(s)) => Ok(s.clone()),
            Some(serde_json::Value::Array(parts)) => {
                let mut out = String::new();
                for p in parts {
                    match p.get("type").and_then(|t| t.as_str()) {
                        Some("text") => out.push_str(p.get("text").and_then(|t| t.as_str()).unwrap_or("")),
                        Some(other) => return Err(format!("unsupported content part type {other:?}")),
                        None => return Err("content part missing \"type\"".into()),
                    }
                }
                Ok(out)
            }
            Some(_) => Err("\"content\" must be a string or an array of parts".into()),
            None => Err("missing \"content\"".into()),
        }
    }

    /// `/v1/completions`. The array form of `prompt` (OpenAI allows batching several prompts in one
    /// request) is rejected with a clear 400 -- this surface serves one completion per request, and
    /// silently taking just the first would answer a different request than the one sent.
    pub fn parse_completion_request(body: &str) -> Result<ParsedGenerate, String> {
        let v: serde_json::Value = serde_json::from_str(body).map_err(|e| format!("invalid JSON: {e}"))?;
        let model = v.get("model").and_then(|m| m.as_str()).map(str::to_string);
        let prompt = match v.get("prompt") {
            Some(serde_json::Value::String(s)) => s.clone(),
            Some(serde_json::Value::Array(_)) =>
                return Err("array \"prompt\" (batched prompts) is not supported; send one string".into()),
            Some(_) => return Err("\"prompt\" must be a string".into()),
            None => return Err("missing \"prompt\"".into()),
        };
        let params = parse_generate_params(&v)?;
        let stream = v.get("stream").and_then(|s| s.as_bool()).unwrap_or(false);
        reject_unsupported(&v, true)?;
        Ok(ParsedGenerate { model, prompt: npu_engine::Prompt::Raw(prompt), params, stream })
    }

    /// The shared sampling surface: OpenAI's fields plus `top_k`/`repetition_penalty`, neither in
    /// OpenAI's schema but both universal among local servers (`GenerateParams`'s own doc comment).
    /// A field absent from the body keeps `GenerateParams::default()` -- OpenAI's defaults (e.g.
    /// `temperature: 1.0`), never a silent substitution of greedy.
    fn parse_generate_params(v: &serde_json::Value) -> Result<npu_engine::GenerateParams, String> {
        let mut p = npu_engine::GenerateParams::default();
        if let Some(x) = v.get("temperature") { p.temperature = as_f32(x, "temperature")?; }
        if let Some(x) = v.get("top_p") { p.top_p = as_f32(x, "top_p")?; }
        if let Some(x) = v.get("top_k") { p.top_k = as_u32(x, "top_k")?; }
        if let Some(x) = v.get("max_tokens") { p.max_tokens = as_u32(x, "max_tokens")?; }
        if let Some(x) = v.get("seed") { p.seed = Some(as_u64(x, "seed")?); }
        if let Some(x) = v.get("presence_penalty") { p.presence_penalty = as_f32(x, "presence_penalty")?; }
        if let Some(x) = v.get("frequency_penalty") { p.frequency_penalty = as_f32(x, "frequency_penalty")?; }
        if let Some(x) = v.get("repetition_penalty") { p.repetition_penalty = as_f32(x, "repetition_penalty")?; }
        p.stop = match v.get("stop") {
            None | Some(serde_json::Value::Null) => Vec::new(),
            Some(serde_json::Value::String(s)) => vec![s.clone()],
            Some(serde_json::Value::Array(items)) => items.iter()
                .map(|i| i.as_str().map(str::to_string)
                    .ok_or_else(|| "\"stop\" array must contain only strings".to_string()))
                .collect::<Result<Vec<_>, _>>()?,
            Some(_) => return Err("\"stop\" must be a string or an array of strings".into()),
        };
        Ok(p)
    }
    fn as_f32(v: &serde_json::Value, field: &str) -> Result<f32, String> {
        v.as_f64().map(|f| f as f32).ok_or_else(|| format!("\"{field}\" must be a number"))
    }
    fn as_u32(v: &serde_json::Value, field: &str) -> Result<u32, String> {
        v.as_u64().and_then(|n| u32::try_from(n).ok())
            .ok_or_else(|| format!("\"{field}\" must be a non-negative integer"))
    }
    fn as_u64(v: &serde_json::Value, field: &str) -> Result<u64, String> {
        v.as_u64().ok_or_else(|| format!("\"{field}\" must be a non-negative integer"))
    }

    /// Spec S6 ("branch freely on *how*, never silently on *what*"): a client-visible parameter this
    /// surface cannot honour must be a 400, not a quiet no-op. `n != 1`, `logprobs`, `logit_bias`,
    /// `tools` and friends all change what the RESPONSE IS; accepting them and ignoring their effect
    /// would answer a request other than the one that was sent, with no trace of the substitution.
    fn reject_unsupported(v: &serde_json::Value, completions: bool) -> Result<(), String> {
        if let Some(n) = v.get("n").and_then(|x| x.as_u64()) {
            if n != 1 { return Err("\"n\" != 1 is not supported".into()); }
        }
        let logprobs_wanted = if completions {
            v.get("logprobs").map(|x| !x.is_null()).unwrap_or(false)
        } else {
            v.get("logprobs").and_then(|x| x.as_bool()).unwrap_or(false)
        };
        if logprobs_wanted { return Err("\"logprobs\" is not supported".into()); }
        for field in ["logit_bias", "tools", "tool_choice", "response_format", "stream_options"] {
            if v.get(field).map(|x| !x.is_null()).unwrap_or(false) {
                return Err(format!("\"{field}\" is not supported"));
            }
        }
        if completions {
            if v.get("echo").and_then(|x| x.as_bool()).unwrap_or(false) {
                return Err("\"echo\" is not supported".into());
            }
            if let Some(b) = v.get("best_of").and_then(|x| x.as_u64()) {
                if b != 1 { return Err("\"best_of\" != 1 is not supported".into()); }
            }
            if v.get("suffix").map(|x| !x.is_null()).unwrap_or(false) {
                return Err("\"suffix\" is not supported".into());
            }
        }
        Ok(())
    }

    /// Wrap mono i16 PCM in a 44-byte canonical WAV header. The rate comes from the model, not a
    /// constant: TTS does not output at the 16 kHz the ASR side works in.
    pub fn wav_from_i16(pcm: &[i16], sample_rate: u32) -> Vec<u8> {
        let data_len = (pcm.len() * 2) as u32;
        let mut w = Vec::with_capacity(44 + data_len as usize);
        w.extend_from_slice(b"RIFF");
        w.extend_from_slice(&(36 + data_len).to_le_bytes());
        w.extend_from_slice(b"WAVEfmt ");
        w.extend_from_slice(&16u32.to_le_bytes());            // PCM fmt chunk size
        w.extend_from_slice(&1u16.to_le_bytes());             // format = PCM
        w.extend_from_slice(&1u16.to_le_bytes());             // channels = mono
        w.extend_from_slice(&sample_rate.to_le_bytes());
        w.extend_from_slice(&(sample_rate * 2).to_le_bytes()); // byte rate = rate * blockalign
        w.extend_from_slice(&2u16.to_le_bytes());             // block align = channels * 2
        w.extend_from_slice(&16u16.to_le_bytes());            // bits per sample
        w.extend_from_slice(b"data");
        w.extend_from_slice(&data_len.to_le_bytes());
        for s in pcm { w.extend_from_slice(&s.to_le_bytes()); }
        w
    }

    fn hex4(b: &[u8], i: usize) -> Result<u16, String> {
        let s = b.get(i..i + 4).ok_or("truncated \\u escape")?;
        let s = std::str::from_utf8(s).map_err(|_| "bad \\u escape".to_string())?;
        u16::from_str_radix(s, 16).map_err(|_| format!("bad \\u escape {s:?}"))
    }
    pub fn json_escape(s: &str) -> String {
        let mut o = String::with_capacity(s.len());
        for ch in s.chars() {
            match ch {
                '"' => o.push_str("\\\""), '\\' => o.push_str("\\\\"),
                '\n' => o.push_str("\\n"), '\r' => o.push_str("\\r"), '\t' => o.push_str("\\t"),
                c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
                c => o.push(c),
            }
        }
        o
    }
    pub fn extract_file_part<'a>(body: &'a [u8], boundary: &str) -> Option<&'a [u8]> {
        if boundary.is_empty() { return None; }
        let delim = format!("--{boundary}");
        for part in split_on(body, delim.as_bytes()) {
            let hdr_end = match find(part, b"\r\n\r\n") { Some(h) => h, None => continue };
            let headers = String::from_utf8_lossy(&part[..hdr_end]).to_ascii_lowercase();
            if headers.contains("name=\"file\"") {
                let mut data = &part[hdr_end + 4..];
                if data.ends_with(b"\r\n") { data = &data[..data.len() - 2]; }
                return Some(data);
            }
        }
        None
    }
    /// Value of a plain (non-file) multipart form field, e.g. `model` on a transcription request.
    /// Parts carrying a `filename=` are skipped: those are uploads, handled by `extract_file_part`.
    pub fn extract_form_field(body: &[u8], boundary: &str, field: &str) -> Option<String> {
        if boundary.is_empty() { return None; }
        let delim = format!("--{boundary}");
        let want = format!("name=\"{}\"", field.to_ascii_lowercase());
        for part in split_on(body, delim.as_bytes()) {
            let hdr_end = match find(part, b"\r\n\r\n") { Some(h) => h, None => continue };
            let headers = String::from_utf8_lossy(&part[..hdr_end]).to_ascii_lowercase();
            if !headers.contains(&want) || headers.contains("filename=") { continue; }
            let mut data = &part[hdr_end + 4..];
            if data.ends_with(b"\r\n") { data = &data[..data.len() - 2]; }
            let v = String::from_utf8_lossy(data).trim().to_string();
            return if v.is_empty() { None } else { Some(v) };
        }
        None
    }
    pub fn split_on<'a>(hay: &'a [u8], sep: &[u8]) -> Vec<&'a [u8]> {
        let mut out = Vec::new();
        let (mut start, mut i) = (0usize, 0usize);
        while i + sep.len() <= hay.len() {
            if &hay[i..i + sep.len()] == sep { out.push(&hay[start..i]); i += sep.len(); start = i; }
            else { i += 1; }
        }
        out.push(&hay[start..]);
        out
    }
    pub fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
        if needle.is_empty() || hay.len() < needle.len() { return None; }
        (0..=hay.len() - needle.len()).find(|&i| &hay[i..i + needle.len()] == needle)
    }
    pub fn parse_wav_i16(wav: &[u8]) -> Option<Vec<i16>> {
        if wav.len() < 12 || &wav[0..4] != b"RIFF" || &wav[8..12] != b"WAVE" { return None; }
        let mut off = 12usize;
        let mut fmt_ok = false;
        let mut data: Option<&[u8]> = None;
        while off + 8 <= wav.len() {
            let id = &wav[off..off + 4];
            let sz = u32::from_le_bytes([wav[off + 4], wav[off + 5], wav[off + 6], wav[off + 7]]) as usize;
            let body_start = off + 8;
            let body_end = body_start.saturating_add(sz).min(wav.len());
            match id {
                b"fmt " if body_end - body_start >= 16 => {
                    let b = &wav[body_start..body_end];
                    let audio_fmt = u16::from_le_bytes([b[0], b[1]]);
                    let channels = u16::from_le_bytes([b[2], b[3]]);
                    let rate = u32::from_le_bytes([b[4], b[5], b[6], b[7]]);
                    let bits = u16::from_le_bytes([b[14], b[15]]);
                    fmt_ok = (audio_fmt == 1 || audio_fmt == 0xFFFE) && bits == 16 && channels == 1 && rate == 16_000;
                }
                b"data" => data = Some(&wav[body_start..body_end]),
                _ => {}
            }
            off = body_start.saturating_add(sz).saturating_add(sz & 1);
        }
        if !fmt_ok { return None; }
        let data = data?;
        let n = data.len() / 2;
        Some((0..n).map(|i| i16::from_le_bytes([data[i * 2], data[i * 2 + 1]])).collect())
    }
    #[cfg(test)]
    mod tests {
        use super::*;
        fn ok(body: &str) -> Vec<String> { parse_inputs(body).expect("should parse") }

        #[test]
        fn parse_inputs_single_and_array() {
            assert_eq!(ok(r#"{"input":"hello"}"#), vec!["hello".to_string()]);
            assert_eq!(ok(r#"{"input":["a","b"]}"#), vec!["a".to_string(), "b".to_string()]);
        }

        /// The three defects found by indexing the KB through the engine (2026-07-27). Each returned
        /// a WRONG answer with HTTP 200, which is worse than an error: `rest.find('[')` treated a
        /// bracket inside a string value as the start of an array, `arr.find(']')` ended the array at
        /// the first bracket inside a string, and `split('"')` on odd indices was escape-unaware.
        #[test]
        fn parse_inputs_survives_the_three_measured_defects() {
            // 1. a bracket in a single input parsed as an array of nothing -> 0 embeddings, HTTP 200.
            // `rest.find('[')` fired on the FIRST bracket wherever it sat, so one is enough to
            // reproduce; the case that found this in the wild carried a doubled-bracket link.
            assert_eq!(ok(r#"{"input":"see [a link] here"}"#),
                vec!["see [a link] here".to_string()]);
            // 2. a `]` inside one element truncated the batch
            let many: Vec<String> = (0..64)
                .map(|i| if i == 7 { "a ] bracket".to_string() } else { format!("t{i}") }).collect();
            let body = format!("{{\"input\":[{}]}}",
                many.iter().map(|t| format!("\"{}\"", t)).collect::<Vec<_>>().join(","));
            assert_eq!(ok(&body), many);
            // 3. an escaped quote split one input into two
            assert_eq!(ok(r#"{"input":["a","he said \"hi\"","c"]}"#),
                vec!["a".to_string(), "he said \"hi\"".to_string(), "c".to_string()]);
        }

        /// Gate 1's corpus. Every case asserts a specific value -- "did not crash" is not a pass,
        /// because the defining bug returned 200 with an empty body.
        #[test]
        fn parse_inputs_adversarial_corpus() {
            // brackets, braces, backslashes, leading dashes
            assert_eq!(ok(r#"{"input":"- a bullet"}"#), vec!["- a bullet".to_string()]);
            assert_eq!(ok(r#"{"input":"-- a flag"}"#), vec!["-- a flag".to_string()]);
            assert_eq!(ok(r#"{"input":"a lone { brace"}"#), vec!["a lone { brace".to_string()]);
            assert_eq!(ok(r#"{"input":"back\\slash"}"#), vec!["back\\slash".to_string()]);
            // whitespace escapes survive as characters, not as literals
            assert_eq!(ok(r#"{"input":"line\nnext\ttab"}"#), vec!["line\nnext\ttab".to_string()]);
            // empty and whitespace-only are inputs, not absences
            assert_eq!(ok(r#"{"input":""}"#), vec![String::new()]);
            assert_eq!(ok(r#"{"input":"   "}"#), vec!["   ".to_string()]);
            // non-ASCII and emoji, literal and \u-escaped, including a surrogate pair
            assert_eq!(ok(r#"{"input":"привет 🌍"}"#), vec!["привет 🌍".to_string()]);
            assert_eq!(ok(r#"{"input":"при"}"#), vec!["при".to_string()]);
            assert_eq!(ok(r#"{"input":"🌍"}"#), vec!["🌍".to_string()]);
            // field order must not matter, and `model` must never be mistaken for the input
            assert_eq!(ok(r#"{"model":"bge","input":"x"}"#), vec!["x".to_string()]);
            assert_eq!(ok(r#"{"input":"x","model":"bge"}"#), vec!["x".to_string()]);
            // a value that merely CONTAINS the key name is not the key
            assert_eq!(ok(r#"{"model":"has \"input\": inside","input":"real"}"#),
                vec!["real".to_string()]);
            // longer than any model window: length is the caller's problem, not the parser's
            let long = "x".repeat(100_000);
            assert_eq!(ok(&format!("{{\"input\":\"{long}\"}}")), vec![long]);
            // a batch mixing all of the above
            assert_eq!(ok(r#"{"input":["[l]","he \"said\"","- b","🌍",""]}"#),
                vec!["[l]".to_string(), "he \"said\"".to_string(), "- b".to_string(),
                     "🌍".to_string(), String::new()]);
            assert_eq!(ok(r#"{"input":[]}"#), Vec::<String>::new());
        }

        /// Malformed input must be an error the route can turn into a 400 -- never a silent empty
        /// `data` list, and never the old "treat the whole body as the text" fallback.
        #[test]
        fn parse_inputs_rejects_malformed_instead_of_guessing() {
            for bad in [
                "",                              // no body at all
                "not json",
                r#"{"model":"bge"}"#,            // no input field
                r#"{"input":}"#,                 // no value
                r#"{"input":"unterminated"#,     // unterminated string
                r#"{"input":["a","b""#,          // unterminated array
                r#"{"input":123}"#,              // wrong type
                r#"{"input":[1,2]}"#,            // wrong element type
                r#"{"input":"bad \q escape"}"#,
                r#"{"input":"\ud83c only a high surrogate"}"#,
            ] {
                assert!(parse_inputs(bad).is_err(), "must reject {bad:?}");
            }
        }
        #[test]
        fn json_escape_quotes_and_newlines() { assert_eq!(json_escape("a\"b\nc"), "a\\\"b\\nc"); }
        #[test]
        fn parse_wav_rejects_non_riff() { assert!(parse_wav_i16(b"not a wav").is_none()); }
        #[test]
        fn form_field_reads_model_and_ignores_the_upload() {
            let b = "X";
            let body = concat!(
                "--X\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\ngigaam\r\n",
                "--X\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n\r\nRIFF\r\n",
                "--X--\r\n").as_bytes();
            assert_eq!(extract_form_field(body, b, "model").as_deref(), Some("gigaam"));
            assert_eq!(extract_form_field(body, b, "language"), None);
            // The file part must not be mistaken for a text field even when asked for by name.
            assert_eq!(extract_form_field(body, b, "file"), None);
            assert_eq!(extract_file_part(body, b), Some(&b"RIFF"[..]));
        }

        #[test]
        fn parse_chat_request_rejects_malformed_bodies() {
            for bad in [
                "not json",
                "{}",                                        // no messages
                r#"{"messages":[]}"#,                         // empty
                r#"{"messages":[{"content":"hi"}]}"#,         // no role
                r#"{"messages":[{"role":"user"}]}"#,          // no content
                r#"{"messages":[{"role":"user","content":123}]}"#, // wrong content type
                r#"{"messages":[{"role":"user","content":"hi"}],"stop":5}"#,
                r#"{"messages":[{"role":"user","content":"hi"}],"temperature":"hot"}"#,
            ] {
                assert!(parse_chat_request(bad).is_err(), "must reject {bad:?}");
            }
        }

        #[test]
        fn parse_completion_request_rejects_malformed_bodies() {
            for bad in ["not json", "{}", r#"{"prompt":123}"#, r#"{"prompt":["a","b"]}"#] {
                assert!(parse_completion_request(bad).is_err(), "must reject {bad:?}");
            }
        }

        #[test]
        fn parse_completion_request_accepts_a_bare_string_prompt() {
            let p = parse_completion_request(r#"{"prompt":"hello"}"#).unwrap();
            assert!(matches!(p.prompt, npu_engine::Prompt::Raw(s) if s == "hello"));
            assert!(!p.stream);
        }
    }
}

#[cfg(test)]
mod route_tests {
    use super::*;
    use crate::actor::start;
    use crate::config::{Config, ModelCfg, ServerCfg};
    use crate::loader::mock::MockLoader;
    use std::collections::BTreeMap;

    fn get(path: &str) -> Request { Request { method: "GET".into(), path: path.into(), boundary: String::new(), body: vec![] } }
    fn post(path: &str, body: &str) -> Request { Request { method: "POST".into(), path: path.into(), boundary: String::new(), body: body.as_bytes().to_vec() } }

    fn mock_handle() -> (Handle, std::thread::JoinHandle<()>, tempfile::TempDir, PathBuf) {
        let mut t = BTreeMap::new();
        t.insert("bge".to_string(), Ok((Capability::EMBED, 1)));
        t.insert("c".to_string(), Ok((Capability::EMBED, 1)));
        let dir = tempfile::tempdir().unwrap();
        let cfg_path = dir.path().join("engine.toml");
        let cfg = Config {
            server: ServerCfg { max_resident: 8, ..Default::default() },
            models: vec![ModelCfg { name: "bge".into(), scenario: "x".into() }],
            ..Default::default()
        };
        cfg.save(&cfg_path).unwrap();
        let (h, j) = start(cfg, Box::new(MockLoader { table: t })).unwrap();
        (h, j, dir, cfg_path)
    }

    #[test]
    fn healthz_models_chat_and_unknown() {
        let (h, j, _d, p) = mock_handle();
        assert_eq!(route(&get("/healthz"), &h, &p).0, 200);
        let (code, body) = route(&get("/v1/models"), &h, &p);
        assert_eq!(code, 200);
        assert!(body.text().contains("\"id\":\"bge\"") && body.text().contains("\"state\":\"loaded\""));
        // A resident model reports how long it has been idle, so a swap is observable from outside.
        assert!(body.text().contains("\"idle_s\":0"), "{body}");
        assert_eq!(route(&get("/nope"), &h, &p).0, 404);
        assert_eq!(route(&get("/health"), &h, &p).1.text(), "{\"status\":\"ok\"}");
        h.shutdown(); j.join().unwrap();
    }
    /// Both new routes resolve a capability through the rail. With no generate or tts model
    /// configured they answer 503 (a server-configuration fact) rather than the old hardcoded 501,
    /// and rather than 400, which would blame the client for the server having no model.
    #[test]
    fn chat_and_speech_report_no_model_as_503() {
        let (h, j, _d, p) = mock_handle();
        let (code, body) = route(&post("/v1/chat/completions",
            r#"{"messages":[{"role":"user","content":"hi"}]}"#), &h, &p);
        assert_eq!(code, 503, "{body}");
        assert!(body.text().contains("no generate model configured"), "{body}");
        let (code, body) = route(&post("/v1/audio/speech", r#"{"input":"hi"}"#), &h, &p);
        assert_eq!(code, 503, "{body}");
        assert!(body.text().contains("no tts model configured"), "{body}");
        // A malformed body is still the client's fault, and is distinguished from the above.
        assert_eq!(route(&post("/v1/chat/completions", "{}"), &h, &p).0, 400);
        assert_eq!(route(&post("/v1/audio/speech", "{}"), &h, &p).0, 400);
        h.shutdown(); j.join().unwrap();
    }

    /// `/v1/audio/speech` against a model that DOES declare the capability: the rail carries it end
    /// to end, and speech comes back as audio bytes rather than JSON. (The equivalent for
    /// `/v1/chat/completions` needs a `TextGenerator`, which `MockModel` does not implement -- see
    /// `mod generate_tests` below.)
    #[test]
    fn speech_serves_when_a_model_declares_the_capability() {
        let mut t = BTreeMap::new();
        t.insert("tts".to_string(), Ok((Capability::TTS, 1)));
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("engine.toml");
        let cfg = Config {
            server: ServerCfg { max_resident: 2, idle_unload_s: 0, ..Default::default() },
            models: vec![ModelCfg { name: "tts".into(), scenario: "y".into() }],
            ..Default::default()
        };
        cfg.save(&p).unwrap();
        let (h, j) = start(cfg, Box::new(MockLoader { table: t })).unwrap();

        let (code, body) = route(&post("/v1/audio/speech", r#"{"input":"hello"}"#), &h, &p);
        assert_eq!(code, 200, "{body}");
        assert_eq!(body.content_type(), "audio/wav");
        let wav = body.bytes();
        assert_eq!(&wav[..4], b"RIFF");
        // The mock speaks at 24 kHz; the header must carry the model's rate, not a 16 kHz constant.
        assert_eq!(u32::from_le_bytes([wav[24], wav[25], wav[26], wav[27]]), 24_000);
        assert_eq!(parse::parse_wav_i16(wav).map(|v| v.len()), None,
            "parse_wav_i16 only accepts the 16 kHz ASR shape, so it must reject 24 kHz speech");
        h.shutdown(); j.join().unwrap();
    }

    /// `/healthz` must go 503 once a model has failed, and must NOT be tripped by a model that is
    /// merely unloaded -- deferral and idle-sweep are deliberate, and a health check that cries wolf
    /// on them gets ignored.
    /// Build a server from (name, load-result) pairs, so a test can put a model in a chosen state.
    fn health_setup(models: &[(&str, bool)], max_resident: usize)
        -> (Handle, std::thread::JoinHandle<()>, tempfile::TempDir, PathBuf) {
        let mut t = BTreeMap::new();
        for (n, ok) in models {
            t.insert((*n).to_string(),
                if *ok { Ok((Capability::EMBED, 1)) } else { Err("no such xclbin".to_string()) });
        }
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("engine.toml");
        let cfg = Config {
            server: ServerCfg { max_resident, idle_unload_s: 0, ..Default::default() },
            models: models.iter()
                .map(|(n, _)| ModelCfg { name: (*n).into(), scenario: "x".into() }).collect(),
            ..Default::default()
        };
        cfg.save(&p).unwrap();
        let (h, j) = start(cfg, Box::new(MockLoader { table: t })).unwrap();
        (h, j, dir, p)
    }

    /// A model left unloaded by `max_resident` is deliberate. A health check that cries wolf on it
    /// gets ignored, which would defeat the point of the one below.
    #[test]
    fn healthz_is_ok_when_a_model_is_merely_deferred() {
        let (h, j, _d, p) = health_setup(&[("bge", true), ("e5", true)], 1);
        let (code, body) = route(&get("/healthz"), &h, &p);
        assert_eq!(code, 200, "{body}");
        assert!(body.text().contains("\"ok\":true") && body.text().contains("\"failed\":[]"), "{body}");
        h.shutdown(); j.join().unwrap();
    }

    /// ...but a model that FAILED to load makes the service unhealthy and names itself. `ok` used to
    /// be the literal `true`, which is how a 5-day outage looked healthy to systemd.
    #[test]
    fn healthz_is_503_and_names_the_model_that_failed() {
        // Two slots so `broken` is actually attempted rather than deferred by capacity.
        let (h, j, _d, p) = health_setup(&[("bge", true), ("broken", false)], 2);
        let (code, body) = route(&get("/healthz"), &h, &p);
        assert_eq!(code, 503, "a failed model must make the service unhealthy: {body}");
        assert!(body.text().contains("\"ok\":false"), "{body}");
        assert!(body.text().contains("\"broken\""), "the failing model is named: {body}");
        assert!(body.text().contains("\"loaded\":1"), "the healthy one is still counted: {body}");
        h.shutdown(); j.join().unwrap();
    }

    #[test]
    fn embeddings_echoes_model() {
        let (h, j, _d, p) = mock_handle();
        let (code, body) = route(&post("/v1/embeddings", r#"{"input":"hi"}"#), &h, &p);
        assert_eq!(code, 200);
        assert!(body.text().contains("\"model\":\"bge\""), "{body}");
        h.shutdown(); j.join().unwrap();
    }
    #[test]
    fn admin_add_then_models_reflects_it() {
        let (h, j, _d, p) = mock_handle();
        let (code, _) = route(&post("/admin/models", r#"{"name":"c","scenario":"z.toml"}"#), &h, &p);
        assert_eq!(code, 200);
        let (_, body) = route(&get("/v1/models"), &h, &p);
        assert!(body.text().contains("\"id\":\"c\""), "added model missing: {body}");
        // and it persisted to the config file
        let cfg = Config::load(&p).unwrap();
        assert!(cfg.find("c").is_some());
        h.shutdown(); j.join().unwrap();
    }
    #[test]
    fn models_shows_a_swap_at_one_slot() {
        // One slot, two configured models: /v1/models is where an operator sees which one holds the
        // device right now, and what happened to the other.
        let mut t = BTreeMap::new();
        t.insert("bge".to_string(), Ok((Capability::EMBED, 1)));
        t.insert("e5".to_string(), Ok((Capability::EMBED, 1)));
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("engine.toml");
        let cfg = Config {
            server: ServerCfg { max_resident: 1, idle_unload_s: 0, ..Default::default() },
            models: vec![
                ModelCfg { name: "bge".into(), scenario: "x".into() },
                ModelCfg { name: "e5".into(), scenario: "y".into() },
            ],
            ..Default::default()
        };
        cfg.save(&p).unwrap();
        let (h, j) = start(cfg, Box::new(MockLoader { table: t })).unwrap();
        let (code, body) = route(&post("/v1/embeddings", r#"{"model":"e5","input":"hi"}"#), &h, &p);
        assert_eq!(code, 200, "{body}");
        assert!(body.text().contains("\"model\":\"e5\""), "{body}");
        let (_, models) = route(&get("/v1/models"), &h, &p);
        assert!(models.text().contains("\"id\":\"e5\",\"object\":\"model\",\"kind\":\"embed\",\"state\":\"loaded\""), "{models}");
        assert!(models.text().contains("\"idle_s\":null"), "the evicted model reports no idle time: {models}");
        h.shutdown(); j.join().unwrap();
    }

    #[test]
    fn diarizations_renders_segments_as_labelled_spans() {
        let segs = vec![
            npu_engine::capability::Segment { start_s: 0.5, end_s: 3.25, speaker: 0 },
            npu_engine::capability::Segment { start_s: 3.0, end_s: 4.0, speaker: 11 },
        ];
        let body = segments_json("pyannote-3.1", &segs);
        assert!(body.contains("\"model\":\"pyannote-3.1\""), "{body}");
        assert!(body.contains("\"start\":0.500"), "{body}");
        assert!(body.contains("\"end\":3.250"), "{body}");
        assert!(body.contains("\"speaker\":\"SPEAKER_00\""), "index 0 renders zero-padded: {body}");
        assert!(body.contains("\"speaker\":\"SPEAKER_11\""), "{body}");
        // Overlapping spans are legal and must both survive.
        assert_eq!(body.matches("\"start\"").count(), 2, "{body}");
    }

    #[test]
    fn an_empty_diarization_is_a_valid_empty_list_not_an_error() {
        assert_eq!(segments_json("m", &[]), "{\"model\":\"m\",\"segments\":[]}");
    }
}

/// Generation-surface tests: `/v1/chat/completions` and `/v1/completions`, buffered and streaming.
/// `MockModel` (used everywhere else in this file) has no `TextGenerator`, so this module builds its
/// own fixture -- a scripted generator that emits a fixed token list with a settable per-token delay,
/// per the task's own prescription for testing this surface without a real decoder.
#[cfg(test)]
mod generate_tests {
    use super::*;
    use crate::actor::start;
    use crate::config::{Config, ModelCfg, ServerCfg};
    use crate::loader::{ModelLoader, Servable, StreamServable};
    use npu_engine::capability::{Capability, Request as EngineReq, Response as EngineResp};
    use npu_engine::{Chunk, FinishReason, GenerateParams, GenerateUsage, Prompt, TextGenerator};
    use npu_engine::EngineError;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    fn post(path: &str, body: &str) -> Request {
        Request { method: "POST".into(), path: path.into(), boundary: String::new(), body: body.as_bytes().to_vec() }
    }
    fn ss(v: &[&str]) -> Vec<String> { v.iter().map(|s| s.to_string()).collect() }
    fn close(a: f32, b: f32) -> bool { (a - b).abs() < 1e-6 }

    type Seen = Arc<Mutex<Option<(Prompt, GenerateParams)>>>;

    /// Emits `tokens` in order, one `Chunk::Text` per call to `sink` (with `delay` between them),
    /// then `Chunk::Done`. Records every `(prompt, params)` it was invoked with, and how many tokens
    /// it managed to SEND before the sink said stop -- what the disconnect test observes.
    struct ScriptedGenerator { tokens: Vec<String>, delay: Duration, sent: Arc<AtomicUsize>, seen: Seen }
    impl TextGenerator for ScriptedGenerator {
        fn generate(&mut self, prompt: &Prompt, params: &GenerateParams,
            sink: &mut dyn FnMut(Chunk<'_>) -> bool) -> Result<(), EngineError> {
            *self.seen.lock().unwrap() = Some((prompt.clone(), params.clone()));
            let mut usage = GenerateUsage::default();
            let cap = (params.max_tokens as usize).min(self.tokens.len());
            for tok in self.tokens.iter().take(cap) {
                if !self.delay.is_zero() { std::thread::sleep(self.delay); }
                self.sent.fetch_add(1, Ordering::SeqCst);
                usage.completion_tokens += 1;
                if !sink(Chunk::Text(tok)) {
                    let _ = sink(Chunk::Done { reason: FinishReason::Aborted, usage });
                    return Ok(());
                }
            }
            let reason = if cap < self.tokens.len() { FinishReason::Length } else { FinishReason::Stop };
            sink(Chunk::Done { reason, usage });
            Ok(())
        }
    }

    struct GenModel { gen: ScriptedGenerator }
    impl Servable for GenModel {
        fn capabilities(&self) -> Capability { Capability::GENERATE }
        fn run(&mut self, _req: EngineReq) -> Result<EngineResp, EngineError> {
            Err(EngineError::Unsupported("use generate_stream".into()))
        }
    }
    impl StreamServable for GenModel {
        fn generate_stream(&mut self, prompt: &Prompt, params: &GenerateParams,
            sink: &mut dyn FnMut(Chunk<'_>) -> bool) -> Result<(), EngineError> {
            self.gen.generate(prompt, params, sink)
        }
    }

    struct GenLoader { tokens: Vec<String>, delay: Duration, sent: Arc<AtomicUsize>, seen: Seen }
    impl ModelLoader for GenLoader {
        fn load(&self, _cfg: &ModelCfg) -> Result<Box<dyn StreamServable>, EngineError> {
            Ok(Box::new(GenModel { gen: ScriptedGenerator {
                tokens: self.tokens.clone(), delay: self.delay, sent: self.sent.clone(), seen: self.seen.clone(),
            }}))
        }
        fn declared_capability(&self, _cfg: &ModelCfg) -> Option<Capability> { Some(Capability::GENERATE) }
    }

    fn gen_handle(tokens: Vec<String>, delay: Duration)
        -> (Handle, std::thread::JoinHandle<()>, tempfile::TempDir, PathBuf, Arc<AtomicUsize>, Seen) {
        let sent = Arc::new(AtomicUsize::new(0));
        let seen: Seen = Arc::new(Mutex::new(None));
        let loader = GenLoader { tokens, delay, sent: sent.clone(), seen: seen.clone() };
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("engine.toml");
        let cfg = Config {
            server: ServerCfg { max_resident: 1, idle_unload_s: 0, ..Default::default() },
            models: vec![ModelCfg { name: "llm".into(), scenario: "x".into() }],
            ..Default::default()
        };
        cfg.save(&p).unwrap();
        let (h, j) = start(cfg, Box::new(loader)).unwrap();
        (h, j, dir, p, sent, seen)
    }

    #[test]
    fn non_streaming_chat_completion_returns_full_text_and_the_whole_message_array() {
        let (h, j, _d, p, _sent, seen) = gen_handle(ss(&["Hello", ", ", "world"]), Duration::ZERO);
        let (code, body) = route(&post("/v1/chat/completions",
            r#"{"messages":[{"role":"system","content":"be terse"},{"role":"user","content":"hi"}]}"#), &h, &p);
        assert_eq!(code, 200, "{body}");
        let v: serde_json::Value = serde_json::from_str(body.text()).unwrap();
        assert_eq!(v["object"], "chat.completion");
        assert_eq!(v["choices"][0]["message"]["role"], "assistant");
        assert_eq!(v["choices"][0]["message"]["content"], "Hello, world");
        assert_eq!(v["choices"][0]["finish_reason"], "stop");
        assert_eq!(v["usage"]["completion_tokens"], 3);
        assert!(v["id"].is_string() && v["created"].is_number());
        // THE bug: the full array (system + user), not just the last turn, must reach the generator.
        let (prompt, _) = seen.lock().unwrap().clone().unwrap();
        match prompt {
            Prompt::Chat(msgs) => {
                assert_eq!(msgs.len(), 2, "{msgs:?}");
                assert_eq!((msgs[0].role.as_str(), msgs[0].content.as_str()), ("system", "be terse"));
                assert_eq!((msgs[1].role.as_str(), msgs[1].content.as_str()), ("user", "hi"));
            }
            Prompt::Raw(_) => panic!("chat completions must produce Prompt::Chat"),
        }
        h.shutdown(); j.join().unwrap();
    }

    #[test]
    fn every_sampling_param_round_trips() {
        let (h, j, _d, p, _sent, seen) = gen_handle(ss(&["ok"]), Duration::ZERO);
        let body = r#"{"messages":[{"role":"user","content":"hi"}],
            "temperature":0.3,"top_p":0.5,"top_k":40,"max_tokens":7,
            "seed":42,"presence_penalty":0.1,"frequency_penalty":0.2,"repetition_penalty":1.3,
            "stop":"STOP"}"#;
        let (code, resp) = route(&post("/v1/chat/completions", body), &h, &p);
        assert_eq!(code, 200, "{resp}");
        let (_, params) = seen.lock().unwrap().clone().unwrap();
        assert!(close(params.temperature, 0.3), "{}", params.temperature);
        assert!(close(params.top_p, 0.5), "{}", params.top_p);
        assert_eq!(params.top_k, 40);
        assert_eq!(params.max_tokens, 7);
        assert_eq!(params.seed, Some(42));
        assert!(close(params.presence_penalty, 0.1), "{}", params.presence_penalty);
        assert!(close(params.frequency_penalty, 0.2), "{}", params.frequency_penalty);
        assert!(close(params.repetition_penalty, 1.3), "{}", params.repetition_penalty);
        assert_eq!(params.stop, vec!["STOP".to_string()]);
        h.shutdown(); j.join().unwrap();
    }

    #[test]
    fn openai_defaults_apply_when_fields_are_absent() {
        let (h, j, _d, p, _sent, seen) = gen_handle(ss(&["ok"]), Duration::ZERO);
        let (code, resp) = route(&post("/v1/chat/completions",
            r#"{"messages":[{"role":"user","content":"hi"}]}"#), &h, &p);
        assert_eq!(code, 200, "{resp}");
        let (_, params) = seen.lock().unwrap().clone().unwrap();
        let d = GenerateParams::default();
        assert!(close(params.temperature, d.temperature), "default temperature must be 1.0, not greedy");
        assert!(close(params.top_p, d.top_p));
        assert_eq!(params.top_k, 0);
        assert_eq!(params.max_tokens, 256);
        assert!(params.stop.is_empty());
        assert_eq!(params.seed, None);
        h.shutdown(); j.join().unwrap();
    }

    #[test]
    fn stop_accepts_a_string_or_an_array() {
        let (h, j, _d, p, _sent, seen) = gen_handle(ss(&["ok"]), Duration::ZERO);
        route(&post("/v1/chat/completions",
            r#"{"messages":[{"role":"user","content":"hi"}],"stop":"END"}"#), &h, &p);
        assert_eq!(seen.lock().unwrap().clone().unwrap().1.stop, vec!["END".to_string()]);
        route(&post("/v1/chat/completions",
            r#"{"messages":[{"role":"user","content":"hi"}],"stop":["A","B"]}"#), &h, &p);
        assert_eq!(seen.lock().unwrap().clone().unwrap().1.stop, vec!["A".to_string(), "B".to_string()]);
        let (code, body) = route(&post("/v1/chat/completions",
            r#"{"messages":[{"role":"user","content":"hi"}],"stop":5}"#), &h, &p);
        assert_eq!(code, 400, "{body}");
        h.shutdown(); j.join().unwrap();
    }

    #[test]
    fn max_tokens_truncates_and_reports_length() {
        let (h, j, _d, p, _sent, _seen) = gen_handle(ss(&["a", "b", "c", "d", "e"]), Duration::ZERO);
        let (code, body) = route(&post("/v1/chat/completions",
            r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":2}"#), &h, &p);
        assert_eq!(code, 200, "{body}");
        let v: serde_json::Value = serde_json::from_str(body.text()).unwrap();
        assert_eq!(v["choices"][0]["finish_reason"], "length");
        assert_eq!(v["choices"][0]["message"]["content"], "ab");
        h.shutdown(); j.join().unwrap();
    }

    /// A multi-byte codepoint split across tokens produces an empty `Chunk::Text` until it completes
    /// -- that must be forwarded, not filtered, and must not corrupt the reassembled text.
    #[test]
    fn an_empty_text_chunk_is_forwarded_without_corrupting_the_result() {
        let (h, j, _d, p, _sent, _seen) = gen_handle(ss(&["", "hi", ""]), Duration::ZERO);
        let (code, body) = route(&post("/v1/chat/completions",
            r#"{"messages":[{"role":"user","content":"x"}]}"#), &h, &p);
        assert_eq!(code, 200, "{body}");
        let v: serde_json::Value = serde_json::from_str(body.text()).unwrap();
        assert_eq!(v["choices"][0]["message"]["content"], "hi");
        h.shutdown(); j.join().unwrap();
    }

    #[test]
    fn unsupported_sampling_params_are_rejected_not_silently_dropped() {
        let (h, j, _d, p, _sent, _seen) = gen_handle(ss(&["x"]), Duration::ZERO);
        for body in [
            r#"{"messages":[{"role":"user","content":"hi"}],"n":2}"#,
            r#"{"messages":[{"role":"user","content":"hi"}],"logprobs":true}"#,
            r#"{"messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function"}]}"#,
            r#"{"messages":[{"role":"user","content":"hi"}],"logit_bias":{"123":10}}"#,
        ] {
            let (code, resp) = route(&post("/v1/chat/completions", body), &h, &p);
            assert_eq!(code, 400, "{body}: got {resp}");
        }
        // ...but explicit no-op values (OpenAI clients send these routinely) must still pass.
        let (code, resp) = route(&post("/v1/chat/completions",
            r#"{"messages":[{"role":"user","content":"hi"}],"n":1,"logprobs":false}"#), &h, &p);
        assert_eq!(code, 200, "{resp}");
        h.shutdown(); j.join().unwrap();
    }

    #[test]
    fn a_multipart_content_array_of_text_parts_is_flattened_and_other_types_are_rejected() {
        let (h, j, _d, p, _sent, seen) = gen_handle(ss(&["ok"]), Duration::ZERO);
        let (code, resp) = route(&post("/v1/chat/completions",
            r#"{"messages":[{"role":"user","content":[{"type":"text","text":"a"},{"type":"text","text":"b"}]}]}"#),
            &h, &p);
        assert_eq!(code, 200, "{resp}");
        match seen.lock().unwrap().clone().unwrap().0 {
            Prompt::Chat(msgs) => assert_eq!(msgs[0].content, "ab"),
            _ => panic!("expected Prompt::Chat"),
        }
        let (code, resp) = route(&post("/v1/chat/completions",
            r#"{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"x"}}]}]}"#),
            &h, &p);
        assert_eq!(code, 400, "an unsupported content part must fail loud, not drop silently: {resp}");
        h.shutdown(); j.join().unwrap();
    }

    #[test]
    fn completions_endpoint_uses_a_raw_untemplated_prompt() {
        let (h, j, _d, p, _sent, seen) = gen_handle(ss(&["once", " upon", " a time"]), Duration::ZERO);
        let (code, body) = route(&post("/v1/completions", r#"{"prompt":"Tell me a story"}"#), &h, &p);
        assert_eq!(code, 200, "{body}");
        let v: serde_json::Value = serde_json::from_str(body.text()).unwrap();
        assert_eq!(v["object"], "text_completion");
        assert_eq!(v["choices"][0]["text"], "once upon a time");
        match seen.lock().unwrap().clone().unwrap().0 {
            Prompt::Raw(s) => assert_eq!(s, "Tell me a story"),
            Prompt::Chat(_) => panic!("/v1/completions must not go through the chat template"),
        }
        h.shutdown(); j.join().unwrap();
    }

    #[test]
    fn completions_rejects_an_array_prompt_with_a_clear_400() {
        let (h, j, _d, p, _sent, _seen) = gen_handle(ss(&["x"]), Duration::ZERO);
        let (code, body) = route(&post("/v1/completions", r#"{"prompt":["a","b"]}"#), &h, &p);
        assert_eq!(code, 400, "{body}");
        assert!(body.text().contains("not supported"), "{body}");
        h.shutdown(); j.join().unwrap();
    }

    /// SSE shape: a role-only preamble, then one content delta per token (verified structurally, not
    /// by exact string match, since `id`/`created` are per-request), then a terminal empty-delta
    /// chunk carrying `finish_reason`. `[DONE]` itself is written by `respond_stream`, which needs a
    /// real socket -- covered by `tests/streaming.rs`.
    #[test]
    fn chat_completion_sse_frames_are_role_then_content_then_finish() {
        let (h, j, _d, p, _sent, _seen) = gen_handle(ss(&["Hello", ", ", "world"]), Duration::ZERO);
        let (code, body) = route(&post("/v1/chat/completions",
            r#"{"messages":[{"role":"user","content":"hi"}],"stream":true}"#), &h, &p);
        assert_eq!(code, 200);
        let Body::Stream(s) = body else { panic!("expected a streaming body") };

        let role: serde_json::Value = serde_json::from_str(&s.render_role()).unwrap();
        assert_eq!(role["object"], "chat.completion.chunk");
        assert_eq!(role["choices"][0]["delta"]["role"], "assistant");
        assert!(role["choices"][0]["delta"].get("content").is_none());
        assert!(role["choices"][0]["finish_reason"].is_null());

        let mut texts = Vec::new();
        let mut got_done = false;
        for item in s.rx.iter() {
            match item {
                StreamItem::Text(t) => {
                    let frame: serde_json::Value = serde_json::from_str(&s.render_text(&t)).unwrap();
                    assert_eq!(frame["object"], "chat.completion.chunk");
                    assert_eq!(frame["choices"][0]["delta"]["content"], t);
                    assert!(frame["choices"][0]["finish_reason"].is_null());
                    texts.push(t);
                }
                StreamItem::Done { reason, .. } => {
                    let frame: serde_json::Value = serde_json::from_str(&s.render_done(reason)).unwrap();
                    assert_eq!(frame["choices"][0]["finish_reason"], reason.as_str());
                    assert_eq!(frame["choices"][0]["delta"], serde_json::json!({}));
                    got_done = true;
                }
                StreamItem::Error(e) => panic!("unexpected error: {e}"),
            }
        }
        assert_eq!(texts.join(""), "Hello, world");
        assert!(got_done);
        h.shutdown(); j.join().unwrap();
    }

    #[test]
    fn completions_endpoint_streams_text_deltas_with_no_role_preamble() {
        let (h, j, _d, p, _sent, _seen) = gen_handle(ss(&["a", "b"]), Duration::ZERO);
        let (code, body) = route(&post("/v1/completions", r#"{"prompt":"x","stream":true}"#), &h, &p);
        assert_eq!(code, 200);
        let Body::Stream(s) = body else { panic!("expected a streaming body") };
        let mut saw_text = false;
        for item in s.rx.iter() {
            if let StreamItem::Text(t) = item {
                let frame: serde_json::Value = serde_json::from_str(&s.render_text(&t)).unwrap();
                assert_eq!(frame["object"], "text_completion");
                assert_eq!(frame["choices"][0]["text"], t);
                saw_text = true;
            }
        }
        assert!(saw_text);
        h.shutdown(); j.join().unwrap();
    }

    /// The core of the disconnect requirement: dropping the receiving end (what `respond_stream` does
    /// on a failed write) must make the actor's next `send` fail, which is the sink's abort signal.
    /// `tests/streaming.rs` covers the real-socket half of this (an actual TCP close).
    #[test]
    fn dropping_the_receiver_aborts_generation() {
        let n = 500;
        let tokens: Vec<String> = (0..n).map(|i| format!("t{i}")).collect();
        let (h, j, _d, p, sent, _seen) = gen_handle(tokens, Duration::from_millis(2));
        let (code, body) = route(&post("/v1/chat/completions",
            r#"{"messages":[{"role":"user","content":"hi"}],"stream":true}"#), &h, &p);
        assert_eq!(code, 200, "{body}");
        let Body::Stream(s) = body else { panic!("expected a streaming body") };
        let _ = s.rx.recv();
        let _ = s.rx.recv();
        drop(s);
        std::thread::sleep(Duration::from_millis(200));
        let got = sent.load(Ordering::SeqCst);
        assert!(got < n, "generation must abort on disconnect; sent {got} of {n} tokens");
        h.shutdown(); j.join().unwrap();
    }

    #[test]
    fn admin_defaults_and_selection_route_generate_correctly() {
        let (h, j, _d, p, _sent, _seen) = gen_handle(ss(&["ok"]), Duration::ZERO);
        let (code, body) = route(&post("/admin/defaults", r#"{"capability":"generate","model":"llm"}"#), &h, &p);
        assert_eq!(code, 200, "{body}");
        let cfg = Config::load(&p).unwrap();
        assert_eq!(cfg.defaults.get(Capability::GENERATE).map(String::as_str), Some("llm"));
        let (code, body) = route(&post("/v1/chat/completions",
            r#"{"messages":[{"role":"user","content":"hi"}]}"#), &h, &p);
        assert_eq!(code, 200, "{body}");
        h.shutdown(); j.join().unwrap();
    }
}
