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
use npu_engine::ModelKind;

const MAX_BODY: usize = 16 * 1024 * 1024;
const SOCKET_TIMEOUT: Duration = Duration::from_secs(60);

/// A parsed request, enough for routing.
pub struct Request {
    pub method: String,
    pub path: String,
    pub boundary: String,
    pub body: Vec<u8>,
}

/// (status code, JSON body).
pub type Response = (u16, String);

/// Pure routing decision. Mutating admin routes load/edit/save the config at `cfg_path` then ask the
/// actor to reconcile. No socket here -> unit-testable with a mock-backed Handle.
pub fn route(req: &Request, handle: &Handle, cfg_path: &Path) -> Response {
    match (req.method.as_str(), req.path.as_str()) {
        ("GET", "/health") => (200, "{\"status\":\"ok\"}".into()),
        ("GET", "/healthz") => {
            let npu = npu_engine::Engine::available();
            let n = handle.status().iter().filter(|s| s.state == LoadState::Loaded).count();
            (200, format!("{{\"ok\":true,\"npu\":{npu},\"loaded\":{n}}}"))
        }
        ("GET", "/v1/models") => (200, models_json(&handle.status())),
        ("POST", "/v1/chat/completions") =>
            (501, "{\"error\":\"not implemented: LLM decode track pending\"}".into()),
        ("POST", "/v1/embeddings") => embeddings(req, handle),
        ("POST", "/v1/audio/transcriptions") => transcriptions(req, handle),
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
pub fn models_json(status: &[ModelStatus]) -> String {
    let mut data = String::new();
    for (i, s) in status.iter().enumerate() {
        if i > 0 { data.push(','); }
        let kind = match s.kind { Some(ModelKind::Asr) => "asr", Some(ModelKind::Embed) => "embed", None => "unknown" };
        let state = match s.state { LoadState::Loaded => "loaded", LoadState::Failed => "failed", LoadState::Unloaded => "unloaded" };
        data.push_str(&format!(
            "{{\"id\":\"{}\",\"object\":\"model\",\"kind\":\"{kind}\",\"state\":\"{state}\",\"detail\":\"{}\",\"bo_bytes\":{}}}",
            s.name, parse::json_escape(&s.detail), s.bo_bytes));
    }
    format!("{{\"object\":\"list\",\"data\":[{data}]}}")
}

fn embeddings(req: &Request, handle: &Handle) -> Response {
    let body = String::from_utf8_lossy(&req.body).to_string();
    let model = extract_str_field(&body, "model");
    let inputs = parse::parse_inputs(&body);
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
            Err(e) => return (400, format!("{{\"error\":\"{}\"}}", parse::json_escape(&e.to_string()))),
        }
    }
    (200, format!("{{\"object\":\"list\",\"data\":[{data}],\"model\":\"{}\"}}", parse::json_escape(&served)))
}

fn transcriptions(req: &Request, handle: &Handle) -> Response {
    let wav = match parse::extract_file_part(&req.body, &req.boundary) {
        Some(w) => w, None => return (400, "{\"error\":\"no file part\"}".into()),
    };
    let samples = match parse::parse_wav_i16(wav) {
        Some(s) if !s.is_empty() => s,
        _ => return (400, "{\"error\":\"bad wav (need 16k mono 16-bit)\"}".into()),
    };
    match handle.transcribe(None, samples, 16_000) {
        Ok(s) => (200, format!("{{\"text\":\"{}\",\"model\":\"{}\"}}",
            parse::json_escape(&s.value), parse::json_escape(&s.model))),
        Err(e) => (500, format!("{{\"error\":\"{}\"}}", parse::json_escape(&e.to_string()))),
    }
}

fn admin_reload(handle: &Handle, cfg_path: &Path) -> Response {
    match Config::load(cfg_path) {
        Ok(cfg) => match handle.reconcile(cfg) {
            Ok(rep) => (200, format!("{{\"loaded\":{},\"unloaded\":{},\"failed\":{}}}",
                rep.loaded.len(), rep.unloaded.len(), rep.failed.len())),
            Err(e) => (500, format!("{{\"error\":\"{}\"}}", parse::json_escape(&e.to_string()))),
        },
        Err(e) => (400, format!("{{\"error\":\"{}\"}}", parse::json_escape(&e))),
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
    mutate_and_reconcile(handle, cfg_path, |cfg| {
        let d = &mut cfg.defaults;
        match cap.as_str() { "asr" => d.asr = Some(model.clone()), "embed" => d.embed = Some(model.clone()), _ => {} }
    })
}

fn mutate_and_reconcile(handle: &Handle, cfg_path: &Path, f: impl FnOnce(&mut Config)) -> Response {
    let mut cfg = match Config::load(cfg_path) { Ok(c) => c, Err(e) => return (400, format!("{{\"error\":\"{}\"}}", parse::json_escape(&e))) };
    f(&mut cfg);
    if let Err(e) = cfg.save(cfg_path) { return (500, format!("{{\"error\":\"{}\"}}", parse::json_escape(&e))); }
    match handle.reconcile(cfg) {
        Ok(rep) => (200, format!("{{\"loaded\":{},\"unloaded\":{},\"failed\":{}}}",
            rep.loaded.len(), rep.unloaded.len(), rep.failed.len())),
        Err(e) => (500, format!("{{\"error\":\"{}\"}}", parse::json_escape(&e.to_string()))),
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
    let addr = format!("127.0.0.1:{port}");
    let listener = TcpListener::bind(&addr)?;
    eprintln!("[npu-serve] ready on http://{addr}");
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
    if content_len > MAX_BODY { return respond(&mut stream, 413, "{\"error\":\"too large\"}"); }
    let mut body = vec![0u8; content_len];
    reader.read_exact(&mut body)?;
    let req = Request { method, path, boundary, body };
    let (code, body) = route(&req, handle, cfg_path);
    respond(&mut stream, code, &body)
}

fn respond(stream: &mut TcpStream, code: u16, body: &str) -> std::io::Result<()> {
    let reason = match code {
        200 => "OK", 400 => "Bad Request", 404 => "Not Found", 413 => "Payload Too Large",
        500 => "Internal Server Error", 501 => "Not Implemented", _ => "Error",
    };
    let resp = format!(
        "HTTP/1.1 {code} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.as_bytes().len());
    stream.write_all(resp.as_bytes())?;
    stream.flush()
}

/// HTTP/JSON/WAV parsing helpers (ported from the C3 npu-serve), pure + unit-tested.
pub mod parse {
    pub fn parse_inputs(body: &str) -> Vec<String> {
        if let Some(idx) = body.find("\"input\"") {
            let rest = &body[idx + 7..];
            if let Some(open) = rest.find('[') {
                let arr = &rest[open + 1..];
                let end = arr.find(']').unwrap_or(arr.len());
                return arr[..end].split('"').enumerate()
                    .filter(|(i, _)| i % 2 == 1).map(|(_, s)| s.to_string()).collect();
            }
            if let Some(q1) = rest.find('"') {
                let s = &rest[q1 + 1..];
                if let Some(q2) = s.find('"') { return vec![s[..q2].to_string()]; }
            }
        }
        vec![body.trim().to_string()]
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
        #[test]
        fn parse_inputs_single_and_array() {
            assert_eq!(parse_inputs(r#"{"input":"hello"}"#), vec!["hello".to_string()]);
            assert_eq!(parse_inputs(r#"{"input":["a","b"]}"#), vec!["a".to_string(), "b".to_string()]);
        }
        #[test]
        fn json_escape_quotes_and_newlines() { assert_eq!(json_escape("a\"b\nc"), "a\\\"b\\nc"); }
        #[test]
        fn parse_wav_rejects_non_riff() { assert!(parse_wav_i16(b"not a wav").is_none()); }
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
        t.insert("bge".to_string(), Ok((ModelKind::Embed, 1)));
        t.insert("c".to_string(), Ok((ModelKind::Embed, 1)));
        let dir = tempfile::tempdir().unwrap();
        let cfg_path = dir.path().join("engine.toml");
        let cfg = Config {
            server: ServerCfg { max_resident: 8, ..Default::default() },
            models: vec![ModelCfg { name: "bge".into(), scenario: "x".into() }],
            ..Default::default()
        };
        cfg.save(&cfg_path).unwrap();
        let (h, j) = start(cfg, Box::new(MockLoader { table: t }));
        (h, j, dir, cfg_path)
    }

    #[test]
    fn healthz_models_chat_and_unknown() {
        let (h, j, _d, p) = mock_handle();
        assert_eq!(route(&get("/healthz"), &h, &p).0, 200);
        let (code, body) = route(&get("/v1/models"), &h, &p);
        assert_eq!(code, 200);
        assert!(body.contains("\"id\":\"bge\"") && body.contains("\"state\":\"loaded\""));
        assert_eq!(route(&post("/v1/chat/completions", "{}"), &h, &p).0, 501);
        assert_eq!(route(&get("/nope"), &h, &p).0, 404);
        assert_eq!(route(&get("/health"), &h, &p), (200, "{\"status\":\"ok\"}".to_string()));
        h.shutdown(); j.join().unwrap();
    }
    #[test]
    fn embeddings_echoes_model() {
        let (h, j, _d, p) = mock_handle();
        let (code, body) = route(&post("/v1/embeddings", r#"{"input":"hi"}"#), &h, &p);
        assert_eq!(code, 200);
        assert!(body.contains("\"model\":\"bge\""), "{body}");
        h.shutdown(); j.join().unwrap();
    }
    #[test]
    fn admin_add_then_models_reflects_it() {
        let (h, j, _d, p) = mock_handle();
        let (code, _) = route(&post("/admin/models", r#"{"name":"c","scenario":"z.toml"}"#), &h, &p);
        assert_eq!(code, 200);
        let (_, body) = route(&get("/v1/models"), &h, &p);
        assert!(body.contains("\"id\":\"c\""), "added model missing: {body}");
        // and it persisted to the config file
        let cfg = Config::load(&p).unwrap();
        assert!(cfg.find("c").is_some());
        h.shutdown(); j.join().unwrap();
    }
}
