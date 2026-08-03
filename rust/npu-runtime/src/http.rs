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
use npu_engine::capability::Capability;

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

fn embeddings(req: &Request, handle: &Handle) -> Response {
    let body = String::from_utf8_lossy(&req.body).to_string();
    let model = extract_str_field(&body, "model");
    let inputs = match parse::parse_inputs(&body) {
        Ok(v) if v.is_empty() => return (400, "{\"error\":\"input is empty\"}".into()),
        Ok(v) => v,
        Err(e) => return (400, format!("{{\"error\":\"{}\"}}", parse::json_escape(&e))),
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
    // OpenAI's transcription request carries `model` as a form field. This route used to drop it and
    // always serve the default, which left ASR -- the capability this engine actually ships -- with
    // no way to pick a model per request even though the actor has always taken one.
    let model = parse::extract_form_field(&req.body, &req.boundary, "model");
    match handle.transcribe(model.as_deref(), samples, 16_000) {
        Ok(s) => (200, format!("{{\"text\":\"{}\",\"model\":\"{}\"}}",
            parse::json_escape(&s.value), parse::json_escape(&s.model))),
        Err(e) => (500, format!("{{\"error\":\"{}\"}}", parse::json_escape(&e.to_string()))),
    }
}

fn admin_reload(handle: &Handle, cfg_path: &Path) -> Response {
    match Config::load(cfg_path) {
        Ok(cfg) => match handle.reconcile(cfg) {
            Ok(rep) => (200, format!("{{\"loaded\":{},\"unloaded\":{},\"failed\":{},\"deferred\":{}}}",
                rep.loaded.len(), rep.unloaded.len(), rep.failed.len(), rep.deferred.len())),
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
    // A capability nothing implements is rejected rather than written: the old match silently
    // dropped anything that was not asr/embed, so `/admin/defaults` reported 200 and changed nothing.
    let cap = match Capability::from_name(&cap) {
        Some(c) => c,
        None => return (400, format!("{{\"error\":\"unknown capability {}\"}}", parse::json_escape(&cap))),
    };
    mutate_and_reconcile(handle, cfg_path, |cfg| cfg.defaults.set(cap, model.clone()))
}

fn mutate_and_reconcile(handle: &Handle, cfg_path: &Path, f: impl FnOnce(&mut Config)) -> Response {
    let mut cfg = match Config::load(cfg_path) { Ok(c) => c, Err(e) => return (400, format!("{{\"error\":\"{}\"}}", parse::json_escape(&e))) };
    f(&mut cfg);
    if let Err(e) = cfg.save(cfg_path) { return (500, format!("{{\"error\":\"{}\"}}", parse::json_escape(&e))); }
    match handle.reconcile(cfg) {
        Ok(rep) => (200, format!("{{\"loaded\":{},\"unloaded\":{},\"failed\":{},\"deferred\":{}}}",
            rep.loaded.len(), rep.unloaded.len(), rep.failed.len(), rep.deferred.len())),
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
        assert!(body.contains("\"id\":\"bge\"") && body.contains("\"state\":\"loaded\""));
        // A resident model reports how long it has been idle, so a swap is observable from outside.
        assert!(body.contains("\"idle_s\":0"), "{body}");
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
        assert!(body.contains("\"model\":\"e5\""), "{body}");
        let (_, models) = route(&get("/v1/models"), &h, &p);
        assert!(models.contains("\"id\":\"e5\",\"object\":\"model\",\"kind\":\"embed\",\"state\":\"loaded\""), "{models}");
        assert!(models.contains("\"idle_s\":null"), "the evicted model reports no idle time: {models}");
        h.shutdown(); j.join().unwrap();
    }
}
