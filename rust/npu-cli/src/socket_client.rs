//! The CLI's transport to the service: every device command (embed, transcribe, transcribe-media,
//! diarize, generate, chat) sends the SAME request an HTTP client would over `/v1/...`, but over
//! `control.sock` instead of a TCP port -- reusing `http::route()` end to end, so there is exactly
//! one implementation of "what does this request do" regardless of which surface asked.
//!
//! No service running means the command FAILS and says how to start one -- there is no in-process
//! fallback left to reach for.
use std::io::{BufRead, BufReader, Read, Write};
use std::os::unix::net::UnixStream;

use anyhow::{Context, Result};

use crate::exit::{Code, Tagged};

/// One message, naming the fix, on every device command run with no service behind the socket.
pub fn refusal() -> anyhow::Error {
    Tagged(Code::NoService, "no xdna-engine service is running, and `npu` needs one -- the service \
        is the single holder of the NPU, so a second in-process copy would take its own hardware \
        context and its own copy of the weights.\n  \
        start:  systemctl --user start xdna-engine\n  \
        status: systemctl --user status xdna-engine\n  \
        logs:   journalctl --user -u xdna-engine -f".to_string()).into()
}

/// No read/write timeout here, deliberately -- unlike `query_control_socket`'s status check, which
/// must answer near-instantly, a device command (a generation, a long transcription) can legitimately
/// take as long as the model takes, exactly the way the actor's own `rx.recv()` never times out
/// waiting for one. A 2s timeout copied from the status-check pattern would fail almost every real
/// generate/transcribe call; refusing when there is no service at all is `connect()` failing
/// immediately (ECONNREFUSED/ENOENT), not a read timing out.
fn connect() -> Option<UnixStream> {
    let path = npu_runtime::control_socket::socket_path()?;
    UnixStream::connect(path).ok()
}

fn write_request(stream: &mut UnixStream, method: &str, path: &str, content_type: Option<&str>,
                  body: &[u8]) -> std::io::Result<()> {
    let mut head = format!("{method} {path} HTTP/1.1\r\n");
    if let Some(ct) = content_type { head.push_str(&format!("Content-Type: {ct}\r\n")); }
    head.push_str(&format!("Content-Length: {}\r\n\r\n", body.len()));
    stream.write_all(head.as_bytes())?;
    stream.write_all(body)
}

fn read_status_line(reader: &mut impl BufRead) -> Result<u16> {
    let mut line = String::new();
    reader.read_line(&mut line)?;
    line.split_whitespace().nth(1).and_then(|s| s.parse().ok())
        .context("control socket: malformed status line")
}

/// The exit code a device command's HTTP status implies. Coarser than `exit::engine_error`'s
/// per-variant match (`engine_err` in http.rs collapses `NotAvailable`+`NoModel` to 503, where the
/// in-process path split them into `Device`/`NoModel`) -- the wire carries a status, not the
/// `EngineError` variant, so this is the closest a socket client can get without a new error field.
fn status_code(status: u16) -> Code {
    match status {
        503 => Code::NoModel,
        400 => Code::Failure,
        _ => Code::Device,
    }
}

/// Turn a non-2xx response into the error every call site raises: the route's own `{"error": ...}`
/// message, tagged with the exit code its status implies.
fn response_error(code: u16, resp: &[u8]) -> anyhow::Error {
    let v: serde_json::Value = serde_json::from_slice(resp).unwrap_or(serde_json::Value::Null);
    let msg = v.get("error").and_then(|e| e.as_str()).map(str::to_string)
        .unwrap_or_else(|| String::from_utf8_lossy(resp).to_string());
    Tagged(status_code(code), msg).into()
}

/// A buffered request/response: connect, send, read the whole body back. Used by every command
/// except the streaming half of `generate`/`chat`.
pub fn call(method: &str, path: &str, content_type: Option<&str>, body: &[u8]) -> Result<(u16, Vec<u8>)> {
    let mut stream = connect().ok_or_else(refusal)?;
    write_request(&mut stream, method, path, content_type, body).map_err(|_| refusal())?;
    let mut reader = BufReader::new(stream);
    let code = read_status_line(&mut reader).map_err(|_| refusal())?;
    let mut content_len = 0usize;
    loop {
        let mut h = String::new();
        if reader.read_line(&mut h)? == 0 { break; }
        let h = h.trim_end();
        if h.is_empty() { break; }
        if let Some(v) = h.to_ascii_lowercase().strip_prefix("content-length:") {
            content_len = v.trim().parse().unwrap_or(0);
        }
    }
    let mut resp_body = vec![0u8; content_len];
    reader.read_exact(&mut resp_body)?;
    Ok((code, resp_body))
}

/// A `POST` with a JSON body -- the common case (embed, generate/chat non-streaming). `Ok` only for
/// a 2xx; the error carries whatever `{"error": "..."}` the route sent, or the raw body if it did
/// not parse as one.
pub fn call_json(path: &str, body: &serde_json::Value) -> Result<serde_json::Value> {
    let bytes = body.to_string();
    let (code, resp) = call("POST", path, Some("application/json"), bytes.as_bytes())?;
    if !(200..300).contains(&code) { return Err(response_error(code, &resp)); }
    serde_json::from_slice(&resp).context("control socket: response was not JSON")
}

/// Like `call_json`, for the one route whose SUCCESS body is not JSON -- `/v1/audio/speech`, which
/// answers audio bytes. Same error handling; only what a 2xx hands back differs.
pub fn call_bytes(path: &str, body: &serde_json::Value) -> Result<Vec<u8>> {
    let bytes = body.to_string();
    let (code, resp) = call("POST", path, Some("application/json"), bytes.as_bytes())?;
    if !(200..300).contains(&code) { return Err(response_error(code, &resp)); }
    Ok(resp)
}

/// A `multipart/form-data` POST, for the two file-upload endpoints (transcribe, diarize). Mirrors
/// exactly what `transcriptions()`/`diarizations()` already parse: a `model` field (if present)
/// then a `file` part -- so a request built here and one built by a real OpenAI client differ only
/// in which library assembled the bytes.
pub fn call_multipart(path: &str, model: Option<&str>, filename: &str, file: &[u8])
    -> Result<serde_json::Value> {
    const BOUNDARY: &str = "npu-cli-boundary";
    let mut body = Vec::new();
    if let Some(m) = model {
        body.extend_from_slice(
            format!("--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n{m}\r\n")
                .as_bytes());
    }
    body.extend_from_slice(format!(
        "--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n\
         Content-Type: application/octet-stream\r\n\r\n").as_bytes());
    body.extend_from_slice(file);
    body.extend_from_slice(format!("\r\n--{BOUNDARY}--\r\n").as_bytes());
    let content_type = format!("multipart/form-data; boundary={BOUNDARY}");
    let (code, resp) = call("POST", path, Some(&content_type), &body)?;
    if !(200..300).contains(&code) { return Err(response_error(code, &resp)); }
    serde_json::from_slice(&resp).context("control socket: response was not JSON")
}

/// A streamed `POST`: the connection to a live SSE body, read one `data:` frame at a time. Generic
/// over the reader so a test can drive it from an in-memory buffer instead of a real socket --
/// `open` below is the only place that ties it to a `UnixStream`.
pub struct SseCall<R> { reader: R }

impl SseCall<BufReader<UnixStream>> {
    pub fn open(path: &str, body: &serde_json::Value) -> Result<Self> {
        let mut stream = connect().ok_or_else(refusal)?;
        let bytes = body.to_string();
        write_request(&mut stream, "POST", path, Some("application/json"), bytes.as_bytes())
            .map_err(|_| refusal())?;
        let mut reader = BufReader::new(stream);
        let code = read_status_line(&mut reader).map_err(|_| refusal())?;
        let mut content_len = 0usize;
        loop {
            let mut h = String::new();
            if reader.read_line(&mut h)? == 0 { break; }
            let h = h.trim_end();
            if h.is_empty() { break; }
            if let Some(v) = h.to_ascii_lowercase().strip_prefix("content-length:") {
                content_len = v.trim().parse().unwrap_or(0);
            }
        }
        if code != 200 {
            // An error response is a normal Content-Length body, not a stream -- read it whole so
            // the caller's message names the real cause instead of "unexpected EOF".
            let mut resp = vec![0u8; content_len];
            reader.read_exact(&mut resp)?;
            return Err(response_error(code, &resp));
        }
        Ok(SseCall { reader })
    }
}

impl<R: BufRead> SseCall<R> {
    /// A stream already positioned at its first `data:` frame -- what a test builds from a
    /// scripted `&[u8]` body, with none of `open`'s socket/status-line machinery.
    #[cfg(test)]
    pub fn from_reader(reader: R) -> Self { SseCall { reader } }

    /// The next frame's JSON payload, or `None` once the stream ends (`[DONE]` or the connection
    /// closing). `Some(Err)` for a line that claims to be a frame but is not valid JSON -- a real
    /// protocol violation, not an end-of-stream signal, so it must not be swallowed as one.
    pub fn next_frame(&mut self) -> Option<Result<serde_json::Value>> {
        loop {
            let mut line = String::new();
            match self.reader.read_line(&mut line) {
                Ok(0) => return None,
                Ok(_) => {}
                Err(e) => return Some(Err(e.into())),
            }
            let line = line.trim_end();
            if line.is_empty() { continue; }
            let Some(payload) = line.strip_prefix("data: ") else { continue };
            if payload == "[DONE]" { return None; }
            return Some(serde_json::from_str(payload).map_err(|e| anyhow::anyhow!("{e}: {payload}")));
        }
    }
}

/// `Prompt::Chat`'s wire shape: `messages`, sent the same way an OpenAI client would build it. The
/// CLI never sends `tool_calls`/`tool_call_id` on an outgoing turn -- those only ever come BACK from
/// the model -- so a message is exactly its role and content.
pub fn chat_messages_json(history: &[npu_engine::ChatMessage]) -> serde_json::Value {
    serde_json::json!(history.iter()
        .map(|m| serde_json::json!({"role": m.role, "content": m.content}))
        .collect::<Vec<_>>())
}

/// `GenerateParams` onto the wire, OMITTING every field the caller did not set -- `None` must not
/// become `null` on the wire, or a server-side default can never apply. `stream`/`stats` are always
/// carried: the CLI always wants the stats-bearing frames (`x_npu_report`) for its own footer,
/// whatever its OWN `--stats` display flag says, and streams by default the same way `npu generate`
/// always has.
pub fn generate_request_json(base: serde_json::Value, model: Option<&str>,
                              params: &npu_engine::GenerateParams, stream: bool) -> serde_json::Value {
    let mut v = base;
    let obj = v.as_object_mut().expect("base is always a JSON object");
    if let Some(m) = model { obj.insert("model".into(), serde_json::json!(m)); }
    if let Some(x) = params.temperature { obj.insert("temperature".into(), serde_json::json!(x)); }
    if let Some(x) = params.top_p { obj.insert("top_p".into(), serde_json::json!(x)); }
    if let Some(x) = params.top_k { obj.insert("top_k".into(), serde_json::json!(x)); }
    if let Some(x) = params.max_tokens { obj.insert("max_tokens".into(), serde_json::json!(x)); }
    if !params.stop.is_empty() { obj.insert("stop".into(), serde_json::json!(params.stop)); }
    if let Some(x) = params.seed { obj.insert("seed".into(), serde_json::json!(x)); }
    if let Some(x) = params.enable_thinking {
        obj.insert("chat_template_kwargs".into(), serde_json::json!({"enable_thinking": x}));
    }
    if let Some(x) = params.presence_penalty { obj.insert("presence_penalty".into(), serde_json::json!(x)); }
    if let Some(x) = params.frequency_penalty { obj.insert("frequency_penalty".into(), serde_json::json!(x)); }
    if let Some(x) = params.repetition_penalty { obj.insert("repetition_penalty".into(), serde_json::json!(x)); }
    if let Some(x) = params.dispatch_log { obj.insert("x_npu_dispatch_log".into(), serde_json::json!(x)); }
    obj.insert("stream".into(), serde_json::json!(stream));
    // Only meaningful when streaming: `render_buffered` always carries the full report regardless
    // of this flag, the same way it always carried `timings`/`x_npu` before this task.
    if stream { obj.insert("stream_options".into(), serde_json::json!({"include_stats": true})); }
    v
}
