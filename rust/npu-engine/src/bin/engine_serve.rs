//! General engine HTTP server. Usage: engine_serve [scenario.toml] [port]
//!   embeddings scenario -> POST /v1/embeddings  {"input": "..."|[...]}  (OpenAI-compatible)
//!   asr scenario        -> POST /v1/audio/transcriptions (Task 12)
//! Single-threaded: the NPU is single-tenant. Run from repo root with flm-asr/voxd stopped.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::time::Duration;

use npu_engine::pipeline::Scenario;
use npu_engine::registry;

const MAX_BODY: usize = 16 * 1024 * 1024;
const SOCKET_TIMEOUT: Duration = Duration::from_secs(30);

fn main() {
    let scenario = std::env::args().nth(1).unwrap_or_else(|| "scenarios/bge-base.toml".into());
    let port: u16 = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(11435);
    let scen = registry::build(Path::new(&scenario), Path::new("."));
    let addr = format!("127.0.0.1:{port}");
    let listener = TcpListener::bind(&addr).unwrap_or_else(|e| panic!("bind {addr}: {e}"));
    eprintln!("[engine_serve] {scenario} ready on http://{addr}");
    for stream in listener.incoming() {
        match stream {
            Ok(s) => { if let Err(e) = handle(s, &scen) { eprintln!("[engine_serve] {e}"); } }
            Err(e) => eprintln!("[engine_serve] accept: {e}"),
        }
    }
}

fn handle(mut stream: TcpStream, scen: &Scenario) -> std::io::Result<()> {
    let _ = stream.set_read_timeout(Some(SOCKET_TIMEOUT));
    let _ = stream.set_write_timeout(Some(SOCKET_TIMEOUT));
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut line = String::new();
    reader.read_line(&mut line)?;
    let request_line = line.trim_end().to_string();
    let mut content_len = 0usize;
    let mut boundary = String::new();
    loop {
        let mut h = String::new();
        if reader.read_line(&mut h)? == 0 { break; }
        let h = h.trim_end();
        if h.is_empty() { break; }
        if let Some(v) = h.to_ascii_lowercase().strip_prefix("content-length:") {
            content_len = v.trim().parse().unwrap_or(0);
        } else if let Some(lower) = {
            let l = h.to_ascii_lowercase();
            if l.starts_with("content-type:") { Some(l) } else { None }
        } {
            if let Some(idx) = lower.find("boundary=") {
                boundary = h[idx + "boundary=".len()..].trim().trim_matches('"').to_string();
            }
        }
    }
    if request_line.starts_with("GET ") && request_line.contains("/health") {
        return respond(&mut stream, 200, "{\"status\":\"ok\"}");
    }
    if content_len > MAX_BODY { return respond(&mut stream, 413, "{\"error\":\"too large\"}"); }
    let mut body_bytes = vec![0u8; content_len];
    reader.read_exact(&mut body_bytes)?;

    match scen {
        Scenario::Embed(pipe) => {
            if !request_line.contains("/v1/embeddings") {
                return respond(&mut stream, 404, "{\"error\":\"not found\"}");
            }
            let body = String::from_utf8_lossy(&body_bytes).to_string();
            let inputs = parse_inputs(&body);
            let mut data = String::new();
            for (i, text) in inputs.iter().enumerate() {
                let v = pipe.embed(text.clone());
                let arr = v.iter().map(|x| format!("{x}")).collect::<Vec<_>>().join(",");
                if i > 0 { data.push(','); }
                data.push_str(&format!("{{\"object\":\"embedding\",\"index\":{i},\"embedding\":[{arr}]}}"));
            }
            let out = format!("{{\"object\":\"list\",\"data\":[{data}],\"model\":\"engine\"}}");
            respond(&mut stream, 200, &out)
        }
        Scenario::Asr(pipe) => {
            if !request_line.contains("/v1/audio/transcriptions") {
                return respond(&mut stream, 404, "{\"error\":\"not found\"}");
            }
            let wav = match extract_file_part(&body_bytes, &boundary) {
                Some(w) => w, None => return respond(&mut stream, 400, "{\"error\":\"no file part\"}"),
            };
            let samples = match parse_wav_i16(wav) {
                Some(s) if !s.is_empty() => s,
                _ => return respond(&mut stream, 400, "{\"error\":\"bad wav\"}"),
            };
            let text = pipe.transcribe(&samples);
            respond(&mut stream, 200, &format!("{{\"text\": \"{}\"}}", json_escape(&text)))
        }
    }
}

/// Minimal extraction of the OpenAI `input` field: a JSON string or array of strings. Falls back to
/// treating the whole body as one plain-text input if it isn't JSON-with-input.
fn parse_inputs(body: &str) -> Vec<String> {
    if let Some(idx) = body.find("\"input\"") {
        let rest = &body[idx + 7..];
        if let Some(open) = rest.find('[') {
            // array of strings
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

fn respond(stream: &mut TcpStream, code: u16, body: &str) -> std::io::Result<()> {
    let reason = if code == 200 { "OK" } else { "Error" };
    let resp = format!(
        "HTTP/1.1 {code} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.as_bytes().len()
    );
    stream.write_all(resp.as_bytes())?;
    stream.flush()
}

fn json_escape(s: &str) -> String {
    let mut o = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o
}

/// Find the multipart part named "file" and return its raw payload bytes (the WAV).
fn extract_file_part<'a>(body: &'a [u8], boundary: &str) -> Option<&'a [u8]> {
    if boundary.is_empty() {
        return None;
    }
    let delim = format!("--{boundary}");
    for part in split_on(body, delim.as_bytes()) {
        let hdr_end = match find(part, b"\r\n\r\n") {
            Some(h) => h,
            None => continue,
        };
        let headers = String::from_utf8_lossy(&part[..hdr_end]).to_ascii_lowercase();
        if headers.contains("name=\"file\"") {
            let mut data = &part[hdr_end + 4..];
            if data.ends_with(b"\r\n") {
                data = &data[..data.len() - 2];
            }
            return Some(data);
        }
    }
    None
}

fn split_on<'a>(hay: &'a [u8], sep: &[u8]) -> Vec<&'a [u8]> {
    let mut out = Vec::new();
    let (mut start, mut i) = (0usize, 0usize);
    while i + sep.len() <= hay.len() {
        if &hay[i..i + sep.len()] == sep {
            out.push(&hay[start..i]);
            i += sep.len();
            start = i;
        } else {
            i += 1;
        }
    }
    out.push(&hay[start..]);
    out
}

fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || hay.len() < needle.len() {
        return None;
    }
    (0..=hay.len() - needle.len()).find(|&i| &hay[i..i + needle.len()] == needle)
}

/// Parse a 16 kHz / mono / 16-bit PCM WAV into little-endian i16 samples.
///
/// Walks the RIFF chunk list properly (id[4] + LE u32 size + word-aligned body) rather than
/// scanning for the bytes `data` — a `LIST`/`INFO` chunk containing the substring "data" before
/// the real `data` chunk would mis-parse the old way. Validates `fmt ` (PCM/extensible, 16-bit,
/// mono, 16 kHz) so a stereo / 24-bit / wrong-rate file is a clean 400, not silent garbage; the
/// mel front-end assumes exactly this format.
fn parse_wav_i16(wav: &[u8]) -> Option<Vec<i16>> {
    if wav.len() < 12 || &wav[0..4] != b"RIFF" || &wav[8..12] != b"WAVE" {
        return None;
    }
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
                // 1 = PCM, 0xFFFE = WAVE_FORMAT_EXTENSIBLE (still LPCM here)
                fmt_ok = (audio_fmt == 1 || audio_fmt == 0xFFFE)
                    && bits == 16
                    && channels == 1
                    && rate == 16_000;
            }
            b"data" => data = Some(&wav[body_start..body_end]),
            _ => {}
        }
        // chunk bodies are word-aligned: an odd size is padded with one byte.
        off = body_start.saturating_add(sz).saturating_add(sz & 1);
    }
    if !fmt_ok {
        return None;
    }
    let data = data?;
    let n = data.len() / 2;
    Some((0..n).map(|i| i16::from_le_bytes([data[i * 2], data[i * 2 + 1]])).collect())
}
