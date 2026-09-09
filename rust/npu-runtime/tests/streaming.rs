// Real-socket coverage for the SSE surface: what `http.rs`'s own unit tests cannot exercise without
// a live `TcpStream` -- the `[DONE]` terminator, and a client that actually hangs up mid-stream.
// Run with: cargo test -p npu-runtime --test streaming
use npu_engine::capability::{Capability, Request as EngineReq, Response as EngineResp};
use npu_engine::{Chunk, EngineError, FinishReason, GenerateParams, GenerateUsage, GenerationReport,
                 Prompt, StepRecord, TextGenerator};
use npu_runtime::actor::start;
use npu_runtime::config::{Config, ModelCfg, ServerCfg};
use npu_runtime::loader::{ModelLoader, Servable, StreamServable};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// Emits `tokens` in order, one `Chunk::Text` per call to `sink`, `delay` apart, then `Chunk::Done`.
/// `sent` counts tokens that made it past the sink -- what the disconnect test watches.
struct ScriptedGenerator { tokens: Vec<String>, delay: Duration, sent: Arc<AtomicUsize> }
impl TextGenerator for ScriptedGenerator {
    fn generate(&mut self, _prompt: &Prompt, params: &GenerateParams,
        sink: &mut dyn FnMut(Chunk<'_>) -> bool) -> Result<(), EngineError> {
        let mut usage = GenerateUsage::default();
        let mut report = GenerationReport::default();
        let t0 = std::time::Instant::now();
        let cap = (params.max_tokens.unwrap_or(npu_engine::DEFAULT_MAX_TOKENS) as usize).min(self.tokens.len());
        for (i, tok) in self.tokens.iter().take(cap).enumerate() {
            if !self.delay.is_zero() { std::thread::sleep(self.delay); }
            self.sent.fetch_add(1, Ordering::SeqCst);
            usage.completion_tokens += 1;
            let t_us = t0.elapsed().as_micros() as u64;
            let rec = StepRecord {
                seq: i as u32,
                token: Some(1000 + i as u32),
                text: tok.clone(),
                emit: tok.clone(),
                t_us,
                dt_us: t_us.saturating_sub(report.steps.last().map(|s| s.t_us).unwrap_or(0)),
                ..StepRecord::default()
            };
            let live = sink(Chunk::Text(&rec.emit)) && sink(Chunk::Step(&rec));
            report.steps.push(rec);
            if !live {
                report.usage = usage;
                let _ = sink(Chunk::Done { reason: FinishReason::Aborted, usage, report: &report });
                return Ok(());
            }
        }
        report.usage = usage;
        report.generate_us = t0.elapsed().as_micros() as u64;
        sink(Chunk::Done { reason: FinishReason::Stop, usage, report: &report });
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

struct GenLoader { tokens: Vec<String>, delay: Duration, sent: Arc<AtomicUsize> }
impl ModelLoader for GenLoader {
    fn load(&self, _cfg: &ModelCfg) -> Result<Box<dyn StreamServable>, EngineError> {
        Ok(Box::new(GenModel { gen: ScriptedGenerator {
            tokens: self.tokens.clone(), delay: self.delay, sent: self.sent.clone(),
        }}))
    }
    fn declared_capability(&self, _cfg: &ModelCfg) -> Option<Capability> { Some(Capability::GENERATE) }
}

/// Start an actor over `loader` and serve it on an OS-assigned port. Returns the address to connect
/// to; the server thread is never joined (it accepts forever) -- it dies with the test process.
fn spawn_server(loader: GenLoader) -> (npu_runtime::actor::Handle, std::thread::JoinHandle<()>, std::net::SocketAddr) {
    let dir = tempfile::tempdir().unwrap();
    let cfg_path = dir.path().join("engine.toml");
    let cfg = Config {
        server: ServerCfg { max_resident: 1, idle_unload_s: 0, ..Default::default() },
        models: vec![ModelCfg { name: "llm".into(), scenario: "x".into(), resident: false }],
        ..Default::default()
    };
    cfg.save(&cfg_path).unwrap();
    let (handle, join) = start(cfg, Box::new(loader)).unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let server_handle = handle.clone();
    // `dir` moves in and is held for the thread's lifetime -- neither request in this file hits an
    // admin route that re-reads `engine.toml`, but keeping the directory alive costs nothing and
    // avoids depending on that.
    std::thread::spawn(move || {
        let _dir = dir;
        let _ = npu_runtime::http::serve_on(listener, server_handle, cfg_path);
    });
    (handle, join, addr)
}

fn post_stream(addr: std::net::SocketAddr, path: &str, body: &str) -> TcpStream {
    let mut client = TcpStream::connect(addr).unwrap();
    let req = format!(
        "POST {path} HTTP/1.1\r\nHost: x\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len());
    client.write_all(req.as_bytes()).unwrap();
    client
}

/// Read the status line and drain headers, leaving the reader positioned at the SSE body.
fn read_status_and_headers(reader: &mut BufReader<TcpStream>) -> String {
    let mut status = String::new();
    reader.read_line(&mut status).unwrap();
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).unwrap() == 0 || line == "\r\n" { break; }
    }
    status
}

/// Collect the `data:` payloads of an SSE response, minus the `[DONE]` terminator.
fn sse_frames(addr: std::net::SocketAddr, body: &str) -> Vec<serde_json::Value> {
    let client = post_stream(addr, "/v1/chat/completions", body);
    let mut reader = BufReader::new(client);
    assert!(read_status_and_headers(&mut reader).contains("200"));
    let mut out = Vec::new();
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).unwrap() == 0 { break; }
        let Some(rest) = line.trim_end().strip_prefix("data: ") else { continue };
        if rest == "[DONE]" { break; }
        out.push(serde_json::from_str(rest).unwrap());
    }
    out
}

/// Without the opt-in the stream is what it always was: role chunk, one content chunk per token,
/// a finish chunk, `[DONE]`. Nothing extra on the wire, no unknown `object` for a strict client to
/// trip over. This is the regression that matters most -- telemetry must be invisible by default.
#[test]
fn a_stream_without_the_opt_in_is_unchanged() {
    let tokens: Vec<String> = ["Hello", ", ", "world"].iter().map(|s| s.to_string()).collect();
    let sent = Arc::new(AtomicUsize::new(0));
    let (handle, join, addr) = spawn_server(GenLoader { tokens, delay: Duration::ZERO, sent });
    let frames = sse_frames(addr, r#"{"messages":[{"role":"user","content":"hi"}],"stream":true}"#);

    assert_eq!(frames.len(), 5, "role + 3 content + finish");
    assert_eq!(frames[0]["choices"][0]["delta"]["role"], "assistant");
    let text: String = frames[1..4].iter()
        .map(|f| f["choices"][0]["delta"]["content"].as_str().unwrap()).collect();
    assert_eq!(text, "Hello, world");
    assert!(frames.iter().all(|f| f["object"] == "chat.completion.chunk"), "no new object kinds");
    assert!(frames.iter().all(|f| f.get("x_npu").is_none()), "no payload nobody asked for");
    assert_eq!(frames[4]["choices"][0]["finish_reason"], "stop");
    handle.shutdown(); let _ = join.join();
}

/// With the opt-in every content frame carries its own measurement and the stream closes with a
/// summary. The frames stay valid OpenAI chunks -- that is the whole premise of the format, and the
/// text has to come out identical to the run without it.
#[test]
fn an_opted_in_stream_carries_per_token_measurements_and_a_summary() {
    let tokens: Vec<String> = ["Hello", ", ", "world"].iter().map(|s| s.to_string()).collect();
    let sent = Arc::new(AtomicUsize::new(0));
    let (handle, join, addr) = spawn_server(GenLoader { tokens, delay: Duration::from_millis(3), sent });
    let frames = sse_frames(addr,
        r#"{"messages":[{"role":"user","content":"hi"}],"stream":true,"stream_options":{"include_stats":true}}"#);

    let content: Vec<&serde_json::Value> = frames.iter()
        .filter(|f| f["object"] == "chat.completion.chunk" && f["x_npu"].is_object()).collect();
    assert_eq!(content.len(), 3, "one measured frame per token");
    let text: String = content.iter()
        .map(|f| f["choices"][0]["delta"]["content"].as_str().unwrap()).collect();
    assert_eq!(text, "Hello, world", "same bytes as the stream without stats");
    for (i, f) in content.iter().enumerate() {
        assert_eq!(f["x_npu"]["det"]["seq"], i as u64);
        assert!(f["x_npu"]["time"]["dt_ms"].as_f64().is_some());
        assert!(f["choices"][0]["finish_reason"].is_null(), "still a normal chunk");
    }
    // The 3 ms sleep per token has to show up, or the numbers on the wire are decoration.
    let dt = content[2]["x_npu"]["time"]["dt_ms"].as_f64().unwrap();
    assert!(dt >= 2.0, "measured gap should reflect the scripted delay, got {dt} ms");

    let summary = frames.iter().find(|f| f["object"] == "npu.run.summary").expect("summary frame");
    assert_eq!(summary["usage"]["completion_tokens"], 3);
    assert!(summary["timings"]["predicted_per_second"].as_f64().unwrap() > 0.0);
    assert_eq!(summary["finish_reason"], "stop");
    // The finish chunk still precedes it: a client that stops at finish_reason sees the same end.
    let fin = frames.iter().position(|f| f["choices"][0]["finish_reason"] == "stop").unwrap();
    let sum = frames.iter().position(|f| f["object"] == "npu.run.summary").unwrap();
    assert!(fin < sum);
    handle.shutdown(); let _ = join.join();
}

#[test]
fn a_client_that_hangs_up_mid_stream_aborts_generation() {
    let n = 500;
    let tokens: Vec<String> = (0..n).map(|i| format!("t{i}")).collect();
    let sent = Arc::new(AtomicUsize::new(0));
    let (handle, join, addr) =
        spawn_server(GenLoader { tokens, delay: Duration::from_millis(2), sent: sent.clone() });

    let client = post_stream(addr, "/v1/chat/completions",
        r#"{"messages":[{"role":"user","content":"hi"}],"stream":true}"#);
    let mut reader = BufReader::new(client);
    let status = read_status_and_headers(&mut reader);
    assert!(status.contains("200"), "{status}");

    // Read a couple of real SSE frames off the wire, then hang up for real.
    let mut frames = 0;
    while frames < 2 {
        let mut line = String::new();
        if reader.read_line(&mut line).unwrap() == 0 { break; }
        if line.starts_with("data:") { frames += 1; }
    }
    drop(reader);

    std::thread::sleep(Duration::from_millis(300));
    let got = sent.load(Ordering::SeqCst);
    assert!(got < n, "a hung-up client must abort generation; sent {got} of {n} tokens");

    handle.shutdown();
    let _ = join.join();
}

#[test]
fn the_sse_body_ends_with_the_done_terminator() {
    let tokens = vec!["hello".to_string(), " world".to_string()];
    let sent = Arc::new(AtomicUsize::new(0));
    let (handle, join, addr) = spawn_server(GenLoader { tokens, delay: Duration::ZERO, sent });

    let mut client = post_stream(addr, "/v1/chat/completions",
        r#"{"messages":[{"role":"user","content":"hi"}],"stream":true}"#);
    let mut resp = String::new();
    client.read_to_string(&mut resp).unwrap();

    assert!(resp.contains("Content-Type: text/event-stream"), "{resp}");
    assert!(!resp.contains("Content-Length:"), "an SSE body must not carry Content-Length: {resp}");
    assert!(resp.contains("\"delta\":{\"role\":\"assistant\"}"), "{resp}");
    assert!(resp.contains("\"delta\":{\"content\":\"hello\"}"), "{resp}");
    assert!(resp.contains("\"finish_reason\":\"stop\""), "{resp}");
    assert!(resp.trim_end().ends_with("data: [DONE]"), "{resp}");

    handle.shutdown();
    let _ = join.join();
}
