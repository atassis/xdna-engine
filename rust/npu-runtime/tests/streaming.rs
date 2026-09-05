// Real-socket coverage for the SSE surface: what `http.rs`'s own unit tests cannot exercise without
// a live `TcpStream` -- the `[DONE]` terminator, and a client that actually hangs up mid-stream.
// Run with: cargo test -p npu-runtime --test streaming
use npu_engine::capability::{Capability, Request as EngineReq, Response as EngineResp};
use npu_engine::{Chunk, EngineError, FinishReason, GenerateParams, GenerateUsage, Prompt, TextGenerator};
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
        sink(Chunk::Done { reason: FinishReason::Stop, usage });
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
        models: vec![ModelCfg { name: "llm".into(), scenario: "x".into() }],
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
