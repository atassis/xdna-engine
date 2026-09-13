// Integration test for the control socket, gated behind `testkit` like tests/actor.rs.
// Run with: cargo test -p npu-runtime --features testkit
#![cfg(feature = "testkit")]
use npu_engine::capability::{Capability, Request, Response};
use npu_engine::EngineError;
use npu_runtime::actor::start_lazy;
use npu_runtime::config::{Config, Defaults, ModelCfg, ServerCfg};
use npu_runtime::control_socket;
use npu_runtime::loader::{ModelLoader, Servable, StreamServable};
use std::io::{BufRead, BufReader, Read, Write};
use std::os::unix::net::UnixStream;
use std::sync::mpsc;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// A model whose `run()` announces it has started, then blocks until the test releases it --
/// standing in for a genuinely long or wedged device dispatch, which occupies the actor thread for
/// `Cmd::Serve`'s whole duration.
struct BlockingModel { started: mpsc::Sender<()>, unblock: mpsc::Receiver<()> }
impl Servable for BlockingModel {
    fn capabilities(&self) -> Capability { Capability::EMBED }
    fn run(&mut self, _req: Request) -> Result<Response, EngineError> {
        let _ = self.started.send(());
        let _ = self.unblock.recv();
        Ok(Response::Vector(vec![0.0; 8]))
    }
}
impl StreamServable for BlockingModel {}

struct BlockingLoader { parts: Mutex<Option<(mpsc::Sender<()>, mpsc::Receiver<()>)>> }
impl ModelLoader for BlockingLoader {
    fn load(&self, _cfg: &ModelCfg) -> Result<Box<dyn StreamServable>, EngineError> {
        let (started, unblock) = self.parts.lock().unwrap().take().expect("loaded twice");
        Ok(Box::new(BlockingModel { started, unblock }))
    }
    fn declared_capability(&self, _cfg: &ModelCfg) -> Option<Capability> { Some(Capability::EMBED) }
}

/// A single `GET /v1/models` request/response over the socket, parsed the same way the CLI's own
/// client will.
fn get_models(path: &std::path::Path) -> (u16, serde_json::Value) {
    let mut stream = UnixStream::connect(path).expect("connect");
    stream.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
    stream.write_all(b"GET /v1/models HTTP/1.1\r\n\r\n").unwrap();
    let mut reader = BufReader::new(stream);
    let mut status_line = String::new();
    reader.read_line(&mut status_line).unwrap();
    let code: u16 = status_line.split_whitespace().nth(1).unwrap().parse().unwrap();
    let mut content_len = 0usize;
    loop {
        let mut h = String::new();
        if reader.read_line(&mut h).unwrap() == 0 { break; }
        let h = h.trim_end();
        if h.is_empty() { break; }
        if let Some(v) = h.to_ascii_lowercase().strip_prefix("content-length:") {
            content_len = v.trim().parse().unwrap();
        }
    }
    let mut body = vec![0u8; content_len];
    reader.read_exact(&mut body).unwrap();
    (code, serde_json::from_slice(&body).unwrap())
}

/// The property the whole design turns on: a status request must be answered while the actor is
/// stuck inside a long device dispatch, not queued behind it -- a real actor, a real blocked
/// `Cmd::Serve`, a real socket, not an assertion about the code shape.
#[test]
fn status_answers_while_the_actor_is_blocked_in_a_long_dispatch() {
    let (started_tx, started_rx) = mpsc::channel();
    let (unblock_tx, unblock_rx) = mpsc::channel();
    let cfg = Config {
        server: ServerCfg { idle_unload_s: 0, ..Default::default() },
        defaults: Defaults::from_pairs([(Capability::EMBED, "bge".to_string())]),
        models: vec![ModelCfg { name: "bge".into(), scenario: "x".into(), resident: false }],
    };
    let loader = BlockingLoader { parts: Mutex::new(Some((started_tx, unblock_rx))) };
    let (handle, join) = start_lazy(cfg, Box::new(loader)).unwrap();

    let dir = tempfile::tempdir().unwrap();
    let sock_path = dir.path().join("control.sock");
    let listener = control_socket::bind(&sock_path).unwrap();
    let serve_handle = handle.clone();
    let live = handle.live_status();
    let cfg_path = dir.path().join("engine.toml");
    std::thread::spawn(move || control_socket::serve(listener, serve_handle, live, cfg_path));

    // Occupy the actor with a dispatch that will not return until this test says so.
    let embed_handle = handle.clone();
    let embed_thread = std::thread::spawn(move || embed_handle.embed(None, "hi"));
    started_rx.recv_timeout(Duration::from_secs(5)).expect("dispatch never started");

    let t0 = Instant::now();
    let (code, doc) = get_models(&sock_path);
    let elapsed = t0.elapsed();

    unblock_tx.send(()).unwrap();
    assert_eq!(embed_thread.join().unwrap().unwrap().model, "bge");

    assert_eq!(code, 200);
    assert!(doc["models"]["data"].is_array(), "{doc}");
    assert!(elapsed < Duration::from_secs(1),
        "status must answer without waiting on the busy actor, took {elapsed:?}");

    handle.shutdown();
    join.join().unwrap();
}
