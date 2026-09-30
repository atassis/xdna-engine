//! S2 head-isolation diagnostic: forward ONE real token through the 48-layer stack (already
//! S1-proven correct on this build), capture its pre-norm hidden row, then run BOTH the on-device
//! head (`RawResidentForward::debug_raw_head_from_hidden`) and the bridge's host float64 oracle on
//! the IDENTICAL bytes, and diff their top tokens. Isolates the head from the layer stack instead
//! of re-running a full generation to guess at the fault.
//!
//! Requires RF_BRIDGE_PY/RF_BRIDGE_PYTHON/RF_BRIDGE_PYTHONPATH (see resident_raw.rs's module doc)
//! and RF_BUILD_DIR (default /mnt/data/xdna/scratch/rf/build/rls1).

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};

use npu_xrt::{unpack_bf16_to_f32, Device};
use serde_json::{json, Value};

fn env_required(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("{name} must be set"))
}

struct Bridge {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<std::process::ChildStdout>,
}

impl Bridge {
    fn spawn() -> Self {
        let bridge_py = env_required("RF_BRIDGE_PY");
        let cwd = std::path::Path::new(&bridge_py).parent().expect("RF_BRIDGE_PY has no parent").to_path_buf();
        let mut child = Command::new(env_required("RF_BRIDGE_PYTHON"))
            .arg(&bridge_py)
            .current_dir(cwd)
            .env("PYTHONPATH", env_required("RF_BRIDGE_PYTHONPATH"))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .expect("spawn bridge");
        let stdin = child.stdin.take().unwrap();
        let stdout = BufReader::new(child.stdout.take().unwrap());
        Bridge { child, stdin, stdout }
    }

    fn call(&mut self, req: Value) -> Value {
        self.stdin.write_all(serde_json::to_string(&req).unwrap().as_bytes()).unwrap();
        self.stdin.write_all(b"\n").unwrap();
        self.stdin.flush().unwrap();
        let mut line = String::new();
        self.stdout.read_line(&mut line).unwrap();
        let resp: Value = serde_json::from_str(&line).unwrap_or_else(|e| panic!("parse {line:?}: {e}"));
        assert!(resp["ok"].as_bool().unwrap_or(false), "bridge error: {resp}");
        resp
    }
}

impl Drop for Bridge {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn build_dir() -> String {
    std::env::var("RF_BUILD_DIR").unwrap_or_else(|_| "/mnt/data/xdna/scratch/rf/build/rls1".to_string())
}

fn main() {
    // The S1-gate prompt's own first token (<bos>) at position 0 -- a real, previously-validated
    // layer-stack input, so any disagreement below is localised to norm+head, not the stack.
    let token: u32 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(2);
    let pos: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(0);

    let mut bridge = Bridge::spawn();

    let dev = std::rc::Rc::new(Device::open(0).expect("open NPU device 0"));
    let mut model = npu_models::llm::RawResidentForward::open(&dev, std::path::Path::new(&build_dir())).expect("open resident forward");

    eprintln!("forwarding token={token} pos={pos} through the 48-layer stack...");
    let hidden_bits = model.debug_forward_hidden(token, pos).expect("forward");
    let mut hidden_f32 = vec![0f32; hidden_bits.len()];
    unpack_bf16_to_f32(&hidden_bits, &mut hidden_f32);
    let (mean, min, max) = (
        hidden_f32.iter().sum::<f32>() / hidden_f32.len() as f32,
        hidden_f32.iter().cloned().fold(f32::INFINITY, f32::min),
        hidden_f32.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
    );
    println!("hidden row: len={} mean={mean:.4} min={min:.4} max={max:.4}", hidden_f32.len());
    println!("hidden row finite: {}", hidden_f32.iter().all(|v| v.is_finite()));

    let normed = model.debug_host_rms_norm(&hidden_f32);
    let (nmean, nmin, nmax) = (
        normed.iter().sum::<f32>() / normed.len() as f32,
        normed.iter().cloned().fold(f32::INFINITY, f32::min),
        normed.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
    );
    println!("host-side normed row: mean={nmean:.4} min={nmin:.4} max={nmax:.4}");

    println!("dispatching the on-device head (h1)...");
    let device_raw = model.debug_raw_head_from_hidden(&hidden_bits).expect("device head");
    let mut device_top: Vec<(usize, f32)> = device_raw.iter().cloned().enumerate().collect();
    device_top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    device_top.truncate(10);
    println!("DEVICE top-10 raw logits: {device_top:?}");

    let bridge_resp = bridge.call(json!({"cmd": "head_debug", "h": hidden_f32}));
    println!("HOST xn_stats: {}", bridge_resp["xn_stats"]);
    println!("HOST top-10 raw logits: {}", bridge_resp["top_raw"]);
    println!("HOST top-10 softcapped: {}", bridge_resp["top_capped"]);

    let device_argmax = device_top[0].0;
    let host_argmax = bridge_resp["top_raw"][0][0].as_u64().unwrap() as usize;
    println!("device argmax={device_argmax} host argmax={host_argmax} match={}", device_argmax == host_argmax);
}
