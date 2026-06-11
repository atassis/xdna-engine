//! Single-binary GigaAM-v3 ASR HTTP service (Task 2, option B — pure Rust, no Python).
//! Runs the official ONNX graphs (mel preprocessor + RNNT decoder/joint) via the onnxruntime
//! C-shim (`npu-onnx`), and our encoder on the NPU (`npu-asr`). FLM-compatible:
//!   POST /v1/audio/transcriptions  (multipart `file` = WAV)  ->  {"text": ...}
//!
//! NPU is single-tenant — flm-asr.service must be stopped. Run from the repo root.
//!   asr_serve [port]        (default 11434)

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::rc::Rc;

use ndarray::prelude::*;
use npu_asr::encoder::{subsample, Encoder};
use npu_asr::weights::WeightStore;
use npu_onnx::{Env, Session, Tensor};
use npu_xrt::Device;

const MEL: usize = 64;
const WIN: usize = 1600;
const D: usize = 768;
const PRED: usize = 320;
const BLANK: i64 = 33;
const MAX_TOK: usize = 3;

struct Pipeline {
    prep: Session,
    decoder: Session,
    joint: Session,
    ws: WeightStore,
    enc: Encoder,
    vocab: HashMap<i64, String>,
}

impl Pipeline {
    fn new(root: &Path) -> Pipeline {
        let env = Env::new().expect("onnx env");
        let asr = root.join("artifacts/asr");
        let load = |f: &str| {
            Session::load(&env, asr.join(f).to_str().unwrap())
                .unwrap_or_else(|e| panic!("load {f}: {e}"))
        };
        let prep = load("preprocessor.onnx");
        let decoder = load("decoder.onnx");
        let joint = load("joint.onnx");
        let ws = WeightStore::load(&root.join("artifacts/encoder")).expect("encoder weights");
        let dev = Rc::new(Device::open(0).expect("open NPU (stop flm-asr first)"));
        let enc = Encoder::new(dev, root, &ws, 16);
        let vocab = load_vocab(&asr.join("vocab.txt"));
        eprintln!("[asr_serve] pipeline ready (onnx preproc/decode + 16-block NPU encoder)");
        Pipeline { prep, decoder, joint, ws, enc, vocab }
    }

    fn transcribe(&self, samples: &[i16]) -> String {
        let wav: Vec<f32> = samples.iter().map(|&s| s as f32 / 32768.0).collect();
        let n = wav.len() as i64;
        let lens = [n];
        let feat = self
            .prep
            .run(
                &[
                    ("waveforms", Tensor::F32(&wav, vec![1, n])),
                    ("waveforms_lens", Tensor::I64(&lens, vec![1])),
                ],
                &["features", "features_lens"],
            )
            .expect("preprocessor");
        let t = feat.shape(0)[2] as usize; // [1,64,T]
        let feats = feat.f32(0); // [64*T] channel-major
        let teff = t.min(WIN);
        let mut audio = Array2::<f32>::zeros((MEL, WIN));
        for c in 0..MEL {
            for ti in 0..teff {
                audio[[c, ti]] = feats[c * t + ti];
            }
        }
        let valid = (teff.max(1) - 1) / 4 + 1;
        let x0 = subsample(&self.ws, &audio);
        let outs = self.enc.forward_blocks(&x0, valid);
        let encoded = outs.last().unwrap(); // [400,768] frame-major
        let ids = self.decode(encoded, valid);
        self.detokenize(&ids)
    }

    // greedy RNNT decode (mirrors onnx-asr _AsrWithTransducerDecoding)
    fn decode(&self, encoded: &Array2<f32>, valid: usize) -> Vec<i64> {
        let mut h = vec![0f32; PRED];
        let mut c = vec![0f32; PRED];
        let mut dec = self.run_decoder(BLANK, &mut h, &mut c);
        let mut tokens: Vec<i64> = Vec::new();
        let (mut t, mut emitted) = (0usize, 0usize);
        while t < valid {
            let frame = encoded.row(t).to_vec(); // [768]
            let logits = self.run_joint(&frame, &dec); // [34]
            let tok = argmax(&logits);
            if tok != BLANK {
                tokens.push(tok);
                emitted += 1;
                dec = self.run_decoder(tok, &mut h, &mut c);
                if emitted == MAX_TOK {
                    t += 1;
                    emitted = 0;
                }
            } else {
                t += 1;
                emitted = 0;
            }
        }
        tokens
    }

    fn run_decoder(&self, x: i64, h: &mut Vec<f32>, c: &mut Vec<f32>) -> Vec<f32> {
        let xv = [x];
        let (dec, nh, nc) = {
            let out = self
                .decoder
                .run(
                    &[
                        ("x", Tensor::I64(&xv, vec![1, 1])),
                        ("h.1", Tensor::F32(h, vec![1, 1, PRED as i64])),
                        ("c.1", Tensor::F32(c, vec![1, 1, PRED as i64])),
                    ],
                    &["dec", "h", "c"],
                )
                .expect("decoder");
            (out.f32(0).to_vec(), out.f32(1).to_vec(), out.f32(2).to_vec())
        };
        *h = nh;
        *c = nc;
        dec
    }

    fn run_joint(&self, enc: &[f32], dec: &[f32]) -> Vec<f32> {
        let out = self
            .joint
            .run(
                &[
                    ("enc", Tensor::F32(enc, vec![1, D as i64, 1])),
                    ("dec", Tensor::F32(dec, vec![1, PRED as i64, 1])),
                ],
                &["joint"],
            )
            .expect("joint");
        out.f32(0).to_vec()
    }

    fn detokenize(&self, ids: &[i64]) -> String {
        let s: String = ids
            .iter()
            .map(|id| self.vocab.get(id).map(|x| x.as_str()).unwrap_or(""))
            .collect();
        s.trim().to_string()
    }
}

fn argmax(v: &[f32]) -> i64 {
    let mut best = 0usize;
    for i in 1..v.len() {
        if v[i] > v[best] {
            best = i;
        }
    }
    best as i64
}

fn load_vocab(path: &Path) -> HashMap<i64, String> {
    let txt = std::fs::read_to_string(path).expect("vocab");
    let mut m = HashMap::new();
    for line in txt.lines() {
        // "<token> <id>"  ; token "▁" -> space
        if let Some((tok, id)) = line.rsplit_once(' ') {
            if let Ok(id) = id.trim().parse::<i64>() {
                m.insert(id, tok.replace('\u{2581}', " "));
            }
        }
    }
    m
}

// ---------------- HTTP (std only; single-threaded — NPU is serial) ----------------

fn main() {
    let port: u16 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(11434);
    let pipe = Pipeline::new(Path::new("."));
    let addr = format!("127.0.0.1:{port}");
    let listener = TcpListener::bind(&addr).unwrap_or_else(|e| panic!("bind {addr}: {e}"));
    eprintln!("[asr_serve] listening on http://{addr}/v1/audio/transcriptions");
    for stream in listener.incoming() {
        match stream {
            Ok(s) => {
                if let Err(e) = handle(s, &pipe) {
                    eprintln!("[asr_serve] request error: {e}");
                }
            }
            Err(e) => eprintln!("[asr_serve] accept error: {e}"),
        }
    }
}

fn handle(mut stream: TcpStream, pipe: &Pipeline) -> std::io::Result<()> {
    let peer = stream.peer_addr().map(|a| a.to_string()).unwrap_or_default();
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut line = String::new();
    reader.read_line(&mut line)?;
    let request_line = line.trim_end().to_string();
    let mut content_len = 0usize;
    let mut boundary = String::new();
    loop {
        let mut h = String::new();
        if reader.read_line(&mut h)? == 0 {
            break;
        }
        let h = h.trim_end();
        if h.is_empty() {
            break;
        }
        let lower = h.to_ascii_lowercase();
        if let Some(v) = lower.strip_prefix("content-length:") {
            content_len = v.trim().parse().unwrap_or(0);
        } else if lower.starts_with("content-type:") {
            if let Some(idx) = lower.find("boundary=") {
                boundary = h[idx + "boundary=".len()..].trim().trim_matches('"').to_string();
            }
        }
    }
    if !request_line.contains("/v1/audio/transcriptions") {
        return respond(&mut stream, 404, "{\"error\":\"not found\"}");
    }
    let mut body = vec![0u8; content_len];
    reader.read_exact(&mut body)?;
    let wav = match extract_file_part(&body, &boundary) {
        Some(w) => w,
        None => return respond(&mut stream, 400, "{\"error\":\"no file part\"}"),
    };
    let samples = match parse_wav_i16(wav) {
        Some(s) => s,
        None => return respond(&mut stream, 400, "{\"error\":\"bad wav\"}"),
    };
    let text = pipe.transcribe(&samples);
    eprintln!("[asr_serve] {peer} -> {} samples -> {:?}", samples.len(), text);
    let body = format!("{{\"text\": \"{}\"}}", json_escape(&text));
    respond(&mut stream, 200, &body)
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

/// Parse a 16-bit PCM WAV: find the `data` chunk and read it as little-endian i16.
fn parse_wav_i16(wav: &[u8]) -> Option<Vec<i16>> {
    let pos = find(wav, b"data")?;
    let size_off = pos + 4;
    if size_off + 4 > wav.len() {
        return None;
    }
    let size = u32::from_le_bytes([wav[size_off], wav[size_off + 1], wav[size_off + 2], wav[size_off + 3]]) as usize;
    let data = &wav[size_off + 4..];
    let n = size.min(data.len()) / 2;
    Some((0..n).map(|i| i16::from_le_bytes([data[i * 2], data[i * 2 + 1]])).collect())
}
