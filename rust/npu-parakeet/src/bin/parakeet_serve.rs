//! Single-binary Parakeet-tdt-0.6b-v3 (multilingual RU+EN) ASR HTTP service.
//! NeMo 128-mel preprocessor + TDT decoder_joint via the onnxruntime C-shim (`npu-onnx`), and our
//! FastConformer encoder on the XDNA2 NPU (`npu-parakeet`). FLM/GigaAM-compatible API:
//!   POST /v1/audio/transcriptions  (multipart `file` = WAV 16k/mono/16-bit)  ->  {"text": ...}
//!
//! NPU is single-tenant — stop the other ASR service first. Run from the repo root.
//!   parakeet_serve [port]        (default 11434)
//!
//! Artifacts (artifacts/parakeet/): preprocessor.onnx (nemo128), decoder_joint.onnx, vocab.txt,
//! encoder/ (extracted weights). NPU xclbins live in mlir-aie/.../whole_array/build (NPU_XCLBIN_ROOT).

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;
use std::time::{Duration, Instant};

use ndarray::prelude::*;
use npu_onnx::{Env, Session, Tensor};
use npu_parakeet::config::ModelCfg;
use npu_parakeet::encoder::FastConformerEncoder;

const MAX_BODY: usize = 64 * 1024 * 1024;
const SOCKET_TIMEOUT: Duration = Duration::from_secs(60);

const MEL: usize = 128;
const D: usize = 1024;
const VOCAB: usize = 8193; // token logits incl. <blk>
const BLANK: i64 = 8192;
const N_DUR: usize = 5; // TDT duration buckets [0,1,2,3,4]
const STATE_DIM: usize = 640;
const STATE_LAYERS: usize = 2; // input_states_* = [2,1,640]
const MAX_TOK: usize = 10; // max_tokens_per_step (config default)
// rel-pos linear_pos needs M = 2T'-1 <= 512 (static window) -> T' <= 256 -> cap mel frames.
const WIN_MEL: usize = 2040;

struct Pipeline {
    prep: Session,
    dj: Session, // decoder_joint
    enc: FastConformerEncoder,
    vocab: HashMap<i64, String>,
}

impl Pipeline {
    fn new(root: &Path) -> Pipeline {
        let env = Env::new().expect("onnx env");
        let pk = root.join("artifacts/parakeet");
        let load = |f: &str| {
            Session::load(&env, pk.join(f).to_str().unwrap()).unwrap_or_else(|e| panic!("load {f}: {e}"))
        };
        let prep = load("preprocessor.onnx");
        let dj = load("decoder_joint.onnx");
        let xroot = std::env::var("NPU_XCLBIN_ROOT").map(std::path::PathBuf::from).unwrap_or_else(|_| root.to_path_buf());
        let enc = FastConformerEncoder::new_npu(&pk.join("encoder"), ModelCfg::PARAKEET_V3, &xroot);
        let vocab = load_vocab(&pk.join("vocab.txt"));
        eprintln!("[parakeet_serve] pipeline ready (nemo128 preproc + 24-block NPU encoder + TDT decode)");
        Pipeline { prep, dj, enc, vocab }
    }

    fn transcribe(&self, samples: &[i16]) -> String {
        let t_mel = Instant::now();
        let wav: Vec<f32> = samples.iter().map(|&s| s as f32 / 32768.0).collect();
        let n = wav.len() as i64;
        let lens = [n];
        let feat = self
            .prep
            .run(
                &[("waveforms", Tensor::F32(&wav, vec![1, n])), ("waveforms_lens", Tensor::I64(&lens, vec![1]))],
                &["features", "features_lens"],
            )
            .expect("preprocessor");
        let t = feat.shape(0)[2] as usize; // [1,128,T]
        let feats = feat.f32(0); // [128*T] channel-major
        let teff = t.min(WIN_MEL);
        let mut mel = Array2::<f32>::zeros((MEL, teff));
        for c in 0..MEL {
            for ti in 0..teff {
                mel[[c, ti]] = feats[c * t + ti];
            }
        }
        let mel_ms = t_mel.elapsed().as_secs_f64() * 1e3;
        let t_enc = Instant::now();
        let encoded = self.enc.encode(&mel); // [T', 1024] on the NPU
        let valid = encoded.nrows();
        let enc_ms = t_enc.elapsed().as_secs_f64() * 1e3;
        let t_dec = Instant::now();
        let ids = self.tdt_decode(&encoded, valid);
        let text = self.detokenize(&ids);
        eprintln!(
            "[timing] mel {mel_ms:.0}ms | encoder {enc_ms:.0}ms | decode {:.0}ms (T'={valid}, {} tok)",
            t_dec.elapsed().as_secs_f64() * 1e3, ids.len()
        );
        text
    }

    /// TDT duration-split greedy decode (mirrors onnx-asr _AsrWithTransducerDecoding + NemoConformerTdt).
    fn tdt_decode(&self, encoded: &Array2<f32>, valid: usize) -> Vec<i64> {
        let mut st1 = vec![0f32; STATE_LAYERS * STATE_DIM];
        let mut st2 = vec![0f32; STATE_LAYERS * STATE_DIM];
        let mut tokens: Vec<i64> = Vec::new();
        let (mut t, mut emitted) = (0usize, 0usize);
        while t < valid {
            let frame = encoded.row(t).to_vec(); // [1024]
            let last = *tokens.last().unwrap_or(&BLANK) as i32;
            let (out, nst1, nst2) = self.run_dj(&frame, last, &st1, &st2);
            let token = argmax(&out[..VOCAB]); // 8193 token logits
            let step = argmax(&out[VOCAB..VOCAB + N_DUR]) as usize; // duration 0..4
            if token != BLANK {
                st1 = nst1; // commit predictor state on emission
                st2 = nst2;
                tokens.push(token);
                emitted += 1;
            }
            if step > 0 {
                t += step;
                emitted = 0;
            } else if token == BLANK || emitted == MAX_TOK {
                t += 1;
                emitted = 0;
            }
        }
        tokens
    }

    fn run_dj(&self, frame: &[f32], last_tok: i32, st1: &[f32], st2: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let targets = [last_tok];
        let tlen = [1i32];
        let sd = vec![STATE_LAYERS as i64, 1, STATE_DIM as i64];
        let out = self
            .dj
            .run(
                &[
                    ("encoder_outputs", Tensor::F32(frame, vec![1, D as i64, 1])),
                    ("targets", Tensor::I32(&targets, vec![1, 1])),
                    ("target_length", Tensor::I32(&tlen, vec![1])),
                    ("input_states_1", Tensor::F32(st1, sd.clone())),
                    ("input_states_2", Tensor::F32(st2, sd)),
                ],
                &["outputs", "output_states_1", "output_states_2"],
            )
            .expect("decoder_joint");
        (out.f32(0).to_vec(), out.f32(1).to_vec(), out.f32(2).to_vec())
    }

    fn detokenize(&self, ids: &[i64]) -> String {
        let s: String = ids.iter().map(|id| self.vocab.get(id).map(|x| x.as_str()).unwrap_or("")).collect();
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
    let port: u16 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(11434);
    let pipe = Pipeline::new(Path::new("."));
    let addr = format!("127.0.0.1:{port}");
    let listener = TcpListener::bind(&addr).unwrap_or_else(|e| panic!("bind {addr}: {e}"));
    eprintln!("[parakeet_serve] listening on http://{addr}/v1/audio/transcriptions");
    for stream in listener.incoming() {
        match stream {
            Ok(s) => {
                if let Err(e) = handle(s, &pipe) {
                    eprintln!("[parakeet_serve] request error: {e}");
                }
            }
            Err(e) => eprintln!("[parakeet_serve] accept error: {e}"),
        }
    }
}

fn handle(mut stream: TcpStream, pipe: &Pipeline) -> std::io::Result<()> {
    let _ = stream.set_read_timeout(Some(SOCKET_TIMEOUT));
    let _ = stream.set_write_timeout(Some(SOCKET_TIMEOUT));
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
    if request_line.starts_with("GET ") && request_line.contains("/health") {
        return respond(&mut stream, 200, "{\"status\":\"ok\"}");
    }
    if !request_line.contains("/v1/audio/transcriptions") {
        return respond(&mut stream, 404, "{\"error\":\"not found\"}");
    }
    if content_len > MAX_BODY {
        return respond(&mut stream, 413, "{\"error\":\"request too large\"}");
    }
    let mut body = vec![0u8; content_len];
    reader.read_exact(&mut body)?;
    let wav = match extract_file_part(&body, &boundary) {
        Some(w) => w,
        None => return respond(&mut stream, 400, "{\"error\":\"no file part\"}"),
    };
    let samples = match parse_wav_i16(wav) {
        Some(s) if !s.is_empty() => s,
        _ => return respond(&mut stream, 400, "{\"error\":\"bad or empty wav\"}"),
    };
    let text = match catch_unwind(AssertUnwindSafe(|| pipe.transcribe(&samples))) {
        Ok(t) => t,
        Err(_) => {
            eprintln!("[parakeet_serve] {peer} -> transcription panicked ({} samples)", samples.len());
            return respond(&mut stream, 500, "{\"error\":\"transcription failed\"}");
        }
    };
    eprintln!("[parakeet_serve] {peer} -> {} samples -> {:?}", samples.len(), text);
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
                fmt_ok = (audio_fmt == 1 || audio_fmt == 0xFFFE) && bits == 16 && channels == 1 && rate == 16_000;
            }
            b"data" => data = Some(&wav[body_start..body_end]),
            _ => {}
        }
        off = body_start.saturating_add(sz).saturating_add(sz & 1);
    }
    if !fmt_ok {
        return None;
    }
    let data = data?;
    let n = data.len() / 2;
    Some((0..n).map(|i| i16::from_le_bytes([data[i * 2], data[i * 2 + 1]])).collect())
}
