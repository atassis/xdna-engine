//! Device gate for the resident-forward S1 driver, against the raw prototype build `rls1`
//! (`rlayer_design m g h arena seg=64,128 nbw=20 1 2 7`): drive gemma4-12b's 48-layer stack
//! through `npu_xrt::ElfResident` one layer/piece at a time, greedy-decode a short prompt, and
//! check the text against the known-good answer.
//!
//! `rls1` is a plain 5-argument full-ELF (`[x, o, w, kvw, kvr]`), not an
//! `aiex.scratchpad_parameter` build. `ElfResident` used to require a ctrl scratchpad section
//! (`get_ctrl_scratchpad_bo()` unconditionally, so a plain-args ELF failed to open at all); it now
//! degrades to `scratchpad_size() == 0` instead (`npu-xrt/shim/xrt_shim.cpp`,
//! `shim_elf_resident_open`/`_open_named`/`_dispatch`), which is the fix this file's first device
//! run against `rls1` needed -- see the S1 commit.
//!
//! The layer stack (open the ELF, boot before every dispatch, per-layer weight/KV BOs, the
//! per-piece x/RoPE/widths byte layout, dispatch, readback) is driven here, in Rust, over real
//! `npu_xrt::Bo`s. The host-only math around it -- tokenizer BPE, the int4 embedding dequant, the
//! RoPE tables, the widths records, and the float64 RMSNorm+head fallback -- is NOT re-derived here:
//! it is asked of a small Python subprocess (a bridge script, path given by `RF_BRIDGE_PY`) that
//! imports the already device-validated, unreleased reference modules unchanged -- not shipped in
//! this repo, so the path is configured, never hardcoded. Hand-porting that math blind, in the same
//! change as a first live run, is exactly the class of silent bug this tree's own doctrine warns
//! about; reusing the validated source is the K038 "computed by the authority, not re-derived" move.
//!
//! Run only through `npu_lock.sh`, with the service stopped for the window (see the task). Every
//! path below is read from the environment: this example ships no machine-specific default for
//! anything outside `/mnt/data` (the repo's existing convention for a device-box scratch path, e.g.
//! `detokenize.rs::qwen3_tokenizer_path`) or anything that would name a sibling private repo.
//! Required: `RF_BRIDGE_PY` (the bridge script), `RF_BRIDGE_PYTHON` (its interpreter, one with
//! pyxrt+numpy), `RF_BRIDGE_PYTHONPATH` (that interpreter's PYTHONPATH for the IRON checkout the
//! bridge imports from). Optional: `RF_BUILD_DIR` (default `/mnt/data/xdna/scratch/rf/build/rls1`).

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, Command, Stdio};

use ndarray::Array1;
use ndarray_npy::read_npy;
use npu_xrt::{unpack_bf16_to_f32, Bo, Device, ElfResident, FLAG_HOST_ONLY};
use serde_json::{json, Value};

fn build_dir() -> String {
    std::env::var("RF_BUILD_DIR").unwrap_or_else(|_| "/mnt/data/xdna/scratch/rf/build/rls1".to_string())
}

fn env_required(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("{name} must be set (see this file's module doc)"))
}

/// A line-JSON subprocess: one request per line in, one response per line out (the bridge script's
/// own protocol doc).
struct Bridge {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<std::process::ChildStdout>,
}

impl Bridge {
    fn spawn() -> Self {
        let bridge_py = env_required("RF_BRIDGE_PY");
        let cwd = Path::new(&bridge_py).parent().expect("RF_BRIDGE_PY has no parent directory").to_path_buf();
        let mut child = Command::new(env_required("RF_BRIDGE_PYTHON"))
            .arg(&bridge_py)
            .current_dir(cwd)
            .env("PYTHONPATH", env_required("RF_BRIDGE_PYTHONPATH"))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .unwrap_or_else(|e| panic!("spawn {bridge_py}: {e}"));
        let stdin = child.stdin.take().expect("bridge stdin");
        let stdout = BufReader::new(child.stdout.take().expect("bridge stdout"));
        Bridge { child, stdin, stdout }
    }

    fn call(&mut self, req: Value) -> Value {
        let line = serde_json::to_string(&req).expect("serialize request");
        self.stdin.write_all(line.as_bytes()).expect("write bridge request");
        self.stdin.write_all(b"\n").expect("write newline");
        self.stdin.flush().expect("flush bridge stdin");
        let mut resp_line = String::new();
        self.stdout.read_line(&mut resp_line).expect("read bridge response");
        assert!(!resp_line.is_empty(), "bridge closed its stdout (request was {req})");
        let resp: Value = serde_json::from_str(&resp_line).unwrap_or_else(|e| panic!("parse bridge response {resp_line:?}: {e}"));
        assert!(resp["ok"].as_bool().unwrap_or(false), "bridge error for {req}: {resp}");
        resp
    }
}

impl Drop for Bridge {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Minimal RFC 4648 base64 decoder (standard alphabet, `=` padding) -- avoids adding a crate
/// dependency to the shipped library for one example's wire format.
fn b64_decode(s: &str) -> Vec<u8> {
    fn val(c: u8) -> u8 {
        match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            _ => panic!("invalid base64 byte {c}"),
        }
    }
    let bytes: Vec<u8> = s.bytes().filter(|&c| c != b'=').collect();
    let mut out = Vec::with_capacity(bytes.len() * 3 / 4 + 3);
    for chunk in bytes.chunks(4) {
        let v: Vec<u8> = chunk.iter().map(|&c| val(c)).collect();
        out.push((v[0] << 2) | (v.get(1).copied().unwrap_or(0) >> 4));
        if v.len() > 2 {
            out.push((v[1] << 4) | (v[2] >> 2));
        }
        if v.len() > 3 {
            out.push((v[2] << 6) | v[3]);
        }
    }
    out
}

fn u16le_bytes(v: &[u16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 2);
    for &x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

fn bytes_to_u16le(b: &[u8]) -> Vec<u16> {
    b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
}

/// `wdir/w{layer}.npy` -> its raw bytes (the npy's own dtype=uint8 payload, header stripped by
/// `read_npy` itself).
fn weight_bytes(wdir: &str, layer: usize) -> Vec<u8> {
    let path = Path::new(wdir).join(format!("w{layer}.npy"));
    let arr: Array1<u8> = read_npy(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    arr.into_raw_vec_and_offset().0
}

/// One piece's dims and geometry-independent inputs, from `rust_bridge.py`'s `piece_inputs`.
struct PieceInputs {
    nt: usize,
    first: usize,
    rope_s: Vec<u8>,
    rope_g: Vec<u8>,
    wr_s: Vec<u8>,
    wr_g: Vec<u8>,
}

/// The fixed-size `%x` buffer for one layer's dispatch: x rows at offset 0, the RoPE table at the
/// fixed offset `xr` (sized for the build's max row-block count, not this piece's `nt`), the
/// widths record at the fixed offset `xbuf - wb_a2` -- the exact layout `rlayer_design.py::xbuf`'s
/// three regions describe, ported byte for byte from `stack_run.py::forward`.
fn build_x_buf(xbuf: usize, xr: usize, wb_a2: usize, x_rows: &[u16], rope: &[u8], widths: &[u8]) -> Vec<u8> {
    let mut buf = vec![0u8; xbuf];
    let x_bytes = u16le_bytes(x_rows);
    buf[..x_bytes.len()].copy_from_slice(&x_bytes);
    buf[xr..xr + rope.len()].copy_from_slice(rope);
    let w_off = xbuf - wb_a2;
    buf[w_off..w_off + widths.len()].copy_from_slice(widths);
    buf
}

/// Loaded init-response fields, named instead of re-reading `Value` at every call site.
struct Dims {
    xbuf: usize,
    obuf: usize,
    kvb_s: usize,
    wb_s: usize,
    kvb_g: usize,
    wb_g: usize,
    kvrow_s: usize,
    kvrow_g: usize,
    gcap: usize,
    seg_nb: usize,
    nlayer: usize,
    pmax: usize,
    d: usize,
    full_attention_layers: Vec<usize>,
    xr: usize,
    wb_a2: usize,
    wdir: String,
    nbw: usize,
    s_cap: usize,
    /// Not read: the build's max row-block capacity is implied by `xr` (already sized to it), kept
    /// only so `Dims` mirrors every field `rust_bridge.py`'s `init` returns.
    #[allow(dead_code)]
    pcap_t: usize,
}

impl Dims {
    fn from_json(v: &Value) -> Dims {
        let u = |k: &str| v[k].as_u64().unwrap_or_else(|| panic!("init response missing `{k}`")) as usize;
        Dims {
            xbuf: u("xbuf"), obuf: u("obuf"), kvb_s: u("kvb_s"), wb_s: u("wb_s"), kvb_g: u("kvb_g"),
            wb_g: u("wb_g"), kvrow_s: u("kvrow_s"), kvrow_g: u("kvrow_g"), gcap: u("gcap"),
            seg_nb: u("seg_nb"), nlayer: u("nlayer"), pmax: u("pmax"), d: u("d"),
            full_attention_layers: v["full_attention_layers"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect(),
            xr: u("xr"), wb_a2: u("wb_a2"), pcap_t: u("pcap_t"),
            wdir: v["wdir"].as_str().unwrap().to_string(), nbw: u("nbw"), s_cap: u("s_cap"),
        }
    }
}

/// Everything the layer loop needs held open for one generation: the device, the shared
/// hw_context's kernels, and every resident BO.
struct Stack {
    dims: Dims,
    /// One `ElfResident` per control code, all sharing ONE registered hw_context
    /// (`open_named`). `rls1` is a plain 5-argument ELF with no `aiex.scratchpad_parameter`
    /// section -- `ElfResident` now opens that shape too (`scratchpad_size() == 0`); `bind` is
    /// called fresh before every dispatch here since the K/V sub-buffer views change every layer,
    /// which its own doc allows ("bind... once (reused every dispatch)" describes the common
    /// case, not a restriction -- the underlying call is a plain `set_arg`, safe to repeat).
    kernels: HashMap<String, ElfResident>,
    xb: Bo,
    ob: Bo,
    wbo: [Bo; 2],
    kv: Vec<Bo>,
}

impl Stack {
    fn open(dev: &Device, dims: Dims) -> Stack {
        let elf = npu_models::llm::artifact::read_elf_bytes(&Path::new(&build_dir()).join("design.elf"))
            .expect("read design.elf");
        let boot = dev.open_elf_resident(&elf, Some("main:boot")).expect("open_elf_resident(boot)");
        let mut names = vec!["p1".to_string(), "p2".to_string()];
        names.push(format!("g1w{}", dims.seg_nb));
        names.push(format!("g2w{}", dims.seg_nb));
        let mut kernels: HashMap<String, ElfResident> = names
            .into_iter()
            .map(|name| {
                let k = boot.open_named(&format!("main:{name}")).unwrap_or_else(|e| panic!("open_named {name}: {e}"));
                (name, k)
            })
            .collect();
        kernels.insert("boot".to_string(), boot);

        let xb = dev.alloc_bo_raw(dims.xbuf, FLAG_HOST_ONLY, 0).expect("alloc xb");
        let ob = dev.alloc_bo_raw(dims.obuf, FLAG_HOST_ONLY, 0).expect("alloc ob");
        let wmax = dims.wb_s.max(dims.wb_g);
        let wbo = [
            dev.alloc_bo_raw(wmax, FLAG_HOST_ONLY, 0).expect("alloc wbo0"),
            dev.alloc_bo_raw(wmax, FLAG_HOST_ONLY, 0).expect("alloc wbo1"),
        ];
        let kv: Vec<Bo> = (0..dims.nlayer)
            .map(|li| {
                let g = dims.full_attention_layers.contains(&li);
                let size = if g { dims.gcap * dims.kvrow_g * 2 } else { dims.s_cap * dims.kvrow_s * 2 };
                let b = dev.alloc_bo_raw(size, FLAG_HOST_ONLY, 0).unwrap_or_else(|e| panic!("alloc kv[{li}]: {e}"));
                b.write_bytes(&vec![0u8; size]).expect("zero kv");
                b.sync_to_device().expect("sync kv");
                b
            })
            .collect();

        Stack { dims, kernels, xb, ob, wbo, kv }
    }

    /// `x_rows` is `P` rows of bf16-bit hidden states (packed u16). Returns the last layer's `P`
    /// rows, same packing, ready either to feed the next piece or to unpack for the head.
    fn forward(&mut self, bridge: &mut Bridge, x_rows: &[u16], s: usize) -> Vec<u16> {
        let p = x_rows.len() / self.dims.d;
        let resp = bridge.call(json!({"cmd": "piece_inputs", "s": s, "p": p}));
        let pin = PieceInputs {
            nt: resp["nt"].as_u64().unwrap() as usize,
            first: resp["first"].as_u64().unwrap() as usize,
            rope_s: b64_decode(resp["rope_s"].as_str().unwrap()),
            rope_g: b64_decode(resp["rope_g"].as_str().unwrap()),
            wr_s: b64_decode(resp["wr_s"].as_str().unwrap()),
            wr_g: b64_decode(resp["wr_g"].as_str().unwrap()),
        };
        let nt = pin.nt;

        // Rows padded to the piece's own 16*nt block (the reference's `xx = np.zeros((16*nt, D))`);
        // the region beyond that up to the build's max (`xr`) is left zero by `build_x_buf`.
        let mut cur = vec![0u16; 16 * nt * self.dims.d];
        cur[..x_rows.len()].copy_from_slice(x_rows);

        for li in 0..self.dims.nlayer {
            let g = self.dims.full_attention_layers.contains(&li);
            let wb_idx = li % 2;

            let wbytes = weight_bytes(&self.dims.wdir, li);
            self.wbo[wb_idx].write_bytes(&wbytes).unwrap_or_else(|e| panic!("write weight[{li}]: {e}"));
            self.wbo[wb_idx].sync_to_device().expect("sync weight");

            let rope = if g { &pin.rope_g } else { &pin.rope_s };
            let widths = if g { &pin.wr_g } else { &pin.wr_s };
            let buf = build_x_buf(self.dims.xbuf, self.dims.xr, self.dims.wb_a2, &cur, rope, widths);
            self.xb.write_bytes(&buf).expect("write xb");
            self.xb.sync_to_device().expect("sync xb");

            let kvrow = if g { self.dims.kvrow_g } else { self.dims.kvrow_s };
            let kvw_len = (if g { self.dims.kvb_g } else { self.dims.kvb_s }) * 2;
            let kvw = self.kv[li].sub(s * kvrow * 2, kvw_len).unwrap_or_else(|e| panic!("kvw[{li}]: {e}"));
            let read_positions = if g { self.dims.gcap } else { self.dims.nbw * 64 };
            let kvr_off = if g { 0 } else { pin.first };
            let kvr = self.kv[li].sub(kvr_off * kvrow * 2, read_positions * kvrow * 2).unwrap_or_else(|e| panic!("kvr[{li}]: {e}"));

            let kernel_name = if g { format!("g{nt}w{}", self.dims.seg_nb) } else { format!("p{nt}") };
            let args: [&Bo; 5] = [&self.xb, &self.ob, &self.wbo[wb_idx], &kvw, &kvr];
            // Boot before every dispatch: a resident image loses its configuration within
            // seconds idle, and a foreign context between dispatches has the same symptom. `bind`
            // is called fresh each time (not once) because `kvw`/`kvr` are new sub-buffer views
            // every layer -- see `Stack::kernels`'s doc.
            let boot = self.kernels.get("boot").expect("boot kernel");
            boot.bind(&args).unwrap_or_else(|e| panic!("bind boot before layer {li}: {e}"));
            boot.dispatch().unwrap_or_else(|e| panic!("boot before layer {li}: {e}"));
            let k = self.kernels.get(&kernel_name).unwrap_or_else(|| panic!("no kernel for {kernel_name}"));
            k.bind(&args).unwrap_or_else(|e| panic!("bind {kernel_name} (layer {li}): {e}"));
            k.dispatch().unwrap_or_else(|e| panic!("dispatch {kernel_name} (layer {li}): {e}"));

            self.ob.sync_from_device().expect("sync ob");
            let mut out_bytes = vec![0u8; self.dims.xr];
            self.ob.read_bytes_at(0, &mut out_bytes).expect("read ob");
            let out_u16 = bytes_to_u16le(&out_bytes);
            // out_u16 is [16*pcap_t, d] row-major; the piece's P real rows are the first P*d
            // elements (`stack_run.py::forward`'s `view(ob, xr)...reshape(16*PCAP_T, D)[:P]`).
            cur = out_u16[..p * self.dims.d].to_vec();

            let mut f32_check = vec![0f32; cur.len()];
            unpack_bf16_to_f32(&cur, &mut f32_check);
            assert!(f32_check.iter().all(|v| v.is_finite()), "layer {li}: non-finite output at s={s}");
        }
        cur
    }
}

fn embed_bf16(bridge: &mut Bridge, ids: &[u32]) -> Vec<u16> {
    let resp = bridge.call(json!({"cmd": "embed", "ids": ids}));
    bytes_to_u16le(&b64_decode(resp["x"].as_str().unwrap()))
}

fn next_token(bridge: &mut Bridge, h: &[f32]) -> u32 {
    let resp = bridge.call(json!({"cmd": "next_token", "h": h}));
    resp["token"].as_u64().unwrap() as u32
}

fn main() {
    let prompt = std::env::args().nth(1).unwrap_or_else(|| "What is the capital of France?".to_string());
    let n_new: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(8);
    let expect = std::env::args().nth(3);

    let mut bridge = Bridge::spawn();
    let mut dims = Dims::from_json(&bridge.call(json!({"cmd": "init"})));
    // Incremental smoke check before the full 48-layer/multi-token gate: `RF_NLAYER_LIMIT=1` runs
    // only layer 0 of the first piece. Not a build knob -- `stack.forward` still runs every layer
    // the loop is told to, so a limited run's output is not comparable to the gate's expected text.
    if let Ok(n) = std::env::var("RF_NLAYER_LIMIT") {
        dims.nlayer = n.parse().expect("RF_NLAYER_LIMIT must be a number");
    }
    let full_prompt = format!("<bos><|turn>user\n{prompt}<turn|>\n<|turn>model\n");
    let ids: Vec<u32> = bridge.call(json!({"cmd": "tokenize", "text": full_prompt}))["ids"]
        .as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
    eprintln!("prompt ids {ids:?}");

    let dev = Device::open(0).expect("open NPU device 0");
    let mut stack = Stack::open(&dev, dims);

    let pmax = stack.dims.pmax;
    let mut s = 0usize;
    let mut h: Vec<u16> = Vec::new();
    for chunk in ids.chunks(pmax) {
        let x = embed_bf16(&mut bridge, chunk);
        h = stack.forward(&mut bridge, &x, s);
        s += chunk.len();
    }

    let mut out_ids = Vec::new();
    for _ in 0..n_new {
        let last_row = &h[h.len() - stack.dims.d..];
        let mut h_f32 = vec![0f32; stack.dims.d];
        unpack_bf16_to_f32(last_row, &mut h_f32);
        let t = next_token(&mut bridge, &h_f32);
        out_ids.push(t);
        eprintln!("token {t}");
        if t == 1 || t == 106 || t == 50 {
            break;
        }
        let x = embed_bf16(&mut bridge, &[t]);
        h = stack.forward(&mut bridge, &x, s);
        s += 1;
    }

    let text = bridge.call(json!({"cmd": "detok", "ids": out_ids}))["text"].as_str().unwrap().to_string();
    println!("TEXT: {text:?}");
    if let Some(expect) = expect {
        if text == expect {
            println!("GATE: PASS");
        } else {
            println!("GATE: FAIL (expected {expect:?})");
            std::process::exit(1);
        }
    }
}
