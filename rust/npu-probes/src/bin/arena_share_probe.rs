//! Kill-if probe for the batched-prefill architecture: **can two ELFs on two hardware contexts
//! share ONE `FusedArena`, and does a device-side scratch write by one become visible to the
//! other?**
//!
//! This is the load-bearing question, because the weights live in the scratch arena
//! (`gen_llm_decode.py` declares only `x`/`rope_global` as inputs and `logits` as output, so all
//! 1.110 GiB of weights and all 224 MiB of KV cache are scratch). If two ELFs cannot share one
//! arena, batched prefill needs a second weight copy AND a host round-trip of the KV cache, and
//! the design in `docs/reference/batched-prefill-architecture.md` collapses.
//!
//! Two phases, both against ONE arena:
//!   A. two `ElfResident`s (= two hw_contexts) from the same ELF. Drive a two-step decode with
//!      step 0 on res1 and step 1 on res2, then the SAME two steps entirely on res1, and compare
//!      the logits BITWISE. Step 1 reads the KV row step 0 wrote, so equality proves cross-context
//!      visibility of a device-side scratch write -- exactly the prefill->decode handoff.
//!   B. one `ElfCtx` + two `rebind`s (= one hw_context, two programs). The free-transition path
//!      that [[transition-cost-is-the-context-not-the-program]] measured.
//!
//! NPU is single-tenant -- run under `xdna-engine-private/journal/scripts/npu_lock.sh`.
//!
//! Usage: arena_share_probe <artifact_dir>

use std::collections::HashMap;
use std::path::Path;

use npu_xrt::{context_report, Arena, Device, FusedArena};
use serde::Deserialize;

#[derive(Deserialize)]
struct BufEntry {
    #[serde(rename = "type")]
    kind: String,
    offset: usize,
    len: usize,
}

#[derive(Deserialize)]
struct ParamSpec {
    byte_offset: usize,
    kind: String,
}

#[derive(Deserialize)]
struct ScratchpadSpec {
    params: HashMap<String, ParamSpec>,
    kv_param: String,
    mask_param: String,
    head_dim: u32,
}

#[derive(Deserialize)]
struct Dims {
    d_model: usize,
    head_dim: usize,
    vocab: usize,
}

#[derive(Deserialize)]
struct Proto {
    rope_theta_global: f64,
}

#[derive(Deserialize)]
struct Meta {
    elf: String,
    input_size: usize,
    output_size: usize,
    scratch_size: usize,
    layout: HashMap<String, BufEntry>,
    #[allow(dead_code)]
    inputs: Vec<String>,
    weights: Vec<String>,
    output: String,
    #[serde(default)]
    scratchpad: Option<ScratchpadSpec>,
    #[serde(default)]
    dims: Option<Dims>,
    #[serde(default)]
    host_protocol: Option<Proto>,
    #[serde(default)]
    cache_buffers: Vec<String>,
}

impl Meta {
    /// Same narrow compat shim `npu-engine/src/llm/artifact.rs:242` carries, and for the same
    /// reason: the shipped artifact predates the generator fix that built `layout` from the graph's
    /// DECLARED args, so `rope_global` -- a real `inputs` entry -- has no layout row. Only for the
    /// exact `rope_global`-immediately-after-`x` packing this generator produces, and it refuses to
    /// guess if that does not hold.
    fn arena_of(&self, name: &str) -> (Arena, usize, usize) {
        if name == "rope_global" && !self.layout.contains_key(name) {
            let x = self.layout.get("x").expect("rope_global shim needs x's layout row");
            let off = x.offset + x.len;
            let len = self.input_size.checked_sub(off)
                .expect("rope_global shim: x runs past input_size");
            assert!(len > 0, "rope_global shim: nothing left after x in the input arena");
            return (Arena::Input, off, len);
        }
        let e = self.layout.get(name).unwrap_or_else(|| panic!("'{name}' not in meta.layout"));
        let a = match e.kind.as_str() {
            "input" => Arena::Input,
            "output" => Arena::Output,
            "scratch" => Arena::Scratch,
            o => panic!("unknown arena '{o}'"),
        };
        (a, e.offset, e.len)
    }
}

fn read(p: &Path) -> Vec<u8> {
    std::fs::read(p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()))
}

/// One position's RoPE angle row, INTERLEAVED [cos, sin, ...] -- the convention
/// `iron/operators/rope/reference.py` documents and `npu_decode.rs::rope_row` implements.
fn rope_row(pos: usize, head_dim: usize, theta: f64) -> Vec<u8> {
    let half = head_dim / 2;
    let mut out = vec![0u8; head_dim * 2];
    for i in 0..half {
        let inv = 1.0 / theta.powf((2 * i) as f64 / head_dim as f64);
        let ang = pos as f64 * inv;
        for (k, v) in [(2 * i, ang.cos()), (2 * i + 1, ang.sin())] {
            let bits = ((v as f32).to_bits() >> 16) as u16;
            out[2 * k..2 * k + 2].copy_from_slice(&bits.to_le_bytes());
        }
    }
    out
}

/// Upload every input + weight buffer from buffers/<name>.bin. Done ONCE for the whole probe --
/// that is half the point: context 2 never gets its own upload.
fn upload_all(meta: &Meta, dir: &Path, arena: &FusedArena) {
    for name in meta.weights.iter() {
        let (a, off, len) = meta.arena_of(name);
        let bytes = read(&dir.join("buffers").join(format!("{name}.bin")));
        assert_eq!(bytes.len(), len, "{name}: blob {} != layout len {len}", bytes.len());
        arena.write_at(a, off, &bytes).unwrap();
    }
}

/// Re-zero every KV cache buffer so a second run starts from the same state as the first.
fn zero_caches(meta: &Meta, arena: &FusedArena) {
    for name in meta.cache_buffers.iter() {
        let (a, off, len) = meta.arena_of(name);
        arena.write_at(a, off, &vec![0u8; len]).unwrap();
    }
}

/// One decode step on `res`: write x + the rope row + the two scratchpad params, dispatch,
/// return the logits bytes.
#[allow(clippy::too_many_arguments)]
fn step(
    meta: &Meta, arena: &FusedArena, res: &npu_xrt::ElfResident,
    embed: &[u8], token: usize, pos: usize, d_model: usize, head_dim: usize, theta: f64,
) -> Vec<u8> {
    let (xa, xo, xl) = meta.arena_of("x");
    assert_eq!(xl, d_model * 2);
    arena.write_at(xa, xo, &embed[token * d_model * 2..(token + 1) * d_model * 2]).unwrap();
    let (ra, ro, rl) = meta.arena_of("rope_global");
    let row = rope_row(pos, head_dim, theta);
    assert_eq!(rl, row.len(), "rope_global len {rl} != head_dim*2 {}", row.len());
    arena.write_at(ra, ro, &row).unwrap();
    arena.sync_input().unwrap();

    let sp = meta.scratchpad.as_ref().unwrap();
    let kv = &sp.params[&sp.kv_param];
    let sm = &sp.params[&sp.mask_param];
    let kv_val = (pos as u32) * sp.head_dim;
    let sm_raw = pos as u32 + 1;
    let sm_val = if sm.kind == "core" { sm_raw << 2 } else { sm_raw };
    res.write_scratchpad(kv.byte_offset, &kv_val.to_le_bytes()).unwrap();
    res.write_scratchpad(sm.byte_offset, &sm_val.to_le_bytes()).unwrap();
    res.dispatch().expect("resident dispatch");
    arena.sync_from_device().unwrap();

    let (oa, oo, ol) = meta.arena_of(&meta.output);
    let mut out = vec![0u8; ol];
    arena.read_at(oa, oo, &mut out).unwrap();
    out
}

fn argmax_bf16(b: &[u8]) -> (usize, f32) {
    let mut best = (0usize, f32::NEG_INFINITY);
    for (i, c) in b.chunks_exact(2).enumerate() {
        let v = f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16);
        if v > best.1 {
            best = (i, v);
        }
    }
    best
}

fn main() {
    let dir = std::env::args().nth(1).unwrap_or_else(|| "artifacts/qwen3-0.6b/decode".into());
    let dir = Path::new(&dir);
    let meta: Meta = serde_json::from_slice(&read(&dir.join("meta.json"))).expect("meta.json");
    let elf = read(&dir.join(&meta.elf));
    println!("artifact {}  elf {}B  arenas in/out/scratch {}/{}/{} B",
             dir.display(), elf.len(), meta.input_size, meta.output_size, meta.scratch_size);

    let dev = Device::open(0).expect("open NPU (device is single-tenant -- use npu_lock.sh)");
    println!("  {}", context_report());

    let arena = FusedArena::new(&dev, meta.input_size, meta.output_size, meta.scratch_size)
        .expect("alloc ONE arena");
    upload_all(&meta, dir, &arena);
    arena.sync_to_device().unwrap();
    println!("  uploaded {} weight buffers ONCE, synced", meta.weights.len());

    // ---- two hw_contexts, one arena ----
    let res1 = dev.open_elf_resident(&elf, Some("main:sequence")).expect("resident 1");
    arena.bind_resident(&res1).expect("bind arena to resident 1");
    let res2 = dev.open_elf_resident(&elf, Some("main:sequence")).expect("resident 2");
    arena.bind_resident(&res2).expect("bind arena to resident 2");
    println!("  TWO residents opened and BOTH bound to the SAME arena. {}", context_report());

    let Some(sp) = meta.scratchpad.as_ref() else {
        println!("  artifact has no scratchpad -- read-sharing shown, write-visibility NOT tested");
        println!("  RESULT: PARTIAL (rerun against a KV-cache artifact for the decisive test)");
        return;
    };
    let dims = meta.dims.as_ref().expect("dims");
    let theta = meta.host_protocol.as_ref().expect("host_protocol").rope_theta_global;
    let (d_model, head_dim) = (dims.d_model, dims.head_dim);
    let embed = read(&dir.join("buffers").join("W_head.bin"));
    assert_eq!(embed.len(), dims.vocab * d_model * 2, "W_head is the tied embedding table");
    let _ = sp;

    // Two tokens, teacher-forced at positions 0 and 1. Step 1 reads the KV row step 0 wrote.
    let (t0, t1) = (100usize, 200usize);

    // Arm AA: both steps on resident 1.
    zero_caches(&meta, &arena);
    arena.sync_to_device().unwrap();
    let _ = step(&meta, &arena, &res1, &embed, t0, 0, d_model, head_dim, theta);
    let aa = step(&meta, &arena, &res1, &embed, t1, 1, d_model, head_dim, theta);

    // Arm AB: step 0 on resident 1, step 1 on resident 2 -- the cross-context handoff.
    zero_caches(&meta, &arena);
    arena.sync_to_device().unwrap();
    let _ = step(&meta, &arena, &res1, &embed, t0, 0, d_model, head_dim, theta);
    let ab = step(&meta, &arena, &res2, &embed, t1, 1, d_model, head_dim, theta);

    // Arm BB: both steps on resident 2, as a control that res2 is not simply degenerate.
    zero_caches(&meta, &arena);
    arena.sync_to_device().unwrap();
    let _ = step(&meta, &arena, &res2, &embed, t0, 0, d_model, head_dim, theta);
    let bb = step(&meta, &arena, &res2, &embed, t1, 1, d_model, head_dim, theta);

    // Optional: dump the AA logits so two artifacts built from different generator states can be
    // compared bitwise. This is the token-identity gate for any change that alters the ELF without
    // intending to alter the arithmetic.
    if let Ok(path) = std::env::var("DUMP_LOGITS") {
        std::fs::write(&path, &aa).unwrap_or_else(|e| panic!("write {path}: {e}"));
        println!("  dumped AA logits ({} bytes) to {path}", aa.len());
    }

    let (ia, va) = argmax_bf16(&aa);
    let (ib, vb) = argmax_bf16(&ab);
    let (ic, vc) = argmax_bf16(&bb);
    println!("  AA (res1,res1) argmax {ia} @ {va}");
    println!("  AB (res1,res2) argmax {ib} @ {vb}   <- the cross-context handoff");
    println!("  BB (res2,res2) argmax {ic} @ {vc}");

    let diff = aa.iter().zip(&ab).filter(|(x, y)| x != y).count();
    let diff_bb = aa.iter().zip(&bb).filter(|(x, y)| x != y).count();
    println!("  bitwise: AA vs AB differ in {diff}/{} bytes; AA vs BB differ in {diff_bb}",
             aa.len());

    if diff == 0 && diff_bb == 0 {
        println!("  *** PASS -- one FusedArena serves two hardware contexts, and a device-side");
        println!("      scratch write in one context is visible to the other. The prefill->decode");
        println!("      KV handoff needs no copy and no second weight arena. ***");
    } else {
        println!("  *** FAIL -- cross-context arena sharing does NOT preserve device-side writes.");
        println!("      The architecture's kill-if fired; batched prefill must fall back to a");
        println!("      single ELF or a host KV copy. ***");
        std::process::exit(1);
    }
}
