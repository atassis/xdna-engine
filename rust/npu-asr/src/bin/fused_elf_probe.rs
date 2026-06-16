//! Generic on-device probe for a *fused full ELF* (IRON FusedMLIROperator) via our new shim +
//! layout-driven `FusedArena`. Meta-driven so it serves every FE1 milestone (2-GEMV spike, LN→QKV,
//! self-attn block, …) without code changes.
//!
//! Proves on device: (1) the shim dispatches a full ELF at all
//! (`xrt::elf`→`hw_context(device,elf)`→`ext::kernel("main:sequence")`→N-BO run); (2) the Rust
//! `FusedArena` + named-offset placement reproduces IRON's `FusedFullELFCallable.get_buffer()`.
//!
//! Artifacts (from the route_b_kernels/decode_fused/*.py generators, host-only IRON compile):
//!   meta.json  — { elf, input_size, output_size, scratch_size, layout{name:{type,offset,len}},
//!                  inputs[], weights[], output }
//!   <elf>, buffers/<name>.bin (raw bf16 for every input/weight + the device golden `output`)
//! Gate: rel-L2(device output, buffers/<output>.bin) <= 0.08.
//!
//! NPU is single-tenant — stop npu-asr.service/voxd.service BEFORE running, restart AFTER.
//!
//! Usage:  fused_elf_probe [artifacts_dir]   (default: artifacts/fused_spike)

use std::collections::HashMap;
use std::path::Path;

use npu_xrt::{unpack_bf16_to_f32, Arena, Device, FusedArena, FusedElfPatcher};
use serde::Deserialize;

/// Optional per-token ELF patch (present for blocks with a KV cache + softmax mask): scan for the
/// magics relative to each cache's byte offset, rewrite KV write-offset + softmax mask, then load.
#[derive(Deserialize)]
struct PatchSpec {
    kv_cache_offsets: Vec<u32>,
    head_dim: u32,
    num_preceding: u32,
}

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
struct Meta {
    elf: String,
    input_size: usize,
    output_size: usize,
    scratch_size: usize,
    layout: HashMap<String, BufEntry>,
    inputs: Vec<String>,
    weights: Vec<String>,
    output: String,
    #[serde(default)]
    patch: Option<PatchSpec>,
    #[serde(default)]
    scratchpad: Option<ScratchpadSpec>,
}

impl Meta {
    fn arena_of(&self, name: &str) -> (Arena, usize, usize) {
        let e = self
            .layout
            .get(name)
            .unwrap_or_else(|| panic!("buffer '{name}' not in meta.layout"));
        let a = match e.kind.as_str() {
            "input" => Arena::Input,
            "output" => Arena::Output,
            "scratch" => Arena::Scratch,
            other => panic!("unknown arena type '{other}'"),
        };
        (a, e.offset, e.len)
    }
}

fn read(p: &Path) -> Vec<u8> {
    std::fs::read(p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()))
}

fn bf16_to_f32(bytes: &[u8]) -> Vec<f32> {
    let u16s: Vec<u16> = bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    let mut out = vec![0f32; u16s.len()];
    unpack_bf16_to_f32(&u16s, &mut out);
    out
}

fn main() {
    let dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "artifacts/fused_spike".to_string());
    let dir = Path::new(&dir);
    println!("[fused_elf_probe] artifacts: {}", dir.display());

    let meta: Meta = serde_json::from_slice(&read(&dir.join("meta.json"))).expect("parse meta.json");
    let mut elf = read(&dir.join(&meta.elf));
    // Per-token ELF patch (KV write offset + softmax mask) if this block has a cache.
    if let Some(ps) = &meta.patch {
        let patcher = FusedElfPatcher::build(&elf, &ps.kv_cache_offsets, ps.head_dim);
        println!(
            "  patcher: {} KV sites, {} softmax sites (num_preceding={})",
            patcher.kv_site_count(),
            patcher.softmax_site_count(),
            ps.num_preceding
        );
        assert!(
            patcher.kv_site_count() > 0 && patcher.softmax_site_count() > 0,
            "no patch sites found — magic/offset mismatch"
        );
        elf = patcher.patch(&elf, ps.num_preceding);
    }
    println!(
        "  {} ({}B)  arenas in/out/scratch = {}/{}/{} B  | inputs={:?} weights={:?} out={}",
        meta.elf, elf.len(), meta.input_size, meta.output_size, meta.scratch_size,
        meta.inputs, meta.weights, meta.output
    );

    let dev = Device::open(0).expect("open NPU (stop npu-asr.service/voxd.service first)");

    // Resident-scratchpad mode (deep-C): set PROBE_POS=<pos> to drive a scratchpad ELF (kv_off/sm_mask)
    // at decode position `pos` (current token at pos, context pos+1). Used to gate the batched decode ELF.
    let resident_pos: Option<u32> = std::env::var("PROBE_POS").ok().and_then(|s| s.parse().ok());
    let use_resident = resident_pos.is_some() && meta.scratchpad.is_some();

    let arena = FusedArena::new(&dev, meta.input_size, meta.output_size, meta.scratch_size)
        .expect("alloc arenas");

    // open kernel / resident handle
    let kern = if use_resident {
        None
    } else {
        let k = dev
            .load_elf_kernel(&elf, Some("main:sequence"))
            .expect("load_elf_kernel — the make-or-break call");
        println!("  load_elf_kernel OK");
        Some(k)
    };
    let resident = if use_resident {
        let res = dev
            .open_elf_resident(&elf, Some("main:sequence"))
            .expect("open_elf_resident (scratchpad ELF)");
        arena.bind_resident(&res).expect("bind resident arena BOs");
        println!("  open_elf_resident OK (PROBE_POS={})", resident_pos.unwrap());
        Some(res)
    } else {
        None
    };

    // Place every input + resident weight buffer by NAME from buffers/<name>.bin.
    for name in meta.inputs.iter().chain(meta.weights.iter()) {
        let (a, off, len) = meta.arena_of(name);
        let bytes = read(&dir.join("buffers").join(format!("{name}.bin")));
        assert_eq!(bytes.len(), len, "{name}: blob {} != layout len {len}", bytes.len());
        arena.write_at(a, off, &bytes).unwrap();
    }
    arena.sync_to_device().unwrap();

    if let Some(res) = &resident {
        let pos = resident_pos.unwrap();
        let sp = meta.scratchpad.as_ref().unwrap();
        let kv = &sp.params[&sp.kv_param];
        let sm = &sp.params[&sp.mask_param];
        let kv_val: u32 = pos * sp.head_dim; // addr-kind: raw element offset
        let sm_raw: u32 = pos + 1; // context length
        let sm_val: u32 = if sm.kind == "core" { sm_raw << 2 } else { sm_raw };
        res.write_scratchpad(kv.byte_offset, &kv_val.to_le_bytes()).unwrap();
        res.write_scratchpad(sm.byte_offset, &sm_val.to_le_bytes()).unwrap();
        res.dispatch().expect("resident scratchpad dispatch");
        println!("  resident dispatch OK (kv_off={kv_val} sm_mask={sm_val})");
    } else {
        arena.dispatch(kern.as_ref().unwrap()).expect("fused ELF dispatch");
        println!("  dispatch OK");
    }
    arena.sync_from_device().unwrap();

    // FUSED_TIME: measure the per-token NPU costs (vs per-dispatch-floor 0.35ms, M1 decode ~200ms/tok,
    // CPU ONNX ~50-82ms/tok). Times: (a) ELF re-registration (load_elf_kernel of the whole ELF), the
    // suspected dominant per-token host cost; (b) dispatch alone (reusing one kernel); (c) the full
    // per-token sequence patch→reload→sync_input→dispatch→sync_out.
    if std::env::var("FUSED_TIME").is_ok() && kern.is_some() {
        use std::time::Instant;
        let kern = kern.as_ref().unwrap();
        let warmup = 3usize;
        let iters = 20usize;
        // (a) re-registration
        for _ in 0..warmup { let _ = dev.load_elf_kernel(&elf, Some("main:sequence")).unwrap(); }
        let t = Instant::now();
        for _ in 0..iters { let _ = dev.load_elf_kernel(&elf, Some("main:sequence")).unwrap(); }
        let reg_ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;
        // (b) dispatch alone
        for _ in 0..warmup { arena.dispatch(kern).unwrap(); }
        let t = Instant::now();
        for _ in 0..iters { arena.dispatch(kern).unwrap(); }
        let disp_ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;
        // (c) full per-token (patch host buffer + reload + sync_input + dispatch + sync_out)
        let base = read(&dir.join(&meta.elf));
        let patcher = meta.patch.as_ref().map(|ps| FusedElfPatcher::build(&base, &ps.kv_cache_offsets, ps.head_dim));
        let t = Instant::now();
        for step in 0..iters {
            let ed = match &patcher { Some(p) => p.patch(&base, step as u32), None => base.clone() };
            let k = dev.load_elf_kernel(&ed, Some("main:sequence")).unwrap();
            arena.sync_input().unwrap();
            arena.dispatch(&k).unwrap();
            arena.sync_from_device().unwrap();
        }
        let tok_ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;
        println!("\n  === per-token timing (12-layer fused decode, S=448, T_enc=1500) ===");
        println!("  ELF re-registration (load_elf_kernel): {reg_ms:.2} ms");
        println!("  dispatch alone (1 NPU dispatch):        {disp_ms:.2} ms");
        println!("  FULL per-token (patch+reload+dispatch): {tok_ms:.2} ms/token");
        println!("  [compare: per-dispatch-floor 0.35ms; M1 NPU decode ~200-260 ms/tok; CPU ONNX ~50-82 ms/tok]");
    }

    // Optional intermediate dump (FUSED_DEBUG="scores,weights,ctx,qkv"): read named scratch buffers
    // post-dispatch to localize a wiring bug. Reads each by its layout offset/len.
    if let Ok(names) = std::env::var("FUSED_DEBUG") {
        for name in names.split(',').filter(|s| !s.is_empty()) {
            let (a, off, len) = meta.arena_of(name);
            let mut b = vec![0u8; len];
            arena.read_at(a, off, &mut b).unwrap();
            let f = bf16_to_f32(&b);
            let show = 8.min(f.len());
            println!("  [dbg] {name} (off {off}, {} elems): {:?}", f.len(), &f[..show]);
        }
    }

    let (oa, ooff, olen) = meta.arena_of(&meta.output);
    let mut out_bytes = vec![0u8; olen];
    arena.read_at(oa, ooff, &mut out_bytes).unwrap();

    let got = bf16_to_f32(&out_bytes);
    let want = bf16_to_f32(&read(&dir.join("buffers").join(format!("{}.bin", meta.output))));
    let n = got.len().min(want.len());
    let (mut num, mut den) = (0f64, 0f64);
    for i in 0..n {
        let d = (got[i] - want[i]) as f64;
        num += d * d;
        den += (want[i] as f64).powi(2);
    }
    let rel = if den > 0.0 { (num / den).sqrt() } else { num.sqrt() };

    println!("\n  first 6 got : {:?}", &got[..6.min(n)]);
    println!("  first 6 want: {:?}", &want[..6.min(n)]);
    println!("\n  rel-L2 = {rel:.5}  ({} elems, gate <= 0.08)", n);
    if rel <= 0.08 {
        println!("  *** PASS — fused ELF dispatch + layout-driven FusedArena correct on device ***");
    } else {
        eprintln!("  *** FAIL — rel-L2 {rel:.5} > 0.08 ***");
        std::process::exit(1);
    }
}
