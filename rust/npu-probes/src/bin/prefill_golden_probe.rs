//! Localise a batched-prefill divergence to a LAYER, by comparing the device's own per-layer KV
//! against the CPU golden the generator emitted from the same inputs.
//!
//! The token gate says the batched KV differs from per-token priming (step-0 logits rel-L2
//! 0.14-0.17). That is 28 layers deep and tells you nothing about WHERE. This runs ONE prefill
//! dispatch, standalone -- no decode artifact, no pairing, no handoff -- using the artifact's own
//! `buffers/{x,rope,sm_widths}.bin`, and diffs `xout` plus every `L{l}_{kc,vc}` slab against
//! `buffers/golden/`. The first layer that diverges is the one to read.
//!
//! It also splits the space in half on its own: if xout and every slab MATCH, the prefill graph is
//! right and the bug is in the handoff or the control; if they differ, the bug is inside the graph.
//!
//! NPU is single-tenant -- run under `xdna-engine-private/journal/scripts/npu_lock.sh`.
//!
//! Usage: prefill_golden_probe <prefill_artifact_dir>

use std::collections::HashMap;
use std::path::Path;

use npu_xrt::{unpack_bf16_to_f32, Arena, Device, FusedArena};
use serde::Deserialize;

#[derive(Deserialize)]
struct BufEntry {
    #[serde(rename = "type")]
    kind: String,
    offset: usize,
    len: usize,
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
    weights_from: String,
    golden: HashMap<String, String>,
    #[serde(default)]
    dims: HashMap<String, serde_json::Value>,
}

fn read(p: &Path) -> Vec<u8> {
    std::fs::read(p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()))
}

fn arena_of(m: &Meta, name: &str) -> (Arena, usize, usize) {
    let e = m.layout.get(name).unwrap_or_else(|| panic!("'{name}' not in layout"));
    let a = match e.kind.as_str() {
        "input" => Arena::Input,
        "output" => Arena::Output,
        "scratch" => Arena::Scratch,
        o => panic!("unknown arena '{o}'"),
    };
    (a, e.offset, e.len)
}

fn f32s(b: &[u8]) -> Vec<f32> {
    let u: Vec<u16> = b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
    let mut out = vec![0f32; u.len()];
    unpack_bf16_to_f32(&u, &mut out);
    out
}

/// rel-L2 plus the index and magnitude of the worst element, because "where" is usually more
/// diagnostic than "how much".
fn diff(got: &[u8], want: &[u8]) -> (f64, usize, f32, f32) {
    let (g, w) = (f32s(got), f32s(want));
    assert_eq!(g.len(), w.len(), "length mismatch {} vs {}", g.len(), w.len());
    let (mut num, mut den, mut worst, mut wi) = (0f64, 0f64, 0f32, 0usize);
    for (i, (&a, &b)) in g.iter().zip(&w).enumerate() {
        num += ((a - b) as f64).powi(2);
        den += (b as f64).powi(2);
        if (a - b).abs() > worst {
            worst = (a - b).abs();
            wi = i;
        }
    }
    ((num.sqrt() / den.sqrt().max(1e-30)), wi, g[wi], w[wi])
}

/// GATE_DUMP_DIR=<dir>: write the raw device bytes for Tier 1 (scripts/gate_numeric.py), which
/// judges them element-wise against a float32 reference. rel-L2 below stays a localiser, not a
/// gate: it is a summary and one structurally wrong element cannot move it.
fn dump(dir: &Option<String>, name: &str, bytes: &[u8]) {
    let Some(d) = dir else { return };
    std::fs::create_dir_all(d).unwrap_or_else(|e| panic!("mkdir {d}: {e}"));
    let p = Path::new(d).join(format!("{name}.bin"));
    std::fs::write(&p, bytes).unwrap_or_else(|e| panic!("write {}: {e}", p.display()));
}

fn main() {
    let dir = std::env::args().nth(1).expect("usage: prefill_golden_probe <prefill_dir>");
    let dir = Path::new(&dir);
    let meta: Meta = serde_json::from_slice(&read(&dir.join("meta.json"))).expect("meta.json");
    let elf = read(&dir.join(&meta.elf));
    let nl = meta.dims.get("layers").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    println!("artifact {}  {} layers  scratch {:.2} GB", dir.display(), nl,
             meta.scratch_size as f64 / 1e9);

    let dev = Device::open(0).expect("open NPU (single-tenant -- use npu_lock.sh)");
    let arena = FusedArena::new(&dev, meta.input_size, meta.output_size, meta.scratch_size)
        .expect("alloc arena");

    // Weights live in the DECODE artifact's buffers/ -- the prefill artifact deliberately emits
    // none, because the two share one arena and a second copy would defeat the point.
    let wdir = Path::new(&meta.weights_from);
    for n in &meta.weights {
        let (a, off, len) = arena_of(&meta, n);
        let b = read(&wdir.join(format!("{n}.bin")));
        assert_eq!(b.len(), len, "{n}: blob {} != layout {len}", b.len());
        arena.write_at(a, off, &b).unwrap();
    }
    // The artifact's OWN inputs -- the ones the golden was computed from. Not host-generated.
    for n in &meta.inputs {
        let (a, off, len) = arena_of(&meta, n);
        let b = read(&dir.join("buffers").join(format!("{n}.bin")));
        assert_eq!(b.len(), len, "{n}: blob {} != layout {len}", b.len());
        arena.write_at(a, off, &b).unwrap();
    }
    arena.sync_to_device().unwrap();
    println!("uploaded {} weights + {} inputs", meta.weights.len(), meta.inputs.len());

    let res = dev.open_elf_resident(&elf, Some("main:sequence")).expect("open prefill resident");
    arena.bind_resident(&res).expect("bind");
    res.dispatch().expect("prefill dispatch");
    arena.sync_from_device().unwrap();
    arena.sync_scratch_from_device().unwrap();
    println!("dispatched, output + scratch synced back\n");

    let gate_dir = std::env::var("GATE_DUMP_DIR").ok();
    let mut worst_layer: Option<(f64, String)> = None;
    println!("{:<10} {:>12} {:>10} {:>12} {:>12}", "tensor", "rel-L2", "worst i", "got", "want");
    // xout first: the graph's end-to-end answer.
    {
        let (a, off, len) = arena_of(&meta, "xout");
        let mut got = vec![0u8; len];
        arena.read_at(a, off, &mut got).unwrap();
        dump(&gate_dir, "xout", &got);
        let want = read(&dir.join(meta.golden.get("xout").expect("golden xout")));
        let (r, i, g, w) = diff(&got, &want);
        println!("{:<10} {:>12.3e} {:>10} {:>12.5} {:>12.5}", "xout", r, i, g, w);
    }
    // Then every layer's KV slab, in order, so the FIRST divergence is visible.
    let m = meta.dims.get("M").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let hkv = meta.dims.get("kv_heads").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let hd = meta.dims.get("head_dim").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let s = meta.dims.get("S").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    for l in 0..nl {
        for which in ["kc", "vc"] {
            let name = format!("L{l}_{which}");
            let Some(gpath) = meta.golden.get(&name) else { continue };
            let (a, off, _len) = arena_of(&meta, &name);
            // The golden is the SLAB cache[h, base:base+M, :] for base=0, not the whole cache.
            let mut got = vec![0u8; hkv * m * hd * 2];
            for h in 0..hkv {
                let src = off + h * s * hd * 2;
                let dst = h * m * hd * 2;
                arena.read_at(a, src, &mut got[dst..dst + m * hd * 2]).unwrap();
            }
            dump(&gate_dir, &name, &got);
            let want = read(&dir.join(gpath));
            let (r, i, g, w) = diff(&got, &want);
            if worst_layer.as_ref().map_or(true, |(br, _)| r > *br) {
                worst_layer = Some((r, name.clone()));
            }
            if l < 3 || r > 1e-2 || l == nl - 1 {
                println!("{:<10} {:>12.3e} {:>10} {:>12.5} {:>12.5}", name, r, i, g, w);
            }
        }
    }
    if let Some((r, n)) = worst_layer {
        println!("\nworst KV slab: {n} at rel-L2 {r:.3e}");
    }
}
