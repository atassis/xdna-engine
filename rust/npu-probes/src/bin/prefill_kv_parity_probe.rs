//! The decisive gate: does batched prefill write the SAME KV bytes as `P` sequential M=1 steps?
//!
//! `prefill_golden_probe` compares the device against a NUMPY golden, which conflates a real bug
//! with ordinary bf16-vs-float32 disagreement -- the shipped MLP block sits at rel-L2 2.5e-2
//! against its own golden and passes. This compares device against device: same silicon, same
//! weights, same bf16, one arena, differing only in HOW the cache was primed. Whatever the two
//! disagree on is the batching, not the arithmetic's reference.
//!
//! Both arms drive the raw residents so nothing in the host's decision logic can differ between
//! them, and both prime exactly the same positions with the same tokens.
//!
//! NPU is single-tenant -- run under `xdna-engine-private/journal/scripts/npu_lock.sh`.
//!
//! Usage: prefill_kv_parity_probe <decode_dir> <prefill_dir> [--n 256]

use std::collections::HashMap;
use std::path::Path;

use npu_xrt::{unpack_bf16_to_f32, Arena, Device, ElfResident, FusedArena};
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
struct Scratchpad {
    params: HashMap<String, ParamSpec>,
    kv_param: String,
    #[serde(default)]
    mask_param: Option<String>,
}
#[derive(Deserialize)]
struct Meta {
    elf: String,
    input_size: usize,
    output_size: usize,
    scratch_size: usize,
    layout: HashMap<String, BufEntry>,
    inputs: Vec<String>,
    #[serde(default)]
    weights: Vec<String>,
    #[serde(default)]
    weights_from: Option<String>,
    #[serde(default)]
    cache_buffers: Vec<String>,
    scratchpad: Scratchpad,
    dims: HashMap<String, serde_json::Value>,
}
impl Meta {
    fn d(&self, k: &str) -> usize {
        self.dims.get(k).and_then(|v| v.as_u64()).unwrap_or(0) as usize
    }
    fn at(&self, n: &str) -> (Arena, usize, usize) {
        let e = self.layout.get(n).unwrap_or_else(|| panic!("'{n}' not in layout"));
        let a = match e.kind.as_str() {
            "input" => Arena::Input,
            "output" => Arena::Output,
            "scratch" => Arena::Scratch,
            o => panic!("arena '{o}'"),
        };
        (a, e.offset, e.len)
    }
}

fn read(p: &Path) -> Vec<u8> {
    std::fs::read(p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()))
}
fn pack(f: &[f32]) -> Vec<u8> {
    let mut b = vec![0u8; f.len() * 2];
    for (i, v) in f.iter().enumerate() {
        b[2 * i..2 * i + 2].copy_from_slice(&((v.to_bits() >> 16) as u16).to_le_bytes());
    }
    b
}
fn rope_row(pos: usize, hd: usize, theta: f64) -> Vec<f32> {
    let half = hd / 2;
    let mut r = vec![0f32; hd];
    for i in 0..half {
        let ang = pos as f64 / theta.powf((2 * i) as f64 / hd as f64);
        r[2 * i] = ang.cos() as f32;
        r[2 * i + 1] = ang.sin() as f32;
    }
    r
}
fn f32s(b: &[u8]) -> Vec<f32> {
    let u: Vec<u16> = b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
    let mut o = vec![0f32; u.len()];
    unpack_bf16_to_f32(&u, &mut o);
    o
}
fn diff(a: &[u8], b: &[u8]) -> (f64, usize, usize) {
    let (x, y) = (f32s(a), f32s(b));
    let (mut num, mut den, mut nne) = (0f64, 0f64, 0usize);
    for (p, q) in x.iter().zip(&y) {
        num += ((p - q) as f64).powi(2);
        den += (*q as f64).powi(2);
        if p != q {
            nne += 1;
        }
    }
    (num.sqrt() / den.sqrt().max(1e-30), nne, x.len())
}

/// Snapshot cache[h, 0..n, :] for every kv head, for one cache buffer.
fn slab(m: &Meta, ar: &FusedArena, name: &str, n: usize) -> Vec<u8> {
    let (hkv, hd, s) = (m.d("kv_heads"), m.d("head_dim"), m.d("S"));
    let (a, off, _) = m.at(name);
    let mut out = vec![0u8; hkv * n * hd * 2];
    for h in 0..hkv {
        let src = off + h * s * hd * 2;
        let dst = h * n * hd * 2;
        ar.read_at(a, src, &mut out[dst..dst + n * hd * 2]).unwrap();
    }
    out
}

fn zero_caches(m: &Meta, ar: &FusedArena) {
    for n in &m.cache_buffers {
        let (a, off, len) = m.at(n);
        ar.write_at(a, off, &vec![0u8; len]).unwrap();
    }
    ar.sync_to_device().unwrap();
}

fn main() {
    let mut it = std::env::args().skip(1);
    let dd = it.next().expect("usage: <decode_dir> <prefill_dir>");
    let pd = it.next().expect("usage: <decode_dir> <prefill_dir>");
    let mut n = 256usize;
    while let Some(f) = it.next() {
        if f == "--n" {
            n = it.next().unwrap().parse().unwrap();
        }
    }
    let (dd, pd) = (Path::new(&dd), Path::new(&pd));
    let dm: Meta = serde_json::from_slice(&read(&dd.join("meta.json"))).unwrap();
    let pm: Meta = serde_json::from_slice(&read(&pd.join("meta.json"))).unwrap();
    let (m_batch, hd, d_model) = (pm.d("M"), pm.d("head_dim"), pm.d("d_model"));
    assert!(n <= m_batch, "--n {n} exceeds the prefill batch {m_batch}");
    let theta = 1_000_000.0f64;

    let dev = Device::open(0).expect("open NPU");
    let arena = FusedArena::new(
        &dev,
        dm.input_size.max(pm.input_size),
        dm.output_size.max(pm.output_size),
        dm.scratch_size.max(pm.scratch_size),
    )
    .expect("alloc one arena for both");

    let wdir = pm.weights_from.clone().map(|s| s.into()).unwrap_or_else(|| dd.join("buffers"));
    let wdir = Path::new(&wdir);
    for nm in &pm.weights {
        let (a, off, len) = pm.at(nm);
        let b = read(&wdir.join(format!("{nm}.bin")));
        assert_eq!(b.len(), len, "{nm}");
        arena.write_at(a, off, &b).unwrap();
    }
    arena.sync_to_device().unwrap();
    let embed = read(&wdir.join("W_head.bin"));

    let d_elf = read(&dd.join(&dm.elf));
    let p_elf = read(&pd.join(&pm.elf));
    let d_res = dev.open_elf_resident(&d_elf, Some("main:sequence")).expect("decode resident");
    let p_res = dev.open_elf_resident(&p_elf, Some("main:sequence")).expect("prefill resident");
    arena.bind_resident(&d_res).unwrap();
    arena.bind_resident(&p_res).unwrap();
    println!("one arena, two residents; comparing KV over {n} positions\n");

    let toks: Vec<u32> = (0..n).map(|i| ((i as u64 * 7919 + 1234) % 151935) as u32 + 1).collect();

    // ---- Arm A: per-token, driving the DECODE resident ----
    zero_caches(&pm, &arena);
    let sp = &dm.scratchpad;
    let kvp = &sp.params[&sp.kv_param];
    let smp = sp.mask_param.as_ref().map(|k| &sp.params[k]);
    for (i, &t) in toks.iter().enumerate() {
        let (a, off, _) = dm.at("x");
        arena.write_at(a, off, &embed[t as usize * d_model * 2..(t as usize + 1) * d_model * 2]).unwrap();
        let (ra, ro, _) = dm.at("rope_global");
        arena.write_at(ra, ro, &pack(&rope_row(i, hd, theta))).unwrap();
        arena.sync_input().unwrap();
        d_res.write_scratchpad(kvp.byte_offset, &((i * hd) as u32).to_le_bytes()).unwrap();
        if let Some(s) = smp {
            let v = (i as u32 + 1) << if s.kind == "core" { 2 } else { 0 };
            d_res.write_scratchpad(s.byte_offset, &v.to_le_bytes()).unwrap();
        }
        d_res.dispatch().expect("decode step");
    }
    arena.sync_scratch_from_device().unwrap();
    let arm_a: Vec<Vec<u8>> =
        pm.cache_buffers.iter().map(|c| slab(&pm, &arena, c, n)).collect();
    println!("arm A (per-token, {n} dispatches) captured");

    // ---- Arm B: one batched prefill dispatch ----
    zero_caches(&pm, &arena);
    let (xa, xo, _) = pm.at("x");
    let mut xbuf = vec![0u8; m_batch * d_model * 2];
    for (i, &t) in toks.iter().enumerate() {
        xbuf[i * d_model * 2..(i + 1) * d_model * 2]
            .copy_from_slice(&embed[t as usize * d_model * 2..(t as usize + 1) * d_model * 2]);
    }
    arena.write_at(xa, xo, &xbuf).unwrap();
    let (ra, ro, _) = pm.at("rope");
    let mut rbuf = Vec::with_capacity(m_batch * hd * 2);
    for i in 0..m_batch {
        rbuf.extend_from_slice(&pack(&rope_row(i, hd, theta)));
    }
    arena.write_at(ra, ro, &rbuf).unwrap();
    if let Some(_) = pm.layout.get("sm_widths") {
        let (wa, wo, _) = pm.at("sm_widths");
        let (hq, s) = (pm.d("q_heads"), pm.d("S"));
        let mut w = Vec::with_capacity(hq * m_batch * 4);
        for _h in 0..hq {
            for i in 0..m_batch {
                w.extend_from_slice(&((i as i32 + 1).clamp(1, s as i32)).to_le_bytes());
            }
        }
        arena.write_at(wa, wo, &w).unwrap();
    }
    arena.sync_input().unwrap();
    let psp = &pm.scratchpad;
    p_res.write_scratchpad(psp.params[&psp.kv_param].byte_offset, &0u32.to_le_bytes()).unwrap();
    p_res.dispatch().expect("prefill dispatch");
    arena.sync_scratch_from_device().unwrap();
    println!("arm B (batched, 1 dispatch) captured\n");

    // Dump both arms so a host-side reference can say which is closer to the TRUE answer. The
    // parity number alone treats GEMV as correct by assumption; it is not obviously the more
    // accurate of the two, and that is the question that decides whether the difference matters.
    if let Ok(d) = std::env::var("DUMP_KV_DIR") {
        for (c, a) in pm.cache_buffers.iter().zip(&arm_a) {
            std::fs::write(format!("{d}/{c}_pertok.bin"), a).unwrap();
            std::fs::write(format!("{d}/{c}_batched.bin"), slab(&pm, &arena, c, n)).unwrap();
        }
        let toks_bytes: Vec<u8> = toks.iter().flat_map(|t| t.to_le_bytes()).collect();
        std::fs::write(format!("{d}/tokens.bin"), &toks_bytes).unwrap();
        println!("dumped both arms' KV + tokens to {d}\n");
    }

    println!("{:<10} {:>12} {:>16}", "cache", "rel-L2", "differing/total");
    let mut first_bad: Option<(String, f64)> = None;
    for (c, a) in pm.cache_buffers.iter().zip(&arm_a) {
        let b = slab(&pm, &arena, c, n);
        let (r, ne, tot) = diff(&b, a);
        if r > 1e-6 && first_bad.is_none() {
            first_bad = Some((c.clone(), r));
        }
        let l: usize = c[1..c.find('_').unwrap()].parse().unwrap();
        if l < 3 || r > 1e-6 {
            println!("{:<10} {:>12.3e} {:>10}/{:<6}", c, r, ne, tot);
        }
    }
    match first_bad {
        None => println!("\n*** IDENTICAL -- batched prefill writes byte-for-byte what P steps write ***"),
        Some((c, r)) => println!("\nfirst divergence: {c} at rel-L2 {r:.3e}"),
    }
}
