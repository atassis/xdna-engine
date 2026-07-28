//! mha_decode parity probe (M1 Task 0) — proves the on-chip single-query MHA kernel
//! matches the host reference `attend_one` (rust/npu-engine/src/asr/whisper_decoder.rs).
//!
//! For each S in {1,30,64,200,448}: fixed-seed random q[768], K[S,768], V[S,768] (rounded
//! to bf16 so host and kernel see identical inputs), compute ctx on the host (a port of
//! attend_one) and on the NPU (the ONE runtime-S mha_decode xclbin), and report the per-S
//! rel-L2 of ctx. GATE: rel-L2 <= 0.08 for every S.
//!
//! RUNTIME S: a SINGLE xclbin (seq=448, fixed n_tiles=7) serves every S<=448. The per-tile
//! real key count is written by the host as an int32 into each tile's header (the 4 bytes
//! after the V-tile). >0 normal, <0 last non-empty (finalize), 0 empty (skip). No zero-pad
//! softmax poison.
//!
//! Kernel I/O (head-major, streaming/flash — see route_b_kernels/mha_decode/):
//!   q   : [12, 64]                       bf16   -> ABI slot 3 (A)
//!   kv  : [12, n_tiles, 2*TKV*64 + 2]    bf16   -> ABI slot 4 (B)
//!          (per tile: K-tile (TKV*64) | V-tile (TKV*64) | int32 s_in_tile (2 bf16), TKV=64)
//!   ctx : [12, 64]                       f32    -> ABI slot 5 (C), read back
//!   run_matmul8(3, instr, n, q, kv, ctx, tmp, trace).
//!
//! NPU is single-tenant — stop npu-asr.service / voxd.service BEFORE, restart AFTER.
//! Run from the worktree root (paths are relative to ".").
//!
//! Usage:  mha_decode_probe [S ...]   (defaults: 1 30 64 200 448)

use std::path::Path;

use npu_xrt::{f32_to_bf16_bits, Device, FLAG_CACHEABLE, FLAG_HOST_ONLY};

const D: usize = 768;
const NHEADS: usize = 12;
const HD: usize = 64;
const TKV: usize = 64; // keys per K/V tile; MUST match the kernel's MHA_TKV.
const S_MAX: usize = 448; // fixed max cache length -> fixed unrolled tile count.
const N_TILES: usize = (S_MAX + TKV - 1) / TKV; // 7; the ONE xclbin always streams this many.
const KV_TILE: usize = 2 * TKV * HD + 2; // K | V | int32 header (2 bf16 lanes).
const GATE: f32 = 0.08;

const MHA_DIR: &str = "mlir-aie/programming_examples/ml/mha_decode/build";

fn u16_bytes(v: &[u16]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

/// Round f32 -> bf16 -> f32 so the host reference sees exactly what the kernel sees.
fn bf16_round(x: f32) -> f32 {
    npu_xrt::bf16_bits_to_f32(f32_to_bf16_bits(x))
}

/// Host reference, ported verbatim from whisper_decoder.rs `attend_one`.
/// q: [768]; k_flat,v_flat: [S,768] row-major; returns ctx [768].
fn attend_one(q: &[f32], k_flat: &[f32], v_flat: &[f32], s: usize) -> Vec<f32> {
    let scale = 1.0 / (HD as f32).sqrt();
    let mut ctx = vec![0f32; D];
    for h in 0..NHEADS {
        let base = h * HD;
        let mut scores = vec![0f32; s];
        for j in 0..s {
            let krow = &k_flat[j * D + base..j * D + base + HD];
            let mut dot = 0f32;
            for d in 0..HD {
                dot += q[base + d] * krow[d];
            }
            scores[j] = dot * scale;
        }
        let mut maxv = f32::NEG_INFINITY;
        for &v in &scores {
            maxv = maxv.max(v);
        }
        let mut sum = 0f32;
        for v in scores.iter_mut() {
            *v = (*v - maxv).exp();
            sum += *v;
        }
        let inv = 1.0 / sum;
        for j in 0..s {
            let p = scores[j] * inv;
            let vrow = &v_flat[j * D + base..j * D + base + HD];
            for d in 0..HD {
                ctx[base + d] += p * vrow[d];
            }
        }
    }
    ctx
}

fn rel_l2(a: &[f32], b: &[f32]) -> f32 {
    let mut num = 0f64;
    let mut den = 0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (*x - *y) as f64;
        num += d * d;
        den += (*y as f64) * (*y as f64);
    }
    (num.sqrt() / den.sqrt().max(1e-30)) as f32
}

fn run_one(dev: &Device, s: usize) -> f32 {
    let root = Path::new(".");
    let dir = root.join(MHA_DIR);
    // ONE runtime-S xclbin (seq=448) for every S.
    let xclbin = dir.join(format!("final_mha_decode_{S_MAX}.xclbin"));
    let insts = dir.join(format!("insts_mha_decode_{S_MAX}.txt"));
    println!("\n[mha_decode_probe] S={s}");
    println!("  xclbin: {}", xclbin.display());

    let kern = dev
        .load_kernel(xclbin.to_str().unwrap(), None)
        .unwrap_or_else(|e| panic!("load {}: {e}", xclbin.display()));
    let ibytes =
        std::fs::read(&insts).unwrap_or_else(|e| panic!("read insts {}: {e}", insts.display()));
    let n_instr = ibytes.len() / 4;
    let g = |i| kern.group_id(i).unwrap();

    // ---- fixed-seed random q[768], K[S,768], V[S,768], bf16-rounded ----
    // Approx-Gaussian (sum of 3 uniforms, centered) so the q.K scores have std ~1 like
    // real attention — a near-uniform softmax (tiny scores) would be a degenerate test
    // that hides reduction/softmax bugs. bf16-round so host and kernel see the same input.
    let mut state: u32 = 0x1234_5678 ^ (s as u32).wrapping_mul(0x9E37_79B9);
    let mut u01 = || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (state >> 8) as f32 / (1u32 << 24) as f32 // [0,1)
    };
    let mut rnd = || bf16_round((u01() + u01() + u01() - 1.5) * 1.1547); // ~N(0,1)-ish, var~1
    let q: Vec<f32> = (0..D).map(|_| rnd()).collect();
    let k_flat: Vec<f32> = (0..s * D).map(|_| rnd()).collect();
    let v_flat: Vec<f32> = (0..s * D).map(|_| rnd()).collect();

    // ---- host reference ----
    let ctx_host = attend_one(&q, &k_flat, &v_flat, s);

    // ---- pack kernel inputs (head-major, tiled; FIXED N_TILES, runtime per-tile count) ----
    let n_real_tiles = s.div_ceil(TKV); // tiles that hold >=1 real key (<= N_TILES)

    // q: [12, 64] bf16
    let mut q_bf = vec![0u16; NHEADS * HD];
    for h in 0..NHEADS {
        for d in 0..HD {
            q_bf[h * HD + d] = f32_to_bf16_bits(q[h * HD + d]);
        }
    }

    // kv: [12, N_TILES, KV_TILE] bf16. Per (head, tile): K-tile (TKV rows) | V-tile (TKV rows)
    // | int32 s_in_tile header. >0 normal, <0 last non-empty (finalize), 0 empty (skipped).
    let kv_elems = NHEADS * N_TILES * KV_TILE;
    let mut kv_bf = vec![0u16; kv_elems];
    let hdr_off = 2 * TKV * HD; // header starts right after the V-tile (2 bf16 = 1 int32)
    for h in 0..NHEADS {
        let base = h * HD;
        for t in 0..N_TILES {
            let off = (h * N_TILES + t) * KV_TILE;
            let k_off = off; // K-tile
            let v_off = off + TKV * HD; // V-tile
            for r in 0..TKV {
                let key = t * TKV + r;
                if key >= s {
                    continue; // padding/empty row -> stays 0, never read by the kernel
                }
                for d in 0..HD {
                    kv_bf[k_off + r * HD + d] = f32_to_bf16_bits(k_flat[key * D + base + d]);
                    kv_bf[v_off + r * HD + d] = f32_to_bf16_bits(v_flat[key * D + base + d]);
                }
            }
            // per-tile runtime count (int32, bit-exact into 2 bf16 lanes).
            let s_in_tile: i32 = if t >= n_real_tiles {
                0 // empty tile
            } else {
                let real = (s - t * TKV).min(TKV) as i32; // real keys in this tile (1..=TKV)
                if t == n_real_tiles - 1 {
                    -real // last non-empty tile: finalize
                } else {
                    real
                }
            };
            let bytes = s_in_tile.to_le_bytes();
            kv_bf[off + hdr_off] = u16::from_le_bytes([bytes[0], bytes[1]]);
            kv_bf[off + hdr_off + 1] = u16::from_le_bytes([bytes[2], bytes[3]]);
        }
    }

    // ---- BOs ----
    let instr = dev.alloc_bo(&kern, ibytes.len(), FLAG_CACHEABLE, g(1)).unwrap();
    instr.write_bytes(&ibytes).unwrap();
    instr.sync_to_device().unwrap();

    let bo_q = dev.alloc_bo(&kern, q_bf.len() * 2, FLAG_HOST_ONLY, g(3)).unwrap();
    let bo_kv = dev.alloc_bo(&kern, kv_bf.len() * 2, FLAG_HOST_ONLY, g(4)).unwrap();
    let bo_ctx = dev.alloc_bo(&kern, NHEADS * HD * 4, FLAG_HOST_ONLY, g(5)).unwrap();
    let bo_tmp = dev.alloc_bo(&kern, 8, FLAG_HOST_ONLY, g(6)).unwrap();
    let bo_tr = dev.alloc_bo(&kern, 4, FLAG_HOST_ONLY, g(7)).unwrap();

    bo_q.write_bytes(u16_bytes(&q_bf)).unwrap();
    bo_q.sync_to_device().unwrap();
    bo_kv.write_bytes(u16_bytes(&kv_bf)).unwrap();
    bo_kv.sync_to_device().unwrap();

    // ---- dispatch ----
    kern.run_matmul8(3, &instr, n_instr, &bo_q, &bo_kv, &bo_ctx, &bo_tmp, &bo_tr)
        .expect("mha_decode dispatch failed");

    // ---- read back ctx [12,64] f32 ----
    bo_ctx.sync_from_device().unwrap();
    let mut cbuf = vec![0u8; NHEADS * HD * 4];
    bo_ctx.read_bytes(&mut cbuf).unwrap();
    let ctx_npu: &[f32] =
        unsafe { std::slice::from_raw_parts(cbuf.as_ptr() as *const f32, NHEADS * HD) };

    let r = rel_l2(ctx_npu, &ctx_host);
    let nz = ctx_npu.iter().filter(|&&x| x != 0.0).count();
    let finite = ctx_npu.iter().all(|x| x.is_finite());
    println!("  ctx rel-L2 vs host : {r:.5}   (gate <= {GATE})  [{nz}/{} nz, finite={finite}]", NHEADS * HD);
    println!("  host ctx[0..4] : {:?}", &ctx_host[..4]);
    println!("  npu  ctx[0..4] : {:?}", &ctx_npu[..4]);
    r
}

fn main() {
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .map(|a| a.parse().expect("S must be a usize"))
        .collect();
    let buckets = if args.is_empty() { vec![1usize, 30, 64, 200, 448] } else { args };

    let dev = Device::open(0).expect("open NPU (stop npu-asr.service/voxd.service first)");

    let mut worst = 0f32;
    let mut results = Vec::new();
    for &s in &buckets {
        let r = run_one(&dev, s);
        worst = worst.max(r);
        results.push((s, r));
    }

    println!("\n=== mha_decode parity summary (gate rel-L2 <= {GATE}) ===");
    let mut all_pass = true;
    for (s, r) in &results {
        let ok = *r <= GATE;
        all_pass &= ok;
        println!("  S={s:<4}  rel-L2={r:.5}  {}", if ok { "PASS" } else { "FAIL" });
    }
    println!("  worst rel-L2 = {worst:.5}  => {}", if all_pass { "GATE PASS" } else { "GATE FAIL" });
    std::process::exit(if all_pass { 0 } else { 1 });
}
