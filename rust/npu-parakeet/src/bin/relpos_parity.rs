//! Standalone device parity for the resident relpos-MHA block (STEP=8), driven from Rust
//! via the same 3-BO ABI the encoder wiring will use (`run_dwconv6` == kernel(op, instr,
//! count, QUV, KPV, CTX)). Proves the Rust dispatch + stream packing == the validated
//! Python runner BEFORE touching encoder.rs::mhsa. Synth spread softmax (the case that
//! exposes the alignment bug). Gate: rel-L2 <= 0.08.
//!
//! Run (NPU quiesced):
//!   cargo run --features npu --bin relpos_parity
//! Env: RELPOS_XCLBIN / RELPOS_INSTS override the default T=172 build artifacts.

use ndarray::{s, Array2};
use npu_xrt::{pack_f32_to_bf16, Device, FLAG_CACHEABLE, FLAG_HOST_ONLY};

const DK: usize = 128;
const T: usize = 172;
const TQ: usize = 8;
const KB: usize = 43;

fn ceil_div(a: usize, b: usize) -> usize { (a + b - 1) / b }

// deterministic pseudo-random f32 in ~[-1,1] (no rand dep); splitmix64-ish per index.
fn rnd(seed: u64) -> f32 {
    let mut z = seed.wrapping_add(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^= z >> 31;
    ((z >> 11) as f32 / (1u64 << 53) as f32) * 2.0 - 1.0 // [-1,1)
}
fn fill(rows: usize, tag: u64) -> Array2<f32> {
    Array2::from_shape_fn((rows, DK), |(i, j)| rnd(tag.wrapping_mul(1_000_003) ^ ((i * DK + j) as u64)))
}

fn bf16_round(x: f32) -> f32 {
    let mut b = [0u16; 1];
    pack_f32_to_bf16(&[x], &mut b);
    f32::from_bits((b[0] as u32) << 16)
}

fn rel_shift_row(bd: &Array2<f32>, i: usize) -> Vec<f32> {
    // device convention: BD_sh[i,j] = BD[i, (T-1-i)+j], j in 0..T
    let base = T - 1 - i;
    (0..T).map(|j| bd[[i, base + j]]).collect()
}

fn host_ctx(qu: &Array2<f32>, qv: &Array2<f32>, k: &Array2<f32>, p: &Array2<f32>, v: &Array2<f32>) -> Array2<f32> {
    let inv = 1.0f32 / (DK as f32).sqrt();
    // bf16-round inputs (device sees bf16) so the reference tracks the device's real precision.
    let br = |m: &Array2<f32>| m.mapv(bf16_round);
    let (qu, qv, k, p, v) = (br(qu), br(qv), br(k), br(p), br(v));
    let ac = qu.dot(&k.t()); // [T,T]
    let bd = qv.dot(&p.t()); // [T,P]
    let mut ctx = Array2::<f32>::zeros((T, DK));
    for i in 0..T {
        let bdr = rel_shift_row(&bd, i);
        let mut s: Vec<f32> = (0..T).map(|j| (ac[[i, j]] + bdr[j]) * inv).collect();
        let mx = s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for e in s.iter_mut() { *e = (*e - mx).exp(); sum += *e; }
        for e in s.iter_mut() { *e /= sum; }
        for d in 0..DK {
            let mut acc = 0.0f32;
            for j in 0..T { acc += s[j] * v[[j, d]]; }
            ctx[[i, d]] = acc;
        }
    }
    ctx
}

fn pad_rows(x: &Array2<f32>, n: usize) -> Array2<f32> {
    let r = x.nrows();
    let mut o = Array2::<f32>::zeros((n, DK));
    o.slice_mut(s![0..r.min(n), ..]).assign(&x.slice(s![0..r.min(n), ..]));
    o
}

fn main() {
    let p_dim = 2 * T - 1;
    // --- synth, rescaled to a NON-DEGENERATE (spread) softmax (mirrors the runner default) ---
    let qu0 = fill(T, 1); let qv0 = fill(T, 2);
    let k = fill(T, 3); let p = fill(p_dim, 4); let v = fill(T, 5);
    // scale qu/qv so scores have unit std -> spread softmax (the bug-exposing regime).
    let ac = qu0.dot(&k.t());
    let bd = qv0.dot(&p.t());
    let mut svals = Vec::new();
    let inv = 1.0f32 / (DK as f32).sqrt();
    for i in 0..T { let bdr = rel_shift_row(&bd, i); for j in 0..T { svals.push((ac[[i, j]] + bdr[j]) * inv); } }
    let mean = svals.iter().sum::<f32>() / svals.len() as f32;
    let std = (svals.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / svals.len() as f32).sqrt() + 1e-6;
    let qu = qu0.mapv(|x| x / std);
    let qv = qv0.mapv(|x| x / std);

    let ctx_ref = host_ctx(&qu, &qv, &k, &p, &v);

    // --- STEP=8 stream packing (port of run_npu_relpos_rowtiled.py --stream) ---
    let n_qt = ceil_div(T, TQ);
    let n_kb = ceil_div(T, KB);
    let n_pb = ceil_div(p_dim, KB);
    let tp = n_kb * KB;
    let pp = n_pb * KB;

    // QUV: [qu_t0, qv_t0, qu_t1, qv_t1, ...], each tile TQ rows (ragged final zero-padded).
    let mut quv = Vec::<f32>::with_capacity(2 * n_qt * TQ * DK);
    for q in 0..n_qt {
        let q0 = q * TQ;
        let take = TQ.min(T - q0);
        let qu_t = pad_rows(&qu.slice(s![q0..q0 + take, ..]).to_owned(), TQ);
        let qv_t = pad_rows(&qv.slice(s![q0..q0 + take, ..]).to_owned(), TQ);
        quv.extend(qu_t.iter());
        quv.extend(qv_t.iter());
    }
    // KPV = k(pad Tp) | p(pad Pp) | V(pad Tp)
    let mut kpv = Vec::<f32>::with_capacity((tp + pp + tp) * DK);
    kpv.extend(pad_rows(&k, tp).iter());
    kpv.extend(pad_rows(&p, pp).iter());
    kpv.extend(pad_rows(&v, tp).iter());
    let ctx_rows = n_qt * TQ;

    // --- device ---
    let xclbin = std::env::var("RELPOS_XCLBIN").unwrap_or_else(|_|
        "mlir-aie/programming_examples/ml/relpos_mha/build/final.xclbin".into());
    let insts = std::env::var("RELPOS_INSTS").unwrap_or_else(|_|
        "mlir-aie/programming_examples/ml/relpos_mha/build/insts.bin".into());
    let instr_bytes = std::fs::read(&insts).expect("read insts.bin");
    let n_instr = instr_bytes.len() / 4;

    let dev = Device::open(0).expect("open NPU (quiesce voice first)");
    let kern = dev.load_kernel(&xclbin, None).expect("load relpos xclbin");
    let g = |i| kern.group_id(i).unwrap();

    let mut quv_bits = vec![0u16; quv.len()];
    let mut kpv_bits = vec![0u16; kpv.len()];
    pack_f32_to_bf16(&quv, &mut quv_bits);
    pack_f32_to_bf16(&kpv, &mut kpv_bits);
    let u16_bytes = |v: &[u16]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };

    let bo_instr = dev.alloc_bo(&kern, instr_bytes.len(), FLAG_CACHEABLE, g(1)).unwrap();
    let bo_quv = dev.alloc_bo(&kern, quv_bits.len() * 2, FLAG_HOST_ONLY, g(3)).unwrap();
    let bo_kpv = dev.alloc_bo(&kern, kpv_bits.len() * 2, FLAG_HOST_ONLY, g(4)).unwrap();
    let cbytes = ctx_rows * DK * 2;
    let bo_ctx = dev.alloc_bo(&kern, cbytes, FLAG_HOST_ONLY, g(5)).unwrap();

    bo_instr.write_bytes(&instr_bytes).unwrap(); bo_instr.sync_to_device().unwrap();
    bo_quv.write_bytes(&u16_bytes(&quv_bits)).unwrap(); bo_quv.sync_to_device().unwrap();
    bo_kpv.write_bytes(&u16_bytes(&kpv_bits)).unwrap(); bo_kpv.sync_to_device().unwrap();

    kern.run_dwconv6(3, &bo_instr, n_instr, &bo_quv, &bo_kpv, &bo_ctx).expect("dispatch");

    bo_ctx.sync_from_device().unwrap();
    let mut cb = vec![0u8; cbytes];
    bo_ctx.read_bytes(&mut cb).unwrap();
    let mut ctx_dev = Array2::<f32>::zeros((T, DK));
    for i in 0..T {
        for d in 0..DK {
            let off = (i * DK + d) * 2;
            let u = u16::from_le_bytes([cb[off], cb[off + 1]]);
            ctx_dev[[i, d]] = f32::from_bits((u as u32) << 16);
        }
    }

    let (mut num, mut den) = (0.0f64, 0.0f64);
    for i in 0..T { for d in 0..DK {
        let e = (ctx_dev[[i, d]] - ctx_ref[[i, d]]) as f64;
        num += e * e; den += (ctx_ref[[i, d]] as f64).powi(2);
    }}
    let rel_l2 = (num.sqrt() / (den.sqrt() + 1e-12)) as f32;
    println!("[relpos_parity] T={T} TQ={TQ} KB={KB}  n_instr={n_instr}");
    println!("[relpos_parity] ctx_dev[0,:4]={:?}", &ctx_dev.slice(s![0, 0..4]).to_vec());
    println!("[relpos_parity] ctx_ref[0,:4]={:?}", &ctx_ref.slice(s![0, 0..4]).to_vec());
    println!("[relpos_parity] rel-L2={rel_l2:.5e}  gate<=0.08  {}",
        if rel_l2 <= 0.08 { "PASS" } else { "FAIL" });
    std::process::exit(if rel_l2 <= 0.08 { 0 } else { 1 });
}
