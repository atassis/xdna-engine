//! On-device parity for the 8-head relpos CONVEYOR via the REAL encoder code path
//! (`NpuMatmul::relpos_mha_conveyor`): proves the Rust belt packing + `run_mha` dispatch +
//! de-interleave == a host relpos-MHA reference. Synthetic per-head q/k/v/p, spread softmax.
//!
//! The conveyor kernel has NO key-mask, so it is exact only at full keys (T == CONV_BUILT_T=176).
//! Default T=176 (PASS expected). Set `CONV_PARITY_T=100` to PROBE the masking gap (expected to
//! FAIL: ~76 zero-score pad keys pollute the softmax denominator).
//!
//! Run (NPU quiesced, from the repo root):
//!   NPU_XCLBIN_ROOT=$PWD cargo run --features npu --release --bin conveyor_parity
//! Needs artifacts/conveyor/single/{final.xclbin,insts.bin} (scripts/conveyor_prebuild.sh) and the
//! resident modal xclbin NpuMatmul::open loads.

use ndarray::{s, Array2, Array3};
use npu_parakeet::npu::NpuMatmul;
use npu_parakeet::ops::rel_shift;
use npu_xrt::pack_f32_to_bf16;
use std::path::Path;

const DK: usize = 128;
const H: usize = 8;
const D: usize = H * DK;
const BUILT_T: usize = 176;

// deterministic pseudo-random f32 in ~[-1,1) (no rand dep).
fn rnd(seed: u64) -> f32 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    ((z >> 11) as f32 / (1u64 << 53) as f32) * 2.0 - 1.0
}
fn synth(rows: usize, cols: usize, tag: u64) -> Array2<f32> {
    Array2::from_shape_fn((rows, cols), |(i, j)| rnd(tag.wrapping_mul(1_000_003) ^ ((i * cols + j) as u64)))
}
fn bf16(x: f32) -> f32 {
    let mut b = [0u16; 1];
    pack_f32_to_bf16(&[x], &mut b);
    f32::from_bits((b[0] as u32) << 16)
}
fn bf(m: &Array2<f32>) -> Array2<f32> { m.mapv(bf16) }

fn main() {
    let t: usize = std::env::var("CONV_PARITY_T").ok().and_then(|s| s.parse().ok()).unwrap_or(BUILT_T);
    assert!(t <= BUILT_T, "T={t} exceeds CONV_BUILT_T={BUILT_T}");
    let p_dim = 2 * t - 1;

    // --- synthetic full-model tensors ---
    let mut q = synth(t, D, 1);
    let k = synth(t, D, 2);
    let v = synth(t, D, 3);
    let pm = synth(p_dim, D, 4);
    let ubias = synth(H, DK, 5).mapv(|x| 0.1 * x);
    let vbias = synth(H, DK, 6).mapv(|x| 0.1 * x);

    // scale q so head-0 scores have unit std -> spread (non-degenerate) softmax, not one-hot.
    let inv = 1.0f32 / (DK as f32).sqrt();
    {
        let qh = q.slice(s![.., 0..DK]).to_owned();
        let kh = k.slice(s![.., 0..DK]).to_owned();
        let ph = pm.slice(s![.., 0..DK]).to_owned();
        let ac = qh.dot(&kh.t());
        let bd = qh.dot(&ph.t());
        let mut sv = Vec::new();
        for i in 0..t { let base = t - 1 - i; for j in 0..t { sv.push((ac[[i, j]] + bd[[i, base + j]]) * inv); } }
        let mean = sv.iter().sum::<f32>() / sv.len() as f32;
        let std = (sv.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / sv.len() as f32).sqrt() + 1e-6;
        q.mapv_inplace(|x| x / std);
    }

    // --- host reference: mirrors relpos_mha_conveyor's host math (f32 dots on bf16-rounded operands,
    //     ops::rel_shift for BD_shifted, bf16 belt carriage), softmax over the REAL t keys only. ---
    let mut bd_all = Array3::<f32>::zeros((H, t, p_dim));
    let mut qu_bf = vec![Array2::<f32>::zeros((t, DK)); H];
    for h in 0..H {
        let col = h * DK;
        let qh = q.slice(s![.., col..col + DK]).to_owned();
        let ph = pm.slice(s![.., col..col + DK]).to_owned();
        let mut qu = qh.clone();
        let mut qv = qh.clone();
        for i in 0..t {
            for c in 0..DK { qu[[i, c]] += ubias[[h, c]]; qv[[i, c]] += vbias[[h, c]]; }
        }
        qu_bf[h] = bf(&qu);
        bd_all.slice_mut(s![h, .., ..]).assign(&qv.dot(&ph.t())); // f32, matches npu.rs
    }
    let bd_sh = rel_shift(&bd_all, t); // [H,t,t]
    let mut ctx_ref = Array2::<f32>::zeros((t, D));
    for h in 0..H {
        let col = h * DK;
        let kh = bf(&k.slice(s![.., col..col + DK]).to_owned());
        let vh = bf(&v.slice(s![.., col..col + DK]).to_owned());
        let ac = qu_bf[h].dot(&kh.t()); // [t,t]
        for i in 0..t {
            let mut sc: Vec<f32> = (0..t).map(|j| (ac[[i, j]] + bf16(bd_sh[[h, i, j]])) * inv).collect();
            let mx = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for e in sc.iter_mut() { *e = (*e - mx).exp(); sum += *e; }
            for e in sc.iter_mut() { *e /= sum; }
            for d in 0..DK {
                let mut acc = 0.0f32;
                for j in 0..t { acc += sc[j] * vh[[j, d]]; }
                ctx_ref[[i, col + d]] = acc;
            }
        }
    }

    // --- device via the real encoder code path ---
    let root = std::env::var("NPU_XCLBIN_ROOT").unwrap_or_else(|_| ".".into());
    let npu = NpuMatmul::open(Path::new(&root));
    let ctx_dev = npu.relpos_mha_conveyor(&q, &k, &v, &pm, &ubias, &vbias, H);
    assert_eq!(ctx_dev.dim(), (t, D), "conveyor returned unexpected ctx shape");

    // --- rel-L2 total + per head ---
    let l2 = |a: &Array2<f32>, b: &Array2<f32>, c0: usize, c1: usize| -> f32 {
        let (mut n, mut den) = (0.0f64, 0.0f64);
        for i in 0..t { for d in c0..c1 {
            let e = (a[[i, d]] - b[[i, d]]) as f64;
            n += e * e; den += (b[[i, d]] as f64).powi(2);
        }}
        (n.sqrt() / (den.sqrt() + 1e-12)) as f32
    };
    let rel = l2(&ctx_dev, &ctx_ref, 0, D);
    let per_head: Vec<f32> = (0..H).map(|h| l2(&ctx_dev, &ctx_ref, h * DK, h * DK + DK)).collect();

    println!("[conveyor_parity] T={t} (BUILT_T={BUILT_T}) H={H}  full-key={}", t == BUILT_T);
    println!("[conveyor_parity] per-head rel-L2: {}", per_head.iter().map(|x| format!("{x:.3e}")).collect::<Vec<_>>().join(" "));
    println!("[conveyor_parity] ctx_dev[0,:3]={:?}  ctx_ref[0,:3]={:?}",
        &ctx_dev.slice(s![0, 0..3]).to_vec(), &ctx_ref.slice(s![0, 0..3]).to_vec());
    let gate = 5e-3f32;
    println!("[conveyor_parity] TOTAL rel-L2={rel:.5e}  gate<={gate:.0e}  {}",
        if rel <= gate { "PASS" } else { "FAIL (expected when T<BUILT_T: no key-mask)" });
    std::process::exit(if rel <= gate { 0 } else { 1 });
}
