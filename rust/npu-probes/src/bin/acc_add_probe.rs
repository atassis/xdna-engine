//! acc_add dtype-arm gate: mixed-precision-budget-sweep candidate #1's OTHER op ("resadd /
//! accadd activation stream f32 -> bf16" -- resadd got a full arm+gate already; accadd's own
//! -DACCADD_B_BF16 arm (narrow the PARTIAL `b` only; `a`/`out` stay f32) existed as
//! infrastructure but had never been taken to a device number).
//!
//! Different numerics regime from affine_cast's two arms on purpose: affine_cast's OUTPUT is
//! bf16 in every arm, so a coarse output absorbs input-rounding ULP disagreements (both its arms
//! came back bit-exact). acc_add's output stays f32 in EVERY arm -- there is no coarse output to
//! collapse a narrowed b's rounding into, so this is a genuinely different test, not expected to
//! inherit affine_cast's free result.
//!
//! Loads final_accadd_512x1024{,_bf16b}.xclbin, deterministic [512,1024] a/b, host golden built
//! from the SAME bf16-rounded b the device sees, best-of-N timing (NOT load-insensitive).
//!
//! NPU single-tenant -- stop npu-serve/npu-vox first. Run from the xdna-engine repo root.

use std::path::Path;
use std::time::Instant;

use npu_xrt::{Device, FLAG_CACHEABLE, FLAG_HOST_ONLY};

const ROWS: usize = 512;
const COLS: usize = 1024;
const LN_DIR: &str = "mlir-aie/programming_examples/ml/layernorm/build";

fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            ((z >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn bf16(x: f32) -> f32 {
    npu_xrt::bf16_bits_to_f32(npu_xrt::f32_to_bf16_bits(x))
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|x| npu_xrt::f32_to_bf16_bits(*x).to_le_bytes())
        .collect()
}

/// out = a + b, f32 throughout; b is bf16-rounded first when gating the bf16b arm (isolates the
/// arm from input quantization, same discipline as resadd/affine_cast).
fn golden(a: &[f32], b: &[f32], b_bf16: bool) -> Vec<f32> {
    a.iter()
        .zip(b.iter())
        .map(|(&av, &bv)| av + if b_bf16 { bf16(bv) } else { bv })
        .collect()
}

fn rel_l2(got: &[f32], want: &[f32]) -> f64 {
    let (mut num, mut den) = (0f64, 0f64);
    for (g, r) in got.iter().zip(want.iter()) {
        num += ((*g - *r) as f64).powi(2);
        den += (*r as f64).powi(2);
    }
    (num / den).sqrt()
}

struct ArmResult {
    rel_l2: f64,
    max_abs_delta: f32,
    n_nan: usize,
    bit_equal: usize,
    total: usize,
    ms_per_dispatch: f64,
}

fn run_arm(dev: &Device, tag: &str, b_bf16_wire: bool, a: &[f32], b: &[f32]) -> ArmResult {
    let ln = Path::new(LN_DIR);
    let xb = ln.join(format!("final_accadd_512x1024{tag}.xclbin"));
    let ib = ln.join(format!("insts_accadd_512x1024{tag}.txt"));
    let kern = dev
        .load_kernel(xb.to_str().unwrap(), None)
        .unwrap_or_else(|e| panic!("load {xb:?}: {e}"));
    let ibytes = std::fs::read(&ib).unwrap_or_else(|e| panic!("read {ib:?}: {e}"));
    let n_instr = ibytes.len() / 4;
    let g = |i| kern.group_id(i).unwrap();

    let instr = dev.alloc_bo(&kern, ibytes.len(), FLAG_CACHEABLE, g(1)).unwrap();
    instr.write_bytes(&ibytes).unwrap();
    instr.sync_to_device().unwrap();

    let b_elem_bytes = if b_bf16_wire { 2 } else { 4 };
    let bo_a = dev.alloc_bo(&kern, ROWS * COLS * 4, FLAG_HOST_ONLY, g(3)).unwrap();
    let bo_b = dev.alloc_bo(&kern, ROWS * COLS * b_elem_bytes, FLAG_HOST_ONLY, g(4)).unwrap();
    let bo_out = dev.alloc_bo(&kern, ROWS * COLS * 4, FLAG_HOST_ONLY, g(5)).unwrap(); // f32 out, every arm
    let bo_tmp = dev.alloc_bo(&kern, 8, FLAG_HOST_ONLY, g(6)).unwrap();
    let bo_tr = dev.alloc_bo(&kern, 4, FLAG_HOST_ONLY, g(7)).unwrap();

    bo_a.write_bytes(&f32_bytes(a)).unwrap();
    bo_a.sync_to_device().unwrap();
    if b_bf16_wire {
        bo_b.write_bytes(&bf16_bytes(b)).unwrap();
    } else {
        bo_b.write_bytes(&f32_bytes(b)).unwrap();
    }
    bo_b.sync_to_device().unwrap();

    for _ in 0..3 {
        kern.run_matmul8(3, &instr, n_instr, &bo_a, &bo_b, &bo_out, &bo_tmp, &bo_tr).unwrap();
    }
    kern.run_matmul8(3, &instr, n_instr, &bo_a, &bo_b, &bo_out, &bo_tmp, &bo_tr).unwrap();
    bo_out.sync_from_device().unwrap();
    let mut obytes = vec![0u8; ROWS * COLS * 4];
    bo_out.read_bytes(&mut obytes).unwrap();
    let got: Vec<f32> = obytes.chunks_exact(4).map(|w| f32::from_le_bytes([w[0], w[1], w[2], w[3]])).collect();

    let want = golden(a, b, b_bf16_wire);
    let rel_l2 = rel_l2(&got, &want);
    let mut max_abs_delta = 0f32;
    let mut n_nan = 0usize;
    let mut bit_equal = 0usize;
    for (g, r) in got.iter().zip(want.iter()) {
        if !g.is_finite() {
            n_nan += 1;
        }
        max_abs_delta = max_abs_delta.max((g - r).abs());
        if g.to_bits() == r.to_bits() {
            bit_equal += 1;
        }
    }

    let mut best = f64::INFINITY;
    for _ in 0..3 {
        let t = Instant::now();
        for _ in 0..200 {
            kern.run_matmul8(3, &instr, n_instr, &bo_a, &bo_b, &bo_out, &bo_tmp, &bo_tr).unwrap();
        }
        let ms = t.elapsed().as_secs_f64() * 1e3 / 200.0;
        best = best.min(ms);
    }

    ArmResult { rel_l2, max_abs_delta, n_nan, bit_equal, total: got.len(), ms_per_dispatch: best }
}

fn report(label: &str, r: &ArmResult) {
    println!(
        "{label}: rel_l2={:.3e} max|d|={:.3e} nan={} bit_equal={}/{} ms/dispatch={:.4}",
        r.rel_l2, r.max_abs_delta, r.n_nan, r.bit_equal, r.total, r.ms_per_dispatch
    );
}

fn main() {
    let dev = Device::open(0).expect("open NPU (stop npu-serve/npu-vox first)");
    let a = fill(ROWS * COLS, 4);
    let b = fill(ROWS * COLS, 5);

    println!("=== acc_add dtype-arm gate, [{ROWS},{COLS}] ===");
    let f32_r = run_arm(&dev, "", false, &a, &b);
    report("f32      ", &f32_r);

    let bf16b_r = run_arm(&dev, "_bf16b", true, &a, &b);
    report("bf16b (b)", &bf16b_r);

    let pct = (bf16b_r.ms_per_dispatch - f32_r.ms_per_dispatch) / f32_r.ms_per_dispatch * 100.0;
    println!("\ndispatch time delta vs f32: {:+.1}%", pct);
}
