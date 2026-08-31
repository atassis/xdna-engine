//! affine_cast dtype-arm gate: does bf16-narrowing an operand change the device answer, and by how
//! much? mixed-precision-budget-sweep candidate #2. Two independent arms, gated separately:
//!   x  (bf16 wire)  -- the ctxLN->affcast inter-op STREAM ("2 MB f32 today").
//!   gb (bf16 wire)  -- gamma|beta, a PER-OP PARAMETER: the same rounded value is reused
//!                      identically across all 512 rows, a different numerics risk (systematic
//!                      per-column bias) than x's independent per-element rounding -- gated on
//!                      its own rel-L2/bit-exactness, not assumed to inherit x's result.
//!
//! Loads each arm's xclbin (final_affcast_512x1024{,_bf16,_gb_bf16}.xclbin -- see
//! affine_cast.cc/affine_cast_iron.py), runs against a deterministic [512,1024] input +
//! [2048] gamma|beta, compares to a host golden computed from the SAME dtype-rounded inputs the
//! device sees (isolates the arm, not input quantization -- same discipline as residual_add's own
//! bf16 gate), and times each (best-of-N, NOT load-insensitive -- see the caller's own caveat on
//! this box being loaded).
//!
//! NPU single-tenant -- stop npu-serve/npu-vox first. Run from the xdna-engine repo root (paths
//! below resolve relative to it).

use std::path::Path;
use std::time::Instant;

use npu_xrt::{Device, FLAG_CACHEABLE, FLAG_HOST_ONLY};

const ROWS: usize = 512;
const COLS: usize = 1024;
const LN_DIR: &str = "mlir-aie/programming_examples/ml/layernorm/build";

/// splitmix64-seeded fill, deterministic and dependency-free -- same style as drain_dtype_probe.rs.
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

/// Host golden: (x*gamma + beta) narrowed to bf16, computed from the SAME dtype-rounded x/gamma/
/// beta the device arm under test sees.
fn golden(x: &[f32], gamma: &[f32], beta: &[f32], x_bf16: bool, gb_bf16: bool) -> Vec<f32> {
    (0..ROWS * COLS)
        .map(|i| {
            let c = i % COLS;
            let xi = if x_bf16 { bf16(x[i]) } else { x[i] };
            let gi = if gb_bf16 { bf16(gamma[c]) } else { gamma[c] };
            let bi = if gb_bf16 { bf16(beta[c]) } else { beta[c] };
            bf16(xi * gi + bi)
        })
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

#[allow(clippy::too_many_arguments)]
fn run_arm(
    dev: &Device,
    tag: &str,
    x_bf16_wire: bool,
    gb_bf16_wire: bool,
    x: &[f32],
    gamma: &[f32],
    beta: &[f32],
) -> ArmResult {
    let ln = Path::new(LN_DIR);
    let xb = ln.join(format!("final_affcast_512x1024{tag}.xclbin"));
    let ib = ln.join(format!("insts_affcast_512x1024{tag}.txt"));
    let kern = dev
        .load_kernel(xb.to_str().unwrap(), None)
        .unwrap_or_else(|e| panic!("load {xb:?}: {e}"));
    let ibytes = std::fs::read(&ib).unwrap_or_else(|e| panic!("read {ib:?}: {e}"));
    let n_instr = ibytes.len() / 4;
    let g = |i| kern.group_id(i).unwrap();

    let instr = dev.alloc_bo(&kern, ibytes.len(), FLAG_CACHEABLE, g(1)).unwrap();
    instr.write_bytes(&ibytes).unwrap();
    instr.sync_to_device().unwrap();

    let x_elem_bytes = if x_bf16_wire { 2 } else { 4 };
    let gb_elem_bytes = if gb_bf16_wire { 2 } else { 4 };
    let bo_x = dev.alloc_bo(&kern, ROWS * COLS * x_elem_bytes, FLAG_HOST_ONLY, g(3)).unwrap();
    let gb_len = 2 * COLS;
    let bo_gb = dev.alloc_bo(&kern, gb_len * gb_elem_bytes, FLAG_HOST_ONLY, g(4)).unwrap();
    let bo_out = dev.alloc_bo(&kern, ROWS * COLS * 2, FLAG_HOST_ONLY, g(5)).unwrap(); // bf16 out, every arm
    let bo_tmp = dev.alloc_bo(&kern, 8, FLAG_HOST_ONLY, g(6)).unwrap();
    let bo_tr = dev.alloc_bo(&kern, 4, FLAG_HOST_ONLY, g(7)).unwrap();

    if x_bf16_wire {
        bo_x.write_bytes(&bf16_bytes(x)).unwrap();
    } else {
        bo_x.write_bytes(&f32_bytes(x)).unwrap();
    }
    bo_x.sync_to_device().unwrap();
    let mut gb = Vec::with_capacity(gb_len);
    gb.extend_from_slice(gamma);
    gb.extend_from_slice(beta);
    if gb_bf16_wire {
        bo_gb.write_bytes(&bf16_bytes(&gb)).unwrap();
    } else {
        bo_gb.write_bytes(&f32_bytes(&gb)).unwrap();
    }
    bo_gb.sync_to_device().unwrap();

    // warm x3, then one measured correctness pass
    for _ in 0..3 {
        kern.run_matmul8(3, &instr, n_instr, &bo_x, &bo_gb, &bo_out, &bo_tmp, &bo_tr).unwrap();
    }
    kern.run_matmul8(3, &instr, n_instr, &bo_x, &bo_gb, &bo_out, &bo_tmp, &bo_tr).unwrap();
    bo_out.sync_from_device().unwrap();
    let mut obytes = vec![0u8; ROWS * COLS * 2];
    bo_out.read_bytes(&mut obytes).unwrap();
    let got: Vec<f32> = obytes
        .chunks_exact(2)
        .map(|w| npu_xrt::bf16_bits_to_f32(u16::from_le_bytes([w[0], w[1]])))
        .collect();

    let want = golden(x, gamma, beta, x_bf16_wire, gb_bf16_wire);
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

    // timing: dispatch-only, best of 3 x 200 (inputs already synced, matches ln_probe's
    // dispatch-only isolation -- excludes host marshal). NOT load-insensitive -- see caller.
    let mut best = f64::INFINITY;
    for _ in 0..3 {
        let t = Instant::now();
        for _ in 0..200 {
            kern.run_matmul8(3, &instr, n_instr, &bo_x, &bo_gb, &bo_out, &bo_tmp, &bo_tr).unwrap();
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
    let x = fill(ROWS * COLS, 1);
    let gamma = fill(COLS, 2);
    let beta = fill(COLS, 3);

    println!("=== affine_cast dtype-arm gate, [{ROWS},{COLS}] ===");
    let f32_r = run_arm(&dev, "", false, false, &x, &gamma, &beta);
    report("f32       ", &f32_r);

    let x_bf16_r = run_arm(&dev, "_bf16", true, false, &x, &gamma, &beta);
    report("bf16 x    ", &x_bf16_r);

    let gb_bf16_r = run_arm(&dev, "_gb_bf16", false, true, &x, &gamma, &beta);
    report("bf16 gb   ", &gb_bf16_r);

    let pct = |a: &ArmResult| (a.ms_per_dispatch - f32_r.ms_per_dispatch) / f32_r.ms_per_dispatch * 100.0;
    println!("\ndispatch time delta vs f32: x={:+.1}%  gb={:+.1}%", pct(&x_bf16_r), pct(&gb_bf16_r));
}
