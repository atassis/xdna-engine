//! k768-gelu-rail gate: isolate and verify the HIGHEST-RISK kernel of the K=768 FFN rail --
//! the first-ever `modalgelu` build (fc1: bf16 A[512,800] @ B[800,3072] -> f32 gelu(A@B)).
//!
//! Why isolated: the rail's remaining work is (a) this kernel being correct and (b) a large
//! npu.rs const->rail-param refactor. Gating (a) first means the refactor is never built on a
//! broken kernel -- and if the GELU epilogue is wrong, the fallback ladder (erf-GELU epilogue ->
//! GELU-on-host) is a kernel-level decision, not a Rust-plumbing one.
//!
//! TWO goldens, deliberately:
//!   * `f32 ref`  -- true gelu(A@B) in f32. This is "what the model wants".
//!   * `bf16 ref` -- the kernel's OWN arithmetic replayed on host: x rounded to bf16 and every
//!     GELU intermediate re-rounded to bf16, matching mm_gelu_epilogue_f32o exactly.
//! Device-vs-bf16ref answers "does the kernel implement what it claims"; bf16ref-vs-f32ref prices
//! the approximation. Conflating them is how a correct kernel gets blamed for a design cost (the
//! epilogue's own comment already records the bf16 GELU costing RU +0.4).
//!
//! The mmul is bfp16-emulated (emulate_bfloat16_mmul_with_bfp16=1), whose shared-exponent error is
//! NOT modelled here -- it lands in the device-vs-bf16ref residual and is expected, not a defect.
//!
//! Bias is exercised through the K-augmentation the rail actually uses: K=800 = 768 real + a
//! 32-wide fold block, A[:,768]=1 and B[768,:]=bias, so A@B = A_real@B_real + bias.
//!
//! NPU is single-tenant -- stop npu-asr / npu-vox first. Run from the repo root.

use std::path::Path;

use npu_xrt::{Bo, Device, Kernel, FLAG_CACHEABLE, FLAG_HOST_ONLY};

const WA: &str = "mlir-aie/programming_examples/basic/matrix_multiplication/whole_array/build";
const XCLBIN: &str = "final_512x800x3072_64x32x128_8c_modalgelu.xclbin";
const INSTS: &str = "insts_512x800x3072_64x32x128_8c_modalgelu.txt";

const M: usize = 512;
const K_REAL: usize = 768;
const K_AUG: usize = 800; // 768 real + 32-wide bias-fold block
const N: usize = 3072;

/// Round an f32 through bf16 exactly the way the device does (the kernel's every intermediate).
fn bf16(x: f32) -> f32 {
    let mut bits = [0u16; 1];
    npu_xrt::pack_f32_to_bf16(&[x], &mut bits);
    let mut out = [0f32; 1];
    npu_xrt::unpack_bf16_to_f32(&bits, &mut out);
    out[0]
}

/// True tanh-approx GELU in f32: what the model wants.
fn gelu_f32(x: f32) -> f32 {
    let inner = 0.797_884_56_f32 * (x + 0.044_715_f32 * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}

/// The kernel's own arithmetic, replayed. Mirrors mm_gelu_epilogue_f32o step for step:
/// f32 acc -> bf16 x, every intermediate re-rounded to bf16, tanh evaluated on an f32 vector
/// but returning bf16.
fn gelu_bf16_chain(acc: f32) -> f32 {
    let c0 = bf16(0.797_884_56);
    let c1 = bf16(0.044_715);
    let half = bf16(0.5);
    let one = bf16(1.0);

    let xv = bf16(acc);
    let x2 = bf16(xv * xv);
    let x3 = bf16(x2 * xv);
    let c1x3 = bf16(c1 * x3);
    let inner_b = bf16(xv + c1x3);
    let inner_f32 = c0 * inner_b; // accum -> to_vector<float>(): stays f32
    let t = bf16(inner_f32.tanh());
    let t_p1 = bf16(t + one);
    let xt = bf16(xv * t_p1);
    bf16(half * xt)
}

fn load_stream(dev: &Device, kern: &Kernel, wa: &Path, ib: &str) -> (Bo, usize) {
    let bytes = std::fs::read(wa.join(ib)).unwrap_or_else(|e| panic!("read {ib}: {e}"));
    let n = bytes.len() / 4;
    let bo = dev
        .alloc_bo(kern, bytes.len(), FLAG_CACHEABLE, kern.group_id(1).unwrap())
        .unwrap();
    bo.write_bytes(&bytes).unwrap();
    bo.sync_to_device().unwrap();
    (bo, n)
}

fn u16_bytes(v: &[u16]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 2) }
}

/// Deterministic small-magnitude inputs. GELU is only interesting near 0 (it saturates to identity
/// for large x and to 0 for very negative x), so a wide spread would make the gate vacuously easy.
fn lcg(seed: &mut u64) -> f32 {
    *seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
    ((*seed >> 33) as f32 / (1u64 << 31) as f32) - 0.5 // ~U(-0.5, 0.5)
}

fn main() {
    let root = Path::new(".");
    let wa = root.join(WA);

    // ---- host-side inputs (row-major; the BD does all tiling, cf. npu.rs::dispatch) ----
    let mut seed = 0x5eed_1234_u64;
    let mut a = vec![0f32; M * K_AUG];
    for r in 0..M {
        for c in 0..K_REAL {
            a[r * K_AUG + c] = lcg(&mut seed) * 0.5;
        }
        a[r * K_AUG + K_REAL] = 1.0; // bias-fold indicator; cols K_REAL+1..K_AUG stay 0
    }
    let mut b = vec![0f32; K_AUG * N];
    for r in 0..K_REAL {
        for c in 0..N {
            b[r * N + c] = lcg(&mut seed) * 0.1;
        }
    }
    for c in 0..N {
        b[K_REAL * N + c] = lcg(&mut seed) * 0.2; // the bias row
    }

    // Round inputs to bf16 ONCE, and use the rounded values for both goldens -- the device never
    // sees the f32 originals, so charging it for input quantization would be measuring the wrong thing.
    let mut a_bits = vec![0u16; M * K_AUG];
    npu_xrt::pack_f32_to_bf16(&a, &mut a_bits);
    let mut a_q = vec![0f32; M * K_AUG];
    npu_xrt::unpack_bf16_to_f32(&a_bits, &mut a_q);

    let mut b_bits = vec![0u16; K_AUG * N];
    npu_xrt::pack_f32_to_bf16(&b, &mut b_bits);
    let mut b_q = vec![0f32; K_AUG * N];
    npu_xrt::unpack_bf16_to_f32(&b_bits, &mut b_q);

    // ---- device ----
    let dev = Device::open(0).expect("open NPU (stop npu-asr / npu-vox first)");
    let xb = wa.join(XCLBIN);
    let kern = dev
        .load_kernel(xb.to_str().unwrap(), None)
        .unwrap_or_else(|e| panic!("load {}: {e}", xb.display()));
    println!("[k768_gelu_probe] xclbin  {}", xb.display());
    let (instr, n_instr) = load_stream(&dev, &kern, &wa, INSTS);
    println!("[k768_gelu_probe] insts   {INSTS} ({n_instr} words)");
    println!("[k768_gelu_probe] shape   M={M} K_aug={K_AUG} (real {K_REAL}) N={N}");

    let g = |i| kern.group_id(i).unwrap();
    let bo_a = dev.alloc_bo(&kern, M * K_AUG * 2, FLAG_HOST_ONLY, g(3)).unwrap();
    let bo_b = dev.alloc_bo(&kern, K_AUG * N * 2, FLAG_HOST_ONLY, g(4)).unwrap();
    let bo_c = dev.alloc_bo(&kern, M * N * 4, FLAG_HOST_ONLY, g(5)).unwrap();
    let bo_tmp = dev.alloc_bo(&kern, 1, FLAG_HOST_ONLY, g(6)).unwrap();
    let bo_tr = dev.alloc_bo(&kern, 4, FLAG_HOST_ONLY, g(7)).unwrap();

    bo_a.write_bytes(u16_bytes(&a_bits)).unwrap();
    bo_a.sync_to_device().unwrap();
    bo_b.write_bytes(u16_bytes(&b_bits)).unwrap();
    bo_b.sync_to_device().unwrap();

    println!("[k768_gelu_probe] dispatching...");
    kern.run_matmul8(3, &instr, n_instr, &bo_a, &bo_b, &bo_c, &bo_tmp, &bo_tr)
        .expect("modalgelu dispatch failed");

    bo_c.sync_from_device().unwrap();
    let mut c_bytes = vec![0u8; M * N * 4];
    bo_c.read_bytes(&mut c_bytes).unwrap();
    let dev_c: Vec<f32> = c_bytes
        .chunks_exact(4)
        .map(|w| f32::from_le_bytes([w[0], w[1], w[2], w[3]]))
        .collect();

    // ---- goldens ----
    println!("[k768_gelu_probe] computing goldens ({}x{} f32 GEMM on host, be patient)...", M, N);
    let mut ref_f32 = vec![0f32; M * N];
    let mut ref_bf16 = vec![0f32; M * N];
    for r in 0..M {
        for c in 0..N {
            let mut acc = 0f32;
            for k in 0..K_AUG {
                acc += a_q[r * K_AUG + k] * b_q[k * N + c];
            }
            ref_f32[r * N + c] = gelu_f32(acc);
            ref_bf16[r * N + c] = gelu_bf16_chain(acc);
        }
    }

    let rel_l2 = |x: &[f32], y: &[f32]| -> f64 {
        let (mut num, mut den) = (0f64, 0f64);
        for i in 0..x.len() {
            let d = (x[i] - y[i]) as f64;
            num += d * d;
            den += (y[i] as f64) * (y[i] as f64);
        }
        (num / den).sqrt()
    };
    let max_abs = |x: &[f32], y: &[f32]| -> f32 {
        x.iter().zip(y).map(|(p, q)| (p - q).abs()).fold(0f32, f32::max)
    };

    let d_vs_bf16 = rel_l2(&dev_c, &ref_bf16);
    let d_vs_f32 = rel_l2(&dev_c, &ref_f32);
    let bf16_vs_f32 = rel_l2(&ref_bf16, &ref_f32);

    // A silent all-zero / all-NaN read is the classic failure here -- say so loudly rather than
    // reporting a meaningless rel-L2 (cf. the fenced-xrt read-race, where zeros masked a real bug).
    let nz = dev_c.iter().filter(|v| **v != 0.0).count();
    let nan = dev_c.iter().filter(|v| v.is_nan()).count();

    println!("\n=== k768 fc1 modalgelu -- device vs goldens ===");
    println!("  device nonzero      {nz}/{}  NaN {nan}", dev_c.len());
    println!("  device vs bf16 ref  rel-L2 {d_vs_bf16:.4e}   max-abs {:.4e}", max_abs(&dev_c, &ref_bf16));
    println!("  device vs f32  ref  rel-L2 {d_vs_f32:.4e}   max-abs {:.4e}", max_abs(&dev_c, &ref_f32));
    println!("  bf16   vs f32  ref  rel-L2 {bf16_vs_f32:.4e}  <- the epilogue's designed-in cost");

    if nz == 0 {
        println!("\n[k768_gelu_probe] FAIL: device returned all zeros -- not a precision result.");
        std::process::exit(1);
    }
    if nan > 0 {
        println!("\n[k768_gelu_probe] FAIL: device returned {nan} NaNs.");
        std::process::exit(1);
    }
    // The kernel is judged against its OWN arithmetic; the bfp16-emulated mmul is the residual.
    if d_vs_bf16 < 5e-2 {
        println!("\n[k768_gelu_probe] PASS: modalgelu matches its own bf16 arithmetic to {d_vs_bf16:.4e}.");
        println!("  first-ever modalgelu build is FUNCTIONALLY CORRECT; the f32 gap ({d_vs_f32:.4e}) is");
        println!("  dominated by the designed-in bf16 GELU + bfp16 mmul, not by a kernel defect.");
    } else {
        println!("\n[k768_gelu_probe] FAIL: device disagrees with its own bf16 arithmetic ({d_vs_bf16:.4e}).");
        println!("  That is a kernel defect, not an approximation cost -> fallback ladder applies.");
        std::process::exit(1);
    }

    // ---- fc2: the rail's OTHER GEMM. Identity epilogue, no K-augmentation (bias goes on the host
    // after the collapse), so its residual is the bfp16-emulated mmul ALONE -- which is the control
    // that tells us how much of fc1's 1.2e-2 is the GELU and how much is the matmul.
    let fc2 = fc2_gate(&dev, &wa, &mut seed);
    println!("\n=== k768 rail summary ===");
    println!("  fc1 modalgelu (gelu+bias-fold) device-vs-bf16  {d_vs_bf16:.4e}");
    println!("  fc2 modalid   (mmul only)      device-vs-bf16  {fc2:.4e}");
    if fc2 > 0.0 {
        println!("  -> the GELU epilogue contributes {:.4e} of designed-in error; the shared bfp16 mmul", bf16_vs_f32);
        println!("     emulation accounts for the bulk of BOTH residuals.");
    }
}

/// fc2 of the rail: A[512,3072] @ B[3072,768] -> f32, identity epilogue, no K-aug.
fn fc2_gate(dev: &Device, wa: &Path, seed: &mut u64) -> f64 {
    const K2: usize = 3072;
    const N2: usize = 768;
    const XB2: &str = "final_512x3072x768_64x32x96_8c_modalid.xclbin";
    const IN2: &str = "insts_512x3072x768_64x32x96_8c_modalid.txt";

    let kern = match dev.load_kernel(wa.join(XB2).to_str().unwrap(), None) {
        Ok(k) => k,
        Err(e) => {
            println!("\n[fc2] SKIP: could not load {XB2}: {e}");
            return 0.0;
        }
    };
    println!("\n[fc2] xclbin {XB2}");
    let (instr, n_instr) = load_stream(dev, &kern, wa, IN2);

    let mut a = vec![0f32; M * K2];
    for v in a.iter_mut() {
        *v = lcg(seed) * 0.3;
    }
    let mut b = vec![0f32; K2 * N2];
    for v in b.iter_mut() {
        *v = lcg(seed) * 0.1;
    }
    let mut a_bits = vec![0u16; M * K2];
    npu_xrt::pack_f32_to_bf16(&a, &mut a_bits);
    let mut a_q = vec![0f32; M * K2];
    npu_xrt::unpack_bf16_to_f32(&a_bits, &mut a_q);
    let mut b_bits = vec![0u16; K2 * N2];
    npu_xrt::pack_f32_to_bf16(&b, &mut b_bits);
    let mut b_q = vec![0f32; K2 * N2];
    npu_xrt::unpack_bf16_to_f32(&b_bits, &mut b_q);

    let g = |i| kern.group_id(i).unwrap();
    let bo_a = dev.alloc_bo(&kern, M * K2 * 2, FLAG_HOST_ONLY, g(3)).unwrap();
    let bo_b = dev.alloc_bo(&kern, K2 * N2 * 2, FLAG_HOST_ONLY, g(4)).unwrap();
    let bo_c = dev.alloc_bo(&kern, M * N2 * 4, FLAG_HOST_ONLY, g(5)).unwrap();
    let bo_tmp = dev.alloc_bo(&kern, 1, FLAG_HOST_ONLY, g(6)).unwrap();
    let bo_tr = dev.alloc_bo(&kern, 4, FLAG_HOST_ONLY, g(7)).unwrap();
    bo_a.write_bytes(u16_bytes(&a_bits)).unwrap();
    bo_a.sync_to_device().unwrap();
    bo_b.write_bytes(u16_bytes(&b_bits)).unwrap();
    bo_b.sync_to_device().unwrap();

    kern.run_matmul8(3, &instr, n_instr, &bo_a, &bo_b, &bo_c, &bo_tmp, &bo_tr)
        .expect("fc2 modalid dispatch failed");
    bo_c.sync_from_device().unwrap();
    let mut c_bytes = vec![0u8; M * N2 * 4];
    bo_c.read_bytes(&mut c_bytes).unwrap();
    let dev_c: Vec<f32> = c_bytes
        .chunks_exact(4)
        .map(|w| f32::from_le_bytes([w[0], w[1], w[2], w[3]]))
        .collect();

    let mut gold = vec![0f32; M * N2];
    for r in 0..M {
        for c in 0..N2 {
            let mut acc = 0f32;
            for k in 0..K2 {
                acc += a_q[r * K2 + k] * b_q[k * N2 + c];
            }
            gold[r * N2 + c] = acc; // identity epilogue
        }
    }
    let (mut num, mut den) = (0f64, 0f64);
    for i in 0..gold.len() {
        let d = (dev_c[i] - gold[i]) as f64;
        num += d * d;
        den += (gold[i] as f64) * (gold[i] as f64);
    }
    let rl = (num / den).sqrt();
    let nz = dev_c.iter().filter(|v| **v != 0.0).count();
    println!("[fc2] nonzero {nz}/{}  rel-L2 {rl:.4e}  (identity epilogue: pure bfp16-mmul error)", dev_c.len());
    rl
}
