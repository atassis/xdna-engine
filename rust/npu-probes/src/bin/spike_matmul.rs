//! Milestone 1 (docs/12): run ONE whole_array bf16->f32 matmul on the XDNA2 NPU from Rust via the
//! XRT shim, and check the result against an in-Rust f32 reference (bf16 in -> f32 accumulate) — the
//! same self-consistency check `scripts/run_npu_matmul_wholearray.py` does in Python. A PASS here
//! de-risks the entire XRT-from-Rust approach.
//!
//! The NPU is single-tenant — stop flm-asr.service/voxd.service before running.
//!
//! Usage (from repo root):
//!   rust/target/release/spike_matmul [M K N]
//! defaults: 512 768 768. Expects the xclbin+insts built by scripts/build_kernels.sh.

use std::time::Instant;

use npu_xrt::{bf16_bits_to_f32, f32_to_bf16_bits, Device, FLAG_CACHEABLE, FLAG_HOST_ONLY};

const WA_DIR: &str =
    "mlir-aie/programming_examples/basic/matrix_multiplication/whole_array/build";

/// Tiny deterministic LCG -> f32 in [-1, 1], so the run is reproducible without a deps.
fn fill_uniform(buf: &mut [f32], seed: u64) {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    for x in buf.iter_mut() {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u = ((s >> 33) as u32) as f32 / (u32::MAX as f32); // [0,1)
        *x = 2.0 * u - 1.0;
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let m: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(512);
    let k: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(768);
    let n: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(768);

    let suffix = format!("{m}x{k}x{n}_32x32x32_8c");
    let xclbin = format!("{WA_DIR}/final_{suffix}.xclbin");
    let insts = format!("{WA_DIR}/insts_{suffix}.txt");
    println!("[cfg] matmul A[{m},{k}] @ B[{k},{n}] -> C[{m},{n}] bf16->f32 (whole_array, 8 cols)");
    println!("[cfg] xclbin={xclbin}");
    println!("[cfg] insts ={insts}");

    // --- host data: A,B as bf16 (raw u16, row-major, no packing), reference in f32 ---
    let mut af = vec![0f32; m * k];
    let mut bf = vec![0f32; k * n];
    fill_uniform(&mut af, 1);
    fill_uniform(&mut bf, 2);
    // round to bf16 (what actually lands on the NPU) and keep the expanded f32 for the reference
    let a_bits: Vec<u16> = af.iter().map(|&x| f32_to_bf16_bits(x)).collect();
    let b_bits: Vec<u16> = bf.iter().map(|&x| f32_to_bf16_bits(x)).collect();
    let a_f32: Vec<f32> = a_bits.iter().map(|&b| bf16_bits_to_f32(b)).collect();
    let b_f32: Vec<f32> = b_bits.iter().map(|&b| bf16_bits_to_f32(b)).collect();

    // reference: C = A.f32 @ B.f32 (bf16 in, f32 accumulate) — matches the NPU convention
    let t_ref = Instant::now();
    let mut cref = vec![0f32; m * n];
    for i in 0..m {
        let arow = &a_f32[i * k..(i + 1) * k];
        let crow = &mut cref[i * n..(i + 1) * n];
        for kk in 0..k {
            let a = arow[kk];
            let brow = &b_f32[kk * n..(kk + 1) * n];
            for j in 0..n {
                crow[j] += a * brow[j];
            }
        }
    }
    println!("[ref] computed in {:.2} s", t_ref.elapsed().as_secs_f64());

    // --- instr words (binary uint32 in the .txt) ---
    let instr_bytes = match std::fs::read(&insts) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("FAIL: cannot read insts {insts}: {e} (build with scripts/build_kernels.sh)");
            std::process::exit(2);
        }
    };
    let instr_words = instr_bytes.len() / 4;
    println!("[artifacts] instr_words={instr_words}");

    // --- drive the NPU ---
    let run = || -> Result<Vec<f32>, String> {
        let dev = Device::open(0)?;
        let kern = dev.load_kernel(&xclbin, None)?;

        // group ids: instr=1, A=3, B=4, C=5, tmp=6, trace=7 (arg0=opcode, arg2=count are scalars)
        let g_instr = kern.group_id(1)?;
        let g_a = kern.group_id(3)?;
        let g_b = kern.group_id(4)?;
        let g_c = kern.group_id(5)?;
        let g_tmp = kern.group_id(6)?;
        let g_tr = kern.group_id(7)?;

        let bo_instr = dev.alloc_bo(&kern, instr_bytes.len(), FLAG_CACHEABLE, g_instr)?;
        let bo_a = dev.alloc_bo(&kern, a_bits.len() * 2, FLAG_HOST_ONLY, g_a)?;
        let bo_b = dev.alloc_bo(&kern, b_bits.len() * 2, FLAG_HOST_ONLY, g_b)?;
        let bo_c = dev.alloc_bo(&kern, m * n * 4, FLAG_HOST_ONLY, g_c)?;
        let bo_tmp = dev.alloc_bo(&kern, 1, FLAG_HOST_ONLY, g_tmp)?;
        let bo_tr = dev.alloc_bo(&kern, 4, FLAG_HOST_ONLY, g_tr)?;

        bo_instr.write_bytes(&instr_bytes)?;
        bo_instr.sync_to_device()?;
        bo_a.write_bytes(u16_as_bytes(&a_bits))?;
        bo_a.sync_to_device()?;
        bo_b.write_bytes(u16_as_bytes(&b_bits))?;
        bo_b.sync_to_device()?;

        // warmup
        kern.run_matmul8(3, &bo_instr, instr_words, &bo_a, &bo_b, &bo_c, &bo_tmp, &bo_tr)?;

        let iters = 50;
        let t0 = Instant::now();
        for _ in 0..iters {
            kern.run_matmul8(3, &bo_instr, instr_words, &bo_a, &bo_b, &bo_c, &bo_tmp, &bo_tr)?;
        }
        let per = t0.elapsed().as_secs_f64() / iters as f64;
        let gflops = 2.0 * m as f64 * k as f64 * n as f64 / per / 1e9;
        println!("[run] device time/iter: {:.3} ms -> {:.1} GFLOP/s", per * 1e3, gflops);

        bo_c.sync_from_device()?;
        let mut cbytes = vec![0u8; m * n * 4];
        bo_c.read_bytes(&mut cbytes)?;
        Ok(bytes_as_f32(&cbytes))
    };

    let c = match run() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("FAIL: NPU dispatch error: {e}");
            std::process::exit(1);
        }
    };

    // --- compare ---
    let mut max_abs = 0f32;
    let mut ref_max = 0f32;
    for i in 0..m * n {
        max_abs = max_abs.max((c[i] - cref[i]).abs());
        ref_max = ref_max.max(cref[i].abs());
    }
    let rel = max_abs / (ref_max + 1e-9);
    let has_nan = c.iter().any(|x| x.is_nan());
    println!("[cmp] C[0,..4]   = {:?}", &c[0..4.min(n)]);
    println!("[cmp] ref[0,..4] = {:?}", &cref[0..4.min(n)]);
    println!("[cmp] max|Δ|={max_abs:.4e}  max_rel={rel:.3e}  nan={has_nan}");
    let ok = rel < 0.03 && !has_nan;
    println!(
        "[result] bf16 whole_array matmul {m}x{k}x{n} on NPU from Rust: {}",
        if ok { "PASS" } else { "FAIL" }
    );
    std::process::exit(if ok { 0 } else { 1 });
}

fn u16_as_bytes(v: &[u16]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 2) }
}

fn bytes_as_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}
