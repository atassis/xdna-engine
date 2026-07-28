//! ctxLN unit gate: does the on-NPU f32 two-pass LayerNorm match the host reference?
//!
//! Loads the ctxLN xclbin (final_ctxln_512x768, the shipped shape), runs a [512,768] f32 input, compares the
//! NPU output to the host `layer_norm_normalize` (the exact reference the encoder rel gate uses).
//! Gate: rel ≤ 1e-3 (f32 two-pass vs f32 scalar should be tight; differs only in reduction order
//! + aie::invsqrt vs 1/sqrt). De-risks the kernel BEFORE encoder integration.
//!
//! ABI (scripts/run_npu_layernorm.py): args 1=instr, 3=in, 4=out, 5=tmp, 6=ctrl, 7=trace; dispatch
//! run_matmul8(3, instr, n, in, out, dummy_c, dummy_tmp, dummy_tr) — OUTPUT is the b/out slot.
//!
//! NPU single-tenant — stop npu-asr/voxd first. Run from MAIN repo root (WA/LN path resolves).

use std::path::Path;

use npu_xrt::{Device, FLAG_CACHEABLE, FLAG_HOST_ONLY};

const LN_DIR: &str = "mlir-aie/programming_examples/ml/layernorm/build";
// 512 = the shipped ctxLN shape (= PAD_M; what build_kernels.sh produces and ctx_ln.rs loads).
const ROWS: usize = 512;
const COLS: usize = 768;
const EPS: f32 = 1e-5;

/// host reference (== npu-asr-host layer_norm_normalize): per-row two-pass centered LN over COLS.
fn host_ln_normalize(x: &[f32]) -> Vec<f32> {
    let mut out = vec![0f32; ROWS * COLS];
    for r in 0..ROWS {
        let row = &x[r * COLS..(r + 1) * COLS];
        let mean = row.iter().sum::<f32>() / COLS as f32;
        let mut vs = 0f32;
        for &v in row {
            let c = v - mean;
            vs += c * c;
        }
        let inv = 1.0 / (vs / COLS as f32 + EPS).sqrt();
        for j in 0..COLS {
            out[r * COLS + j] = (row[j] - mean) * inv;
        }
    }
    out
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn main() {
    let root = Path::new(".");
    let ln = root.join(LN_DIR);
    let dev = Device::open(0).expect("open NPU (stop npu-asr/voxd first)");

    let xb = ln.join(format!("final_ctxln_{ROWS}x{COLS}.xclbin"));
    let ib = ln.join(format!("insts_ctxln_{ROWS}x{COLS}.txt"));
    let kern = dev.load_kernel(xb.to_str().unwrap(), None).expect("load ctxLN xclbin");
    let ibytes = std::fs::read(&ib).expect("read insts");
    let n_instr = ibytes.len() / 4;
    let g = |i| kern.group_id(i).unwrap();

    let instr = dev.alloc_bo(&kern, ibytes.len(), FLAG_CACHEABLE, g(1)).unwrap();
    instr.write_bytes(&ibytes).unwrap();
    instr.sync_to_device().unwrap();

    // in=gid3, out=gid4, then dummy tmp/ctrl/trace at 5/6/7 (placeholders, must be live).
    let bo_in = dev.alloc_bo(&kern, ROWS * COLS * 4, FLAG_HOST_ONLY, g(3)).unwrap();
    let bo_out = dev.alloc_bo(&kern, ROWS * COLS * 4, FLAG_HOST_ONLY, g(4)).unwrap();
    let bo_c = dev.alloc_bo(&kern, 64, FLAG_HOST_ONLY, g(5)).unwrap();
    let bo_tmp = dev.alloc_bo(&kern, 8, FLAG_HOST_ONLY, g(6)).unwrap();
    let bo_tr = dev.alloc_bo(&kern, 4, FLAG_HOST_ONLY, g(7)).unwrap();

    // deterministic, non-trivial input: per-row varied magnitudes so mean/var are meaningful.
    let x: Vec<f32> = (0..ROWS * COLS)
        .map(|i| {
            let r = (i / COLS) as f32;
            let c = (i % COLS) as f32;
            (c * 0.013 + r * 0.007).sin() * 2.0 + ((i * 7 + 3) % 101) as f32 / 101.0 - 0.5
        })
        .collect();

    bo_in.write_bytes(&f32_bytes(&x)).unwrap();
    bo_in.sync_to_device().unwrap();

    // warm + run
    for _ in 0..3 {
        kern.run_matmul8(3, &instr, n_instr, &bo_in, &bo_out, &bo_c, &bo_tmp, &bo_tr).unwrap();
    }
    kern.run_matmul8(3, &instr, n_instr, &bo_in, &bo_out, &bo_c, &bo_tmp, &bo_tr).unwrap();
    bo_out.sync_from_device().unwrap();
    let mut obytes = vec![0u8; ROWS * COLS * 4];
    bo_out.read_bytes(&mut obytes).unwrap();
    let got: Vec<f32> = obytes.chunks_exact(4).map(|w| f32::from_le_bytes([w[0], w[1], w[2], w[3]])).collect();

    // --- SWITCH-FREE cost: time the full per-call LN (write+sync+dispatch+sync+read), ONE hwctx,
    // NO context switch. This is the absolute floor co-residency could ever reach. Compare to host
    // LN = 23.5 ms / 96 calls = 0.245 ms/call. The encoder makes 96 LN calls/inference.
    use std::time::Instant;
    let bench = |n: usize| -> f64 {
        let t = Instant::now();
        for _ in 0..n {
            bo_in.write_bytes(&f32_bytes(&x)).unwrap();
            bo_in.sync_to_device().unwrap();
            kern.run_matmul8(3, &instr, n_instr, &bo_in, &bo_out, &bo_c, &bo_tmp, &bo_tr).unwrap();
            bo_out.sync_from_device().unwrap();
            let mut o = vec![0u8; ROWS * COLS * 4];
            bo_out.read_bytes(&mut o).unwrap();
        }
        t.elapsed().as_secs_f64() * 1e3 / n as f64
    };
    let per_call = bench(200);
    // dispatch+compute ONLY (reuse already-synced bo_in, don't marshal): isolates the per-dispatch
    // floor from the 3 MB f32 round-trip. run_matmul8 is synchronous (blocks until the NPU finishes).
    let dispatch_only = {
        let t = Instant::now();
        for _ in 0..200 {
            kern.run_matmul8(3, &instr, n_instr, &bo_in, &bo_out, &bo_c, &bo_tmp, &bo_tr).unwrap();
        }
        t.elapsed().as_secs_f64() * 1e3 / 200.0
    };
    let marshal = per_call - dispatch_only;
    println!("\n=== SWITCH-FREE per-call cost (one hwctx, NO context switch) ===");
    println!("  ctxLN full call (write+dispatch+read, [{ROWS},{COLS}] f32) = {per_call:.4} ms");
    println!("    - dispatch+compute only (no host marshal)               = {dispatch_only:.4} ms");
    println!("    - host marshal (write 1.5MB + read 1.5MB f32)           = {marshal:.4} ms");
    println!("  projected 96 LN calls/inference, switch-free:");
    println!("    full (as-implemented, per-op marshal) = {:.1} ms", per_call * 96.0);
    println!("    dispatch-only (resident-activation hypothetical) = {:.1} ms", dispatch_only * 96.0);
    println!("  host LN for comparison: 23.5 ms / 96 calls = 0.245 ms/call");
    println!("  => full {:.1}x host; dispatch-only {:.1}x host.", per_call / 0.245, dispatch_only / 0.245);

    let refr = host_ln_normalize(&x);
    let (mut maxd, mut maxabs) = (0f32, 0f32);
    for (g, r) in got.iter().zip(refr.iter()) {
        maxd = maxd.max((g - r).abs());
        maxabs = maxabs.max(r.abs());
    }
    let rel = maxd / (maxabs + 1e-9);

    println!("=== ctxLN unit gate: NPU f32 two-pass LN vs host layer_norm_normalize ===");
    println!("  shape [{ROWS},{COLS}], eps {EPS:.0e}");
    println!("  row0[0..4]  got={:?}", &got[..4]);
    println!("  row0[0..4]  ref={:?}", &refr[..4]);
    println!("  max|Δ| = {maxd:.3e}   max|ref| = {maxabs:.3e}   rel = {rel:.3e}");
    if rel <= 1e-3 && maxabs > 1e-3 {
        println!("  ✅ PASS: ctxLN matches host LN (rel {rel:.3e} ≤ 1e-3).");
    } else {
        println!("  ❌ FAIL: rel {rel:.3e} (> 1e-3) or trivial output.");
    }

    // === numerical robustness: adversarial distributions (LN's hazard is magnitude/variance — the
    // exact regime the bf16 E[x²]-mean² kernel failed on, docs/06 §3). f32 two-pass should survive. ===
    let run_ln = |inp: &[f32]| -> Vec<f32> {
        bo_in.write_bytes(&f32_bytes(inp)).unwrap();
        bo_in.sync_to_device().unwrap();
        kern.run_matmul8(3, &instr, n_instr, &bo_in, &bo_out, &bo_c, &bo_tmp, &bo_tr).unwrap();
        bo_out.sync_from_device().unwrap();
        let mut o = vec![0u8; ROWS * COLS * 4];
        bo_out.read_bytes(&mut o).unwrap();
        o.chunks_exact(4).map(|w| f32::from_le_bytes([w[0], w[1], w[2], w[3]])).collect()
    };
    let cmp = |label: &str, inp: &[f32]| {
        let got = run_ln(inp);
        let refr = host_ln_normalize(inp);
        let (mut md, mut ma, mut finite) = (0f32, 0f32, true);
        for (g, r) in got.iter().zip(refr.iter()) {
            if !g.is_finite() {
                finite = false;
            }
            md = md.max((g - r).abs());
            ma = ma.max(r.abs());
        }
        let ok = finite && md / (ma + 1e-9) <= 1e-3;
        println!("  {:<26} max|Δ|={md:.3e} rel={:.3e} finite={finite}  {}", label, md / (ma + 1e-9), if ok { "✅" } else { "❌" });
    };
    println!("\n=== numerical robustness (adversarial input distributions vs host) ===");
    cmp("constant rows (var=0)", &vec![5.0; ROWS * COLS]); // inv=1/sqrt(eps); host out=0 — div-by-~0
    cmp("large magnitude (±19200)", &(0..ROWS * COLS).map(|i| ((i % COLS) as f32 - 384.0) * 50.0).collect::<Vec<_>>());
    cmp("tiny magnitude (±0.038)", &(0..ROWS * COLS).map(|i| ((i % COLS) as f32 - 384.0) * 1e-4).collect::<Vec<_>>());
    let mut adv_out = vec![0.1f32; ROWS * COLS];
    for r in 0..ROWS {
        adv_out[r * COLS] = 1000.0; // one big outlier per row
    }
    cmp("single outlier/row", &adv_out);
}
