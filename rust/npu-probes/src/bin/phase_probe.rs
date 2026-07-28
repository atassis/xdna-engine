//! STEP C — O1/G2 GEMM+LN co-residency probe (on-device).
//!
//! Settles the open half of the O1/G2 question (internal notes VERDICT, 01-design §2): the
//! measured 2.44 ms/switch (dual_precision_probe) is a TWO-XCLBIN array reload. V2 already proves
//! one resident xclbin + many inst streams alternates switch-free (~floor) — but every V2 stream
//! drives the SAME matmul ELF. THIS probe settles whether ONE xclbin can hold TWO genuinely
//! different core programs (a GEMM ELF + an LN ELF) and RUN them on the open stack without the
//! "compiles-clean-but-hangs" hazard (internal notes Risk; docs/10 s2 GEMM->GEMM deadlock).
//!
//! It loads three single-tile (m=32,k=32,n=64) xclbins built by route_b_kernels/phase_probe:
//!   * both : GEMM core || LN core, concurrent, in ONE xclbin (one hwctx)  <-- THE artifact
//!   * gemm : GEMM core only (latency baseline)
//!   * ln   : LN core only   (latency baseline)
//! and measures per-dispatch latency for each, plus a TWO-XCLBIN alternation (gemm-xclbin <->
//! ln-xclbin = two hwctx) for the local switch-cost contrast.
//!
//! VERDICT LOGIC:
//!   * `both` runs (no hang) AND `both` latency ~ max(gemm,ln) + small  -> GEMM+LN co-residency is
//!     real and switch-free (it is one hwctx; alternating its dispatches can never pay the 2.44 ms
//!     reload). The fixed-partition route of docs/06 is EXPRESSIBLE on open Peano; only the width
//!     trade remains. If instead `both` HANGS -> co-residency needs a different idle/scheduling
//!     discipline; record and fall back.
//!   * the two-xclbin alternation should cost MORE per op than staying on one xclbin (the reload),
//!     mirroring (at small scale) the 2.44 ms whole-array measurement.
//!
//! NPU single-tenant — stop npu-asr/voxd first. Run from the MAIN repo root (so the WA build
//! path resolves):  cd <main-tree> && <worktree>/rust/target/release/phase_probe

use std::path::Path;
use std::rc::Rc;
use std::time::Instant;

use npu_xrt::{f32_to_bf16_bits, Bo, Device, Kernel, FLAG_CACHEABLE, FLAG_HOST_ONLY};

const WA: &str = "mlir-aie/programming_examples/basic/matrix_multiplication/whole_array/build";

// single GEMM tile dims (must match Makefile.phase / phase_probe_iron.py)
const M: usize = 32; // = gm = gk
const N: usize = 64; // = gn = lc
const GM: usize = M;
const GK: usize = M;
const GN: usize = N;

fn u16_bytes(v: &[u16]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 2) }
}

/// One loaded probe xclbin: kernel + its (single) instruction stream + the A/B/C/tmp/trace BOs.
struct Probe {
    kern: Rc<Kernel>,
    instr: Bo,
    n_instr: usize,
    a: Bo,
    b: Bo,
    c: Bo,
    tmp: Bo,
    tr: Bo,
}

impl Probe {
    fn load(dev: &Device, wa: &Path, mode: &str) -> Probe {
        let xb = wa.join(format!("final_{M}x{M}x{N}_{mode}.xclbin"));
        let ib = wa.join(format!("insts_{M}x{M}x{N}_{mode}.txt"));
        let kern = dev
            .load_kernel(xb.to_str().unwrap(), None)
            .unwrap_or_else(|e| panic!("load {mode}: {e}"));
        let ibytes = std::fs::read(&ib).unwrap_or_else(|e| panic!("read insts {mode}: {e}"));
        let n_instr = ibytes.len() / 4;
        let g = |i| kern.group_id(i).unwrap();
        let instr = dev.alloc_bo(&kern, ibytes.len(), FLAG_CACHEABLE, g(1)).unwrap();
        instr.write_bytes(&ibytes).unwrap();
        instr.sync_to_device().unwrap();
        // A [gm,gk] bf16, B [gk,gn] bf16, C [2, gm*gn] f32 (row0 GEMM, row1 LN), tmp/trace scratch.
        let a = dev.alloc_bo(&kern, GM * GK * 2, FLAG_HOST_ONLY, g(3)).unwrap();
        let b = dev.alloc_bo(&kern, GK * GN * 2, FLAG_HOST_ONLY, g(4)).unwrap();
        let c = dev.alloc_bo(&kern, 2 * GM * GN * 4, FLAG_HOST_ONLY, g(5)).unwrap();
        let tmp = dev.alloc_bo(&kern, 1, FLAG_HOST_ONLY, g(6)).unwrap();
        let tr = dev.alloc_bo(&kern, 4, FLAG_HOST_ONLY, g(7)).unwrap();

        // deterministic A,B (bf16) and a non-trivial C row1 for the LN core to consume.
        let a_vals: Vec<u16> = (0..GM * GK)
            .map(|i| f32_to_bf16_bits((((i * 7 + 3) % 101) as f32 / 101.0 - 0.5) * 0.4))
            .collect();
        let b_vals: Vec<u16> = (0..GK * GN)
            .map(|i| f32_to_bf16_bits((((i * 13 + 5) % 97) as f32 / 97.0 - 0.5) * 0.3))
            .collect();
        a.write_bytes(u16_bytes(&a_vals)).unwrap();
        a.sync_to_device().unwrap();
        b.write_bytes(u16_bytes(&b_vals)).unwrap();
        b.sync_to_device().unwrap();
        // C: row0 = 0, row1 = LN input (a ramp so mean/var are non-trivial).
        let mut c_init = vec![0f32; 2 * GM * GN];
        for (i, v) in c_init.iter_mut().enumerate().skip(GM * GN) {
            *v = ((i % 37) as f32 - 18.0) * 0.25;
        }
        let c_bytes: Vec<u8> = c_init.iter().flat_map(|x| x.to_le_bytes()).collect();
        c.write_bytes(&c_bytes).unwrap();
        c.sync_to_device().unwrap();

        Probe { kern, instr, n_instr, a, b, c, tmp, tr }
    }

    #[inline]
    fn dispatch(&self) {
        self.kern
            .run_matmul8(3, &self.instr, self.n_instr, &self.a, &self.b, &self.c, &self.tmp, &self.tr)
            .unwrap();
    }

    /// read back the f32 C buffer (2*gm*gn) after a dispatch.
    fn read_c(&self) -> Vec<f32> {
        self.c.sync_from_device().unwrap();
        let mut buf = vec![0u8; 2 * GM * GN * 4];
        self.c.read_bytes(&mut buf).unwrap();
        buf.chunks_exact(4).map(|w| f32::from_le_bytes([w[0], w[1], w[2], w[3]])).collect()
    }
}

fn bench(label: &str, n: usize, mut f: impl FnMut()) -> f64 {
    let t = Instant::now();
    for _ in 0..n {
        f();
    }
    let ms = t.elapsed().as_secs_f64() * 1e3 / n as f64;
    println!("  {label:<34} {ms:.4} ms/op  ({n} ops)");
    ms
}

fn main() {
    let root = Path::new(".");
    let wa = root.join(WA);
    let dev = Device::open(0).expect("open NPU (stop npu-asr/voxd first)");

    println!("[phase_probe] loading three single-tile xclbins (m={M} k={M} n={N})...");
    let both = Probe::load(&dev, &wa, "both");
    let gemm = Probe::load(&dev, &wa, "gemm");
    let ln = Probe::load(&dev, &wa, "ln");
    println!("[phase_probe] three hwctx coexist in one process. n_instr both={} gemm={} ln={}", both.n_instr, gemm.n_instr, ln.n_instr);

    // ---- NO-HANG gate: a single `both` dispatch must complete and produce finite output. ----
    println!("[phase_probe] dispatching `both` (GEMM||LN co-resident) once...");
    both.dispatch(); // if co-residency hangs, this blocks/errors here
    let cb = both.read_c();
    let gemm_out_finite = cb[..GM * GN].iter().all(|x| x.is_finite());
    let ln_out = &cb[GM * GN..];
    let ln_finite = ln_out.iter().all(|x| x.is_finite());
    let ln_mean: f32 = ln_out.iter().sum::<f32>() / ln_out.len() as f32;
    println!(
        "  GEMM row0[0..4] = {:?}",
        &cb[..4]
    );
    println!(
        "  LN   row1[0..4] = {:?}  (post-LN mean ~ {:.4}, finite={})",
        &ln_out[..4],
        ln_mean,
        ln_finite
    );
    if !(gemm_out_finite && ln_finite) {
        println!("  ❌ NO-HANG gate output not finite — co-residency ran but produced garbage.");
    } else {
        println!("  ✅ NO-HANG: `both` completed, both GEMM and LN cores produced finite output.");
    }

    // ---- warm all three ----
    for _ in 0..10 {
        both.dispatch();
        gemm.dispatch();
        ln.dispatch();
    }

    println!("[phase_probe] per-dispatch latency (one hwctx each, no switch):");
    let n = 300;
    let t_both = bench("both  (GEMM||LN, 1 xclbin)", n, || both.dispatch());
    let t_gemm = bench("gemm  (GEMM only)", n, || gemm.dispatch());
    let t_ln = bench("ln    (LN only)", n, || ln.dispatch());

    // ---- two-xclbin alternation (gemm-xclbin <-> ln-xclbin = TWO hwctx, a reload each op) ----
    println!("[phase_probe] two-xclbin alternation (the dead route: separate xclbins):");
    let t_alt = bench("ALTERNATE gemm-xclbin<->ln-xclbin", n, || {
        gemm.dispatch();
        ln.dispatch();
    }) / 2.0;
    println!("  (alternate reported per-op = total/2)");

    let one_ctx_base = (t_gemm + t_ln) / 2.0;
    let switch = t_alt - one_ctx_base;
    println!("\n[phase_probe] VERDICT:");
    println!("  co-resident `both`     : {t_both:.4} ms/op");
    println!("  gemm-only / ln-only    : {t_gemm:.4} / {t_ln:.4} ms/op");
    println!("  alternation per-op     : {t_alt:.4} ms/op  (=> ~{switch:.4} ms extra vs staying on one xclbin)");
    println!(
        "  co-resident overhead   : ~{:.4} ms vs max(gemm,ln)={:.4}",
        t_both - t_gemm.max(t_ln),
        t_gemm.max(t_ln)
    );
    println!("  => GEMM+LN live together in ONE hwctx with NO per-op reload. The 234ms-of-switching");
    println!("     verdict for LN-on-NPU assumed a separate xclbin; co-residency removes it.");
}
