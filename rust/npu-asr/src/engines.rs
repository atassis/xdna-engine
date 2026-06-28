//! NPU engines: the weight-bound whole-array fused matmul (`WAEpilogue`) and the depthwise-conv
//! (`DwconvEngine`). Mirrors `npu_asr/fused.py` (WAEpilogue) and `npu_asr/ops.py` (DwconvEngine).
//!
//! WAEpilogue is WEIGHT-BOUND: the K-augmented weight `B_aug` and the instruction buffer are built
//! and synced ONCE in `new`; the activation/output buffers are allocated once and reused, so
//! `forward` only writes the activation tile and dispatches (task-15 levers 1+2).

use std::cell::RefCell;
use std::path::Path;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use ndarray::prelude::*;
use npu_xrt::{bf16_bits_to_f32, f32_to_bf16_bits, Bo, Device, Kernel, FLAG_CACHEABLE, FLAG_HOST_ONLY};
use rayon::prelude::*;

pub const WA_SUBDIR: &str =
    "mlir-aie/programming_examples/basic/matrix_multiplication/whole_array/build";
pub const DW_SUBDIR: &str = "mlir-aie/programming_examples/ml/dwconv1d/build";

pub const PAD_M: usize = 512;
const TILE: usize = 32;

// --- dispatch profiling (cumulative NPU .wait() time + dispatch count) ---
static NPU_NS: AtomicU64 = AtomicU64::new(0);
static NPU_DISP: AtomicU64 = AtomicU64::new(0);

pub fn reset_prof() {
    NPU_NS.store(0, Ordering::Relaxed);
    NPU_DISP.store(0, Ordering::Relaxed);
}
/// (seconds spent in NPU dispatch, number of dispatches)
pub fn prof() -> (f64, u64) {
    (
        NPU_NS.load(Ordering::Relaxed) as f64 / 1e9,
        NPU_DISP.load(Ordering::Relaxed),
    )
}
fn record(dt: std::time::Duration) {
    NPU_NS.fetch_add(dt.as_nanos() as u64, Ordering::Relaxed);
    NPU_DISP.fetch_add(1, Ordering::Relaxed);
}

/// crate-internal alias so other engine modules (`ctx2`) record into the same NPU profiler, and
/// attribute the dispatch's NPU-stall to the current op (`marsh::add_stall`).
#[allow(dead_code)]
pub(crate) fn prof_record(dt: std::time::Duration) {
    record(dt);
    marsh::add_stall(dt);
}

/// Per-dispatch HOST marshaling profiler, attributed PER OP (q/k/v/out/fc1/fc2). The ~162ms
/// "round-trip" pool bracketing each NPU dispatch, split by op x stage so we can rank which seam
/// (which op's convert/readback/epilogue) a resident dataflow should eliminate first. Timing is
/// always accumulated (Instant+atomic, negligible); the breakdown prints only when `NPU_MARSH_PROF`
/// is set. The current op is a thread-local tag set by the encoder before each matmul (mirrors the
/// `ENC_PEROP` pattern); all recording runs on the dispatching thread, so the tag is race-free.
pub mod marsh {
    use std::cell::Cell;
    use std::sync::atomic::{AtomicU64, Ordering};

    // stages (host round-trip sub-stages bracketing one dispatch)
    pub const CONV: usize = 0; // activation bf16 convert + write_bytes into the host buffer
    pub const SYNC_TO: usize = 1; // bo_a.sync_to_device (host->device)
    pub const SYNC_FROM: usize = 2; // bo_c.sync_from_device (device->host)
    pub const READ: usize = 3; // read_bytes out of the device BO
    pub const EPI: usize = 4; // host epilogue (bias/accumulate/dequant)
    const NS: usize = 5;

    // ops (the encoder's per-layer matmuls; OTHER = default / non-encoder dispatches e.g. decode)
    pub const Q: usize = 0;
    pub const K: usize = 1;
    pub const V: usize = 2;
    pub const OUT: usize = 3;
    pub const FC1: usize = 4;
    pub const FC2: usize = 5;
    pub const OTHER: usize = 6;
    const NO: usize = 7;
    pub const OP_NAMES: [&str; NO] = ["q", "k", "v", "out", "fc1", "fc2", "other"];

    static STAGE: [[AtomicU64; NS]; NO] = [const { [const { AtomicU64::new(0) }; NS] }; NO];
    static DISP: [AtomicU64; NO] = [const { AtomicU64::new(0) }; NO];
    static STALL: [AtomicU64; NO] = [const { AtomicU64::new(0) }; NO];

    thread_local! {
        static CUR_OP: Cell<usize> = const { Cell::new(OTHER) };
    }

    /// Tag subsequent dispatches with `op` (one of the consts above). Cheap; always called.
    pub fn set_op(op: usize) {
        CUR_OP.with(|c| c.set(op));
    }
    pub fn cur_op() -> usize {
        CUR_OP.with(|c| c.get())
    }

    /// Record a marshaling stage against the current op. `CONV` is recorded exactly once per
    /// dispatch (first stage in every path), so it doubles as the per-op dispatch counter.
    #[allow(dead_code)] // only the two_ctx path (ctx2.rs) records; reset/dump are used in both
    pub(crate) fn add(stage: usize, dt: std::time::Duration) {
        let op = cur_op();
        STAGE[op][stage].fetch_add(dt.as_nanos() as u64, Ordering::Relaxed);
        if stage == CONV {
            DISP[op].fetch_add(1, Ordering::Relaxed);
        }
    }
    /// Record one dispatch's NPU-stall (`.wait()`) against the current op (called by `prof_record`).
    #[allow(dead_code)]
    pub(crate) fn add_stall(dt: std::time::Duration) {
        STALL[cur_op()].fetch_add(dt.as_nanos() as u64, Ordering::Relaxed);
    }

    pub fn reset() {
        for op in 0..NO {
            for s in 0..NS {
                STAGE[op][s].store(0, Ordering::Relaxed);
            }
            DISP[op].store(0, Ordering::Relaxed);
            STALL[op].store(0, Ordering::Relaxed);
        }
    }

    // accessors (analysis + tests)
    pub fn stage_ns(op: usize, stage: usize) -> u64 {
        STAGE[op][stage].load(Ordering::Relaxed)
    }
    pub fn disp(op: usize) -> u64 {
        DISP[op].load(Ordering::Relaxed)
    }
    pub fn stall_ns(op: usize) -> u64 {
        STALL[op].load(Ordering::Relaxed)
    }
    /// Sum of all marshaling stages across all ops (the host round-trip total), used by
    /// [`super::dump_dispatch_prof`] for the round-trip-vs-stall ratio.
    pub fn total_ns() -> u64 {
        (0..NO)
            .flat_map(|op| (0..NS).map(move |s| STAGE[op][s].load(Ordering::Relaxed)))
            .sum()
    }

    /// Print the per-op x per-stage marshaling table (per pass) when `NPU_MARSH_PROF` is set.
    pub fn dump(iters: usize) {
        if std::env::var("NPU_MARSH_PROF").is_err() {
            return;
        }
        let it = iters.max(1) as f64;
        eprintln!("\n=== encoder seam attribution (per pass; {iters} passes; ms) ===");
        eprintln!(
            "  {:>5} {:>6} {:>10} {:>8} {:>9} {:>10} {:>8} | {:>10} {:>7}",
            "op", "disp", "conv+wr", "sync_to", "syncfrom", "read_byt", "epi", "rndtrip", "stall"
        );
        let ms = |ns: u64| ns as f64 / 1e6 / it;
        let mut tot = [0f64; NS];
        let mut tot_disp = 0f64;
        let mut tot_stall = 0f64;
        for op in 0..NO {
            let d = disp(op) as f64 / it;
            if d == 0.0 && stall_ns(op) == 0 {
                continue; // skip ops that never ran (keeps OTHER out when unused)
            }
            let st: [f64; NS] = std::array::from_fn(|s| ms(stage_ns(op, s)));
            let rt: f64 = st.iter().sum();
            let stall = ms(stall_ns(op));
            for s in 0..NS {
                tot[s] += st[s];
            }
            tot_disp += d;
            tot_stall += stall;
            eprintln!(
                "  {:>5} {:>6.0} {:>10.2} {:>8.2} {:>9.2} {:>10.2} {:>8.2} | {:>10.2} {:>7.2}",
                OP_NAMES[op], d, st[CONV], st[SYNC_TO], st[SYNC_FROM], st[READ], st[EPI], rt, stall
            );
        }
        let rt_tot: f64 = tot.iter().sum();
        eprintln!(
            "  {:>5} {:>6.0} {:>10.2} {:>8.2} {:>9.2} {:>10.2} {:>8.2} | {:>10.2} {:>7.2}",
            "TOTAL", tot_disp, tot[CONV], tot[SYNC_TO], tot[SYNC_FROM], tot[READ], tot[EPI], rt_tot,
            tot_stall
        );
    }
}

/// Step 0 (resident full-NPU spec): split the NPU dispatch stream into NPU-compute (the `.wait()`
/// stall = on-chip compute + DMA) vs host marshaling (the round-trip = bf16 convert/write + syncs +
/// readback + epilogue). The cascade un-park gate is "marshaling-dominated => round-trip-bound", so
/// this prints both totals + the per-dispatch split. Prints only when `NPU_MARSH_PROF` is set.
/// `iters` = number of passes (encoder runs) to average over. Call [`reset_prof`] + [`marsh::reset`]
/// before the timed region.
pub fn dump_dispatch_prof(iters: usize) {
    if std::env::var("NPU_MARSH_PROF").is_err() {
        return;
    }
    let (npu_s, disp) = prof();
    let it = iters.max(1) as f64;
    let npu_ms = npu_s * 1e3 / it;
    let marsh_ms = marsh::total_ns() as f64 / 1e6 / it;
    let disp_per = disp as f64 / it;
    let total = (npu_ms + marsh_ms).max(1e-9);
    let dp = disp_per.max(1.0);
    eprintln!("\n=== NPU dispatch split (per pass; {iters} passes; {disp_per:.0} dispatches/pass) ===");
    eprintln!(
        "  NPU-compute (stall, .wait())   {npu_ms:8.2} ms  ({:4.1}%)   {:.3} ms/dispatch",
        100.0 * npu_ms / total,
        npu_ms / dp,
    );
    eprintln!(
        "  host marshaling (round-trip)   {marsh_ms:8.2} ms  ({:4.1}%)   {:.3} ms/dispatch",
        100.0 * marsh_ms / total,
        marsh_ms / dp,
    );
    marsh::dump(iters); // per-stage marshaling breakdown
}

pub(crate) fn u16_bytes(v: &[u16]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 2) }
}

pub(crate) fn read_instr_words(path: &Path) -> (Vec<u8>, usize) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read instr {}: {e}", path.display()));
    let words = bytes.len() / 4;
    (bytes, words)
}

/// Weight-bound whole-array fused matmul with on-chip bias(+SiLU). bias rides a K-augmented
/// extra k-block. Output is bf16 (the epilogue narrows f32->bf16 on-chip).
pub struct WAEpilogue {
    dev: Rc<Device>,
    kern: Rc<Kernel>,
    k: usize,
    n: usize,
    kaug: usize,
    n_instr: usize,
    bo_instr: Bo,
    bo_b: Bo, // K-augmented weight (synced once)
    bo_a: Bo, // activation (reused)
    bo_c: Bo, // output bf16 (reused)
    bo_tmp: Bo,
    bo_tr: Bo,
    a_aug: RefCell<Vec<u16>>, // host activation buffer, col K preset to bf16(1.0)
    cbuf: RefCell<Vec<u8>>,   // reused output read buffer
    inited: std::cell::Cell<bool>,
}

impl WAEpilogue {
    /// `mode` = "silu" or "bias". `b_real` is [K, N]; `bias` is length N.
    pub fn new(
        dev: Rc<Device>,
        root: &Path,
        mode: &str,
        k: usize,
        n: usize,
        b_real: &Array2<f32>,
        bias: &[f32],
    ) -> Self {
        assert!(mode == "silu" || mode == "bias");
        let kaug = k + TILE;
        let suffix = format!("{PAD_M}x{kaug}x{n}_{TILE}x{TILE}x{TILE}_8c_{mode}");
        let wa = root.join(WA_SUBDIR);
        let xclbin = wa.join(format!("final_{suffix}.xclbin"));
        let insts = wa.join(format!("insts_{suffix}.txt"));
        let kern = dev
            .load_kernel(xclbin.to_str().unwrap(), None)
            .unwrap_or_else(|e| panic!("load {}: {e}", xclbin.display()));
        let (instr_bytes, n_instr) = read_instr_words(&insts);

        // constant K-augmented weight, built once: rows 0..K = weight, row K = bias, rest 0
        let mut b_bits = vec![0u16; kaug * n];
        for kk in 0..k {
            for nn in 0..n {
                b_bits[kk * n + nn] = f32_to_bf16_bits(b_real[[kk, nn]]);
            }
        }
        for nn in 0..n {
            b_bits[k * n + nn] = f32_to_bf16_bits(bias[nn]);
        }

        let g_instr = kern.group_id(1).unwrap();
        let g_a = kern.group_id(3).unwrap();
        let g_b = kern.group_id(4).unwrap();
        let g_c = kern.group_id(5).unwrap();
        let g_tmp = kern.group_id(6).unwrap();
        let g_tr = kern.group_id(7).unwrap();

        let bo_instr = dev.alloc_bo(&kern, instr_bytes.len(), FLAG_CACHEABLE, g_instr).unwrap();
        bo_instr.write_bytes(&instr_bytes).unwrap();
        bo_instr.sync_to_device().unwrap();
        let bo_b = dev.alloc_bo(&kern, b_bits.len() * 2, FLAG_HOST_ONLY, g_b).unwrap();
        bo_b.write_bytes(u16_bytes(&b_bits)).unwrap();
        bo_b.sync_to_device().unwrap();
        let bo_a = dev.alloc_bo(&kern, PAD_M * kaug * 2, FLAG_HOST_ONLY, g_a).unwrap();
        let bo_c = dev.alloc_bo(&kern, PAD_M * n * 2, FLAG_HOST_ONLY, g_c).unwrap();
        let bo_tmp = dev.alloc_bo(&kern, 1, FLAG_HOST_ONLY, g_tmp).unwrap();
        let bo_tr = dev.alloc_bo(&kern, 4, FLAG_HOST_ONLY, g_tr).unwrap();

        // host activation buffer: zeros, col K preset to bf16(1.0) for ALL rows (the bias rider)
        let mut a_aug = vec![0u16; PAD_M * kaug];
        let one = f32_to_bf16_bits(1.0);
        for r in 0..PAD_M {
            a_aug[r * kaug + k] = one;
        }

        WAEpilogue {
            dev,
            kern,
            k,
            n,
            kaug,
            n_instr,
            bo_instr,
            bo_b,
            bo_a,
            bo_c,
            bo_tmp,
            bo_tr,
            a_aug: RefCell::new(a_aug),
            cbuf: RefCell::new(vec![0u8; PAD_M * n * 2]),
            inited: std::cell::Cell::new(false),
        }
    }

    /// a_real is [Mp, K] (Mp <= PAD_M). Returns [Mp, N] f32.
    pub fn forward(&self, a_real: &Array2<f32>) -> Array2<f32> {
        let (mp, kk) = a_real.dim();
        assert_eq!(kk, self.k);
        assert!(mp <= PAD_M);
        {
            let mut a = self.a_aug.borrow_mut();
            let (k, kaug) = (self.k, self.kaug);
            // convert this call's `mp` real rows f32->bf16 in parallel (disjoint row chunks)
            a.par_chunks_mut(kaug).take(mp).enumerate().for_each(|(r, row)| {
                for c in 0..k {
                    row[c] = f32_to_bf16_bits(a_real[[r, c]]);
                }
            });
            // First call writes the full buffer (sets the constant padding rows + bias column);
            // later calls only need to refresh the `mp` real rows.
            if self.inited.get() {
                self.bo_a.write_bytes(&u16_bytes(&a)[..mp * self.kaug * 2]).unwrap();
            } else {
                self.bo_a.write_bytes(u16_bytes(&a)).unwrap();
                self.inited.set(true);
            }
        }
        self.bo_a.sync_to_device().unwrap();

        let t0 = Instant::now();
        self.kern
            .run_matmul8(
                3,
                &self.bo_instr,
                self.n_instr,
                &self.bo_a,
                &self.bo_b,
                &self.bo_c,
                &self.bo_tmp,
                &self.bo_tr,
            )
            .unwrap();
        record(t0.elapsed());

        self.bo_c.sync_from_device().unwrap();
        // output is bf16 row-major [M,N]; the first mp rows are the first mp*N elements
        let n = self.n;
        {
            let mut cb = self.cbuf.borrow_mut();
            self.bo_c.read_bytes(&mut cb[..mp * n * 2]).unwrap();
        }
        let cb = self.cbuf.borrow();
        let bytes: &[u8] = &cb[..mp * n * 2];
        // bf16 -> f32 in parallel over all mp*N elements
        let data: Vec<f32> = (0..mp * n)
            .into_par_iter()
            .map(|i| {
                let idx = i * 2;
                bf16_bits_to_f32(u16::from_le_bytes([bytes[idx], bytes[idx + 1]]))
            })
            .collect();
        Array2::from_shape_vec((mp, n), data).unwrap()
    }
}

/// Chained FFN: mm1 (silu epilogue, K-augmented bias rider) writes the 3072-wide intermediate H
/// into a **device BO** that mm2 (plain matmul, f32 out) reads directly as its A input — H never
/// round-trips to the host. bias2 is added on host after mm2 (the plain mm2 has no bias rider).
///
/// Mirrors the validated `scripts/proto_ffn_chain.py`: bo_H is allocated against mm1's C
/// `group_id(5)` and passed to mm2 as its A `group_id(3)` (XRT allows sharing host_only BOs across
/// the two whole-array hw-context dispatches). mm1 is weight-bound exactly like `WAEpilogue::new`
/// (B_aug + instr synced once, activation/output buffers reused).
pub struct ChainedFFN {
    dev: Rc<Device>,
    kern1: Rc<Kernel>,
    kern2: Rc<Kernel>,
    k1: usize, // mm1 K (768)
    kaug1: usize,
    n2: usize, // mm2 N (768)
    n_instr1: usize,
    n_instr2: usize,
    // mm1 buffers
    bo_instr1: Bo,
    bo_b1: Bo, // K-augmented W1 (synced once)
    bo_a1: Bo, // mm1 activation (reused)
    // shared intermediate: mm1's C output AND mm2's A input (bf16 [PAD_M, n1])
    bo_h: Bo,
    bo_tmp1: Bo,
    bo_tr1: Bo,
    // mm2 buffers
    bo_instr2: Bo,
    bo_b2: Bo, // plain W2 (synced once)
    bo_c2: Bo, // mm2 output f32 [PAD_M, n2] (reused)
    bo_tmp2: Bo,
    bo_tr2: Bo,
    bias2: Vec<f32>, // host-side bias add after mm2, length n2
    a_aug1: RefCell<Vec<u16>>,
    cbuf: RefCell<Vec<u8>>, // reused f32 output read buffer (PAD_M*n2*4 bytes)
    inited: std::cell::Cell<bool>,
}

impl ChainedFFN {
    /// mm1: silu epilogue, `w1` is [k1, n1] (LN affine pre-folded), `b1` length n1 (bias rider).
    /// mm2: plain matmul, `w2` is [n1, n2], `b2` length n2 (added on host).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dev: Rc<Device>,
        root: &Path,
        k1: usize,
        n1: usize,
        n2: usize,
        w1: &Array2<f32>,
        b1: &[f32],
        w2: &Array2<f32>,
        b2: &[f32],
    ) -> Self {
        assert_eq!(w1.dim(), (k1, n1));
        assert_eq!(w2.dim(), (n1, n2));
        let kaug1 = k1 + TILE;
        let wa = root.join(WA_SUBDIR);

        // --- mm1: silu epilogue xclbin (K-augmented) ---
        let suffix1 = format!("{PAD_M}x{kaug1}x{n1}_{TILE}x{TILE}x{TILE}_8c_silu");
        let xclbin1 = wa.join(format!("final_{suffix1}.xclbin"));
        let insts1 = wa.join(format!("insts_{suffix1}.txt"));
        let kern1 = dev
            .load_kernel(xclbin1.to_str().unwrap(), None)
            .unwrap_or_else(|e| panic!("load {}: {e}", xclbin1.display()));
        let (instr1_bytes, n_instr1) = read_instr_words(&insts1);

        // --- mm2: PLAIN matmul xclbin (no mode suffix), f32 out ---
        let suffix2 = format!("{PAD_M}x{n1}x{n2}_{TILE}x{TILE}x{TILE}_8c");
        let xclbin2 = wa.join(format!("final_{suffix2}.xclbin"));
        let insts2 = wa.join(format!("insts_{suffix2}.txt"));
        let kern2 = dev
            .load_kernel(xclbin2.to_str().unwrap(), None)
            .unwrap_or_else(|e| panic!("load {}: {e}", xclbin2.display()));
        let (instr2_bytes, n_instr2) = read_instr_words(&insts2);

        // mm1 K-augmented weight: rows 0..k1 = W1, row k1 = b1, rest 0
        let mut b1_bits = vec![0u16; kaug1 * n1];
        for kk in 0..k1 {
            for nn in 0..n1 {
                b1_bits[kk * n1 + nn] = f32_to_bf16_bits(w1[[kk, nn]]);
            }
        }
        for nn in 0..n1 {
            b1_bits[k1 * n1 + nn] = f32_to_bf16_bits(b1[nn]);
        }

        // mm2 plain weight W2 [n1, n2] bf16 (no augmentation, no bias rider)
        let mut b2_bits = vec![0u16; n1 * n2];
        for kk in 0..n1 {
            for nn in 0..n2 {
                b2_bits[kk * n2 + nn] = f32_to_bf16_bits(w2[[kk, nn]]);
            }
        }

        // mm1 group ids
        let g1_instr = kern1.group_id(1).unwrap();
        let g1_a = kern1.group_id(3).unwrap();
        let g1_b = kern1.group_id(4).unwrap();
        let g1_c = kern1.group_id(5).unwrap();
        let g1_tmp = kern1.group_id(6).unwrap();
        let g1_tr = kern1.group_id(7).unwrap();
        // mm2 group ids (A=gid3 is where bo_h is passed, but bo_h is allocated against mm1's C gid)
        let g2_instr = kern2.group_id(1).unwrap();
        let g2_b = kern2.group_id(4).unwrap();
        let g2_c = kern2.group_id(5).unwrap();
        let g2_tmp = kern2.group_id(6).unwrap();
        let g2_tr = kern2.group_id(7).unwrap();

        let bo_instr1 = dev.alloc_bo(&kern1, instr1_bytes.len(), FLAG_CACHEABLE, g1_instr).unwrap();
        bo_instr1.write_bytes(&instr1_bytes).unwrap();
        bo_instr1.sync_to_device().unwrap();
        let bo_b1 = dev.alloc_bo(&kern1, b1_bits.len() * 2, FLAG_HOST_ONLY, g1_b).unwrap();
        bo_b1.write_bytes(u16_bytes(&b1_bits)).unwrap();
        bo_b1.sync_to_device().unwrap();
        let bo_a1 = dev.alloc_bo(&kern1, PAD_M * kaug1 * 2, FLAG_HOST_ONLY, g1_a).unwrap();
        // shared intermediate H: bf16 [PAD_M, n1], allocated against mm1's C group (proto's bo_H)
        let bo_h = dev.alloc_bo(&kern1, PAD_M * n1 * 2, FLAG_HOST_ONLY, g1_c).unwrap();
        let bo_tmp1 = dev.alloc_bo(&kern1, 1, FLAG_HOST_ONLY, g1_tmp).unwrap();
        let bo_tr1 = dev.alloc_bo(&kern1, 4, FLAG_HOST_ONLY, g1_tr).unwrap();

        let bo_instr2 = dev.alloc_bo(&kern2, instr2_bytes.len(), FLAG_CACHEABLE, g2_instr).unwrap();
        bo_instr2.write_bytes(&instr2_bytes).unwrap();
        bo_instr2.sync_to_device().unwrap();
        let bo_b2 = dev.alloc_bo(&kern2, b2_bits.len() * 2, FLAG_HOST_ONLY, g2_b).unwrap();
        bo_b2.write_bytes(u16_bytes(&b2_bits)).unwrap();
        bo_b2.sync_to_device().unwrap();
        // mm2 output: f32 [PAD_M, n2] (4 bytes/elem)
        let bo_c2 = dev.alloc_bo(&kern2, PAD_M * n2 * 4, FLAG_HOST_ONLY, g2_c).unwrap();
        let bo_tmp2 = dev.alloc_bo(&kern2, 1, FLAG_HOST_ONLY, g2_tmp).unwrap();
        let bo_tr2 = dev.alloc_bo(&kern2, 4, FLAG_HOST_ONLY, g2_tr).unwrap();

        // host mm1 activation buffer: zeros, col k1 = bf16(1.0) (bias rider) for all rows
        let mut a_aug1 = vec![0u16; PAD_M * kaug1];
        let one = f32_to_bf16_bits(1.0);
        for r in 0..PAD_M {
            a_aug1[r * kaug1 + k1] = one;
        }

        ChainedFFN {
            dev,
            kern1,
            kern2,
            k1,
            kaug1,
            n2,
            n_instr1,
            n_instr2,
            bo_instr1,
            bo_b1,
            bo_a1,
            bo_h,
            bo_tmp1,
            bo_tr1,
            bo_instr2,
            bo_b2,
            bo_c2,
            bo_tmp2,
            bo_tr2,
            bias2: b2.to_vec(),
            a_aug1: RefCell::new(a_aug1),
            cbuf: RefCell::new(vec![0u8; PAD_M * n2 * 4]),
            inited: std::cell::Cell::new(false),
        }
    }

    /// a_real is [Mp, k1] (Mp <= PAD_M). Returns [Mp, n2] f32 (mm2 + bias2 on host).
    pub fn forward(&self, a_real: &Array2<f32>) -> Array2<f32> {
        let (mp, kk) = a_real.dim();
        assert_eq!(kk, self.k1);
        assert!(mp <= PAD_M);

        // --- write/convert mm1 activation (same scheme as WAEpilogue::forward) ---
        {
            let mut a = self.a_aug1.borrow_mut();
            let (k1, kaug1) = (self.k1, self.kaug1);
            a.par_chunks_mut(kaug1).take(mp).enumerate().for_each(|(r, row)| {
                for c in 0..k1 {
                    row[c] = f32_to_bf16_bits(a_real[[r, c]]);
                }
            });
            if self.inited.get() {
                self.bo_a1.write_bytes(&u16_bytes(&a)[..mp * self.kaug1 * 2]).unwrap();
            } else {
                self.bo_a1.write_bytes(u16_bytes(&a)).unwrap();
                self.inited.set(true);
            }
        }
        self.bo_a1.sync_to_device().unwrap();

        // --- mm1: writes H into bo_h on device; NO sync_from_device of H ---
        let t0 = Instant::now();
        self.kern1
            .run_matmul8(
                3,
                &self.bo_instr1,
                self.n_instr1,
                &self.bo_a1,
                &self.bo_b1,
                &self.bo_h,
                &self.bo_tmp1,
                &self.bo_tr1,
            )
            .unwrap();
        record(t0.elapsed());

        // --- mm2: reads bo_h as A (NO host write/sync of A); writes f32 C ---
        let t1 = Instant::now();
        self.kern2
            .run_matmul8(
                3,
                &self.bo_instr2,
                self.n_instr2,
                &self.bo_h,
                &self.bo_b2,
                &self.bo_c2,
                &self.bo_tmp2,
                &self.bo_tr2,
            )
            .unwrap();
        record(t1.elapsed());

        // --- read mm2 f32 output, add bias2 (broadcast over rows) on host ---
        self.bo_c2.sync_from_device().unwrap();
        let n2 = self.n2;
        {
            let mut cb = self.cbuf.borrow_mut();
            self.bo_c2.read_bytes(&mut cb[..mp * n2 * 4]).unwrap();
        }
        let cb = self.cbuf.borrow();
        let bytes: &[u8] = &cb[..mp * n2 * 4];
        let bias2 = &self.bias2;
        let data: Vec<f32> = (0..mp * n2)
            .into_par_iter()
            .map(|i| {
                let idx = i * 4;
                let v = f32::from_le_bytes([bytes[idx], bytes[idx + 1], bytes[idx + 2], bytes[idx + 3]]);
                v + bias2[i % n2]
            })
            .collect();
        Array2::from_shape_vec((mp, n2), data).unwrap()
    }

    pub fn device(&self) -> &Rc<Device> {
        &self.dev
    }
}

/// depthwise conv1d k=5 'same' on [768,400]; one channel per ObjectFifo tile.
pub struct DwconvEngine {
    dev: Rc<Device>,
    kern: Rc<Kernel>,
    n_instr: usize,
    bo_instr: Bo,
    bo_x: Bo,
    bo_w: Bo,
    bo_y: Bo,
    ch: usize,
    t: usize,
}

impl DwconvEngine {
    pub fn new(dev: Rc<Device>, root: &Path, ch: usize, t: usize) -> Self {
        let dw = root.join(DW_SUBDIR);
        let xclbin = dw.join("final.xclbin");
        let insts = dw.join("insts.bin");
        let kern = dev
            .load_kernel(xclbin.to_str().unwrap(), None)
            .unwrap_or_else(|e| panic!("load {}: {e}", xclbin.display()));
        let (instr_bytes, n_instr) = read_instr_words(&insts);

        let g_instr = kern.group_id(1).unwrap();
        let g_x = kern.group_id(3).unwrap();
        let g_w = kern.group_id(4).unwrap();
        let g_y = kern.group_id(5).unwrap();

        let bo_instr = dev.alloc_bo(&kern, instr_bytes.len(), FLAG_CACHEABLE, g_instr).unwrap();
        bo_instr.write_bytes(&instr_bytes).unwrap();
        bo_instr.sync_to_device().unwrap();
        let bo_x = dev.alloc_bo(&kern, ch * t * 2, FLAG_HOST_ONLY, g_x).unwrap();
        let bo_w = dev.alloc_bo(&kern, ch * 16 * 2, FLAG_HOST_ONLY, g_w).unwrap();
        let bo_y = dev.alloc_bo(&kern, ch * t * 2, FLAG_HOST_ONLY, g_y).unwrap();

        DwconvEngine {
            dev,
            kern,
            n_instr,
            bo_instr,
            bo_x,
            bo_w,
            bo_y,
            ch,
            t,
        }
    }

    /// x is [ch, T] f32; taps is [ch, 5] f32. Returns [ch, T] f32.
    pub fn dwconv(&self, x: &Array2<f32>, taps: &Array2<f32>) -> Array2<f32> {
        let (ch, t) = x.dim();
        assert_eq!((ch, t), (self.ch, self.t));
        // weights padded to [ch,16]: cols 0..5 = taps, rest 0
        let mut w_bits = vec![0u16; ch * 16];
        for c in 0..ch {
            for ki in 0..5 {
                w_bits[c * 16 + ki] = f32_to_bf16_bits(taps[[c, ki]]);
            }
        }
        self.bo_w.write_bytes(u16_bytes(&w_bits)).unwrap();
        self.bo_w.sync_to_device().unwrap();
        // x bf16 row-major flat
        let mut x_bits = vec![0u16; ch * t];
        for c in 0..ch {
            for ti in 0..t {
                x_bits[c * t + ti] = f32_to_bf16_bits(x[[c, ti]]);
            }
        }
        self.bo_x.write_bytes(u16_bytes(&x_bits)).unwrap();
        self.bo_x.sync_to_device().unwrap();

        let t0 = Instant::now();
        self.kern
            .run_dwconv6(3, &self.bo_instr, self.n_instr, &self.bo_x, &self.bo_w, &self.bo_y)
            .unwrap();
        record(t0.elapsed());

        self.bo_y.sync_from_device().unwrap();
        let mut yb = vec![0u8; ch * t * 2];
        self.bo_y.read_bytes(&mut yb).unwrap();
        let mut out = Array2::<f32>::zeros((ch, t));
        for c in 0..ch {
            for ti in 0..t {
                let idx = (c * t + ti) * 2;
                let bits = u16::from_le_bytes([yb[idx], yb[idx + 1]]);
                out[[c, ti]] = bf16_bits_to_f32(bits);
            }
        }
        out
    }
}

// keep Rc<Device> field used even if some accessors are unused
impl WAEpilogue {
    pub fn device(&self) -> &Rc<Device> {
        &self.dev
    }
}
impl DwconvEngine {
    pub fn device(&self) -> &Rc<Device> {
        &self.dev
    }
}

#[cfg(test)]
mod marsh_tests {
    use super::marsh;
    use std::time::Duration;

    #[test]
    fn per_op_attribution_routes_by_cur_op() {
        marsh::reset();
        marsh::set_op(marsh::FC2);
        marsh::add(marsh::EPI, Duration::from_nanos(100));
        marsh::add(marsh::CONV, Duration::from_nanos(50)); // CONV also counts one dispatch
        marsh::set_op(marsh::Q);
        marsh::add(marsh::CONV, Duration::from_nanos(10));

        assert_eq!(marsh::stage_ns(marsh::FC2, marsh::EPI), 100);
        assert_eq!(marsh::stage_ns(marsh::FC2, marsh::CONV), 50);
        assert_eq!(marsh::disp(marsh::FC2), 1);
        assert_eq!(marsh::stage_ns(marsh::Q, marsh::CONV), 10);
        assert_eq!(marsh::disp(marsh::Q), 1);
        assert_eq!(marsh::disp(marsh::V), 0);
        // total_ns sums all op x stage
        assert_eq!(marsh::total_ns(), 160);
    }
}
