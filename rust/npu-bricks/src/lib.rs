//! Shared NPU dispatch rails: byte-marshaling helpers and dispatch profiling with zero
//! model-specific semantics. Extracted from `npu-asr::engines` (`extract-npu-bricks-crate` step 1)
//! -- `PAD_M`/`WA_SUBDIR`/`u16_bytes` had independent byte-identical copies in `npu-asr`,
//! `npu-parakeet` and `npu-whisper`; this crate is their one definition.

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

pub const WA_SUBDIR: &str =
    "mlir-aie/programming_examples/basic/matrix_multiplication/whole_array/build";
pub const DW_SUBDIR: &str = "mlir-aie/programming_examples/ml/dwconv1d/build";

/// The resident-kernel row-tile forced by the whole-array kernel shape.
pub const PAD_M: usize = 512;

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
pub fn record(dt: std::time::Duration) {
    NPU_NS.fetch_add(dt.as_nanos() as u64, Ordering::Relaxed);
    NPU_DISP.fetch_add(1, Ordering::Relaxed);
}

/// Record one dispatch's NPU-stall time, attributed to the current `marsh` op.
pub fn prof_record(dt: std::time::Duration) {
    record(dt);
    marsh::add_stall(dt);
}

/// Per-dispatch HOST marshaling profiler, attributed PER OP (q/k/v/out/fc1/fc2). The ~162ms
/// "round-trip" pool bracketing each NPU dispatch, split by op x stage so we can rank which seam
/// (which op's convert/readback/epilogue) a resident dataflow should eliminate first. Timing is
/// always accumulated (Instant+atomic, negligible); the breakdown prints only when `NPU_MARSH_PROF`
/// is set. The current op is a thread-local tag set by the encoder before each matmul; all
/// recording runs on the dispatching thread, so the tag is race-free.
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
    #[allow(dead_code)] // only the two_ctx path (npu-asr's ctx2.rs) records; reset/dump are used in both
    pub fn add(stage: usize, dt: std::time::Duration) {
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

pub fn u16_bytes(v: &[u16]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 2) }
}

pub fn read_instr_words(path: &Path) -> (Vec<u8>, usize) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read instr {}: {e}", path.display()));
    let words = bytes.len() / 4;
    (bytes, words)
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
