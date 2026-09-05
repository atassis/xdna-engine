//! M2: what does host-side cache maintenance (`Bo::sync_to_device`/`sync_from_device`) actually
//! cost, as a function of range size -- and does a RANGED flush of a small region on a large BO
//! cost less than a WHOLE-BO flush of that same large BO?
//!
//! Two experiments, one process:
//!   1. STANDALONE sweep: fresh host_only BOs at sizes from 2304 B (the Qwen3 fused-decode input
//!      arena) to ~2.14 GB (its scratch arena), each freshly dirtied then synced. Answers: is cost
//!      linear in bytes, and what is the per-byte / per-cache-line rate.
//!   2. RANGED-vs-WHOLE: one 2.14 GB BO (the standalone sweep's own top entry, reused). Time
//!      syncing the WHOLE thing, then time syncing `.sub(0, 2304)` and `.sub(0, 303872)` views of
//!      IT -- the exact geometry the one-way-buffer design argument is about (does maintenance
//!      scale with the allocation or with what was written).
//!
//! Both directions are timed per size: on x86 CLFLUSH is documented as write-back+invalidate
//! regardless of requested direction, so TO_DEVICE and FROM_DEVICE are expected to cost about the
//! same -- this probe checks that rather than assuming it.
//!
//! Path exercised: `Bo::sync_*` -> `shim_bo_sync_*` -> `xrt::bo::sync()`. No `xrt.ini` exists on
//! this box (`Debug.force_driver_sync` defaults false), so this is the userspace CLFLUSH path, NOT
//! the kernel `SYNC_BO` ioctl. Says so again at the bottom of the report.
//!
//! Pure host-side timing: allocates BOs via `alloc_bo_raw` and calls `sync_*`, no kernel load, no
//! dispatch. Still touches `/dev/accel/accel0` (opens the device) -- run under `npu_lock.sh`.
//!
//! Usage: clflush_cost_probe

use std::time::{Duration, Instant};

use npu_xrt::{Bo, Device, FLAG_HOST_ONLY};

const MIN_REPS: usize = 5;
const MAX_REPS: usize = 100;
const TARGET_BUDGET: Duration = Duration::from_millis(1000);

const LINE: usize = 64; // x86-64 cache line

/// Sweep sizes, ascending: the two Qwen3 fused-decode arena sizes (input=2304 B, output=303872 B)
/// bracketed by a log-ish sweep from 4 KiB to the ~2.14 GB scratch arena.
const SIZES: &[usize] = &[
    2_304,
    4_096,
    16_384,
    65_536,
    262_144,
    303_872,
    1_048_576,
    4_194_304,
    16_777_216,
    67_108_864,
    268_435_456,
    1_073_741_824,
    2_140_000_000,
];

fn mean_std(xs: &[f64]) -> (f64, f64) {
    let n = xs.len();
    let mean = xs.iter().sum::<f64>() / n as f64;
    if n <= 1 {
        return (mean, 0.0);
    }
    let var = xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1) as f64;
    (mean, var.sqrt())
}

/// 95% CI half-width via normal approximation (n is 5-100 here, not small enough to need a t-table
/// for the purpose of this report; flagged as approximate in the printed header).
fn ci95(std: f64, n: usize) -> f64 {
    1.96 * std / (n as f64).sqrt()
}

/// Write `bo`'s ENTIRE extent with a repeating pattern, chunked through `pattern` (reused, not
/// reallocated per call). This is what makes the subsequent sync measure a REAL flush of dirty,
/// cache-resident lines across the whole range rather than a flush of untouched/clean memory.
/// Untimed by the caller (called before starting the clock).
fn dirty_whole(bo: &Bo, pattern: &[u8]) {
    let total = bo.nbytes();
    let mut off = 0usize;
    while off < total {
        let n = pattern.len().min(total - off);
        if off == 0 {
            bo.write_bytes(&pattern[..n]).expect("write_bytes at offset 0");
        } else {
            bo.sub(off, n).expect("sub").write_bytes(&pattern[..n]).expect("write_bytes at offset");
        }
        off += n;
    }
}

/// Time `reps` (adaptive, bounded by MIN/MAX_REPS and TARGET_BUDGET) dirty+sync cycles in each
/// direction. Returns ((mean_to_us, std_to_us, n_to), (mean_from_us, std_from_us, n_from)).
fn measure(bo: &Bo, pattern: &[u8]) -> ((f64, f64, usize), (f64, f64, usize)) {
    let mut to_us = Vec::new();
    let budget_start = Instant::now();
    while to_us.len() < MIN_REPS
        || (to_us.len() < MAX_REPS && budget_start.elapsed() < TARGET_BUDGET)
    {
        dirty_whole(bo, pattern);
        let t0 = Instant::now();
        bo.sync_to_device().expect("sync_to_device");
        to_us.push(t0.elapsed().as_secs_f64() * 1e6);
    }

    let mut from_us = Vec::new();
    let budget_start = Instant::now();
    while from_us.len() < MIN_REPS
        || (from_us.len() < MAX_REPS && budget_start.elapsed() < TARGET_BUDGET)
    {
        // Redirty + re-push to device first so the range is cache-resident going into the
        // FROM_DEVICE timing too (isolates flush cost from write/dispatch precondition, matching
        // the TO_DEVICE measurement's precondition rather than measuring an already-clean range).
        dirty_whole(bo, pattern);
        bo.sync_to_device().expect("pre-sync for from_device measurement");
        let t0 = Instant::now();
        bo.sync_from_device().expect("sync_from_device");
        from_us.push(t0.elapsed().as_secs_f64() * 1e6);
    }

    let (m_to, s_to) = mean_std(&to_us);
    let (m_from, s_from) = mean_std(&from_us);
    ((m_to, s_to, to_us.len()), (m_from, s_from, from_us.len()))
}

fn fmt_row(label: &str, size: usize, to: (f64, f64, usize), from: (f64, f64, usize)) -> String {
    let (m_to, s_to, n_to) = to;
    let (m_from, s_from, n_from) = from;
    let per_byte_to_ns = m_to * 1000.0 / size as f64;
    let per_line_to_ns = per_byte_to_ns * LINE as f64;
    let per_byte_from_ns = m_from * 1000.0 / size as f64;
    let per_line_from_ns = per_byte_from_ns * LINE as f64;
    format!(
        "{label:<28} {size:>12} B  \
to: {m_to:>10.2} +/- {ci_to:>7.2} us (n={n_to:<3}) {per_byte_to_ns:>8.4} ns/B {per_line_to_ns:>8.2} ns/line  \
from: {m_from:>10.2} +/- {ci_from:>7.2} us (n={n_from:<3}) {per_byte_from_ns:>8.4} ns/B {per_line_from_ns:>8.2} ns/line",
        ci_to = ci95(s_to, n_to),
        ci_from = ci95(s_from, n_from),
    )
}

fn main() {
    println!("[clflush_cost_probe] host-side Bo::sync_to_device/sync_from_device cost vs range size");
    println!("[clflush_cost_probe] CI = 95% normal-approx half-width (mean +/- ci), not a t-interval");

    let dev = Device::open(0).expect("open NPU (run under npu_lock.sh; do not stop services)");

    // Reusable dirty pattern, big enough to amortize the per-chunk write_bytes/sub call overhead
    // without itself dominating the untimed pre-touch phase.
    let pattern: Vec<u8> = (0..16 * 1024 * 1024).map(|i| (i as u8).wrapping_mul(31).wrapping_add(7)).collect();

    println!("\n=== Experiment 1: standalone BOs, cost vs range size ===");
    let mut big_bo: Option<Bo> = None;
    for &size in SIZES {
        let bo = dev.alloc_bo_raw(size, FLAG_HOST_ONLY, 0).unwrap_or_else(|e| panic!("alloc_bo_raw({size}): {e}"));
        let (to, from) = measure(&bo, &pattern);
        println!("{}", fmt_row("standalone", size, to, from));
        if size == *SIZES.last().unwrap() {
            big_bo = Some(bo); // keep the 2.14 GB BO alive for experiment 2; all smaller ones drop here
        }
    }

    println!("\n=== Experiment 2: ranged flush of a small region on the SAME 2.14 GB BO vs whole-BO flush ===");
    let big = big_bo.expect("top sweep size did not run");
    println!("(whole-BO cost for this exact allocation is the {} B row above)", big.nbytes());
    for &size in &[2_304usize, 303_872usize] {
        let view = big.sub(0, size).unwrap_or_else(|e| panic!("sub(0,{size}) of the 2.14 GB BO: {e}"));
        let (to, from) = measure(&view, &pattern);
        println!("{}", fmt_row("view-of-2.14GB-BO", size, to, from));
    }

    println!("\n[clflush_cost_probe] no xrt.ini found on this box -> Debug.force_driver_sync defaults false");
    println!("[clflush_cost_probe] path exercised: userspace CLFLUSH (xrt::bo::sync()), NOT the kernel SYNC_BO ioctl");
}
