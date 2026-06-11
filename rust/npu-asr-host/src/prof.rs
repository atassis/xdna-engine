//! Tiny per-op host profiler. Accumulates wall-clock ns per named bucket across the whole run,
//! enabled only when the env var `NPU_HOST_PROF` is set (so it costs nothing in production).
//!
//! Usage:
//!   npu_asr_host::prof::reset();
//!   let out = npu_asr_host::prof::time("mha", || mha(...));   // or prof::scope("mha")
//!   npu_asr_host::prof::dump();   // prints the breakdown, sorted, once at the end
//!
//! Buckets are a fixed small set keyed by &'static str; we use a Mutex<Vec> guarded by a cheap
//! "enabled" flag so the disabled path is a single atomic load.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::Instant;

static ENABLED: AtomicBool = AtomicBool::new(false);
static INIT: std::sync::Once = std::sync::Once::new();

static BUCKETS: Mutex<Vec<(&'static str, u128, u64)>> = Mutex::new(Vec::new());

fn ensure_init() {
    INIT.call_once(|| {
        if std::env::var_os("NPU_HOST_PROF").is_some() {
            ENABLED.store(true, Ordering::Relaxed);
        }
    });
}

#[inline]
pub fn enabled() -> bool {
    ensure_init();
    ENABLED.load(Ordering::Relaxed)
}

/// Clear all buckets (call after warmup, before the timed loop).
pub fn reset() {
    ensure_init();
    if let Ok(mut b) = BUCKETS.lock() {
        b.clear();
    }
}

fn add(name: &'static str, ns: u128) {
    if let Ok(mut b) = BUCKETS.lock() {
        if let Some(e) = b.iter_mut().find(|e| e.0 == name) {
            e.1 += ns;
            e.2 += 1;
        } else {
            b.push((name, ns, 1));
        }
    }
}

/// Time a closure into bucket `name` (no-op overhead when disabled apart from the call itself).
#[inline]
pub fn time<T>(name: &'static str, f: impl FnOnce() -> T) -> T {
    if !enabled() {
        return f();
    }
    let t0 = Instant::now();
    let r = f();
    add(name, t0.elapsed().as_nanos());
    r
}

/// RAII scope timer: `let _g = prof::scope("foo");` times until end of the lexical block.
pub struct Guard {
    name: &'static str,
    t0: Instant,
    on: bool,
}
impl Drop for Guard {
    fn drop(&mut self) {
        if self.on {
            add(self.name, self.t0.elapsed().as_nanos());
        }
    }
}
#[inline]
pub fn scope(name: &'static str) -> Guard {
    Guard {
        name,
        t0: Instant::now(),
        on: enabled(),
    }
}

/// Print the breakdown (sorted by total time desc). `iters` divides the totals to per-run ms.
pub fn dump(iters: usize) {
    if !enabled() {
        return;
    }
    let b = BUCKETS.lock().unwrap();
    let mut rows: Vec<_> = b.clone();
    rows.sort_by(|a, c| c.1.cmp(&a.1));
    let total: u128 = rows.iter().map(|r| r.1).sum();
    let it = iters.max(1) as f64;
    eprintln!("\n=== HOST per-op profile (per run, {iters} iters) ===");
    eprintln!("  {:<22} {:>10}  {:>8}  {:>6}", "op", "ms/run", "calls/run", "%");
    for (name, ns, calls) in &rows {
        let ms = *ns as f64 / 1e6 / it;
        let pct = if total > 0 { *ns as f64 / total as f64 * 100.0 } else { 0.0 };
        eprintln!(
            "  {:<22} {:>10.2}  {:>8}  {:>5.1}%",
            name,
            ms,
            *calls as f64 / it,
            pct
        );
    }
    eprintln!(
        "  {:<22} {:>10.2}",
        "TOTAL host (profiled)",
        total as f64 / 1e6 / it
    );
}
