//! Tiny thread-local host profiler (enable by wrapping ops in `prof::time`). Used to locate the
//! host-glue hotspots before optimizing. Zero cost when the labels aren't hit.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::time::Instant;

thread_local!(static P: RefCell<BTreeMap<String, (f64, usize)>> = RefCell::new(BTreeMap::new()));

pub fn time<T>(label: &str, f: impl FnOnce() -> T) -> T {
    let t = Instant::now();
    let r = f();
    let dt = t.elapsed().as_secs_f64();
    P.with(|p| {
        let mut m = p.borrow_mut();
        let e = m.entry(label.to_string()).or_insert((0.0, 0));
        e.0 += dt;
        e.1 += 1;
    });
    r
}

pub fn report() -> String {
    P.with(|p| {
        let m = p.borrow();
        let mut rows: Vec<_> = m.iter().collect();
        rows.sort_by(|a, b| b.1 .0.partial_cmp(&a.1 .0).unwrap());
        rows.iter()
            .map(|(k, (s, n))| format!("  {k:18} {s:7.3}s  x{n}", s = s, n = n))
            .collect::<Vec<_>>()
            .join("\n")
    })
}

pub fn reset() {
    P.with(|p| p.borrow_mut().clear());
}

/// Phase-timing profiler: attributes encode-path time to one of three buckets
/// ([`Bucket::Npu`], [`Bucket::Host`], [`Bucket::Marshal`]). Zero overhead unless
/// the env var `PARAKEET_PHASE_TIMING` is set. Kept as a submodule because its
/// `reset`/`report` names deliberately differ from the legacy host profiler above.
pub mod phase {
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::sync::OnceLock;
    use std::time::{Duration, Instant};

    /// Which layer of the stack a measured span is charged to.
    #[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
    pub enum Bucket {
        /// On-device kernel dispatch.
        Npu,
        /// Host-side CPU math.
        Host,
        /// Host<->device DMA / sync with no compute.
        Marshal,
    }

    type Key = (&'static str, Bucket);
    thread_local!(static ACC: RefCell<HashMap<Key, (Duration, u64)>> = RefCell::new(HashMap::new()));

    /// Cached `PARAKEET_PHASE_TIMING` presence check (read once per process).
    static ENV_ON: OnceLock<bool> = OnceLock::new();

    #[cfg(test)]
    thread_local!(static FORCE_ON: std::cell::Cell<bool> = const { std::cell::Cell::new(false) });

    /// True when phase timing is enabled. Reads the env var at most once, then caches it.
    pub fn timing_on() -> bool {
        #[cfg(test)]
        {
            if FORCE_ON.with(|f| f.get()) {
                return true;
            }
        }
        *ENV_ON.get_or_init(|| std::env::var_os("PARAKEET_PHASE_TIMING").is_some())
    }

    /// Test-only: force [`timing_on`] to return true on this thread (no env needed).
    #[cfg(test)]
    pub(crate) fn force_on_for_test() {
        FORCE_ON.with(|f| f.set(true));
    }

    /// Inject a measurement straight into the accumulator, bypassing the env gate.
    /// Used by [`PhaseScope::drop`] and by unit tests.
    pub(crate) fn record_raw(stage: &'static str, bucket: Bucket, dur: Duration, calls: u64) {
        ACC.with(|a| {
            let mut m = a.borrow_mut();
            let e = m.entry((stage, bucket)).or_insert((Duration::ZERO, 0));
            e.0 += dur;
            e.1 += calls;
        });
    }

    /// Clear the accumulator. Call before each measured pass.
    pub fn reset() {
        ACC.with(|a| a.borrow_mut().clear());
    }

    /// RAII guard: times its lifetime and charges the elapsed span to `(stage, bucket)`
    /// on drop. A true no-op (never reads the clock) when [`timing_on`] is false.
    pub struct PhaseScope {
        stage: &'static str,
        bucket: Bucket,
        start: Option<Instant>,
        active: bool,
    }

    impl PhaseScope {
        pub fn new(stage: &'static str, bucket: Bucket) -> Self {
            let active = timing_on();
            PhaseScope {
                stage,
                bucket,
                start: if active { Some(Instant::now()) } else { None },
                active,
            }
        }
    }

    impl Drop for PhaseScope {
        fn drop(&mut self) {
            if self.active {
                if let Some(start) = self.start {
                    record_raw(self.stage, self.bucket, start.elapsed(), 1);
                }
            }
        }
    }

    /// A summarized pass: per-bucket totals plus residual/overlap vs the end-to-end wall time.
    pub struct PhaseReport {
        pub e2e_ms: f64,
        pub npu_ms: f64,
        pub host_ms: f64,
        pub marshal_ms: f64,
        /// `e2e - (npu+host+marshal)`, clamped at 0: wall time not attributed to any bucket.
        pub residual_ms: f64,
        /// `(npu+host+marshal) - e2e`, clamped at 0: >0 => concurrency, ~0 => serial.
        pub overlap_ms: f64,
        /// `(stage, bucket, ms, calls)` sorted descending by ms.
        pub rows: Vec<(String, Bucket, f64, u64)>,
    }

    /// Sum per bucket, rank the `(stage, bucket)` rows, and compute residual + overlap
    /// against the supplied end-to-end wall clock.
    pub fn report(e2e: Duration) -> PhaseReport {
        ACC.with(|a| {
            let m = a.borrow();
            let (mut npu_ms, mut host_ms, mut marshal_ms) = (0.0f64, 0.0f64, 0.0f64);
            let mut rows: Vec<(String, Bucket, f64, u64)> = Vec::with_capacity(m.len());
            for (&(stage, bucket), &(dur, calls)) in m.iter() {
                let ms = dur.as_secs_f64() * 1000.0;
                match bucket {
                    Bucket::Npu => npu_ms += ms,
                    Bucket::Host => host_ms += ms,
                    Bucket::Marshal => marshal_ms += ms,
                }
                rows.push((stage.to_string(), bucket, ms, calls));
            }
            rows.sort_by(|x, y| y.2.partial_cmp(&x.2).unwrap_or(std::cmp::Ordering::Equal));
            let e2e_ms = e2e.as_secs_f64() * 1000.0;
            let attributed = npu_ms + host_ms + marshal_ms;
            PhaseReport {
                e2e_ms,
                npu_ms,
                host_ms,
                marshal_ms,
                residual_ms: (e2e_ms - attributed).max(0.0),
                overlap_ms: (attributed - e2e_ms).max(0.0),
                rows,
            }
        })
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::time::Duration;
        #[test]
        fn phase_report_sums_and_ranks() {
            force_on_for_test();
            reset();
            record_raw("ff1", Bucket::Npu, Duration::from_micros(600), 4);
            record_raw("mhsa_softmax", Bucket::Host, Duration::from_micros(900), 4);
            record_raw("ff1", Bucket::Marshal, Duration::from_micros(300), 4);
            let r = report(Duration::from_micros(1800));
            assert!((r.npu_ms - 0.6).abs() < 1e-6);
            assert!((r.host_ms - 0.9).abs() < 1e-6);
            assert!((r.marshal_ms - 0.3).abs() < 1e-6);
            assert_eq!(r.rows[0].0, "mhsa_softmax"); // largest single (stage,bucket) first
            assert!(r.overlap_ms.abs() < 1e-6); // sum 1.8ms == e2e 1.8ms -> serial
            assert!(r.residual_ms.abs() < 1e-6);
        }
    }
}
