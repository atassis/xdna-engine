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
