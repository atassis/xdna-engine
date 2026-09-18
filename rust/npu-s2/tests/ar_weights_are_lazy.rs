//! Proves `ArWeights` never materializes the whole 4.56B-element AR checkpoint, by MEASURING
//! `/proc/self/status`'s `VmHWM` (peak resident-set size -- monotonic, so it survives any freeing
//! between checkpoints) rather than trusting the "lazy" claim. The number that matters: the AR
//! weight set is 17.0 GiB dequantized to f32 (4,561,852,416 elements); this box carries ~18 GiB
//! available. An implementation that eagerly decoded every tensor at open time would push peak RSS
//! to roughly that 17 GiB plus the ~4.2 GiB the checkpoint's raw bytes already occupy once opened
//! (`GgufFile::open` reads the whole file) -- past the box's own budget before a single token is
//! produced. This test opens the model, touches several tensors across both stacks and multiple
//! layers, and asserts peak RSS stays within a single-digit-GiB band, nowhere near that 21+ GiB.
//!
//! Skips (does not fail) when the GGUF file isn't available or `/proc` isn't readable (non-Linux).

use std::path::PathBuf;

fn gguf_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("S2_GGUF") {
        let p = PathBuf::from(p);
        return p.is_file().then_some(p);
    }
    // Same sibling-checkout layout `scripts/codec_paths.py` resolves, walked without a python
    // dependency so this test doesn't require python3 to prove a memory property.
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut anc = Some(root.as_path());
    while let Some(dir) = anc {
        let cand = dir.join("s2.cpp/models/s2-pro-q6_k.gguf");
        if cand.is_file() {
            return Some(cand);
        }
        anc = dir.parent();
    }
    None
}

/// Peak RSS ever reached by this process, in KiB, from `/proc/self/status`'s `VmHWM` line. Peak,
/// not current, on purpose: freeing a decoded tensor between checkpoints must not hide a transient
/// blow-up, and an eager loader's failure mode here IS a transient peak (OOM or swap thrash) before
/// steady state.
fn vm_hwm_kib() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            return rest.split_whitespace().next()?.parse().ok();
        }
    }
    None
}

#[test]
fn touching_a_handful_of_tensors_stays_far_below_an_eager_load() {
    let Some(gguf) = gguf_path() else {
        eprintln!("skip: no GGUF available (set $S2_GGUF to force)");
        return;
    };
    let Some(baseline_kib) = vm_hwm_kib() else {
        eprintln!("skip: /proc/self/status VmHWM unreadable (non-Linux?)");
        return;
    };

    let w = npu_s2::ar::ArWeights::open(&gguf).expect("ArWeights::open");
    let after_open_kib = vm_hwm_kib().unwrap();

    // Multiple layers from BOTH stacks, plus the small top-level tensors and a couple of embedding
    // row-range reads -- the pattern a real AR step would touch, not the whole model.
    let mut held = Vec::new();
    for il in [0usize, 17, 35] {
        held.push(w.slow_layer(il).expect("slow_layer"));
    }
    for il in [0usize, 3] {
        held.push(w.fast_layer(il).expect("fast_layer"));
    }
    let _norm = w.norm().expect("norm");
    let _fast_norm = w.fast_norm().expect("fast_norm");
    let _fast_output = w.fast_output().expect("fast_output");
    let _emb = w.embedding_rows(0, 8).expect("embedding_rows");
    let _cb = w.codebook_embedding_rows(0, 8).expect("codebook_embedding_rows");
    let after_touch_kib = vm_hwm_kib().unwrap();

    let gib = |kib: u64| kib as f64 / (1024.0 * 1024.0);
    println!(
        "VmHWM: baseline={:.2} GiB, after ArWeights::open={:.2} GiB, after touching 5 layers \
         + 3 top-level + 2 row-range reads={:.2} GiB",
        gib(baseline_kib), gib(after_open_kib), gib(after_touch_kib)
    );

    // An eager whole-model decode needs ~17.0 GiB of f32 arrays ON TOP of the ~4.2 GiB the raw
    // checkpoint bytes already occupy once opened -- ~21 GiB, past this box's ~18 GiB available.
    // 5 touched layers are at most 5*403 MiB=~2 GiB of decoded f32; 10 GiB leaves a wide margin
    // above that while staying far short of the eager figure, so this bound distinguishes "lazy"
    // from "eager" without being sensitive to incidental allocator/test-harness overhead.
    const EAGER_LOAD_KIB: u64 = 21 * 1024 * 1024;
    const LAZY_BOUND_KIB: u64 = 10 * 1024 * 1024;
    assert!(
        after_touch_kib < LAZY_BOUND_KIB,
        "peak RSS after touching a handful of tensors was {:.2} GiB, expected < {:.1} GiB -- \
         an eager full-model load would need ~{:.1} GiB",
        gib(after_touch_kib), gib(LAZY_BOUND_KIB), gib(EAGER_LOAD_KIB)
    );
    drop(held);
}
