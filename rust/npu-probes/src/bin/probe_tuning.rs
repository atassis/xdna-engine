//! Single-box tuning probe. Run from repo root, NPU idle (stop other ASR/embeddings services).
//! For each boolean knob: A/B the GigaAM encoder (knob on vs off) and report the e2e delta, so each
//! knob's class can be confirmed. Also feeds profiles/xdna2.toml.
//! Usage: probe_tuning [iters]   (default 20)
//!
//! Cross-box variance is OUT OF SCOPE (needs >=2 machines). The CPU<->NPU crossover and the
//! weight-swap-vs-reload timing are follow-on measurements (see the spec) — this binary covers the
//! per-knob A/B classification, which is what one box can settle.
use std::path::Path;
use std::rc::Rc;
use std::time::Instant;

use ndarray::Array2;
use npu_asr::ctx2::Precision;
use npu_asr::encoder::Encoder;
use npu_asr::tuning::TuningConfig;
use npu_asr::weights::WeightStore;
use npu_xrt::Device;

fn time_encoder(
    dev: Rc<Device>, root: &Path, ws: &WeightStore, cfg: &TuningConfig,
    x0: &Array2<f32>, valid: usize, iters: usize,
) -> f64 {
    let enc = Encoder::new_with_tuning(dev, root, ws, ws.nblocks(), cfg);
    let mut best = f64::INFINITY;
    for _ in 0..iters {
        let t = Instant::now();
        let _ = enc.forward_last(x0, valid);
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
    }
    best
}

fn main() {
    let iters: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(20);
    let root = Path::new(".");
    let ws = WeightStore::load(&root.join("artifacts/encoder")).expect("encoder weights");
    let dev = Rc::new(Device::open(0).expect("open NPU (stop other services first)"));
    let x0 = Array2::<f32>::zeros((400, 768));
    let valid = 400usize;
    let base = TuningConfig::baked_default(Precision::FastBf16);

    let knobs: Vec<(&str, fn(TuningConfig) -> TuningConfig)> = vec![
        ("modal_epilogue",   |mut c| { c.modal_epilogue = !c.modal_epilogue; c }),
        ("subsample_on_npu", |mut c| { c.subsample_on_npu = !c.subsample_on_npu; c }),
        ("layernorm_on_npu", |mut c| { c.layernorm_on_npu = !c.layernorm_on_npu; c }),
        ("glu_fused",        |mut c| { c.glu_fused = !c.glu_fused; c }),
        ("qkv_overlap",      |mut c| { c.qkv_overlap = !c.qkv_overlap; c }),
        ("mm2_pipeline",     |mut c| { c.mm2_pipeline = !c.mm2_pipeline; c }),
    ];

    let base_ms = time_encoder(dev.clone(), root, &ws, &base, &x0, valid, iters);
    println!("baseline (all baked defaults) = {base_ms:.1} ms  (iters={iters}, min-of-N)");
    println!("knob                 flipped_ms   delta_ms   verdict");
    for (name, flip) in knobs {
        let b = time_encoder(dev.clone(), root, &ws, &base, &x0, valid, iters);
        let f = time_encoder(dev.clone(), root, &ws, &flip(base), &x0, valid, iters);
        let delta = f - b;
        let verdict = if delta > 1.0 { "DEFAULT WINS (keep baked)" }
                      else if delta < -1.0 { "FLIP WINS (revisit default!)" }
                      else { "neutral (Class 0 either way)" };
        println!("{name:<20} {f:>9.1}  {delta:>+9.1}   {verdict}");
    }
    println!("\nAccuracy: re-run `verify_encoder` for any knob whose flip changes output; record numbers in a internal notes note.");
}
