//! How long does priming a prompt actually take, batched vs one token at a time?
//!
//! The rail has no whole-stack prefill number: the ~90x in the architecture note is a COST MODEL,
//! not a measurement. This times the thing itself -- the complete priming phase, `prefill()` plus
//! whatever tail still goes through `step()` -- so the two arms are compared on identical work.
//!
//! The flag is resolved once per process, so an arm is a process. Run the two ALTERNATED and take
//! paired medians: this box drifts ~33% over hours, so an unpaired A-then-B is not a measurement
//! (this box drifts ~33% over hours; quote a delta against a contemporaneous alternated
//! control, and name the power mode). `--reps` is per invocation; alternation is the caller's job
//! (scripts/time_prefill.sh does it).
//!
//! NPU is single-tenant -- stop `npu serve` and serialise against any other device user.
//!
//! Usage: prefill_time_probe <decode_dir> <prefill_dir> [--reps N] [--lens 256,512,1024]

use std::path::Path;
use std::rc::Rc;
use std::time::Instant;

use npu_engine::llm::generator::DecodeStep;
use npu_engine::llm::npu_decode::NpuDecodeStep;
use npu_xrt::Device;

fn prompt_ids(len: usize, vocab: u32) -> Vec<u32> {
    (0..len).map(|i| ((i as u64 * 7919 + 1234) % (vocab as u64 - 1)) as u32 + 1).collect()
}

fn median(v: &mut Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) / 2.0 }
}

fn main() {
    let mut a = std::env::args().skip(1);
    let decode_dir = a.next().expect("usage: prefill_time_probe <decode_dir> <prefill_dir>");
    let prefill_dir = a.next().expect("usage: prefill_time_probe <decode_dir> <prefill_dir>");
    let mut reps = 3usize;
    let mut lens = vec![256usize, 512, 1024];
    while let Some(f) = a.next() {
        match f.as_str() {
            "--reps" => reps = a.next().and_then(|s| s.parse().ok()).unwrap_or(reps),
            "--lens" => lens = a.next().unwrap().split(',').filter_map(|s| s.parse().ok()).collect(),
            other => panic!("unknown flag {other}"),
        }
    }

    let dev = Rc::new(Device::open(0).expect("open NPU (single-tenant -- use npu_lock.sh)"));
    let mut step = NpuDecodeStep::with_prefill(&dev, Path::new(&decode_dir), Path::new(&prefill_dir))
        .expect("load decode+prefill pair into one arena");
    let batched = step.prefill_batch();
    let arm = if batched.is_some() { "batched" } else { "pertok" };
    println!("arm={arm}  prefill_batch={batched:?}  reps={reps}");
    // `dispatches` used to be `primed.div_ceil(batch) + stepwise` -- a MODEL of the count printed
    // under a header that reads as an observation, which is what a per-submission time was then
    // divided out of. npu_xrt::dispatch_log already counts and times every blocking dispatch
    // together, tagged by kernel (method-time-every-dispatch rule 1), so take both from it.
    let split = std::env::var("NPU_DISPATCH_LOG").map(|v| v != "0").unwrap_or(false);
    if !split {
        eprintln!("[time] NPU_DISPATCH_LOG unset: dispatch count/ms columns will read 0. \
                   Set NPU_DISPATCH_LOG=1 to measure per-submission time.");
    }
    println!("{:>6} {:>10} {:>12} {:>12} {:>10} {:>12} {:>12}",
             "P", "batched", "ms_median", "ms/token", "disp_n", "disp_ms_tot", "ms/disp_max");

    let vocab = 151936u32;
    for &p in &lens {
        // One untimed pass first: the first dispatch after a load pays a cost the steady state
        // does not, and mixing it into the median measures the load, not the prefill.
        step.reset().expect("reset");
        let ids = prompt_ids(p, vocab);
        let _ = step.prefill(&ids, 0).expect("warmup prefill");

        let mut ms = Vec::with_capacity(reps);
        let mut primed_n = 0usize;
        let mut snap: Vec<(String, u32, f64)> = Vec::new();
        for _ in 0..reps {
            step.reset().expect("reset KV between reps");
            // Zero AFTER reset: reset itself dispatches on some paths, and booking those here
            // would inflate the prefill's own per-submission time.
            npu_xrt::dispatch_log::reset();
            let t = Instant::now();
            let primed = step.prefill(&ids, 0).expect("prefill");
            for (i, &tok) in ids[primed..].iter().enumerate() {
                step.step(tok, primed + i).expect("prime step");
            }
            ms.push(t.elapsed().as_secs_f64() * 1e3);
            primed_n = primed;
            snap = npu_xrt::dispatch_log::per_kernel_snapshot();
        }
        let m = median(&mut ms);
        let disp_n: u32 = snap.iter().map(|(_, n, _)| *n).sum();
        let disp_ms: f64 = snap.iter().map(|(_, _, s)| *s).sum::<f64>() * 1e3;
        // The gate is the SLOWEST submission against the 2 s TDR, not the mean -- a mean hides
        // exactly the outlier that would trip it. Per-kernel mean is the best bound this log
        // gives (it sums, it does not keep a max), so report it as a LOWER bound and say so.
        let per_disp_max = snap.iter()
            .map(|(_, n, s)| if *n > 0 { s * 1e3 / *n as f64 } else { 0.0 })
            .fold(0.0f64, f64::max);
        println!("{p:>6} {primed_n:>10} {m:>12.2} {:>12.3} {disp_n:>10} {disp_ms:>12.1} {per_disp_max:>12.2}",
                 m / p as f64);
        for (k, n, s) in &snap {
            println!("{:>6} {:>10} {:>12} {:>12} {:>10} {:>12.1} {:>12.2}  {k}",
                     "", "", "", "", n, s * 1e3, if *n > 0 { s * 1e3 / *n as f64 } else { 0.0 });
        }
    }
}
