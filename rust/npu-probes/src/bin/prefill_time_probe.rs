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
    println!("{:>6} {:>10} {:>12} {:>12} {:>12}", "P", "batched", "ms_median", "ms/token", "dispatches");

    let vocab = 151936u32;
    for &p in &lens {
        // One untimed pass first: the first dispatch after a load pays a cost the steady state
        // does not, and mixing it into the median measures the load, not the prefill.
        step.reset().expect("reset");
        let ids = prompt_ids(p, vocab);
        let _ = step.prefill(&ids, 0).expect("warmup prefill");

        let mut ms = Vec::with_capacity(reps);
        let mut primed_n = 0usize;
        for _ in 0..reps {
            step.reset().expect("reset KV between reps");
            let t = Instant::now();
            let primed = step.prefill(&ids, 0).expect("prefill");
            for (i, &tok) in ids[primed..].iter().enumerate() {
                step.step(tok, primed + i).expect("prime step");
            }
            ms.push(t.elapsed().as_secs_f64() * 1e3);
            primed_n = primed;
        }
        let m = median(&mut ms);
        let disp = match batched {
            Some(b) => primed_n.div_ceil(b) + (p - primed_n),
            None => p,
        };
        println!("{p:>6} {primed_n:>10} {m:>12.2} {:>12.3} {disp:>12}", m / p as f64);
    }
}
