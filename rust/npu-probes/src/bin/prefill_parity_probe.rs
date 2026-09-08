//! Device gate for batched prefill: does priming the KV cache in batches of M produce the SAME
//! tokens as priming it one position at a time?
//!
//! That is the gate the project's own rule names -- token identity at temperature 0, over several
//! prompt lengths INCLUDING ones that do not divide M. rel-L2 is a note, never a blocker.
//!
//! Run it TWICE, once with `NPU_LLM_PREFILL_BATCHED=0` (the per-token control) and once with it on,
//! and diff the two token streams. Two processes rather than two instances in one, because the env
//! flag is resolved once per process and because two `with_prefill` instances would hold four
//! hardware contexts and two 2 GB arenas for no reason.
//!
//! NPU is single-tenant -- run under `xdna-engine-private/journal/scripts/npu_lock.sh`.
//!
//! Usage: prefill_parity_probe <decode_dir> <prefill_dir> [--gen N] [--lens 64,255,256,257,600]

use std::path::Path;
use std::rc::Rc;

use npu_engine::llm::generator::DecodeStep;
use npu_engine::llm::npu_decode::NpuDecodeStep;
use npu_xrt::Device;

fn argmax(v: &[f32]) -> u32 {
    let mut best = (0usize, f32::NEG_INFINITY);
    for (i, &x) in v.iter().enumerate() {
        if x > best.1 {
            best = (i, x);
        }
    }
    best.0 as u32
}

/// A deterministic synthetic prompt. Real text would need the tokenizer; what this gate tests is
/// that two priming STRATEGIES agree, and for that any fixed id sequence in range does.
fn prompt_ids(len: usize, vocab: u32) -> Vec<u32> {
    (0..len).map(|i| ((i as u64 * 7919 + 1234) % (vocab as u64 - 1)) as u32 + 1).collect()
}

fn main() {
    let mut a = std::env::args().skip(1);
    let decode_dir = a.next().expect("usage: prefill_parity_probe <decode_dir> <prefill_dir>");
    let prefill_dir = a.next().expect("usage: prefill_parity_probe <decode_dir> <prefill_dir>");
    let mut n_gen = 8usize;
    let mut lens = vec![64usize, 255, 256, 257, 600];
    while let Some(f) = a.next() {
        match f.as_str() {
            "--gen" => n_gen = a.next().and_then(|s| s.parse().ok()).unwrap_or(n_gen),
            "--lens" => {
                lens = a.next().unwrap().split(',').filter_map(|s| s.parse().ok()).collect()
            }
            other => panic!("unknown flag {other}"),
        }
    }

    let dev = Rc::new(Device::open(0).expect("open NPU (single-tenant -- use npu_lock.sh)"));
    let mut step = NpuDecodeStep::with_prefill(&dev, Path::new(&decode_dir), Path::new(&prefill_dir))
        .expect("load decode+prefill pair into one arena");
    let batched = step.prefill_batch();
    println!("prefill_batch = {batched:?}   (None = per-token control arm)");

    let vocab = 151936u32; // qwen3-0.6b; only used to keep synthetic ids in range
    for &p in &lens {
        step.reset().expect("reset KV between prompt lengths");
        let ids = prompt_ids(p, vocab);
        // Prime everything but the last token; the last token is what `step` samples from.
        let primed = step.prefill(&ids[..p - 1]).expect("prefill");
        for (i, &t) in ids[primed..p - 1].iter().enumerate() {
            step.step(t, primed + i).expect("prime step");
        }
        let mut pos = p - 1;
        let mut tok = ids[p - 1];
        let mut out = Vec::with_capacity(n_gen);
        let mut first: Option<Vec<f32>> = None;
        for _ in 0..n_gen {
            let logits = step.step(tok, pos).expect("decode step");
            if first.is_none() {
                first = Some(logits.clone());
            }
            tok = argmax(&logits);
            out.push(tok);
            pos += 1;
        }
        let div = if batched.map_or(false, |m| p % m == 0) { "exact" } else { "ragged" };
        // The FIRST step after priming isolates the KV state from the free-running cascade: one
        // argmax flip there explains every later token, so comparing logits at step 0 is what
        // separates "the caches differ" from "one knife-edge tie went the other way".
        if let Ok(dir) = std::env::var("DUMP_LOGITS_DIR") {
            let arm = if batched.is_some() { "batched" } else { "pertok" };
            let f = first.as_ref().unwrap();
            let mut bytes = Vec::with_capacity(f.len() * 4);
            for v in f {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            std::fs::write(format!("{dir}/logits_{arm}_P{p}.bin"), &bytes).expect("dump logits");
        }
        println!(
            "P={p:<5} primed_batched={primed:<5} {div:<7} tokens={:?}",
            out
        );
    }
}
