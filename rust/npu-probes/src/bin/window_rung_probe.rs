//! Device gate for window rungs: does an ELF carrying N named control codes produce the SAME
//! tokens as the single-window build, ACROSS a rung boundary?
//!
//! A rung is a narrower attention window served by its own `aie.runtime_sequence` in the same full
//! ELF, selected by `main:<name>` on the one registered hw_context. It quantises the SHIM's KV
//! fill; the core keeps its runtime window. So a rung must be arithmetically invisible: the same
//! positions are attended either way, only fewer bytes are streamed to attend them.
//!
//! THE GATE HAS TO PROVE THE ARM CHANGED. Comparing two artifacts that both ran `main:sequence`
//! would pass while testing nothing, which is exactly how a rung set that was silently ignored
//! would look. So this asserts the selected kernel NAME differs side to side at the crossing
//! before it compares any token.
//!
//! NPU is single-tenant: stop `npu serve` and serialise against any other device user before
//! running, or the timing below measures contention rather than the arms.
//!
//! Usage: window_rung_probe <rung_decode_dir> <control_decode_dir> [--prime N] [--gen N]

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

/// A deterministic synthetic prompt. What this gate tests is that two BUILDS agree, and for that
/// any fixed id sequence in range does; real text would drag in the tokenizer for nothing.
fn prompt_ids(len: usize, vocab: u32) -> Vec<u32> {
    (0..len).map(|i| ((i as u64 * 7919 + 1234) % (vocab as u64 - 1)) as u32 + 1).collect()
}

/// Prime `prime` positions, then free-run `gen` greedy tokens. Returns the tokens and the
/// `(window, kernel)` actually selected at the first and last generated position.
fn run(step: &mut NpuDecodeStep, prime: usize, gen: usize, vocab: u32)
    -> (Vec<u32>, (usize, String), (usize, String)) {
    step.reset().expect("reset KV");
    let ids = prompt_ids(prime, vocab);
    for (i, &t) in ids.iter().enumerate().take(prime - 1) {
        step.step(t, i).expect("prime step");
    }
    let mut pos = prime - 1;
    let mut tok = ids[prime - 1];
    let first_arm = step.bucket_for(pos);
    let mut out = Vec::with_capacity(gen);
    for _ in 0..gen {
        let logits = step.step(tok, pos).expect("decode step");
        tok = argmax(&logits);
        out.push(tok);
        pos += 1;
    }
    let last_arm = step.bucket_for(pos - 1);
    (out, first_arm, last_arm)
}

/// ALTERNATED timing at a position the rung covers, which is where a rung is supposed to pay.
///
/// Alternated rather than arm-after-arm because this box drifts and the effect being measured is
/// smaller than the drift over a session (D027). One step per arm per round, medians reported with
/// the spread, so a reader can see whether the arms' distributions even separate.
fn timing(rung: &mut NpuDecodeStep, ctl: &mut NpuDecodeStep, vocab: u32) {
    const PRIME: usize = 200;
    const REPS: usize = 40;
    for s in [&mut *rung, &mut *ctl] {
        s.reset().expect("reset");
        let ids = prompt_ids(PRIME, vocab);
        for (i, &t) in ids.iter().enumerate().take(PRIME) {
            s.step(t, i).expect("prime");
        }
    }
    println!(
        "\nalternated timing at pos {PRIME}: rung arm {:?}, control arm {:?}",
        rung.bucket_for(PRIME).1,
        ctl.bucket_for(PRIME).1
    );
    let (mut rt, mut ct) = (Vec::new(), Vec::new());
    for _ in 0..REPS {
        for (s, out) in [(&mut *rung, &mut rt), (&mut *ctl, &mut ct)] {
            let t = std::time::Instant::now();
            s.step(1234, PRIME).expect("timed step");
            out.push(t.elapsed().as_secs_f64() * 1e3);
        }
    }
    let med = |v: &mut Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        (v[v.len() / 2], v[0], v[v.len() - 1])
    };
    let (rm, rlo, rhi) = med(&mut rt);
    let (cm, clo, chi) = med(&mut ct);
    println!("  rung    median {rm:.3} ms  [{rlo:.3}, {rhi:.3}]");
    println!("  control median {cm:.3} ms  [{clo:.3}, {chi:.3}]");
    println!(
        "  delta {:+.3} ms  {:+.1}%   modes {}",
        rm - cm,
        100.0 * (rm - cm) / cm,
        if rhi < clo || chi < rlo { "DISJOINT" } else { "overlap -- treat as indicative" }
    );
}

fn main() {
    let mut a = std::env::args().skip(1);
    let rung_dir = a.next().expect("usage: window_rung_probe <rung_dir> <control_dir>");
    let ctl_dir = a.next().expect("usage: window_rung_probe <rung_dir> <control_dir>");
    let (mut prime, mut gen) = (900usize, 200usize);
    while let Some(f) = a.next() {
        match f.as_str() {
            "--prime" => prime = a.next().and_then(|s| s.parse().ok()).unwrap_or(prime),
            "--gen" => gen = a.next().and_then(|s| s.parse().ok()).unwrap_or(gen),
            other => panic!("unknown flag {other}"),
        }
    }
    let vocab = 151936u32;

    let dev = Rc::new(Device::open(0).expect("open NPU (single-tenant -- serialise against other device users)"));
    let mut rung = NpuDecodeStep::new(&dev, Path::new(&rung_dir)).expect("load rung artifact");
    println!("rung buckets:    {:?}", rung.bucket_kernels());
    let mut ctl = NpuDecodeStep::new(&dev, Path::new(&ctl_dir)).expect("load control artifact");
    println!("control buckets: {:?}", ctl.bucket_kernels());

    let (r_tok, r_first, r_last) = run(&mut rung, prime, gen, vocab);
    let (c_tok, c_first, c_last) = run(&mut ctl, prime, gen, vocab);
    println!("positions {}..{}", prime - 1, prime - 2 + gen);
    println!("  rung    first {r_first:?}  last {r_last:?}");
    println!("  control first {c_first:?}  last {c_last:?}");

    // 1. The rung arm must actually CROSS: a different control code at the start than at the end.
    //    Without this the parity below could pass on an artifact whose rungs were never selected.
    assert_ne!(
        r_first.1, r_last.1,
        "no rung crossing in positions {}..{} -- widen --gen or lower --prime; this gate is \
         vacuous without one",
        prime - 1,
        prime - 2 + gen
    );
    // 2. The control must NOT cross -- it is one window, so its arm is constant. If it changed,
    //    the two sides are not the comparison this claims to be.
    assert_eq!(c_first.1, c_last.1, "control changed arms; it is not a single-window build");
    // 3. And only then, token identity.
    let bad: Vec<usize> = (0..gen).filter(|&i| r_tok[i] != c_tok[i]).collect();
    if bad.is_empty() {
        println!("PASS -- {gen} tokens identical across a rung crossing ({} -> {})", r_first.1, r_last.1);
        timing(&mut rung, &mut ctl, vocab);
    } else {
        println!("FAIL -- {} of {gen} tokens differ, first at index {}", bad.len(), bad[0]);
        for &i in bad.iter().take(8) {
            println!("   [{i}] rung {} control {}", r_tok[i], c_tok[i]);
        }
        std::process::exit(1);
    }
}
