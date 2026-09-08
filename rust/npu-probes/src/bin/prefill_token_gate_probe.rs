//! TIER 2 for the BATCHED PREFILL path: greedy-decode from a prompt primed in batches, and emit
//! the same JSON `scripts/gate_token_set.py` judges.
//!
//! Why this exists at all: `gate_llm.sh --tier2` drives `verify_llm_decode.py`, which says in its
//! own header "the graph is decode-only: there is no prefill". So the end-to-end gate the rail
//! adopted has never run the prefill path, and flipping `NPU_LLM_PREFILL_BATCHED` on the strength
//! of it would be K019 -- a gate passing without touching its subject -- applied to a product
//! default. This closes that.
//!
//! It drives the PRODUCTION types (`NpuDecodeStep::with_prefill`, the same `prefill()`/`step()` the
//! engine calls), not a re-typed runlist, so what passes here is what ships. The arm is whatever
//! `NPU_LLM_PREFILL_BATCHED` selects, which makes the per-token arm a control run by the same
//! binary over the same references.
//!
//! One prompt length per invocation, because a reference is per length and because the gate must
//! be able to name WHICH length failed. `reset()` re-zeroes every KV buffer between runs.
//!
//! SCOPE, and it is deliberately STRICTER than production: this calls `DecodeStep::prefill` direct,
//! so it batches at EVERY length, while `LlmGenerator` gates batching behind
//! `PREFILL_MIN_CHUNKS` and today primes anything under `M` per token. So a P=64 run here exercises
//! a geometry the shipped policy never reaches -- 3 real rows of padding for every real one. That is
//! the right way round for a gate: it holds if the policy is later loosened, which the measured
//! break-even argues it should be.
//!
//! NPU is single-tenant -- run under `xdna-engine-private/journal/scripts/npu_lock.sh`.
//!
//! Takes ONE OR MORE references per invocation and loops them inside a single process: binding the
//! pair costs a 1.99 GB arena upload, and paying that once per prompt length would dominate the run.
//!
//! Usage: prefill_token_gate_probe <decode_dir> <prefill_dir> --outdir <d> --ref <a.json> [--ref b.json ...]

use std::path::Path;
use std::rc::Rc;

use npu_engine::llm::generator::DecodeStep;
use npu_engine::llm::npu_decode::NpuDecodeStep;
use npu_xrt::Device;

/// Top-k indices, highest logit first. Not a full sort: k is 5 and the vocab is 151936.
fn topk(v: &[f32], k: usize) -> Vec<u32> {
    let mut idx: Vec<u32> = (0..v.len() as u32).collect();
    idx.select_nth_unstable_by(k.min(v.len()) - 1, |&a, &b| {
        v[b as usize].partial_cmp(&v[a as usize]).unwrap()
    });
    idx.truncate(k.min(v.len()));
    idx.sort_by(|&a, &b| v[b as usize].partial_cmp(&v[a as usize]).unwrap());
    idx
}

/// Minimal field reads off the reference JSON. A dependency on a JSON crate is not worth it for
/// two integer arrays, but a WRONG read silently gates against the wrong prompt -- so every
/// accessor here fails loudly rather than defaulting.
fn json_ints(src: &str, key: &str) -> Vec<u32> {
    let at = src
        .find(&format!("\"{key}\""))
        .unwrap_or_else(|| panic!("reference JSON has no `{key}`"));
    let open = src[at..].find('[').expect("`{key}` is not an array") + at;
    let close = src[open..].find(']').expect("unterminated array") + open;
    src[open + 1..close]
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().unwrap_or_else(|_| panic!("non-integer in `{key}`: {s:?}")))
        .collect()
}

fn json_int(src: &str, key: &str) -> usize {
    let at = src
        .find(&format!("\"{key}\""))
        .unwrap_or_else(|| panic!("reference JSON has no `{key}`"));
    let colon = src[at..].find(':').expect("malformed") + at;
    let rest = &src[colon + 1..];
    let end = rest.find(|c: char| c == ',' || c == '}').expect("malformed");
    rest[..end].trim().parse().expect("non-integer")
}

fn main() {
    let mut a = std::env::args().skip(1);
    let decode_dir = a.next().expect("usage: prefill_token_gate_probe <decode_dir> <prefill_dir>");
    let prefill_dir = a.next().expect("usage: prefill_token_gate_probe <decode_dir> <prefill_dir>");
    let mut refs: Vec<String> = Vec::new();
    let mut outdir = String::new();
    while let Some(f) = a.next() {
        match f.as_str() {
            "--ref" => refs.push(a.next().expect("--ref needs a path")),
            "--outdir" => outdir = a.next().expect("--outdir needs a path"),
            other => panic!("unknown flag {other}"),
        }
    }
    assert!(!refs.is_empty() && !outdir.is_empty(), "--ref (>=1) and --outdir are both required");

    let dev = Rc::new(Device::open(0).expect("open NPU (single-tenant -- use npu_lock.sh)"));
    let mut step = NpuDecodeStep::with_prefill(&dev, Path::new(&decode_dir), Path::new(&prefill_dir))
        .expect("load decode+prefill pair into one arena");
    let batched = step.prefill_batch();
    let arm = if batched.is_some() { "batched" } else { "pertok" };
    eprintln!("[gate] arm={arm} prefill_batch={batched:?} refs={}", refs.len());

    for ref_path in &refs {
    let src = std::fs::read_to_string(ref_path).expect("read reference JSON");
    let prompt_ids = json_ints(&src, "prompt_ids");
    let n_tokens = json_int(&src, "n_tokens");
    let k = json_int(&src, "k");
    let p = prompt_ids.len();
    assert!(p >= 2, "a prompt of {p} token(s) has nothing to prime");
    let stem = Path::new(ref_path).file_stem().unwrap().to_string_lossy().to_string();
    let out_path = format!("{outdir}/{stem}_arm{}.json", if batched.is_some() { 1 } else { 0 });

    // Every length starts from a zeroed cache: without this, length N+1 would decode against
    // length N's history and the gate would be measuring the reset, not the prefill.
    step.reset().expect("reset KV before the run");
    // Prime everything but the LAST prompt token: prefill emits no logits, so that one goes through
    // the decode ELF -- which is also what leaves the KV in the state P sequential steps would.
    let primed = step.prefill(&prompt_ids[..p - 1]).expect("prefill");
    for (i, &t) in prompt_ids[primed..p - 1].iter().enumerate() {
        step.step(t, primed + i).expect("prime step");
    }


    // FREE-RUNNING: feed back the device's own argmax. This is what the adopted rule judges, and
    // it judges exactly ONE step -- the first divergence -- because after that the two runs are on
    // different trajectories and a comparison would measure the trajectory.
    let mut gen_ids = Vec::with_capacity(n_tokens);
    let mut topk_ids = Vec::with_capacity(n_tokens);
    let mut tok = prompt_ids[p - 1];
    let mut pos = p - 1;
    for _ in 0..n_tokens {
        let logits = step.step(tok, pos).expect("decode step");
        let tk = topk(&logits, k);
        tok = tk[0];
        gen_ids.push(tok);
        topk_ids.push(tk);
        pos += 1;
    }

    // TEACHER-FORCED: re-run the same prompt, but feed the REFERENCE's tokens. Every step then sees
    // the state the reference saw, so all n_tokens are independently comparable instead of one --
    // which is what turns "the first divergence was survivable" into "no step's distribution has
    // moved". Emitted alongside, never instead: the free-running arm is the adopted rule.
    let ref_gen = json_ints(&src, "gen_ids");
    let mut tf_topk: Vec<Vec<u32>> = Vec::with_capacity(n_tokens);
    if !ref_gen.is_empty() {
        step.reset().expect("reset KV before the teacher-forced pass");
        let primed_tf = step.prefill(&prompt_ids[..p - 1]).expect("prefill (teacher-forced)");
        for (i, &t) in prompt_ids[primed_tf..p - 1].iter().enumerate() {
            step.step(t, primed_tf + i).expect("prime step (teacher-forced)");
        }
        let mut tok = prompt_ids[p - 1];
        let mut pos = p - 1;
        for i in 0..n_tokens.min(ref_gen.len()) {
            let logits = step.step(tok, pos).expect("decode step (teacher-forced)");
            tf_topk.push(topk(&logits, k));
            tok = ref_gen[i]; // the REFERENCE's token, not ours
            pos += 1;
        }
    }

    let arr = |v: &[u32]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", ");
    let mut out = String::new();
    out.push_str("{\n");
    out.push_str(&format!("  \"spec\": \"qwen3-0.6b\",\n  \"backend\": \"npu:{arm}\",\n"));
    out.push_str(&format!("  \"prompt\": \"<pinned P={p}>\",\n"));
    out.push_str(&format!("  \"prompt_ids\": [{}],\n", arr(&prompt_ids)));
    out.push_str(&format!("  \"n_tokens\": {n_tokens},\n  \"k\": {k},\n"));
    out.push_str(&format!("  \"primed_batched\": {primed},\n"));
    out.push_str(&format!("  \"gen_ids\": [{}],\n", arr(&gen_ids)));
    out.push_str("  \"topk_ids\": [\n");
    for (i, t) in topk_ids.iter().enumerate() {
        out.push_str(&format!("    [{}]{}\n", arr(t), if i + 1 < topk_ids.len() { "," } else { "" }));
    }
    out.push_str("  ],\n  \"teacher_forced_topk_ids\": [\n");
    for (i, t) in tf_topk.iter().enumerate() {
        out.push_str(&format!("    [{}]{}\n", arr(t), if i + 1 < tf_topk.len() { "," } else { "" }));
    }
    out.push_str("  ]\n}\n");
    std::fs::write(&out_path, out).expect("write npu JSON");
    eprintln!("[gate] P={p} primed={primed} -> {out_path}");
    }
}
