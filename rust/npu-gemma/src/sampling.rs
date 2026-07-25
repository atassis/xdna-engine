//! Token-level sampling over a logits vector: temperature, top-k, top-p (nucleus), and
//! repetition/frequency/presence penalties. Host-CPU, model-agnostic (any `[vocab]` logits slice --
//! the Gemma lm-head output, on-NPU or host).
//!
//! **Default is GREEDY** (`SamplingConfig::default()` has `temperature: 0.0`), matching the shipped
//! decode baseline (`scripts/gemma_ref_generate.py` generates with `do_sample=False`). `sample()`
//! short-circuits straight to [`argmax`] when `temperature <= 0.0` -- no RNG draw, no penalty/top-k/
//! top-p pass, byte-for-byte the same greedy path this crate already validates against the host
//! oracle. Flipping the shipped decode default from greedy to sampling is an OWNER-GATED product
//! decision; this module only adds the capability.
//!
//! Pipeline order (mirrors the common llama.cpp/HF convention -- reshape the distribution AFTER
//! penalizing already-seen tokens):
//! 1. repetition (CTRL-style, multiplicative) / frequency / presence (OpenAI-style, additive) penalties
//!    over the token history
//! 2. temperature scaling
//! 3. top-k filter
//! 4. top-p (nucleus) filter
//! 5. softmax -> categorical draw (inverse-CDF) via the seeded [`SplitMix64`] PRNG
//!
//! All internal math (penalties, softmax, the CDF walk) runs in f64: `f32` logits/config are promoted
//! to f64 immediately at the API boundary and stay there, matching the host-reference policy in
//! `scripts/gemma_sampling_ref.py` bit-for-bit (same PRNG, same operation order, same libm `exp`) --
//! that script is the correctness oracle the tests below bake in as fixed-seed expected constants.

use std::collections::HashMap;

/// One sampling configuration. `Default` is greedy (temperature 0, all filters/penalties disabled).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingConfig {
    /// `<= 0.0` means greedy argmax (no sampling). Otherwise divides logits before filtering.
    pub temperature: f32,
    /// 0 disables top-k filtering (keep all candidates).
    pub top_k: usize,
    /// `>= 1.0` disables nucleus filtering (keep all candidates).
    pub top_p: f32,
    /// CTRL-style multiplicative penalty over tokens already in the history. `1.0` = no-op.
    pub repetition_penalty: f32,
    /// OpenAI-style additive penalty scaled by how many times a token has appeared. `0.0` = no-op.
    pub frequency_penalty: f32,
    /// OpenAI-style additive penalty applied once per token that has appeared at all. `0.0` = no-op.
    pub presence_penalty: f32,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        SamplingConfig {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
        }
    }
}

/// Deterministic 64-bit PRNG (Vigna's splitmix64) -- chosen over pulling in the `rand` crate so the
/// correctness oracle (`scripts/gemma_sampling_ref.py`) can reproduce it bit-for-bit in ~10 lines of
/// pure Python with no numpy/torch dependency.
#[derive(Debug, Clone, Copy)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        SplitMix64 { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform draw in `[0, 1)` from the top 53 bits (standard PRNG-to-float recipe).
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

/// Index of the largest value (strict `>`, first wins on ties -- matches the host-argmax idiom used
/// by the on-NPU decode gate elsewhere in this engine).
pub fn argmax(logits: &[f32]) -> u32 {
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    best_i as u32
}

/// Numerically-stable softmax in f64 (see module docs for why f64 throughout).
fn softmax_f64(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits
        .iter()
        .map(|&v| if v.is_finite() { (v - max).exp() } else { 0.0 })
        .collect();
    let sum: f64 = exps.iter().sum();
    exps.into_iter().map(|e| e / sum).collect()
}

/// Apply repetition/frequency/presence penalties over tokens present in `history`, in place. No-op
/// (and no history scan) when all three are at their identity value.
fn apply_penalties(logits: &mut [f64], history: &[u32], cfg: &SamplingConfig) {
    let (rep, freq, pres) = (cfg.repetition_penalty as f64, cfg.frequency_penalty as f64, cfg.presence_penalty as f64);
    if rep == 1.0 && freq == 0.0 && pres == 0.0 {
        return;
    }
    let mut counts: HashMap<u32, u32> = HashMap::new();
    for &t in history {
        *counts.entry(t).or_insert(0) += 1;
    }
    for (tok, count) in counts {
        if let Some(l) = logits.get_mut(tok as usize) {
            if rep != 1.0 {
                *l = if *l > 0.0 { *l / rep } else { *l * rep };
            }
            *l -= freq * count as f64;
            *l -= pres;
        }
    }
}

/// Keep only the `k` largest logits; mask the rest to `-inf`. `k == 0` disables the filter.
fn filter_top_k(logits: &mut [f64], k: usize) {
    if k == 0 || k >= logits.len() {
        return;
    }
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_unstable_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    for &i in &idx[k..] {
        logits[i] = f64::NEG_INFINITY;
    }
}

/// Nucleus filter: keep the smallest prefix (by descending probability) whose cumulative mass is
/// `>= top_p`; mask the rest to `-inf`. `top_p >= 1.0` disables the filter.
fn filter_top_p(logits: &mut [f64], top_p: f64) {
    if top_p >= 1.0 {
        return;
    }
    let probs = softmax_f64(logits);
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_unstable_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
    let mut cum = 0.0f64;
    let mut cutoff = idx.len();
    for (pos, &i) in idx.iter().enumerate() {
        cum += probs[i];
        if cum >= top_p {
            cutoff = pos + 1;
            break;
        }
    }
    for &i in &idx[cutoff..] {
        logits[i] = f64::NEG_INFINITY;
    }
}

/// Draw one token id from `logits` given the decode `history` (token ids generated so far, used only
/// by the penalties). Greedy (argmax, no RNG draw) when `cfg.temperature <= 0.0`; otherwise runs the
/// full penalty -> temperature -> top-k -> top-p -> softmax -> inverse-CDF pipeline.
pub fn sample(logits: &[f32], history: &[u32], cfg: &SamplingConfig, rng: &mut SplitMix64) -> u32 {
    if cfg.temperature <= 0.0 {
        return argmax(logits);
    }
    let mut work: Vec<f64> = logits.iter().map(|&v| v as f64).collect();
    apply_penalties(&mut work, history, cfg);
    let temperature = cfg.temperature as f64;
    for v in work.iter_mut() {
        *v /= temperature;
    }
    filter_top_k(&mut work, cfg.top_k);
    filter_top_p(&mut work, cfg.top_p as f64);
    let probs = softmax_f64(&work);
    let u = rng.next_f64();
    let mut cum = 0.0f64;
    for (i, &p) in probs.iter().enumerate() {
        cum += p;
        if u < cum {
            return i as u32;
        }
    }
    (probs.len() - 1) as u32 // float edge case: u landed in the last-ulp gap
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mirrored 1:1 in scripts/gemma_sampling_ref.py -- keep both in sync if either changes.
    const SAMPLE_LOGITS: [f32; 8] = [1.0, 3.0, 2.0, 0.5, 4.0, -1.0, 2.5, 0.0];

    #[test]
    fn greedy_default_is_argmax_and_ignores_rng_and_history() {
        // The shipped-default gate: SamplingConfig::default() must be greedy.
        let cfg = SamplingConfig::default();
        assert_eq!(cfg.temperature, 0.0, "default must stay greedy (temperature 0) -- do not flip");
        let mut rng = SplitMix64::new(123);
        let tok = sample(&SAMPLE_LOGITS, &[1, 1, 4, 4, 4], &cfg, &mut rng);
        assert_eq!(tok, argmax(&SAMPLE_LOGITS));
        assert_eq!(tok, 4); // index of the 4.0 logit
    }

    #[test]
    fn argmax_first_wins_on_ties() {
        assert_eq!(argmax(&[1.0, 2.0, 2.0, 0.0]), 1);
    }

    #[test]
    fn top_k_one_collapses_to_the_argmax_token_for_any_seed() {
        // Only one candidate survives top_k=1, so the categorical draw is degenerate: every seed
        // must pick it. This is a strong sanity check independent of any host-reference cross-check.
        let cfg = SamplingConfig { temperature: 1.0, top_k: 1, ..SamplingConfig::default() };
        let want = argmax(&SAMPLE_LOGITS);
        for seed in [0u64, 1, 42, 999_999, u64::MAX] {
            let mut rng = SplitMix64::new(seed);
            assert_eq!(sample(&SAMPLE_LOGITS, &[], &cfg, &mut rng), want, "seed {seed}");
        }
    }

    #[test]
    fn top_k_filter_keeps_exactly_k_finite_entries() {
        let mut work: Vec<f64> = SAMPLE_LOGITS.iter().map(|&v| v as f64).collect();
        filter_top_k(&mut work, 3);
        let finite = work.iter().filter(|v| v.is_finite()).count();
        assert_eq!(finite, 3);
        // the 3 largest are indices 4 (4.0), 1 (3.0), 6 (2.5)
        for i in [4usize, 1, 6] {
            assert!(work[i].is_finite(), "index {i} should survive top_k=3");
        }
        for i in [0usize, 2, 3, 5, 7] {
            assert!(work[i].is_infinite(), "index {i} should be masked by top_k=3");
        }
    }

    #[test]
    fn top_p_filter_matches_hand_computed_nucleus() {
        // Two-candidate logits, well-separated softmax mass: [10.0, 0.0] -> probs ~ [0.9999546, 4.5e-5].
        let mut work = vec![10.0f64, 0.0];
        filter_top_p(&mut work, 0.99);
        assert!(work[0].is_finite());
        assert!(work[1].is_infinite(), "the near-zero-mass tail must be dropped at top_p=0.99");
    }

    #[test]
    fn repetition_penalty_shrinks_positive_logit_toward_zero() {
        let cfg = SamplingConfig { repetition_penalty: 2.0, ..SamplingConfig::default() };
        let mut work: Vec<f64> = vec![4.0, -4.0];
        apply_penalties(&mut work, &[0, 1], &cfg);
        assert!((work[0] - 2.0).abs() < 1e-12, "positive logit divided by penalty: {}", work[0]);
        assert!((work[1] - (-8.0)).abs() < 1e-12, "negative logit multiplied by penalty: {}", work[1]);
    }

    #[test]
    fn frequency_and_presence_penalties_are_additive_and_count_scaled() {
        let cfg = SamplingConfig { frequency_penalty: 0.5, presence_penalty: 0.1, ..SamplingConfig::default() };
        let mut work: Vec<f64> = vec![10.0, 10.0, 10.0];
        apply_penalties(&mut work, &[0, 0, 0, 1], &cfg); // token 0 x3, token 1 x1, token 2 unseen
        // expected values widened through f32 first (cfg fields are f32), matching apply_penalties'
        // own `as f64` boundary bit-for-bit -- comparing against raw f64 literals here would fail on
        // the ~1e-9 f32-rounding noise in 0.1f32, not a real bug.
        let (freq, pres) = (cfg.frequency_penalty as f64, cfg.presence_penalty as f64);
        assert!((work[0] - (10.0 - freq * 3.0 - pres)).abs() < 1e-12);
        assert!((work[1] - (10.0 - freq * 1.0 - pres)).abs() < 1e-12);
        assert!((work[2] - 10.0).abs() < 1e-12, "unseen token must be untouched");
    }

    #[test]
    fn identity_config_is_a_true_no_op_before_softmax() {
        let cfg = SamplingConfig::default();
        let mut work: Vec<f64> = SAMPLE_LOGITS.iter().map(|&v| v as f64).collect();
        let before = work.clone();
        apply_penalties(&mut work, &[1, 1, 4], &cfg);
        assert_eq!(work, before, "identity penalties must not touch logits at all");
    }

    /// Cross-language host-reference gate. Fixed seed, fixed logits/history/config; the expected
    /// token id is the literal stdout of `python3 scripts/gemma_sampling_ref.py`
    /// (`cross_lang_full_pipeline: seed=42 -> 6`). Re-run that script by hand if this ever needs
    /// re-deriving -- it is the oracle, not this test.
    #[test]
    fn cross_lang_full_pipeline_matches_python_oracle_seed_42() {
        let cfg = SamplingConfig {
            temperature: 0.8,
            top_k: 4,
            top_p: 0.9,
            repetition_penalty: 1.2,
            frequency_penalty: 0.3,
            presence_penalty: 0.1,
        };
        let mut rng = SplitMix64::new(42);
        let tok = sample(&SAMPLE_LOGITS, &[1, 1, 4], &cfg, &mut rng);
        assert_eq!(tok, 6, "must match scripts/gemma_sampling_ref.py's seed=42 oracle output");
    }

    /// Same gate, a second fixed seed/config pair, and an 8-draw sequence (checks the PRNG advances
    /// identically call-over-call, not just on the first draw). Oracle:
    /// `cross_lang_plain_softmax_seq: seed=7, n=8 -> [4, 0, 6, 4, 4, 2, 4, 4]`.
    #[test]
    fn cross_lang_plain_softmax_sequence_matches_python_oracle_seed_7() {
        let cfg = SamplingConfig { temperature: 1.0, top_k: 0, top_p: 1.0, ..SamplingConfig::default() };
        let mut rng = SplitMix64::new(7);
        let got: Vec<u32> = (0..8).map(|_| sample(&SAMPLE_LOGITS, &[], &cfg, &mut rng)).collect();
        assert_eq!(got, vec![4, 0, 6, 4, 4, 2, 4, 4]);
    }

    /// The greedy short-circuit path also matches the oracle (and, trivially, plain host argmax) --
    /// belt-and-suspenders for the "must not regress the shipped baseline" constraint.
    #[test]
    fn cross_lang_greedy_matches_python_oracle() {
        let cfg = SamplingConfig { temperature: 0.0, ..Default::default() };
        let mut rng = SplitMix64::new(0);
        assert_eq!(sample(&SAMPLE_LOGITS, &[1, 1, 4], &cfg, &mut rng), 4);
    }
}
