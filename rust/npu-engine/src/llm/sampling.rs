//! Token-level sampling over a logits VIEW: temperature, top-k, top-p (nucleus), and
//! repetition/frequency/presence penalties.
//!
//! Copied from `npu-gemma::sampling` and fixed for the defect recorded in
//! `docs/kb/an-id-indexed-array-api-degrades-silently-on-a-subset.md`: the original took a bare
//! `&[f32]` indexed by TOKEN ID, so `sample()`/`argmax()` returned a raw array index (a token id
//! only for a full-vocabulary array) and `apply_penalties`'s `logits.get_mut(tok as usize)` silently
//! dropped the penalty for any history token outside the array. [`LogitView`] makes the index space
//! explicit -- full vocab is the identity view, a top-k slice is an explicit one -- so every return
//! is a real token id and a history token missing from the view is a counted, reported decision
//! ([`SampleOutcome::penalties_skipped`]), never a silent no-op.
//!
//! **Default is GREEDY** (`SamplingConfig::default()` has `temperature: 0.0`). `sample()`
//! short-circuits straight to [`argmax`] when `temperature <= 0.0` -- no RNG draw, no penalty/top-k/
//! top-p pass, byte-for-byte the same greedy path `npu-gemma::sampling` validates against the host
//! oracle.
//!
//! Pipeline order (mirrors the common llama.cpp/HF convention): penalties -> temperature -> top-k ->
//! top-p -> softmax -> inverse-CDF draw via the seeded [`SplitMix64`] PRNG. All internal math runs in
//! f64, matching `scripts/gemma_sampling_ref.py` bit-for-bit.

use std::collections::HashMap;

/// Logits paired with the token ids they stand for. `ids: None` is the identity view
/// (`values[i]` is the logit for token id `i`) -- the full-vocabulary case. `ids: Some(ids)` is a
/// narrowed view, e.g. a device-side top-k slice, where `values[i]` is the logit for `ids[i]`.
#[derive(Debug, Clone, Copy)]
pub struct LogitView<'a> {
    values: &'a [f32],
    ids: Option<&'a [u32]>,
}

impl<'a> LogitView<'a> {
    /// Full-vocabulary view: `values[i]` is the logit for token id `i`.
    pub fn full(values: &'a [f32]) -> Self {
        LogitView { values, ids: None }
    }

    /// A narrowed view: `values[i]` is the logit for token `ids[i]`. Panics on a length mismatch --
    /// a mismatched pair is a caller bug, not a runtime case to recover from.
    pub fn subset(values: &'a [f32], ids: &'a [u32]) -> Self {
        assert_eq!(values.len(), ids.len(), "LogitView::subset: values/ids length mismatch");
        LogitView { values, ids: Some(ids) }
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// The logit at rank `r` -- a position WITHIN THIS VIEW, not a token id.
    pub fn value_at_rank(&self, r: usize) -> f32 {
        self.values[r]
    }

    /// The token id standing at rank `r`.
    pub fn id_at_rank(&self, r: usize) -> u32 {
        match self.ids {
            Some(ids) => ids[r],
            None => r as u32,
        }
    }

    /// The rank token `id` occupies in this view, if it is present at all. `None` is the
    /// representable "this id is absent from this view" case the caller must decide about, rather
    /// than a silent skip.
    pub fn rank_of_id(&self, id: u32) -> Option<usize> {
        match self.ids {
            None => ((id as usize) < self.values.len()).then_some(id as usize),
            Some(ids) => ids.iter().position(|&x| x == id),
        }
    }
}

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

/// The result of one draw. `token` is always a real token id, never a rank.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SampleOutcome {
    pub token: u32,
    /// Distinct history token ids that were absent from the view and so could not be penalised.
    /// Always 0 on the greedy short-circuit (penalties never run) and always 0 for a
    /// full-vocabulary view (nothing can be absent from it). Non-zero only means "a narrowed view
    /// was handed a history token it cannot see" -- the caller decides whether that is acceptable
    /// (it is, for a genuine top-k slice) or should trigger a full-logits fallback.
    pub penalties_skipped: u32,
}

/// Deterministic 64-bit PRNG (Vigna's splitmix64) -- chosen so `scripts/gemma_sampling_ref.py` can
/// reproduce it bit-for-bit in pure Python with no numpy/torch dependency.
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

/// The token id at the largest logit (strict `>`, first rank wins on ties).
pub fn argmax(view: LogitView) -> u32 {
    let mut best_r = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for r in 0..view.len() {
        let v = view.value_at_rank(r);
        if v > best_v {
            best_v = v;
            best_r = r;
        }
    }
    view.id_at_rank(best_r)
}

/// Numerically-stable softmax in f64, over ranks.
fn softmax_f64(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits
        .iter()
        .map(|&v| if v.is_finite() { (v - max).exp() } else { 0.0 })
        .collect();
    let sum: f64 = exps.iter().sum();
    exps.into_iter().map(|e| e / sum).collect()
}

/// Apply repetition/frequency/presence penalties over `history` token ids, in place over
/// rank-indexed `work`. Resolves each history id through `view` -- an id absent from the view is
/// counted, not silently skipped. No-op (and no history scan) when all three penalties are at their
/// identity value, matching the pre-fix behaviour exactly.
fn apply_penalties(work: &mut [f64], view: &LogitView, history: &[u32], cfg: &SamplingConfig) -> u32 {
    let (rep, freq, pres) =
        (cfg.repetition_penalty as f64, cfg.frequency_penalty as f64, cfg.presence_penalty as f64);
    if rep == 1.0 && freq == 0.0 && pres == 0.0 {
        return 0;
    }
    let mut counts: HashMap<u32, u32> = HashMap::new();
    for &t in history {
        *counts.entry(t).or_insert(0) += 1;
    }
    let mut skipped = 0u32;
    for (tok, count) in counts {
        match view.rank_of_id(tok) {
            Some(r) => {
                let l = &mut work[r];
                if rep != 1.0 {
                    *l = if *l > 0.0 { *l / rep } else { *l * rep };
                }
                *l -= freq * count as f64;
                *l -= pres;
            }
            None => skipped += 1,
        }
    }
    skipped
}

/// Keep only the `k` largest logits (by rank); mask the rest to `-inf`. `k == 0` disables the filter.
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

/// Draw one token from `view` given the decode `history` (token ids generated so far, used only by
/// the penalties). Greedy (argmax, no RNG draw, no penalty/top-k/top-p pass) when
/// `cfg.temperature <= 0.0`; otherwise runs the full pipeline.
pub fn sample(view: LogitView, history: &[u32], cfg: &SamplingConfig, rng: &mut SplitMix64) -> SampleOutcome {
    if cfg.temperature <= 0.0 {
        return SampleOutcome { token: argmax(view), penalties_skipped: 0 };
    }
    let mut work: Vec<f64> = (0..view.len()).map(|r| view.value_at_rank(r) as f64).collect();
    let penalties_skipped = apply_penalties(&mut work, &view, history, cfg);
    let temperature = cfg.temperature as f64;
    for v in work.iter_mut() {
        *v /= temperature;
    }
    filter_top_k(&mut work, cfg.top_k);
    filter_top_p(&mut work, cfg.top_p as f64);
    let probs = softmax_f64(&work);
    let u = rng.next_f64();
    let mut cum = 0.0f64;
    for (r, &p) in probs.iter().enumerate() {
        cum += p;
        if u < cum {
            return SampleOutcome { token: view.id_at_rank(r), penalties_skipped };
        }
    }
    SampleOutcome { token: view.id_at_rank(probs.len() - 1), penalties_skipped } // last-ulp edge case
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mirrored 1:1 in scripts/gemma_sampling_ref.py -- keep both in sync if either changes.
    const SAMPLE_LOGITS: [f32; 8] = [1.0, 3.0, 2.0, 0.5, 4.0, -1.0, 2.5, 0.0];

    #[test]
    fn greedy_default_is_argmax_and_ignores_rng_and_history() {
        let cfg = SamplingConfig::default();
        assert_eq!(cfg.temperature, 0.0, "default must stay greedy (temperature 0) -- do not flip");
        let mut rng = SplitMix64::new(123);
        let out = sample(LogitView::full(&SAMPLE_LOGITS), &[1, 1, 4, 4, 4], &cfg, &mut rng);
        assert_eq!(out.token, argmax(LogitView::full(&SAMPLE_LOGITS)));
        assert_eq!(out.token, 4); // index of the 4.0 logit
        assert_eq!(out.penalties_skipped, 0);
    }

    #[test]
    fn argmax_first_wins_on_ties() {
        assert_eq!(argmax(LogitView::full(&[1.0, 2.0, 2.0, 0.0])), 1);
    }

    #[test]
    fn top_k_one_collapses_to_the_argmax_token_for_any_seed() {
        let cfg = SamplingConfig { temperature: 1.0, top_k: 1, ..SamplingConfig::default() };
        let want = argmax(LogitView::full(&SAMPLE_LOGITS));
        for seed in [0u64, 1, 42, 999_999, u64::MAX] {
            let mut rng = SplitMix64::new(seed);
            let out = sample(LogitView::full(&SAMPLE_LOGITS), &[], &cfg, &mut rng);
            assert_eq!(out.token, want, "seed {seed}");
        }
    }

    #[test]
    fn top_k_filter_keeps_exactly_k_finite_entries() {
        let mut work: Vec<f64> = SAMPLE_LOGITS.iter().map(|&v| v as f64).collect();
        filter_top_k(&mut work, 3);
        let finite = work.iter().filter(|v| v.is_finite()).count();
        assert_eq!(finite, 3);
        for i in [4usize, 1, 6] {
            assert!(work[i].is_finite(), "index {i} should survive top_k=3");
        }
        for i in [0usize, 2, 3, 5, 7] {
            assert!(work[i].is_infinite(), "index {i} should be masked by top_k=3");
        }
    }

    #[test]
    fn top_p_filter_matches_hand_computed_nucleus() {
        let mut work = vec![10.0f64, 0.0];
        filter_top_p(&mut work, 0.99);
        assert!(work[0].is_finite());
        assert!(work[1].is_infinite(), "the near-zero-mass tail must be dropped at top_p=0.99");
    }

    #[test]
    fn repetition_penalty_shrinks_positive_logit_toward_zero() {
        let cfg = SamplingConfig { repetition_penalty: 2.0, ..SamplingConfig::default() };
        let view = LogitView::full(&[4.0, -4.0]);
        let mut work: Vec<f64> = vec![4.0, -4.0];
        apply_penalties(&mut work, &view, &[0, 1], &cfg);
        assert!((work[0] - 2.0).abs() < 1e-12, "positive logit divided by penalty: {}", work[0]);
        assert!((work[1] - (-8.0)).abs() < 1e-12, "negative logit multiplied by penalty: {}", work[1]);
    }

    #[test]
    fn frequency_and_presence_penalties_are_additive_and_count_scaled() {
        let cfg = SamplingConfig { frequency_penalty: 0.5, presence_penalty: 0.1, ..SamplingConfig::default() };
        let view = LogitView::full(&[10.0, 10.0, 10.0]);
        let mut work: Vec<f64> = vec![10.0, 10.0, 10.0];
        apply_penalties(&mut work, &view, &[0, 0, 0, 1], &cfg); // token 0 x3, token 1 x1, token 2 unseen
        let (freq, pres) = (cfg.frequency_penalty as f64, cfg.presence_penalty as f64);
        assert!((work[0] - (10.0 - freq * 3.0 - pres)).abs() < 1e-12);
        assert!((work[1] - (10.0 - freq * 1.0 - pres)).abs() < 1e-12);
        assert!((work[2] - 10.0).abs() < 1e-12, "unseen token must be untouched");
    }

    #[test]
    fn identity_config_is_a_true_no_op_before_softmax() {
        let cfg = SamplingConfig::default();
        let view = LogitView::full(&SAMPLE_LOGITS);
        let mut work: Vec<f64> = SAMPLE_LOGITS.iter().map(|&v| v as f64).collect();
        let before = work.clone();
        apply_penalties(&mut work, &view, &[1, 1, 4], &cfg);
        assert_eq!(work, before, "identity penalties must not touch logits at all");
    }

    /// Cross-language host-reference gate, preserved from `npu-gemma::sampling`. Fixed seed,
    /// fixed logits/history/config; expected id is `scripts/gemma_sampling_ref.py`'s literal stdout
    /// (`cross_lang_full_pipeline: seed=42 -> 6`). A full-vocabulary `LogitView` must reproduce the
    /// pre-fix crate's output bit-for-bit -- this IS the "identity view is bit-identical" gate.
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
        let out = sample(LogitView::full(&SAMPLE_LOGITS), &[1, 1, 4], &cfg, &mut rng);
        assert_eq!(out.token, 6, "must match scripts/gemma_sampling_ref.py's seed=42 oracle output");
    }

    /// Same gate, a second fixed seed/config pair, an 8-draw sequence (PRNG advances identically
    /// call-over-call). Oracle: `cross_lang_plain_softmax_seq: seed=7, n=8 -> [4, 0, 6, 4, 4, 2, 4, 4]`.
    #[test]
    fn cross_lang_plain_softmax_sequence_matches_python_oracle_seed_7() {
        let cfg = SamplingConfig { temperature: 1.0, top_k: 0, top_p: 1.0, ..SamplingConfig::default() };
        let mut rng = SplitMix64::new(7);
        let got: Vec<u32> =
            (0..8).map(|_| sample(LogitView::full(&SAMPLE_LOGITS), &[], &cfg, &mut rng).token).collect();
        assert_eq!(got, vec![4, 0, 6, 4, 4, 2, 4, 4]);
    }

    #[test]
    fn cross_lang_greedy_matches_python_oracle() {
        let cfg = SamplingConfig { temperature: 0.0, ..Default::default() };
        let mut rng = SplitMix64::new(0);
        let out = sample(LogitView::full(&SAMPLE_LOGITS), &[1, 1, 4], &cfg, &mut rng);
        assert_eq!(out.token, 4);
    }

    /// A fixed seed is reproducible: two independent `SplitMix64` streams from the same seed must
    /// draw the identical token sequence.
    #[test]
    fn fixed_seed_is_reproducible() {
        let cfg = SamplingConfig { temperature: 0.9, top_k: 3, ..SamplingConfig::default() };
        let run = |seed| {
            let mut rng = SplitMix64::new(seed);
            (0..5).map(|_| sample(LogitView::full(&SAMPLE_LOGITS), &[1, 4], &cfg, &mut rng).token).collect::<Vec<_>>()
        };
        assert_eq!(run(7), run(7));
    }

    /// The load-bearing fix: a narrowed view returns REAL token ids (not ranks), and a history
    /// token absent from the view is counted rather than silently dropped.
    #[test]
    fn narrowed_view_returns_token_ids_and_counts_skipped_penalties() {
        // Rank 0 -> token 100, rank 1 -> token 200, rank 2 -> token 300 (ids deliberately far from
        // their ranks, so returning a rank instead of an id would fail immediately).
        let ids = [100u32, 200, 300];
        let values = [1.0f32, 5.0, 2.0];
        let view = LogitView::subset(&values, &ids);

        // Greedy: argmax over the view must report the WINNING TOKEN ID (200), never rank 1.
        assert_eq!(argmax(view), 200);

        // Penalized sample: history has 200 (in view), 300 (in view), and 999/1000 (both absent).
        // temperature > 0 so the penalty pass actually runs (greedy short-circuits past it entirely).
        let cfg = SamplingConfig { temperature: 1.0, repetition_penalty: 1.5, ..SamplingConfig::default() };
        let mut rng = SplitMix64::new(1);
        let out = sample(view, &[200, 300, 999, 1000], &cfg, &mut rng);
        assert!([100, 200, 300].contains(&out.token), "must return a real id from the view, got {}", out.token);
        assert_eq!(out.penalties_skipped, 2, "999 and 1000 are absent from the view and must be counted");
    }

    #[test]
    fn identity_view_and_full_slice_agree_on_rank_lookup() {
        let view = LogitView::full(&SAMPLE_LOGITS);
        for i in 0..SAMPLE_LOGITS.len() as u32 {
            assert_eq!(view.rank_of_id(i), Some(i as usize));
        }
        assert_eq!(view.rank_of_id(SAMPLE_LOGITS.len() as u32), None, "out-of-range id is absent, not a panic");
    }
}
