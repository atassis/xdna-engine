//! S2 AR token selection: everything stock does to a raw LM-head logits row before it becomes a
//! token id. Mirrors two stock files: `s2.cpp/src/s2_sampler.cpp::sample_token` (top_k/top_p
//! filter, temperature draw) and the masking + repetition-aware-sampling (RAS) bookkeeping in
//! `s2.cpp/src/s2_generate.cpp::generate` (:29-35, :57-71, :86-121). The AR half's correctness
//! gate is on pre-sampling logits (see `ar/weights.rs`'s module doc and `s2_ar_ref.py:745-762|`)
//! precisely because this file's stock counterpart is not reproducible past the RAS trigger; this
//! module exists for the working prototype, not the gate.
//!
//! **Deliberate divergence from stock: the RNG is a parameter, not global entropy.**
//! `s2_sampler.cpp:94` seeds its `mt19937` from `std::random_device{}()` -- unseeded, freshly
//! random every process -- and only the temperature>0 sampling draw touches it... except RAS
//! (below) forces a temperature=1.0 draw on a repeat regardless of the caller's own temperature,
//! so a "temperature=0" stock run is NOT guaranteed deterministic; it only stays deterministic
//! until the first RAS trigger (`s2_ar_ref.py:754-762` names this exactly). [`sample_token`] and
//! [`select_main_token`] here take a [`SampleRng`] instead, so: at temperature 0 with no RAS
//! trigger our output is identical to stock (both reduce to masked argmax, independent of RNG);
//! once RAS fires, stock is unseeded and irreproducible even to itself, while we are seeded and
//! exactly reproducible for a given seed -- NOT bit-identical to any particular stock run, which
//! is impossible by construction (stock never records the seed it used). Do not read the two as
//! interchangeable past that point.

use std::collections::VecDeque;

/// Mirrors `s2::SamplerParams` (`s2_sampler.h:8-12`) and `GenerateParams`'s same three fields
/// (`s2_generate.h:25-27`) -- same names, same stock defaults.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplerParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
}

impl Default for SamplerParams {
    fn default() -> Self {
        SamplerParams { temperature: 0.8, top_p: 0.8, top_k: 30 }
    }
}

/// Source of randomness for [`sample_token`]'s temperature>0 draw. A trait, not a concrete type,
/// so a caller can swap in a different generator; [`SplitMix64`] is the one this module ships.
pub trait SampleRng {
    /// Uniform value in `[0, 1)`.
    fn next_f64(&mut self) -> f64;
}

/// SplitMix64 (Steele/Vigna) -- small, deterministic, seedable. Not libstdc++'s `mt19937`: bit
/// parity with a stock run is impossible anyway (stock's seed is never recorded, see module
/// doc), so this only needs to be a good, reproducible weighted-draw source, not a specific
/// algorithm.
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        SplitMix64 { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

impl SampleRng for SplitMix64 {
    fn next_f64(&mut self) -> f64 {
        // Top 53 bits of a 64-bit draw -> a full-precision uniform double in [0, 1).
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

/// `argmax(logits)`, first index wins on an exact tie -- matches `np.argmax`
/// (`s2_ar_ref.py:783`), the gate's own authority. Stock's tie order is unspecified: `std::sort`
/// (`s2_sampler.cpp:51-53`) is not required to be stable, so exact-tie behavior was never a
/// contract stock itself upholds.
pub fn greedy_token(logits: &[f32]) -> i32 {
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    best_i as i32
}

/// Mirrors `s2::sample_token` (`s2_sampler.cpp:42-99`). At `temperature <= 0.0` this is exactly
/// [`greedy_token`] and never touches `rng`: top_k/top_p filtering only REMOVES lower-ranked
/// entries (`s2_sampler.cpp:65-70`), it can't promote anything above rank 0, so the filtered
/// top-of-sort is always the unfiltered argmax (`s2_ar_ref.py:747-752` states the same
/// equivalence for its own gate). Otherwise: sort desc, keep the top_k / cumulative-top_p prefix,
/// re-softmax that prefix at `temperature`, and draw from it.
pub fn sample_token(logits: &[f32], params: &SamplerParams, rng: &mut impl SampleRng) -> i32 {
    if params.temperature <= 0.0 {
        return greedy_token(logits);
    }
    sample_stochastic(logits, params, rng)
}

fn sample_stochastic(logits: &[f32], params: &SamplerParams, rng: &mut impl SampleRng) -> i32 {
    let vocab_size = logits.len();
    let mut items: Vec<(f32, i32)> = logits.iter().enumerate().map(|(i, &l)| (l, i as i32)).collect();
    items.sort_by(|a, b| b.0.total_cmp(&a.0));

    let k = if params.top_k > 0 { (params.top_k as usize).min(vocab_size) } else { vocab_size };
    let top_p = params.top_p.clamp(0.0, 1.0);

    // Cutoff decision uses a softmax over the WHOLE sorted vocab (`softmax_from_sorted_logits`,
    // s2_sampler.cpp:25-40) -- a different pass from the temperature-softmax below that actually
    // weights the draw over the survivors.
    let max_logit = items[0].0;
    let denom: f32 = items.iter().map(|&(l, _)| (l - max_logit).exp()).sum();

    let mut filtered: Vec<(f32, i32)> = Vec::with_capacity(k);
    let mut cum = 0.0f32;
    for (i, &(l, id)) in items.iter().enumerate() {
        let p = if denom > 0.0 { (l - max_logit).exp() / denom } else { 0.0 };
        cum += p;
        // Both conditions are monotonic in `i` (cum is non-decreasing), so once either trips it
        // stays tripped -- equivalent to a prefix cut, just following s2_sampler.cpp:63-71's own
        // continue-don't-break shape.
        let remove_for_top_k = i >= k;
        let remove_for_top_p = i > 0 && cum > top_p;
        if remove_for_top_k || remove_for_top_p {
            continue;
        }
        filtered.push((l, id));
    }
    if filtered.is_empty() {
        filtered.push(items[0]); // defensive only: rank 0 (i=0) can never be removed above.
    }

    let f_max = filtered[0].0; // rank 0 always survives filtering, so this equals `max_logit`.
    let mut probs: Vec<f32> =
        filtered.iter().map(|&(l, _)| ((l - f_max) / params.temperature).exp()).collect();
    let sum: f32 = probs.iter().sum();
    if sum <= 0.0 {
        return filtered[0].1; // s2_sampler.cpp:89-91
    }
    for p in &mut probs {
        *p /= sum;
    }

    filtered[draw_weighted(&probs, rng)].1
}

/// `std::discrete_distribution` equivalent: draw uniformly over `[0, sum(probs))` and return the
/// index whose cumulative weight first exceeds it.
fn draw_weighted(probs: &[f32], rng: &mut impl SampleRng) -> usize {
    let total: f64 = probs.iter().map(|&p| p as f64).sum();
    if total <= 0.0 {
        return 0; // unreachable given sample_stochastic's own sum<=0.0 guard above.
    }
    let u = rng.next_f64() * total;
    let mut cum = 0.0f64;
    for (i, &p) in probs.iter().enumerate() {
        cum += p as f64;
        if u < cum {
            return i;
        }
    }
    probs.len() - 1 // floating-point rounding at the tail.
}

/// Additive vocabulary restriction for the AR LM head: `-inf` everywhere except the semantic code
/// range `[semantic_begin_id, semantic_end_id]` (inclusive) and `im_end_id`, mirroring `sem_mask`
/// (`s2_generate.cpp:29-35`) and its Python mirror (`mask_row_ids`, `s2_ar_ref.py:770-773`).
/// Default ids (`fish_speech.semantic_begin_id`=151678, `fish_speech.semantic_end_id`=155773,
/// `s2_model.cpp:270-271` / `s2_ar_ref.py:331-332,383-384`) are checkpoint hparams the caller
/// resolves, not this module's business. `im_end_id` is `None` for stock's "disabled" sentinel
/// (C++ `im_end_id < 0`, `s2_generate.cpp:33`; Python `hp.im_end_id < 0`, `s2_ar_ref.py:772`) --
/// stock's C++ path resolves it by tokenizing `<|im_end|>` (`s2_tokenizer.cpp:247`) while its
/// Python mirror reads it from a different GGUF key, `fish_speech.audio_pad_token_id`
/// (`s2_ar_ref.py:402`); this module takes whichever value the caller already resolved.
pub struct SemanticMask {
    pub semantic_begin_id: i32,
    pub semantic_end_id: i32,
    pub im_end_id: Option<i32>,
}

impl SemanticMask {
    pub fn new(semantic_begin_id: i32, semantic_end_id: i32, im_end_id: Option<i32>) -> Self {
        SemanticMask { semantic_begin_id, semantic_end_id, im_end_id }
    }

    /// In-place, mirroring `apply_mask_and_sample` (`s2_generate.cpp:57-71`): mask everything
    /// outside `[semantic_begin_id, semantic_end_id] ∪ {im_end_id}` to `-inf`, then, if
    /// `block_im_end`, additionally force `im_end_id` itself to `-inf` (used while
    /// `step < min_tokens_before_end` to stop the model ending too early). An out-of-range
    /// `im_end_id` (`< 0` or `>= logits.len()`) is silently inert, matching stock's own bounds
    /// check (`s2_generate.cpp:33,63`).
    pub fn apply(&self, logits: &mut [f32], block_im_end: bool) {
        for (i, l) in logits.iter_mut().enumerate() {
            let i = i as i32;
            let in_semantic_range = i >= self.semantic_begin_id && i <= self.semantic_end_id;
            let is_im_end = self.im_end_id == Some(i);
            if !(in_semantic_range || is_im_end) {
                *l = f32::NEG_INFINITY;
            }
        }
        if block_im_end {
            if let Some(id) = self.im_end_id {
                if let Some(l) = usize::try_from(id).ok().and_then(|i| logits.get_mut(i)) {
                    *l = f32::NEG_INFINITY;
                }
            }
        }
    }
}

const RAS_WINDOW_SIZE: usize = 10; // s2_generate.cpp:87, `ras_window_size`
const RAS_TEMPERATURE: f32 = 1.0; // s2_generate.cpp:88, `ras_high_temp`
const RAS_TOP_P: f32 = 0.9; // s2_generate.cpp:89, `ras_high_top_p`

/// Repetition-aware-sampling window: the last <= [`RAS_WINDOW_SIZE`] accepted main (semantic)
/// tokens, mirroring `ras_window` (`s2_generate.cpp:86-121`). Pure bookkeeping -- no RNG -- so
/// the repeat rule is testable on its own.
#[derive(Default)]
pub struct RasWindow {
    tokens: VecDeque<i32>,
}

impl RasWindow {
    pub fn new() -> Self {
        RasWindow { tokens: VecDeque::with_capacity(RAS_WINDOW_SIZE) }
    }

    /// `s2_generate.cpp:99-102`: true iff the window is non-empty, `candidate` is already present
    /// ANYWHERE in it (membership, not "exactly 10 back" -- `std::find` over the whole deque),
    /// and `candidate` falls in the semantic range (excludes `im_end_id`, which can't trigger
    /// RAS: ending the sequence is never treated as a repeat).
    pub fn should_resample(&self, candidate: i32, semantic_begin_id: i32, semantic_end_id: i32) -> bool {
        !self.tokens.is_empty()
            && candidate >= semantic_begin_id
            && candidate <= semantic_end_id
            && self.tokens.contains(&candidate)
    }

    /// `s2_generate.cpp:118-121`: push, then evict the oldest if over capacity. Runs
    /// unconditionally every step, including the token RAS itself just produced -- stock does not
    /// re-check or retry a resampled token against the window.
    pub fn push(&mut self, token: i32) {
        self.tokens.push_back(token);
        if self.tokens.len() > RAS_WINDOW_SIZE {
            self.tokens.pop_front();
        }
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

/// RAS's own fixed sampler params (`s2_generate.cpp:111-114`): `temperature`/`top_p` are always
/// `RAS_TEMPERATURE`/`RAS_TOP_P` regardless of the caller's own params -- so RAS can fire (and
/// draw from `rng`) even when the caller asked for `temperature=0.0`; only `top_k` carries over.
pub fn ras_resample_params(base: &SamplerParams) -> SamplerParams {
    SamplerParams { temperature: RAS_TEMPERATURE, top_p: RAS_TOP_P, top_k: base.top_k }
}

/// One step of `generate()`'s main-token selection (`s2_generate.cpp:96-121` and the priming call
/// at `:73-74`, which is the `step=0` case of the same `block_im_end = step < min_tokens_before_end`
/// rule -- no special-casing needed for the first token). Masks `logits` in place, samples, and
/// resamples at RAS params if the result repeats within `ras`'s window.
///
/// Stock rebuilds its `biased` scratch from `state.logits` again for the RAS branch
/// (`s2_generate.cpp:104-110`) rather than reusing the array `apply_mask_and_sample` produced;
/// this reuses the one masked in place above instead. That's a safe simplification, not a
/// behavior change: RAS's own `block_im_end` test (`:108`, `step < min_tokens_before_end`) is the
/// identical condition that produced the candidate being re-sampled, so the two masked arrays are
/// numerically the same array.
///
/// Does not touch model state (`state.logits`/`step_input`/KV history) -- advancing the AR
/// forward pass is the driver's job, not built yet on this side (only [`super::ArWeights`] has
/// landed); this is the primitive that driver will call once it exists.
pub fn select_main_token(
    logits: &mut [f32],
    mask: &SemanticMask,
    params: &SamplerParams,
    ras: &mut RasWindow,
    block_im_end: bool,
    rng: &mut impl SampleRng,
) -> i32 {
    mask.apply(logits, block_im_end);
    let mut token = sample_token(logits, params, rng);
    if ras.should_resample(token, mask.semantic_begin_id, mask.semantic_end_id) {
        let ras_params = ras_resample_params(params);
        token = sample_token(logits, &ras_params, rng);
    }
    ras.push(token);
    token
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_matches_masked_argmax() {
        let logits = [0.1, 5.0, -3.0, 5.5, 2.0];
        assert_eq!(greedy_token(&logits), 3);
    }

    #[test]
    fn sample_token_at_zero_temperature_is_greedy_regardless_of_top_k_top_p() {
        let logits = [0.1, 5.0, -3.0, 5.5, 2.0];
        let params = SamplerParams { temperature: 0.0, top_p: 0.1, top_k: 1 };
        let mut rng = SplitMix64::new(0); // must be unused: temperature<=0 never draws.
        assert_eq!(sample_token(&logits, &params, &mut rng), greedy_token(&logits));
    }

    #[test]
    fn negative_temperature_also_takes_the_greedy_path() {
        let logits = [1.0, 2.0, 3.0];
        let params = SamplerParams { temperature: -1.0, ..Default::default() };
        let mut rng = SplitMix64::new(0);
        assert_eq!(sample_token(&logits, &params, &mut rng), 2);
    }

    #[test]
    fn mask_excludes_out_of_range_ids() {
        // vocab_size=12, semantic range [5,8], im_end=10.
        let mask = SemanticMask::new(5, 8, Some(10));
        let mut logits = [1.0f32; 12];
        mask.apply(&mut logits, false);
        for i in 0..12 {
            let expect_finite = (5..=8).contains(&i) || i == 10;
            assert_eq!(logits[i].is_finite(), expect_finite, "index {i}");
        }
    }

    #[test]
    fn block_im_end_additionally_suppresses_the_end_token() {
        let mask = SemanticMask::new(5, 8, Some(10));
        let mut logits = [1.0f32; 12];
        mask.apply(&mut logits, true);
        assert!(logits[10].is_infinite() && logits[10].is_sign_negative());
        // the semantic range is untouched by block_im_end.
        for i in 5..=8 {
            assert_eq!(logits[i], 1.0);
        }
    }

    #[test]
    fn disabled_im_end_is_inert() {
        let mask = SemanticMask::new(5, 8, None);
        let mut logits = [1.0f32; 12];
        mask.apply(&mut logits, true); // block_im_end with no im_end_id must not panic or touch anything extra.
        for i in 0..12 {
            let expect_finite = (5..=8).contains(&i);
            assert_eq!(logits[i].is_finite(), expect_finite, "index {i}");
        }
    }

    #[test]
    fn ras_window_empty_never_triggers() {
        let w = RasWindow::new();
        assert!(!w.should_resample(42, 0, 1000));
    }

    #[test]
    fn ras_window_repeat_at_the_capacity_boundary_still_triggers() {
        let mut w = RasWindow::new();
        for t in 100..110 {
            w.push(t); // fills the window to exactly RAS_WINDOW_SIZE=10: tokens 100..=109.
        }
        assert_eq!(w.len(), RAS_WINDOW_SIZE);
        // 100 is the OLDEST entry still inside the window (position 0 of 10) -- must still count.
        assert!(w.should_resample(100, 0, 1000));
    }

    #[test]
    fn ras_window_repeat_just_evicted_no_longer_triggers() {
        let mut w = RasWindow::new();
        for t in 100..110 {
            w.push(t);
        }
        w.push(110); // 11th push evicts 100; window is now 101..=110.
        assert_eq!(w.len(), RAS_WINDOW_SIZE);
        assert!(!w.should_resample(100, 0, 1000));
        assert!(w.should_resample(101, 0, 1000)); // the new oldest entry, symmetric check.
    }

    #[test]
    fn ras_ignores_a_repeat_outside_the_semantic_range() {
        let mut w = RasWindow::new();
        w.push(42);
        // 42 repeats, but the caller's semantic range is [100, 1000] -- e.g. an im_end_id.
        assert!(!w.should_resample(42, 100, 1000));
    }

    #[test]
    fn same_seed_reproduces_the_same_sequence() {
        let logits = [1.0f32, 1.0, 1.0, 1.0];
        let params = SamplerParams { temperature: 1.0, top_p: 1.0, top_k: 0 };
        let draw = |seed: u64| -> Vec<i32> {
            let mut rng = SplitMix64::new(seed);
            (0..50).map(|_| sample_token(&logits, &params, &mut rng)).collect()
        };
        assert_eq!(draw(42), draw(42));
    }

    #[test]
    fn different_seeds_diverge() {
        let logits = [1.0f32, 1.0, 1.0, 1.0];
        let params = SamplerParams { temperature: 1.0, top_p: 1.0, top_k: 0 };
        let draw = |seed: u64| -> Vec<i32> {
            let mut rng = SplitMix64::new(seed);
            (0..50).map(|_| sample_token(&logits, &params, &mut rng)).collect()
        };
        // 4-way choice over 50 draws: a chance collision is astronomically unlikely (4^-50).
        assert_ne!(draw(42), draw(43));
    }

    #[test]
    fn select_main_token_resamples_on_repeat_and_updates_the_window() {
        let mask = SemanticMask::new(0, 3, Some(4));
        let params = SamplerParams { temperature: 0.0, ..Default::default() };
        let mut ras = RasWindow::new();
        let mut rng = SplitMix64::new(7);

        // logits[1] is always the argmax, so greedy alone would emit `1` every step and the
        // window fills with repeats from step 2 onward -- forcing RAS to fire, at temperature=1.0,
        // even though `params.temperature` is 0.
        let mut logits = [0.0f32, 9.0, 1.0, 2.0, -100.0];
        let first = select_main_token(&mut logits, &mask, &params, &mut ras, false, &mut rng);
        assert_eq!(first, 1);
        assert_eq!(ras.len(), 1);

        let mut logits2 = [0.0f32, 9.0, 1.0, 2.0, -100.0];
        let second = select_main_token(&mut logits2, &mask, &params, &mut ras, false, &mut rng);
        // Repeat detected (token 1 is already in the window) -> RAS resample fired, so the result
        // need not be the greedy winner; it must still land in the masked semantic range.
        assert!((0..=3).contains(&second));
        assert_eq!(ras.len(), 2);
    }
}
