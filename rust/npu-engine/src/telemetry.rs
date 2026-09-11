//! Per-token measurement records, and the report they roll up into.
//!
//! One record per decoded token, produced by the generation loop and handed to the sink next to the
//! text that token produced. Every other surface -- the CLI overlay, the OpenAI `timings` object,
//! the JSONL run log -- is a projection of these records, so no two of them can derive a different
//! tok/s.
//!
//! Unconditional, not flag-gated. Three `Instant::now()` and one small struct per token against a
//! decode step measured in tens of milliseconds: MEASURED 2026-09-09 on qwen3-0.6b, a 64-token
//! generation is byte-identical and within run-to-run noise of the same binary without any of this
//! (5251-6221 ms against 5286-6140 over two rounds). An instrument you have to remember to switch
//! on is never armed for the run you end up caring about. The expensive tiers (device trace) stay
//! opt-in and are not what this module produces.
//!
//! What this module deliberately does NOT do is invent phases it cannot see. The split below is the
//! one the generation loop can measure honestly; the device's own internal breakdown lives under
//! `DecodeStep::step` and a backend that can measure it reports it through its own hooks.

use serde::{Deserialize, Serialize};

use crate::pipeline::GenerateUsage;

pub mod wire;

/// Host-visible split of the work between one token and the next.
///
/// `step_us` is attributed to the token whose logits that dispatch produced, NOT to the iteration
/// it ran in: the loop samples token N from logits, then dispatches to get the logits for N+1. The
/// first completion token's logits come from priming, so its `step_us` is 0 and the cost sits in
/// [`PrefillRecord`] -- charging it to the token would double-count prefill.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct StepPhases {
    /// `DecodeStep::step`: the device dispatch that produced this token's logits, plus whatever
    /// host glue the backend wraps around it.
    pub step_us: u64,
    /// Logit sampling: penalties, top-k/top-p, the draw. See [`SamplePhases`] for the split.
    pub sample_us: u64,
    /// Incremental detokenization and stop-sequence matching.
    pub detok_us: u64,
    /// The four stages inside `sample_us`, when the backend supplies them. `None` today: it needs
    /// `sampling::SampleOutcome` to carry per-stage timing, which this crate does not yet -- see
    /// [`SamplePhases`]'s own doc.
    pub sample_phases: Option<SamplePhases>,
}

impl StepPhases {
    pub fn sum_us(&self) -> u64 {
        self.step_us + self.sample_us + self.detok_us
    }
}

/// The stages inside [`StepPhases::sample_us`]: penalties, top-k, top-p, the draw -- pipeline order,
/// matching `sampling::sample`'s own. MEASURED 2026-09-09: the default Qwen3 arm (temperature 0.6,
/// top_p 0.95, top_k 20) costs 2.725-3.079 ms/token against 0.132 at temperature 0, a 21x spread an
/// A/B was needed to find. This struct is the fix -- once populated, the spread is one `--stats`
/// call, not an A/B. `None` on [`StepPhases::sample_phases`] until `sampling::SampleOutcome` carries
/// these; `Some` including on the greedy path, where each stage is a true, measured zero (greedy
/// skips them, not "did not measure them").
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SamplePhases {
    pub penalties_us: u64,
    pub top_k_us: u64,
    pub top_p_us: u64,
    pub draw_us: u64,
}

/// What one decoded token cost, and what it produced.
///
/// `token` is `None` for the single trailing record a stop-sequence flush produces: text can leave
/// the generator without a token behind it, and a log that silently dropped those frames would not
/// replay to the same bytes.
#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StepRecord {
    pub seq: u32,
    pub token: Option<u32>,
    /// This token's own detokenized text. Empty when the token completes no codepoint -- a
    /// multi-byte character split across two BPE tokens produces text only on the second.
    pub text: String,
    /// What actually reached the sink at this step, which is not always `text`: a stop matcher
    /// holds text back until it knows the sequence did not start, then releases it in one piece.
    pub emit: String,
    /// Microseconds since the start of the generation.
    pub t_us: u64,
    /// Microseconds since the previous record. For `seq == 0` this is measured from the end of
    /// prefill, so it is a first-token latency and not an inter-token one -- which is why
    /// [`Summary`] computes its percentiles over `seq >= 1`.
    pub dt_us: u64,
    pub phases: StepPhases,
    /// Device dispatches this token cost, when the backend counts them (`NPU_DISPATCH_LOG`).
    /// `None` means "not measured", never "zero" -- the two answer different questions.
    pub dispatches: Option<u32>,
    /// Hardware-context transitions this token cost, same availability rule as `dispatches`.
    pub transitions: Option<u32>,
}

impl StepRecord {
    /// Time this token spent outside every phase the loop can name. A large residual is a finding:
    /// it means the cost is somewhere nothing is timing, which is the one thing a phase table can
    /// otherwise hide.
    pub fn residual_us(&self) -> u64 {
        self.dt_us.saturating_sub(self.phases.sum_us())
    }
}

/// Prompt-side work. Deliberately not a per-token structure: prefill runs as a handful of batched
/// dispatches over many positions, so a per-token timing there would be an average dressed up as a
/// measurement.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrefillRecord {
    /// Prompt length in tokens.
    pub tokens: u32,
    /// Positions primed by the batched prefill path.
    pub batched: u32,
    /// Positions walked one at a time through `step`, including the final position, which always
    /// goes through `step` because only `step` returns the logits the first sample reads.
    pub stepwise: u32,
    pub us: u64,
    pub dispatches: Option<u32>,
}

/// Static conditions the run happened under.
///
/// Every number in a report is a measurement at a moment on a machine in a power state, not a
/// property of the hardware. On this device that is not pedantry: the AIE core clock is a DPM
/// ladder and the power mode is per-boot and often unpinned, so two honest runs can differ by
/// double digits with no code change between them. A report without this block cannot be compared
/// to another one, and the commonest "regression" is a `power_mode` difference.
#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunConditions {
    pub engine_version: String,
    pub model: String,
    /// `xrt-smi`'s performance mode, when the caller could read it. `None` is honest; a guess is not.
    pub power_mode: Option<String>,
    /// Whether the model was already resident when the request arrived. A cold first token is a
    /// different measurement from a warm one and must not be averaged with it.
    pub resident: Option<bool>,
    pub kernel: Option<String>,
    pub started_unix: i64,
}

/// One (xclbin, instruction-stream) pair's blocking dispatch time inside a generation, from
/// `npu_xrt::dispatch_log`. `label` is `npu_xrt::Kernel`'s own label (an xclbin stem), not a
/// device-agnostic design name -- two designs sharing an xclbin (`SHARE_DESIGNS`) collapse to one
/// row here, and that collapse is itself a finding, not a loss of information. And the log can only
/// ever split by STREAM: a fused decode issues every layer as one instruction stream inside one XRT
/// dispatch, so this can legitimately be a single row that IS the whole device row -- see
/// `stats::table`'s one-stream case, which renders that as a fact rather than as a completed split.
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
pub struct DesignCost {
    pub label: String,
    pub dispatches: u32,
    pub secs: f64,
}

/// `after` minus `before`, matched by `label`. `LlmGenerator::generate` uses this to scope the
/// device-by-stream breakdown to the decode window: `before` is the log right after prefill,
/// `after` is the log as of the last token this generation actually attributed a dispatch to --
/// same boundary `StepPhases::step_us`'s own per-token sum stops at (see the stop-token comment in
/// `generator.rs`), which is what keeps a diffed row from exceeding the device row it explains. A
/// label present only in `after` keeps its full count; `saturating_sub`/`.max(0.0)` guard the
/// reverse, which cannot happen in practice (dispatches never un-happen) but must never underflow
/// or go negative if it somehow did.
pub fn diff_design_breakdown(before: &[DesignCost], after: &[DesignCost]) -> Vec<DesignCost> {
    let base: std::collections::HashMap<&str, &DesignCost> =
        before.iter().map(|d| (d.label.as_str(), d)).collect();
    after
        .iter()
        .map(|a| {
            let (bd, bs) = base.get(a.label.as_str()).map_or((0, 0.0), |b| (b.dispatches, b.secs));
            DesignCost {
                label: a.label.clone(),
                dispatches: a.dispatches.saturating_sub(bd),
                secs: (a.secs - bs).max(0.0),
            }
        })
        .collect()
}

/// Which build produced this run's numbers: precision, quantization, the fusion flags baked into
/// the artifact, and its identity on disk. The fix for the project's most expensive recurring
/// error -- differencing two absolute numbers from two builds measured hours apart with nothing in
/// the output saying they were different builds at all. Every field is `None`/empty until a backend
/// supplies it (`DecodeStep::provenance`); `n_past` is the one field the generator itself fills, from
/// the KV position this generation actually reached.
#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArmProvenance {
    pub head_dtype: Option<String>,
    pub mlp_dtype: Option<String>,
    pub quant_group: Option<u32>,
    /// Fusion/env flags active in this artifact's build (`FUSE_QKV_DP`, `TMV_CTX`, ...).
    pub fusion_flags: Vec<String>,
    /// The artifact's built `dims.S` -- the KV window it was compiled for, not necessarily the one
    /// this generation used.
    pub max_seq: Option<u32>,
    /// KV position reached by the end of THIS generation (prompt + completion).
    pub n_past: Option<u32>,
    pub artifact_path: Option<String>,
    pub artifact_hash: Option<String>,
    pub toolchain_pin_hash: Option<String>,
}

/// Everything one generation measured. Produced by the generation loop; the fields the loop cannot
/// see (`queue_us`, `load_us`, `conditions`) are filled by the layer that can.
///
/// No longer `Eq`: `design_breakdown`'s `secs: f64` cannot be. Nothing depended on it -- `PartialEq`
/// is what every test here (`assert_eq!`) actually needs.
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerationReport {
    pub conditions: RunConditions,
    /// Time the request waited for the device-owning actor. Real and currently invisible: one
    /// thread owns the device, so a second request queues behind the first for its whole generation.
    pub queue_us: u64,
    /// Model load/residency work done inside this request. Zero when the model was already warm.
    pub load_us: u64,
    pub tokenize_us: u64,
    pub prefill: PrefillRecord,
    pub steps: Vec<StepRecord>,
    /// Wall clock inside `generate`, from entry to the last token.
    pub generate_us: u64,
    pub usage: GenerateUsage,
    /// NPU package power in microwatts, sampled once at each end of the generation. TWO SAMPLES,
    /// not an integral: the counter costs ~785 us to read, which is 3.7% of a decode step, so it
    /// cannot be sampled per token without corrupting the interval it would be describing. Do not
    /// turn this pair into a J/token figure -- it cannot carry one.
    pub npu_power_start_uw: Option<u64>,
    pub npu_power_end_uw: Option<u64>,
    /// Per-stream blocking time inside [`StepPhases::step_us`], scoped to the DECODE window only --
    /// see `DecodeStep::design_breakdown` and [`diff_design_breakdown`], which `LlmGenerator::generate`
    /// uses to subtract prefill's dispatches out of it. Empty means either the log was off for this
    /// generation or the backend dispatches through no `npu_xrt::Kernel` at all; both render as
    /// nothing to show.
    pub design_breakdown: Vec<DesignCost>,
    /// Which build produced these numbers. See [`ArmProvenance`].
    pub provenance: ArmProvenance,
}

/// Which layer the run was actually spending its time in.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Bound {
    /// Waiting for the device to be free at all.
    Queue,
    /// Loading or making the model resident.
    Load,
    Tokenize,
    Prefill,
    /// The decode dispatch: device time plus backend host glue.
    Device,
    Sampling,
    Detokenize,
    /// Time inside decode that no phase claimed. The default, because a summary nothing measured
    /// has attributed nothing -- naming a real bucket there would be a guess with a confident face.
    #[default]
    Unattributed,
}

impl Bound {
    pub fn as_str(self) -> &'static str {
        match self {
            Bound::Queue => "queue",
            Bound::Load => "load",
            Bound::Tokenize => "tokenize",
            Bound::Prefill => "prefill",
            Bound::Device => "device",
            Bound::Sampling => "sampling",
            Bound::Detokenize => "detokenize",
            Bound::Unattributed => "unattributed",
        }
    }

    /// What to go and fix. A metric that does not point at a lever is noise, so the verdict carries
    /// the lever rather than leaving the reader to map it.
    pub fn lever(self) -> &'static str {
        match self {
            Bound::Queue => "another request owns the device; check concurrency, not the model",
            Bound::Load => "cold start -- pin the model resident to take this off the request path",
            Bound::Tokenize => "host tokenizer; unusual, check the chat template",
            Bound::Prefill => "prompt-side; batched prefill and prompt length are the levers",
            Bound::Device => "on-device dispatch -- the decode-step cost itself",
            Bound::Sampling => "host sampling; penalties and vocabulary size are the levers",
            Bound::Detokenize => "host detokenize/stop-matching",
            Bound::Unattributed => "time no phase claimed -- instrument before optimizing",
        }
    }
}

/// The rolled-up numbers every surface renders. Computed in one place on purpose.
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
pub struct Summary {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    /// Request start to the first byte of text. Includes queue, load, tokenize and prefill, which
    /// is why those are also reported separately -- a slow first token has four possible causes.
    pub ttft_us: u64,
    pub queue_us: u64,
    pub load_us: u64,
    pub tokenize_us: u64,
    pub prefill_us: u64,
    /// First token to last token. The denominator for `tok_per_s`.
    pub decode_us: u64,
    pub total_us: u64,
    pub tok_per_s: f64,
    /// Prompt tokens per second over the prefill window. `None` when nothing was prefilled.
    pub prompt_tok_per_s: Option<f64>,
    /// Inter-token latency over `seq >= 1`, in microseconds. `mean` here is the same quantity
    /// vLLM calls TPOT; the percentiles are what it calls ITL, and they are not interchangeable --
    /// a mean cannot show a stall.
    pub itl_mean_us: u64,
    pub itl_p50_us: u64,
    pub itl_p95_us: u64,
    pub itl_p99_us: u64,
    pub itl_max_us: u64,
    /// Summed decode phases across every token.
    pub phases: StepPhases,
    /// Decode time no phase claimed.
    pub residual_us: u64,
    pub dispatches: Option<u32>,
    pub transitions: Option<u32>,
    pub bound: Bound,
    /// `bound`'s share of `total_us`, 0..1.
    pub bound_share: f64,
    /// See [`GenerationReport::design_breakdown`]. Copied through verbatim: it is a single
    /// per-generation aggregate, not something derived from `steps`.
    pub design_breakdown: Vec<DesignCost>,
    /// See [`GenerationReport::provenance`].
    pub provenance: ArmProvenance,
}

/// Nearest-rank percentile over an already-sorted slice. Exact, because at a few thousand samples
/// per request there is no reason to approximate: the whole vector is smaller than the response.
fn percentile(sorted: &[u64], p: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let rank = ((p / 100.0) * sorted.len() as f64).ceil().max(1.0) as usize;
    sorted[rank.min(sorted.len()) - 1]
}

impl GenerationReport {
    /// Records that carry a real token, i.e. excluding a stop-flush tail.
    pub fn token_steps(&self) -> impl Iterator<Item = &StepRecord> {
        self.steps.iter().filter(|s| s.token.is_some())
    }

    /// Roll the records up. Total is measured, not summed: `total_us` is the wall clock, and
    /// `residual_us` is what the named phases failed to account for. Reporting the residual rather
    /// than distributing it is the whole point -- a breakdown whose parts are defined to add up
    /// cannot tell you that the cost is somewhere you are not looking.
    pub fn summarize(&self) -> Summary {
        let toks: Vec<&StepRecord> = self.token_steps().collect();
        let first_emit_us = self.steps.iter().find(|s| !s.emit.is_empty()).map(|s| s.t_us);
        let last_us = self.steps.last().map(|s| s.t_us).unwrap_or(0);
        let decode_us = match (toks.first(), toks.last()) {
            (Some(f), Some(l)) => l.t_us.saturating_sub(f.t_us),
            _ => 0,
        };
        let total_us = self.queue_us + self.load_us + self.generate_us;

        // ITL over seq >= 1: the first record's gap is measured from the end of prefill, so it is a
        // first-token latency wearing an inter-token name. Folding it in is how a long prompt turns
        // into a fake p99.
        let mut gaps: Vec<u64> = toks.iter().filter(|s| s.seq >= 1).map(|s| s.dt_us).collect();
        gaps.sort_unstable();
        let itl_mean_us = if gaps.is_empty() { 0 } else { gaps.iter().sum::<u64>() / gaps.len() as u64 };

        let mut phases = StepPhases::default();
        let mut sample_sum = SamplePhases::default();
        let mut any_sample_phases = false;
        for s in &toks {
            phases.step_us += s.phases.step_us;
            phases.sample_us += s.phases.sample_us;
            phases.detok_us += s.phases.detok_us;
            if let Some(sp) = s.phases.sample_phases {
                any_sample_phases = true;
                sample_sum.penalties_us += sp.penalties_us;
                sample_sum.top_k_us += sp.top_k_us;
                sample_sum.top_p_us += sp.top_p_us;
                sample_sum.draw_us += sp.draw_us;
            }
        }
        // Same `None`-means-unmeasured rule as `sum_opt` below: only real if at least one step
        // actually carried it, never a zeroed struct standing in for "nobody reported this".
        phases.sample_phases = any_sample_phases.then_some(sample_sum);
        // Against the decode window, not against the phase sum: anything the phases missed has to
        // show up somewhere, and here is where.
        let residual_us = decode_us.saturating_sub(phases.sum_us());

        let sum_opt = |f: fn(&StepRecord) -> Option<u32>, base: Option<u32>| -> Option<u32> {
            let mut any = base.is_some();
            let mut acc = base.unwrap_or(0);
            for s in &toks {
                if let Some(v) = f(s) {
                    any = true;
                    acc += v;
                }
            }
            any.then_some(acc)
        };

        let (bound, bound_share) = {
            let cands = [
                (Bound::Queue, self.queue_us),
                (Bound::Load, self.load_us),
                (Bound::Tokenize, self.tokenize_us),
                (Bound::Prefill, self.prefill.us),
                (Bound::Device, phases.step_us),
                (Bound::Sampling, phases.sample_us),
                (Bound::Detokenize, phases.detok_us),
                (Bound::Unattributed, residual_us),
            ];
            // `cands` is a fixed array so the max always exists; the fallback keeps a panic path
            // out of a reporting function entirely rather than relying on that.
            let (b, v) = cands.iter().copied().max_by_key(|(_, v)| *v).unwrap_or((Bound::Unattributed, 0));
            (b, if total_us == 0 { 0.0 } else { v as f64 / total_us as f64 })
        };

        Summary {
            prompt_tokens: self.usage.prompt_tokens,
            completion_tokens: self.usage.completion_tokens,
            ttft_us: self.queue_us + self.load_us + first_emit_us.unwrap_or(last_us),
            queue_us: self.queue_us,
            load_us: self.load_us,
            tokenize_us: self.tokenize_us,
            prefill_us: self.prefill.us,
            decode_us,
            total_us,
            // n-1 gaps span n tokens, so the rate is over the gaps that were actually measured.
            // Dividing n tokens by the span between the first and last overstates by 1/(n-1).
            tok_per_s: if decode_us == 0 { 0.0 } else { gaps.len() as f64 * 1e6 / decode_us as f64 },
            prompt_tok_per_s: (self.prefill.us > 0)
                .then(|| self.prefill.tokens as f64 * 1e6 / self.prefill.us as f64),
            itl_mean_us,
            itl_p50_us: percentile(&gaps, 50.0),
            itl_p95_us: percentile(&gaps, 95.0),
            itl_p99_us: percentile(&gaps, 99.0),
            itl_max_us: gaps.last().copied().unwrap_or(0),
            phases,
            residual_us,
            dispatches: sum_opt(|s| s.dispatches, self.prefill.dispatches),
            transitions: sum_opt(|s| s.transitions, None),
            bound,
            bound_share,
            design_breakdown: self.design_breakdown.clone(),
            provenance: self.provenance.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn step(seq: u32, dt_us: u64, step_us: u64) -> StepRecord {
        StepRecord {
            seq,
            token: Some(100 + seq),
            text: "x".into(),
            emit: "x".into(),
            t_us: 1_000 + dt_us * seq as u64,
            dt_us,
            phases: StepPhases { step_us, sample_us: 10, detok_us: 5, sample_phases: None },
            ..StepRecord::default()
        }
    }

    fn report(steps: Vec<StepRecord>) -> GenerationReport {
        let generate_us = steps.last().map(|s| s.t_us).unwrap_or(0);
        GenerationReport {
            steps,
            generate_us,
            usage: GenerateUsage { prompt_tokens: 8, completion_tokens: 4 },
            prefill: PrefillRecord { tokens: 8, batched: 0, stepwise: 8, us: 1_000, dispatches: None },
            ..GenerationReport::default()
        }
    }

    #[test]
    fn percentiles_are_exact_nearest_rank() {
        let v: Vec<u64> = (1..=100).collect();
        assert_eq!(percentile(&v, 50.0), 50);
        assert_eq!(percentile(&v, 95.0), 95);
        assert_eq!(percentile(&v, 99.0), 99);
        assert_eq!(percentile(&v, 100.0), 100);
        assert_eq!(percentile(&[], 99.0), 0);
        assert_eq!(percentile(&[7], 99.0), 7);
    }

    #[test]
    fn the_first_gap_is_excluded_from_inter_token_latency() {
        // seq 0's gap is first-token latency. Folding it in would let a long prompt masquerade as
        // a decode stall -- 90_000 here would become the max and the p99.
        let s = report(vec![step(0, 90_000, 80), step(1, 20, 5), step(2, 20, 5), step(3, 20, 5)])
            .summarize();
        assert_eq!(s.itl_max_us, 20);
        assert_eq!(s.itl_p99_us, 20);
        assert_eq!(s.itl_mean_us, 20);
    }

    #[test]
    fn a_stall_survives_into_the_tail_but_not_the_mean() {
        let mut steps: Vec<StepRecord> = (0..100).map(|i| step(i, 20, 5)).collect();
        steps[57].dt_us = 5_000;
        let s = report(steps).summarize();
        assert_eq!(s.itl_p50_us, 20, "a single stall must not move the median");
        assert_eq!(s.itl_max_us, 5_000, "and must not be lost either");
        assert!(s.itl_mean_us < 100, "mean stays near the floor: {}", s.itl_mean_us);
    }

    #[test]
    fn unclaimed_decode_time_lands_in_the_residual() {
        // 1000 us between tokens, 200 of it claimed by phases: 800 is unattributed, and saying so
        // is the point. A breakdown that renormalized to 100% would report "device 100%" here.
        let steps: Vec<StepRecord> = (0..5)
            .map(|i| StepRecord { phases: StepPhases { step_us: 185, sample_us: 10, detok_us: 5, sample_phases: None }, ..step(i, 1_000, 185) })
            .collect();
        let s = report(steps).summarize();
        assert_eq!(s.decode_us, 4_000);
        assert_eq!(s.phases.sum_us(), 1_000);
        assert_eq!(s.residual_us, 3_000);
    }

    #[test]
    fn the_verdict_names_the_biggest_bucket_and_its_share() {
        let mut r = report((0..4).map(|i| step(i, 1_000, 900)).collect());
        r.queue_us = 0;
        let s = r.summarize();
        assert_eq!(s.bound, Bound::Device);
        // A queue wait dwarfing everything flips the verdict: the model is not the problem.
        r.queue_us = 10_000_000;
        assert_eq!(r.summarize().bound, Bound::Queue);
    }

    #[test]
    fn a_flush_record_carries_text_without_a_token() {
        // Stop-sequence flush: text leaves the generator with no token behind it. It must not be
        // counted as a token, and must not be dropped either -- a replay has to reproduce the bytes.
        let mut steps: Vec<StepRecord> = (0..3).map(|i| step(i, 20, 5)).collect();
        steps.push(StepRecord { seq: 3, token: None, text: String::new(), emit: "tail".into(),
                                t_us: 1_100, dt_us: 5, ..StepRecord::default() });
        let r = report(steps);
        assert_eq!(r.token_steps().count(), 3);
        assert_eq!(r.summarize().itl_max_us, 20, "the flush gap is not an inter-token gap");
    }

    #[test]
    fn counters_stay_none_when_nothing_measured_them() {
        // None means "not measured", 0 means "measured and it was zero". Collapsing them would let
        // a disabled dispatch log read as a backend that dispatches nothing.
        let s = report((0..3).map(|i| step(i, 20, 5)).collect()).summarize();
        assert_eq!(s.dispatches, None);
        let mut steps: Vec<StepRecord> = (0..3).map(|i| step(i, 20, 5)).collect();
        steps[0].dispatches = Some(1);
        steps[1].dispatches = Some(1);
        steps[2].dispatches = Some(1);
        assert_eq!(report(steps).summarize().dispatches, Some(3));
    }

    #[test]
    fn sample_phases_sums_across_steps_and_stays_none_when_unmeasured() {
        // None when no step measured it -- same rule as dispatches/transitions above.
        let s = report((0..3).map(|i| step(i, 20, 5)).collect()).summarize();
        assert_eq!(s.phases.sample_phases, None);

        let mut steps: Vec<StepRecord> = (0..3).map(|i| step(i, 20, 5)).collect();
        for st in &mut steps {
            st.phases.sample_phases =
                Some(SamplePhases { penalties_us: 1, top_k_us: 2, top_p_us: 3, draw_us: 4 });
        }
        let sp = report(steps).summarize().phases.sample_phases.expect("measured on every step");
        assert_eq!(sp, SamplePhases { penalties_us: 3, top_k_us: 6, top_p_us: 9, draw_us: 12 });
    }
}
