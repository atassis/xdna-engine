//! The decode loop: prompt tokenization -> per-token device step -> sampling -> stop conditions.
//! The device call sits behind [`DecodeStep`] so the whole loop is testable with no NPU; the
//! XRT/`ElfResident` backend against a measured fused-decode artifact is a later agent's job.

use std::collections::VecDeque;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use crate::api::EngineError;
use crate::llm::config::ModelConfig;
use crate::llm::detokenize::{IncrementalDetokenizer, StopFeed, StopMatcher};
use crate::llm::sampling::{self, LogitView, SamplingConfig, SplitMix64};
use crate::pipeline::{Chunk, FinishReason, GenerateParams, GenerateUsage, Prompt, TextGenerator};
use crate::telemetry::{
    diff_design_breakdown, ArmProvenance, DesignCost, GenerationReport, PrefillRecord, SamplePhases,
    StepPhases, StepRecord,
};

/// One decode step against whatever backend holds the model: feed `token` at KV-cache position
/// `pos`, get back full-vocabulary logits. `pos` is 0 for the first prompt token; the caller (this
/// module) drives it, so an implementation is stateless about position.
pub trait DecodeStep {
    fn step(&mut self, token: u32, pos: usize) -> Result<Vec<f32>, EngineError>;

    /// Drop any per-generation state before a new one starts. A device backend holds a KV cache
    /// that only grows with `pos`, so without this the second request continues the first one's
    /// context and answers differently -- which is what an end-to-end run caught after every layer
    /// passed its own tests: the scripted mock has no KV state, so no unit test could see it.
    /// Default no-op, so a stateless implementation needs no change.
    fn reset(&mut self) -> Result<(), EngineError> {
        Ok(())
    }

    /// The largest number of token positions this backend's KV cache can hold, or `None` when the
    /// backend has no window (a host implementation whose cache is a growable `Vec`). The device
    /// backend reads it from the artifact's `dims.S`, the capacity of `kc`/`vc`.
    ///
    /// This is a REQUIRED bound, not a hint. `step`'s `pos` becomes a `kv_off` element offset into
    /// a cache sized for exactly `S` positions, and `pos == S` addresses the element one past its
    /// end: still inside the arena, past no check, silently answering from whatever it overwrote.
    /// Enforced in [`LlmGenerator::generate`].
    fn max_context(&self) -> Option<usize> {
        None
    }

    /// The batch one dispatch of this backend's prefill ELF covers, or `None` when it has no
    /// batched-prefill path (every host backend, and a device backend loaded without a prefill
    /// artifact or with `NPU_LLM_PREFILL_BATCHED=0`). The generator uses it to decide whether a
    /// prompt is long enough to be worth the padding, so a backend that returns `None` behaves
    /// exactly as this rail did before prefill existed.
    fn prefill_batch(&self) -> Option<usize> {
        None
    }

    /// Prime the KV cache for `tokens` at positions `[0, tokens.len())` in batches of
    /// [`prefill_batch`](DecodeStep::prefill_batch), and return how many positions were primed.
    ///
    /// Returning fewer than `tokens.len()` (including 0) is legal and is how a backend declines:
    /// the caller resumes the per-token loop at the returned position. It produces no logits --
    /// prefill's product is the KV cache -- so the caller must keep at least the prompt's LAST
    /// token for [`step`](DecodeStep::step), which is what it samples from.
    fn prefill(&mut self, tokens: &[u32]) -> Result<usize, EngineError> {
        let _ = tokens;
        Ok(0)
    }

    /// Live device BO bytes this backend holds, or 0 for a host backend that holds none.
    ///
    /// Reported so `npu models` can weigh the biggest resident thing on the box. An LLM's weights,
    /// KV cache and scratch are the dominant device allocation in this engine, and until this
    /// existed the MEM column read `-` for exactly the model most worth measuring.
    fn bo_bytes(&self) -> u64 {
        0
    }

    /// Cumulative (dispatches, hardware-context transitions) since [`DecodeStep::reset`], or
    /// `None` when this backend does not count them. The generator differences consecutive reads
    /// to charge each token what it actually cost, so an implementation returns running totals and
    /// never has to know about tokens.
    ///
    /// `None` propagates all the way to the report as `null`, which is not the same as `0`: one
    /// says nobody measured, the other says the backend dispatched nothing. Collapsing them makes
    /// a switched-off counter look like a device that did no work.
    fn counters(&self) -> Option<(u32, u32)> {
        None
    }

    /// Per-generation device accounting, or `None` when the backend has none or it is not enabled.
    /// Emitted by [`LlmGenerator::generate`] after the loop, paired with [`DecodeStep::reset`]
    /// before it, so the numbers cover exactly one generation. A host-side backend returns `None`;
    /// nothing in the loop branches on the answer.
    fn dispatch_report(&self) -> Option<String> {
        None
    }

    /// Per-design blocking time this generation cost, `(label, dispatch count, total seconds)`.
    /// Every backend's device dispatches funnel through `npu_xrt::Kernel`, which is what makes this
    /// default correct for ALL of them with no override needed: a backend that never touches the
    /// device (the scripted mock, in tests) truthfully reports nothing, and one that does gets it
    /// for free. Scoped to one generation the same way `counters()` is, by
    /// [`DecodeStep::reset`]/[`npu_xrt::dispatch_log::reset`].
    fn design_breakdown(&self) -> Vec<DesignCost> {
        npu_xrt::dispatch_log::enabled()
            .then(npu_xrt::dispatch_log::per_kernel_snapshot)
            .unwrap_or_default()
            .into_iter()
            .map(|(label, dispatches, secs)| DesignCost { label, dispatches, secs })
            .collect()
    }

    /// Which build this backend is. Default: nothing known. A device backend overrides it with what
    /// its artifact's `meta.json` recorded; `n_past` is filled by the generator itself afterward,
    /// from the KV position this generation actually reached.
    fn provenance(&self) -> ArmProvenance {
        ArmProvenance::default()
    }
}

/// Difference two cumulative counter reads. `None` on either side stays `None` all the way to the
/// report: a backend that does not count is not a backend that dispatched nothing.
fn counter_delta(prev: &mut Option<(u32, u32)>, now: Option<(u32, u32)>) -> (Option<u32>, Option<u32>) {
    let d = match (*prev, now) {
        (Some((pd, pt)), Some((nd, nt))) => (Some(nd.saturating_sub(pd)), Some(nt.saturating_sub(pt))),
        _ => (None, None),
    };
    *prev = now;
    d
}

/// Tokenize a prompt. `Prompt::Chat` renders through the model's chat template first;
/// `Prompt::Raw` tokenizes directly. The returned length is the TRUE tokenized prompt length --
/// never recover it later by filtering EOS out of a padded buffer: EOS doubles as the chat
/// template's own turn separator, so that recovery undercounts and desyncs every position after it.
pub fn tokenize_prompt(
    cfg: &ModelConfig,
    prompt: &Prompt,
    enable_thinking: Option<bool>,
) -> Result<Vec<u32>, EngineError> {
    let (text, add_special_tokens) = match prompt {
        Prompt::Chat(messages) => {
            let tmpl = cfg
                .chat_template
                .as_ref()
                .ok_or_else(|| EngineError::Unsupported("model has no chat_template for Prompt::Chat".to_string()))?;
            // The template already writes out the literal special-token text (`<|im_start|>`, ...);
            // asking the tokenizer to ALSO add its own would duplicate them.
            (tmpl.render_with(messages, true, enable_thinking)?, false)
        }
        Prompt::Raw(s) => (s.clone(), true),
    };
    let enc = cfg
        .tokenizer
        .encode(text, add_special_tokens)
        .map_err(|e| EngineError::Load(format!("tokenize prompt: {e}")))?;
    Ok(enc.get_ids().to_vec())
}

/// A scripted [`DecodeStep`] for tests: replays a fixed queue of logits vectors, one per call,
/// ignoring `token`/`pos`. Not a model -- a way to drive [`LlmGenerator`]'s loop deterministically.
pub struct ScriptedDecodeStep {
    steps: VecDeque<Vec<f32>>,
    max_context: Option<usize>,
    prefill_batch: Option<usize>,
    /// Every `prefill()` call's token count, in order. The batched path produces no logits, so a
    /// test cannot see it in the output at all -- this is what makes "did the generator take it,
    /// and with how much of the prompt" observable without a device.
    pub prefill_calls: Vec<usize>,
    /// Sleep inside `step`, so the report's `step_us` can be checked against a duration the test
    /// chose. An instrument nobody put a known input through is not a measurement.
    step_delay: std::time::Duration,
    /// Cumulative dispatch/transition counts, when the mock is asked to keep them.
    counters: Option<(u32, u32)>,
    /// Cumulative (dispatches, secs), when the mock is asked to track it -- deliberately advanced
    /// by every `step()` call, prefill-stepwise included, to mirror `npu_xrt::dispatch_log`'s own
    /// single accumulator closely enough to reproduce the prefill-contamination bug in a test.
    design: Option<(u32, f64)>,
    design_secs_per_dispatch: f64,
}

impl ScriptedDecodeStep {
    pub fn new(steps: Vec<Vec<f32>>) -> Self {
        ScriptedDecodeStep {
            steps: steps.into(),
            max_context: None,
            prefill_batch: None,
            prefill_calls: Vec::new(),
            step_delay: std::time::Duration::ZERO,
            counters: None,
            design: None,
            design_secs_per_dispatch: 0.0,
        }
    }

    /// Make every `step` take a known amount of time.
    pub fn with_step_delay(mut self, d: std::time::Duration) -> Self {
        self.step_delay = d;
        self
    }

    /// Count dispatches the way a device backend does: one per `step`, reset with the cache.
    pub fn with_counters(mut self) -> Self {
        self.counters = Some((0, 0));
        self
    }

    /// Track a `DesignCost` breakdown the way `npu_xrt::dispatch_log` does: one shared cumulative
    /// counter, advanced `secs_per_dispatch` on every `step()` call regardless of whether it ran
    /// during prefill or decode. Lets a test reproduce -- and check the generator differences away
    /// -- a prefill dispatch landing in the same accumulator the decode window is later read from.
    pub fn with_design_tracking(mut self, secs_per_dispatch: f64) -> Self {
        self.design = Some((0, 0.0));
        self.design_secs_per_dispatch = secs_per_dispatch;
        self
    }

    /// Give the mock a finite KV window, so the bound in [`LlmGenerator::generate`] is testable
    /// without a device.
    pub fn with_max_context(mut self, max_context: usize) -> Self {
        self.max_context = Some(max_context);
        self
    }

    /// Give the mock a batched-prefill path of batch `m`, so [`LlmGenerator::generate`]'s threshold
    /// and its handoff back to the per-token loop are testable without a device.
    pub fn with_prefill_batch(mut self, m: usize) -> Self {
        self.prefill_batch = Some(m);
        self
    }
}

impl DecodeStep for ScriptedDecodeStep {
    fn step(&mut self, _token: u32, _pos: usize) -> Result<Vec<f32>, EngineError> {
        if !self.step_delay.is_zero() {
            std::thread::sleep(self.step_delay);
        }
        if let Some((d, _)) = self.counters.as_mut() {
            *d += 1;
        }
        if let Some((d, s)) = self.design.as_mut() {
            *d += 1;
            *s += self.design_secs_per_dispatch;
        }
        self.steps.pop_front().ok_or_else(|| EngineError::Device("scripted decode exhausted".to_string()))
    }

    fn counters(&self) -> Option<(u32, u32)> {
        self.counters
    }

    fn design_breakdown(&self) -> Vec<DesignCost> {
        self.design
            .map(|(dispatches, secs)| vec![DesignCost { label: "scripted".to_string(), dispatches, secs }])
            .unwrap_or_default()
    }

    fn max_context(&self) -> Option<usize> {
        self.max_context
    }

    fn prefill_batch(&self) -> Option<usize> {
        self.prefill_batch
    }

    fn prefill(&mut self, tokens: &[u32]) -> Result<usize, EngineError> {
        self.prefill_calls.push(tokens.len());
        Ok(tokens.len())
    }
}

/// An autoregressive text generator over a model directory's tokenizer/chat-template/stop-tokens and
/// a [`DecodeStep`] backend.
pub struct LlmGenerator<D: DecodeStep> {
    cfg: ModelConfig,
    decode: D,
    /// The scenario's `[generation]` block -- the tier between the request and the checkpoint.
    scenario_defaults: crate::pipeline::GenerationDefaults,
}

impl<D: DecodeStep> LlmGenerator<D> {
    pub fn new(cfg: ModelConfig, decode: D) -> Self {
        LlmGenerator { cfg, decode, scenario_defaults: Default::default() }
    }

    /// Set the scenario's generation defaults. A request that names a field still wins; these
    /// apply only under the fields it leaves out, and above the checkpoint's own settings.
    pub fn with_scenario_defaults(mut self, d: crate::pipeline::GenerationDefaults) -> Self {
        self.scenario_defaults = d;
        self
    }

    /// The decode backend, for callers that need to read what it recorded (the scripted mock's
    /// `prefill_calls`, a device backend's artifact dims).
    pub fn backend(&self) -> &D {
        &self.decode
    }
}

/// The fewest batchable prompt tokens that make the batched-prefill path worth taking.
///
/// A MEASURED break-even, not a derived one. The old rule was one whole chunk
/// (`batchable >= M`, 255 here) and was justified by ROWS -- a prefill dispatch costs a padded
/// chunk of `M` whatever the prompt is, so a short prompt pays for rows it never uses. That
/// prices the wrong resource: what the per-token path spends is DISPATCHES, one per token, and
/// each one also computes an lm-head whose logits `generate()` discards on every pass but the
/// last.
///
/// Measured 2026-09-10 on qwen3-0.6b, M=256 S=2048, two alternated rounds x 3 reps
/// (`scripts/time_prefill.sh 2 3 10,32,64,128,255`, medians, Power Mode Default):
///
/// | batchable | per-token | batched |        |
/// |-----------|-----------|---------|--------|
/// |        10 |    315 ms |  362 ms | -47 ms |
/// |        32 |   1013 ms |  362 ms |  2.8x  |
/// |        64 |   2027 ms |  362 ms |  5.6x  |
/// |       128 |   4100 ms |  362 ms |  11.3x |
/// |       255 |   8102 ms |  361 ms |  22.4x |
///
/// The batched arm is FLAT at ~362 ms -- one padded chunk, one dispatch, whatever the prompt --
/// and the per-token arm is 31.5-31.9 ms/token. So the crossover is 362/31.6 = 11.5 tokens, and
/// 12 is the smallest threshold that cannot lose. Batching below it is a real regression: at 10
/// tokens it costs 47 ms to avoid nothing.
///
/// KILL-IF: both terms are artifact-specific. A different `M`, a different window, or a decode
/// step that gets faster moves the crossover, and this constant is then wrong in whichever
/// direction that went. Re-run the sweep rather than trusting it across an artifact change.
///
/// `NPU_LLM_PREFILL_MIN_TOKENS` overrides it -- raise it to the artifact's `dims.M` for the
/// pre-2026-09-10 behaviour, or higher to bisect a suspected short-prompt prefill bug.
fn prefill_min_tokens() -> usize {
    prefill_min_tokens_from(std::env::var("NPU_LLM_PREFILL_MIN_TOKENS").ok().as_deref())
}

/// The measured crossover above. Named rather than spelled `12` at the call site.
const PREFILL_BREAK_EVEN_TOKENS: usize = 12;

/// Split out from the env read so it can be tested without touching the process environment --
/// these tests run in parallel threads of one process, and a test that sets a variable another
/// test's default depends on fails the OTHER test, intermittently.
fn prefill_min_tokens_from(raw: Option<&str>) -> usize {
    raw.and_then(|v| v.parse().ok()).unwrap_or(PREFILL_BREAK_EVEN_TOKENS).max(1)
}

/// A seed for when the caller does not ask for reproducibility. Not cryptographic -- just distinct
/// across requests so "no seed" does not silently mean "the same sequence every time", which
/// `GenerateParams::seed: Option<u64>` promises only when the caller actually supplies one.
fn default_seed() -> u64 {
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0);
    nanos ^ ((std::process::id() as u64) << 32)
}

impl<D: DecodeStep> TextGenerator for LlmGenerator<D> {
    fn bo_bytes(&self) -> u64 {
        self.decode.bo_bytes()
    }

    fn generate(
        &mut self,
        prompt: &Prompt,
        params: &GenerateParams,
        sink: &mut dyn FnMut(Chunk<'_>) -> bool,
    ) -> Result<(), EngineError> {
        // Per-request override, set BEFORE `reset()` so its own `dispatch_log::enabled()` check
        // (which zeroes the log for this generation) already sees it, not the previous request's.
        npu_xrt::dispatch_log::set_override(params.dispatch_log);
        // One clock for the whole generation: every span in the report is measured against `t0`,
        // so the parts and the wall clock cannot drift apart and the residual means something.
        let t0 = Instant::now();
        // Every generation starts from an empty context. The device backend's KV cache only
        // grows with `pos`, so without this each request continues the previous one's. Zeroing it
        // is prompt-side setup, so its cost belongs to the prefill window rather than to a span of
        // its own; tokenization is carved back out of the middle of that window below.
        self.decode.reset()?;
        let mut prefill_us = t0.elapsed().as_micros() as u64;
        let t_tokenize = Instant::now();
        let prompt_ids = tokenize_prompt(&self.cfg, prompt, params.enable_thinking)?;
        let tokenize_us = t_tokenize.elapsed().as_micros() as u64;
        if prompt_ids.is_empty() {
            return Err(EngineError::Unsupported("prompt tokenized to zero tokens".to_string()));
        }
        let prompt_tokens = prompt_ids.len() as u32;
        // The KV window is a HARD bound, and crossing it is silent rather than loud: `pos` becomes
        // a `kv_off` element offset into a cache holding exactly S positions, so position S lands
        // one element past its end -- in-arena, past the artifact's own bounds check, answering
        // from what it overwrote. Priming walks positions `0..prompt_len`, so the prompt must fit.
        let max_context = self.decode.max_context();
        if let Some(max_ctx) = max_context {
            if prompt_ids.len() > max_ctx {
                return Err(EngineError::Unsupported(format!(
                    "prompt is {} tokens but this model's context window is {max_ctx} \
                     (the decode artifact was built at max_seq={max_ctx}); \
                     shorten the prompt or load an artifact built with a larger window",
                    prompt_ids.len()
                )));
            }
        }

        // The one place the tiers collapse: request -> scenario -> checkpoint -> engine.
        let gen = params.resolve(&self.scenario_defaults, &self.cfg.checkpoint_defaults);
        let sampling_cfg = SamplingConfig {
            temperature: gen.temperature,
            top_k: gen.top_k as usize,
            top_p: gen.top_p,
            repetition_penalty: gen.repetition_penalty,
            frequency_penalty: gen.frequency_penalty,
            presence_penalty: gen.presence_penalty,
        };
        let mut rng = SplitMix64::new(params.seed.unwrap_or_else(default_seed));
        // Penalties see the whole context, prompt included -- OpenAI's own wording ("existing
        // frequency in the text so far") and every mainstream server read this as prompt+completion.
        let mut history: Vec<u32> = prompt_ids.clone();
        let mut detok = IncrementalDetokenizer::new();
        let mut stopper = StopMatcher::new(params.stop.clone());

        // Prime the KV cache over the prompt. The last position always goes through `step`,
        // because only `step` returns logits and those are what the first `sample()` reads; every
        // position before it is a pure KV side effect, which is exactly what the batched path
        // produces. `primed <= prompt_ids.len() - 1` therefore holds by construction and the loop
        // below always runs at least once.
        let t_prefill = Instant::now();
        let mut counters = self.decode.counters();
        let batchable = prompt_ids.len() - 1;
        let mut primed = 0usize;
        // `prefill_batch()` is the CAPABILITY probe -- `Some` means a batched prefill artifact is
        // loaded. Its `M` no longer gates the decision: `prime()` pads a partial chunk itself, and
        // paying for the padding beats paying for the dispatches. See `prefill_min_tokens()`.
        if self.decode.prefill_batch().is_some() && batchable >= prefill_min_tokens() {
            primed = self.decode.prefill(&prompt_ids[..batchable])?;
        }
        let mut logits = Vec::new();
        for (i, &tok) in prompt_ids.iter().enumerate().skip(primed) {
            logits = self.decode.step(tok, i)?;
        }
        prefill_us += t_prefill.elapsed().as_micros() as u64;
        let (prefill_dispatches, _) = counter_delta(&mut counters, self.decode.counters());
        let prefill = PrefillRecord {
            tokens: prompt_tokens,
            batched: primed as u32,
            stepwise: (prompt_ids.len() - primed) as u32,
            us: prefill_us,
            dispatches: prefill_dispatches,
        };
        // Floor for the decode-only design breakdown: everything dispatched up to here is
        // prefill's, on the SAME accumulator `reset()` zeroed before prefill ran. `decode_design`
        // tracks forward from it at the per-token cadence below (mirroring `counters`), so the diff
        // at the end can never include a prefill dispatch and stops at the boundary `phases.step_us`
        // itself stops at.
        let prefill_design = self.decode.design_breakdown();
        let mut decode_design = prefill_design.clone();
        let mut steps: Vec<StepRecord> = Vec::new();
        let mut last_t = t0.elapsed().as_micros() as u64;
        let mut pending_step_us = 0u64;

        let max_tokens = gen.max_tokens;

        let mut completion_tokens = 0u32;
        let finish: FinishReason;
        let mut pos = prompt_ids.len();

        // Checked BOTH before sampling (so `max_tokens: 0` never samples at all) and again right
        // after a token is accepted (so the loop never pays for a device dispatch it will not use).
        'decode: loop {
            if completion_tokens >= max_tokens {
                finish = FinishReason::Length;
                break;
            }
            // `outcome.penalties_skipped` is always 0 here: a full-vocabulary `LogitView` has no
            // absent ids. It becomes live traffic once a device-side top-k slice
            // (`llm-onchip-topk-feedback`) replaces `LogitView::full` -- the field already exists so
            // that swap does not need a new signature to carry the count.
            let t_sample = Instant::now();
            let outcome = sampling::sample(LogitView::full(&logits), &history, &sampling_cfg, &mut rng);
            let sample_us = t_sample.elapsed().as_micros() as u64;
            let tok = outcome.token;

            // The stop token gets no record: it is never emitted, and the dispatch that produced
            // its logits falls outside the first-to-last-token window the decode rate spans.
            if self.cfg.stop.is_stop(tok) {
                finish = FinishReason::Stop;
                break;
            }
            history.push(tok);
            completion_tokens += 1;

            let t_detok = Instant::now();
            let text = detok.push(tok, &self.cfg.tokenizer)?;
            let (emit, matched) = match stopper.feed(&text) {
                StopFeed::Emit(t) => (t, false),
                StopFeed::Matched(t) => (t, true),
            };
            let detok_us = t_detok.elapsed().as_micros() as u64;

            let t_us = t0.elapsed().as_micros() as u64;
            let (dispatches, transitions) = counter_delta(&mut counters, self.decode.counters());
            // Same boundary as `dispatches` above, so the two can never disagree about how far the
            // decode window extends.
            decode_design = self.decode.design_breakdown();
            let rec = StepRecord {
                seq: completion_tokens - 1,
                token: Some(tok),
                text,
                emit,
                t_us,
                dt_us: t_us.saturating_sub(last_t),
                // `pending_step_us` is the dispatch from the END of the previous iteration -- the
                // one that produced the logits this token was sampled from. The first token's is 0:
                // its logits came from priming, and charging them here would double-count prefill.
                // `Some` even on the greedy short-circuit: `outcome.timings` is a true measured zero
                // there, not an absent measurement -- see `SamplePhases`'s doc.
                phases: StepPhases {
                    step_us: std::mem::take(&mut pending_step_us),
                    sample_us,
                    detok_us,
                    sample_phases: Some(SamplePhases {
                        penalties_us: outcome.timings.penalties_ns / 1_000,
                        top_k_us: outcome.timings.top_k_ns / 1_000,
                        top_p_us: outcome.timings.top_p_ns / 1_000,
                        draw_us: outcome.timings.draw_ns / 1_000,
                    }),
                },
                dispatches,
                transitions,
            };
            last_t = t_us;

            // The text frame first, then its measurement, so a consumer that renders from `Step`
            // alone sees the same order a consumer rendering from `Text` does. Both are asked
            // whether the client is still there; a token that completed no codepoint emits no text,
            // and `Step` is then the only place a disconnect can be noticed.
            let live = rec.emit.is_empty() || sink(Chunk::Text(&rec.emit));
            let live = sink(Chunk::Step(&rec)) && live;
            steps.push(rec);
            if matched {
                // A matched stop ends the generation on its own terms, so a client that hangs up on
                // this last frame still reports `Stop` -- which is what the loop did before.
                finish = FinishReason::Stop;
                break 'decode;
            }
            if !live {
                finish = FinishReason::Aborted;
                break 'decode;
            }

            if completion_tokens >= max_tokens {
                finish = FinishReason::Length;
                break;
            }
            // Same bound, at the other end: the next dispatch would write position `pos`, so stop
            // while `pos` is still inside the window rather than after it has been overwritten.
            if max_context.is_some_and(|m| pos >= m) {
                finish = FinishReason::Length;
                break;
            }
            let t_step = Instant::now();
            logits = self.decode.step(tok, pos)?;
            pending_step_us = t_step.elapsed().as_micros() as u64;
            pos += 1;
        }

        let tail = stopper.flush();
        if !tail.is_empty() {
            // Text with no token behind it. It still gets a record, with `token: None`: a log that
            // dropped these frames would not replay to the same bytes, and a rate that counted them
            // as tokens would be wrong in the other direction.
            let t_us = t0.elapsed().as_micros() as u64;
            let rec = StepRecord {
                seq: steps.len() as u32,
                token: None,
                emit: tail,
                t_us,
                dt_us: t_us.saturating_sub(last_t),
                ..StepRecord::default()
            };
            sink(Chunk::Text(&rec.emit));
            sink(Chunk::Step(&rec));
            steps.push(rec);
        }
        // `pos` is the KV position the next dispatch would have written -- i.e. the one this
        // generation actually reached, prompt included. The backend supplies everything else about
        // the build that produced these numbers; this is the one field only the loop can see.
        let provenance = ArmProvenance { n_past: Some(pos as u32), ..self.decode.provenance() };
        let report = GenerationReport {
            tokenize_us,
            prefill,
            steps,
            generate_us: t0.elapsed().as_micros() as u64,
            usage: GenerateUsage { prompt_tokens, completion_tokens },
            design_breakdown: diff_design_breakdown(&prefill_design, &decode_design),
            provenance,
            ..GenerationReport::default()
        };
        sink(Chunk::Done { reason: finish, usage: report.usage, report: &report });
        // After Done, never before: the report is diagnostics, and a caller streaming to a socket
        // must get its terminator whatever the backend has to say. stderr, so it cannot land in
        // an SSE body.
        if let Some(r) = self.decode.dispatch_report() {
            eprintln!("[dispatch] {prompt_tokens} prompt + {completion_tokens} completion tokens\n{r}");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::telemetry::Bound;
    use crate::llm::chat_template::ChatTemplate;
    use crate::pipeline::GenerationDefaults;
    use crate::llm::config::StopTokens;
    use crate::pipeline::ChatMessage;
    use std::collections::HashMap;
    use tokenizers::models::wordlevel::WordLevelBuilder;
    use tokenizers::pre_tokenizers::whitespace::Whitespace;
    use tokenizers::decoders::wordpiece::WordPiece as WordPieceDecoder;
    use tokenizers::Tokenizer;

    /// vocab ids: 0 <unk>, 1 hello, 2 world, 3 stop_word, 4 im_end (EOS), plus room for tests.
    fn build_cfg(chat_template: Option<&str>) -> ModelConfig {
        let vocab: HashMap<String, u32> = [
            ("<unk>", 0u32),
            ("hello", 1),
            ("world", 2),
            ("stop_word", 3),
            ("<|im_end|>", 4),
            ("foo", 5),
            ("bar", 6),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
        let model = WordLevelBuilder::new().vocab(vocab).unk_token("<unk>".to_string()).build().unwrap();
        let mut tok = Tokenizer::new(model);
        tok.with_pre_tokenizer(Some(Whitespace {}));
        tok.with_decoder(Some(WordPieceDecoder::default()));
        let stop = StopTokens { chat_eos: 4, generation_eos: None };
        ModelConfig::new(tok, chat_template.map(|s| ChatTemplate::new(s.to_string())), stop)
    }

    // Priming consumes exactly `prompt_ids.len()` script entries; the LAST one primed is what the
    // FIRST `sample()` call in the decode loop sees. Each further accepted token consumes one more
    // entry before the NEXT sample() call. All tests below use `temperature: 0.0` (pure argmax) so
    // logits are peaked deliberately and the outcome never depends on the RNG/seed.

    #[test]
    fn stops_on_eos_token_and_reports_usage() {
        let cfg = build_cfg(None);
        // prompt "hello" (1 token) -> priming consumes entry 0, which is also the first sample().
        let decode = ScriptedDecodeStep::new(vec![
            vec![0.0, 0.0, 9.0, 0.0, 0.0], // sample -> id 2 ("world")
            vec![0.0, 0.0, 0.0, 0.0, 9.0], // sample -> id 4 (EOS)
        ]);
        let mut gen = LlmGenerator::new(cfg, decode);
        let params = GenerateParams { max_tokens: Some(50), temperature: Some(0.0), ..GenerateParams::default() };
        let (text, reason, usage) = gen.generate_to_string(&Prompt::Raw("hello".to_string()), &params).unwrap();
        assert_eq!(text, "world");
        assert_eq!(reason, FinishReason::Stop);
        assert_eq!(usage.prompt_tokens, 1);
        assert_eq!(usage.completion_tokens, 1, "the EOS token itself must not be counted");
    }

    #[test]
    fn a_configured_default_applies_when_the_request_omits_max_tokens() {
        let cfg = build_cfg(None);
        let peak = |id: usize| { let mut v = vec![0.0; 7]; v[id] = 9.0; v };
        let decode = ScriptedDecodeStep::new(vec![peak(2), peak(5), peak(6), peak(1)]);
        let mut gen = LlmGenerator::new(cfg, decode).with_scenario_defaults(GenerationDefaults { max_tokens: Some(2), ..Default::default() });
        // max_tokens unset -- the model's own budget of 2 decides, not the engine's 256.
        let params = GenerateParams { temperature: Some(0.0), ..GenerateParams::default() };
        let (text, reason, usage) =
            gen.generate_to_string(&Prompt::Raw("hello".to_string()), &params).unwrap();
        assert_eq!(reason, FinishReason::Length);
        assert_eq!(usage.completion_tokens, 2);
        assert_eq!(text, "world foo");
    }

    #[test]
    fn an_explicit_request_value_beats_the_configured_default_in_both_directions() {
        let peak = |id: usize| { let mut v = vec![0.0; 7]; v[id] = 9.0; v };
        // Asking for MORE than the model default gets more: the default is a default, not a cap.
        let decode = ScriptedDecodeStep::new(vec![peak(2), peak(5), peak(6), peak(1)]);
        let mut gen = LlmGenerator::new(build_cfg(None), decode).with_scenario_defaults(GenerationDefaults { max_tokens: Some(1), ..Default::default() });
        let params = GenerateParams { max_tokens: Some(3), temperature: Some(0.0), ..GenerateParams::default() };
        let (_, _, usage) =
            gen.generate_to_string(&Prompt::Raw("hello".to_string()), &params).unwrap();
        assert_eq!(usage.completion_tokens, 3, "an explicit request must be able to exceed the default");

        // And asking for less gets less.
        let decode = ScriptedDecodeStep::new(vec![peak(2), peak(5), peak(6), peak(1)]);
        let mut gen = LlmGenerator::new(build_cfg(None), decode).with_scenario_defaults(GenerationDefaults { max_tokens: Some(3), ..Default::default() });
        let params = GenerateParams { max_tokens: Some(1), temperature: Some(0.0), ..GenerateParams::default() };
        let (_, _, usage) =
            gen.generate_to_string(&Prompt::Raw("hello".to_string()), &params).unwrap();
        assert_eq!(usage.completion_tokens, 1);
    }

    #[test]
    fn with_no_configured_default_the_engine_default_still_applies() {
        let cfg = build_cfg(None);
        // 256 is a lot of script, so assert the resolution rather than the count: an unset request
        // against an unset model default must not collapse to zero tokens.
        let decode = ScriptedDecodeStep::new(vec![vec![0.0, 0.0, 9.0, 0.0, 0.0], vec![0.0, 0.0, 0.0, 0.0, 9.0]]);
        let mut gen = LlmGenerator::new(cfg, decode);
        let params = GenerateParams { temperature: Some(0.0), ..GenerateParams::default() };
        assert_eq!(params.max_tokens, None);
        let (text, reason, _) =
            gen.generate_to_string(&Prompt::Raw("hello".to_string()), &params).unwrap();
        assert_eq!(reason, FinishReason::Stop);
        assert_eq!(text, "world");
    }

    #[test]
    fn a_prompt_longer_than_the_kv_window_is_refused_naming_both_numbers() {
        let cfg = build_cfg(None);
        // Window of 2, prompt of 3 -- priming alone would walk positions 0,1,2 and write past the
        // last row of head 0. Nothing downstream can detect that, so it must be refused here.
        let decode = ScriptedDecodeStep::new(vec![vec![0.0, 0.0, 9.0, 0.0, 0.0]]).with_max_context(2);
        let mut gen = LlmGenerator::new(cfg, decode);
        let params = GenerateParams { max_tokens: Some(50), temperature: Some(0.0), ..GenerateParams::default() };
        let err = gen
            .generate_to_string(&Prompt::Raw("hello world foo".to_string()), &params)
            .expect_err("a prompt that does not fit the window must not run");
        let msg = err.to_string();
        assert!(msg.contains('3'), "must name the prompt length: {msg}");
        assert!(msg.contains('2'), "must name the window: {msg}");
    }

    #[test]
    fn generation_stops_at_the_kv_window_before_overwriting_it() {
        let cfg = build_cfg(None);
        // Window of 3, prompt "hello" (1 token). Priming consumes entry 0 at position 0, so `pos`
        // enters the loop at 1. Positions 1 and 2 are legal; the step that would write position 3
        // must not happen. That allows exactly 3 accepted tokens, and the script is deliberately
        // LONGER than that -- if the bound did not fire the loop would happily keep going.
        let peak = |id: usize| {
            let mut v = vec![0.0; 7];
            v[id] = 9.0;
            v
        };
        let decode = ScriptedDecodeStep::new(vec![peak(2), peak(5), peak(6), peak(1), peak(2)])
            .with_max_context(3);
        let mut gen = LlmGenerator::new(cfg, decode);
        let params = GenerateParams { max_tokens: Some(50), temperature: Some(0.0), ..GenerateParams::default() };
        let (text, reason, usage) =
            gen.generate_to_string(&Prompt::Raw("hello".to_string()), &params).unwrap();
        assert_eq!(reason, FinishReason::Length, "hitting the window is a length stop");
        assert_eq!(usage.completion_tokens, 3, "one token per legal position, and not one more");
        assert_eq!(text, "world foo bar");
    }

    #[test]
    fn a_prompt_exactly_filling_the_window_still_emits_from_its_last_logits() {
        let cfg = build_cfg(None);
        // prompt_len == max_context is legal: priming walks 0..=1 and stops inside the window. The
        // last primed logits are real, so one token is sampled from them before the loop stops.
        let decode = ScriptedDecodeStep::new(vec![vec![0.0; 7], {
            let mut v = vec![0.0; 7];
            v[2] = 9.0;
            v
        }])
        .with_max_context(2);
        let mut gen = LlmGenerator::new(cfg, decode);
        let params = GenerateParams { max_tokens: Some(50), temperature: Some(0.0), ..GenerateParams::default() };
        let (text, reason, usage) =
            gen.generate_to_string(&Prompt::Raw("hello world".to_string()), &params).unwrap();
        assert_eq!(reason, FinishReason::Length);
        assert_eq!(usage.completion_tokens, 1);
        assert_eq!(text, "world");
    }

    #[test]
    fn a_backend_with_no_window_is_unbounded_as_before() {
        let cfg = build_cfg(None);
        // max_context() defaults to None, so nothing about the existing host path changes.
        let decode = ScriptedDecodeStep::new(vec![
            vec![0.0, 0.0, 9.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 9.0],
        ]);
        assert!(decode.max_context().is_none());
        let mut gen = LlmGenerator::new(cfg, decode);
        let params = GenerateParams { max_tokens: Some(50), temperature: Some(0.0), ..GenerateParams::default() };
        let (_, reason, _) =
            gen.generate_to_string(&Prompt::Raw("hello".to_string()), &params).unwrap();
        assert_eq!(reason, FinishReason::Stop);
    }

    // ------------------------------------------------------------------------------------
    // Batched prefill wiring. The batched path returns no logits, so nothing about it is visible
    // in the generated text -- these assert on the script the per-token loop still consumes and on
    // the mock's recorded prefill calls.
    // ------------------------------------------------------------------------------------

    #[test]
    fn a_prompt_of_at_least_one_batch_is_primed_by_prefill_except_its_last_token() {
        let cfg = build_cfg(None);
        // 14 prompt tokens against a batch of 2 -- several whole chunks. Prefill takes the first
        // thirteen; only the last position goes through `step`, so the script needs exactly two
        // entries (that priming step, then one more after the first accepted token). The prompt is
        // long enough to clear the break-even threshold, which is a separate decision.
        let peak = |id: usize| { let mut v = vec![0.0; 7]; v[id] = 9.0; v };
        let decode = ScriptedDecodeStep::new(vec![peak(2), peak(4)]).with_prefill_batch(2);
        let mut gen = LlmGenerator::new(cfg, decode);
        let params = GenerateParams { max_tokens: Some(50), temperature: Some(0.0), ..GenerateParams::default() };
        let (text, reason, usage) =
            gen.generate_to_string(&Prompt::Raw("a b c d e f g h i j k l m n".to_string()), &params).unwrap();
        assert_eq!(gen.backend().prefill_calls, vec![13], "prefill takes the prompt minus its last token");
        assert_eq!(usage.prompt_tokens, 14);
        assert_eq!(text, "world");
        assert_eq!(reason, FinishReason::Stop);
    }

    #[test]
    fn a_prompt_under_the_break_even_keeps_the_per_token_path() {
        let cfg = build_cfg(None);
        // 3 prompt tokens: 2 batchable, under the measured crossover, so all three go through
        // `step`. Not a capability limit -- `prime()` pads a partial chunk perfectly well -- but a
        // padded chunk costs one whole prefill dispatch and two decode steps cost less than that.
        let peak = |id: usize| { let mut v = vec![0.0; 7]; v[id] = 9.0; v };
        let decode = ScriptedDecodeStep::new(vec![peak(0), peak(0), peak(2), peak(4)]).with_prefill_batch(4);
        let mut gen = LlmGenerator::new(cfg, decode);
        let params = GenerateParams { max_tokens: Some(50), temperature: Some(0.0), ..GenerateParams::default() };
        let (text, _, usage) =
            gen.generate_to_string(&Prompt::Raw("hello world foo".to_string()), &params).unwrap();
        assert!(gen.backend().prefill_calls.is_empty(), "under the break-even, step() is cheaper");
        assert_eq!(usage.prompt_tokens, 3);
        assert_eq!(text, "world");
    }

    #[test]
    fn a_partial_chunk_over_the_break_even_is_primed_in_one_call() {
        // 14 prompt tokens against a batch of 256: 13 batchable, over the crossover but far under
        // one chunk. The whole point of the 2026-09-10 flip -- the old `batchable >= M` rule sent
        // this prompt through 14 sequential dispatches to avoid padding one.
        let cfg = build_cfg(None);
        let peak = |id: usize| { let mut v = vec![0.0; 7]; v[id] = 9.0; v };
        let decode = ScriptedDecodeStep::new(vec![peak(2), peak(4)]).with_prefill_batch(256);
        let mut gen = LlmGenerator::new(cfg, decode);
        let params = GenerateParams { max_tokens: Some(50), temperature: Some(0.0), ..GenerateParams::default() };
        let prompt = Prompt::Raw("a b c d e f g h i j k l m n".to_string());
        let (_, _, usage) = gen.generate_to_string(&prompt, &params).unwrap();
        assert_eq!(usage.prompt_tokens, 14);
        assert_eq!(gen.backend().prefill_calls, vec![13], "one call, not 13 steps");
    }

    #[test]
    fn the_threshold_is_the_measured_break_even_and_never_zero() {
        // The default is the measured crossover, not 1: below it the padded chunk costs more than
        // the dispatches it replaces (10 batchable tokens measured 315 ms per-token, 362 batched).
        assert_eq!(prefill_min_tokens_from(None), PREFILL_BREAK_EVEN_TOKENS);
        assert!(PREFILL_BREAK_EVEN_TOKENS > 1, "1 would batch prompts the sweep says to step");
        // The override is the bisect handle for a suspected short-prompt prefill bug -- raise it
        // to the artifact's M to get the pre-2026-09-10 behaviour back.
        assert_eq!(prefill_min_tokens_from(Some("256")), 256);
        // 0 would mean "batch a zero-token prompt"; garbage is not a threshold at all.
        assert_eq!(prefill_min_tokens_from(Some("0")), 1);
        assert_eq!(prefill_min_tokens_from(Some("lots")), PREFILL_BREAK_EVEN_TOKENS);
    }

    #[test]
    fn a_backend_with_no_batched_path_primes_exactly_as_before() {
        let cfg = build_cfg(None);
        // prefill_batch() defaults to None, so the priming loop is byte-for-byte the old one: four
        // script entries consumed for a four-token prompt.
        let peak = |id: usize| { let mut v = vec![0.0; 7]; v[id] = 9.0; v };
        let decode = ScriptedDecodeStep::new(vec![peak(0), peak(0), peak(0), peak(2), peak(4)]);
        assert!(decode.prefill_batch().is_none());
        let mut gen = LlmGenerator::new(cfg, decode);
        let params = GenerateParams { max_tokens: Some(50), temperature: Some(0.0), ..GenerateParams::default() };
        let (text, _, usage) =
            gen.generate_to_string(&Prompt::Raw("hello world foo bar".to_string()), &params).unwrap();
        assert_eq!(usage.prompt_tokens, 4);
        assert_eq!(text, "world");
    }

    #[test]
    fn the_last_prompt_token_always_goes_through_step_even_at_an_exact_multiple() {
        let cfg = build_cfg(None);
        // 14 prompt tokens, batch 1: prefill could cover all fourteen, and must not -- only `step`
        // returns logits and the first sample() reads them. It takes thirteen.
        let peak = |id: usize| { let mut v = vec![0.0; 7]; v[id] = 9.0; v };
        let decode = ScriptedDecodeStep::new(vec![peak(2), peak(4)]).with_prefill_batch(1);
        let mut gen = LlmGenerator::new(cfg, decode);
        let params = GenerateParams { max_tokens: Some(50), temperature: Some(0.0), ..GenerateParams::default() };
        let (text, _, _) =
            gen.generate_to_string(&Prompt::Raw("a b c d e f g h i j k l m n".to_string()), &params).unwrap();
        assert_eq!(gen.backend().prefill_calls, vec![13]);
        assert_eq!(text, "world", "the sampled logits came from step(pos=13), not from prefill");
    }

    #[test]
    fn a_backend_that_declines_prefill_falls_back_with_no_positions_skipped() {
        // What `NPU_LLM_PREFILL_BATCHED=0` and a missing prefill artifact both look like from here:
        // `prefill_batch()` is Some, `prefill()` primes 0, and the per-token loop covers everything.
        struct Declining(ScriptedDecodeStep);
        impl DecodeStep for Declining {
            fn step(&mut self, t: u32, p: usize) -> Result<Vec<f32>, EngineError> { self.0.step(t, p) }
            fn prefill_batch(&self) -> Option<usize> { Some(1) }
            fn prefill(&mut self, _tokens: &[u32]) -> Result<usize, EngineError> { Ok(0) }
        }
        let peak = |id: usize| { let mut v = vec![0.0; 7]; v[id] = 9.0; v };
        let decode = Declining(ScriptedDecodeStep::new(vec![peak(0), peak(0), peak(2), peak(4)]));
        let mut gen = LlmGenerator::new(build_cfg(None), decode);
        let params = GenerateParams { max_tokens: Some(50), temperature: Some(0.0), ..GenerateParams::default() };
        let (text, _, usage) =
            gen.generate_to_string(&Prompt::Raw("hello world foo".to_string()), &params).unwrap();
        assert_eq!(usage.prompt_tokens, 3);
        assert_eq!(text, "world");
    }

    #[test]
    fn max_tokens_stops_the_loop_with_length_reason() {
        let cfg = build_cfg(None);
        let decode = ScriptedDecodeStep::new(vec![
            vec![0.0, 0.0, 9.0, 0.0, 0.0], // -> "world"
            vec![0.0, 0.0, 9.0, 0.0, 0.0], // -> "world" again
        ]);
        let mut gen = LlmGenerator::new(cfg, decode);
        let params = GenerateParams { max_tokens: Some(2), temperature: Some(0.0), ..GenerateParams::default() };
        let (text, reason, usage) = gen.generate_to_string(&Prompt::Raw("hello".to_string()), &params).unwrap();
        assert_eq!(text, "world world");
        assert_eq!(reason, FinishReason::Length, "must stop WITHOUT an extra wasted device dispatch");
        assert_eq!(usage.completion_tokens, 2);
    }

    /// Control the instrument before trusting it: drive the loop with a backend whose step time
    /// the test chose, and check the report says so. Without this every number downstream is
    /// plausible and unverified.
    #[test]
    fn the_report_measures_a_known_step_delay() {
        use std::time::Duration;
        let cfg = build_cfg(None);
        // Four decode steps, none of them the stop token, then the script runs out -- so the loop
        // ends on max_tokens with four tokens recorded.
        let decode = ScriptedDecodeStep::new(vec![vec![0.0, 0.0, 9.0, 0.0, 0.0]; 5])
            .with_step_delay(Duration::from_millis(5))
            .with_counters();
        let mut gen = LlmGenerator::new(cfg, decode);
        let params = GenerateParams { max_tokens: Some(4), temperature: Some(0.0), ..GenerateParams::default() };
        let mut steps: Vec<StepRecord> = Vec::new();
        let mut report = GenerationReport::default();
        gen.generate(&Prompt::Raw("hello".to_string()), &params, &mut |c| {
            match c {
                Chunk::Step(r) => steps.push(r.clone()),
                Chunk::Done { report: r, .. } => report = r.clone(),
                Chunk::Text(_) => {}
            }
            true
        })
        .unwrap();

        assert_eq!(steps.len(), 4, "one record per token");
        assert_eq!(report.steps, steps, "the streamed records and the final report are one object");
        assert_eq!(report.usage.completion_tokens, 4);
        assert!(steps.iter().enumerate().all(|(i, s)| s.seq == i as u32), "seq is dense and in order");
        assert!(steps.iter().all(|s| s.token == Some(2)), "the scripted argmax is token 2");

        // The first token's logits came from priming, so its step is prefill's, not its own.
        assert_eq!(steps[0].phases.step_us, 0);
        for s in &steps[1..] {
            assert!(
                (4_000..50_000).contains(&s.phases.step_us),
                "5 ms sleep should land in step_us, got {} us",
                s.phases.step_us
            );
            assert!(s.dt_us >= s.phases.step_us, "a gap cannot be shorter than the phase inside it");
        }

        // The prompt walk is prefill: one dispatch per prompt token, none of them charged to a
        // completion token. `hello` is one token in the test tokenizer.
        assert_eq!(report.prefill.tokens, 1);
        assert_eq!(report.prefill.stepwise, 1);
        assert_eq!(report.prefill.dispatches, Some(1));
        assert_eq!(steps[0].dispatches, Some(0), "no dispatch produced the first token's logits");
        assert!(steps[1..].iter().all(|s| s.dispatches == Some(1)));

        let sum = report.summarize();
        assert_eq!(sum.completion_tokens, 4);
        assert_eq!(sum.bound, Bound::Device, "a 5 ms device step against microsecond host work");
        assert!(sum.tok_per_s > 0.0 && sum.tok_per_s < 400.0, "tok/s: {}", sum.tok_per_s);
        assert!(sum.total_us >= sum.decode_us);
    }

    /// Reproduces the production bug directly: a stepwise prefill and the decode loop both dispatch
    /// through the SAME `design_breakdown()` accumulator (mirroring `npu_xrt::dispatch_log`, which
    /// is reset once at generation start, before prefill runs). Before the fix this test's report
    /// would show 4 dispatches / 0.04s -- prefill's 3 plus decode's 1 -- against a device row built
    /// from one 20ms decode dispatch, i.e. a device-by-stream total exceeding the device row it is
    /// supposed to be a breakdown of, exactly what produced 109.9% in production.
    #[test]
    fn design_breakdown_excludes_prefill_dispatches() {
        use std::time::Duration;
        let cfg = build_cfg(None);
        let peak = |id: usize| { let mut v = vec![0.0; 7]; v[id] = 9.0; v };
        // "hello world foo" -- 3 prompt tokens, no batched prefill, so priming walks all 3 positions
        // through `step()` (3 dispatches) before the first sample. One more decode dispatch fetches
        // the second token's logits, and `max_tokens: 2` stops the loop right after -- no wasted
        // trailing dispatch, so every decode dispatch here is cleanly attributed.
        let decode = ScriptedDecodeStep::new(vec![vec![0.0; 7], vec![0.0; 7], peak(2), peak(5)])
            .with_step_delay(Duration::from_millis(20))
            .with_design_tracking(0.01);
        let mut gen = LlmGenerator::new(cfg, decode);
        let params = GenerateParams { max_tokens: Some(2), temperature: Some(0.0), ..GenerateParams::default() };
        let mut report = GenerationReport::default();
        gen.generate(&Prompt::Raw("hello world foo".to_string()), &params, &mut |c| {
            if let Chunk::Done { report: r, .. } = c {
                report = r.clone();
            }
            true
        })
        .unwrap();

        assert_eq!(report.usage.completion_tokens, 2);
        assert_eq!(report.prefill.stepwise, 3);
        assert_eq!(report.design_breakdown.len(), 1);
        assert_eq!(
            report.design_breakdown[0].dispatches, 1,
            "prefill's 3 dispatches must not be in the decode-window breakdown: {:?}", report.design_breakdown
        );
        assert!(
            (report.design_breakdown[0].secs - 0.01).abs() < 1e-9,
            "expected one decode dispatch's worth of secs, got {:?}", report.design_breakdown
        );

        let sum = report.summarize();
        let design_total_us: f64 = sum.design_breakdown.iter().map(|d| d.secs * 1e6).sum();
        assert!(
            design_total_us <= sum.phases.step_us as f64,
            "device-by-stream total ({design_total_us} us) must never exceed the device row \
             ({} us) it explains -- this inequality failing is what 109.9% looked like",
            sum.phases.step_us
        );
    }

    /// The records must reproduce the bytes the text frames carried. A log that cannot do this is
    /// not a log of the run, and every replay built on it is fiction.
    #[test]
    fn concatenated_emits_reproduce_the_streamed_text() {
        let cfg = build_cfg(None);
        let decode = ScriptedDecodeStep::new(vec![vec![0.0, 0.0, 9.0, 0.0, 0.0]; 4]);
        let mut gen = LlmGenerator::new(cfg, decode);
        let params = GenerateParams { max_tokens: Some(3), temperature: Some(0.0), ..GenerateParams::default() };
        let mut streamed = String::new();
        let mut from_records = String::new();
        gen.generate(&Prompt::Raw("hello".to_string()), &params, &mut |c| {
            match c {
                Chunk::Text(t) => streamed.push_str(t),
                Chunk::Step(r) => from_records.push_str(&r.emit),
                Chunk::Done { .. } => {}
            }
            true
        })
        .unwrap();
        assert!(!streamed.is_empty());
        assert_eq!(streamed, from_records);
    }

    #[test]
    fn sink_returning_false_aborts_with_aborted_reason() {
        let cfg = build_cfg(None);
        let decode = ScriptedDecodeStep::new(vec![vec![0.0, 0.0, 9.0, 0.0, 0.0]]);
        let mut gen = LlmGenerator::new(cfg, decode);
        let params = GenerateParams { max_tokens: Some(50), temperature: Some(0.0), ..GenerateParams::default() };
        let mut finish = None;
        gen.generate(&Prompt::Raw("hello".to_string()), &params, &mut |c| match c {
            Chunk::Text(_) => false, // abort on the very first text chunk
            Chunk::Step(_) => true,
            Chunk::Done { reason, .. } => {
                finish = Some(reason);
                true
            }
        })
        .unwrap();
        assert_eq!(finish, Some(FinishReason::Aborted));
    }

    #[test]
    fn stop_string_ends_generation_and_excludes_the_match() {
        let cfg = build_cfg(None);
        let decode = ScriptedDecodeStep::new(vec![
            vec![0.0, 0.0, 9.0, 0.0, 0.0], // -> "world"
            vec![0.0, 0.0, 0.0, 9.0, 0.0], // -> "stop_word"
        ]);
        let mut gen = LlmGenerator::new(cfg, decode);
        let params = GenerateParams {
            max_tokens: Some(50),
            temperature: Some(0.0),
            stop: vec!["stop_word".to_string()],
            ..GenerateParams::default()
        };
        let (text, reason, _) = gen.generate_to_string(&Prompt::Raw("hello".to_string()), &params).unwrap();
        assert_eq!(text, "world ", "stop string itself must not appear in the output");
        assert_eq!(reason, FinishReason::Stop);
    }

    #[test]
    fn greedy_is_deterministic_regardless_of_the_unseeded_default() {
        // No `seed` supplied (so each run picks its own `default_seed()`) -- greedy must still be
        // identical every time, because temperature<=0 never touches the RNG.
        let make = || {
            let cfg = build_cfg(None);
            let decode = ScriptedDecodeStep::new(vec![
                vec![0.0, 0.0, 9.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.0, 0.0, 9.0],
            ]);
            LlmGenerator::new(cfg, decode)
        };
        let params = GenerateParams { max_tokens: Some(50), temperature: Some(0.0), ..GenerateParams::default() };
        let (t1, ..) = make().generate_to_string(&Prompt::Raw("hello".to_string()), &params).unwrap();
        let (t2, ..) = make().generate_to_string(&Prompt::Raw("hello".to_string()), &params).unwrap();
        assert_eq!(t1, t2);
    }

    #[test]
    fn fixed_seed_sampling_is_reproducible() {
        // `max_tokens: 2` makes termination deterministic on its own, so the test isolates exactly
        // one thing: does the SAME seed draw the SAME two tokens from non-degenerate softmax logits.
        let script = || vec![vec![1.0, 1.0, 1.0, 1.0, 0.0], vec![1.0, 2.0, 3.0, 1.0, 0.0]];
        let params =
            GenerateParams { max_tokens: Some(2), temperature: Some(0.9), seed: Some(1234), ..GenerateParams::default() };
        let run = || {
            let cfg = build_cfg(None);
            let decode = ScriptedDecodeStep::new(script());
            let mut gen = LlmGenerator::new(cfg, decode);
            gen.generate_to_string(&Prompt::Raw("hello".to_string()), &params).unwrap().0
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn sample_phases_is_some_and_zeroed_on_the_greedy_path() {
        let cfg = build_cfg(None);
        let decode = ScriptedDecodeStep::new(vec![
            vec![0.0, 0.0, 9.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 9.0],
        ]);
        let mut gen = LlmGenerator::new(cfg, decode);
        let params = GenerateParams { max_tokens: Some(50), temperature: Some(0.0), ..GenerateParams::default() };
        let mut steps: Vec<StepRecord> = Vec::new();
        gen.generate(&Prompt::Raw("hello".to_string()), &params, &mut |c| {
            if let Chunk::Step(r) = c {
                steps.push(r.clone());
            }
            true
        })
        .unwrap();
        assert_eq!(steps.len(), 1);
        assert_eq!(
            steps[0].phases.sample_phases,
            Some(crate::telemetry::SamplePhases::default()),
            "greedy must report a measured zero, not an absent measurement"
        );
    }

    #[test]
    fn sample_phases_is_populated_on_the_sampled_path() {
        // Wiring check only: is `outcome.timings` reaching the record at all as `Some`. Magnitude
        // (that a stage's cost is really nonzero) is sampling::sample's own test, at its native
        // nanosecond precision -- at this vocab size (5) the microsecond-truncated fields here can
        // legitimately all read 0, which would make an `> 0` assertion at THIS layer flaky, not wrong.
        let cfg = build_cfg(None);
        let decode = ScriptedDecodeStep::new(vec![vec![1.0, 2.0, 3.0, 1.0, 0.0]]);
        let mut gen = LlmGenerator::new(cfg, decode);
        let params = GenerateParams {
            max_tokens: Some(1),
            temperature: Some(0.9),
            top_k: Some(3),
            seed: Some(7),
            ..GenerateParams::default()
        };
        let mut steps: Vec<StepRecord> = Vec::new();
        gen.generate(&Prompt::Raw("hello".to_string()), &params, &mut |c| {
            if let Chunk::Step(r) = c {
                steps.push(r.clone());
            }
            true
        })
        .unwrap();
        assert!(steps[0].phases.sample_phases.is_some(), "sampling must report its phases");
    }

    #[test]
    fn chat_prompt_renders_through_the_template_before_tokenizing() {
        // Template folds every message into just its content, space-joined -- proves Prompt::Chat
        // goes through render() (not raw-tokenized) and the true prompt length is the RENDERED
        // token count, not the raw message count.
        let cfg = build_cfg(Some("{%- for m in messages -%}{{ m.content }} {%- endfor -%}"));
        // prompt "hello world " (2 tokens) -> priming consumes both entries; entry 1 is what the
        // first sample() sees.
        let decode = ScriptedDecodeStep::new(vec![
            vec![0.0, 0.0, 0.0, 0.0, 0.0], // priming pos 0 ("hello") -- overwritten before sampling
            vec![0.0, 0.0, 0.0, 0.0, 9.0], // priming pos 1 ("world") -> first sample() -> EOS
        ]);
        let mut gen = LlmGenerator::new(cfg, decode);
        let params = GenerateParams { max_tokens: Some(5), temperature: Some(0.0), ..GenerateParams::default() };
        let prompt = Prompt::Chat(vec![ChatMessage::new("user", "hello world")]);
        let (text, reason, usage) = gen.generate_to_string(&prompt, &params).unwrap();
        assert_eq!(text, "");
        assert_eq!(reason, FinishReason::Stop);
        assert_eq!(usage.prompt_tokens, 2, "rendered prompt \"hello world \" tokenizes to 2 ids");
    }

    #[test]
    fn raw_prompt_never_goes_through_a_chat_template() {
        let cfg = build_cfg(Some("SHOULD NOT BE USED"));
        let decode = ScriptedDecodeStep::new(vec![vec![0.0, 0.0, 0.0, 0.0, 9.0]]); // -> EOS immediately
        let mut gen = LlmGenerator::new(cfg, decode);
        let params = GenerateParams { max_tokens: Some(5), temperature: Some(0.0), ..GenerateParams::default() };
        let (_, _, usage) = gen.generate_to_string(&Prompt::Raw("hello".to_string()), &params).unwrap();
        assert_eq!(usage.prompt_tokens, 1);
    }
}
