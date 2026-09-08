//! The decode loop: prompt tokenization -> per-token device step -> sampling -> stop conditions.
//! The device call sits behind [`DecodeStep`] so the whole loop is testable with no NPU; the
//! XRT/`ElfResident` backend against a measured fused-decode artifact is a later agent's job.

use std::collections::VecDeque;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::api::EngineError;
use crate::llm::config::ModelConfig;
use crate::llm::detokenize::{IncrementalDetokenizer, StopFeed, StopMatcher};
use crate::llm::sampling::{self, LogitView, SamplingConfig, SplitMix64};
use crate::pipeline::{Chunk, FinishReason, GenerateParams, GenerateUsage, Prompt, TextGenerator};

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
    /// backend reads it from the artifact's `dims.S`, which sizes `kc`/`vc` as `[Hkv, S, HD]`.
    ///
    /// This is a REQUIRED bound, not a hint. `step`'s `pos` becomes `kv_off = pos * head_dim` into
    /// that layout, so `pos == S` lands exactly on head 1's row 0: inside the arena, past no check,
    /// and silently answering from an overwritten cache. Enforced in [`LlmGenerator::generate`].
    fn max_context(&self) -> Option<usize> {
        None
    }

    /// Per-generation device accounting, or `None` when the backend has none or it is not enabled.
    /// Emitted by [`LlmGenerator::generate`] after the loop, paired with [`DecodeStep::reset`]
    /// before it, so the numbers cover exactly one generation. A host-side backend returns `None`;
    /// nothing in the loop branches on the answer.
    fn dispatch_report(&self) -> Option<String> {
        None
    }
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
}

impl ScriptedDecodeStep {
    pub fn new(steps: Vec<Vec<f32>>) -> Self {
        ScriptedDecodeStep { steps: steps.into(), max_context: None }
    }

    /// Give the mock a finite KV window, so the bound in [`LlmGenerator::generate`] is testable
    /// without a device.
    pub fn with_max_context(mut self, max_context: usize) -> Self {
        self.max_context = Some(max_context);
        self
    }
}

impl DecodeStep for ScriptedDecodeStep {
    fn step(&mut self, _token: u32, _pos: usize) -> Result<Vec<f32>, EngineError> {
        self.steps.pop_front().ok_or_else(|| EngineError::Device("scripted decode exhausted".to_string()))
    }

    fn max_context(&self) -> Option<usize> {
        self.max_context
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
}

/// A seed for when the caller does not ask for reproducibility. Not cryptographic -- just distinct
/// across requests so "no seed" does not silently mean "the same sequence every time", which
/// `GenerateParams::seed: Option<u64>` promises only when the caller actually supplies one.
fn default_seed() -> u64 {
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0);
    nanos ^ ((std::process::id() as u64) << 32)
}

impl<D: DecodeStep> TextGenerator for LlmGenerator<D> {
    fn generate(
        &mut self,
        prompt: &Prompt,
        params: &GenerateParams,
        sink: &mut dyn FnMut(Chunk<'_>) -> bool,
    ) -> Result<(), EngineError> {
        // Every generation starts from an empty context. The device backend's KV cache only
        // grows with `pos`, so without this each request continues the previous one's.
        self.decode.reset()?;
        let prompt_ids = tokenize_prompt(&self.cfg, prompt, params.enable_thinking)?;
        if prompt_ids.is_empty() {
            return Err(EngineError::Unsupported("prompt tokenized to zero tokens".to_string()));
        }
        let prompt_tokens = prompt_ids.len() as u32;
        // The KV window is a HARD bound, and crossing it is silent rather than loud: `pos` becomes
        // `kv_off = pos * head_dim` into a `[Hkv, S, HD]` cache, so position S lands on head 1's
        // row 0 -- in-arena, past the artifact's own bounds check, answering from a cache it just
        // overwrote. Priming walks positions `0..prompt_len`, so the prompt alone must fit.
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

        // Prime the KV cache over the prompt, one dispatch per token; only the last position's
        // logits are sampled from.
        let mut logits = Vec::new();
        for (i, &tok) in prompt_ids.iter().enumerate() {
            logits = self.decode.step(tok, i)?;
        }

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
            let outcome = sampling::sample(LogitView::full(&logits), &history, &sampling_cfg, &mut rng);
            let tok = outcome.token;

            if self.cfg.stop.is_stop(tok) {
                finish = FinishReason::Stop;
                break;
            }
            history.push(tok);
            completion_tokens += 1;

            let text = detok.push(tok, &self.cfg.tokenizer)?;
            match stopper.feed(&text) {
                StopFeed::Emit(t) => {
                    if !t.is_empty() && !sink(Chunk::Text(&t)) {
                        finish = FinishReason::Aborted;
                        break 'decode;
                    }
                }
                StopFeed::Matched(t) => {
                    if !t.is_empty() {
                        sink(Chunk::Text(&t));
                    }
                    finish = FinishReason::Stop;
                    break 'decode;
                }
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
            logits = self.decode.step(tok, pos)?;
            pos += 1;
        }

        let tail = stopper.flush();
        if !tail.is_empty() {
            sink(Chunk::Text(&tail));
        }
        sink(Chunk::Done { reason: finish, usage: GenerateUsage { prompt_tokens, completion_tokens } });
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

    #[test]
    fn sink_returning_false_aborts_with_aborted_reason() {
        let cfg = build_cfg(None);
        let decode = ScriptedDecodeStep::new(vec![vec![0.0, 0.0, 9.0, 0.0, 0.0]]);
        let mut gen = LlmGenerator::new(cfg, decode);
        let params = GenerateParams { max_tokens: Some(50), temperature: Some(0.0), ..GenerateParams::default() };
        let mut finish = None;
        gen.generate(&Prompt::Raw("hello".to_string()), &params, &mut |c| match c {
            Chunk::Text(_) => false, // abort on the very first text chunk
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
        let prompt = Prompt::Chat(vec![ChatMessage { role: "user".to_string(), content: "hello world".to_string() }]);
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
