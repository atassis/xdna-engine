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
}

impl ScriptedDecodeStep {
    pub fn new(steps: Vec<Vec<f32>>) -> Self {
        ScriptedDecodeStep { steps: steps.into() }
    }
}

impl DecodeStep for ScriptedDecodeStep {
    fn step(&mut self, _token: u32, _pos: usize) -> Result<Vec<f32>, EngineError> {
        self.steps.pop_front().ok_or_else(|| EngineError::Device("scripted decode exhausted".to_string()))
    }
}

/// An autoregressive text generator over a model directory's tokenizer/chat-template/stop-tokens and
/// a [`DecodeStep`] backend.
pub struct LlmGenerator<D: DecodeStep> {
    cfg: ModelConfig,
    decode: D,
}

impl<D: DecodeStep> LlmGenerator<D> {
    pub fn new(cfg: ModelConfig, decode: D) -> Self {
        LlmGenerator { cfg, decode }
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

        let sampling_cfg = SamplingConfig {
            temperature: params.temperature,
            top_k: params.top_k as usize,
            top_p: params.top_p,
            repetition_penalty: params.repetition_penalty,
            frequency_penalty: params.frequency_penalty,
            presence_penalty: params.presence_penalty,
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

        let mut completion_tokens = 0u32;
        let finish: FinishReason;
        let mut pos = prompt_ids.len();

        // Checked BOTH before sampling (so `max_tokens: 0` never samples at all) and again right
        // after a token is accepted (so the loop never pays for a device dispatch it will not use).
        'decode: loop {
            if completion_tokens >= params.max_tokens {
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

            if completion_tokens >= params.max_tokens {
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
        let params = GenerateParams { max_tokens: 50, temperature: 0.0, ..GenerateParams::default() };
        let (text, reason, usage) = gen.generate_to_string(&Prompt::Raw("hello".to_string()), &params).unwrap();
        assert_eq!(text, "world");
        assert_eq!(reason, FinishReason::Stop);
        assert_eq!(usage.prompt_tokens, 1);
        assert_eq!(usage.completion_tokens, 1, "the EOS token itself must not be counted");
    }

    #[test]
    fn max_tokens_stops_the_loop_with_length_reason() {
        let cfg = build_cfg(None);
        let decode = ScriptedDecodeStep::new(vec![
            vec![0.0, 0.0, 9.0, 0.0, 0.0], // -> "world"
            vec![0.0, 0.0, 9.0, 0.0, 0.0], // -> "world" again
        ]);
        let mut gen = LlmGenerator::new(cfg, decode);
        let params = GenerateParams { max_tokens: 2, temperature: 0.0, ..GenerateParams::default() };
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
        let params = GenerateParams { max_tokens: 50, temperature: 0.0, ..GenerateParams::default() };
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
            max_tokens: 50,
            temperature: 0.0,
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
        let params = GenerateParams { max_tokens: 50, temperature: 0.0, ..GenerateParams::default() };
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
            GenerateParams { max_tokens: 2, temperature: 0.9, seed: Some(1234), ..GenerateParams::default() };
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
        let params = GenerateParams { max_tokens: 5, temperature: 0.0, ..GenerateParams::default() };
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
        let params = GenerateParams { max_tokens: 5, temperature: 0.0, ..GenerateParams::default() };
        let (_, _, usage) = gen.generate_to_string(&Prompt::Raw("hello".to_string()), &params).unwrap();
        assert_eq!(usage.prompt_tokens, 1);
    }
}
