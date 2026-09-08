//! The 3-stage model abstraction. `Encoder` is the shared NPU stage (object-safe); `Frontend`
//! and `Head` are per-domain host/ONNX glue. A `Scenario` is one assembled pipeline.

use ndarray::Array2;

use crate::api::EngineError;

/// The genuinely-shared, genuinely-hard NPU stage. INTERFACE CONTRACT for sibling models
/// (GigaAM Conformer, Parakeet FastConformer, BERT): implement this and the registry can host it.
pub trait Encoder {
    /// `x` is [M, D_in] (bf16-valued f32); `valid_len` = non-padded rows. Returns [M, D].
    fn forward_last(&self, x: &Array2<f32>, valid_len: usize) -> Array2<f32>;
}

/// A full ASR backend: raw PCM samples -> text. Different ASR models (GigaAM RNNT, Parakeet TDT)
/// have different preproc + decode, so the whole transcription path is the trait; the encoder stage
/// inside each still implements `Encoder`. Fallible: an ONNX session or a decode step can fail at
/// runtime (a moved artifact, a malformed clip), and that must reach the caller as `Err`, not a panic
/// (engine-errors-are-real).
pub trait AsrModel {
    fn transcribe(&self, samples: &[i16]) -> Result<String, EngineError>;
}

/// Raw input -> encoder input activations + valid_len.
pub trait Frontend {
    type Input;
    fn run(&self, input: Self::Input) -> (Array2<f32>, usize);
}

/// Encoder output -> final result.
pub trait Head {
    type Output;
    fn run(&self, encoded: &Array2<f32>, valid_len: usize) -> Self::Output;
}

/// A text/sequence embedder: input string -> embedding vector. Implemented by both the BERT
/// (`bert::EmbedPipeline`) and ESM-2 (`esm::EsmEmbedPipeline`) pipelines so the registry can host
/// either behind one `Scenario::Embed` arm. Distinct method name (`embed_one`) delegates to each
/// pipeline's inherent `embed` (no recursion, inherent methods preserved for the verify bins).
pub trait Embedder {
    fn embed_one(&self, text: String) -> Result<Vec<f32>, EngineError>;
}

/// Speaker diarization: PCM in, speaker-attributed spans out. Same `&self` shape as `AsrModel`;
/// no interior mutability is needed because the ONNX sessions behind it are stateless per call.
pub trait Diarizer {
    fn diarize(&self, pcm: &[i16]) -> Result<Vec<crate::capability::Segment>, EngineError>;
}

/// One assembled, ready-to-serve pipeline. The registry returns this; `engine_serve` matches on it.
pub enum Scenario {
    Asr(Box<dyn AsrModel>),
    Embed(Box<dyn Embedder>),
    Diarize(Box<dyn Diarizer>),
    Generate(Box<dyn TextGenerator>),
}
// ---------------------------------------------------------------------------------------------
// Text generation (decoder-LLM). Added for `llm-serve-openai-surface`.
// ---------------------------------------------------------------------------------------------

/// One turn of a chat conversation. `role` is OpenAI's vocabulary (`system`/`user`/`assistant`);
/// it stays a String because the set is the wire protocol's, not ours, and a model's chat template
/// is free to recognise roles we have never heard of.
#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// What the caller wants continued. The distinction is not cosmetic: `Chat` must go through the
/// model's own chat template (Qwen3 is ChatML, Gemma is not) before tokenizing, and `Raw` must NOT
/// -- that is exactly the difference between `/v1/chat/completions` and `/v1/completions`.
#[derive(Debug, Clone)]
pub enum Prompt {
    Chat(Vec<ChatMessage>),
    Raw(String),
}

/// OpenAI's sampling surface. Defaults are OpenAI's defaults, not greedy: a client that sends no
/// `temperature` expects 1.0, and silently substituting greedy would make our answers differ from
/// every other server for the same request.
#[derive(Debug, Clone)]
pub struct GenerateParams {
    /// Every sampling field is an Option, and `None` means "the caller did not ask" rather than
    /// any particular value. That distinction is what lets a default exist at all: while these were
    /// plain numbers, a request arrived with every field already looking specified, so there was
    /// nowhere for a per-model or per-checkpoint default to apply. See [`GenerationDefaults`].
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    /// 0 = disabled. Not in OpenAI's schema but universal in local servers, and the device-side
    /// top-k slice (`llm-onchip-topk-feedback`) makes it the cheapest of the three to honour.
    pub top_k: Option<u32>,
    pub max_tokens: Option<u32>,
    pub stop: Vec<String>,
    pub seed: Option<u64>,
    /// `chat_template_kwargs.enable_thinking`, the one template kwarg the reasoning-model families
    /// (Qwen3, and the same spelling in vLLM/SGLang) read. `None` leaves the template's own
    /// default, which for Qwen3 is thinking ON -- and at a 256-token budget that spends the whole
    /// thing reasoning and never emits an answer. Only meaningful for `Prompt::Chat`.
    pub enable_thinking: Option<bool>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub repetition_penalty: Option<f32>,
}

impl Default for GenerateParams {
    /// Everything unset. The concrete numbers live in [`ENGINE_DEFAULTS`] and are applied at
    /// resolution time, not here -- baking them in here is exactly what made "unset" unreadable.
    fn default() -> Self {
        GenerateParams {
            temperature: None,
            top_p: None,
            top_k: None,
            max_tokens: None,
            stop: Vec::new(),
            seed: None,
            enable_thinking: None,
            presence_penalty: None,
            frequency_penalty: None,
            repetition_penalty: None,
        }
    }
}

/// Why generation stopped. OpenAI reports `stop` for both an EOS token and a matched stop string,
/// so `Stop` covers both and the distinction stays internal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// EOS token, or a `stop` sequence matched.
    Stop,
    /// `max_tokens` reached.
    Length,
    /// The sink asked to stop -- client disconnected mid-stream.
    Aborted,
}

/// One tier of generation defaults. The same shape serves the scenario's `[generation]` block and
/// the checkpoint's `generation_config.json`, so the resolution below is one `or` chain rather
/// than a special case per source.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GenerationDefaults {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub max_tokens: Option<u32>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub repetition_penalty: Option<f32>,
}

/// The last tier: what applies when nobody -- request, scenario, or checkpoint -- said anything.
/// OpenAI's numbers, kept as the final fallback rather than as the only answer.
pub const ENGINE_DEFAULTS: ResolvedGeneration = ResolvedGeneration {
    temperature: 1.0,
    top_p: 1.0,
    top_k: 0,
    max_tokens: 256,
    presence_penalty: 0.0,
    frequency_penalty: 0.0,
    repetition_penalty: 1.0,
};

/// Backwards-compatible alias for the engine's completion budget.
pub const DEFAULT_MAX_TOKENS: u32 = ENGINE_DEFAULTS.max_tokens;

/// Every sampling knob with a concrete value, after the tiers have been collapsed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResolvedGeneration {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub max_tokens: u32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub repetition_penalty: f32,
}

impl GenerateParams {
    /// Collapse the tiers, highest priority first:
    ///
    ///   request -> scenario `[generation]` -> checkpoint `generation_config.json` -> engine
    ///
    /// The checkpoint tier is there because the checkpoint is the only source that knows what the
    /// model was TUNED for. `generation_config.json` is a serialized `GenerationConfig` and stating
    /// defaults is what it is for: qwen3-0.6b asks for `temperature 0.6, top_p 0.95, top_k 20`, and
    /// answering it at the engine's 1.0/1.0/0 is unfiltered sampling at full entropy from a model
    /// that asked for a narrowed nucleus.
    pub fn resolve(&self, scenario: &GenerationDefaults, checkpoint: &GenerationDefaults) -> ResolvedGeneration {
        macro_rules! pick {
            ($f:ident) => {
                self.$f.or(scenario.$f).or(checkpoint.$f).unwrap_or(ENGINE_DEFAULTS.$f)
            };
        }
        ResolvedGeneration {
            temperature: pick!(temperature),
            top_p: pick!(top_p),
            top_k: pick!(top_k),
            max_tokens: pick!(max_tokens),
            presence_penalty: pick!(presence_penalty),
            frequency_penalty: pick!(frequency_penalty),
            repetition_penalty: pick!(repetition_penalty),
        }
    }

    /// Range-check the fields a caller set, against OpenAI's documented ranges. Shared by every
    /// surface ON PURPOSE: HTTP and the CLI validating separately is how the two drift, and an
    /// out-of-range value is silently degenerate rather than loud -- a negative temperature reads
    /// as greedy, a `top_p` above 1 disables the filter, and neither is what was asked for.
    ///
    /// `None` is never an error here: unset means a lower tier decides, and the tiers are ours.
    pub fn validate(&self) -> Result<(), String> {
        fn range(name: &str, v: Option<f32>, lo: f32, hi: f32) -> Result<(), String> {
            match v {
                Some(x) if !x.is_finite() =>
                    Err(format!("\"{name}\" must be a finite number")),
                Some(x) if x < lo || x > hi =>
                    Err(format!("\"{name}\" must be between {lo} and {hi} (got {x})")),
                _ => Ok(()),
            }
        }
        range("temperature", self.temperature, 0.0, 2.0)?;
        range("top_p", self.top_p, 0.0, 1.0)?;
        range("presence_penalty", self.presence_penalty, -2.0, 2.0)?;
        range("frequency_penalty", self.frequency_penalty, -2.0, 2.0)?;
        // Not an OpenAI field, so the bound is the one every local server uses: values <= 0 invert
        // the penalty into a BONUS for repeated tokens, which no caller means to ask for.
        range("repetition_penalty", self.repetition_penalty, 0.01, 2.0)?;
        Ok(())
    }
}

impl FinishReason {
    /// The wire name. `Aborted` reports as `stop`: OpenAI has no vocabulary for "the client hung
    /// up", and by the time it matters nobody is reading the field.
    pub fn as_str(self) -> &'static str {
        match self {
            FinishReason::Stop | FinishReason::Aborted => "stop",
            FinishReason::Length => "length",
        }
    }
}

/// Token accounting, OpenAI's `usage` object.
#[derive(Debug, Clone, Copy, Default)]
pub struct GenerateUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
}

/// One piece of a generation, handed to the sink as it is produced.
#[derive(Debug)]
pub enum Chunk<'a> {
    /// Decoded text for the token(s) just produced. Borrowed, so a non-streaming caller that just
    /// appends to a String never allocates per token. May be empty: a multi-byte UTF-8 codepoint
    /// split across BPE tokens produces nothing until it completes.
    Text(&'a str),
    /// Terminal. Emitted exactly once, after the last `Text`.
    Done { reason: FinishReason, usage: GenerateUsage },
}

/// An autoregressive text model. ONE method serves both the streaming and the buffered surface --
/// a non-streaming caller is a streaming caller whose sink appends to a String. Two entry points
/// would be two code paths to keep in agreement, and the buffered one is the trivial case.
///
/// `&mut self`, not `&self`: a decoder owns a KV cache and a resident device context, and every
/// shipped model in this crate is already `&mut` in truth (see `capability::Servable`'s note on the
/// 2026-07-29 boundary inventory -- whisper/parakeet/gigaam launder mutability through `RefCell`).
/// Declaring it honestly here means an implementation never has to.
///
/// The sink returns `false` to abort (client gone). An implementation MUST stop promptly and MUST
/// still deliver `Chunk::Done` with `FinishReason::Aborted`, so accounting and cleanup have exactly
/// one path regardless of how generation ended.
pub trait TextGenerator {
    fn generate(
        &mut self,
        prompt: &Prompt,
        params: &GenerateParams,
        sink: &mut dyn FnMut(Chunk<'_>) -> bool,
    ) -> Result<(), EngineError>;

    /// Collect a whole generation into a String. Provided, not required: this is the buffered
    /// surface expressed in terms of the streaming one, which is the point of the single method.
    fn generate_to_string(
        &mut self,
        prompt: &Prompt,
        params: &GenerateParams,
    ) -> Result<(String, FinishReason, GenerateUsage), EngineError> {
        let mut out = String::new();
        let mut fin = FinishReason::Stop;
        let mut usage = GenerateUsage::default();
        self.generate(prompt, params, &mut |c| {
            match c {
                Chunk::Text(t) => out.push_str(t),
                Chunk::Done { reason, usage: u } => {
                    fin = reason;
                    usage = u;
                }
            }
            true
        })?;
        Ok((out, fin, usage))
    }
}


#[cfg(test)]
mod generation_tests {
    use super::*;

    fn ckpt() -> GenerationDefaults {
        // qwen3-0.6b's real generation_config.json, verbatim.
        GenerationDefaults { temperature: Some(0.6), top_p: Some(0.95), top_k: Some(20), ..Default::default() }
    }

    #[test]
    fn a_silent_request_gets_what_the_checkpoint_asks_for() {
        let g = GenerateParams::default().resolve(&GenerationDefaults::default(), &ckpt());
        assert_eq!(g.temperature, 0.6, "not the engine's 1.0 -- the model asked for 0.6");
        assert_eq!(g.top_p, 0.95);
        assert_eq!(g.top_k, 20);
        // Fields the checkpoint says nothing about still fall through to the engine.
        assert_eq!(g.max_tokens, ENGINE_DEFAULTS.max_tokens);
        assert_eq!(g.repetition_penalty, ENGINE_DEFAULTS.repetition_penalty);
    }

    #[test]
    fn with_no_checkpoint_the_engine_defaults_still_apply() {
        let g = GenerateParams::default()
            .resolve(&GenerationDefaults::default(), &GenerationDefaults::default());
        assert_eq!(g.temperature, ENGINE_DEFAULTS.temperature);
        assert_eq!(g.top_p, ENGINE_DEFAULTS.top_p);
        assert_eq!(g.top_k, ENGINE_DEFAULTS.top_k);
    }

    #[test]
    fn the_scenario_outranks_the_checkpoint_and_the_request_outranks_both() {
        let scenario = GenerationDefaults { temperature: Some(0.2), ..Default::default() };
        let g = GenerateParams::default().resolve(&scenario, &ckpt());
        assert_eq!(g.temperature, 0.2, "scenario beats checkpoint");
        assert_eq!(g.top_p, 0.95, "and does not disturb what it says nothing about");

        let req = GenerateParams { temperature: Some(1.5), ..GenerateParams::default() };
        let g = req.resolve(&scenario, &ckpt());
        assert_eq!(g.temperature, 1.5, "request beats both");
    }

    #[test]
    fn an_explicit_zero_is_a_value_not_an_absence() {
        // The whole reason these are Options: `temperature: 0` means greedy and MUST outrank the
        // checkpoint's 0.6. Under the old plain-f32 fields this was indistinguishable from unset.
        let req = GenerateParams { temperature: Some(0.0), ..GenerateParams::default() };
        let g = req.resolve(&GenerationDefaults::default(), &ckpt());
        assert_eq!(g.temperature, 0.0);
    }

    #[test]
    fn validate_accepts_the_endpoints_and_rejects_outside_them() {
        let p = |t: Option<f32>, tp: Option<f32>, pp: Option<f32>| GenerateParams {
            temperature: t, top_p: tp, presence_penalty: pp, ..GenerateParams::default()
        };
        assert!(p(Some(0.0), Some(0.0), Some(-2.0)).validate().is_ok());
        assert!(p(Some(2.0), Some(1.0), Some(2.0)).validate().is_ok());
        assert!(p(None, None, None).validate().is_ok(), "unset is never an error");
        assert!(p(Some(-0.1), None, None).validate().unwrap_err().contains("temperature"));
        assert!(p(Some(2.1), None, None).validate().unwrap_err().contains("temperature"));
        assert!(p(None, Some(1.1), None).validate().unwrap_err().contains("top_p"));
        assert!(p(None, None, Some(-2.1)).validate().unwrap_err().contains("presence_penalty"));
        assert!(p(Some(f32::NAN), None, None).validate().unwrap_err().contains("finite"));
        assert!(p(Some(f32::INFINITY), None, None).validate().unwrap_err().contains("finite"));
    }
}
