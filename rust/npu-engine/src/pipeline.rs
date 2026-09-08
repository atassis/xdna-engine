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
    pub temperature: f32,
    pub top_p: f32,
    /// 0 = disabled. Not in OpenAI's schema but universal in local servers, and the device-side
    /// top-k slice (`llm-onchip-topk-feedback`) makes it the cheapest of the three to honour.
    pub top_k: u32,
    /// `None` = the caller did not ask, so the model's configured default applies (and failing
    /// that, [`DEFAULT_MAX_TOKENS`]). It is an Option so that "unset" and "explicitly 256" stay
    /// distinguishable all the way to the generator -- without that a per-model default cannot
    /// exist, because by the time the request arrives every field already looks specified.
    pub max_tokens: Option<u32>,
    pub stop: Vec<String>,
    pub seed: Option<u64>,
    /// `chat_template_kwargs.enable_thinking`, the one template kwarg the reasoning-model families
    /// (Qwen3, and the same spelling in vLLM/SGLang) read. `None` leaves the template's own
    /// default, which for Qwen3 is thinking ON -- and at the 256-token default that spends the
    /// whole budget reasoning and never emits an answer. Only meaningful for `Prompt::Chat`.
    pub enable_thinking: Option<bool>,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub repetition_penalty: f32,
}

impl Default for GenerateParams {
    fn default() -> Self {
        GenerateParams {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            max_tokens: None,
            stop: Vec::new(),
            seed: None,
            enable_thinking: None,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repetition_penalty: 1.0,
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

/// The completion budget when neither the request nor the model's config names one. OpenAI's
/// number, kept as the final fallback rather than as the only answer.
pub const DEFAULT_MAX_TOKENS: u32 = 256;

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

