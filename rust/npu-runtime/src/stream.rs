//! What crosses from the actor thread (where `TextGenerator::generate` runs) to the socket thread
//! (which owns the client connection). Kept format-agnostic on purpose: `http.rs` renders these into
//! SSE/JSON, a CLI can just print the text, and neither has to know about the other.
use npu_engine::{EngineError, FinishReason, GenerateUsage, GenerationReport, StepRecord};

pub enum StreamItem {
    /// Decoded text for the token(s) just produced. May be empty (a multi-byte codepoint split
    /// across tokens produces nothing until it completes) -- forwarded as-is, not filtered.
    Text(String),
    /// What the token just produced cost. One per DECODED TOKEN, which is not one per `Text`: a
    /// token completing no codepoint emits no text, and a stop-sequence flush emits text with no
    /// token. A consumer counting tokens must count these, and a consumer that does not want them
    /// drops them -- they are produced either way, because an instrument switched on per request
    /// is not armed for the request that turns out to matter.
    Step(StepRecord),
    /// Terminal on a clean end (stop condition, `max_tokens`, or an aborted sink). Exactly one,
    /// always last.
    ///
    /// `usage` stays its own field: it is the OpenAI contract and every consumer needs it, while
    /// the report is for the ones that want the timeline. Boxed because it carries one record per
    /// token and this enum is moved through a channel per chunk.
    Done { reason: FinishReason, usage: GenerateUsage, report: Box<GenerationReport> },
    /// A real failure mid-generation -- not a client hangup, which just drops the receiver and lets
    /// the next `send` fail instead.
    ///
    /// Carries the `EngineError` rather than its rendered text so a consumer can classify it. It
    /// used to be a `String`, which made every in-generation failure a 500 -- including the ones
    /// that are the caller's doing, like a prompt too long for the model's context window. That is
    /// not an HTTP dependency creeping into this module: `EngineError` is already the vocabulary
    /// this crate speaks, and each surface still decides its own rendering.
    Error(EngineError),
}
