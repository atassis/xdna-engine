//! What crosses from the actor thread (where `TextGenerator::generate` runs) to the socket thread
//! (which owns the client connection). Kept format-agnostic on purpose: `http.rs` renders these into
//! SSE/JSON, a CLI can just print the text, and neither has to know about the other.
use npu_engine::{EngineError, FinishReason, GenerateUsage};

pub enum StreamItem {
    /// Decoded text for the token(s) just produced. May be empty (a multi-byte codepoint split
    /// across tokens produces nothing until it completes) -- forwarded as-is, not filtered.
    Text(String),
    /// Terminal on a clean end (stop condition, `max_tokens`, or an aborted sink). Exactly one,
    /// always last.
    Done { reason: FinishReason, usage: GenerateUsage },
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
