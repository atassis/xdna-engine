//! What crosses from the actor thread (where `TextGenerator::generate` runs) to the socket thread
//! (which owns the client connection). Kept format-agnostic on purpose: `http.rs` renders these into
//! SSE/JSON, a CLI can just print the text, and neither has to know about the other.
use npu_engine::{FinishReason, GenerateUsage};

pub enum StreamItem {
    /// Decoded text for the token(s) just produced. May be empty (a multi-byte codepoint split
    /// across tokens produces nothing until it completes) -- forwarded as-is, not filtered.
    Text(String),
    /// Terminal on a clean end (stop condition, `max_tokens`, or an aborted sink). Exactly one,
    /// always last.
    Done { reason: FinishReason, usage: GenerateUsage },
    /// A real failure mid-generation -- not a client hangup, which just drops the receiver and lets
    /// the next `send` fail instead.
    Error(String),
}
