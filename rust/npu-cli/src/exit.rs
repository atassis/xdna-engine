//! Closed set of process exit codes `npu` returns, additive over the historical "everything is
//! 1": scripts, CI, and the planned ffmpeg-filter coprocess need to tell "service is down" from
//! "no such model" from "device/engine error" from "user cancelled" without parsing stderr text.
//!
//! Classification happens in `main.rs`, at each call site that still has a typed cause in hand
//! (an `EngineError` variant, a connection failure) -- before it gets flattened to a display
//! string. `main` reads the result back exactly once, via [`of`], which is the single place any
//! error becomes a process exit code.

use npu_engine::EngineError;

/// One of the six exit codes `npu` can return. Anything not confidently matched to 2-5 stays at
/// the `Failure` default -- deliberate, not a gap. See `main.rs`'s call sites for what is and is
/// not classified, and why.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Code {
    /// Success.
    Success = 0,
    /// Generic / unclassified failure. The default bucket: every error not confidently matched
    /// to one of the codes below lands here.
    Failure = 1,
    /// No service running, or the CLI could not reach it. `npu reload`/`load`/`unload` talk to a
    /// running server over HTTP; a connection failure there means: start one (`npu serve`).
    NoService = 2,
    /// No such model, or no model configured for the requested capability.
    NoModel = 3,
    /// A device or engine error (no NPU present, an actor/device fault).
    Device = 4,
    /// Cancelled by the user. No call site produces this yet: nothing in this CLI catches
    /// SIGINT or prompts for confirmation today. Declared now so a future one does not need a
    /// renumbering -- see main.rs's report on why this is expected.
    #[allow(dead_code)]
    Cancelled = 5,
}

/// Tags an error with the exit code its caller should surface, without changing anything a user
/// sees: `Display` shows only the wrapped message, exactly like the `&str`/`String` context it
/// replaces at every call site. [`of`] downcasts for it; nothing else needs to know it exists.
#[derive(Debug)]
pub struct Tagged(pub Code, pub String);

impl std::fmt::Display for Tagged {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str(&self.1) }
}
impl std::error::Error for Tagged {}

/// Which exit code an `EngineError` variant should surface as, read at a call site that still has
/// the typed error in hand. `WrongKind`/`Unsupported`/`Load` stay the generic default: `Load`
/// alone covers both "name not in the config" (would want `NoModel`) and genuine load/build
/// failures, indistinguishable by variant -- splitting them would mean matching message text
/// instead of the type, which this task deliberately does not do.
pub fn engine_error(e: &EngineError) -> Code {
    match e {
        EngineError::NotAvailable | EngineError::Device(_) => Code::Device,
        EngineError::NoModel(_) => Code::NoModel,
        EngineError::WrongKind { .. } | EngineError::Unsupported(_) | EngineError::Load(_) =>
            Code::Failure,
    }
}

/// The single conversion point: what `main` returns for a failed command. Reads the [`Tagged`]
/// left on the error chain -- attached directly, or via `.context(Tagged(..))`, which anyhow
/// recurses through (see the test below) -- and defaults untagged errors to `Failure`.
pub fn of(e: &anyhow::Error) -> Code {
    e.downcast_ref::<Tagged>().map(|t| t.0).unwrap_or(Code::Failure)
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Context;
    use npu_engine::capability::Capability;

    #[test]
    fn engine_error_classifies_device_and_no_model() {
        assert_eq!(engine_error(&EngineError::NotAvailable), Code::Device);
        assert_eq!(engine_error(&EngineError::Device("x".into())), Code::Device);
        assert_eq!(engine_error(&EngineError::NoModel(Capability::ASR)), Code::NoModel);
    }

    /// The conservative default: variants that could plausibly be one bucket or another stay
    /// generic rather than guessing from message text.
    #[test]
    fn engine_error_leaves_ambiguous_variants_generic() {
        assert_eq!(engine_error(&EngineError::Load("x".into())), Code::Failure);
        assert_eq!(engine_error(&EngineError::Unsupported("x".into())), Code::Failure);
        assert_eq!(engine_error(&EngineError::WrongKind {
            wanted: Capability::ASR, got: Capability::EMBED }), Code::Failure);
    }

    #[test]
    fn of_reads_a_directly_tagged_error() {
        let e: anyhow::Error = Tagged(Code::NoService, "down".into()).into();
        assert_eq!(of(&e), Code::NoService);
        assert_eq!(e.to_string(), "down", "the tag must not appear in the visible message");
    }

    /// `.context(Tagged(..))` must still classify, and must keep chaining the ORIGINAL cause --
    /// this is what the `reload`/`load`/`unload` call sites rely on (human text and the tag from
    /// one `.context()` call, the connection failure still shown as the cause).
    #[test]
    fn of_reads_a_tag_added_via_context_and_keeps_the_chain() {
        let root: anyhow::Error = anyhow::anyhow!("connection refused");
        let e = Err::<(), anyhow::Error>(root)
            .context(Tagged(Code::NoService, "reload (is the server running?)".into()))
            .unwrap_err();
        assert_eq!(of(&e), Code::NoService);
        assert_eq!(e.to_string(), "reload (is the server running?)");
        assert!(format!("{e:?}").contains("connection refused"), "must still show the cause: {e:?}");
    }

    #[test]
    fn of_defaults_to_failure_for_an_untagged_error() {
        assert_eq!(of(&anyhow::anyhow!("boom")), Code::Failure);
    }
}
