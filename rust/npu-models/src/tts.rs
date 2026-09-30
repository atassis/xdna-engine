//! `kind = "tts"`: speech synthesis. COMPOSES pieces other lanes own -- a prompt template, a
//! Slow AR decoder, a Fast AR decoder, a neural codec -- rather than any of them, so it builds
//! from any `tts` scenario and reports exactly what is missing at synthesis time instead of
//! failing to load, or worse, emitting a stub tone under a 200.

use std::path::Path;

use crate::api::EngineError;
use crate::cancel::Cancel;
use crate::capability::{Capability, Request, Response};
use crate::config::ScenarioConfig;
use crate::pipeline::TtsModel;

pub struct TtsPipeline {
    name: String,
}

impl TtsPipeline {
    /// Host-only: no device is opened and no artifact is read, because nothing here composes yet
    /// (see `synthesize`) -- a `tts` scenario must not take a hardware context away from a model
    /// that coexists with it just to register a capability it cannot serve.
    pub fn build(cfg: &ScenarioConfig, _root: &Path) -> Result<TtsPipeline, EngineError> {
        Ok(TtsPipeline { name: cfg.scenario.name.clone() })
    }
}

impl TtsModel for TtsPipeline {
    fn synthesize(&mut self, _text: &str, cancel: &Cancel) -> Result<(Vec<i16>, u32), EngineError> {
        if cancel.is_cancelled() {
            return Err(EngineError::Unsupported(format!(
                "{}: synthesis cancelled before it started", self.name)));
        }
        Err(EngineError::Unsupported(format!(
            "{}: text-to-speech is not implemented -- no prompt template, Slow AR, Fast AR or \
             codec is wired in yet", self.name)))
    }
}

/// The open-contract instance: `Request::Text(String) -> Response::Audio`. Not on the wired
/// dispatch path (`Model::synthesize` via `pipeline::TtsModel` is), but a real third instance for
/// `capability::Servable`'s two-instance probe -- see that module's doc comment. Uses an
/// unpublished `Cancel`, so a call through this trait cannot be interrupted; `Model::synthesize`
/// is the entry point that carries a caller's own token.
impl crate::capability::Servable for TtsPipeline {
    fn capabilities(&self) -> Capability { Capability::TTS }
    fn run(&mut self, req: Request) -> Result<Response, EngineError> {
        match req {
            Request::Text(text) => {
                let (pcm, sample_rate) = self.synthesize(&text, &Cancel::new())?;
                Ok(Response::Audio { pcm, sample_rate })
            }
            other => Err(EngineError::Unsupported(format!("tts cannot serve a {} request", other.shape()))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> ScenarioConfig {
        ScenarioConfig::from_str(
            "[scenario]\nkind = \"tts\"\nname = \"kokoro\"\n[artifacts]\n"
        ).unwrap()
    }

    #[test]
    fn build_never_fails_and_never_touches_a_device() {
        // No device, no artifacts on disk at all -- `Model::available()`-style device checks
        // would fail here on a CI box with no NPU, and build() must not care.
        let p = TtsPipeline::build(&cfg(), Path::new("/nonexistent/root"));
        assert!(p.is_ok(), "a tts scenario must register its capability even with nothing composed yet");
    }

    #[test]
    fn synthesize_is_an_honest_unsupported_not_a_stub_tone() {
        let mut p = TtsPipeline::build(&cfg(), Path::new("/nonexistent/root")).unwrap();
        let err = p.synthesize("hello", &Cancel::new()).unwrap_err();
        match err {
            EngineError::Unsupported(msg) => {
                assert!(msg.contains("kokoro"), "{msg}");
                assert!(msg.contains("Slow AR") || msg.contains("codec"), "{msg}");
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    #[test]
    fn a_cancelled_token_is_refused_before_any_work() {
        let mut p = TtsPipeline::build(&cfg(), Path::new("/nonexistent/root")).unwrap();
        let c = Cancel::new();
        c.cancel(crate::cancel::CancelReason::Operator);
        let err = p.synthesize("hello", &c).unwrap_err();
        assert!(matches!(err, EngineError::Unsupported(_)));
    }

    #[test]
    fn servable_shape_answers_text_with_audio_shaped_unsupported() {
        let mut p = TtsPipeline::build(&cfg(), Path::new("/nonexistent/root")).unwrap();
        assert_eq!(crate::capability::Servable::capabilities(&p), Capability::TTS);
        let err = crate::capability::Servable::run(&mut p, Request::Text("hi".into())).unwrap_err();
        assert!(matches!(err, EngineError::Unsupported(_)));
    }
}
