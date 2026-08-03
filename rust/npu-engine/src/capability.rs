//! The open request surface (`engine-open-capability-contract`, task family `substrate`).
//!
//! PROBE, not the finished contract: derived from adapting exactly two real instances --
//! `bert::EmbedPipeline` (text in, vector out, genuinely `&self`, already inside the engine) and
//! `npu_sr::SrEngine` (image in, image out, genuinely `&mut self`, currently a full island: its own
//! `SrError`, its own `load`/`upscale_*` ABI, zero wiring into `npu-runtime`'s `Inference`/
//! `ModelLoader`/`Cmd`/`resolve()`). Per the doctrine's "earn generality from instances" rule this
//! module does NOT wire into `npu-runtime` yet and does NOT enumerate capabilities beyond the two
//! that are earned -- that is next work, once a third instance actually asks for it.
//!
//! Named `Servable`, not `Model`: the task's own `next` field proposed `trait Model { .. }` as a
//! candidate name, but `npu_engine::api::Model` (the loaded-scenario struct every caller already
//! uses) occupies that name in this crate's type namespace. A trait and a struct of the same name
//! in the same crate collide the moment anything does `use npu_engine::{Model, capability::Model}`
//! (or a glob import of both) -- a real compile error, not a style nit, found by trying the
//! candidate name during this probe.

use crate::api::EngineError;

/// A capability name, e.g. `"embed"`, `"image-sr"`. Deliberately a string, not an enum variant --
/// that is the exact thing this task exists to open. `ModelKind { Asr, Embed }` is a CLOSED enum;
/// every new capability (here: image super-resolution) means editing the enum plus every match on
/// it (`registry.rs`, `select.rs`, `actor.rs`'s `Cmd`, ...). A string costs a new value, never an
/// edit to an existing one. `Ord`/`Hash` are free now (derive) and are exactly what the parent
/// task's eventual `defaults: BTreeMap<Capability, String>` needs -- not built here, just not
/// blocked by this type when it is.
///
/// `npu-runtime` routes on this type; the `Capability` alias for `ModelKind` that used to live in
/// `npu_runtime::registry` is gone, so only one type by that name remains.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Capability(pub &'static str);

impl std::fmt::Display for Capability {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str(self.0) }
}

impl Capability {
    pub const ASR: Capability = Capability("asr");
    pub const EMBED: Capability = Capability("embed");
    pub const GENERATE: Capability = Capability("generate");
    pub const TTS: Capability = Capability("tts");
    pub const IMAGE_SR: Capability = Capability("image-sr");

    /// Every capability this binary can serve. `&'static str` is kept rather than widened to an
    /// owned string precisely because of this list: a capability name that no compiled-in model
    /// implements cannot be served, so an arbitrary runtime string would only ever name something
    /// that fails. Adding a capability is one const plus one entry HERE -- one file, versus the five
    /// (`api`, `pipeline`, `loader`, `actor`, `select`) that a `ModelKind` variant used to cost.
    pub const ALL: &'static [Capability] =
        &[Self::ASR, Self::EMBED, Self::GENERATE, Self::TTS, Self::IMAGE_SR];

    /// Resolve a capability written in config -- an `/admin/defaults` body, a `?capability=` -- against
    /// `ALL`. `None` for a name nothing implements.
    pub fn from_name(s: &str) -> Option<Capability> {
        Self::ALL.iter().find(|c| c.0 == s).copied()
    }

    /// A scenario's `[scenario] kind`, which is NOT the same vocabulary as a capability name: every
    /// shipped embedding scenario says `kind = "embeddings"` while the capability is `embed`. Kept as
    /// an alias rather than renamed on either side -- the scenario files are user data.
    pub fn from_scenario_kind(s: &str) -> Option<Capability> {
        match s {
            "embeddings" => Some(Self::EMBED),
            other => Self::from_name(other),
        }
    }
}

/// The request payload. An enum over data SHAPES, not a generic-per-capability associated type --
/// deliberate, not a default: `Box<dyn Servable>` (the whole point of collapsing `Scenario`'s two
/// variants into one trait) requires the trait to be object-safe, and an associated `type Input`
/// that varies per impl breaks that immediately.
///
/// Variants are earned per instance, never added speculatively. `Text`/`Image` came from
/// `EmbedPipeline` and `SrEngine`; `Audio` from wiring the shipped ASR models onto this trait --
/// `Text` cannot carry PCM. Note `Text` serves three capabilities (embed, generate, tts): the enum
/// discriminates SHAPES, not capabilities, so a new capability that reuses a shape costs nothing.
#[derive(Debug)]
pub enum Request {
    /// `EmbedPipeline`/`EsmEmbedPipeline`'s shape: one string in.
    Text(String),
    /// `Model::transcribe`'s shape: mono PCM plus the rate it was sampled at. The rate travels with
    /// the samples rather than being assumed 16 kHz -- the models resample, and a silently wrong
    /// rate is a wrong transcript, not an error.
    Audio { pcm: Vec<i16>, sample_rate: u32 },
    /// `SrEngine::upscale_rgb8`'s shape: interleaved RGB8 + explicit dims (the real per-frame ABI
    /// the C ABI / ffmpeg filter already calls -- `npu-sr-capi/src/lib.rs:90`, not the lower-level
    /// `upscale_plane`/`upscale_planar_rgb` used only by the parity gates).
    Image { rgb: Vec<u8>, w: usize, h: usize },
}

impl Request {
    /// The variant name, for error messages. `Debug` is not usable there: `Audio` holds the whole
    /// PCM buffer and would render a mismatch as megabytes of samples.
    pub fn shape(&self) -> &'static str {
        match self {
            Request::Text(_) => "text",
            Request::Audio { .. } => "audio",
            Request::Image { .. } => "image",
        }
    }
}

/// The response payload. Same rule as `Request`: shapes, earned per instance.
#[derive(Debug)]
pub enum Response {
    /// `EmbedPipeline::embed`'s shape: one embedding vector out.
    Vector(Vec<f32>),
    /// A transcript or a generated completion -- ASR and generate differ in what produces the text,
    /// not in its shape.
    Text(String),
    /// Synthesised speech. Carries its own rate for the same reason `Request::Audio` does, and
    /// because TTS output is not the 16 kHz the ASR side works in.
    Audio { pcm: Vec<i16>, sample_rate: u32 },
    /// `SrEngine::upscale_rgb8`'s shape: interleaved RGB8 + the OUTPUT dims (upscaled, so these
    /// differ from the request's `w`/`h` by the schedule's integer scale factor).
    Image { rgb: Vec<u8>, w: usize, h: usize },
}

impl Response {
    /// The variant name, for error messages. Same reason as `Request::shape`.
    pub fn shape(&self) -> &'static str {
        match self {
            Response::Vector(_) => "vector",
            Response::Text(_) => "text",
            Response::Audio { .. } => "audio",
            Response::Image { .. } => "image",
        }
    }
}

/// The open, object-safe request-surface trait. `Box<dyn Servable>` is meant to be able to replace
/// `pipeline::Scenario`'s closed two-variant enum and `npu-runtime`'s `loader::Inference` /
/// `actor::Cmd` (five edit sites total, per this task's own boundary inventory) -- NOT done in this
/// pass; this module proves the trait shape fits two real, structurally different models first.
///
/// `&mut self` on `run`, not `&self`: settled by the 2026-07-29 boundary inventory across all 7
/// shipped model implementations -- every one is already `&mut` in truth (whisper/parakeet/gigaam
/// launder a KV/context cache through `RefCell` so their outer method can stay `&self`; `SrEngine`
/// is the one model that says so honestly today). Declaring `&mut self` here deletes that laundering
/// requirement for any future adapter instead of adding a new one.
pub trait Servable {
    /// What this instance serves. Singular, not a set: neither earned instance below declares more
    /// than one capability. A multi-capability model is a foreseeable extension this trait ADMITS
    /// (nothing here rules it out) but does not build -- YAGNI, per doctrine item 4.
    fn capabilities(&self) -> Capability;

    /// Best-effort device-memory footprint in bytes, 0 if unknown. Same shape and same honest "0
    /// until measured" convention as `npu_runtime::loader::Inference::bo_bytes` (`loader.rs:34`) --
    /// not yet measured per-model there either (backlog R11(f)); this trait does not invent a
    /// stricter contract than the one caller that already has this method actually keeps.
    fn footprint(&self) -> u64 { 0 }

    /// Serve one request. `Err` for a capability mismatch (wrong `Request` variant for this model,
    /// e.g. `Request::Image` sent to a text embedder) as well as for a genuine runtime failure --
    /// both are `EngineError`, matching how `Model::transcribe`/`Model::embed` already report both
    /// classes of failure through the same error type (`api.rs`).
    fn run(&mut self, req: Request) -> Result<Response, EngineError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The trait must stay object-safe: `Box<dyn Servable>` is the entire point (it is what would
    /// let a registry hold heterogeneous models the way `Box<dyn AsrModel>`/`Box<dyn Embedder>` do
    /// today). A signature change that breaks object-safety fails to compile right here, not later
    /// at some call site far from this module.
    #[allow(dead_code)]
    fn _assert_object_safe(_: &dyn Servable) {}

    #[test]
    fn capability_is_a_cheap_orderable_key() {
        // Exercises the derive this module leans on for the eventual BTreeMap<Capability, String>
        // (not built here -- see the parent task's target shape) without building that map yet.
        let mut v = vec![Capability("embed"), Capability("image-sr")];
        v.sort();
        assert_eq!(v, vec![Capability("embed"), Capability("image-sr")]);
        assert_ne!(Capability("embed"), Capability("image-sr"));
    }
}
