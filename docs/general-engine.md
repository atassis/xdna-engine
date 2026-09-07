# What makes this a general engine

This is about the software boundary, not the hardware one: which code is shared across every
model this engine serves, and which code is written once per model. The data-movement
argument for why the NPU can be fast at all is a separate question, already made in
[data-movement-thesis.md](data-movement-thesis.md) and applied per-op in
[execution-graph.md](execution-graph.md); this note does not repeat it. Read
[../ARCHITECTURE.md](../ARCHITECTURE.md) alongside this for the crate layout and its own "Known
seams" section, which lists several of the same gaps from the crate-layering angle.

The claim is not "one binary runs any model with zero new code." It is narrower and, I think,
more honest: there is one interface a model's NPU stage implements, one config format that
describes a model declaratively, one weight-baking path, one serving substrate (control plane
+ HTTP), and those four things are genuinely shared today across ASR, embeddings, and (partially)
generation. What is not shared is named explicitly below, because pretending otherwise would
make the next model's estimate wrong.

## The shared contract: `Frontend` / `Encoder` / `Head`

`rust/npu-engine/src/pipeline.rs` defines the traits every model sits behind:

```rust
pub trait Encoder {
    fn forward_last(&self, x: &Array2<f32>, valid_len: usize) -> Array2<f32>;
}
```

`Encoder` is deliberately the only trait with a comment calling it "the genuinely-shared,
genuinely-hard NPU stage." Its contract is a plain shape: `[M, D_in]` bf16-valued activations
in, `[M, D]` out. It says nothing about attention type, position encoding, layer count, or
kernel precision -- a Conformer block, a Transformer-XL block, and a BERT block all satisfy it.
Five concrete encoders implement it today:

- `ConformerEncoder` (GigaAM-v3), `rust/npu-engine/src/asr/mod.rs`
- `FastConformerEncoder` (Parakeet), `rust/npu-engine/src/asr/parakeet.rs`
- `BertEncoder`, `rust/npu-engine/src/bert/encoder.rs`
- `EsmEncoder` and `EsmEncoderNative`, `rust/npu-engine/src/esm/encoder.rs`

Everything upstream and downstream of that one call is per-domain:

- `Frontend<Input>` turns raw input into `(Array2<f32>, valid_len)` -- log-mel framing for
  audio, tokenization for text.
- `Head<Output>` turns encoder output into a result -- mean/CLS pooling for embeddings,
  RNNT-style decoding for ASR.
- `Embedder`, `Diarizer`, `TextGenerator` are additional per-capability traits for the models
  that do not fit the encoder-then-head shape.
- `Scenario` is the enum (`Asr | Embed | Diarize | Generate`) that lets the registry hold any
  of them behind one object and the HTTP/CLI layer dispatch without knowing which model it got.

`AsrModel` is its own trait, not `Encoder` + a generic `Head`, because RNNT decoding
(`AsrPipeline::decode` in `asr/mod.rs`, and the equivalent in `parakeet.rs`/`whisper.rs`) is a
stateful loop over a decoder network and a joint network -- it does not reduce to a stateless
`Head::run`. So ASR's shared piece is the encoder stage only; decode is per-model by necessity,
not by omission.

## What is actually reused

**The `Encoder` trait boundary.** Any model whose forward pass is layer-normalize -> attend ->
project -> layer-normalize -> feed-forward, batched over a sequence, can be dropped in behind
this trait and the rest of the pipeline (residency handling, HTTP routing, config loading,
CLI) does not change. This is what let BERT, ESM-2, GigaAM, and Parakeet ship as the same kind
of thing to the registry despite four different attention/position schemes.

**The scenario config format.** `rust/npu-engine/src/config.rs`'s `ScenarioConfig` is one TOML
shape -- `[scenario]` (kind, name), an optional `[model]` block (hidden/ff/heads/layers/
precision/kernel -- optional because non-transformer models like the diarization PyanNet/
ResNet34 graphs have none of these fields), `[artifacts]`, and a per-kind block
(`[embeddings]`, `[diarization]`). Every shipped model is data against this one schema; see
`scenarios/*.toml`. Adding a model with an already-supported `kind` is a new TOML file, not new
Rust, provided its encoder already exists.

**Weight baking.** `npu-weights` bakes Hugging Face safetensors or ONNX into a content-addressed
mmap bf16 checkpoint through one declarative path: a scenario names `artifacts.source` (`hf:
<repo>[@rev]` or `path:/abs`) and `artifacts.arch`, and `Artifacts::model_spec` resolves it.
The `arch` transform is the one per-model piece here -- `bert|esm|vit|opt|whisper|
fastconformer|gigaam` are the values in use -- but the bake-on-miss, fingerprinting, and
parity-check machinery around it is one implementation.

**Dispatch rails.** `npu-dispatch` is explicitly "byte-marshaling helpers and dispatch
profiling with zero model-specific semantics" -- its own doc comment. It used to be three
byte-identical copies inside `npu-asr`, `npu-parakeet`, and `npu-whisper`; it is now the one
definition every encoder crate depends on for padding constants and NPU-time profiling.

**The serving substrate.** `npu-runtime`'s device actor (`rust/npu-runtime/src/actor.rs`) is a
single thread owning one `Registry` of loaded models, reachable through a cloneable `Handle`.
It knows nothing about ASR, embeddings, or LLMs -- it knows `Scenario` variants, capability
routing, residency (`max_resident`, LRU eviction, idle-unload, idle-release), and reconciling a
`Config` against what is currently loaded. The HTTP server (`rust/npu-runtime/src/http.rs`) and
the `npu` CLI are both thin clients of this one `Handle`; adding a model that reuses an
existing `Encoder`/`Frontend`/`Head` set costs zero new lines in either. The NPU is
single-tenant, so this actor is also the thing that keeps every model's device access
serialized -- one owner, one thread, no lock, regardless of which model is running.

**The resident-dataflow design principles.** The techniques described in the data-movement
thesis and the execution graph -- resident weights/activations, fused dispatch, precision
laddering -- are argued once, generically, in terms of "the M dimension sets the regime." They
apply to any model's transformer or conv block by construction, because the argument never
refers to a specific model. What is *built* per model is still per model (below); what is
*true* about how to build it is shared.

## What does not generalize (named explicitly, not glossed over)

**ASR decode is per-model, not merely per-capability.** GigaAM and Parakeet both decode via an
RNNT predictor + joint network (`asr/mod.rs`, `asr/parakeet.rs`); Whisper decodes via a
transformer decoder fused into a single resident ELF dispatch (`asr/whisper.rs`,
`asr/whisper_decoder.rs`). These are three different decode algorithms behind the same
`AsrModel` trait, not three configurations of one decoder.

**Whisper's encoder does not go through the shared trait.** `WhisperAsr` holds a
`npu_whisper::encoder::WhisperEncoder` and calls its inherent `forward_last` method directly
(`asr/whisper.rs`); that type does not implement `npu_engine::pipeline::Encoder`. It has the
same shape by convention, not by the compiler enforcing it -- Whisper's model code is also
split across two crates (`npu-whisper` for the reference/decoder, `npu-engine::asr` for the
NPU wiring), where GigaAM and Parakeet each keep their non-generic parts in one dedicated
crate (`npu-asr`, `npu-parakeet`). `ARCHITECTURE.md`'s "Known seams" section names this same
inconsistency from the crate-layout side.

**Diarization is host-only today.** `rust/npu-engine/src/diarize/mod.rs`'s own doc comment:
"Host-only in v1; the NPU embedder swaps in behind `SpeakerEmbedder` without touching this
file." The `Diarizer` trait and the `DiarizePipeline` shape (segmenter -> embedder -> clusterer,
each swappable) generalize across the two shipped manifests (pyannote 3.1 and
community-1); none of the three stages currently dispatches to the NPU. `registry::try_build`
does not even open the device for a diarize scenario, on purpose, so it cannot take a hardware
context away from a co-resident ASR model.

**The capability set is closed, not open.** `ModelKind { Asr, Embed, Diarize, Generate }`
(`api.rs`) is threaded through five places (`api`, `pipeline`, `loader`, `actor`, `select`);
adding a fifth capability means editing all five. `rust/npu-engine/src/capability.rs`'s
`Capability`/`Servable`/`Request`/`Response` types are an open replacement for exactly this
-- its own doc comment calls it a "PROBE, not the finished contract," validated against two
instances (`bert::EmbedPipeline` and `npu_sr::SrEngine`) and not yet wired into the registry
the HTTP server and CLI actually use.

**Video super-resolution is a second, separate pipeline.** `npu-sr` does not implement
`Encoder`/`Frontend`/`Head` at all. It has its own frame-in/frame-out ABI, its own error type,
and drives `npu-xrt` directly rather than going through `npu-engine` or `npu-runtime`. It
shares the device and the weight-loading approach, not the request path -- it is not reachable
through the HTTP server or the `Model`/`Scenario` types described here or in
[api.md](api.md).

**Small-LLM decode reuses primitives, not code, across models.** Qwen3 is served today through
`npu_engine::llm` (`NpuDecodeStep` driving a fused-decode ELF, wired into the registry's
`Generate` arm). Gemma 3's bring-up lives in the separate `npu-gemma` crate, which
`ARCHITECTURE.md` marks a "SCAFFOLD: reference math, a `Brick` decode schedule and sampling,
but no serving path -- nothing depends on it and it depends on no other crate here." Both reuse
the same conceptual primitives (resident FFN, fused decode, a KV cache) named in the top-level
README, but that reuse is currently at the level of technique, not shared Rust: Gemma is not
reachable through a scenario TOML or the registry the way GigaAM, Parakeet, BERT, and ESM-2
are.

**Kernel-binary selection is a per-model hardcoded path, not a lookup.** A model crate names
its xclbin as a string literal encoding shape, tile, column count, and precision variant in the
filename (`ARCHITECTURE.md`'s example: `final_512x1024x4096_64x32x128_8c_modalsilu.xclbin`).
There is no machine-readable statement of which operations a model needs that a dynamic
binary-selection layer could consume; which precision tiers exist for a given shape is decided
by which kernel someone has already built and WER-gated for that model, not derived
automatically from a policy.

## The honest summary

The generalization that is real: one interface (`Encoder`) for the part of every model that is
genuinely hard and genuinely shared, one declarative config format, one weight-bake path, and
one serving substrate that does not know or care which model it is running. The parts that do
not generalize are the parts that cannot, by the nature of the model (RNNT vs. transformer
decode, fused-ELF LLM decode vs. host-orchestrated ASR decode), plus a few that could but have
not been done yet (Whisper's encoder wiring, the closed capability enum, per-model kernel
naming). Both categories are worth keeping distinct: the first will still be true after any
amount of engineering; the second is a to-do list.
