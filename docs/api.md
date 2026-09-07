# API surface

Two surfaces exist: a Rust/C library you embed in-process, and an HTTP server you run
alongside your own client. Both sit on the same underlying pipeline described in
[general-engine.md](general-engine.md). This document only describes what the code implements
-- where a route or field's behavior was unclear from the source, it says so rather than
guessing.

## Rust: `npu-engine`

`rust/npu-engine/src/lib.rs` states its own contract: "Public API: `Engine`, `Model`,
`ModelKind`, `EngineError`. ... Everything else in this crate is implementation detail
(`#[doc(hidden)]`) and may change without notice." That is the surface documented here.

```rust
use npu_engine::{Engine, Model, ModelKind, EngineError};
```

- `Engine::available() -> bool` -- true if `/dev/accel/accel0` exists. A cheap file check, not
  an open.
- `Model::load(scenario: impl AsRef<Path>) -> Result<Model, EngineError>` -- load a model from
  a scenario TOML, resolving artifact paths against the current working directory.
- `Model::load_in(scenario, root) -> Result<Model, EngineError>` -- same, with an explicit repo
  root (where `artifacts/` lives).
- `Model::kind(&self) -> ModelKind` -- one of `Asr | Embed | Diarize | Generate`.
- `Model::embed_dim(&self) -> Option<usize>` -- the embedding width for an `Embed` model
  (its scenario's configured `hidden`); `None` otherwise.
- `Model::transcribe(&self, pcm: &[i16], sample_rate: u32) -> Result<String, EngineError>` --
  16 kHz mono PCM in, text out. Any other `sample_rate` is `Err(Unsupported)`.
- `Model::embed(&self, text: &str) -> Result<Vec<f32>, EngineError>`
- `Model::embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EngineError>` -- calls
  `embed` per element; not a batched device dispatch.
- `Model::diarize(&self, pcm: &[i16], sample_rate: u32) -> Result<Vec<Segment>, EngineError>`
  -- speaker-attributed spans (`Segment { start_s, end_s, speaker }`, `speaker` a cluster
  index, not a label).
- `Model::generate(&mut self, prompt: &Prompt, params: &GenerateParams, sink: &mut dyn FnMut(Chunk<'_>) -> bool) -> Result<(), EngineError>`
  -- streaming text generation; `&mut self` because a decoder owns a KV cache and a device
  context. `sink` returning `false` aborts generation.
- `TextGenerator::generate_to_string` -- the buffered form, built on `generate`: collects a
  full completion into one `(String, FinishReason, GenerateUsage)`.

Calling a method against the wrong model kind returns `Err(EngineError::WrongKind { wanted,
got })` rather than panicking. `EngineError` variants: `NotAvailable`, `Load(String)`,
`WrongKind { wanted, got }`, `NoModel(Capability)`, `Unsupported(String)`, `Device(String)`.

`Prompt` is `Chat(Vec<ChatMessage>)` or `Raw(String)` -- the former goes through the model's
chat template before tokenizing, the latter does not. `GenerateParams` defaults match OpenAI's
(`temperature: 1.0`, `top_p: 1.0`, `max_tokens: 256`, greedy is not the default); it adds
`top_k` and `repetition_penalty`, which are outside OpenAI's schema but common among local
servers.

A `Model` is not `Send`/`Sync` and holds device resources; a single instance must not be
driven concurrently (the NPU is single-tenant, and the control-plane layer below serializes
for exactly this reason).

Everything under `npu_engine::{pipeline, registry, bert, esm, asr, diarize, llm, capability,
config, tuning_profile}` is `#[doc(hidden)]` and, per the crate's own comment, may change
without notice. `npu_engine::capability::{Capability, Servable, Request, Response}` in
particular is described in its own source comment as a "PROBE, not the finished contract" --
validated against two model instances, not wired into the registry the HTTP server and CLI
actually use. It is not part of the stable surface.

## C ABI: `npu-capi`

`rust/npu-capi/src/lib.rs` exposes a handle-based C ABI (builds as a cdylib/staticlib with a
cbindgen header) over the same two layers: a single in-process model, and the multi-model
control plane.

Every function catches Rust panics at the FFI boundary and reports failure through a
thread-local `npu_last_error()` string rather than unwinding into C.

**Single model:**

- `int npu_available(void)`
- `NpuModel *npu_model_load(const char *scenario_path)` -- NULL on error
- `int npu_model_kind(const NpuModel *m)` -- 0=asr, 1=embed, 2=diarize, 3=generate, -1=error
- `char *npu_transcribe(NpuModel *m, const int16_t *pcm, size_t n, uint32_t sample_rate)` --
  caller frees with `npu_string_free`
- `char *npu_diarize(...)` -- returns `{"segments":[{"start":0.500,"end":3.200,"speaker":0}]}`
- `int npu_embed(NpuModel *m, const char *text, float *out, size_t out_cap)` -- call with
  `out == NULL` or `out_cap == 0` first to get the dimension, then again with a buffer of at
  least that size; returns the dimension written, or -1
- `void npu_string_free(char *s)`, `void npu_model_free(NpuModel *m)`
- `const char *npu_last_error(void)` -- valid until the next call on the same thread

**Control plane** (multi-model, config-driven, the same actor the HTTP server uses):

- `NpuRuntime *npu_runtime_start(const char *config_path)` -- loads the config and reconciles
  its models
- `char *npu_runtime_transcribe(NpuRuntime *rt, const char *model, const int16_t *pcm, size_t n, uint32_t sample_rate)`
  -- `model` NULL selects the configured default for ASR
- `int npu_runtime_embed(NpuRuntime *rt, const char *model, const char *text, float *out, size_t out_cap)`
- `int npu_runtime_reload(NpuRuntime *rt)` -- re-reads the config file and reconciles
- `char *npu_runtime_models_json(NpuRuntime *rt)` -- the same JSON shape as `GET /v1/models`
  below
- `void npu_runtime_stop(NpuRuntime *rt)`

## HTTP server

`rust/npu-runtime/src/http.rs`: a blocking, single-flight HTTP/1.1 server (`npu serve`,
default port `11434`, configurable via `[server] port` in `engine.toml`). Single-flight because
the NPU is single-tenant -- one request is served at a time by the same device actor described
in [general-engine.md](general-engine.md). Route dispatch is a pure function (`route()`) over a
mock-backed handle, separate from socket I/O, and its own test suite is the most precise
description of behavior at the edges; what follows is the route table it implements.

Bodies are capped at 16 MiB. Streaming responses use Server-Sent Events with no
`Content-Length`; the socket is closed at the end of the response either way
(`Connection: close`).

### Health and inventory

- `GET /health` -- `{"status":"ok"}`, unconditionally.
- `GET /healthz` -- `{"ok": bool, "npu": bool, "loaded": <n>, "failed": [<model names>]}`.
  `ok` is computed from whether any configured model is in the `Failed` state, not asserted --
  it returns 503 rather than 200 when a model failed to load, and 200 otherwise. `npu` reports
  whether an NPU device is present at all.
- `GET /v1/models` -- `{"object":"list","data":[{"id","object":"model","kind","state",
  "detail","bo_bytes","idle_s"}, ...]}` for every model named in the config. `state` is one of
  `loaded | failed | unloaded`; `idle_s` is seconds since the model last served a request, or
  `null` while not resident.

### Inference (OpenAI-compatible shape)

- `POST /v1/chat/completions` -- serves `Capability::GENERATE`. Body: `model` (optional,
  selects the config default when absent), `messages` (required, full array -- system prompt
  and history, not just the last turn), the sampling fields below, and `stream` (bool). Message
  `content` is a plain string or OpenAI's multi-part array form where every part is
  `{"type":"text","text":...}`; any other part type is rejected with 400, not silently dropped.
  Non-streaming response: `chat.completion` object with `choices[0].message.content` and a
  `usage` object (`prompt_tokens`, `completion_tokens`, `total_tokens`). Streaming: SSE frames
  of `chat.completion.chunk` objects (a role-announcement chunk first, then content deltas,
  terminated by a `finish_reason` chunk and a literal `data: [DONE]`).
- `POST /v1/completions` -- same generation path over a raw, non-templated prompt. Body:
  `model`, `prompt` (string only -- an array of prompts, OpenAI's batching form, is rejected
  with 400), sampling fields, `stream`. Response shape mirrors chat completions with
  `text_completion` objects and a `text` field instead of `message.content`/`delta.content`.
- `POST /v1/embeddings` -- serves `Capability::EMBED`. Body: `model` (optional), `input` (a
  string, or an array of strings). Response: `{"object":"list","data":[{"object":"embedding",
  "index","embedding":[...]}],"model":"<served model name>"}`.
- `POST /v1/audio/speech` -- serves `Capability::TTS`. Body: `model`, `input` (text). Response
  is a `audio/wav` byte body (44-byte canonical header + 16-bit PCM), not JSON. `voice` is
  accepted and currently ignored -- no configured model reads it.
- `POST /v1/audio/transcriptions` -- multipart upload, `file` part (any container ffmpeg can
  decode, not only 16 kHz mono WAV) plus an optional `model` form field. Response:
  `{"text":"...","model":"..."}`.
- `POST /v1/audio/diarizations` -- not an OpenAI endpoint; mirrors the shape of
  `/v1/audio/transcriptions` (multipart `file` + `model` form field) for
  `Capability::DIARIZE`. Response: `{"model":"...","segments":[{"start":0.500,"end":3.200,
  "speaker":"SPEAKER_00"}, ...]}`.

Shared sampling fields (chat and text completions): `temperature`, `top_p`, `top_k`,
`max_tokens`, `seed`, `stop` (string or array of strings), `presence_penalty`,
`frequency_penalty`, `repetition_penalty`, and `chat_template_kwargs.enable_thinking` (the
vLLM/SGLang spelling, read by Qwen3-family templates -- any other key inside
`chat_template_kwargs` is a 400). A field absent from the body keeps
`GenerateParams::default()` -- OpenAI's own defaults, never a silent substitution of greedy
decoding.

Fields the surface does not implement are rejected with 400, not silently ignored: `n != 1`,
`logprobs`, `logit_bias`, `tools`, `tool_choice`, `response_format`, `stream_options`, and,
completions-only, `echo`, `best_of != 1`, `suffix`.

### Errors

Every inference route maps `EngineError` to a status code the same way: `NoModel` and
`NotAvailable` -> 503 (nothing is wrong with the request; the server has no model configured
for that capability, or no NPU is present), `WrongKind`/`Unsupported` -> 400, `Load`/`Device`
-> 500. Bodies are `{"error":"<message>"}`.

### Admin (desired-state config)

These routes edit `engine.toml` on disk and then ask the device actor to reconcile against it
-- config is the persistence layer; a restart re-reads the same file.

- `POST /admin/reload` -- re-read the config and reconcile with no edit.
- `POST /admin/models` -- body `{"name","scenario"}`; add or replace a model entry (new models
  default to `resident: false`).
- `DELETE /admin/models/<name>`
- `POST /admin/defaults` -- body `{"capability","model"}`; set which model serves a capability
  when a request does not name one. `capability` must be one of `Capability::ALL` (currently
  `asr`, `embed`, `generate`, `tts`, `image-sr`, `diarize`); an unrecognized value is a 400
  rather than a silent no-op.

All four return `{"loaded","unloaded","failed","deferred"}` counts from the reconcile report on
success.

## Scenario and config files

Scenario TOML (`scenarios/*.toml`) is what a `Model::load` or a config's `[[model]]` entry
points at; it is not a code entry point but is the primary way this engine is actually driven,
so it is worth naming here rather than only in general-engine.md. Shape: `[scenario]`
(`kind`, `name`), an optional `[model]` block (`hidden`, `ff`, `n_heads`, `head_dim`,
`n_layers`, `max_seq`, `precision`, `kernel`; absent for non-transformer models), `[artifacts]`
(weight/tokenizer/checkpoint paths, or `decode`/`tokenizer_dir` for `kind = "generate"`), and a
per-kind block (`[embeddings]`: `pooling`, `normalize`; `[diarization]`: `manifest`). See
`rust/npu-engine/src/config.rs` for the full field list and `scenarios/*.toml` for shipped
examples across every kind.

`engine.toml` (the control-plane config `npu serve` reads) is a separate, smaller schema:
`[server]` (`port`, `max_resident`, `idle_unload_s`, `sweep_interval_s`, `idle_release_s`,
`evict_policy`, `memory_ceiling_mb`), `[defaults]` (capability name -> model name), and
`[[model]]` entries (`name`, `scenario` path, `resident` bool). See
`rust/npu-runtime/src/config.rs`.

## What is not documented here

Two crates expose their own device-facing ABIs that do not go through `npu-engine` or
`npu-runtime` at all: `npu-sr` (frame-in/frame-out video super-resolution, its own `SrEngine`
type and error) and `npu-sr-capi` (a C ABI over it, `libxdna_sr.so`, for the ffmpeg
`vf_xdna_sr` filter). They are real, shipped surfaces, but a separate one from everything
above -- see [general-engine.md](general-engine.md) for why they sit outside this pipeline.
