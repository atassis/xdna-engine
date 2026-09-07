# Porting a model onto xdna-engine

This is the real sequence for bringing a new model onto the engine, derived from how the
five models already here (GigaAM, Parakeet, Whisper, the BERT family, ESM-2) are actually
wired, not from a generic template. Every path and symbol below exists in the tree today.

A new model is four pieces of work: weight conversion, the pipeline contract, kernel
reuse-vs-authoring, and a correctness gate. Do them in that order -- each one depends on
the previous.

## 0. Decide the shape

Two different things count as "a new model" here, and they cost differently:

- **A new encoder** that fits the shared `Frontend / Encoder / Head` pipeline
  (`rust/npu-engine/src/pipeline.rs`) -- another embedding model, another ASR encoder,
  a vision backbone. This is the cheap path: BERT-family embeddings live entirely as
  three small files inside `npu-engine` (`rust/npu-engine/src/bert/`).
- **A new architecture family** -- an autoregressive decoder, or anything whose control
  flow doesn't fit `Encoder::forward_last`'s single forward pass. This gets its own
  crate (`npu-whisper`, `npu-parakeet`, `npu-gemma` are each a crate), and reuses the
  fused-decode / KV-cache primitives rather than the `Encoder` trait.

`ARCHITECTURE.md`'s "Known seams" section is honest about the cost of the second path:
model placement has no single rule (GigaAM is `npu-asr`, Parakeet is `npu-parakeet`, BERT
and ESM are modules inside `npu-engine`), and the capability set (`ModelKind`/`Scenario`)
is closed, so a genuinely new modality means editing `registry.rs`'s dispatch, not just
adding a file. Read that section before choosing where a new model lives.

## 1. Weight conversion

Weight conversion is a `npu_weights::arch::Arch` implementation, one file under
`rust/npu-weights/src/arch/`:

```rust
pub trait Arch {
    fn name(&self) -> &'static str;
    fn required_tensors(&self, n_layers: usize) -> Vec<String>;
    fn transform(&self, src: &BTreeMap<String, RawTensor>) -> anyhow::Result<BTreeMap<String, OutTensor>>;
}
```

(`rust/npu-weights/src/arch/mod.rs`). `transform` takes the source tensor bag (row-major
f32, whatever names the source checkpoint uses) and returns the engine's own tensor
names/layout, each marked `bf16: bool` for the baked dtype. Register the new arch in
`arch::get()`'s match in the same file.

`rust/npu-weights/src/arch/bert.rs` is the simplest real example: it infers encoder depth
by scanning for the highest `encoder.layer.{i}.` index present, transposes every linear
weight (`transpose2d`, so GEMM weights are laid out the way the engine's matmul wrappers
expect), keeps norm gamma/beta and biases as f32, and marks GEMM weights `bf16: true`.
`fastconformer.rs`, `whisper.rs`, `opt.rs`, `vit.rs` etc. are more of the same shape for
their own tensor-naming conventions.

Wire it up declaratively, not by hand-running a script. `npu_weights::spec::ModelSpec`
(`rust/npu-weights/src/spec.rs`) is `{ source: Source::Hf{repo,rev} | Source::Path(..),
arch: String, checkpoint: Option<PathBuf> }`. `ModelSpec::ensure_checkpoint()` resolves
the source files, fingerprints them (sha256), and bakes a `.safetensors` checkpoint under
`artifacts/checkpoints/<arch>__<source>__<fp12>.safetensors` if one isn't already there --
this is the single entry point both the engine and the `npu weights bake` /
`npu bake <model>` CLI commands call (`rust/npu-cli/src/main.rs`).

A scenario TOML opts into this by setting, under `[artifacts]`:

```toml
source = "hf:org/repo"      # or "path:/abs/dir"
arch = "your_arch_name"
```

(`rust/npu-engine/src/config.rs`'s `Artifacts::model_spec()`). This is additive -- a
scenario that instead sets the legacy `weights = "artifacts/..."` npy directory keeps
working unchanged; every shipped scenario in `scenarios/` still uses that path except
where a declarative one has been proven (`rust/npu-engine/tests/declarative_weights.rs`
is the host-only, no-NPU proof that a synthetic source checkpoint round-trips through
`transform` -> bake -> load).

Your weight-store type then needs a `load_checkpoint()` reading
`npu_weights::checkpoint::load(path, arch)` the way `BertWeights::load_checkpoint()`
does (`rust/npu-engine/src/bert/weights.rs`) -- same tensor-name scheme your `transform`
wrote (`emb/<k>` and `L{i}/<k>` for BERT).

If you need a host-side reference to check the on-NPU path against during bring-up,
`[artifacts] onnx_ref` points at an exported ONNX graph, run through `npu-onnx`
(`rust/npu-onnx`, "oracles + fallback" per `ARCHITECTURE.md`) via a thin C shim over the
system onnxruntime.

## 2. Implement the pipeline contract

`pipeline::Encoder` is the one trait every sibling encoder implements -- its own doc
comment calls it "the genuinely-shared, genuinely-hard NPU stage. INTERFACE CONTRACT for
sibling models (GigaAM Conformer, Parakeet FastConformer, BERT)":

```rust
pub trait Encoder {
    fn forward_last(&self, x: &Array2<f32>, valid_len: usize) -> Array2<f32>;
}
```

`x` is `[M, D_in]` (bf16-valued f32), `valid_len` the non-padded row count, return is
`[M, D]`. `Frontend` (raw input -> `(Array2<f32>, valid_len)`) and `Head`
(encoded -> `Self::Output`) are per-domain host glue with an associated type, not shared
across models -- look at `rust/npu-engine/src/bert/frontend.rs` (WordPiece tokenize ->
summed word/position/type embeddings -> LayerNorm) and `bert/head.rs` (mean/CLS pooling +
optional L2-normalize) for what "small" actually means: 62 and 67 lines.

Assemble the three into one pipeline struct with a `build(cfg, root, dev)` constructor,
the way `bert::EmbedPipeline` does: it owns a `Frontend`, an `Encoder`, and a `Head`, and
its own `embed()` method chains `frontend.run() -> Encoder::forward_last() -> head.run()`.

Then implement exactly one of the outer capability traits, chosen by modality, from the
same file:

- `Embedder::embed_one(&self, text: String)` -- text embedding (BERT, ESM).
- `AsrModel::transcribe(&self, samples: &[i16])` -- full ASR (raw PCM -> text); the
  encoder stage inside still implements `Encoder`.
- `Diarizer::diarize(&self, pcm: &[i16])` -- speaker-attributed spans.
- `TextGenerator::generate(&mut self, prompt, params, sink)` -- autoregressive decode,
  `&mut self` because a decoder genuinely owns a KV cache and a resident device context
  (see the trait's own doc comment on why the other three models launder that mutability
  through `RefCell` instead of declaring it).

Finally wire the build path into `registry::try_build()` (`rust/npu-engine/src/registry.rs`).
Today's dispatch is `ModelKind::from_scenario_kind(&cfg.scenario.kind)` and then, for ASR
and Embed, a `cfg.scenario.name.to_lowercase().contains("parakeet"/"whisper"/"esm")`
match with a generic fallback. This is the seam `ARCHITECTURE.md` calls out ("the
capability set is closed") -- adding a model that fits an existing `ModelKind` is a match
arm here; adding a genuinely new modality means also touching `capability.rs`'s
`Capability::ALL` and `Request`/`Response` shapes.

Add the scenario file itself under `scenarios/`. Fill `[model]`
(`hidden`/`ff`/`n_heads`/`head_dim`/`n_layers`/`max_seq`/`precision`/`kernel`, see
`ScenarioConfig::ModelCfg` in `config.rs`) if the model is transformer-shaped; omit the
block entirely otherwise -- `scenarios/diarize-pyannote-3.1.toml` has none, and
`config.rs`'s own test (`a_scenario_without_a_model_block_parses_and_carries_its_diarization_manifest`)
is the proof that's a supported shape, not an oversight.

## 3. Which kernels get reused vs authored

Decide the compute regime before touching a kernel: `docs/execution-graph.md` section 0
is the lookup table. `M >= 8` (encoder, prefill, vision, batch>=128 decode) wants
`aie::mmul`; `M = 1` (single-stream autoregressive decode) wants GEMV plus residency and
op-count reduction, not a bigger matmul. Getting this wrong means authoring or wiring the
wrong kernel shape.

**Reuse path.** `bert::encoder.rs`'s own header says it plainly: "BERT encoder on the NPU:
post-norm Transformer layers reusing the npu_asr matmul engines." It authors zero new
kernels -- `BertBlock` is built from `npu_asr::ctx2::{CtxAOp, FfnMm2, SharedCtxA}` (the
GEMM+bias+epilogue engine originally built for GigaAM) plus `npu_asr_host::{gelu,
layer_norm, mha}` for the pieces still running on host. If your model's ops (linear
projection, LayerNorm/RMSNorm, softmax, GELU/SiLU, standard MHA) already have a kernel in
`aie_kernels/` and a Rust wrapper in an existing crate, wire the wrapper the way
`BertBlock::forward` does -- this is most of what "porting a model" actually is.
`aie_kernels/INDEX.md` is the full inventory of the 48 kernels that exist, their dtype
and shape contract, and (honestly) which have never been verified on device.

**Gap path.** When an op has no kernel -- the depthwise-2D conv gap
`docs/execution-graph.md` section 4 calls out, blocking MobileNet/EfficientNet-shaped
convs -- you author: an AIE C++ kernel under `aie_kernels/<name>/` (a `.cc` with its
dtype/shape contract documented at the top, `extern "C"` entry points, following the
pattern of the existing 48), and, if it needs a multi-core dataflow rather than a single
kernel call, an IRON generator under `designs/<name>/`. `designs/ffn_gemm2/ffn_gemm2_iron.py`
is a concrete worked example: two cores pipelined through an on-chip `ObjectFifo`
(`H = A@W1` never leaves the device before `C = H@W2` consumes it), with its own
`Makefile.ffn`. Build the xclbin through the pinned toolchain: `scripts/toolchain_up.sh`
resolves/builds the instance, and `scripts/build_kernels.sh` shows the actual build
pattern (`make -C <mlir-aie-example-dir> NPU2=1`) against the `mlir-aie` submodule after
`scripts/sync_kernels.sh` copies the canonical kernel/design sources in.

For a decoder-only LLM specifically, don't estimate the job from scratch -- read
`rust/npu-gemma/src/lib.rs`'s module doc comment first. It is a worked port-map table for
Gemma 3: most rows are REUSE (fused-decode GEMV, KV-cache write via `StridedCopy`,
softmax, on-chip argmax -- all already shipped for the Whisper decoder) or WIRE (an
operator that exists in the vendored IRON operator library but isn't plumbed into this
engine yet, e.g. `rms_norm`, `rope`, `swiglu_decode`), and exactly one row needed new
kernel authoring (`head_dim=256` in the prefill flash-attention kernel, which hardcodes
`d=64`). Write the equivalent op-by-op table for your model before estimating the work.

## 4. Gating correctness

Four independent gates, each catching a different class of mistake:

1. **Per-kernel golden.** A NumPy/PyTorch reference at the exact op shape. The bar,
   from `docs/execution-graph.md`'s correctness contract: rel-L2 <= 0.08 (standard),
   < 0.02 (tight bf16-GEMM bar), or corr >= 0.99 (cascade ELF). `aie_kernels/INDEX.md`
   records, per kernel, whether a `golden.py` exists (31 of 48 do) and which `verify_*.py`
   harness runs it -- and is honest where a kernel has never actually run on device.
2. **Weight-checkpoint parity.** `max rel-err < 5e-2` (the bf16 floor sits around
   3.89e-3), enforced in code by `checkpoint::verify_against_npy` (`npu weights verify`,
   `rust/npu-cli/src/main.rs`) and by one `rust/npu-weights/tests/parity_<arch>.rs` per
   registered arch -- write the analogous test for a new one. These run with no NPU.
3. **Device numeric parity**, when a change touches an existing on-NPU path:
   `scripts/encoder_parity.py`. It compares candidate/shipped/f32-truth encodes on four
   statistics (mean, worst-frame, worst-burst rel-L2, plus a `new-burst` delta) rather
   than a single mean, specifically because a whole-clip mean can average a short,
   localized error spike into invisibility. Its own docstring shows the usage pattern:
   capture three encode dirs with a `npu-probes` binary
   (`cargo run -p npu-probes --bin parakeet_encode_npu -- ...`), then gate them.
4. **End-to-end task metric.** `scripts/wer_eval.py` for ASR: transcribes a clip set
   against the live service and a CPU oracle, reports WER for both. There is no generic
   equivalent yet for embeddings or vision in this tree -- plan to write one (a small
   labeled set, or agreement against a known CPU/ONNX baseline) rather than skip this
   gate.

**CI coverage.** `.github/workflows/rust-ci.yml` builds, clippies and tests exactly three
crates on a hosted runner -- `npu-asr-host`, `npu-gemma`, `npu-weights` -- because nothing
else in the workspace compiles without XRT and onnxruntime headers/libs. This is exactly
why a new `Arch` implementation and its parity test belong in `npu-weights`: it's the one
place a new model's weight-conversion correctness is machine-checked without hardware.
Anything that touches the NPU-dependent crates is gated locally with `scripts/ci_gate.sh`
on a box that has the device (see `CONTRIBUTING.md`).
