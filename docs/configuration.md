# Configuration

The engine has three layers of configuration:

1. **`engine.toml`** -- desired state: which models are configured, which capability each
   default resolves to, and server-wide policy (residency, port). This is what `npu serve`
   reads and what `npu config` / the `/admin/*` HTTP routes edit.
2. **Scenario TOML** (`scenarios/*.toml`) -- one file per model: its shape, its precision,
   and where its weights live. `engine.toml` points at these by path.
3. **Environment variables** -- install-time paths and a handful of runtime overrides.

## `engine.toml`

Resolved in this order: the `--config` flag, then `$NPU_CONFIG`, then
`~/.config/npu/engine.toml`. A missing file is not an error -- `Config::load` returns the
default empty config, which serves nothing until you add a model.

```toml
[server]
port = 11434
max_resident = 2
idle_unload_s = 900
evict_policy = "lru"

[defaults]
asr = "parakeet"
embed = "bge-base"

[[model]]
name = "parakeet"
scenario = "scenarios/asr.toml"

[[model]]
name = "bge-base"
scenario = "scenarios/bge-base.toml"
```

### `[server]`

Every key is optional; an absent one takes its default.

| Key | Default | Meaning |
| --- | --- | --- |
| `port` | `11434` | HTTP listen port. Shared with FLM and ollama by convention -- `npu serve` refuses to bind if something else is already answering there and it is not another xdna-engine instance. |
| `max_resident` | `1` | How many models may be resident (loaded on the device) at once. Not a hard cap on config size -- a request for another model evicts one per `evict_policy`. |
| `idle_unload_s` | `900` | Unload a model that has served no request for this long, freeing the device. `0` disables idle unload. |
| `sweep_interval_s` | `30` | How often the device actor checks for idle models. Only checked between commands, so this is also the worst-case delay before an idle model is released. Clamped to at least 1s. |
| `idle_release_s` | `1800` | A second, deeper idleness level: after this long since the last *request* (not just since unload), the memory an unload freed but the allocator kept is given back. `0` disables it. |
| `memory_ceiling_mb` | `4096` | Ceiling on summed device buffer-object bytes across resident models. **Partial**: Parakeet reports its real footprint; every other shipped model still reports `0` and is not bounded by this. A model without a measured footprint says so in its status detail rather than passing silently. |
| `evict_policy` | `"lru"` | What to evict when `max_resident` is full and a new model needs the slot: `"lru"` (drop the least-recently-used resident model) or `"none"` (refuse the new load instead). |

### `[defaults]`

Maps a capability name to the model that serves it when a request does not name one
explicitly (`{"model": "..."}` absent from an HTTP body, or `--model` absent on the CLI).

```toml
[defaults]
asr = "parakeet"
embed = "bge-base"
diarize = "pyannote-3.1"
generate = "qwen3-0.6b"
```

The capability names are `asr`, `embed`, `diarize`, `generate` (also `tts` and
`image-sr`, which no shipped model currently implements). **Note the one mismatch**: the
capability is `embed`, but an embeddings scenario's own `[scenario] kind` is
`"embeddings"` -- those are two different vocabularies (scenario kind selects which
pipeline builder runs; capability is what request routing matches on).

### `[[model]]`

One entry per model this install knows about.

```toml
[[model]]
name = "parakeet"
scenario = "scenarios/asr.toml"
resident = false
```

- `name` -- an arbitrary label. This is what `--model`, a request's `"model"` field, and
  `[defaults]` values refer to.
- `scenario` -- path to the scenario TOML, resolved against the engine root (see below)
  if relative.
- `resident` (default `false`) -- pin this model so it is never chosen as an eviction
  victim and never idle-unloaded. Set it with `npu config pin <model>` / `npu config unpin
  <model>`, or by hand.

  **A pin is an exemption, not an entitlement.** It keeps a model that IS resident from being
  swept or evicted; it does not win it a slot it would not otherwise have had. Boot admission
  is still the first `max_resident` models in config order, so a pin listed after enough
  others is simply not loaded at startup -- it loads on the first request that routes to it
  and then stays. `npu config show`, `npu config pin` and the server's startup log all say so
  when that is the case, rather than leaving the config stating an intent the runtime declined.

  Pinning every model leaves no eviction victim at all; `npu config show` and the config
  summary warn when pinned models are `>= max_resident`.

### Editing it

```
npu config show
npu config add-model <name> <scenario-path>
npu config remove-model <name>
npu config set-default <capability> <model>
npu config pin <model>                # resident = true
npu config unpin <model>              # resident = false
npu config set <key> <value>          # one [server] key; `npu config set --help` lists them
```

`npu config set` covers the residency knobs: `max_resident`, `idle_unload_s` (`0` switches
idle unload off entirely), `idle_release_s`, `sweep_interval_s`, `evict_policy`, plus `port`
and `memory_ceiling_mb`. The key list is closed -- an unrecognised key is refused rather than
written, because a key nothing reads produces a file that still parses and silently does
nothing.

Each of these edits `engine.toml` **in place** and saves it atomically (temp file + rename).
In place, not re-serialized from the parsed struct: the struct does not carry comments, so
rewriting the file from it deleted every one of them -- including the comments the generated
config ships with. Re-running `add-model` on a name already present updates that entry's
scenario and leaves its other keys, including `resident`, alone.

### Residency at runtime

`engine.toml` is desired state. To change what is on the device **now**, without editing the config
or restarting:

```
npu load <model>          # make it resident now
npu unload <model>        # give its device memory back now
```

`npu load` **refuses** rather than evicting when the server is already at `max_resident`, and the
refusal names what is holding the slots. That is deliberate and is the one place the operator path
differs from the request path: a request names a capability, so swapping a model in to serve it is
right; an explicit load is a statement about *capacity*, and honouring it by dropping a model
someone else pinned would answer a different question. Free a slot with `npu unload`, or raise the
cap with `npu config set max_resident <n>`.

Both are idempotent, and neither touches `engine.toml` -- so neither survives a restart. For
residency that does, pin the model.

> **`memory_ceiling_mb` bounds real bytes only for Parakeet today.** It sums `Servable::footprint()`
> across resident models; Parakeet reports its actual pinned device BO total, and every other shipped
> model still returns a hardcoded `0` (`npu-engine/src/pipeline.rs`'s trait defaults, unwired for
> Whisper, the generic GigaAM ASR path, BERT/ESM embed, and text generation). A ceiling with a
> non-Parakeet model resident is not enforcing anything for that model's share. `npu load` says which
> resident models the accountant could not weigh, rather than letting a ceiling you just set look
> like it covers everything that is loaded. `max_resident` (a model COUNT) is the only limit
> enforced uniformly across every model kind.

A running service does not pick up a config edit until reloaded:

```
npu reload                       # POST /admin/reload on the running server
```

The same edits are available over HTTP for a running server: `POST /admin/models`
(`{"name":..., "scenario":...}`), `DELETE /admin/models/<name>`, `POST /admin/defaults`
(`{"capability":..., "model":...}`), each of which saves the config and reconciles
immediately, replying with `{"loaded":N,"unloaded":N,"failed":N,"deferred":N}`.

## Scenario files

A scenario is one TOML file describing one model. `scenarios/` in the repo holds the
shipped examples; a scenario referenced from `engine.toml` by a relative path resolves
against the engine root.

```toml
[scenario]
kind = "asr"                 # "asr" | "embeddings" | "diarize" | "generate"
name = "parakeet-tdt-0.6b-v3"

[model]
hidden = 1024
ff = 4096
n_heads = 8
head_dim = 128
n_layers = 24
max_seq = 2040
precision = "bf16"

[artifacts]
weights = "artifacts/parakeet"
```

### `[model]`

Optional -- present for transformer-shaped models, absent for scenarios whose model has
none of these fields (diarization's PyanNet/ResNet34, for instance). A builder that
needs it and finds it missing gets a loud error naming the scenario, not six fabricated
numbers.

| Field | Default | Meaning |
| --- | --- | --- |
| `hidden`, `ff`, `n_heads`, `head_dim`, `n_layers`, `max_seq` | -- (required) | Model shape. |
| `n_mels` | `80` | Log-mel filterbank channels the frontend produces (80 for pre-large-v3 Whisper, 128 from large-v3 on). |
| `n_decoder_layers` | `n_layers` | Decoder depth, when the model has one and it differs from the encoder depth (Whisper-turbo: 32 encoder / 4 decoder). |
| `precision` | `"bf16"` | `native` \| `bf16` \| `int8`. |
| `kernel` | `"zeropad"` | Matmul-shape strategy; ESM's scenarios also use `"native"`. |

### `[artifacts]`

| Field | Meaning |
| --- | --- |
| `weights` | Legacy npy/f32 weights directory. Still the default when no `source` is set. |
| `tokenizer` | Tokenizer path (a single file for most models). |
| `onnx_ref` | ONNX reference graph, used for parity checks. |
| `decode` | `kind = "generate"` only: directory holding a fused-decode ELF (`meta.json` + `decode.elf` + `buffers/`). |
| `tokenizer_dir` | `kind = "generate"` only: directory holding `tokenizer.json` / `tokenizer_config.json` / `generation_config.json`. |
| `source` | Declarative weight source: `"hf:<repo>[@rev]"` or `"path:/abs"`. When set, the engine resolves and bakes (if missing) a `npu-weights` checkpoint instead of reading the npy `weights` dir. |
| `arch` | Required when `source` is set. One of: `bert`, `clip`, `dinov2`, `edsr`, `espcn`, `esm`, `fastconformer`, `gigaam`, `modernbert`, `opt`, `resnet`, `vit`, `whisper`. |
| `checkpoint` (alias `arena`) | Explicit `.safetensors` path. Left empty, it is derived as `${XDNA_CHECKPOINT_DIR:-<root>/artifacts/checkpoints}/<arch>__<source>__<fingerprint>.safetensors`. |

All shipped scenarios still use the plain `weights` npy path; `source`/`arch` are an
additive, opt-in alternative.

### `[embeddings]`

```toml
[embeddings]
pooling = "mean"      # default "mean"; "cls" also used
normalize = true       # default true
```

### `[diarization]`

```toml
[diarization]
manifest = "artifacts/pyannote/speaker-diarization-3.1/diarize.json"
```

One field on purpose: every pyannote hyperparameter lives in the manifest the export
script writes, alongside the upstream source it came from, so nothing gets retyped here.

`kind = "generate"` scenarios (small LLMs) need no `[model]` block at all -- the fused
decode ELF's own `meta.json` carries every dimension. Only `[artifacts] decode`,
`weights`, and `tokenizer_dir` are read. See `scenarios/generate-qwen3-0.6b.toml`.

## Root and path resolution

Every relative path in a scenario or in `engine.toml` resolves against an **engine
root** -- the directory holding `scenarios/` and `artifacts/`. Resolution order:

1. `$XDNA_ENGINE_ROOT`, if set (this is what the installed systemd unit sets).
2. An absolute `.../scenarios/x.toml` scenario path in the config names its own root.
3. The config file's own directory.
4. The current working directory.
5. `${XDG_DATA_HOME:-~/.local/share}/xdna-engine` (where `install.sh` stages a production
   root).

Each candidate is only accepted if it actually contains a `scenarios/` directory.

## Environment variables

### Install-time (`install.sh`)

| Variable | Default | Meaning |
| --- | --- | --- |
| `ONNX_ASR_VENV` | searches `./.venv`, then `~/.local/share/xdna-engine/onnx-asr-venv` | Python venv with `onnx_asr` importable; used to run the service and generate ASR artifacts. |
| `EXPORT_VENV` | `$REPO/.venv` | Venv used to (re)generate encoder artifacts via the export scripts. |
| `XRT_INC_DIR` | `/usr/include` | Where to find `xrt/xrt_bo.h`. |
| `XRT_LIB_DIR` | `/usr/lib` | Where to find `libxrt_coreutil.so*`. |
| `MODEL` | `parakeet` | `parakeet` or `gigaam`; selects which artifact set the install preflight checks for. |
| `ENGINE_BIN_DIR` | `~/.local/bin` | Where the `npu` binary is installed. |
| `ENGINE_CONFIG` | `~/.config/npu/engine.toml` | The config the installed unit is pointed at. |
| `ENGINE_ROOT` | `${XDG_DATA_HOME:-~/.local/share}/xdna-engine` | The stable production root staged for the service (`XDNA_ENGINE_ROOT` in the unit). |
| `ENGINE_ARTIFACTS` | `$REPO/artifacts` | What `$ENGINE_ROOT/artifacts` symlinks to. Point this elsewhere if weights live on another partition. |
| `ENGINE_MLIR_AIE` | `$REPO/mlir-aie` | Source for the kernel publish step. |
| `STABLE_LIB_DIR` | `~/.local/lib/xdna-engine` | Where the hardened `libonnxruntime.so` copy lives. |
| `COMPLETION_DIR` | first writable zsh `$fpath` entry, else `${XDG_DATA_HOME:-~/.local/share}/zsh/site-functions` | Where shell completions are written. |

### Runtime (the `npu` binary and service)

| Variable | Meaning |
| --- | --- |
| `NPU_CONFIG` | Config path, when `--config` is not passed. |
| `XDNA_ENGINE_ROOT` | Engine root override (see "Root and path resolution"). |
| `NPU_QUIET` | Set to suppress a one-shot CLI command's load-time banners (the CLI sets this to `1` for itself by default; `NPU_QUIET=0 npu embed ...` brings the banners back). |
| `NPU_PRECISION` | `native` \| `bf16` (default) \| `int8` -- runtime precision selector read by the ASR/Whisper NPU paths. |
| `NPU_ASR_MAX_SPAN_S` | Default `18.0`. Longest audio span sent to ASR in one call from `transcribe-media`. |
| `NPU_DIARIZE_THREADS` | Thread pool size for the diarization embedder (default: measured-optimal for a 10-core/20-thread box; falls back on `0` or an unparseable value). |
| `NPU_DIARIZE_MEM_MB` | Memory budget for diarization embedding batch sizing (~55 MB/crop is the measured slope). |
| `NPU_WEIGHTS_CHECKPOINT` | Path to a bf16-baked `npu-weights` checkpoint, as an opt-in alternate weight source to the npy path. |
| `XDNA_CHECKPOINT_DIR` | Base directory for derived checkpoint paths (`<arch>__<source>__<fingerprint>.safetensors`), when a scenario's `[artifacts] checkpoint` is unset. |
| `NPU_XCLBIN_ROOT` | Overrides where the Parakeet NPU path resolves its resident xclbins from (default: the engine root). |
| `NPU_KERNEL_MANIFEST_VERIFY` | Set (to any value) to re-hash kernel artifacts against their manifest at load time. Default off. |
| `NPU_HOST_PROF` | Set to enable the per-op host-side profiler (ASR host reference path). Default off, zero cost. |
| `NPU_DISPATCH_LOG` | Set to log per-`(xclbin, insts)` dispatch blocking time and hw_context-transition counts. Covers EVERY kernel dispatch in the engine, not only LLM decode. |

This is the operator-facing surface. The kernel and dataflow crates carry additional
environment-gated switches used for research and ablation during kernel development;
those are not part of the stable configuration surface and are not covered here.

## Weight checkpoints (`npu weights`)

```
npu weights bake --source hf:facebook/opt-125m --arch opt [--checkpoint PATH] [--force]
npu weights load --checkpoint PATH --arch opt
npu weights verify --checkpoint PATH --arch opt --refs DIR
```

`bake` skips the work if a checkpoint already exists and is fresh, unless `--force`.
`verify` checks a baked checkpoint's tensors against a directory of reference `.npy`
files within a `5e-2` max relative-error tolerance and prints `PARITY PASS` or fails
loud with the offending tensor's error. `npu bake <model-name>` is a shortcut that reads
the source/arch from a model already in `engine.toml`.

## HTTP API surface

OpenAI-compatible routes accept the subset of the request schema the engine actually
implements. Sending any of `n != 1`, `logprobs`, `logit_bias`, `tools`, `tool_choice`,
`response_format`, or `stream_options` gets a `400`, naming the field, rather than a
silent no-op.

Status codes carry meaning: `503` means "the server has no model for this capability, or
no NPU device" (nothing a client can fix by retrying differently); `400` means the
request itself is malformed or asks for something unsupported; `500` means a device or
load error.
