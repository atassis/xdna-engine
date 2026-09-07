# Troubleshooting

Real failure modes, each traceable to a specific guard or message in the tree. If you
hit something not listed here, `journalctl --user -u xdna-engine.service -f` and the
error text itself are the next place to look -- error strings in this codebase are
written to name the cause, not just fail.

## Build and install

**`cargo not found on PATH`** -- install Rust (rustup.rs). `install.sh` checks this
before anything else.

**`XRT headers not found under $XRT_INC_DIR`** / **`XRT libs not found under
$XRT_LIB_DIR`** -- the `amdxdna` XRT package is not installed, or it lives somewhere
other than `/usr/include` / `/usr/lib`. Override with `XRT_INC_DIR=` / `XRT_LIB_DIR=`.

**`'import onnx_asr' failed in <venv>`** -- the venv passed as `ONNX_ASR_VENV` (or found
by the default search) does not have `onnx_asr` installed. Point at a venv that does.

**`engine config missing: $ENGINE_CONFIG`** -- `install.sh`'s final preflight found no
`~/.config/npu/engine.toml`. Create one (see [configuration.md](configuration.md)) or
set `ENGINE_CONFIG=` at an existing file.

**`engine config references a missing scenario`** / **`scenario '...' points at
missing/empty weights`** -- a `[[model]]` in `engine.toml` names a scenario file, or a
scenario names a weights directory, that does not exist. The installer refuses rather
than shipping a service that would start and serve nothing; the message names the exact
path it could not find. Generate the missing artifacts (see "Model artifacts" in
[getting-started.md](getting-started.md)) or fix the path.

**Parakeet NPU xclbin missing from the published set** -- the install preflight looks
for a resident xclbin matching a specific naming pattern
(`final_512x1024x{n}_64x32x128_8c{variant}.xclbin`) under the published kernel set. If
none of the accepted variants exist, build them: `scripts/build_parakeet_kernels.sh`
(needs the mlir-aie toolchain), then re-run `install.sh` to republish.

**Everything reports success but the running service is still the old binary** -- `git
pull` alone does not update `~/.local/bin/npu`; re-run `install.sh`, which rebuilds and
reinstalls it. The install script's own step 3a exists specifically because an earlier
version of it built but never installed.

## The NPU device

**`no XDNA2 NPU device available`** / **`no XDNA2 NPU device at /dev/accel/accel0 (is
the amdxdna driver loaded?)`** -- `/dev/accel/accel0` does not exist. Check the
`amdxdna` kernel module is loaded (`lsmod | grep amdxdna`) and that the PCI device
(`1022:17f0`) is bound.

**`device error: open NPU (stop other ASR/embeddings service first): ...`** -- the NPU
is single-tenant; another process already holds a hardware context. Find it with
`fuser -v /dev/accel/accel0` and stop it (a running `xdna-engine.service`, FLM, or a
leftover probe/benchmark process), then retry.

**`port <N> is already served by an xdna-engine instance`** -- another `npu serve` (very
likely the systemd unit) already owns the port. `systemctl --user status xdna-engine`
to check, `systemctl --user stop xdna-engine` to free it, or `npu serve --port <other>`.

**`port <N> is already in use by another process`** (and it did not answer `/healthz` as
an xdna-engine) -- port 11434 is a shared default with FLM and ollama. `ss -ltnp 'sport
= :11434'` to identify it. The installed unit declares `Conflicts=flm-asr.service`, so
starting `xdna-engine.service` stops FLM automatically; this message is for the process
neither of those knows about.

**The NPU is wedged** -- a hung kernel can leave every later submission failing with
`DRM_IOCTL_AMDXDNA_GET_INFO IOCTL failed (err=-22)`, and a plain `modprobe -r amdxdna &&
modprobe amdxdna` leaves the PCI device present-but-unbound (`/dev/accel/accel0` gone).
`sudo bash scripts/reset_npu.sh` does a full PCI remove + rescan + driver reload in one
sequence, which forces a clean firmware re-init. If `/dev/accel/accel0` is still absent
afterward, a reboot is the guaranteed fix.

**The `amdxdna` module did not actually rebuild after a kernel update** -- on a DKMS
install this can fail silently against a non-default compiler toolchain while the
package manager still reports success. `scripts/driver_status.sh` reports which module
is actually loaded (DKMS vs in-tree) and whether it matches the pinned commit in
`driver.lock`; run it after any kernel update.

## Service startup

**Service refuses to bind, listing `FAILED <model>: <reason>`** -- one or more
configured models failed to load. `npu serve` refuses to bind the port in this case
(rather than starting and answering every request with an error, which is how a past
outage went unnoticed for days) unless you pass `--allow-degraded`, in which case it
binds anyway and `/healthz` reports `503`.

**`/healthz` returns `503`** -- at least one configured model is in the `Failed` state.
`{"ok":false,"npu":true,"loaded":N,"failed":["name", ...]}` names which. `Unloaded` is
not a failure (a model deferred by `max_resident`, or idle-swept) and never triggers
this.

**A model stays `Unloaded` and never serves** -- either `max_resident` has no free slot
and `evict_policy = "none"` is set (so nothing is ever evicted to make room), or pinned
models (`resident = true`) already fill every slot. `npu config show` prints a warning
when pinned models are `>= max_resident`.

**Alternating between two capabilities is slow every time** (e.g. ASR then diarization,
back and forth) -- with `max_resident = 1` (the default), each request for the
not-currently-resident model evicts and reloads. Raise `max_resident` to fit both models
resident at once.

## CLI

**`npu: error while loading shared libraries: libonnxruntime.so.1`** -- you have a
`npu` binary that was built without the RPATH bake `install.sh` performs (a bare `cargo
build` or a binary copied out from a different build). Reinstall via `install.sh`,
which bakes an RPATH to the stable onnxruntime directory; the systemd unit works around
the same issue for the service by setting `LD_LIBRARY_PATH` directly.

**`this model has no chat template -- use npu generate --raw`** -- `npu generate`
sends the prompt through the model's chat template by default (the correct default for
an instruction-tuned model). A base model with no chat template needs `--raw` for plain
continuation instead.

**`sample_rate <N> (need 16000)`** -- ASR paths require 16 kHz mono PCM. `npu
transcribe` and `transcribe-media` decode through ffmpeg automatically for anything
that is not already a 16 kHz mono 16-bit WAV; calling the engine API directly with a
different rate hits this error.

## HTTP API

**`400` on a chat/completions request** naming a specific field (`"n" != 1 is not
supported`, `"logprobs" is not supported`, or one of `logit_bias` / `tools` /
`tool_choice` / `response_format` / `stream_options`) -- these fields are recognized but
not implemented; drop them from the request rather than expecting a silent no-op.

**`503` on `/v1/chat/completions` or `/v1/audio/speech`** -- no model configured for that
capability (`generate` / `tts`), or no NPU device. This is a server-configuration fact,
not a bad request, which is why it is `503` and not `400`.

## Weight checkpoints

**`parity FAILED: max rel-err ... >= 5e-2`** from `npu weights verify` -- the baked
checkpoint's tensors disagree with the reference `.npy` files beyond tolerance. This
means the bake transform for that `arch` is wrong for this checkpoint, not a flaky
threshold; check the arch transform against the source checkpoint's actual layout.

## Where errors are worth reading verbatim

Most failures in this codebase carry the fix in the message: preflight failures name
the exact missing path or scenario, device errors say what to stop first, and load
failures are attributed to the model that caused them (`/v1/models` and `npu models`
report `state` and `detail` per model). Read the message before searching for a
workaround.
