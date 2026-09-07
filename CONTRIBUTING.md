# Contributing

This is a solo project -- I wrote it and I review everything that comes in. There's no
team and no SLA on review turnaround; I'll get to a PR when I get to it.

## Building

```
cd rust
cargo build              # the product workspace (default-members in rust/Cargo.toml)
cargo build -p npu-probes             # dev-only: device probes/parity checks, not built by default
cargo build -p npu-vod-chain-skeleton # dev-only: a wiring proof-of-concept, not built by default
```

Building the Rust workspace needs XRT headers/libs for the `amdxdna` driver
(`xrt/xrt_bo.h`, `libxrt_coreutil.so*`) and, for the ASR crates, an `onnx-asr` Python venv
with `onnx_asr` importable. `install.sh` preflights both and documents exactly what it
checks and where (`XRT_INC_DIR`/`XRT_LIB_DIR`/`ONNX_ASR_VENV`) -- read its top comment
before debugging a build failure that's actually a missing prerequisite.

If you don't have XRT or onnxruntime available, three crates build and test standalone
with no NPU and no ONNX runtime: `npu-asr-host`, `npu-gemma`, `npu-weights`. That's also
exactly the subset `.github/workflows/rust-ci.yml` runs on a hosted runner.

Building AIE kernels/xclbins is a separate toolchain path (`toolchain.lock`,
`scripts/toolchain_up.sh`) and is not needed to build or run the engine itself -- it runs
against prebuilt xclbins under `artifacts/`. See `docs/porting-amd-npu.md` if you're
touching the toolchain pin, and `docs/porting-model.md` if you're adding a kernel.

## Before submitting

For anything touching `rust/`:

```
scripts/ci_gate.sh
```

This runs `cargo clippy --workspace --all-targets`, `cargo test --workspace`, and
`cargo check -p npu-parakeet --no-default-features` (the last one guards npu-parakeet's
contract that it builds standalone, without XRT). It needs XRT and onnxruntime, so run it
on a machine that has them -- it's the real gate; the hosted CI workflow is a weaker
subset that covers only the three host-only crates.

If your change is confined to those three crates and you don't have XRT/onnxruntime:

```
cd rust
cargo clippy -p npu-asr-host -p npu-gemma -p npu-weights --all-targets
cargo test -p npu-asr-host -p npu-gemma -p npu-weights
```

For a kernel or IRON design change under `aie_kernels/`/`designs/`, run
`scripts/toolchain_smoke.sh` first (CPU-only: proves the toolchain still places and
builds an xclbin) and, where one exists, the kernel's own `verify_*.py` golden gate
(`aie_kernels/INDEX.md` lists which kernels have one and which script runs it). See
`docs/porting-model.md` section 4 for the full set of correctness gates a new op or
model should pass.

If your change touches the actual device: the NPU is single-tenant (one hardware
context at a time in practice, see `README.md`), so stop any other service holding
`/dev/accel/accel0` before running a device probe or test.

Once, after cloning, run `git config core.hooksPath hooks` -- it activates a pre-push
guard (`hooks/pre-push`) that blocks pushes carrying certain leaked material into this
public tree. It won't fire on ordinary contributions; it exists for my own workflow
across a private companion project.

## Code style

- Match the style of the file you're editing over an abstract rule. There's no
  `rustfmt.toml`/`clippy.toml` beyond `cargo`'s defaults, and comment density varies
  deliberately by directory (compare `aie_kernels/*/*.cc` to `rust/npu-engine/src/*.rs`)
  -- look at neighboring files before adding new ones, not a fixed word count.
- Prefer a doc comment that states a real constraint, a real shape contract, or the
  reason a non-obvious choice was made over one that restates what the code already
  shows.
- New AIE kernel/design files should carry an `SPDX-License-Identifier` header matching
  their neighbors (`Apache-2.0`, or the upstream license when the file is derived from a
  vendored `mlir-aie` example) -- most files under `aie_kernels/`/`designs/` already do.
  Rust source files in this repo don't carry per-file headers; don't add one.

## Licensing

Apache-2.0 (see `LICENSE`). There's no CLA and no DCO sign-off requirement -- by opening
a pull request you're contributing under the same Apache-2.0 terms the rest of the repo
is under. If your change carries third-party weights, code, or data with its own license,
say so in the PR and follow the pattern in `NOTICE` (attribution + what changed, weights
not redistributed if their license doesn't allow it).
