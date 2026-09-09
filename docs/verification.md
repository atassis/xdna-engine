# Verification

No single gate in this tree proves "this model is correct on the NPU." Each layer below
proves a narrower thing than that, and I list what each one does and does not prove so a
claim can be traced to the check that actually backs it, rather than to a green run of
something adjacent. They stack, in roughly the order I would run them.

## 1. Software correctness -- does the code even work

`scripts/ci_gate.sh` is the real gate for anything touching `rust/`: `cargo clippy
--workspace --all-targets`, `cargo test --workspace --no-fail-fast`, and a standalone
`cargo check -p npu-parakeet --no-default-features` (the decoupling contract -- that
crate must build without XRT). It needs XRT and `onnxruntime` on the box, so it runs
locally, not on a hosted runner.

What it does not prove: a fixture-guarded test (the pattern is `eprintln!("SKIP: ...")` +
an early `return` when an artifact is missing) reports `ok` to `cargo test` even though it
executed nothing. `ci_gate.sh` runs with `--nocapture` and greps the log for `^SKIP` lines
specifically so a passing-by-skipping test doesn't read as a real pass -- but only if you
read that part of the output.

**Public CI** (`.github/workflows/rust-ci.yml`) builds, lints, and tests exactly three
crates on a hosted runner -- `npu-asr-host`, `npu-gemma`, `npu-weights` -- because a
hosted runner has neither XRT nor `onnxruntime` and cannot link anything else in the
workspace. The workflow's own comment calls this "deliberately the WEAKER half of the
gate." It runs 73 of the repo's workspace tests, none of which touch real NPU hardware.

## 2. Weight-checkpoint parity -- is the conversion correct

`rust/npu-weights/tests/parity_<arch>.rs`, one per registered architecture (`bert`, `esm`,
`whisper`, `gigaam`, `parakeet`, `opt`, `vit`, `dinov2`, `resnet`, `clip`, `modernbert`,
`minilm`, `espcn`, `edsr`). Each bakes a real HF/ONNX checkpoint through the Rust `Arch`
transform and diffs every output tensor against a Python-oracle `.npy`. The threshold,
enforced identically by the `npu weights verify` CLI command
(`rust/npu-cli/src/main.rs`) and by the shared test helper
(`rust/npu-weights/tests/common/mod.rs`) so the tool and the tests can't drift apart:

```
max rel-err < 5e-2   (checkpoint::verify_against_npy, rust/npu-weights/src/checkpoint.rs)
```

This is a no-NPU, no-XRT test -- it is why the public CI job above can cover it.

What it proves: the baked bf16 checkpoint's tensors match the source model's, within the
bf16 quantization floor. What it does not prove: anything about inference. Several
architectures in [model-support.md](model-support.md) -- MiniLM, ModernBERT, DINOv2, CLIP --
have *only* this gate; no forward pass, host or NPU, exists for them anywhere in this
tree, so passing this test is the entire correctness story for those checkpoints today.

## 3. Per-kernel goldens -- is one kernel's math right at one shape

Each kernel under `aie_kernels/<name>/` can carry a `golden.py`: a NumPy/PyTorch
reference for that kernel's exact op and shape, meant to be diffed against the on-chip
output by a `verify_*.py` script. `aie_kernels/INDEX.md` tracks this honestly, per
kernel: 31 of the 48 kernel directories have a `golden.py`, 17 do not (built and verified
only through the multi-core designs that consume them, never independently). Device
status is tracked separately from the golden's existence, and it is mostly absent: 44 of
the 48 rows read "unverified in this index" -- meaning no PASS, FAIL, or device-run
record could be found in the tree for that kernel in isolation, not that it is known
broken or known working. Where the index found a real device record, it says what kind
(a golden-vs-kernel gate, or only a rounding-mode A/B) rather than folding a weaker
result into a bare PASS.

The correctness bar, when a golden-vs-device gate does run
(`docs/execution-graph.md`): rel-L2 <= 0.08 (standard), < 0.02 (the tighter bf16-GEMM
bar), or corr >= 0.99 (the cascade-accumulator path, which is correlation-gated rather
than rel-L2-gated for its own numerical reasons).

## 4. Device numeric parity -- does the model's NPU path match a reference

This is the layer that actually dispatches a real forward pass to the NPU and compares
it against a CPU/ONNX/PyTorch oracle, at the rel-L2 bar from section 3. It lives in
`rust/npu-probes` (`verify_encoder.rs`, `verify_embeddings.rs`, `verify_esm.rs`,
`verify_parakeet.rs`, `verify_whisper.rs`, `verify_patch_embed.rs --npu`, and others) and
in `rust/npu-sr/tests/npu_gate.rs` / `edsr_npu_gate.rs` for the super-resolution nets.
`scripts/verify_bge_parity.py` is the equivalent for BGE embeddings, checked against a
mean-pooled, L2-normalized HF f32 reference.

This layer needs the physical device, plus whatever artifacts (xclbins, checkpoints,
fixtures) that model's gate names -- most of these scripts and binaries skip cleanly
(printing why) rather than failing when a prerequisite is missing, which is the same
"SKIP reads as pass" caveat as section 1. It does not run in CI. It is also the layer
that is simply absent for every model in [model-support.md](model-support.md) marked
Host-only, since there is no NPU dispatch to check.

## 5. Encoder-level regression gate -- catching what a mean hides

`scripts/encoder_parity.py` exists because a whole-clip mean rel-L2 can average a short,
localized error burst into invisibility: its own measurement found the shipped Parakeet
encoder carrying contiguous 1-5-frame runs at 15-79% per-frame relative error on 16 of 17
clips, while the whole-clip mean sat at 0.089. It gates a candidate change on four
statistics against a shipped baseline -- mean, worst-frame, worst-burst (max mean error
over a sliding window), and `new-burst` (how much error the worst-hurt frame *gained*
versus baseline, an absolute delta rather than a relative one, specifically to catch a
change that turns a clean frame bad without moving the relative ranking). A change passes
only if all four do.

## 6. End-to-end task metric -- WER

`scripts/wer_eval.py` and its per-model variants (`parakeet_wer_eval.py`,
`whisper_decode_wer.py`, `whisper_npu_wer.py`, `int8_wer_eval.py`, ...) hit a live
`npu serve` instance over HTTP, transcribe a clip set, and report WER against ground
truth and against a CPU oracle. The clip set (`scripts/fetch_wer_clips.py`) is small by
construction -- about 13 Russian + 4 English FLEURS clips, 17 total.

The measured limitation, from `scripts/encoder_parity.py`'s own docstring: perturbing the
shipped encoder by a meaningless +/-1e-5 per-element noise swings greedy-decode WER over
this 17-clip set across an ~8.2-9.2 band. A WER delta smaller than that swing proves
nothing about whether a change is correct. Treat a WER number over this clip count as a
coarse smoke signal, not a tight regression gate -- section 5 exists specifically to
give the encoder path a gate that does not have this problem.

## 7. Determinism gates -- the actual bar for a decode-path change

For a change to an already-shipped autoregressive decode path, the project's own
standing rule (stated directly in `scripts/lever3_determinism_gate.sh`) is that neither
rel-L2 nor the 17-clip WER is the correctness gate -- it is **1:1 determinism against the
CPU reference at temperature 0**. Decode is greedy/argmax, so "temperature 0" is
structural: two arms agree iff every argmax matches, checked by exact byte equality of
the emitted text (no normalization, since normalization is exactly what would hide a
token flip). The script runs three arms over the full 17-clip bank -- the ONNX/CPU
reference, the shipped resident decode, and a candidate -- with two repetitions per NPU
arm to also check run-to-run determinism, and requires: candidate == shipped on every
clip, each NPU arm equal to itself across reps, and each NPU arm byte-equal to the ONNX
reference.

For a change that is supposed to alter nothing about decode at all -- instrumentation,
plumbing, a refactor -- the same bar applies with the shipped binary as its own reference:
run the candidate and the pre-change build over one prompt at a fixed seed and temperature
0 and require byte-identical output, plus repetitions of each to separate a real
difference from run-to-run drift. A JSONL run log (`npu generate --stats-log`, see
[measurement.md](measurement.md)) makes that comparison token-level rather than
text-level: `npu stats --diff` aligns two runs on `seq`, reports the first divergent
token id before it prints any timing, and refuses to read a speed difference as a speedup
when the two runs did not compute the same thing or ran under different power modes.

## What none of this proves

- **Cross-hardware generalization.** Every gate above ran on the one Krackan/XDNA2 box
  named in [hardware-support.md](hardware-support.md). Nothing has been measured on a
  second device.
- **Sustained / thermal behavior.** These are correctness gates, not the benchmark
  suite; see [benchmark-methodology.md](benchmark-methodology.md) for why a burst
  measurement can look better than steady-state.
- **Anything about a Host-only model in [model-support.md](model-support.md)** beyond
  its weight-conversion parity or its host-CPU reference, since there is no NPU path for
  any of these gates to exercise.

## Reproducing this yourself

- Software gate: `scripts/ci_gate.sh` (needs XRT + `onnxruntime` on the box).
- Weight-conversion parity for one architecture, no NPU needed:
  `cargo test -p npu-weights --test parity_<arch>` (falls back to `SKIP` and prints the
  export command if the oracle fixture is missing -- read the printed line, not just the
  exit code).
- Device parity for a model that has one: build `npu-probes`
  (`cargo build -p npu-probes --release`) and run its `verify_*` binary against real
  artifacts, or `cargo test -p npu-sr` for the super-resolution gates. The NPU is
  single-tenant -- stop `xdna-engine.service` and confirm `/dev/accel/accel0` is free
  (`fuser /dev/accel/accel0`) first, the way `scripts/lever3_determinism_gate.sh` does.
- End-to-end WER: start the service (`npu serve` or the systemd unit), then
  `scripts/wer_eval.py --clips artifacts/wer_clips` (or `--no-service` to smoke-test just
  the CPU oracle path, no NPU needed).
- Determinism gate for a decode-path change: `scripts/lever3_determinism_gate.sh`,
  adjusted for the artifacts you are comparing.
