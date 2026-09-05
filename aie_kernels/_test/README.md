# aie_kernels/_test

Device-verify harnesses for the kernels in `aie_kernels/<op>/`. Two kinds of script live
here on purpose, and only one of them is the suite:

- **`verify_*.py` (40 files) are the gates.** Each builds one or more bricks' kernels
  against the pinned toolchain, runs them on aie2p, and checks rel-L2 against a numpy
  golden. Run these to know whether a brick is correct on device. Four of them are
  family harnesses covering several bricks in one device session:
  - `verify_norm_elementwise_f32.py` -- rmsnorm, layernorm, qk-norm, relu2, geglu, swiglu
  - `verify_gemm_int8.py` -- gemm-int8, gemv-int8
  - `verify_gemm_int8xint4.py` -- gemm-int8xint4, gemv-int8xint4, gemm-int8xint4-dequant
  - `verify_specials.py` -- transpose-dma (the other specials named in its docstring are
    parked, not covered)

  The rest are one brick each; grep `aie_kernels/_test` for a kernel's directory name to
  find its gate. `drain_device.py` / `drain_mods.py` are the runners that import these
  modules and execute every `do_*()` device gate they expose (a module with no `do_*()`
  is script-style and must run as a subprocess -- see `drain_device.py`'s own docstring
  for why conflating the two under-covered the suite before).

  **The `verify_` prefix is not a guarantee.** `drain_device.py` names four
  `verify_*.py` files that are bisect probes wearing the gate prefix -- they print a
  number and exit 0 by design, so an exit-status tally would score them as passes that
  mean nothing: `verify_dequant_f32.py`, `verify_int4_shapes.py`, `verify_int4_stride.py`,
  `verify_rounding_ab.py`. `verify_conv_1d_realdata.py` is excluded from the drain for an
  unrelated reason (it takes a captured stage dump as an argv arg, and `run.sh` forwards
  none). Trust `drain_device.py`'s TARGETS/SCRIPT_TARGETS lists over the filename.

- **`probe_*.py` (89 files) are one-off investigations, kept for their findings, not
  gates.** Bisects, A/Bs, rail comparisons, minimal repros of toolchain defects. Do not
  expect them to pass or to mean anything as a pass/fail signal -- several print numbers
  and exit 0 by design. They are kept because the KB cites their measurements, not
  because they verify anything today.

They deliberately share this one directory rather than living apart: 62 probes `import
bricklib` directly (same-directory import, no package/path setup), and some probes import
a `verify_*` module outright for its helpers or its golden wiring -- e.g. `probe_rope.py`
opens with `import verify_rope_lut as m`. Splitting probes out would break both.

## Other files

- **`bricklib.py`** -- the shared build/dispatch rail every `verify_*`/`probe_*` script
  uses: compiles a brick's `.cc` for aie2p, wires an `@iron.jit` design that streams rows
  through it, and gates rel-L2 vs the numpy golden. Read its own docstring before adding a
  new verify script.
- **`run.sh`** -- the entry point: `./run.sh verify_xxx.py`. It takes the AIE **toolchain
  instance from THIS worktree's `toolchain.lock`** (the instance is content-addressed on
  that file, so borrowing a sibling worktree's instance would silently run against the
  sibling's pin) but **borrows a `.venv-iron` from wherever one exists** (this worktree
  does not have its own; the venv carries `ml_dtypes` and the Peano install, and isn't
  worktree-specific the way the toolchain pin is). Override the venv with `BRICK_VENV=` if
  needed. Serializes on the shared NPU lock when `NPU_LOCK_SH` is set.
- **`compile_check.sh`** -- CPU-only compile check for a brick `.cc` (no device). A clean
  compile is necessary, not sufficient (says nothing about spill/aliasing codegen), but it
  catches most authoring mistakes without touching the single-tenant NPU.
- **`gen/`** -- scratch build artifacts (xclbins, generated shims); not source.
