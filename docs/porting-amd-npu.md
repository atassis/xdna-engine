# Porting to a different AMD NPU

This engine has been built and measured on exactly one device: an AMD Ryzen AI 9 465
(Krackan Point), XDNA2, `amdxdna` driver, `/dev/accel/accel0`. This is what is specific to
that target, where the assumption is written down in code rather than just implied, and
what is genuinely untested elsewhere.

## What this device actually is, to the driver and the toolchain

The `amdxdna` driver identifies this silicon by PCI ID `0x17f0:0x10`, which resolves to
its `dev_npu4_info` device-family table (`driver.lock`, `rust/npu-xrt/src/lib.rs`) --
"npu4" is the driver's own name for this generation, not a name this project invented.
Two behaviors follow directly from being npu4 rather than an earlier generation:

- `.hwctx_limit` is 16 hardware-context slots on npu4, versus 6 on what the driver calls
  npu1 (`rust/npu-xrt/src/lib.rs`, the comment on `open`'s context-cache accounting).
- npu4 sets `AIE2_TEMPORAL_ONLY` (`npu4_regs.c`), so hardware contexts do not spatially
  partition the compute array -- `aie2_ctx.c` ignores a context's requested column list
  and always requests every column, and contexts instead time-slice the whole array.
  A generation without that flag could partition space instead of time; nothing in this
  engine's device-actor model (`rust/npu-runtime`, one thread serializing all NPU work)
  assumes or exploits partitioning either way, but it has only ever run under
  time-slicing.

On the open-toolchain side, IRON's own device model recognizes two array shapes
(`designs/whole_array_fused/whole_array_iron.py`):

```python
# npu is a 4 row x 4 col array
if dev == "npu" and n_aie_cols > 4:
    raise AssertionError("Invalid configuration: NPU (Phoenix/Hawk) has 4 columns")
# npu2 is a 4 row x 8 col array
if dev == "npu2" and n_aie_cols > 8:
    raise AssertionError("Invalid configuration: NPU2 (Strix/Strix Halo/Krackan) has 8 columns")
```

Every design and kernel in this tree targets `npu2` -- 4 rows x 8 columns, 32 compute
tiles, plus 8 MemTiles and 8 shim tiles (`docs/aie2p-architecture-and-roofline.md`'s
hardware-ground-truth section). That 32 is not read from the device at load time; it is
compiled in. `rust/npu-asr/src/ctx2.rs` has `const N_AIE_CORES: usize = 32; // 4 rows x
8 cols -- one rtp[0] write packet each`, and `rust/npu-asr/src/conv_npu.rs` has
`const MT: usize = 512; // M-tile = m*n_aie_rows*n_aie_cols = 16*4*8 (kernel-fixed)`. Every
IRON generator under `designs/` that builds a whole-array kernel picks `n_aie_cols` (and
asserts it against the array-shape check above) rather than discovering it.

**What a port to `npu` (Phoenix/Hawk Point, 4 columns, XDNA1) would need to change,
concretely:** every hardcoded `8`/`32` tile-count constant across `rust/npu-asr`,
`rust/npu-parakeet`, and the `designs/*/`.py generators; the `dev`/`--device` argument
each IRON generator already accepts (several designs, e.g. `designs/ctx_ln/*.py`, already
parameterize `NPU1()` vs `NPU2()` from `aie.iron.device`); and the per-generation
microkernel MAC tile dims IRON keys by device (`microkernel_mac_dim_map` in
`whole_array_iron.py`: `npu`'s bf16 tile is `(4,8,4)`, `npu2`'s is `(4,8,8)` or `(8,8,8)`
depending on whether bf16-via-bfp16 emulation is on). None of this has been tried here --
say so plainly rather than guessing it would "just work" with a config flip. The one
honest data point in the other direction: the IRON device check groups Strix, Strix Halo
and Krackan together as the same 8-column `npu2` shape, which is what this codebase
currently believes about that family -- untested beyond the one Krackan box it runs on.

## Format and precision are AIE2P-specific

`docs/aie2p-architecture-and-roofline.md`'s format table: bf16 matmul on this hardware is
**emulated** (128 MAC/cycle/core, via 32-lane FMA + shuffle), while `bfp16ebs8` (block
floating point, 8 elements sharing one exponent) gets the **true systolic** array at
512 MAC/cycle/core -- a roughly 4x difference in achievable throughput between two
formats that look similar on paper. That distinction, and the `bfp16ebs8` type itself,
is a property of the AIE2P generation this chip implements. A prior AIE2 (non-P) part
would need its own accuracy and throughput measurement of whatever native matmul path it
actually has -- nothing here transfers by assumption. The `-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16`
build flag and `emulate_bf16_mmul_with_bfp16` generator argument that select between the
two bf16 paths are also part of this AIE2P-specific surface.

## Toolchain pin

`toolchain.lock` is the single source of truth for the exact compiler stack a build uses,
and every field in it is generation-specific in some way:

- `MLIR_AIE_FORK_COMMIT` -- the `atassis/mlir-aie` fork commit, itself tracking
  `Xilinx/mlir-aie` upstream. This is the place-tiles compiler and the IRON runtime;
  its device model (`NPU1`/`NPU2`, tile counts, target chip strings) is what encodes
  "which AMD NPU" at the compiler level.
- `PEANO_FORK_COMMIT` / `PEANO_DIST` -- the `atassis/llvm-aie` (Peano) fork and its
  install seed. Peano is the AIE-target LLVM backend; its `aie2p` intrinsics
  (`aie2pintrin.h`, the `aie2p_aie_api_compat.h` header providing block-FP type stubs)
  are specific to this chip generation's ISA. A different-generation target needs a
  Peano build for that generation's backend, not a re-pin of this one.
- `MLIR_DISTRO_WHEEL` -- the prebuilt core LLVM/MLIR framework aiecc itself is built on
  (separate from Peano, which only compiles the AIE kernel code).
- `IRON_FORK_COMMIT` -- a floor commit for `amd/IRON`, resolved as a merge-base rather
  than an exact pin (the file's own comment explains why: every local IRON checkout in
  this workspace carries commits on top).
- `NANOBIND` -- the Python/C++ binding layer version aiecc's Python side needs.

Bring-up is `scripts/toolchain_up.sh` (build or locate the instance for the current pin)
gated by `scripts/toolchain_smoke.sh` (a CPU-only check: the modal generator must emit
`aie.logical_tile`, the place-tiles pass must place it, and a full xclbin must build --
no device needed). Kernel source lives in this repo's `aie_kernels/`/`designs/` and is
copied into the `mlir-aie` submodule's example tree by `scripts/sync_kernels.sh` before a
build; the submodule itself is pinned separately (`.gitmodules`, `ignore = all` -- it is
not populated by a normal clone, `scripts/toolchain_up.sh` resolves the toolchain that
compiles against it).

None of this toolchain machinery is AMD-vendor-locked at the API level -- it's the open
MLIR-AIE / IRON stack AMD publishes -- but the *pins* (which commit, which target, which
device string) are chosen for this one chip.

## Driver expectations

`driver.lock` pins the exact `amd/xdna-driver` commit this engine has been validated
against, kept deliberately separate from `toolchain.lock`: the driver binds to the
running Linux kernel, not to the AIE compiler, so a kernel update forces a driver rebuild
with the toolchain pin untouched, and the two have different gates (driver: does the NPU
still enumerate and allocate a hardware context; toolchain: does a kernel still hit its
accuracy bar). A different NPU generation likely needs a different minimum driver version
for its own PCI ID / device-family table entry -- this repo has only ever run against the
one pinned here, against `dev_npu4_info`.

## What's untested, stated plainly

- **XDNA1 (Phoenix/Hawk Point, the driver's "npu1"/IRON's `NPU1()`, 4-column array):**
  never built, never run. The array-geometry constants are hardcoded for 8 columns
  throughout, and would need to be re-derived, not reconfigured, per the concrete list
  above.
- **Any AIE2 (non-P) chip:** the `bfp16ebs8` true-systolic path and its 512 MAC/cycle
  peak are AIE2P properties; nothing here has verified what an AIE2 part's native matmul
  path actually supports.
- **Strix Halo, or any other member the toolchain currently buckets into the same
  8-column `npu2` shape as this Krackan box:** believed to share the array geometry per
  the toolchain's own device check, never independently measured on that silicon. Power
  and thermal behavior are explicitly called out elsewhere in this repo
  (`docs/benchmark-methodology.md`) as workload- and chassis-specific and not assumed to
  transfer even within the same NPU generation.
- **Spatial (non-time-sliced) hardware-context partitioning:** not exercised, since npu4
  does not offer it.

If you're porting this to different AMD NPU silicon, treat every one of the above as a
re-measurement, not a re-configuration: re-derive the array-geometry constants from the
target's actual tile count, re-pin `toolchain.lock` to a Peano/mlir-aie build that targets
that generation's backend, re-pin `driver.lock` to a driver version that recognizes its
PCI ID, and re-run every gate in `docs/porting-model.md` section 4 before trusting any
number that comes out.
