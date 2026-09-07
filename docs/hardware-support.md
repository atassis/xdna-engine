# Hardware support

I have built and measured this on exactly one machine. Everything below states that
plainly rather than implying broader coverage. For the deeper technical story of what a
port to different AMD NPU silicon would actually require, see
[porting-amd-npu.md](porting-amd-npu.md); this page is the shorter reference: what I
target, what I run, and what I have not touched.

## The tested device

| Property | Value | Source |
| --- | --- | --- |
| Machine | AMD Ryzen AI 9 465 (Krackan Point) | `README.md`, `docs/porting-amd-npu.md` |
| NPU generation | XDNA2 | `README.md` |
| PCI ID | `0x17f0:0x10` (also written `1022:17f0`, vendor:device) | `driver.lock`, `docs/troubleshooting.md` |
| Driver's internal name | `npu4` (`dev_npu4_info` device-family table) | `driver.lock`, `rust/npu-xrt/src/lib.rs` |
| Array shape | 4 rows x 8 columns = 32 AIE compute tiles, plus 8 MemTiles and 8 shim tiles; IRON calls this array shape `npu2` | `docs/porting-amd-npu.md`, `docs/aie2p-architecture-and-roofline.md` |
| Hardware-context slots | 16 (versus 6 on the driver's `npu1`, i.e. Phoenix/Hawk Point) | `rust/npu-xrt/src/lib.rs` |
| Context model | `AIE2_TEMPORAL_ONLY` -- contexts time-slice the whole array; the array is never spatially partitioned between contexts | `npu4_regs.c`, `aie2_ctx.c`, cited in `docs/porting-amd-npu.md` |
| Device node | `/dev/accel/accel0` | `README.md`, `docs/troubleshooting.md` |
| Kernel driver | `amdxdna` | `README.md` |

Every kernel and design in this tree is written against the 8-column `npu2` shape;
tile counts like `32` and `512` (`M-tile = m*n_aie_rows*n_aie_cols`) are compiled-in
constants, not read from the device at load time (`docs/porting-amd-npu.md` has the exact
file/line list). Nothing here has ever run against a different array shape.

## OS, kernel, and driver

Developed and gated on an Arch-derived distribution (CachyOS); `driver.lock` records the
exact kernel version the current driver pin was last validated against, and the
`amdxdna` module is installed as a DKMS package built from AMD's own packaging
(`build/arch/PKGBUILD-amdxdna-driver` in the `xdna-driver` tree, per
`scripts/driver_status.sh`). The underlying `amdxdna` driver and the open MLIR-AIE/IRON
compiler stack are not distro-specific, but the install *recipe* this repo scripts
(DKMS, `makepkg`, `pacman`) is Arch-specific; a different distribution needs the
equivalent DKMS/kmod packaging step, not a different driver.

Two things worth knowing before touching the driver:

- **DKMS can fail silently against a non-default compiler toolchain.** On a clang/LLD-built
  kernel, `dkms.conf`'s default build selects no compiler and the build dies on
  `-mllvm`/`-mstack-alignment=8` flags the kbuild passes -- while the package manager
  still reports success, leaving the *old* in-tree module loaded. `driver.lock`'s own
  note: measured on this box, a clang-built kernel failed under the distribution's DKMS
  install hook while a gcc-built kernel of the same package succeeded. `scripts/driver_status.sh`
  reports which module is actually loaded (`dkms` vs `in-tree`) and whether it matches
  the pin; run it after every kernel update, not just after a driver bump.
- **`driver.lock` and `toolchain.lock` are pinned and gated independently, on purpose.**
  The driver binds to the running kernel (a kernel update forces a driver rebuild, with
  the AIE compiler pin untouched); the toolchain binds to the AIE compiler (a Peano/MLIR-AIE
  bump must not touch the driver). Their pass/fail bars differ too: a driver swap gates
  on "does the NPU still enumerate and allocate a hardware context," a toolchain bump
  gates on kernel accuracy (rel-L2 against a golden). See `driver.lock`'s own header
  comment for the full reasoning.

## What running the engine needs (prebuilt artifacts, no AIE toolchain)

The engine runs against prebuilt xclbins in `artifacts/` and does not need the AIE
compiler for normal use (`README.md`, `docs/getting-started.md`). What `install.sh`
actually preflights:

- **Rust** -- `cargo` on `PATH`.
- **XRT headers and libs** -- `xrt/xrt_bo.h` under `XRT_INC_DIR` (default `/usr/include`)
  and `libxrt_coreutil.so*` under `XRT_LIB_DIR` (default `/usr/lib`). This is the
  userspace side of the `amdxdna` driver stack.
- **An `onnx-asr` venv** with `onnx_asr` importable, and its `libonnxruntime.so.*` --
  every ASR/diarization model's preprocessing, and everything that decodes off the NPU
  (RNNT/TDT predictor-joint networks, Whisper's default decoder, pyannote segmentation
  and embedding), runs through this.
- **Model artifacts** under `artifacts/` -- weights, ONNX graphs, and the xclbins
  themselves, produced by the export/build scripts in `scripts/`.

None of the above is generation-specific in the way the array-shape constants are; they
are ordinary build/runtime dependencies. What *is* generation-specific is baked into the
xclbins and the Rust constants that assume them (previous section).

## What building kernels from source needs (the AIE toolchain)

Only needed if you are changing a kernel or building artifacts for a model that has
none yet (`docs/getting-started.md`). `toolchain.lock` is the single source of truth for
the exact pin -- read it directly rather than trusting a version number quoted here,
since it moves. The fields it carries, and what each one is:

| Field | What it pins |
| --- | --- |
| `MLIR_AIE_FORK_COMMIT` | The `atassis/mlir-aie` fork commit (tracking `Xilinx/mlir-aie` upstream) -- the place-tiles compiler and the IRON runtime. Its device model (`NPU1`/`NPU2`, tile counts, target-chip strings) is what encodes "which AMD NPU" at the compiler level. |
| `PEANO_FORK_COMMIT` / `PEANO_DIST` | The `atassis/llvm-aie` (Peano) fork and its install seed -- the AIE-target LLVM backend. Its `aie2p` intrinsics are specific to this chip generation's ISA; a different generation needs a Peano build for that generation's backend, not a re-pin of this one. |
| `MLIR_DISTRO_WHEEL` | The prebuilt core LLVM/MLIR framework `aiecc` itself is built on (separate from Peano, which only compiles the AIE kernel code). |
| `IRON_FORK_COMMIT` | A floor commit for `amd/IRON`, resolved as a merge-base rather than an exact pin, since every IRON checkout in a working tree here carries local commits on top. |
| `NANOBIND` | The Python/C++ binding-layer version `aiecc`'s Python side needs. |

Bring-up is `scripts/toolchain_up.sh` (build or locate the instance for the current
pin); it is gated by `scripts/toolchain_smoke.sh`, a CPU-only check (the modal generator
must emit `aie.logical_tile`, the place-tiles pass must place it, and a full xclbin must
build) -- no device needed to validate the compiler itself. Kernel source in this repo's
`aie_kernels/`/`designs/` is copied into the `mlir-aie` submodule's example tree by
`scripts/sync_kernels.sh` before a build; the submodule is otherwise an unpopulated
placeholder (`.gitmodules`, `ignore = all`).

None of this toolchain machinery is vendor-locked at the API level -- it is the open
MLIR-AIE/IRON stack AMD publishes -- but the specific commits, target strings, and
device pins in `toolchain.lock` are chosen for this one chip generation.

## Format and precision, specific to this silicon (AIE2P)

From `docs/aie2p-architecture-and-roofline.md`'s own measurement of the mmul/accumulator
paths: bf16 matmul on this hardware is **emulated** (128 MAC/cycle/core, via 32-lane
FMA + shuffle), while `bfp16ebs8` (block floating point, 8 elements sharing one exponent)
gets the **true systolic** array (512 MAC/issue in the mixed bf16 x bfp16 case) -- roughly
a 4x difference in achievable throughput between two formats that look similar on paper.
This distinction, and the `bfp16ebs8` type itself, is an AIE2P property. A prior AIE2
(non-P) part would need its own measurement of whatever native matmul path it actually
has; nothing here transfers by assumption (`docs/porting-amd-npu.md`).

## What's untested, stated plainly

- **XDNA1 (Phoenix/Hawk Point, the driver's `npu1`, 4-column array).** Never built,
  never run. Every array-geometry constant in this tree is hardcoded for 8 columns and
  would need to be re-derived, not reconfigured.
- **Any AIE2 (non-P) chip.** The `bfp16ebs8` true-systolic path and its throughput
  numbers above are AIE2P-specific; no measurement exists here of an AIE2 part's actual
  matmul path.
- **Strix Halo, or any other silicon the toolchain currently buckets into the same
  8-column `npu2` shape as this Krackan box.** Believed to share the array geometry, per
  the toolchain's own device check; never independently measured. Power and thermal
  behavior are explicitly workload- and chassis-specific per
  `docs/benchmark-methodology.md`, and are not assumed to transfer even within one NPU
  generation.
- **Spatial (non-time-sliced) hardware-context partitioning.** Not exercised, since
  `npu4` does not offer it.
- **Any driver/kernel combination other than the one `driver.lock` names as last
  validated.** A kernel major-version bump warrants a re-gate even if `driver.lock`'s
  commit pin does not change, since the driver tracks internal kernel APIs that are not
  stable across kernel releases.

If you are trying this on different AMD NPU silicon, treat every one of the items above
as a re-measurement, not a re-configuration -- re-derive the array-geometry constants
from the target's actual tile count, re-pin `toolchain.lock` to a Peano/MLIR-AIE build
targeting that generation's backend, re-pin `driver.lock` to a driver that recognizes its
PCI ID, and re-run the gates in [verification.md](verification.md) before trusting any
number that comes out. `docs/porting-amd-npu.md` has the concrete, file-by-file list of
what would need to change for a Phoenix/Hawk Point (XDNA1) port specifically.
