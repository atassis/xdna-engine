# PR-DRAFT (for owner review -- NOT filed): adf-free cascade accessor in aie_api

Status: prepped, verified against source. **Owner-gated: nothing filed.** Decide target repo + voice, then I finalize.

## One-line

`aie_api` exposes the AIE core-to-core cascade only through the ADF graph API, which `#include <adf.h>`; that
header is absent in bare-metal / IRON / mlir-aie flows, so no straight-line `.cc` kernel can issue a cascade
put/get -- even though the hardware path is already exposed, adf-free, by the Peano `mcd_write`/`scd_read`
builtins. Add a thin adf-free `aie::cascade_out()/cascade_in()` wrapper over those builtins.

## The gap (verified, file:line)

- `mlir-aie/third_party/aie_api/include/aie_api/adf/stream.hpp:10` -> `#include <adf.h>` (same in `adf/window.hpp:10`, `adf/io_buffer.hpp:10`). These are the ONLY cascade-bearing headers in aie_api, all under `adf/`, all reached via `aie_api/aie_adf.hpp`.
- `adf.h` is **not vendored anywhere** in the toolchain (`find . -name adf.h` = empty).
- Net: any kernel that includes the aie_api cascade path fails to compile; IRON is then forced to lower cross-core K-reduction as `npu_cascade` channel buffer-copies + software-add instead of a native cascade-accumulator FIFO.

## Evidence (3 compile checks, llvm-aie clang acc2a72c, --target=aie2p-none-unknown-elf)

1. **Gap reproduced** -- `gap_adf_cascade.cc` (includes `aie_api/aie_adf.hpp`) -> `fatal error: 'adf.h' file not found`.
2. **Adf-free path exists in the ISA** -- `BuiltinsAIE2P.def` declares `__builtin_aie2p_mcd_write_{acc32,vec}` (put) and `__builtin_aie2p_scd_read_{acc32,vec}` / `scd_ACC2048` / `scd_expand_*` (get). 14 cascade builtins, no adf dependency.
3. **Fix compiles clean** -- `fix_adf_free_cascade.cc` and `test_wrapper.cc` (using the candidate `aie::cascade_out`/`cascade_in_i32`) both build to a valid AIE2P object with rc=0. Cascade is reachable adf-free; aie_api just doesn't wrap it.

## Proposed fix

`aie_cascade_bare.hpp` (this dir): a header in `namespace aie` wrapping the `mcd_write`/`scd_read` builtins,
no `<adf.h>`. The int32-vector accessors are compile-verified. The **acc32 accessors are the ones fused
K-reduction wants** (partial sums travel as accumulators); the only open detail is the `aie::accum<acc32,16>`
<-> native `v16acc32` conversion at the builtin boundary -- to finalize on review.

## Why it matters

Unblocks the **fused cascade-accumulator** -- the brick catalog's ranked #3 lever (~5-10x on multi-core
K-reduction for GEMM/FFN). It converts the cascade from "dumb buffer-transport" (what we ship today) into the
native accumulator FIFO the hardware provides. Surfaced by a real kernel (`cascade-kreduce` brick, wave-1 fan-out).

## Test plan

A compile-only lit test in the aie_api / mlir-aie test suite: a kernel that includes the adf-free header and
issues `cascade_out`/`cascade_in`, `RUN: %clang ... --target=aie2p ... -c`, `expected: rc==0`. (No device
needed; this is a toolchain-exposure fix. Numerical behavior is covered separately by the on-NPU
`brick-wave1-device-verify` cascade-kreduce entry.)

## BASE IS STALE -- re-verify before filing

Our `aie_api` submodule is pinned at `2a40805` (2025-02-19, ~5 months old) with remote
`github.com/jgmelber/aie_api` (a personal fork -- and, notably, what `Xilinx/mlir-aie`'s own `.gitmodules`
vendors from). Canonical `Xilinx/aie_api` HEAD is `bec000fd`. **Before filing:** re-verify the adf.h gap on
canonical-latest (5 months of churn may have restructured the cascade path), and rebase the diff there
([[validate-upstream-pr-against-matching-base]], [[prefer-latest-over-stale-toolchain]]). The gap + evidence
in this dir are verified only against our stale pin so far.

## Open decisions for the owner

1. **Target repo.** `Xilinx/aie_api` (canonical, `bec000fd`) vs `jgmelber/aie_api` (what mlir-aie tracks) vs a
   fork-carried patch vs an IRON-side helper. You know the upstream relationships; this is your call.
2. **Scope -- RECOMMENDED: typed free-function accessors, matching the idiom.** aie_api exposes primitives as
   TYPED FREE-FUNCTION TEMPLATES (`load_v`/`store_v`/`broadcast`), and the ADF cascade dispatches by size with
   `if constexpr`. So the convention-matching design is `aie::cascade_out<T>(const T&)` / `aie::cascade_in<T>()`
   templated on the `accum`/`vector` type -> right builtin. NOT a new `cascade<T>` object type (over-building),
   NOT raw untyped builtins (under-idiomatic). Position it explicitly as the NON-ADF path (does not compete
   with the existing ADF cascade). Float the API shape in a short issue/RFC PR before the full drop.
3. Voice/base per [[upstream-pr-hygiene]]: ASCII owner voice, base off canonical upstream/main, validate
   against a matching base, confirm-before-file.

## Artifacts in this dir
`gap_adf_cascade.cc` (repro) - `fix_adf_free_cascade.cc` (adf-free proof) - `aie_cascade_bare.hpp` (candidate fix) - `test_wrapper.cc` (wrapper compile test).
