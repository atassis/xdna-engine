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

## Future work (PR-body text -- goes in the description; NOT a commitment)

> This PR is deliberately the minimal, non-ADF accessor for the raw cascade put/get -- enough for a
> bare-metal / IRON kernel to hand-write a cross-core reduction, which is not possible through aie_api today.
> It stands on its own: it exposes a capability that is simply missing outside the ADF flow.
>
> A natural follow-on -- which I'd be glad to raise as a separate RFC if there's interest -- is folding the
> cascade into the `mmul` / `accum` path so multi-core K-reduction fuses natively (today it lowers to
> buffer-copy + software-add). The hand-written reduction this accessor enables is exactly the instance that
> would inform that larger design, so it makes sense to land this first and discuss the fused form separately.
> Happy to align on the API shape (typed free functions vs a typed port type) before expanding.

Tone rules for this footer (per [[upstream-pr-hygiene]]): PR 1 is complete-and-valuable on its own; the
follow-on is an ENHANCEMENT, never a correction of PR 1; opt-in ("if there's interest"), not a roadmap
dictated to the maintainers; grounded in the K-reduction use case; invites their design ownership. Do NOT
over-promise or make merge of PR 1 contingent on PR 2.

## CANONICAL RE-VERIFY (done 2026-07-19) -- the fix is DESIGN-LEVEL, RFC-first

Re-verified against a fresh clone of canonical `Xilinx/aie_api` `bec000f` ("Sync to latest 2026.1",
2026-05-13): **gap CONFIRMED** -- `adf/stream.hpp:12` still `#include <adf.h>` (moved from :10, minor reorg),
`adf.h` still unvendored, no adf-free cascade primitive added upstream.

**But canonical code shows the mergeable fix is bigger than a wrapper:**
- aie_api's cascade is ADF-NATIVE throughout -- the only surface is `readincr_v<N>(input_cascade)` /
  `writeincr(output_cascade,...)` over ADF stream TYPES (adf.hpp: `accum<acc48,8> = readincr_v<8>(input_cascade)`).
  No framework-agnostic cascade primitive exists to extend.
- aie_api NEVER calls `__builtin_aie2p_*` (0 in detail/). Intrinsics are feature-macro-guarded
  (`__AIE_API_HAS_*`), multi-COMPILER (chess + peano) and multi-ARCH (aie2/2p/2ps). Our PoC calls the raw
  Peano builtin -> compiles, but is NOT idiomatic and would fail under chess / other arches.
- => an adf-free cascade primitive is a NEW intrinsic-access path across their compiler x arch matrix =
  maintainer-design territory, not a droppable diff.

**Split the deliverable:**
1. **Canonical `Xilinx/aie_api` -> RFC/issue FIRST** (owner target choice, confirmed). Document the gap +
   show the adf-free path is reachable (PoC = evidence) + ask how they'd structure an adf-free cascade
   primitive across chess/peano/arch. NOT a code PR until they weigh in. (Note: `Xilinx/aie_api` reads as a
   periodic release-sync MIRROR, so the channel may be issue + internal port, not a direct merge -- confirm.)
2. **Our flow -> fork-carried patch** in `mlir-aie/third_party/aie_api` (aie2p/peano only), the builtin
   wrapper here, to unblock K-reduction now -- explicitly separate from the upstream RFC.

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
