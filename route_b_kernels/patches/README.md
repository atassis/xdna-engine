# Vendored toolchain patches (pinned + tethered)

Out-of-tree patches against the external AMD toolchains we build on, carried here as the durable
"pinned + tethered patch" artifact (upstream PRs are deferred until the service is finished — owner call).
Each patch names the upstream repo + the pinned commit it applies against.

| Patch | Upstream repo | Pinned at | What it does |
|---|---|---|---|
| `mlir-aie-cachyos.patch` | `mlir-aie` submodule | (submodule SHA) | bf16 BFP16 fast-path microkernel + CachyOS build fixes |
| `iron-transpose-num-batches.patch` | `~/repositories/ns/amd/IRON` | `5503a95` | Adds `num_batches` to the IRON `Transpose` operator (head-batches B contiguous (M,N) transposes into ONE dispatch, mirroring GEMV's batching). Backward-compatible: `num_batches=1` (the default) is behaviorally identical, so existing callers are unaffected. |
| `iron-gemm-fusion-prefix.patch` | `~/repositories/ns/amd/IRON` | `devel` | Makes the IRON `GEMM` operator fusable under `FusedMLIROperator`: applies the injected `func_prefix` to BOTH the kernel-object filename and the kernel **symbol** names (`zero_*`, `matmul_*`, `convert_copy_*`) in `gemm/design.py`, matching the fusion layer's per-op `op{idx}_` filename + `prefix_symbols` rename. GEMV already did this; GEMM never did because deep-C never fused a GEMM. Backward-compatible: with `func_prefix=""` (non-fused) the strings are unchanged. |

## `iron-transpose-num-batches.patch` — apply

```bash
cd ~/repositories/ns/amd/IRON          # pinned at 5503a95
git apply /path/to/route_b_kernels/patches/iron-transpose-num-batches.patch
```

**Why:** lever #3 ([[lever3-dispatch-overhead-bound]]) — the fused Whisper decode dispatch is
launch-overhead-bound, and the per-head V transposes were ~47% of the ~612 micro-launches/token because
`Transpose` (unlike `GEMV`/`StridedCopy`) couldn't batch over heads, forcing a Python unroll over H=12.
With `num_batches=H` the 12 self-attn transposes collapse to 1 launch/layer. Touches
`iron/operators/transpose/{op.py,design.py}` only; the `.cc` kernel (transposes s×s sub-tiles) is unchanged.

**Validation status:** **builds clean on the NEW mlir-aie 1.3.2 stack** (12-layer coalesced resident ELF
via `scripts/build_deepc_decode.sh`; scratch 310→281.7 MB = the 12× `vcTc` cross-transpose buffers removed,
confirming the coalescing took effect). On-device numerical (rel-L2 via `verify_fused_decode_sp.py`) + WER
(gate 0.1172) + dispatch-timing A/B is PENDING the single-tenant NPU (see [[lever3-dispatch-overhead-bound]]
"Next" + `scripts/lever3_coalesce_ab.sh`).

**Stack note (2026-06-16, post-deep-C):** deep-C landed the migration to the new vendored mlir-aie 1.3.2 +
constant resident ELF ([[deepc-resident-scratchpad-result]]). This `num_batches` change **ported forward
cleanly**: it is orthogonal to `amd-IRON-deepc.patch` (which doesn't touch `transpose`), so both apply to the
same `amd/IRON` checkout. `build_deepc_decode.sh` applies the deep-C patch; this transpose patch is applied
on top (the lever-3 A/B script does so idempotently). The old `ironenv` 0.0.1 is retired. **Durable carry:**
fold this into the deep-C IRON patch set when convenient (owner call); for now it is a separate tethered patch.

## `iron-gemm-fusion-prefix.patch` — apply

```bash
cd ~/repositories/ns/amd/IRON
git apply /path/to/route_b_kernels/patches/iron-gemm-fusion-prefix.patch
```

**Why:** lever #3 vector-(b) batching needs the IRON `GEMM` op (skinny-N `out[M,N]=W[M,K]@X[K,N]`) inside a
fused full ELF. On the new stack, fusing a GEMM failed two ways: (1) the `link_with` kernel-object name was
left unprefixed while fusion renamed the `.o` to `op0_…` → "could not copy gemm_…o"; (2) the matmul/zero
kernel **symbols** were unprefixed while `prefix_symbols="op0_"` renamed them in the object → `undefined symbol:
matmul_bf16_bf16`. The fix prepends `func_prefix` in both places (GEMV's `op.py`/`design.py` already do this).
Orthogonal to the deep-C + transpose patches; all three apply to the same checkout. `scripts/build_gemm_probe.sh`
applies it idempotently. **Validation:** the N∈{16,32,64,128} fc1-GEMM probe ELFs build clean (compile-only);
on-device correctness (rel-L2 ≤ 0.08 via `fused_elf_probe`) + the dispatch-amortisation sweep are PENDING the
single-tenant NPU (Milestone 0 of `internal notes`).
