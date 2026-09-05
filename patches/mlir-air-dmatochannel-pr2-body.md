# [AIRDmaToChannel] Fix iterator invalidation when hoisting broadcast DMAs out of deeply-nested herds

## Problem

`aircc` SIGSEGVs while compiling the in-tree example
`programming_examples/matrix_vector_multiplication/bf16_cascade/matvec_cascade.py`
(at both the default `M=2048` and smaller sizes):

```
make -C programming_examples/matrix_vector_multiplication/bf16_cascade run \
  M=2048 K=8192 TILE_M=2 M_INPUT=1 HERD_COLS=8 N_CASCADE=4
```

The crash is in `DmaToChannelPass`, inside
`AIRHoistExternalAIRChannelPattern<HerdOp>::matchAndRewrite`.

## Root cause

When hoisting the external half of a data movement out of a herd, the pass
gathers a backward slice of the channel ops, then augments that slice with the
constant operands referenced by the slice's region-bearing ops:

```cpp
for (auto o : backwardSlice) {
  for (auto &region : o->getRegions()) {
    visitUsedValuesDefinedAbove(region, [&backwardSlice](OpOperand *use) {
      if (getConstantIntValue(use->get()))
        backwardSlice.insert(use->get().getDefiningOp());   // mutate while iterating
    });
  }
}
```

`backwardSlice` is a `SetVector`, and the lambda inserts into it while the
`for (auto o : backwardSlice)` range-for is iterating it. When the herd body
nests the hoisted channel under enough `affine.if`/`scf.if` regions (e.g. a
cascade GEMV, where the broadcast B-vector DMA sits under two `affine.if` and an
`scf.if`), the backward slice is large enough that an insert reallocates the
SetVector's backing vector mid-iteration. The range-for's iterator is then
dangling, and the next dereference reads freed memory and crashes.

It is heap-layout dependent (classic iterator-invalidation UB), which is why it
reproduces deterministically inside the full `aircc` pipeline but not from a
fresh `air-opt` invocation on the same IR.

## Fix

Collect the constant defining ops into a temporary `SmallVector` first, then
insert them into `backwardSlice` after the loop completes. No insertion happens
during iteration, so the iterator cannot be invalidated.

The result is unchanged: every value added is a constant (selected by
`getConstantIntValue`), and constants have no regions, so they would contribute
nothing even if the original loop had re-visited them. The final `backwardSlice`
contains exactly the same operations.

## No regression

The set of hoisted ops is identical; only the order of inserting the constants
changes (and SetVector de-duplicates regardless). All existing
`test/Transform/AIRDmaToChannel` lit tests pass unchanged.

## Test

Adds `cascade_broadcast_hoist_constant_slice.mlir`, a reduced (2x2 herd, K=128)
cascade GEMV that drives the broadcast-DMA hoist through the same nested
`affine.if`/`scf.if` structure, and checks the channelization
(`air-opt -air-dma-to-channel`). The crash itself is heap-layout dependent
(iterator-invalidation UB): it reproduces deterministically in the full `aircc`
pipeline but not from a fresh `air-opt` on the same IR, so the lit test guards
the channelization of this path rather than asserting a segfault.

## Validation

With the fix:
- `matvec_cascade.py` now compiles (previously SIGSEGV) and **runs correctly on
  NPU2** (the example's own check against the `np.dot` golden, rtol=0.04) at both
  `M=16` and the default `M=2048`.
- The full `test/Transform/AIRDmaToChannel` lit suite passes, including the new
  test.
