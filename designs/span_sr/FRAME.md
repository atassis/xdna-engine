# Phase 2 M1: frame-width design, status

Spec: `2026-09-28-span-frame-width-design.md` (journal, commit `fdb39178`). Task:
`fused-espcn-streaming-pipeline`. Toolchain pin `79c1952`, instance `83c26a8138af`.

## Channel-count check (done first, per task)

The spec's Blocker-2 fix (S4) is "drop `ObjectFifoHandle.join()`; each skip source's producer
feeds its own fifo directly" into `conv_cat`. That means `conv_cat`'s core reads 4 source fifos +
1 weights fifo = **5 input DMA channels**, against the per-core cap of **2 in / 2 out**
(`npu-resource-budgets.md`). Does not fit -- checked before building, not after.

**The natural MemTile-side fallback (a shallow second join re-combining 4 independently-sized
per-source rings) is ALSO blocked**, not by budget but by the toolchain's own verifier:
`ObjectFifoLinkOp::verify()` (`AIEDialect.cpp:~1498`) rejects an objectfifo that appears in more
than one `ObjectFifoLinkOp` ("objectfifo cannot be in more than one ObjectFifoLinkOp"). A per-
source ring that is itself the dst of one join cannot then be a src of a second -- so joins do not
chain in this pin, and "join of joins" is not a legal construction here, confirmed by reading the
verifier, not just inferred from net_design.py's own "one shared pool" comment (which independently
says the same thing: a join's many-side subfifos never get separate physical buffers).

Only the remaining option from the task's own list fits both constraints: **split `conv1x1_cat`'s
1x1 into two partial-sum cores** (int32 accumulator per pair of sources, one MemTile-side rejoin,
one shared final requant on a third core) -- bit-exact by the same associativity argument the spec
already makes for its own 4-source design, but requiring a real kernel addition (a partial-
accumulator entry point + a combine-and-requant entry point), not a wiring change. **Scoped, not
built this session** -- see "Not done" below.

## b1c1/b1c2 split (S5-iii): built, verified

Generalized `net_design.py`'s `split_gate` mechanism (previously gate-only, BALANCE.md option (a))
to `silu16`/`silu_i16` (`SPLIT_KINDS`), reusing `net_layout.split_gate_params` as-is (already kind-
agnostic -- it only slices the shared weight/bias/mult blob format). New: `_split_layout()` (byte
split of a stage's own COUT-indexed segment, silu16's int16 result or silu_i16's int8 result, plus
where an x-forward attaches on the hi half only) and `_shim_split_half()` (kind-dispatched call,
generalizing `_shim_gate_half`). `build()`'s `split_gate=` now accepts b1c1/b1c2 with zero change
to any other wiring (the kernel-arg construction and the join-based rejoin were already generic).
Default behaviour unchanged (`split_gate=frozenset()`), so every existing byte-identical path is
untouched.

**Compile-only gate** (`compile_span_split_b1c1c2.py`, no device): `split_gate={"b1c1","b1c2"}`
places within L1/MemTile/DMA at **W=32** -- full aiecc pipeline, address allocation included.

**Device gate** (`verify_span_split_b1c1c2.py`): full net (24 cores, upto=`up`) on a real-photo
32x64 strip crop, exact vs `span_int`: **32768/32768 -> PASS**.

## Max W this milestone: still 32 -- ring (Blocker 2) is the binding wall, confirmed on real aiecc

W must be a multiple of 16 (`conv3x3_u8.cc`'s own constraint). With b1c1/b1c2 split applied, swept
W=48/64/80/112 through the SAME compile-only harness: **all fail**, every one at the join's MemTile
allocation, not at any core's L1 -- e.g. at W=48:

```
error: 'aie.objectfifo.pool' op iterate_bds needs one 552960-byte buffer on its MemTile,
which has 524288 bytes free
```

`552,960 = 5*3072*36` (`CAT_HALVES(5) x half(48)(3072) x max(skip_depths)(36)`) -- exactly the
spec's own S3 formula, and the ring's own predicted ceiling (`w<=44.7`, so W=48 is the first
16-multiple past it) lands exactly where aiecc actually rejects it. **b1c1/b1c2 is real, correct,
and frees L1 as designed, but the ring is the tighter wall once it's fixed, unchanged from before
it was fixed** -- symmetric to spec S5's own finding the other way around. Reaching the spec's
target strip widths (96-112) needs the ring fix; that fix, as specified, doesn't fit this
toolchain (see above), and its viable replacement (partial-sum cores) isn't built. So the milestone
gate (bit-exact strip at the largest W the fixes admit, then the pace probe) has nothing new to run
at a wider W than the existing baseline -- W=32 today is the same ceiling as before this session,
now with one of the two blockers understood and the other (b1c1/b1c2) actually fixed.

## Byte table at W=32 (h=half(32)=2304)

| stage/structure | bytes | vs budget |
|---|---|---|
| b1c1 unsplit (tile 0,3), pre-fix | 65,344 / 65,536 | 99.7% (spec S2, unchanged, reference) |
| b1c1_lo/b1c1_hi, b1c2_lo/b1c2_hi | not individually queried this session -- compiles clean, see "Not done" | -- |
| conv_cat join ring (`cat_in`), unchanged | 5*2304*36 = 414,720 / 524,288 | 79.0% (spec S3, unchanged -- not fixed) |
| conv_cat join ring at W=48 (swept, fails) | 5*3072*36 = 552,960 / 524,288 | 105.5% -- confirmed by aiecc, not estimated |

## Not done (honest gap, not swept under "compiles")

- Per-core L1 byte tables for `b1c1_lo/hi`, `b1c2_lo/hi` individually -- verified only as "the
  whole 24-core net places", not queried per-tile. Would need an aiecc `--print-alloc`-style dump;
  not run this session.
- The ring fix (Blocker 2): unbuilt. Partial-sum-core kernel (int32 accumulator split +
  combine-and-requant) scoped above, not written.
- Bit-exact gate at any W > 32: nothing to gate -- the ring wall is untouched, so no wider strip
  compiles yet.
- Pace probe (b1c1/b1c2 halves' cyc/px, whole-net rate at W=32): not run. BALANCE.md's own A/B on
  the b2c3 gate split found NO measured whole-net win from a channel split at this MAIN_DEPTH (the
  skip-ring/lockstep dynamics mask a per-core compute cut) -- the same result is plausible here and
  should be measured, not assumed, before extending this split to production.
