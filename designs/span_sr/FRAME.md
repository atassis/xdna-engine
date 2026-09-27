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

## Variant B (spill long-lead skips to DDR): legal at the API/toolchain level, checked on real aiecc

Spec's "After milestone 1" section: S4's join restructure (4 independent per-source rings) doesn't
fit (5 input DMA channels on `conv_cat` against the 2-in cap; joins don't chain, see "Channel-count
check" above). Variant B was left open ("whether an IRON join accepts shim-fed subfifos; breaks
'never leaves L2' for two skip streams"). Checked both halves.

**join() accepts a shim-fed subfifo producer -- confirmed by API, not inference.**
`ObjectFifoHandle.join()` (`python/iron/dataflow/objectfifo.py:1110`) only wires each new
subfifo's CONSUMER side into the link (`subfifo_cons = [s.cons(...) ...]`, `ObjectFifoLink(...)`,
:1188-1193); the PRODUCER side is left untouched, exactly like `f_in`/`out[names[-1]]` in
`net_design.py`. `ObjectFifo.prod()` (:327) takes `tile:` and documents it directly: "When this
handle drives an ObjectFifo from the runtime (passed in `Runtime` `fn_args`), the shim tile its
host-side DMA binds to." Nothing in `join()`, `ObjectFifoLink.__init__` (:1398-1441, only asserts
src/dst counts and offset-list lengths) or `ObjectFifoLinkOp::verify()` constrains a subfifo's
producer tile TYPE -- the join and the shim-feed are orthogonal mechanisms, and `AnyShimTile`
(`python/iron/device/tile.py:123`) is a first-class producer tile like `AnyMemTile`/
`AnyComputeTile`.

**Delayed self-read (write row now, read it back later) is an existing Runtime primitive.**
`python/iron/runtime/runtime.py`'s own module docstring: "it can use native `range_`/`if_` control
flow with `fill`/`drain` verbs nested inside." Ordering a read-fill to happen only after a
write-drain completes (rather than racing two independent shim queues) is `TaskGroup`
(`python/iron/runtime/taskgroup.py`): `group=` on `fill()`/`drain()`, `wait=True` on the write,
`tg.finish()` awaits it before the read issues. `IronRuntimeError`'s only constraint found here
(`python/iron/runtime/runtime.py:151`): mixing the implicit default group with an explicit one is
refused ("Mixing explicit task groups and the default task group is prohibited") -- once every
fill/drain in a sequence with any explicit group is put in an explicit group, it resolves.

**Compile-only proof, full aiecc pipeline, CPU-only** (`aie_kernels/_test/compile_span_spill_join.py`):
2 producer cores -> 2 shim WRITE drains into 2 DDR spill buffers (`TaskGroup`, `wait=True`) -> 2
shim READ fills back out of those SAME buffers into a 2-way `.join()` on one `AnyMemTile` -> 1
consumer core -> shim drain to `y`. Ran via `./run.sh compile_span_spill_join.py`:
```
[span_spill_join] OK: shim-fed join subfifo + DDR write-then-read compiles clean through aiecc
address allocation
```
No rejection at the API level, the MLIR verifier, or aiecc's address allocation. **7 shim-side DMA
transfers total** (4 MM2S fills: 2 host x-in + 2 spill-read; 3 S2MM drains: 2 spill-write + 1
y-out), auto-placed by `AnyShimTile`/`AnyMemTile` -- exact tile/channel assignment not dumped this
session (no `--print-alloc`-equivalent run; see "Not done"). This is a minimal 2-source probe of
the WIRING, not the real 4-source net with real row counts/LEAD; row-granular (per-`lead`-rows)
fill/drain, as opposed to this probe's whole-buffer fill/drain matching every other `Runtime` call
in this repo, is unbuilt and untimed -- see "Not done".

**Byte/ring budget under variant B, re-derived from `net_layout.py` (not copied from W=32/48):**
conv_1 and b1c3 (`rows_ahead` 20, 17 -> current `skip_depth` 36, 33) are the two spilled sources;
their on-chip join depth drops to `SKIP_SLACK` alone (16, no `rows_ahead` term -- the DDR round
trip absorbs the lead, the ring only needs handshake slack). conv_2 (`skip_depth` 17) and b6c1
(`skip_depth` 20) are unchanged (still core -> join directly). The join is still ONE
`.join()` call / one shared MemTile pool (variant B does not touch that constraint -- only S4 did,
and S4 is dead), so `ring_depth = max(17, 20, 16, 16) = 20` (b6c1-dominated, not conv_1 anymore).

| W | half(w) | ring bytes (`5*half(w)*20`) | vs 524,288 (dedicated MemTile) |
|---|---|---|---|
| 32 | 2,304 | 230,400 | 43.9% |
| 48 | 3,072 | 307,200 | 58.6% |
| 96 | 5,376 | 537,600 | **102.5% -- still over** |
| 112 | 6,144 | 614,400 | **117.2% -- still over** |

Closed form `ring(w) = 5*(w+16)*48*20 = 4,800*(w+16)`; solving `<=524,288` (best case: the shallow
join alone on a dedicated MemTile, no weight-group sharing) gives **`w <= 93.2`** -- narrower than
96, well short of 112. This is *tighter* than the spec's own W=96-112 target but *looser* than the
b1c1/b1c2 split's own L1 ceiling (`w<=89`, spec S5-iii, ESTIMATED). So variant B does what the
spec claims -- it demotes the ring from binding wall to non-binding (89 < 93.2) -- but it does not,
on this arithmetic, make W=96 or W=112 compile outright; L1 (b1c1/b1c2 split) becomes the tighter
wall again, consistent with the spec's own "W then binds on core L1 (~89 px)" line. Getting to
96-112 needs BOTH variant B (or equivalent) AND a wider L1 fix than S5-iii, or a smaller INNER.

**DDR spill channel count, and which caps it does/doesn't trip.** Each spilled source costs 2 new
shim-side DMA transfers (1 S2MM write, 1 MM2S read) -- 4 total for conv_1+b1c3, confirmed
compiling in the 2-source probe above. This is a DIFFERENT resource from both caps that blocked
S4: the compute-tile 2-in/2-out cap (`conv_cat`'s own channels, untouched -- variant B keeps
`conv_cat` at 1 input, the joined `cat_in`) and the MemTile's 6-MM2S `WEIGHT_GROUP` cap (governs
weight-group fan-out, not skip traffic; the spill/read channels sit on a shim tile, not on the
join's MemTile's weight-serving side). So variant B trips neither of the caps S4 tripped -- its
binding constraint is the ring's own byte budget above, not a channel count.
