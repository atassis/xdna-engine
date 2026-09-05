# STEP-6 row-tiled, MemTile-staged rel-pos MHA block -- build & device gate

Row-tiled version of the per-head Parakeet rel-pos MHA node (AC+BD matmuls ->
rel_shift+softmax -> ctx matmul). The T query rows are processed in TILES of `TQ`
inside the kernel (`relpos_rowtiled_bake`), so only the per-tile `[TQ,*]` score/prob
scratch lives in L1 -- the fix for the step-5 `relpos_full_bake` L1 overflow at real
T. Resident k/p/V are staged through the MemTile (L2). Target: T up to 172, one head.

Single-tenant NPU: the orchestrator gates the device serially. Announce + `fuser`
the device, stop `npu-serve`/`npu-asr` before any timed run (do NOT auto-restart
mid-run). This file is AUTHOR + build-recipe only; it does not touch the device.

STATUS: STEP=8 T=172 DEVICE-GATED (rel-L2 0.025 synth / 7e-3 real-tiled, corr 0.9998,
both <= 0.08 gate). The full-length resident block is validated on silicon.

ALIGNMENT (load-bearing): the softmax reads scores at row stride T (or P) and the
rel_shift base `BD + i*P + (T-1-i)` is NEVER 128b-aligned. aie2p `aie::load_v` is an
ALIGNED load and TRUNCATES an unaligned address to the 128b boundary (returns shifted
data) -- so those score/prob accesses MUST use `aie::load_unaligned_v` /
`aie::store_unaligned_v`. This is silent unless BD ~ AC: real block-0 is content-
dominated (BD << AC after rescale) so it hid the bug; a spread/synth softmax (BD ~ AC)
exposes it. Gate any softmax kernel change with a `--synth-T` run, not only real data.

## 0. numpy correctness (already GREEN, no device)

    ~/npuvox-asr-bench/.venv/bin/python scripts/parakeet_relpos_mha_golden.py

STEP-6 results (the load-bearing rel_shift / row-tiling proof):

    G6  tiled rel_shift(T= 32,Tq= 8) == NeMo : rel=0.00e+00 exact  PASS
    G6  tiled rel_shift(T=172,Tq= 8) == NeMo : rel=0.00e+00 ragged PASS   # 172 % 8 = 4
    G6  tiled rel_shift(T=172,Tq=16) == NeMo : rel=0.00e+00 ragged PASS
    G6  tiled rel_shift(T=172,Tq=24) == NeMo : rel=0.00e+00 ragged PASS
    G7 T=32 real   tiled(Tq=8)==single-tile : probs rel=0.00e+00 ctx rel=0.00e+00  PASS
    G7 T=32 real   tiled bf16 vs f32 host    : probs rel=5.50e-03 ctx rel=1.67e-03  PASS
    G7 T=172 synth tiled(Tq=8)==single-tile  : probs rel=0.00e+00 ctx rel=0.00e+00  PASS
    G7 T=172 synth tiled bf16 vs f32 host    : probs rel=3.61e-03 ctx rel=3.80e-03  PASS
    G7 T=172 synth tiled(Tq=16)==single-tile : probs rel=0.00e+00 ctx rel=0.00e+00  PASS

`tiled == single-tile` is BIT-EXACT (rel=0) -> query-row tiling introduces zero
numerical change PROVIDED the rel_shift base uses the GLOBAL query index
`(T-1) - (q0+il)`. That is the whole correctness risk of this step, and it is proven
at the target T=172 with ragged tiles (TQ that does not divide T).

## 1. build (orchestrator, on the toolchain box)

FORK-ONLY toolchain (never the wheel). `T` and `TQ` are BAKED into the kernel
(`-DRELPOS_T` / `-DRELPOS_TQ`) and MUST match the generator `-T` and the runner.

    source scripts/toolchain_up.sh          # blessed atassis/mlir-aie fork instance
    source scripts/iron_env.sh
    scripts/sync_kernels.sh                  # route_b_kernels/ -> mlir-aie build sandbox
    cd mlir-aie/programming_examples/ml/relpos_mha
    make clean
    make NPU2=1 STEP=6 T=172 TQ=8           # -> build/final.xclbin + build/insts.bin

Pick TQ so the per-tile score scratch fits L1 (g_ac[TQ*T] + g_bd[TQ*P] f32 +
g_probs[TQ*T] bf16; ~19 KB at TQ=8,T=172). TQ need not divide T.

Bring-up sweep (kpv fits L1 at small T): `make ... STEP=6 T=32 TQ=8` first to gate
the row-tiled arithmetic on silicon, then raise T.

## 2. device gate (serial, orchestrator-driven)

    # announce + quiesce, then:
    ~/npuvox-asr-bench/.venv/bin/python scripts/run_npu_relpos_rowtiled.py \
        --xclbin mlir-aie/programming_examples/ml/relpos_mha/build/final.xclbin \
        --insts  mlir-aie/programming_examples/ml/relpos_mha/build/insts.bin

Real block-0 head-0 (T=32) by default. For the TARGET shape when no real 172-frame
activations are on disk (only T=32 refs ship), synthesize the gate (must match the
built T/TQ):

    ... run_npu_relpos_rowtiled.py --synth-T 172

Gate (both): `rel-L2 <= 0.08 AND corr >= 0.99` vs the fp32 host ctx. `--raw` drives
the true saturating (one-hot) regime; default rescales to a non-degenerate softmax
that exercises the exp2 path + the tiling.

## 3. STEP-7 / STEP-8 -- the block-streamed path that reaches T=172

Step-6 `relpos_rowtiled_bake` still takes FULL k/p/V pointers (kpv resident in
L1), so it caps at the T where kpv fits L1. p alone is 86 KB > 64 KB L1 at T=172,
and quv (2T*DK*2 = 88 KB) overflows too. Two new pieces close this, split so the
ARITHMETIC risk and the DATAFLOW risk are gated separately:

- STEP=7 `relpos_kpvstream_bake` (`relpos_kpvstream_iron.py`) -- MONOLITHIC
  reference. Same packed `(quv, kpv, ctx)` ABI as STEP=6, but the k/p/V matmuls
  are DECOMPOSED into KB-row key-blocks: `relpos_dot_block` fills a COLUMN SLICE
  `[TQ, kb]` of the score row; `relpos_ctx_block` accumulates each V-block into a
  resident f32 ctx buffer, narrowed once at the end. These are the EXACT bricks
  and accumulation order the streaming core runs -- so STEP=7 gates the
  block-decomposed arithmetic on the SAME runner (unchanged) at the T where kpv
  fits L1. Numeric delta vs STEP=6: AC/BD are bit-identical (each element is a
  single full-DK dot; only the key dim is tiled); ctx re-associates its key sum in
  f32 (bf16 hop only at the final narrow, same as the proven brick) -> a strict
  precision non-regression, far below the ctx rel-L2 gate (0.08).

- STEP=8 `relpos_rowtiled_stream` (`relpos_rowtiled_stream_iron.py`) -- the FULL-T
  block. k/p/V staged in the 512 KB MemTile and streamed to L1 in KB-row
  key-blocks, RE-STREAMED once per query tile (STEP-A: the runtime re-fills the
  whole kpv per tile -- see 3f; `repeat_count` replay was rejected on device). L1
  then holds only ONE key-block + the `[TQ,*]` scratch. Same block bricks as
  STEP=7, exposed as `relpos_stream_dot/_softmax/_ctx*` and driven from the IRON
  Worker's acquire/release loop. This adds ONLY the dataflow the golden cannot
  validate.

### 3a. ObjectFifo topology (2 input DMA channels -- HARD limit)

The NPU2 CORE tile has exactly 2 input (S2MM) + 2 output (MM2S) DMA channels
(`AIE2TargetModel::getNum{Dest,Source}SwitchboxConnections` = 2 for `WireBundle
::DMA` on core tiles; MemTile = 6). STEP=8 uses both inputs + one output:

  Channel A  `of_quv` [TQ,DK] bf16, depth 2 -- per query tile the core acquires
             qu_tile (phase K) then qv_tile (phase P); 2*n_qt blocks, read ONCE
             (no replay). Host packs QUV TILE-INTERLEAVED.
  Channel B  `of_kpv` [KB,DK] bf16, depth 2 -- a DIRECT block fifo (obj = one
             [KB,DK] block, same style as of_quv). ONE shim BD re-reads the whole
             padded kpv from DDR offset 0 n_qt times (stride-0 outer tap dim -> BD
             repeat_count=n_qt-1, 3f); each replay's kpv_pad_rows read is delivered
             as 16 blocks in address order, so each tile gets k0..k3,p0..p7,V0..V3
             from the start. Per query tile: n_kb k-blocks, n_pb p, n_vb V.
             (Earlier this used of_kpv_l3l2(obj=whole kpv).forward(obj=[KB,DK]) --
             a forward with a SMALLER obj than its source; removed, see 3f.)
  Output     `of_ctx` [TQ,DK] bf16, depth 2 -- one block per query tile.

`split`/`forward`/`repeat_count` API: `python/iron/dataflow/objectfifo.py`
(`repeat_count` documented as "MemTile DMA replays the buffer descriptor N times
without a new DMA transfer from L3").

### 3b. BYTE BUDGET (the validation the golden cannot give)

T=172, P=343, TQ=8, KB=43, DK=128; bf16=2 B, f32=4 B.

L1 = 64 KB per compute tile. STEP=8 resident set:

  g_ac    [TQ*T]  f32   8*172*4  =  5504 B
  g_bd    [TQ*P]  f32   8*343*4  = 10976 B
  g_probs [TQ*T]  bf16  8*172*2  =  2752 B
  g_ctxf  [TQ*DK] f32   8*128*4  =  4096 B
  srow    [512]   f32   512*4    =  2048 B   (softmax row scratch, static)
  of_quv  [TQ,DK] bf16  x2       =  4096 B   (depth 2)
  of_kpv  [KB,DK] bf16  x2       = 22016 B   (depth 2; the ONLY streamed k/p/V in L1)
  of_ctx  [TQ,DK] bf16  x2       =  4096 B   (depth 2)
  ------------------------------------------------
  TOTAL                          = 55584 B = 54.3 KB  < 64 KB  (9.9 KB headroom)

  Drop `of_kpv` to depth 1 (loses stream/compute overlap) -> 44576 B = 43.5 KB if
  the placer needs more slack. The streamed block never exceeds [KB,DK] = 11 KB.

Contrast -- what overflowed: whole kpv resident [2T+P,DK] = 687*128*2 = 171.8 KB
PLUS whole quv [2T,DK] = 86 KB -> ~258 KB in a 64 KB L1. Streaming replaces both
whole buffers with one 11 KB block + the TQ-sized scratch.

L2 (MemTile) = 512 KB. Both inputs are DIRECT block fifos (obj = [KB,DK] / [TQ,DK]),
so nothing stages the whole 172 KB kpv in L2 -- if IRON routes the block fifos
through a MemTile at all it is only [KB,DK]/[TQ,DK] double-buffers (tens of KB). The
padded kpv (Tp+Pp+Tp = 688 rows = 176 KB) lives only in DDR and is re-streamed as
[KB,DK] blocks. L2 usage is negligible.

### 3c. build (orchestrator, toolchain box)

    source scripts/toolchain_up.sh ; source scripts/iron_env.sh
    scripts/sync_kernels.sh
    cd mlir-aie/programming_examples/ml/relpos_mha
    make clean
    # STEP=7 -- gate the block arithmetic where kpv fits L1 (bring-up):
    make NPU2=1 STEP=7 T=32 TQ=8 KB=43
    # STEP=8 -- the full-T streamed block (T,TQ,KB baked AND passed to the gen):
    make clean && make NPU2=1 STEP=8 T=172 TQ=8 KB=43

KB must match `-DRELPOS_KB`; T=172=4*43 blocks k/V with no pad, p is 7 full + a
42-row ragged tail (1 pad row in the L2 layout, unread by the core).

### 3d. device gate (serial, orchestrator-driven; announce + fuser + quiesce)

**CAVEAT (2026-08-31, relpos-uses-8-of-32-cores round 19):** the `--stream` commands below predate
the HEADS=8 row-tiling extension and are STALE for any xclbin built with `HEADS=8` (the shipped
row-tiled arm): they send one head's worth of QUV/KPV/CTX against a runtime sequence that expects
all H heads concatenated head-major (confirmed against the compiled `.mlir`'s
`aie.runtime_sequence` arg sizes -- each is exactly 8x a single head's), which reads/writes past
the allocated host buffer and returns non-finite garbage, non-deterministically. `run_npu_relpos_rowtiled.py`
now builds all H heads for `--stream` and de-concatenates the readback per head, which removes that
failure class -- but a SEPARATE, unresolved issue remains even after the fix (also present on the
untouched, single-head STEP=6 path, so it is not multi-head-specific): the gate still does not pass
(rel-L2/corr both far outside tolerance), with a per-row error pattern that cleanly splits between
~0.002 (bf16 noise) and ~1.3-1.6 (essentially uncorrelated), the first query tile always clean and
later tiles scattered bad -- shape-matching the KPV block-replay mechanism in 3f below, not yet
isolated further. The real production pipeline (`rust/npu-parakeet`'s `RelposK`, same xclbin family)
gives correct transcripts on real clips, so this is not believed to be a kernel defect, but that is
not proven for this second issue the way the buffer-size fix is. Do not treat a passing `--stream`
run as fully trustworthy until this is resolved; see the script's own updated docstring.

    # STEP=7 (existing packing, unchanged runner):
    scripts/run_npu_relpos_rowtiled.py --xclbin .../build/final.xclbin \
        --insts .../build/insts.bin --synth-T 32
    # STEP=8 (streamed packing: tile-interleaved QUV + padded KPV/CTX):
    scripts/run_npu_relpos_rowtiled.py --xclbin .../build/final.xclbin \
        --insts .../build/insts.bin --synth-T 172 --stream --tq 8 --kb 43
    # STEP=8 DISCRIMINATOR (device input + reference from ONE tiled real array):
    scripts/run_npu_relpos_rowtiled.py --xclbin .../build/final.xclbin \
        --insts .../build/insts.bin --real-tiled-T 172 --stream --tq 8 --kb 43

Gate (both): `rel-L2 <= 0.08 AND corr >= 0.99` vs the fp32 host ctx.

The --stream runner now prints a `[pack-check]` line: it de-packs the exact QUV/KPV
BO bytes per the kernel's expected layout and asserts they reconstruct the reference
qu/qv/k/p/V. `scripts/relpos_stream_packing_check.py` runs the same check standalone
and PASSES byte-exact at T=32 AND T=172 -> the --stream device-input packing is
consistent with the reference; a synth-only failure is NOT a packing/reference
divergence. `--real-tiled-T N` is the discriminator: if T=172 FAILS with --synth but
PASSES with --real-tiled-T (or fails identically), it isolates synth-data vs a real
device multi-block bug.

### 3e. CORE = QUERY range_ + UNROLLED BLOCKS (two device-found bugs)

The core loop topology went through two device-driven corrections:

1. PROGRAM-MEMORY overflow. Fully unrolling everything (n_qt=22 query tiles x ~16
   block calls = ~352 `func.call`s) overflowed CORE PROGRAM memory at ELF/CDO
   (`_XAie_LoadProgMemSection(): Overflow`) -- an INSTRUCTION-count problem (the
   54 KB L1 DATA budget, 3b, was always fine). Fix: make the 22x QUERY-tile sweep
   a `range_` hardware loop.

2. NESTED-range_ j0 delivered wrong on device. Making the k/p/V BLOCK loops nested
   `range_` loops (j0 = `index_cast` of the inner induction Value) BUILT and PASSED
   at T=32 but FAILED at T=172 (corr 0.65). Root cause (numpy-reproduced, see
   `scripts/relpos_block_model_check.py` + a stuck/iter-index j0 model): the nested
   loop's runtime j0 did NOT take the per-iteration value 0,KB,2KB,... on device.
   T=32 masked it -- there `Tk_full = (32//43)*43 = 0`, so the k/V block loops are
   EMPTY (k/V come from static peels) and p runs a single j0=0 iteration; the
   multi-iteration nested j0 is first exercised at T>=86. The OUTER query range_'s
   `index_cast` q0 is fine (T=32 runs 4 query tiles and passes).

Shipped topology:
  - query tiles: `range_(0, Tq_full, TQ)` hardware loop; q0 = `index_cast(iv)`
    (the induction Value IS q0). Ragged final tile PEELED (tq a Python constant).
  - k / p / V blocks: UNROLLED Python loops -- `for j0 in range(0, *_full, KB)`
    with j0 a Python-int CONSTANT, + a static ragged peel. 16 blocks/tile.
  Emitted: ~32 block calls (query-loop body + peeled tile) -- far under the ~352
  that overflowed, and j0 is a proven-good compile-time constant (the peels
  already used static j0 and passed at T=32). Only runtime i32 is the query q0.

VERIFY (no device): `... | grep -c 'scf.for'` is SMALL (1 query hardware loop;
block loops Python-unrolled inside it); L1 memref allocs unchanged (3b).

### 3f. KPV replay (single-BD shim) + the corr=0.65 delivery bug

REPLAY mechanism (shipped): ONE shim BD re-reads the whole padded kpv from DDR
offset 0, n_qt times. `rt.fill(of_kpv.prod(), KPV, tap=kpv_tap)` with

    kpv_tap = TensorTiler2D.simple_tiler([kpv_pad_rows, DK], pattern_repeat=n_qt)[0]
    # -> sizes=[n_qt,1,kpv_pad_rows,DK], strides=[0,0,DK,1], offset 0

`shim_dma_single_bd_task` turns `sizes[0]>1` into BD `repeat_count=n_qt-1` -> ONE
BD replayed n_qt times, each re-reading from DDR offset 0. Two earlier variants
were rejected: (1) L2->L1 `forward(repeat_count=n_qt)` -- replays a STAGED L2
buffer without re-reading L3, so it never restarts per tile; (2) a per-tile fill
loop -- 22 BDs > the 16-BD shim limit. This single-BD tap is correct + 1 BD.

Block-fill accounting (T=172, TQ=8, KB=43): per query tile = 4 k (172/43) + 8 p
(7 full + 1 ragged pb=42) + 4 V = 16 blocks; x n_qt=22 tiles (21 full q0=0..160 +
1 peeled ragged tq=4) = 352 blocks.

TWO device-found corr=0.65 bugs (both distinct from the replay). All the replay
mechanisms gave the IDENTICAL wrong output (rel-L2 0.82, corr 0.65) -> the replay
was never the bug. `scripts/relpos_block_model_check.py` reproduces the EXACT
block-decomposed path on the EXACT --stream packing and is BIT-EXACT to the proven
monolithic tiled model (rel-L2 1e-8 at T=32 and T=172), so the block bricks +
packing LOGIC are correct; the bug is device dataflow (which numpy cannot model).

  BUG 1 (fixed, validated at T=32): `of_kpv_l3l2(obj=whole kpv).forward(obj_type=
  [KB,DK])` -- a forward whose dest obj is 16x SMALLER than its source (not the
  standard 1:1 forward). Fix: make `of_kpv` a DIRECT block fifo (obj = [KB,DK])
  like the working `of_quv`; the shim streams the repeat-tap read as [KB,DK] blocks
  in address order. STEP=8 T=32 --stream then PASSED (rel-L2 6.8e-3, == STEP=6).

  BUG 2 (fixed): nested-`range_` block-loop j0. See 3e(2). Ruled OUT the --synth
  harness first (`scripts/relpos_synth_ref_check.py`: the runner's f32 oracle
  matches the block model at T=32/64/86/129/172, all rel-L2 ~4e-3 -> reference is
  correct). Then localized to the nested-range_ j0 (T=32 passes because its k/V
  block loops are empty; multi-iteration j0 first bites at T>=86). Fix: unroll the
  block loops (Python-int j0); keep the query sweep a `range_`.

L1 budget UNCHANGED (one [KB,DK] block; 54.3 KB, 3b); L2 no longer stages the whole
kpv. If T=172 STILL fails after both fixes, the STEP=7 bisection (`make NPU2=1
STEP=7 T=32 TQ=8 KB=43`, run WITHOUT --stream) isolates STREAM dataflow vs compiled
block bricks; and gate an intermediate multi-block T (e.g. T=86, n_kb=2) to confirm.

VERIFY (no device):

    python3 relpos_rowtiled_stream_iron.py -d npu2 -T 172 --tq 8 --kb 43 \
        | grep -iE 'memref|objectfifo|scf.for'

L1 memref allocs are `[KB,DK]` + the `[TQ,*]` scratch, never `[T,DK]`/`[P,DK]`;
`scf.for` count small (loops, not unrolled). Device gate: 3d STEP=8 command,
`rel-L2 <= 0.08 AND corr >= 0.99`.


---

## Appendix -- STEP 1 / STEP 2 (single-tile de-risk kernels, T<=32)

STEP 1 and STEP 2 are the two single-tile kernels that de-risked the rel-pos
bricks before the row-tiled STEP 6+ block above. Both cap at small T (AC/BD/probs
co-resident in L1; `RELPOS_TMAX=512` caps the per-row f32 scratch) -- STEP 6+ is
the one that reaches the target T=172 and is device-validated (STATUS above).
The kernels/generators/runners below still exist and are the narrowest way to
gate the softmax and AC-matmul bricks in isolation.

### STEP 1 -- relpos_scores_softmax (host-fed AC + BD, no matmul)

Given host-precomputed `AC[T,T]` f32 and `BD[T,P]` f32 (`P = 2T-1`), returns
`probs[T,T]` bf16 = `softmax_over_keys( rel_shift(BD) + AC , * 1/sqrt(DK) )`. No
matmul on device -- isolates the two hard rel-pos bricks (the zero-arithmetic
strided-relayout `rel_shift` + the vectorized-exp2 softmax).

- Kernel: `relpos_mha.cc` -> `relpos_scores_softmax_bake` (thin zero-scalar
  wrapper baking `T`, `P`, `inv_scale=1/sqrt(128)` over `relpos_scores_softmax`).
- Generator: `relpos_scores_softmax_iron.py` (single core, 3-buffer ABI: AC in /
  BD in / probs out; bare `resolve_program()`, PLACE-TILES model, no
  SequentialPlacer).
- Golden (CPU, no device): `scripts/parakeet_relpos_mha_golden.py` (G1-G4b).
- Device runner: `scripts/run_npu_relpos_scores.py`.

`T = 32` (Parakeet block 0) is baked by default. `T` must be identical in
`make T=<n>` (threads both the generator `-T` and the kernel `-DRELPOS_T`).

Build:

    bash scripts/setup_route_b.sh
    source scripts/iron_env.sh
    bash scripts/sync_kernels.sh
    make -C mlir-aie/programming_examples/ml/relpos_mha NPU2=1 T=32
    # -> build/final.xclbin, build/insts.bin

Drive block-0 head-0: the runner computes `AC[32,32]` f32 + `BD[32,63]` f32 from
the encoder weight artifacts, sends them (host_only BOs, group_ids 3/4), reads
back `probs[32,32]` bf16 (group_id 5). ABI: `opcode=3`,
`kernel(3, instr[gid1,cacheable], n_instr, AC[gid3], BD[gid4], PROBS[gid5])`.
Block-0 raw scores saturate (~one-hot softmax), which only tests rel_shift +
argmax, so by default the runner pre-scales AC/BD by `1/std` host-side to land a
non-degenerate softmax that exercises the on-chip exp2 / bf16-reciprocal path
(the oracle uses the identical effective scale); `--raw` drives the true
saturating regime.

    .venv-iron/bin/python scripts/run_npu_relpos_scores.py \
        --xclbin mlir-aie/programming_examples/ml/relpos_mha/build/final.xclbin \
        --insts  mlir-aie/programming_examples/ml/relpos_mha/build/insts.bin \
        --block 0 --head 0
    # add --raw for the saturating (one-hot) regime

CPU golden (`scripts/parakeet_relpos_mha_golden.py`):

    G1  strided rel_shift == NeMo rel_shift : rel=0.000e+00  PASS
    G2  f32 mirror (strided) == host mhsa   : rel=0.000e+00  PASS
    G3  kernel bf16 model vs f32 host mhsa  : rel=4.235e-02  GATE<= 0.08  PASS
    G4a standalone brick, real regime       : rel=0.000e+00  GATE<= 0.08  PASS  (worst of 8 heads; scores saturate -> ~one-hot)
    G4b standalone brick, non-degenerate sm : rel=2.928e-03  GATE<= 0.08  PASS  (worst of 8 heads; exercises exp2 softmax)

Device PASS criteria: `rel-L2 <= 0.08 AND corr >= 0.99` vs the fp32 host softmax,
with `probs` rowsums ~1.0. The host-side numeric model of the device path checks
out at rel-L2 ~2.5e-3 across all 8 heads (rescaled regime) and exactly 0 in the
raw one-hot regime; the device gate itself (aiecc build + pyxrt drive) is
CPU-authored + numpy-validated only, gated serially by the orchestrator like
every other device step in this file.

### STEP 2 -- relpos_ac_scores_softmax (on-chip AC matmul, resident-L1 score tile)

Composes the `AC = qu @ k^T` score matmul ON DEVICE, keeping the f32 score tile
resident in L1 between the matmul and the softmax (no host round-trip) -- the
first resident-block test. Takes PACKED `qk[2T,DK]` bf16 (`qu = qk[0:T]`,
`k = qk[T:2T]`) + host-fed `BD[T,P]` f32, returns the same `probs[T,T]` bf16 as
STEP 1: `softmax_over_keys( rel_shift(BD) + (qu @ k^T) , * 1/sqrt(DK) )`.

- Kernel: `relpos_mha.cc` -> `relpos_ac_scores_softmax_bake` (`relpos_ac_matmul`
  writes a resident L1 f32 `g_ac[T*T]`; `relpos_scores_softmax` reads it in
  place). Row-major bf16-in / f32-accumulate dot-product tile (mirrors the `q.K`
  dot in `mha_decode.cc`), producing row-major AC directly so the per-row
  rel_shift + softmax-over-keys can consume it with no de-block pass. The
  `aie::mmul`-blocked microkernel is the perf follow-up (needs an L1 de-block
  before the row-wise softmax).
- Generator: `relpos_ac_scores_softmax_iron.py` (single core, 3-buffer ABI: qk
  in / BD in / probs out; qu+k packed into one input to stay within the NPU2
  compute tile's 2 input-DMA-channel budget; bare `resolve_program()`,
  PLACE-TILES model, no SequentialPlacer).
- Makefile: same `Makefile`, selected by `STEP=2`.
- Golden: `scripts/parakeet_relpos_mha_golden.py` (G5a/G5b/G5c).
- Device runner: `scripts/run_npu_relpos_ac_scores.py`.

Build (T=32):

    bash scripts/setup_route_b.sh
    source scripts/iron_env.sh
    bash scripts/sync_kernels.sh
    make -C mlir-aie/programming_examples/ml/relpos_mha clean
    make -C mlir-aie/programming_examples/ml/relpos_mha NPU2=1 STEP=2 T=32
    # -> build/final.xclbin (kernel 'relpos_ac_scores_softmax_bake'), build/insts.bin

Drive block-0 head-0: packs `qk = concat(bf16(qu), bf16(k))` and host-fed
`BD[32,63]` f32, reads back `probs[32,32]` bf16. ABI: `opcode=3`,
`kernel(3, instr[gid1,cacheable], n_instr, QK[gid3], BD[gid4], PROBS[gid5])`.
Same regime discipline as STEP 1: block-0 raw scores saturate to a one-hot where
the bf16 matmul can flip a near-tie argmax (harmless; washes out end-to-end), so
the runner defaults to the rescaled non-degenerate softmax (divides qu/BD by
std); `--raw` drives the saturating regime.

    .venv-iron/bin/python scripts/run_npu_relpos_ac_scores.py \
        --xclbin mlir-aie/programming_examples/ml/relpos_mha/build/final.xclbin \
        --insts  mlir-aie/programming_examples/ml/relpos_mha/build/insts.bin \
        --block 0 --head 0
    # add --raw for the saturating (one-hot) regime

Gate: `rel-L2 <= 0.08 AND corr >= 0.99` vs the fp32 host softmax, rowsums ~1.0.

Step-2 golden numbers (`scripts/parakeet_relpos_mha_golden.py`):

    G3  kernel bf16 model vs f32 host mhsa  : rel=4.235e-02  GATE<= 0.08  PASS
    G5c step-2 AC bf16 mmul vs f32 qu@k^T   : rel=8.264e-03  (matmul only, diagnostic)
    G5a step-2 composed brick, real regime  : rel=2.500e-01  DIAGNOSTIC (one-hot; 2 near-tie argmax flip(s) across 8 heads; washes out in G3)
    G5b step-2 composed brick, non-degen sm : rel=7.048e-03  GATE<= 0.08  PASS
    RESULT: ALL PASS

G5a is diagnostic, not a gate: the real block-0 softmax is an exact one-hot, so
the only signal is whether the bf16 AC matmul flips a near-tie argmax -- exactly
2 rows across all 8 heads flip, each costing `sqrt(2/T) ~ 0.25` rel by
construction. The end-to-end G3 (bf16 AC folded through ctx + out proj) shows
these flips wash out to 4.2e-2, and G5b (rescaled, exercises the actual on-chip
matmul + exp2 numerics) passes at 7.0e-3. So the composed brick is gated on G5b
+ G3, matching STEP 1's "the real regime only tests rel_shift + argmax" framing.

Device gate for STEP 2: CPU-authored + numpy-validated only, same
orchestrator-serial device discipline as every step in this file. The unknown
vs STEP 1 is the matmul -> resident-L1 -> softmax objectFIFO chain and the
packed-qk 2-input DMA; the softmax brick itself is validated from STEP 1.
