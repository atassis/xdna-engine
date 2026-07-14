# STEP-6 row-tiled, MemTile-staged rel-pos MHA block -- build & device gate

Row-tiled version of the per-head Parakeet rel-pos MHA node (AC+BD matmuls ->
rel_shift+softmax -> ctx matmul). The T query rows are processed in TILES of `TQ`
inside the kernel (`relpos_rowtiled_bake`), so only the per-tile `[TQ,*]` score/prob
scratch lives in L1 -- the fix for the step-5 `relpos_full_bake` L1 overflow at real
T. Resident k/p/V are staged through the MemTile (L2). Target: T up to 172, one head.

Single-tenant NPU: the orchestrator gates the device serially. Announce + `fuser`
the device, stop `npu-serve`/`npu-asr` before any timed run (do NOT auto-restart
mid-run). This file is AUTHOR + build-recipe only; it does not touch the device.

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
  Channel B  `of_kpv` [KB,DK] bf16, depth 2 -- `of_kpv_l3l2` stages the whole
             padded kpv L3->L2; `.forward(obj_type=[KB,DK])` delivers it L2->L1 in
             16 KB-blocks per fill. The runtime re-fills the whole kpv per query
             tile (STREAM-A, 3f), so each tile gets k0..k3,p0..p7,V0..V3 from the
             start. Per query tile: n_kb k-blocks, n_pb p-blocks, n_vb V-blocks.
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

L2 (MemTile) = 512 KB. Staged padded kpv = (Tp + Pp + Tp)*DK*2 with Tp=n_kb*KB=172
(no pad), Pp=n_pb*KB=344 (1 pad row) -> 688*128*2 = 176128 B = 172.0 KB < 512 KB.
quv streams L3->L1 (no whole-L2 stage); of_ctx is transient. So L2 ~172 KB, ample.

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

    # STEP=7 (existing packing, unchanged runner):
    scripts/run_npu_relpos_rowtiled.py --xclbin .../build/final.xclbin \
        --insts .../build/insts.bin --synth-T 32
    # STEP=8 (streamed packing: tile-interleaved QUV + padded KPV/CTX):
    scripts/run_npu_relpos_rowtiled.py --xclbin .../build/final.xclbin \
        --insts .../build/insts.bin --synth-T 172 --stream --tq 8 --kb 43

Gate (both): `rel-L2 <= 0.08 AND corr >= 0.99` vs the fp32 host ctx.

### 3e. CORE = REAL HARDWARE LOOPS (program-memory fix)

The STEP=8 core sweeps are `aie.iron` `range_` loops (the whole_array_modal
core_fn pattern: ObjectFifo acquire/release INSIDE `range_`), NOT static Python
unrolling. The first cut unrolled everything (n_qt=22 query tiles x ~16 block
calls = ~352 `func.call`s) and the ELF/CDO stage overflowed CORE PROGRAM memory
(`_XAie_LoadProgMemSection(): Overflow` / `XAie_LoadElf failed`) -- an
INSTRUCTION-count problem; the 54 KB L1 DATA budget (3b) was fine and the MLIR
generated. Loop topology now (bounded to a handful of call sites):

  - query tiles: `range_(0, Tq_full, TQ)` -- the induction Value IS q0 (0, TQ,
    2*TQ, ...; NO multiply). tq == TQ every iteration. The ragged final query
    tile (tq = T % TQ = 4 at T=172) is PEELED as ONE static iteration so tq stays
    a Python constant (no runtime min()).
  - k / V blocks: `range_(0, Tk_full, KB)` (j0 = the induction Value); at
    T=172,KB=43 there is no ragged block (172 = 4*43).
  - p blocks: `range_(0, Pp_full, KB)` for the 7 full blocks + a peeled ragged
    final block (pb = 42). 
  Runtime i32 scalars (q0, j0) = `index_cast(induction_value, to=i32)` -- the
  exact helper `python/helpers/dialects/scf.py` uses. tq/kb are Python constants
  throughout (peeling avoids runtime `min`). So the emitted core is ~1 query-loop
  body + 1 peeled tile = a small, fixed number of `func.call`s regardless of T.

VERIFY (no device): `... | grep -c 'scf.for'` is SMALL (4 range_ loops per
emitted body: query + k + p + V), NOT ~352 unrolled calls; L1 memref allocs
unchanged (3b).

PROBE 1 (do int32 kernel scalars lower?) is now RESOLVED: the static build
generated valid MLIR and reached the ELF stage, so int scalar args lower fine;
the loop version passes q0/j0 as `index_cast`'d i32 Values (same operand slot).

### 3f. KPV REPLAY = STREAM-A (repeat_count rejected on device)

PROBE 2 (resident-L2 `repeat_count` replay) was tried and REJECTED on silicon:
`of_kpv.forward(obj_type=[KB,DK], repeat_count=n_qt)` from a whole-kpv L2 buffer
BUILT (8 scf.for, ELF/CDO pass) but FAILED parity -- rel-L2 0.82, corr 0.65: the
L2 read did NOT restart per query tile, so tile q consumed the wrong key-blocks
(systematic, not garbage).

SHIPPED FIX = STREAM-A. Drop `repeat_count`; the runtime RE-FILLS the whole padded
kpv ONCE PER QUERY TILE:

    for _q in range(n_qt):
        rt.fill(of_kpv_l3l2.prod(), KPV)   # each fill -> forward -> 16 blocks

so every query tile deterministically gets kpv from the START (k0..k3, p0..p7,
V0..V3). `of_kpv_l3l2` (whole padded kpv, depth 1) -> `of_kpv` (forward, obj
[KB,DK], depth 2) delivers 16 blocks per fill; n_qt fills = 22 x 16 = 352 blocks =
exactly what the core acquires. L1 budget UNCHANGED (one [KB,DK] block at a time;
54.3 KB from 3b). Cost: kpv is streamed from DDR n_qt times (the documented
STREAM-A data-movement tradeoff) -- correctness ships first; a working resident-L2
replay to cut the re-fetch is a later optimization.

Block-fill accounting (T=172, TQ=8, KB=43): per query tile = 4 k (172/43) + 8 p
(7 full + 1 ragged pb=42) + 4 V = 16 blocks; x n_qt=22 tiles (21 full q0=0..160 +
1 peeled ragged tq=4) = 352 blocks.

VERIFY (no device):

    python3 relpos_rowtiled_stream_iron.py -d npu2 -T 172 --tq 8 --kb 43 \
        | grep -iE 'memref|objectfifo|scf.for'

L1 memref allocs are `[KB,DK]` + the `[TQ,*]` scratch, never `[T,DK]`/`[P,DK]`;
`scf.for` count small (loops, not unrolled). Device gate: 3d STEP=8 command,
`rel-L2 <= 0.08 AND corr >= 0.99`.
