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
  block. k/p/V staged ONCE in the 512 KB MemTile, streamed to L1 in KB-row
  key-blocks, REPLAYED once per query tile via ObjectFifo `repeat_count`. L1 then
  holds only ONE key-block + the `[TQ,*]` scratch. Same block bricks as STEP=7,
  exposed as `relpos_stream_dot/_softmax/_ctx*` and driven from the IRON Worker's
  acquire/release loop. This adds ONLY the dataflow the golden cannot validate.

### 3a. ObjectFifo topology (2 input DMA channels -- HARD limit)

The NPU2 CORE tile has exactly 2 input (S2MM) + 2 output (MM2S) DMA channels
(`AIE2TargetModel::getNum{Dest,Source}SwitchboxConnections` = 2 for `WireBundle
::DMA` on core tiles; MemTile = 6). STEP=8 uses both inputs + one output:

  Channel A  `of_quv` [TQ,DK] bf16, depth 2 -- per query tile the core acquires
             qu_tile (phase K) then qv_tile (phase P); 2*n_qt blocks, read ONCE
             (no replay). Host packs QUV TILE-INTERLEAVED.
  Channel B  `of_kpv` [KB,DK] bf16, depth 2 -- `of_kpv_l3l2` stages the whole
             padded kpv L3->L2 once; `.forward(obj_type=[KB,DK], repeat_count=n_qt)`
             re-streams it L2->L1 in KB-blocks, n_qt times, WITHOUT re-fetching
             from DDR. Per query tile: n_kb k-blocks, n_pb p-blocks, n_vb V-blocks.
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

### 3e. TWO BUILD PROBES -- the only things not numpy-validatable (STEP=8)

Both are pure toolchain-lowering questions; neither can be closed without emitting
MLIR / building. Characterized, not hand-waved:

- PROBE 1 (scalar kernel args). The block bricks take int32 scalars (tq, kb, j0,
  q0, ncol). `relpos_rowtiled_stream_iron.py` unrolls the core statically (n_qt,
  n_kb, n_pb, n_vb are compile-time) and passes each as a Python int.
  `Kernel.__call__` forwards non-Buffer args untouched to `func.call`
  (python/iron/kernel.py). CONFIRM a Python int materializes as an i32
  `arith.constant` operand; if not, wrap with an explicit `arith.constant`.
- PROBE 2 (resident-L2 replay in blocks). `of_kpv_l3l2` stages the whole padded
  kpv in L2 (one L3->L2 fill); `of_kpv` forwards it to L1 with `obj_type=[KB,DK]`
  (smaller than the source) + `repeat_count=n_qt`. CONFIRM this lowers to an
  `aie.memtile_dma` that keeps kpv resident and emits KB-blocks n_qt times, NOT
  n_qt fresh L3 DMAs. If forward-with-smaller-obj_type + repeat_count does not
  lower that way, fall back to STREAM-A: runtime re-fills kpv blocks per query
  tile (the proven whole_array pattern; correct, same L1 budget, re-fetches kpv
  from DDR each tile -> worse data movement, to be optimized back to replay).

SMALLEST PROBE (toolchain box, no bench):

    python3 relpos_rowtiled_stream_iron.py -d npu2 -T 172 --tq 8 --kb 43 \
        | grep -iE 'memref|memtile_dma|objectfifo|repeat|arith.constant'

Inspect (a) L1 memref allocs are `[KB,DK]` + the `[TQ,*]` scratch and NEVER
`[T,DK]`/`[P,DK]` (proves the byte budget on silicon), (b) the kpv path is one
resident MemTile buffer with a replayed BD (PROBE 2), and (c) the scalar args
appear as `arith.constant` operands to the `func.call`s (PROBE 1).

Until PROBES 1-2 are closed by a build, STEP=7 gates the block-decomposed
arithmetic on silicon (the compute half of STEP=8, kpv-fits-L1); STEP=8 is the
dataflow-only remainder, and the byte budget above proves it FITS once it lowers.
