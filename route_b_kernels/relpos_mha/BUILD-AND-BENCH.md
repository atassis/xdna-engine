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

## 3. DEVICE GATE / OPEN ITEM (the one thing the toolchain must close)

The kernel arithmetic and the row-tiling index math are proven (section 0). The
remaining device-side item is purely a DATAFLOW question and needs the toolchain to
resolve (it cannot be numpy-validated):

- `relpos_rowtiled_iron.py` as written forwards the WHOLE `kpv` (k+p+V) buffer to
  L1. That fits L1 only up to a MODERATE T -- at T=172, `p` alone is 86 KB > 64 KB
  L1, so the whole-forward overflows exactly like step-5 did. The Tq-tiling already
  fixed the SCORE scratch; the resident kpv is the last L1 pressure.
- To reach T=172, kpv must be STREAMED from the MemTile in KEY-BLOCKS (never fully
  L1-resident), with the L2 buffer REPLAYED once per query tile. The API is
  confirmed present: `aie.iron.ObjectFifo(..., repeat_count=n_qtiles)` and
  `.forward(..., repeat_count=...)` -- "replays the MemTile buffer descriptor N
  times without a new L3 DMA" (python/iron/dataflow/objectfifo.py). This needs the
  bake decomposed into per-block kernels (dot-matmul-into-column-slice of AC/BD,
  softmax-over-full-row, ctx-accumulate-over-key-block) driven from an
  acquire/release core loop -- structurally the whole_array_modal core_fn pattern.
- SMALLEST NEXT PROBE (on the toolchain box, no bench): generate the STEP=6 MLIR at
  T=172 and inspect the L1 allocation --
    `python3 relpos_rowtiled_iron.py -d npu2 -T 172 | grep -i 'memref\|buffer'`
  If the kpv L1 buffer is the overflow, wire kpv as a MemTile split into `T/kb`
  `[kb,DK]` sub-objects with `repeat_count = ceil(T/TQ)` and confirm the generated
  `aie.memtile_dma` replays; then re-gate section 2 at `--synth-T 172`.

Until that lands, STEP=6 gates on-device at the T where kpv fits L1 (proves the
row-tiled arithmetic + GLOBAL-index rel_shift on silicon); the block-streamed kpv
is the remaining scaling wire-up, not an arithmetic risk.
