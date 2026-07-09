# AIE2P hardware brick catalog - the periodic table for building from scratch

The AMD XDNA2 (Strix) NPU is an AIE2P array. Its performance is not a mystery once you
stop thinking in FLOPs and start thinking in *bricks*: the fixed set of hardware
primitives the silicon actually provides, spread across five layers - COMPUTE, MOVEMENT,
MEMORY, ORCHESTRATION, and FORMAT. Each layer has a small number of specialized bricks,
and the recurring, costly mistake is reaching for a *generic* brick where a *specialized*
one exists. Swapping a hand-rolled `mac`+`reduce_add` matmul for the systolic `mmul`
brick, for example, is a single-brick change worth roughly 5x.

This is the complete inventory - every brick, what it does, whether it is being used
optimally, and where it applies - so that designing an algorithm for this hardware is a
*lookup*, not a post-hoc discovery. The other half of the doc is the pick-by-regime rule:
a brick only helps in the regime it was built for, and applying a compute brick to an
overhead-bound decode step buys nothing.

Legend: **OK** = used optimally - **~** = used suboptimally - **X** = unused (left on the table).

## Hardware ground-truth

- Array: **32 compute tiles (8 cols x 4 rows) + 8 MemTiles (1 row) + 8 shim tiles**.
- **L1 = 64 KB/core**, 4 banks, 16 locks, 16 BDs, **2 MM2S + 2 S2MM DMA channels** (the
  2-input-DMA wall). MemTile **L2 = 512 KB** x8 = **4 MB** pool, 8 banks, 64 locks, 48 BDs,
  6+6 channels. **L3 LPDDR ~120 GB/s vs on-chip ~800 GB/s (6.7x)**.
- **Vector reg = 512-bit** (32 bf16 / 64 int8 / 128 int4 lanes). **Accumulator reg =
  2048-bit** (64 accfloat / 64 acc32 / 32 acc64). **Cascade bus = 512-bit/beat**, adjacent
  cores only.
- **Compute peak: 128 bf16 MAC/cyc/core (EMULATED) | 512 bfp16 (TRUE systolic) | 1024
  int8**. bf16 mmul is emulated 32-lane FMA+shuffle; the real 512-MAC array is bfp16/int only.

---

## Layer 1 - COMPUTE (what each core calculates)

Source: `aie_api/{aie.hpp,detail/mmul.hpp,sliding_mul.hpp,accum.hpp}`, the aie2p vector
multiply intrinsics.

| Brick | What | Status | Where / note |
|---|---|---|---|
| **mmul** (systolic matrix) | M*K*N tile/issue, K-sum stays in acc, no reduce | **OK** in the patch-embed and repro paths; GEMV path still **~** | encoder/FFN/prefix GEMM (M>=8). The ~5x win. Batched cascade-FFN GEMM tile is still mac+reduce and remains a candidate. Shapes per format below |
| **mac / mul** (vector FMA) | elementwise into acc | **OK** (norm/bias/act); **~** for matmul (paired w/ reduce_add) | M=1 GEMV is correctly mac; batched should be mmul |
| `aie::accumulate` (col-broadcast) | r outputs from a broadcast, no per-out reduce | n/a | better GEMV than reduce_add but still < mmul |
| **sliding_mul / _sym** (FIR) | sliding-window MAC, K-sum in acc, window reuse | **OK** | `dwconv1d.cc` uses `aie::sliding_mul_ops::mul` (k=5/k=9 Conformer depthwise). The ~Lanes x win is realized |
| **reduce_add** (horizontal) | synthesized: shifts + ones-MAC + extract, **per output** | **~** (the GEMV tail cost) | the mechanism mmul eliminates |
| reduce_add_v | reduce <=4 vectors at once | **X** | cheap mitigation where reductions unavoidable |
| max_cmp / min_cmp | value **+ index mask** co-produced | **X** (scalar argmax scan is deliberate - non-bottleneck, ~6us/core x8 parallel) | **free argmax**, softmax row-max, ReLU=max(x,0) |
| **exp2** (SFU, 16/issue) | only exp on chip; exp(x)=exp2(x*log2e) | **OK**; **~** only in M=1 decode | M=1 decode softmax issues a 16-lane SFU per scalar key (inherent to the online softmax recurrence; M=1 so not the bottleneck). The M=T relative-position MHA vectorizes exp2 over keys - the intended form |
| **tanh** (SFU) | GELU tanh-approx | **OK** | gelu |
| **invsqrt** (SFU, 1 op) | 1/sqrt | **OK** in most LN sites; **~** one residual | `ln_2pass`/`mm_ln_epilogue`/`norm_gemv_prologue` use `aie::invsqrt`. One layernorm-rows site still does `1.0f/aie::sqrt` (2 SFU) -> swap pending |
| inv (1/x) | reciprocal | **~** | softmax `1.0f/g` scalar divide -> aie::inv |
| **linear_approx** (LUT, 4/cyc) | piecewise-linear any function | **X** | erf/sigmoid/log/custom act the SFU lacks |
| **parallel_lookup** (gather LUT) | table gather <=32 lanes | **X** | embedding lookup, dequant codebook, RoPE tables |
| SRS `to_vector<T>(shift)` | acc->vec drain w/ requant shift | **OK** | every drain; shift = requant scale |
| UPS `from_vector` | vec->acc widen | **OK** | bf16->f32 widen |
| to_fixed/to_float, pack/unpack | quant/dequant, int-width step | **X** | int8/int4 activation quant glue |
| add/sub/mul, select/clamp/abs/neg, shift/shuffle | elementwise + lane-routing | add/sub/mul **OK**; rest **X** | shuffle = on-chip realign (vs re-DMA); clamp = logit cap / int sat |

mmul tile shapes by format: bf16 `8x8x8`(emul)/`4x8x8`/`8x1x8`; **bfp16 `8x8x8`/`8x8x16`
(TRUE, ~4x)**; int8 `8x8x8`/`4x8x8` + sparse `4x16x8`; **int8xint4 `4x16x16` (1024
MAC/issue, densest)**; int16 small-K.

## Layer 2 - MOVEMENT + INTERCONNECT (how data flows)

Source: the AIE / AIEX / AIR dialect op definitions.

| Brick | What | Status | Where / note |
|---|---|---|---|
| `dma_bd` (n-D strided + pad) | 3-D core / 4-D memtile strided slice, zero-pad gather | **OK** | transpose/relayout/im2col FOR FREE in the DMA |
| **BD-chain loop** (`next_bd`+`repeat_count`) | the ONLY on-chip hardware loop (BD next-ptr goto) | **X** | **#1 MOVEMENT LEVER: whole decode layer as ONE dispatch -> kills the ~91% inter-op dispatch overhead** |
| bd_chain templates (`dma_start_bd_chain`) | reusable BD chain re-bound to rotating buffers | **X** | ping-pong / resident double-buffering |
| dma_memcpy_nd (4-D, host) | host<->shim staging | **~** | one issued per op instead of folding into a resident chain |
| **cascade put/get** (MCD/SCD, 512-bit) | accumulator core->core, adjacent, NOT a DMA channel | **~ (dumb buffer-transport)** | **fused mmul+cascade K-reduce (~5-10x); escapes the 2-input-DMA wall.** Currently streams buffers + software-add instead |
| circuit `connect`/`flow` (spatial) | dedicated contention-free lane between tiles | **X** | persistent SPATIAL dataflow graph (the systolic-pipeline idea) |
| packet_flow (5-bit ID routing) | many low-bw flows time-share a link | **X** | when you can't afford a circuit per pair |
| multicast / broadcast | 1->N on the fabric (`broadcast_shape` -> multi-consumer objectFIFO) | **OK (available, in use)** | cascade-FFN uses it for `inX`. NOT the weight-restream lever - that's RESIDENCY (BD-chain) |
| get/put_stream (core<->fabric) | direct stream port, bypass DMA/locks | **X** | fine-grained streaming |
| **objectFIFO** (DMA+lock circular buf) | the primary movement abstraction | **OK** point-to-point; **~** driven op-by-op | rarely exploits its repeat_count/link/broadcast -> degenerates to 1 DMA/op |
| objectfifo.link (memtile relay) | join/distribute/broadcast through L2 | **~** plain staging | resident relayout+broadcast hub (kill inter-op host hops) |

## Layer 3 - MEMORY (where data lives)

Source: the target model + `aie_api/{accum,vector,ld_st}.hpp`.

| Brick | What | Status | Where / note |
|---|---|---|---|
| L1 64KB/core (4 banks) | core working set | **OK** (N-stationary tiling fits it) | overflow = build fail; M-stationary correctly rejected (~4.15x slower) |
| L2 MemTile 512KB x8 = 4MB | staging / resident-intermediate home | **~ (forced)** | FFN 3MB intermediate > 512KB -> must tile / DDR round-trip |
| L3 LPDDR ~120 GB/s | the only physics wall | **OK as the lens** | decode sits far above the bandwidth floor = engineering overhead |
| Accumulator reg 2048-bit (64 accfloat) | MAC/mmul partial target, f32 | **OK** | the cascade carries this; bias-as-acc-init unused |
| Vector reg 512-bit | SIMD operands | **OK**; **bfp16ebs8 type X** | block-FP vector type unused |
| load_v/store_v (alignment 16/32/64B) | vec mem<->reg | **OK** | **footgun**: misaligned aligned-load silently truncates address (no fault, asserts off) - watch in new strided kernels |

## Layer 4 - ORCHESTRATION (how it's synchronized + controlled)

Source: the lock/objectfifo + runtime-sequence op definitions, plus the decode generator.

| Brick | What | Status | Where / note |
|---|---|---|---|
| locks (counting semaphores, AcquireGreaterEqual/Release = P/V) | buffer credits | **OK** (via objectFIFO) | 16/core, 64/memtile, max 63 |
| **objectFIFO depth** | N buffers = compute/DMA overlap | **~ the known miss** | depth-1 (no overlap) hit in cascade-FFN; recovering depth-2 without the w1+w2 over-credit needs a per-sub-stream knob the AIR Python API lacks -> C++ pass / upstream |
| runtime_sequence (npu-insts) | the host control program | **OK** (1 fused 12-layer seq) | flat/loop-incapable -> the unroll-at-scale build wall |
| RTP scratchpad | per-token params, constant ELF | **OK** | kv_off + sm_mask, 2 words/token (replaced a 27MB ELF patch) |
| load_pdi (cascade re-arm) | device-image / cascade reset | **OK** (0 reloads/token) | the cascade-reentrancy bug lives in the AIR lowering path we upstream |
| dispatch (1 ELF/token) | whole decoder = 1 dispatch | **OK** | ~91% of per-token is intra-dispatch inter-op overhead -> on-chip cascade dataflow next |
| **trace events** | per-op compute/DMA/stall counters | **X** | **the per-op occupancy TOOL - build it to gate every lever** |

## Layer 5 - FORMAT (number representation)

Source: the mmul/accum/block_vector detail headers + int8 accuracy logs. WER numbers are
against a bf16 baseline of 0.1172.

| Format | repr | mmul tiles / throughput | accuracy (WER, base 0.1172) | Status |
|---|---|---|---|---|
| fp32 | emulated via bf16 | 4x8x4 only, tiny | range only | **OK as accum** |
| **bf16** | 1/8/7 | 8x8x8 emul, **128 MAC/cyc** | base 0.1172 | **OK** default; GEMM path **~** (not in mmul) |
| **bfp16ebs8/16** | block-FP, 8 share 1 exp | **8x8x8/8x8x16 TRUE systolic, ~4x bf16** | block-exp loss, untested | **X - top untapped compute lever** (encoder GEMM, WER-gate it) |
| bf16 x bfp16 | mixed | 8x8x8, 512/issue | B-side block loss | **X** (bf16 act x bfp16 weight, no act-quant) |
| **int8** | 8b + scale | 8x8x8 + sparse, **1024 MAC/cyc ~8x** | sweet 0.1245 / all 0.1319 (compounds) | **BUILT, gated OFF - latency-negative @M=1** (adds ops; energy-only) |
| int16 | 16b | small-K, low tput | hi-precision int | **X** (niche) |
| int4 | 4b + group scale/zp | (paired only) | AWQ, untested | **X** except AWQ ref; decode weight ENERGY lever |
| **int8 x int4** | 8b x 4b + AWQ | **4x16x16, 1024 MAC/issue densest** | AWQ, untested | **X** (ref dequants to bf16 instead) |

Precision rule: accumulate f32 (accfloat) / acc32; narrow to bf16 only in the epilogue;
fold quant scales POST-mmul at GROUP granularity (AWQ); set `conv_even` rounding for bfp16
(else ~-0.07/K bias).

---

## What is left on the table - ranked across all layers

1. **BD-chain on-chip loop** (MOVEMENT) - whole decode layer as ONE dispatch; attacks the
   ~91% / ~48ms-per-token inter-op dispatch overhead. The single biggest brick-miss.
2. **mmul for batched GEMM** (COMPUTE) - ~5x proven; encoder/FFN/prefill/vision (M>=8
   compute-bound only).
3. **Fused cascade-accumulator** (MOVEMENT) - ~5-10x on multi-core K-reduction; the bus is
   currently used as dumb transport.
4. **bfp16 systolic** (FORMAT) - ~4x the bf16 array for compute-bound GEMM, WER-gated; the
   top untapped *format*.
5. **Residency across dispatches** (MOVEMENT) - stop re-streaming weights per op; this is
   the BD-chain loop (#1), NOT broadcast (broadcast already works and isn't the lever).
6. **objectFIFO depth-2 overlap recovery** (ORCH) - recover weight DMA/compute overlap
   (needs an AIR knob).
7. **Trace-event per-op occupancy harness** (ORCH) - the tool that gates every lever
   (compute vs DMA vs stall).
8. **Small COMPUTE swaps**: invsqrt (LN 2->1 SFU op, one residual site), max_cmp (free
   argmax where it is the bottleneck), LUT linear_approx (erf/sigmoid), parallel_lookup
   (embedding/RoPE), reduce_add_v, vectorize the M=1 decode exp2.
9. **int8 / int8xint4** (FORMAT) - energy levers for the M=1 weight stream
   (latency-negative; energy-gated).

## Where the bricks come from - three tiers

A useful way to reason about what can and cannot be improved is to split the stack into
three tiers, because the tier determines whether the fix is in my kernels, in reusing
reference kernels, or in the toolchain.

**Tier 1 - HARDWARE bricks (the silicon: mmul array, cascade ports, DMA engines, SFU,
memory). FIXED - they can only be USED, not improved.** The Peano backend
(`__builtin_aie2p_*` intrinsics) and `aie_api` expose every one of them; nothing in the
catalog above is missing from the tooling. The gap here was pure non-use - mmul,
sliding_mul, cascade-accumulator, bfp16, and the BD-chain loop all exist, and generic
substitutes shipped in their place. You do not "make a better mmul"; it is the array. The
fix is in the kernels.

**Tier 2 - REFERENCE KERNELS (the `aie_kernels/aie2p/{mm,mha,conv2dk1}.cc` set and the
programming examples). GOOD - mmul-based and vendor-tuned.** Hand-written kernels that
reinvented these tended to reinvent them *worse*. The fix is to REUSE them, not to
"improve" them. Where a kernel genuinely must differ - the fused cascade-FFN - copy the
idiom, not the exact kernel.

**Tier 3 - TOOLCHAIN EXPOSURE (the dialects, lowerings, passes, and abstractions that let
you AUTHOR the bricks). This is where the real gaps and bugs live - and where improvements
are durable.** Confirmed:

- **`npu_cascade` lowers to buffer-transport + software-add, NOT the fused
  mmul-accumulator form** -> adding a fused-cascade-accumulator lowering makes the cascade
  brick actually usable.
- **objectFIFO has no per-sub-stream DEPTH knob** in the Python API -> a C++ pass to
  recover depth-2 overlap without the over-credit merge.
- **BD-chain on-chip loop:** the hardware brick exists, but in-tree authoring is limited
  (npu-insts is flat / loop-incapable) -> exposing it is a toolchain opportunity. The
  dynamic-runtime-sequences work upstream is already moving in this direction.
- **Compiler bugs found + fixed/staged:** a cascade re-entrancy bug, an AIRDmaToChannel
  SIGSEGV, quadratic aiecc passes, and an int->float->int Peano miscompile. Each is a
  brick the toolchain exposed incorrectly until it was fixed.

**Method:** Tier-3 gaps are found by *reading the source to deduce suboptimality*, not by
surveying issue trackers - the device is not needed to *discover* a structural or
algorithmic problem, only to *validate* a fix later (often just a lit test). Every gap so
far came from reading source: the quadratic aiecc passes, the `npu_cascade`
buffer-copy+software-add lowering, the AIRDmaToChannel SIGSEGV, the int->float->int
miscompile, the scalar `dwconv1d` vs `sliding_mul`. Issue trackers are a reactive
secondary cross-check to avoid duplication; the edge is proactive code-reading.

**Net:** the hardware bricks and good reference kernels are provided - the job there is to
USE and REUSE them. The durable edge is Tier 3: the toolchain does not always *expose* the
bricks well (fused cascade, depth knob, on-chip loop) or *correctly* (the bugs), and
improving that exposure both unblocks the kernels and is the high-value upstream
contribution. So: do not "replace the hardware bricks" - USE them, REUSE the good kernels,
and IMPROVE the toolchain exposure where it fails the bricks.

## Regime rule (so a brick is not mis-applied - the mmul lesson)

COMPUTE bricks (mmul, bfp16, sliding_mul, int8-tile) win ONLY when the op is
**compute-bound** = M>=8 batched (encoder GEMM, prefill, speculative-verify, vision conv).
At **M=1 decode** the levers are MOVEMENT (BD-chain loop, cascade, broadcast,
kill-transpose) + op-count, and FORMAT only for ENERGY - NOT faster MACs. Gate every
COMPUTE/FORMAT win on a per-op compute-vs-DMA occupancy measurement (the trace-event
harness). A hand-written *compute* kernel can be ~18x off peak while looking fine to the
data-movement lens, so the two lenses are complementary: count bytes moved and
shape-reloads for the movement picture, and % of the 128 bf16 MAC/cyc/core peak for the
compute picture.
