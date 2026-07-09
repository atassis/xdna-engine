# The NPU is data-movement-bound, not compute-bound

The single most useful thing I learned building an inference engine on the AMD
XDNA2 NPU is this: the pipeline is limited by moving bytes and reloading array
shapes, not by arithmetic. Compute is essentially free. If you optimize FLOPs
you are optimizing the wrong axis. You optimize bytes-moved and shape-reloads.

This note lays out the thesis, the measurements that back it, and the levers
that follow from it.

## The one thesis that organizes everything

Compute is essentially free; the cost is moving bytes across LPDDR and reloading
array shapes between operations. I arrived at this from three independent
measurements that all point the same way.

| evidence | what it shows |
|---|---|
| Encoder context-switch cost | ~340 of ~427 ms of the encoder NPU pool is pure ~2.67 ms shape-reload, not compute (~0.1-0.3 ms) or DMA |
| On-NPU decode attention | one fused decode-attention dispatch is ~759 us, with compute approximately equal to the DMA floor (delta ~4 us) -> roughly 100% data-movement; the attention math is free, the cost is re-streaming 16.8 MB/token |
| Per-op occupancy trace | the #1 latency sink does 0% vector compute; attention GEMVs are 56-81% stall-bound; the NPU compute array is ~84% idle |
| Bandwidth ratio | on-chip SRAM ~800 GB/s vs LPDDR ~120 GB/s datasheet -> residency is a large bandwidth multiplier |

So the NPU's 32 compute tiles (against a 20-core CPU) sit ~84% idle - not
because the hardware is weak, but because I feed it from DRAM and reload shapes
between ops. "CPU parity" at batch=1 is a data-movement tie, not a compute
verdict. Raise data-movement efficiency and the 32-core compute headroom becomes
real.

## We are overhead-bound, not bandwidth-bound

There are two floors, and the interesting one is far from where we sit.

- Physics floor (LPDDR bandwidth): decode must read the model's weights per
  token. At the 120 GB/s datasheet number that is a ~1.65 ms/token floor. But
  the datasheet number is not achievable on the shim-DMA path: measured
  achievable LPDDR read bandwidth is ~47-57 GB/s (pure DMA saturates around
  47 GB/s past a few columns; the encoder GEMM path reaches ~57 GB/s). So the
  real floor is ~2x higher, roughly 3.5-4.2 ms/token. The bytes are the bytes;
  no loop trick changes them.
- Where we actually are: measured decode is ~75 ms/token, about 18-21x above
  that measured-achievable floor.

That gap is the point. Decode today is overhead-bound - dispatch boundaries,
inter-op transitions, on-chip choreography - not bandwidth-bound. The win is not
"NPU faster than CPU." The win is deleting the choreography overhead that per-op
reference kernels leave on the table. That is where the ~10x-plus of headroom
lives, and it is engineering headroom, not physics.

The attack order follows directly:

- Phase 1 (now, where the win is): eliminate inter-op and dispatch overhead.
- Phase 2 (only once phase 1 nears the floor): lower the floor itself by moving
  fewer bytes - quantize to int8/int4, keep small models fully resident, use
  sparse/windowed attention, speculative decode, vocab pruning. Optimizing
  bandwidth before overhead is premature when you are ~20x away from it
  mattering.

The only true physics wall is LPDDR bandwidth, and we are far from it. Every
other "wall" is an engineering wall. Treat a "can't" on an engineering wall as a
signal to find the next angle, not as a verdict.

### Where the on-chip dispatch time actually goes

Strict per-phase timing of the fused decode (quiesced box) breaks the per-token
cost down as:

- on-chip dispatch: 47.95 ms (91%)
- output projection / lm_head: 4.65 ms (8.8%)
- all host<->NPU synchronization (sync in/out, small readback, patch, embed):
  ~0.02 ms total, roughly 0%

This falsifies the intuitive lever. "Close the per-token host<->NPU loop" is not
the win - that loop is already closed. The resident design pays for a token with
two scratchpad writes, one dispatch, and a readback, all in microseconds. The
overhead to delete is on-chip, between operations, not across the host boundary.

So the ranked levers are:

1. Reduce the on-chip dispatch (47.95 ms, 91%, ~29x above the datasheet
   bandwidth floor). It is on-chip inter-op overhead plus per-op stalls
   (51 ops/layer x 12 layers; the V-transpose does 0% compute, the GEMVs are
   56-81% stall). Levers: micro-op fusion, an on-chip dataflow loop that kills
   inter-op boundaries, and residency.
2. Output projection / lm_head (8.8%) - minor; on-NPU paths already exist.
3. A depth-independent block-loop for build generality and minimum resident
   footprint, plus the on-chip dataflow loop that removes inter-op boundaries
   within a single dispatch.

The method that caught the mis-attribution: build a strict-evidence dataflow
map of where cycles go before attacking anything.

### The on-chip loop primitive

The only loop you can author directly on this NPU is the DMA descriptor
(BD) chain's next-pointer plus repeat-count - a hardware "goto" living in the
descriptor chain, below the firmware. Two tempting alternatives are dead ends and
not worth re-litigating: there is no instruction-stream branch (the instruction
stream is a flat config-write language with no branch opcode and no access to the
firmware program counter), and the microcontroller/firmware loop is closed. The
usable ladder is: host-driven block-loop (de-risked) -> BD-chain dataflow loop
(the loop pushed into the DMA engine: a layer-strided repeating descriptor plus
steady-state cores plus a recurring activation buffer collapse to one dispatch,
a one-layer resident image, no host hop).

Honest positioning: fused autoregressive LLM decode on XDNA2 with open kernels
already exists in the ecosystem (host-driven per-layer multi-launch designs
running small Llama / SmolLM2 / int4-AWQ style models). Run-a-small-LLM is not a
differentiator and I do not frame it as novel. The measured delta that is real:
those designs run roughly 33 host dispatches per token with attention on the CPU,
whereas this engine's 12-layer decoder runs the whole decode - attention
included - on-NPU as a resident image at 1.00 dispatch per token. Reaching one
dispatch per token requires the whole model as a single monolith image, which
runs into a build-scale wall; the multi-launch designs dodge it, and I could
build the monolith only because the build-scale toolchain was fixed. Caveats I
keep attached to that result: a 12-layer ASR decoder is not a 16-24 layer LLM, so
monolith scaling to LLM depth is unproven (the block-loop is the hedge); and one
dispatch per token won the host shuffle but moved the remaining battle on-chip,
where ~91% of the time now lives.

## Per-op occupancy: a different prescription per op

From an isolated-op hardware trace on a representative tile. Here "compute" means
vector instructions present, "stall" is memory-stall plus lock-stall coverage,
and "DMA" is port-running coverage. Caveat: these are isolated standalone ops,
not the in-fused context, so the inter-op overhead itself is not captured here.

| op class | vector compute | DMA | stall | reading |
|---|---|---|---|---|
| V-transpose (#1 sink) | 0% | 40-50% | 60-79% | pure data shuffle, no math - eliminating it loses nothing |
| attention GEMV (score/ctx) | present | 10-36% | 56-81% | memory-dependency stall-bound (scattered K/V reads), not DMA-saturated - a layout/residency fix |
| projection GEMV (qkv/proj/fc1/fc2) | ~90% | ~90% | 24-51% | DMA-bandwidth-bound but well overlapped (near-optimal) - only fewer bytes help |
| softmax | 82-92% | 13-16% | 7-18% | compute-bound-ish, well pipelined - fine |
| LayerNorm / GELU | 36-53% | 2-6% | 43-63% | small; half compute, half startup stall |

The actionable split, and why "it's all bandwidth-bound" is too coarse:

- Transposes: remove the op. Zero compute value.
- Attention GEMVs: fix the access pattern / make K/V resident. This is a stall,
  not a bandwidth wall.
- Projection GEMVs: quantize. Fewer bytes is the only thing that helps a
  well-overlapped bandwidth-bound op.
- Softmax and LayerNorm: already fine.

## The two dominant data-movement costs and their levers

### Cost A: shape-reloads (the ~2.67 ms context-switch)

The single biggest encoder latency term. Decode already solved this: one
constant resident image, zero context switches across tokens. The encoder has
not - it switches image shapes per matmul/layer.

Lever: apply decode's pattern to the encoder - a resident, small-shape-set
(ideally single-shape) multi-layer encoder image. The goal is to minimize the
number of distinct shapes, not the dispatch count. Decode is the existence proof
that this is possible; it is a multi-month effort.

### Cost B: byte streaming from LPDDR

Weights, activations, and KV get re-streamed from DRAM on every dispatch or
token. The on-NPU attention result shows this is the cost - the compute is free.
On-chip memory is far faster than LPDDR, so the prize is keeping bytes on-chip
and moving fewer of them. The master lever is to eliminate bytes, not accept
them:

1. Resident weights/KV - keep them on-chip across tokens rather than
   re-streaming. Blocked by capacity (weights per token exceed on-chip SRAM), so
   it needs streaming-under-compute, a smaller model, or quantization.
2. Quantization to int8/int4 - in the bandwidth-bound regime (batched decode,
   encoder weight-streaming) this is close to a linear speedup on the dominant
   cost; int4 weights mean 4x less streaming. Earlier int8 attempts lost only on
   host-side quant/dequant; a resident int8/int4 dataflow (quantize once, stay
   low-precision, dequantize only at the f32 norm islands) wins there. Regime
   caveat: at M=1 single-stream decode this lever is latency-negative, because
   decode is overhead/op-count-bound rather than bandwidth-bound - the saved
   bytes are not on the critical path, while the int8 scale-muls and GEMV
   widening add on-chip ops (measured +11-16% slower). So int8/int4 is the
   highest-leverage byte lever for throughput and energy, not for M=1 latency.
3. Batch B>=128 - amortize the constant weight read across streams. This is the
   throughput win, and where the compute headroom finally shows.
4. Sparse / windowed attention - ASR alignment is roughly monotonic, so there is
   no need to read all 1500 encoder frames per token. Read less K/V.
   Algorithmic, accuracy-gated.
5. Speculative / parallel decode - verify K tokens per weight-read to amortize
   the decode weight stream.
6. Vocab pruning / hierarchical lm_head - do not compute all 51865 logits when
   you only need argmax or top-k.

## Does this generalize across modalities? Yes

The data-movement overhead is structural to the "GEMMs plus host glue" design,
not specific to any one model, so the fix is horizontal. Two generic resident
primitives cover essentially everything worth running:

- Fused resident transformer block (LN -> QKV -> attention -> proj -> LN -> FFN
  -> residual). Serves the ASR encoder and decoder, small LLMs, ViT/CLIP, and
  protein models - all the same structure, only the dimensions and sequence
  length differ. This is the M-stationary GEMM plus fused-epilogue lever.
- Fused resident conv block (conv -> norm -> activation -> pool). Serves vision
  conv nets.

The hard constraint: these must be generic primitives, not per-model hand-fused
kernels. Per-model kernels would rebuild a closed catalog and erode the
run-any-model property that is the whole point. The data-movement frame is
itself the generalization - "minimize bytes plus reloads" is model-agnostic.

## Engineering walls vs the one physics wall

| wall | physics or engineering | implication |
|---|---|---|
| ~2.67 ms context-switch | engineering (shape-reload) | minimize distinct shapes -> resident constant image (decode did it) |
| 2-input-DMA per tile | hardware, for the naive all-to-all DMA mapping | GEMM->GEMM single-dispatch fusion via a second DMA input is out; the softmax-free FFN/GEMM K-reduce escapes it via the on-chip cascade accumulator (core-to-core, not a DMA channel); still blocks fused attention, since a cascade cannot span a softmax |
| MemTile 512 KB / L1 64 KB | hardware capacity | big intermediates (FFN hidden ~3 MB) cannot stay resident un-tiled -> tile along M |
| N-stationary cannot fuse reductions | engineering (kernel design) | an M-stationary GEMM fixes it, or cascade-reduce across columns |
| no NPU FFT / depthwise-conv kernel | engineering (missing kernel) | log-mel and conv stay on host (small) or need custom kernels |
| LPDDR bandwidth | physics (the DRAM) | the only true wall - beat it by moving fewer bytes, not faster ones |

Only LPDDR bandwidth is physics, and even that is beaten by moving fewer bytes.
Everything else is engineering: multi-week to multi-month, but not impossible.

## What I still cannot see

I try not to conclude from partial data. The honest gaps, and how each closes:

1. In-fused inter-op breakdown - the large fraction of the M=1 decode dispatch
   that is not op compute. Is it DMA wait, stream stall, or dependency? Needs an
   in-fused trace that spans the fused multi-device structure; today's per-device
   trace pass cannot, so I lean on subtraction-attribution.
2. On-chip vs LPDDR achievable bandwidth on this exact silicon - the 800/120
   ratio is datasheet; the achievable read side is already measured at ~47-57
   GB/s, and the on-chip side still wants its own micro-benchmark.
3. True on-chip resident-weight decode for a small model - weights that stay
   on-chip across tokens. Demonstrated nowhere; int4 is needed to fit. Measures
   the residency ceiling.
4. M-stationary GEMM utilization and epilogue fusion at one encoder shape. The
   fused-norm epilogue is the proven part; the M-stationary GEMM itself only
   pays at very large M (measured 4.15x slower at the FFN shape).
5. Whether the advertised hardware softmax/LayerNorm primitives beat our
   kernels. In practice only exp2/tanh/invsqrt/inv primitives exist, and they
   are already used, so there is no dedicated softmax/LN primitive to win with.
6. The encoder floor once the shape-set is minimized - a single-shape encoder
   spike.
7. int4 / resident-quantized dataflow viability - the highest-leverage
   byte-reduction, still unbuilt. The GEMV path is bf16-only, so it needs a new
   kernel; the axis is throughput and energy.

## Bottom line

- "CPU parity, can't improve" is wrong. About 84% of the NPU is idle and the
  bottleneck is data movement, not compute. There is large headroom, gated by
  data-movement engineering, not physics.
- The master frame: count bytes and shape-reloads, not FLOPs. Every lever earns
  its place by how many bytes it removes from LPDDR or how many shape-reloads it
  kills.
- The master lever: do not optimize a bandwidth-bound op - make it not happen
  (resident, quantized, batched, sparse). The V-transpose (0% compute) is the
  proof case: the fastest way to run an op that does no math is to delete it.
- Decode is the existence proof - resident, one constant image. The program is
  to extend that to the encoder and drive the byte count down with quantization
  and residency, generically, across all modalities.
