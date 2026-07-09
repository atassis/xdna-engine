# Execution graph: which hardware primitive to use at every node

This is a node-by-node plan for running transformer and vision models on the AMD
XDNA2 NPU: for each step of every model, the optimal compute primitive, the data
layout, the on-chip dataflow, and the correctness gate. The point is that kernel
work becomes a lookup rather than a rediscovery.

The organizing insight is that the M dimension (batch or sequence length) sets the
regime, and the regime sets the primitive. Everything else follows from that one
decision.

## 0. The organizing principle - M sets the regime sets the primitive

| regime | who | bound by | primitive policy |
|---|---|---|---|
| **M >= 8 (batched)** | encoder (seq=1500), vision (patches), prefill, lm_head-as-GEMM, B>=128 decode | COMPUTE (fillable matmul array) | **`aie::mmul`** family: 8x8x8 + 2x2 register-block + output-stationary + pre-tiled DMA layout + fused epilogue (bias/act/norm) + on-chip cascade-accumulator K-reduction + resident intermediate (no host round-trip) |
| **M = 1 (autoregressive)** | Whisper/LLM decode token step | OVERHEAD (on-chip dispatch ~78-95%) + LPDDR weight stream; NOT compute | **GEMV** (mmul cannot fill the tile); levers = op-count reduction (fuse/eliminate launches), residency (weights/KV/encoder-K-V stay on-chip), kill the transpose (0% compute), quantization int8/int4 for ENERGY (latency-negative at M=1) |

At M >= 8 the matmul array is the right hammer: `aie::mmul` is about 5x faster than
a hand-written dot product, measured on silicon. At M = 1 the array cannot be filled
(you cannot pack an 8x8x8 tile from a single row), the step is dominated by dispatch
overhead and weight streaming, and the levers move entirely to data movement and
op-count.

A subtle but load-bearing point about the current codebase: every matmul that is
hand-written here (cascade-FFN, decode GEMV, mha_decode) uses `aie::mac` + `reduce_add`.
`aie::mmul` lives only in the vendored AMD kernels (`mm.cc`, `mha.cc`, `conv2dk1`).
That is exactly why the encoder's whole_array GEMMs are already fast and the custom
kernels are not - the encoder inherited the systolic primitive, the custom kernels
did not.

### Corollary catalog (use this as the lookup)

| op-class | M>=8 primitive | M=1 primitive |
|---|---|---|
| linear / GEMM (qkv, proj, fc1, fc2, lm_head) | `mmul` + pre-tiled + fused epilogue | GEMV, weight-resident, fewer launches |
| multi-core K-reduction (split-K GEMM, fc2) | hardware cascade-accumulator (`get_scd`/`put_mcd` into the mmul acc) - NEVER host-f32 or buffer-passing | single-core or cascade-acc; usually not split at M=1 |
| activation (GELU/SiLU/softmax-tail) | fuse into the GEMM `to_vector` epilogue | cheap on-chip ChanneledUnary, fuse into prior op |
| norm (LN/RMS) | fuse as GEMV/GEMM prologue or epilogue; affine folds into the next/prev weight | on-chip, affine folded into weights |
| attention (MHA) | flash-attention mmul (QK^T, softmax, PV all mmul) | GEMV flash (head-batched), KV-resident, windowed/sparse later |
| transpose / reshape | eliminate via layout (produce the next op's required layout directly) | eliminate (#1 decode sink, 0% compute) |
| conv | im2col -> `mmul` GEMM (patch-embed bakes to GEMM); depthwise-2D = missing kernel | n/a |
| bias / residual add | fold into GEMM epilogue / K-aug weight column | fuse into adjacent op (bias-fold done) |
| argmax / lm_head tail | GEMV vocab-as-M + on-chip argmax | same |

## 1. Correctness contract (the "byte-to-byte" reframe)

A faster kernel changes float accumulation order (mmul tiles vs dot-product,
cascade-accumulate vs host-f32, bf16 vs bfp16), and float add is non-associative,
so a rewrite is almost never bit-identical. Demanding bit-identity would forbid the
optimization outright. The achievable, rigorous contract is:

- **Per-node golden gate:** a numpy/torch bf16-faithful golden, metric rel-L2
  <= 0.08 (standard), < 0.02 (tight bf16-GEMM bar), or corr >= 0.99 (cascade ELF).
  Build it with a reference at the exact node shape.
- **End-to-end accuracy backstop:** WER 0.1172 on the M=1 decode, plus the
  automatable argmax-token parity surrogate (16/16 identical). Every variant is
  WER-gated.
- **Weight-load gate:** weight-arena parity max-rel-err < 5e-2 (bf16 floor ~3.89e-3).
- **Precision is a per-node POLICY, not a default:** bf16 (emulated mmul) where we
  want parity; bfp16 (true 512-MAC systolic) or int8/int4 only where the WER gate
  says the accuracy cost is affordable (decode weights are an energy lever; encoder
  attention is WER-expensive in int8).

The artifact is therefore a correctness-gated re-architecture, not a bit-exact
transliteration.

## 2. ENCODER execution graph (Whisper-small, M=1500, 12 layers) - the batched/mmul regime

The 6 GEMMs (q/k/v/out/fc1/fc2) already use `mmul` via the vendored whole_array
`mm.cc`, so the encoder GEMM compute is fine. The cost is host glue, non-resident
intermediates, and host-f32 K-reduction: 324 dispatches, 69.8% marshaling. So the
encoder re-architecture is about DATAFLOW (residency and fusion), not the GEMM
kernel.

| node | now | target primitive + dataflow | gate | ROI / status |
|---|---|---|---|---|
| conv stem | host im2col+GELU | im2col -> mmul band-GEMM (built), GELU fused | rel-L2 vs golden | med; built, gate it on |
| LN1/LN2/ln_post | host f32 | on-chip norm prologue fused into the next GEMV/GEMM; affine folded into weight | rel-L2 | high (kills host hops) |
| q/k/v/out/fc1 GEMM | mmul, 36 disp each, host bias/epilogue | keep mmul; fuse bias+act into the to_vector epilogue; resident bf16 ln1 shared across q/k/v (-11ms) | rel-L2 < 0.02 | high; partially built |
| **fc1->fc2 intermediate** | [1500,3072] round-trips to HOST (+host GELU) = ~88ms | resident intermediate: keep it in a device BO (or on-chip), GELU fused on-chip; two mmul GEMMs sharing the mmul-tiled layout (do not re-tile) | rel-L2; WER | **#1 ENCODER LEVER**, ~88ms; mostly plumbing |
| **fc2 K-split reduction** | mmul partials + host f32 accumulate (4x) | on-chip cascade-accumulator K-reduction (`get_scd`/`put_mcd` into the mmul acc) - removes the host accumulate and the 60ms fc2 marshaling | rel-L2; WER | high; needs the fused-cascade kernel |
| MHA | host f32 (default) / StaticMHA gated | flash-attention mmul resident (StaticMHA exists, 1 disp/layer); bf16 layer-cap for WER | rel-L2 < 0.02; WER | high; built, WER-gate the depth |
| residual adds | host | on-chip, fused into the epilogue of the producing GEMM | rel-L2 | med |

**Encoder end-state:** a resident transformer block -
ln -> qkv(mmul) -> flash-attn(mmul) -> proj -> ln -> fc1(mmul) -> GELU ->
fc2(mmul, cascade-acc K-reduce) -> residual, with activations staying on-chip or
device-resident across ops and the host doing I/O only. This generic resident block
is the moat. The cascade-FFN work is the fc1->fc2 sub-slice of this, and its lesson
(honor the hardware: mmul + cascade-acc) is exactly this end-state's target.

## 3. DECODE execution graph (Whisper-small, M=1, 12 layers, 51 ops/layer) - the GEMV/overhead regime

Current state: 1 resident ELF, 1 dispatch/token, all ops are GEMV or elementwise
(no mmul, which is correct at M=1). 95% on-chip dispatch overhead. mmul is NOT the
lever here. The levers are:

| node-class | now | target lever | gate | ROI / status |
|---|---|---|---|---|
| op-count (51/layer) | bias-fold + GELU-fuse shipped (32/layer, ~12% faster) | continue the fusion ladder -> ~16/layer (residual(+)LN fuse) | WER | high; partially done |
| **V-transpose (#1 sink, 0% compute, 37% op-cycles)** | per-head un-batched Transpose, re-transposes FIXED encoder V every token | eliminate: store encoder V pre-transposed once (cross), incremental compute-transpose (self); cross-elim validated -5.6% | argmax parity | **#1 DECODE LEVER**; cross done, self needs a layout/kernel change to clear the DMA wall |
| attention GEMV (56-81% stall) | head-batched GEMV | KV-resident layout; windowed/sparse attention (algorithmic, WER-gated) | WER | med-high |
| weight stream (fc1/fc2/qkv GEMV) | bf16 GEMV, ~198MB/token | residency (cannot fit) + int8/int4 for ENERGY (latency-negative at M=1) | WER + energy | energy-only at M=1 |
| proj_out + ln_post + argmax | DONE on-NPU (GEMV vocab-as-M + fused argmax) | -- | WER 0.1172 exact, argmax parity | DONE |

**Decode end-state:** the resident ELF is already near the right shape; the remaining
wins are op-count (fuse), kill-the-transpose, and (energy) quantization - NOT a GEMM
rewrite. Honoring the hardware here means honoring the M=1 reality: it is
overhead-bound, so delete launches and bytes rather than chasing the matmul array.

## 4. lm_head / vision / other

- **lm_head:** DONE (GEMV vocab-as-M 52224x832, K-aug bias+ln fold, on-chip argmax).
  The vocab-as-M GEMV (not GEMM) is forced by the AIE shim 2^20 DMA-stride limit at
  N > ~3840 - a real layout constraint to remember.
- **Vision (ViT/DINOv2/CLIP/ResNet):** all conv lowers to im2col -> the one mmul GEMM
  (patch-embed bakes to GEMM at convert time; ResNet strided conv via runtime im2col).
  Same mmul primitive as the encoder. One real gap: depthwise-2D conv has no kernel
  anywhere, which gates MobileNet/EfficientNet/YOLO plus the Parakeet/Conformer conv
  module (depthwise k9). Write it as a dedicated kernel (not im2col - depthwise is
  bandwidth-bound).
- **Parakeet (FastConformer):** extra node-classes Whisper lacks - relative-position
  (Transformer-XL) attention, a conv module
  (pointwise -> GLU -> depthwise k9 -> BN -> SiLU -> pointwise), and Macaron half-FFNs.
  Same mmul GEMMs plus the depthwise gap.

## 5. The high-value COMBINATIONS (why composition, not just per-op, is the win)

1. **Shared mmul-tiled layout across fc1->fc2** (and qkv share ln1): tile/convert
   ONCE, reuse -> no re-tile between ops. The layout is the expensive part; amortize
   it across the block.
2. **Fused epilogue** (bias + GELU/SiLU + the start of the next LN) at the
   `to_vector` point: the activation and norm ride the GEMM's accumulator readout,
   zero extra passes.
3. **On-chip cascade-accumulator K-reduction** instead of host-f32 or buffer-passing
   (encoder fc2; any split-K): the cross-core add happens in the MAC datapath
   (~5-10x). This is the biggest single dataflow gap.
4. **Resident intermediate / resident block:** activations never touch host across a
   transformer block (encoder #1 lever, ~88ms). Decode already does this
   (1 dispatch/token).
5. **Eliminate transposes via layout:** produce each op's output already in the next
   op's required tile order.
6. **Precision laddering:** bf16 (parity) -> bfp16 (encoder GEMMs, WER-gated) ->
   int8/int4 (decode, energy).

## 6. ROI-ordered build sequence

1. Encoder resident-intermediate FFN (fc1->fc2 device-resident + GELU fused).
2. On-chip cascade-accumulator for fc2 K-reduction.
3. mmul-ize hand-written GEMMs where M >= 8.
4. Decode op-count ladder + self-V-transpose elimination.
5. Depthwise-2D conv kernel.
6. bfp16 / int8 precision (WER-gated).

Gate discipline per step: build -> per-node golden gate -> on-device ->
e2e WER/argmax parity -> single-source the number.

## 7. Steerable execution - the speed/quality knob

The honest reframe of "byte-to-byte": instead of demanding bit-identity (impossible
for a faster float kernel), declare an accuracy TOLERANCE per model and let the
engine pick the FASTEST execution that meets it. Some models dial toward quality
(tight tolerance -> bf16/f32-accumulate), some toward speed
(loose -> bfp16/int8/int4). This is steering by profile selection against a gate,
not a single magic runtime-precision kernel: precision is mostly a compile-time
choice (different mmul instantiations, operand types, and layouts), so a single ELF
cannot branch across all of them at runtime.

**Two steering layers, matched to how the hardware actually works:**

1. **Compile-time VARIANTS (the heavy/structural knobs: precision, tile, fusion).**
   Template the kernel on precision and tile (the reference `mm.cc` is already
   templated on `T_in`; `aie::mmul<r,s,t,TA,TB>` plus
   `-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16` give the precision tiers). Build the
   tiers a model needs and register them: a kernel registry maps
   `(op, precision_tier, fusion_level) -> ELF`. Selection is at load time, not in the
   inner loop. Cost: N variants to build and N gates to validate.
2. **Runtime RTP PARAMETERS (the light knobs: activation kind, mask width, bias
   on/off, K-offset).** We already do this - the decode ELF takes per-token
   scratchpad words (`kv_off`, `sm_mask`) and the encoder GEMM selects activation by
   a runtime mode word (rtp[0]: 0=identity / 1=silu / 2=gelu). These cost ~nothing
   and need no rebuild. Use them for anything that does not change the datapath or
   layout.

**The PROFILE (what you steer).** A small enumerated set per model/capability, NOT a
free cross-product:

| profile | precision | fusion / residency | when |
|---|---|---|---|
| `parity` | bf16 in, f32 accumulate | full epilogue fusion, resident | default; WER must equal baseline |
| `fast` | bfp16 (true 512-MAC systolic) | full fusion, resident | quality-tolerant models; WER-gated delta |
| `lean` | int8 weights (+ int8 KV at decode) | resident, energy-tuned | bandwidth/energy-bound (decode), WER-gated |
| `tiny` | int4 weights | resident | small-LLM Tier-2 only |

**The gate makes it SAFE (no silent quality loss).** Each profile is validated
offline against the per-node golden (rel-L2/corr) AND the e2e gate
(WER 0.1172 / argmax parity), and records its measured accuracy so the profile
carries its own quality cost (for example `fast` = WER 0.1209, `lean` = 0.1245). The
engine then refuses a profile that fails its declared tolerance - you cannot
accidentally ship fast-but-wrong.

**Where it lives:** the control plane (`engine.toml`) already does per-model
desired-state config, so we add an `execution_profile` (or `precision` /
`max_wer_delta`) field per model. The control plane selects the registered kernel
variants for that profile at load; a residency-agnostic `max_resident` knob composes
with it. No redesign - it is a new config field plus a kernel registry keyed on the
profile plus the existing gates as the admission test. This realizes the
multi-precision general-engine thesis: one engine, per-model speed/quality dialed by
a gated profile.
