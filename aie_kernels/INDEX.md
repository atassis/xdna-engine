# aie_kernels/ index

One row per kernel directory (48). Dtypes and shape contract are read from each
kernel's own `.cc` header and, where present, its `golden.py` -- not inferred from the
directory name. `golden.py` is a straight yes/no: 31 of 48 dirs have one, 17 do not
(the 17 were built and verified only through the designs that consume them, never
independently -- see the `golden.py` column). Covering harness comes from
`aie_kernels/_test/` after the family-harness rename (see `_test/README.md`).

**Device status is sourced, not guessed.** A cell states a result only where this repo
records one, with the file and (where given) a date; most cells say **"unverified in
this index"**, meaning no PASS, FAIL, or device-run record could be found here -- not
that the kernel is broken, and not that it works. Where a `verify_*.py` exists but its
own docstring says it was never actually run on device, that is quoted rather than
inferred. Where a device run happened but only A/B'd a rounding mode (not a golden-vs-
kernel gate), that is called out as such rather than folded into a PASS/FAIL.

Two of these ops (`softmax`, `swiglu`) already exist in the AMD tree at
`mlir-aie/aie_kernels/aie2p/`; the other 46 do not. Checked 2026-09-05 against the
populated sibling checkout -- note the `mlir-aie` submodule inside this repo is an
unpopulated placeholder (`ignore = all`), so it cannot answer this question.

| Op | golden.py | Dtypes | Shape contract | Covering harness | Device status |
|---|---|---|---|---|---|
| acc-add | no | f32 (b: optional bf16 arm, `ACCADD_B_BF16`) | 1 row of `cols`, cols%N==0; 2-input ABI (a,b)->out | none found | unverified in this index |
| affine-cast | no | f32 in (bf16 arm `AFFCAST_X_BF16`), gb f32 (bf16 arm `AFFCAST_GB_BF16`) -> bf16 out | 1 row of `cols`, cols%N==0; gb=[gamma\|beta], len 2*cols | none found | unverified in this index |
| cascade-kreduce | yes | bf16 A/B, f32 (accfloat) cascade accumulate, bf16 C | `aie::mmul` 8x8x8 and 8x64x8 tiles; HEAD/MIDDLE/TAIL cascade roles, N-core depth | none in `_test/` | unverified in this index (its own `device_test/` validated only the generic `aie_api` cascade-accessor primitive, PASS 2026-07-19 -- not this kernel file) |
| cast-f32-bf16 | no | f32 -> bf16 (conv_even rounding) | 1 row of `cols`, cols%N==0 | none found | unverified in this index |
| cast-quant-bf16-int8 | yes | bf16<->int8 (per-tensor scale), bf16<->f32 | 1 row of `cols`, cols%N==0; 4 entry points | `verify_cast_quant.py` | unverified in this index |
| census-coresident-body | no | f32 | 1 row of `cols`; measurement-only, "not a shipped path" per header | none found | unverified in this index |
| conformer-epilogues | yes | f32 accumulator in -> bf16 out (SiLU/GLU/BatchNorm/residual-add) | per-tile, N=1024 lanes (d_model); GLU pairs col j with col N+j (layout-dependent) | `conf_epi_silu_design.py` (own dir, not `_test/`; SiLU node only; IRON build harness, no golden-comparison logic in the file) | unverified in this index |
| conv-1d | yes | f32 | ONE output channel per call; x:[c_in,t], w_row:[c_in*k]; causal, dilation-aware | `verify_conv_1d.py` | unverified in this index |
| conv-transpose-1d | yes | f32 | ONE output channel per call, scalar (avoids unaligned vector stores at strides 8/8/4/2) | `verify_conv_transpose_1d.py` | GREEN, rel-L2 3.894e-08 (cited in `verify_sin.py`'s 2026-07-31 docstring) |
| dequant-int4-group | yes | int4 (packed 2/byte) -> bf16; scale f32; zero-point int8 (optional) | 1 row of `cols` int4-packed values; GROUP granularity, GROUP%N==0, cols%GROUP==0 | `verify_dequant_int4_group.py` | unverified in this index |
| gatedeltanet | yes | k/v/q bf16; state S and gates f32 (accfloat-equivalent) | [T,DK]/[T,DV] tiles; D_V is the only data-parallel axis, T is sequential; SPIKE, not a landed brick | `verify_gatedeltanet.py` | unverified in this index |
| gather-rows | yes | f32 codebook/out, int32 indices | codebook [n_rows,D] resident (D=8; gated at n_rows=1024/tile), indices streamed | `verify_gather_rows.py` | unverified in this index (file's own docstring: "NOT device-verified by the agent that authored this file -- no NPU access for that task") |
| geglu | yes | f32 in/out; bf16 tanh-output intermediate only | 1 row of `cols` out; input row width 2*cols = [up\|gate] | `verify_norm_elementwise_f32.py` | unverified in this index |
| gelu-erf | yes | f32 | exactly 16-wide per call (no element count, no internal loop -- a loop of any trip count miscompiled); M carries volume | `verify_gelu_erf.py` | unverified in this index (own header records THREE rounds of device failure on earlier revisions, rel-L2 0.70/0.7043/permuted-output; current revision's own device result is not recorded here -- see note below on sin) |
| gemm-bf16xbfp16 | yes | A bf16, B bfp16ebs8 (pre-quantized off-core), C bf16 (accfloat accumulate) | M,K,N compile-time; native `aie::mmul<8,8,8,...>` tile | `verify_bfp16.py` | unverified in this index (`verify_rounding_ab.py` ran a rounding-mode-only A/B on device 2026-07-28: all 4 arms bit-identical -- not a golden-vs-kernel gate) |
| gemm-bfp16-ebs8 | yes | bf16 resident stream in/out; true-systolic bfp16ebs8 x bfp16ebs8 internally | M,K,N multiples of 8; native 8x8x8 tile | `verify_bfp16.py`, `verify_upscaler_bf16_conv2d.py`, `verify_upscaler_bf16_gemm.py`, `verify_upscaler_espcn_image.py`, `verify_upscaler_espcn_wholenet.py` | unverified in this index (`verify_crrnd_scope_ab.py` ran a rounding-mode-only A/B on device 2026-08-30: BREAKS across rounding arms, max 240 ULP -- not a golden-vs-kernel gate) |
| gemm-int8 | yes | int8 x int8 -> int32 | M,K,N multiples of 8; native `aie::mmul<8,8,8,int8,int8,accauto>` tile | `verify_gemm_int8.py`, `verify_upscaler_conv2d.py`, `verify_upscaler_conv2d_ktile.py` | unverified in this index |
| gemm-int8xint4 | yes | int8 (A) x int4, packed 2/byte (B) -> int32 | native `mmul_8_4` tile, 4x16x16 (M,K,N multiples thereof) | `verify_gemm_int8xint4.py` (`verify_int4_probe.py`/`_shapes.py`/`_stride.py` are layout bisect probes despite the `verify_` prefix -- not gates, not in `drain_device.py`) | unverified in this index |
| gemm-int8xint4-dequant | yes | int8 (A) x int4 group-quantized (B), f32 per-group scale -> bf16 | K split into groups (multiple of native 16-wide K tile); M,K,N multiples of native tile | `verify_gemm_int8xint4.py` (`verify_dequant_f32.py`/`verify_dequant_probe.py` are bisect probes despite the `verify_` prefix -- not gates) | unverified in this index |
| gemv-int8 | yes | int8 x int8 -> int32, or fused single-scalar dequant -> f32 | GEMV_M/K/N compile-time; M padded to native r=8 (row 0 = real query) | `verify_gemm_int8.py` | unverified in this index |
| gemv-int8xint4 | yes | int8 (A, M_NATIVE=4-padded) x int4 (B, 16x16 sub-blocks) -> int32 | GEMV_K/N multiples of native 16x16 tile | `verify_gemm_int8xint4.py` | unverified in this index |
| glu | no | f32 in/out; bf16 tanh-output intermediate only | 1 row of `cols`; input row width 2*cols = [a\|g] | none found | unverified in this index |
| layernorm | yes | f32 in/out (gamma/beta f32 if `Affine`) | 1 row of `cols` (=D), cols%N==0 (N=16 default); `UseMean`/`Affine` compile-time bools select LN/RMS x affine-or-not | `verify_norm_elementwise_f32.py` | unverified in this index |
| lm-head-argmax | yes | hidden/W bf16, accumulator f32, out (value f32, index int32) | [1,HIDDEN]@[HIDDEN,VOCAB] tiled by K_TILE/N_TILE; M padded to M_PAD (row 0 real) | `verify_lm_head_argmax.py`, `verify_argmax_slice.py` (latter not wired into `drain_device.py`'s TARGETS/SCRIPT_TARGETS) | unverified in this index |
| ln-2pass | no | f32 | 1 row of `cols`, N=16; normalize-only (no affine) | none found | unverified in this index |
| ln-affine-cast | no | f32 in, f32 gb -> bf16 out | 1 row of `cols`, cols%N==0; gb=[gamma\|beta], len 2*cols | `verify_crrnd_scope_ab.py` (rounding-mode A/B only, not golden-gated) | unverified in this index (ran on device 2026-08-30 for rounding only: BREAKS, 30/64 elements rounding-sensitive, max 1 ULP) |
| ln-affine-f32 | no | f32 in, f32 gb, f32 out | 1 row of `cols`, cols%N==0; gb=[gamma\|beta], len 2*cols | none found | unverified in this index |
| ln-cheap-prologue | no | mu f32-precise (double-bf16 or f32-byte-reinterpreted), inv bf16-tolerant, a (in place) bf16 | PRO_M rows per A-tile, mmul-blocked [r,s] tiled coords | none found | unverified in this index (header states outright: "NOT DEVICE-VALIDATED (numpy-only)") |
| mm-ln-epilogue | no | f32 accumulator in -> bf16 out | DIM_M/DIM_N tile (defaults 16/64), DIM_R/DIM_T mmul sub-tile blocking (defaults 4/8); normalize-only | none found | unverified in this index |
| mm-ln-prologue | no | bf16 A in/out (streamed store), f32 accumulators | DIM_M/DIM_K = mmul r,s sub-tile; K re-streamed in k=32 blocks, two-pass stats-then-apply | none found | unverified in this index |
| mm-mode-lnaffcast | no | bf16 A-tile bytes reinterpreted as f32, f32 gb -> bf16 out | rides the modal GEMM's own A/B/C tile shapes (32x128xf32 `cacc`); mode of the modal GEMM xclbin, not a standalone ABI | none found | unverified in this index |
| mm-mode-resadd2a | no | bf16 a/b -> f32 c (GEMM C-tile accumulator, scale baked as compile-time bits) | rides the modal GEMM's C-tile staging; closed-form index map over (m,n,r,k) | none found | unverified in this index |
| mm-silu-epilogue | no | f32 accumulator in -> bf16 out | m*n C-tile, walked 16-wide; bias folded via host K-augmentation (no kernel arg) | none found | unverified in this index |
| moe-topk-router | yes | hidden/Wg bf16, accumulator f32, out (values f32, indices int) | [1,HIDDEN]@[HIDDEN,EXPERTS]; K iterations of selection-sort-by-argmax | `verify_moe_topk_router.py` | unverified in this index |
| norm-gemv-prologue | no | a (in place) bf16; f32 reduction/normalize math | EPI_MK = m*K resident tile (default 12288), EPI_K=768 (real row length); decode M=1 padded to m=64 | none found | unverified in this index |
| prefill-attn | yes | Q/K/V bf16 (dot-product loop); softmax f32 internally (composes `softmax.cc` via `#include`) | M<=11 queries (S2 fast-decoder prefill bound); causal, GQA; ONE query row per call | `verify_prefill_attn.py` | GREEN, 2026-08-31: 32/32 heads, worst rel-L2 8.540e-08 (`drain_device.py`) |
| qk-norm | yes | f32 in/out; gamma f32[cols] (optional, nullptr for weight-free) | `rows` x `cols` (cols=head_dim); ONE dispatch normalizes the WHOLE resident tile, not one row | `verify_norm_elementwise_f32.py` | unverified in this index |
| relu2 | yes | f32 | 1 row of `cols`, cols%N==0 | `verify_norm_elementwise_f32.py` | unverified in this index |
| residual-add | no | f32 default (bf16 arms: `RESADD_BF16` all-bf16, `RESADD_B_BF16` b-only) | 1 row of `cols`, cols%N==0; `scale` baked at compile time (one xclbin per scale value) | `verify_crrnd_scope_ab.py` (rounding-mode A/B only, not golden-gated) | unverified in this index (ran on device 2026-08-30 for rounding only: BREAKS, 20/64 elements rounding-sensitive, max 1 ULP) |
| rmsnorm | yes | f32 in/out; gamma f32[cols] required, beta f32[cols] optional | 1 row of `cols`, cols%N==0 (N=16 default) | `verify_norm_elementwise_f32.py`, `verify_rmsnorm.py` | unverified in this index |
| rope-interleaved | yes | qk bf16 (in place); cossin f32 (host-precomputed) | ROPE_D total dim, ROPE_ROT<=ROPE_D rotary width (partial-rotary passthrough), ROPE_M rows/call | `verify_rope_interleaved.py` | unverified in this index |
| rope-lut | yes | qk bf16, pos int32, inv_freq f32 (resident); LUT bf16, 256-entry | ROPE_D/ROPE_ROT/ROPE_M, ROPE_SCALE_INV compile-time NTK-scaling knob | `verify_rope_lut.py`, `verify_rope_lut_tables_repro.py` | unverified in this index |
| sin | yes | f32 | exactly 64 elements per call (aliased/unbounded output past that -- open toolchain hazard, not a delivery limit) | `verify_sin.py`, `verify_snake.py` (composes `sin_v` via `#include`) | **DISPUTED, unresolved.** The harness's committed config (256x16) records GREEN rel-L2 1.275e-04, 2026-07-31, three fresh builds. An earlier record (2026-07-30) has it FAILING at 7.091e-01 -- but at **64x64**, a different shape, and it attributes the 1.275e-04 to a stale cache, which the later note rebuts by name. Different shapes, so both can be true. Not re-run here. |
| snake | yes | f32; alpha f32 (per-channel scalar) | ONE channel row of t=64 elements per call (matches sin's per-call ceiling) | `verify_snake.py` | GREEN, rel-L2 5.143e-06 (cited in `verify_sin.py`'s 2026-07-31 docstring: "snake stays green ... it composes `sin_v` directly and never used `sin_core`") |
| softmax | yes | f32 | 1 row of `cols` (the reduction axis), axis-agnostic; 3-pass (max, sum-exp, normalize), own polynomial for exp (not the hardware exp2 SFU) | `verify_softmax.py`; also compiled into `prefill-attn` on-device via `#include "../softmax/softmax.cc"` | unverified in this index for `verify_softmax.py` itself; the composed instance inside `prefill-attn` is part of that kernel's GREEN 2026-08-31 run above |
| swiglu | yes | f32 in/out; bf16 exp2-SFU-output intermediate | two f32 [tile,D] input streams (gate,up) -> one f32 [tile,D] output; SWIGLU_SIZE compile-time, multiple of 16 | `verify_norm_elementwise_f32.py` | unverified in this index |
| transpose-dma | yes | bf16 (uint16) default, or 32-bit via `TPOSE_i32`; pure DMA relayout, 0% compute | up to 4 [count,stride] pairs per side; gated at 32x32 int32 | `verify_specials.py` | unverified in this index |
| transpose-tile | no | bf16 (uint16) default, or 32-bit via `TPOSE_i32` | [mb,nb] row-major -> [nb,mb] row-major; mb,nb runtime int32 | none found | unverified in this index |
