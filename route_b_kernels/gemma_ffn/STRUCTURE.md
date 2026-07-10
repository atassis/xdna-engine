# Gemma gated-GeGLU FFN sub-block -- structure map (r1 validation spike)

The "own the primitive" map the kernel + generator (Task 3) consume. Validated host-side on the cached
Gemma-3-270m (same architecture as the Gemma-4-E2B target; E2B swaps only dims). Numbers below are the
270m proof; re-capture on E2B when its download lands.

## The sub-block (what stays resident on the NPU)

    x_in  ->  pre_feedforward_layernorm (RMSNorm)
          ->  gate = normed @ gate_proj.T ;  up = normed @ up_proj.T      (two D->I GEMMs)
          ->  h = gelu_tanh(gate) * up                                    (gated GeGLU)
          ->  down = h @ down_proj.T                                      (one I->D GEMM, reduces over I)
          ->  post_feedforward_layernorm (RMSNorm)
          ->  x_out = x_in + (post-norm result)                          (residual from resident L1 buffer)

Confirmed against HF `Gemma3DecoderLayer`: sandwich norms (`pre/post_feedforward_layernorm`) + residual
wrap the `mlp` (gate/up/down + `GELUTanh`). The oracle hooks the pre-norm INPUT and post-norm OUTPUT and
closes the block with `x_out = x_in + post_norm_out`.

## Dims (270m proof / E2B target)

| | d_model (D) | intermediate (I) | act | rms_eps |
|---|---|---|---|---|
| Gemma-3-270m | 640 | 2048 | gelu_pytorch_tanh | 1e-6 |
| Gemma-4-E2B | (from meta.json on capture) | | gelu_pytorch_tanh | |

gate_proj/up_proj = [I, D]; down_proj = [D, I]; norm gammas = [D].

## Gemma specifics the kernel MUST match (audit)

- **RMSNorm**: sum-of-squares in **float32**, scale by **(1 + gamma)** (gamma init 0). Dropping the +1 or
  doing bf16 ssq is a known bug (mlir-air `attention_decode/` already fixes the ssq case).
- **Activation**: `gelu_pytorch_tanh` = `0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))`.
- **Gated GeGLU** = TWO parallel D->I projections then elementwise `gelu(gate)*up`, unlike Whisper/cascade's
  single `fc1 -> gelu`.

## Kernel base decision (audit -- supersedes the spec's cascade_ffn framing)

- **PREFILL (M>=8) base = the whole-array `aie::mmul` GEMM + fused epilogue**
  (`route_b_kernels/whole_array_fused/whole_array_silu_iron.py`, `decode_fused/gen_ffn_batched.py`), NOT
  `cascade_ffn`'s M=1 dot-product kernel (`mac`+`reduce_add`, ~150 us/row, prefill-nonviable). Brick-first
  regime rule: mmul wins only at M>=8.
- **DMA channels**: gate/up/down stream sequentially on the one weight channel -> gated GeGLU adds NO DMA
  input channel. The 2-input-DMA wall does NOT bite this softmax-free FFN (it is the already-cleared
  cascade/join case); it only bites fused attention (stage 2).
- **Residual + post-norm** live in the block TAIL, fed from a resident L1 buffer (not a 3rd DMA input).
- **L1 budget / single-herd**: keep single-herd to avoid the `ndn-build-cap` N_div_n=9 wall; E2B's larger I
  + the doubled (gate+up) h-slab shrink the all-L1 M_TILE ceiling -> expect multi-dispatch chunking over I.

## Host proof (`tests/test_gemma_ffn_golden.py`, both models)

- 270m fp32 golden vs HF oracle: **rel_L2 5e-7, corr 1.0** (formula exact); bf16 **3.4e-3**.
- E2B (bf16 oracle) corr **0.99999**; the GEMM/GeGLU math isolated exact (**3.9e-7**) -- only the RMSNorm
  convention differed (Gemma4 = normed*weight vs Gemma3 normed*(1+weight); handled by dumping effective gamma).
- E2B sub-block boundary = pre_ff_norm input -> `mlp` module OUTPUT (pure dense gated-GeGLU); EXCLUDES E2B's
  MoE combine (`enable_moe_block`) + PLE (`per_layer_input_*`) + `layer_scalar`, which are layer-level plumbing.
- E2B real FFN intermediate I = **12288** (gate_proj [12288,1536]), NOT config intermediate_size=6144.

## Predicted movement gate (`movement_model.py`, before build)

Grounded in audit DMA constants (91us dispatch floor, 17.6us/MB ~57GB/s, bf16). resident=1 dispatch (ideal):

| model | M=8 | M=512 | note |
|---|---|---|---|
| gemma3-270m (gated) | 2.60x | 3.41x | weight-light -> big relative win |
| gemma4-e2b (gated) | 1.18x | 1.82x | weight-floor bounded (113MB stream); win grows with M |
| parakeet-enc (ungated, genericity) | 1.71x | 2.08x | SAME primitive, different shape+gating |

Dispatch-elimination is the constant primary lever; intermediate-byte saving grows with M; weights are the
shared floor both arms pay. **Measure at realistic prefill M, not M=6.** E2B caveat: I=12288 forces multi-
dispatch chunking (~4x), shrinking the dispatch saving -> validate the real ELF dispatch count vs generate_taps.

## Build plan (Task 3+, toolchain-gated)

EXISTS (no new kernel): **GELU epilogue** (`route_b_kernels/aie_kernels/mm_silu_epilogue.cc` -> `mm_gelu_epilogue_f32o`,
tanh-approx; + modal `rtp[0]==2` = gelu) and the **whole-array mmul GEMM** (`whole_array_silu_iron.py`). So gate =
whole_array + gelu epilogue; up/down = whole_array + identity epilogue.

NON-RESIDENT BASELINE (Task 5 denominator, buildable from existing kernels): 3 whole_array matmuls (gate/up/down)
+ host geglu + host RMSNorm; each op 1 dispatch, intermediates round-trip LPDDR.

RESIDENT (fused) arm -- the hard part, needs on-device iteration (do NOT write blind): one dataflow keeping
normed/gate/up/h on-chip across gate->up->geglu->down; needs on-chip `gelu(gate)*up` elementwise + head RMSNorm
(f32 ssq; the m_stationary / decode_norm_gemv reduction primitives are the bridge), single-herd, multi-dispatch
chunk over I for E2B. Build via FORK instance (`toolchain_up.sh`->aiecc, `toolchain_smoke.sh` first). Gate:
rel_L2<=0.08 + corr>=0.99 vs the bf16 golden (native bf16 path; BFP16_IREE needs 0.65 bar + risks #847 -O1/-O2).
