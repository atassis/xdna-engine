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

## Host proof (Gemma-3-270m, `tests/test_gemma_ffn_golden.py`)

- fp32 golden vs HF oracle: **rel_L2 1.8e-8, corr 1.000000** (formula exact).
- bf16 golden vs HF oracle: **rel_L2 1.9e-4, corr 1.000000** (well inside the codebase gate rel_L2<=0.08).
