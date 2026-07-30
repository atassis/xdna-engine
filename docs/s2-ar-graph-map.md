# S2 Dual-AR graph map: text tokens -> [10, T] codec codes

The codec DECODER half of Fish-Audio S2 (codes -> audio) is already ported and partly running on
the NPU (`route_b_kernels/codec_block/`, `feat/s2-codec-bricks`). The autoregressive half that
PRODUCES those codes had no reference, no shape inventory, and no kernel plan before this doc and
its companion `scripts/s2_ar_ref.py`. This is that inventory: every op in the real forward pass,
with real tensor names and shapes pulled from `s2.cpp/models/s2-pro-q6_k.gguf`, which of it already
has a brick in `route_b_kernels/`, a parity/gating plan, and an L1 sizing note for what's new.

Everything here was cross-checked against `s2.cpp/src/s2_model.cpp` (`SlowARModel::eval_cached` /
`::fast_decode`, the ggml graph builders) and `s2.cpp/src/s2_generate.cpp` (the sampling loop) --
not inferred from a paper or a different Fish-Speech version. Where a claim is inferred rather than
read off the source, it's labelled INFERRED.

## 0. Headline finding: the quantization blocker is real, and it's fully unblocked

Every AR weight matrix -- `embeddings.weight`, `codebook_embeddings.weight`, every
`wqkv`/`wo`/`w1`/`w2`/`w3` in all 36 slow + 4 fast layers, `fast_embeddings.weight`,
`fast_output.weight` -- is GGUF type **q6_k**. Only the RMSNorm gamma vectors (`attention_norm`,
`ffn_norm`, `q_norm`, `k_norm`, `fast_norm`, all length-2560-or-128 vectors) are f16.
`scripts/gguf_extract.py` (borrowed from `feat/s2-codec-bricks`, also copied into this worktree)
only decodes f32/f16/bf16 -- so, unmodified, it reads NOTHING useful for this half of the model.

Confirmed by `python3 scripts/s2_ar_ref.py --report-types`:

```
358 AR tensors needed, by GGUF type: {'q6_k': 204, 'f16': 154}
```

Rather than stub the graph with random weights, `scripts/s2_ar_ref.py` ports ggml's q6_k dequant
(`ggml/src/ggml-quants.c:dequantize_row_q6_K`, block layout `ggml/src/ggml-common.h:352-358`) to
vectorized numpy, verified byte-exact against a scalar transliteration of the same C function on
real bytes from the shipped GGUF (`_selftest_q6k_matches_scalar_ref` in that script). Every weight
this graph needs is therefore readable, and the reference in this repo runs end to end on REAL
trained weights -- verified: a full 36-layer slow forward pass + full 4-layer fast forward pass +
a 2-frame greedy-generation smoke test, all producing finite, in-range outputs (semantic argmax
landed inside `[semantic_begin_id, semantic_end_id]`; see that script's `--full-forward` output).

**Recommendation for the eventual NPU port: do this conversion on the HOST, once, at load time.**
q6_k's 6-bit/16-scales-per-256-superblock format has no relationship to this project's existing
`dequant-int4-group` brick (int4, one f32 scale + zero-point per group) -- porting q6_k dequant to
an AIE2P kernel would be new, fiddly, low-value work for a conversion that only needs to happen
once per model load, not once per token. Dequantize to bf16 on the host (numpy, as validated here,
~55s for the full ~4.5B-element AR weight set) and ship a resident bf16 (or re-quantized int8/int4)
weight blob to the device -- this is a preprocessing decision, not a new hardware brick.

## 1. Architecture, from the GGUF's own metadata (`fish_speech.*` / `fish-speech.*` KV keys)

Dual-AR ("Slow" text/semantic transformer + "Fast" residual-codebook transformer), both
Llama-family (RMSNorm, RoPE, SwiGLU, GQA), fused QKV, causal.

| | Slow | Fast |
|---|---|---|
| purpose | text/prompt tokens -> 1 semantic token/frame + hidden state | hidden + prior codebook tokens -> 9 residual RVQ tokens/frame |
| layers | 36 (`block_count`) | 4 (`fast_block_count`) |
| embedding_length | 2560 | 2560 (`fast_embedding_length`; no `fast_project_in` tensor in this GGUF, so hidden feeds Fast directly) |
| feed_forward_length | 9728 | 9728 |
| head_count / head_count_kv | 32 / 8 (GQA 4:1) | 32 / 8 (GQA 4:1) |
| head_dim | 128 (from `q_norm`/`k_norm` shape, NOT `embedding_length/head_count`=80) | 128 (`fast_head_dim`) |
| q_size = head_count*head_dim | 4096 (> embedding_length=2560 -- wide attention, confirmed by tensor shapes, not a modeling assumption) | 4096 |
| attention_qk_norm | **True** | **False** |
| rope_freq_base | 1e6 | 1e6 |
| rms_norm_eps | ~1e-6 | ~1e-6 |
| causal mask | yes, KV-cached across the whole generation | yes, fresh per frame (`fast_context_length`=11, no cross-frame cache) |
| output | tied: `logits = hidden @ embeddings.weight^T` (`tie_word_embeddings=True`) over `vocab_size=155776` | untied: `logits = hidden @ fast_output.weight^T` over `codebook_size=4096` (`fast_tie_word_embeddings=False`) |

Codebook bookkeeping: `num_codebooks=10` (1 semantic + `quantizer_residual_codebooks=9`, per the
codec's own KV keys), `codebook_size=4096`, `semantic_begin_id=151678`, `semantic_end_id=155773`
(4096-wide, matches `codebook_size`), `scale_codebook_embeddings=True` (verified True in THIS
checkpoint -- easy to assume False since that's the struct default in `s2_model.cpp`, but the GGUF
overrides it).

## 2. Slow-transformer op graph (per `eval_cached`, `s2_model.cpp:860-1097`)

Real tensor names/shapes (GGUF `ne`, then numpy shape = reversed, matching `gguf_extract.py`'s
convention -- (out,in) for weight matrices):

```
INPUT: flat_tokens[n_tokens, 11]   -- row t = [semantic_or_text_id, cb0_id, .., cb9_id]
       (11 = num_codebooks+1; codebook slots are ignored/masked-to-zero for non-semantic rows)

1. embed        = get_rows(embeddings.weight[2560,155776] q6_k, semantic_ids)          -> [n,2560]
2. codebook_sum = sum_{cb=0..9} get_rows(codebook_embeddings.weight[2560,40960] q6_k,
                                          cb_token_id[cb] + cb*4096)                    -> [n,2560]
3. codebook_sum = codebook_sum * is_semantic_mask[n,1]     (only added on semantic-token rows)
4. x = embed + codebook_sum
5. x = x * token_scale[n,1]     (token_scale = 1/sqrt(11) on semantic rows, 1.0 elsewhere --
                                  scale_codebook_embeddings=True in this checkpoint; scales the
                                  WHOLE summed embedding, not just the codebook part)

   per layer il in 0..35 (each weight below is layers.{il}.*, q6_k unless noted):
6.  attn_in = rmsnorm(x, attention_norm[2560] f16, eps) * attention_norm
7.  qkv     = attn_in @ wqkv[6144,2560]^T                                              -> [n,6144]
8.  q,k,v   = split(qkv, [4096, 1024, 1024])  -> reshape [n,32,128] / [n,8,128] / [n,8,128]
9.  q = rmsnorm_per_head(q, q_norm[128] f16, eps); k = rmsnorm_per_head(k, k_norm[128] f16, eps)
10. q,k = RoPE_interleaved(q, pos, base=1e6); RoPE_interleaved(k, pos, base=1e6)   -- mode=NORMAL,
        ADJACENT-PAIR rotation, NOT NeoX split-half (ggml.h:250,1809; see #5 below)
11. k,v -> KV cache write; k_rep,v_rep = repeat_interleave(k,4), repeat_interleave(v,4)  GQA 8->32
12. scores = causal_softmax(q @ k_rep^T / sqrt(128))                                   -> [n,32,n]
13. ctx     = scores @ v_rep  -> reshape [n,4096]
14. attn_out = ctx @ wo[2560,4096]^T                                                   -> [n,2560]
15. h = x + attn_out
16. ff_in = rmsnorm(h, ffn_norm[2560] f16, eps)
17. gate = ff_in @ w1[9728,2560]^T ; up = ff_in @ w3[9728,2560]^T
18. ff_h = silu(gate) * up                                                             -> [n,9728]
19. ff_out = ff_h @ w2[2560,9728]^T                                                    -> [n,2560]
20. x = h + ff_out

21. slow_out = rmsnorm(x, norm.weight[2560] f16, eps)
22. hidden_last = slow_out[-1, :]                                                      -> [2560]
23. logits = hidden_last @ embeddings.weight[2560,155776]^T  (TIED)                    -> [155776]
```

## 3. Fast-transformer op graph (per `fast_decode`, `s2_model.cpp:1099-1255`), called 9x/frame

```
INPUT: hidden[2560] (slow model's hidden_last for this frame), prefix_codes = codebooks
       decided so far this frame (starts empty, grows to length 8 by the last call)

1. x0 = hidden        (no fast_project_in in this GGUF: fast_embedding_length == embedding_length)
2. prefix_emb = get_rows(fast_embeddings.weight[2560,4096] q6_k, prefix_codes)     -> [len,2560]
3. x = concat([x0, prefix_emb], axis=0)                                          -> [1+len, 2560]

   per layer il in 0..3 (fast_layers.{il}.*, same op sequence as slow steps 6-20, but:
     - NO qk_norm (fast_attention_qk_norm=False)
     - causal mask is fresh (n_past=0 every call, no cross-call KV cache)
     - fast_rms_norm_eps, fast_rope_freq_base=1e6 (own hparams, numerically ~same as slow here))

23'. fast_out = rmsnorm(x, fast_norm.weight[2560] f16, eps)
24'. last = fast_out[-1, :]                                                        -> [2560]
25'. logits = last @ fast_output.weight[2560,4096]^T  (UNTIED)                     -> [4096]
```

Sampling (outside both graphs, `s2_generate.cpp`/`s2_sampler.cpp`): mask non-semantic vocab to
`-inf`, top_k/top_p filter, softmax, `std::discrete_distribution` sample -- OR, if
`temperature<=0`, deterministic argmax of the filtered (== unfiltered top-1) logits. See #3 below
for why this matters for gating.

## 4. Brick-vocabulary mapping (`route_b_kernels/bricks/` + `route_b_kernels/` generally)

| AR op | existing brick | verdict |
|---|---|---|
| RMSNorm (attention_norm/ffn_norm/fast_norm, D=2560) | `bricks/rmsnorm` | **REUSE**, D-generic `[tile,D]` contract already matches |
| RMSNorm on Q/K per head (D=128) | `bricks/qk-norm` | **REUSE**, same contract, smaller D |
| SwiGLU (`silu(gate)*up`) | `bricks/swiglu` | **REUSE**, exact match (`out=silu(gate)*up`, same as ggml's `ggml_vec_swiglu_f32`) |
| wqkv / wo / w1 / w2 / w3 GEMMs | `bricks/gemm-bfp16-ebs8` (or `gemm-bf16xbfp16`) | **REUSE the kernel**, but needs q6_k->bf16 weight conversion first (see #0) -- not a kernel gap, a preprocessing gap |
| masked argmax over restricted vocab (greedy sampling) | `bricks/lm-head-argmax` | **REUSE with reparameterization**: brick's `HIDDEN=256,VOCAB=512` are compile-time defaults, not fixed; needs `VOCAB=4096` (fast, exactly `codebook_size`) or `~4097` (slow, `semantic_end-semantic_begin+1` plus `im_end_id`) |
| RoPE (Q, K, both stacks) | `bricks/rope-lut` | **LOOKS-SOLVED-BUT-ISN'T -- concrete gap.** `rope-lut`'s golden (`golden.py:88-92`) rotates `x1=qk[0:half]` against `x2=qk[half:rot]` -- **NEOX split-half** convention. S2 calls `ggml_rope_ext(..., mode=0, ...)` = `GGML_ROPE_TYPE_NORMAL` = **adjacent-pair** rotation (`ggml.h:1809`, diagrammed `[cscs0000]`). These are different math; reusing `rope-lut` as-is on S2 would silently rotate the wrong element pairs. Needs a NEW adjacent-pair variant (same LUT/sizing, different pairing in the kernel loop) -- see brick-first doctrine: "using a generic brick where the hardware needs a specialized one" is the recurring costly mistake; this is its RoPE-convention cousin. |
| GQA-aware causal self-attention, decode (slow model, incremental M=1 step) | `route_b_kernels/mha_decode/mha_decode.cc` | **PARTIAL -- new kernel instance needed.** `mha_decode.cc` hardcodes `HD=64` (`static constexpr int HD = 64;`, Whisper D=768/12 heads) and has no GQA repeat (Whisper MHA: `n_head==n_head_kv`). S2 needs `HD=128` and an 8->32 GQA expand before/inside the flash loop. The flash-attention STRUCTURE (online softmax, streamed K/V tiles, f32 accumulator) is directly reusable; the compile-time constants and the missing repeat are not. |
| GQA-aware causal self-attention, prefill (fast model, M<=10 short sequence; slow model's initial multi-token prompt) | none exactly | **NEW.** Neither `mha_decode` (M=1 decode) nor `relpos_mha` (Parakeet encoder, M=T, bidirectional + relative-position bias, no causal mask, no GQA) matches a short **causal, GQA, no-relative-position** prefill. Given fast_context_length<=11, this is small enough that a naive resident (not flash) SxS score matrix is fine -- see sizing #6. |
| embedding / codebook-table gather (`get_rows`) | none | **NEW.** No brick in the catalog does an indexed row-gather from a huge (155776x2560 / 40960x2560) DDR/L3-resident table. Every existing brick's `[tile,D]` contract assumes the WHOLE tile streams in order; this needs index-driven DMA. Small and cheap per-op (a handful of ~2-5KB row reads per token), but a genuinely new MOVEMENT pattern, not a reparameterization of an existing one. |
| masked multi-table-gather-sum + optional scale (embedding+codebook fusion, steps 2-5 above) | none | **NEW**, small. S2-specific fusion (sum 10 gathered rows, mask by per-row semantic flag, add to token embedding, conditionally scale the sum). Structurally similar to the elementwise fuse patterns already in `route_b_kernels/ctx_ln/` (resadd/affine-cast), just with a gather feeding it instead of a stream. |
| q6_k weight dequant | `bricks/dequant-int4-group` (wrong format) | **NOT NEEDED on-device** -- see #0's recommendation (host, load-time, once). |
| MoE router | `bricks/moe-topk-router` | **N/A** -- S2 has no MoE layer (dense FFN only); not part of this graph. |

## 5. Parity story: where the gate goes

The codec side gates at `codec_codes.bin` -> `codec_audio.bin` (`s2_dump_raw`, `s2_codec.cpp:1305`,
called for `codec_codes` at `s2_codec.cpp:1328`, for `codec_audio`/`codec_latent` at
`s2_codec.cpp:1333`/`1435`) -- "given the SAME codes, the ported block must reproduce the SAME
audio." The natural AR-half cut, by the same logic, is "given the SAME input tokens, does the
ported graph reproduce the SAME logits" -- NOT the same sampled codes, because sampling is
stochastic and NOT reproducible even between two runs of unmodified `s2.cpp`:
`s2_sampler.cpp:94` seeds `std::mt19937` from `std::random_device{}()` with no seed override
anywhere in the CLI/API. Two real findings that shape the gate:

1. **`--temperature 0` makes generation fully deterministic AND argmax-equivalent**, with no RNG
   involvement -- verified by reading `sample_token` (`s2_sampler.cpp:42-99`): it sorts logits
   descending, top_k/top_p FILTERS (removes low-ranked entries, never reorders), then
   `if (params.temperature <= 0.0f) return filtered[0].second;` (line 77-79). `filtered[0]` is
   always the pre-filter argmax regardless of `top_p`/`top_k` values, because filtering can only
   remove candidates ranked below it. So `--temperature 0` end-to-end code-matrix parity is a real,
   achievable target, not just a per-step logits check.
2. **...except the RAS window can override it.** `s2_generate.cpp:99-116`: if the just-picked
   semantic token repeats within the last 10 semantic tokens, the code re-samples at
   `temperature=1.0, top_p=0.9` -- WITH the same unseeded RNG -- even inside a `--temperature 0`
   run. There is no CLI flag to disable this. So exact end-to-end code-matrix parity holds only
   for however many frames elapse before RAS first triggers on a given prompt; it is not a
   permanent guarantee. `scripts/s2_ar_ref.py`'s `generate_greedy` does NOT implement the RAS
   escape hatch (documented in its docstring) -- deliberately, since matching libstdc++'s
   `mt19937`/`discrete_distribution` bit-for-bit from numpy isn't achievable anyway.

**Recommendation: gate on pre-sampling logits, not on sampled codes**, sidestepping RAS entirely.
This needs ONE new dump point s2.cpp does not currently have (task constraint: do not add it,
just specify it):

- **`s2_model.cpp`, inside `SlowARModel::eval_cached`, right after line 1091**
  (`ggml_backend_tensor_get(logits, result.logits.data(), 0, hparams_.vocab_size * sizeof(float));`):
  dump `result.hidden` (2560 floats) as `ar_slow_hidden` and `result.logits` (155776 floats) as
  `ar_slow_logits`, guarded the same way `s2_dump_raw` already is (`S2_DUMP_DIR` env, same
  `<name>.bin`/`<name>.shape` convention). This call fires on BOTH the prefill and every
  subsequent per-token step; for a clean single-shot gate, tag the dump with a step counter
  (`n_past_` at entry is 0 exactly for the prefill call) so only the FIRST call needs comparing.
- **`s2_model.cpp`, inside `SlowARModel::fast_decode`, right after line 1250**
  (`ggml_backend_tensor_get(logits, logits_out.data(), 0, hparams_.codebook_size * sizeof(float));`):
  dump `logits_out` as `ar_fast_logits_cb{cb_idx}` and the `prefix_tokens` argument as
  `ar_fast_prefix_cb{cb_idx}` (both tiny). `hidden_in` doesn't need its own dump -- it's exactly
  the `ar_slow_hidden` from the same frame's slow-model call.
- **`s2_generate.cpp`, right before line 51** (`model.prefill_fast(prompt_tm, ...)`): dump
  `prompt_tm` (the flattened `[n_tokens, 11]` int32 array already built at lines 39-44) as
  `ar_prompt_tokens` -- this is the deterministic INPUT the numpy reference needs to reproduce
  `ar_slow_logits` without reimplementing the tokenizer/prompt template (explicitly out of scope,
  same line the codec side already drew between "codes -> audio" and "text -> codes -> audio").

Given those three dump points, the gate is: run s2.cpp once with `S2_DUMP_DIR` set (any
temperature, since only the PRE-sampling prefill call is being compared) on a fixed prompt, feed
`ar_prompt_tokens` into `s2_ar_ref.slow_transformer_forward(hp, w, ar_prompt_tokens,
logits_row_ids=None)`, and rel-L2 the result against `ar_slow_logits`/`ar_slow_hidden` -- same
rel-L2 convention the codec side and every `route_b_kernels/bricks/*/golden.py` already use.
`scripts/s2_ar_ref.py` doesn't wire this comparison up yet (the dump point doesn't exist to
compare against), but every piece it needs -- the forward pass, a `rel_l2`-style helper -- is
already in the script and proven correct on real weights; wiring a `--verify-dump DIR` flag is a
same-day follow-up once the dump point lands.

## 6. Sizing note: new kernels, operand shapes, L1 fit

AIE2P core-tile data memory (L1) is 64 KB. Measured calibration point from the codec work (same
family of streamed-design-with-one-resident-operand kernels): at `c=96` rows, a `[c,64]` f32
resident operand FITS; `[c,128]` f32 EXCEEDS. That's used below as the practical boundary (not
re-derived here -- this task is host-only, no device access) rather than the raw 64 KB arithmetic,
since other L1 consumers (streamed-operand double-buffer, code, stack) already eat into the budget
at `[c,128]` f32 even though `96*128*4=48 KB < 64 KB` on paper.

| new/changed kernel | resident operand | shape | bytes (f32) | bytes (bf16) | fits `[c<=96,D<=64]`-f32 budget? |
|---|---|---|---|---|---|
| embedding/codebook gather | none (table stays in DDR/L3; only the gathered rows move) | per-row `[1,2560]` | 10 KB/row | 5 KB/row | **trivial** -- the constraint here is DMA random-access count (11 gathers/slow-token, up to 9/fast-frame-step), not L1 capacity |
| masked codebook-sum+scale | the up-to-11 gathered rows, summed in place | `[11,2560]` transient | 112.6 KB (all 11 co-resident) | 56.3 KB | **borderline at f32** (over budget if all 11 rows are held simultaneously instead of accumulated one at a time into a single `[1,2560]` running sum -- accumulate-in-place is the fix, trivial) |
| wqkv GEMM (K=2560,N=6144) | weight tile `[Ktile,Ntile]`, activation resident-M convention from `gemm-bfp16-ebs8` | e.g. `[96,64]` bf16 tile of the `[2560,6144]` matrix | 24 KB (f32 equiv.) | **12 KB** | fits comfortably at bf16; the FULL matrix (6144x2560, 31.5 MB bf16) never needs to be resident, only one `[Ktile,Ntile]` tile at a time -- this is exactly what `gemm-bfp16-ebs8` already tiles for |
| wo GEMM (K=4096,N=2560) | same tiling | `[96,64]` bf16 tile of `[4096,2560]` | 24 KB | **12 KB** | fits |
| w1/w3 GEMM (K=2560,N=9728) | same tiling | `[96,64]` bf16 tile of `[2560,9728]` | 24 KB | **12 KB** | fits |
| w2 GEMM (K=9728,N=2560) | same tiling | `[96,64]` bf16 tile of `[9728,2560]` | 24 KB | **12 KB** | fits |
| RoPE (interleaved-pair variant) | `inv_freq[64]` + 256-entry sin/cos LUTs (same as `rope-lut`) | `[64]` f32 + 2x`[256]` bf16 | 256 B + 1 KB | -- | **trivial**, sizing was never the issue for this brick, correctness (pairing convention) was |
| GQA-aware decode attention (HD=128, M=1) | one streamed K/V tile, `TKV` keys x `HD=128` x 2 (K and V) | e.g. `TKV=64`: `[64,128]`x2 bf16 | -- | 32 KB **per tile**, **64.01 KB live** | `TKV=64` does **NOT** fit; `TKV=32` does, and is FORCED. Built and device-gated 2026-07-31 at 1.689e-02. The 32 KB above is ONE tile, but an objectFIFO is double-buffered (depth 2), so the live cost is `2 * (2*TKV*HD + 2)` bf16 = **64.01 KB** at `TKV=64`, against a 64 KB core tile -- over budget on the KV fifo alone, before q/ctx/accumulator/stack. `TKV=32` holds `TKV*HD` at 4096 and lands at 32.01 KB, byte-identical to the shipped HD=64/`TKV=64` footprint. The "roughly halving" instinct in this row was right; the "fits" verdict next to `TKV=64` was not. |
| GQA-aware causal prefill attention (fast model, M<=10) | full `[M,M]` score matrix, `M<=10`, all 32 heads batched or looped | `[10,10]` f32 x 32 heads = 12.8 KB total, or 400 B/head looped | 400 B/head | -- | **trivial** -- M<=10 is small enough that flash-attention's online-softmax complexity isn't needed here at all, a naive resident scores matrix is fine per-head |
| lm-head-argmax reparam (VOCAB=4096 or ~4097) | weight tile `[HIDDEN_tile, VOCAB_tile]`, same family as the GEMMs above | e.g. `[96,64]` bf16 of `[2560,4096]` | 24 KB | 12 KB | fits, same reasoning as the GEMMs |

The two structurally different-from-anything-existing pieces are the embedding gather (index-driven
DMA, not stream-order) and the RoPE pairing-convention fix (a correctness bug waiting to happen if
`rope-lut` were reused unmodified) -- both cheap in L1 terms, neither cheap to get wrong silently.

## 7. What's verified vs inferred (repeated from the script, for the doc-only reader)

**VERIFIED**: op graph and every shape above (read from `s2_model.cpp` + the GGUF's own tensor
table/KV metadata -- `read_ar_hparams` reads the identical keys/defaults as `load_shared`, so a
drifted GGUF fails loud rather than silently mis-sizing). q6_k dequant (byte-exact vs a scalar
port of the ggml C source, on real GGUF bytes). RoPE convention mismatch (read both `ggml.h`'s
mode constants/diagram AND `rope-lut/golden.py`'s actual indexing). `--temperature 0` determinism
and the RAS override (read `s2_sampler.cpp` and `s2_generate.cpp` directly). `mha_decode.cc`'s
`HD=64` hardcoding and lack of GQA (read the kernel source).

**INFERRED**: that a from-scratch non-KV-cached recompute (what `s2_ar_ref.py` does) is
mathematically equivalent to `eval_cached`'s incremental cache -- follows from attention being a
pure function of the causal history, not independently diffed step-by-step against a live run
(needs the new dump points in #5 to close that loop).

**NOT BUILT**: the tokenizer/prompt-template (text -> initial `flat_tokens`), on-device kernels
for any of the "NEW" rows in #4/#6 (this was a host-only, no-NPU-access task by explicit
constraint), and the `--verify-dump` wiring in `s2_ar_ref.py` (the dump points it would consume
don't exist yet).
