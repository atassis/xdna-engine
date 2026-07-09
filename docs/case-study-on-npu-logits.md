# Moving the Whisper decoder's logits and argmax onto the NPU

This is a case study about a small, stubborn piece of a speech-recognition decode
loop, and what it took to move it fully onto the AMD XDNA2 NPU. The headline result
is undramatic on purpose: the decode loop now returns a token id directly from the
device, word-error rate is exactly equal to baseline, and steady-state decode is a
few percent faster. The interesting part is the path there, because the first two
things I built were both worse than the host code I was trying to replace, and the
version that finally won did so by moving less data rather than by computing faster.

## The problem

Whisper's greedy decode loop, per generated token, does two things at the tail:

1. A large matrix-vector multiply against the language-model head (`proj_out`),
   projecting the decoder's hidden state (K=768) up to the full vocabulary
   (about 51,865 entries). That is roughly 40M multiply-accumulates per token.
2. An `argmax` over that logits vector to pick the next token id.

On the host, the `proj_out` matmul in f32 costs about 6 ms per token. It also
produces a wide logits vector that then has to be read back and scanned. My goal
was to push both the matmul and the argmax onto the NPU so that the decode loop
never materializes a full logits vector on the host at all - it just receives a
token id. The payoff I was chasing was CPU offload (power, freeing the host) and,
if the arithmetic cooperated, a latency win too.

Two facts framed the whole exercise, and one of them I only learned by doing it.

First, this is an M=1 workload. Decode generates one token at a time, so every
matmul is really a GEMV with a batch dimension of one. At M=1 the NPU is not
compute-bound - the compute is nearly free - it is dominated by per-dispatch
overhead. The practical rule that fell out of this work: at M=1, latency is
roughly proportional to the number of kernel launches, not the number of FLOPs.

Second, the default decode precision on this path is bf16 (bf16 inputs,
f32 accumulate), not int8. That mattered because I had been warned that the argmax
was "make or break" for correctness under int8 quantization, where a rounding
error can flip which vocabulary entry wins. Under the actual default bf16 path,
that alarm mostly does not apply: bf16 logits with an f32 accumulator hold up well.

## Attempt one: reuse the resident encoder kernel, chunked

The engine already has a resident GEMV kernel that the encoder uses. The obvious
first move was to reuse it for `proj_out` and avoid building anything new.

That kernel has a hard cap: its maximum served output-stream width is 3072
columns. The vocabulary is far wider than that, so a single dispatch covering all
~52k logits was impossible on this kernel. The fallback was to tile the output
into 17 chunks of 3072 (17 x 3072 = 52,224, padding the last chunk's spare columns
to zero weight and dropping them), and run 17 separate GEMV dispatches per token,
concatenating the results into the logits vector.

I also folded the post-layernorm affine transform into the projection weights so
it costs nothing at runtime. Layernorm applies a per-channel scale and shift
before the projection: `logits = ((h - mean)/std * gamma + beta) . W`. That
rearranges exactly, in f32, to `nrm . (gamma (elementwise) W) + (beta . W)`, so I
premultiply the weights by gamma and precompute the `beta . W` bias once. The
GEMV then just consumes the affine-free normalized hidden state. This fold is
algebraically exact, and the transcripts later confirmed it: 16 of 17 test clips
came out bit-identical.

Correctness was fine. WER came in at 0.1209 versus a baseline of 0.1172, a +0.37%
absolute difference caused by exactly one flipped token across the 17 clips - a
Russian word where two candidates were a near-tie under bf16 rounding. That is
inside the marginal-pass band.

But latency was a loss. The 17 chunked dispatches cost about 14.5 ms per token,
against roughly 6 ms for the host f32 matmul they replaced - about +8 ms per token,
or roughly +15% slower decode. This is exactly the M=1 overhead lens biting: I had
replaced one host operation with 17 device launches, and launch count is what
costs. So attempt one was a genuine CPU offload but a latency regression.

## Attempt two: one wide dispatch, blocked by a DMA-stride wall

If 17 launches lose and 1 launch would win, the fix is obvious: build a single
wide kernel that emits all ~52k logits in one dispatch. The arithmetic was
encouraging - a single ~1.5 ms dispatch would beat both the 14.5 ms chunked path
and the 6 ms host matmul, turning the +15% regression into roughly an 8% win.

So I tried to build the wide version of the resident-style (whole-array) kernel at
the full output width. It refused to build, and the error pointed at a real
hardware limit:

```
aie.dma_bd op Stride 3 exceeds the [1:1048576] range.  <size=2, stride=13369344>
```

The whole-array GEMM tiles its `[M, N]` output into 256-row blocks, which makes the
row-block DMA stride `256 * N`. That stride has to fit in a 20-bit field, i.e.
stay at or below 1,048,576. So `256 * N <= 2^20` forces `N < 4096`. That single
constraint is the actual root cause of the 3072 cap I had hit in attempt one
(256 x 3072 = 786,432, which fits; 256 x 4608 = 1.18M, which does not). The
maximum this shape can ever reach is about N = 3840, still around 14 chunks.

So the wide whole-array kernel was not slow, it was impossible. A 2D output tiling
of this width cannot produce a single wide dispatch on this hardware, full stop.
I stopped chasing it.

## What worked: a standalone proj_out GEMV, laid out as vocab-over-M

The stride wall is a property of the 2D output tiling. The way around it is to make
the output one-dimensional. Instead of laying the projection out as a wide `[M, N]`
matrix, lay it out as `GEMV(M = vocab_pad, K = 768)` with the vocabulary as the M
dimension. Then the output is a contiguous vector of length `vocab_pad` - there are
no 2D output strides at all, so there is no stride wall, and it builds at large M.

This GEMV form had actually been the original foundation of the projection work;
the earlier design had moved away from where it ran, not from the math. The one
real obstacle the earlier inline attempt had hit was memory arena size: an inline
elementwise bias-add had forced the padded vocabulary up to 65,536, blowing the
arena past 400 MB.

I fixed both problems at once:

- I built it as a standalone ELF dispatched once per token after the main decode
  ELF, rather than inlining it. On its own the arena is just weight plus the hidden
  input plus the logits output, about 80 MB, with none of the ~250 MB decode
  scratch.
- I folded the bias into the GEMV itself using K-augmentation instead of a
  separate add. I extend K from 768 to 832: the extra weight column carries the
  precomputed `beta . W` bias, and the input carries a constant 1 in the matching
  slot, written once. Padding rows get a large negative bias so they never win.
  Dropping the separate add let the padded vocabulary be 52,224 (a multiple of 8,
  the GEMV tiling constraint) instead of 65,536.

The result: one dispatch, about 80 MB, no stride wall. The generator builds the
ELF in about 30 seconds; the ELF itself is tiny (about 94 KB, the weights live
separately). In the engine it loads once and dispatches per token with no
scratchpad churn.

Latency came out as predicted. The proj_out step dropped from about 6.0 ms on the
host to about 2.0 ms as a single GEMV dispatch, and per-token decode went from
about 52.7 ms to about 48.4 ms - roughly an 8% faster decode. One launch beats both
the host matmul and the 17-chunk path, which is the M=1 lens stated as a win rather
than a regression: latency tracks launch count, and this is one launch.

There was one more correctness detail. When the bias was still being applied on the
host in f32 after the bf16 GEMV output, WER sat at 0.1209 - one fragile
out-of-vocabulary clip wobbled ("Lakka" became "Lakas"). Once I folded the bias
into the GEMV so it accumulated in f32 *before* the single bf16 rounding of the
output, that flip disappeared and WER landed at 0.1172, exactly baseline. Folding
the bias in earlier removed a rounding boundary, and correctness improved.

## Closing the loop: argmax on the device

With complete logits now produced on the NPU, the last piece was the argmax, so the
host could receive a token id instead of a ~104 KB logits vector every token.

I wrote a scalar, tie-exact argmax that matches the host semantics precisely
(strict greater-than, so the first maximum wins, exactly like the reference
argmax). The design avoids any on-device cross-core reduction: each of 8 columns
independently scans its contiguous slice of `vocab_pad / 8 = 6528` entries (which
fits in L1) and emits one packed `[value, index]` partial. The host does a trivial
8-way reduce over 8 partials (64 bytes total) to get the global winner. No streaming
state, no on-chip reduction tree.

One framework detail forced a small trick. The fused operator's runtime sequence
sizes every buffer at 2 bytes per element (uniform bf16), so a raw i32/f32 output
buffer fails a size assertion. I packed each `(value, index)` pair into four
bf16-typed slots (8 bytes): the kernel writes the raw f32/i32 bytes, and the host
reinterprets them. With that, it fuses cleanly into the proj_out ELF.

The argmax is fused into the same ELF, which now emits both the 64-byte partials
and the full logits (the logits are still needed at step 0 for language detection).
In steady state the decode loop reads only the 64 bytes of partials and returns a
token id, dropping the ~104 KB logits readback entirely. Step 0 keeps the host
logits path as a carve-out for language selection.

Validation: running both paths and comparing gave 0 argmax mismatches across 636
tokens, and the wired token-id path measured WER 0.1172 - exactly baseline, with
0 of 17 clips differing.

## Result

The full projection-plus-argmax tail of the decode loop now runs on the NPU. The
host hands off a hidden state and gets back a token id; only step 0 ever reads a
full logits vector. WER is 0.1172, identical to the host-f32 baseline. Steady-state
decode is about 8% faster than the host path, and the per-token host-to-device
logits readback (about 104 KB, and the argmax scan behind it) is gone.

The payoff was mostly in what got deleted, not in what got sped up. The winning
kernel was not a faster matmul - it was the same arithmetic reshaped so it could
ship in a single dispatch and hand back 64 bytes instead of 104 KB. The 17-chunk
version computed the same logits and was correct, but it lost because it launched
17 times; the wide whole-array version could not exist at all because of a DMA
stride limit. On an overhead-bound M=1 workload, eliminating dispatches and bytes
moved beat every attempt to make the operation itself go faster.

The 17-chunk path is kept as a correctness reference and fallback. And the one
result I would still like to nail down is the energy delta: CPU offload was a
primary motivation, but a clean energy A/B needs a quiesced machine, so that number
remains unmeasured.
