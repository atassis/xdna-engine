# Cheap on-chip LayerNorm prologue -- research verdict (feat/r1-ln-cheap)

Question: is there a THIRD path for a pre-matmul LayerNorm that is CHEAP on-chip (~one A-stream,
like the SiLU epilogue), after two prior on-chip attempts failed on device?

Reproduce: `python3 route_b_kernels/ctx_ln/ln_cheap_study.py` (pure numpy, no device).

## Verdict

A cheap on-chip normalize BRICK does exist and is authored (`aie_kernels/ln_cheap_prologue.cc`):
single-pass, one A-stream, ~SiLU-epilogue class cost, NOT the 9 ms the re-streaming 2-pass cost.
BUT it should NOT be shipped as-is. It is a LATERAL move, not a win, for two independent reasons,
and it carries a precision risk gated on an unmeasured property of the real activations:

1. STRUCTURAL (same reason A5 host-fold lost): the host still owns the expensive parts -- the per-row
   reduction (stats) AND the raw-A bf16 pack AND the [T,1024] upload. Moving only the elementwise
   normalize onto the (84%-idle) NPU removes ZERO host->device bytes and only one host arithmetic
   pass. By the data-movement lens there is no bytes-saved win. A5 already measured this exact
   relocation as en break-even / ru +17 ms WORSE (commit 574eca9).
2. PRECISION (new finding): a single-pass prologue must pack RAW x (pre-center) to bf16, and a
   DC-heavy row loses its AC signal in bf16. The normalize rel err scales with |mean|/std (table
   below): fine when |mean|/std <~ 2, breaks the 1e-2 gate by |mean|/std ~ 4-13. The host LayerNorm
   avoids this by centering BEFORE the bf16 cast. Whether the real Parakeet encoder pre-norm inputs
   are benign is UNMEASURED -- that is the gating device experiment, and it only decides correctness,
   not the (already-negative) structural economics.

So: on-chip LN stays on the host. The brick + study are kept as a proven-cheap primitive that the
device gate can pick up IF a future design removes the host reduction too (see "what would change
the verdict").

## Angles tried and why each fails on this HW

| angle | mechanism | why it fails |
|---|---|---|
| 2-pass on-chip reduction (prior, `mm_ln_prologue.cc`) | stats pass re-streams A, then normalize+matmul | full-K A row [64,1024] bf16 = 128 KB > 64 KB L1 -> A re-streamed from L3 -> ~9 ms/dispatch (NPU 98 ms -> ~900 ms). DEAD: L1 capacity. |
| epilogue-correction (prior) | raw x@W' in bf16, then -mean*colsum(W'), *inv | catastrophic cancellation. Confirmed below. DEAD: numerics. |
| K-augmentation (subtract mean in the f32 accumulator) | A_aug=[x|mu], W_aug=[W;-colsum] -> (x-mu)@W in f32 acc | mu must enter the bf16 datapath; bf16-mu * large colsum(W) cancels just as badly (table A2: rel 16.6). DEAD: numerics (bf16 mean). |
| small-M tile (m=8 so [8,1024]=16 KB fits L1) | full-K row resident -> 2-pass local, no L3 re-stream | m=8 vs m=64 = 8x more weight re-streams. fc1 weight [1024,4096] bf16 = 8 MB -> +56 MB/dispatch @ ~50 GB/s ~ +1.1 ms of weight DMA, to save an LN the host does in ~1-2 ms total. DEAD: trades LN re-stream for a bigger weight re-stream (data-movement lens). |
| **single-pass prologue-apply (this branch)** | host does cheap stats [T,2]; NPU applies (x-mu)*inv in one fused A pass; stats ride IN-BAND on the A stream | CHEAP and bounded -- but structurally lateral (reason 1) + |mean|-gated precision (reason 2). Viable-but-not-worth-it. |

## Numbers (bf16-emulated numpy; rel err = L2 relative to the true-f32-LN reference)

Well-conditioned weight (W ~ randn/sqrt(K)):

```
                              |mean|/std   epilogue-corr   prologue(f32)   prologue(bf16-mean)   kaug(bf16-mean)
benign                            ~0        2.9e-3          2.9e-3          3.5e-3                2.3e-3
mild                              ~1        4.1e-3          3.1e-3          3.9e-3                3.8e-3
high                              ~3        1.3e-2          7.4e-3          9.6e-3                1.1e-2
extreme                           ~13       6.1e-2          2.9e-2          4.0e-2                4.7e-2
```

DC-biased weight (W ~ randn/sqrt(K) + 0.5, so colsum(W) ~ 512 -- the realistic catastrophe):

```
                              |mean|/std   epilogue-corr   kaug(bf16-mean)
benign                            ~0        5.9e-2          5.2e-2
mild                              ~1        1.6e+0          1.3e+0
high                              ~3        5.1e+0          4.3e+0
extreme                           ~13       1.75e+1         1.66e+1
```

Reading:
- Epilogue-correction is DEAD (confirmed): rel err grows with |mean|/std even on a benign weight,
  and EXPLODES to > 1 (garbage) with any DC-biased weight. Matches the prior 33x report.
- K-augmentation with a bf16 mean dies the SAME way -- putting the subtraction in the f32 accumulator
  does NOT help, because the mean itself is bf16-rounded before it multiplies the large colsum.
- prologue-apply is the ONLY approach that stays bounded. mean-delivery precision (f32 vs double-bf16
  vs bf16) is a MINOR factor (double-bf16 ~= f32); the DOMINANT error is packing raw DC-heavy x to
  bf16, so prologue error tracks the bf16-store floor (~3e-3) in benign/mild and drifts to ~1e-2 by
  |mean|/std ~ 4, ~3e-2 by ~13. It clears the 1e-2 gate ONLY while |mean|/std <~ 3-4.

## Cost argument for the cheap brick (why ~0.1 ms, not 9 ms)

The 9 ms 2-pass cost was: (i) a full stats pass over A + (ii) a SECOND full A-stream from L3 for the
apply pass, because the full-K row does not fit L1. The cheap brick deletes BOTH:
- stats moves to the host (cheap [T,2] reduction, the same one host LN already does);
- the apply is one elementwise mul-add per A element, fused INSIDE the existing single A DMA
  (no second stream) -- the input-side twin of the SiLU epilogue, which is already absorbed in the
  98 ms bucket, not a +800 ms event.
Extra DMA: one prepended stats "k-block" (3*PRO_M bf16 = 384 B/tile) in-band on the A channel -- no
3rd input channel (the compute tile has 2, both used by A/B; same constraint that forced bias via
K-augmentation). On-chip work per output tile = one A-tile touch (~one k-block x K/k) ~= the SiLU
epilogue's per-tile cost -> ~0.1 ms class, not 9 ms.

The cost is genuinely cheap. What is NOT cheap-positive is the SYSTEM: the host keeps the reduction,
the raw-A pack, and the upload, so wall-clock e2e is expected break-even-to-negative (reason 1).

## What would change the verdict (the only path to a real win)

The brick only pays if the HOST reduction also leaves the host -- i.e. compute the per-row stats
on-chip too, WITHOUT a second A-stream. That needs the full-K row co-resident in L1, which it is not
at m=64/K=1024. Two futures could unlock it, both out of scope here:
- a fused resident block that keeps A resident across LN+matmul (the r1-resident-encoder-block
  direction) so the stats pass reads L1, not L3 -- then stats+apply are both ~free on-chip and the
  whole LN leaves the host;
- an L2-staged reduction if a future array topology gives the compute tile a 3rd input.

## Device gate (IF the owner wants to measure -- NOT run here; single-tenant NPU, orchestrator-gated)

This is authored + numpy-validated ONLY. Before any device work, the decisive cheap experiment is a
HOST-ONLY probe (no xclbin): dump the real Parakeet encoder pre-norm activations and print the
per-row |mean|/std distribution across the fc1/pw1 LN sites. If the p99 is <~ 3, the prologue is
numerically safe; if not, it is dead on correctness and the structural argument is moot anyway.

Only if the probe is green AND a design removes the host reduction is a full build warranted:
1. generator: prepend a stats k-block to the A objectFIFO; core peels it via `ln_cheap_load` into
   core-local mu/inv, then `ln_cheap_apply` before each `matmul(...)`. End the generator with a bare
   `Program(dev, rt).resolve_program()` (never `SequentialPlacer` -- stale wheel API).
2. Rust: reuse the A5 `single_lazy` seam (npu.rs) -- host computes [T,2] stats (double-bf16 mu_hi/mu_lo
   + bf16 inv), packs RAW A, prepends the stats block; gamma folds into W', beta into b'.
3. build the xclbin via `scripts/toolchain_up.sh` + the whole_array modal Makefile (fork toolchain).
4. matched A/B vs the plain host-LN baseline isolating the LN sites; WER gate 8.5% (char-identical to
   A1), rel <= 1e-2 vs `layer_norm_normalize`.

Kernel `template`-keyword hygiene (dependent `to_vector`) is already applied in
`aie_kernels/ln_cheap_prologue.cc`.
