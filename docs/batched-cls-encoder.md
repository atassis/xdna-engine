# Packed CLS encoder (source-level, device gate pending)

BERT supports an opt-in instance that packs independent passages into shared-weight
Q/K/V, output-projection and FFN GEMMs. Attention is bidirectional **within each
passage only**. Attention, LayerNorm and GELU remain host stages; this is not an
all-NPU encoder or a measured throughput improvement.

## Instance contract

Use the complete opt-in [BGE CLS scenario](../scenarios/bge-base-cls.toml), or set
these fields in a separately selected BERT scenario:

```toml
[model]
# Other BERT-base dimensions remain 768/3072, 12 heads of 64, max_seq <= 512.
precision = "native"

[embeddings]
pooling = "cls"
normalize = true
batch_rows = 1024
```

The snippet shows fields to merge, not a complete model manifest. The opt-in scenario
declares the full three-stem closure below. Existing `bge-base.toml` remains mean-pooled;
no alias, registry selection or service default is changed.
`batch_rows` must fit one complete passage and be a positive multiple of 128
(the native 32-row tile times four array rows). Each capacity has distinct actual
program and instruction-stream identities. A missing matching artifact is an error,
not a fallback to 512-row instructions.

The packed instance uses **local native BF16, tile 32x32x32**, regardless of the
single-text instance's process-global precision/tile selectors. Explicit `bf16`
(block-floating emulation) and `int8` model precision are rejected for this mode.
Int8 has a matrix-global activation scale; block-floating passage independence is
not validated here. These are implementation support boundaries, not silicon limits.

B is the document count. Useful M is the sum of token lengths including tokenizer
special tokens. Positions reset to zero per passage. There is no inter-passage
padding in this native mode. Every GEMM still executes the configured physical
capacity, so padding work is `batch_rows - useful_M`. Collections are split at
passage boundaries, preserving order and tails. Overlength passages are rejected
before executing any request prefix; tokenizer-config truncation/padding is disabled
at this boundary. Empty strings retain tokenizer special tokens. The Rust batch API
returns an empty vector for an empty collection; HTTP retains its existing 400 for
`input: []`.

Omitting `batch_rows` retains single-text/truncation behavior and explicit serial
array compatibility, including the mean scenario. ESM arrays remain explicitly
serial. Fallible `Embedder::embed_one` and `Servable::Text` calls on a configured CLS instance
use the singleton packed path, including full token-length validation. The infallible
inherent `embed` method remains a legacy/internal truncating control, not the strict
serving API. A singleton packed call uses the instance's native/M artifacts; it does
not allocate a second 512-row context or duplicate device weights. The per-passage
limit is still 512 **including special tokens**, not the physical capacity of 1024.

## Artifact recipe and preflight

For capacity 1024 the closure is:

- `1024x800x3072_32x32x32_8c_modalsilunat` (resident program)
- `1024x800x768_32x32x32_8c_modalidnat`
- `1024x800x3072_32x32x32_8c_modalidnat`

The complete `scenarios/bge-base-cls.toml` declares exactly these stems under
`[kernels.whole_array]`, using the existing `scripts/kernel_families/whole_array.sh`
recipe and its source closure.
With a lock-pinned, source-synchronized compiler checkout, the build commands are:

```sh
# Compiler/resource authorization and toolchain setup precede these commands.
for stem in \
  1024x800x3072_32x32x32_8c_modalsilunat \
  1024x800x768_32x32x32_8c_modalidnat \
  1024x800x3072_32x32x32_8c_modalidnat; do
  bash scripts/kernel_families/whole_array.sh build "$stem" "$MLIR_AIE_ROOT" || exit $?
done
```

The recipe forwards M through `Makefile.modal` to
`designs/whole_array_fused/whole_array_modal_iron.py`. Its native contract recorder
checks the **emitted** runtime operand extents and hashes the built xclbin/stream.
`contract_<stem>.json` must accompany each pair in the resolved whole-array kernel
directory. Runtime preflight verifies extents, precision, tile and current byte
hashes before opening the device through the registry. Copying/renaming only an old
512-row pair is insufficient. Receipts are build evidence, not device correctness.
Each stem needs `final_<stem>.xclbin`, runtime-selected `insts_<stem>.txt` (or `.bin`
when txt is absent), and the original `contract_<stem>.json`. If both instruction
files exist, txt wins: a receipt hashing only bin cannot certify a stale preferred txt.

The existing publisher preserves receipt bytes and asks the runtime-owned validator
to check the selected declared Native source closure before copying, then validates
the actual destination bytes before generating source-bound pair manifests. It never
synthesizes receipts at publish time. Declared verification rejects missing receipts
as missing and malformed/shape/hash mismatches as invalid contracts, even if a new
pair manifest hashes the wrong bytes consistently. Root and family toolchain stamps
remain required. Legacy Fast bundles keep their pair/source provenance semantics;
unselected Native experiments are not a new gate.

The existing build/publish/verify installation route can package this scenario; the
scenario alone does not register or deploy it. Publication into a reused directory
is not transactional, and a failed stage must not be selected by a running service.
Keep the mean scenario and its matching artifacts available for rollback.

For 1024 rows, one activation BO is `1024*800*2 = 1,638,400` bytes and one output
BO is `1024*3072*4 = 12,582,912` bytes; the existing optional two pipeline slots
have the same capacities. The local tile working set does not grow with M: at the
native defaults A/B each use `2*32*32*2 = 4096` bytes, C uses
`1*32*32*4 = 4096` bytes, plus the generator's 3328-byte stack policy and RTP storage.
Actual ELF stack frames, L1 allocation, 16-KB program budget, routing and DMA streams
still require compiler checks. BERT uses nine matrix dispatches per block per packed
batch (four attention projections, FFN expansion, four FFN contraction partials).
Weights are reused across all packed rows and bounded batches.

## Verification boundary

CPU fixtures cover per-passage positions, full packed projections/FFNs with segmented
attention, neighboring-passage mutation, B=1, reordering/repeats, normalized CLS offsets,
empty strings, overlength, useful M>512 and tails. A naive concatenated-attention
control fails the isolation predicate. Fake API/service tests verify one ordered
scalar/singleton/ordered-array requests, failure propagation and output-count checks.
Recipe/preflight fixtures reject invalid grids, emitted 512-row shapes, missing receipts
and replaced bytes. Staged-root fixtures resolve the declared closure without a build
tree; subprocess publisher fixtures use only disposable roots and a fake manifest seam.

Source/offline fixtures do not establish real served CPU/Native numerical agreement,
HTTP transport stability, installation success or deployment readiness. Real service
scalar/singleton/multi comparisons and 513-token rejection remain separately gated.
These checks do not compile or execute larger-M kernels. Before adoption, build with
the pinned compiler and check stack/L1/program budgets, then authorize an isolated
device gate comparing repeated serial **Native** controls against packed Native on
the same frozen texts. Also report the actual FastBF16 baseline so precision cost
is not hidden. Record artifact identities, determinism/numeric tolerance, isolation,
passages/s, useful tokens/s, physical padding, latency, memory and dispatch counts.
No performance claim is established by the CPU fixtures.
