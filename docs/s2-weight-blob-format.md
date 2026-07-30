# S2 AR weight blob: format, manifest schema, and verification

`scripts/s2_weight_prep.py` reads the shipped S2-Pro GGUF and writes a resident **bf16 weight
blob** plus a **JSON manifest** describing it. This is a host-side, load-time preprocessing step:
it dequantizes every q6_k weight matrix once, up front, so a future device-side loader can mmap
the blob and upload tensors by name/offset without ever parsing GGUF or running dequant itself.
This doc specifies the on-disk format and how to verify a blob independently of the tool that
produced it. It does not cover a device-side loader or upload path -- neither exists yet.

## Why dequantize on the host, once

Every large AR weight matrix in S2-Pro (`embeddings.weight`, `codebook_embeddings.weight`, every
`wqkv`/`wo`/`w1`/`w2`/`w3` across all 36 slow + 4 fast transformer layers, `fast_embeddings.weight`,
`fast_output.weight` -- 204 tensors) is GGUF type **q6_k**: 6-bit quants plus 16 signed 8-bit
sub-block scales per 256-element superblock (`ggml/src/ggml-quants.c:dequantize_row_q6_K`, block
layout `ggml/src/ggml-common.h:352-358`). Only the small RMSNorm gamma vectors (154 tensors) are
f16. `scripts/s2_ar_ref.py` ports that dequant to vectorized numpy and validates it byte-exact
against a scalar transliteration of the ggml C source (see that script's own self-test); see
`docs/s2-ar-graph-map.md` section 0 for the full account.

q6_k's superblock layout has no relationship to this project's `dequant-int4-group` AIE2P brick
(int4, one f32 scale + zero-point per group) -- porting q6_k dequant to an AIE2P kernel would be
new, fiddly work for a conversion that runs once per model load, not once per token. Doing it on
the host with numpy, once, and shipping the already-dense result is strictly simpler and costs
tens of seconds at load time. `scripts/s2_weight_prep.py` is that host conversion tool.

## On-disk layout

Running the tool against an output directory produces exactly two files:

```
<outdir>/
  s2_ar_weights.blob   # flat binary: every tensor's raw bytes, back to back, 64B-aligned
  manifest.json         # format version, source GGUF identity, alignment, per-tensor index
```

The blob has no header of its own -- it is pure tensor payload. Every byte range inside it is
located exclusively through `manifest.json`'s per-tensor `offset`/`nbytes` pair. This keeps the
blob itself a dumb, mmap-able byte array; all the structure lives in the (small, human-readable)
manifest.

### Alignment

Every tensor starts at an offset that is a multiple of **64 bytes**, the `alignment_bytes` field
in the manifest. Padding between tensors is zero-filled.

64B is not an arbitrary round number -- it is a real AIE2P hardware alignment tier already
documented in `docs/aie2p-brick-catalog.md`'s `load_v`/`store_v` row (**alignment 16/32/64B**;
that same doc also flags "misaligned aligned-load silently truncates the address" as a live
footgun, no fault, no assert). A 64B boundary is exactly the size of one 16-lane f32 (or 32-lane
bf16) vector register store. Aligning every tensor to 64B means a future device-side loader can
issue a DMA straight off the blob's mmap for any tensor, at any vector width this project uses
today, with no bounce-copy to fix up alignment first. It is deliberately the *floor* named in this
project's own hardware notes, not a number picked for this tool alone.

### Element order and dtype

Each tensor's bytes are its numpy array, `.tobytes()`, C-contiguous (row-major), in the dtype
recorded in its manifest entry. There is no byte-swapping or repacking beyond the dtype cast
itself -- what you get by `np.frombuffer(blob_bytes, dtype=<manifest dtype>).reshape(<manifest
numpy_shape>)` is the tensor.

## The `ne` vs numpy-shape convention

GGUF stores tensor dimensions as ggml `ne`, **fastest-axis first**. This tool (via
`s2_ar_ref.tensor_numpy_shape`) reverses `ne` to get a numpy C-order shape, matching
`scripts/gguf_extract.py`'s existing convention: a ggml `ne=[k, c_out, c_in]` weight becomes numpy
shape `[c_in, c_out, k]`, and a 2D linear weight's numpy shape is `(out_features, in_features)`
(PyTorch `nn.Linear` style). Concretely, in this checkpoint: `layers.0.attention.wqkv.weight` has
`ne=[2560, 6144]` and numpy shape `(6144, 2560)`.

This convention is validated, not assumed: `s2_ar_ref.py` checked it against real stage-4 weights
via an adjoint identity at 1.6e-07 relative error. Getting a transpose wrong here is a real,
previously-hit failure class in this project (the conv-1d weight transpose, the RoPE pairing
convention) -- both passed every self-consistent check and were only caught by an end-to-end
oracle. That is why this tool's self-test (below) asserts the **shape tuple**, not just the
element count, against `tensor_numpy_shape(ne)` recomputed independently from the source GGUF: an
element-count check alone passes a silently-transposed tensor.

## Dtype policy: why bf16 for weights, native f16 for norm gammas

| source GGUF type | output dtype | reasoning |
|---|---|---|
| q6_k (14) | bf16 | the format this tool exists to unblock: 6-bit quantized -> dense bf16, ~8 bits of mantissa precision, matmul-ready. |
| f16 (1) | f16 (native passthrough) | RMSNorm gamma vectors are already f16 in the source. Casting them to bf16 would trade 10 mantissa bits for 7 -- a pure precision loss -- for **zero** size benefit, since both are 2 bytes/element. There is no reason to do it. A consumer widens f16 to f32 for the elementwise multiply exactly as easily as it would widen bf16. |

Any other source GGUF type makes the tool raise loud rather than guess: this checkpoint was
observed to use only q6_k and f16 for every AR tensor (358 total: 204 q6_k + 154 f16, confirmed by
`s2_ar_ref.py --report-types`), and a future checkpoint using a different quant type should fail
here with a clear message, not silently emit wrong bytes.

## Manifest schema

`manifest.json` is a single JSON object:

| field | type | meaning |
|---|---|---|
| `format_version` | int | this document describes version `1`. |
| `alignment_bytes` | int | tensor offset alignment (64). |
| `blob_file` | string | filename of the blob, relative to the manifest's directory. |
| `blob_bytes_total` | int | total blob size in bytes (== the blob file's size on disk). |
| `tensor_count` | int | number of tensors in `tensors`. |
| `full_set` | bool | `true` if this manifest covers the complete 358-tensor AR weight set; `false` if produced with `--layers` (a subset -- see "Partial conversions" below). |
| `slow_layers_included` / `fast_layers_included` | int | how many of the 36 slow / 4 fast transformer layers this manifest covers. |
| `slow_layers_total` / `fast_layers_total` | int | the model's actual layer counts, from GGUF hparams -- lets a reader tell a smoke-test manifest from a full one even without `full_set`. |
| `generated_at_unix` | float | when this blob was written. |
| `source_gguf` | object | see below. |
| `hparams` | object | a copy of the AR architecture hyperparameters (`block_count`, `embedding_length`, `head_count`, `vocab_size`, `codebook_size`, etc.) read from the GGUF's own KV metadata -- convenience for a downstream loader that wants shapes without re-reading the GGUF. |
| `tensors` | array | one entry per tensor, see below. |

`source_gguf` identifies exactly which GGUF this blob was converted from, without hashing the
multi-gigabyte tensor payload:

| field | meaning |
|---|---|
| `path` | resolved (symlink-followed) path to the source GGUF at conversion time. |
| `size_bytes` | full size of the source GGUF file. |
| `data_start` | byte offset where GGUF tensor data begins (i.e. header + KV metadata + tensor directory, padded to the GGUF's own alignment). |
| `header_sha256` | SHA-256 over exactly the first `data_start` bytes of the source file -- the header/metadata/directory region, **not** the multi-GB tensor payload. Two GGUFs with the same architecture but different weights will differ here only if their metadata differs; this is an identity check for "which file/config", not a full-weight checksum. |
| `header_hash_covers_bytes` | same value as `data_start`, named for clarity at the call site. |

Each entry in `tensors` is:

| field | type | meaning |
|---|---|---|
| `name` | string | GGUF tensor name, e.g. `layers.3.attention.wqkv.weight`. |
| `dtype` | string | `"bf16"` or `"f16"` (see dtype policy above). |
| `numpy_shape` | int array | `tensor_numpy_shape(ggml_ne)` -- numpy C-order shape. |
| `ggml_ne` | int array | the source GGUF's `ne` (fastest-axis-first) for this tensor, unreversed. |
| `source_gguf_type` | string | the GGUF quant type name this tensor was read as (`"q6_k"` or `"f16"`). |
| `offset` | int | byte offset into the blob file, always a multiple of `alignment_bytes`. |
| `nbytes` | int | byte length of this tensor's region (`prod(numpy_shape) * itemsize`, no padding included). |

Example entry (from a real conversion run):

```json
{
  "name": "layers.0.attention.wqkv.weight",
  "dtype": "bf16",
  "numpy_shape": [6144, 2560],
  "ggml_ne": [2560, 6144],
  "source_gguf_type": "q6_k",
  "offset": 1049251840,
  "nbytes": 31457280
}
```

To read a tensor from a manifest + blob pair (reference reader, numpy/`ml_dtypes`):

```python
import json, numpy as np, ml_dtypes

manifest = json.load(open("manifest.json"))
blob = open(manifest["blob_file"], "rb")

def read(name):
    e = next(t for t in manifest["tensors"] if t["name"] == name)
    np_dtype = ml_dtypes.bfloat16 if e["dtype"] == "bf16" else np.float16
    blob.seek(e["offset"])
    raw = blob.read(e["nbytes"])
    return np.frombuffer(raw, dtype=np_dtype).reshape(e["numpy_shape"])
```

## Partial conversions (`--layers N`)

`s2_weight_prep.py --layers N` caps both the slow and fast layer counts at `N` (fast only has 4 to
begin with) while always including every top-level tensor (`embeddings.weight`,
`codebook_embeddings.weight`, `norm.weight`, `fast_embeddings.weight`, `fast_norm.weight`,
`fast_output.weight`). This exists for smoke-testing the tool itself without paying for all 358
tensors; a partial manifest sets `full_set: false` and records exactly how many layers it covers,
so nothing downstream can mistake it for a complete weight set.

## Full-set size: the blob is bigger than the q6_k source

This is worth stating in bytes, not just noting: converting the complete 358-tensor AR weight set
(4,561,852,416 elements across the two source dtypes) produces a blob of about **8.5 GiB**
(9,123,704,832 bytes), against a q6_k + f16 source of about **3.5 GiB** (3,742,403,072 bytes) as
those same tensors sit in the GGUF -- a **2.44x expansion**. That ratio is exactly q6_k's packed
density: ~6.56 bits/weight (210 bytes per 256-element superblock) versus bf16's 16 bits/weight,
16/6.56 = 2.44. This is arithmetic, not a measurement artifact, and it is reproducible from the
GGUF's tensor directory alone (no dequant needed -- `s2_weight_prep.py` prints it every run via an
analytic scan, `analytic_size_report()`).

8.5 GiB is the number a device-side loader will eventually need to stage from LPDDR at model load
time. It is a one-time load cost, not a per-token cost, but it is real: this project counts bytes
moved, and a ~4.5B-element bf16 weight set is the thing that has to move. Whether some or all of
that 8.5 GiB should instead be a re-quantized int8/int4 blob is exactly the kind of FORMAT-lever
tradeoff `docs/aie2p-brick-catalog.md` catalogs generically -- this tool deliberately leaves that
slot empty (see "What this tool does not do" below) rather than deciding it here.

## Self-test: how a blob is verified

`s2_weight_prep.py` always runs a self-test after writing (`--no-selftest` to skip, not
recommended). For a random sample of tensors (`--selftest-samples`, default 12; `--selftest-seed`,
default 0) it:

1. Recomputes `tensor_numpy_shape(ggml_ne)` independently from the source GGUF's tensor directory
   and asserts it equals the manifest's `numpy_shape` **and** that the bytes read back from the
   blob reshape to that exact shape. This is the check that catches a transposed tensor -- see the
   `ne`-vs-numpy-shape section above; a check based on element count alone would pass a transpose.
2. Re-reads the tensor's raw bytes directly from the blob file at its manifest `offset`/`nbytes`.
3. Independently re-dequantizes the same tensor from the source GGUF (a fresh call to
   `s2_ar_ref.read_tensor`, not anything cached from the write pass) and applies the same dtype
   cast the writer used.
4. Asserts the blob bytes are **bit-exact** equal to that independent re-cast -- not within a
   numeric tolerance. The cast (float32 -> bf16 or float32 -> f16) is deterministic, so anything
   short of exact equality means the write or read path is wrong (bad offset, wrong dtype view, a
   padding off-by-one), not that "rounding happened."
5. Also reports, as an informational secondary metric, the max relative error between the blob's
   bf16/f16 values and the original float32 dequant -- this is expected to be nonzero (bf16 keeps
   ~8 bits of mantissa precision; a max relative error around `3.9e-3` -- `2^-8` -- on a large
   matrix is normal bf16 rounding, not a bug) and is reported separately from the pass/fail
   bit-exactness check in step 4.

To verify a blob independently of `s2_weight_prep.py` itself (e.g. from a different tool or a
future device-side loader's own test), reuse the reference reader shown above and compare against
`s2_ar_ref.read_tensor(gg, name).astype(np.float32)` from a fresh `s2_ar_ref.open_gguf()` call on
the same source GGUF -- the manifest's `source_gguf.header_sha256` confirms it is the same file.

## What this tool does not do

- **No device-side loader or upload path.** This tool produces a blob + manifest on the host
  filesystem. Nothing here mmaps it into device-visible memory, DMAs it onto the NPU, or wires it
  into `route_b_kernels/` or the engine's dataflow. That is separate, not-yet-built work.
- **No int8/int4 re-quantization.** bf16 is a deliberate, conservative first target -- it is a
  format every matmul kernel in this project already consumes, so a blob in this format is usable
  the moment a loader exists. Re-quantizing to int8/int4 is a real FORMAT-lever option this
  manifest's per-tensor `dtype` field already admits (a loader can accept mixed dtypes per
  tensor), but choosing precision per weight matrix is a separate, gated decision with its own
  accuracy-gate work, out of scope here.
- **No engine integration.** This is a standalone host tool; it does not touch
  `route_b_kernels/`, `rust/npu-engine/`, or any runtime dataflow.
