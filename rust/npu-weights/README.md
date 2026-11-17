# CPU source-native weight views

`npu_weights::source_native` is an opt-in CPU library for reading encoded weights without baking
another checkpoint or expanding a whole tensor/model into floating-point arrays. Existing
`ModelSpec` baking and runtime defaults are unchanged.

Supported inputs:

- Safetensors F32, F16, BF16, I32 and I64 sections, including local sharded indexes. I32 code/zero-point
  and I64 metadata tensors remain integers; I64 is admitted as container metadata only, so it is
  never a dense weight encoding. Unsupported dtypes are rejected, not silently skipped.
- Explicitly selected AutoAWQ GEMM u4 tensor triples: `qweight[K,N/8]`,
  `qzeros[K/group_size,N/8]`, `scales[K/group_size,N]`. Scale storage may be F16, BF16 or F32.
  The constructor selects this packing contract explicitly; it does not infer compatibility from
  an AWQ filename. GPTQ zero conventions and other AWQ packing variants are not aliases.
- Little-endian GGUF v3 Q4_0, Q4_1, Q6_K and F32 tensors. The reader validates every tensor's type;
  a nominal Q4_0 filename is not a guarantee that every tensor uses that codec. Higher-rank tensors
  and other GGML encodings require additional declared readers.
- Explicitly selected compressed-tensors `pack-quantized` symmetric u4 section groups:
  `weight_packed[rows,cols/8]` of little-endian I32 words holding eight unsigned nibbles each,
  consecutive along K from the lowest nibble, `weight_shape[2]` I64 metadata holding the logical
  `[rows,cols]`, and `weight_scale[rows,cols/group_size]` BF16. The signed source code of nibble `c`
  is `c - 8`, so no zero point is stored, and scale bits are preserved exactly. A declared zero
  point, a group index map, another strategy, bit width, symmetry, scale type or packing order is
  rejected rather than guessed at.

All views implement `WeightTensor`. Its descriptor is logical `[output_rows, input_columns]`;
`decode_row` reconstructs only a caller-requested row slice into the caller's buffer. It is a CPU
reference operation, not a high-throughput matmul implementation or a claim of NPU support.

## Read encoded sources

```rust,no_run
use npu_weights::source_native::{
    awq::AwqTensor, safetensors::SafetensorsSet, Limits, WeightTensor,
};

# fn example() -> anyhow::Result<()> {
let source = SafetensorsSet::open_index("model.safetensors.index.json", Limits::default())?;
let name = "model.layers.1.mlp.down_proj";
let weight = AwqTensor::new(
    name,
    source.get(&format!("{name}.qweight"))?.clone(),
    source.get(&format!("{name}.qzeros"))?.clone(),
    source.get(&format!("{name}.scales"))?.clone(),
    128, // From the reviewed source config, not an inferred default.
)?;
let mut values = [0.0; 32];
weight.decode_row(0, 0, &mut values)?;
# Ok(())
# }
```

For GGUF, open `source_native::gguf::GgufFile` with the same limits, then select a tensor with
`get(name)`. Floating safetensors entries can be wrapped with `DenseTensor::new`; shape/byte lengths
are checked rather than reinterpreted based on size alone.

## Declare compressed-tensors parameters explicitly

```rust,no_run
use npu_weights::source_native::{
    ct::{PackQuantizedFormat, PackQuantizedTensor, WeightSections},
    safetensors::SafetensorsFile, Limits, ScalarType, WeightTensor,
};

# fn example() -> anyhow::Result<()> {
let source = SafetensorsFile::open("model.safetensors", Limits::default())?;
let prefix = "model.layers.0.self_attn.q_proj";
let codec = PackQuantizedFormat::declared(
    "group", // From the reviewed source config, never inferred from names or shapes.
    4,
    true,
    128,
    ScalarType::Bf16,
)?;
let weight = PackQuantizedTensor::new(
    prefix,
    WeightSections {
        weight_packed: source.get(&format!("{prefix}.weight_packed"))?.clone(),
        weight_shape: source.get(&format!("{prefix}.weight_shape"))?.clone(),
        weight_scale: source.get(&format!("{prefix}.weight_scale"))?.clone(),
        weight_zero_point: None,
        weight_g_idx: None,
    },
    codec,
)?;
let mut values = [0.0; 32];
weight.decode_row(0, 0, &mut values)?;
# Ok(())
# }
```

`PackQuantizedFormat::declared` records a caller claim, and because its fields are public,
`PackQuantizedTensor::new` revalidates the group size and the BF16 scale type it can actually decode.
The declared column count and the scale section's width do constrain a group size - one that does not
divide the columns or does not match the scale width is rejected - but they cannot prove which size
the publisher recorded, and safetensors discards `__metadata__`, so nothing at this layer ties a
declaration to a file's own `quantization_config` or revision: importers must verify strategy, bit
width, symmetry, group size and scale type against the selected source config before constructing a
view, and route the companion sections themselves. The reader owns bit-exact codes and scale bits
plus the exact product, not value, payload, model or device admission: a non-finite or huge scale
propagates instead of being rejected, saturated or dropped, which stays the higher boundary's concern
(as `plan`'s scaled-BF16 gate is). No local requantization, code rebase or scale reselection happens
here.

## Stream into caller-owned storage

```rust,no_run
use npu_weights::source_native::{write_encoded, WeightTensor};
use std::io::Write;

fn copy_weight(weight: &dyn WeightTensor, destination: &mut dyn Write)
    -> anyhow::Result<()> {
    let mut scratch = [0u8; 64 * 1024];
    let sections = write_encoded(weight, destination, &mut scratch)?;
    // Keep weight.descriptor() and these relative section placements with the destination.
    // They are not a self-describing checkpoint or an automatically published cache.
    assert_eq!(sections.len(), weight.sections().len());
    Ok(())
}
```

The writer copies encoded bytes unchanged: GGUF blocks or AWQ words, zero points and stored scale
bytes, or the packed words, `weight_shape` metadata and scale bytes of a compressed-tensors view. It
does not quantize, clip, replace zero points, cast scales or allocate an unpacked tensor.
Transformation scratch is the caller's buffer; parsed header/schema memory is separately bounded
by `Limits`. A caller that writes into a `Vec` still owns the resulting output allocation. The
codec reference decoders use fixed-size word/block scratch, apart from the requested output.

Sources must remain unchanged while their views are alive. Each span retains its file handle;
dropping a container reader does not invalidate its tensors. Source paths/byte spans are available
for caller provenance, but paths alone are not content hashes or immutable model revisions. Shard
index paths are confined to the index directory, including symlink resolution. Snapshot layouts that
symlink shards into an external blob cache must use explicitly opened per-shard `SafetensorsFile`
views instead of `SafetensorsSet::open_index`.

Destination publication, partial-write cleanup, source revision/hash verification and cache retention
belong to the caller. An I/O error may leave a partial destination. This library does not download
models, select a different quantized release, map architecture-specific tensor names, create device
buffers, execute inference or remove existing artifacts. Those require separate integration and
acceptance against the selected source model.

## Bind graph-owned BF16 input requirements

`source_native::plan::WeightContract` describes ordered row selections/concatenation, explicit
float32 scaling followed by BF16 round-to-nearest-even, and zero fill. The graph records these
requirements at its transformation sites, using the architecture's tensor bindings; the loader
does not infer them from buffer names or lengths. Version 1 accepts BF16 rank-1/2 source tensors.
It is not a GGUF/AWQ-to-device adapter or a quantization operation.

`WeightPlan::bind_config_json` validates checkpoint configuration, source types/shapes, selections,
complete required-buffer coverage and exact lengths against independently supplied program extents.
`write_buffer` streams into a buffer-relative `(offset, bytes)` sink using caller-owned scratch
(at least two bytes). Copy operations preserve bits; scaling rejects non-finite results. A failed
read, scale or sink may leave a partial destination. The retained embedding span can be mapped
without reopening its path; the unsafe mapping API requires source immutability for its lifetime.

The fused-decode generator records an embedded `weight_inputs` ABI for its supported tied-BF16
contract. Other graph/encoding combinations do not claim support merely because sizes coincide.
`npu-models::llm::NpuDecodeStep::from_checkpoint` is an explicit opt-in entry point using that ABI,
one shared decode/prefill arena, and source-backed host embedding rows. It currently opens a local
`model.safetensors` plus `config.json`; missing/invalid contracts never select prepared blobs as a
fallback. Existing constructors/defaults are unchanged. Runtime records the source path and small
config/contract hashes separately from program identity. Ordinary opens do not scan the checkpoint
for a content fingerprint: source provenance reports `sha256: null`. Structural validation is not
verification against a trusted expected digest, and a path is not a content identity. Sources must
still remain unchanged while their views are alive. For an explicit offline fingerprint, use
`sha256sum checkpoint/model.safetensors`; the loader does not cache or silently reuse that result.
CPU byte comparison is not fresh compiler-emission, device, cold-load or quality acceptance.

The host-only profiling example separates raw reads, source-reader reads, explicit SHA-256,
contract-driven transformations and prepared-buffer reads. Use an optimized build and control
page-cache state separately; repeated reads may measure memory rather than storage:

```sh
cargo run --release -p npu-weights --example profile_source_native -- \
    read checkpoint artifact/meta.json
cargo run --release -p npu-weights --example profile_source_native -- \
    hash checkpoint artifact/meta.json
```

These probes do not allocate device storage or time a complete model open.

## Checks

```sh
cargo test -p npu-weights --test source_native --test source_native_safetensors \
    --test source_native_awq --test source_native_ct --test source_native_gguf \
    --test source_native_shared --test source_native_plan
```

Fixtures use hand-built encoded bytes and independent expected values, including group/word/block
boundaries, malformed headers and exact encoded streaming. They do not substitute for real-source
numerical quality or device correctness/performance acceptance.
