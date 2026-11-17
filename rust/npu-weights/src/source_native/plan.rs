//! Graph-owned BF16 and packed CT input requirements, bound once to lazy checkpoint spans.
//! Operations concatenate in declaration order; offsets passed to the sink are buffer-relative.
//! Failed execution may leave a partial destination. Sources must remain immutable while bound.
use super::{
    ct::{ChainTile, PackQuantizedFormat, PackQuantizedTensor, WeightSections},
    safetensors::TensorEntry,
    FileSpan, ScalarType, WeightTensor,
};
use anyhow::{ensure, Context, Result};
use half::bf16;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// The chain kernel's int4 scale group: the width `ct` groups one chain packet's K by, so a grid's
/// column start must land on it for every tile to carry whole source scale groups. The matching
/// per-tile extent rule is enforced by `ChainTile::encoded_bytes`, not repeated here.
const PACKET_GROUP: usize = 32;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TensorInput {
    pub tensor: String,
    pub shape: Vec<usize>,
}

/// The order a tile grid's tiles occupy in the buffer. Both are spelled explicitly because neither
/// follows from the geometry: transposing a grid's order is the same bytes re-emitted permuted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CtTileOrder {
    /// Row tile outer, column tile inner.
    RowMajor,
    /// Column tile outer, row tile inner.
    ColumnMajor,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum WeightOp {
    Rows {
        tensor: String,
        shape: Vec<usize>,
        first: usize,
        count: usize,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        scale: Option<f32>,
    },
    Zero {
        bytes: usize,
    },
    /// A rectangular region of one `pack-quantized` matrix, tiled into chain packets. The operation
    /// is one compact record: only the region, the tiling and the order are recorded, never the
    /// tiles, so a full-model matrix stays one operation whatever its packet count.
    CtTiles {
        weight_packed: String,
        weight_shape: String,
        weight_scale: String,
        /// The full logical matrix the selected region is a window of, in source coordinates.
        shape: Vec<usize>,
        first_row: usize,
        first_col: usize,
        rows: usize,
        cols: usize,
        tile_rows: usize,
        tile_cols: usize,
        tile_order: CtTileOrder,
    },
}

/// Physical requirements emitted at the graph's transformation sites. Tensor names are the
/// architecture's semantic bindings, not checkpoint paths or weight-content fingerprints.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WeightContract {
    pub version: u32,
    pub config: BTreeMap<String, serde_json::Value>,
    pub buffers: BTreeMap<String, Vec<WeightOp>>,
    pub embedding: TensorInput,
}

#[derive(Debug)]
enum BoundOp {
    Rows {
        span: FileSpan,
        scale: Option<f32>,
    },
    Zero(usize),
    /// A validated grid: the view holds only spans, and the remaining fields are the arithmetic
    /// needed to re-derive each tile's start in the declared order. No tile list, packet cache or
    /// projected payload is built at bind or held across buffers.
    CtTiles {
        view: PackQuantizedTensor,
        first_row: usize,
        first_col: usize,
        tile_rows: usize,
        tile_cols: usize,
        row_tiles: usize,
        col_tiles: usize,
        /// Every tile of a grid has the same extents, so one checked product covers the grid.
        packet_bytes: usize,
        order: CtTileOrder,
    },
}

#[derive(Debug)]
pub struct WeightPlan {
    buffers: BTreeMap<String, Vec<BoundOp>>,
    embedding: TensorEntry,
}

fn checked_tensor<'a>(
    tensors: &'a BTreeMap<String, TensorEntry>,
    name: &str,
    shape: &[usize],
) -> Result<&'a TensorEntry> {
    let tensor = tensors
        .get(name)
        .with_context(|| format!("missing tensor {name}"))?;
    ensure!(
        !shape.is_empty() && shape.len() <= 2 && shape.iter().all(|&n| n > 0),
        "{name}: expected positive rank-1/2 shape"
    );
    ensure!(
        tensor.dtype == ScalarType::Bf16,
        "{name}: BF16 source required, got {:?}",
        tensor.dtype
    );
    ensure!(
        tensor.shape == shape,
        "{name}: shape {:?}, expected {shape:?}",
        tensor.shape
    );
    let bytes = shape
        .iter()
        .try_fold(2usize, |p, n| p.checked_mul(*n))
        .context("tensor extent overflow")?;
    ensure!(
        u64::try_from(bytes)? == tensor.span.len(),
        "{name}: tensor byte extent mismatch"
    );
    Ok(tensor)
}

/// Route explicit companions to the checked view for the symmetric int4/g32/BF16 format declared
/// by this operation. This checks representation, not publisher provenance or source completeness.
fn ct_view(
    buffer: &str,
    tensors: &BTreeMap<String, TensorEntry>,
    weight_packed: &str,
    weight_shape: &str,
    weight_scale: &str,
) -> Result<PackQuantizedTensor> {
    let named = |label: &str, tensor: &str| -> Result<TensorEntry> {
        tensors
            .get(tensor)
            .cloned()
            .with_context(|| format!("{buffer}: missing {label} tensor {tensor}"))
    };
    PackQuantizedTensor::new(
        buffer,
        WeightSections {
            weight_packed: named("weight_packed", weight_packed)?,
            weight_shape: named("weight_shape", weight_shape)?,
            weight_scale: named("weight_scale", weight_scale)?,
            weight_zero_point: None,
            weight_g_idx: None,
        },
        PackQuantizedFormat::declared("group", 4, true, PACKET_GROUP, ScalarType::Bf16)
            .context("pack-quantized chain packet format")?,
    )
    .with_context(|| format!("{buffer}: CT companion group"))
}

impl WeightContract {
    /// Parse the embedded ABI without allowing duplicate JSON keys to overwrite requirements.
    pub fn from_metadata_json(bytes: &[u8]) -> Result<Option<Self>> {
        let mut metadata = super::safetensors::parse_unique(bytes)?;
        metadata
            .as_object_mut()
            .context("artifact metadata must be an object")?
            .remove("weight_inputs")
            .map(serde_json::from_value)
            .transpose()
            .map_err(Into::into)
    }
}

impl WeightPlan {
    pub fn bind_config_json(
        contract: &WeightContract,
        tensors: &BTreeMap<String, TensorEntry>,
        expected: &BTreeMap<String, usize>,
        config: &[u8],
    ) -> Result<Self> {
        let config = super::safetensors::parse_unique(config)?;
        Self::bind(contract, tensors, expected, &config)
    }

    /// Validate all names, shapes, scalar semantics, selections and output lengths before writing.
    /// `expected` comes from the validated program layout, not from this contract's operations.
    pub fn bind(
        contract: &WeightContract,
        tensors: &BTreeMap<String, TensorEntry>,
        expected: &BTreeMap<String, usize>,
        config: &serde_json::Value,
    ) -> Result<Self> {
        ensure!(
            contract.version == 1 || contract.version == 2,
            "unsupported weight-input contract version {}",
            contract.version
        );
        ensure!(
            !contract.config.is_empty(),
            "missing checkpoint configuration requirements"
        );
        for (key, required) in &contract.config {
            let actual = config
                .get(key)
                .with_context(|| format!("missing checkpoint config {key}"))?;
            ensure!(
                actual == required
                    || actual
                        .as_f64()
                        .zip(required.as_f64())
                        .is_some_and(|(a, b)| a == b),
                "incompatible checkpoint config {key}: got {actual}, required {required}"
            );
        }
        // CT operations require the complete declared quantization configuration to participate in
        // the equality check above. Configuration equality does not authenticate the weight payload.
        if contract
            .buffers
            .values()
            .flatten()
            .any(|op| matches!(op, WeightOp::CtTiles { .. }))
        {
            let declared = contract
                .config
                .get("quantization_config")
                .context("CT tile grids need a declared quantization_config")?;
            let declared = declared
                .as_object()
                .context("declared quantization_config must be an object")?;
            ensure!(
                !declared.is_empty(),
                "declared quantization_config must not be empty"
            );
        }
        ensure!(
            !expected.is_empty() && contract.buffers.keys().eq(expected.keys()),
            "weight-input buffer coverage mismatch"
        );
        let embedding = checked_tensor(
            tensors,
            &contract.embedding.tensor,
            &contract.embedding.shape,
        )?
        .clone();
        ensure!(embedding.shape.len() == 2, "embedding must be a matrix");
        let mut buffers = BTreeMap::new();
        for (name, ops) in &contract.buffers {
            ensure!(!ops.is_empty(), "{name}: empty weight-input operations");
            let mut bound = Vec::with_capacity(ops.len());
            let mut total = 0usize;
            for op in ops {
                let (part, len) = match op {
                    WeightOp::Zero { bytes } => {
                        ensure!(
                            *bytes > 0 && bytes % 2 == 0,
                            "{name}: zero extent must contain whole BF16 values"
                        );
                        (BoundOp::Zero(*bytes), *bytes)
                    }
                    WeightOp::Rows {
                        tensor,
                        shape,
                        first,
                        count,
                        scale,
                    } => {
                        let entry = checked_tensor(tensors, tensor, shape)?;
                        ensure!(
                            *count > 0
                                && first.checked_add(*count).context("row extent overflow")?
                                    <= shape[0],
                            "{tensor}: invalid row selection"
                        );
                        if let Some(scale) = scale {
                            ensure!(scale.is_finite(), "{tensor}: scale must be finite");
                        }
                        let width = if shape.len() == 2 {
                            shape[1].checked_mul(2).context("row width overflow")?
                        } else {
                            2
                        };
                        let start = first.checked_mul(width).context("row offset overflow")?;
                        let len = count
                            .checked_mul(width)
                            .context("selected extent overflow")?;
                        let offset = entry
                            .span
                            .offset()
                            .checked_add(u64::try_from(start)?)
                            .context("source offset overflow")?;
                        let span = entry.span.source().span(offset, u64::try_from(len)?)?;
                        (
                            BoundOp::Rows {
                                span,
                                scale: *scale,
                            },
                            len,
                        )
                    }
                    WeightOp::CtTiles {
                        weight_packed,
                        weight_shape,
                        weight_scale,
                        shape,
                        first_row,
                        first_col,
                        rows,
                        cols,
                        tile_rows,
                        tile_cols,
                        tile_order,
                    } => {
                        ensure!(
                            contract.version == 2,
                            "{name}: a CT tile grid requires contract version 2, got {}",
                            contract.version
                        );
                        let view =
                            ct_view(name, tensors, weight_packed, weight_shape, weight_scale)?;
                        let source = view.descriptor();
                        ensure!(
                            shape.len() == 2 && shape[0] > 0 && shape[1] > 0,
                            "{name}: a CT tile grid needs a positive rank-2 logical shape, got {shape:?}"
                        );
                        ensure!(
                            shape[0] == source.rows && shape[1] == source.cols,
                            "{name}: CT grid declares logical shape [{}, {}], the source declares [{}, {}]",
                            shape[0],
                            shape[1],
                            source.rows,
                            source.cols
                        );
                        ensure!(
                            *rows > 0 && *cols > 0,
                            "{name}: a CT grid region needs positive extents, got {rows}x{cols}"
                        );
                        ensure!(
                            *tile_rows > 0 && *tile_cols > 0,
                            "{name}: a CT grid needs positive tile extents, got {tile_rows}x{tile_cols}"
                        );
                        // The region is a window of the declared matrix, and it is tiled exactly:
                        // nothing is padded, clamped or transposed to make an extent fit.
                        let last_row = first_row
                            .checked_add(*rows)
                            .context("CT grid row extent overflow")?;
                        let last_col = first_col
                            .checked_add(*cols)
                            .context("CT grid column extent overflow")?;
                        ensure!(
                            last_row <= source.rows,
                            "{name}: CT grid rows {first_row}..{last_row} exceed the {} declared rows",
                            source.rows
                        );
                        ensure!(
                            last_col <= source.cols,
                            "{name}: CT grid columns {first_col}..{last_col} exceed the {} declared columns",
                            source.cols
                        );
                        ensure!(
                            rows % tile_rows == 0 && cols % tile_cols == 0,
                            "{name}: the {rows}x{cols} region is not a whole number of {tile_rows}x{tile_cols} tiles"
                        );
                        ensure!(
                            first_col % PACKET_GROUP == 0,
                            "{name}: CT grid column start {first_col} is not a {PACKET_GROUP}-column scale group start"
                        );
                        // One legal representative tile decides the whole grid: legality is about the
                        // shared extents, so an illegal extent or an uncomputable packet size rejects
                        // the operation before any output length is claimed.
                        let packet_bytes = ChainTile {
                            first_row: 0,
                            first_col: 0,
                            rows: *tile_rows,
                            cols: *tile_cols,
                        }
                        .encoded_bytes()
                        .with_context(|| format!("{name}: CT tile geometry"))?;
                        let row_tiles = rows / tile_rows;
                        let col_tiles = cols / tile_cols;
                        let len = row_tiles
                            .checked_mul(col_tiles)
                            .and_then(|tiles| tiles.checked_mul(packet_bytes))
                            .context("CT grid byte total overflow")?;
                        (
                            BoundOp::CtTiles {
                                view,
                                first_row: *first_row,
                                first_col: *first_col,
                                tile_rows: *tile_rows,
                                tile_cols: *tile_cols,
                                row_tiles,
                                col_tiles,
                                packet_bytes,
                                order: *tile_order,
                            },
                            len,
                        )
                    }
                };
                total = total.checked_add(len).context("output length overflow")?;
                bound.push(part);
            }
            ensure!(
                total == expected[name],
                "{name}: planned {total} bytes, program requires {}",
                expected[name]
            );
            buffers.insert(name.clone(), bound);
        }
        Ok(Self { buffers, embedding })
    }

    pub fn embedding(&self) -> &TensorEntry {
        &self.embedding
    }

    /// Caller scratch is the only payload storage. BF16 scaling is explicitly float32 multiply
    /// followed by round-to-nearest-even; copies preserve bits, including signed zeros and NaNs.
    /// A CT packet is never split across calls, so scratch must already hold the buffer's largest
    /// packet; that is checked against every operation before the sink is called even once.
    pub fn write_buffer(
        &self,
        name: &str,
        scratch: &mut [u8],
        mut sink: impl FnMut(usize, &[u8]) -> Result<()>,
    ) -> Result<()> {
        let capacity = scratch.len() / 2 * 2;
        let ops = self
            .buffers
            .get(name)
            .with_context(|| format!("unknown weight buffer {name}"))?;
        let mut required = 2usize;
        for op in ops {
            if let BoundOp::CtTiles { packet_bytes, .. } = op {
                required = required.max(*packet_bytes);
            }
        }
        ensure!(
            capacity >= required,
            "weight preparation needs at least {required} scratch bytes, the caller provides {capacity}"
        );
        let mut output = 0usize;
        for op in ops {
            match op {
                BoundOp::Zero(n) => {
                    let mut offset = 0usize;
                    while offset < *n {
                        let len = capacity.min(*n - offset);
                        let chunk = &mut scratch[..len];
                        chunk.fill(0);
                        sink(output, chunk)?;
                        output += len;
                        offset += len;
                    }
                }
                BoundOp::Rows { span, scale } => {
                    let len = usize::try_from(span.len())?;
                    let mut offset = 0usize;
                    while offset < len {
                        let n = capacity.min(len - offset);
                        let chunk = &mut scratch[..n];
                        span.read_exact_at(u64::try_from(offset)?, chunk)?;
                        if let Some(scale) = scale {
                            for bytes in chunk.chunks_exact_mut(2) {
                                let value =
                                    bf16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]]))
                                        .to_f32()
                                        * scale;
                                let rounded = bf16::from_f32(value);
                                ensure!(
                                    rounded.is_finite(),
                                    "{name}: non-finite scaled BF16 value"
                                );
                                bytes.copy_from_slice(&rounded.to_bits().to_le_bytes());
                            }
                        }
                        sink(output, chunk)?;
                        output += n;
                        offset += n;
                    }
                }
                BoundOp::CtTiles {
                    view,
                    first_row,
                    first_col,
                    tile_rows,
                    tile_cols,
                    row_tiles,
                    col_tiles,
                    packet_bytes,
                    order,
                } => {
                    let (first_row, first_col, tile_rows, tile_cols) =
                        (*first_row, *first_col, *tile_rows, *tile_cols);
                    let (row_tiles, col_tiles, packet_bytes) =
                        (*row_tiles, *col_tiles, *packet_bytes);
                    // Every product and sum below stays in range: bind checked the region's end
                    // against the declared matrix, the tile extents against the packet rule, and the
                    // grid's whole byte total against the program's buffer length.
                    let tiles = row_tiles
                        .checked_mul(col_tiles)
                        .context("CT tile count overflow")?;
                    for index in 0..tiles {
                        // One flat index decomposed by the declared order, so the emitted sequence is
                        // that order by construction rather than by two copies of a loop nest.
                        let (row, col) = match order {
                            CtTileOrder::RowMajor => (index / col_tiles, index % col_tiles),
                            CtTileOrder::ColumnMajor => (index % row_tiles, index / row_tiles),
                        };
                        let tile = ChainTile {
                            first_row: first_row + row * tile_rows,
                            first_col: first_col + col * tile_cols,
                            rows: tile_rows,
                            cols: tile_cols,
                        };
                        // The preflight admitted this packet into the caller's scratch, and
                        // `write_chain_tile` fills exactly `packet_bytes` of it: no second copy, no
                        // encoder of our own, and one packet per sink call.
                        let out = &mut scratch[..packet_bytes];
                        let written = view.write_chain_tile(&tile, out)?;
                        debug_assert_eq!(written, packet_bytes);
                        sink(output, out)?;
                        output += written;
                    }
                }
            }
        }
        Ok(())
    }
}
