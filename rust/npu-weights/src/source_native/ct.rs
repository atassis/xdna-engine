//! Source-preserving reader for compressed-tensors `pack-quantized` symmetric INT4 weights.
//!
//! The represented tensor is the section group compressed-tensors writes for a packed weight:
//! `weight_packed[rows, cols/8]` of little-endian I32 words holding eight unsigned 4-bit codes
//! each, consecutive along K from the lowest nibble upward, `weight_shape[2]` of I64 metadata
//! holding the logical `[rows, cols]`, and `weight_scale[rows, cols/group_size]` of scale elements.
//! The signed source code of an unsigned nibble `c` is `c - 8`, so no zero point is stored, and a
//! view reconstructs `code * scale` without clipping, rebasing or reselecting a scale.
//!
//! Two explicit boundaries. `PackQuantizedFormat` records the caller's declaration, and because its
//! fields are public, `new` revalidates the group width and scale type it can actually decode. The
//! declared column count and the scale section's width do constrain a group size - a size that does
//! not divide the columns or does not match the scale width is rejected - but the source bytes cannot
//! prove which size the publisher's `quantization_config` recorded, and safetensors containers drop
//! `__metadata__`, so nothing at this layer ties a declaration to a file's own config or revision.
//! Importers must verify those values against the selected source before constructing a view.
//! Likewise this reader owns bit-exact preservation of codes and scale bits, not value, payload,
//! model or device admission: non-finite or huge scales propagate through the exact product instead
//! of being rejected here, the same division of labour as `plan`'s scaled-BF16 admission gate.
//!
//! Anything else - an asymmetric zero point, a group index map, another strategy, bit width,
//! symmetry, scale type, packing order or extent - is rejected rather than guessed at.
use super::safetensors::TensorEntry;
use super::{
    validate_row, Encoding, FileSpan, ScalarType, Section, TensorDescriptor, WeightTensor,
};
use anyhow::{ensure, Context, Result};
use half::bf16;

/// Four-bit codes: the number of nibbles in one I32 packed word.
const PACK_FACTOR: usize = 8;
/// The signed range one unsigned nibble rebases into: the code of nibble `c` is `c - 8`.
const CODE_BIAS: i16 = 8;
/// The chain kernel's int4 scale group (`GROUP` in `aie_kernels/aie2p/chain_mm_bfp16.cc`), which
/// must equal the source group width for a repack to be exact; it also fixes the column alignment
/// and the column multiple a tile must satisfy.
const PACKET_GROUP: usize = 32;
/// A bfloat16 scale is carried as its two raw source bytes, in the source's own order.
const SCALE_BYTES: usize = 2;
/// Kernel subtile blocking (`aie_kernels/aie2p/chain_mm_bfp16.cc`,
/// `designs/resident_forward/chain_ref.py`): 8x8 subtiles of 32 bytes, and the
/// static asserts require `NB % 16 == 0` and `SUB_K % 32 == 0`.
const SUBTILE_ROWS: usize = 8;
const SUBTILE_COLS: usize = 8;
const SUBTILE_BYTES: usize = 32;
const SUBTILE_PAIR_ROWS: usize = 16;
/// One output row inside a subtile: eight adjacent-K nibbles, i.e. four packet bytes.
const SUBTILE_ROW_BYTES: usize = SUBTILE_BYTES / SUBTILE_ROWS;
/// A packet byte carries two adjacent-K nibbles, so rebasing both nibbles at once is `^ 0x88`.
const NIBBLE_BIAS_PAIR: u8 = 0x88;

/// The declared compression parameters of one `pack-quantized` tensor.
///
/// A caller obtains these from the selected source's own `quantization_config`, verified at import
/// time. The fields are public so a caller can name them directly, which is why `new` revalidates
/// them: only what this codec can actually read is accepted and reported.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PackQuantizedFormat {
    pub group_size: usize,
    pub scale_type: ScalarType,
}

impl PackQuantizedFormat {
    /// Represent a declared group strategy, bit width, symmetry, group width and scale type.
    pub fn declared(
        strategy: &str,
        weight_bits: u32,
        symmetric: bool,
        group_size: usize,
        scale_type: ScalarType,
    ) -> Result<Self> {
        ensure!(
            strategy == "group",
            "compressed-tensors strategy {strategy:?}: only \"group\" is represented"
        );
        ensure!(
            weight_bits == 4,
            "compressed-tensors weight_bits {weight_bits}: only 4-bit codes are represented"
        );
        ensure!(
            symmetric,
            "asymmetric compressed-tensors codes need a zero point, which is not represented"
        );
        ensure!(
            group_size > 0,
            "compressed-tensors group size must be positive"
        );
        ensure!(
            scale_type == ScalarType::Bf16,
            "compressed-tensors scales must be BF16, got {scale_type:?}"
        );
        Ok(Self {
            group_size,
            scale_type,
        })
    }
}

/// The companion sections of one packed tensor, as routed by the caller.
///
/// Name selection is the importer's routing decision; this codec performs none, and it passes a
/// companion tensor that exists through as a declaration it must either represent or reject.
#[derive(Clone, Debug)]
pub struct WeightSections {
    pub weight_packed: TensorEntry,
    pub weight_shape: TensorEntry,
    pub weight_scale: TensorEntry,
    pub weight_zero_point: Option<TensorEntry>,
    pub weight_g_idx: Option<TensorEntry>,
}

/// One rectangular window of a `pack-quantized` matrix, shaped as one chain-kernel int4 sub-block:
/// `rows` is the kernel's `NB`, `cols` its `SUB_K`, and the start is in logical source coordinates.
///
/// The extents are hardware packet constraints on the output geometry (`NB % 16 == 0`,
/// `SUB_K % 32 == 0`), not new model policy. `first_col` must land on a source scale group because a
/// packet's K groups have to coincide with source groups; `first_row` needs no alignment, since a
/// group depends only on K and a packet carries its own subtile blocking relative to `first_row`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChainTile {
    pub first_row: usize,
    pub first_col: usize,
    pub rows: usize,
    pub cols: usize,
}

impl ChainTile {
    /// The exact byte count this tile's layout occupies: `cols / 32` bfloat16 scale entries (two raw
    /// bytes each) per output row, then one 32-byte subtile per (8 output rows, 8 K columns). Only
    /// kernel-legal, positive extents have a size, and every count is checked, so an absurd aligned
    /// extent is a rejection rather than an overflow or an allocation request.
    pub fn encoded_bytes(&self) -> Result<usize> {
        ensure!(
            self.rows > 0 && self.cols > 0,
            "a chain tile needs positive extents, got {}x{}",
            self.rows,
            self.cols
        );
        ensure!(
            self.rows % SUBTILE_PAIR_ROWS == 0,
            "a chain tile's {} rows are not a whole number of 16-row subtile pairs",
            self.rows
        );
        ensure!(
            self.cols % PACKET_GROUP == 0,
            "a chain tile's {} columns are not a whole number of 32-column scale groups",
            self.cols
        );
        let header = (self.cols / PACKET_GROUP)
            .checked_mul(self.rows)
            .and_then(|slots| slots.checked_mul(SCALE_BYTES))
            .context("chain tile scale header overflow")?;
        let payload = self
            .rows
            .checked_mul(self.cols / 2)
            .context("chain tile subtile count overflow")?;
        header
            .checked_add(payload)
            .context("chain tile packet size overflow")
    }

    /// Byte offset of the subtile holding output row `row` and K column `column`, relative to the
    /// start of the payload: subtiles run n-block major, `[rows / 8][cols / 8]`.
    fn subtile_offset(&self, row: usize, column: usize) -> usize {
        debug_assert!(row < self.rows && column < self.cols);
        debug_assert!(self.cols % SUBTILE_COLS == 0);
        ((row / SUBTILE_ROWS) * (self.cols / SUBTILE_COLS) + column / SUBTILE_COLS) * SUBTILE_BYTES
    }
}

/// A view over one explicitly selected `pack-quantized` section group.
#[derive(Clone, Debug)]
pub struct PackQuantizedTensor {
    descriptor: TensorDescriptor,
    sections: Vec<Section>,
    packed: FileSpan,
    scales: FileSpan,
    packed_cols: usize,
    cols: usize,
    groups: usize,
    group_size: usize,
}

impl PackQuantizedTensor {
    /// Build a view from already-routed entries plus a caller-verified format declaration.
    pub fn new(
        name: impl Into<String>,
        sections: WeightSections,
        format: PackQuantizedFormat,
    ) -> Result<Self> {
        let name = name.into();
        let WeightSections {
            weight_packed,
            weight_shape,
            weight_scale,
            weight_zero_point,
            weight_g_idx,
        } = sections;
        // The fields are public, so a struct literal can bypass `declared`; revalidate the two
        // parameters this codec depends on before anything is reported through the descriptor.
        ensure!(
            format.group_size > 0,
            "{name}: compressed-tensors group size must be positive"
        );
        ensure!(
            format.scale_type == ScalarType::Bf16,
            "{name}: format declares {:?} scales, but weight_scale is BF16 and only BF16 scales are decoded",
            format.scale_type
        );
        ensure!(
            weight_zero_point.is_none(),
            "{name}: a declared weight_zero_point means asymmetric codes, which are not represented"
        );
        ensure!(
            weight_g_idx.is_none(),
            "{name}: a declared weight_g_idx means a permuted K axis, which is not represented"
        );
        ensure!(
            weight_packed.dtype == ScalarType::I32 && weight_packed.shape.len() == 2,
            "{name}: weight_packed must be rank-2 I32 words, got {:?} {:?}",
            weight_packed.dtype,
            weight_packed.shape
        );
        ensure!(
            weight_scale.dtype == ScalarType::Bf16 && weight_scale.shape.len() == 2,
            "{name}: weight_scale must be rank-2 BF16, got {:?} {:?}",
            weight_scale.dtype,
            weight_scale.shape
        );
        validate_entry(&name, "weight_packed", &weight_packed)?;
        validate_entry(&name, "weight_shape", &weight_shape)?;
        validate_entry(&name, "weight_scale", &weight_scale)?;
        let (packed_rows, packed_cols) = (weight_packed.shape[0], weight_packed.shape[1]);
        let (scale_rows, scale_groups) = (weight_scale.shape[0], weight_scale.shape[1]);
        ensure!(
            packed_rows > 0 && packed_cols > 0 && scale_rows > 0 && scale_groups > 0,
            "{name}: packed words and scale rows/columns must be positive"
        );
        // The shape metadata is the only payload a constructor reads, and it is rank-1 [2] I64.
        ensure!(
            weight_shape.dtype == ScalarType::I64 && weight_shape.shape == [2],
            "{name}: weight_shape must be rank-1 I64 [2] metadata, got {:?} {:?}",
            weight_shape.dtype,
            weight_shape.shape
        );
        let [rows, cols] = read_shape(&name, &weight_shape.span)?;
        let groups = groups_per_row(&name, cols, format.group_size)?;
        // Geometry is validated row by row: agreement of the total element count would also accept a
        // packed tensor reshaped into other extents, whose row strides then disagree with the
        // declared logical matrix and with the scales.
        ensure!(
            packed_rows == rows,
            "{name}: {packed_rows} packed rows disagree with the {rows} declared rows"
        );
        let codes_per_packed_row = packed_cols
            .checked_mul(PACK_FACTOR)
            .context("packed column count overflow")?;
        ensure!(
            codes_per_packed_row == cols,
            "{name}: {packed_cols} packed words per row hold {codes_per_packed_row} codes, not the declared {cols} columns; padded words are not accepted"
        );
        // Scale extents are cross-checked against both the declared shape and the group width.
        ensure!(
            scale_rows == rows && scale_groups == groups,
            "{name}: weight_scale shape [{scale_rows}, {scale_groups}] disagrees with the {rows} declared rows and {cols} columns at group size {}",
            format.group_size
        );

        let descriptor = TensorDescriptor {
            name,
            rows,
            cols,
            encoding: Encoding::CtPackQuantized {
                group_size: format.group_size,
                scale_type: format.scale_type,
            },
        };
        Ok(Self {
            sections: vec![
                Section {
                    name: "weight_packed".into(),
                    span: weight_packed.span.clone(),
                },
                Section {
                    name: "weight_shape".into(),
                    span: weight_shape.span.clone(),
                },
                Section {
                    name: "weight_scale".into(),
                    span: weight_scale.span.clone(),
                },
            ],
            descriptor,
            packed: weight_packed.span,
            scales: weight_scale.span,
            packed_cols,
            cols,
            groups,
            group_size: format.group_size,
        })
    }

    /// The source code at one logical position: the unsigned nibble rebased by `c - 8`.
    pub fn code(&self, row: usize, column: usize) -> Result<i8> {
        let element = self.locate("code", row, column)?;
        Ok(signed_code(nibble_of(
            self.word(element / PACK_FACTOR)?,
            element % PACK_FACTOR,
        )))
    }

    /// The exact source scale bits of the group holding one logical column.
    pub fn scale_bits(&self, row: usize, column: usize) -> Result<u16> {
        self.validate("scale", row, column)?;
        let group = column / self.group_size;
        debug_assert!(group < self.groups);
        let element = row
            .checked_mul(self.groups)
            .and_then(|base| base.checked_add(group))
            .context("scale index overflow")?;
        let mut bytes = [0u8; 2];
        self.scales
            .read_exact_at(byte_offset(element, 2, "scale offset")?, &mut bytes)?;
        Ok(u16::from_le_bytes(bytes))
    }

    /// Row/column validation plus the logical code index of a position.
    fn locate(&self, what: &str, row: usize, column: usize) -> Result<usize> {
        self.validate(what, row, column)?;
        column
            .checked_add(row.checked_mul(self.cols).context("code index overflow")?)
            .context("code index overflow")
    }

    fn validate(&self, what: &str, row: usize, column: usize) -> Result<()> {
        // `validate_row` with a one-element window is exactly `row < rows && column < cols`, so a
        // CT position is bounded by the same shared rule every other codec here uses.
        validate_row(&self.descriptor, row, column, 1)
            .with_context(|| format!("{}: {what} position", self.descriptor.name))
    }

    /// One whole little-endian I32 packed word. A code only exists relative to its word, so callers
    /// read the full 32 bits and shift before masking anything.
    fn word(&self, element: usize) -> Result<u32> {
        let mut bytes = [0u8; 4];
        self.packed
            .read_exact_at(byte_offset(element, 4, "word offset")?, &mut bytes)?;
        Ok(u32::from_le_bytes(bytes))
    }
}

impl PackQuantizedTensor {
    /// Repack one tile into the chain kernel's int4/BF16-scale sub-block layout, byte for byte:
    /// `[cols / 32][rows]` little-endian bfloat16 scale bits, then `[rows / 8][cols / 8]` 32-byte
    /// subtiles with nibble lane `i = (n = i / 8, k = i % 8)`, low nibble first, holding a signed
    /// two's-complement s4.
    ///
    /// Only the data bytes are adapted. A source nibble `c` represents the signed code `c - 8`, so
    /// the packet's s4 nibble is `c ^ 8` - byte-wise `^ 0x88`, always legal here because every packet
    /// byte carries two adjacent-K nibbles of one whole source word - and a scale is copied as its
    /// raw 16 bits, so signed or unsigned zero, subnormals, infinities and quiet or signaling NaN
    /// payloads survive unchanged. A scale is read from the group of its own source column, never
    /// from a floored window start. Nothing is clipped, requantized, dequantized or widened, and the
    /// kernel converts weights to BFP16 internally, so producing this layout is neither numerical,
    /// program nor device acceptance and authorizes nothing about that compute path.
    ///
    /// Every validation runs before the first write, so a rejected request leaves `out` untouched;
    /// only a source read error can leave a partial packet, and publication or rollback of that
    /// caller-owned storage stays the caller's decision. `out` may be longer than the packet: bytes
    /// past the returned count are not read or written. The return value is exactly
    /// `tile.encoded_bytes()?`.
    pub fn write_chain_tile(&self, tile: &ChainTile, out: &mut [u8]) -> Result<usize> {
        let name = self.descriptor.name.as_str();
        let required = tile.encoded_bytes().with_context(|| {
            format!(
                "{name}: chain tile at ({}, {})",
                tile.first_row, tile.first_col
            )
        })?;
        ensure!(
            self.group_size == PACKET_GROUP,
            "{name}: a chain packet groups K by {PACKET_GROUP}, but the source declares group size {}; repacking would share or splice source scales",
            self.group_size
        );
        ensure!(
            tile.first_col % PACKET_GROUP == 0,
            "{name}: chain tile column start {} is not a source group start; the packet would need a scale for K columns whose group begins before the window",
            tile.first_col
        );
        // The tile is a window of the declared logical matrix. Extents are checked additions against
        // the same bounds every other codec here uses, so an overflowing, off-by-one or otherwise
        // out-of-range start is rejected before any destination byte is touched.
        let last_row = tile
            .first_row
            .checked_add(tile.rows)
            .with_context(|| format!("{name}: chain tile row range overflow"))?;
        let last_col = tile
            .first_col
            .checked_add(tile.cols)
            .with_context(|| format!("{name}: chain tile column range overflow"))?;
        ensure!(
            last_row <= self.descriptor.rows,
            "{name}: chain tile rows {}..{} exceed the {} declared rows",
            tile.first_row,
            last_row,
            self.descriptor.rows
        );
        ensure!(
            last_col <= self.descriptor.cols,
            "{name}: chain tile columns {}..{} exceed the {} declared columns",
            tile.first_col,
            last_col,
            self.descriptor.cols
        );
        ensure!(
            out.len() >= required,
            "{name}: chain tile {}x{} needs {required} bytes, the destination has {}",
            tile.rows,
            tile.cols,
            out.len()
        );

        // Both parts below are exactly the counts `encoded_bytes` checked, so neither product can
        // overflow: the header is smaller than the packet and the payload is the rest of it.
        let header = (tile.cols / PACKET_GROUP) * tile.rows * SCALE_BYTES;
        debug_assert_eq!(header + tile.rows * (tile.cols / 2), required);
        for group in 0..tile.cols / PACKET_GROUP {
            for row in 0..tile.rows {
                // `scale_bits` bounds the position and indexes the group of this exact column, which
                // is `first_col / 32 + group` because the start is group-aligned.
                let bits =
                    self.scale_bits(tile.first_row + row, tile.first_col + group * PACKET_GROUP)?;
                let at = (group * tile.rows + row) * SCALE_BYTES;
                out[at..at + SCALE_BYTES].copy_from_slice(&bits.to_le_bytes());
            }
        }
        // A tile's K start is 32-aligned, so every 8-column K block is a whole set of source words,
        // and one word is exactly one subtile row's four packet bytes. The whole 32 bits are read and
        // then byte-rebased; no nibble is cached or shifted a second time.
        for row_block in 0..tile.rows / SUBTILE_ROWS {
            for col_block in 0..tile.cols / SUBTILE_COLS {
                let column = tile.first_col + col_block * SUBTILE_COLS;
                for row in 0..SUBTILE_ROWS {
                    let source_row = tile.first_row + row_block * SUBTILE_ROWS + row;
                    // `rows * packed_cols` is the constructor-validated word count, so a window row's
                    // index is in range and cannot overflow.
                    let word = source_row * self.packed_cols + column / PACK_FACTOR;
                    let bytes = self.word(word)?.to_le_bytes();
                    let at = header
                        + tile.subtile_offset(
                            row_block * SUBTILE_ROWS + row,
                            col_block * SUBTILE_COLS,
                        )
                        + row * SUBTILE_ROW_BYTES;
                    for (offset, byte) in bytes.iter().enumerate() {
                        out[at + offset] = byte ^ NIBBLE_BIAS_PAIR;
                    }
                }
            }
        }
        Ok(required)
    }
}

impl WeightTensor for PackQuantizedTensor {
    fn descriptor(&self) -> &TensorDescriptor {
        &self.descriptor
    }

    fn sections(&self) -> &[Section] {
        &self.sections
    }

    fn decode_row(&self, row: usize, column: usize, out: &mut [f32]) -> Result<()> {
        validate_row(&self.descriptor, row, column, out.len())?;
        if out.is_empty() {
            return Ok(());
        }
        let mut element = row
            .checked_mul(self.packed_cols)
            .and_then(|base| base.checked_add(column / PACK_FACTOR))
            .context("word index overflow")?;
        let mut word = self.word(element)?;
        // The word currently held, counted from the start of this row's packed words.
        let mut held_word = column / PACK_FACTOR;
        for (offset, value) in out.iter_mut().enumerate() {
            let column = column.checked_add(offset).context("column overflow")?;
            debug_assert!(column < self.cols);
            if column / PACK_FACTOR != held_word {
                // Consecutive K columns occupy consecutive words, so a window never skips one.
                debug_assert_eq!(column / PACK_FACTOR, held_word + 1);
                held_word += 1;
                element += 1;
                word = self.word(element)?;
            }
            let code = signed_code(nibble_of(word, column % PACK_FACTOR));
            // The scale is widened exactly, so the f32 product is the exact product whenever it is
            // representable; non-finite or overflowing scales propagate instead of being clipped.
            *value = f32::from(code) * bf16::from_bits(self.scale_bits(row, column)?).to_f32();
        }
        Ok(())
    }
}

/// The nibble `slot` of a full packed word, counting from the lowest four bits. The shift happens on
/// the whole word and the mask is applied only afterwards, so no other slot of the word is lost.
fn nibble_of(word: u32, slot: usize) -> u8 {
    debug_assert!(slot < PACK_FACTOR);
    ((word >> (slot * 4)) & 0x0f) as u8
}

/// The signed source code of a masked nibble: `c - 8`, never a two's-complement reinterpretation
/// of the low nibble, which would map 8..15 to negative values again.
fn signed_code(nibble: u8) -> i8 {
    debug_assert!(nibble < 16);
    // A masked nibble is 0..15, so `c - 8` always lands in `-8..=7`.
    (i16::from(nibble) - CODE_BIAS) as i8
}

/// Logical groups per row, rejecting a declared group width that cannot tile the declared columns.
fn groups_per_row(name: &str, cols: usize, group_size: usize) -> Result<usize> {
    let groups = cols
        .checked_div(group_size)
        .context("group size overflow")?;
    ensure!(
        groups * group_size == cols,
        "{name}: group size {group_size} does not divide {cols} declared columns"
    );
    Ok(groups)
}

/// Byte offset of one element, with a static label because anyhow context must be `'static`.
fn byte_offset(element: usize, width: usize, label: &'static str) -> Result<u64> {
    let offset = element.checked_mul(width).context(label)?;
    Ok(u64::try_from(offset)?)
}

/// Read the two positive logical dimensions out of `[2]` I64 metadata. Representability is the
/// host's business, not a policy of this reader: dimensions must be positive and must fit in `usize`,
/// and every product derived from them uses checked arithmetic, so an absurd value either fails
/// conversion or fails to agree with the real section extents.
fn read_shape(name: &str, span: &FileSpan) -> Result<[usize; 2]> {
    let mut shape = [0i64; 2];
    for (index, value) in shape.iter_mut().enumerate() {
        let mut bytes = [0; 8];
        span.read_exact_at((index * 8) as u64, &mut bytes)
            .with_context(|| format!("{name}: weight_shape element {index}"))?;
        *value = i64::from_le_bytes(bytes);
    }
    let [rows, cols] = shape;
    ensure!(
        rows > 0 && cols > 0,
        "{name}: weight_shape dimensions must be positive, got [{rows}, {cols}]"
    );
    Ok([
        addressable(name, "rows", rows)?,
        addressable(name, "columns", cols)?,
    ])
}

/// One declared dimension as a `usize`: the host's addressable range is the only bound, and every
/// product derived from it is checked arithmetic.
fn addressable(name: &str, label: &str, value: i64) -> Result<usize> {
    usize::try_from(value).with_context(|| {
        format!("{name}: declared {label} {value} is not addressable on this host")
    })
}

fn validate_entry(name: &str, label: &str, entry: &TensorEntry) -> Result<()> {
    let elements = entry
        .shape
        .iter()
        .try_fold(1usize, |count, dim| count.checked_mul(*dim))
        .with_context(|| format!("{name}: {label} element count overflow"))?;
    let bytes = elements
        .checked_mul(entry.dtype.byte_width())
        .with_context(|| format!("{name}: {label} byte length overflow"))?;
    ensure!(
        !entry.span.is_empty(),
        "{name}: {label} span must not be empty"
    );
    ensure!(
        entry.span.len() == u64::try_from(bytes)?,
        "{name}: {label} byte length does not match dtype and shape"
    );
    Ok(())
}
