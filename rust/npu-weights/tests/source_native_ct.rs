//! Source-preserving reader contract for compressed-tensors `pack-quantized` INT4 weights.
//!
//! Every expectation here is computed independently of the reader under test: codes come from
//! synthetic nibble tables and from two synthetic I32 constants chosen as an ascending nibble run,
//! scale expectations are literal f32 bit patterns widened by shifting a BF16 pattern into an f32,
//! and every file is a tiny synthetic safetensors image read through the production header parser.
//! No constant here is an in-repo vendor fixture: nothing in this checkout carries the upstream
//! packing implementation, and no publisher payload is read. Nothing here quantizes, requantizes or
//! clips a value: there is no `quantize_weight` dependency and no source grid change.
//!
//! The one representation exercised here is the pair recorded as reviewed header evidence for the
//! selected source (notes outside this checkout, not in-repo data): symmetric 4-bit group codes,
//! unsigned nibbles packed low-to-high along K into I32 words, an I64 `[2]` `weight_shape` metadata
//! tensor holding the logical `[rows, columns]`, and BF16 `weight_scale` per row and group. Anything
//! else - an asymmetric zero point, a group index map, another strategy, bit width, symmetry, scale
//! type or packing order - is rejected rather than guessed at.
//!
//! Two boundaries this file deliberately does not move. `PackQuantizedFormat::declared` is a
//! *caller claim*: the safetensors container discards `__metadata__`, so nothing at this layer can
//! tie a declaration to the file's own `quantization_config`. Those values must be importer-verified
//! against the selected source config at import time. The source does constrain a declared group
//! width - it must divide the declared column count and match the scale section's width - but those
//! dimensions cannot prove which width the publisher's config recorded for this revision, so any
//! divisor with a matching scale section is representable here; see the group-size cases below.
//! Likewise this reader's contract is bit-exact scale preservation and an exact product; admission
//! of finite-valued weights is a higher-boundary concern (as `plan`'s scaled-BF16 gate already is),
//! not a per-tensor rescan here, and certainly not a whole-model scan.
//!
//! Accepting `"I64"` as a safetensors dtype is a container decision, not a CT one, and it is
//! pinned where that decision lives: `tests/source_native_safetensors.rs`. This file only
//! consumes such a metadata entry through the reader, and keeps refusing I64 as numeric weight
//! storage.

use anyhow::Result;
use npu_weights::source_native::ct::{PackQuantizedFormat, PackQuantizedTensor, WeightSections};
use npu_weights::source_native::safetensors::SafetensorsFile;
use npu_weights::source_native::{
    write_encoded, DenseTensor, Encoding, Limits, ScalarType, WeightTensor,
};
use serde_json::json;
use std::io::{Seek, SeekFrom, Write};
use tempfile::NamedTempFile;

/// Nibbles stored in one I32 word at four bits per code.
const PACK_FACTOR: usize = 8;
/// Group width used by the small fixtures.
const GROUP: usize = 4;
/// Fixture geometry: three rows, three I32 words per row, six groups per row.
const ROWS: usize = 3;
const COLS: usize = 24;

/// One tensor as it appears in a synthetic image.
#[derive(Clone)]
struct Raw {
    name: String,
    dtype: String,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

fn raw(name: String, dtype: &str, shape: &[usize], bytes: Vec<u8>) -> Raw {
    Raw {
        name,
        dtype: dtype.to_string(),
        shape: shape.to_vec(),
        bytes,
    }
}

fn i32_bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

fn i64_bytes(values: &[i64]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn bf16_bytes(bits: &[u16]) -> Vec<u8> {
    bits.iter().flat_map(|bits| bits.to_le_bytes()).collect()
}

/// Writes a real safetensors image - header length, header JSON, contiguous payload - so the
/// reader exercises the production parser, including its duplicate-key and extent enforcement.
fn image(entries: &[Raw]) -> Result<NamedTempFile> {
    let mut header = serde_json::Map::new();
    let mut offset = 0u64;
    for entry in entries {
        let len = u64::try_from(entry.bytes.len())?;
        header.insert(
            entry.name.clone(),
            json!({
                "dtype": entry.dtype,
                "shape": entry.shape,
                "data_offsets": [offset, offset + len],
            }),
        );
        offset += len;
    }
    let header = serde_json::Value::Object(header).to_string();
    let mut file = NamedTempFile::new()?;
    file.write_all(&(header.len() as u64).to_le_bytes())?;
    file.write_all(header.as_bytes())?;
    for entry in entries {
        file.write_all(&entry.bytes)?;
    }
    file.flush()?;
    Ok(file)
}

/// The three required compression parameters at the geometry a fixture declares.
fn ct_raws(
    prefix: &str,
    rows: usize,
    cols: usize,
    words: &[u32],
    shape_meta: [i64; 2],
    scales: &[u16],
) -> Vec<Raw> {
    vec![
        raw(
            format!("{prefix}.weight_packed"),
            "I32",
            &[rows, cols / PACK_FACTOR],
            i32_bytes(words),
        ),
        raw(
            format!("{prefix}.weight_shape"),
            "I64",
            &[2],
            i64_bytes(&shape_meta),
        ),
        raw(
            format!("{prefix}.weight_scale"),
            "BF16",
            &[rows, scales.len() / rows.max(1)],
            bf16_bytes(scales),
        ),
    ]
}

/// The format declaration the tests hand to the reader. In production these values are the
/// importer's, verified against the selected source's own config before any tensor is read;
/// `declared` only refuses what cannot be represented, it does not authenticate the claim.
fn group_format(group_size: usize) -> Result<PackQuantizedFormat> {
    PackQuantizedFormat::declared("group", 4, true, group_size, ScalarType::Bf16)
}

/// Collects each declared parameter by its source name. A companion tensor that exists is passed
/// through as a declaration the reader must either represent or reject, never silently drop. Name
/// selection is the importer's routing decision; this codec performs none.
fn sections(source: &SafetensorsFile, prefix: &str) -> Result<WeightSections> {
    Ok(WeightSections {
        weight_packed: source.get(&format!("{prefix}.weight_packed"))?.clone(),
        weight_shape: source.get(&format!("{prefix}.weight_shape"))?.clone(),
        weight_scale: source.get(&format!("{prefix}.weight_scale"))?.clone(),
        weight_zero_point: source
            .get(&format!("{prefix}.weight_zero_point"))
            .ok()
            .cloned(),
        weight_g_idx: source.get(&format!("{prefix}.weight_g_idx")).ok().cloned(),
    })
}

/// Builds one view from already-routed entries plus an importer-supplied format declaration.
fn view(source: &SafetensorsFile, prefix: &str, group_size: usize) -> Result<PackQuantizedTensor> {
    PackQuantizedTensor::new(prefix, sections(source, prefix)?, group_format(group_size)?)
}

/// Builds a view from a synthetic image. A rejection at any layer - header, extent or codec - is an
/// error, so callers can assert rejection without a panic from a missing entry.
fn build(entries: &[Raw], prefix: &str, group_size: usize) -> Result<PackQuantizedTensor> {
    let file = image(entries)?;
    let source = SafetensorsFile::open(file.path(), Limits::default())?;
    view(&source, prefix, group_size)
}

/// The CT nibble for a signed four-bit code: the two's-complement low nibble rebased by XOR 8,
/// which holds the same value as `code + 8`.
fn nibble_of(code: i8) -> u32 {
    u32::from(((code as u8) & 0x0f) ^ 8)
}

/// Independent signed reconstruction of an unsigned nibble `c`: `c - 8`, never `c ^ 8` and never a
/// two's-complement reinterpretation of the low nibble.
fn signed_of(nibble: u8) -> i8 {
    nibble as i8 - 8
}

/// Deterministic code pattern: every row contains all sixteen codes, and every row and every word
/// inside a row gets a different nibble sequence, so a row, word or slot confusion cannot pass.
fn source_code(row: usize, column: usize) -> i8 {
    let word = column / PACK_FACTOR;
    let slot = column % PACK_FACTOR;
    -8 + ((row + word * 4 + slot * 7) % 16) as i8
}

/// Packs one row/word with consecutive columns in consecutive nibbles, lowest nibble first.
fn source_word(row: usize, word: usize) -> u32 {
    let mut packed = 0u32;
    for slot in 0..PACK_FACTOR {
        packed |= nibble_of(source_code(row, word * PACK_FACTOR + slot)) << (slot * 4);
    }
    packed
}

/// Packs explicit codes into one word, lowest nibble first.
fn word_of(codes: [i8; PACK_FACTOR]) -> u32 {
    let mut packed = 0u32;
    for (slot, code) in codes.into_iter().enumerate() {
        packed |= nibble_of(code) << (slot * 4);
    }
    packed
}

/// Every BF16 pattern used by a fixture is nonzero and distinct, so a row or group addressing
/// error changes the observed bits instead of coincidentally matching.
fn scale_bits(row: usize, group: usize) -> u16 {
    (row * 16 + group + 1) as u16
}

/// The f32 product of a code and a BF16 scale, widened by moving the BF16 pattern into the high
/// half of an f32. The product is exact for every four-bit code while the scale magnitude is at
/// most 2^124; beyond that the IEEE-754 f32 product overflows to an infinity.
fn exact(code: i8, bits: u16) -> f32 {
    f32::from(code) * f32::from_bits(u32::from(bits) << 16)
}

fn exact_bits(code: i8, bits: u16) -> u32 {
    exact(code, bits).to_bits()
}

fn bits_of(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn codes_at(tensor: &PackQuantizedTensor, row: usize, columns: std::ops::Range<usize>) -> Vec<i8> {
    columns
        .map(|column| tensor.code(row, column).expect("in-range code"))
        .collect()
}

/// The AutoAWQ GEMM lane-to-slot permutation, restated only to prove it is not a CT convention:
/// applying it to a CT word yields a different column order.
fn autoawq_lane_order(word: u32) -> Vec<i8> {
    const LANE_TO_SLOT: [usize; 8] = [0, 4, 1, 5, 2, 6, 3, 7];
    LANE_TO_SLOT
        .iter()
        .map(|slot| signed_of(((word >> (slot * 4)) & 0x0f) as u8))
        .collect()
}

/// Records only the rejection, so a call that unexpectedly succeeds still fails the test.
fn assert_rejected<T>(result: Result<T>, label: &str) {
    assert!(result.is_err(), "{label}: expected rejection");
}

fn assert_rejects_view(entries: &[Raw], label: &str, group_size: usize) {
    assert_rejected(build(entries, "proj", group_size), label);
}

fn assert_rejects_slice(tensor: &PackQuantizedTensor, row: usize, column: usize, len: usize) {
    // The caller's buffer length also carries the requested width, so the reader cannot widen a
    // selection past the declared shape in order to fill a buffer.
    assert!(
        tensor
            .decode_row(row, column, &mut vec![0.0f32; len])
            .is_err(),
        "row {row}, column {column}, {len} values: expected rejection"
    );
}

/// The fixture shared by the geometric tests: three rows, three words, six groups.
fn grid() -> (Vec<u32>, Vec<u16>) {
    let words: Vec<u32> = (0..ROWS)
        .flat_map(|row| (0..COLS / PACK_FACTOR).map(move |word| source_word(row, word)))
        .collect();
    let scales: Vec<u16> = (0..ROWS)
        .flat_map(|row| (0..COLS / GROUP).map(move |group| scale_bits(row, group)))
        .collect();
    (words, scales)
}

#[test]
fn all_sixteen_source_codes_reconstruct_the_full_signed_grid() -> Result<()> {
    // Two synthetic words whose nibbles run ascending, written as source bytes. Nibble slot 0 is
    // the lowest four bits, so the slots read low to high as 0..15 and reconstruct as -8..7.
    let file = image(&ct_raws(
        "proj",
        1,
        16,
        &[0x7654_3210, 0xFEDC_BA98],
        [1, 16],
        &[0x3f80],
    ))?;
    let source = SafetensorsFile::open(file.path(), Limits::default())?;
    let tensor = view(&source, "proj", 16)?;

    assert_eq!(tensor.descriptor().rows, 1);
    assert_eq!(tensor.descriptor().cols, 16);
    assert_eq!(
        tensor.descriptor().encoding,
        Encoding::CtPackQuantized {
            group_size: 16,
            scale_type: ScalarType::Bf16,
        }
    );
    assert_eq!(codes_at(&tensor, 0, 0..16), (-8..8).collect::<Vec<i8>>());

    // With a scale of exactly 1.0 the reconstruction is the source grid itself: no midpoint shift,
    // no clip and no requantization.
    let mut decoded = [0.0f32; 16];
    tensor.decode_row(0, 0, &mut decoded)?;
    let expected = [
        -8.0f32, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
    ];
    assert_eq!(decoded, expected);
    assert_eq!(tensor.scale_bits(0, 0)?, 0x3f80);
    assert_eq!(tensor.scale_bits(0, 15)?, 0x3f80);
    Ok(())
}

#[test]
fn ct_nibble_order_is_not_the_autoawq_gemm_permutation() -> Result<()> {
    let word = 0x7654_3210u32;
    let file = image(&ct_raws("proj", 1, 8, &[word], [1, 8], &[0x3f80]))?;
    let source = SafetensorsFile::open(file.path(), Limits::default())?;
    let tensor = view(&source, "proj", 8)?;

    // CT stores consecutive K columns in consecutive low-to-high nibbles.
    assert_eq!(codes_at(&tensor, 0, 0..8), (-8..0).collect::<Vec<i8>>());
    // AutoAWQ GEMM puts lane 1 in slot 4, so the same word reads a different column order.
    assert_ne!(autoawq_lane_order(word), (-8..0).collect::<Vec<i8>>());
    Ok(())
}

#[test]
fn nibble_slots_word_rows_and_group_boundaries_match_source_tables() -> Result<()> {
    let (words, scales) = grid();
    let file = image(&ct_raws(
        "proj",
        ROWS,
        COLS,
        &words,
        [ROWS as i64, COLS as i64],
        &scales,
    ))?;
    let source = SafetensorsFile::open(file.path(), Limits::default())?;
    let tensor = view(&source, "proj", GROUP)?;

    assert_eq!(tensor.descriptor().rows, ROWS);
    assert_eq!(tensor.descriptor().cols, COLS);
    assert_eq!(
        tensor
            .sections()
            .iter()
            .map(|section| section.name.as_str())
            .collect::<Vec<_>>(),
        ["weight_packed", "weight_shape", "weight_scale"]
    );

    for row in 0..ROWS {
        // Every source code appears in every row of this fixture.
        let mut seen: Vec<i8> = codes_at(&tensor, row, 0..COLS);
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(
            seen,
            (-8..8).collect::<Vec<i8>>(),
            "row {row} code coverage"
        );
        for column in 0..COLS {
            assert_eq!(
                tensor.code(row, column)?,
                source_code(row, column),
                "code({row}, {column})"
            );
            assert_eq!(
                tensor.scale_bits(row, column)?,
                scale_bits(row, column / GROUP),
                "scale selection at ({row}, {column})"
            );
        }
        let mut decoded = vec![0.0f32; COLS];
        tensor.decode_row(row, 0, &mut decoded)?;
        for (column, value) in decoded.iter().enumerate() {
            assert_eq!(
                value.to_bits(),
                exact_bits(source_code(row, column), scale_bits(row, column / GROUP)),
                "row {row} column {column}"
            );
        }
    }

    // The highest slot of one word and the lowest slot of the next are consecutive columns, so a
    // word-length error would shift a whole row.
    assert_eq!(
        tensor.code(2, 7)?,
        signed_of(((source_word(2, 0) >> 28) & 0x0f) as u8)
    );
    assert_eq!(
        tensor.code(2, 8)?,
        signed_of((source_word(2, 1) & 0x0f) as u8)
    );

    // A window that straddles a group boundary switches scales exactly at that boundary.
    let mut across_group = [0.0f32; 6];
    tensor.decode_row(1, 2, &mut across_group)?;
    assert_eq!(
        bits_of(&across_group),
        (2..8)
            .map(|column| exact_bits(source_code(1, column), scale_bits(1, column / GROUP)))
            .collect::<Vec<u32>>()
    );

    // A window that straddles a word boundary reads both words.
    let mut across_word = [0.0f32; 4];
    tensor.decode_row(2, 6, &mut across_word)?;
    assert_eq!(
        bits_of(&across_word),
        (6..10)
            .map(|column| exact_bits(source_code(2, column), scale_bits(2, column / GROUP)))
            .collect::<Vec<u32>>()
    );

    // Bitwise equivalence against the source bytes themselves, read through the parsed entry spans
    // rather than through the reader: one whole packed little-endian word supplies the nibble and the
    // scale element supplies the raw BF16 pattern, so the expected product is rebuilt independently.
    // A decoder that held only part of a word agrees with nothing here except slot zero.
    let packed_entry = source.get("proj.weight_packed")?;
    let scale_entry = source.get("proj.weight_scale")?;
    for (row, column) in [
        (0usize, 0usize),
        (0, 1),
        (0, 7),
        (1, 8),
        (1, 11),
        (2, 15),
        (2, 23),
    ] {
        let word_bytes = (row * (COLS / PACK_FACTOR) + column / PACK_FACTOR) * 4;
        let mut raw_word = [0u8; 4];
        packed_entry
            .span
            .read_exact_at(word_bytes as u64, &mut raw_word)?;
        let nibble = ((u32::from_le_bytes(raw_word) >> ((column % PACK_FACTOR) * 4)) & 0x0f) as u8;
        let scale_bytes = (row * (COLS / GROUP) + column / GROUP) * 2;
        let mut raw_scale = [0u8; 2];
        scale_entry
            .span
            .read_exact_at(scale_bytes as u64, &mut raw_scale)?;
        let bits = u16::from_le_bytes(raw_scale);
        let mut decoded = [0.0f32; 1];
        tensor.decode_row(row, column, &mut decoded)?;
        assert_eq!(
            decoded[0].to_bits(),
            exact_bits(signed_of(nibble), bits),
            "source-byte product at ({row}, {column})"
        );
        assert_eq!(tensor.code(row, column)?, signed_of(nibble));
        assert_eq!(tensor.scale_bits(row, column)?, bits);
    }
    Ok(())
}

#[test]
fn ct_selections_stay_within_the_declared_shape() -> Result<()> {
    let (words, scales) = grid();
    let file = image(&ct_raws(
        "proj",
        ROWS,
        COLS,
        &words,
        [ROWS as i64, COLS as i64],
        &scales,
    ))?;
    let source = SafetensorsFile::open(file.path(), Limits::default())?;
    let tensor = view(&source, "proj", GROUP)?;

    // The last row, column, word, slot and group are all addressable.
    assert_eq!(
        tensor.code(ROWS - 1, COLS - 1)?,
        source_code(ROWS - 1, COLS - 1)
    );
    assert_eq!(
        tensor.scale_bits(ROWS - 1, COLS - 1)?,
        scale_bits(ROWS - 1, COLS / GROUP - 1)
    );
    tensor.decode_row(0, COLS - 1, &mut [0.0f32; 1])?;

    assert_rejected(tensor.code(ROWS, 0), "row beyond the declared shape");
    assert_rejected(tensor.code(0, COLS), "column beyond the declared shape");
    assert_rejected(tensor.code(0, usize::MAX), "column offset overflow");
    assert_rejected(tensor.scale_bits(ROWS, 0), "scale row beyond the shape");
    assert_rejected(tensor.scale_bits(0, COLS), "scale column beyond the shape");
    assert_rejects_slice(&tensor, ROWS, 0, 1);
    assert_rejects_slice(&tensor, 0, COLS, 1);
    assert_rejects_slice(&tensor, 0, COLS - GROUP, GROUP + 1);
    assert_rejects_slice(&tensor, 0, usize::MAX, 1);
    // The shared row-slice rule of every source codec here accepts an empty selection anywhere in
    // range, including one that starts just past the last column, and rejects an invalid row or a
    // start past that. The CT view inherits it; it invents no stricter rule of its own.
    tensor.decode_row(0, 0, &mut [])?;
    tensor.decode_row(0, COLS - 1, &mut [])?;
    tensor.decode_row(0, COLS, &mut [])?;
    assert_rejects_slice(&tensor, 0, COLS + 1, 0);
    assert_rejects_slice(&tensor, ROWS, 0, 0);
    Ok(())
}

#[test]
fn row_slices_are_bounded_and_a_large_tensor_never_expands_to_floating_point() -> Result<()> {
    let (words, scales) = grid();
    let file = image(&ct_raws(
        "proj",
        ROWS,
        COLS,
        &words,
        [ROWS as i64, COLS as i64],
        &scales,
    ))?;
    let source = SafetensorsFile::open(file.path(), Limits::default())?;
    let tensor = view(&source, "proj", GROUP)?;

    // Partial row and column selections of every shape, verified element by element.
    for (row, column, len) in [
        (0usize, 0usize, 1usize),
        (0, 2, 1),
        (1, 3, 10),
        (2, 20, 4),
        (1, 8, 8),
        (2, 0, COLS),
    ] {
        let mut window = vec![0.0f32; len];
        tensor.decode_row(row, column, &mut window)?;
        assert_eq!(
            bits_of(&window),
            (column..column + len)
                .map(|k| exact_bits(source_code(row, k), scale_bits(row, k / GROUP)))
                .collect::<Vec<u32>>(),
            "window at ({row}, {column}) with {len} values"
        );
    }
    drop(tensor);
    drop(source);

    // A declared two-million-row projection: eight megabytes of packed words and four megabytes
    // of scales, of which only the shape metadata and the last row's scale are ever written.
    // Reading the last row must not materialize a 64 MB floating-point tensor, and an out-of-range
    // row must fail.
    const BIG_ROWS: usize = 2_000_000;
    const PACKED_BYTES: u64 = BIG_ROWS as u64 * 4;
    const SHAPE_BYTES: u64 = 16;
    const SCALE_BYTES: u64 = BIG_ROWS as u64 * 2;
    let packed_end = PACKED_BYTES;
    let shape_end = packed_end + SHAPE_BYTES;
    let scale_end = shape_end + SCALE_BYTES;
    let rows = BIG_ROWS;
    let header = format!(
        r#"{{"big.weight_packed":{{"dtype":"I32","shape":[{rows},1],"data_offsets":[0,{packed_end}]}},"big.weight_shape":{{"dtype":"I64","shape":[2],"data_offsets":[{packed_end},{shape_end}]}},"big.weight_scale":{{"dtype":"BF16","shape":[{rows},1],"data_offsets":[{shape_end},{scale_end}]}}}}"#
    );
    let mut big = NamedTempFile::new()?;
    big.write_all(&(header.len() as u64).to_le_bytes())?;
    big.write_all(header.as_bytes())?;
    let base = 8 + header.len() as u64;
    big.seek(SeekFrom::Start(base + packed_end))?;
    big.write_all(&i64_bytes(&[BIG_ROWS as i64, PACK_FACTOR as i64]))?;
    // Only the last row's scale is ever written; every other source byte stays zero.
    big.seek(SeekFrom::Start(base + scale_end - 2))?;
    big.write_all(&bf16_bytes(&[0x3f80]))?;
    big.flush()?;

    let source = SafetensorsFile::open(big.path(), Limits::default())?;
    let tensor = view(&source, "big", PACK_FACTOR)?;
    assert_eq!(tensor.descriptor().rows, BIG_ROWS);
    assert_eq!(tensor.descriptor().cols, PACK_FACTOR);
    assert_eq!(tensor.scale_bits(BIG_ROWS - 1, 0)?, 0x3f80);
    let mut window = [0.0f32; 2];
    tensor.decode_row(BIG_ROWS - 1, PACK_FACTOR - 2, &mut window)?;
    // Unwritten source words are zero nibbles, which reconstruct as -8 under that row's scale.
    assert_eq!(window, [-8.0; 2]);
    assert_rejects_slice(&tensor, BIG_ROWS, 0, 2);
    Ok(())
}

#[test]
fn scale_sections_are_read_unchanged_including_adversarial_bf16_patterns() -> Result<()> {
    // Smallest subnormals of both signs, signed zero, the largest finite value, 1.0 and its
    // successor. Row-major: one pattern per row and group, group width 8.
    const PATTERNS: [[u16; 3]; ROWS] = [
        [0x0001, 0x0003, 0x0004],
        [0x3f80, 0x3f81, 0x8000],
        [0x8001, 0x3f00, 0x7f7f],
    ];
    let words: Vec<u32> = (0..ROWS)
        .flat_map(|row| (0..COLS / PACK_FACTOR).map(move |word| source_word(row, word)))
        .collect();
    let scales: Vec<u16> = PATTERNS.iter().flatten().copied().collect();
    let file = image(&ct_raws(
        "proj",
        ROWS,
        COLS,
        &words,
        [ROWS as i64, COLS as i64],
        &scales,
    ))?;
    let source = SafetensorsFile::open(file.path(), Limits::default())?;
    let tensor = view(&source, "proj", 8)?;

    // Every scale element keeps its exact source bits.
    for row in 0..ROWS {
        for group in 0..3 {
            for column in group * 8..group * 8 + 8 {
                assert_eq!(
                    tensor.scale_bits(row, column)?,
                    PATTERNS[row][group],
                    "scale bits at ({row}, {column})"
                );
            }
        }
    }
    assert_rejected(
        tensor.scale_bits(0, COLS),
        "scale beyond the declared shape",
    );

    // Literal f32 bit patterns derived from BF16 semantics rather than from the reader: subnormal
    // scales stay subnormal, a signed-zero scale keeps the sign of the code, and the largest
    // finite BF16 scale only yields infinities once the code magnitude exceeds one.
    for (row, column, expected) in [
        (0usize, 0usize, 0x8008_0000u32),
        (0, 4, 0x0004_0000),
        (0, 7, 0x8007_0000),
        (0, 8, 0x800c_0000),
        (0, 12, 0x8018_0000),
        (0, 15, 0x8009_0000),
        (0, 16, 0x0000_0000),
        (0, 20, 0x8010_0000),
        (0, 23, 0x0004_0000),
        (1, 0, 0xc0e0_0000),
        (1, 4, 0x40a0_0000),
        (1, 7, 0xc0c0_0000),
        (1, 8, 0xc041_8000),
        (1, 12, 0xc0e1_c000),
        (1, 15, 0xc001_0000),
        (1, 16, 0x8000_0000),
        (1, 20, 0x0000_0000),
        (1, 23, 0x8000_0000),
        (2, 0, 0x0006_0000),
        (2, 4, 0x8006_0000),
        (2, 7, 0x0005_0000),
        (2, 8, 0xbf80_0000),
        (2, 12, 0xc040_0000),
        (2, 15, 0xbf00_0000),
        (2, 16, 0x7f80_0000),
        (2, 20, 0xff80_0000),
        (2, 23, 0x7f80_0000),
    ] {
        let mut value = [0.0f32; 1];
        tensor.decode_row(row, column, &mut value)?;
        assert_eq!(
            value[0].to_bits(),
            expected,
            "code {} at ({row}, {column}) with scale {:#06x}",
            source_code(row, column),
            PATTERNS[row][column / 8]
        );
    }

    // Documented validity policy: the reader owns bit-exact preservation, not value policy. The
    // exact product of a code and a scale is representable in f32 for every four-bit code while
    // |scale| <= 2^124; larger scales overflow to an infinity and non-finite scales propagate
    // instead of being rejected, saturated or dropped.
    for (bits, code, expected) in [
        (0x7f7fu16, 1i8, 0x7f7f_0000u32),
        (0x7f7f, -1, 0xff7f_0000),
        (0x7f7f, 2, 0x7f80_0000),
        (0x7f7f, -8, 0xff80_0000),
        (0x7f80, -8, 0xff80_0000),
        (0xff80, -8, 0x7f80_0000),
        (0x3f00, -8, 0xc080_0000),
        (0x8000, 7, 0x8000_0000),
    ] {
        let word = word_of([code; PACK_FACTOR]);
        let file = image(&ct_raws("edge", 1, 8, &[word], [1, 8], &[bits]))?;
        let source = SafetensorsFile::open(file.path(), Limits::default())?;
        let tensor = view(&source, "edge", 8)?;
        let mut value = [0.0f32; PACK_FACTOR];
        tensor.decode_row(0, 0, &mut value)?;
        assert_eq!(
            bits_of(&value),
            vec![expected; PACK_FACTOR],
            "code {code} with scale {bits:#06x}"
        );
        assert_eq!(tensor.scale_bits(0, 0)?, bits, "scale {bits:#06x} bits");
    }
    // Infinities and NaNs propagate from the scale instead of silently becoming zeros.
    for bits in [0x7f80u16, 0xff80, 0x7f81, 0x7fc0, 0x7fc1, 0xffc1] {
        let file = image(&ct_raws(
            "edge",
            1,
            8,
            &[word_of([-8, -7, -6, -5, -4, -3, -2, -1])],
            [1, 8],
            &[bits],
        ))?;
        let source = SafetensorsFile::open(file.path(), Limits::default())?;
        let tensor = view(&source, "edge", 8)?;
        let mut value = [0.0f32; PACK_FACTOR];
        tensor.decode_row(0, 0, &mut value)?;
        assert!(
            value
                .iter()
                .all(|value| value.is_nan() || value.is_infinite()),
            "scale {bits:#06x} produced {value:?}"
        );
        assert_eq!(tensor.scale_bits(0, 0)?, bits, "scale {bits:#06x} bits");
    }
    Ok(())
}

#[test]
fn writer_copies_all_compressed_sections_unchanged_with_one_byte_scratch() -> Result<()> {
    let (words, scales) = grid();
    let file = image(&ct_raws(
        "proj",
        ROWS,
        COLS,
        &words,
        [ROWS as i64, COLS as i64],
        &scales,
    ))?;
    let source = SafetensorsFile::open(file.path(), Limits::default())?;
    let tensor = view(&source, "proj", GROUP)?;

    // The destination holds every compressed section - packed words, the I64 shape metadata and
    // the scales - byte for byte and in this order. It is a faithful source image, not a
    // self-describing one: the returned section names and placements plus the descriptor/format are
    // what make it interpretable, exactly as for the other codecs.
    let mut expected = i32_bytes(&words);
    expected.extend(i64_bytes(&[ROWS as i64, COLS as i64]));
    expected.extend(bf16_bytes(&scales));
    struct Bounded {
        bytes: Vec<u8>,
        max_write: usize,
    }
    impl Write for Bounded {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.max_write = self.max_write.max(bytes.len());
            self.bytes.extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut out = Bounded {
        bytes: Vec::new(),
        max_write: 0,
    };
    let written = write_encoded(&tensor, &mut out, &mut [0u8; 1])?;
    assert_eq!(out.bytes, expected, "source payload changed");
    assert_eq!(out.max_write, 1, "writer exceeded one-byte scratch");
    assert_eq!(
        written
            .iter()
            .map(|section| (section.name.as_str(), section.offset, section.len))
            .collect::<Vec<_>>(),
        vec![
            ("weight_packed", 0u64, 36u64),
            ("weight_shape", 36u64, 16u64),
            ("weight_scale", 52u64, 36u64),
        ]
    );
    assert!(write_encoded(&tensor, &mut Vec::new(), &mut []).is_err());
    Ok(())
}

#[test]
fn rejects_unrepresentable_declared_formats() -> Result<()> {
    // Only symmetric 4-bit group codes with BF16 scales are represented. Any other declaration is
    // rejected rather than reinterpreted, so no caller can silently claim another format.
    for (strategy, bits, symmetric, group_size, scale_type) in [
        ("group", 8, true, GROUP, ScalarType::Bf16),
        ("group", 2, true, GROUP, ScalarType::Bf16),
        ("group", 4, false, GROUP, ScalarType::Bf16),
        ("group", 4, true, 0, ScalarType::Bf16),
        ("group", 4, true, GROUP, ScalarType::F16),
        ("group", 4, true, GROUP, ScalarType::F32),
        ("group", 4, true, GROUP, ScalarType::I32),
        ("group", 4, true, GROUP, ScalarType::I64),
        ("channel", 4, true, GROUP, ScalarType::Bf16),
        ("block", 4, true, GROUP, ScalarType::Bf16),
        ("tensor", 4, true, GROUP, ScalarType::Bf16),
        ("GROUP", 4, true, GROUP, ScalarType::Bf16),
        ("grouped", 4, true, GROUP, ScalarType::Bf16),
        ("sub-channel", 4, true, GROUP, ScalarType::Bf16),
        ("", 4, true, GROUP, ScalarType::Bf16),
    ] {
        assert!(
            PackQuantizedFormat::declared(strategy, bits, symmetric, group_size, scale_type).is_err(),
            "accepted {strategy:?} bits {bits} symmetric {symmetric} group {group_size} scales {scale_type:?}"
        );
    }

    // Any declared group size that divides the source columns is accepted with a matching scale
    // section; the reader does not substitute a default group width.
    let (words, _) = grid();
    for group_size in [4usize, 8, 12, 24] {
        let scale_values = vec![0x3f80u16; ROWS * COLS / group_size];
        let entries = vec![
            raw(
                "proj.weight_packed".into(),
                "I32",
                &[ROWS, COLS / PACK_FACTOR],
                i32_bytes(&words),
            ),
            raw(
                "proj.weight_shape".into(),
                "I64",
                &[2],
                i64_bytes(&[ROWS as i64, COLS as i64]),
            ),
            raw(
                "proj.weight_scale".into(),
                "BF16",
                &[ROWS, COLS / group_size],
                bf16_bytes(&scale_values),
            ),
        ];
        assert!(
            build(&entries, "proj", group_size).is_ok(),
            "declared group size {group_size}"
        );
    }
    let (words, scales) = grid();
    let entries = ct_raws(
        "proj",
        ROWS,
        COLS,
        &words,
        [ROWS as i64, COLS as i64],
        &scales,
    );
    assert_rejects_view(&entries, "group size that is not a divisor", 5);
    assert_rejects_view(&entries, "group size wider than the row", 48);
    Ok(())
}

#[test]
fn rejects_inconsistent_or_unsupported_weight_sections() -> Result<()> {
    let (words, scales) = grid();

    // A declared zero point means asymmetric codes; a declared group index map means a permuted
    // K axis. Neither is representable here, and neither may be dropped on the floor.
    for name in ["weight_zero_point", "weight_g_idx"] {
        let mut entries = ct_raws(
            "proj",
            ROWS,
            COLS,
            &words,
            [ROWS as i64, COLS as i64],
            &scales,
        );
        entries.push(raw(
            format!("proj.{name}"),
            "I32",
            &[ROWS],
            i32_bytes(&[0u32; ROWS]),
        ));
        assert_rejects_view(&entries, &format!("{name} declared"), GROUP);
    }

    // Each section below is internally consistent, so the parser accepts the image and only the
    // reader's cross-check of packed words, declared shape and scale groups can reject it.
    for (label, packed_rows, scale_rows, shape, scale_cols, group_size) in [
        (
            "packed rows disagree with shape metadata",
            ROWS + 1,
            ROWS + 1,
            [ROWS as i64, COLS as i64],
            COLS / GROUP,
            GROUP,
        ),
        (
            "shape rows disagree with packed rows",
            ROWS,
            ROWS,
            [ROWS as i64 + 1, COLS as i64],
            COLS / GROUP,
            GROUP,
        ),
        (
            "scale rows disagree",
            ROWS,
            ROWS - 1,
            [ROWS as i64, COLS as i64],
            COLS / GROUP,
            GROUP,
        ),
        (
            "packed words leave padded nibbles",
            ROWS,
            ROWS,
            [ROWS as i64, COLS as i64 - 1],
            COLS / GROUP,
            GROUP,
        ),
        (
            "columns are not a whole group",
            ROWS,
            ROWS,
            [ROWS as i64, COLS as i64 - 2],
            COLS / GROUP,
            GROUP,
        ),
        (
            "scale groups disagree",
            ROWS,
            ROWS,
            [ROWS as i64, COLS as i64],
            COLS / GROUP - 1,
            GROUP,
        ),
        (
            "zero packed rows",
            0,
            0,
            [0, COLS as i64],
            COLS / GROUP,
            GROUP,
        ),
        (
            "shape metadata is absurd",
            ROWS,
            ROWS,
            [i64::MAX, i64::MAX],
            COLS / GROUP,
            GROUP,
        ),
        (
            "negative shape metadata",
            ROWS,
            ROWS,
            [-1, COLS as i64],
            COLS / GROUP,
            GROUP,
        ),
    ] {
        let entries = vec![
            raw(
                "proj.weight_packed".into(),
                "I32",
                &[packed_rows, COLS / PACK_FACTOR],
                i32_bytes(
                    &words
                        .iter()
                        .copied()
                        .cycle()
                        .take(packed_rows * COLS / PACK_FACTOR)
                        .collect::<Vec<u32>>(),
                ),
            ),
            raw("proj.weight_shape".into(), "I64", &[2], i64_bytes(&shape)),
            raw(
                "proj.weight_scale".into(),
                "BF16",
                &[scale_rows, scale_cols],
                bf16_bytes(
                    &scales
                        .iter()
                        .copied()
                        .cycle()
                        .take(scale_rows * scale_cols)
                        .collect::<Vec<u16>>(),
                ),
            ),
        ];
        assert_rejects_view(&entries, label, group_size);
    }

    // A scale section whose byte extent does not match its declared dtype and shape must be caught
    // by the parser before any reader sees it.
    let mut entries = ct_raws(
        "proj",
        ROWS,
        COLS,
        &words,
        [ROWS as i64, COLS as i64],
        &scales,
    );
    entries[2].bytes.truncate(scales.len() * 2 - 2);
    assert_rejects_view(&entries, "scale section truncated", GROUP);

    // Wrong dtypes and ranks are rejected on their own merits, not by an extent coincidence.
    for (label, index, dtype, shape, bytes) in [
        (
            "packed codes are rank 1",
            0usize,
            "I32",
            vec![ROWS * COLS / PACK_FACTOR],
            i32_bytes(&words),
        ),
        (
            "scales are rank 1",
            2,
            "BF16",
            vec![ROWS * COLS / GROUP],
            bf16_bytes(&scales),
        ),
        (
            "packed codes are not I32",
            0usize,
            "I64",
            vec![ROWS, COLS / PACK_FACTOR],
            i64_bytes(&[0i64; ROWS * COLS / PACK_FACTOR]),
        ),
        (
            "scales are not BF16",
            2,
            "I32",
            vec![ROWS, COLS / GROUP],
            i32_bytes(&[0u32; ROWS * COLS / GROUP]),
        ),
        (
            "shape metadata is not I64",
            1,
            "I32",
            vec![2],
            i32_bytes(&[ROWS as u32, COLS as u32]),
        ),
        (
            "shape metadata is not one-dimensional",
            1,
            "I64",
            vec![1, 2],
            i64_bytes(&[ROWS as i64, COLS as i64]),
        ),
        (
            "shape metadata holds more than two dimensions",
            1,
            "I64",
            vec![3],
            i64_bytes(&[ROWS as i64, COLS as i64, 1]),
        ),
    ] {
        let mut entries = ct_raws(
            "proj",
            ROWS,
            COLS,
            &words,
            [ROWS as i64, COLS as i64],
            &scales,
        );
        entries[index].dtype = dtype.to_string();
        entries[index].shape = shape;
        entries[index].bytes = bytes;
        assert_rejects_view(&entries, label, GROUP);
    }
    Ok(())
}

#[test]
fn rejects_packed_words_whose_rows_disagree_with_the_declared_shape() -> Result<()> {
    // Agreement of the total nibble count is not per-row geometry. The same nine fixture words
    // reshaped into one row of nine, or into nine rows of one word, still cover the 72 nibbles of
    // the declared [3, 24] and leave the scale section valid, so only a row-wise cross-check of the
    // packed rows against the declared shape can reject them. Both images are otherwise perfectly
    // valid safetensors objects that the production parser accepts, so any rejection has to come
    // from the reader rather than from a forged entry or a header coincidence.
    let (words, scales) = grid();
    for (label, packed_shape) in [
        (
            "one packed row holding all three source rows",
            [1usize, ROWS * COLS / PACK_FACTOR],
        ),
        (
            "nine packed rows of one word each",
            [ROWS * COLS / PACK_FACTOR, 1usize],
        ),
    ] {
        assert_eq!(
            packed_shape[0] * packed_shape[1],
            ROWS * COLS / PACK_FACTOR,
            "{label}: the packed word count must equal the accepted image's"
        );
        let entries = vec![
            raw(
                "proj.weight_packed".into(),
                "I32",
                &packed_shape,
                i32_bytes(&words),
            ),
            raw(
                "proj.weight_shape".into(),
                "I64",
                &[2],
                i64_bytes(&[ROWS as i64, COLS as i64]),
            ),
            raw(
                "proj.weight_scale".into(),
                "BF16",
                &[ROWS, COLS / GROUP],
                bf16_bytes(&scales),
            ),
        ];
        assert_rejects_view(&entries, label, GROUP);
    }

    // Control: those exact bytes are accepted under the declared row/word geometry, so the
    // rejections above are about per-row agreement and not about the packed payload.
    let control = ct_raws(
        "proj",
        ROWS,
        COLS,
        &words,
        [ROWS as i64, COLS as i64],
        &scales,
    );
    assert!(
        build(&control, "proj", GROUP).is_ok(),
        "the correctly shaped packed image was rejected"
    );
    Ok(())
}

#[test]
fn rejects_a_literal_format_declaring_a_scale_type_other_than_bf16() -> Result<()> {
    // PackQuantizedFormat's fields are public, so a caller can construct the declaration directly
    // instead of calling `declared`. The routed sections below stay the accepted BF16 fixture, so
    // only the declared scale type can distinguish these constructions from the accepted one: a
    // view may not report a scale type it did not verify it can read.
    let (words, scales) = grid();
    let file = image(&ct_raws(
        "proj",
        ROWS,
        COLS,
        &words,
        [ROWS as i64, COLS as i64],
        &scales,
    ))?;
    let source = SafetensorsFile::open(file.path(), Limits::default())?;
    assert!(
        PackQuantizedTensor::new("proj", sections(&source, "proj")?, group_format(GROUP)?).is_ok(),
        "the declared BF16 control was rejected"
    );

    for scale_type in [
        ScalarType::F16,
        ScalarType::F32,
        ScalarType::I32,
        ScalarType::I64,
    ] {
        let format = PackQuantizedFormat {
            group_size: GROUP,
            scale_type,
        };
        assert_rejected(
            PackQuantizedTensor::new("proj", sections(&source, "proj")?, format),
            "literal format declaring a scale type the codec does not decode",
        );
    }
    Ok(())
}

#[test]
fn rejects_truncated_contained_and_duplicate_compressed_headers() -> Result<()> {
    let (words, scales) = grid();
    let valid = ct_raws(
        "proj",
        ROWS,
        COLS,
        &words,
        [ROWS as i64, COLS as i64],
        &scales,
    );
    let payload: Vec<u8> = i32_bytes(&words)
        .into_iter()
        .chain(i64_bytes(&[ROWS as i64, COLS as i64]))
        .chain(bf16_bytes(&scales))
        .collect();
    let packed_len = words.len() * 4;
    let shape_end = packed_len + 16;
    let scale_end = shape_end + scales.len() * 2;
    assert_eq!(payload.len(), scale_end);

    let header = format!(
        r#"{{"proj.weight_packed":{{"dtype":"I32","shape":[{ROWS},3],"data_offsets":[0,{packed_len}]}},"proj.weight_shape":{{"dtype":"I64","shape":[2],"data_offsets":[{packed_len},{shape_end}]}},"proj.weight_scale":{{"dtype":"BF16","shape":[{ROWS},6],"data_offsets":[{shape_end},{scale_end}]}}}}"#
    );
    // The reference image is accepted before the mutations below.
    let mut file = NamedTempFile::new()?;
    file.write_all(&(header.len() as u64).to_le_bytes())?;
    file.write_all(header.as_bytes())?;
    file.write_all(&payload)?;
    file.flush()?;
    assert!(view(
        &SafetensorsFile::open(file.path(), Limits::default())?,
        "proj",
        GROUP
    )
    .is_ok());

    for truncate in [1usize, 3, packed_len + 1] {
        let mut file = NamedTempFile::new()?;
        file.write_all(&(header.len() as u64).to_le_bytes())?;
        file.write_all(header.as_bytes())?;
        file.write_all(&payload[..payload.len() - truncate])?;
        file.flush()?;
        assert!(
            SafetensorsFile::open(file.path(), Limits::default()).is_err(),
            "accepted a source truncated by {truncate} bytes"
        );
    }
    let overlap_start = packed_len - 2;
    // The duplicate-name image is otherwise perfectly valid - both entries describe the same
    // span and every offset and byte lines up - so only duplicate-key detection can reject it.
    // The overlap entry stays a separate span/gap rule.
    for (label, header) in [
        (
            "duplicate tensor name",
            format!(
                r#"{{"proj.weight_packed":{{"dtype":"I32","shape":[{ROWS},3],"data_offsets":[0,{packed_len}]}},"proj.weight_packed":{{"dtype":"I32","shape":[{ROWS},3],"data_offsets":[0,{packed_len}]}},"proj.weight_shape":{{"dtype":"I64","shape":[2],"data_offsets":[{packed_len},{shape_end}]}},"proj.weight_scale":{{"dtype":"BF16","shape":[{ROWS},6],"data_offsets":[{shape_end},{scale_end}]}}}}"#
            ),
        ),
        (
            "duplicate tensor field",
            format!(
                r#"{{"proj.weight_packed":{{"dtype":"I32","dtype":"I32","shape":[{ROWS},3],"data_offsets":[0,{packed_len}]}},"proj.weight_scale":{{"dtype":"BF16","shape":[{ROWS},6],"data_offsets":[{shape_end},{scale_end}]}}}}"#
            ),
        ),
        (
            "overlapping sections",
            format!(
                r#"{{"proj.weight_packed":{{"dtype":"I32","shape":[{ROWS},3],"data_offsets":[0,{packed_len}]}},"proj.weight_scale":{{"dtype":"BF16","shape":[{ROWS},6],"data_offsets":[{overlap_start},{scale_end}]}}}}"#
            ),
        ),
    ] {
        let mut file = NamedTempFile::new()?;
        file.write_all(&(header.len() as u64).to_le_bytes())?;
        file.write_all(header.as_bytes())?;
        file.write_all(&payload)?;
        file.flush()?;
        assert!(
            SafetensorsFile::open(file.path(), Limits::default()).is_err(),
            "accepted {label}"
        );
    }

    // A missing parameter or an empty span is a rejection, not a default.
    for label in [
        "packed missing",
        "shape missing",
        "scale missing",
        "empty span",
    ] {
        let mut entries = valid.clone();
        match label {
            "packed missing" => entries.retain(|entry| entry.name != "proj.weight_packed"),
            "shape missing" => entries.retain(|entry| entry.name != "proj.weight_shape"),
            "scale missing" => entries.retain(|entry| entry.name != "proj.weight_scale"),
            _ => entries[1].bytes = Vec::new(),
        }
        assert_rejects_view(&entries, label, GROUP);
    }
    Ok(())
}

#[test]
fn i64_is_shape_metadata_only_and_never_a_weight_encoding() -> Result<()> {
    let (words, scales) = grid();
    let file = image(&ct_raws(
        "proj",
        ROWS,
        COLS,
        &words,
        [ROWS as i64, COLS as i64],
        &scales,
    ))?;
    let source = SafetensorsFile::open(file.path(), Limits::default())?;
    let shape = source.get("proj.weight_shape")?;
    assert_eq!(shape.dtype, ScalarType::I64);
    assert_eq!(shape.shape, [2]);
    assert_eq!(shape.span.len(), 16);
    let mut metadata = [0i64; 2];
    for (index, value) in metadata.iter_mut().enumerate() {
        let mut bytes = [0; 8];
        shape.span.read_exact_at((index * 8) as u64, &mut bytes)?;
        *value = i64::from_le_bytes(bytes);
    }
    assert_eq!(metadata, [ROWS as i64, COLS as i64]);

    // Accepting I64 for a metadata tensor must not admit I64 as dense weight storage or change any
    // other codec contract. Both rejections below have matching byte extents, so only the declared
    // scalar type can cause them.
    let metadata_span = source.get("proj.weight_shape")?.span.clone();
    assert_rejected(
        DenseTensor::new("metadata", 2, 1, ScalarType::I64, metadata_span.clone()),
        "I64 dense tensor",
    );
    assert_rejected(
        DenseTensor::new("metadata", 1, 4, ScalarType::I32, metadata_span.clone()),
        "I32 dense tensor",
    );
    // The declared I32 codes and BF16 scales still read as themselves.
    assert_eq!(source.get("proj.weight_packed")?.dtype, ScalarType::I32);
    assert_eq!(source.get("proj.weight_scale")?.dtype, ScalarType::Bf16);

    // A metadata tensor cannot be presented as the packed codes or as the scales.
    let mut swapped = sections(&source, "proj")?;
    std::mem::swap(&mut swapped.weight_packed, &mut swapped.weight_shape);
    assert_rejected(
        PackQuantizedTensor::new("proj", swapped, group_format(GROUP)?),
        "shape metadata presented as packed codes",
    );
    let mut swapped = sections(&source, "proj")?;
    std::mem::swap(&mut swapped.weight_scale, &mut swapped.weight_shape);
    assert_rejected(
        PackQuantizedTensor::new("proj", swapped, group_format(GROUP)?),
        "shape metadata presented as scales",
    );
    Ok(())
}

#[test]
fn view_keeps_its_owned_spans_after_the_reader_is_dropped() -> Result<()> {
    // A view must not depend on the container that produced it: each FileSpan keeps the source
    // open on its own, which is what lets a producer stream long after parsing.
    let (words, scales) = grid();
    let file = image(&ct_raws(
        "proj",
        ROWS,
        COLS,
        &words,
        [ROWS as i64, COLS as i64],
        &scales,
    ))?;
    let source = SafetensorsFile::open(file.path(), Limits::default())?;
    let tensor = view(&source, "proj", GROUP)?;
    drop(source);

    let row = ROWS - 1;
    let column = COLS - 1;
    assert_eq!(tensor.code(row, column)?, source_code(row, column));
    assert_eq!(
        tensor.scale_bits(row, column)?,
        scale_bits(row, column / GROUP)
    );
    let mut decoded = vec![0.0f32; COLS];
    tensor.decode_row(row, 0, &mut decoded)?;
    assert_eq!(
        bits_of(&decoded),
        (0..COLS)
            .map(|k| exact_bits(source_code(row, k), scale_bits(row, k / GROUP)))
            .collect::<Vec<u32>>()
    );
    let mut copy = Vec::new();
    write_encoded(&tensor, &mut copy, &mut [0u8; 3])?;
    assert_eq!(copy.len(), words.len() * 4 + 16 + scales.len() * 2);
    Ok(())
}

#[test]
fn reads_the_selected_column_geometry_with_synthetic_rows() -> Result<()> {
    // The column geometry recorded as reviewed header evidence for a selected projection (I32 packed
    // [rows, 480], I64 shape metadata [2] holding [rows, 3840], BF16 scales [rows, 120], group size
    // 32) rebuilt with two synthetic rows, because the real projections have thousands of rows. No
    // publisher payload is read, and none of these numbers come from an in-repo fixture: they are
    // test inputs chosen to match the recorded geometry.
    const ROWS: usize = 2;
    const COLS: usize = 3840;
    const GROUPS: usize = 120;
    let packed_cols = COLS / PACK_FACTOR;
    let mut words = vec![0u32; ROWS * packed_cols];
    // Row 1, columns 0..8 hold the codes 0..7; every other code is -8 (a zero nibble).
    words[packed_cols] = 0xFEDC_BA98;
    let mut scales = vec![0x3f80u16; ROWS * GROUPS];
    scales[GROUPS - 1 + GROUPS] = 0x3f00;
    let prefix = "model.layers.0.self_attn.q_proj";
    let entries = vec![
        raw(
            format!("{prefix}.weight_packed"),
            "I32",
            &[ROWS, packed_cols],
            i32_bytes(&words),
        ),
        raw(
            format!("{prefix}.weight_shape"),
            "I64",
            &[2],
            i64_bytes(&[ROWS as i64, COLS as i64]),
        ),
        raw(
            format!("{prefix}.weight_scale"),
            "BF16",
            &[ROWS, GROUPS],
            bf16_bytes(&scales),
        ),
    ];
    let file = image(&entries)?;
    let source = SafetensorsFile::open(file.path(), Limits::default())?;
    let tensor = view(&source, prefix, 32)?;

    assert_eq!(tensor.descriptor().rows, ROWS);
    assert_eq!(tensor.descriptor().cols, COLS);
    assert_eq!(
        tensor.descriptor().encoding,
        Encoding::CtPackQuantized {
            group_size: 32,
            scale_type: ScalarType::Bf16,
        }
    );

    // The last word of a row and the last group of a row are inside the declared extents.
    assert_eq!(codes_at(&tensor, 0, COLS - 8..COLS), vec![-8; 8]);
    assert_eq!(tensor.scale_bits(0, COLS - 1)?, 0x3f80);
    assert_eq!(tensor.scale_bits(1, COLS - 1)?, 0x3f00);
    assert_eq!(codes_at(&tensor, 1, 0..9), vec![0, 1, 2, 3, 4, 5, 6, 7, -8]);

    // One whole group of columns decodes with that group's scale only, and the neighbouring group
    // with its own.
    let mut group = vec![0.0f32; 32];
    tensor.decode_row(1, COLS - 32, &mut group)?;
    assert_eq!(group, vec![-4.0; 32], "code -8 under a 0.5 scale");
    let mut neighbour = vec![0.0f32; 32];
    tensor.decode_row(1, COLS - 64, &mut neighbour)?;
    assert_eq!(neighbour, vec![-8.0; 32], "code -8 under a 1.0 scale");
    let mut first = [0.0f32; 8];
    tensor.decode_row(1, 0, &mut first)?;
    assert_eq!(first, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    assert_rejects_slice(&tensor, ROWS, COLS - 32, 32);
    assert_rejects_slice(&tensor, ROWS - 1, COLS - 31, 32);
    Ok(())
}

#[test]
fn preserves_independent_bf16_embedding_and_head_regions() -> Result<()> {
    // A tied configuration does not establish payload equality: these two BF16 regions hold
    // deliberately different values and must stay independent, unchanged read views.
    let embed = [0x3f80u16, 0x3f80, 0x3f80, 0x3f80, 0x3f80, 0x3f80];
    let head = [0xbf80u16, 0x3f80, 0x4000, 0x4040, 0x3f80, 0x0001];
    let (words, scales) = grid();
    let mut entries = vec![
        raw(
            "model.embed_tokens.weight".into(),
            "BF16",
            &[2, 3],
            bf16_bytes(&embed),
        ),
        raw("lm_head.weight".into(), "BF16", &[2, 3], bf16_bytes(&head)),
    ];
    entries.extend(ct_raws(
        "proj",
        ROWS,
        COLS,
        &words,
        [ROWS as i64, COLS as i64],
        &scales,
    ));
    let file = image(&entries)?;
    let source = SafetensorsFile::open(file.path(), Limits::default())?;

    let embedding = source.get("model.embed_tokens.weight")?.clone();
    let head_region = source.get("lm_head.weight")?.clone();
    assert_eq!(embedding.dtype, ScalarType::Bf16);
    assert_eq!(head_region.dtype, ScalarType::Bf16);
    assert_ne!(
        embedding.span.offset(),
        head_region.span.offset(),
        "two authoritative regions were aliased to one span"
    );

    let embed_tensor = DenseTensor::new("embed", 2, 3, embedding.dtype, embedding.span.clone())?;
    let head_tensor = DenseTensor::new("head", 2, 3, head_region.dtype, head_region.span.clone())?;
    let mut embed_rows = [0.0f32; 6];
    let mut head_rows = [0.0f32; 6];
    embed_tensor.decode_row(0, 0, &mut embed_rows[..3])?;
    embed_tensor.decode_row(1, 0, &mut embed_rows[3..])?;
    head_tensor.decode_row(0, 0, &mut head_rows[..3])?;
    head_tensor.decode_row(1, 0, &mut head_rows[3..])?;
    assert_eq!(bits_of(&embed_rows), vec![0x3f80_0000; 6]);
    assert_eq!(
        bits_of(&head_rows),
        [
            0xbf80_0000,
            0x3f80_0000,
            0x4000_0000,
            0x4040_0000,
            0x3f80_0000,
            0x0001_0000,
        ]
    );

    let mut embed_bytes = Vec::new();
    let mut head_bytes = Vec::new();
    write_encoded(&embed_tensor, &mut embed_bytes, &mut [0u8; 1])?;
    write_encoded(&head_tensor, &mut head_bytes, &mut [0u8; 1])?;
    assert_eq!(embed_bytes, bf16_bytes(&embed));
    assert_eq!(head_bytes, bf16_bytes(&head));
    assert_ne!(
        embed_bytes, head_bytes,
        "a tied configuration must not make these payloads equal"
    );

    // Building the CT view neither consumes nor reinterprets either BF16 region.
    let tensor = view(&source, "proj", GROUP)?;
    let mut embed_after = [0.0f32; 3];
    let mut head_after = [0.0f32; 3];
    embed_tensor.decode_row(1, 0, &mut embed_after)?;
    head_tensor.decode_row(1, 0, &mut head_after)?;
    assert_eq!(bits_of(&embed_after), vec![0x3f80_0000; 3]);
    assert_eq!(
        bits_of(&head_after),
        [0x4040_0000, 0x3f80_0000, 0x0001_0000]
    );
    assert_eq!(tensor.descriptor().cols, COLS);
    assert_eq!(
        tensor
            .sections()
            .iter()
            .map(|section| section.name.as_str())
            .collect::<Vec<_>>(),
        ["weight_packed", "weight_shape", "weight_scale"]
    );
    Ok(())
}

#[test]
fn keeps_source_codes_and_scales_instead_of_requantizing() -> Result<()> {
    // One CT encoding whose codes and scales are exactly the ones the test wrote. A CT view must
    // report those codes and those scale bits; any requantization, clip search or scale selection
    // would change at least one of them.
    let codes = [-8i8, -3, -1, 0, 1, 5, 7, -8, 4, 4, -2, 6, -5, 0, 7, 3];
    let scale_bits = [0x3f80u16, 0x3f00, 0x3f81, 0x8000];
    let words: Vec<u32> = (0..2)
        .map(|word| {
            let mut block = [0i8; PACK_FACTOR];
            block.copy_from_slice(&codes[word * PACK_FACTOR..(word + 1) * PACK_FACTOR]);
            word_of(block)
        })
        .collect();
    assert_eq!(words, vec![0x0fd9_8750, 0xbf83_e6cc]);
    let mut entries = ct_raws("proj", 1, 16, &words, [1, 16], &scale_bits);
    // A separate authoritative BF16 region, unrelated to what the CT reader must report.
    entries.push(raw(
        "twin".into(),
        "BF16",
        &[1, 16],
        bf16_bytes(&[0x3f80u16; 16]),
    ));
    let file = image(&entries)?;
    let source = SafetensorsFile::open(file.path(), Limits::default())?;

    let tensor = view(&source, "proj", GROUP)?;
    assert_eq!(
        codes_at(&tensor, 0, 0..16),
        codes,
        "source codes were altered"
    );
    for (index, expected) in scale_bits.iter().enumerate() {
        for column in index * GROUP..index * GROUP + GROUP {
            assert_eq!(tensor.scale_bits(0, column)?, *expected);
        }
    }
    let mut decoded = [0.0f32; 16];
    tensor.decode_row(0, 0, &mut decoded)?;
    assert_eq!(
        bits_of(&decoded),
        (0..16)
            .map(|column| exact_bits(codes[column], scale_bits[column / GROUP]))
            .collect::<Vec<u32>>()
    );

    // The compressed sections are byte-identical to the source image, in order.
    let mut copy = Vec::new();
    write_encoded(&tensor, &mut copy, &mut [0u8; 7])?;
    let mut expected = i32_bytes(&words);
    expected.extend(i64_bytes(&[1, 16]));
    expected.extend(bf16_bytes(&scale_bits));
    assert_eq!(copy, expected, "compressed sections were not copied");

    // The unrelated BF16 region is untouched by the CT read.
    let twin = source.get("twin")?;
    assert_eq!(twin.span.len(), 32);
    let mut twin_read = vec![0u8; 32];
    twin.span.read_exact_at(0, &mut twin_read)?;
    assert_eq!(twin_read, bf16_bytes(&[0x3f80u16; 16]));
    Ok(())
}
