//! Exact source-preserving repack of one compressed-tensors `pack-quantized` tile into the chain
//! kernel's int4/BF16-scale 8x8-subtile packet layout.
//!
//! The packet contract is owned outside this crate and appears here only as an expectation:
//! `designs/resident_forward/chain_ref.py` (`sub_bytes`, `decode_sub`) and the pinned IRON kernel
//! `aie_kernels/aie2p/chain_mm_bfp16.cc` (`SUB_BYTES`, `convert_sub`, whose static asserts require
//! `NB % 16 == 0` and `SUB_K % 32 == 0` at a 32-column scale group). One packet is `[cols/32][rows]`
//! little-endian BF16 scale bits, then `[rows/8][cols/8]` 32-byte subtiles with nibble lane
//! `i = (n = i/8, k = i%8)`, low nibble first, holding a signed two's-complement s4. Those are
//! hardware packet constraints on the output geometry, not new model policy, and every extent here is
//! a test input rather than a model shape.
//!
//! What this adapts is DATA BYTES ONLY: a source nibble `c` represents the signed code `c - 8`, so the
//! packet's s4 nibble is `c ^ 8` - byte-wise `^ 0x88`, always legal here because every packet byte
//! carries two adjacent-K nibbles - and the source scale's raw 16 bits are copied verbatim, with no
//! clipping, requantization, dequantization or float widening. The kernel converts weights to BFP16
//! internally, so producing this packet layout is neither numerical, program nor device acceptance and
//! authorizes nothing about that compute path; BF16 embedding and head regions stay BF16.
//!
//! This is not the other packing in this workspace: `iron/common/quant.py`'s GEMM helper uses a
//! row-major F32-scale header and declines BF16, so its layout is unused and no quantizer is imported
//! or called. The only external decode is the maintained `chain_ref.decode_sub` CPU reference, for a
//! finite-value reconstruction golden; raw-bit assertions stay separate from it so a NaN or infinity
//! scale pattern cannot hide a cast.
//!
//! One alignment rule the fixtures honour: a packet's K groups must coincide with source scale groups,
//! so `first_col % 32 == 0` is required before any byte is written, and every expectation indexes the
//! source scale by `(first_col + k) / 32` instead of assuming a floored start. An unaligned column
//! start - including 8 - cuts a real source group in half and is a rejection, not a guess or padding. A
//! row start needs no alignment: groups depend on K, and a packet carries its own 8-row subtile
//! blocking relative to `first_row`.

use anyhow::{ensure, Context, Result};
use npu_weights::source_native::ct::{
    ChainTile, PackQuantizedFormat, PackQuantizedTensor, WeightSections,
};
use npu_weights::source_native::safetensors::SafetensorsFile;
use npu_weights::source_native::{Limits, ScalarType, WeightTensor};
use serde_json::json;
use std::io::Write;
use std::path::PathBuf;
use tempfile::NamedTempFile;

/// Nibbles per source I32 packed word.
const PACK_FACTOR: usize = 8;
/// The source group width and the packet scale group: the two must be the same number.
const GROUP: usize = 32;
/// Fixture extents: 24 words and six groups per row, so tiles sub-select on both axes.
const ROWS: usize = 48;
const COLS: usize = 192;
// Two valid kernel geometries: 192 scale bytes + 1536 code, and 32 scale bytes + 256 code.
const TALL_ROWS: usize = 32;
const TALL_COLS: usize = 96;
const TALL_BYTES: usize = 1728;
const SMALL_ROWS: usize = 16;
const SMALL_COLS: usize = 32;
const SMALL_BYTES: usize = 288;
/// The byte every unfilled caller buffer starts at, so "nothing was written" is observable.
const CANARY: u8 = 0x5a;

#[derive(Clone)]
struct Raw {
    name: String,
    dtype: String,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

fn raw(name: &str, dtype: &str, shape: &[usize], bytes: Vec<u8>) -> Raw {
    Raw {
        name: name.to_string(),
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

/// A real tiny safetensors image, so the production header parser and its span accounting run.
fn image(entries: &[Raw]) -> Result<NamedTempFile> {
    let mut header = serde_json::Map::new();
    let mut offset = 0u64;
    for entry in entries {
        let len = u64::try_from(entry.bytes.len())?;
        let span = json!([offset, offset + len]);
        header.insert(
            entry.name.clone(),
            json!({"dtype": entry.dtype, "shape": entry.shape, "data_offsets": span}),
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

fn opened(
    rows: usize,
    cols: usize,
    words: &[u32],
    scales: &[u16],
) -> Result<(NamedTempFile, SafetensorsFile)> {
    let packed = raw(
        "proj.weight_packed",
        "I32",
        &[rows, cols / PACK_FACTOR],
        i32_bytes(words),
    );
    let shape = raw(
        "proj.weight_shape",
        "I64",
        &[2],
        i64_bytes(&[rows as i64, cols as i64]),
    );
    let scale = raw(
        "proj.weight_scale",
        "BF16",
        &[rows, scales.len() / rows],
        bf16_bytes(scales),
    );
    let file = image(&[packed, shape, scale])?;
    let source = SafetensorsFile::open(file.path(), Limits::default())?;
    Ok((file, source))
}

/// The view under test, from the same fixture bytes at a declared group width.
fn tile_view(
    rows: usize,
    cols: usize,
    group: usize,
    words: &[u32],
    scales: &[u16],
) -> Result<(NamedTempFile, PackQuantizedTensor)> {
    let (file, source) = opened(rows, cols, words, scales)?;
    let sections = WeightSections {
        weight_packed: source.get("proj.weight_packed")?.clone(),
        weight_shape: source.get("proj.weight_shape")?.clone(),
        weight_scale: source.get("proj.weight_scale")?.clone(),
        weight_zero_point: None,
        weight_g_idx: None,
    };
    let format = PackQuantizedFormat::declared("group", 4, true, group, ScalarType::Bf16)?;
    Ok((file, PackQuantizedTensor::new("proj", sections, format)?))
}

fn tile(first_row: usize, first_col: usize, rows: usize, cols: usize) -> ChainTile {
    ChainTile {
        first_row,
        first_col,
        rows,
        cols,
    }
}

fn canary(len: usize) -> Vec<u8> {
    vec![CANARY; len]
}

/// The unsigned nibble of a signed four-bit code.
fn nibble_of(code: i8) -> u32 {
    u32::from(((code as u8) & 0x0f) ^ 8)
}

/// Independent signed reconstruction of an unsigned nibble `c`: `c - 8`.
fn signed_of(nibble: u8) -> i8 {
    nibble as i8 - 8
}

/// Unique words for this fixture's row/word pairs: odd multiplication is invertible modulo2^32.
/// Unlike a modulo16 row pattern, these distinguish reusing payload rows16..31 from rows0..15.
fn source_word(row: usize, word: usize) -> u32 {
    assert!(word < 32, "fixture row stride must exceed its word count");
    ((row * 32 + word) as u32)
        .wrapping_mul(0x9e37_79b9)
        .wrapping_add(0x243f_6a88)
}

fn grid(rows: usize, cols: usize) -> (Vec<u32>, Vec<u16>) {
    let words: Vec<u32> = (0..rows)
        .flat_map(|row| (0..cols / PACK_FACTOR).map(move |word| source_word(row, word)))
        .collect();
    // Distinct, finite, nonzero bits for every (row, group) in this fixture.
    let scales: Vec<u16> = (0..rows)
        .flat_map(|row| {
            (0..cols / GROUP).map(move |group| 0x3f80 + (row * (cols / GROUP) + group) as u16)
        })
        .collect();
    (words, scales)
}

/// The source code of a matrix position, decoded from the fixture's own word table.
fn tile_code(words: &[u32], cols: usize, row: usize, column: usize) -> i8 {
    let word = words[row * (cols / PACK_FACTOR) + column / PACK_FACTOR];
    signed_of(((word >> ((column % PACK_FACTOR) * 4)) & 0x0f) as u8)
}

fn source_nibble(words: &[u32], cols: usize, row: usize, column: usize) -> u8 {
    let word = words[row * (cols / PACK_FACTOR) + column / PACK_FACTOR];
    ((word >> ((column % PACK_FACTOR) * 4)) & 0x0f) as u8
}

/// Where one tile position's nibble pair lands, written from `chain_ref.decode_sub`'s reshaping and
/// the kernel's `convert_sub` offsets rather than from the code under test.
fn packet_index(header: usize, tile: &ChainTile, n: usize, k: usize) -> usize {
    header + (n / 8 * (tile.cols / 8) + k / 8) * 32 + (n % 8) * 4 + (k % 8) / 2
}

/// Where one (group, output row) lands in the scale header: `[group][output_row]`.
fn header_slot(rows: usize, group: usize, n: usize) -> usize {
    (group * rows + n) * 2
}

/// The whole expected packet. Scale selection uses the source group of the actual column, so it is
/// correct for any start; only an aligned start is a legal request, and these geometry preconditions
/// are asserted here so a bug cannot hide by flooring a column start.
fn expected_packet(words: &[u32], scales: &[u16], cols: usize, tile: &ChainTile) -> Vec<u8> {
    assert_eq!(
        tile.first_col % GROUP,
        0,
        "fixture rule: tiles start on a source group"
    );
    assert_eq!(
        tile.rows % 16,
        0,
        "fixture rule: NB is a whole number of subtile pairs"
    );
    assert_eq!(
        tile.cols % GROUP,
        0,
        "fixture rule: SUB_K holds whole groups"
    );
    let groups = cols / GROUP;
    let header = tile.cols / GROUP * tile.rows * 2;
    let mut packet = vec![0u8; header + tile.rows * tile.cols / 2];
    for group in 0..tile.cols / GROUP {
        for n in 0..tile.rows {
            let bits =
                scales[(tile.first_row + n) * groups + (tile.first_col + group * GROUP) / GROUP];
            let at = header_slot(tile.rows, group, n);
            packet[at] = bits as u8;
            packet[at + 1] = (bits >> 8) as u8;
        }
    }
    for n in 0..tile.rows {
        for k in 0..tile.cols {
            // The packet's signed s4 nibble is the signed code's low nibble, i.e. source nibble ^ 8.
            let nibble = source_nibble(words, cols, tile.first_row + n, tile.first_col + k) ^ 8;
            let at = packet_index(header, tile, n, k);
            if k % 2 == 0 {
                packet[at] |= nibble;
            } else {
                packet[at] |= nibble << 4;
            }
        }
    }
    packet
}

/// Payload written by a packer that kept the source's row-major K order: same nibbles, wrong order.
fn naive_row_major(words: &[u32], cols: usize, tile: &ChainTile) -> Vec<u8> {
    let mut payload = vec![0u8; tile.rows * tile.cols / 2];
    for n in 0..tile.rows {
        for k in 0..tile.cols {
            let nibble = source_nibble(words, cols, tile.first_row + n, tile.first_col + k) ^ 8;
            let at = n * (tile.cols / 2) + k / 2;
            payload[at] |= if k % 2 == 0 { nibble } else { nibble << 4 };
        }
    }
    payload
}

/// Misinterpreting CT words with AWQ's lane-to-slot permutation, then emitting the target subtiles.
/// This is not an AWQ source image: AWQ packs output lanes, whereas CT packs consecutive K columns.
fn autoawq_payload(words: &[u32], cols: usize, tile: &ChainTile) -> Vec<u8> {
    const LANE_TO_SLOT: [usize; 8] = [0, 4, 1, 5, 2, 6, 3, 7];
    let mut payload = vec![0u8; tile.rows * tile.cols / 2];
    for n in 0..tile.rows {
        for lane in 0..tile.cols {
            let column = tile.first_col + lane / 8 * 8 + LANE_TO_SLOT[lane % 8];
            let nibble = source_nibble(words, cols, tile.first_row + n, column) ^ 8;
            let at = packet_index(0, tile, n, lane);
            payload[at] |= if lane % 2 == 0 { nibble } else { nibble << 4 };
        }
    }
    payload
}

/// Every nibble of a byte range, sorted: a multiset, so it says nothing about order.
fn sorted_nibbles(bytes: &[u8]) -> Vec<u8> {
    let mut nibbles: Vec<u8> = bytes
        .iter()
        .flat_map(|byte| [byte & 0x0f, byte >> 4])
        .collect();
    nibbles.sort_unstable();
    nibbles
}

fn source_nibbles(words: &[u32], cols: usize, tile: &ChainTile) -> Vec<u8> {
    let mut nibbles: Vec<u8> = (0..tile.rows)
        .flat_map(|n| (0..tile.cols).map(move |k| (n, k)))
        .map(|(n, k)| source_nibble(words, cols, tile.first_row + n, tile.first_col + k) ^ 8)
        .collect();
    nibbles.sort_unstable();
    nibbles
}

/// Rejection must be decided before any byte of caller storage is touched.
fn reject(tensor: &PackQuantizedTensor, tile: &ChainTile, len: usize, label: &str) {
    let mut packet = canary(len);
    assert!(
        tensor.write_chain_tile(tile, &mut packet).is_err(),
        "accepted {label}"
    );
    assert!(
        packet.iter().all(|byte| *byte == CANARY),
        "wrote bytes for {label}"
    );
}

#[test]
fn fixture_distinguishes_payload_only_sixteen_row_folding() {
    let (words, scales) = grid(ROWS, COLS);
    let selected = tile(0, 0, TALL_ROWS, TALL_COLS);
    let expected = expected_packet(&words, &scales, COLS, &selected);
    let header = TALL_COLS / GROUP * TALL_ROWS * 2;
    let mut folded = vec![0u8; expected.len()];
    folded[..header].copy_from_slice(&expected[..header]);
    // Keep the correct scales, but incorrectly reuse the first16 payload rows for the second16.
    for n in 0..TALL_ROWS {
        for k in 0..TALL_COLS {
            let nibble = source_nibble(&words, COLS, n % 16, k) ^ 8;
            let at = packet_index(header, &selected, n, k);
            folded[at] |= if k % 2 == 0 { nibble } else { nibble << 4 };
        }
    }
    assert_ne!(
        expected, folded,
        "fixture cannot expose a payload-only row-folding bug"
    );
}

#[test]
fn a_whole_32x96_tile_is_the_kernels_1728_byte_packet() -> Result<()> {
    let (words, scales) = grid(ROWS, COLS);
    let (_file, tensor) = tile_view(ROWS, COLS, GROUP, &words, &scales)?;
    let (_, source) = opened(ROWS, COLS, &words, &scales)?;
    let scales_at = &source.get("proj.weight_scale")?.span;
    let words_at = &source.get("proj.weight_packed")?.span;
    let selected = tile(0, 0, TALL_ROWS, TALL_COLS);
    assert_eq!(selected.encoded_bytes()?, TALL_BYTES);
    // Exact-sized caller storage on the stack; internal allocation bounds need implementation review.
    let mut packet = [CANARY; TALL_BYTES];
    assert_eq!(tensor.write_chain_tile(&selected, &mut packet)?, TALL_BYTES);
    assert_eq!(
        packet.as_slice(),
        expected_packet(&words, &scales, COLS, &selected).as_slice()
    );
    assert_eq!(tensor.descriptor().rows, ROWS);
    assert_eq!(tensor.descriptor().cols, COLS);
    // The header is `[group][output_row]` and holds the source scale's two bytes in the source's own
    // little-endian order, read back through the parsed entry span rather than through the codec.
    for (group, n) in [(0usize, 0usize), (0, 31), (1, 7), (2, 0), (2, 31)] {
        let mut raw_bits = [0u8; 2];
        scales_at.read_exact_at((n * (COLS / GROUP) + group) as u64 * 2, &mut raw_bits)?;
        let at = header_slot(TALL_ROWS, group, n);
        assert_eq!(
            &packet[at..at + 2],
            &raw_bits[..],
            "header slot group {group} row {n}"
        );
    }
    // One whole subtile straight from the source words: eight output rows by eight K columns, each
    // row's four bytes being that word's little-endian bytes with every byte XOR 0x88.
    let header = TALL_COLS / GROUP * TALL_ROWS * 2;
    for n in 0..8usize {
        let mut raw_word = [0u8; 4];
        words_at.read_exact_at((n * (COLS / PACK_FACTOR)) as u64 * 4, &mut raw_word)?;
        let expected: Vec<u8> = raw_word.iter().map(|byte| byte ^ 0x88).collect();
        let at = packet_index(header, &selected, n, 0);
        assert_eq!(&packet[at..at + 4], &expected[..], "subtile row {n}");
    }
    Ok(())
}

#[test]
fn packet_sizes_are_derived_and_checked_and_16x32_is_the_second_parameterization() -> Result<()> {
    let (words, scales) = grid(ROWS, COLS);
    let (_file, tensor) = tile_view(ROWS, COLS, GROUP, &words, &scales)?;
    let small = tile(0, 0, SMALL_ROWS, SMALL_COLS);
    assert_eq!(small.encoded_bytes()?, SMALL_BYTES);
    let mut packet = canary(SMALL_BYTES);
    assert_eq!(tensor.write_chain_tile(&small, &mut packet)?, SMALL_BYTES);
    assert_eq!(packet, expected_packet(&words, &scales, COLS, &small));
    // Every size is the header plus half a byte per code, derived from the extent, never a constant.
    let sizes: [(usize, usize, usize); 5] = [
        (16, 32, 288),
        (16, 64, 576),
        (16, 96, 864),
        (32, 96, 1728),
        (48, 192, 5184),
    ];
    for (rows, cols, bytes) in sizes {
        assert_eq!(
            tile(0, 0, rows, cols).encoded_bytes()?,
            bytes,
            "{rows}x{cols} size"
        );
    }
    // A selection that cannot form a whole packet has no size, and an absurd extent is a checked
    // rejection rather than an overflow or an allocation.
    let illegal: [(usize, usize); 8] = [
        (0, 32),
        (16, 0),
        (8, 32),
        (24, 32),
        (40, 96),
        (16, 16),
        (16, 48),
        (16, 80),
    ];
    for (rows, cols) in illegal {
        assert!(
            tile(0, 0, rows, cols).encoded_bytes().is_err(),
            "sized a {rows}x{cols} selection"
        );
    }
    assert!(tile(0, 0, usize::MAX, 32).encoded_bytes().is_err());
    assert!(tile(0, 0, 16, usize::MAX).encoded_bytes().is_err());
    // Aligned extents must reach checked byte-count arithmetic, not just fail a modulus check.
    assert!(tile(0, 0, usize::MAX & !15, 32).encoded_bytes().is_err());
    assert!(tile(0, 0, 16, usize::MAX & !31).encoded_bytes().is_err());
    Ok(())
}

#[test]
fn payload_is_nblock_major_subtiles_and_neither_row_major_nor_autoawq() -> Result<()> {
    let (words, scales) = grid(ROWS, COLS);
    let (_file, tensor) = tile_view(ROWS, COLS, GROUP, &words, &scales)?;
    let selected = tile(0, 0, SMALL_ROWS, SMALL_COLS);
    let mut packet = canary(SMALL_BYTES);
    tensor.write_chain_tile(&selected, &mut packet)?;
    let header = SMALL_COLS / GROUP * SMALL_ROWS * 2;
    assert_eq!(packet, expected_packet(&words, &scales, COLS, &selected));
    // The required payload order is neither a row-major K run nor the AutoAWQ lane permutation, even
    // though all three are permutations of the same nibbles.
    assert_ne!(
        &packet[header..],
        &naive_row_major(&words, COLS, &selected)[..],
        "kept row-major order"
    );
    assert_ne!(
        &packet[header..],
        &autoawq_payload(&words, COLS, &selected)[..],
        "used the AutoAWQ order"
    );
    // Both alternatives must retain every nibble; a malformed comparator is not an ordering test.
    let expected_multiset = sorted_nibbles(&packet[header..]);
    assert_eq!(
        sorted_nibbles(&naive_row_major(&words, COLS, &selected)),
        expected_multiset
    );
    assert_eq!(
        sorted_nibbles(&autoawq_payload(&words, COLS, &selected)),
        expected_multiset
    );
    // Lane `i = (n = i/8, k = i%8)`: byte 4 of a subtile is output row 1, so it differs from byte 0 -
    // the rows carry different nibble sequences, so an n/k collapse cannot reproduce this byte.
    assert_ne!(
        packet[header],
        packet[header + 4],
        "n and k lanes were collapsed"
    );
    // The last subtile row of the last 8-column K block is a different word again.
    let at = packet_index(header, &selected, SMALL_ROWS - 1, SMALL_COLS - 8);
    let word = words[(SMALL_ROWS - 1) * (COLS / PACK_FACTOR) + 3];
    assert_eq!(
        packet[at],
        word.to_le_bytes()[0] ^ 0x88,
        "last subtile row, last K block"
    );
    Ok(())
}

#[test]
fn a_hand_written_ascending_word_tile_has_the_signed_s4_bytes_predicted_by_hand() -> Result<()> {
    // Every source word here is 0x7654_3210: unsigned nibbles 0..7 in consecutive low-to-high slots,
    // which the source format reads as the signed codes -8..-1. The other fixture covers all16 codes.
    // The packet's nibble for code c is the
    // two's-complement low nibble of c, i.e. source nibble ^ 8, and a byte carries adjacent K columns
    // 0 and 1 in its low and high nibbles. So the first payload byte is 0x08 | 0x09 << 4 = 0x98, then
    // 0xba, 0xdc, 0xfe - the word's little-endian bytes XOR 0x88. All 64 payload groups are identical:
    // a prediction written out by hand, with no helper and no shared permutation involved.
    let words = vec![0x7654_3210u32; SMALL_ROWS * (SMALL_COLS / PACK_FACTOR)];
    let scales = vec![0x3f80u16; SMALL_ROWS];
    let (_file, tensor) = tile_view(SMALL_ROWS, SMALL_COLS, GROUP, &words, &scales)?;
    let mut packet = canary(SMALL_BYTES);
    let small = tile(0, 0, SMALL_ROWS, SMALL_COLS);
    assert_eq!(tensor.write_chain_tile(&small, &mut packet)?, SMALL_BYTES);
    let header = SMALL_COLS / GROUP * SMALL_ROWS * 2;
    // The header is 0x3f80 (a scale of exactly 1.0) once per (group, output row), little-endian.
    let expected_header = vec![0x80u8, 0x3f].repeat(header / 2);
    assert_eq!(
        &packet[..header],
        &expected_header[..],
        "scale header bytes"
    );
    for group in packet[header..].chunks(4) {
        assert_eq!(
            group,
            &[0x98u8, 0xba, 0xdc, 0xfe][..],
            "payload group {group:?}"
        );
    }
    Ok(())
}

#[test]
fn tiles_at_nonzero_rows_and_aligned_columns_are_their_own_source_bytes() -> Result<()> {
    let (words, scales) = grid(ROWS, COLS);
    let (_file, tensor) = tile_view(ROWS, COLS, GROUP, &words, &scales)?;
    // Row starts need no alignment; column starts must be group-aligned. Every case crosses an 8-row
    // subtile, an 8-column K block or a scale-group boundary inside the tile.
    let tiles: [(usize, usize, usize, usize); 6] = [
        (16, 64, TALL_ROWS, TALL_COLS),
        (10, 32, SMALL_ROWS, SMALL_COLS),
        (8, 160, SMALL_ROWS, SMALL_COLS),
        (0, 96, SMALL_ROWS, TALL_COLS),
        (16, 0, SMALL_ROWS, COLS),
        (1, 32, 16, 64),
    ];
    for (first_row, first_col, rows, cols) in tiles {
        let selected = tile(first_row, first_col, rows, cols);
        let mut packet = canary(selected.encoded_bytes()?);
        assert_eq!(
            tensor.write_chain_tile(&selected, &mut packet)?,
            packet.len()
        );
        assert_eq!(
            packet,
            expected_packet(&words, &scales, COLS, &selected),
            "tile at ({first_row}, {first_col}) {rows}x{cols}"
        );
    }
    // Same-size tiles at different starts are different bytes: a selection that ignored `first_row` or
    // `first_col`, or that clamped to a row-16 / column-0 tile, cannot pass.
    let mut origin = canary(SMALL_BYTES);
    let mut moved_row = canary(SMALL_BYTES);
    let mut moved_column = canary(SMALL_BYTES);
    tensor.write_chain_tile(&tile(0, 0, SMALL_ROWS, SMALL_COLS), &mut origin)?;
    tensor.write_chain_tile(&tile(16, 32, SMALL_ROWS, SMALL_COLS), &mut moved_row)?;
    tensor.write_chain_tile(&tile(0, 32, SMALL_ROWS, SMALL_COLS), &mut moved_column)?;
    assert_ne!(origin, moved_row, "row selection ignored");
    assert_ne!(origin, moved_column, "column selection ignored");
    assert_ne!(moved_row, moved_column);
    Ok(())
}

#[test]
fn all_sixteen_codes_and_adversarial_bf16_scale_bits_survive_byte_for_byte() -> Result<()> {
    // Signed and unsigned zero, the smallest subnormals, the largest finite values, ordinary values,
    // both infinities and quiet/signaling NaN payloads: one scale per row of a16x32 tile's header.
    const PATTERNS: [u16; 16] = [
        0x0000, 0x8000, 0x0001, 0x8001, 0x0003, 0x7f81, 0x7f7f, 0xff7f, 0x7f80, 0xff80, 0x7fc0,
        0x7fc1, 0xffc1, 0x3f80, 0x3f00, 0xff81,
    ];
    let (words, _) = grid(SMALL_ROWS, SMALL_COLS);
    let scales = PATTERNS.to_vec();
    let (_file, tensor) = tile_view(SMALL_ROWS, SMALL_COLS, GROUP, &words, &scales)?;
    let selected = tile(0, 0, SMALL_ROWS, SMALL_COLS);
    let mut packet = canary(SMALL_BYTES);
    assert_eq!(
        tensor.write_chain_tile(&selected, &mut packet)?,
        SMALL_BYTES
    );
    assert_eq!(
        packet,
        expected_packet(&words, &scales, SMALL_COLS, &selected)
    );
    let header = SMALL_COLS / GROUP * SMALL_ROWS * 2;
    // Every scale keeps its exact two source bytes, including signaling NaNs, and the
    // header is still two bytes per 32 K columns.
    for n in 0..SMALL_ROWS {
        let at = header_slot(SMALL_ROWS, 0, n);
        let observed = [packet[at], packet[at + 1]];
        let expected = [PATTERNS[n] as u8, (PATTERNS[n] >> 8) as u8];
        assert_eq!(observed, expected, "scale {:#06x}", PATTERNS[n]);
    }
    // The infinities and the three NaN payload patterns are still these literal bytes at their own
    // row's slot: quieting, sign-dropping or widening would change them.
    let inf_row = header_slot(SMALL_ROWS, 0, 8);
    assert_eq!(
        &packet[inf_row..inf_row + 2],
        &[0x80u8, 0x7f][..],
        "+inf scale"
    );
    let neg_inf_row = header_slot(SMALL_ROWS, 0, 9);
    assert_eq!(
        &packet[neg_inf_row..neg_inf_row + 2],
        &[0x80u8, 0xff][..],
        "-inf scale"
    );
    for (row, expected) in [
        (10usize, [0xc0u8, 0x7f]),
        (11, [0xc1, 0x7f]),
        (12, [0xc1, 0xff]),
    ] {
        let at = header_slot(SMALL_ROWS, 0, row);
        assert_eq!(
            &packet[at..at + 2],
            &expected[..],
            "NaN payload {expected:?} at row {row}"
        );
    }
    // Permutation-independent code check: the payload's nibble multiset is exactly the window's source
    // nibbles XOR 8, so clipping, requantizing or losing an extreme level would change it.
    let payload = sorted_nibbles(&packet[header..]);
    assert_eq!(payload, source_nibbles(&words, SMALL_COLS, &selected));
    for nibble in 0u8..16 {
        assert!(
            payload.contains(&nibble),
            "nibble {nibble:#02x} is missing from the packet"
        );
    }
    // The two extremes named: unsigned 0 is code -8 and unsigned 15 is code 7, so neither a midpoint
    // shift nor a clamp may move either one.
    assert_eq!((signed_of(0), signed_of(15)), (-8, 7));
    assert_eq!((nibble_of(-8), nibble_of(7)), (0, 15));
    Ok(())
}

#[test]
fn rejects_unaligned_columns_bad_extents_bad_ranges_and_short_buffers_before_writing() -> Result<()>
{
    let (words, scales) = grid(ROWS, COLS);
    let (_file, tensor) = tile_view(ROWS, COLS, GROUP, &words, &scales)?;
    let good = tile(16, 64, TALL_ROWS, TALL_COLS);
    assert_eq!(good.encoded_bytes()?, TALL_BYTES);
    // An unaligned K start cuts a real source group in half: the packet would need a scale for columns
    // whose group begins before the window. Rejected, never guessed, padded or resampled. Every case
    // is inside the declared shape, so alignment - not a range coincidence - causes each rejection.
    let unaligned: [usize; 8] = [8, 16, 24, 31, 40, 72, 104, COLS - SMALL_COLS - 8];
    for first_col in unaligned {
        reject(
            &tensor,
            &tile(0, first_col, SMALL_ROWS, SMALL_COLS),
            TALL_BYTES,
            &format!("column start {first_col}"),
        );
    }
    // Extents that are not positive, not whole subtile pairs or not whole groups.
    let extents: [(usize, usize); 7] = [
        (0, 96),
        (32, 0),
        (8, 96),
        (40, 96),
        (32, 16),
        (32, 48),
        (32, 80),
    ];
    for (rows, cols) in extents {
        reject(
            &tensor,
            &tile(0, 0, rows, cols),
            TALL_BYTES,
            &format!("{rows}x{cols}"),
        );
    }
    // Rows and columns outside the declared shape, including off-by-one and overflow arithmetic.
    let outside: [(usize, usize, usize, usize); 9] = [
        (ROWS, 0, SMALL_ROWS, SMALL_COLS),
        (ROWS - SMALL_ROWS + 1, 0, SMALL_ROWS, SMALL_COLS),
        (ROWS - SMALL_ROWS, COLS, SMALL_ROWS, SMALL_COLS),
        (0, COLS - SMALL_COLS + GROUP, SMALL_ROWS, SMALL_COLS),
        (
            ROWS - SMALL_ROWS,
            COLS - SMALL_COLS,
            SMALL_ROWS,
            SMALL_COLS + GROUP,
        ),
        (usize::MAX, 0, SMALL_ROWS, SMALL_COLS),
        (0, usize::MAX, SMALL_ROWS, SMALL_COLS),
        (usize::MAX - 8, 0, SMALL_ROWS, SMALL_COLS),
        (0, 0, usize::MAX, SMALL_COLS),
    ];
    for (first_row, first_col, rows, cols) in outside {
        let label = format!("{first_row},{first_col} {rows}x{cols} against a {ROWS}x{COLS} source");
        reject(
            &tensor,
            &tile(first_row, first_col, rows, cols),
            TALL_BYTES,
            &label,
        );
    }
    // A smaller buffer is a rejection decided before any write; an oversized buffer keeps its tail,
    // because caller-owned storage stays caller-owned.
    for short in [TALL_BYTES - 1, TALL_BYTES / 2, 1, 0] {
        reject(&tensor, &good, short, &format!("{short}-byte buffer"));
    }
    let mut roomy = canary(TALL_BYTES + 64);
    assert_eq!(tensor.write_chain_tile(&good, &mut roomy)?, TALL_BYTES);
    assert_eq!(
        &roomy[..TALL_BYTES],
        &expected_packet(&words, &scales, COLS, &good)[..]
    );
    assert!(
        roomy[TALL_BYTES..].iter().all(|byte| *byte == CANARY),
        "wrote past the packet"
    );
    // Two calls into one buffer agree exactly, so a reused packet is fully overwritten.
    let mut again = canary(TALL_BYTES);
    assert_eq!(tensor.write_chain_tile(&good, &mut again)?, TALL_BYTES);
    assert_eq!(again, roomy[..TALL_BYTES].to_vec());
    Ok(())
}

#[test]
fn refuses_a_source_whose_group_is_not_the_packet_scale_group() -> Result<()> {
    // The same columns at another declared group width are still a readable CT view - that is the
    // reader's contract - but a 32-column packet group would then share or splice source scales, so
    // repacking refuses rather than resampling a scale.
    let (words, _) = grid(ROWS, COLS);
    let tiles: [(usize, usize, usize, usize); 3] = [
        (0, 0, TALL_ROWS, TALL_COLS),
        (0, 0, SMALL_ROWS, SMALL_COLS),
        (16, 64, SMALL_ROWS, SMALL_COLS),
    ];
    for group in [16usize, 64, 96] {
        let scales: Vec<u16> = (0..ROWS)
            .flat_map(|row| (0..COLS / group).map(move |g| 0x3f80 + (row * 3 + g) as u16))
            .collect();
        let (_file, tensor) = tile_view(ROWS, COLS, group, &words, &scales)?;
        for (first_row, first_col, rows, cols) in tiles {
            let label = format!("group-{group} source at ({first_row}, {first_col}) {rows}x{cols}");
            reject(
                &tensor,
                &tile(first_row, first_col, rows, cols),
                TALL_BYTES,
                &label,
            );
        }
    }
    // Control: at group 32 the identical selections are accepted.
    let scales: Vec<u16> = (0..ROWS)
        .flat_map(|row| (0..COLS / GROUP).map(move |group| 0x3f80 + (row * 3 + group) as u16))
        .collect();
    let (_file, tensor) = tile_view(ROWS, COLS, GROUP, &words, &scales)?;
    let mut packet = canary(SMALL_BYTES);
    let small = tile(0, 0, SMALL_ROWS, SMALL_COLS);
    assert_eq!(tensor.write_chain_tile(&small, &mut packet)?, SMALL_BYTES);
    assert_eq!(packet, expected_packet(&words, &scales, COLS, &small));
    Ok(())
}

/// The maintained CPU reference for this packet layout, used only to confirm that a finite fixture
/// reconstructs the source tile. It is a decoder owned outside this crate, so it shares no index
/// arithmetic with the codec or with the expectations above, and it never quantizes anything.
fn chain_ref_reference(packet: &[u8], sub_k: usize, nb: usize) -> Result<Vec<f32>> {
    const SCRIPT: &str = "import sys
import numpy as np
sys.path.insert(0, sys.argv[1])
import chain_ref
sub_k, nb = int(sys.argv[2]), int(sys.argv[3])
raw = bytes.fromhex(open(sys.argv[4]).read().strip())
assert len(raw) == chain_ref.sub_bytes(sub_k, nb)
values = chain_ref.decode_sub(np.frombuffer(raw, np.uint8), sub_k, nb)
print(np.ascontiguousarray(values.T).view(np.uint32).tobytes().hex())
";
    let designs = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../designs/resident_forward");
    ensure!(
        designs.join("chain_ref.py").is_file(),
        "missing chain_ref.py at {}",
        designs.display()
    );
    let mut hex = NamedTempFile::new()?;
    for byte in packet {
        write!(hex, "{byte:02x}")?;
    }
    hex.flush()?;
    let output = std::process::Command::new("python3")
        .arg("-c")
        .arg(SCRIPT)
        .arg(designs)
        .arg(sub_k.to_string())
        .arg(nb.to_string())
        .arg(hex.path())
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .env("OMP_NUM_THREADS", "2")
        .env("OPENBLAS_NUM_THREADS", "2")
        .env("MKL_NUM_THREADS", "2")
        .output()
        .context("launch explicit CPU NumPy golden")?;
    ensure!(
        output.status.success(),
        "chain_ref failed ({}): {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = std::str::from_utf8(&output.stdout)?.trim();
    ensure!(
        stdout.len() == sub_k * nb * 8,
        "wrong oracle hex extent: {}",
        stdout.len()
    );
    let bytes: Vec<u8> = stdout
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| Ok(u8::from_str_radix(std::str::from_utf8(pair)?, 16)?))
        .collect::<Result<_>>()?;
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

#[test]
#[ignore = "explicit CPU NumPy reference: run this test with --ignored --exact"]
fn a_packed_tile_reconstructs_its_source_tile_through_the_pinned_chain_ref() -> Result<()> {
    // Finite scale patterns only, so exact bit equality is meaningful here; the non-finite patterns are
    // pinned as raw header bytes in another test, so a stray float conversion still fails there.
    let (words, scales) = grid(ROWS, COLS);
    let (_file, tensor) = tile_view(ROWS, COLS, GROUP, &words, &scales)?;
    let tiles: [(usize, usize, usize, usize); 2] = [
        (16, 64, TALL_ROWS, TALL_COLS),
        (10, 32, SMALL_ROWS, SMALL_COLS),
    ];
    for (first_row, first_col, rows, cols) in tiles {
        let selected = tile(first_row, first_col, rows, cols);
        let mut packet = canary(selected.encoded_bytes()?);
        assert_eq!(
            tensor.write_chain_tile(&selected, &mut packet)?,
            packet.len()
        );
        let values = chain_ref_reference(&packet, cols, rows)?;
        assert_eq!(
            values.len(),
            rows * cols,
            "chain_ref returned the wrong extent"
        );
        for n in 0..rows {
            for k in 0..cols {
                let code = tile_code(&words, COLS, first_row + n, first_col + k);
                // The source scale of this exact column, widened by moving its bits into an f32.
                let bits = scales[(first_row + n) * (COLS / GROUP) + (first_col + k) / GROUP];
                let expected = f32::from(code) * f32::from_bits(u32::from(bits) << 16);
                assert_eq!(
                    values[n * cols + k].to_bits(),
                    expected.to_bits(),
                    "chain_ref reconstruction at row {n} column {k}"
                );
            }
        }
    }
    Ok(())
}
