//! Compact graph-owned CT tile grids streamed through caller-owned packet scratch.
//! Original packed codes and raw BF16 scale bits are checked against a per-position formula,
//! literal anchors and the maintained packet encoder.

use anyhow::Result;
use npu_weights::source_native::ct::{
    ChainTile, PackQuantizedFormat, PackQuantizedTensor, WeightSections,
};
use npu_weights::source_native::plan::{WeightContract, WeightOp, WeightPlan};
use npu_weights::source_native::safetensors::SafetensorsFile;
use npu_weights::source_native::{Limits, ScalarType};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::io::Write;
use std::path::PathBuf;
use tempfile::NamedTempFile;

/// Nibbles per source I32 packed word.
const PACK_FACTOR: usize = 8;
/// The source group width and the packet scale group; the chain packet refuses any other.
const GROUP: usize = 32;
/// Fixture logical extents: 24 words and six scale groups per row.
const ROWS: usize = 48;
const COLS: usize = 192;
const GROUPS: usize = COLS / GROUP;
/// The selected region: an arbitrary valid row start, a group-aligned column start, and extents
/// that are whole multiples of both supported tile geometries.
const REGION_FIRST_ROW: usize = 5;
const REGION_FIRST_COL: usize = 32;
const REGION_ROWS: usize = 32;
const REGION_COLS: usize = 128;
/// The two packet geometries: 16x32 is 32 scale bytes + 256 codes, 32x96 is 192 + 1536.
const TILE_ROWS: usize = 16;
const TILE_COLS: usize = 32;
const TILE_BYTES: usize = 288;
const WIDE_TILE_ROWS: usize = 32;
const WIDE_TILE_COLS: usize = 96;
const WIDE_TILE_BYTES: usize = 1728;
/// Two row tiles x four column tiles of 288 bytes for the standard region.
const REGION_BYTES: usize = 2304;
/// The scratch the model runtime provides, and the sizes a whole packet can go out in.
const RUNTIME_SCRATCH_BYTES: usize = 64 * 1024;
/// The byte every unfilled caller buffer starts at, so "nothing was written" is observable.
const CANARY: u8 = 0x5a;

/// Explicit companion names for one CT op. Two fixtures name the same CT bytes with unrelated
/// names, so name selection is demonstrably the contract's and no prefix or suffix is guessed.
struct Names {
    packed: &'static str,
    shape: &'static str,
    scale: &'static str,
}
const STANDARD: Names = Names {
    packed: "proj.weight_packed",
    shape: "proj.weight_shape",
    scale: "proj.weight_scale",
};
const UNRELATED: Names = Names {
    packed: "ct.packed",
    shape: "ct.meta",
    scale: "ct.scales",
};

/// One region selection plus its tiling and explicit order, exactly as the contract spells it.
#[derive(Clone)]
struct Selection {
    first_row: usize,
    first_col: usize,
    rows: usize,
    cols: usize,
    tile_rows: usize,
    tile_cols: usize,
    order: &'static str,
}

fn standard(order: &'static str) -> Selection {
    Selection {
        first_row: REGION_FIRST_ROW,
        first_col: REGION_FIRST_COL,
        rows: REGION_ROWS,
        cols: REGION_COLS,
        tile_rows: TILE_ROWS,
        tile_cols: TILE_COLS,
        order,
    }
}

/// The wide 32x96 geometry over the region its single tile covers.
fn wide(order: &'static str) -> Selection {
    Selection {
        tile_rows: WIDE_TILE_ROWS,
        tile_cols: WIDE_TILE_COLS,
        cols: WIDE_TILE_COLS,
        ..standard(order)
    }
}

#[derive(Clone)]
struct Raw {
    name: &'static str,
    dtype: &'static str,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

fn raw(name: &'static str, dtype: &'static str, shape: &[usize], bytes: Vec<u8>) -> Raw {
    Raw {
        name,
        dtype,
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

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_bits().to_le_bytes())
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
            entry.name.to_string(),
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

/// Unique words for this fixture's row/word pairs: odd multiplication is invertible modulo 2^32, so
/// a packet that folded, shifted or reused payload rows or words cannot reproduce these bytes.
fn source_word(row: usize, word: usize) -> u32 {
    assert!(
        word < COLS / PACK_FACTOR,
        "fixture row stride must exceed its word count"
    );
    ((row * 32 + word) as u32)
        .wrapping_mul(0x9e37_79b9)
        .wrapping_add(0x243f_6a88)
}

/// Raw BF16 scale bits per (row, group). Four positions carry adversarial bit patterns - a
/// signaling NaN, -infinity, a subnormal and -0.0 - which must survive verbatim in the stream.
fn scale_bits(row: usize, group: usize) -> u16 {
    assert!(row < ROWS && group < GROUPS, "fixture scale table bounds");
    match (row, group) {
        (5, 2) => 0x7f81,
        (21, 3) => 0xff80,
        (5, 4) => 0x0001,
        (21, 1) => 0x8000,
        _ => 0x3f80 + (row * GROUPS + group) as u16,
    }
}

/// The fixture's scale table for a source that groups K by `width`. The adversarial patterns belong
/// to the 32-column fixture, so only that width reports them.
fn grid_scales(width: usize) -> Vec<u16> {
    let groups = COLS / width;
    (0..ROWS)
        .flat_map(|row| {
            (0..groups).map(move |index| {
                if width == GROUP {
                    scale_bits(row, index)
                } else {
                    0x3f80 + (row * groups + index) as u16
                }
            })
        })
        .collect()
}

fn grid_words() -> Vec<u32> {
    (0..ROWS)
        .flat_map(|row| (0..COLS / PACK_FACTOR).map(move |word| source_word(row, word)))
        .collect()
}

fn ct_raws(names: &Names, meta: &[i64], width: usize) -> Vec<Raw> {
    let scales = grid_scales(width);
    vec![
        raw(
            names.packed,
            "I32",
            &[ROWS, COLS / PACK_FACTOR],
            i32_bytes(&grid_words()),
        ),
        raw(names.shape, "I64", &[meta.len()], i64_bytes(meta)),
        raw(
            names.scale,
            "BF16",
            &[ROWS, COLS / width],
            bf16_bytes(&scales),
        ),
    ]
}

fn bf16_raws() -> Vec<Raw> {
    // Distinct BF16 regions that a tied configuration must not make equal.
    let embedding: Vec<u16> = (0..24).map(|index| 0x3f80 + index).collect();
    let head: Vec<u16> = (0..24).map(|index| 0x3f80 - index).collect();
    let norm: Vec<u16> = (0..16).map(|index| 0x3dcd + index).collect();
    vec![
        raw(
            "model.embed_tokens.weight",
            "BF16",
            &[6, 4],
            bf16_bytes(&embedding),
        ),
        raw("lm_head.weight", "BF16", &[6, 4], bf16_bytes(&head)),
        raw("model.norm.weight", "BF16", &[8, 2], bf16_bytes(&norm)),
    ]
}

/// The opened image plus the file it came from, kept alive together: spans must outlive the reader.
struct Source {
    _file: NamedTempFile,
    reader: SafetensorsFile,
}

impl Source {
    fn path(&self) -> PathBuf {
        self._file.path().to_path_buf()
    }
    fn bytes(&self) -> Vec<u8> {
        std::fs::read(self.path()).expect("fixture file")
    }
}

fn opened(entries: &[Raw]) -> Result<Source> {
    let file = image(entries)?;
    let reader = SafetensorsFile::open(file.path(), Limits::default())?;
    Ok(Source {
        _file: file,
        reader,
    })
}

/// Fixture A: standard companion names, BF16 regions first, logical shape metadata 48x192.
fn fixture() -> Result<Source> {
    let mut entries = bf16_raws();
    entries.extend(ct_raws(&STANDARD, &[ROWS as i64, COLS as i64], GROUP));
    opened(&entries)
}

/// Fixture B: the same CT bytes under unrelated names, with the CT sections laid out first, so the
/// relative span order differs from fixture A and no shared-prefix name is available to guess.
fn unrelated_fixture() -> Result<Source> {
    let mut entries = ct_raws(&UNRELATED, &[ROWS as i64, COLS as i64], GROUP);
    entries.extend(bf16_raws());
    opened(&entries)
}

/// An image whose `weight_shape` metadata disagrees with the contract's expected logical shape.
fn meta_fixture(meta: &[i64]) -> Result<Source> {
    let mut entries = bf16_raws();
    entries.extend(ct_raws(&STANDARD, meta, GROUP));
    opened(&entries)
}

/// The declared quantization configuration. `WeightPlan` binds this object by deep equality and
/// parses none of its members: the operation itself declares the codec, and publisher
/// authentication is a later, separate gate. The keys are the shape a compressed-tensors publisher
/// uses, and a 4-bit integer payload reports type `int`.
fn quantization_config() -> Value {
    json!({
        "config_groups": {
            "group_0": {
                "format": "pack-quantized",
                "targets": ["Linear"],
                "weights": {
                    "group_size": GROUP,
                    "num_bits": 4,
                    "strategy": "group",
                    "symmetric": true,
                    "type": "int",
                },
            },
        },
        "format": "pack-quantized",
        "group_size": GROUP,
        "num_bits": 4,
        "producer": {"name": "synthetic", "version": "0"},
        "symmetric": true,
        "type": "int",
    })
}

/// A configuration pair: what the contract requires and what the source reports, so a case can
/// hold the same value on BOTH sides and prove the contract rule rather than a mismatch.
fn config_pair(quantization: Option<Value>) -> (Value, Vec<u8>) {
    let mut config = serde_json::Map::new();
    config.insert("model_type".into(), json!("fixture"));
    config.insert("tie_word_embeddings".into(), json!(true));
    if let Some(quantization) = quantization {
        config.insert("quantization_config".into(), quantization);
    }
    let config = Value::Object(config);
    let source = serde_json::to_vec(&config).expect("source configuration");
    (config, source)
}

/// The valid pair, plus the declared configuration for the `Value`-based binding path.
fn declared_config() -> Value {
    config_pair(Some(quantization_config())).0
}

fn declared_config_bytes() -> Vec<u8> {
    config_pair(Some(quantization_config())).1
}

/// A complete contract root, so a schema case is rejected for the field under test and not for a
/// missing version, configuration, buffers or embedding.
fn contract_json(version: u32, config: Value, buffers: Value) -> Value {
    json!({
        "version": version,
        "config": config,
        "buffers": buffers,
        "embedding": {"tensor": "model.embed_tokens.weight", "shape": [6, 4]},
    })
}

fn ct_contract(buffers: Value) -> WeightContract {
    contract(2, declared_config(), buffers)
}

fn contract(version: u32, config: Value, buffers: Value) -> WeightContract {
    serde_json::from_value(contract_json(version, config, buffers)).expect("contract JSON")
}

fn contract_from_op(op: Value) -> WeightContract {
    ct_contract(json!({"k_proj": [op]}))
}

fn ct_op_json(names: &Names, selection: &Selection, shape: Value) -> Value {
    json!({
        "op": "ct_tiles",
        "weight_packed": names.packed,
        "weight_shape": names.shape,
        "weight_scale": names.scale,
        "shape": shape,
        "first_row": selection.first_row,
        "first_col": selection.first_col,
        "rows": selection.rows,
        "cols": selection.cols,
        "tile_rows": selection.tile_rows,
        "tile_cols": selection.tile_cols,
        "tile_order": selection.order,
    })
}

/// The CT op with the fixture's own logical shape, which is the only shape this op may declare.
fn ct_op(names: &Names, selection: &Selection) -> Value {
    ct_op_json(names, selection, json!([ROWS, COLS]))
}

fn standard_op() -> Value {
    ct_op(&STANDARD, &standard("row_major"))
}

fn rows_op(tensor: &str, shape: &[usize], first: usize, count: usize) -> Value {
    json!({"op": "rows", "tensor": tensor, "shape": shape, "first": first, "count": count})
}

fn buffers(op: Value) -> Value {
    json!({"k_proj": [op]})
}

fn buffers_named(pairs: &[(&str, Value)]) -> Value {
    let mut map = serde_json::Map::new();
    for (name, ops) in pairs {
        map.insert((*name).to_string(), ops.clone());
    }
    Value::Object(map)
}

fn expected(pairs: &[(&str, usize)]) -> BTreeMap<String, usize> {
    pairs
        .iter()
        .map(|(name, len)| ((*name).to_string(), *len))
        .collect()
}

fn bind(
    contract: &WeightContract,
    reader: &SafetensorsFile,
    expected: &BTreeMap<String, usize>,
) -> Result<WeightPlan> {
    WeightPlan::bind(contract, reader.tensors(), expected, &declared_config())
}

fn bind_json(
    contract: &WeightContract,
    reader: &SafetensorsFile,
    expected: &BTreeMap<String, usize>,
    config: &[u8],
) -> Result<WeightPlan> {
    WeightPlan::bind_config_json(contract, reader.tensors(), expected, config)
}

/// Streaming state: buffer-relative offsets, emitted bytes, a hard chunk bound, optional failure.
struct Sink {
    bytes: Vec<u8>,
    calls: usize,
    max_chunk: usize,
    fail_after: Option<usize>,
}

impl Sink {
    fn new(scratch_len: usize) -> Self {
        Self {
            bytes: Vec::new(),
            calls: 0,
            max_chunk: scratch_len / 2 * 2,
            fail_after: None,
        }
    }
    fn write(&mut self, offset: usize, chunk: &[u8]) -> Result<()> {
        assert_eq!(offset, self.bytes.len(), "buffer-relative offset");
        assert!(
            !chunk.is_empty() && chunk.len() <= self.max_chunk,
            "a {}-byte chunk exceeds the {}-byte scratch",
            chunk.len(),
            self.max_chunk
        );
        self.calls += 1;
        if self.fail_after.is_some_and(|limit| self.calls > limit) {
            anyhow::bail!("sink failed");
        }
        self.bytes.extend_from_slice(chunk);
        Ok(())
    }
}

fn write(plan: &WeightPlan, name: &str, scratch: &mut [u8], sink: &mut Sink) -> Result<()> {
    plan.write_buffer(name, scratch, |offset, chunk| sink.write(offset, chunk))
}

/// Streams one buffer and returns the emitted bytes and the sink call count.
fn stream(plan: &WeightPlan, name: &str, scratch_len: usize) -> Result<(Vec<u8>, usize)> {
    let mut scratch = vec![0u8; scratch_len];
    let mut sink = Sink::new(scratch_len);
    write(plan, name, &mut scratch, &mut sink)?;
    Ok((sink.bytes, sink.calls))
}

/// The qualified encoder, used as a second producer for the same tiles.
fn packets(view: &PackQuantizedTensor, tiles: &[ChainTile]) -> Result<Vec<Vec<u8>>> {
    tiles
        .iter()
        .map(|tile| {
            let mut packet = vec![CANARY; tile.encoded_bytes()?];
            let len = view.write_chain_tile(tile, &mut packet)?;
            packet.truncate(len);
            Ok(packet)
        })
        .collect()
}

fn ct_view(source: &Source, names: &Names) -> Result<PackQuantizedTensor> {
    let sections = WeightSections {
        weight_packed: source.reader.get(names.packed)?.clone(),
        weight_shape: source.reader.get(names.shape)?.clone(),
        weight_scale: source.reader.get(names.scale)?.clone(),
        // Neither optional companion is in these images, and the CT op schema has no field where a
        // caller could even name one, so nothing can be silently dropped or requantized.
        weight_zero_point: None,
        weight_g_idx: None,
    };
    PackQuantizedTensor::new(
        "k_proj",
        sections,
        PackQuantizedFormat::declared("group", 4, true, GROUP, ScalarType::Bf16)?,
    )
}

fn tile(first_row: usize, first_col: usize, rows: usize, cols: usize) -> ChainTile {
    ChainTile {
        first_row,
        first_col,
        rows,
        cols,
    }
}

/// Tile starts in the declared order: `row_major` runs the row tile outer and the column tile
/// inner, `column_major` the reverse. No order is ever inferred.
fn tile_starts(selection: &Selection) -> Vec<ChainTile> {
    let row_tiles = selection.rows / selection.tile_rows;
    let col_tiles = selection.cols / selection.tile_cols;
    let pairs: Vec<(usize, usize)> = match selection.order {
        "row_major" => (0..row_tiles)
            .flat_map(|row| (0..col_tiles).map(move |col| (row, col)))
            .collect(),
        "column_major" => (0..col_tiles)
            .flat_map(|col| (0..row_tiles).map(move |row| (row, col)))
            .collect(),
        other => unreachable!("fixture rule: unknown order {other}"),
    };
    pairs
        .into_iter()
        .map(|(row, col)| {
            tile(
                selection.first_row + row * selection.tile_rows,
                selection.first_col + col * selection.tile_cols,
                selection.tile_rows,
                selection.tile_cols,
            )
        })
        .collect()
}

fn reference_bytes(selection: &Selection) -> Vec<u8> {
    let mut expected = Vec::new();
    for tile in tile_starts(selection) {
        expected.extend_from_slice(&expected_tile(&tile));
    }
    expected
}

/// The whole expected packet for one tile, written from the kernel's documented layout -
/// `[cols/32][rows]` little-endian BF16 scale bits, then `[rows/8][cols/8]` 32-byte subtiles with
/// nibble lane `i = (n = i / 8, k = i % 8)`, low nibble first - and never from the plan or codec.
/// A source nibble `c` carries the signed code `c - 8`, so the packet's s4 nibble is exactly `c ^ 8`.
fn expected_tile(t: &ChainTile) -> Vec<u8> {
    assert_eq!(
        t.first_col % GROUP,
        0,
        "fixture rule: tiles start on a group"
    );
    assert_eq!(
        t.rows % 16,
        0,
        "fixture rule: NB is a whole subtile pair count"
    );
    assert_eq!(t.cols % GROUP, 0, "fixture rule: SUB_K holds whole groups");
    let header = t.cols / GROUP * t.rows * 2;
    let mut packet = vec![0u8; header + t.rows * t.cols / 2];
    for group in 0..t.cols / GROUP {
        for row in 0..t.rows {
            let bits = scale_bits(t.first_row + row, (t.first_col + group * GROUP) / GROUP);
            let at = (group * t.rows + row) * 2;
            packet[at] = bits as u8;
            packet[at + 1] = (bits >> 8) as u8;
        }
    }
    for row in 0..t.rows {
        for column in 0..t.cols {
            let word = source_word(t.first_row + row, (t.first_col + column) / PACK_FACTOR);
            let nibble = ((word >> (((t.first_col + column) % PACK_FACTOR) * 4)) & 0x0f) as u8 ^ 8;
            let at = header
                + ((row / 8) * (t.cols / 8) + column / 8) * 32
                + (row % 8) * 4
                + (column % 8) / 2;
            if column % 2 == 0 {
                packet[at] |= nibble;
            } else {
                packet[at] |= nibble << 4;
            }
        }
    }
    packet
}

/// Literal expected stream bytes for `[at, at + len)` of a tile concatenation, computed from the
/// fixture's own word and scale tables per absolute buffer offset. It never reads the source file,
/// so it is independent of container layout and of both producers under test.
fn expected_span(at: usize, len: usize, tiles: &[ChainTile]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(len);
    let mut base = 0usize;
    for t in tiles {
        let packet = expected_tile(t);
        for index in 0..packet.len() {
            let absolute = base + index;
            if absolute >= at && bytes.len() < len {
                bytes.push(packet[index]);
            }
        }
        base += packet.len();
    }
    assert_eq!(bytes.len(), len, "span past the end of the grid");
    bytes
}

/// Streams a contract's single CT buffer with the runtime's scratch and returns the bytes.
fn stream_ct(contract: &WeightContract, source: &Source, buffer: &str, total: usize) -> Vec<u8> {
    let plan =
        bind(contract, &source.reader, &expected(&[(buffer, total)])).expect("valid CT binding");
    let (bytes, _) = stream(&plan, buffer, RUNTIME_SCRATCH_BYTES).expect("stream");
    bytes
}

#[test]
fn a_ct_grid_is_one_weight_op_with_a_stable_json_roundtrip() -> Result<()> {
    let source = fixture()?;
    let contract = ct_contract(buffers(standard_op()));
    let ops = contract.buffers.get("k_proj").expect("k_proj buffer");
    // A whole 2304-byte, eight-tile grid stays ONE operation: metadata does not grow per tile, so
    // a full-model matrix keeps one operation instead of one per packet.
    assert_eq!(ops.len(), 1, "a tile grid is one compact operation");
    assert!(
        matches!(ops[0], WeightOp::CtTiles { .. }),
        "the grid operation is not a CT tile operation"
    );
    // The operation's own byte total comes from the tiles, not from a per-tile operation list.
    let total: usize = tile_starts(&standard("row_major"))
        .iter()
        .map(|tile| tile.encoded_bytes())
        .collect::<Result<Vec<_>>>()?
        .iter()
        .sum();
    assert_eq!(total, REGION_BYTES);
    assert_eq!(
        stream_ct(&contract, &source, "k_proj", total).len(),
        REGION_BYTES
    );

    // Roundtrip: nothing is skipped on serialization, so a re-emitted contract still carries every
    // selector explicitly and reparses to the same operation count.
    let emitted = serde_json::to_value(&contract)?;
    let op = &emitted["buffers"]["k_proj"][0];
    assert_eq!(op["op"], json!("ct_tiles"));
    assert_eq!(
        op.as_object().expect("op fields").len(),
        12,
        "every selector must survive serialization: {op}"
    );
    for key in [
        "weight_packed",
        "weight_shape",
        "weight_scale",
        "shape",
        "first_row",
        "first_col",
        "rows",
        "cols",
        "tile_rows",
        "tile_cols",
        "tile_order",
    ] {
        assert!(!op[key].is_null(), "roundtrip dropped {key}");
    }
    let reparsed: WeightContract = serde_json::from_value(emitted)?;
    assert_eq!(reparsed.buffers.get("k_proj").expect("k_proj").len(), 1);
    assert_eq!(reparsed.version, 2);

    // The embedded ABI path, including what a duplicate key does to it.
    let wrapped = serde_json::to_vec(&json!({"weight_inputs": contract}))?;
    assert!(WeightContract::from_metadata_json(&wrapped)
        .expect("metadata ABI")
        .is_some());
    assert!(WeightContract::from_metadata_json(br#"{}"#)
        .expect("no ABI")
        .is_none());
    // An ABI present but empty is a malformed contract, and a duplicated selector key is refused by
    // the same unique-key parsing that guards a source configuration.
    assert!(WeightContract::from_metadata_json(br#"{"weight_inputs":{}}"#).is_err());
    let mut duplicated = String::from_utf8(wrapped).expect("utf8");
    let once = "\"tile_order\":\"row_major\"";
    assert_eq!(duplicated.matches(once).count(), 1);
    duplicated = duplicated.replace(once, &format!("{once},{once}"));
    assert!(WeightContract::from_metadata_json(duplicated.as_bytes()).is_err());
    Ok(())
}

#[test]
fn a_ct_grid_streams_exact_packets_through_bounded_caller_scratch() -> Result<()> {
    let source = fixture()?;
    let before = source.bytes();
    let selection = standard("row_major");
    let tiles = tile_starts(&selection);
    assert_eq!(tiles.len(), 8);
    // The buffer's required length is exactly the sum of the tiles' own packet sizes.
    let total: usize = tiles
        .iter()
        .map(|tile| tile.encoded_bytes())
        .collect::<Result<Vec<_>>>()?
        .iter()
        .sum();
    assert_eq!(total, REGION_BYTES);
    assert_eq!(total, 8 * TILE_BYTES);

    let contract = ct_contract(buffers(ct_op(&STANDARD, &selection)));
    let plan = bind(&contract, &source.reader, &expected(&[("k_proj", total)]))?;
    // Dropping the reader proves the spans outlive it, which is what makes the plan lazy here; it
    // says nothing about whether some other read happened to occur at bind time.
    let path = source.path();
    drop(source.reader);

    // Include exact packet capacity and spare capacity that is not a multiple of packet size.
    // The suite bounds chunks without fixing the batching policy.
    let mut streams = Vec::new();
    for scratch_len in [RUNTIME_SCRATCH_BYTES, TILE_BYTES, 600] {
        let (bytes, calls) = stream(&plan, "k_proj", scratch_len)?;
        assert_eq!(bytes.len(), total, "{scratch_len}-byte scratch");
        assert!(
            calls >= total.div_ceil(scratch_len),
            "{scratch_len}-byte scratch"
        );
        assert_eq!(bytes, reference_bytes(&selection), "{scratch_len} scratch");
        streams.push(bytes);
    }
    assert!(streams.windows(2).all(|pair| pair[0] == pair[1]));
    let bytes = streams.remove(0);

    // Literal anchors from the fixture tables: tile 0's scale header is one group of 16 rows, so it
    // runs (5,1) 0x3f9f, (6,1) 0x3fa5, (7,1) 0x3fab. Its first payload byte is one whole little
    // endian source word's low byte, rebased as a pair of nibbles by ^ 0x88.
    assert_eq!(
        &bytes[0..8],
        [0x9f, 0x3f, 0xa5, 0x3f, 0xab, 0x3f, 0xb1, 0x3f]
    );
    assert_eq!(bytes[32], source_word(5, 4).to_le_bytes()[0] ^ 0x88);
    // Tile 1 starts at 288: row-major puts (5,2) - the signaling NaN 0x7f81 - first.
    assert_eq!(
        &bytes[288..296],
        [0x81, 0x7f, 0xa6, 0x3f, 0xac, 0x3f, 0xb2, 0x3f]
    );
    // The adversarial scale bits land at their own tile starts: subnormal 0x0001 at 864, -0.0
    // 0x8000 at 1152, -infinity 0xff80 at 1728, and tile 7's plain bits at 2016.
    assert_eq!(&bytes[576..580], [0xa1, 0x3f, 0xa7, 0x3f]);
    assert_eq!(&bytes[864..866], [0x01, 0x00]);
    assert_eq!(&bytes[1152..1154], [0x00, 0x80]);
    assert_eq!(&bytes[1728..1730], [0x80, 0xff]);
    assert_eq!(&bytes[2016..2018], [0x02, 0x40]);
    // The whole stream, byte for byte, is the fixture's words and scales in tile order.
    assert_eq!(bytes, expected_span(0, total, &tiles));

    // The qualified encoder produces the same eight packets in the same order.
    let source = fixture()?;
    let view = ct_view(&source, &STANDARD)?;
    let encoded: Vec<u8> = packets(&view, &tiles)?.into_iter().flatten().collect();
    assert_eq!(encoded.len(), REGION_BYTES);
    assert_eq!(encoded, reference_bytes(&selection));
    assert_eq!(std::fs::read(path)?, before, "the source stays immutable");
    Ok(())
}

#[test]
fn ct_companions_are_selected_by_explicit_names_with_a_different_file_layout() -> Result<()> {
    let source = unrelated_fixture()?;
    let selection = standard("row_major");
    // The three companions are named one by one; none shares a prefix with the others here, so a
    // suffix guess such as `{prefix}.weight_packed` cannot resolve any of them.
    let contract = ct_contract(buffers(ct_op(&UNRELATED, &selection)));
    let plan = bind(
        &contract,
        &source.reader,
        &expected(&[("k_proj", REGION_BYTES)]),
    )?;
    let packed = source.reader.get(UNRELATED.packed)?;
    let meta = source.reader.get(UNRELATED.shape)?;
    let scales = source.reader.get(UNRELATED.scale)?;
    assert_eq!(packed.dtype, ScalarType::I32);
    assert_eq!(meta.dtype, ScalarType::I64);
    assert_eq!(scales.dtype, ScalarType::Bf16);
    assert_eq!(meta.span.len(), 16);
    // Different layout from fixture A: the packed section starts before every BF16 region here.
    let embedding = source.reader.get("model.embed_tokens.weight")?;
    let head = source.reader.get("lm_head.weight")?;
    assert!(packed.span.offset() < scales.span.offset());
    assert!(scales.span.offset() < embedding.span.offset());
    assert!(meta.span.offset() > packed.span.offset());
    assert!(meta.span.offset() < scales.span.offset());
    assert_ne!(embedding.span.offset(), head.span.offset());

    let (bytes, _) = stream(&plan, "k_proj", RUNTIME_SCRATCH_BYTES)?;
    assert_eq!(bytes, reference_bytes(&selection));
    // Naming a companion that does not exist is a binding failure, never a fallback to a similar
    // name: this image has no `proj.*` tensor at all.
    let mut op = ct_op(&UNRELATED, &selection);
    op["weight_scale"] = json!(STANDARD.scale);
    let missing = ct_contract(buffers(op));
    assert!(bind(
        &missing,
        &source.reader,
        &expected(&[("k_proj", REGION_BYTES)])
    )
    .is_err());
    Ok(())
}

#[test]
fn tile_orders_are_distinct_and_boundary_tiles_use_their_own_source_bytes() -> Result<()> {
    let source = fixture().expect("fixture");
    let row_major = standard("row_major");
    let column_major = standard("column_major");
    // Stream position 288 is tile (row 0, column 1) for row_major and tile (row 1, column 0) for
    // column_major, and the two tiles' scale headers begin with different literal bits.
    let cases = [
        (&row_major, [0x81u8, 0x7f], [0xa1, 0x3f], [0x01, 0x00]),
        (&column_major, [0x00, 0x80], [0x81, 0x7f], [0x00, 0x40]),
    ];
    let mut streams = Vec::new();
    for (selection, at288, at576, at864) in cases {
        let total = reference_bytes(selection).len();
        let contract = ct_contract(buffers(ct_op(&STANDARD, selection)));
        let bytes = stream_ct(&contract, &source, "k_proj", total);
        assert_eq!(bytes, reference_bytes(selection));
        assert_eq!(bytes.len(), REGION_BYTES);
        assert_eq!(&bytes[0..2], &[0x9f, 0x3f]);
        assert_eq!(&bytes[288..290], &at288[..]);
        assert_eq!(&bytes[576..578], &at576[..]);
        assert_eq!(&bytes[864..866], &at864[..]);
        // The final tile is (row 1, column 3) for BOTH orders, so the stream ends with the same
        // payload bytes - not a scale header, which leads each packet.
        assert_eq!(&bytes[2302..2304], &[0x89, 0x54]);
        assert_eq!(&bytes[2016..2018], &[0x02, 0x40]);
        streams.push(bytes);
    }
    assert_ne!(
        streams[0], streams[1],
        "the two grid orders emitted the same concatenation"
    );
    assert_eq!(
        sorted_nibbles(&streams[0]),
        sorted_nibbles(&streams[1]),
        "the orders should differ only in tile order"
    );

    // Boundary rows and columns: the region starts at row 5 (an arbitrary valid row start) and
    // ends at column 159, so no tile may borrow a scale group or a word from outside it. Shifting
    // the selection by one group is a different byte string of the same length.
    let mut shifted = standard("row_major");
    shifted.first_col = 64;
    let shifted_total = reference_bytes(&shifted).len();
    assert_eq!(shifted_total, REGION_BYTES);
    let shifted_contract = ct_contract(buffers(ct_op(&STANDARD, &shifted)));
    let shifted_bytes = stream_ct(&shifted_contract, &source, "k_proj", shifted_total);
    assert_eq!(shifted_bytes, reference_bytes(&shifted));
    assert_ne!(shifted_bytes, streams[0]);

    // Full coverage of the same matrix is exactly the packed payload plus the scale payload.
    let full = Selection {
        first_row: 0,
        first_col: 0,
        rows: ROWS,
        cols: COLS,
        tile_rows: TILE_ROWS,
        tile_cols: TILE_COLS,
        order: "row_major",
    };
    let full_total = reference_bytes(&full).len();
    assert_eq!(full_total, ROWS * COLS / 2 + ROWS * GROUPS * 2);
    let full_bytes = stream_ct(
        &ct_contract(buffers(ct_op(&STANDARD, &full))),
        &source,
        "k_proj",
        full_total,
    );
    assert_eq!(full_bytes, reference_bytes(&full));
    assert_eq!(
        full_bytes.len(),
        ROWS * COLS / PACK_FACTOR * 4 + ROWS * GROUPS * 2
    );
    Ok(())
}

#[test]
fn both_tile_geometries_stream_their_own_packet_sizes() -> Result<()> {
    let source = fixture().expect("fixture");
    let small = standard("row_major");
    let large = wide("column_major");
    let contract = ct_contract(buffers_named(&[
        ("small_grid", json!([ct_op(&STANDARD, &small)])),
        ("wide_grid", json!([ct_op(&STANDARD, &large)])),
    ]));
    assert_eq!(contract.buffers.len(), 2);
    let plan = bind(
        &contract,
        &source.reader,
        &expected(&[("small_grid", REGION_BYTES), ("wide_grid", WIDE_TILE_BYTES)]),
    )
    .expect("valid bindings");

    let tiles = tile_starts(&large);
    assert_eq!(tiles.len(), 1);
    assert_eq!(tiles[0].encoded_bytes().expect("legal"), WIDE_TILE_BYTES);
    let view = ct_view(&source, &STANDARD).expect("view");
    let encoded = packets(&view, &tiles).expect("encode");
    assert_eq!(encoded[0].len(), WIDE_TILE_BYTES);

    let (small_bytes, _) = stream(&plan, "small_grid", RUNTIME_SCRATCH_BYTES).expect("small");
    let (wide_bytes, _) = stream(&plan, "wide_grid", WIDE_TILE_BYTES).expect("wide");
    assert_eq!(small_bytes, reference_bytes(&small));
    assert_eq!(wide_bytes, reference_bytes(&large));
    assert_eq!(wide_bytes, encoded[0]);
    // A 32x96 header holds three groups x 32 rows of raw bits, so group 2 row 16 is source row 21,
    // group 3: -infinity 0xff80 at byte 160.
    assert_eq!(&wide_bytes[0..4], &[0x9f, 0x3f, 0xa5, 0x3f]);
    assert_eq!(&wide_bytes[160..162], &[0x80, 0xff]);
    assert_ne!(small_bytes.len(), wide_bytes.len());
    assert_ne!(&small_bytes[0..64], &wide_bytes[0..64]);
    Ok(())
}

#[test]
fn one_buffer_mixes_rows_a_ct_grid_zero_and_bf16_head_rows() -> Result<()> {
    let source = fixture()?;
    let before = source.bytes();
    // norm rows 2..5 (16 bytes), a 16x64 CT grid (2 x 288), six zero bytes, head rows 1..2 (16).
    let grid = Selection {
        rows: 16,
        cols: 64,
        ..standard("row_major")
    };
    let grid_bytes = reference_bytes(&grid).len();
    assert_eq!(grid_bytes, 2 * TILE_BYTES);
    let ops = json!([
        rows_op("model.norm.weight", &[8, 2], 2, 4),
        ct_op(&STANDARD, &grid),
        {"op": "zero", "bytes": 6},
        rows_op("lm_head.weight", &[6, 4], 1, 2),
    ]);
    let total = 16 + grid_bytes + 6 + 16;
    let contract = ct_contract(json!({"weights": ops}));
    assert_eq!(contract.buffers["weights"].len(), 4);
    let plan = bind(&contract, &source.reader, &expected(&[("weights", total)])).expect("mixed");
    let (bytes, _) = stream(&plan, "weights", TILE_BYTES).expect("mixed stream");
    assert_eq!(bytes.len(), total);
    // Norm elements 4..11 are the fixture's own BF16 bits, in declaration order.
    assert_eq!(
        &bytes[0..16],
        [
            0xd1, 0x3d, 0xd2, 0x3d, 0xd3, 0x3d, 0xd4, 0x3d, 0xd5, 0x3d, 0xd6, 0x3d, 0xd7, 0x3d,
            0xd8, 0x3d
        ]
    );
    assert_eq!(
        &bytes[16..16 + grid_bytes],
        reference_bytes(&grid).as_slice()
    );
    assert_eq!(&bytes[16 + grid_bytes..16 + grid_bytes + 6], &[0; 6]);
    // Head elements 4..11 follow the zeros, still raw BF16.
    assert_eq!(
        &bytes[16 + grid_bytes + 6..],
        [
            0x7c, 0x3f, 0x7b, 0x3f, 0x7a, 0x3f, 0x79, 0x3f, 0x78, 0x3f, 0x77, 0x3f, 0x76, 0x3f,
            0x75, 0x3f
        ]
    );

    // A tied configuration (`tie_word_embeddings` is in both configurations above) does not make
    // the BF16 regions equal or aliased, and building the CT plan left both readable as BF16.
    let embedding = plan.embedding();
    assert_eq!(embedding.dtype, ScalarType::Bf16);
    assert_eq!(embedding.shape, [6, 4]);
    let mut embed = vec![0u8; 48];
    embedding.span.read_exact_at(0, &mut embed)?;
    let mut head = vec![0u8; 48];
    source
        .reader
        .get("lm_head.weight")?
        .span
        .read_exact_at(0, &mut head)?;
    assert_ne!(embed, head, "a tied name is not payload equality");
    assert_eq!(
        &embed[0..4],
        &[0x80, 0x3f, 0x81, 0x3f],
        "embedding stays BF16"
    );
    assert_eq!(&head[0..4], &[0x80, 0x3f, 0x7f, 0x3f], "head stays BF16");
    assert_eq!(&source.bytes(), &before, "the source stays immutable");
    Ok(())
}

#[test]
fn scratch_must_hold_the_largest_packet_before_any_byte_is_written() -> Result<()> {
    let source = fixture()?;
    let small = standard("row_major");
    let large = wide("row_major");
    // The exact largest packet is accepted: 288 bytes for a 16x32 grid.
    let one = ct_contract(buffers(ct_op(&STANDARD, &small)));
    let plan = bind(&one, &source.reader, &expected(&[("k_proj", REGION_BYTES)]))?;
    for scratch_len in [TILE_BYTES, 600, RUNTIME_SCRATCH_BYTES] {
        let (bytes, _) = stream(&plan, "k_proj", scratch_len)?;
        assert_eq!(bytes.len(), REGION_BYTES, "{scratch_len}-byte scratch");
    }
    // One byte short of a packet, four bytes, and no scratch at all: refused, with nothing emitted.
    for scratch_len in [TILE_BYTES - 1, 4, 1] {
        let mut scratch = vec![0u8; scratch_len];
        let mut sink = Sink::new(scratch_len);
        assert!(
            write(&plan, "k_proj", &mut scratch, &mut sink).is_err(),
            "accepted a {scratch_len}-byte scratch for a 288-byte packet"
        );
        assert_eq!(
            sink.calls, 0,
            "wrote bytes for a {scratch_len}-byte scratch"
        );
        assert!(sink.bytes.is_empty());
    }

    // Preflight covers every operation of the whole buffer: the leading `rows` region and the
    // trailing `zero` region would each fit a 288-byte scratch, but the third operation's 1728-byte
    // packet does not, so the entire buffer is refused before a single sink call.
    let mixed = ct_contract(json!({"k_proj": [
        rows_op("model.norm.weight", &[8, 2], 0, 1),
        ct_op(&STANDARD, &small),
        ct_op(&STANDARD, &large),
        {"op": "zero", "bytes": 4},
    ]}));
    let total = 4 + REGION_BYTES + WIDE_TILE_BYTES + 4;
    let plan = bind(&mixed, &source.reader, &expected(&[("k_proj", total)]))?;
    for scratch_len in [TILE_BYTES, 600] {
        let mut scratch = vec![0u8; scratch_len];
        let mut sink = Sink::new(scratch_len);
        assert!(write(&plan, "k_proj", &mut scratch, &mut sink).is_err());
        assert_eq!(
            sink.calls, 0,
            "wrote bytes with a {scratch_len}-byte scratch"
        );
    }
    // The largest packet fits; the leading BF16 row and trailing zeros remain in declaration order.
    let (bytes, _) = stream(&plan, "k_proj", WIDE_TILE_BYTES)?;
    assert_eq!(bytes.len(), total);
    assert_eq!(&bytes[0..4], &[0xcd, 0x3d, 0xce, 0x3d]);
    assert_eq!(&bytes[total - 4..], &[0; 4]);
    assert_eq!(
        &bytes[4..4 + REGION_BYTES],
        reference_bytes(&small).as_slice()
    );

    // An unknown buffer name is refused without writing.
    let mut scratch = vec![0u8; RUNTIME_SCRATCH_BYTES];
    let mut sink = Sink::new(RUNTIME_SCRATCH_BYTES);
    assert!(write(&plan, "absent", &mut scratch, &mut sink).is_err());
    assert_eq!(sink.calls, 0);
    Ok(())
}

#[test]
fn version_one_rows_and_zero_keep_their_existing_semantics() -> Result<()> {
    let source = fixture().expect("source");
    // A v1 contract needs no quantization configuration and keeps its two-byte scratch minimum and
    // its whole-BF16 zero extents.
    let ops = json!([
        rows_op("model.norm.weight", &[8, 2], 6, 2),
        {"op": "zero", "bytes": 4},
    ]);
    let v1 = contract(1, json!({"model_type": "fixture"}), json!({"out": ops}));
    let plan = bind(&v1, &source.reader, &expected(&[("out", 12)])).expect("v1 binding");
    // A three-byte scratch still streams in two-byte chunks; the odd byte is unused, not emitted.
    let mut scratch = [0u8; 3];
    let mut sink = Sink::new(3);
    write(&plan, "out", &mut scratch, &mut sink).expect("three-byte scratch streams");
    assert_eq!(sink.max_chunk, 2);
    assert_eq!(sink.bytes.len(), 12);
    assert_eq!(
        sink.bytes,
        [0xd9, 0x3d, 0xda, 0x3d, 0xdb, 0x3d, 0xdc, 0x3d, 0, 0, 0, 0]
    );
    let mut scratch = [0u8; 1];
    let mut sink = Sink::new(1);
    assert!(write(&plan, "out", &mut scratch, &mut sink).is_err());
    assert_eq!(sink.calls, 0);
    // An odd zero extent is still refused for v1.
    let odd = contract(
        1,
        json!({"model_type": "fixture"}),
        json!({"out": [{"op": "zero", "bytes": 3}]}),
    );
    assert!(bind(&odd, &source.reader, &expected(&[("out", 3)])).is_err());

    // Version handling: v1 admits only v1 operations, v2 admits both kinds, an unknown version is
    // refused for a contract that also carries a CT grid.
    let grid = json!([ct_op(&STANDARD, &standard("row_major"))]);
    let at_one = contract(1, declared_config(), json!({"k_proj": grid}));
    assert!(bind(
        &at_one,
        &source.reader,
        &expected(&[("k_proj", REGION_BYTES)])
    )
    .is_err());
    let at_three = contract(3, declared_config(), json!({"k_proj": grid.clone()}));
    assert!(bind(
        &at_three,
        &source.reader,
        &expected(&[("k_proj", REGION_BYTES)])
    )
    .is_err());
    for version in [0, 4, u32::MAX] {
        let unknown = contract(version, declared_config(), json!({"k_proj": grid.clone()}));
        assert!(bind(
            &unknown,
            &source.reader,
            &expected(&[("k_proj", REGION_BYTES)])
        )
        .is_err());
    }
    // v2 binds the same v1 operations with the same required length.
    let v2 = contract(
        2,
        declared_config(),
        json!({"out": [rows_op("model.norm.weight", &[8, 2], 6, 2), {"op": "zero", "bytes": 4}]}),
    );
    let plan = bind(&v2, &source.reader, &expected(&[("out", 12)])).expect("v2 v1 operations");
    let (bytes, _) = stream(&plan, "out", 2).expect("two-byte scratch");
    assert_eq!(bytes.len(), 12);

    // Schema version 2 alone does not imply a quantized source or a CT operation.
    let bf16_only = contract(
        2,
        json!({"model_type": "fixture"}),
        json!({"out": [rows_op("model.norm.weight", &[8, 2], 6, 2), {"op": "zero", "bytes": 4}]}),
    );
    let plan = bind_json(
        &bf16_only,
        &source.reader,
        &expected(&[("out", 12)]),
        br#"{"model_type":"fixture"}"#,
    )
    .expect("BF16-only v2 does not require quantization_config");
    let (without_quantization, _) = stream(&plan, "out", 2)?;
    assert_eq!(without_quantization, bytes);
    Ok(())
}

#[test]
fn a_ct_contract_requires_a_declared_quantization_config_object() -> Result<()> {
    let source = fixture()?;
    // The rule is about the CONTRACT's declared object, so each of these cases puts the same value
    // on both sides: only the rule can reject them, not a mismatch against the source.
    for (label, quantization) in [
        ("absent", None),
        ("empty", Some(json!({}))),
        ("a string", Some(json!("group"))),
        ("null", Some(json!(null))),
        ("an array", Some(json!([quantization_config()]))),
    ] {
        let (config, source_config) = config_pair(quantization.clone());
        let contract = contract(2, config, buffers(standard_op()));
        assert!(
            bind_json(
                &contract,
                &source.reader,
                &expected(&[("k_proj", REGION_BYTES)]),
                &source_config
            )
            .is_err(),
            "accepted a declared quantization config that is {label}"
        );
    }
    // A v1 contract's requirements are untouched by this rule: no quantization config, still binds.
    let (config, source_config) = config_pair(None);
    let v1 = serde_json::from_value::<WeightContract>(contract_json(
        1,
        config,
        json!({"out": [{"op": "zero", "bytes": 2}]}),
    ))?;
    assert!(bind_json(
        &v1,
        &source.reader,
        &expected(&[("out", 2)]),
        &source_config
    )
    .is_ok());

    // Deep equality, three levels into the declared object, and a missing nested member.
    let mut nested = declared_config();
    nested["quantization_config"]["config_groups"]["group_0"]["weights"]["group_size"] = json!(64);
    assert!(bind(
        &contract(2, nested, buffers(standard_op())),
        &source.reader,
        &expected(&[("k_proj", REGION_BYTES)])
    )
    .is_err());
    let mut missing_key = declared_config();
    missing_key["quantization_config"]
        .as_object_mut()
        .expect("object")
        .remove("producer");
    assert!(bind(
        &contract(2, missing_key, buffers(standard_op())),
        &source.reader,
        &expected(&[("k_proj", REGION_BYTES)])
    )
    .is_err());

    // Valid control on the same two paths, so the rejections above are not fixture artifacts.
    let valid = ct_contract(buffers(standard_op()));
    let bytes = stream_ct(&valid, &source, "k_proj", REGION_BYTES);
    assert_eq!(bytes, reference_bytes(&standard("row_major")));
    let plan = bind_json(
        &valid,
        &source.reader,
        &expected(&[("k_proj", REGION_BYTES)]),
        &declared_config_bytes(),
    )
    .expect("byte configuration");
    let (again, _) = stream(&plan, "k_proj", TILE_BYTES)?;
    assert_eq!(again, bytes);
    // A source configuration that is not an object at all, and one with duplicate keys, are
    // refused by the existing unique-key parsing of the source configuration bytes.
    assert!(bind_json(
        &valid,
        &source.reader,
        &expected(&[("k_proj", REGION_BYTES)]),
        br#""group""#
    )
    .is_err());
    let duplicated = format!(
        "{{\"model_type\":\"fixture\",\"tie_word_embeddings\":true,\"quantization_config\":{},\"quantization_config\":{}}}",
        serde_json::to_string(&quantization_config())?,
        serde_json::to_string(&quantization_config())?,
    );
    assert!(bind_json(
        &valid,
        &source.reader,
        &expected(&[("k_proj", REGION_BYTES)]),
        duplicated.as_bytes()
    )
    .is_err());
    Ok(())
}

#[test]
fn invalid_ct_sources_are_rejected_before_any_write() -> Result<()> {
    let selection = standard("row_major");
    let names = &STANDARD;

    // A self-consistent 64-column-group source: the packet groups K by 32, so its scales would have
    // to be shared or spliced. The declared group width stays a plain configuration member here;
    // `WeightPlan` binds the object, the codec is the operation's own declaration.
    let mut group64 = declared_config();
    group64["quantization_config"]["group_size"] = json!(64);
    group64["quantization_config"]["config_groups"]["group_0"]["weights"]["group_size"] = json!(64);
    let source_config = serde_json::to_vec(&group64)?;
    let mut entries = bf16_raws();
    entries.extend(ct_raws(names, &[ROWS as i64, COLS as i64], 64));
    let source = opened(&entries)?;
    let contract = contract(2, group64, buffers(ct_op(names, &selection)));
    assert!(bind_json(
        &contract,
        &source.reader,
        &expected(&[("k_proj", REGION_BYTES)]),
        &source_config,
    )
    .is_err());
    // Same geometry, declared as the 32-group the source does not have.
    let inconsistent = ct_contract(buffers(ct_op(names, &selection)));
    assert!(bind(
        &inconsistent,
        &source.reader,
        &expected(&[("k_proj", REGION_BYTES)])
    )
    .is_err());
    // Valid control on this builder path, so no rejection below is a fixture artifact.
    let control = ct_contract(buffers(ct_op(names, &standard("row_major"))));
    assert_eq!(
        stream_ct(&control, &fixture()?, "k_proj", REGION_BYTES).len(),
        REGION_BYTES
    );

    // One mutated companion per case, sized so the container parser admits it and the plan is the
    // layer that refuses it.
    let words = i32_bytes(&grid_words());
    let scales = bf16_bytes(&grid_scales(GROUP));
    let meta = i64_bytes(&[ROWS as i64, COLS as i64]);
    let packed_shape = [ROWS, COLS / PACK_FACTOR];
    let cases: Vec<(&str, Vec<Raw>)> = vec![
        (
            "f32 scales",
            vec![
                raw(names.packed, "I32", &packed_shape, words.clone()),
                raw(names.shape, "I64", &[2], meta.clone()),
                raw(
                    names.scale,
                    "F32",
                    &[ROWS, GROUPS],
                    f32_bytes(&vec![1.0; ROWS * GROUPS]),
                ),
            ],
        ),
        (
            "rank-1 scales",
            vec![
                raw(names.packed, "I32", &packed_shape, words.clone()),
                raw(names.shape, "I64", &[2], meta.clone()),
                raw(names.scale, "BF16", &[ROWS * GROUPS], scales.clone()),
            ],
        ),
        (
            "scales for a narrower matrix",
            vec![
                raw(names.packed, "I32", &packed_shape, words.clone()),
                raw(names.shape, "I64", &[2], meta.clone()),
                raw(
                    names.scale,
                    "BF16",
                    &[ROWS, GROUPS - 1],
                    bf16_bytes(&grid_scales(GROUP)[..ROWS * (GROUPS - 1)]),
                ),
            ],
        ),
        (
            "i64 packed words",
            vec![
                raw(
                    names.packed,
                    "I64",
                    &packed_shape,
                    i64_bytes(&vec![7; ROWS * COLS / PACK_FACTOR]),
                ),
                raw(names.shape, "I64", &[2], meta.clone()),
                raw(names.scale, "BF16", &[ROWS, GROUPS], scales.clone()),
            ],
        ),
        (
            "rank-1 packed words",
            vec![
                raw(
                    names.packed,
                    "I32",
                    &[ROWS * COLS / PACK_FACTOR],
                    words.clone(),
                ),
                raw(names.shape, "I64", &[2], meta.clone()),
                raw(names.scale, "BF16", &[ROWS, GROUPS], scales.clone()),
            ],
        ),
        (
            "padded packed words",
            vec![
                raw(
                    names.packed,
                    "I32",
                    &[ROWS, COLS / PACK_FACTOR + 1],
                    i32_bytes(&vec![0x0123_4567; ROWS * (COLS / PACK_FACTOR + 1)]),
                ),
                raw(names.shape, "I64", &[2], meta.clone()),
                raw(names.scale, "BF16", &[ROWS, GROUPS], scales.clone()),
            ],
        ),
        (
            "f32 shape metadata",
            vec![
                raw(names.packed, "I32", &packed_shape, words.clone()),
                raw(names.shape, "F32", &[2], f32_bytes(&[48.0, 192.0])),
                raw(names.scale, "BF16", &[ROWS, GROUPS], scales.clone()),
            ],
        ),
        (
            "rank-2 shape metadata",
            vec![
                raw(names.packed, "I32", &packed_shape, words.clone()),
                raw(names.shape, "I64", &[1, 2], meta.clone()),
                raw(names.scale, "BF16", &[ROWS, GROUPS], scales.clone()),
            ],
        ),
        (
            "four shape metadata elements",
            vec![
                raw(names.packed, "I32", &packed_shape, words),
                raw(
                    names.shape,
                    "I64",
                    &[4],
                    i64_bytes(&[ROWS as i64, COLS as i64, 1, 1]),
                ),
                raw(names.scale, "BF16", &[ROWS, GROUPS], scales),
            ],
        ),
    ];
    for (label, mut entries) in cases {
        let mut with_bf16 = bf16_raws();
        with_bf16.append(&mut entries);
        let source = opened(&with_bf16).expect("container admitted the mutation");
        let contract = ct_contract(buffers(ct_op(names, &selection)));
        assert!(
            bind(
                &contract,
                &source.reader,
                &expected(&[("k_proj", REGION_BYTES)])
            )
            .is_err(),
            "accepted {label}"
        );
    }

    // A span shorter than its own dtype and shape never reaches the plan: the container parser is
    // the layer that refuses it, and this asserts exactly that.
    let truncated = {
        let mut entries = bf16_raws();
        entries.extend(ct_raws(names, &[ROWS as i64, COLS as i64], GROUP));
        entries.last_mut().expect("scale entry").bytes.truncate(2);
        let file = image(&entries).expect("image");
        SafetensorsFile::open(file.path(), Limits::default())
    };
    assert!(truncated.is_err(), "a truncated span passed the parser");

    // Valid control again, on the same code path the rejections above used.
    let source = fixture()?;
    let plan = bind(
        &ct_contract(buffers(ct_op(names, &selection))),
        &source.reader,
        &expected(&[("k_proj", REGION_BYTES)]),
    )?;
    let (bytes, _) = stream(&plan, "k_proj", TILE_BYTES)?;
    assert_eq!(bytes, reference_bytes(&selection));
    Ok(())
}

#[test]
fn a_ct_grid_rejects_bad_ranges_alignment_orders_and_unknown_fields() -> Result<()> {
    let source = fixture().expect("source");
    let selection = standard("row_major");

    // The schema decides these, on a complete contract root, so the field under test is the reason.
    let mut op = ct_op(&STANDARD, &selection);
    op["tile_order"] = json!("diagonal");
    assert!(serde_json::from_value::<WeightContract>(contract_json(
        2,
        declared_config(),
        buffers(op)
    ))
    .is_err());
    let mut op = ct_op(&STANDARD, &selection);
    op.as_object_mut().expect("object").remove("tile_order");
    assert!(serde_json::from_value::<WeightContract>(contract_json(
        2,
        declared_config(),
        buffers(op)
    ))
    .is_err());
    // Two spellings of one order key: the duplicate is refused, not resolved by last-wins.
    let raw = serde_json::to_string(&contract_json(2, declared_config(), buffers(standard_op())))?;
    let once = "\"tile_order\":\"row_major\"";
    assert!(raw.contains(once));
    assert!(WeightContract::from_metadata_json(
        format!(
            "{{\"weight_inputs\":{}}}",
            raw.replace(once, &format!("{once},{once}"))
        )
        .as_bytes()
    )
    .is_err());

    // A companion or numeric the schema has no field for: a zero point, a group index or a per-op
    // scale must be refused as an unknown field, never ignored or used to rebase or requantize.
    for (field, value) in [
        ("weight_zero_point", json!("proj.weight_zero_point")),
        ("weight_g_idx", json!("proj.weight_g_idx")),
        ("scale", json!(0.5)),
        ("group_size", json!(GROUP)),
        ("subtiles", json!(true)),
        ("encoding", json!("ct")),
        ("transpose", json!(true)),
    ] {
        let mut op = ct_op(&STANDARD, &selection);
        op[field] = value.clone();
        assert!(
            serde_json::from_value::<WeightContract>(contract_json(
                2,
                declared_config(),
                buffers(op)
            ))
            .is_err(),
            "accepted CT op field {field} {value}"
        );
    }
    // Every selector is required: dropping any one of them is a schema failure.
    for field in [
        "weight_packed",
        "weight_shape",
        "weight_scale",
        "shape",
        "first_row",
        "first_col",
        "rows",
        "cols",
        "tile_rows",
        "tile_cols",
    ] {
        let mut op = ct_op(&STANDARD, &selection);
        op.as_object_mut().expect("object").remove(field);
        assert!(
            serde_json::from_value::<WeightContract>(contract_json(
                2,
                declared_config(),
                buffers(op)
            ))
            .is_err(),
            "accepted a CT op without {field}"
        );
    }
    // A malformed expected logical shape is refused wherever the implementation notices - schema or
    // binding - but never accepted.
    for shape in [
        json!([ROWS]),
        json!([ROWS, COLS, 1]),
        json!([]),
        json!("48x192"),
    ] {
        let raw = contract_json(
            2,
            declared_config(),
            buffers(ct_op_json(&STANDARD, &selection, shape.clone())),
        );
        if let Ok(contract) = serde_json::from_value::<WeightContract>(raw) {
            assert!(
                bind(
                    &contract,
                    &source.reader,
                    &expected(&[("k_proj", REGION_BYTES)])
                )
                .is_err(),
                "accepted expected logical shape {shape}"
            );
        }
    }

    // Rejections that need the source: region, alignment, extents and the declared shape.
    let mut off_grid = selection.clone();
    off_grid.cols = 130;
    let mut partial_rows = selection.clone();
    partial_rows.tile_rows = 24;
    let mut unaligned_region = selection.clone();
    unaligned_region.first_col = 8;
    let mut unaligned_tile = selection.clone();
    unaligned_tile.first_col = 40;
    let mut past_bottom = selection.clone();
    past_bottom.rows = 48;
    let mut past_right = selection.clone();
    past_right.cols = 192;
    let mut overflowing = selection.clone();
    overflowing.rows = usize::MAX;
    for (label, selection) in [
        ("region not a whole number of tiles", off_grid),
        ("tile rows do not divide the region", partial_rows),
        ("region starts mid scale group", unaligned_region),
        ("a tile would start mid scale group", unaligned_tile),
        ("region exceeds the declared rows", past_bottom),
        ("region exceeds the declared columns", past_right),
        ("region arithmetic would overflow", overflowing),
    ] {
        let contract = ct_contract(buffers(ct_op(&STANDARD, &selection)));
        assert!(
            bind(
                &contract,
                &source.reader,
                &expected(&[("k_proj", REGION_BYTES)])
            )
            .is_err(),
            "accepted {label}"
        );
    }
    for (field, value) in [
        ("rows", json!(0)),
        ("cols", json!(0)),
        ("tile_rows", json!(0)),
        ("tile_cols", json!(0)),
        ("first_row", json!(ROWS)),
        ("first_col", json!(COLS)),
        ("first_col", json!(usize::MAX)),
    ] {
        let mut op = ct_op(&STANDARD, &selection);
        op[field] = value.clone();
        let contract = contract_from_op(op);
        assert!(
            bind(
                &contract,
                &source.reader,
                &expected(&[("k_proj", REGION_BYTES)])
            )
            .is_err(),
            "accepted {field} {value}"
        );
    }

    // The declared logical shape must agree with the source's own `weight_shape` metadata.
    let contract = ct_contract(buffers(standard_op()));
    for meta in [
        vec![ROWS as i64, (COLS / 2) as i64],
        vec![0, COLS as i64],
        vec![-(ROWS as i64), COLS as i64],
    ] {
        let source = meta_fixture(&meta)?;
        assert!(
            bind(
                &contract,
                &source.reader,
                &expected(&[("k_proj", REGION_BYTES)])
            )
            .is_err(),
            "accepted weight_shape {meta:?}"
        );
    }
    // A `weight_shape` of four elements never reaches the plan: the codec requires rank-1 [2] I64,
    // and its view constructor is the layer that refuses it.
    let view = ct_view(&meta_fixture(&[ROWS as i64, COLS as i64, 1, 1])?, &STANDARD);
    assert!(view.is_err(), "a rank-2 weight_shape built a view");

    // Valid control in the same test, so the rejections above are not unknown-operation artifacts.
    let plan = bind(
        &ct_contract(buffers(standard_op())),
        &source.reader,
        &expected(&[("k_proj", REGION_BYTES)]),
    )
    .expect("control");
    let (bytes, _) = stream(&plan, "k_proj", TILE_BYTES)?;
    assert_eq!(bytes, reference_bytes(&standard("row_major")));
    Ok(())
}

#[test]
fn output_length_and_tile_counts_are_checked_without_a_fixed_ceiling() -> Result<()> {
    let source = fixture().expect("source");
    let selection = standard("row_major");
    let contract = ct_contract(buffers(standard_op()));
    let tiles = tile_starts(&selection);
    let sum: usize = tiles
        .iter()
        .map(|tile| tile.encoded_bytes().expect("legal tile"))
        .sum();
    assert_eq!(sum, REGION_BYTES);
    // Any requirement other than the sum of the tiles' own sizes is a binding failure.
    for total in [
        REGION_BYTES - 1,
        REGION_BYTES + 1,
        REGION_BYTES + TILE_BYTES,
        2 * REGION_BYTES,
        tiles.len(),
        0,
    ] {
        assert!(
            bind(&contract, &source.reader, &expected(&[("k_proj", total)])).is_err(),
            "accepted a program requirement of {total}"
        );
    }
    // Buffer coverage is the program's, exactly: a renamed or extra requirement is refused.
    assert!(bind(
        &contract,
        &source.reader,
        &expected(&[("k_prox", REGION_BYTES)])
    )
    .is_err());
    assert!(bind(
        &contract,
        &source.reader,
        &expected(&[("k_proj", REGION_BYTES), ("v_proj", TILE_BYTES)])
    )
    .is_err());

    // Counts come from the region divided by the tiling and from checked arithmetic, not from an
    // arbitrary cap: the whole 48x192 matrix at 16x32 tiles (18 packets) binds and streams, and so
    // does a single 32x96 packet, while extents that cannot be computed at all are refused.
    let full = Selection {
        first_row: 0,
        first_col: 0,
        rows: ROWS,
        cols: COLS,
        tile_rows: TILE_ROWS,
        tile_cols: TILE_COLS,
        order: "column_major",
    };
    assert_eq!(tile_starts(&full).len(), 18);
    let full_total = full.rows * full.cols / 2 + full.rows * (full.cols / GROUP) * 2;
    let full_contract = ct_contract(buffers(ct_op(&STANDARD, &full)));
    let plan = bind(
        &full_contract,
        &source.reader,
        &expected(&[("k_proj", full_total)]),
    )
    .expect("18 tile grid");
    let (bytes, _) = stream(&plan, "k_proj", TILE_BYTES)?;
    assert_eq!(bytes.len(), full_total);
    assert_eq!(bytes, reference_bytes(&full));

    let single = ct_contract(buffers(ct_op(&STANDARD, &wide("row_major"))));
    assert!(bind(
        &single,
        &source.reader,
        &expected(&[("k_proj", WIDE_TILE_BYTES)])
    )
    .is_ok());
    for (field, value) in [
        ("cols", usize::MAX),
        ("rows", usize::MAX),
        ("first_row", usize::MAX),
    ] {
        let mut selection = standard("row_major");
        match field {
            "cols" => selection.cols = value,
            "rows" => selection.rows = value,
            _ => selection.first_row = value,
        }
        let contract = ct_contract(buffers(ct_op(&STANDARD, &selection)));
        assert!(
            bind(&contract, &source.reader, &expected(&[("k_proj", 1)])).is_err(),
            "accepted an uncomputable {field}"
        );
    }
    Ok(())
}

#[test]
fn a_sink_failure_stops_after_the_partial_prefix_it_reached() -> Result<()> {
    let source = fixture().expect("source");
    let selection = standard("row_major");
    let contract = ct_contract(buffers(standard_op()));
    let plan = bind(
        &contract,
        &source.reader,
        &expected(&[("k_proj", REGION_BYTES)]),
    )
    .expect("binding");
    let reference = reference_bytes(&selection);

    // Establish the real call count for this scratch size before choosing failure thresholds,
    // because the suite does not fix how a plan batches chunks.
    let (_, calls) = stream(&plan, "k_proj", 600)?;
    assert!(calls > 2, "a 600-byte scratch needs more than two chunks");
    for limit in [1usize, calls / 2, calls - 1] {
        let mut scratch = vec![0u8; 600];
        let mut sink = Sink::new(600);
        sink.fail_after = Some(limit);
        let error = write(&plan, "k_proj", &mut scratch, &mut sink)
            .expect_err("a failing sink must be reported");
        assert!(error.to_string().contains("sink failed"), "{error}");
        assert_eq!(sink.calls, limit + 1);
        assert!(sink.bytes.len() < REGION_BYTES);
        // The destination is caller-owned: bytes already handed over may stay, and nothing after
        // the failure is emitted.
        assert_eq!(&reference[..sink.bytes.len()], sink.bytes.as_slice());
    }

    // The same plan streams completely afterwards, into a destination that accepts everything: the
    // failure is not a poisoned plan, and no other buffer's bytes appear in it.
    let (bytes, _) = stream(&plan, "k_proj", RUNTIME_SCRATCH_BYTES)?;
    assert_eq!(bytes, reference);
    assert!(!bytes.iter().all(|byte| *byte == CANARY));
    Ok(())
}

/// Every nibble of a byte range, sorted: a multiset, so it says nothing about order. It is used
/// only to show the two grid orders carry the same data in a different tile order.
fn sorted_nibbles(bytes: &[u8]) -> Vec<u8> {
    let mut nibbles: Vec<u8> = bytes
        .iter()
        .flat_map(|byte| [byte & 0x0f, byte >> 4])
        .collect();
    nibbles.sort_unstable();
    nibbles
}
