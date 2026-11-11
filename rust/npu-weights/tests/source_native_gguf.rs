use npu_weights::source_native::{gguf::GgufFile, write_encoded, Encoding, Limits, WeightTensor};
use std::fs;
use tempfile::NamedTempFile;

fn put_string(out: &mut Vec<u8>, text: &str) {
    out.extend_from_slice(&(text.len() as u64).to_le_bytes());
    out.extend_from_slice(text.as_bytes());
}

struct Tensor<'a> {
    name: &'a str,
    dims: &'a [u64],
    ty: u32,
    offset: u64,
    data: Vec<u8>,
}

/// Write a GGUF v3 fixture directly from the format fields, rather than encoding through the
/// implementation under test. `metadata` is already encoded as key/type/value records.
fn fixture(metadata: Vec<u8>, tensors: Vec<Tensor<'_>>) -> NamedTempFile {
    let count = usize::from(!metadata.is_empty());
    fixture_with_metadata_count(metadata, count, tensors)
}

fn fixture_with_metadata_count(
    metadata: Vec<u8>,
    metadata_count: usize,
    tensors: Vec<Tensor<'_>>,
) -> NamedTempFile {
    let mut header = Vec::new();
    header.extend_from_slice(b"GGUF");
    header.extend_from_slice(&3u32.to_le_bytes());
    header.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    header.extend_from_slice(&(metadata_count as u64).to_le_bytes());
    header.extend_from_slice(&metadata);
    for tensor in &tensors {
        put_string(&mut header, tensor.name);
        header.extend_from_slice(&(tensor.dims.len() as u32).to_le_bytes());
        for dim in tensor.dims {
            header.extend_from_slice(&dim.to_le_bytes());
        }
        header.extend_from_slice(&tensor.ty.to_le_bytes());
        header.extend_from_slice(&tensor.offset.to_le_bytes());
    }
    let data_start = (header.len() + 31) & !31;
    header.resize(data_start, 0);
    let data_end = tensors
        .iter()
        .map(|t| t.offset as usize + t.data.len())
        .max()
        .unwrap_or(0);
    header.resize(data_start + data_end, 0);
    for tensor in tensors {
        let start = data_start + tensor.offset as usize;
        header[start..start + tensor.data.len()].copy_from_slice(&tensor.data);
    }
    let file = NamedTempFile::new().unwrap();
    fs::write(file.path(), header).unwrap();
    file
}

fn alignment_metadata(alignment: u32) -> Vec<u8> {
    let mut metadata = Vec::new();
    put_string(&mut metadata, "general.alignment");
    metadata.extend_from_slice(&4u32.to_le_bytes()); // UINT32
    metadata.extend_from_slice(&alignment.to_le_bytes());
    metadata
}

#[test]
fn decodes_q4_variants_f32_and_preserves_encoded_section() {
    // f16 0x3e00 is 1.5 (but is not the same value when read as a BF16), exercising f16 scales.
    let mut q40 = vec![0, 0x3e];
    q40.extend((0..16).map(|i| (15 - i) << 4 | i));
    let mut q41 = vec![0, 0x3c, 0, 0xc0]; // d=1, m=-2
    q41.extend((0..16).map(|i| (15 - i) << 4 | i));
    let f32_data = [1.25f32, -2.5, 9.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    let encoded = q40.clone();
    let file = fixture(
        alignment_metadata(32),
        vec![
            Tensor {
                name: "q40",
                dims: &[32],
                ty: 2,
                offset: 0,
                data: q40,
            },
            Tensor {
                name: "q41",
                dims: &[32],
                ty: 3,
                offset: 32,
                data: q41,
            },
            Tensor {
                name: "plain",
                dims: &[3],
                ty: 0,
                offset: 64,
                data: f32_data,
            },
        ],
    );
    let gguf = GgufFile::open(file.path(), Limits::default()).unwrap();
    let q40 = gguf.get("q40").unwrap();
    assert_eq!(q40.descriptor().encoding, Encoding::GgufQ4_0);
    let mut values = [0.0; 3];
    q40.decode_row(0, 0, &mut values).unwrap();
    assert_eq!(values, [-12.0, -10.5, -9.0]);
    q40.decode_row(0, 16, &mut values).unwrap();
    assert_eq!(values, [10.5, 9.0, 7.5]);
    let mut q41_values = [0.0; 2];
    gguf.get("q41")
        .unwrap()
        .decode_row(0, 0, &mut q41_values)
        .unwrap();
    assert_eq!(q41_values, [-2.0, -1.0]);
    let mut dense = [0.0; 2];
    gguf.get("plain")
        .unwrap()
        .decode_row(0, 1, &mut dense)
        .unwrap();
    assert_eq!(dense, [-2.5, 9.0]);
    let mut copied = Vec::new();
    let sections = write_encoded(q40, &mut copied, &mut [0u8; 5]).unwrap();
    assert_eq!(copied, encoded);
    assert_eq!(sections[0].len, 18);
}

#[test]
fn decodes_q6k_official_lane_layout_and_partial_slices() {
    // block_q6_K is ql[128], qh[64], signed scales[16], f16 d.  These literal fields select
    // every qh pair and both 128-element halves; expected values are hand-calculated below.
    let mut q6 = vec![0u8; 210];
    q6[0] = 0xa1; // lanes 0(low=1) and 64(high=10)
    q6[32] = 0x74; // lanes 32(low=4) and 96(high=7)
    q6[64] = 0x02; // lane 128 low=2
    q6[96] = 0xd0; // lane 224 high=13
    q6[128] = 0b10_01_11_10; // lane high bits: 0=>2,32=>3,64=>1,96=>2
    q6[160] = 0b00_00_00_01; // lane 128=>1
    q6[192..208].copy_from_slice(&[
        1,
        -2i8 as u8,
        3,
        -4i8 as u8,
        5,
        -6i8 as u8,
        7,
        -8i8 as u8,
        9,
        -10i8 as u8,
        11,
        -12i8 as u8,
        13,
        -14i8 as u8,
        15,
        -16i8 as u8,
    ]);
    q6[208..210].copy_from_slice(&0x3800u16.to_le_bytes()); // d=0.5
    let file = fixture(
        alignment_metadata(32),
        vec![Tensor {
            name: "q6",
            dims: &[256],
            ty: 14,
            offset: 0,
            data: q6,
        }],
    );
    let gguf = GgufFile::open(file.path(), Limits::default()).unwrap();
    let tensor = gguf.get("q6").unwrap();
    let mut lane = [0.0];
    for (index, expected) in [
        (0, 0.5),
        (32, 30.0),
        (64, -15.0),
        (96, 24.5),
        (128, -63.0),
        (224, -142.5),
    ] {
        tensor.decode_row(0, index, &mut lane).unwrap();
        assert_eq!(lane[0], expected, "lane {index}");
    }
    let mut slice = [0.0; 3];
    tensor.decode_row(0, 31, &mut slice).unwrap();
    assert_eq!(slice, [32.0, 30.0, -48.0]);
}

#[test]
fn tensor_outlives_container_and_validates_rows() {
    let data = [3.0f32, 4.0, 5.0, 6.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    let file = fixture(
        vec![],
        vec![Tensor {
            name: "matrix",
            dims: &[2, 2],
            ty: 0,
            offset: 0,
            data,
        }],
    );
    let tensor = {
        let reader = GgufFile::open(file.path(), Limits::default()).unwrap();
        reader.get("matrix").unwrap().clone()
    };
    assert_eq!((tensor.descriptor().rows, tensor.descriptor().cols), (2, 2));
    let mut row = [0.0; 2];
    tensor.decode_row(1, 0, &mut row).unwrap();
    assert_eq!(row, [5.0, 6.0]);
    assert!(tensor.decode_row(2, 0, &mut []).is_err());
    tensor.decode_row(1, 2, &mut []).unwrap();
}

#[test]
fn tokenizer_arrays_are_bounded_by_header_bytes_not_top_level_key_count() {
    let mut metadata = Vec::new();
    put_string(&mut metadata, "tokenizer.ggml.tokens");
    metadata.extend_from_slice(&9u32.to_le_bytes());
    metadata.extend_from_slice(&8u32.to_le_bytes());
    metadata.extend_from_slice(&151936u64.to_le_bytes());
    metadata.resize(metadata.len() + 151936 * 8, 0); // Empty strings, each with a u64 length.
    let file = fixture(metadata, vec![]);
    let mut limits = Limits::default();
    limits.max_metadata_entries = 1;
    GgufFile::open(file.path(), limits).unwrap();
    limits.max_header_bytes = 1024;
    assert!(GgufFile::open(file.path(), limits).is_err());
}

#[test]
fn rejects_unknown_empty_array_type_and_non_u32_alignment() {
    let mut metadata = Vec::new();
    put_string(&mut metadata, "array");
    metadata.extend_from_slice(&9u32.to_le_bytes());
    metadata.extend_from_slice(&99u32.to_le_bytes());
    metadata.extend_from_slice(&0u64.to_le_bytes());
    let file = fixture(metadata, vec![]);
    assert!(GgufFile::open(file.path(), Limits::default()).is_err());
    let mut metadata = Vec::new();
    put_string(&mut metadata, "general.alignment");
    metadata.extend_from_slice(&0u32.to_le_bytes());
    metadata.push(32);
    let file = fixture(metadata, vec![]);
    assert!(GgufFile::open(file.path(), Limits::default()).is_err());
}

#[test]
fn rejects_malformed_headers_and_tensor_layouts() {
    let bad_boolean = {
        let mut metadata = Vec::new();
        put_string(&mut metadata, "flag");
        metadata.extend_from_slice(&7u32.to_le_bytes());
        metadata.push(2);
        fixture(metadata, vec![])
    };
    assert!(GgufFile::open(bad_boolean.path(), Limits::default()).is_err());

    let invalid_alignment = fixture(alignment_metadata(3), vec![]);
    assert!(GgufFile::open(invalid_alignment.path(), Limits::default()).is_err());

    let mut duplicate_metadata = alignment_metadata(32);
    duplicate_metadata.extend_from_slice(&alignment_metadata(64));
    let duplicate_metadata = fixture_with_metadata_count(duplicate_metadata, 2, vec![]);
    assert!(GgufFile::open(duplicate_metadata.path(), Limits::default()).is_err());

    let nested_array = {
        let mut metadata = Vec::new();
        put_string(&mut metadata, "nested");
        metadata.extend_from_slice(&9u32.to_le_bytes()); // ARRAY
        metadata.extend_from_slice(&9u32.to_le_bytes()); // ARRAY element type
        metadata.extend_from_slice(&0u64.to_le_bytes());
        fixture(metadata, vec![])
    };
    assert!(GgufFile::open(nested_array.path(), Limits::default()).is_err());

    let unknown_metadata_type = {
        let mut metadata = Vec::new();
        put_string(&mut metadata, "unknown");
        metadata.extend_from_slice(&99u32.to_le_bytes());
        fixture(metadata, vec![])
    };
    assert!(GgufFile::open(unknown_metadata_type.path(), Limits::default()).is_err());

    let unknown_tensor_type = fixture(
        vec![],
        vec![Tensor {
            name: "unknown",
            dims: &[1],
            ty: 99,
            offset: 0,
            data: vec![],
        }],
    );
    assert!(GgufFile::open(unknown_tensor_type.path(), Limits::default()).is_err());

    let partial_row = fixture(
        vec![],
        vec![Tensor {
            name: "bad",
            dims: &[33],
            ty: 2,
            offset: 0,
            data: vec![0; 36],
        }],
    );
    assert!(GgufFile::open(partial_row.path(), Limits::default()).is_err());

    let overlap = fixture(
        vec![],
        vec![
            Tensor {
                name: "a",
                dims: &[32],
                ty: 2,
                offset: 0,
                data: vec![0; 18],
            },
            Tensor {
                name: "b",
                dims: &[32],
                ty: 2,
                offset: 0,
                data: vec![0; 18],
            },
        ],
    );
    assert!(GgufFile::open(overlap.path(), Limits::default()).is_err());

    let duplicate_tensor = fixture(
        vec![],
        vec![
            Tensor {
                name: "same",
                dims: &[32],
                ty: 2,
                offset: 0,
                data: vec![0; 18],
            },
            Tensor {
                name: "same",
                dims: &[32],
                ty: 2,
                offset: 32,
                data: vec![0; 18],
            },
        ],
    );
    assert!(GgufFile::open(duplicate_tensor.path(), Limits::default()).is_err());

    let truncated = fixture(
        vec![],
        vec![Tensor {
            name: "short",
            dims: &[32],
            ty: 2,
            offset: 0,
            data: vec![0; 4],
        }],
    );
    assert!(GgufFile::open(truncated.path(), Limits::default()).is_err());
}

#[test]
fn all_q6k_lanes_match_pinned_ggml_oracle_across_rows_and_blocks() {
    // Golden values generated with dequantize_row_q6_K from ggml-org/llama.cpp
    // 1648d641174927634dfb72135b220e4052393729, ggml/src/ggml-quants.c.
    // The oracle used the field bytes below and d=1.25; it did not call this crate's decoder.
    const EXPECTED: [f32; 256] = [
        -4320.0, 5120.0, 1760.0, -1600.0, -4960.0, 4480.0, 1120.0, -2240.0, -3040.0, 3840.0, 480.0,
        -320.0, -3680.0, 3200.0, 2400.0, -960.0, 4286.25, -5080.0, -1746.25, 1587.5, 4921.25,
        -4445.0, -1111.25, 2222.5, 3016.25, -3810.0, -476.25, 317.5, 3651.25, -3175.0, -2381.25,
        952.5, 6.25, -20.0, 33.75, 7.5, -18.75, 35.0, 8.75, -17.5, -23.75, 10.0, -16.25, -22.5,
        31.25, -15.0, -21.25, 32.5, -0.0, 0.0, -0.0, -0.0, 0.0, -0.0, -0.0, 0.0, 0.0, -0.0, 0.0,
        0.0, -0.0, 0.0, 0.0, -0.0, -40.0, 23.75, 26.25, 8.75, -8.75, -5.0, -22.5, 20.0, 3.75, 6.25,
        -11.25, -27.5, -25.0, 37.5, 1.25, 3.75, 27.5, 60.0, 55.0, -70.0, -35.0, -2.5, 32.5, 67.5,
        -60.0, -65.0, -30.0, 2.5, 37.5, 72.5, -55.0, -60.0, -55.0, -47.5, -2.5, 2.5, 47.5, -65.0,
        -20.0, -15.0, 32.5, 77.5, -77.5, -30.0, 15.0, 20.0, 67.5, -47.5, 20.0, -40.0, -400.0,
        -440.0, 480.0, 100.0, -260.0, -620.0, -360.0, 560.0, 200.0, -180.0, -540.0, 380.0, 640.0,
        280.0, 540.0, -640.0, -220.0, 200.0, 620.0, -560.0, -140.0, 280.0, 380.0, -480.0, -60.0,
        40.0, 460.0, -400.0, -300.0, 120.0, -1080.0, 1280.0, 440.0, -400.0, -1240.0, 1120.0, 280.0,
        -560.0, -760.0, 960.0, 120.0, -80.0, -920.0, 800.0, 600.0, -240.0, -200.0, 640.0, -1080.0,
        -240.0, 600.0, -1120.0, -280.0, 560.0, 760.0, -320.0, 520.0, 720.0, -1000.0, 480.0, 680.0,
        -1040.0, 400.0, -1280.0, 2160.0, 480.0, -1200.0, 2240.0, 560.0, -1120.0, -1520.0, 640.0,
        -1040.0, -1440.0, 2000.0, -960.0, -1360.0, 2080.0, 320.0, -720.0, -560.0, -1680.0, 2320.0,
        1280.0, 160.0, -960.0, -2000.0, -1840.0, 2160.0, 1120.0, 0.0, -1120.0, -2160.0, -2000.0,
        -3968.75, -1905.0, -2222.5, 2540.0, 4762.5, 4286.25, -3651.25, -1428.75, 635.0, 317.5,
        5080.0, -3016.25, -3333.75, -1111.25, 952.5, 635.0, 2205.0, 2677.5, -4567.5, -1732.5,
        -1417.5, 1575.0, 4410.0, -2835.0, -2362.5, 472.5, 787.5, 3780.0, -3465.0, -630.0, 2362.5,
        2677.5, -71.25, 97.5, 30.0, -37.5, -105.0, 63.75, 116.25, 48.75, -22.5, -90.0, 82.5, 71.25,
        3.75, -3.75, -75.0, 97.5,
    ];
    let mut block = vec![0; 210];
    for i in 0..128 {
        block[i] = (i * 37 + 11) as u8;
    }
    for i in 0..64 {
        block[128 + i] = (i * 53 + 7) as u8;
    }
    let scales: [i8; 16] = [
        -128, 127, -1, 0, 1, -2, 2, -16, 16, -32, 32, -64, 64, -127, 126, -3,
    ];
    for i in 0..16 {
        block[192 + i] = scales[i] as u8;
    }
    block[208..210].copy_from_slice(&0x3d00u16.to_le_bytes());
    let mut data = block.clone();
    data.extend_from_slice(&block);
    // The second row differs, catching an implementation that always reads row zero.
    let mut second = block.clone();
    second[208..210].copy_from_slice(&0x4100u16.to_le_bytes());
    data.extend_from_slice(&second);
    data.extend_from_slice(&second);
    let file = fixture(
        vec![],
        vec![Tensor {
            name: "q6",
            dims: &[512, 2],
            ty: 14,
            offset: 0,
            data,
        }],
    );
    let source = GgufFile::open(file.path(), Limits::default()).unwrap();
    let tensor = source.get("q6").unwrap();
    for row in 0..2 {
        let mut actual = [0.0; 512];
        tensor.decode_row(row, 0, &mut actual).unwrap();
        for k in 0..512 {
            assert_eq!(
                actual[k],
                EXPECTED[k % 256] * (row + 1) as f32,
                "row {row} lane {k}"
            );
        }
        let mut boundary = [0.0; 9];
        tensor.decode_row(row, 252, &mut boundary).unwrap();
        assert_eq!(boundary, actual[252..261]);
    }
}
