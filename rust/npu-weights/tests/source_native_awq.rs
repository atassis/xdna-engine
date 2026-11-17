use anyhow::Result;
use half::{bf16, f16};
use npu_weights::source_native::awq::AwqTensor;
use npu_weights::source_native::safetensors::TensorEntry;
use npu_weights::source_native::{write_encoded, FileSource, ScalarType, WeightTensor};
use std::io::Write;
use tempfile::NamedTempFile;

struct Entries {
    _file: NamedTempFile,
    qweight: TensorEntry,
    qzeros: TensorEntry,
    scales: TensorEntry,
}

fn scale_bytes(dtype: ScalarType, values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::new();
    for &value in values {
        match dtype {
            ScalarType::F16 => {
                bytes.extend_from_slice(&f16::from_f32(value).to_bits().to_le_bytes())
            }
            ScalarType::Bf16 => {
                bytes.extend_from_slice(&bf16::from_f32(value).to_bits().to_le_bytes())
            }
            ScalarType::F32 => bytes.extend_from_slice(&value.to_le_bytes()),
            ScalarType::I32 | ScalarType::I64 => panic!("scales must be floating point"),
        }
    }
    bytes
}

fn entries(
    qweight_words: &[u32],
    qweight_shape: Vec<usize>,
    qzeros_words: &[u32],
    qzeros_shape: Vec<usize>,
    scale_type: ScalarType,
    scale_values: &[f32],
    scale_shape: Vec<usize>,
) -> Result<Entries> {
    let mut qweight = Vec::new();
    for &word in qweight_words {
        qweight.extend_from_slice(&word.to_le_bytes());
    }
    let mut qzeros = Vec::new();
    for &word in qzeros_words {
        qzeros.extend_from_slice(&word.to_le_bytes());
    }
    let scales = scale_bytes(scale_type, scale_values);
    let mut file = NamedTempFile::new()?;
    file.write_all(&qweight)?;
    file.write_all(&qzeros)?;
    file.write_all(&scales)?;
    file.flush()?;

    let source = FileSource::open(file.path())?;
    let qweight_len = qweight.len() as u64;
    let qzeros_len = qzeros.len() as u64;
    Ok(Entries {
        _file: file,
        qweight: TensorEntry {
            dtype: ScalarType::I32,
            shape: qweight_shape,
            span: source.span(0, qweight_len)?,
        },
        qzeros: TensorEntry {
            dtype: ScalarType::I32,
            shape: qzeros_shape,
            span: source.span(qweight_len, qzeros_len)?,
        },
        scales: TensorEntry {
            dtype: scale_type,
            shape: scale_shape,
            span: source.span(qweight_len + qzeros_len, scales.len() as u64)?,
        },
    })
}

/// Packs logical lane values according to the AutoAWQ GEMM slot order.
fn awq_word(lanes: [u8; 8]) -> u32 {
    const SLOT_TO_LANE: [usize; 8] = [0, 2, 4, 6, 1, 3, 5, 7];
    SLOT_TO_LANE
        .into_iter()
        .enumerate()
        .fold(0u32, |word, (slot, lane)| {
            word | ((lanes[lane] as u32) << (slot * 4))
        })
}

#[test]
fn decodes_autoawq_slots_unsigned_extrema_and_group_boundaries() -> Result<()> {
    // K=4, N=8, group_size=2.  The first two K values use scale 1, the second
    // two scale 2.  q=0/z=15 and q=15/z=0 exercise both unsigned extrema.
    let qweight = [
        awq_word([0, 15, 1, 2, 3, 4, 5, 6]),
        awq_word([7, 8, 9, 10, 11, 12, 13, 14]),
        awq_word([15, 0, 1, 2, 3, 4, 5, 6]),
        awq_word([7, 8, 9, 10, 11, 12, 13, 14]),
    ];
    let qzeros = [
        awq_word([15, 0, 0, 0, 0, 0, 0, 0]),
        awq_word([15, 0, 0, 0, 0, 0, 0, 0]),
    ];
    let entries = entries(
        &qweight,
        vec![4, 1],
        &qzeros,
        vec![2, 1],
        ScalarType::F16,
        &[1.0; 8].into_iter().chain([2.0; 8]).collect::<Vec<_>>(),
        vec![2, 8],
    )?;
    let tensor = AwqTensor::new("linear", entries.qweight, entries.qzeros, entries.scales, 2)?;

    assert_eq!(tensor.descriptor().rows, 8);
    assert_eq!(tensor.descriptor().cols, 4);
    // Logical lane 0 is slot 0, lane 1 is slot 4, lane 2 is slot 1, etc.
    let mut first_column = [0.0; 8];
    for (row, value) in first_column.iter_mut().enumerate() {
        tensor.decode_row(row, 0, std::slice::from_mut(value))?;
    }
    assert_eq!(first_column, [-15.0, 15.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let mut boundary = [0.0; 3];
    tensor.decode_row(0, 1, &mut boundary)?;
    assert_eq!(boundary, [-8.0, 0.0, -16.0]);
    let mut second_lane = [0.0; 1];
    tensor.decode_row(1, 2, &mut second_lane)?;
    assert_eq!(second_lane, [0.0]);
    Ok(())
}

#[test]
fn distinguishes_packed_words_and_group_dependent_zero_points() -> Result<()> {
    let entries = entries(
        &[
            0x11111111, 0xeeeeeeee, 0x11111111, 0xeeeeeeee, 0x11111111, 0xeeeeeeee, 0x11111111,
            0xeeeeeeee,
        ],
        vec![4, 2],
        &[0x00000000, 0x88888888, 0x22222222, 0xdddddddd],
        vec![2, 2],
        ScalarType::F16,
        &[1.0; 16].into_iter().chain([2.0; 16]).collect::<Vec<_>>(),
        vec![2, 16],
    )?;
    let tensor = AwqTensor::new(
        "multiword",
        entries.qweight,
        entries.qzeros,
        entries.scales,
        2,
    )?;
    let mut first = [0.0; 4];
    let mut second = [0.0; 4];
    tensor.decode_row(0, 0, &mut first)?;
    tensor.decode_row(8, 0, &mut second)?;
    assert_eq!(first, [1.0, 1.0, -2.0, -2.0]);
    assert_eq!(second, [6.0, 6.0, 2.0, 2.0]);
    Ok(())
}

#[test]
fn accepts_f16_bf16_and_f32_scales_without_lossy_conversion() -> Result<()> {
    let qweight = [awq_word([2, 2, 2, 2, 2, 2, 2, 2])];
    let qzeros = [awq_word([1, 1, 1, 1, 1, 1, 1, 1])];
    let f16_scale = 1.000_976_6;
    for scale_type in [ScalarType::F16, ScalarType::Bf16, ScalarType::F32] {
        let entries = entries(
            &qweight,
            vec![1, 1],
            &qzeros,
            vec![1, 1],
            scale_type,
            &[f16_scale; 8],
            vec![1, 8],
        )?;
        let tensor = AwqTensor::new("scales", entries.qweight, entries.qzeros, entries.scales, 1)?;
        let mut decoded = [0.0];
        tensor.decode_row(0, 0, &mut decoded)?;
        let expected = match scale_type {
            ScalarType::F16 => f16::from_f32(f16_scale).to_f32(),
            ScalarType::Bf16 => bf16::from_f32(f16_scale).to_f32(),
            ScalarType::F32 => f16_scale,
            ScalarType::I32 | ScalarType::I64 => unreachable!(),
        };
        assert_eq!(decoded, [expected]);
    }
    assert_ne!(
        f16::from_f32(f16_scale).to_f32(),
        bf16::from_f32(f16_scale).to_f32()
    );
    Ok(())
}

#[test]
fn writer_preserves_all_encoded_source_sections_with_tiny_scratch() -> Result<()> {
    let qweight = [awq_word([1, 2, 3, 4, 5, 6, 7, 8])];
    let qzeros = [awq_word([8, 7, 6, 5, 4, 3, 2, 1])];
    let entries = entries(
        &qweight,
        vec![1, 1],
        &qzeros,
        vec![1, 1],
        ScalarType::F32,
        &[3.25; 8],
        vec![1, 8],
    )?;
    let expected = [
        qweight[0].to_le_bytes().as_slice(),
        qzeros[0].to_le_bytes().as_slice(),
        &scale_bytes(ScalarType::F32, &[3.25; 8]),
    ]
    .concat();
    let tensor = AwqTensor::new("writer", entries.qweight, entries.qzeros, entries.scales, 1)?;
    let mut output = Vec::new();
    let written = write_encoded(&tensor, &mut output, &mut [0u8; 1])?;
    assert_eq!(output, expected);
    assert_eq!(
        written
            .iter()
            .map(|section| section.name.as_str())
            .collect::<Vec<_>>(),
        ["qweight", "qzeros", "scales"]
    );
    assert_eq!(
        written
            .iter()
            .map(|section| section.len)
            .collect::<Vec<_>>(),
        [4, 4, 32]
    );
    assert_eq!(
        written
            .iter()
            .map(|section| section.offset)
            .collect::<Vec<_>>(),
        [0, 4, 8]
    );
    Ok(())
}

#[test]
fn tensor_keeps_file_source_alive_after_builder_source_is_dropped() -> Result<()> {
    let qweight = [awq_word([4, 4, 4, 4, 4, 4, 4, 4])];
    let qzeros = [awq_word([1, 1, 1, 1, 1, 1, 1, 1])];
    let entries = entries(
        &qweight,
        vec![1, 1],
        &qzeros,
        vec![1, 1],
        ScalarType::F16,
        &[2.0; 8],
        vec![1, 8],
    )?;
    let tensor = AwqTensor::new(
        "lifetime",
        entries.qweight,
        entries.qzeros,
        entries.scales,
        1,
    )?;
    let mut decoded = [0.0];
    tensor.decode_row(0, 0, &mut decoded)?;
    assert_eq!(decoded, [6.0]);
    Ok(())
}

#[test]
fn rejects_malformed_triples_and_invalid_slices() -> Result<()> {
    let valid = || {
        entries(
            &[0],
            vec![1, 1],
            &[0],
            vec![1, 1],
            ScalarType::F16,
            &[1.0; 8],
            vec![1, 8],
        )
    };

    let bad_rank = valid()?;
    let mut qweight = bad_rank.qweight;
    qweight.shape = vec![1];
    assert!(AwqTensor::new("bad-rank", qweight, bad_rank.qzeros, bad_rank.scales, 1).is_err());

    let bad_dtype = valid()?;
    let mut qzeros = bad_dtype.qzeros;
    qzeros.dtype = ScalarType::F16;
    assert!(AwqTensor::new("bad-dtype", bad_dtype.qweight, qzeros, bad_dtype.scales, 1).is_err());

    let mismatched = valid()?;
    let mut scales = mismatched.scales;
    scales.shape = vec![1, 7];
    assert!(AwqTensor::new("mismatch", mismatched.qweight, mismatched.qzeros, scales, 1).is_err());

    let integer_scales = valid()?;
    let mut scales = integer_scales.scales;
    scales.dtype = ScalarType::I32;
    assert!(AwqTensor::new(
        "integer-scales",
        integer_scales.qweight,
        integer_scales.qzeros,
        scales,
        1
    )
    .is_err());

    let bad_group = valid()?;
    assert!(AwqTensor::new(
        "group-zero",
        bad_group.qweight,
        bad_group.qzeros,
        bad_group.scales,
        0
    )
    .is_err());

    let nondivisible_group = valid()?;
    assert!(AwqTensor::new(
        "group-shape",
        nondivisible_group.qweight,
        nondivisible_group.qzeros,
        nondivisible_group.scales,
        2
    )
    .is_err());

    let short = valid()?;
    let source = FileSource::open(short._file.path())?;
    let qweight = TensorEntry {
        dtype: ScalarType::I32,
        shape: vec![1, 1],
        span: source.span(0, 3)?,
    };
    assert!(AwqTensor::new("short", qweight, short.qzeros, short.scales, 1).is_err());

    let overflow = valid()?;
    let mut qweight = overflow.qweight;
    qweight.shape = vec![usize::MAX, 2];
    assert!(AwqTensor::new("overflow", qweight, overflow.qzeros, overflow.scales, 1).is_err());

    let slice = valid()?;
    let tensor = AwqTensor::new("slice", slice.qweight, slice.qzeros, slice.scales, 1)?;
    assert!(tensor.decode_row(8, 0, &mut []).is_err());
    assert!(tensor.decode_row(0, 2, &mut []).is_err());
    assert!(tensor.decode_row(0, 0, &mut [0.0; 2]).is_err());
    tensor.decode_row(0, 1, &mut [])?;
    Ok(())
}
