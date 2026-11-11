use npu_weights::source_native::{
    awq::AwqTensor, gguf::GgufFile, safetensors::SafetensorsFile, write_encoded, Limits,
    WeightTensor,
};
use std::io::Write;

fn string(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
}

#[test]
fn both_codecs_share_source_views_writer_and_independent_matrix_reference() {
    let mut awq_file = tempfile::NamedTempFile::new().unwrap();
    let header = r#"{"w.qweight":{"dtype":"I32","shape":[32,1],"data_offsets":[0,128]},"w.qzeros":{"dtype":"I32","shape":[1,1],"data_offsets":[128,132]},"w.scales":{"dtype":"F16","shape":[1,8],"data_offsets":[132,148]}}"#;
    awq_file
        .write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    awq_file.write_all(header.as_bytes()).unwrap();
    let mut awq_raw = Vec::new();
    for _ in 0..32 {
        awq_raw.extend_from_slice(&0x75316420u32.to_le_bytes());
    }
    awq_raw.extend_from_slice(&0x88888888u32.to_le_bytes());
    for _ in 0..8 {
        awq_raw.extend_from_slice(&0x3c00u16.to_le_bytes());
    }
    awq_file.write_all(&awq_raw).unwrap();
    let parsed = SafetensorsFile::open(awq_file.path(), Limits::default()).unwrap();
    let awq = AwqTensor::new(
        "w",
        parsed.get("w.qweight").unwrap().clone(),
        parsed.get("w.qzeros").unwrap().clone(),
        parsed.get("w.scales").unwrap().clone(),
        32,
    )
    .unwrap();
    drop(parsed);

    let mut gguf = Vec::new();
    gguf.extend_from_slice(b"GGUF");
    gguf.extend_from_slice(&3u32.to_le_bytes());
    gguf.extend_from_slice(&1u64.to_le_bytes());
    gguf.extend_from_slice(&0u64.to_le_bytes());
    string(&mut gguf, "w");
    gguf.extend_from_slice(&2u32.to_le_bytes());
    gguf.extend_from_slice(&32u64.to_le_bytes());
    gguf.extend_from_slice(&8u64.to_le_bytes());
    gguf.extend_from_slice(&2u32.to_le_bytes());
    gguf.extend_from_slice(&0u64.to_le_bytes());
    gguf.resize((gguf.len() + 31) / 32 * 32, 0);
    let mut gguf_raw = Vec::new();
    for row in 0u8..8 {
        gguf_raw.extend_from_slice(&0x3c00u16.to_le_bytes());
        gguf_raw.extend_from_slice(&[row | (row << 4); 16]);
    }
    gguf.extend_from_slice(&gguf_raw);
    let mut gguf_file = tempfile::NamedTempFile::new().unwrap();
    gguf_file.write_all(&gguf).unwrap();
    let gguf = GgufFile::open(gguf_file.path(), Limits::default()).unwrap();
    let gt = gguf.get("w").unwrap();

    for (tensor, encoded) in [
        (&awq as &dyn WeightTensor, &awq_raw),
        (gt as &dyn WeightTensor, &gguf_raw),
    ] {
        assert_eq!(
            (tensor.descriptor().rows, tensor.descriptor().cols),
            (8, 32)
        );
        let mut copied = Vec::new();
        let sections = write_encoded(tensor, &mut copied, &mut [0; 5]).unwrap();
        assert_eq!(&copied, encoded);
        assert_eq!(
            sections.iter().map(|s| s.len).sum::<u64>(),
            encoded.len() as u64
        );
        for row in 0..8 {
            let mut values = [0.0; 7];
            tensor.decode_row(row, 13, &mut values).unwrap();
            assert_eq!(values, [(row as f32) - 8.0; 7]);
            // Independent scalar GEMV and two-row GEMM reference over the same logical weights.
            let dot: f32 = values
                .iter()
                .enumerate()
                .map(|(k, w)| w * (k as f32 - 2.0))
                .sum();
            assert_eq!(dot, (row as f32 - 8.0) * 7.0);
            let second: f32 = values.iter().map(|w| w * 2.0).sum();
            assert_eq!(second, (row as f32 - 8.0) * 14.0);
        }
    }
}
