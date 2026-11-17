use npu_weights::source_native::{
    safetensors::{SafetensorsFile, SafetensorsSet},
    write_encoded, DenseTensor, Limits, ScalarType, WeightTensor,
};
use std::io::Write;

fn source(header: &str, payload: &[u8]) -> tempfile::NamedTempFile {
    let mut file = tempfile::NamedTempFile::new().unwrap();
    file.write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(header.as_bytes()).unwrap();
    file.write_all(payload).unwrap();
    file
}

#[test]
fn keeps_integer_spans_and_exact_float_source_types() {
    let h = r#"{"codes":{"dtype":"I32","shape":[1,1],"data_offsets":[0,4]},"scale":{"dtype":"F16","shape":[1,1],"data_offsets":[4,6]},"__metadata__":{"producer":"test"}}"#;
    let raw = [0x10, 0x32, 0x54, 0x76, 0x01, 0x3c];
    let file = source(h, &raw);
    let parsed = SafetensorsFile::open(file.path(), Limits::default()).unwrap();
    assert_eq!(parsed.get("codes").unwrap().dtype, ScalarType::I32);
    let scale = parsed.get("scale").unwrap().clone();
    drop(parsed);
    let t = DenseTensor::new("scale", 1, 1, scale.dtype, scale.span).unwrap();
    let mut f = [0.0];
    t.decode_row(0, 0, &mut f).unwrap();
    assert_eq!(f[0], 1.0009765625);
    let mut bytes = vec![];
    write_encoded(&t, &mut bytes, &mut [0; 1]).unwrap();
    assert_eq!(bytes, [0x01, 0x3c]);
}

#[test]
fn rejects_duplicate_names_fields_and_metadata() {
    for h in [
        r#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
        r#"{"x":{"dtype":"F16","dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
        r#"{"__metadata__":{"producer":"one","producer":"two"},"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
    ] {
        let f = source(h, &[0; 4]);
        assert!(
            SafetensorsFile::open(f.path(), Limits::default()).is_err(),
            "accepted {h}"
        );
    }
}

#[test]
fn rejects_malformed_ranges_and_types_before_data_reads() {
    for h in [
        r#"{"x":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}"#,
        r#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}"#,
        r#"{"x":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}}"#,
        r#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"y":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
        r#"{"x":{"dtype":"F32","shape":[18446744073709551615,2],"data_offsets":[0,4]}}"#,
        r#"{"__metadata__":{"producer":4},"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
        r#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}} garbage"#,
    ] {
        let f = source(h, &[0; 4]);
        assert!(
            SafetensorsFile::open(f.path(), Limits::default()).is_err(),
            "accepted {h}"
        );
    }
}

#[test]
fn loads_sharded_index_without_expanding_tensors() {
    let dir = tempfile::tempdir().unwrap();
    for name in ["a", "b"] {
        let h = format!(r#"{{"{name}":{{"dtype":"I32","shape":[1],"data_offsets":[0,4]}}}}"#);
        let f = source(&h, &[0x10, 0x32, 0x54, 0x76]);
        std::fs::copy(f.path(), dir.path().join(format!("{name}.safetensors"))).unwrap();
    }
    let index = dir.path().join("model.safetensors.index.json");
    std::fs::write(
        &index,
        r#"{"metadata":{"total_size":8},"weight_map":{"a":"a.safetensors","b":"b.safetensors"}}"#,
    )
    .unwrap();
    let set = SafetensorsSet::open_index(&index, Limits::default()).unwrap();
    assert_eq!(set.tensors().len(), 2);
    let mut raw = [0; 4];
    set.get("a")
        .unwrap()
        .span
        .read_exact_at(0, &mut raw)
        .unwrap();
    assert_eq!(raw, [0x10, 0x32, 0x54, 0x76]);
    assert!(set.get("missing").is_err());
    for index_text in [
        r#"{"weight_map":{"a":"../escape.safetensors"}}"#,
        r#"{"weight_map":{"a":"/absolute.safetensors"}}"#,
        r#"{"weight_map":{"wrong":"a.safetensors"}}"#,
        r#"{"weight_map":{"a":"a.safetensors","a":"b.safetensors"}}"#,
    ] {
        std::fs::write(&index, index_text).unwrap();
        assert!(
            SafetensorsSet::open_index(&index, Limits::default()).is_err(),
            "accepted {index_text}"
        );
    }
}

#[test]
fn enforces_header_and_schema_limits() {
    let h = r#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
    let f = source(h, &[0; 4]);
    for limits in [
        Limits {
            max_header_bytes: 1,
            ..Limits::default()
        },
        Limits {
            max_tensors: 0,
            ..Limits::default()
        },
        Limits {
            max_string_bytes: 0,
            ..Limits::default()
        },
    ] {
        assert!(SafetensorsFile::open(f.path(), limits).is_err());
    }
    let mut truncated = tempfile::NamedTempFile::new().unwrap();
    truncated.write_all(&u64::MAX.to_le_bytes()).unwrap();
    assert!(SafetensorsFile::open(truncated.path(), Limits::default()).is_err());
}

#[cfg(unix)]
#[test]
fn sharded_index_rejects_symlinks_outside_source_root() {
    let root = tempfile::tempdir().unwrap();
    let outside = source(
        r#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
        &[0; 4],
    );
    std::os::unix::fs::symlink(outside.path(), root.path().join("shard.safetensors")).unwrap();
    let index = root.path().join("model.safetensors.index.json");
    std::fs::write(&index, r#"{"weight_map":{"x":"shard.safetensors"}}"#).unwrap();
    assert!(SafetensorsSet::open_index(&index, Limits::default()).is_err());
    // Explicit shard access is a separate caller selection, not authority granted by an index.
    assert!(
        SafetensorsFile::open(root.path().join("shard.safetensors"), Limits::default()).is_ok()
    );
}

#[test]
fn admits_i64_metadata_bytes_extents_ranks_and_counts() {
    // I64 belongs to the container as integer metadata: eight bytes per element, read back
    // unchanged, with the parser owning byte extent, overflow and rank. Which metadata tensor means
    // what, and whether a reader accepts rank 1, is the reader's contract, not this one.
    let metadata = [3i64, 24];
    let bytes: Vec<u8> = metadata
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let file = source(
        r#"{"meta":{"dtype":"I64","shape":[2],"data_offsets":[0,16]}}"#,
        &bytes,
    );
    let parsed = SafetensorsFile::open(file.path(), Limits::default()).unwrap();
    let entry = parsed.get("meta").unwrap();
    assert_eq!(entry.dtype, ScalarType::I64);
    assert_eq!(entry.shape, [2]);
    assert_eq!(entry.span.len(), 16, "I64 elements are eight bytes");
    let mut raw = [0u8; 16];
    entry.span.read_exact_at(0, &mut raw).unwrap();
    assert_eq!(raw.as_slice(), bytes.as_slice());
    let mut values = [0i64; 2];
    for (index, value) in values.iter_mut().enumerate() {
        let mut element = [0u8; 8];
        entry
            .span
            .read_exact_at((index * 8) as u64, &mut element)
            .unwrap();
        *value = i64::from_le_bytes(element);
    }
    assert_eq!(values, metadata);

    // Rank stays with the reader: a rank-2 I64 entry with the same extent is a valid container
    // object. The CT view that requires rank 1 rejects it, tested where that contract lives.
    let file = source(
        r#"{"meta":{"dtype":"I64","shape":[1,2],"data_offsets":[0,16]}}"#,
        &bytes,
    );
    assert!(SafetensorsFile::open(file.path(), Limits::default()).is_ok());

    for header in [
        // Eight bytes per element, not four: the same shape with a four-byte extent is malformed.
        r#"{"meta":{"dtype":"I64","shape":[2],"data_offsets":[0,8]}}"#,
        r#"{"meta":{"dtype":"I64","shape":[2],"data_offsets":[0,15]}}"#,
        r#"{"meta":{"dtype":"I64","shape":[1],"data_offsets":[0,16]}}"#,
        r#"{"meta":{"dtype":"I64","shape":[3],"data_offsets":[0,16]}}"#,
        // Count and rank limits stay parser rules for every dtype.
        r#"{"meta":{"dtype":"I64","shape":[18446744073709551615,2],"data_offsets":[0,16]}}"#,
        r#"{"meta":{"dtype":"I64","shape":[2,2,2,2,2,2,2,2,2],"data_offsets":[0,4096]}}"#,
        r#"{"meta":{"dtype":"I64","shape":[-1],"data_offsets":[0,8]}}"#,
    ] {
        let file = source(header, &bytes);
        assert!(
            SafetensorsFile::open(file.path(), Limits::default()).is_err(),
            "accepted {header}"
        );
    }
}
