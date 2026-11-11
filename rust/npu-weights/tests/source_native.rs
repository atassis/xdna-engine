use npu_weights::source_native::{
    write_encoded, DenseTensor, Encoding, FileSource, ScalarType, WeightTensor,
};
use std::io::{self, Write};

#[test]
fn dense_source_streams_original_bytes_and_decodes_bounded_rows() {
    let mut file = tempfile::NamedTempFile::new().unwrap();
    let bytes: Vec<u8> = [1.0f32, -2.0, 3.5, 4.0]
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    file.write_all(&bytes).unwrap();
    let source = FileSource::open(file.path()).unwrap();
    let tensor =
        DenseTensor::new("weight", 2, 2, ScalarType::F32, source.span(0, 16).unwrap()).unwrap();
    assert_eq!(
        tensor.descriptor().encoding,
        Encoding::Dense(ScalarType::F32)
    );
    drop(source);
    let mut row = [0.0; 1];
    tensor.decode_row(1, 0, &mut row).unwrap();
    assert_eq!(row, [3.5]);
    assert!(tensor.decode_row(2, 0, &mut row).is_err());
    assert!(tensor.decode_row(0, 2, &mut row).is_err());
    let mut copied = Vec::new();
    let written = write_encoded(&tensor, &mut copied, &mut [0; 3]).unwrap();
    assert_eq!(copied, bytes);
    assert_eq!(written[0].offset, 0);
    assert_eq!(written[0].len, 16);
    assert!(write_encoded(&tensor, &mut Vec::new(), &mut []).is_err());
}

#[test]
fn streaming_large_span_never_exceeds_caller_scratch() {
    let file = tempfile::NamedTempFile::new().unwrap();
    file.as_file().set_len(1024 * 1024).unwrap();
    let source = FileSource::open(file.path()).unwrap();
    let tensor = DenseTensor::new(
        "large",
        1024,
        256,
        ScalarType::F32,
        source.span(0, source.len()).unwrap(),
    )
    .unwrap();
    struct BoundedSink {
        bytes: usize,
        max_write: usize,
    }
    impl Write for BoundedSink {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.bytes += bytes.len();
            self.max_write = self.max_write.max(bytes.len());
            assert!(bytes.iter().all(|b| *b == 0));
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let mut out = BoundedSink {
        bytes: 0,
        max_write: 0,
    };
    write_encoded(&tensor, &mut out, &mut [0; 1023]).unwrap();
    assert_eq!(out.bytes, 1024 * 1024);
    assert_eq!(out.max_write, 1023);
}

#[test]
fn propagates_partial_destination_failure() {
    let mut file = tempfile::NamedTempFile::new().unwrap();
    file.write_all(&[0; 8]).unwrap();
    let source = FileSource::open(file.path()).unwrap();
    let t = DenseTensor::new("x", 1, 2, ScalarType::F32, source.span(0, 8).unwrap()).unwrap();
    struct Fails;
    impl Write for Fails {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> {
            Err(io::Error::new(io::ErrorKind::Other, "destination failed"))
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    assert!(write_encoded(&t, &mut Fails, &mut [0; 2]).is_err());
    file.as_file().set_len(0).unwrap();
    assert!(write_encoded(&t, &mut Vec::new(), &mut [0; 2]).is_err());
}

#[test]
fn source_rejects_overflow_short_ranges_and_wrong_dense_geometry() {
    let mut file = tempfile::NamedTempFile::new().unwrap();
    file.write_all(&[0; 8]).unwrap();
    let source = FileSource::open(file.path()).unwrap();
    assert!(source.span(u64::MAX, 2).is_err());
    assert!(source.span(7, 2).is_err());
    assert!(source.read_exact_at(8, &mut [0]).is_err());
    let span = source.span(0, 8).unwrap();
    assert!(DenseTensor::new("bad", 2, 2, ScalarType::F32, span.clone()).is_err());
    assert!(DenseTensor::new("bad", 0, 2, ScalarType::F32, span.clone()).is_err());
    assert!(DenseTensor::new("bad", 1, 2, ScalarType::I32, span.clone()).is_err());
    assert!(span.read_exact_at(u64::MAX, &mut [0]).is_err());
}

#[test]
fn dense_half_types_widen_their_own_bits_exactly() {
    for (dtype, bits, expected) in [
        (
            ScalarType::F16,
            [0x3c01u16, 0xbc01],
            [1.0009765625, -1.0009765625],
        ),
        (
            ScalarType::Bf16,
            [0x3f81u16, 0xbf81],
            [1.0078125, -1.0078125],
        ),
    ] {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        for value in bits {
            file.write_all(&value.to_le_bytes()).unwrap();
        }
        let source = FileSource::open(file.path()).unwrap();
        let span = source.span(0, 4).unwrap();
        assert_eq!(span.offset(), 0);
        assert_eq!(span.source().path(), file.path());
        let tensor = DenseTensor::new("half", 1, 2, dtype, span).unwrap();
        let mut out = [0.0; 2];
        tensor.decode_row(0, 0, &mut out).unwrap();
        assert_eq!(out, expected);
    }
}
