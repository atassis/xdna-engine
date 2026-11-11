//! Bounded, source-preserving GGUF v3 reader for the CPU weight path.
//!
//! GGUF offsets are relative to the aligned data section.  The parser retains only tensor
//! descriptors and source spans; metadata values (including arrays) are streamed past.
use super::{
    validate_row, Encoding, FileSource, FileSpan, Limits, Section, TensorDescriptor, WeightTensor,
};
use anyhow::{bail, ensure, Context, Result};
use half::f16;
use std::collections::{BTreeMap, HashSet};
use std::path::Path;

const GGUF_V3: u32 = 3;
const DEFAULT_ALIGNMENT: u64 = 32;
const TYPE_F32: u32 = 0;
const TYPE_Q4_0: u32 = 2;
const TYPE_Q4_1: u32 = 3;
const TYPE_Q6_K: u32 = 14;

#[derive(Clone, Debug)]
struct Codec {
    encoding: Encoding,
    block_elements: usize,
    block_bytes: usize,
}

fn codec(type_id: u32, name: &str) -> Result<Codec> {
    match type_id {
        TYPE_F32 => Ok(Codec {
            encoding: Encoding::Dense(super::ScalarType::F32),
            block_elements: 1,
            block_bytes: 4,
        }),
        TYPE_Q4_0 => Ok(Codec {
            encoding: Encoding::GgufQ4_0,
            block_elements: 32,
            block_bytes: 18,
        }),
        TYPE_Q4_1 => Ok(Codec {
            encoding: Encoding::GgufQ4_1,
            block_elements: 32,
            block_bytes: 20,
        }),
        TYPE_Q6_K => Ok(Codec {
            encoding: Encoding::GgufQ6K,
            block_elements: 256,
            block_bytes: 210,
        }),
        other => bail!("{name}: unsupported GGUF tensor type {other}"),
    }
}

/// A GGUF tensor remains readable after its containing [`GgufFile`] is dropped.
#[derive(Clone, Debug)]
pub struct GgufTensor {
    descriptor: TensorDescriptor,
    sections: Vec<Section>,
    codec: Codec,
}

impl GgufTensor {
    fn new(name: String, rows: usize, cols: usize, codec: Codec, span: FileSpan) -> Self {
        Self {
            descriptor: TensorDescriptor {
                name,
                rows,
                cols,
                encoding: codec.encoding.clone(),
            },
            sections: vec![Section {
                name: "data".into(),
                span,
            }],
            codec,
        }
    }

    fn decode_quantized_lane(&self, block: &[u8], lane: usize) -> f32 {
        match &self.codec.encoding {
            Encoding::GgufQ4_0 => {
                let d = f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
                let q = if lane < 16 {
                    block[2 + lane] & 0x0f
                } else {
                    block[2 + lane - 16] >> 4
                };
                d * (q as i32 - 8) as f32
            }
            Encoding::GgufQ4_1 => {
                let d = f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
                let m = f16::from_bits(u16::from_le_bytes([block[2], block[3]])).to_f32();
                let q = if lane < 16 {
                    block[4 + lane] & 0x0f
                } else {
                    block[4 + lane - 16] >> 4
                };
                d * q as f32 + m
            }
            Encoding::GgufQ6K => {
                // Official ggml block_q6_K layout: ql[128], qh[64], scales[16], d.
                // qh packs the two high bits for four lanes; scales are signed int8.
                let group = lane / 128;
                let l = lane % 32;
                let quarter = (lane % 128) / 32;
                let ql_base = group * 64;
                let qh = block[128 + group * 32 + l];
                let low = match quarter {
                    0 | 2 => block[ql_base + l],
                    1 | 3 => block[ql_base + 32 + l],
                    _ => unreachable!(),
                };
                let low_nibble = if quarter < 2 { low & 0x0f } else { low >> 4 };
                let high = (qh >> (quarter * 2)) & 0x03;
                let q = (low_nibble | (high << 4)) as i32 - 32;
                let scale_index = group * 8 + (l / 16) + quarter * 2;
                let scale = block[192 + scale_index] as i8 as f32;
                let d = f16::from_bits(u16::from_le_bytes([block[208], block[209]])).to_f32();
                d * scale * q as f32
            }
            _ => unreachable!("only quantized codecs call decode_quantized_lane"),
        }
    }
}

impl WeightTensor for GgufTensor {
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
        let cols = self.descriptor.cols;
        let row_base = row.checked_mul(cols).context("GGUF row offset overflow")?;
        if self.codec.block_elements == 1 {
            for (i, value) in out.iter_mut().enumerate() {
                let element = row_base
                    .checked_add(column)
                    .and_then(|x| x.checked_add(i))
                    .context("GGUF element offset overflow")?;
                let offset = element
                    .checked_mul(4)
                    .context("GGUF F32 byte offset overflow")?;
                let mut raw = [0u8; 4];
                self.sections[0]
                    .span
                    .read_exact_at(u64::try_from(offset)?, &mut raw)?;
                *value = f32::from_le_bytes(raw);
            }
            return Ok(());
        }

        let first_block = column / self.codec.block_elements;
        let last_column = column
            .checked_add(out.len() - 1)
            .context("GGUF row slice overflow")?;
        let last_block = last_column / self.codec.block_elements;
        let blocks_per_row = cols / self.codec.block_elements;
        let mut scratch = [0u8; 210];
        for block_in_row in first_block..=last_block {
            let ordinal = row
                .checked_mul(blocks_per_row)
                .and_then(|x| x.checked_add(block_in_row))
                .context("GGUF block ordinal overflow")?;
            let byte_offset = ordinal
                .checked_mul(self.codec.block_bytes)
                .context("GGUF block offset overflow")?;
            self.sections[0].span.read_exact_at(
                u64::try_from(byte_offset)?,
                &mut scratch[..self.codec.block_bytes],
            )?;
            let lane_start = if block_in_row == first_block {
                column % self.codec.block_elements
            } else {
                0
            };
            let lane_end = if block_in_row == last_block {
                last_column % self.codec.block_elements + 1
            } else {
                self.codec.block_elements
            };
            for lane in lane_start..lane_end {
                let output = block_in_row
                    .checked_mul(self.codec.block_elements)
                    .and_then(|x| x.checked_add(lane))
                    .and_then(|x| x.checked_sub(column))
                    .context("GGUF output offset overflow")?;
                out[output] = self.decode_quantized_lane(&scratch, lane);
            }
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct GgufFile {
    tensors: BTreeMap<String, GgufTensor>,
}
impl GgufFile {
    pub fn open(path: impl AsRef<Path>, limits: Limits) -> Result<Self> {
        let source = FileSource::open(path)?;
        let mut reader = HeaderReader::new(&source, limits)?;
        let mut magic = [0u8; 4];
        reader.read_into(&mut magic)?;
        ensure!(&magic == b"GGUF", "not a GGUF file");
        ensure!(
            reader.u32()? == GGUF_V3,
            "only GGUF v3 little-endian is supported"
        );
        let raw_tensor_count = reader.u64()?;
        let tensor_count = reader.usize_count(raw_tensor_count, limits.max_tensors, "tensor")?;
        let raw_metadata_count = reader.u64()?;
        let metadata_count =
            reader.usize_count(raw_metadata_count, limits.max_metadata_entries, "metadata")?;

        let mut metadata_keys = HashSet::with_capacity(metadata_count);
        let mut alignment = DEFAULT_ALIGNMENT;
        for _ in 0..metadata_count {
            let key = reader.string()?;
            ensure!(
                metadata_keys.insert(key.clone()),
                "duplicate GGUF metadata key {key:?}"
            );
            let value_type = reader.u32()?;
            let alignment_value = reader.skip_value(value_type)?;
            if key == "general.alignment" {
                ensure!(
                    value_type == 4,
                    "general.alignment must have GGUF UINT32 type"
                );
                alignment =
                    alignment_value.context("general.alignment must be an unsigned integer")?;
            }
        }
        ensure!(
            alignment.is_power_of_two() && alignment > 0,
            "invalid GGUF alignment {alignment}"
        );

        struct Pending {
            name: String,
            rows: usize,
            cols: usize,
            codec: Codec,
            offset: u64,
            bytes: u64,
        }
        let mut pending = Vec::with_capacity(tensor_count);
        let mut tensor_names = HashSet::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let name = reader.string()?;
            ensure!(
                tensor_names.insert(name.clone()),
                "duplicate GGUF tensor name {name:?}"
            );
            let rank = reader.u32()?;
            ensure!(
                rank == 1 || rank == 2,
                "{name}: unsupported GGUF tensor rank {rank}"
            );
            let raw_cols = reader.u64()?;
            let cols = reader.usize_dimension(raw_cols, &name)?;
            let rows = if rank == 2 {
                let raw_rows = reader.u64()?;
                reader.usize_dimension(raw_rows, &name)?
            } else {
                1
            };
            let type_id = reader.u32()?;
            let codec = codec(type_id, &name)?;
            ensure!(
                cols > 0 && rows > 0,
                "{name}: tensor dimensions must be positive"
            );
            ensure!(
                codec.block_elements == 1 || cols % codec.block_elements == 0,
                "{name}: partial source quantization block per row"
            );
            let blocks = cols
                .checked_div(codec.block_elements)
                .and_then(|x| x.checked_mul(rows))
                .context("GGUF tensor block count overflow")?;
            let bytes = blocks
                .checked_mul(codec.block_bytes)
                .context("GGUF tensor byte size overflow")?;
            let offset = reader.u64()?;
            ensure!(
                offset % alignment == 0,
                "{name}: tensor offset is not aligned"
            );
            pending.push(Pending {
                name,
                rows,
                cols,
                codec,
                offset,
                bytes: u64::try_from(bytes)?,
            });
        }
        let data_start = align_up(reader.position, alignment)?;
        ensure!(data_start <= source.len(), "truncated GGUF header padding");

        let mut ranges = Vec::with_capacity(pending.len());
        let mut tensors = BTreeMap::new();
        for entry in pending {
            let absolute = data_start
                .checked_add(entry.offset)
                .context("GGUF tensor offset overflow")?;
            let end = absolute
                .checked_add(entry.bytes)
                .context("GGUF tensor extent overflow")?;
            ensure!(
                end <= source.len(),
                "{}: truncated GGUF tensor data",
                entry.name
            );
            ranges.push((absolute, end, entry.name.clone()));
            let span = source.span(absolute, entry.bytes)?;
            tensors.insert(
                entry.name.clone(),
                GgufTensor::new(entry.name, entry.rows, entry.cols, entry.codec, span),
            );
        }
        ranges.sort_unstable_by_key(|range| range.0);
        let mut previous_end = data_start;
        for (start, end, name) in ranges {
            ensure!(
                start >= previous_end,
                "{name}: GGUF tensor data overlaps a previous tensor"
            );
            previous_end = end;
        }
        Ok(Self { tensors })
    }

    pub fn tensors(&self) -> &BTreeMap<String, GgufTensor> {
        &self.tensors
    }
    pub fn get(&self, name: &str) -> Result<&GgufTensor> {
        self.tensors
            .get(name)
            .with_context(|| format!("missing GGUF tensor {name:?}"))
    }
}

struct HeaderReader<'a> {
    source: &'a FileSource,
    position: u64,
    end: u64,
    limits: Limits,
}
impl<'a> HeaderReader<'a> {
    fn new(source: &'a FileSource, limits: Limits) -> Result<Self> {
        let end = 24u64
            .checked_add(u64::try_from(limits.max_header_bytes)?)
            .context("GGUF header limit overflow")?;
        Ok(Self {
            source,
            position: 0,
            end: end.min(source.len()),
            limits,
        })
    }
    fn read_into(&mut self, out: &mut [u8]) -> Result<()> {
        let next = self
            .position
            .checked_add(u64::try_from(out.len())?)
            .context("GGUF header offset overflow")?;
        ensure!(
            next <= self.end,
            "GGUF header exceeds configured byte limit or is truncated"
        );
        self.source.read_exact_at(self.position, out)?;
        self.position = next;
        Ok(())
    }
    fn u8(&mut self) -> Result<u8> {
        let mut b = [0; 1];
        self.read_into(&mut b)?;
        Ok(b[0])
    }
    fn u16(&mut self) -> Result<u16> {
        let mut b = [0; 2];
        self.read_into(&mut b)?;
        Ok(u16::from_le_bytes(b))
    }
    fn u32(&mut self) -> Result<u32> {
        let mut b = [0; 4];
        self.read_into(&mut b)?;
        Ok(u32::from_le_bytes(b))
    }
    fn u64(&mut self) -> Result<u64> {
        let mut b = [0; 8];
        self.read_into(&mut b)?;
        Ok(u64::from_le_bytes(b))
    }
    fn skip(&mut self, len: u64) -> Result<()> {
        let next = self
            .position
            .checked_add(len)
            .context("GGUF header offset overflow")?;
        ensure!(
            next <= self.end,
            "GGUF header exceeds configured byte limit or is truncated"
        );
        self.position = next;
        Ok(())
    }
    fn string(&mut self) -> Result<String> {
        let len = self.u64()?;
        ensure!(
            len <= u64::try_from(self.limits.max_string_bytes)?,
            "GGUF string exceeds configured limit"
        );
        let size = usize::try_from(len).context("GGUF string length not addressable")?;
        let mut bytes = vec![0; size];
        self.read_into(&mut bytes)?;
        String::from_utf8(bytes).context("GGUF string is not UTF-8")
    }
    fn skip_string(&mut self) -> Result<()> {
        let len = self.u64()?;
        ensure!(
            len <= u64::try_from(self.limits.max_string_bytes)?,
            "GGUF string exceeds configured limit"
        );
        self.skip(len)
    }
    fn usize_count(&self, value: u64, limit: usize, kind: &str) -> Result<usize> {
        let count =
            usize::try_from(value).with_context(|| format!("GGUF {kind} count not addressable"))?;
        ensure!(count <= limit, "GGUF {kind} count exceeds configured limit");
        Ok(count)
    }
    fn usize_dimension(&self, value: u64, name: &str) -> Result<usize> {
        usize::try_from(value).with_context(|| format!("{name}: GGUF dimension not addressable"))
    }
    /// Skip one GGUF metadata value. Arrays are streamed element-by-element and cannot nest.
    /// The returned value is only used to validate `general.alignment`.
    fn skip_value(&mut self, value_type: u32) -> Result<Option<u64>> {
        match value_type {
            0 => Ok(Some(self.u8()? as u64)),
            1 => {
                self.u8()?;
                Ok(None)
            }
            2 => Ok(Some(self.u16()? as u64)),
            3 => {
                self.u16()?;
                Ok(None)
            }
            4 => Ok(Some(self.u32()? as u64)),
            5 | 6 => {
                self.u32()?;
                Ok(None)
            }
            7 => {
                let value = self.u8()?;
                ensure!(value <= 1, "malformed GGUF boolean");
                Ok(None)
            }
            8 => {
                self.skip_string()?;
                Ok(None)
            }
            9 => {
                let element_type = self.u32()?;
                let min_width = match element_type {
                    0 | 1 | 7 => 1,
                    2 | 3 => 2,
                    4 | 5 | 6 => 4,
                    8 | 10 | 11 | 12 => 8,
                    9 => bail!("nested GGUF metadata arrays are unsupported"),
                    other => bail!("unknown GGUF array element type {other}"),
                };
                let count = self.u64()?;
                // Token vocabularies can exceed the top-level metadata-key limit. Bound their
                // count by remaining header bytes and the minimum source element width instead.
                ensure!(
                    count <= (self.end - self.position) / min_width,
                    "GGUF metadata array exceeds remaining header budget"
                );
                if element_type == 8 {
                    for _ in 0..count {
                        self.skip_string()?;
                    }
                } else if element_type == 7 {
                    let mut remaining = count;
                    let mut scratch = [0; 4096];
                    while remaining > 0 {
                        let n = remaining.min(scratch.len() as u64) as usize;
                        self.read_into(&mut scratch[..n])?;
                        ensure!(
                            scratch[..n].iter().all(|v| *v <= 1),
                            "malformed GGUF boolean array"
                        );
                        remaining -= n as u64;
                    }
                } else {
                    self.skip(
                        count
                            .checked_mul(min_width)
                            .context("GGUF array byte size overflow")?,
                    )?;
                }
                Ok(None)
            }
            10 => Ok(Some(self.u64()?)),
            11 | 12 => {
                self.u64()?;
                Ok(None)
            }
            other => bail!("unknown GGUF metadata value type {other}"),
        }
    }
}

fn align_up(value: u64, alignment: u64) -> Result<u64> {
    value
        .checked_add(alignment - 1)
        .map(|x| x & !(alignment - 1))
        .context("GGUF data alignment overflow")
}
