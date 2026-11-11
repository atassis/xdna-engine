//! Source-preserving decoder for AutoAWQ's unsigned GEMM-packed 4-bit weights.
//!
//! The source tensors are `qweight[K, N/8]`, `qzeros[K/group_size, N/8]`, and
//! `scales[K/group_size, N]`.  This view exposes the logical matrix as `[N, K]`:
//! a requested output row is one logical output channel and its columns are input
//! channels.  It retains the three original source spans unchanged; decoding is
//! intentionally a bounded reference operation rather than a materialization.
use super::safetensors::TensorEntry;
use super::{
    validate_row, Encoding, FileSpan, ScalarType, Section, TensorDescriptor, WeightTensor,
};
use anyhow::{ensure, Context, Result};
use half::{bf16, f16};

/// AutoAWQ GEMM writes logical output lanes into nibble slots in this order.
const SLOT_TO_LANE: [usize; 8] = [0, 2, 4, 6, 1, 3, 5, 7];
/// Maps a logical output lane to its containing nibble slot.
const LANE_TO_SLOT: [usize; 8] = [0, 4, 1, 5, 2, 6, 3, 7];

#[derive(Clone, Debug)]
pub struct AwqTensor {
    descriptor: TensorDescriptor,
    sections: Vec<Section>,
    qweight: FileSpan,
    qzeros: FileSpan,
    scales: FileSpan,
    packed_cols: usize,
    group_size: usize,
    scale_type: ScalarType,
}

impl AwqTensor {
    /// Create a view over one explicitly selected AutoAWQ GEMM tensor triple.
    ///
    /// This constructor deliberately does not infer AWQ from a model label or
    /// choose companion tensors.  The caller supplies the exact source triple
    /// and group size, which are recorded in the descriptor's `AwqGemm` encoding.
    pub fn new(
        name: impl Into<String>,
        qweight: TensorEntry,
        qzeros: TensorEntry,
        scales: TensorEntry,
        group_size: usize,
    ) -> Result<Self> {
        ensure!(group_size > 0, "AWQ group size must be positive");
        ensure!(qweight.shape.len() == 2, "AWQ qweight must be rank 2");
        ensure!(qzeros.shape.len() == 2, "AWQ qzeros must be rank 2");
        ensure!(scales.shape.len() == 2, "AWQ scales must be rank 2");
        ensure!(
            qweight.dtype == ScalarType::I32,
            "AWQ qweight must have I32 words"
        );
        ensure!(
            qzeros.dtype == ScalarType::I32,
            "AWQ qzeros must have I32 words"
        );
        ensure!(
            matches!(
                scales.dtype,
                ScalarType::F16 | ScalarType::Bf16 | ScalarType::F32
            ),
            "AWQ scales must have F16, BF16, or F32 elements"
        );

        validate_entry("AWQ qweight", &qweight)?;
        validate_entry("AWQ qzeros", &qzeros)?;
        validate_entry("AWQ scales", &scales)?;

        let k = qweight.shape[0];
        let packed_cols = qweight.shape[1];
        ensure!(
            k > 0 && packed_cols > 0,
            "AWQ qweight dimensions must be positive"
        );
        ensure!(
            k % group_size == 0,
            "AWQ K dimension must be divisible by group size"
        );
        let n = packed_cols
            .checked_mul(8)
            .context("AWQ N dimension overflow")?;
        let groups = k / group_size;
        ensure!(
            qzeros.shape == [groups, packed_cols],
            "AWQ qzeros shape must be [K/group_size, N/8]"
        );
        ensure!(
            scales.shape == [groups, n],
            "AWQ scales shape must be [K/group_size, N]"
        );

        let scale_type = scales.dtype;
        let descriptor = TensorDescriptor {
            name: name.into(),
            rows: n,
            cols: k,
            encoding: Encoding::AwqGemm {
                group_size,
                scale_type,
            },
        };
        let sections = vec![
            Section {
                name: "qweight".into(),
                span: qweight.span.clone(),
            },
            Section {
                name: "qzeros".into(),
                span: qzeros.span.clone(),
            },
            Section {
                name: "scales".into(),
                span: scales.span.clone(),
            },
        ];
        Ok(Self {
            descriptor,
            sections,
            qweight: qweight.span,
            qzeros: qzeros.span,
            scales: scales.span,
            packed_cols,
            group_size,
            scale_type,
        })
    }

    fn read_word(span: &FileSpan, element: usize) -> Result<u32> {
        let offset = element.checked_mul(4).context("AWQ word offset overflow")?;
        let mut bytes = [0u8; 4];
        span.read_exact_at(u64::try_from(offset)?, &mut bytes)?;
        Ok(u32::from_le_bytes(bytes))
    }

    fn read_scale(&self, element: usize) -> Result<f32> {
        let width = self.scale_type.byte_width();
        let offset = element
            .checked_mul(width)
            .context("AWQ scale offset overflow")?;
        let mut bytes = [0u8; 4];
        self.scales
            .read_exact_at(u64::try_from(offset)?, &mut bytes[..width])?;
        Ok(match self.scale_type {
            ScalarType::F16 => f16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32(),
            ScalarType::Bf16 => bf16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32(),
            ScalarType::F32 => f32::from_le_bytes(bytes),
            ScalarType::I32 => unreachable!("AWQ constructor rejects I32 scales"),
        })
    }
}

impl WeightTensor for AwqTensor {
    fn descriptor(&self) -> &TensorDescriptor {
        &self.descriptor
    }

    fn sections(&self) -> &[Section] {
        &self.sections
    }

    fn decode_row(&self, row: usize, column: usize, out: &mut [f32]) -> Result<()> {
        validate_row(&self.descriptor, row, column, out.len())?;
        let packed_column = row / 8;
        let slot = LANE_TO_SLOT[row % 8];
        debug_assert_eq!(SLOT_TO_LANE[slot], row % 8);
        for (offset, value) in out.iter_mut().enumerate() {
            let k = column + offset;
            let group = k / self.group_size;
            let qweight_element = k
                .checked_mul(self.packed_cols)
                .and_then(|base| base.checked_add(packed_column))
                .context("AWQ qweight index overflow")?;
            let qzero_element = group
                .checked_mul(self.packed_cols)
                .and_then(|base| base.checked_add(packed_column))
                .context("AWQ qzeros index overflow")?;
            let scale_element = group
                .checked_mul(self.descriptor.rows)
                .and_then(|base| base.checked_add(row))
                .context("AWQ scale index overflow")?;
            let q = (Self::read_word(&self.qweight, qweight_element)? >> (slot * 4)) & 0x0f;
            let z = (Self::read_word(&self.qzeros, qzero_element)? >> (slot * 4)) & 0x0f;
            let scale = self.read_scale(scale_element)?;
            *value = (q as i32 - z as i32) as f32 * scale;
        }
        Ok(())
    }
}

fn validate_entry(label: &str, entry: &TensorEntry) -> Result<()> {
    let elements = entry
        .shape
        .iter()
        .try_fold(1usize, |count, dim| count.checked_mul(*dim))
        .with_context(|| format!("{label} element count overflow"))?;
    let bytes = elements
        .checked_mul(entry.dtype.byte_width())
        .with_context(|| format!("{label} byte length overflow"))?;
    ensure!(
        entry.span.len() == u64::try_from(bytes)?,
        "{label} byte length does not match dtype and shape"
    );
    Ok(())
}
