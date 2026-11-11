//! Graph-owned BF16 input requirements, bound once to lazy checkpoint spans.
//! Operations concatenate in declaration order; offsets passed to the sink are buffer-relative.
//! Failed execution may leave a partial destination. Sources must remain immutable while bound.
use super::{safetensors::TensorEntry, FileSpan, ScalarType};
use anyhow::{ensure, Context, Result};
use half::bf16;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TensorInput {
    pub tensor: String,
    pub shape: Vec<usize>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum WeightOp {
    Rows {
        tensor: String,
        shape: Vec<usize>,
        first: usize,
        count: usize,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        scale: Option<f32>,
    },
    Zero {
        bytes: usize,
    },
}

/// Physical requirements emitted at the graph's transformation sites. Tensor names are the
/// architecture's semantic bindings, not checkpoint paths or weight-content fingerprints.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WeightContract {
    pub version: u32,
    pub config: BTreeMap<String, serde_json::Value>,
    pub buffers: BTreeMap<String, Vec<WeightOp>>,
    pub embedding: TensorInput,
}

#[derive(Debug)]
enum BoundOp {
    Rows { span: FileSpan, scale: Option<f32> },
    Zero(usize),
}

#[derive(Debug)]
pub struct WeightPlan {
    buffers: BTreeMap<String, Vec<BoundOp>>,
    embedding: TensorEntry,
}

fn checked_tensor<'a>(
    tensors: &'a BTreeMap<String, TensorEntry>,
    name: &str,
    shape: &[usize],
) -> Result<&'a TensorEntry> {
    let tensor = tensors
        .get(name)
        .with_context(|| format!("missing tensor {name}"))?;
    ensure!(
        !shape.is_empty() && shape.len() <= 2 && shape.iter().all(|&n| n > 0),
        "{name}: expected positive rank-1/2 shape"
    );
    ensure!(
        tensor.dtype == ScalarType::Bf16,
        "{name}: BF16 source required, got {:?}",
        tensor.dtype
    );
    ensure!(
        tensor.shape == shape,
        "{name}: shape {:?}, expected {shape:?}",
        tensor.shape
    );
    let bytes = shape
        .iter()
        .try_fold(2usize, |p, n| p.checked_mul(*n))
        .context("tensor extent overflow")?;
    ensure!(
        u64::try_from(bytes)? == tensor.span.len(),
        "{name}: tensor byte extent mismatch"
    );
    Ok(tensor)
}

impl WeightContract {
    /// Parse the embedded ABI without allowing duplicate JSON keys to overwrite requirements.
    pub fn from_metadata_json(bytes: &[u8]) -> Result<Option<Self>> {
        let mut metadata = super::safetensors::parse_unique(bytes)?;
        metadata
            .as_object_mut()
            .context("artifact metadata must be an object")?
            .remove("weight_inputs")
            .map(serde_json::from_value)
            .transpose()
            .map_err(Into::into)
    }
}

impl WeightPlan {
    pub fn bind_config_json(
        contract: &WeightContract,
        tensors: &BTreeMap<String, TensorEntry>,
        expected: &BTreeMap<String, usize>,
        config: &[u8],
    ) -> Result<Self> {
        let config = super::safetensors::parse_unique(config)?;
        Self::bind(contract, tensors, expected, &config)
    }

    /// Validate all names, shapes, scalar semantics, selections and output lengths before writing.
    /// `expected` comes from the validated program layout, not from this contract's operations.
    pub fn bind(
        contract: &WeightContract,
        tensors: &BTreeMap<String, TensorEntry>,
        expected: &BTreeMap<String, usize>,
        config: &serde_json::Value,
    ) -> Result<Self> {
        ensure!(
            !contract.config.is_empty(),
            "missing checkpoint configuration requirements"
        );
        for (key, required) in &contract.config {
            let actual = config
                .get(key)
                .with_context(|| format!("missing checkpoint config {key}"))?;
            ensure!(
                actual == required
                    || actual
                        .as_f64()
                        .zip(required.as_f64())
                        .is_some_and(|(a, b)| a == b),
                "incompatible checkpoint config {key}: got {actual}, required {required}"
            );
        }
        ensure!(
            contract.version == 1,
            "unsupported weight-input contract version {}",
            contract.version
        );
        ensure!(
            !expected.is_empty() && contract.buffers.keys().eq(expected.keys()),
            "weight-input buffer coverage mismatch"
        );
        let embedding = checked_tensor(
            tensors,
            &contract.embedding.tensor,
            &contract.embedding.shape,
        )?
        .clone();
        ensure!(embedding.shape.len() == 2, "embedding must be a matrix");
        let mut buffers = BTreeMap::new();
        for (name, ops) in &contract.buffers {
            ensure!(!ops.is_empty(), "{name}: empty weight-input operations");
            let mut bound = Vec::with_capacity(ops.len());
            let mut total = 0usize;
            for op in ops {
                let (part, len) = match op {
                    WeightOp::Zero { bytes } => {
                        ensure!(
                            *bytes > 0 && bytes % 2 == 0,
                            "{name}: zero extent must contain whole BF16 values"
                        );
                        (BoundOp::Zero(*bytes), *bytes)
                    }
                    WeightOp::Rows {
                        tensor,
                        shape,
                        first,
                        count,
                        scale,
                    } => {
                        let entry = checked_tensor(tensors, tensor, shape)?;
                        ensure!(
                            *count > 0
                                && first.checked_add(*count).context("row extent overflow")?
                                    <= shape[0],
                            "{tensor}: invalid row selection"
                        );
                        if let Some(scale) = scale {
                            ensure!(scale.is_finite(), "{tensor}: scale must be finite");
                        }
                        let width = if shape.len() == 2 {
                            shape[1].checked_mul(2).context("row width overflow")?
                        } else {
                            2
                        };
                        let start = first.checked_mul(width).context("row offset overflow")?;
                        let len = count
                            .checked_mul(width)
                            .context("selected extent overflow")?;
                        let offset = entry
                            .span
                            .offset()
                            .checked_add(u64::try_from(start)?)
                            .context("source offset overflow")?;
                        let span = entry.span.source().span(offset, u64::try_from(len)?)?;
                        (
                            BoundOp::Rows {
                                span,
                                scale: *scale,
                            },
                            len,
                        )
                    }
                };
                total = total.checked_add(len).context("output length overflow")?;
                bound.push(part);
            }
            ensure!(
                total == expected[name],
                "{name}: planned {total} bytes, program requires {}",
                expected[name]
            );
            buffers.insert(name.clone(), bound);
        }
        Ok(Self { buffers, embedding })
    }

    pub fn embedding(&self) -> &TensorEntry {
        &self.embedding
    }

    /// Caller scratch is the only payload storage. BF16 scaling is explicitly float32 multiply
    /// followed by round-to-nearest-even; copies preserve bits, including signed zeros and NaNs.
    pub fn write_buffer(
        &self,
        name: &str,
        scratch: &mut [u8],
        mut sink: impl FnMut(usize, &[u8]) -> Result<()>,
    ) -> Result<()> {
        let capacity = scratch.len() / 2 * 2;
        ensure!(
            capacity > 0,
            "weight preparation needs at least two scratch bytes"
        );
        let ops = self
            .buffers
            .get(name)
            .with_context(|| format!("unknown weight buffer {name}"))?;
        let mut output = 0usize;
        for op in ops {
            let len = match op {
                BoundOp::Rows { span, .. } => usize::try_from(span.len())?,
                BoundOp::Zero(n) => *n,
            };
            let mut offset = 0usize;
            while offset < len {
                let n = capacity.min(len - offset);
                let chunk = &mut scratch[..n];
                match op {
                    BoundOp::Zero(_) => chunk.fill(0),
                    BoundOp::Rows { span, scale } => {
                        span.read_exact_at(u64::try_from(offset)?, chunk)?;
                        if let Some(scale) = scale {
                            for bytes in chunk.chunks_exact_mut(2) {
                                let value =
                                    bf16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]]))
                                        .to_f32()
                                        * scale;
                                let rounded = bf16::from_f32(value);
                                ensure!(
                                    rounded.is_finite(),
                                    "{name}: non-finite scaled BF16 value"
                                );
                                bytes.copy_from_slice(&rounded.to_bits().to_le_bytes());
                            }
                        }
                    }
                }
                sink(output, chunk)?;
                output += n;
                offset += n;
            }
        }
        Ok(())
    }
}
