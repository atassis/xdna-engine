//! Source-preserving CPU weight views and bounded streaming into caller-owned storage.
//!
//! These readers do not quantize, bake checkpoints, allocate device buffers or select models.
//! Sources must remain unchanged while open. The file handle is retained by each span, so views
//! outlive container readers. Output publication/rollback belongs to the caller: a failed write
//! may leave a partial destination. Local decode is a reference operation, not a model export.
#![doc = include_str!("../../README.md")]

use anyhow::{ensure, Context, Result};
use half::{bf16, f16};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

pub mod awq;
pub mod ct;
pub mod gguf;
pub mod plan;
pub mod safetensors;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScalarType {
    F32,
    F16,
    Bf16,
    I32,
    /// Fixed-width integer metadata storage (for example a safetensors `weight_shape` tensor).
    /// It is never decoded as a floating weight.
    I64,
}
impl ScalarType {
    pub fn byte_width(self) -> usize {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::F16 | Self::Bf16 => 2,
            Self::I64 => 8,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Encoding {
    Dense(ScalarType),
    GgufQ4_0,
    GgufQ4_1,
    GgufQ6K,
    AwqGemm {
        group_size: usize,
        scale_type: ScalarType,
    },
    CtPackQuantized {
        group_size: usize,
        scale_type: ScalarType,
    },
}

#[derive(Clone, Debug)]
pub struct TensorDescriptor {
    pub name: String,
    pub rows: usize,
    pub cols: usize,
    pub encoding: Encoding,
}

#[derive(Clone, Copy, Debug)]
pub struct Limits {
    pub max_header_bytes: usize,
    pub max_tensors: usize,
    pub max_metadata_entries: usize,
    pub max_string_bytes: usize,
}
impl Default for Limits {
    fn default() -> Self {
        Self {
            max_header_bytes: 16 * 1024 * 1024,
            max_tensors: 100_000,
            max_metadata_entries: 100_000,
            max_string_bytes: 8 * 1024 * 1024,
        }
    }
}

#[derive(Clone, Debug)]
pub struct FileSource {
    file: Arc<Mutex<File>>,
    len: u64,
    path: PathBuf,
}
impl FileSource {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path.as_ref())
            .with_context(|| format!("open weight source {:?}", path.as_ref()))?;
        ensure!(
            file.metadata()?.is_file(),
            "weight source is not a regular file"
        );
        let len = file.metadata()?.len();
        Ok(Self {
            file: Arc::new(Mutex::new(file)),
            len,
            path: path.as_ref().to_path_buf(),
        })
    }
    /// The caller-supplied source location, not a content fingerprint or a pinned revision.
    pub fn path(&self) -> &Path {
        &self.path
    }
    pub fn len(&self) -> u64 {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn read_exact_at(&self, offset: u64, out: &mut [u8]) -> Result<()> {
        let end = offset
            .checked_add(u64::try_from(out.len())?)
            .context("source range overflow")?;
        ensure!(
            end <= self.len,
            "source range [{offset}, {end}) exceeds {}",
            self.len
        );
        let mut file = self
            .file
            .lock()
            .map_err(|_| anyhow::anyhow!("source file lock poisoned"))?;
        file.seek(SeekFrom::Start(offset))?;
        file.read_exact(out)
            .context("truncated or unreadable weight source")
    }
    pub fn span(&self, offset: u64, len: u64) -> Result<FileSpan> {
        ensure!(
            offset.checked_add(len).context("span overflow")? <= self.len,
            "span exceeds source"
        );
        Ok(FileSpan {
            source: self.clone(),
            offset,
            len,
        })
    }
}

#[derive(Clone, Debug)]
pub struct FileSpan {
    source: FileSource,
    offset: u64,
    len: u64,
}
impl FileSpan {
    /// Map the retained inode's span without reopening its path or exposing the shared seek cursor.
    ///
    /// # Safety
    /// The source file must not be modified or truncated for the returned mapping's lifetime.
    pub unsafe fn mmap(&self) -> Result<memmap2::Mmap> {
        ensure!(!self.is_empty(), "cannot map an empty source span");
        let file = self
            .source
            .file
            .lock()
            .map_err(|_| anyhow::anyhow!("source file lock poisoned"))?;
        unsafe {
            memmap2::MmapOptions::new()
                .offset(self.offset)
                .len(usize::try_from(self.len)?)
                .map(&*file)
        }
        .context("map weight source span")
    }

    pub fn source(&self) -> &FileSource {
        &self.source
    }
    pub fn offset(&self) -> u64 {
        self.offset
    }
    pub fn len(&self) -> u64 {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn read_exact_at(&self, relative: u64, out: &mut [u8]) -> Result<()> {
        ensure!(
            relative
                .checked_add(u64::try_from(out.len())?)
                .context("span read overflow")?
                <= self.len,
            "read exceeds tensor span"
        );
        self.source.read_exact_at(
            self.offset
                .checked_add(relative)
                .context("absolute offset overflow")?,
            out,
        )
    }
    pub fn copy_to(&self, out: &mut dyn Write, scratch: &mut [u8]) -> Result<()> {
        ensure!(!scratch.is_empty(), "streaming scratch must not be empty");
        let mut offset = 0;
        while offset < self.len {
            let n = (self.len - offset).min(scratch.len() as u64) as usize;
            self.read_exact_at(offset, &mut scratch[..n])?;
            out.write_all(&scratch[..n])?;
            offset += n as u64;
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct Section {
    pub name: String,
    pub span: FileSpan,
}

pub trait WeightTensor {
    fn descriptor(&self) -> &TensorDescriptor;
    fn sections(&self) -> &[Section];
    fn decode_row(&self, row: usize, column: usize, out: &mut [f32]) -> Result<()>;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WrittenSection {
    pub name: String,
    pub offset: u64,
    pub len: u64,
}

/// Copy encoded sections unchanged. Placements are relative to the start of this write, not a
/// seek position; retain the descriptor/codec alongside them to interpret the destination.
pub fn write_encoded(
    tensor: &dyn WeightTensor,
    out: &mut dyn Write,
    scratch: &mut [u8],
) -> Result<Vec<WrittenSection>> {
    ensure!(!scratch.is_empty(), "streaming scratch must not be empty");
    let mut offset = 0u64;
    let mut placed = Vec::with_capacity(tensor.sections().len());
    for section in tensor.sections() {
        let end = offset
            .checked_add(section.span.len())
            .context("destination extent overflow")?;
        section.span.copy_to(out, scratch)?;
        placed.push(WrittenSection {
            name: section.name.clone(),
            offset,
            len: section.span.len(),
        });
        offset = end;
    }
    Ok(placed)
}

pub fn validate_row(d: &TensorDescriptor, row: usize, column: usize, len: usize) -> Result<()> {
    ensure!(row < d.rows, "{}: row {row} exceeds {}", d.name, d.rows);
    ensure!(
        column.checked_add(len).context("row slice overflow")? <= d.cols,
        "{}: column slice exceeds {}",
        d.name,
        d.cols
    );
    Ok(())
}

#[derive(Clone, Debug)]
pub struct DenseTensor {
    descriptor: TensorDescriptor,
    sections: Vec<Section>,
    dtype: ScalarType,
}
impl DenseTensor {
    pub fn new(
        name: impl Into<String>,
        rows: usize,
        cols: usize,
        dtype: ScalarType,
        span: FileSpan,
    ) -> Result<Self> {
        ensure!(
            rows > 0 && cols > 0,
            "dense tensor dimensions must be positive"
        );
        ensure!(
            !matches!(dtype, ScalarType::I32 | ScalarType::I64),
            "integer scalars are encoded metadata, not a dense floating tensor"
        );
        let bytes = rows
            .checked_mul(cols)
            .and_then(|x| x.checked_mul(dtype.byte_width()))
            .context("dense tensor size overflow")?;
        ensure!(
            span.len() == u64::try_from(bytes)?,
            "dense tensor byte length mismatch"
        );
        Ok(Self {
            descriptor: TensorDescriptor {
                name: name.into(),
                rows,
                cols,
                encoding: Encoding::Dense(dtype),
            },
            sections: vec![Section {
                name: "data".into(),
                span,
            }],
            dtype,
        })
    }
}
impl WeightTensor for DenseTensor {
    fn descriptor(&self) -> &TensorDescriptor {
        &self.descriptor
    }
    fn sections(&self) -> &[Section] {
        &self.sections
    }
    fn decode_row(&self, row: usize, column: usize, out: &mut [f32]) -> Result<()> {
        validate_row(&self.descriptor, row, column, out.len())?;
        let width = self.dtype.byte_width();
        let first = row * self.descriptor.cols + column;
        for (i, value) in out.iter_mut().enumerate() {
            let mut raw = [0u8; 4];
            self.sections[0]
                .span
                .read_exact_at(((first + i) * width) as u64, &mut raw[..width])?;
            *value = match self.dtype {
                ScalarType::F32 => f32::from_le_bytes(raw),
                ScalarType::F16 => f16::from_bits(u16::from_le_bytes([raw[0], raw[1]])).to_f32(),
                ScalarType::Bf16 => bf16::from_bits(u16::from_le_bytes([raw[0], raw[1]])).to_f32(),
                ScalarType::I32 | ScalarType::I64 => {
                    unreachable!("the constructor rejects integer dense tensors")
                }
            };
        }
        Ok(())
    }
}
