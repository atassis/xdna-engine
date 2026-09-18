//! Minimal GGUF tensor-directory reader: enough to pull named f32-valued tensors out of the S2
//! decoder and AR checkpoints. Mirrors `scripts/gguf_extract.py`'s exact byte layout (same KV-skip
//! table, same alignment handling, same `ne`-reversal convention: ggml's `ne` is fastest-dim-first,
//! and ggml's flat tensor bytes already sit in C-order for the REVERSED shape, so no data
//! permutation is needed -- only the logical shape changes) so a Rust read and a `gguf_extract.load()`
//! read of the same tensor agree by construction. Decodes F32/F16/BF16/Q6K: F32/F16/BF16 cover the
//! codec decoder (every decoder tensor in the S2-Pro checkpoint is F16, confirmed by inspection);
//! Q6K covers the AR half, where every large weight (embeddings, attention, feed-forward) is GGUF
//! type q6_k. Anything else fails loud rather than silently misreading a quantized tensor.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::S2Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GgmlType {
    F32,
    F16,
    Bf16,
    /// Block-quantized; byte size is per-block (see [`Q6K_BLOCK_BYTES`]), not per-element.
    Q6K,
}

impl GgmlType {
    fn from_id(id: u32) -> Option<Self> {
        match id {
            0 => Some(GgmlType::F32),
            1 => Some(GgmlType::F16),
            14 => Some(GgmlType::Q6K),
            30 => Some(GgmlType::Bf16),
            _ => None,
        }
    }
}

/// ggml's `QK_K`: output elements per quantized super-block, shared by every `q*_k` type.
const QK_K: usize = 256;
/// `sizeof(block_q6_K)` (`ggml/src/ggml-common.h`): 128 `ql` (low 4 bits) + 64 `qh` (high 2 bits)
/// + 16 `scales` (i8, one per 16 elements) + 2 `d` (f16 super-block scale) = 210 bytes / 256 elems.
const Q6K_BLOCK_BYTES: usize = QK_K / 2 + QK_K / 4 + QK_K / 16 + 2;

/// Dequantize `n` q6_k elements (`n % QK_K == 0`) from one or more `block_q6_K`s.
///
/// Ported from `ggml/src/ggml-quants.c:dequantize_row_q6_K` via `scripts/s2_ar_ref.py`'s
/// `q6k_dequant_blocks` (verified there byte-exact against a scalar transliteration of the same
/// C function on real GGUF bytes). Each 256-element block splits into two 128-element halves;
/// within a half, `l` in `0..32` yields 4 outputs at `l`, `l+32`, `l+64`, `l+96`, each a 6-bit
/// value (4 low bits from `ql`, 2 high bits from `qh`) reassembled and re-centered by `-32`
/// (q6_k is a signed 6-bit code, range -32..31), then scaled by `d * scales[is + {0,2,4,6}]`
/// where `is = l / 16` selects one of the two i8 sub-block scales in that half.
fn dequantize_q6_k(bytes: &[u8], n: usize) -> Vec<f32> {
    debug_assert_eq!(n % QK_K, 0, "q6_k element count must be a multiple of {QK_K}");
    let mut out = vec![0f32; n];
    for (block, y) in bytes.chunks_exact(Q6K_BLOCK_BYTES).zip(out.chunks_exact_mut(QK_K)) {
        let ql = &block[0..128];
        let qh = &block[128..192];
        let sc = &block[192..208];
        let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));
        for half in 0..2 {
            let ql_h = &ql[half * 64..half * 64 + 64];
            let qh_h = &qh[half * 32..half * 32 + 32];
            let sc_h = &sc[half * 8..half * 8 + 8];
            let y_h = &mut y[half * 128..half * 128 + 128];
            for l in 0..32 {
                let is = l / 16;
                let q1 = ((ql_h[l] & 0x0F) | ((qh_h[l] & 0x03) << 4)) as i32 - 32;
                let q2 = ((ql_h[l + 32] & 0x0F) | (((qh_h[l] >> 2) & 0x03) << 4)) as i32 - 32;
                let q3 = ((ql_h[l] >> 4) | (((qh_h[l] >> 4) & 0x03) << 4)) as i32 - 32;
                let q4 = ((ql_h[l + 32] >> 4) | (((qh_h[l] >> 6) & 0x03) << 4)) as i32 - 32;
                y_h[l] = d * (sc_h[is] as i8) as f32 * q1 as f32;
                y_h[l + 32] = d * (sc_h[is + 2] as i8) as f32 * q2 as f32;
                y_h[l + 64] = d * (sc_h[is + 4] as i8) as f32 * q3 as f32;
                y_h[l + 96] = d * (sc_h[is + 6] as i8) as f32 * q4 as f32;
            }
        }
    }
    out
}

struct TensorInfo {
    /// ggml `ne`, fastest-varying dimension first (file order, NOT numpy/row-major order).
    ne: Vec<u64>,
    /// `None` for a ggml type this reader doesn't decode (e.g. a quantized block format).
    /// Checked lazily, only when [`GgufFile::tensor_f32`] is actually called for that tensor --
    /// the S2-Pro checkpoint carries quantized tensors elsewhere in the same file (embeddings),
    /// and scanning the directory must not fail on those just because nobody asked for them.
    ty: Option<GgmlType>,
    ty_id: u32,
    /// Byte offset from `data_start`, as stored in the file.
    offset: u64,
}

pub struct GgufFile {
    path: PathBuf,
    data: Vec<u8>,
    data_start: usize,
    tensors: HashMap<String, TensorInfo>,
    /// Every UINT32 metadata KV. The header walk has to decode these anyway to find
    /// `general.alignment`, so keeping them costs nothing and saves re-parsing the file for a
    /// scalar the model declares (`fish_speech.codec.sample_rate`). Non-UINT32 KVs are skipped.
    u32_meta: HashMap<String, u32>,
}

fn err(path: &Path, msg: impl Into<String>) -> S2Error {
    S2Error::Shape(format!("{}: {}", path.display(), msg.into()))
}

struct Reader<'a> {
    d: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(d: &'a [u8]) -> Self {
        Reader { d, pos: 0 }
    }
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let s = self.d.get(self.pos..self.pos + n)?;
        self.pos += n;
        Some(s)
    }
    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }
    fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }
    fn string(&mut self) -> Option<String> {
        let n = self.u64()? as usize;
        Some(String::from_utf8_lossy(self.take(n)?).into_owned())
    }
    /// Skip one KV value of GGUF value-type `t` (0..12; see the GGUF spec / gguf_extract.py's
    /// `skip_val`). `9` (array) recurses: element type then `n` elements of that type.
    fn skip_value(&mut self, t: u32) -> Option<()> {
        let fixed = match t {
            0 | 1 | 7 => 1usize,      // uint8/int8/bool
            2 | 3 => 2,               // uint16/int16
            4 | 5 | 6 => 4,           // uint32/int32/float32
            10 | 11 | 12 => 8,        // uint64/int64/float64
            _ => 0,
        };
        if fixed > 0 {
            self.take(fixed)?;
            return Some(());
        }
        match t {
            8 => {
                self.string()?;
            }
            9 => {
                let et = self.u32()?;
                let n = self.u64()?;
                for _ in 0..n {
                    self.skip_value(et)?;
                }
            }
            _ => return None,
        }
        Some(())
    }
    /// Like `skip_value` but returns the value when `t == want_u32_type` (used only for
    /// `general.alignment`, a UINT32 KV -- every other key is skipped and discarded).
    fn read_u32_or_skip(&mut self, t: u32) -> Option<Option<u32>> {
        if t == 4 {
            return Some(Some(self.u32()?));
        }
        self.skip_value(t)?;
        Some(None)
    }
}

impl GgufFile {
    pub fn open(path: &Path) -> crate::Result<Self> {
        let data = std::fs::read(path).map_err(|e| S2Error::Io(path.to_path_buf(), e))?;
        let mut r = Reader::new(&data);
        if r.take(4) != Some(b"GGUF") {
            return Err(err(path, "missing GGUF magic"));
        }
        let _version = r.u32().ok_or_else(|| err(path, "truncated header (version)"))?;
        let n_tensors = r.u64().ok_or_else(|| err(path, "truncated header (n_tensors)"))?;
        let n_kv = r.u64().ok_or_else(|| err(path, "truncated header (n_kv)"))?;

        let mut alignment: u64 = 32;
        let mut u32_meta = HashMap::new();
        for _ in 0..n_kv {
            let key = r.string().ok_or_else(|| err(path, "truncated KV key"))?;
            let ty = r.u32().ok_or_else(|| err(path, "truncated KV type"))?;
            let v = r
                .read_u32_or_skip(ty)
                .ok_or_else(|| err(path, format!("truncated KV value for `{key}` (type {ty})")))?;
            if let Some(u) = v {
                if key == "general.alignment" {
                    alignment = u as u64;
                }
                u32_meta.insert(key, u);
            }
        }

        let mut tensors = HashMap::with_capacity(n_tensors as usize);
        for _ in 0..n_tensors {
            let name = r.string().ok_or_else(|| err(path, "truncated tensor name"))?;
            let nd = r.u32().ok_or_else(|| err(path, format!("truncated tensor `{name}` ndim")))? as usize;
            let mut ne = Vec::with_capacity(nd);
            for _ in 0..nd {
                ne.push(r.u64().ok_or_else(|| err(path, format!("truncated tensor `{name}` dims")))?);
            }
            let ty_id = r.u32().ok_or_else(|| err(path, format!("truncated tensor `{name}` type")))?;
            let offset = r.u64().ok_or_else(|| err(path, format!("truncated tensor `{name}` offset")))?;
            let ty = GgmlType::from_id(ty_id);
            tensors.insert(name, TensorInfo { ne, ty, ty_id, offset });
        }

        let pad = if alignment == 0 { 0 } else { (alignment - (r.pos as u64) % alignment) % alignment };
        let data_start = r.pos + pad as usize;

        Ok(GgufFile { path: path.to_path_buf(), data, data_start, tensors, u32_meta })
    }

    /// A UINT32 metadata value, or `None` if the model does not declare `key` (or declares it
    /// with another type). Callers must decide what an absent key means -- a default belongs at
    /// the call site with its reason, not here.
    pub fn meta_u32(&self, key: &str) -> Option<u32> {
        self.u32_meta.get(key).copied()
    }

    /// `name`'s numpy/row-major shape: ggml's `ne` reversed (ggml is fastest-dim-first).
    pub fn shape(&self, name: &str) -> crate::Result<Vec<usize>> {
        let info = self.tensors.get(name).ok_or_else(|| err(&self.path, format!("no tensor `{name}`")))?;
        Ok(info.ne.iter().rev().map(|&d| d as usize).collect())
    }

    /// Read `name` as flat f32, already in the row-major order `shape()` describes (ggml's flat
    /// bytes need no permutation to match a C-order reshape onto the reversed `ne` -- see module
    /// doc). F16/BF16/Q6K decode to f32; F32 tensors pass through.
    pub fn tensor_f32(&self, name: &str) -> crate::Result<Vec<f32>> {
        let info = self.tensors.get(name).ok_or_else(|| err(&self.path, format!("no tensor `{name}`")))?;
        let ty = info.ty.ok_or_else(|| {
            err(&self.path, format!("tensor `{name}`: unsupported ggml type id {} (only F32/F16/BF16/Q6K decode)", info.ty_id))
        })?;
        let n: u64 = info.ne.iter().product();
        let n = n as usize;
        let start = self.data_start + info.offset as usize;
        let nbytes = match ty {
            GgmlType::F32 => n * 4,
            GgmlType::F16 | GgmlType::Bf16 => n * 2,
            GgmlType::Q6K => {
                if n % QK_K != 0 {
                    return Err(err(&self.path, format!("tensor `{name}`: {n} elements not a multiple of q6_k block size {QK_K}")));
                }
                (n / QK_K) * Q6K_BLOCK_BYTES
            }
        };
        let bytes = self.data.get(start..start + nbytes).ok_or_else(|| {
            err(&self.path, format!("tensor `{name}`: {nbytes} bytes at offset {start} run past EOF ({})", self.data.len()))
        })?;
        Ok(match ty {
            GgmlType::Q6K => dequantize_q6_k(bytes, n),
            other => decode_dense(other, bytes),
        })
    }

    /// Numpy-shape rows `[row_lo, row_hi)` of a 2-D tensor, flat f32, `(row_hi-row_lo)` rows of
    /// `shape()[1]` elements each, row-major. A ggml row is `ne[0]` contiguous elements (the
    /// fastest-varying axis -- see the module doc), so this is a byte-range read, not a full
    /// decode: an embedding-table gather touches KB, not the whole table. Mirrors
    /// `s2_ar_ref.py`'s `read_tensor(..., row_range=...)`, including its q6_k precondition
    /// (`ne[0] % QK_K == 0`, true of every 2-D tensor the AR checkpoint carries).
    pub fn tensor_f32_rows(&self, name: &str, row_lo: usize, row_hi: usize) -> crate::Result<Vec<f32>> {
        let info = self.tensors.get(name).ok_or_else(|| err(&self.path, format!("no tensor `{name}`")))?;
        if info.ne.len() != 2 {
            return Err(err(&self.path, format!("tensor `{name}`: row range needs a 2-D tensor, got ne={:?}", info.ne)));
        }
        let row_elems = info.ne[0] as usize; // fastest axis = one numpy row's element count
        let n_rows = info.ne[1] as usize;
        if row_lo >= row_hi || row_hi > n_rows {
            return Err(err(&self.path, format!("tensor `{name}`: row range [{row_lo},{row_hi}) out of bounds for {n_rows} rows")));
        }
        let ty = info.ty.ok_or_else(|| {
            err(&self.path, format!("tensor `{name}`: unsupported ggml type id {} (only F32/F16/BF16/Q6K decode)", info.ty_id))
        })?;
        let nrows = row_hi - row_lo;
        let (row_bytes, first_row_offset) = match ty {
            GgmlType::F32 => (row_elems * 4, row_lo * row_elems * 4),
            GgmlType::F16 | GgmlType::Bf16 => (row_elems * 2, row_lo * row_elems * 2),
            GgmlType::Q6K => {
                if row_elems % QK_K != 0 {
                    return Err(err(&self.path, format!("tensor `{name}`: row width {row_elems} not a multiple of q6_k block size {QK_K}")));
                }
                let blocks_per_row = row_elems / QK_K;
                (blocks_per_row * Q6K_BLOCK_BYTES, row_lo * blocks_per_row * Q6K_BLOCK_BYTES)
            }
        };
        let start = self.data_start + info.offset as usize + first_row_offset;
        let nbytes = nrows * row_bytes;
        let bytes = self.data.get(start..start + nbytes).ok_or_else(|| {
            err(&self.path, format!("tensor `{name}`: {nbytes} bytes at offset {start} run past EOF ({})", self.data.len()))
        })?;
        Ok(match ty {
            GgmlType::Q6K => dequantize_q6_k(bytes, nrows * row_elems),
            other => decode_dense(other, bytes),
        })
    }
}

/// Flat f32 decode for the three non-block-quantized types (`Q6K` goes through
/// [`dequantize_q6_k`] instead, since it isn't a fixed-stride element format).
fn decode_dense(ty: GgmlType, bytes: &[u8]) -> Vec<f32> {
    match ty {
        GgmlType::F32 => bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
        GgmlType::F16 => bytes
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes(c.try_into().unwrap())))
            .collect(),
        GgmlType::Bf16 => bytes
            .chunks_exact(2)
            .map(|c| f32::from_bits((u16::from_le_bytes(c.try_into().unwrap()) as u32) << 16))
            .collect(),
        GgmlType::Q6K => unreachable!("Q6K is block-quantized; see dequantize_q6_k"),
    }
}

/// IEEE-754 half -> f32 (sign/exp/mantissa widening, subnormal and inf/nan handled). No external
/// crate: this is the one conversion this module needs, and getting it wrong is exactly the kind
/// of thing the real-GGUF cross-check test (`tests/gguf_matches_python_rail.rs`) exists to catch.
fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            // Subnormal half -> normalized f32.
            let mut e = -1i32;
            let mut m = mant;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            let f32_exp = (127 - 15 + e + 2) as u32;
            (sign << 31) | (f32_exp << 23) | (m << 13)
        }
    } else if exp == 0x1f {
        (sign << 31) | (0xff << 23) | (mant << 13) // inf / nan
    } else {
        (sign << 31) | ((exp + (127 - 15)) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_known_values() {
        // 0x3C00 = 1.0, 0xC000 = -2.0, 0x0000 = 0.0, 0x7C00 = +inf.
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        assert_eq!(f16_to_f32(0xC000), -2.0);
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert!(f16_to_f32(0x7C00).is_infinite());
        // Subnormal: 0x0001 = smallest positive subnormal half = 2^-24.
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24));
        // 0x0200 = mantissa 0x200 (leading bit at position 9) = 512 * 2^-24 = 2^-15, exact.
        assert_eq!(f16_to_f32(0x0200), 2f32.powi(-15));
    }

    /// Hand-built minimal GGUF: format-conformance only (magic/header/one KV/one tensor), not a
    /// substitute for the real-GGUF cross-check in `tests/gguf_matches_python_rail.rs` -- this
    /// tests parser adherence to the documented byte layout, not against Python's own reading.
    #[test]
    fn parses_a_hand_built_minimal_file() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes()); // version
        buf.extend_from_slice(&1u64.to_le_bytes()); // n_tensors
        buf.extend_from_slice(&1u64.to_le_bytes()); // n_kv
        // KV: general.alignment = 8 (u32)
        let key = b"general.alignment";
        buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
        buf.extend_from_slice(key);
        buf.extend_from_slice(&4u32.to_le_bytes()); // type = UINT32
        buf.extend_from_slice(&8u32.to_le_bytes()); // value
        // Tensor info: name "w", ne=[3,2] (numpy shape reversed -> [2,3]), type F32, offset 0
        let name = b"w";
        buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
        buf.extend_from_slice(name);
        buf.extend_from_slice(&2u32.to_le_bytes()); // n_dims
        buf.extend_from_slice(&3u64.to_le_bytes());
        buf.extend_from_slice(&2u64.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes()); // type = F32
        buf.extend_from_slice(&0u64.to_le_bytes()); // offset
        // pad to 8-byte alignment
        while buf.len() % 8 != 0 {
            buf.push(0);
        }
        let vals: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        for v in vals {
            buf.extend_from_slice(&v.to_le_bytes());
        }

        let td = tempfile::tempdir().unwrap();
        let path = td.path().join("mini.gguf");
        std::fs::write(&path, &buf).unwrap();

        let g = GgufFile::open(&path).unwrap();
        assert_eq!(g.shape("w").unwrap(), vec![2, 3]);
        assert_eq!(g.tensor_f32("w").unwrap(), vals);
        assert!(g.tensor_f32("missing").is_err());

        // Row 0 = [1,2,3], row 1 = [4,5,6] -- each numpy row is `ne[0]`=3 contiguous elements.
        assert_eq!(g.tensor_f32_rows("w", 0, 1).unwrap(), vec![1.0, 2.0, 3.0]);
        assert_eq!(g.tensor_f32_rows("w", 1, 2).unwrap(), vec![4.0, 5.0, 6.0]);
        assert_eq!(g.tensor_f32_rows("w", 0, 2).unwrap(), vals);
        assert!(g.tensor_f32_rows("w", 2, 3).is_err(), "row 2 is out of bounds (only 2 rows)");
        assert!(g.tensor_f32_rows("w", 1, 1).is_err(), "empty range must be rejected");
    }
}
