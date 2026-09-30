//! S2: serves the resident-forward layer-stack prototype build (`rls1`-shaped: a plain 5-argument
//! full ELF, per-layer combined weight files, no content-addressed store for the layer weights)
//! through `DecodeStep`, with every per-request host computation done in Rust -- no Python
//! subprocess in this path. `rust_bridge.py` (private repo) now serves only as a test oracle for
//! the functions here, per the S2 instruction.
//!
//! What moved from the bridge to here: the int4 embedding-row dequant (`EmbedHeadPack`), the RoPE
//! tables (`rope_row_sliding`/`rope_row_global`), the widths records
//! (`sliding_widths_record`/`global_widths_record`, matching the `RF_ATTN_H=1` reshape the build
//! was compiled with), and class/rung selection (`sliding_class`/`global_class`). The on-device LM
//! head (`h1`) replaces the host float64 fallback S1 used; its 566 MB weight STREAM is still
//! produced by the existing Python packer (`rhead.head_stream()`, itself bit-level lane/nibble
//! packing already gated by its own smoke test) -- this reads that packer's OUTPUT bytes the same
//! way it already reads the per-layer weight files, not its packing algorithm. Reimplementing that
//! packer blind, in the same change as the first served request, would be exactly the class of
//! silent bug this tree's doctrine warns about, for a build step that already has a cheaper fix
//! (materialise once, read the bytes) than a from-scratch port.

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::rc::Rc;

use ndarray::{Array1, ArrayD};
use ndarray_npy::read_npy;
use npu_xrt::{unpack_bf16_to_f32, Bo, Device, ElfResident, FLAG_HOST_ONLY};

use crate::api::EngineError;
use crate::llm::generator::{CacheState, DecodeStep};
use crate::llm::npu_decode::unpack_bf16_bytes;

// ---------------------------------------------------------------------------------------------
// Embedding + head weights: int4 dequant, ported from head_ref.py (unplanar_chunk/dequant_f64/
// row_group_for_k), read a single row at a time rather than materialising the 566 MB pack.
// ---------------------------------------------------------------------------------------------

/// `row_group_for_k` (head_ref.py): the smallest row-group that keeps the packed block's payload
/// and total stride aligned to a 32-byte (64-nibble) SIMD load, re-derived from `k` rather than
/// trusted from a stored field computed for a different `k` (that module's own docstring). Panics
/// (as the Python does) if none aligns within 64 -- a genuine packer-format assumption, not an
/// input to validate per call.
fn row_group_for_k(k: usize, group_size: usize, vec_size: usize) -> usize {
    let load = vec_size / 2;
    let payload = k / 2;
    let n_groups = k / group_size;
    let header = 2 * n_groups;
    let stride = header + payload;
    for g in 1..=64usize {
        if payload.is_multiple_of(load) && (g * stride).is_multiple_of(load) {
            return g;
        }
    }
    panic!("no row_group <= 64 aligns int4/g{group_size} vec={vec_size} at K={k}");
}

/// The manifest's `embedding` entry, parsed directly (not through `StoreManifest`, which only
/// keeps `{blob, offset, length, layout}` -- the dequant needs `vocab`/`hidden`/`group_size` too).
struct EmbeddingMeta {
    blob_path: PathBuf,
    offset: usize,
    hidden: usize, // K
    group_size: usize,
    row_group: usize,
}

fn store_blob_path(store_dir: &Path, blob: &str) -> PathBuf {
    store_dir.join("blobs").join(format!("{blob}.bin"))
}

fn load_embedding_meta(store_dir: &Path) -> Result<EmbeddingMeta, EngineError> {
    let manifest_path = store_dir.join("manifest.json");
    let bytes = fs::read(&manifest_path).map_err(|e| EngineError::Load(format!("read {}: {e}", manifest_path.display())))?;
    let v: serde_json::Value = serde_json::from_slice(&bytes).map_err(|e| EngineError::Load(format!("parse {}: {e}", manifest_path.display())))?;
    let e = v.get("embedding").ok_or_else(|| EngineError::Load(format!("{}: missing `embedding`", manifest_path.display())))?;
    let layout = e.get("layout").and_then(|x| x.as_str()).unwrap_or("");
    if layout != "planar_int4g32_headpack" {
        return Err(EngineError::Load(format!("embedding layout {layout:?}, expected planar_int4g32_headpack")));
    }
    let blob = e.get("blob").and_then(|x| x.as_str()).ok_or_else(|| EngineError::Load("embedding: missing `blob`".to_string()))?;
    let offset = e.get("offset").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
    let hidden = e.get("hidden").and_then(|x| x.as_u64()).ok_or_else(|| EngineError::Load("embedding: missing `hidden`".to_string()))? as usize;
    let group_size = e.get("group_size").and_then(|x| x.as_u64()).ok_or_else(|| EngineError::Load("embedding: missing `group_size`".to_string()))? as usize;
    let row_group_declared = e.get("row_group").and_then(|x| x.as_u64()).ok_or_else(|| EngineError::Load("embedding: missing `row_group`".to_string()))? as usize;
    let row_group = row_group_for_k(hidden, group_size, 64);
    if row_group != row_group_declared {
        return Err(EngineError::Load(format!(
            "embedding row_group: derived {row_group} for K={hidden} disagrees with manifest's stored {row_group_declared}"
        )));
    }
    Ok(EmbeddingMeta { blob_path: store_blob_path(store_dir, blob), offset, hidden, group_size, row_group })
}

/// Reads one dequantised embedding row at a time from the tied int4 head pack, by seeking to its
/// row-group block rather than mapping the whole 566 MB pack. Port of `head_ref.unplanar_chunk` +
/// `dequant_f64`, narrowed to a single row (`stack_run.py::Embed.__call__`'s own access pattern).
pub struct EmbedHeadPack {
    file: File,
    meta: EmbeddingMeta,
}

impl EmbedHeadPack {
    pub fn open(store_dir: &Path) -> Result<EmbedHeadPack, EngineError> {
        let meta = load_embedding_meta(store_dir)?;
        let file = File::open(&meta.blob_path).map_err(|e| EngineError::Load(format!("open {}: {e}", meta.blob_path.display())))?;
        Ok(EmbedHeadPack { file, meta })
    }

    pub fn hidden(&self) -> usize {
        self.meta.hidden
    }

    /// The dequantised row for `token`, scaled by `sqrt(hidden)` (the embedding's own convention,
    /// `stack_run.py::Embed.__call__`), as bf16 bits ready for an input buffer.
    pub fn embed_row_bf16(&mut self, token: u32) -> Result<Vec<u16>, EngineError> {
        let k = self.meta.hidden;
        let group = self.meta.group_size;
        let rg = self.meta.row_group;
        let ng = k / group;
        let pay = k / 2;
        let stride = ng * 2 + pay;
        let row0 = (token as usize) - (token as usize) % rg;
        let local = (token as usize) - row0;
        let block_off = self.meta.offset + (row0 / rg) * rg * stride;

        let mut block = vec![0u8; rg * stride];
        self.file.seek(SeekFrom::Start(block_off as u64)).map_err(|e| EngineError::Load(format!("seek embedding block: {e}")))?;
        self.file.read_exact(&mut block).map_err(|e| EngineError::Load(format!("read embedding block: {e}")))?;

        // Block layout: [rg rows of `pay` payload bytes][rg rows of `2*ng` scale bytes] --
        // head_ref.unplanar_chunk's `rows[:, 2*ng:]` (payload) / `rows[:, :2*ng]` (scale) split,
        // after its own transpose from [payload-major][scale-major] storage into per-row form.
        let payload_row = &block[local * pay..(local + 1) * pay];
        let scale_off = rg * pay + local * 2 * ng;
        let scale_bytes = &block[scale_off..scale_off + 2 * ng];
        let scale_u16: Vec<u16> = scale_bytes.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
        let mut scale_f32 = vec![0f32; ng];
        unpack_bf16_to_f32(&scale_u16, &mut scale_f32);

        let mut q = vec![0i8; k];
        for (i, &byte) in payload_row.iter().enumerate() {
            let lo = (byte & 0x0F) as i8;
            let hi = ((byte >> 4) & 0x0F) as i8;
            q[2 * i] = if lo >= 8 { lo - 16 } else { lo };
            q[2 * i + 1] = if hi >= 8 { hi - 16 } else { hi };
        }

        let sqrt_k = (k as f64).sqrt() as f32;
        let mut row_f32 = vec![0f32; k];
        for i in 0..k {
            row_f32[i] = (q[i] as f32) * scale_f32[i / group] * sqrt_k;
        }
        let mut bits = vec![0u16; k];
        npu_xrt::pack_f32_to_bf16(&row_f32, &mut bits);
        Ok(bits)
    }
}

/// `final_norm`: `{blob, offset, length, layout: raw_f32, shape}` -- a plain f32 vector, no dequant.
pub fn load_final_norm(store_dir: &Path) -> Result<Vec<f32>, EngineError> {
    let manifest_path = store_dir.join("manifest.json");
    let bytes = fs::read(&manifest_path).map_err(|e| EngineError::Load(format!("read {}: {e}", manifest_path.display())))?;
    let v: serde_json::Value = serde_json::from_slice(&bytes).map_err(|e| EngineError::Load(format!("parse {}: {e}", manifest_path.display())))?;
    let e = v.get("final_norm").ok_or_else(|| EngineError::Load("manifest missing `final_norm`".to_string()))?;
    let blob = e.get("blob").and_then(|x| x.as_str()).ok_or_else(|| EngineError::Load("final_norm: missing `blob`".to_string()))?;
    let offset = e.get("offset").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
    let length = e.get("length").and_then(|x| x.as_u64()).ok_or_else(|| EngineError::Load("final_norm: missing `length`".to_string()))? as usize;
    let path = store_blob_path(store_dir, blob);
    let mut f = File::open(&path).map_err(|e| EngineError::Load(format!("open {}: {e}", path.display())))?;
    f.seek(SeekFrom::Start(offset as u64)).map_err(|e| EngineError::Load(format!("seek final_norm: {e}")))?;
    let mut bytes = vec![0u8; length];
    f.read_exact(&mut bytes).map_err(|e| EngineError::Load(format!("read final_norm: {e}")))?;
    Ok(bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

// ---------------------------------------------------------------------------------------------
// RoPE tables: pos -> a bf16-bit angle row in the resident layer image's CONCATENATED [cos|sin]
// convention (qkv_ref.rope_table / glob_ref.rope_table). NOT `npu_decode::rope_row`'s INTERLEAVED
// `[cos,sin,cos,sin,...]` convention -- a different device build, a different packing; porting the
// wrong one here would compile and dispatch and produce plausible-looking wrong attention.
// ---------------------------------------------------------------------------------------------

/// The sliding layer's table (`qkv_ref.rope_table`): theta=10000, head_dim=256, every one of the
/// 128 frequency pairs real (no partial rotary). `[cos_128 | sin_128]` bf16 bits.
pub fn rope_row_sliding(pos: usize) -> Vec<u16> {
    rope_row_concat(pos, 256, 10000.0, 128)
}

/// The global layer's table (`glob_ref.rope_table`), TRUNCATED to just the real frequencies
/// (`stack_run.py::forward`'s own `rope_g[:P] = rg.reshape(P,2,256)[:,:,:64].reshape(P,128)`):
/// theta=1e6, head_dim=512 (so the exponent divides by 512 even though only 64 of 256 pairs are
/// real -- the "proportional" partial-rotary shape), 64 real frequencies, 192 elided instead of
/// written as the zero-frequency (cos=1,sin=0) identity rows the untruncated table would carry.
/// `[cos_64 | sin_64]` bf16 bits, half the untruncated table's width.
pub fn rope_row_global(pos: usize) -> Vec<u16> {
    rope_row_concat(pos, 512, 1_000_000.0, 64)
}

/// `[cos_half | sin_half]` bf16 bits for `n_real` of `head_dim/2` frequency pairs (the rest, when
/// `n_real < head_dim/2` and the caller does not truncate, are the identity rotation -- see
/// `npu_decode::rope_angles`'s doc for the same proportional-RoPE shape). Here the caller always
/// wants exactly `n_real` values per half (sliding: all of them; global: pre-truncated), so this
/// only ever computes real frequencies.
fn rope_row_concat(pos: usize, head_dim: usize, theta: f64, n_real: usize) -> Vec<u16> {
    let mut angles = vec![0f32; 2 * n_real];
    for i in 0..n_real {
        let inv = 1.0 / theta.powf((2 * i) as f64 / head_dim as f64);
        let ang = pos as f64 * inv;
        angles[i] = ang.cos() as f32;
        angles[n_real + i] = ang.sin() as f32;
    }
    let mut bits = vec![0u16; 2 * n_real];
    npu_xrt::pack_f32_to_bf16(&angles, &mut bits);
    bits
}

// ---------------------------------------------------------------------------------------------
// Widths records: rld_run.widths + rlo_run.wrec (sliding, RF_ATTN_H reshape) and glob_ref.widths +
// stack_run.py's duplicate-stack (global). Both produce `[nt][2][2][16]` i32, row-major, matching
// this build's compiled kernels exactly (RF_ATTN_H=1 -- see resident.rs's S1 report for what
// happens when this reshape is skipped).
// ---------------------------------------------------------------------------------------------

/// `rld_run.widths` then `rlo_run.wrec`'s per-head reshape, flattened to `i32` LE bytes.
/// `win` is the sliding attention window (1024); `first` is the ring's read-window start
/// (`(n_past - (win-1)).max(0) / 64 * 64`, the same value `resident.rs::ring_read_first` computes
/// for the S0/S1 design -- reused here since it is the identical formula, not re-derived).
pub fn sliding_widths_record(n_past: usize, p_len: usize, nt: usize, first: usize, win: usize) -> Vec<u8> {
    // w[tb][hi_lo][row_of_32], row_of_32 duplicated at [r] and [16+r] (two query heads).
    let mut w = vec![0i32; nt * 2 * 32];
    for tb in 0..nt {
        for r in 0..16 {
            let p = n_past + 16 * tb + r;
            let q = p.min(n_past + p_len - 1);
            let hi = (q as i64 - first as i64 + 1) as i32;
            let lo = (q as i64 - (win as i64 - 1) - first as i64).max(0) as i32;
            for idx in [r, 16 + r] {
                w[tb * 64 + idx] = hi; // hi_lo == 0
                w[tb * 64 + 32 + idx] = lo; // hi_lo == 1
            }
        }
    }
    // reshape (nt,2,32) -> (nt,2,2,16): out[tb][half][hi_lo][row] = w[tb][hi_lo][half*16+row]
    // (np.stack([w[:,:,:16], w[:,:,16:]], axis=1)).
    let mut out = vec![0i32; nt * 2 * 2 * 16];
    for tb in 0..nt {
        for half in 0..2 {
            for hi_lo in 0..2 {
                for row in 0..16 {
                    out[tb * 64 + half * 32 + hi_lo * 16 + row] = w[tb * 64 + hi_lo * 32 + half * 16 + row];
                }
            }
        }
    }
    i32s_to_le_bytes(&out)
}

/// `glob_ref.widths` then `stack_run.py`'s `np.stack([wg, wg], axis=1)` duplication (the global
/// layer has no window, so both query heads share the same `[hi, lo=0]` record; the duplication
/// only exists to match the sliding layer's `[nt][2][2][16]` shape).
pub fn global_widths_record(n_past: usize, p_len: usize, nt: usize) -> Vec<u8> {
    let mut wg = vec![0i32; nt * 2 * 16]; // [tb][hi_lo][row], lo stays 0
    for tb in 0..nt {
        for r in 0..16 {
            let p = n_past + 16 * tb + r;
            let q = p.min(n_past + p_len - 1);
            wg[tb * 32 + r] = (q + 1) as i32; // hi_lo == 0
        }
    }
    let mut out = vec![0i32; nt * 2 * 2 * 16];
    for tb in 0..nt {
        for dup in 0..2 {
            for hi_lo in 0..2 {
                for row in 0..16 {
                    out[tb * 64 + dup * 32 + hi_lo * 16 + row] = wg[tb * 32 + hi_lo * 16 + row];
                }
            }
        }
    }
    i32s_to_le_bytes(&out)
}

fn i32s_to_le_bytes(v: &[i32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for &x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

fn u16s_to_le_bytes(v: &[u16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 2);
    for &x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

fn le_bytes_to_u16s(b: &[u8]) -> Vec<u16> {
    b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
}

// ---------------------------------------------------------------------------------------------
// The raw build's own meta -- not the S0/S1 `ResidentArtifact` schema (which assumes a
// content-addressed weight store and a uniform class table). `rls1`'s shape is a research
// prototype's, not the shipped one: per-layer combined weight files, two independent class
// tables (sliding `p{nt}`, global `g{nt}w{seg}`). Documented separately rather than forced into
// the S0 schema's assumptions.
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct RawResidentMeta {
    pub dir: PathBuf,
    pub elf_name: String,
    pub boot: String,
    pub head_kernel: String,
    pub weight_dir: PathBuf,
    pub embedding_store: PathBuf,
    pub d_model: usize,
    pub nlayer: usize,
    pub full_attention_layers: Vec<usize>,
    pub row_block: usize,
    pub pcap_t: usize,
    pub xbuf: usize,
    pub obuf: usize,
    pub kvb_s: usize,
    pub kvb_g: usize,
    pub wb_s: usize,
    pub wb_g: usize,
    pub kvrow_s: usize,
    pub kvrow_g: usize,
    pub gcap: usize,
    pub s_cap: usize,
    pub nbw: usize,
    pub seg_nb: usize,
    pub xr: usize,
    pub widths_bytes: usize,
    pub sliding_window: usize,
    /// `nt -> "p{nt}"` for the sliding layers' class codes.
    pub sliding_classes: HashMap<usize, String>,
    /// `nt -> "g{nt}w{seg}"` for the global layers' class codes.
    pub global_classes: HashMap<usize, String>,
    pub vocab: usize,
    pub head_ncol: usize,
    pub head_wcol: usize,
    pub head_outb: usize,
    pub logit_softcap: Option<f64>,
    pub rms_norm_eps: f64,
}

impl RawResidentMeta {
    /// Load `dir/meta.json` (this module's own schema -- see its doc). Parse-don't-validate,
    /// same convention as `resident_artifact.rs`.
    pub fn load(dir: &Path) -> Result<RawResidentMeta, EngineError> {
        let meta_path = dir.join("meta.json");
        let bytes = fs::read(&meta_path).map_err(|e| EngineError::Load(format!("read {}: {e}", meta_path.display())))?;
        let v: serde_json::Value = serde_json::from_slice(&bytes).map_err(|e| EngineError::Load(format!("parse {}: {e}", meta_path.display())))?;
        let ctx = |msg: String| EngineError::Load(format!("{}: {msg}", meta_path.display()));
        let kind = v.get("kind").and_then(|x| x.as_str()).ok_or_else(|| ctx("missing `kind`".into()))?;
        if kind != "resident_forward_raw" {
            return Err(ctx(format!("kind {kind:?}, expected \"resident_forward_raw\"")));
        }
        let s = |k: &str| -> Result<String, EngineError> {
            v.get(k).and_then(|x| x.as_str()).map(|s| s.to_string()).ok_or_else(|| ctx(format!("missing `{k}`")))
        };
        let u = |k: &str| -> Result<usize, EngineError> {
            v.get(k).and_then(|x| x.as_u64()).map(|x| x as usize).ok_or_else(|| ctx(format!("missing/non-numeric `{k}`")))
        };
        let path = |k: &str| -> Result<PathBuf, EngineError> { Ok(PathBuf::from(s(k)?)) };

        let full_attention_layers: Vec<usize> = v
            .get("full_attention_layers")
            .and_then(|x| x.as_array())
            .ok_or_else(|| ctx("missing `full_attention_layers`".into()))?
            .iter()
            .map(|x| x.as_u64().unwrap() as usize)
            .collect();

        let classes_of = |key: &str| -> Result<HashMap<usize, String>, EngineError> {
            let arr = v.get(key).and_then(|x| x.as_array()).ok_or_else(|| ctx(format!("missing `{key}`")))?;
            let mut out = HashMap::new();
            for c in arr {
                let code = c.get("code").and_then(|x| x.as_str()).ok_or_else(|| ctx(format!("{key}[]: missing `code`")))?;
                let nt = c.get("nt").and_then(|x| x.as_u64()).ok_or_else(|| ctx(format!("{key}[]: missing `nt`")))? as usize;
                out.insert(nt, code.to_string());
            }
            Ok(out)
        };

        Ok(RawResidentMeta {
            dir: dir.to_path_buf(),
            elf_name: s("elf")?,
            boot: s("boot")?,
            head_kernel: s("head_kernel")?,
            weight_dir: path("weight_dir")?,
            embedding_store: path("embedding_store")?,
            d_model: u("d_model")?,
            nlayer: u("nlayer")?,
            full_attention_layers,
            row_block: u("row_block")?,
            pcap_t: u("pcap_t")?,
            xbuf: u("xbuf")?,
            obuf: u("obuf")?,
            kvb_s: u("kvb_s")?,
            kvb_g: u("kvb_g")?,
            wb_s: u("wb_s")?,
            wb_g: u("wb_g")?,
            kvrow_s: u("kvrow_s")?,
            kvrow_g: u("kvrow_g")?,
            gcap: u("gcap")?,
            s_cap: u("s_cap")?,
            nbw: u("nbw")?,
            seg_nb: u("seg_nb")?,
            xr: u("xr")?,
            widths_bytes: u("widths_bytes")?,
            sliding_window: u("sliding_window")?,
            sliding_classes: classes_of("classes")?,
            global_classes: classes_of("global_classes")?,
            vocab: u("vocab")?,
            head_ncol: u("head_ncol")?,
            head_wcol: u("head_wcol")?,
            head_outb: u("head_outb")?,
            logit_softcap: v.get("logit_softcap").and_then(|x| x.as_f64()),
            rms_norm_eps: v.get("rms_norm_eps").and_then(|x| x.as_f64()).unwrap_or(1e-6),
        })
    }

    pub fn elf_path(&self) -> PathBuf {
        self.dir.join(&self.elf_name)
    }
}

/// `nt` for a piece of `p_len` rows (spec table: `ceil(p_len / row_block)`).
pub fn piece_nt(p_len: usize, row_block: usize) -> usize {
    p_len.div_ceil(row_block)
}

/// The resident-forward `DecodeStep` backend for a raw layer-stack build (`rls1`-shaped): drives
/// the 48-layer stack plus the on-device LM head (`h1`) over `ElfResident`, with every per-request
/// host value computed in this module.
pub struct RawResidentForward {
    meta: RawResidentMeta,
    embed: EmbedHeadPack,
    fnorm: Vec<f32>,
    arena_xb: Bo,
    arena_ob: Bo,
    arena_wbo: [Bo; 2],
    kv_arena: Bo,
    layer_kv_offset: Vec<usize>,
    head_xb: Bo,
    head_wb: Bo,
    head_ob: Bo,
    kernels: HashMap<String, ElfResident>,
    n_written: usize,
}

impl RawResidentForward {
    pub fn open(dev: &Rc<Device>, dir: &Path) -> Result<RawResidentForward, EngineError> {
        let meta = RawResidentMeta::load(dir)?;
        let embed = EmbedHeadPack::open(&meta.embedding_store)?;
        let fnorm = load_final_norm(&meta.embedding_store)?;

        let elf = crate::llm::artifact::read_elf_bytes(&meta.elf_path())?;
        let boot = dev
            .open_elf_resident(&elf, Some(&format!("main:{}", meta.boot)))
            .map_err(|e| EngineError::Load(format!("open_elf_resident (boot): {e}")))?;

        let mut kernels = HashMap::new();
        for code in meta.sliding_classes.values().chain(meta.global_classes.values()) {
            if kernels.contains_key(code) {
                continue;
            }
            let k = boot.open_named(&format!("main:{code}")).map_err(|e| EngineError::Load(format!("open_named {code}: {e}")))?;
            kernels.insert(code.clone(), k);
        }
        let head_k = boot
            .open_named(&format!("main:{}", meta.head_kernel))
            .map_err(|e| EngineError::Load(format!("open_named {}: {e}", meta.head_kernel)))?;
        kernels.insert(meta.head_kernel.clone(), head_k);
        kernels.insert(meta.boot.clone(), boot);

        let arena_xb = dev.alloc_bo_raw(meta.xbuf, FLAG_HOST_ONLY, 0).map_err(|e| EngineError::Load(format!("alloc xb: {e}")))?;
        let arena_ob = dev.alloc_bo_raw(meta.obuf, FLAG_HOST_ONLY, 0).map_err(|e| EngineError::Load(format!("alloc ob: {e}")))?;
        let wmax = meta.wb_s.max(meta.wb_g);
        let arena_wbo = [
            dev.alloc_bo_raw(wmax, FLAG_HOST_ONLY, 0).map_err(|e| EngineError::Load(format!("alloc wbo0: {e}")))?,
            dev.alloc_bo_raw(wmax, FLAG_HOST_ONLY, 0).map_err(|e| EngineError::Load(format!("alloc wbo1: {e}")))?,
        ];
        // ONE arena for every layer's K/V, not 48 separate top-level allocations: 48 live BOs plus
        // xb/ob/wbo/head triggered `DRM_IOCTL_AMDXDNA_CREATE_BO ... Resource temporarily
        // unavailable` on the NEXT model load in the same service process (device-confirmed,
        // rf-engine-integration S2) -- a driver-level BO-count limit, not a bytes one (this
        // arena's total bytes are unchanged from the 48-BO version). Per-layer views are `Bo::sub`
        // into this one allocation, the same mechanism `run_piece`'s kvw/kvr already use within a
        // layer's own buffer.
        let mut layer_kv_offset = Vec::with_capacity(meta.nlayer);
        let mut kv_total = 0usize;
        for li in 0..meta.nlayer {
            let g = meta.full_attention_layers.contains(&li);
            let size = if g { meta.gcap * meta.kvrow_g * 2 } else { meta.s_cap * meta.kvrow_s * 2 };
            layer_kv_offset.push(kv_total);
            kv_total += size;
        }
        let kv_arena = dev.alloc_bo_raw(kv_total, FLAG_HOST_ONLY, 0).map_err(|e| EngineError::Load(format!("alloc kv arena ({kv_total} bytes): {e}")))?;
        kv_arena.write_bytes(&vec![0u8; kv_total]).map_err(|e| EngineError::Load(format!("zero kv arena: {e}")))?;
        kv_arena.sync_to_device().map_err(|e| EngineError::Load(format!("sync kv arena: {e}")))?;

        let head_xb = dev.alloc_bo_raw(meta.xbuf, FLAG_HOST_ONLY, 0).map_err(|e| EngineError::Load(format!("alloc head xb: {e}")))?;
        let head_wb = dev
            .alloc_bo_raw(meta.head_ncol * meta.head_wcol, FLAG_HOST_ONLY, 0)
            .map_err(|e| EngineError::Load(format!("alloc head wb: {e}")))?;
        let head_ob = dev.alloc_bo_raw(meta.head_outb, FLAG_HOST_ONLY, 0).map_err(|e| EngineError::Load(format!("alloc head ob: {e}")))?;
        let head_weight_path = meta.dir.join("w_head.npy");
        // rhead.head_stream()'s own shape is 5-D ([column][group][row][GW elements]-ish, C-contiguous);
        // read it shape-agnostically and take the flat bytes in storage order, which is what the
        // device wants regardless of how numpy's own shape reads.
        let head_weight: ArrayD<u8> = read_npy(&head_weight_path).map_err(|e| EngineError::Load(format!("read {}: {e}", head_weight_path.display())))?;
        let head_weight_bytes = head_weight.into_raw_vec_and_offset().0;
        head_wb.write_bytes(&head_weight_bytes).map_err(|e| EngineError::Load(format!("write head weight: {e}")))?;
        head_wb.sync_to_device().map_err(|e| EngineError::Load(format!("sync head weight: {e}")))?;

        Ok(RawResidentForward { meta, embed, fnorm, arena_xb, arena_ob, arena_wbo, kv_arena, layer_kv_offset, head_xb, head_wb, head_ob, kernels, n_written: 0 })
    }

    /// One piece of `p_len` rows starting at `s`, `x_bits` its bf16-bit input rows (`p_len * d`
    /// elements) -- drives the 48-layer stack and returns the last layer's `p_len` rows, same
    /// packing.
    fn forward(&mut self, x_bits: &[u16], s: usize, p_len: usize) -> Result<Vec<u16>, EngineError> {
        let d = self.meta.d_model;
        let nt = piece_nt(p_len, self.meta.row_block);
        let first = s.saturating_sub(self.meta.sliding_window.saturating_sub(1)) / 64 * 64;
        let widths_s = sliding_widths_record(s, p_len, nt, first, self.meta.sliding_window);
        let widths_g = global_widths_record(s, p_len, nt);

        let mut cur = vec![0u16; 16 * nt * d];
        cur[..x_bits.len()].copy_from_slice(x_bits);

        for li in 0..self.meta.nlayer {
            let g = self.meta.full_attention_layers.contains(&li);
            let wb_idx = li % 2;

            let weight_path = self.meta.weight_dir.join(format!("w{li}.npy"));
            let wbytes: Array1<u8> = read_npy(&weight_path).map_err(|e| EngineError::Load(format!("read {}: {e}", weight_path.display())))?;
            let wbytes = wbytes.into_raw_vec_and_offset().0;
            self.arena_wbo[wb_idx].write_bytes(&wbytes).map_err(|e| EngineError::Device(format!("write weight[{li}]: {e}")))?;
            self.arena_wbo[wb_idx].sync_to_device().map_err(|e| EngineError::Device(format!("sync weight[{li}]: {e}")))?;

            // Pad rows (row index >= p_len, only when nt > 1 -- p_len < 16*nt) get a ZERO rope
            // row, not a real angle at their own (invalid, past-the-piece) position: stack_run.py's
            // `rope_s = np.zeros((16*nt,256)); rope_s[:P] = qkv_ref.rope_table(pos)` only ever
            // fills the first P rows and leaves the rest at the array's zero-init. Missing this
            // only breaks pieces where p_len is not a multiple of 16, so nt=1 (p_len==16 exactly)
            // never exercises it -- which is exactly why it survived every check before the
            // server's own 20-token chat-template prompt (nt=2) hit it.
            let row_width = if g { 128 } else { 256 };
            let mut rope_rows = vec![0u16; 16 * nt * row_width];
            for row_idx in 0..p_len {
                let pos = s + row_idx;
                let row = if g { rope_row_global(pos) } else { rope_row_sliding(pos) };
                rope_rows[row_idx * row_width..(row_idx + 1) * row_width].copy_from_slice(&row);
            }
            let rope_bytes = u16s_to_le_bytes(&rope_rows);
            let widths = if g { &widths_g } else { &widths_s };

            let mut buf = vec![0u8; self.meta.xbuf];
            let x_bytes = u16s_to_le_bytes(&cur);
            buf[..x_bytes.len()].copy_from_slice(&x_bytes);
            buf[self.meta.xr..self.meta.xr + rope_bytes.len()].copy_from_slice(&rope_bytes);
            let w_off = self.meta.xbuf - self.meta.widths_bytes;
            buf[w_off..w_off + widths.len()].copy_from_slice(widths);
            self.arena_xb.write_bytes(&buf).map_err(|e| EngineError::Device(format!("write x: {e}")))?;
            self.arena_xb.sync_to_device().map_err(|e| EngineError::Device(format!("sync x: {e}")))?;

            let kvrow = if g { self.meta.kvrow_g } else { self.meta.kvrow_s };
            let layer_off = self.layer_kv_offset[li];
            let kvw_len = (if g { self.meta.kvb_g } else { self.meta.kvb_s }) * 2;
            let kvw = self.kv_arena.sub(layer_off + s * kvrow * 2, kvw_len).map_err(|e| EngineError::Device(format!("kvw[{li}]: {e}")))?;
            let read_positions = if g { self.meta.gcap } else { self.meta.nbw * 64 };
            let kvr_off = if g { 0 } else { first };
            let kvr = self.kv_arena.sub(layer_off + kvr_off * kvrow * 2, read_positions * kvrow * 2).map_err(|e| EngineError::Device(format!("kvr[{li}]: {e}")))?;

            let class = if g { &self.meta.global_classes } else { &self.meta.sliding_classes };
            let code = class.get(&nt).ok_or_else(|| EngineError::Unsupported(format!("no class for nt={nt} ({})", if g { "global" } else { "sliding" })))?;

            let args: [&Bo; 5] = [&self.arena_xb, &self.arena_ob, &self.arena_wbo[wb_idx], &kvw, &kvr];
            let boot = self.kernels.get(&self.meta.boot).expect("boot kernel");
            boot.bind(&args).map_err(|e| EngineError::Device(format!("bind boot before layer {li}: {e}")))?;
            boot.dispatch().map_err(EngineError::Device)?;
            let k = self.kernels.get(code).ok_or_else(|| EngineError::Load(format!("no kernel for {code}")))?;
            k.bind(&args).map_err(|e| EngineError::Device(format!("bind {code} (layer {li}): {e}")))?;
            k.dispatch().map_err(EngineError::Device)?;

            self.arena_ob.sync_from_device().map_err(|e| EngineError::Device(format!("sync ob: {e}")))?;
            let mut out_bytes = vec![0u8; self.meta.xr];
            self.arena_ob.read_bytes_at(0, &mut out_bytes).map_err(|e| EngineError::Device(format!("read ob: {e}")))?;
            let out_u16 = le_bytes_to_u16s(&out_bytes);
            cur = out_u16[..p_len * d].to_vec();

            let mut check = vec![0f32; cur.len()];
            unpack_bf16_to_f32(&cur, &mut check);
            if !check.iter().all(|v| v.is_finite()) {
                return Err(EngineError::Device(format!("layer {li}: non-finite output at s={s}")));
            }
        }
        self.n_written = s + p_len;
        Ok(cur)
    }

    /// The on-device LM head: `hidden` is one row's bf16-valued float32 (already unpacked), row 0
    /// of `head_xb`'s own 16-row block (`hrun.py::Head.__call__`). Returns raw (un-softcapped)
    /// f32 logits.
    fn head_logits(&mut self, hidden_bf16: &[u16]) -> Result<Vec<f32>, EngineError> {
        let mut buf = vec![0u8; self.meta.xbuf];
        let row_bytes = u16s_to_le_bytes(hidden_bf16);
        buf[..row_bytes.len()].copy_from_slice(&row_bytes);
        self.head_xb.write_bytes(&buf).map_err(|e| EngineError::Device(format!("write head x: {e}")))?;
        self.head_xb.sync_to_device().map_err(|e| EngineError::Device(format!("sync head x: {e}")))?;

        let args: [&Bo; 3] = [&self.head_xb, &self.head_ob, &self.head_wb];
        let boot = self.kernels.get(&self.meta.boot).expect("boot kernel");
        boot.bind(&args).map_err(|e| EngineError::Device(format!("bind boot before head: {e}")))?;
        boot.dispatch().map_err(EngineError::Device)?;
        let h1 = self.kernels.get(&self.meta.head_kernel).expect("head kernel");
        h1.bind(&args).map_err(|e| EngineError::Device(format!("bind head: {e}")))?;
        h1.dispatch().map_err(EngineError::Device)?;

        self.head_ob.sync_from_device().map_err(|e| EngineError::Device(format!("sync head ob: {e}")))?;
        let row_bytes_len = self.meta.vocab * 2;
        let mut row0 = vec![0u8; row_bytes_len];
        self.head_ob.read_bytes_at(0, &mut row0).map_err(|e| EngineError::Device(format!("read head ob: {e}")))?;
        Ok(unpack_bf16_bytes(&row0))
    }

    /// `rms_norm_f64` (head_ref.py: "Gemma4RMSNorm: normed * weight, no +1" -- this checkpoint's
    /// `final_norm` weights run mean ~20, max ~604, not the near-1 range a `(1+w)` convention
    /// implies; this formula is what `hrun.py`'s own device smoke test validates against, not just
    /// what the docstring claims). Computed once here rather than on-device -- a single 3840-wide
    /// reduction, cheap on the host and the same precision the device path itself does not name a
    /// lower one for.
    fn rms_norm(&self, h: &[f32]) -> Vec<f32> {
        let n = h.len() as f64;
        let ms: f64 = h.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>() / n + self.meta.rms_norm_eps;
        let scale = ms.powf(-0.5);
        h.iter().zip(&self.fnorm).map(|(&x, &w)| ((x as f64) * scale * (w as f64)) as f32).collect()
    }

    /// Diagnostic (S2 head-isolation): forward one token/position through the 48-layer stack ONLY,
    /// returning the raw (pre-norm) hidden row's bf16 bits -- the exact value `step()` hands to
    /// `head_logits`. Lets a caller capture this row once and feed the identical bytes to both the
    /// on-device head (`debug_raw_head_from_hidden`) and a host oracle, instead of re-running the
    /// (already S1-proven-correct) 48-layer stack twice to compare the two heads.
    pub fn debug_forward_hidden(&mut self, token: u32, pos: usize) -> Result<Vec<u16>, EngineError> {
        let x_bits = self.embed.embed_row_bf16(token)?;
        self.forward(&x_bits, pos, 1)
    }

    /// Diagnostic: the on-device head on a RAW (pre-norm) hidden row's bf16 bits, without softcap
    /// -- `step()`'s own path from `debug_forward_hidden`'s output, exposed so a captured row can
    /// be replayed here instead of re-forwarding. Do NOT pre-normalize the row before calling this
    /// -- `h1` applies its own RMSNorm + `final_norm` gain internally (see `step()`'s doc); the S2
    /// head-isolation bug was exactly a caller (this one, originally) doing that host-side first.
    pub fn debug_raw_head_from_hidden(&mut self, hidden_bits: &[u16]) -> Result<Vec<f32>, EngineError> {
        self.head_logits(hidden_bits)
    }

    /// Diagnostic: the HOST-SIDE `rms_norm_f64` oracle computation on a raw hidden row -- never
    /// fed to the device (see `step()`'s doc) -- so its stats can be diffed against the bridge
    /// oracle's `head_debug` command's `xn_stats` as a norm-math cross-check independent of `h1`.
    pub fn debug_host_rms_norm(&self, hidden_f32: &[f32]) -> Vec<f32> {
        self.rms_norm(hidden_f32)
    }

    fn apply_softcap(&self, mut logits: Vec<f32>) -> Vec<f32> {
        if let Some(cap) = self.meta.logit_softcap {
            let cap = cap as f32;
            for v in &mut logits {
                *v = (*v / cap).tanh() * cap;
            }
        }
        logits
    }
}

impl DecodeStep for RawResidentForward {
    fn step(&mut self, token: u32, pos: usize) -> Result<Vec<f32>, EngineError> {
        let hidden = self.debug_forward_hidden(token, pos)?;
        // The RAW (un-normalized) hidden row, not `debug_head_from_hidden`'s host-normed one:
        // `h1` applies its OWN RMSNorm + `final_norm` gain internally (`rhead.py`'s `NORM_PRE`/
        // `GAINS` phases, the gain baked into `head_stream()`'s weight pack). `hrun.py`'s own
        // reference confirms this -- `head(h)` feeds the device the raw row; `rms_norm_f64(h, ...)`
        // is applied only on the HOST SIDE for comparison, never sent to the device. Feeding an
        // already-normed row here double-applies the norm, which is exactly the S2 bug this
        // comment exists to prevent reintroducing: measured device logits ~12x the host oracle's
        // magnitude (316 vs 26.5 raw) before this fix, with `debug_rms_norm`'s own output otherwise
        // bit-identical to the host oracle's `xn_stats` -- i.e. the norm math was right, only
        // applying it twice was wrong.
        let raw = self.head_logits(&hidden)?;
        Ok(self.apply_softcap(raw))
    }

    fn reset(&mut self) -> Result<CacheState, EngineError> {
        let size = self.kv_arena.nbytes();
        self.kv_arena.write_bytes(&vec![0u8; size]).map_err(|e| EngineError::Device(format!("zero kv arena on reset: {e}")))?;
        self.kv_arena.sync_to_device().map_err(|e| EngineError::Device(format!("sync kv arena on reset: {e}")))?;
        self.n_written = 0;
        Ok(CacheState::Cleared)
    }

    fn max_context(&self) -> Option<usize> {
        // The narrower of the two linear caches this build actually allocated (no ring wraparound
        // in this raw driver yet -- see the S2 worklog's known gaps).
        Some(self.meta.s_cap.min(self.meta.gcap))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::process::{Command, Stdio};

    #[test]
    fn row_group_for_k_matches_the_manifests_declared_value() {
        // The real gemma4-12b manifest declares row_group=2 at K=3840, group_size=32.
        assert_eq!(row_group_for_k(3840, 32, 64), 2);
    }

    #[test]
    fn rope_row_sliding_at_pos_zero_is_the_identity_rotation() {
        // ang = 0 for every frequency at pos=0: cos=1, sin=0, for all 128 pairs.
        let row = rope_row_sliding(0);
        assert_eq!(row.len(), 256);
        let mut f = vec![0f32; 256];
        unpack_bf16_to_f32(&row, &mut f);
        assert!(f[..128].iter().all(|&v| (v - 1.0).abs() < 1e-6), "{:?}", &f[..8]);
        assert!(f[128..].iter().all(|&v| v.abs() < 1e-6), "{:?}", &f[128..136]);
    }

    #[test]
    fn rope_row_global_is_half_the_untruncated_tables_width() {
        // 64 real frequencies, truncated (not the untruncated 256+256=512-wide table).
        let row = rope_row_global(0);
        assert_eq!(row.len(), 128);
    }

    #[test]
    fn rope_row_global_first_frequency_matches_theta_1e6_at_head_dim_512() {
        // i=0: inv = 1/theta^0 = 1, ang = pos*1 = pos. Tolerance is bf16 precision (~2^-8 relative
        // near 1.0), not float32 -- 0.5390625 is bf16(cos(1)) exactly, not an approximation error.
        let row = rope_row_global(1);
        let mut f = vec![0f32; 128];
        unpack_bf16_to_f32(&row, &mut f);
        assert!((f[0] - 1.0_f64.cos() as f32).abs() < 5e-3, "cos(1) mismatch: {}", f[0]);
        assert!((f[64] - 1.0_f64.sin() as f32).abs() < 5e-3, "sin(1) mismatch: {}", f[64]);
    }

    #[test]
    fn sliding_widths_record_shape_and_boundary() {
        // nt=1, n_past=0, P=9, first=0, win=1023 (a hand-worked case matching the S0 attention
        // widths test): row 0 -> p=0, q=min(0,8)=0, hi=1, lo=0.
        let bytes = sliding_widths_record(0, 9, 1, 0, 1023);
        assert_eq!(bytes.len(), 1 * 2 * 2 * 16 * 4);
        let words: Vec<i32> = bytes.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        // out[tb=0][half=0][hi_lo=0][row=0] = hi at row 0 = 1.
        assert_eq!(words[0], 1);
        // out[tb=0][half=0][hi_lo=1][row=0] = lo at row 0 = 0.
        assert_eq!(words[16], 0);
    }

    #[test]
    fn global_widths_record_lo_is_always_zero() {
        let bytes = global_widths_record(0, 9, 1);
        let words: Vec<i32> = bytes.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        assert_eq!(words[0], 1); // hi at row 0
        assert_eq!(words[16], 0); // lo always 0
        // Duplicated at dup=1 (offset 32): identical to dup=0.
        assert_eq!(&words[32..48], &words[0..16]);
    }

    fn store_dir() -> Option<PathBuf> {
        let dir = std::env::var_os("RESIDENT_STORE_DIR").map(PathBuf::from).unwrap_or_else(|| {
            std::env::var_os("XDNA_DATA").map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data"))
                .join("artifacts/gemma4-12b/store")
        });
        if dir.join("manifest.json").is_file() {
            Some(dir)
        } else {
            eprintln!("skip: no gemma4-12b store at {} (set RESIDENT_STORE_DIR or XDNA_DATA)", dir.display());
            None
        }
    }

    #[test]
    fn embed_row_bf16_is_finite_and_the_right_width_against_the_real_store() {
        let Some(dir) = store_dir() else { return };
        let mut embed = EmbedHeadPack::open(&dir).expect("open the real embedding store");
        assert_eq!(embed.hidden(), 3840);
        for &token in &[0u32, 1, 23391, 262143] {
            let bits = embed.embed_row_bf16(token).expect("dequant a real row");
            assert_eq!(bits.len(), 3840);
            let mut f = vec![0f32; 3840];
            unpack_bf16_to_f32(&bits, &mut f);
            assert!(f.iter().all(|v| v.is_finite()), "token {token}: non-finite embedding row");
        }
    }

    #[test]
    fn final_norm_is_the_right_width_against_the_real_store() {
        let Some(dir) = store_dir() else { return };
        let w = load_final_norm(&dir).expect("load the real final_norm");
        assert_eq!(w.len(), 3840);
        assert!(w.iter().all(|v| v.is_finite()));
    }

    /// Oracle path: `rust_bridge.py` (private repo, not shipped) as a test-only cross-check against
    /// the SAME reference functions `embed_row_bf16`/`rope_row_*`/`*_widths_record` port. Skips
    /// (not fails) when the bridge or its interpreter is not configured on this box -- this is a
    /// cross-check, not a build requirement.
    fn bridge_call(req: &str) -> Option<serde_json::Value> {
        let bridge_py = std::env::var("RF_BRIDGE_PY").ok()?;
        let python = std::env::var("RF_BRIDGE_PYTHON").ok()?;
        let pythonpath = std::env::var("RF_BRIDGE_PYTHONPATH").ok()?;
        let cwd = Path::new(&bridge_py).parent()?.to_path_buf();
        let mut child = Command::new(&python)
            .arg(&bridge_py)
            .current_dir(cwd)
            .env("PYTHONPATH", pythonpath)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .ok()?;
        child.stdin.take()?.write_all(format!("{req}\n").as_bytes()).ok()?;
        let out = child.wait_with_output().ok()?;
        serde_json::from_slice(&out.stdout.split(|&b| b == b'\n').next()?).ok()
    }

    #[test]
    fn embed_row_matches_the_bridge_oracle() {
        let Some(resp) = bridge_call(r#"{"cmd":"embed","ids":[23391]}"#) else {
            eprintln!("skip: RF_BRIDGE_PY/RF_BRIDGE_PYTHON/RF_BRIDGE_PYTHONPATH not set");
            return;
        };
        assert!(resp["ok"].as_bool().unwrap_or(false), "bridge error: {resp}");
        let want_bits: Vec<u16> = {
            let raw = resp["x"].as_str().expect("x field");
            crate::llm::resident_raw::tests_b64::decode(raw)
        };
        let Some(dir) = store_dir() else { return };
        let mut embed = EmbedHeadPack::open(&dir).expect("open the real embedding store");
        let got = embed.embed_row_bf16(23391).expect("dequant row 23391");
        assert_eq!(got, want_bits, "Rust embed_row_bf16 disagrees with the bridge oracle");
    }

    #[test]
    fn rope_rows_match_the_bridge_oracle() {
        let Some(resp) = bridge_call(r#"{"cmd":"piece_inputs","s":5,"p":1}"#) else {
            eprintln!("skip: RF_BRIDGE_PY/RF_BRIDGE_PYTHON/RF_BRIDGE_PYTHONPATH not set");
            return;
        };
        assert!(resp["ok"].as_bool().unwrap_or(false), "bridge error: {resp}");
        // nt=1, p=1: row 0 of rope_s/rope_g is the real position-5 row; rows 1..15 are zero pad.
        let want_s = tests_b64::decode(resp["rope_s"].as_str().unwrap());
        let want_g = tests_b64::decode(resp["rope_g"].as_str().unwrap());
        assert_eq!(&want_s[..256], rope_row_sliding(5).as_slice(), "sliding rope disagrees with the bridge oracle");
        assert_eq!(&want_g[..128], rope_row_global(5).as_slice(), "global rope disagrees with the bridge oracle");
    }

    #[test]
    fn widths_records_match_the_bridge_oracle() {
        let Some(resp) = bridge_call(r#"{"cmd":"piece_inputs","s":5,"p":9}"#) else {
            eprintln!("skip: RF_BRIDGE_PY/RF_BRIDGE_PYTHON/RF_BRIDGE_PYTHONPATH not set");
            return;
        };
        assert!(resp["ok"].as_bool().unwrap_or(false), "bridge error: {resp}");
        let nt = resp["nt"].as_u64().unwrap() as usize;
        let first = resp["first"].as_u64().unwrap() as usize;
        let want_s = tests_b64::decode(resp["wr_s"].as_str().unwrap());
        let want_g = tests_b64::decode(resp["wr_g"].as_str().unwrap());
        // wr_s/wr_g are int32; tests_b64::decode packs pairs of bytes into u16, so reinterpret.
        let want_s_bytes: Vec<u8> = want_s.iter().flat_map(|v| v.to_le_bytes()).collect();
        let want_g_bytes: Vec<u8> = want_g.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(want_s_bytes, sliding_widths_record(5, 9, nt, first, 1024), "sliding widths disagree with the bridge oracle");
        assert_eq!(want_g_bytes, global_widths_record(5, 9, nt), "global widths disagree with the bridge oracle");
    }
}

/// Minimal base64 decoder, test-only (the oracle cross-check's wire format) -- avoids adding a
/// crate dependency to the shipped library for one test's decoding.
#[cfg(test)]
mod tests_b64 {
    pub fn decode(s: &str) -> Vec<u16> {
        fn val(c: u8) -> u8 {
            match c {
                b'A'..=b'Z' => c - b'A',
                b'a'..=b'z' => c - b'a' + 26,
                b'0'..=b'9' => c - b'0' + 52,
                b'+' => 62,
                b'/' => 63,
                _ => panic!("invalid base64 byte {c}"),
            }
        }
        let bytes: Vec<u8> = s.bytes().filter(|&c| c != b'=').collect();
        let mut out = Vec::with_capacity(bytes.len() * 3 / 4 + 3);
        for chunk in bytes.chunks(4) {
            let v: Vec<u8> = chunk.iter().map(|&c| val(c)).collect();
            out.push((v[0] << 2) | (v.get(1).copied().unwrap_or(0) >> 4));
            if v.len() > 2 {
                out.push((v[1] << 4) | (v[2] >> 2));
            }
            if v.len() > 3 {
                out.push((v[2] << 6) | v[3]);
            }
        }
        out.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
    }
}
