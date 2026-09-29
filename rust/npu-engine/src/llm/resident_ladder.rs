//! `rf48L`-shaped backend: the 256k-context ladder. Extends `resident_onecmd.rs`'s one-command
//! design (one `%x`/`%o`/`%s` FusedArena, weights/KV resident for the model's whole lifetime) with
//! TWO things that build did not have:
//!
//! - **A rung ladder, not one fixed command per phase.** The global window is a property of the
//!   compiled sequence (`rlayer_design.emit`'s naming), so each reach (8192..262144 keys for
//!   decode, 4096..65536 for prefill) is its own control code. The host picks the smallest rung
//!   whose window holds `n_past + P` keys -- `fwd_host.Forward.rung`, ported verbatim below.
//! - **A mirrored sliding-cache ring (`s_ring` = 1280).** The device writes a new row at BOTH
//!   `pos % C` and `pos % C + C`; the host only ever computes `pos % C` (`fwd_host.Forward.ring`).
//!   Context can now pass the old 2048-row linear cap. A prefill piece must not cross a multiple
//!   of `C` -- `fwd_host.Forward.max_piece` -- so pieces are cut there, not just at `pmax`.
//!
//! Layout is read from `meta.json` (`kind: "resident_forward_ladder"`), itself generated from
//! `fwd_layout.json` (the build's own authority) plus the architecture-constant `%x`/`%o` offsets
//! `rforward.Plan` computes identically for every rung (confirmed: this build's `XF`/`OF` equal
//! `rf48s`'s `xbuf`/`obuf_f1` bit for bit) -- see `scripts/gen_rf48L_meta.py`. Nothing here hand-
//! copies a byte offset.
//!
//! Reuses `resident_raw.rs`'s embedding/RoPE/widths functions unchanged, same as `resident_onecmd.rs`.
//!
//! **Prefill past the largest f2 rung (65536 keys) is not implemented.** `fwd_host.py`'s own
//! per-token cost table prices an f2 command at 262144 keys past the TDR budget (~2.9s) and notes
//! it "is not built" there either -- the device lane's own per-layer fallback (`p2`/`g2w4096`) is
//! a DIFFERENT, independently-sized dispatch path (`rlayer_design`'s standalone per-layer buffers,
//! not this arena) that has not been verified against this build's own emit() parameters. Rather
//! than guess at an unverified layout, `prefill()` here DECLINES past 65536 keys (returns its
//! progress unchanged, the trait's documented decline) and lets the generator's per-token loop
//! finish priming through `step()`, which reaches the full 262144 via the f1 ladder alone -- slower
//! for that tail, but exercises only the rungs this module verifies. See `max_context`'s doc for
//! why this keeps the 262144 bound honest anyway.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use ndarray::Array1;
use ndarray_npy::read_npy;
use npu_xrt::{Bo, Device, ElfResident, FLAG_HOST_ONLY};

use crate::api::EngineError;
use crate::llm::generator::{CacheState, DecodeStep};
use crate::llm::npu_decode::unpack_bf16_bytes;
use crate::llm::resident_raw::{global_widths_record, piece_nt, rope_row_global, rope_row_sliding, sliding_widths_record, EmbedHeadPack};

/// One entry of `fwd_layout.json`'s `rungs` array / `meta.json`'s copy of it: a compiled control
/// code and the window it covers. `kind` is `"split"` (f1-family, decode, carries the split-lane
/// per-column overlay) or `"seg"` (f2-family, prefill, no overlay) -- `fwd_host.command`'s own
/// branch on `P == 1`.
#[derive(Debug, Clone)]
pub struct Rung {
    pub name: String,
    pub nt: usize,
    pub blocks: usize,
    pub keys: usize,
    pub kind: String,
}

/// `fwd_host.Forward.rung`: the smallest rung of the given `nt` whose window holds `keys_needed`
/// keys. Ported verbatim (linear scan over an ascending list, first match wins) -- `rungs` is
/// small (6 f1 + 3 f2) and read once per dispatch, so this is not worth a binary search.
pub fn rung_for<'a>(rungs: &'a [Rung], nt: usize, keys_needed: usize) -> Result<&'a Rung, EngineError> {
    rungs.iter().filter(|r| r.nt == nt).find(|r| r.keys >= keys_needed).ok_or_else(|| {
        let largest = rungs.iter().filter(|r| r.nt == nt).last().map(|r| r.keys).unwrap_or(0);
        EngineError::Unsupported(format!("no nt={nt} rung holds {keys_needed} keys (largest {largest})"))
    })
}

/// `fwd_host.Forward.ring`: a sliding-cache row's index in the mirrored ring's first copy.
pub fn ring_pos(pos: usize, c: usize) -> usize {
    pos % c
}

/// `fwd_host.Forward.max_piece`: the rows of a piece starting at `s` that do not cross a multiple
/// of the ring size `c` -- `min(p, c - s % c)`. A caller cuts a longer piece at this boundary.
pub fn max_piece(s: usize, p: usize, c: usize) -> usize {
    p.min(c - s % c)
}

/// `rf48L/meta.json`'s schema (`kind: "resident_forward_ladder"`). See module doc for provenance.
#[derive(Debug, Clone)]
pub struct LadderMeta {
    pub dir: PathBuf,
    pub elf_name: String,
    pub boot: String,
    pub rungs: Vec<Rung>,
    pub weight_dir: PathBuf,
    pub embedding_store: PathBuf,
    pub d_model: usize,
    pub nlayer: usize,
    pub full_attention_layers: Vec<usize>,
    pub row_block: usize,
    pub pcap_t: usize,
    pub pmax: usize,
    pub sliding_window: usize,
    pub s_cap: usize,
    pub s_ring: usize,
    pub s_rows: usize,
    pub g_cap: usize,
    pub kvrow_s: usize,
    pub kvrow_g: usize,
    pub param_unit_bytes: usize,
    pub xrows: usize,
    pub rope_s_off: usize,
    pub rope_g_off: usize,
    pub widths_s_off: usize,
    pub widths_g_off: usize,
    pub widths_bytes: usize,
    pub xbuf: usize,
    pub hidden_slot_bytes: usize,
    pub logits_off: usize,
    pub obuf_f1: usize,
    pub obuf_f2: usize,
    pub layer_weight_off: HashMap<usize, usize>,
    pub layer_kv_off: HashMap<usize, usize>,
    pub head_off: usize,
    pub scratch_bytes: usize,
    pub vocab: usize,
    pub logit_softcap: Option<f64>,
    pub scratchpad_params: HashMap<String, usize>,
    pub split_widths_x_off: Vec<usize>,
}

impl LadderMeta {
    pub fn load(dir: &Path) -> Result<LadderMeta, EngineError> {
        let meta_path = dir.join("meta.json");
        let bytes = fs::read(&meta_path).map_err(|e| EngineError::Load(format!("read {}: {e}", meta_path.display())))?;
        let v: serde_json::Value = serde_json::from_slice(&bytes).map_err(|e| EngineError::Load(format!("parse {}: {e}", meta_path.display())))?;
        let ctx = |msg: String| EngineError::Load(format!("{}: {msg}", meta_path.display()));
        let kind = v.get("kind").and_then(|x| x.as_str()).ok_or_else(|| ctx("missing `kind`".into()))?;
        if kind != "resident_forward_ladder" {
            return Err(ctx(format!("kind {kind:?}, expected \"resident_forward_ladder\"")));
        }
        let s = |k: &str| -> Result<String, EngineError> {
            v.get(k).and_then(|x| x.as_str()).map(|s| s.to_string()).ok_or_else(|| ctx(format!("missing `{k}`")))
        };
        let u = |k: &str| -> Result<usize, EngineError> {
            v.get(k).and_then(|x| x.as_u64()).map(|x| x as usize).ok_or_else(|| ctx(format!("missing/non-numeric `{k}`")))
        };
        let path = |k: &str| -> Result<PathBuf, EngineError> { Ok(PathBuf::from(s(k)?)) };
        let usize_map = |k: &str| -> Result<HashMap<usize, usize>, EngineError> {
            let obj = v.get(k).and_then(|x| x.as_object()).ok_or_else(|| ctx(format!("missing `{k}`")))?;
            obj.iter()
                .map(|(key, val)| {
                    let ik: usize = key.parse().map_err(|_| ctx(format!("{k}: non-numeric key {key:?}")))?;
                    let iv = val.as_u64().ok_or_else(|| ctx(format!("{k}[{key}]: non-numeric value")))? as usize;
                    Ok((ik, iv))
                })
                .collect()
        };

        let full_attention_layers: Vec<usize> = v
            .get("full_attention_layers")
            .and_then(|x| x.as_array())
            .ok_or_else(|| ctx("missing `full_attention_layers`".into()))?
            .iter()
            .map(|x| x.as_u64().unwrap() as usize)
            .collect();
        let scratchpad_params: HashMap<String, usize> = v
            .get("scratchpad_params")
            .and_then(|x| x.as_object())
            .ok_or_else(|| ctx("missing `scratchpad_params`".into()))?
            .iter()
            .map(|(k, val)| (k.clone(), val.as_u64().unwrap() as usize))
            .collect();
        let split_widths_x_off: Vec<usize> = v
            .get("split_widths_x_off")
            .and_then(|x| x.as_array())
            .ok_or_else(|| ctx("missing `split_widths_x_off`".into()))?
            .iter()
            .map(|x| x.as_u64().unwrap() as usize)
            .collect();
        let rungs: Vec<Rung> = v
            .get("rungs")
            .and_then(|x| x.as_array())
            .ok_or_else(|| ctx("missing `rungs`".into()))?
            .iter()
            .map(|r| -> Result<Rung, EngineError> {
                Ok(Rung {
                    name: r.get("name").and_then(|x| x.as_str()).ok_or_else(|| ctx("rung missing `name`".into()))?.to_string(),
                    nt: r.get("nt").and_then(|x| x.as_u64()).ok_or_else(|| ctx("rung missing `nt`".into()))? as usize,
                    blocks: r.get("blocks").and_then(|x| x.as_u64()).ok_or_else(|| ctx("rung missing `blocks`".into()))? as usize,
                    keys: r.get("keys").and_then(|x| x.as_u64()).ok_or_else(|| ctx("rung missing `keys`".into()))? as usize,
                    kind: r.get("kind").and_then(|x| x.as_str()).ok_or_else(|| ctx("rung missing `kind`".into()))?.to_string(),
                })
            })
            .collect::<Result<_, _>>()?;

        Ok(LadderMeta {
            dir: dir.to_path_buf(),
            elf_name: s("elf")?,
            boot: s("boot")?,
            rungs,
            weight_dir: path("weight_dir")?,
            embedding_store: path("embedding_store")?,
            d_model: u("d_model")?,
            nlayer: u("nlayer")?,
            full_attention_layers,
            row_block: u("row_block")?,
            pcap_t: u("pcap_t")?,
            pmax: u("pmax")?,
            sliding_window: u("sliding_window")?,
            s_cap: u("s_cap")?,
            s_ring: u("s_ring")?,
            s_rows: u("s_rows")?,
            g_cap: u("g_cap")?,
            kvrow_s: u("kvrow_s")?,
            kvrow_g: u("kvrow_g")?,
            param_unit_bytes: u("param_unit_bytes")?,
            xrows: u("xrows")?,
            rope_s_off: u("rope_s_off")?,
            rope_g_off: u("rope_g_off")?,
            widths_s_off: u("widths_s_off")?,
            widths_g_off: u("widths_g_off")?,
            widths_bytes: u("widths_bytes")?,
            xbuf: u("xbuf")?,
            hidden_slot_bytes: u("hidden_slot_bytes")?,
            logits_off: u("logits_off")?,
            obuf_f1: u("obuf_f1")?,
            obuf_f2: u("obuf_f2")?,
            layer_weight_off: usize_map("layer_weight_off")?,
            layer_kv_off: usize_map("layer_kv_off")?,
            head_off: u("head_off")?,
            scratch_bytes: u("scratch_bytes")?,
            vocab: u("vocab")?,
            logit_softcap: v.get("logit_softcap").and_then(|x| x.as_f64()),
            scratchpad_params,
            split_widths_x_off,
        })
    }

    pub fn elf_path(&self) -> PathBuf {
        self.dir.join(&self.elf_name)
    }

    /// The largest key range any implemented rung reaches, for a given `nt` -- 0 if none.
    fn largest_keys(&self, nt: usize) -> usize {
        self.rungs.iter().filter(|r| r.nt == nt).map(|r| r.keys).max().unwrap_or(0)
    }
}

fn u16s_to_le_bytes(v: &[u16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 2);
    for &x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

fn i32s_to_le_bytes(v: &[i32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for &x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

/// The `DecodeStep` backend driving `rf48L`. One `ElfResident` per rung (9: 6 f1-family + 3
/// f2-family), all sharing one `hw_context`, opened and bound once at `open`.
pub struct LadderResidentForward {
    meta: LadderMeta,
    embed: EmbedHeadPack,
    xb: Bo,
    ob: Bo,
    #[allow(dead_code)] // kept alive: %s is referenced only through the bound rung kernels
    sb: Bo,
    boot: ElfResident,
    rungs: HashMap<String, ElfResident>,
    n_written: usize,
}

impl LadderResidentForward {
    pub fn open(dev: &Rc<Device>, dir: &Path) -> Result<LadderResidentForward, EngineError> {
        let meta = LadderMeta::load(dir)?;
        let embed = EmbedHeadPack::open(&meta.embedding_store)?;

        let xb = dev.alloc_bo_raw(meta.xbuf, FLAG_HOST_ONLY, 0).map_err(|e| EngineError::Load(format!("alloc xb: {e}")))?;
        let ob_size = meta.obuf_f1.max(meta.obuf_f2);
        let ob = dev.alloc_bo_raw(ob_size, FLAG_HOST_ONLY, 0).map_err(|e| EngineError::Load(format!("alloc ob: {e}")))?;
        let sb = dev.alloc_bo_raw(meta.scratch_bytes, FLAG_HOST_ONLY, 0).map_err(|e| EngineError::Load(format!("alloc scratch ({} bytes): {e}", meta.scratch_bytes)))?;

        // Every layer's weight stream and its K/V cache (zeroed), plus the head's weight stream,
        // go into %s ONCE here -- resident for the model's whole lifetime. Same as
        // `resident_onecmd.rs::open`; ring rows and global rows share this one region regardless
        // of which rung's dispatch touches them.
        for li in 0..meta.nlayer {
            let weight_path = meta.weight_dir.join(format!("w{li}.npy"));
            let wbytes: Array1<u8> = read_npy(&weight_path).map_err(|e| EngineError::Load(format!("read {}: {e}", weight_path.display())))?;
            let wbytes = wbytes.into_raw_vec_and_offset().0;
            let woff = *meta.layer_weight_off.get(&li).ok_or_else(|| EngineError::Load(format!("meta.json: no layer_weight_off for layer {li}")))?;
            sb.sub(woff, wbytes.len()).map_err(|e| EngineError::Load(format!("sb.sub weight[{li}]: {e}")))?
                .write_bytes(&wbytes).map_err(|e| EngineError::Load(format!("write weight[{li}]: {e}")))?;

            let kv_off = *meta.layer_kv_off.get(&li).ok_or_else(|| EngineError::Load(format!("meta.json: no layer_kv_off for layer {li}")))?;
            let next_off = if li + 1 < meta.nlayer {
                *meta.layer_weight_off.get(&(li + 1)).ok_or_else(|| EngineError::Load(format!("meta.json: no layer_weight_off for layer {}", li + 1)))?
            } else {
                meta.head_off
            };
            let kv_size = next_off - kv_off;
            sb.sub(kv_off, kv_size).map_err(|e| EngineError::Load(format!("sb.sub kv[{li}]: {e}")))?
                .write_bytes(&vec![0u8; kv_size]).map_err(|e| EngineError::Load(format!("zero kv[{li}]: {e}")))?;
        }

        let head_weight_path = meta.dir.join("w_head.npy");
        let head_weight: ndarray::ArrayD<u8> = read_npy(&head_weight_path).map_err(|e| EngineError::Load(format!("read {}: {e}", head_weight_path.display())))?;
        let head_weight_bytes = head_weight.into_raw_vec_and_offset().0;
        sb.sub(meta.head_off, head_weight_bytes.len()).map_err(|e| EngineError::Load(format!("sb.sub head: {e}")))?
            .write_bytes(&head_weight_bytes).map_err(|e| EngineError::Load(format!("write head weight: {e}")))?;

        sb.sync_to_device().map_err(|e| EngineError::Load(format!("sync scratch: {e}")))?;

        let elf = std::fs::read(meta.elf_path()).map_err(|e| EngineError::Load(format!("read {}: {e}", meta.elf_path().display())))?;
        let boot = dev.open_elf_resident(&elf, Some(&format!("main:{}", meta.boot))).map_err(|e| EngineError::Load(format!("open_elf_resident (boot): {e}")))?;
        boot.bind(&[&xb, &ob, &sb, &sb, &sb]).map_err(|e| EngineError::Load(format!("bind boot: {e}")))?;

        let mut rungs = HashMap::new();
        for r in &meta.rungs {
            let kern = boot.open_named(&format!("main:{}", r.name)).map_err(|e| EngineError::Load(format!("open_named {}: {e}", r.name)))?;
            kern.bind(&[&xb, &ob, &sb]).map_err(|e| EngineError::Load(format!("bind {}: {e}", r.name)))?;
            rungs.insert(r.name.clone(), kern);
        }

        Ok(LadderResidentForward { meta, embed, xb, ob, sb, boot, rungs, n_written: 0 })
    }

    /// Build `%x` for a piece of `p_len` rows at position `s` and dispatch the rung named
    /// `rung_name` (already selected by the caller), returning `first` (the sliding ring's
    /// read-window start, needed for `kvr_s`).
    fn dispatch(&mut self, rung_name: &str, split_nb: Option<usize>, x_bits: &[u16], s: usize, p_len: usize) -> Result<usize, EngineError> {
        let nt = piece_nt(p_len, self.meta.row_block);
        let first = s.saturating_sub(self.meta.sliding_window.saturating_sub(1)) / 64 * 64;

        let mut buf = vec![0u8; self.meta.xbuf];
        let x_bytes = u16s_to_le_bytes(x_bits);
        buf[..x_bytes.len()].copy_from_slice(&x_bytes);

        let mut rope_s = vec![0u16; 16 * nt * 256];
        let mut rope_g_full = vec![0u16; 16 * nt * 128];
        for row in 0..p_len {
            let pos = s + row;
            rope_s[row * 256..(row + 1) * 256].copy_from_slice(&rope_row_sliding(pos));
            rope_g_full[row * 128..(row + 1) * 128].copy_from_slice(&rope_row_global(pos));
        }
        let rope_s_bytes = u16s_to_le_bytes(&rope_s);
        let rope_g_bytes = u16s_to_le_bytes(&rope_g_full);
        buf[self.meta.rope_s_off..self.meta.rope_s_off + rope_s_bytes.len()].copy_from_slice(&rope_s_bytes);
        buf[self.meta.rope_g_off..self.meta.rope_g_off + rope_g_bytes.len()].copy_from_slice(&rope_g_bytes);

        let widths_s = sliding_widths_record(s, p_len, nt, first, self.meta.sliding_window);
        let widths_g = global_widths_record(s, p_len, nt);
        buf[self.meta.widths_s_off..self.meta.widths_s_off + widths_s.len()].copy_from_slice(&widths_s);
        buf[self.meta.widths_g_off..self.meta.widths_g_off + widths_g.len()].copy_from_slice(&widths_g);

        // The split lane's per-column widths overlay -- decode only. Offsets are architecture
        // constants (independent of which f1-family rung was picked, see module doc); only `nbc`
        // (the SELECTED rung's own `blocks` field) varies per rung. Same broadcast shape as
        // `resident_onecmd.rs` (two repeats of [16 x hi, 16 x zero], 64 i32 total).
        if let Some(nb) = split_nb {
            let nbc = nb / 8;
            for (c, &off) in self.meta.split_widths_x_off.iter().enumerate() {
                let hi = ((s + 1) as i64 - (c * nbc * 64) as i64).clamp(0, (nbc * 64) as i64) as i32;
                let block: Vec<i32> = std::iter::repeat_n(hi, 16).chain(std::iter::repeat_n(0, 16)).collect();
                let w: Vec<i32> = block.iter().chain(block.iter()).copied().collect();
                let wbytes = i32s_to_le_bytes(&w);
                let dst = self.meta.rope_g_off + off - self.meta.xrows;
                buf[dst..dst + wbytes.len()].copy_from_slice(&wbytes);
            }
        }

        self.xb.write_bytes(&buf).map_err(|e| EngineError::Device(format!("write x: {e}")))?;
        self.xb.sync_to_device().map_err(|e| EngineError::Device(format!("sync x: {e}")))?;

        self.boot.dispatch().map_err(EngineError::Device)?;

        let kern = self.rungs.get(rung_name).ok_or_else(|| EngineError::Load(format!("no rung `{rung_name}` bound")))?;
        let unit = self.meta.param_unit_bytes;
        let write_param = |name: &str, byte_value: usize| -> Result<(), EngineError> {
            let idx = *self.meta.scratchpad_params.get(name).ok_or_else(|| EngineError::Load(format!("meta.json: no scratchpad param `{name}`")))?;
            if byte_value % unit != 0 {
                return Err(EngineError::Device(format!("{name}: byte offset {byte_value} not a multiple of param_unit_bytes {unit}")));
            }
            let word = (byte_value / unit) as u64;
            kern.write_scratchpad(idx * unit, &word.to_le_bytes()[..unit.min(8)]).map_err(|e| EngineError::Device(format!("write scratchpad {name}: {e}")))
        };
        let ring_s = ring_pos(s, self.meta.s_ring);
        let ring_first = ring_pos(first, self.meta.s_ring);
        write_param("kvw_s", ring_s * self.meta.kvrow_s * 2)?;
        write_param("kvw_g", s * self.meta.kvrow_g * 2)?;
        write_param("kvr_s", ring_first * self.meta.kvrow_s * 2)?;

        kern.dispatch().map_err(EngineError::Device)?;
        self.ob.sync_from_device().map_err(|e| EngineError::Device(format!("sync ob: {e}")))?;

        self.n_written = s + p_len;
        Ok(first)
    }

    fn read_logits(&self) -> Result<Vec<f32>, EngineError> {
        let mut bytes = vec![0u8; self.meta.vocab * 2];
        self.ob.read_bytes_at(self.meta.logits_off, &mut bytes).map_err(|e| EngineError::Device(format!("read logits: {e}")))?;
        Ok(unpack_bf16_bytes(&bytes))
    }
}

impl DecodeStep for LadderResidentForward {
    fn step(&mut self, token: u32, pos: usize) -> Result<Vec<f32>, EngineError> {
        let rung = rung_for(&self.meta.rungs, 1, pos + 1)?;
        let (name, blocks) = (rung.name.clone(), rung.blocks);
        let x_bits = self.embed.embed_row_bf16(token)?;
        self.dispatch(&name, Some(blocks), &x_bits, pos, 1)?;
        let mut logits = self.read_logits()?;
        if let Some(cap) = self.meta.logit_softcap {
            let cap = cap as f32;
            for v in &mut logits {
                *v = (*v / cap).tanh() * cap;
            }
        }
        Ok(logits)
    }

    fn prefill(&mut self, tokens: &[u32], from: usize) -> Result<usize, EngineError> {
        let batchable = tokens.len().saturating_sub(1); // the last token always goes through step()
        let f2_reach = self.meta.largest_keys(2);
        let mut at = from;
        while at < batchable {
            // Cut the piece at the ring boundary and the fixed row cap, same order fwd_host uses
            // (max_piece, then pick a rung for the cut piece's own end).
            let room = max_piece(at, self.meta.pmax, self.meta.s_ring);
            let end = (at + room).min(batchable);
            let p_len = end - at;
            let keys_needed = end;
            // Past the largest implemented f2 rung: decline here (return `at`, unchanged progress
            // for this call) and let the generator's per-token loop finish via `step()`'s f1
            // ladder, which reaches 262144 -- see module doc.
            if keys_needed > f2_reach {
                break;
            }
            let rung = rung_for(&self.meta.rungs, 2, keys_needed)?;
            let rung_name = rung.name.clone();
            let piece = &tokens[at..end];
            let mut x_bits = Vec::with_capacity(piece.len() * self.meta.d_model);
            for &tok in piece {
                x_bits.extend_from_slice(&self.embed.embed_row_bf16(tok)?);
            }
            self.dispatch(&rung_name, None, &x_bits, at, p_len)?;
            at = end;
        }
        Ok(at)
    }

    fn prefill_batch(&self) -> Option<usize> {
        Some(self.meta.pmax)
    }

    fn reset(&mut self) -> Result<CacheState, EngineError> {
        for li in 0..self.meta.nlayer {
            let kv_off = self.meta.layer_kv_off[&li];
            let next_off = if li + 1 < self.meta.nlayer { self.meta.layer_weight_off[&(li + 1)] } else { self.meta.head_off };
            let size = next_off - kv_off;
            self.sb.sub(kv_off, size).map_err(|e| EngineError::Device(format!("sb.sub kv[{li}] on reset: {e}")))?
                .write_bytes(&vec![0u8; size]).map_err(|e| EngineError::Device(format!("zero kv[{li}] on reset: {e}")))?;
        }
        self.sb.sync_to_device().map_err(|e| EngineError::Device(format!("sync scratch on reset: {e}")))?;
        self.n_written = 0;
        Ok(CacheState::Cleared)
    }

    fn max_context(&self) -> Option<usize> {
        Some(self.meta.largest_keys(1))
    }

    /// Real device BO bytes: `%x` + `%o` + `%s`, so the generic `memory_ceiling_mb` accounting
    /// (`npu-runtime::loader::EngineLoader`/`registry::ensure_resident`) can evict THIS model or
    /// evict the shipped `gemma4-12b` to make room, instead of reporting the default 0 and letting
    /// both look free to load at once -- the residency conflict the coordinator flagged for
    /// `rf48L`'s ~11.8 GB scratch.
    fn bo_bytes(&self) -> u64 {
        (self.meta.xbuf + self.meta.obuf_f1.max(self.meta.obuf_f2) + self.meta.scratch_bytes) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rf48l_rungs() -> Vec<Rung> {
        // The real rf48L ladder (fwd_layout.json, 2026-09-29): 6 f1-family (decode) + 3 f2-family
        // (prefill) rungs, ascending.
        let split = |name: &str, blocks: usize, keys: usize| Rung { name: name.into(), nt: 1, blocks, keys, kind: "split".into() };
        let seg = |name: &str, blocks: usize, keys: usize| Rung { name: name.into(), nt: 2, blocks, keys, kind: "seg".into() };
        vec![
            split("f1", 128, 8192),
            split("f1s256", 256, 16384),
            split("f1s512", 512, 32768),
            split("f1s1024", 1024, 65536),
            split("f1s2048", 2048, 131072),
            split("f1s4096", 4096, 262144),
            seg("f2", 64, 4096),
            seg("f2w256", 256, 16384),
            seg("f2w1024", 1024, 65536),
        ]
    }

    #[test]
    fn rung_selection_picks_the_smallest_window_that_holds_the_keys() {
        let rungs = rf48l_rungs();
        assert_eq!(rung_for(&rungs, 1, 1).unwrap().name, "f1");
        assert_eq!(rung_for(&rungs, 1, 8192).unwrap().name, "f1");
        assert_eq!(rung_for(&rungs, 1, 8193).unwrap().name, "f1s256");
        assert_eq!(rung_for(&rungs, 1, 65536).unwrap().name, "f1s1024");
        assert_eq!(rung_for(&rungs, 1, 262144).unwrap().name, "f1s4096");
        assert_eq!(rung_for(&rungs, 2, 4096).unwrap().name, "f2");
        assert_eq!(rung_for(&rungs, 2, 4097).unwrap().name, "f2w256");
        assert_eq!(rung_for(&rungs, 2, 65536).unwrap().name, "f2w1024");
    }

    #[test]
    fn rung_selection_refuses_past_the_largest_rung_naming_both_numbers() {
        let rungs = rf48l_rungs();
        let e = rung_for(&rungs, 1, 262145).unwrap_err();
        let msg = e.to_string();
        assert!(msg.contains("262145") && msg.contains("262144"), "{msg}");
        let e = rung_for(&rungs, 2, 65537).unwrap_err();
        let msg = e.to_string();
        assert!(msg.contains("65537") && msg.contains("65536"), "{msg}");
    }

    #[test]
    fn ring_pos_wraps_at_the_ring_size() {
        assert_eq!(ring_pos(0, 1280), 0);
        assert_eq!(ring_pos(1279, 1280), 1279);
        assert_eq!(ring_pos(1280, 1280), 0);
        assert_eq!(ring_pos(6083, 1280), 6083 % 1280);
    }

    #[test]
    fn max_piece_is_uncut_away_from_a_boundary() {
        // A 32-row piece starting well clear of the next multiple of 1280 fits whole.
        assert_eq!(max_piece(0, 32, 1280), 32);
        assert_eq!(max_piece(100, 32, 1280), 32);
    }

    #[test]
    fn max_piece_cuts_a_piece_that_would_cross_a_ring_multiple() {
        // Starting 10 rows before the 1280 boundary, only 10 rows fit before it.
        assert_eq!(max_piece(1270, 32, 1280), 10);
        // Starting exactly on a boundary: the ring's own multiple, not the previous one.
        assert_eq!(max_piece(1280, 32, 1280), 32);
        // Two ring widths in, same shape.
        assert_eq!(max_piece(2556, 32, 1280), 4);
    }

    #[test]
    fn max_piece_never_returns_zero() {
        // s % c is always < c, so c - s % c is always >= 1: a piece always makes SOME progress.
        for s in [0usize, 1, 1279, 1280, 2559, 1_000_000] {
            assert!(max_piece(s, 32, 1280) >= 1, "s={s}");
        }
    }

    #[test]
    fn largest_keys_reads_the_max_across_an_nt_family() {
        let meta_rungs = rf48l_rungs();
        let largest_f1 = meta_rungs.iter().filter(|r| r.nt == 1).map(|r| r.keys).max().unwrap();
        let largest_f2 = meta_rungs.iter().filter(|r| r.nt == 2).map(|r| r.keys).max().unwrap();
        assert_eq!(largest_f1, 262144);
        assert_eq!(largest_f2, 65536);
    }
}
