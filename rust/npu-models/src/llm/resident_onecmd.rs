//! S2 follow-up: serves the device lane's ONE-COMMAND resident-forward build (`rf48s`-shaped),
//! replacing `resident_raw.rs`'s per-layer-dispatch driver as `gemma4-12b-resident`'s backend.
//!
//! `rf48s` splices all 48 layers (+ the LM head, `f1` only) into ONE runtime sequence per piece --
//! `f1` (48 layers + head, one per decode token) or `f2` (48 layers, prefill pieces up to 32 rows,
//! no head) -- on THREE fixed arena buffers (`%x` input, `%o` output, `%s` scratch, FusedArena
//! order) plus three `aiex.scratchpad_parameter` words (`kvw_s`/`kvw_g`/`kvr_s`, Option C -- unlike
//! `rls1`'s plain-args ABI, so this uses `ElfResident::write_scratchpad`, not `Bo::sub` kv views).
//! Every layer's weights, its K/V cache, and the head's weight stream live in `%s` RESIDENT for the
//! model's whole lifetime -- uploaded once at `open`, never re-read per dispatch, unlike the
//! per-layer driver's per-token weight-file reads. Position-dependent addressing is three
//! scratchpad words instead of `Bo::sub` views into a big KV arena.
//!
//! Reuses `resident_raw.rs`'s embedding dequant and RoPE/widths functions unchanged -- same
//! reference functions (`qkv_ref`/`glob_ref`/`rld_run`/`rlo_run`), same conventions, verified
//! against the same oracle. The on-device head still normalizes internally (this module inherits
//! the S2 double-norm lesson: never RMSNorm a row on the host before this build's head phase).

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
use crate::llm::resident_raw::{artifact_relative_path, global_widths_record, piece_nt, rope_row_global, rope_row_sliding, sliding_widths_record, EmbedHeadPack};

/// `rf48s/meta.json`'s own schema (`kind: "resident_forward_onecmd"`) -- the byte layout of the
/// three arena buffers plus the scratchpad param indices, all computed once by
/// `rforward.Plan`/`rsplitl` (the authority) and dumped, not re-derived here. See that module's
/// doc for the field meanings; names mirror `rforward.py`'s own (`plan.XF`, `plan.wbase`, ...).
#[derive(Debug, Clone)]
pub struct OneCmdMeta {
    pub dir: PathBuf,
    pub elf_name: String,
    pub boot: String,
    pub f1: String,
    pub f2: String,
    pub weight_dir: PathBuf,
    pub embedding_store: PathBuf,
    pub d_model: usize,
    pub nlayer: usize,
    pub full_attention_layers: Vec<usize>,
    pub row_block: usize,
    pub pcap_t: usize,
    pub pmax: usize,
    pub split_nb: usize,
    pub nbw: usize,
    pub sliding_window: usize,
    pub s_cap: usize,
    pub g_cap: usize,
    pub kvrow_s: usize,
    pub kvrow_g: usize,
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
    pub head_ncol: usize,
    pub head_wcol: usize,
    pub head_outb: usize,
    pub vocab: usize,
    pub logit_softcap: Option<f64>,
    pub scratchpad_params: HashMap<String, usize>,
    pub split_widths_x_off: Vec<usize>,
    pub split_kvr_elems: usize,
}

impl OneCmdMeta {
    pub fn load(dir: &Path) -> Result<OneCmdMeta, EngineError> {
        let meta_path = dir.join("meta.json");
        let bytes = fs::read(&meta_path).map_err(|e| EngineError::Load(format!("read {}: {e}", meta_path.display())))?;
        let v: serde_json::Value = serde_json::from_slice(&bytes).map_err(|e| EngineError::Load(format!("parse {}: {e}", meta_path.display())))?;
        let ctx = |msg: String| EngineError::Load(format!("{}: {msg}", meta_path.display()));
        let kind = v.get("kind").and_then(|x| x.as_str()).ok_or_else(|| ctx("missing `kind`".into()))?;
        if kind != "resident_forward_onecmd" {
            return Err(ctx(format!("kind {kind:?}, expected \"resident_forward_onecmd\"")));
        }
        let s = |k: &str| -> Result<String, EngineError> {
            v.get(k).and_then(|x| x.as_str()).map(|s| s.to_string()).ok_or_else(|| ctx(format!("missing `{k}`")))
        };
        let u = |k: &str| -> Result<usize, EngineError> {
            v.get(k).and_then(|x| x.as_u64()).map(|x| x as usize).ok_or_else(|| ctx(format!("missing/non-numeric `{k}`")))
        };
        let path = |k: &str| -> Result<PathBuf, EngineError> { Ok(artifact_relative_path(dir, &s(k)?)) };
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

        Ok(OneCmdMeta {
            dir: dir.to_path_buf(),
            elf_name: s("elf")?,
            boot: s("boot")?,
            f1: s("f1")?,
            f2: s("f2")?,
            weight_dir: path("weight_dir")?,
            embedding_store: path("embedding_store")?,
            d_model: u("d_model")?,
            nlayer: u("nlayer")?,
            full_attention_layers,
            row_block: u("row_block")?,
            pcap_t: u("pcap_t")?,
            pmax: u("pmax")?,
            split_nb: u("split_nb")?,
            nbw: u("nbw")?,
            sliding_window: u("sliding_window")?,
            s_cap: u("s_cap")?,
            g_cap: u("g_cap")?,
            kvrow_s: u("kvrow_s")?,
            kvrow_g: u("kvrow_g")?,
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
            head_ncol: u("head_ncol")?,
            head_wcol: u("head_wcol")?,
            head_outb: u("head_outb")?,
            vocab: u("vocab")?,
            logit_softcap: v.get("logit_softcap").and_then(|x| x.as_f64()),
            scratchpad_params,
            split_widths_x_off,
            split_kvr_elems: u("split_kvr_elems")?,
        })
    }

    pub fn elf_path(&self) -> PathBuf {
        self.dir.join(&self.elf_name)
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

/// The `DecodeStep` backend driving `rf48s`. One `ElfResident` per control code
/// (`boot`/`f1`/`f2`), all sharing ONE hw_context, all bound ONCE at `open` (the three arena BOs
/// never change identity across dispatches -- only their CONTENTS and the three scratchpad words
/// do).
pub struct OneCommandResidentForward {
    meta: OneCmdMeta,
    embed: EmbedHeadPack,
    xb: Bo,
    ob: Bo,
    #[allow(dead_code)] // kept alive: %s is referenced only through `boot`/`f1`/`f2`'s bound args
    sb: Bo,
    boot: ElfResident,
    f1: ElfResident,
    f2: ElfResident,
    n_written: usize,
}

impl OneCommandResidentForward {
    pub fn open(dev: &Rc<Device>, dir: &Path) -> Result<OneCommandResidentForward, EngineError> {
        let meta = OneCmdMeta::load(dir)?;
        let embed = EmbedHeadPack::open(&meta.embedding_store)?;

        let xb = dev.alloc_bo_raw(meta.xbuf, FLAG_HOST_ONLY, 0).map_err(|e| EngineError::Load(format!("alloc xb: {e}")))?;
        let ob_size = meta.obuf_f1.max(meta.obuf_f2);
        let ob = dev.alloc_bo_raw(ob_size, FLAG_HOST_ONLY, 0).map_err(|e| EngineError::Load(format!("alloc ob: {e}")))?;
        let sb = dev.alloc_bo_raw(meta.scratch_bytes, FLAG_HOST_ONLY, 0).map_err(|e| EngineError::Load(format!("alloc scratch ({} bytes): {e}", meta.scratch_bytes)))?;

        // Every layer's weight stream, its K/V cache (zeroed), and the head's weight stream go
        // into %s ONCE here -- resident for the model's whole lifetime, not re-read per dispatch
        // (`resident_raw.rs`'s per-layer driver re-read every layer's weight file every token;
        // this build's whole point is that %s is the resident arena the S0 design always meant).
        for li in 0..meta.nlayer {
            let weight_path = meta.weight_dir.join(format!("w{li}.npy"));
            let wbytes: Array1<u8> = read_npy(&weight_path).map_err(|e| EngineError::Load(format!("read {}: {e}", weight_path.display())))?;
            let wbytes = wbytes.into_raw_vec_and_offset().0;
            let woff = *meta.layer_weight_off.get(&li).ok_or_else(|| EngineError::Load(format!("meta.json: no layer_weight_off for layer {li}")))?;
            sb.sub(woff, wbytes.len()).map_err(|e| EngineError::Load(format!("sb.sub weight[{li}]: {e}")))?
                .write_bytes(&wbytes).map_err(|e| EngineError::Load(format!("write weight[{li}]: {e}")))?;

            let kv_off = *meta.layer_kv_off.get(&li).ok_or_else(|| EngineError::Load(format!("meta.json: no layer_kv_off for layer {li}")))?;
            // The per-layer kv byte count is `layer_weight_off[li+1] - kv_off` (or, for the last
            // layer, `head_off - kv_off`): the NEXT boundary `rforward.Plan` actually allocated,
            // not a size recomputed from g_cap/split_kvr_elems -- the two agree when the split
            // lane's window is the wider one, but reading the boundary directly needs no such
            // case split and cannot silently drift from what was allocated.
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

        let elf = crate::llm::artifact::read_elf_bytes(&meta.elf_path())?;
        let boot = dev.open_elf_resident(&elf, Some(&format!("main:{}", meta.boot))).map_err(|e| EngineError::Load(format!("open_elf_resident (boot): {e}")))?;
        // boot's own kernel ABI is 5 args (the shared "main" signature every control code in this
        // ELF carries); fwd_stack.py pads slots 3/4 with the SAME %s buffer boot never reads --
        // matched here rather than guessed, since a 3-arg bind on `boot` specifically was never
        // gated by anything in this tree.
        boot.bind(&[&xb, &ob, &sb, &sb, &sb]).map_err(|e| EngineError::Load(format!("bind boot: {e}")))?;
        let f1 = boot.open_named(&format!("main:{}", meta.f1)).map_err(|e| EngineError::Load(format!("open_named f1: {e}")))?;
        f1.bind(&[&xb, &ob, &sb]).map_err(|e| EngineError::Load(format!("bind f1: {e}")))?;
        let f2 = boot.open_named(&format!("main:{}", meta.f2)).map_err(|e| EngineError::Load(format!("open_named f2: {e}")))?;
        f2.bind(&[&xb, &ob, &sb]).map_err(|e| EngineError::Load(format!("bind f2: {e}")))?;

        Ok(OneCommandResidentForward { meta, embed, xb, ob, sb, boot, f1, f2, n_written: 0 })
    }

    /// Build `%x` for a piece of `p_len` rows (`x_bits`, `p_len * d_model` bf16-bit elements)
    /// starting at position `s`, dispatch `code` (`"f1"` or `"f2"`), and return `first` (the
    /// sliding ring's read-window start, needed by the caller for `kvr_s`).
    fn dispatch(&mut self, code_is_f1: bool, x_bits: &[u16], s: usize, p_len: usize) -> Result<usize, EngineError> {
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

        // The split lane's per-column widths overlay -- decode only (P == 1), written into the
        // global RoPE region past its one-block table (`rsplitl.widths_x`'s own doc). Absolute
        // offsets are `rope_g_off + split_widths_x_off[c] - xrows`: `split_widths_x_off` is
        // `rsplitl.widths_x`'s own single-layer-xbuf-relative value (its docstring: "%x byte
        // offset... in the global RoPE region"), and `rforward.retarget` remaps every RoPE offset
        // the same way (`plan.RG + off - plan.XROWS`) -- this mirrors that remap, not a new one.
        if self.meta.split_nb > 0 && p_len == 1 {
            let nbc = self.meta.split_nb / 8;
            for (c, &off) in self.meta.split_widths_x_off.iter().enumerate() {
                let hi = ((s + 1) as i64 - (c * nbc * 64) as i64).clamp(0, (nbc * 64) as i64) as i32;
                // `w = np.zeros((1,2,2,16), np.int32); w[:,:,0] = hi` -- numpy broadcasts the
                // scalar over axes 0 and 3, leaving axis 2 as the only real split: flattened
                // (C-order) that is two repeats of [16 x hi, 16 x zero], 64 i32 total, NOT an
                // interleaved [hi,0,hi,0,...] (the bug this replaces: right element, wrong shape).
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

        let kern = if code_is_f1 { &self.f1 } else { &self.f2 };
        let unit = 4usize; // %s is i32 words
        let write_param = |name: &str, byte_value: usize| -> Result<(), EngineError> {
            let idx = *self.meta.scratchpad_params.get(name).ok_or_else(|| EngineError::Load(format!("meta.json: no scratchpad param `{name}`")))?;
            let word = (byte_value / unit) as u32;
            kern.write_scratchpad(idx * unit, &word.to_le_bytes()).map_err(|e| EngineError::Device(format!("write scratchpad {name}: {e}")))
        };
        write_param("kvw_s", s * self.meta.kvrow_s * 2)?;
        write_param("kvw_g", s * self.meta.kvrow_g * 2)?;
        write_param("kvr_s", first * self.meta.kvrow_s * 2)?;

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

impl DecodeStep for OneCommandResidentForward {
    fn step(&mut self, token: u32, pos: usize) -> Result<Vec<f32>, EngineError> {
        let x_bits = self.embed.embed_row_bf16(token)?;
        self.dispatch(true, &x_bits, pos, 1)?;
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
        let mut at = from;
        while at < batchable {
            let end = (at + self.meta.pmax).min(batchable);
            let piece = &tokens[at..end];
            let mut x_bits = Vec::with_capacity(piece.len() * self.meta.d_model);
            for &tok in piece {
                x_bits.extend_from_slice(&self.embed.embed_row_bf16(tok)?);
            }
            self.dispatch(false, &x_bits, at, piece.len())?;
            at = end;
        }
        Ok(at)
    }

    fn prefill_batch(&self) -> Option<usize> {
        Some(self.meta.pmax)
    }

    fn reset(&mut self) -> Result<CacheState, EngineError> {
        // Every layer's K/V region is re-zeroed the same way `open` zeroed it initially.
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
        Some(max_context_bound(self.meta.s_cap, self.meta.pmax, self.meta.nbw, self.meta.sliding_window))
    }

    /// Real device BO bytes (`%x`+`%o`+`%s`) -- see `resident_ladder::LadderResidentForward::bo_bytes`
    /// for why this exists: the default `0` let this model's ~7.6 GB scratch look free to the
    /// `memory_ceiling_mb` accountant, which is exactly the residency conflict rf48L raised.
    fn bo_bytes(&self) -> u64 {
        (self.meta.xbuf + self.meta.obuf_f1.max(self.meta.obuf_f2) + self.meta.scratch_bytes) as u64
    }
}

/// The hard, tight bound `max_context()` reports: the largest position at which a dispatch is
/// still guaranteed to keep every K/V write inside the sliding cache's allocated `s_cap` rows.
///
/// The sliding layers' cache is a LINEAR (non-wrapping) buffer of `s_cap` rows (`rld_run.py`'s
/// own docstring: "the cache is one host buffer", addressed at `n_past * KVROW * 2` with no
/// modulo anywhere in that module or `rforward.py`'s scratchpad writes) -- confirmed from
/// source, not assumed.
///
/// `s_cap` alone is NOT the safe bound: `f2` (prefill) is compiled for a FIXED `nt=2`
/// (`rlayer_design.emit`'s naming -- "f1"/"f2" name the compiled `nt`, not a step index), so
/// its K/V write BD length is a COMPILE-TIME constant of `2 * 16 * 256` elements per head
/// (`rlayer_design.py`'s `len = {2 * 16 * nt * 256}`), i.e. exactly `pmax` (32) rows, for EVERY
/// `f2` dispatch regardless of the piece's real row count -- a 5-row piece still writes 32 rows
/// of (finite, zero-padded) K/V starting at its own position. So the last piece dispatched at
/// position `at < max_context` writes through `at + pmax`, and that must not exceed `s_cap`.
/// Subtracting `pmax` up front makes every `at < max_context` satisfy `at + pmax <= s_cap` by
/// construction, with no separate per-dispatch check needed -- and covers decode (`f1`, a fixed
/// single-row write) with room to spare, since it needs only 1 row of slack, not `pmax`.
///
/// The window READ is bounded too: each dispatch reads `nbw` 64-row blocks of the sliding
/// cache from `first` (`rlayer_design.py`'s `sizes = [NBW, 4, 64, 64]`), so `first + nbw * 64`
/// must stay within `s_cap`. Past it the read takes the next region's bytes as masked keys, and
/// a non-finite one poisons the x.V MAC (0 x NaN). On device (rfl6, s_cap 2048, nbw 20) the
/// first non-finite position is exactly 1855, the value this returns for those numbers.
pub fn max_context_bound(s_cap: usize, pmax: usize, nbw: usize, sliding_window: usize) -> usize {
    let write_bound = s_cap.saturating_sub(pmax);
    let Some(last_first) = s_cap.checked_sub(nbw * 64) else { return 0 };
    let read_bound = last_first / 64 * 64 + 63 + sliding_window;
    write_bound.min(read_bound)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_context_bound_is_s_cap_minus_the_largest_piece() {
        // rf48s: s_cap=2048, pmax=32, nbw=20, window 1024 -- the real gemma4-12b-resident numbers.
        // The window read binds first: 2016 would read up to 192 rows past the cache.
        assert_eq!(max_context_bound(2048, 32, 20, 1024), 1855);
        // A 16-block read never passes the rows the window itself needs; the write bound binds.
        assert_eq!(max_context_bound(4096, 32, 16, 1024), 4064);
        assert_eq!(max_context_bound(1024, 32, 20, 1024), 0);
    }

    #[test]
    fn max_context_bound_never_underflows_a_tiny_cache() {
        assert_eq!(max_context_bound(16, 32, 20, 1024), 0);
    }
}
