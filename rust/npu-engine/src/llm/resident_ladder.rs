//! `rf48L`-shaped backend: the 256k-context ladder. Extends `resident_onecmd.rs`'s one-command
//! design (one `%x`/`%o`/`%s` FusedArena, weights/KV resident for the model's whole lifetime) with
//! TWO things that build did not have:
//!
//! - **A rung ladder, not one fixed command per phase.** The global window is a property of the
//!   compiled sequence (`rlayer_design.emit`'s naming), so each reach (8192..262144 keys for
//!   decode, 4096..65536 for prefill) is its own control code. The host picks the smallest rung
//!   whose window holds `n_past + P` keys -- `fwd_host.Forward.rung`, ported verbatim below.
//! - **One sliding-cache ring per layer (`s_ring` = 1280) in 64-row blocks** (prototype `kvring.py`).
//!   Each command writes its rows and reads its window in two spans, each a runtime offset plus a
//!   runtime granule count; a write span never crosses a block, so never the ring end. The span
//!   rule is `ring_params`; `ring_fits` bounds every span by `meta.json`'s `s_ring_layout`, which the
//!   generator wrote from the same region size it built against.
//! - **Two scratch BOs**: `%s` holds the weights (static BDs only), `%k` the caches (every BD whose
//!   address moves with the position), so a runtime-offset transfer cannot reach a weight byte.
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
use crate::llm::resident::ring_read_first;
use crate::telemetry::ArmProvenance;
use crate::llm::multimodal::MediaEmbeds;
use crate::llm::resident_raw::{global_widths_record, piece_nt, rope_row_global, rope_row_sliding, sliding_widths_record, EmbedHeadPack};

/// Pause before each re-run of a non-finite dispatch. A policy, measured 2026-09-30 on rf48C under
/// desktop load: failures come in bursts, 63% clear on an immediate re-run, and 7 of 8 that failed
/// four immediate re-runs cleared after a 0.5-4 s pause (rf-forward-intermittent-nonfinite).
const NONFINITE_BACKOFF_MS: [u64; 8] = [0, 0, 250, 500, 1000, 2000, 4000, 8000];

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
/// small (rf48L: 6 f1 + 3 f2) and read once per dispatch, so this is not worth a binary search.
pub fn rung_for<'a>(rungs: &'a [Rung], nt: usize, keys_needed: usize) -> Result<&'a Rung, EngineError> {
    rungs.iter().filter(|r| r.nt == nt).find(|r| r.keys >= keys_needed).ok_or_else(|| {
        let largest = rungs.iter().filter(|r| r.nt == nt).last().map(|r| r.keys).unwrap_or(0);
        EngineError::Unsupported(format!("no nt={nt} rung holds {keys_needed} keys (largest {largest})"))
    })
}

/// `fwd_host.Forward.ring`: a sliding-cache position's ring slot.
pub fn ring_pos(pos: usize, c: usize) -> usize {
    pos % c
}

/// `fwd_host.Forward.piece` + `family`: the next prefill piece at `at` with `left` rows to go. Its
/// rows come from the largest family whose windows reach past `at`: `row_block * nt`, cut at that
/// family's largest window. Its rung is the smallest family covering those rows, then that
/// family's smallest window. `None` past every prefill rung's window.
pub fn prefill_piece(rungs: &[Rung], at: usize, left: usize, row_block: usize) -> Option<(usize, &Rung)> {
    let seg = || rungs.iter().filter(|r| r.kind == "seg");
    let reach = |nt: usize| seg().filter(|r| r.nt == nt).map(|r| r.keys).max().unwrap_or(0);
    let mut nts: Vec<usize> = seg().map(|r| r.nt).collect();
    nts.sort_unstable();
    nts.dedup();
    let big = *nts.iter().rev().find(|&&nt| reach(nt) > at)?;
    let p = left.min(row_block * big).min(reach(big) - at);
    let nt = *nts.iter().find(|&&nt| row_block * nt >= p && reach(nt) >= at + p)?;
    rung_for(rungs, nt, at + p).ok().map(|r| (p, r))
}

/// One ring BD family of `s_ring_layout` (`kvring.describe`), in bytes: the static offsets of its
/// BDs, the extent one granule touches, and the step between granules.
#[derive(Debug, Clone)]
pub struct RingBd {
    pub min_off: usize,
    pub max_off: usize,
    pub extent: usize,
    pub granule_stride: usize,
}

/// `meta.json`'s `s_ring_layout` (`kvring.describe`): the blocked ring the generator built.
#[derive(Debug, Clone)]
pub struct RingLayout {
    pub block_rows: usize,
    pub blocks: usize,
    pub window_blocks: usize,
    pub slab_bytes: usize,
    pub block_bytes: usize,
    pub row_bytes: usize,
    pub region_bytes: usize,
    pub write: RingBd,
    pub read: RingBd,
}

/// `kvring.write_spans`: (first slot, rows) of span A (to the end of `s`'s block, at most
/// `rows - 1`) and span B (the rest, from the next position's slot).
pub fn write_spans(s: usize, rows: usize, c: usize, blk: usize) -> [(usize, usize); 2] {
    let r = s % c;
    let a = (rows - 1).min(blk - r % blk);
    [(r, a), ((s + a) % c, rows - a)]
}

/// `kvring.read_spans`: (first block, blocks) of span A (to the ring end, at most `nbw - 1`) and
/// span B (the rest, from block 0) of the `nbw`-block window from `first`.
pub fn read_spans(first: usize, nbw: usize, c: usize, blk: usize) -> [(usize, usize); 2] {
    let (nb, b0) = (c / blk, first % c / blk);
    let na = nbw.min(nb - b0);
    if na == nbw {
        return [(b0, nbw - 1), ((b0 + nbw - 1) % nb, 1)];
    }
    [(b0, na), (0, nbw - na)]
}

/// Keep the caches across requests unless `NPU_RESIDENT_REUSE_KV=0` (see the flag's entry).
fn reuse_kv() -> bool {
    std::env::var("NPU_RESIDENT_REUSE_KV").ok().as_deref() != Some("0")
}

/// The positions `[lo, hi)` whose sliding-ring slots hold their own K/V, as dispatches write them.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RingValid {
    pub lo: usize,
    pub hi: usize,
}

impl RingValid {
    /// A dispatch at `s` writes `p_len` real rows and pads to `rows`, in a ring of `c` slots. Its
    /// rows land on the slots of `[s + rows - c, s)` too, and its padding rows are not positions.
    /// Old valid positions stay contiguous with the new ones only if the write starts inside them.
    pub fn write(self, s: usize, p_len: usize, rows: usize, c: usize) -> RingValid {
        let lo = if self.lo <= s && s <= self.hi { self.lo } else { s };
        RingValid { lo: lo.max((s + rows).saturating_sub(c)), hi: s + p_len }
    }

    /// Whether a resume at `r` finds every sliding position its window reads, `[first(r), r)`.
    pub fn keeps_the_window(self, r: usize, window: usize) -> bool {
        r == 0 || (ring_read_first(r, window) >= self.lo && r <= self.hi)
    }
}

/// What `npu stats` names this build by: the ELF's content hash, the toolchain instance it was built
/// with (`gen_env.txt`'s `RF_INST`, a directory named by the `toolchain.lock` hash), and its
/// `rlayer_design` flags (`gen_args.txt`) plus the `RF_*` build environment.
pub fn build_provenance(dir: &Path, elf: &[u8], max_seq: usize) -> ArmProvenance {
    use sha2::{Digest, Sha256};
    let read = |f: &str| fs::read_to_string(dir.join(f)).unwrap_or_default();
    let env = read("gen_env.txt");
    let toolchain = env.lines().find_map(|l| l.strip_prefix("RF_INST="))
        .and_then(|p| Path::new(p.trim()).file_name()).map(|n| n.to_string_lossy().into_owned());
    let mut flags: Vec<String> = env.lines().map(str::trim)
        .filter(|l| l.starts_with("RF_") && !l.starts_with("RF_INST=")).map(str::to_string).collect();
    flags.extend(read("gen_args.txt").split_whitespace().map(str::to_string));
    ArmProvenance {
        fusion_flags: flags,
        max_seq: u32::try_from(max_seq).ok(),
        artifact_path: Some(dir.display().to_string()),
        artifact_hash: Some(Sha256::digest(elf).iter().take(6).map(|b| format!("{b:02x}")).collect()),
        toolchain_pin_hash: toolchain,
        ..ArmProvenance::default()
    }
}

/// One scratchpad parameter's value: a byte offset (written as `bytes / param_unit_bytes`) or a
/// count of granules past the BD's static one (written raw).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RingParam {
    Bytes(usize),
    Count(usize),
}

/// `kvring.params`: every ring parameter of a command writing `rows` rows at `s` and reading the
/// window from `first`, in `layout`'s bytes.
pub fn ring_params(l: &RingLayout, s: usize, rows: usize, first: usize) -> Vec<(&'static str, RingParam)> {
    let c = l.blocks * l.block_rows;
    let slot = |t: usize| t / l.block_rows * l.block_bytes + t % l.block_rows * l.row_bytes;
    let [(wa, na), (wb, nb)] = write_spans(s, rows, c, l.block_rows);
    let [(ra, ma), (rb, mb)] = read_spans(first, l.window_blocks, c, l.block_rows);
    vec![
        ("kvw_a", RingParam::Bytes(slot(wa))),
        ("kvw_an", RingParam::Count(na - 1)),
        ("kvw_b", RingParam::Bytes(slot(wb))),
        ("kvw_bn", RingParam::Count(nb - 1)),
        ("kvr_a", RingParam::Bytes(ra * l.block_bytes)),
        ("kvr_an", RingParam::Count(ma - 1)),
        ("kvr_b", RingParam::Bytes(rb * l.block_bytes)),
        ("kvr_bn", RingParam::Count(mb - 1)),
    ]
}

/// K059 (`kvring.fits`): every span of `p` inside the ring region, a write span inside one slab,
/// a read span on whole blocks -- bounded by the layout the generator recorded, not re-derived.
pub fn ring_fits(l: &RingLayout, p: &[(&str, RingParam)]) -> Result<(), String> {
    let get = |k: &str| p.iter().find(|(n, _)| *n == k).map(|(_, v)| *v);
    for (bd, pre, write) in [(&l.write, "kvw", true), (&l.read, "kvr", false)] {
        for sp in ["a", "b"] {
            let (Some(RingParam::Bytes(off)), Some(RingParam::Count(n))) = (get(&format!("{pre}_{sp}")), get(&format!("{pre}_{sp}n"))) else {
                return Err(format!("{pre}_{sp}: missing or mistyped parameter"));
            };
            let hi = bd.max_off + off + n * bd.granule_stride + bd.extent;
            if hi > l.region_bytes {
                return Err(format!("{pre}_{sp}: [{}, {hi}) outside the {}-byte ring region", bd.min_off + off, l.region_bytes));
            }
            if write && off % l.slab_bytes + n * bd.granule_stride + bd.extent > l.slab_bytes {
                return Err(format!("{pre}_{sp}: {} rows from byte {off} cross a slab", n + 1));
            }
            if !write && (off % l.block_bytes != 0 || n + 1 > l.blocks) {
                return Err(format!("{pre}_{sp}: {} blocks from byte {off} are not whole blocks of the ring", n + 1));
            }
        }
    }
    Ok(())
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
    pub cache_bytes: usize,
    pub s_ring_layout: RingLayout,
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
        let rl = v.get("s_ring_layout").ok_or_else(|| ctx("missing `s_ring_layout` (a mirrored-ring artifact predates the single ring)".into()))?;
        let ru = |o: &serde_json::Value, k: &str| -> Result<usize, EngineError> {
            o.get(k).and_then(|x| x.as_u64()).map(|x| x as usize).ok_or_else(|| ctx(format!("s_ring_layout: missing/non-numeric `{k}`")))
        };
        let ring_bd = |k: &str| -> Result<RingBd, EngineError> {
            let o = rl.get(k).ok_or_else(|| ctx(format!("s_ring_layout: missing `{k}`")))?;
            Ok(RingBd { min_off: ru(o, "min_off")?, max_off: ru(o, "max_off")?, extent: ru(o, "extent")?, granule_stride: ru(o, "granule_stride")? })
        };
        let s_ring_layout = RingLayout {
            block_rows: ru(rl, "block_rows")?,
            blocks: ru(rl, "blocks")?,
            window_blocks: ru(rl, "window_blocks")?,
            slab_bytes: ru(rl, "slab_bytes")?,
            block_bytes: ru(rl, "block_bytes")?,
            row_bytes: ru(rl, "row_bytes")?,
            region_bytes: ru(rl, "region_bytes")?,
            write: ring_bd("write")?,
            read: ring_bd("read")?,
        };
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

        let (c, rl) = (u("s_ring")?, &s_ring_layout);
        if rl.blocks * rl.block_rows != c || rl.region_bytes != u("s_rows")? * u("kvrow_s")? * 2 {
            return Err(ctx(format!("s_ring_layout ({} blocks of {} rows, {} bytes) is not the ring s_ring/s_rows/kvrow_s size", rl.blocks, rl.block_rows, rl.region_bytes)));
        }
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
            cache_bytes: u("cache_bytes")?,
            s_ring_layout,
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
    pub(crate) fn largest_keys(&self, nt: usize) -> usize {
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
    kb: Bo,
    boot: ElfResident,
    rungs: HashMap<String, ElfResident>,
    /// What a resume is measured against ([`RingValid::keeps_the_window`]).
    ring_valid: RingValid,
    /// A dispatch failed or stayed non-finite, so the caches may hold NaN rows (K034: a masked NaN
    /// V row still poisons the output). The next `reset` zeroes them.
    poisoned: bool,
    provenance: ArmProvenance,
    /// This generation's tower rows, gathered in place of the text embedding at their positions.
    media: MediaEmbeds,
}

impl LadderResidentForward {
    pub fn open(dev: &Rc<Device>, dir: &Path) -> Result<LadderResidentForward, EngineError> {
        let meta = LadderMeta::load(dir)?;
        let embed = EmbedHeadPack::open(&meta.embedding_store)?;

        let xb = dev.alloc_bo_raw(meta.xbuf, FLAG_HOST_ONLY, 0).map_err(|e| EngineError::Load(format!("alloc xb: {e}")))?;
        let ob_size = meta.obuf_f1.max(meta.obuf_f2);
        let ob = dev.alloc_bo_raw(ob_size, FLAG_HOST_ONLY, 0).map_err(|e| EngineError::Load(format!("alloc ob: {e}")))?;
        let sb = dev.alloc_bo_raw(meta.scratch_bytes, FLAG_HOST_ONLY, 0).map_err(|e| EngineError::Load(format!("alloc weights ({} bytes): {e}", meta.scratch_bytes)))?;
        let kb = dev.alloc_bo_raw(meta.cache_bytes, FLAG_HOST_ONLY, 0).map_err(|e| EngineError::Load(format!("alloc caches ({} bytes): {e}", meta.cache_bytes)))?;

        // Every layer's weight stream, then the head's, go into %s ONCE here -- resident for the
        // model's whole lifetime; every cache starts zeroed in %k.
        for li in 0..meta.nlayer {
            let weight_path = meta.weight_dir.join(format!("w{li}.npy"));
            let wbytes: Array1<u8> = read_npy(&weight_path).map_err(|e| EngineError::Load(format!("read {}: {e}", weight_path.display())))?;
            let wbytes = wbytes.into_raw_vec_and_offset().0;
            let woff = *meta.layer_weight_off.get(&li).ok_or_else(|| EngineError::Load(format!("meta.json: no layer_weight_off for layer {li}")))?;
            sb.sub(woff, wbytes.len()).map_err(|e| EngineError::Load(format!("sb.sub weight[{li}]: {e}")))?
                .write_bytes(&wbytes).map_err(|e| EngineError::Load(format!("write weight[{li}]: {e}")))?;
        }
        kb.write_bytes(&vec![0u8; meta.cache_bytes]).map_err(|e| EngineError::Load(format!("zero caches: {e}")))?;
        kb.sync_to_device().map_err(|e| EngineError::Load(format!("sync caches: {e}")))?;

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
            kern.bind(&[&xb, &ob, &sb, &kb]).map_err(|e| EngineError::Load(format!("bind {}: {e}", r.name)))?;
            rungs.insert(r.name.clone(), kern);
        }

        let provenance = build_provenance(&meta.dir, &elf, meta.largest_keys(1));
        Ok(LadderResidentForward { meta, embed, xb, ob, sb, kb, boot, rungs, ring_valid: RingValid::default(), poisoned: false, provenance, media: MediaEmbeds::default() })
    }

    /// Build `%x` for a piece of `p_len` rows at position `s` and dispatch the rung named
    /// `rung_name` (already selected by the caller), returning `first` (the sliding window's
    /// first key, 64-aligned).
    fn dispatch(&mut self, rung_name: &str, split_nb: Option<usize>, x_bits: &[u16], s: usize, p_len: usize) -> Result<usize, EngineError> {
        let nt = piece_nt(p_len, self.meta.row_block);
        let first = ring_read_first(s, self.meta.sliding_window);

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
        let write_param = |name: &str, byte_value: RingParam| -> Result<(), EngineError> {
            let idx = *self.meta.scratchpad_params.get(name).ok_or_else(|| EngineError::Load(format!("meta.json: no scratchpad param `{name}`")))?;
            let value = match byte_value {
                RingParam::Bytes(b) if b % unit != 0 => {
                    return Err(EngineError::Device(format!("{name}: byte offset {b} not a multiple of param_unit_bytes {unit}")));
                }
                RingParam::Bytes(b) => b / unit,
                RingParam::Count(n) => n,
            };
            // Each slot is an i32 (params.txt); `unit` is only the scale of an offset it holds.
            let word = u32::try_from(value).map_err(|_| EngineError::Device(format!("{name}: {value} does not fit an i32 slot")))?;
            kern.write_scratchpad(idx * 4, &word.to_le_bytes()).map_err(|e| EngineError::Device(format!("write scratchpad {name}: {e}")))
        };
        let rung_nt = self.meta.rungs.iter().find(|r| r.name == rung_name).map(|r| r.nt)
            .ok_or_else(|| EngineError::Load(format!("meta.json: no rung `{rung_name}`")))?;
        let ring = ring_params(&self.meta.s_ring_layout, s, rung_nt * self.meta.row_block, first);
        ring_fits(&self.meta.s_ring_layout, &ring).map_err(|e| EngineError::Device(format!("{rung_name} at {s}: {e}")))?;
        write_param("kvw_g", RingParam::Bytes(s * self.meta.kvrow_g * 2))?;
        for (name, v) in ring {
            write_param(name, v)?;
        }

        kern.dispatch().map_err(EngineError::Device)?;
        self.ob.sync_from_device().map_err(|e| EngineError::Device(format!("sync ob: {e}")))?;

        let c = self.meta.s_ring_layout.blocks * self.meta.s_ring_layout.block_rows;
        self.ring_valid = self.ring_valid.write(s, p_len, rung_nt * self.meta.row_block, c);
        Ok(first)
    }

    /// Whether the last layer's output rows are finite. A dispatch that goes non-finite has
    /// written NaN only into its own new K/V rows, which re-running it at the same position
    /// overwrites (rf-forward-intermittent-nonfinite: ~5% of f1 dispatches at 6k, both pins).
    fn out_rows_finite(&self, p_len: usize) -> Result<bool, EngineError> {
        let mut bytes = vec![0u8; p_len * self.meta.d_model * 2];
        let slot = (self.meta.nlayer % 2) * self.meta.hidden_slot_bytes;
        self.ob.read_bytes_at(slot, &mut bytes).map_err(|e| EngineError::Device(format!("read x_out: {e}")))?;
        Ok(unpack_bf16_bytes(&bytes).iter().all(|v| v.is_finite()))
    }

    /// `dispatch`, repeated while its output comes back non-finite.
    fn dispatch_finite(&mut self, rung_name: &str, split_nb: Option<usize>, x_bits: &[u16], s: usize, p_len: usize) -> Result<(), EngineError> {
        self.poisoned = true;
        self.dispatch(rung_name, split_nb, x_bits, s, p_len)?;
        if self.out_rows_finite(p_len)? {
            self.poisoned = false;
            return Ok(());
        }
        let dump = std::env::var("NPU_RESIDENT_NF_DUMP").ok();
        for (retry, &pause_ms) in NONFINITE_BACKOFF_MS.iter().enumerate() {
            eprintln!("[resident] {rung_name} at {s}: non-finite output, retry {}/{} after {pause_ms} ms", retry + 1, NONFINITE_BACKOFF_MS.len());
            if retry == 2 {
                if let Some(path) = &dump {
                    self.dump_nonfinite(path, rung_name, s, p_len);
                }
            }
            std::thread::sleep(std::time::Duration::from_millis(pause_ms));
            self.dispatch(rung_name, split_nb, x_bits, s, p_len)?;
            if self.out_rows_finite(p_len)? {
                self.poisoned = false;
                return Ok(());
            }
        }
        Err(EngineError::Device(format!("{rung_name} at {s}: output non-finite after {} retries", NONFINITE_BACKOFF_MS.len())))
    }

    /// `NPU_RESIDENT_NF_DUMP`: every layer's non-finite cache byte ranges (bf16 exponent all ones),
    /// appended to `path` as one JSON line per burst (written at the third retry).
    fn dump_nonfinite(&self, path: &str, rung_name: &str, s: usize, p_len: usize) {
        let _ = self.kb.sync_from_device();
        let mut layers = serde_json::Map::new();
        let mut offs: Vec<(usize, usize)> = self.meta.layer_kv_off.iter().map(|(&l, &o)| (l, o)).collect();
        offs.sort_by_key(|&(_, o)| o);
        for (i, &(layer, off)) in offs.iter().enumerate() {
            let end = offs.get(i + 1).map_or(self.meta.cache_bytes, |&(_, o)| o);
            let mut ranges: Vec<(usize, usize)> = Vec::new();
            let mut buf = vec![0u8; 1 << 24];
            let mut at = off;
            while at < end {
                let n = buf.len().min(end - at);
                if self.kb.read_bytes_at(at, &mut buf[..n]).is_err() {
                    break;
                }
                for (j, c) in buf[..n].chunks_exact(2).enumerate() {
                    if u16::from_le_bytes([c[0], c[1]]) & 0x7f80 == 0x7f80 {
                        let b = at - off + 2 * j;
                        match ranges.last_mut() {
                            Some(r) if r.1 == b => r.1 = b + 2,
                            _ => ranges.push((b, b + 2)),
                        }
                    }
                }
                at += n;
            }
            if !ranges.is_empty() {
                let total: usize = ranges.iter().map(|r| r.1 - r.0).sum();
                ranges.truncate(256);
                layers.insert(layer.to_string(), serde_json::json!({"bytes": total, "ranges": ranges}));
            }
        }
        let line = serde_json::json!({"rung": rung_name, "s": s, "p_len": p_len, "ring_valid": [self.ring_valid.lo, self.ring_valid.hi], "layers": layers});
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
            use std::io::Write;
            let _ = writeln!(f, "{line}");
        }
    }

    fn zero_caches(&mut self) -> Result<(), EngineError> {
        self.kb.write_bytes(&vec![0u8; self.meta.cache_bytes]).map_err(|e| EngineError::Device(format!("zero caches: {e}")))?;
        self.kb.sync_to_device().map_err(|e| EngineError::Device(format!("sync caches: {e}")))?;
        self.ring_valid = RingValid::default();
        self.poisoned = false;
        Ok(())
    }

    /// The input row at `pos`: the tower's row where media was scattered, else the text embedding.
    fn embed_at(&mut self, token: u32, pos: usize) -> Result<Vec<u16>, EngineError> {
        match self.media.row(pos) {
            Some(b) => Ok(b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()),
            None => self.embed.embed_row_bf16(token),
        }
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
        let x_bits = self.embed_at(token, pos)?;
        self.dispatch_finite(&name, Some(blocks), &x_bits, pos, 1)?;
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
            // Past every prefill rung: decline here (return `at`, unchanged progress for this call)
            // and let the generator's per-token loop finish via `step()`'s f1 ladder, which reaches
            // 262144 -- see module doc.
            let Some((p_len, rung)) = prefill_piece(&self.meta.rungs, at, batchable - at, self.meta.row_block) else {
                break;
            };
            let end = at + p_len;
            let rung_name = rung.name.clone();
            let piece = &tokens[at..end];
            let mut x_bits = Vec::with_capacity(piece.len() * self.meta.d_model);
            for (i, &tok) in piece.iter().enumerate() {
                x_bits.extend_from_slice(&self.embed_at(tok, at + i)?);
            }
            self.dispatch_finite(&rung_name, None, &x_bits, at, p_len)?;
            at = end;
        }
        Ok(at)
    }

    fn prefill_batch(&self) -> Option<usize> {
        Some(self.meta.pmax)
    }

    /// Keeps the caches across requests (`NPU_RESIDENT_REUSE_KV=0` disables): rows past a new write
    /// position are masked, and a stale finite row there contributes exactly what a zero does. A
    /// possibly-NaN cache is zeroed either way.
    fn reset(&mut self) -> Result<CacheState, EngineError> {
        if !self.poisoned && reuse_kv() {
            return Ok(CacheState::Retained);
        }
        self.zero_caches()?;
        Ok(CacheState::Cleared)
    }

    fn resume_limit(&self, reused: usize) -> usize {
        match self.ring_valid.keeps_the_window(reused, self.meta.sliding_window) {
            true => reused,
            false => 0,
        }
    }

    fn batched_resume_gated(&self) -> bool {
        reuse_kv()
    }


    fn provenance(&self) -> ArmProvenance {
        self.provenance.clone()
    }

    fn set_media(&mut self, media: MediaEmbeds) {
        self.media = media;
    }

    fn max_context(&self) -> Option<usize> {
        Some(self.meta.largest_keys(1))
    }

    /// Real device BO bytes: `%x` + `%o` + `%s` + `%k`, so the generic `memory_ceiling_mb` accounting
    /// (`npu-runtime::loader::EngineLoader`/`registry::ensure_resident`) can evict THIS model or
    /// evict the shipped `gemma4-12b` to make room, instead of reporting the default 0 and letting
    /// both look free to load at once -- the residency conflict the coordinator flagged for
    /// `rf48L`'s ~11.8 GB scratch.
    fn bo_bytes(&self) -> u64 {
        (self.meta.xbuf + self.meta.obuf_f1.max(self.meta.obuf_f2) + self.meta.scratch_bytes + self.meta.cache_bytes) as u64
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
    fn provenance_reads_the_build_flags_and_the_toolchain_instance() {
        let dir = std::env::temp_dir().join(format!("rf-prov-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("gen_env.txt"), "RF_ATTN_H=1\nRF_INST=/x/instances/259cac5e9ca8\nRF_FAST=1\n").unwrap();
        fs::write(dir.join("gen_args.txt"), "rlayer_design m g h nbw=20\n").unwrap();
        let p = build_provenance(&dir, b"elf", 262144);
        assert_eq!(p.toolchain_pin_hash.as_deref(), Some("259cac5e9ca8"));
        assert_eq!(p.fusion_flags, ["RF_ATTN_H=1", "RF_FAST=1", "rlayer_design", "m", "g", "h", "nbw=20"]);
        assert_eq!(p.max_seq, Some(262144));
        assert_eq!(p.artifact_hash.as_deref().map(str::len), Some(12));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_resume_is_kept_only_while_the_ring_still_holds_its_window() {
        let (w, c) = (1024, 1280);
        // A fresh prime of 2240 rows then no wrap onto [960, 2000): kept; one more row lands on 960.
        let v = RingValid::default().write(0, 2240, 2240, c);
        assert!(v.keeps_the_window(2000, w));
        assert!(!RingValid::default().write(0, 2241, 2241, c).keeps_the_window(2000, w));
        // Padding rows destroy slots too, and are not positions themselves.
        let v = RingValid::default().write(0, 2224, 2241, c);
        assert!(!v.keeps_the_window(2000, w));
        assert!(!v.keeps_the_window(2230, w), "a padding row is not a written position");
        assert!(RingValid::default().keeps_the_window(0, w));
    }

    /// The 2026-09-30 service gate's sequence: a long request, then one re-primed from 0 that shares
    /// only a few tokens, then its follow-up. The re-prime rewrote every slot the follow-up needs.
    #[test]
    fn a_re_prime_from_zero_makes_its_own_follow_up_resumable() {
        let (w, c) = (1024, 1280);
        let steps = |mut v: RingValid, from: usize, to: usize| {
            for s in from..to { v = v.write(s, 1, 16, c); }
            v
        };
        let prime = |v: RingValid, n: usize| {
            let mut v = v;
            for a in (0..n).step_by(32) { let p = (n - a).min(32); v = v.write(a, p, 32, c); }
            v
        };
        let long = steps(prime(RingValid::default(), 2797), 2797, 2842);
        assert!(!long.keeps_the_window(2475, w), "the long reply overwrote the follow-up's window");
        let qa1 = steps(prime(long, 2484), 2484, 2508);
        assert!(qa1.keeps_the_window(2481, w), "qa1 re-primed [0, 2508): its follow-up must resume");
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

    fn f7_rungs() -> Vec<Rung> {
        let seg = |name: &str, nt: usize, blocks: usize| Rung { name: name.into(), nt, blocks, keys: blocks * 64, kind: "seg".into() };
        let mut r = rf48l_rungs();
        r.extend([seg("f7", 7, 64), seg("f7w256", 7, 256)]);
        r
    }

    #[test]
    fn prefill_piece_takes_the_largest_family_its_window_allows() {
        let r = f7_rungs();
        let pick = |at, left| prefill_piece(&r, at, left, 16).map(|(p, g)| (p, g.name.clone()));
        assert_eq!(pick(0, 5440), Some((112, "f7".into())));
        assert_eq!(pick(4032, 5440), Some((112, "f7w256".into())), "the piece's end picks the window");
        assert_eq!(pick(1232, 5440), Some((112, "f7".into())), "the ring's wrap does not cut a piece");
        assert_eq!(pick(0, 20), Some((20, "f2".into())), "a short tail runs on the smallest family");
        assert_eq!(pick(16320, 5440), Some((64, "f7w256".into())), "cut at f7's largest window");
        assert_eq!(pick(16384, 5440), Some((32, "f2w1024".into())), "past f7's windows only f2 holds the end");
        assert_eq!(pick(65536, 10), None, "past every prefill window");
    }

    #[test]
    fn prefill_piece_on_the_f2_only_ladder_is_32_rows() {
        let r = rf48l_rungs();
        for at in [0usize, 100, 1270, 4090, 60000] {
            let (p, g) = prefill_piece(&r, at, 5440, 16).unwrap();
            assert_eq!(p, 32, "at {at}");
            assert_eq!(g.name, rung_for(&r, 2, at + p).unwrap().name, "at {at}");
        }
    }

    /// rf48L's ring: `kvring.describe(1280, 20, 1280 * 4096)`.
    fn rf_ring() -> RingLayout {
        RingLayout {
            block_rows: 64,
            blocks: 20,
            window_blocks: 20,
            slab_bytes: 32768,
            block_bytes: 524288,
            row_bytes: 128,
            region_bytes: 10485760,
            write: RingBd { min_off: 0, max_off: 491520, extent: 24704, granule_stride: 128 },
            read: RingBd { min_off: 0, max_off: 491520, extent: 32768, granule_stride: 524288 },
        }
    }

    #[test]
    fn ring_spans_match_kvring() {
        // kvring.write_spans / read_spans on the same inputs (python, 2026-09-29).
        assert_eq!(write_spans(1279, 16, 1280, 64), [(1279, 1), (0, 15)]);
        assert_eq!(write_spans(0, 16, 1280, 64), [(0, 15), (15, 1)]);
        assert_eq!(write_spans(60, 32, 1280, 64), [(60, 4), (64, 28)]);
        assert_eq!(read_spans(0, 20, 1280, 64), [(0, 19), (19, 1)]);
        assert_eq!(read_spans(64 * 7, 20, 1280, 64), [(7, 13), (0, 7)]);
        let p = ring_params(&rf_ring(), 1279, 16, 0);
        assert_eq!(p[0], ("kvw_a", RingParam::Bytes(4984768 * 2)));
        assert_eq!(p[3], ("kvw_bn", RingParam::Count(14)));
        assert_eq!(p[6], ("kvr_b", RingParam::Bytes(4980736 * 2)));
    }

    #[test]
    fn ring_params_fit_at_every_slot_and_the_check_can_fail() {
        let l = rf_ring();
        for s in 0usize..3 * 1280 {
            for rows in [16, 32] {
                let first = s.saturating_sub(1023) / 64 * 64;
                ring_fits(&l, &ring_params(&l, s, rows, first)).unwrap_or_else(|e| panic!("s {s} rows {rows}: {e}"));
            }
        }
        // the unsplit 16-row write from slot 1279 (the mirrored ring's overrun) must not pass
        let mut p = ring_params(&l, 1279, 16, 0);
        p[1] = ("kvw_an", RingParam::Count(15));
        assert!(ring_fits(&l, &p).unwrap_err().contains("outside"));
        // nor may a span that stays in the region but runs into the next slab
        let mut p = ring_params(&l, 100, 16, 0);
        p[1] = ("kvw_an", RingParam::Count(40));
        assert!(ring_fits(&l, &p).unwrap_err().contains("cross a slab"));
        let mut p = ring_params(&l, 0, 16, 0);
        p[4] = ("kvr_a", RingParam::Bytes(2 * l.block_bytes));
        assert!(ring_fits(&l, &p).unwrap_err().contains("outside"));
    }

    #[test]
    fn largest_keys_reads_the_max_across_an_nt_family() {
        let meta_rungs = rf48l_rungs();
        let largest_f1 = meta_rungs.iter().filter(|r| r.nt == 1).map(|r| r.keys).max().unwrap();
        let largest_f2 = meta_rungs.iter().filter(|r| r.nt == 2).map(|r| r.keys).max().unwrap();
        assert_eq!(largest_f1, 262144);
        assert_eq!(largest_f2, 65536);
    }
    /// Device gate (`NPU_LLM_DEVICE_GATE=1`, under `npu_lock.sh queue --`; `NPU_RESIDENT_DIR` overrides
    /// the build): a prompt resumed on the retained cache after another that shares its first 150
    /// tokens gives the same logits, bit for bit, as the same prompt primed on a zeroed cache.
    #[test]
    fn a_resumed_prefix_matches_a_fresh_prime_on_device() {
        if std::env::var("NPU_LLM_DEVICE_GATE").is_err() {
            eprintln!("SKIP: set NPU_LLM_DEVICE_GATE=1 to run (opens the NPU device -- wrap with npu_lock.sh queue --)");
            return;
        }
        let dir = std::env::var("NPU_RESIDENT_DIR")
            .unwrap_or_else(|_| "/mnt/data/xdna/artifacts/gemma4-12b/resident_rf48C_p7148a7".to_string());
        let dev = Rc::new(Device::open(0).expect("open the NPU"));
        let mut f = LadderResidentForward::open(&dev, Path::new(&dir)).expect("open the resident forward");
        let ids = |seed: u32, n: usize| (0..n as u32).map(|i| 1000 + (i * 7919 + seed * 104729) % 200_000).collect::<Vec<u32>>();
        let p = ids(1, 200);
        let mut q = p[..150].to_vec();
        q.extend(ids(2, 61));
        let prime = |f: &mut LadderResidentForward, t: &[u32], from: usize| -> Vec<f32> {
            let at = f.prefill(t, from).expect("prefill").max(from);
            let mut logits = Vec::new();
            for (i, &tok) in t.iter().enumerate().skip(at) {
                logits = f.step(tok, i).expect("step");
            }
            logits
        };
        prime(&mut f, &p, 0);
        assert_eq!(f.resume_limit(150), 150, "no wrap at these lengths, the resume must be kept");
        let resumed = prime(&mut f, &q, 150);
        f.zero_caches().unwrap();
        // The same piece boundary at 150 as the resumed run, so only the cache's history differs.
        f.prefill(&q[..151], 0).expect("prefill to 150");
        let fresh = prime(&mut f, &q, 150);
        let differ = |a: &[f32], b: &[f32]| a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
        assert_eq!(differ(&resumed, &fresh), 0, "resumed vs fresh logits differ");
        // Negative control: the same resume over a cache holding a different prefix must differ.
        f.zero_caches().unwrap();
        prime(&mut f, &ids(3, 200), 0);
        let wrong = prime(&mut f, &q, 150);
        assert!(differ(&wrong, &fresh) > 0, "a resume over the wrong prefix matched: the check cannot fail");
    }

    /// Bench (`NPU_LLM_DEVICE_GATE=1 NPU_RESIDENT_BENCH=1`): the boot dispatch alone, a decode step
    /// (boot + f1), and batched prefill per token, at each `NPU_RESIDENT_BENCH_P` prompt length.
    #[test]
    fn bench_boot_decode_and_prefill_on_device() {
        if std::env::var("NPU_LLM_DEVICE_GATE").is_err() || std::env::var("NPU_RESIDENT_BENCH").is_err() {
            eprintln!("SKIP: set NPU_LLM_DEVICE_GATE=1 NPU_RESIDENT_BENCH=1 (opens the NPU device)");
            return;
        }
        let dir = std::env::var("NPU_RESIDENT_DIR")
            .unwrap_or_else(|_| "/mnt/data/xdna/artifacts/gemma4-12b/resident_rf48C_p7148a7".to_string());
        let lens: Vec<usize> = std::env::var("NPU_RESIDENT_BENCH_P").unwrap_or_else(|_| "4096".into())
            .split(',').map(|v| v.parse().unwrap()).collect();
        let dev = Rc::new(Device::open(0).expect("open the NPU"));
        let mut f = LadderResidentForward::open(&dev, Path::new(&dir)).expect("open the resident forward");
        let median = |mut v: Vec<f64>| { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] };
        let boots: Vec<f64> = (0..20).map(|_| {
            let t = std::time::Instant::now();
            f.boot.dispatch().unwrap();
            t.elapsed().as_secs_f64() * 1e3
        }).collect();
        println!("BENCH boot median {:.2} ms (min {:.2})", median(boots.clone()), boots.iter().cloned().fold(f64::MAX, f64::min));
        for p in lens {
            f.zero_caches().unwrap();
            let ids: Vec<u32> = (0..p as u32).map(|i| 1000 + (i * 7919) % 200_000).collect();
            let t = std::time::Instant::now();
            let at = f.prefill(&ids, 0).expect("prefill");
            let pre = t.elapsed().as_secs_f64();
            let mut steps = Vec::new();
            for i in at..at + 24 {
                let t = std::time::Instant::now();
                f.step(ids[i % p], i).expect("step");
                steps.push(t.elapsed().as_secs_f64() * 1e3);
            }
            println!("BENCH P {p}: prefill {at} tok in {pre:.1} s = {:.2} ms/tok; decode step median {:.1} ms (min {:.1})",
                     pre * 1e3 / at as f64, median(steps.clone()), steps.iter().cloned().fold(f64::MAX, f64::min));
        }
    }
}
