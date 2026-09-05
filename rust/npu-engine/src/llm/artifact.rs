//! A validated reader for a fused-decode-ELF's `meta.json`, parse-don't-validate: every name the
//! artifact itself declares live (`inputs`/`weights`/`output`) must resolve to a `layout` entry, every
//! buffer must fit inside the arena size its own type declares, and no two buffers may overlap.
//! Construction FAILS LOUD naming the gap rather than silently inferring it.
//!
//! `gen_llm_decode.py:306` hand-writes `["x", "logits"] + wnames` instead of asking IRON for every
//! declared input, so `rope_global` -- a real `inputs` entry -- has no `layout` row (the generator's
//! own bug, not a device fact; see
//! `docs/superpowers/specs/2026-09-05-llm-serving-and-residency-design.md` §4 and its placement
//! ledger). [`LlmArtifact::load`] refuses to guess that gap open-endedly: it fails loud on ANY missing
//! name, and separately -- ONLY for the exact `rope_global`-after-`x` shape this generator produces --
//! applies a narrow, logged compatibility placement. Rebuilding the artifact with a fixed generator
//! removes the shim's reason to exist, not its correctness.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use npu_xrt::Arena;

use crate::api::EngineError;

/// (arena, byte-offset, byte-len) of a named buffer, as declared by `meta.json`'s `layout` map.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BufLoc {
    pub arena: Arena,
    pub off: usize,
    pub len: usize,
}

/// One ctrl-scratchpad parameter (`aiex.scratchpad_parameter`, Option C). `core: true` means the
/// firmware's UPDATE_REG convention applies -- the host must shift the value left by 2 bits before
/// writing it (see `asr::whisper_decoder`'s `dispatch_resident`, which this mirrors); `core: false`
/// ("addr" kind) is written raw.
#[derive(Clone, Copy, Debug)]
pub struct ScratchpadParam {
    pub byte_offset: usize,
    pub core: bool,
}

/// How the host scales `embed[token]` before writing it to the device -- `host_protocol.embed_scale`
/// in `meta.json`, a real per-model choice (Whisper embeds unscaled; some LLM families multiply by
/// `sqrt(d_model)`) and therefore a branch on *what*, not *how*: an unrecognised value fails loud
/// rather than silently defaulting to one arm (design spec §6).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmbedScale {
    None,
    SqrtDModel,
}

/// A validated, ready-to-drive fused decode ELF: every buffer the artifact declares has a checked
/// location, and the scratchpad protocol (`kv_off`/`sm_mask`) is resolved to concrete offsets.
#[derive(Debug)]
pub struct LlmArtifact {
    pub decode_dir: PathBuf,
    pub elf_name: String,
    pub kernel_name: String,
    pub input_size: usize,
    pub output_size: usize,
    pub scratch_size: usize,
    pub layout: HashMap<String, BufLoc>,
    pub weights: Vec<String>,
    pub output: String,
    /// `meta.json`'s `cache_buffers` -- the KV-cache scratch buffers a fresh generation must
    /// re-zero (`NpuDecodeStep::reset`). Absent (empty) is legal: a model with no on-device cache
    /// buffer still validates.
    pub cache_buffers: Vec<String>,
    pub kv_off: ScratchpadParam,
    pub sm_mask: ScratchpadParam,
    pub head_dim: usize,
    pub d_model: usize,
    pub vocab: usize,
    pub n_layers: usize,
    pub embed_scale: EmbedScale,
    pub rope_theta_global: f64,
}

impl LlmArtifact {
    /// Load and validate `decode_dir/meta.json`. `decode_dir` also holds `decode.elf` and
    /// `buffers/<name>.bin` for every name in `weights`.
    pub fn load(decode_dir: &Path) -> Result<LlmArtifact, EngineError> {
        let meta_path = decode_dir.join("meta.json");
        let bytes = fs::read(&meta_path)
            .map_err(|e| EngineError::Load(format!("read {}: {e}", meta_path.display())))?;
        let meta: serde_json::Value = serde_json::from_slice(&bytes)
            .map_err(|e| EngineError::Load(format!("parse {}: {e}", meta_path.display())))?;
        let ctx = |msg: String| EngineError::Load(format!("{}: {msg}", meta_path.display()));

        let usz = |k: &str| -> Result<usize, EngineError> {
            meta.get(k)
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .ok_or_else(|| ctx(format!("missing/non-numeric top-level `{k}`")))
        };
        let str_field = |k: &str| -> Result<String, EngineError> {
            meta.get(k)
                .and_then(|v| v.as_str())
                .map(str::to_string)
                .ok_or_else(|| ctx(format!("missing/non-string top-level `{k}`")))
        };
        let str_list = |k: &str| -> Result<Vec<String>, EngineError> {
            meta.get(k)
                .and_then(|v| v.as_array())
                .ok_or_else(|| ctx(format!("missing/non-array top-level `{k}`")))?
                .iter()
                .map(|v| v.as_str().map(str::to_string).ok_or_else(|| ctx(format!("`{k}` has a non-string entry"))))
                .collect()
        };

        let input_size = usz("input_size")?;
        let output_size = usz("output_size")?;
        let scratch_size = usz("scratch_size")?;
        let elf_name = str_field("elf")?;
        let kernel_name = str_field("kernel_name")?;
        let output = str_field("output")?;
        let inputs = str_list("inputs")?;
        let weights = str_list("weights")?;
        let cache_buffers = meta.get("cache_buffers").map(|_| str_list("cache_buffers")).transpose()?.unwrap_or_default();

        let mut layout = HashMap::new();
        let layout_obj = meta
            .get("layout")
            .and_then(|v| v.as_object())
            .ok_or_else(|| ctx("missing/non-object top-level `layout`".to_string()))?;
        for (name, e) in layout_obj {
            let arena = match e.get("type").and_then(|v| v.as_str()) {
                Some("input") => Arena::Input,
                Some("output") => Arena::Output,
                Some("scratch") => Arena::Scratch,
                other => return Err(ctx(format!("layout[{name}].type = {other:?}, want input/output/scratch"))),
            };
            let off = e
                .get("offset")
                .and_then(|v| v.as_u64())
                .ok_or_else(|| ctx(format!("layout[{name}] has no numeric `offset`")))? as usize;
            let len = e
                .get("len")
                .and_then(|v| v.as_u64())
                .ok_or_else(|| ctx(format!("layout[{name}] has no numeric `len`")))? as usize;
            layout.insert(name.clone(), BufLoc { arena, off, len });
        }

        let dims = meta.get("dims").ok_or_else(|| ctx("missing top-level `dims`".to_string()))?;
        let dim = |k: &str| -> Result<usize, EngineError> {
            dims.get(k).and_then(|v| v.as_u64()).map(|v| v as usize).ok_or_else(|| ctx(format!("dims.{k} missing/non-numeric")))
        };
        let head_dim = dim("head_dim")?;
        let d_model = dim("d_model")?;
        let vocab = dim("vocab")?;
        let n_layers = dim("layers")?;

        let hp = meta.get("host_protocol").ok_or_else(|| ctx("missing top-level `host_protocol`".to_string()))?;
        let embed_scale = match hp.get("embed_scale").and_then(|v| v.as_str()) {
            Some("none") => EmbedScale::None,
            Some("sqrt_d_model") => EmbedScale::SqrtDModel,
            other => return Err(ctx(format!("host_protocol.embed_scale = {other:?}, want \"none\" or \"sqrt_d_model\""))),
        };
        let rope_theta_global = hp
            .get("rope_theta_global")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| ctx("host_protocol.rope_theta_global missing/non-numeric".to_string()))?;

        let sp = meta.get("scratchpad").ok_or_else(|| ctx("missing top-level `scratchpad`".to_string()))?;
        let sp_params = sp.get("params").ok_or_else(|| ctx("scratchpad.params missing".to_string()))?;
        let kv_param_name = sp
            .get("kv_param")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ctx("scratchpad.kv_param missing".to_string()))?;
        let mask_param_name = sp
            .get("mask_param")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ctx("scratchpad.mask_param missing".to_string()))?;
        let read_param = |name: &str| -> Result<ScratchpadParam, EngineError> {
            let p = sp_params.get(name).ok_or_else(|| ctx(format!("scratchpad.params has no entry `{name}`")))?;
            let byte_offset = p
                .get("byte_offset")
                .and_then(|v| v.as_u64())
                .ok_or_else(|| ctx(format!("scratchpad.params.{name}.byte_offset missing/non-numeric")))? as usize;
            let core = match p.get("kind").and_then(|v| v.as_str()) {
                Some("core") => true,
                Some("addr") => false,
                other => return Err(ctx(format!("scratchpad.params.{name}.kind = {other:?}, want \"core\" or \"addr\""))),
            };
            Ok(ScratchpadParam { byte_offset, core })
        };
        let kv_off = read_param(kv_param_name)?;
        let sm_mask = read_param(mask_param_name)?;

        // Compat shim, narrow and logged: ONLY for `rope_global` immediately following `x`, the exact
        // shape `gen_llm_decode.py` currently emits. Any other gap still fails loud below.
        if inputs.iter().any(|n| n == "rope_global") && !layout.contains_key("rope_global") {
            let x = *layout
                .get("x")
                .ok_or_else(|| ctx("rope_global compat shim needs `x`'s layout entry, and it is ALSO missing -- refusing to guess".to_string()))?;
            let len = head_dim * 2; // bf16 bytes
            let off = x.off + x.len;
            if off + len > input_size {
                return Err(ctx(format!(
                    "rope_global compat shim would place it at [{off}, {}), past input_size {input_size} -- \
                     artifact shape does not match the assumed contiguous [x, rope_global] packing, refusing to guess",
                    off + len
                )));
            }
            eprintln!(
                "[llm artifact] {}: `rope_global` is a declared input with no `layout` entry \
                 (gen_llm_decode.py:306 hand-writes [\"x\",\"logits\"]+weights instead of asking IRON \
                 for every declared input -- see the 2026-09-05 serving spec §4/placement ledger). \
                 Compat shim: placing it at input offset {off}, len {len}. Rebuild the artifact with a \
                 fixed generator to remove this shim.",
                meta_path.display()
            );
            layout.insert("rope_global".to_string(), BufLoc { arena: Arena::Input, off, len });
        }

        // Every declared name must now resolve.
        let mut missing: Vec<&str> = inputs
            .iter()
            .chain(weights.iter())
            .chain(std::iter::once(&output))
            .map(String::as_str)
            .filter(|n| !layout.contains_key(*n))
            .collect();
        if !missing.is_empty() {
            missing.sort_unstable();
            missing.dedup();
            return Err(ctx(format!("declared buffer(s) with no `layout` entry: {}", missing.join(", "))));
        }

        // Per-arena extents.
        let arena_size = |a: Arena| match a {
            Arena::Input => input_size,
            Arena::Output => output_size,
            Arena::Scratch => scratch_size,
        };
        for (name, loc) in &layout {
            let bound = arena_size(loc.arena);
            let end = loc.off.checked_add(loc.len).ok_or_else(|| ctx(format!("buffer `{name}` offset+len overflows")))?;
            if end > bound {
                return Err(ctx(format!(
                    "buffer `{name}` at {:?}[{}, {end}) exceeds its arena size {bound}",
                    loc.arena, loc.off
                )));
            }
        }

        // No overlaps within an arena (gaps -- e.g. alignment padding -- are fine).
        for arena in [Arena::Input, Arena::Output, Arena::Scratch] {
            let mut spans: Vec<(usize, usize, &str)> =
                layout.iter().filter(|(_, l)| l.arena == arena).map(|(n, l)| (l.off, l.off + l.len, n.as_str())).collect();
            spans.sort_unstable_by_key(|&(off, _, _)| off);
            for w in spans.windows(2) {
                let (_, end0, n0) = w[0];
                let (off1, end1, n1) = w[1];
                if off1 < end0 {
                    return Err(ctx(format!("buffers `{n0}` and `{n1}` overlap in {arena:?}: [.., {end0}) vs [{off1}, {end1})")));
                }
            }
        }

        Ok(LlmArtifact {
            decode_dir: decode_dir.to_path_buf(),
            elf_name,
            kernel_name,
            input_size,
            output_size,
            scratch_size,
            layout,
            weights,
            output,
            cache_buffers,
            kv_off,
            sm_mask,
            head_dim,
            d_model,
            vocab,
            n_layers,
            embed_scale,
            rope_theta_global,
        })
    }

    pub fn elf_path(&self) -> PathBuf {
        self.decode_dir.join(&self.elf_name)
    }

    pub fn weight_blob_path(&self, name: &str) -> PathBuf {
        self.decode_dir.join("buffers").join(format!("{name}.bin"))
    }

    pub fn loc(&self, name: &str) -> &BufLoc {
        &self.layout[name]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// A minimal well-formed `meta.json` matching the real Qwen3-0.6B artifact's SHAPE (one input
    /// `x` covering [0, x_len), the declared `rope_global` input deliberately OMITTED from `layout`
    /// to exercise the compat shim), so every test below is one deliberate mutation away from it.
    fn base_meta(x_len: usize, head_dim: usize, extra_layout: serde_json::Value) -> serde_json::Value {
        let input_size = x_len + head_dim * 2;
        let mut layout = serde_json::json!({
            "x": {"type": "input", "offset": 0, "len": x_len},
            "logits": {"type": "output", "offset": 0, "len": 8},
            "W": {"type": "scratch", "offset": 0, "len": 16},
        });
        for (k, v) in extra_layout.as_object().unwrap() {
            layout.as_object_mut().unwrap().insert(k.clone(), v.clone());
        }
        serde_json::json!({
            "elf": "decode.elf", "kernel_name": "main:sequence",
            "input_size": input_size, "output_size": 8, "scratch_size": 16,
            "layout": layout,
            "inputs": ["x", "rope_global"], "weights": ["W"], "output": "logits",
            "cache_buffers": [],
            "scratchpad": {
                "params": {"kv_off": {"byte_offset": 0, "kind": "addr"}, "sm_mask": {"byte_offset": 4, "kind": "core"}},
                "kv_param": "kv_off", "mask_param": "sm_mask",
            },
            "dims": {"layers": 1, "d_model": x_len / 2, "vocab": 4, "head_dim": head_dim},
            "host_protocol": {"embed_scale": "none", "rope_theta_global": 1_000_000.0},
        })
    }

    fn write_meta(dir: &Path, meta: &serde_json::Value) {
        let mut f = fs::File::create(dir.join("meta.json")).unwrap();
        f.write_all(serde_json::to_string(meta).unwrap().as_bytes()).unwrap();
    }

    #[test]
    fn compat_shim_places_rope_global_right_after_x_and_logs() {
        let dir = tempfile::tempdir().unwrap();
        write_meta(dir.path(), &base_meta(8, 4, serde_json::json!({})));
        let art = LlmArtifact::load(dir.path()).expect("shim should place rope_global");
        let rope = art.layout["rope_global"];
        assert_eq!(rope.arena, Arena::Input);
        assert_eq!(rope.off, 8, "must sit immediately after x's [0,8)");
        assert_eq!(rope.len, 8, "head_dim(4)*2 bf16 bytes");
    }

    #[test]
    fn a_gap_other_than_rope_global_fails_loud_naming_it() {
        let dir = tempfile::tempdir().unwrap();
        let mut meta = base_meta(8, 4, serde_json::json!({"rope_global": {"type": "input", "offset": 8, "len": 8}}));
        // Declare a weight the layout never describes.
        meta["weights"] = serde_json::json!(["W", "GHOST"]);
        write_meta(dir.path(), &meta);
        let err = LlmArtifact::load(dir.path()).unwrap_err().to_string();
        assert!(err.contains("GHOST"), "must name the missing buffer: {err}");
    }

    #[test]
    fn missing_x_makes_the_shim_refuse_rather_than_guess() {
        let dir = tempfile::tempdir().unwrap();
        let mut meta = base_meta(8, 4, serde_json::json!({}));
        meta["layout"].as_object_mut().unwrap().remove("x");
        write_meta(dir.path(), &meta);
        let err = LlmArtifact::load(dir.path()).unwrap_err().to_string();
        assert!(err.contains("rope_global"), "{err}");
        assert!(err.contains("also missing") || err.contains("ALSO missing"), "{err}");
    }

    #[test]
    fn overlapping_buffers_fail_loud_naming_both() {
        let dir = tempfile::tempdir().unwrap();
        let meta = base_meta(
            8,
            4,
            serde_json::json!({
                "rope_global": {"type": "input", "offset": 8, "len": 8},
                "OVERLAP": {"type": "scratch", "offset": 4, "len": 4},
            }),
        );
        write_meta(dir.path(), &meta);
        let err = LlmArtifact::load(dir.path()).unwrap_err().to_string();
        assert!(err.contains("overlap"), "{err}");
        assert!(err.contains("W") && err.contains("OVERLAP"), "{err}");
    }

    #[test]
    fn extent_past_arena_size_fails_loud() {
        let dir = tempfile::tempdir().unwrap();
        let mut meta = base_meta(8, 4, serde_json::json!({"rope_global": {"type": "input", "offset": 8, "len": 8}}));
        meta["layout"]["W"] = serde_json::json!({"type": "scratch", "offset": 0, "len": 999});
        write_meta(dir.path(), &meta);
        let err = LlmArtifact::load(dir.path()).unwrap_err().to_string();
        assert!(err.contains("exceeds its arena size"), "{err}");
        assert!(err.contains('W'), "{err}");
    }

    #[test]
    fn well_formed_artifact_with_explicit_rope_global_needs_no_shim() {
        let dir = tempfile::tempdir().unwrap();
        let meta = base_meta(8, 4, serde_json::json!({"rope_global": {"type": "input", "offset": 8, "len": 8}}));
        write_meta(dir.path(), &meta);
        let art = LlmArtifact::load(dir.path()).unwrap();
        assert_eq!(art.head_dim, 4);
        assert_eq!(art.embed_scale, EmbedScale::None);
        assert!(!art.kv_off.core);
        assert!(art.sm_mask.core);
    }

    #[test]
    fn unknown_embed_scale_fails_loud_rather_than_defaulting() {
        let dir = tempfile::tempdir().unwrap();
        let mut meta = base_meta(8, 4, serde_json::json!({"rope_global": {"type": "input", "offset": 8, "len": 8}}));
        meta["host_protocol"]["embed_scale"] = serde_json::json!("mystery");
        write_meta(dir.path(), &meta);
        let err = LlmArtifact::load(dir.path()).unwrap_err().to_string();
        assert!(err.contains("embed_scale"), "{err}");
    }
}
