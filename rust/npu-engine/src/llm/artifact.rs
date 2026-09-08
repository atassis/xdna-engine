//! A validated reader for a fused-decode-ELF's `meta.json`, parse-don't-validate: every name the
//! artifact itself declares live (`inputs`/`weights`/`output`) must resolve to a `layout` entry, every
//! buffer must fit inside the arena size its own type declares, and no two buffers may overlap.
//! Construction FAILS LOUD naming the gap rather than silently inferring it.
//!
//! `gen_llm_decode.py:306` hand-writes `["x", "logits"] + wnames` instead of asking IRON for every
//! declared input, so `rope_global` -- a real `inputs` entry -- has no `layout` row (the generator's
//! own bug, not a device fact). [`LlmArtifact::load`] refuses to guess that gap open-endedly: it
//! fails loud on ANY missing
//! name, and separately -- ONLY for the exact `rope_global`-after-`x` shape this generator produces --
//! applies a narrow, logged compatibility placement. Rebuilding the artifact with a fixed generator
//! removes the shim's reason to exist, not its correctness.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use npu_asr::kernel_registry;
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
    /// `meta.json`'s `embed_blob`; see [`Self::embed_blob`]. `None` in pre-2026-09-08 artifacts.
    pub embed_blob: Option<String>,
    pub kv_off: ScratchpadParam,
    pub sm_mask: ScratchpadParam,
    pub head_dim: usize,
    pub d_model: usize,
    pub vocab: usize,
    pub n_layers: usize,
    pub embed_scale: EmbedScale,
    pub rope_theta_global: f64,
    /// The LOCAL RoPE base, for models with interleaved local/global attention (Gemma-3). `None` on
    /// a global-only model (Qwen3), which is why it is optional rather than defaulted: the presence
    /// of this field is exactly what decides whether the artifact declares a `rope_local` input, so
    /// a default would make a two-input and a three-input artifact indistinguishable here.
    pub rope_theta_local: Option<f64>,
    /// `meta.json`'s `toolchain.hash` -- the toolchain.lock semantic hash this ELF was compiled
    /// against (`gen_llm_decode.py`, added 2026-09-05). `None` on any artifact built before this
    /// field existed. See [`LlmArtifact::load`]'s freshness check below.
    pub toolchain_hash: Option<String>,
}

/// Verdict from comparing an artifact's [`LlmArtifact::toolchain_hash`] against the currently
/// pinned toolchain. Three ways to NOT be a hash match, because they mean different things: a
/// pre-this-change artifact was never stamped, a stamped one couldn't be checked (no
/// `toolchain.lock` above the artifact dir -- expected in a production install, which must not
/// depend on that dev-infra file), and a stamped one that actively disagrees with the current pin.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolchainFreshness {
    Fresh { hash: String },
    Stale { built_hash: String, current_hash: String },
    Unstamped,
    Unverifiable { built_hash: String, reason: String },
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
        let embed_blob = meta.get("embed_blob").and_then(|v| v.as_str()).map(str::to_owned);
        let toolchain_hash = meta
            .get("toolchain")
            .and_then(|t| t.get("hash"))
            .and_then(|v| v.as_str())
            .map(str::to_string);

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
        // Absent on a global-only model; present and numeric, or the artifact is malformed. A
        // non-numeric value must not read as "global-only" -- that would silently drop the local
        // RoPE write and leave the local layers rotating at the wrong base.
        // ABSENT and PRESENT-AS-NULL are the same thing -- single-theta RoPE. The generator emits
        // the key for every spec and writes `null` where there is no local base, so a global-only
        // artifact carries an explicit null rather than omitting the field; matching only on
        // absence sent Qwen3-0.6B, the pinned default, down the malformed branch. A non-numeric
        // value that is not null is still an error, which is the case the check is for.
        let rope_theta_local = match hp.get("rope_theta_local") {
            None | Some(serde_json::Value::Null) => None,
            Some(v) => Some(v.as_f64().ok_or_else(||
                ctx("host_protocol.rope_theta_local present but non-numeric".to_string()))?),
        };

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

        // Toolchain freshness: fail loud on an ACTIVE mismatch (the pin moved, nobody rebuilt this
        // artifact -- a stale ELF answers with a plausible WRONG token, silently). Anything short of a
        // confirmed mismatch is reported, never fatal -- a shipped consumer must not require
        // toolchain.lock to exist (that is dev-infra; see kernel_registry::check_toolchain_freshness's
        // doc comment for the encoder-side sibling of this same rule).
        match Self::check_toolchain_freshness(&toolchain_hash, decode_dir) {
            ToolchainFreshness::Stale { built_hash, current_hash } => {
                return Err(ctx(format!(
                    "toolchain-stale: built against {built_hash}, current toolchain.lock is {current_hash} \
                     -- rebuild with scripts/build_llm_decode.sh"
                )));
            }
            ToolchainFreshness::Unstamped => eprintln!(
                "[llm artifact] {}: no toolchain provenance recorded (built before this check existed) \
                 -- freshness UNVERIFIED",
                meta_path.display()
            ),
            ToolchainFreshness::Unverifiable { built_hash, reason } => eprintln!(
                "[llm artifact] {}: built against {built_hash}, but {reason} -- freshness UNVERIFIED",
                meta_path.display()
            ),
            ToolchainFreshness::Fresh { .. } => {}
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
            embed_blob,
            kv_off,
            sm_mask,
            head_dim,
            d_model,
            vocab,
            n_layers,
            embed_scale,
            rope_theta_global,
            rope_theta_local,
            toolchain_hash,
        })
    }

    /// Compare `toolchain_hash` (from `meta.json`) against the toolchain.lock found by walking up
    /// from `decode_dir`. Never touches an env var or a fixed repo path -- a production install has
    /// no toolchain.lock at all, and that is a valid, expected state, not an error.
    fn check_toolchain_freshness(toolchain_hash: &Option<String>, decode_dir: &Path) -> ToolchainFreshness {
        let Some(built) = toolchain_hash else {
            return ToolchainFreshness::Unstamped;
        };
        match Self::resolve_current_pin_hash(decode_dir) {
            Ok(Some(current)) if &current == built => ToolchainFreshness::Fresh { hash: current },
            Ok(Some(current)) => ToolchainFreshness::Stale { built_hash: built.clone(), current_hash: current },
            Ok(None) => ToolchainFreshness::Unverifiable {
                built_hash: built.clone(),
                reason: "no toolchain.lock found above the artifact dir (not a dev checkout)".to_string(),
            },
            Err(e) => ToolchainFreshness::Unverifiable { built_hash: built.clone(), reason: format!("resolving toolchain.lock failed: {e}") },
        }
    }

    /// Walk up from `start` for a `toolchain.lock` and hash it with
    /// `kernel_registry::current_toolchain_hash` -- the ONE derivation
    /// (`scripts/kernel_sandbox.sh::current_toolchain_hash`, `toolchain_up.sh`'s LOCKHASH, and this
    /// function all agree byte-for-byte). `Ok(None)` means no lock was found, which is the normal
    /// shape for a production install and must not be treated as an error.
    fn resolve_current_pin_hash(start: &Path) -> std::io::Result<Option<String>> {
        let mut dir = start.canonicalize()?;
        loop {
            if dir.join("toolchain.lock").is_file() {
                return kernel_registry::current_toolchain_hash(&dir).map(Some);
            }
            if !dir.pop() {
                return Ok(None);
            }
        }
    }

    pub fn elf_path(&self) -> PathBuf {
        self.decode_dir.join(&self.elf_name)
    }

    /// The bf16 `[vocab, d_model]` blob the host embedding gather reads. `W_head` itself unless
    /// the lm-head was quantised; then the generator emits a bf16 sidecar and names it here.
    /// Older artifacts have no such field, so they resolve to `W_head` exactly as before.
    pub fn embed_blob(&self) -> &str {
        self.embed_blob.as_deref().unwrap_or("W_head")
    }

    pub fn weight_blob_path(&self, name: &str) -> PathBuf {
        self.decode_dir.join("buffers").join(format!("{name}.bin"))
    }

    pub fn loc(&self, name: &str) -> &BufLoc {
        &self.layout[name]
    }

    /// Bidirectional companion to the missing-`layout`-entry check above (ported from
    /// `asr::whisper_decoder`'s `parse_layout`, `xdna-engine f446a50`): an `Arena::Input` buffer that
    /// `per_token_writes` does not name is what a missing host write looks like from the layout side.
    /// The caller passes the literal list of buffer names its own per-token step function writes
    /// (`NpuDecodeStep::step` -> `["x", "rope_global"]` today), so this catches an artifact whose
    /// declared inputs the decoder does not handle, not a generic property of `meta.json` alone.
    ///
    /// Scoped deliberately narrow: this guards the MISSING-WRITE class only. It does NOT guard
    /// artifact staleness -- a buffer can be written every token and still hold stale-toolchain
    /// bytes, which is a separate cause with the same symptom (one wrong token at a small-margin
    /// step) and is `check_toolchain_freshness`'s job, not this one's.
    pub fn check_per_token_writes(&self, per_token_writes: &[&str]) -> Result<(), EngineError> {
        let mut missing: Vec<&str> = self
            .layout
            .iter()
            .filter(|(_, loc)| loc.arena == Arena::Input)
            .map(|(name, _)| name.as_str())
            .filter(|name| !per_token_writes.contains(name))
            .collect();
        if missing.is_empty() {
            return Ok(());
        }
        missing.sort_unstable();
        Err(EngineError::Load(format!(
            "input-arena buffer(s) with no per-token host write: {} -- either the host write is \
             missing or the buffer does not belong in the input arena",
            missing.join(", ")
        )))
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

    // ------------------------------------------------------------------------------------
    // Toolchain freshness (2026-09-05): closes the hole a stale fused decode ELF exploited
    // silently -- it read 7/8 teacher-forced and looked like a precision tie. The
    // ELF that shipped 2026-09-03 was built against toolchain 9da6356ac521 (the pin one commit
    // before the 2026-09-04 re-pin, per `git log --follow -- toolchain.lock`); the values below
    // are that real pair, not invented ones.
    // ------------------------------------------------------------------------------------

    #[test]
    fn toolchain_fresh_when_stamp_matches_current_pin() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("toolchain.lock"), b"PIN=abc\n").unwrap();
        let current = kernel_registry::current_toolchain_hash(dir.path()).unwrap();
        let mut meta = base_meta(8, 4, serde_json::json!({"rope_global": {"type": "input", "offset": 8, "len": 8}}));
        meta["toolchain"] = serde_json::json!({"hash": current});
        write_meta(dir.path(), &meta);
        let art = LlmArtifact::load(dir.path()).expect("a matching stamp must load");
        assert_eq!(art.toolchain_hash.as_deref(), Some(current.as_str()));
    }

    /// THE regression this task closes: a decode.elf built against the pin active before the
    /// 2026-09-04 re-pin (9da6356ac521), loaded against the CURRENT pin (a6c6331c41b6, this repo's
    /// real toolchain.lock as of 2026-09-05) -- must fail loud, naming both hashes, exactly the
    /// shape the real `artifacts-qwen3-0.6b/decode/` artifact is in right now (that one predates
    /// the stamp field entirely and hits `toolchain_unstamped_artifact_loads_with_a_warning_not_an_error`
    /// below instead; this test is what re-running its build under this change would produce).
    #[test]
    fn toolchain_stale_stamp_fails_loud_naming_both_hashes() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("toolchain.lock"), b"PIN=abc\n").unwrap();
        let current = kernel_registry::current_toolchain_hash(dir.path()).unwrap();
        assert_ne!(current, "9da6356ac521", "test setup must actually disagree with the real old pin");
        let mut meta = base_meta(8, 4, serde_json::json!({"rope_global": {"type": "input", "offset": 8, "len": 8}}));
        meta["toolchain"] = serde_json::json!({"hash": "9da6356ac521"});
        write_meta(dir.path(), &meta);
        let err = LlmArtifact::load(dir.path()).unwrap_err().to_string();
        assert!(err.contains("toolchain-stale"), "{err}");
        assert!(err.contains("9da6356ac521"), "{err}");
        assert!(err.contains(&current), "{err}");
    }

    #[test]
    fn toolchain_unstamped_artifact_loads_with_a_warning_not_an_error() {
        let dir = tempfile::tempdir().unwrap();
        // No toolchain.lock, no `toolchain` key -- the real shape of every artifact built before
        // this change, e.g. the shipped artifacts-qwen3-0.6b/decode/meta.json.
        let meta = base_meta(8, 4, serde_json::json!({"rope_global": {"type": "input", "offset": 8, "len": 8}}));
        write_meta(dir.path(), &meta);
        let art = LlmArtifact::load(dir.path()).expect("unstamped must be a warning, not fatal");
        assert_eq!(art.toolchain_hash, None);
    }

    // ------------------------------------------------------------------------------------
    // check_per_token_writes -- the inverse arena check ported from whisper_decoder.rs
    // (xdna-engine f446a50): an Arena::Input buffer nothing writes per token is an error.
    // ------------------------------------------------------------------------------------

    /// A global-only artifact has no `rope_theta_local`, and must keep reading as global-only --
    /// that absence is what makes the decoder's write list derivable instead of a literal.
    #[test]
    fn rope_theta_local_is_absent_on_a_global_only_artifact() {
        let dir = tempfile::tempdir().unwrap();
        write_meta(dir.path(), &base_meta(8, 4, serde_json::json!({})));
        assert_eq!(LlmArtifact::load(dir.path()).unwrap().rope_theta_local, None);
    }

    /// Gemma-3 interleaves local and global attention, carries both bases, and declares a THIRD
    /// input. Parsing the local base is what lets the decoder write it: before this the artifact
    /// declared `rope_local`, nothing wrote it, and check_per_token_writes refused the load with
    /// "input-arena buffer(s) with no per-token host write: rope_local".
    #[test]
    fn a_local_attention_artifact_parses_its_local_base_and_declares_a_third_input() {
        let dir = tempfile::tempdir().unwrap();
        let mut meta = base_meta(8, 4, serde_json::json!({
            "rope_global": {"type": "input", "offset": 8,  "len": 8},
            "rope_local":  {"type": "input", "offset": 16, "len": 8},
        }));
        meta["host_protocol"]["rope_theta_local"] = serde_json::json!(10_000.0);
        // A third input needs room for it: the fixture's default input arena fits exactly two.
        meta["input_size"] = serde_json::json!(24);
        write_meta(dir.path(), &meta);
        let art = LlmArtifact::load(dir.path()).unwrap();
        assert_eq!(art.rope_theta_local, Some(10_000.0));

        // The decoder derives its write list from exactly this, so pin both directions.
        assert!(art.check_per_token_writes(&["x", "rope_global"]).is_err(),
            "two writes must NOT satisfy a three-input artifact");
        art.check_per_token_writes(&["x", "rope_global", "rope_local"])
            .expect("naming all three inputs is what the decoder now does");
    }

    /// The shape the GENERATOR actually emits for a global-only spec: the key is present and
    /// `null`, not absent. Qwen3-0.6B's shipped meta.json is exactly this, and matching only on
    /// absence sent the pinned default model down the malformed branch -- `npu generate --model
    /// qwen3-0.6b` failed to load with "rope_theta_local present but non-numeric" while every unit
    /// test passed, because no fixture carried the real form.
    #[test]
    fn a_present_null_local_base_reads_as_global_only() {
        let dir = tempfile::tempdir().unwrap();
        let mut meta = base_meta(8, 4, serde_json::json!({}));
        meta["host_protocol"]["rope_theta_local"] = serde_json::Value::Null;
        write_meta(dir.path(), &meta);
        assert_eq!(LlmArtifact::load(dir.path()).unwrap().rope_theta_local, None);
    }

    /// A non-numeric value must not read as "global-only": that would silently skip the local write
    /// and leave the local layers rotating at the global base -- wrong output, no error.
    #[test]
    fn a_malformed_local_base_is_an_error_not_a_silent_global_only() {
        let dir = tempfile::tempdir().unwrap();
        let mut meta = base_meta(8, 4, serde_json::json!({}));
        meta["host_protocol"]["rope_theta_local"] = serde_json::json!("10000");
        write_meta(dir.path(), &meta);
        let e = LlmArtifact::load(dir.path()).unwrap_err().to_string();
        assert!(e.contains("rope_theta_local"), "{e}");
    }

    #[test]
    fn per_token_writes_passes_when_every_input_buffer_is_named() {
        let dir = tempfile::tempdir().unwrap();
        let meta = base_meta(8, 4, serde_json::json!({"rope_global": {"type": "input", "offset": 8, "len": 8}}));
        write_meta(dir.path(), &meta);
        let art = LlmArtifact::load(dir.path()).unwrap();
        art.check_per_token_writes(&["x", "rope_global"]).expect("x and rope_global are both named");
    }

    #[test]
    fn per_token_writes_passes_via_the_rope_global_compat_shim_too() {
        // rope_global absent from `layout` (shimmed in by `load`) must still be seen as an
        // Arena::Input entry by this check -- it runs against the resolved layout, not raw meta.json.
        let dir = tempfile::tempdir().unwrap();
        write_meta(dir.path(), &base_meta(8, 4, serde_json::json!({})));
        let art = LlmArtifact::load(dir.path()).unwrap();
        art.check_per_token_writes(&["x", "rope_global"]).expect("shimmed rope_global still counts");
    }

    #[test]
    fn per_token_writes_fails_loud_naming_an_unwritten_input_buffer() {
        let dir = tempfile::tempdir().unwrap();
        let mut meta = base_meta(
            8,
            4,
            serde_json::json!({
                "rope_global": {"type": "input", "offset": 8, "len": 8},
                "positions": {"type": "input", "offset": 16, "len": 4},
            }),
        );
        // base_meta's input_size only covers [x, rope_global); widen it to also fit `positions`.
        meta["input_size"] = serde_json::json!(20);
        write_meta(dir.path(), &meta);
        let art = LlmArtifact::load(dir.path()).unwrap();
        // The decoder only knows to write x and rope_global -- `positions` is a declared input
        // buffer the decoder does not handle, exactly the missing-write shape this guards.
        let err = art.check_per_token_writes(&["x", "rope_global"]).unwrap_err().to_string();
        assert!(err.contains("positions"), "{err}");
    }

    #[test]
    fn toolchain_stamped_but_unresolvable_pin_loads_with_a_warning_not_an_error() {
        let dir = tempfile::tempdir().unwrap();
        // Stamped, but no toolchain.lock anywhere above this dir -- the production-install shape
        // the design constraint requires: a shipped consumer must not need toolchain.lock to exist.
        let mut meta = base_meta(8, 4, serde_json::json!({"rope_global": {"type": "input", "offset": 8, "len": 8}}));
        meta["toolchain"] = serde_json::json!({"hash": "9da6356ac521"});
        write_meta(dir.path(), &meta);
        let art = LlmArtifact::load(dir.path()).expect("an unresolvable pin must be a warning, not fatal");
        assert_eq!(art.toolchain_hash.as_deref(), Some("9da6356ac521"));
    }
}
