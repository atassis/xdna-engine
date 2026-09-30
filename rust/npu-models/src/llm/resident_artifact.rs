//! Readers for the resident-forward artifact's `meta.json` and the weight store's
//! `manifest.json`, parse-don't-validate as `LlmArtifact::load` does (see that module's doc). No
//! device access here -- these two types only turn on-disk JSON into checked Rust values; loading
//! bytes into an [`npu_xrt::Arena`] is a later stage.
//!
//! The artifact's `meta.json` names weight regions by the manifest key of a separate,
//! content-addressed weight store (`manifest.json` + `blobs/<sha256>.bin`): the store owns what
//! the bytes are, the artifact owns where they land in the scratch arena, and a region whose
//! manifest layout or length disagrees with what the artifact claims is refused at load
//! (`check_weights_against_store`) -- a region name is a claim about its bytes, and that is the
//! check that it is the same claim.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::api::EngineError;
use crate::llm::artifact::BufLoc;

/// One `classes` entry: a control code that carries `nt` row-blocks (K035 stopgap -- see the spec).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidentClass {
    pub code: String,
    pub nt: usize,
}

/// One `segments` entry: a contiguous layer range dispatched under one class code, optionally
/// carrying the input embed (`x_in`) and/or the LM head (`head`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidentSegment {
    pub layers: (usize, usize),
    pub x_in: bool,
    pub head: bool,
    pub suffix: String,
}

/// `meta.json`'s `dims` block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidentDims {
    pub layers: usize,
    pub d_model: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim_sliding: usize,
    pub head_dim_global: usize,
    pub ffn: usize,
    pub vocab: usize,
    pub sliding_window: usize,
    pub row_block: usize,
    pub pcap_blocks: usize,
}

/// One `derived` entry: a size the loader trusts only because it names the bound it was computed
/// from (K038). `source: "assumed"` is refused at load time -- see [`ResidentArtifact::load`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DerivedSize {
    pub value: usize,
    pub bound: String,
    pub source: String,
}

/// One `scratchpad.params` entry: a per-request scalar the device reads out of a declared range
/// (K035/K040), with an optional compiled maximum for a value that drives a runtime repeat (K043).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScratchpadParam {
    pub byte_offset: usize,
    pub kind: String, // "addr" | "core"
    pub range: (i64, i64),
    pub compiled_max: Option<usize>,
}

/// `meta.json`'s `store` block: which manifest this artifact's weight regions resolve against.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoreRef {
    pub manifest_sha256: String,
    pub path: PathBuf,
}

/// A `meta.json` `layout` entry naming a weight region: where it lands in the scratch arena, and
/// the manifest key + layout string it must agree with (§1.3: "a region name is a claim about its
/// bytes, and this is the check that it is the same claim").
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeightRegion {
    pub manifest_key: String,
    pub scratch_off: usize,
    pub len: usize,
    pub layout: String,
}

/// A parsed, checked `resident_forward` `meta.json`. Construction fails loud on every check this
/// stage owns (§1.2's loader-check list); it does not open a device or read the store's blobs.
#[derive(Debug, Clone)]
pub struct ResidentArtifact {
    pub schema: u64,
    /// Directory this was loaded from -- ELF and store paths (`store.path`) resolve relative to it.
    pub dir: PathBuf,
    /// `meta.json`'s `elf` field (the file's basename inside `dir`), mirroring
    /// `LlmArtifact::elf_name`/`elf_path`.
    pub elf_name: String,
    pub boot: String,
    pub classes: Vec<ResidentClass>,
    pub segments: Vec<ResidentSegment>,
    pub dims: ResidentDims,
    pub derived: HashMap<String, DerivedSize>,
    pub scratchpad_params: HashMap<String, ScratchpadParam>,
    pub canary_bytes_per_memtile: usize,
    pub store: StoreRef,
    pub weights: HashMap<String, WeightRegion>,
    /// The three `FusedArena` sizes ([`npu_xrt::FusedArena::new`]'s `input_size`/`output_size`/
    /// `scratch_size`), same top-level fields `LlmArtifact::load` reads.
    pub input_size: usize,
    pub output_size: usize,
    pub scratch_size: usize,
    /// Non-weight input/output buffers (`x_in`, the RoPE tables, the canary halves, the logits
    /// output) by name, in [`crate::llm::artifact::BufLoc`]'s arena/offset/length shape -- the same
    /// buffer-location convention `LlmArtifact` uses, so a resident driver addresses them the same
    /// way `NpuDecodeStep` addresses `x`/`logits`.
    pub io: HashMap<String, BufLoc>,
}

impl ResidentArtifact {
    /// Load and validate `dir/meta.json`. Checks enforced here (§1.2):
    /// - `kind` is `resident_forward`;
    /// - `classes` has no gap: every `nt` in `1..=max(nt)` is present exactly once;
    /// - `segments` tiles `[0, dims.layers)` with no gap and no overlap;
    /// - no `derived` entry has `source: "assumed"`.
    pub fn load(dir: &Path) -> Result<ResidentArtifact, EngineError> {
        let meta_path = dir.join("meta.json");
        let bytes = fs::read(&meta_path)
            .map_err(|e| EngineError::Load(format!("read {}: {e}", meta_path.display())))?;
        let meta: serde_json::Value = serde_json::from_slice(&bytes)
            .map_err(|e| EngineError::Load(format!("parse {}: {e}", meta_path.display())))?;
        let ctx = |msg: String| EngineError::Load(format!("{}: {msg}", meta_path.display()));

        let kind = meta.get("kind").and_then(|v| v.as_str()).ok_or_else(|| ctx("missing `kind`".into()))?;
        if kind != "resident_forward" {
            return Err(ctx(format!("`kind` is {kind:?}, expected \"resident_forward\"")));
        }
        let schema = meta.get("schema").and_then(|v| v.as_u64()).ok_or_else(|| ctx("missing `schema`".into()))?;
        let boot = meta.get("boot").and_then(|v| v.as_str()).ok_or_else(|| ctx("missing `boot`".into()))?.to_string();
        let elf_name = meta.get("elf").and_then(|v| v.as_str()).ok_or_else(|| ctx("missing `elf`".into()))?.to_string();
        let usz_top = |k: &str| -> Result<usize, EngineError> {
            meta.get(k).and_then(|v| v.as_u64()).map(|v| v as usize).ok_or_else(|| ctx(format!("missing/non-numeric top-level `{k}`")))
        };
        let input_size = usz_top("input_size")?;
        let output_size = usz_top("output_size")?;
        let scratch_size = usz_top("scratch_size")?;

        let classes = parse_classes(&meta, &ctx)?;
        check_classes_no_gap(&classes).map_err(&ctx)?;

        let dims = parse_dims(&meta, &ctx)?;
        let segments = parse_segments(&meta, &ctx)?;
        check_segments_tile(&segments, dims.layers).map_err(&ctx)?;

        let derived = parse_derived(&meta, &ctx)?;
        for (name, d) in &derived {
            if d.source == "assumed" {
                return Err(ctx(format!("derived[{name}].source is \"assumed\" -- not a real bound")));
            }
        }

        let scratchpad_params = parse_scratchpad_params(&meta, &ctx)?;

        let canary_bytes_per_memtile = meta
            .get("canary")
            .and_then(|c| c.get("bytes_per_memtile"))
            .and_then(|v| v.as_u64())
            .ok_or_else(|| ctx("missing `canary.bytes_per_memtile`".into()))? as usize;

        let store = parse_store(&meta, dir, &ctx)?;
        let weights = parse_weight_layout(&meta, &ctx)?;
        let io = parse_io(&meta, &ctx)?;

        Ok(ResidentArtifact {
            schema,
            dir: dir.to_path_buf(),
            elf_name,
            boot,
            classes,
            segments,
            dims,
            derived,
            scratchpad_params,
            canary_bytes_per_memtile,
            store,
            weights,
            input_size,
            output_size,
            scratch_size,
            io,
        })
    }

    /// `dir/<elf>` (or `<elf>.zst`, via the shared free function), mirroring
    /// `LlmArtifact::elf_path`/`read_elf_bytes`.
    pub fn elf_path(&self) -> PathBuf {
        self.dir.join(&self.elf_name)
    }

    pub fn read_elf_bytes(&self) -> Result<Vec<u8>, EngineError> {
        crate::llm::artifact::read_elf_bytes(&self.elf_path())
    }

    /// Check every `layout` region against the store manifest it names: a region whose manifest
    /// `layout` string or `length` disagrees is refused (§1.3 -- "a region name is a claim about
    /// its bytes"). Does not check `store.manifest_sha256` against the manifest's own hash; that
    /// is a follow-up once a full store loader (mmap + dequant) lands.
    pub fn check_weights_against_store(&self, store: &StoreManifest) -> Result<(), EngineError> {
        for (name, region) in &self.weights {
            let entry = store
                .resolve(&region.manifest_key)
                .ok_or_else(|| EngineError::Load(format!("weight region `{name}`: manifest has no key `{}`", region.manifest_key)))?;
            if entry.layout != region.layout {
                return Err(EngineError::Load(format!(
                    "weight region `{name}`: layout {:?} in meta.json but {:?} in the store manifest",
                    region.layout, entry.layout
                )));
            }
            if entry.length != region.len {
                return Err(EngineError::Load(format!(
                    "weight region `{name}`: len {} in meta.json but {} in the store manifest",
                    region.len, entry.length
                )));
            }
        }
        Ok(())
    }
}

fn check_classes_no_gap(classes: &[ResidentClass]) -> Result<(), String> {
    if classes.is_empty() {
        return Err("`classes` is empty".into());
    }
    let max_nt = classes.iter().map(|c| c.nt).max().unwrap();
    for nt in 1..=max_nt {
        let count = classes.iter().filter(|c| c.nt == nt).count();
        if count != 1 {
            return Err(format!("`classes` has {count} entries for nt={nt}, expected exactly 1"));
        }
    }
    Ok(())
}

fn check_segments_tile(segments: &[ResidentSegment], layers: usize) -> Result<(), String> {
    if segments.is_empty() {
        return Err("`segments` is empty".into());
    }
    let mut sorted: Vec<(usize, usize)> = segments.iter().map(|s| s.layers).collect();
    sorted.sort_unstable();
    let mut next = 0usize;
    for (lo, hi) in &sorted {
        if *lo != next {
            return Err(format!("`segments` has a gap or overlap before layer {lo} (expected {next})"));
        }
        if hi <= lo {
            return Err(format!("`segments` entry [{lo}, {hi}) is empty or inverted"));
        }
        next = *hi;
    }
    if next != layers {
        return Err(format!("`segments` cover [0, {next}), expected [0, {layers}) from `dims.layers`"));
    }
    Ok(())
}

fn parse_classes(
    meta: &serde_json::Value, ctx: &impl Fn(String) -> EngineError,
) -> Result<Vec<ResidentClass>, EngineError> {
    let arr = meta.get("classes").and_then(|v| v.as_array()).ok_or_else(|| ctx("missing/non-array `classes`".into()))?;
    arr.iter()
        .map(|c| {
            let code = c.get("code").and_then(|v| v.as_str()).ok_or_else(|| ctx("classes[]: missing `code`".into()))?;
            let nt = c.get("nt").and_then(|v| v.as_u64()).ok_or_else(|| ctx("classes[]: missing `nt`".into()))?;
            Ok(ResidentClass { code: code.to_string(), nt: nt as usize })
        })
        .collect()
}

fn parse_segments(
    meta: &serde_json::Value, ctx: &impl Fn(String) -> EngineError,
) -> Result<Vec<ResidentSegment>, EngineError> {
    let arr = meta.get("segments").and_then(|v| v.as_array()).ok_or_else(|| ctx("missing/non-array `segments`".into()))?;
    arr.iter()
        .map(|s| {
            let layers = s.get("layers").and_then(|v| v.as_array()).ok_or_else(|| ctx("segments[]: missing `layers`".into()))?;
            let lo = layers.first().and_then(|v| v.as_u64()).ok_or_else(|| ctx("segments[].layers: missing lo".into()))?;
            let hi = layers.get(1).and_then(|v| v.as_u64()).ok_or_else(|| ctx("segments[].layers: missing hi".into()))?;
            let x_in = s.get("x_in").and_then(|v| v.as_bool()).unwrap_or(false);
            let head = s.get("head").and_then(|v| v.as_bool()).unwrap_or(false);
            let suffix = s.get("suffix").and_then(|v| v.as_str()).unwrap_or("").to_string();
            Ok(ResidentSegment { layers: (lo as usize, hi as usize), x_in, head, suffix })
        })
        .collect()
}

fn parse_dims(
    meta: &serde_json::Value, ctx: &impl Fn(String) -> EngineError,
) -> Result<ResidentDims, EngineError> {
    let d = meta.get("dims").ok_or_else(|| ctx("missing `dims`".into()))?;
    let usz = |k: &str| -> Result<usize, EngineError> {
        d.get(k).and_then(|v| v.as_u64()).map(|v| v as usize).ok_or_else(|| ctx(format!("dims: missing/non-numeric `{k}`")))
    };
    Ok(ResidentDims {
        layers: usz("layers")?,
        d_model: usz("d_model")?,
        q_heads: usz("q_heads")?,
        kv_heads: usz("kv_heads")?,
        head_dim_sliding: usz("head_dim_sliding")?,
        head_dim_global: usz("head_dim_global")?,
        ffn: usz("ffn")?,
        vocab: usz("vocab")?,
        sliding_window: usz("sliding_window")?,
        row_block: usz("row_block")?,
        pcap_blocks: usz("pcap_blocks")?,
    })
}

fn parse_derived(
    meta: &serde_json::Value, ctx: &impl Fn(String) -> EngineError,
) -> Result<HashMap<String, DerivedSize>, EngineError> {
    let d = meta.get("derived").and_then(|v| v.as_object()).ok_or_else(|| ctx("missing/non-object `derived`".into()))?;
    let mut out = HashMap::new();
    for (name, v) in d {
        // `derived.segments` (the fit table) is a nested object, not a `{value, bound, source}`
        // scalar (§1.2); it carries no loader check yet and is skipped here rather than
        // misparsed as one.
        if name == "segments" {
            continue;
        }
        let value = v.get("value").and_then(|x| x.as_u64()).ok_or_else(|| ctx(format!("derived[{name}]: missing `value`")))? as usize;
        let bound = v.get("bound").and_then(|x| x.as_str()).ok_or_else(|| ctx(format!("derived[{name}]: missing `bound`")))?.to_string();
        let source = v.get("source").and_then(|x| x.as_str()).unwrap_or("").to_string();
        out.insert(name.clone(), DerivedSize { value, bound, source });
    }
    Ok(out)
}

fn parse_scratchpad_params(
    meta: &serde_json::Value, ctx: &impl Fn(String) -> EngineError,
) -> Result<HashMap<String, ScratchpadParam>, EngineError> {
    let obj = meta
        .get("scratchpad")
        .and_then(|s| s.get("params"))
        .and_then(|v| v.as_object())
        .ok_or_else(|| ctx("missing/non-object `scratchpad.params`".into()))?;
    let mut out = HashMap::new();
    for (name, p) in obj {
        let byte_offset = p.get("byte_offset").and_then(|v| v.as_u64()).ok_or_else(|| ctx(format!("scratchpad.params[{name}]: missing `byte_offset`")))? as usize;
        let kind = p.get("kind").and_then(|v| v.as_str()).ok_or_else(|| ctx(format!("scratchpad.params[{name}]: missing `kind`")))?.to_string();
        let range = p.get("range").and_then(|v| v.as_array()).ok_or_else(|| ctx(format!("scratchpad.params[{name}]: missing `range`")))?;
        let lo = range.first().and_then(|v| v.as_i64()).ok_or_else(|| ctx(format!("scratchpad.params[{name}].range: missing lo")))?;
        let hi = range.get(1).and_then(|v| v.as_i64()).ok_or_else(|| ctx(format!("scratchpad.params[{name}].range: missing hi")))?;
        let compiled_max = p.get("compiled_max").and_then(|v| v.as_u64()).map(|v| v as usize);
        out.insert(name.clone(), ScratchpadParam { byte_offset, kind, range: (lo, hi), compiled_max });
    }
    Ok(out)
}

fn parse_store(
    meta: &serde_json::Value, dir: &Path, ctx: &impl Fn(String) -> EngineError,
) -> Result<StoreRef, EngineError> {
    let s = meta.get("store").ok_or_else(|| ctx("missing `store`".into()))?;
    let manifest_sha256 = s.get("manifest").and_then(|v| v.as_str()).ok_or_else(|| ctx("store: missing `manifest`".into()))?.to_string();
    let path_str = s.get("path").and_then(|v| v.as_str()).ok_or_else(|| ctx("store: missing `path`".into()))?;
    Ok(StoreRef { manifest_sha256, path: dir.join(path_str) })
}

/// `meta.json`'s `io` object: non-weight buffers (`x_in`, RoPE tables, canary halves, `logits`),
/// same `{"type": "input"|"output"|"scratch", "offset", "len"}` shape `LlmArtifact::load` uses for
/// its `layout` map (`artifact.rs` around its own `layout_obj` loop).
fn parse_io(
    meta: &serde_json::Value, ctx: &impl Fn(String) -> EngineError,
) -> Result<HashMap<String, BufLoc>, EngineError> {
    use npu_xrt::Arena;
    let obj = meta.get("io").and_then(|v| v.as_object()).ok_or_else(|| ctx("missing/non-object `io`".into()))?;
    let mut out = HashMap::new();
    for (name, e) in obj {
        let arena = match e.get("type").and_then(|v| v.as_str()) {
            Some("input") => Arena::Input,
            Some("output") => Arena::Output,
            Some("scratch") => Arena::Scratch,
            other => return Err(ctx(format!("io[{name}].type = {other:?}, want input/output/scratch"))),
        };
        let off = e.get("offset").and_then(|v| v.as_u64()).ok_or_else(|| ctx(format!("io[{name}]: missing `offset`")))? as usize;
        let len = e.get("len").and_then(|v| v.as_u64()).ok_or_else(|| ctx(format!("io[{name}]: missing `len`")))? as usize;
        out.insert(name.clone(), BufLoc { arena, off, len });
    }
    Ok(out)
}

fn parse_weight_layout(
    meta: &serde_json::Value, ctx: &impl Fn(String) -> EngineError,
) -> Result<HashMap<String, WeightRegion>, EngineError> {
    let obj = meta.get("layout").and_then(|v| v.as_object()).ok_or_else(|| ctx("missing/non-object `layout`".into()))?;
    let mut out = HashMap::new();
    for (name, w) in obj {
        let manifest_key = w.get("manifest_key").and_then(|v| v.as_str()).ok_or_else(|| ctx(format!("layout[{name}]: missing `manifest_key`")))?.to_string();
        let scratch_off = w.get("scratch_off").and_then(|v| v.as_u64()).ok_or_else(|| ctx(format!("layout[{name}]: missing `scratch_off`")))? as usize;
        let len = w.get("len").and_then(|v| v.as_u64()).ok_or_else(|| ctx(format!("layout[{name}]: missing `len`")))? as usize;
        let layout = w.get("layout").and_then(|v| v.as_str()).ok_or_else(|| ctx(format!("layout[{name}]: missing `layout`")))?.to_string();
        out.insert(name.clone(), WeightRegion { manifest_key, scratch_off, len, layout });
    }
    Ok(out)
}

/// One resolved entry from the weight store's `manifest.json`: `(layer, matrix) -> blob region`,
/// flattened to a single string key (see [`StoreManifest::resolve`]) since S0 only needs lookup by
/// the same key `meta.json`'s `layout` names, not the store's own nested traversal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoreEntry {
    pub blob: String,
    pub offset: usize,
    pub length: usize,
    pub layout: String,
}

/// A parsed weight store `manifest.json`. This reads it far enough to resolve a region's
/// blob/offset/length/layout for the cross-check against `meta.json`; it does not mmap `blobs/`
/// or dequantise anything (that is a follow-up loader).
#[derive(Debug, Clone)]
pub struct StoreManifest {
    pub model: String,
    pub num_layers: usize,
    /// The store directory this was loaded from -- [`Self::blob_path`] resolves a blob against it.
    dir: PathBuf,
    entries: HashMap<String, StoreEntry>,
}

impl StoreManifest {
    /// Load `dir/manifest.json`. Flattens `layers.<i>.matrices.<name>` to the key
    /// `"layer<i>.<name>"`, plus top-level `"embedding"` and `"final_norm"`, since those are the
    /// only two shapes `meta.json`'s `layout[].manifest_key` needs to name (§1.3).
    pub fn load(dir: &Path) -> Result<StoreManifest, EngineError> {
        let manifest_path = dir.join("manifest.json");
        let bytes = fs::read(&manifest_path)
            .map_err(|e| EngineError::Load(format!("read {}: {e}", manifest_path.display())))?;
        let meta: serde_json::Value = serde_json::from_slice(&bytes)
            .map_err(|e| EngineError::Load(format!("parse {}: {e}", manifest_path.display())))?;
        let ctx = |msg: String| EngineError::Load(format!("{}: {msg}", manifest_path.display()));

        let model = meta.get("model").and_then(|v| v.as_str()).ok_or_else(|| ctx("missing `model`".into()))?.to_string();
        let num_layers = meta.get("num_layers").and_then(|v| v.as_u64()).ok_or_else(|| ctx("missing `num_layers`".into()))? as usize;

        let mut entries = HashMap::new();
        let layers = meta.get("layers").and_then(|v| v.as_object()).ok_or_else(|| ctx("missing/non-object `layers`".into()))?;
        for (layer_idx, layer) in layers {
            let matrices = layer.get("matrices").and_then(|v| v.as_object()).ok_or_else(|| ctx(format!("layers[{layer_idx}]: missing `matrices`")))?;
            for (matrix_name, m) in matrices {
                // Some layer entries carry a list (a chunked passthrough tensor, §"a list of
                // these, one per K-chunk"); S0 does not need chunk detail, so those are skipped
                // rather than misparsed as a single region.
                if m.is_array() {
                    continue;
                }
                if let Some(entry) = parse_store_entry(m) {
                    entries.insert(format!("layer{layer_idx}.{matrix_name}"), entry);
                }
            }
        }
        if let Some(e) = meta.get("embedding").and_then(parse_store_entry) {
            entries.insert("embedding".to_string(), e);
        }
        if let Some(e) = meta.get("final_norm").and_then(parse_store_entry) {
            entries.insert("final_norm".to_string(), e);
        }

        Ok(StoreManifest { model, num_layers, dir: dir.to_path_buf(), entries })
    }

    pub fn resolve(&self, key: &str) -> Option<&StoreEntry> {
        self.entries.get(key)
    }

    /// `dir/blobs/<entry.blob>.bin` -- where an entry's bytes actually live on disk.
    pub fn blob_path(&self, entry: &StoreEntry) -> PathBuf {
        self.dir.join("blobs").join(format!("{}.bin", entry.blob))
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

fn parse_store_entry(v: &serde_json::Value) -> Option<StoreEntry> {
    Some(StoreEntry {
        blob: v.get("blob")?.as_str()?.to_string(),
        offset: v.get("offset")?.as_u64()? as usize,
        length: v.get("length")?.as_u64()? as usize,
        layout: v.get("layout")?.as_str()?.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Real store manifest packed on this box. Override with `RESIDENT_STORE_DIR` on a box that
    /// keeps it elsewhere; the repo's own convention for a device-box fixture is a defaulted,
    /// overridable path (`detokenize.rs::qwen3_tokenizer_path`).
    fn store_dir() -> PathBuf {
        std::env::var("RESIDENT_STORE_DIR").map(PathBuf::from).unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/artifacts/gemma4-12b/store")
        })
    }

    #[test]
    fn loads_the_real_gemma4_12b_store_manifest() {
        let m = StoreManifest::load(&store_dir()).expect("load the real packed manifest");
        assert_eq!(m.model, "gemma4-12b");
        assert_eq!(m.num_layers, 48);
        assert!(!m.is_empty());
        // Every sliding layer has these three; layer 0 is sliding (not in full_attention_layers).
        for key in ["layer0.mlp_stream", "layer0.attn_in_stream", "layer0.attn_out_stream"] {
            let e = m.resolve(key).unwrap_or_else(|| panic!("manifest missing {key}"));
            assert!(e.length > 0, "{key}: zero-length blob region");
            assert!(!e.blob.is_empty());
        }
        let embedding = m.resolve("embedding").expect("manifest missing `embedding`");
        assert_eq!(embedding.layout, "planar_int4g32_headpack");
    }

    fn fixture_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/resident_meta_ok")
    }

    #[test]
    fn loads_a_seeded_valid_meta() {
        let a = ResidentArtifact::load(&fixture_dir()).expect("a well-formed fixture must load");
        assert_eq!(a.schema, 1);
        assert_eq!(a.boot, "boot");
        assert_eq!(a.classes.len(), 2);
        assert_eq!(a.dims.layers, 2);
        assert_eq!(a.segments.len(), 1);
        assert_eq!(a.elf_name, "resident.elf");
        assert_eq!(a.input_size, 4096);
        assert!(a.io.contains_key("x_in"));
        assert!(a.io.contains_key("logits"));
    }

    // Negative tests (a seeded gap in `classes`, a seeded gap in `segments`, an `assumed`
    // `derived.source`) and the differential tests against the Python value builders are
    // deferred -- see the S0 worklog, "land first, test later".
}
