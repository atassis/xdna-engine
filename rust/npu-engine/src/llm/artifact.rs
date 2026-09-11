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

/// The causal mask of a batched-prefill artifact, which is a per-row WIDTH VECTOR and not a
/// triangle: the scores buffer is `[q_heads*M, S]`, so row `r = h*M + i` is token `i` under head
/// `h`, and it may attend `base + i + 1` positions -- the same count under every head. Softmax
/// masks each row past its own width (`vector_size_source="rows"`), which is why there is no
/// separate mask buffer, no additive triangle, and no scalar width in this mode.
///
/// An ordinary input buffer, rewritten per chunk like `x` and the angle tables, NOT a scratchpad
/// parameter.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MaskWidths {
    /// The declared input buffer holding the widths -- read from `meta.json`, never a literal, for
    /// the same reason the RoPE buffers are.
    pub buffer: String,
    /// `dims.q_heads`. The widths repeat once per head.
    pub heads: usize,
    /// `heads * M` int32 values; the buffer is four times this in bytes.
    pub rows: usize,
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

/// Which half of the LLM rail an artifact drives.
///
/// A decode artifact declares one position per dispatch, an output the host samples from, and every
/// model constant it needs. A prefill artifact declares `dims.M` positions per dispatch and is
/// deliberately allowed to omit the model constants the DECODE half is the authority for: it runs
/// only as one half of a checked pair, and inheriting them is what keeps the pair from carrying two
/// copies of a number that must agree ([`LlmArtifact::check_prefill_pairing`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArtifactRole {
    Decode,
    Prefill,
}

/// Which RoPE angle table a declared input buffer holds. Gemma-3 interleaves local and global
/// attention layers and the ELF reads a separate table for each; a global-only model (Qwen3) has
/// one, and may name it either `rope_global` or `rope`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RopeBase {
    Global,
    Local,
}

/// A validated, ready-to-drive fused decode ELF: every buffer the artifact declares has a checked
/// location, and the scratchpad protocol (`kv_off`/`sm_mask`) is resolved to concrete offsets.
///
/// `Clone` is metadata-only and paid once at load: window bucketing keeps one of these per bucket
/// so every bucket reads its OWN scratchpad offsets rather than the primary's. It clones maps and
/// paths, never a buffer -- the arena and the ELF bytes live elsewhere.
#[derive(Debug, Clone)]
pub struct LlmArtifact {
    pub role: ArtifactRole,
    pub decode_dir: PathBuf,
    pub elf_name: String,
    pub kernel_name: String,
    pub input_size: usize,
    pub output_size: usize,
    pub scratch_size: usize,
    pub layout: HashMap<String, BufLoc>,
    pub weights: Vec<String>,
    /// The buffer the host reads after a dispatch. `None` on a prefill artifact: prefill's whole
    /// product is the KV cache it leaves in scratch, so it declares no output and the host reads
    /// nothing back. Required on a decode artifact -- one whose logits buffer cannot be located is
    /// not drivable at all.
    pub output: Option<String>,
    /// `meta.json`'s `cache_buffers` -- the KV-cache scratch buffers a fresh generation must
    /// re-zero (`NpuDecodeStep::reset`). Absent (empty) is legal: a model with no on-device cache
    /// buffer still validates.
    pub cache_buffers: Vec<String>,
    /// `meta.json`'s `embed_blob`; see [`Self::embed_blob`]. `None` in pre-2026-09-08 artifacts.
    pub embed_blob: Option<String>,
    pub kv_off: ScratchpadParam,
    /// The scalar causal-width parameter. Required on a decode artifact. `None` on every prefill
    /// artifact the current generator emits, in BOTH arms and for two different reasons: the
    /// causal one masks with [`Self::mask_widths`] instead, and the non-causal control masks
    /// nothing. Either way `scratchpad.mask_param` is `null` and there is nothing to write.
    pub sm_mask: Option<ScratchpadParam>,
    /// The attention-window scratchpad parameter (`scratchpad.window_param`, resolved the same
    /// way as [`Self::kv_off`]/[`Self::sm_mask`]). `None` on every artifact today -- the window
    /// is still a build-time constant and the engine ships four separately-compiled designs, one
    /// per window, selected per token the way [`crate::llm::npu_decode`]'s bucket selector
    /// already does. Present only on an artifact built for the dynamic-window design, and always
    /// together with [`Self::window_granule`] -- see the cross-check at the end of [`Self::load_role`].
    pub attn_window: Option<ScratchpadParam>,
    /// `meta.json`'s `dims.window_granule` -- the unit the host rounds an attended length up to
    /// before writing [`Self::attn_window`] (`crate::llm::npu_decode::window_len`). Present
    /// exactly when `attn_window` is: a scratchpad pointer with no granule has no unit to round
    /// against, and a granule with no pointer has nothing to write it to.
    pub window_granule: Option<usize>,
    /// `meta.json`'s `window_rungs`: the NAMED control codes this one ELF carries besides
    /// `main:sequence`, each a decode-layer design at a narrower attention window over the SAME KV
    /// capacity and the SAME arena, as `(kernel subname, window)` sorted ascending by window.
    ///
    /// A rung is not another artifact. aiecc emits one control code per `aie.runtime_sequence` and
    /// XRT resolves them by `main:<name>` against the ONE `hw_context` the ELF registers, so the
    /// rungs cost extra ELF and neither a context nor a rebuild -- which is the whole difference
    /// from the bucket-artifact ladder this supersedes. Empty on every artifact built before rungs
    /// existed, and empty is exactly "one window, the old behaviour".
    pub window_rungs: Vec<(String, usize)>,
    /// The per-row causal widths, when `meta.json` says `causal: true`. Prefill only -- decode is
    /// M=1, where one scalar width says everything there is to say. See [`MaskWidths`].
    pub mask_widths: Option<MaskWidths>,
    /// The declared input buffers holding RoPE angle tables, and which base each wants. Derived
    /// from `inputs` rather than a literal `["rope_global", "rope_local"]`, which is what lets a
    /// single-table prefill artifact name its buffer `rope` without a second code path.
    pub rope_inputs: Vec<(String, RopeBase)>,
    pub head_dim: usize,
    pub d_model: usize,
    /// `dims.vocab`. Required on a decode artifact -- it sizes the logits read and bounds the
    /// embedding gather. `None` on a prefill artifact, which has no lm-head and emits no logits.
    pub vocab: Option<usize>,
    pub n_layers: usize,
    /// `meta.json`'s `dims.S` -- how many token positions the on-device KV cache holds. `kc`/`vc`
    /// are laid out `[S/kv_block, Hkv, kv_block, HD]` (see [`Self::kv_block`]), so this is an
    /// exact capacity, not a hint, and it is a BUILD parameter: the cache's own strides depend on
    /// it, so changing the window means a different artifact.
    ///
    /// Required, like its sibling dims, deliberately. It was recorded here and read by nobody,
    /// which left `pos` unbounded all the way to the dispatch -- and the overrun is silent, since
    /// position S lands on head 1's row 0 rather than outside the arena. An artifact that cannot
    /// say how big its window is cannot have that window enforced, so it fails to load instead.
    pub max_seq: usize,
    /// `meta.json`'s `dims.kv_heads` -- the KV cache's head count, needed (alongside
    /// [`Self::max_seq`] and [`Self::head_dim`]) to compute [`crate::llm::kv_layout::kv_off`]'s
    /// block term. Was already emitted in `meta.json` and simply never read into this struct
    /// before the KV cache had more than one block to address.
    pub kv_heads: usize,
    /// `meta.json`'s `dims.kv_block` -- `iron.common.kv_layout.KVLayout`'s `T`: how many
    /// positions share one contiguous run per head before the cache layout returns to head 0's
    /// next block. Equal to [`Self::max_seq`] (one block, the pre-blocking flat layout) on any
    /// artifact built before this field existed, via a default rather than a parse failure -- an
    /// artifact with no `dims.kv_block` at all IS a flat-layout one, not a malformed one.
    pub kv_block: usize,
    /// `meta.json`'s `dims.M` -- how many token positions ONE dispatch of this ELF covers. 1 on a
    /// decode artifact (absent from its meta, and the decode graph IS the M=1 instance of the
    /// prefill graph); the batch on a prefill artifact, where it is required and where every
    /// per-dispatch host buffer is sized by it.
    pub batch: usize,
    /// `host_protocol.embed_scale`. Required on a decode artifact. `None` on a prefill artifact that
    /// does not declare it: the batched path gathers through the DECODE artifact's embedding table
    /// and its scale, deliberately -- one gather function, so the two paths cannot drift on the
    /// scale-then-narrow, which is exactly what the token-identity gate would report as a
    /// divergence. A prefill artifact that DOES declare it must agree.
    pub embed_scale: Option<EmbedScale>,
    /// `host_protocol.rope_theta_global`. Required on a decode artifact. `None` on a prefill
    /// artifact that does not declare it -- see [`Self::embed_scale`]; the angle rows are computed
    /// from the decode half's base for the same reason.
    pub rope_theta_global: Option<f64>,
    /// The LOCAL RoPE base, for models with interleaved local/global attention (Gemma-3). `None` on
    /// a global-only model (Qwen3), which is why it is optional rather than defaulted: the presence
    /// of this field is exactly what decides whether the artifact declares a `rope_local` input, so
    /// a default would make a two-input and a three-input artifact indistinguishable here.
    pub rope_theta_local: Option<f64>,
    /// `meta.json`'s `toolchain.hash` -- the toolchain.lock semantic hash this ELF was compiled
    /// against (`gen_llm_decode.py`, added 2026-09-05). `None` on any artifact built before this
    /// field existed. See [`LlmArtifact::load`]'s freshness check below.
    pub toolchain_hash: Option<String>,
    /// `meta.json`'s `dims.prefill_break_even_tokens` -- the measured prompt-length crossover
    /// above which one batched dispatch (a fixed cost, independent of how many of its `dims.M`
    /// rows are real tokens) beats priming per-token. A property of the ARTIFACT, not a Rust
    /// constant: a fixed dispatch cost that changes with tiling/M/S changes across a rebuild used
    /// to live in `generator.rs::PREFILL_BREAK_EVEN_TOKENS`, and went stale the first time an
    /// artifact rebuilt without a matching re-sweep -- caught 2026-09-11 when a 13-token prompt
    /// used the batched path at ~1.6x what per-token priming would have cost. `None` on a decode
    /// artifact (irrelevant there) and on any prefill artifact built before this field existed;
    /// the caller falls back to a hardcoded default in that case.
    pub prefill_break_even_tokens: Option<usize>,
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
        Self::load_role(decode_dir, ArtifactRole::Decode)
    }

    /// Load and validate a batched-prefill artifact. Same `meta.json` schema as a decode artifact,
    /// with two role differences enforced here rather than assumed by the caller: `dims.M` is
    /// required (it sizes every per-dispatch host write), and `output` is optional because prefill
    /// produces no logits.
    pub fn load_prefill(prefill_dir: &Path) -> Result<LlmArtifact, EngineError> {
        Self::load_role(prefill_dir, ArtifactRole::Prefill)
    }

    fn load_role(decode_dir: &Path, role: ArtifactRole) -> Result<LlmArtifact, EngineError> {
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
        let output = match role {
            ArtifactRole::Decode => Some(str_field("output")?),
            // Present-and-null and absent are the same thing, matching how `rope_theta_local` is
            // read below: a generator that emits every key for every role writes `null` here.
            ArtifactRole::Prefill => match meta.get("output") {
                None | Some(serde_json::Value::Null) => None,
                Some(v) => Some(
                    v.as_str()
                        .map(str::to_string)
                        .ok_or_else(|| ctx("`output` present but non-string".to_string()))?,
                ),
            },
        };
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
        // Prefill has no lm-head, so it has no vocabulary to declare. Decode reads its logits
        // buffer against this and bounds the embedding gather with it, so there it is required.
        let vocab = match role {
            ArtifactRole::Decode => Some(dim("vocab")?),
            ArtifactRole::Prefill => dims.get("vocab").and_then(|v| v.as_u64()).map(|v| v as usize),
        };
        let n_layers = dim("layers")?;
        let max_seq = dim("S")?;
        // Both absent on any artifact built before blocking landed (including this file's own
        // pre-blocking test fixtures). Default kv_block to max_seq -- one block, the pre-blocking
        // flat layout, exactly what such an artifact IS -- rather than failing to load an
        // otherwise valid old artifact over a field it had no reason to carry. kv_heads then
        // defaults harmlessly to 0: kv_off's block term multiplies by kv_heads only when
        // `pos / kv_block > 0`, which cannot happen while kv_block == max_seq and pos < max_seq,
        // so an unknown kv_heads is provably never read in that case.
        let kv_block = match dims.get("kv_block") {
            None | Some(serde_json::Value::Null) => max_seq,
            Some(v) => v
                .as_u64()
                .ok_or_else(|| ctx("dims.kv_block present but non-numeric".to_string()))?
                as usize,
        };
        let kv_heads = match dims.get("kv_heads") {
            None | Some(serde_json::Value::Null) if kv_block == max_seq => 0,
            other => other
                .and_then(|v| v.as_u64())
                .ok_or_else(|| ctx("dims.kv_heads missing/non-numeric".to_string()))?
                as usize,
        };
        // Absent on every artifact today -- the dynamic-window design this pairs with
        // (`scratchpad.window_param`, read below) hasn't shipped one yet. `null` and absent both
        // mean "no granule", the same convention `kv_block`/`rope_theta_local` use.
        let window_granule = match dims.get("window_granule") {
            None | Some(serde_json::Value::Null) => None,
            Some(v) => Some(
                v.as_u64().ok_or_else(|| ctx("dims.window_granule present but non-numeric".to_string()))? as usize,
            ),
        };
        // `dims.M` is what makes a prefill artifact drivable: it sizes `x` and the RoPE angle
        // block, it is the padded chunk width, and it is the batch the causal width is derived
        // from. A decode artifact does not carry it and does not need to -- decode IS M=1.
        let batch = match role {
            ArtifactRole::Decode => 1,
            ArtifactRole::Prefill => match dim("M") {
                Ok(0) => return Err(ctx("dims.M = 0".to_string())),
                other => other?,
            },
        };

        // See `prefill_break_even_tokens`'s own doc comment for why this lives here rather than
        // as a Rust constant. Absent (pre-2026-09-11 artifacts, and every decode artifact) is a
        // valid state, not an error -- the caller supplies its own default.
        let prefill_break_even_tokens = match role {
            ArtifactRole::Decode => None,
            ArtifactRole::Prefill => dims.get("prefill_break_even_tokens").and_then(|v| v.as_u64()).map(|v| v as usize),
        };

        // Optional as a whole only for prefill, whose model constants come from the decode half.
        static NO_HP: serde_json::Value = serde_json::Value::Null;
        let hp = match role {
            ArtifactRole::Decode => meta
                .get("host_protocol")
                .ok_or_else(|| ctx("missing top-level `host_protocol`".to_string()))?,
            ArtifactRole::Prefill => meta.get("host_protocol").unwrap_or(&NO_HP),
        };
        let embed_scale = match hp.get("embed_scale").and_then(|v| v.as_str()) {
            Some("none") => Some(EmbedScale::None),
            Some("sqrt_d_model") => Some(EmbedScale::SqrtDModel),
            None if role == ArtifactRole::Prefill => None,
            other => return Err(ctx(format!("host_protocol.embed_scale = {other:?}, want \"none\" or \"sqrt_d_model\""))),
        };
        let rope_theta_global = match hp.get("rope_theta_global").and_then(|v| v.as_f64()) {
            Some(t) => Some(t),
            None if role == ArtifactRole::Prefill => None,
            None => return Err(ctx("host_protocol.rope_theta_global missing/non-numeric".to_string())),
        };
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
        // `null` is a real answer, not an omission: a non-causal prefill bring-up build has no
        // scalar causal width to write. Decode always has one -- without it every position would
        // attend the whole compiled window.
        let mask_param_name = match sp.get("mask_param") {
            Some(serde_json::Value::Null) | None if role == ArtifactRole::Prefill => None,
            other => Some(
                other
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| ctx("scratchpad.mask_param missing".to_string()))?,
            ),
        };
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
        let sm_mask = mask_param_name.map(read_param).transpose()?;
        // The third scratchpad pointer, mirroring `kv_param`/`mask_param`: absent (or explicit
        // `null`) on every artifact today, since the window is still a build-time constant. A
        // window pointer with no granule to round against -- or a granule with nothing to write
        // it to -- is a half-wired artifact, so the two are required together rather than each
        // silently defaulting to "not declared" on its own.
        let window_param_name = match sp.get("window_param") {
            Some(serde_json::Value::Null) | None => None,
            other => Some(
                other
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| ctx("scratchpad.window_param present but non-string".to_string()))?,
            ),
        };
        let attn_window = window_param_name.map(read_param).transpose()?;
        if attn_window.is_some() != window_granule.is_some() {
            return Err(ctx(format!(
                "scratchpad.window_param ({window_param_name:?}) and dims.window_granule \
                 ({window_granule:?}) must both be present or both absent -- one without the \
                 other computes an attended length against an undefined unit"
            )));
        }

        // Rungs are validated here rather than trusted, because a bad one is a plausible wrong
        // answer and never an error: a rung claiming a window it was not built at would attend
        // short and return a believable token. A rung wider than `S` is refused for the same
        // reason the generator refuses to build one.
        let mut window_rungs: Vec<(String, usize)> = Vec::new();
        if let Some(v) = meta.get("window_rungs") {
            let obj = v
                .as_object()
                .ok_or_else(|| ctx("window_rungs present but not an object".to_string()))?;
            for (name, w) in obj {
                let w = w
                    .as_u64()
                    .ok_or_else(|| ctx(format!("window_rungs[{name}] is non-numeric")))?
                    as usize;
                if w == 0 || w > max_seq {
                    return Err(ctx(format!(
                        "window_rungs[{name}] = {w} is not a window inside dims.S = {max_seq}"
                    )));
                }
                window_rungs.push((name.clone(), w));
            }
            window_rungs.sort_by_key(|(_, w)| *w);
            if attn_window.is_none() {
                return Err(ctx(
                    "window_rungs without scratchpad.window_param: a rung quantises the shim's \
                     FILL and relies on the core taking its own window at runtime, so a rung set \
                     with no runtime window would attend the rung's whole width at every position"
                        .to_string(),
                ));
            }
        }

        // The RoPE angle buffers, resolved from what the artifact DECLARES. `rope` and
        // `rope_global` are the same thing under two spellings -- the decode generator emits the
        // second, the prefill generator the first, and a single-table model has exactly one.
        let mut rope_inputs: Vec<(String, RopeBase)> = Vec::new();
        for n in &inputs {
            match n.as_str() {
                "rope" | "rope_global" => rope_inputs.push((n.clone(), RopeBase::Global)),
                "rope_local" => rope_inputs.push((n.clone(), RopeBase::Local)),
                // Anything else is caught by `check_per_token_writes`, which compares the input
                // arena against the write list this derives -- so an unrecognised input buffer is
                // reported as an unwritten one rather than skipped here in silence.
                _ => {}
            }
        }

        // Compat shim, narrow and logged: ONLY for `rope_global` immediately following `x`, the exact
        // shape `gen_llm_decode.py` currently emits. Any other gap still fails loud below.
        if inputs.iter().any(|n| n == "rope_global") && !layout.contains_key("rope_global") {
            let x = *layout
                .get("x")
                .ok_or_else(|| ctx("rope_global compat shim needs `x`'s layout entry, and it is ALSO missing -- refusing to guess".to_string()))?;
            let len = batch * head_dim * 2; // bf16 bytes, one angle row per position in the batch
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
            .chain(output.iter())
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

        // A prefill dispatch's host writes are all `dims.M` rows wide, so a layout that disagrees
        // with `dims.M` means the host would either short-write the tail rows (garbage KV, no
        // error) or overrun into the next input buffer. Checked for the prefill role only: the
        // decode artifacts already in the field are not re-validated by this change.
        if role == ArtifactRole::Prefill {
            let rope_names = rope_inputs.iter().map(|(n, _)| (n.as_str(), head_dim, "head_dim"));
            for (name, unit, what) in std::iter::once(("x", d_model, "d_model")).chain(rope_names) {
                let Some(loc) = layout.get(name) else { continue };
                let want = batch * unit * 2;
                if loc.len != want {
                    return Err(ctx(format!(
                        "layout[{name}].len = {} but dims.M({batch}) * {what}({unit}) * 2 = {want}",
                        loc.len
                    )));
                }
            }
        }

        // The causal mask. `causal: true` says this graph masks with a per-row width VECTOR --
        // an ordinary input buffer, not a scratchpad scalar -- so everything the host needs to
        // fill it is checked here rather than assumed at dispatch. It is read for the prefill role
        // only: `causal` on a decode artifact would be describing its scalar `sm_mask`, a
        // different mechanism, and reading it as this one would demand a widths buffer decode
        // does not have.
        let mask_widths = match (role, meta.get("causal").and_then(|v| v.as_bool())) {
            (ArtifactRole::Prefill, Some(true)) => {
                let mw = meta.get("mask_widths").filter(|v| !v.is_null()).ok_or_else(|| {
                    ctx("`causal` is true but there is no `mask_widths` block naming the widths buffer".to_string())
                })?;
                let buffer = mw
                    .get("buffer")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| ctx("mask_widths.buffer missing/non-string".to_string()))?
                    .to_string();
                // Stated, not assumed. IRON's AIERuntimeArgSpec defaults to bfloat16 and sizes the
                // device buffer off its dtype, so a widths spec that forgot its dtype allocates
                // half of what the host is about to write -- silently, into the next buffer.
                match mw.get("dtype").and_then(|v| v.as_str()) {
                    Some("int32") => {}
                    other => {
                        return Err(ctx(format!(
                            "mask_widths.dtype = {other:?}, want \"int32\" -- a widths buffer of any \
                             other width is not the one the host writes"
                        )))
                    }
                }
                let q_heads = dims
                    .get("q_heads")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize)
                    .ok_or_else(|| ctx("dims.q_heads missing/non-numeric, and the widths are one int32 per (head, token) row".to_string()))?;
                if !inputs.iter().any(|n| n == &buffer) {
                    return Err(ctx(format!(
                        "mask_widths.buffer `{buffer}` is not a declared input, so nothing would \
                         place it in the input arena"
                    )));
                }
                let loc = layout
                    .get(&buffer)
                    .ok_or_else(|| ctx(format!("mask_widths.buffer `{buffer}` has no `layout` entry")))?;
                let want = q_heads * batch * 4;
                if loc.len != want {
                    return Err(ctx(format!(
                        "layout[{buffer}].len = {} but dims.q_heads({q_heads}) * dims.M({batch}) * 4 = {want}",
                        loc.len
                    )));
                }
                if let Some(declared) = mw.get("len").and_then(|v| v.as_u64()) {
                    if declared as usize != want {
                        return Err(ctx(format!("mask_widths.len = {declared}, but the layout says {want}")));
                    }
                }
                // Two mask sources cannot both be right: the rows softmax takes no scalar, and a
                // graph that carried both would be masking twice at two different widths.
                if sm_mask.is_some() {
                    return Err(ctx(format!(
                        "artifact declares both the widths buffer `{buffer}` and a scalar \
                         scratchpad width -- one graph, two disagreeing masks"
                    )));
                }
                Some(MaskWidths { buffer, heads: q_heads, rows: q_heads * batch })
            }
            _ => None,
        };

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
            role,
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
            attn_window,
            window_granule,
            window_rungs,
            mask_widths,
            rope_inputs,
            head_dim,
            d_model,
            vocab,
            n_layers,
            max_seq,
            kv_heads,
            kv_block,
            batch,
            embed_scale,
            rope_theta_global,
            rope_theta_local,
            toolchain_hash,
            prefill_break_even_tokens,
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
        // ABSOLUTE, NOT CANONICAL -- do not resolve symlinks here. `install.sh` stages a production
        // root holding its own `toolchain.lock` (the pin the artifacts were BUILT against) beside an
        // `artifacts` SYMLINK into the dev checkout. Canonicalizing follows that symlink, so the
        // walk-up sails past the staged lock and lands on whatever the developer's tree is pinned at
        // right now -- gating a shipped artifact on an unrelated working tree.
        //
        // Measured 2026-09-09: a re-pin in the checkout made every installed artifact fail to load
        // with `toolchain-stale`, while the staged lock sitting directly above them still hashed to
        // exactly what they were built with. The artifact was fine and the right answer was one
        // directory up; canonicalize() walked past it.
        //
        // Lexical walk-up gives each artifact the lock of the tree it LIVES in, which is the
        // question this check is actually asking. A dev-tree artifact still resolves the dev pin.
        let mut dir = std::path::absolute(start)?;
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

    /// The declared output buffer, or a load-shaped error naming the artifact. Only a decode
    /// artifact has one; a prefill artifact's product is the KV cache it leaves behind.
    pub fn output_name(&self) -> Result<&str, EngineError> {
        self.output.as_deref().ok_or_else(|| {
            EngineError::Load(format!(
                "{}: artifact declares no `output` buffer (role {:?})",
                self.decode_dir.display(),
                self.role
            ))
        })
    }

    /// The precondition for sharing ONE `FusedArena` between this artifact's ELF and `other`'s.
    ///
    /// Arena offsets in IRON are emergent from runlist order (`iron/common/sequence.py`'s
    /// `add_buffers` packs input args, then output args, then every other buffer in first-appearance
    /// order), so two independently generated graphs agreeing on where `L0_kc` lives is a property
    /// of how they were built, never a guarantee. Both generators declaring the same
    /// `scratch_order` is what makes it true; this is what makes it CHECKED, and it fails loud
    /// naming the first divergent buffer rather than letting one ELF's weights land on the other's
    /// KV cache.
    ///
    /// Three checks, in the order a divergence is most likely to appear:
    ///
    /// 1. every scratch buffer both artifacts name sits at the same `(offset, len)`;
    /// 2. every one of this artifact's `cache_buffers` is declared by `other` at all -- prefill
    ///    that does not name the cache decode reads is not writing it;
    /// 3. no buffer `other` declares under a DIFFERENT name overlaps one of this artifact's
    ///    weight/cache buffers.
    ///
    /// The input and output arenas are deliberately out of scope: both are rewritten by whichever
    /// path is about to dispatch, and they legitimately differ in size (prefill's `x` is `M` rows).
    ///
    /// KNOWN GAP, and it is the generator's to close, not this function's: `meta.json` lists only
    /// the buffers the graph DECLARES. IRON also packs undeclared launch-to-launch intermediates
    /// into scratch, and those are invisible here -- check 3 can only see what is named. What keeps
    /// them off the weights is `scratch_order` putting `[*weights, *caches]` in a common prefix.
    pub fn check_shared_layout_agrees(&self, other: &LlmArtifact) -> Result<(), EngineError> {
        let label = |a: &LlmArtifact| a.decode_dir.display().to_string();
        let mine = label(self);
        let theirs = label(other);

        let mut shared: Vec<&str> = self
            .layout
            .keys()
            .filter(|n| other.layout.contains_key(n.as_str()))
            .map(String::as_str)
            .filter(|n| self.loc(n).arena == Arena::Scratch || other.layout[*n].arena == Arena::Scratch)
            .collect();
        shared.sort_unstable();
        for name in shared {
            let a = *self.loc(name);
            let b = other.layout[name];
            if a != b {
                return Err(EngineError::Load(format!(
                    "shared-arena layout mismatch on `{name}`: {mine} places it at {:?}[{}, {}) \
                     but {theirs} places it at {:?}[{}, {}) -- the two ELFs cannot share one arena; \
                     regenerate both with the same declared scratch_order",
                    a.arena, a.off, a.off + a.len, b.arena, b.off, b.off + b.len
                )));
            }
        }

        let mut absent: Vec<&str> =
            self.cache_buffers.iter().map(String::as_str).filter(|n| !other.layout.contains_key(*n)).collect();
        absent.sort_unstable();
        if let Some(name) = absent.first() {
            return Err(EngineError::Load(format!(
                "{theirs} declares no buffer `{name}`, which {mine} lists in `cache_buffers` \
                 ({} more missing) -- the two ELFs do not agree on the KV cache",
                absent.len() - 1
            )));
        }

        // Protected = everything this artifact needs to survive the other ELF running: its weights
        // (which include the caches -- the generator lists a cache buffer in both) and its caches.
        let mut protected: Vec<(usize, usize, &str)> = self
            .weights
            .iter()
            .chain(self.cache_buffers.iter())
            .map(String::as_str)
            .filter_map(|n| {
                let l = self.layout.get(n)?;
                (l.arena == Arena::Scratch).then_some((l.off, l.off + l.len, n))
            })
            .collect();
        protected.sort_unstable();
        protected.dedup();
        let mut theirs_spans: Vec<(usize, usize, &str)> = other
            .layout
            .iter()
            .filter(|(_, l)| l.arena == Arena::Scratch)
            .map(|(n, l)| (l.off, l.off + l.len, n.as_str()))
            .collect();
        theirs_spans.sort_unstable();
        for &(off, end, name) in &protected {
            for &(o2, e2, n2) in &theirs_spans {
                if n2 != name && o2 < end && off < e2 {
                    return Err(EngineError::Load(format!(
                        "shared-arena collision: {theirs}'s `{n2}` at scratch[{o2}, {e2}) overlaps \
                         {mine}'s `{name}` at scratch[{off}, {end}) -- one ELF would overwrite the \
                         other's weights or KV cache"
                    )));
                }
            }
        }
        Ok(())
    }

    /// The input buffers the host rewrites before every dispatch: `x`, whatever RoPE tables this
    /// artifact declares, and the causal widths on a causal one. Derived, never a literal -- the
    /// list is model-shaped (Gemma-3 has a third table) and role-shaped (the prefill generator
    /// names its single table `rope`).
    /// Every `Arena::Input` BUFFER the caller must write each token. Scratchpad REGISTERS
    /// (`kv_off`, `sm_mask`, `attn_window`) are deliberately absent: `check_per_token_writes`
    /// validates this list against `layout` entries, a register has no `layout` entry, so naming
    /// one here would read as a guard while checking nothing.
    pub fn per_dispatch_writes(&self) -> Vec<&str> {
        std::iter::once("x")
            .chain(self.rope_inputs.iter().map(|(n, _)| n.as_str()))
            .chain(self.mask_widths.iter().map(|m| m.buffer.as_str()))
            .collect()
    }

    /// Each declared RoPE buffer paired with the angle base to fill it from, resolved ONCE at load
    /// so no dispatch path carries a lookup or an unwrap. `bases` is the authority for the two
    /// constants: for a decode artifact that is itself, and for a prefill artifact it is the decode
    /// half of its pair -- prefill may leave them undeclared, and computing its angle rows from
    /// anything but decode's base is what a disagreement would look like on device.
    pub fn rope_writes(&self, bases: &LlmArtifact) -> Result<Vec<(BufLoc, f64)>, EngineError> {
        self.rope_inputs
            .iter()
            .map(|(name, which)| {
                let theta = match which {
                    RopeBase::Global => bases.rope_theta_global,
                    RopeBase::Local => bases.rope_theta_local,
                };
                let theta = theta.ok_or_else(|| {
                    EngineError::Load(format!(
                        "{}: declares input `{name}` but {} has no host_protocol base for {which:?}",
                        self.decode_dir.display(),
                        bases.decode_dir.display()
                    ))
                })?;
                let loc = *self.layout.get(name).ok_or_else(|| {
                    EngineError::Load(format!(
                        "{}: input `{name}` has no layout entry",
                        self.decode_dir.display()
                    ))
                })?;
                Ok((loc, theta))
            })
            .collect()
    }

    /// The prefill/decode agreements a shared arena does not cover: everything that changes what
    /// the KV bytes MEAN rather than where they sit. `self` is the decode half.
    ///
    /// Checked at the point the pair is formed (K007), each naming its own numbers, because none of
    /// these produces an error on the device -- a disagreeing RoPE base or `S` yields a plausible
    /// wrong token and nothing else.
    pub fn check_prefill_pairing(&self, prefill: &LlmArtifact) -> Result<(), EngineError> {
        let mut checks: Vec<(&str, usize, usize)> = vec![
            // `S` is the capacity both halves address, so a disagreement puts prefill's KV rows
            // under decode's head boundaries -- in-arena, past every bounds check.
            ("dims.S", self.max_seq, prefill.max_seq),
            ("dims.head_dim", self.head_dim, prefill.head_dim),
            ("dims.d_model", self.d_model, prefill.d_model),
            ("dims.layers", self.n_layers, prefill.n_layers),
            // The block size, which is what the shared bytes MEAN. It defaults to `S` when the
            // artifact does not declare it, so this also catches the case that cost 2026-09-10: a
            // decode blocked at 128 paired with a prefill silent about blocking, priming the right
            // values at flat addresses. Every generation came back as one token repeated, and
            // nothing between the two halves compared the one number that differed.
            ("dims.kv_block", self.kv_block, prefill.kv_block),
        ];
        // 0 means "not declared, and provably never read" -- see the loader. Only compare two
        // artifacts that both state it.
        if self.kv_heads != 0 && prefill.kv_heads != 0 {
            checks.push(("dims.kv_heads", self.kv_heads, prefill.kv_heads));
        }
        // Optional on the prefill half (it has no lm-head), checked when declared.
        if let (Some(d), Some(p)) = (self.vocab, prefill.vocab) {
            checks.push(("dims.vocab", d, p));
        }
        for (what, d, p) in checks {
            if d != p {
                return Err(EngineError::Load(format!(
                    "prefill/decode disagree on {what}: decode {d}, prefill {p}"
                )));
            }
        }
        // The model constants prefill may inherit rather than declare. Where it DOES declare one,
        // it must agree -- a second copy of a number that must match is only useful if it is
        // compared, and a disagreeing RoPE base produces plausible wrong text and nothing else.
        if prefill.rope_theta_global.is_some() && self.rope_theta_global != prefill.rope_theta_global
            || prefill.rope_theta_local.is_some() && self.rope_theta_local != prefill.rope_theta_local
        {
            return Err(EngineError::Load(format!(
                "prefill/decode disagree on the RoPE base: decode ({:?}, {:?}), prefill ({:?}, {:?})",
                self.rope_theta_global, self.rope_theta_local,
                prefill.rope_theta_global, prefill.rope_theta_local
            )));
        }
        if prefill.embed_scale.is_some() && self.embed_scale != prefill.embed_scale {
            return Err(EngineError::Load(format!(
                "prefill/decode disagree on host_protocol.embed_scale: decode {:?}, prefill {:?}",
                self.embed_scale, prefill.embed_scale
            )));
        }
        // Both halves must ask for the same RoPE tables, whatever they call the buffers. A prefill
        // artifact declaring only a global table for a model with local layers would leave the
        // local ones rotating at whatever its buffer last held.
        let bases = |a: &LlmArtifact| {
            let mut b: Vec<RopeBase> = a.rope_inputs.iter().map(|(_, r)| *r).collect();
            b.sort_unstable_by_key(|r| *r as u8);
            b
        };
        if bases(self) != bases(prefill) {
            return Err(EngineError::Load(format!(
                "prefill/decode declare different RoPE tables: decode {:?}, prefill {:?}",
                self.rope_inputs, prefill.rope_inputs
            )));
        }
        // The final chunk of a prompt is padded to `M`, so a prefill run covers `ceil(n/M)*M`
        // positions. With `S % M == 0` that can never exceed `S` for any prompt the window already
        // admits (`n <= S-1` => `ceil(n/M)*M <= S`), which is what lets the chunk loop skip a bound
        // it could not act on anyway -- declining a long prompt after priming half of it is worse
        // than refusing the pair at load.
        if !self.max_seq.is_multiple_of(prefill.batch) {
            return Err(EngineError::Load(format!(
                "prefill batch M={} does not divide the KV window S={}: the padded final chunk \
                 would write past the end of the cache for prompts near the window",
                prefill.batch, self.max_seq
            )));
        }
        Ok(())
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
            "dims": {"layers": 1, "d_model": x_len / 2, "vocab": 4, "head_dim": head_dim, "S": 8},
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
        assert_eq!(art.embed_scale, Some(EmbedScale::None));
        assert!(!art.kv_off.core);
        assert!(art.sm_mask.unwrap().core);
    }

    // ------------------------------------------------------------------------------------
    // `attn_window` / `window_granule` -- the third scratchpad parameter. Absent on every
    // artifact today (base_meta's default scratchpad block); a dynamic-window artifact adds
    // scratchpad.params.attn_window + scratchpad.window_param + dims.window_granule together.
    // ------------------------------------------------------------------------------------

    #[test]
    fn no_window_fields_yields_none_for_both() {
        let dir = tempfile::tempdir().unwrap();
        let meta = base_meta(8, 4, serde_json::json!({"rope_global": {"type": "input", "offset": 8, "len": 8}}));
        write_meta(dir.path(), &meta);
        let art = LlmArtifact::load(dir.path()).unwrap();
        assert!(art.attn_window.is_none());
        assert_eq!(art.window_granule, None);
    }

    #[test]
    fn all_three_window_fields_parse() {
        let dir = tempfile::tempdir().unwrap();
        let mut meta = base_meta(8, 4, serde_json::json!({"rope_global": {"type": "input", "offset": 8, "len": 8}}));
        meta["scratchpad"]["params"]["attn_window"] =
            serde_json::json!({"byte_offset": 8, "kind": "core"});
        meta["scratchpad"]["window_param"] = serde_json::json!("attn_window");
        meta["dims"]["window_granule"] = serde_json::json!(128);
        write_meta(dir.path(), &meta);
        let art = LlmArtifact::load(dir.path()).unwrap();
        let aw = art.attn_window.expect("window_param declared");
        assert_eq!(aw.byte_offset, 8);
        assert!(aw.core, "attn_window is kind: core");
        assert_eq!(art.window_granule, Some(128));
    }

    // ------------------------------------------------------------------------------------
    // `window_rungs` -- the named control codes this ELF carries besides `main:sequence`.
    // ------------------------------------------------------------------------------------

    /// A dynamic-window meta, which is the only kind a rung set is legal on.
    fn dynwindow_meta() -> serde_json::Value {
        let mut meta = base_meta(8, 4, serde_json::json!({"rope_global": {"type": "input", "offset": 8, "len": 8}}));
        meta["scratchpad"]["params"]["attn_window"] =
            serde_json::json!({"byte_offset": 8, "kind": "core"});
        meta["scratchpad"]["window_param"] = serde_json::json!("attn_window");
        meta["dims"]["window_granule"] = serde_json::json!(128);
        meta
    }

    #[test]
    fn no_rungs_is_the_old_single_window_behaviour() {
        let dir = tempfile::tempdir().unwrap();
        write_meta(dir.path(), &dynwindow_meta());
        let art = LlmArtifact::load(dir.path()).unwrap();
        assert!(art.window_rungs.is_empty(), "a meta with no rungs must declare none");
    }

    #[test]
    fn rungs_parse_and_sort_ascending_by_window() {
        let dir = tempfile::tempdir().unwrap();
        let mut meta = dynwindow_meta();
        // Deliberately out of order: JSON object order is not window order, and the selector
        // walks this list assuming ascending, so the sort is load-bearing rather than cosmetic.
        meta["window_rungs"] = serde_json::json!({"sequence_w4": 4, "sequence_w2": 2});
        write_meta(dir.path(), &meta);
        let art = LlmArtifact::load(dir.path()).unwrap();
        assert_eq!(
            art.window_rungs,
            vec![("sequence_w2".to_string(), 2), ("sequence_w4".to_string(), 4)]
        );
    }

    #[test]
    fn a_rung_wider_than_the_top_window_fails_loud() {
        let dir = tempfile::tempdir().unwrap();
        let mut meta = dynwindow_meta();
        let s = meta["dims"]["S"].as_u64().unwrap();
        meta["window_rungs"] = serde_json::json!({"sequence_wide": s + 1});
        write_meta(dir.path(), &meta);
        let err = LlmArtifact::load(dir.path()).unwrap_err().to_string();
        assert!(err.contains("window_rungs") && err.contains("dims.S"), "{err}");
    }

    #[test]
    fn rungs_without_a_runtime_window_fail_loud() {
        // A rung quantises the SHIM's fill; the CORE still needs its runtime window or every
        // dispatch attends the rung's full width. Accepting this pair would be a plausible wrong
        // answer -- correct tokens near a rung boundary, silently over-attending elsewhere.
        let dir = tempfile::tempdir().unwrap();
        let mut meta = base_meta(8, 4, serde_json::json!({"rope_global": {"type": "input", "offset": 8, "len": 8}}));
        meta["window_rungs"] = serde_json::json!({"sequence_w4": 4});
        write_meta(dir.path(), &meta);
        let err = LlmArtifact::load(dir.path()).unwrap_err().to_string();
        assert!(err.contains("window_rungs") && err.contains("window_param"), "{err}");
    }

    #[test]
    fn window_param_without_granule_fails_loud_rather_than_guessing_a_unit() {
        let dir = tempfile::tempdir().unwrap();
        let mut meta = base_meta(8, 4, serde_json::json!({"rope_global": {"type": "input", "offset": 8, "len": 8}}));
        meta["scratchpad"]["params"]["attn_window"] =
            serde_json::json!({"byte_offset": 8, "kind": "core"});
        meta["scratchpad"]["window_param"] = serde_json::json!("attn_window");
        // dims.window_granule deliberately left undeclared.
        write_meta(dir.path(), &meta);
        let err = LlmArtifact::load(dir.path()).unwrap_err().to_string();
        assert!(err.contains("window_param") && err.contains("window_granule"), "{err}");
    }

    #[test]
    fn granule_without_window_param_fails_loud_rather_than_guessing_an_offset() {
        let dir = tempfile::tempdir().unwrap();
        let mut meta = base_meta(8, 4, serde_json::json!({"rope_global": {"type": "input", "offset": 8, "len": 8}}));
        meta["dims"]["window_granule"] = serde_json::json!(128);
        // scratchpad.window_param deliberately left undeclared.
        write_meta(dir.path(), &meta);
        let err = LlmArtifact::load(dir.path()).unwrap_err().to_string();
        assert!(err.contains("window_param") && err.contains("window_granule"), "{err}");
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

    #[test]
    fn an_artifact_that_does_not_declare_its_window_fails_to_load() {
        // The migration cost of making `dims.S` required, asserted rather than left implicit. An
        // artifact built before the window was recorded loads into an engine that cannot bound
        // `pos`, and the overrun is silent -- so refusing it is the safer failure of the two.
        let dir = tempfile::tempdir().unwrap();
        let mut meta = base_meta(8, 4, serde_json::json!({"rope_global": {"type": "input", "offset": 8, "len": 8}}));
        meta["dims"].as_object_mut().unwrap().remove("S");
        write_meta(dir.path(), &meta);
        let err = LlmArtifact::load(dir.path()).unwrap_err().to_string();
        assert!(err.contains("dims.S"), "must name the missing field: {err}");
    }

    #[test]
    fn the_declared_window_is_what_the_artifact_reports() {
        let dir = tempfile::tempdir().unwrap();
        let mut meta = base_meta(8, 4, serde_json::json!({"rope_global": {"type": "input", "offset": 8, "len": 8}}));
        meta["dims"]["S"] = serde_json::json!(512);
        write_meta(dir.path(), &meta);
        assert_eq!(LlmArtifact::load(dir.path()).unwrap().max_seq, 512);
    }

    #[test]
    fn an_artifact_with_no_dims_kv_block_reads_as_the_flat_pre_blocking_layout() {
        // base_meta() (and every fixture in this file that predates the KV-blocked-layout task)
        // declares neither dims.kv_block nor dims.kv_heads -- exactly what a real artifact built
        // before that task looks like. It must still load, with kv_block defaulting to max_seq.
        let dir = tempfile::tempdir().unwrap();
        let mut meta = base_meta(8, 4, serde_json::json!({"rope_global": {"type": "input", "offset": 8, "len": 8}}));
        meta["dims"]["S"] = serde_json::json!(512);
        assert!(meta["dims"].get("kv_block").is_none());
        write_meta(dir.path(), &meta);
        let art = LlmArtifact::load(dir.path()).unwrap();
        assert_eq!(art.kv_block, 512);
        assert_eq!(art.kv_block, art.max_seq);
    }

    #[test]
    fn a_blocked_artifact_declares_both_kv_block_and_kv_heads() {
        let dir = tempfile::tempdir().unwrap();
        let mut meta = base_meta(8, 4, serde_json::json!({"rope_global": {"type": "input", "offset": 8, "len": 8}}));
        meta["dims"]["S"] = serde_json::json!(4096);
        meta["dims"]["kv_block"] = serde_json::json!(128);
        meta["dims"]["kv_heads"] = serde_json::json!(8);
        write_meta(dir.path(), &meta);
        let art = LlmArtifact::load(dir.path()).unwrap();
        assert_eq!(art.kv_block, 128);
        assert_eq!(art.kv_heads, 8);
    }

    #[test]
    fn a_blocked_artifact_missing_kv_heads_fails_loud_rather_than_guessing() {
        // kv_block < max_seq with kv_heads absent CANNOT default harmlessly (unlike the flat
        // case): the block term is genuinely read once pos crosses one block. Silently defaulting
        // to 0 would compute a wrong kv_off instead of refusing to load.
        let dir = tempfile::tempdir().unwrap();
        let mut meta = base_meta(8, 4, serde_json::json!({"rope_global": {"type": "input", "offset": 8, "len": 8}}));
        meta["dims"]["S"] = serde_json::json!(4096);
        meta["dims"]["kv_block"] = serde_json::json!(128);
        write_meta(dir.path(), &meta);
        let err = LlmArtifact::load(dir.path()).unwrap_err().to_string();
        assert!(err.contains("kv_heads"), "must name the missing field: {err}");
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

    /// A staged production root shadows the dev checkout its `artifacts` symlink points into.
    /// This is the 2026-09-09 outage: a re-pin in the checkout made every INSTALLED artifact fail
    /// to load, because the walk-up canonicalized through the symlink and found the dev pin rather
    /// than the staged lock those artifacts were built against.
    #[test]
    fn staged_install_root_wins_over_the_checkout_its_artifacts_symlink_into() {
        let root = tempfile::tempdir().unwrap();
        let checkout = root.path().join("checkout");
        let staged = root.path().join("staged");
        fs::create_dir_all(checkout.join("artifacts/qwen3/decode")).unwrap();
        fs::create_dir_all(&staged).unwrap();

        // The two trees are pinned DIFFERENTLY -- that disagreement is the whole test.
        fs::write(checkout.join("toolchain.lock"), b"PIN=dev_moved_on\n").unwrap();
        fs::write(staged.join("toolchain.lock"), b"PIN=what_it_was_built_with\n").unwrap();
        let staged_hash = kernel_registry::current_toolchain_hash(&staged).unwrap();
        let dev_hash = kernel_registry::current_toolchain_hash(&checkout).unwrap();
        assert_ne!(staged_hash, dev_hash, "test setup must actually disagree");

        // install.sh's shape: artifacts is a SYMLINK into the checkout, beside a staged lock.
        std::os::unix::fs::symlink(checkout.join("artifacts"), staged.join("artifacts")).unwrap();

        // Stamp the artifact with the STAGED pin -- it was built when that was current.
        let mut meta = base_meta(8, 4, serde_json::json!({"rope_global": {"type": "input", "offset": 8, "len": 8}}));
        meta["toolchain"] = serde_json::json!({ "hash": staged_hash });
        write_meta(&checkout.join("artifacts/qwen3/decode"), &meta);

        // Loaded by its INSTALLED path, it must resolve the staged lock and be Fresh.
        LlmArtifact::load(&staged.join("artifacts/qwen3/decode"))
            .expect("installed artifact must be gated on the staged pin, not the dev checkout's");

        // Same bytes reached through the CHECKOUT path are a dev artifact and still get the dev
        // pin -- the check keeps its teeth where it has them.
        let err = LlmArtifact::load(&checkout.join("artifacts/qwen3/decode"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("toolchain-stale"), "{err}");
        assert!(err.contains(&dev_hash), "{err}");
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

    // ------------------------------------------------------------------------------------
    // The prefill role, and the arena-sharing precondition. The check these exercise is the one
    // thing standing between "two ELFs, one arena, one weight upload" and one graph's activations
    // landing on the other's KV cache -- an offset in IRON is emergent from runlist order, so the
    // two agreeing is a property of how they were generated and nothing in the artifacts asserts it.
    // ------------------------------------------------------------------------------------

    const D_MODEL: usize = 4;
    const HEAD_DIM: usize = 4;

    /// A decode/prefill `meta.json` PAIR whose scratch layouts agree: `W` at [0,16) and the cache
    /// `L0_kc` at [16,32) in BOTH. Every test below is one deliberate mutation away from it.
    ///
    /// `W_head` at [32,48) is declared by the DECODE half only -- the lm-head, which prefill has no
    /// use for (it needs logits at no position). That asymmetry is real and the fixture carries it,
    /// because it is what makes the shared-name comparison insufficient on its own.
    ///
    /// The input arena deliberately does NOT agree: prefill's `x` is `M` rows and its `rope_global`
    /// therefore starts elsewhere. That is legal and must stay legal -- each path rewrites the whole
    /// input arena before its own dispatch -- so a check that demanded agreement there would refuse
    /// every real pair.
    fn pair_metas(m: usize) -> (serde_json::Value, serde_json::Value) {
        let scratch = serde_json::json!({
            "W": {"type": "scratch", "offset": 0, "len": 16},
            "L0_kc": {"type": "scratch", "offset": 16, "len": 16},
        });
        let with_io = |x_len: usize, rope_len: usize, extra: serde_json::Value| {
            let mut lay = scratch.clone();
            lay.as_object_mut().unwrap().insert(
                "x".into(),
                serde_json::json!({"type": "input", "offset": 0, "len": x_len}),
            );
            lay.as_object_mut().unwrap().insert(
                "rope_global".into(),
                serde_json::json!({"type": "input", "offset": x_len, "len": rope_len}),
            );
            for (k, v) in extra.as_object().unwrap() {
                lay.as_object_mut().unwrap().insert(k.clone(), v.clone());
            }
            lay
        };
        let common = |lay: serde_json::Value, input_size: usize| {
            serde_json::json!({
                "kernel_name": "main:sequence",
                "input_size": input_size, "scratch_size": 48,
                "layout": lay,
                "inputs": ["x", "rope_global"],
                "weights": ["W", "L0_kc"], "cache_buffers": ["L0_kc"],
                "scratchpad": {
                    "params": {"kv_off": {"byte_offset": 0, "kind": "addr"},
                               "sm_mask": {"byte_offset": 4, "kind": "core"}},
                    "kv_param": "kv_off", "mask_param": "sm_mask",
                },
                "dims": {"layers": 1, "d_model": D_MODEL, "vocab": 4, "head_dim": HEAD_DIM, "S": 8},
                "host_protocol": {"embed_scale": "none", "rope_theta_global": 1_000_000.0},
            })
        };
        let mut dec = common(with_io(D_MODEL * 2, HEAD_DIM * 2, serde_json::json!({
            "logits": {"type": "output", "offset": 0, "len": 8},
            "W_head": {"type": "scratch", "offset": 32, "len": 16},
        })), D_MODEL * 2 + HEAD_DIM * 2);
        dec["elf"] = serde_json::json!("decode.elf");
        dec["output"] = serde_json::json!("logits");
        dec["output_size"] = serde_json::json!(8);
        dec["weights"] = serde_json::json!(["W", "L0_kc", "W_head"]);

        let mut pre = common(with_io(m * D_MODEL * 2, m * HEAD_DIM * 2, serde_json::json!({})),
                             m * (D_MODEL + HEAD_DIM) * 2);
        pre["elf"] = serde_json::json!("prefill.elf");
        pre["output_size"] = serde_json::json!(0);
        pre["dims"]["M"] = serde_json::json!(m);
        (dec, pre)
    }

    /// `(decode, prefill)` loaded from two temp dirs. The dirs are returned so they outlive the
    /// artifacts' `decode_dir` paths.
    fn load_pair(
        dec: &serde_json::Value,
        pre: &serde_json::Value,
    ) -> (tempfile::TempDir, tempfile::TempDir, LlmArtifact, LlmArtifact) {
        let (d, p) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
        write_meta(d.path(), dec);
        write_meta(p.path(), pre);
        let da = LlmArtifact::load(d.path()).expect("decode half of the pair");
        let pa = LlmArtifact::load_prefill(p.path()).expect("prefill half of the pair");
        (d, p, da, pa)
    }

    #[test]
    fn a_prefill_artifact_carries_its_batch_and_needs_no_output() {
        let (dec, pre) = pair_metas(2);
        let (_d, _p, da, pa) = load_pair(&dec, &pre);
        assert_eq!(pa.role, ArtifactRole::Prefill);
        assert_eq!(pa.batch, 2);
        assert_eq!(pa.output, None, "prefill's product is the KV cache, not a buffer");
        assert!(pa.output_name().is_err(), "asking for it must fail loud, not return a placeholder");
        assert_eq!(da.batch, 1, "the decode graph IS the M=1 instance of the prefill graph");
        assert_eq!(da.output.as_deref(), Some("logits"));
    }

    #[test]
    fn a_decode_artifact_still_requires_its_output() {
        let (mut dec, _) = pair_metas(2);
        dec.as_object_mut().unwrap().remove("output");
        let dir = tempfile::tempdir().unwrap();
        write_meta(dir.path(), &dec);
        let err = LlmArtifact::load(dir.path()).unwrap_err().to_string();
        assert!(err.contains("output"), "{err}");
    }

    #[test]
    fn a_prefill_artifact_without_dims_m_fails_to_load() {
        let (_, mut pre) = pair_metas(2);
        pre["dims"].as_object_mut().unwrap().remove("M");
        let dir = tempfile::tempdir().unwrap();
        write_meta(dir.path(), &pre);
        let err = LlmArtifact::load_prefill(dir.path()).unwrap_err().to_string();
        assert!(err.contains("dims.M"), "must name the missing field: {err}");
    }

    #[test]
    fn a_prefill_input_buffer_sized_for_the_wrong_batch_fails_loud() {
        // The silent failure this closes: `x` sized for M=1 while dims.M says 2 means the host
        // writes 2 rows into a 1-row buffer -- an overrun into `rope_global`, no error, garbage KV.
        let (_, mut pre) = pair_metas(2);
        pre["layout"]["x"] = serde_json::json!({"type": "input", "offset": 0, "len": D_MODEL * 2});
        let dir = tempfile::tempdir().unwrap();
        write_meta(dir.path(), &pre);
        let err = LlmArtifact::load_prefill(dir.path()).unwrap_err().to_string();
        assert!(err.contains("layout[x].len"), "{err}");
        assert!(err.contains("dims.M(2)"), "must name the batch it disagrees with: {err}");
    }

    #[test]
    fn an_agreeing_pair_passes_the_shared_arena_check() {
        let (dec, pre) = pair_metas(2);
        let (_d, _p, da, pa) = load_pair(&dec, &pre);
        da.check_shared_layout_agrees(&pa).expect("the pair was generated to agree");
        // Symmetric: neither side is privileged, so a caller may check in either direction.
        pa.check_shared_layout_agrees(&da).expect("and in the other direction");
    }

    #[test]
    fn a_weight_at_a_different_offset_fails_loud_naming_it() {
        let (dec, mut pre) = pair_metas(2);
        pre["layout"]["W"] = serde_json::json!({"type": "scratch", "offset": 16, "len": 16});
        pre["layout"]["L0_kc"] = serde_json::json!({"type": "scratch", "offset": 0, "len": 16});
        let (_d, _p, da, pa) = load_pair(&dec, &pre);
        let err = da.check_shared_layout_agrees(&pa).unwrap_err().to_string();
        assert!(err.contains("`W`") || err.contains("`L0_kc`"), "must name a divergent buffer: {err}");
        assert!(err.contains("scratch_order"), "must say how to fix it: {err}");
    }

    #[test]
    fn a_weight_of_a_different_length_fails_loud_naming_it() {
        let (dec, mut pre) = pair_metas(2);
        pre["layout"]["W"] = serde_json::json!({"type": "scratch", "offset": 0, "len": 8});
        let (_d, _p, da, pa) = load_pair(&dec, &pre);
        let err = da.check_shared_layout_agrees(&pa).unwrap_err().to_string();
        assert!(err.contains("`W`"), "{err}");
        assert!(err.contains("[0, 16)") && err.contains("[0, 8)"), "must name both extents: {err}");
    }

    #[test]
    fn a_prefill_that_does_not_declare_the_kv_cache_fails_loud_naming_it() {
        let (dec, mut pre) = pair_metas(2);
        pre["layout"].as_object_mut().unwrap().remove("L0_kc");
        pre["weights"] = serde_json::json!(["W"]);
        pre["cache_buffers"] = serde_json::json!([]);
        let (_d, _p, da, pa) = load_pair(&dec, &pre);
        let err = da.check_shared_layout_agrees(&pa).unwrap_err().to_string();
        assert!(err.contains("L0_kc"), "{err}");
        assert!(err.contains("cache_buffers"), "{err}");
    }

    #[test]
    fn a_differently_named_prefill_buffer_landing_on_a_weight_fails_loud_naming_both() {
        // The failure a name-by-name comparison alone cannot see, and the one that actually
        // corrupts: prefill's [M, FF] intermediate is bigger than decode's [FF] one, so a
        // generator that packs by first-appearance rather than a declared scratch_order slides
        // everything after it -- under a NEW name, so no shared name disagrees.
        let (dec, mut pre) = pair_metas(2);
        pre["layout"]["P0_h"] = serde_json::json!({"type": "scratch", "offset": 32, "len": 16});
        let (_d, _p, da, pa) = load_pair(&dec, &pre);
        let err = da.check_shared_layout_agrees(&pa).unwrap_err().to_string();
        assert!(err.contains("P0_h"), "must name the intruder: {err}");
        assert!(err.contains("W_head"), "and what it lands on: {err}");
    }

    #[test]
    fn the_input_arena_is_allowed_to_disagree() {
        // Prefill's `x` is M rows and its `rope_global` therefore starts elsewhere. Both paths
        // rewrite the whole input arena before dispatching, so this is not a conflict -- and a
        // check that flagged it would refuse every real pair.
        let (dec, pre) = pair_metas(4);
        let (_d, _p, da, pa) = load_pair(&dec, &pre);
        assert_ne!(da.loc("rope_global"), pa.loc("rope_global"), "the fixture must actually differ");
        da.check_shared_layout_agrees(&pa).expect("an input-arena difference is not a collision");
    }

    /// The shape `designs/decode_fused/gen_llm_prefill.py` emits, key for key, as of 2026-09-08 --
    /// so that a generator change this loader cannot read shows up here rather than at model load.
    /// It differs from a decode meta in five ways, and each one is a deliberate allowance above:
    /// the RoPE input is `rope` (not `rope_global`), `dims` carries no `vocab`, `host_protocol`
    /// carries prose rather than the two model constants, `mask_param` is `null` (the causal arm
    /// masks with the widths VECTOR of `causal_prefill_meta` below, and the control masks not at
    /// all), and the weight blobs live in the decode artifact's `buffers/` (`weights_from`)
    /// rather than its own.
    fn generator_shaped_prefill_meta(m: usize, mask_param: serde_json::Value) -> serde_json::Value {
        serde_json::json!({
            "spec": "qwen3-0.6b", "elf": "prefill.elf", "kernel_name": "main:sequence",
            "input_size": m * (D_MODEL + HEAD_DIM) * 2, "output_size": m * D_MODEL * 2,
            "scratch_size": 48,
            "layout": {
                "x": {"type": "input", "offset": 0, "len": m * D_MODEL * 2},
                "rope": {"type": "input", "offset": m * D_MODEL * 2, "len": m * HEAD_DIM * 2},
                "xout": {"type": "output", "offset": 0, "len": m * D_MODEL * 2},
                "W": {"type": "scratch", "offset": 0, "len": 16},
                "L0_kc": {"type": "scratch", "offset": 16, "len": 16},
            },
            "inputs": ["x", "rope"], "output": "xout",
            "weights": ["W", "L0_kc"],
            "weights_from": "/somewhere/decode/buffers",
            "cache_buffers": ["L0_kc"],
            "arena_shared": true,
            "scratchpad": {
                "params": {"kv_off": {"byte_offset": 0, "kind": "addr"},
                           "sm_mask": {"byte_offset": 4, "kind": "core"}},
                "kv_param": "kv_off", "mask_param": mask_param,
                "head_dim": HEAD_DIM, "kv_heads": 1,
            },
            "dims": {"layers": 1, "M": m, "S": 8, "d_model": D_MODEL, "ffn": 8,
                     "q_heads": 1, "kv_heads": 1, "head_dim": HEAD_DIM, "q_dim": HEAD_DIM},
            "host_protocol": {
                "batch": m,
                "x": "[M, D] bf16 token-major embeddings for this chunk",
                "rope": "[M, HD] bf16, one row per absolute position base..base+M-1",
                "kv_off": "base * head_dim, element units, addr kind, written raw",
                "sm_mask": "base + M (core kind; the host writes it <<2)",
            },
        })
    }

    /// The generator's shape with `--causal rows`: a third input buffer carrying one int32 per
    /// softmax row, `causal: true`, and no scalar `mask_param` anywhere.
    fn causal_prefill_meta(m: usize) -> serde_json::Value {
        let mut pre = generator_shaped_prefill_meta(m, serde_json::Value::Null);
        let q_heads = pre["dims"]["q_heads"].as_u64().unwrap() as usize;
        let (off, len) = (m * (D_MODEL + HEAD_DIM) * 2, q_heads * m * 4);
        pre["layout"]["sm_widths"] = serde_json::json!({"type": "input", "offset": off, "len": len});
        pre["inputs"] = serde_json::json!(["x", "rope", "sm_widths"]);
        pre["input_size"] = serde_json::json!(off + len);
        pre["causal"] = serde_json::json!(true);
        pre["causal_mode"] = serde_json::json!("rows");
        pre["mask_widths"] = serde_json::json!({
            "buffer": "sm_widths", "dtype": "int32", "rows": q_heads * m, "len": len,
        });
        pre
    }

    fn load_causal_prefill(meta: &serde_json::Value) -> Result<LlmArtifact, EngineError> {
        let dir = tempfile::tempdir().unwrap();
        write_meta(dir.path(), meta);
        LlmArtifact::load_prefill(dir.path())
    }

    #[test]
    fn a_causal_prefill_artifact_resolves_its_widths_buffer() {
        let art = load_causal_prefill(&causal_prefill_meta(2)).expect("causal prefill artifact");
        let mw = art.mask_widths.as_ref().expect("causal: true must resolve a widths buffer");
        assert_eq!(mw.buffer, "sm_widths");
        assert_eq!(mw.heads, 1);
        assert_eq!(mw.rows, 2, "q_heads(1) * M(2)");
        assert_eq!(art.loc("sm_widths").len, mw.rows * 4, "int32, not bf16");
        assert!(art.sm_mask.is_none(), "the rows softmax takes no scalar width");
        // The write list is what `check_per_token_writes` gates on, so a widths buffer left off it
        // would load fine and then be dispatched as an all-zero mask -- every row all -inf.
        assert_eq!(art.per_dispatch_writes(), vec!["x", "rope", "sm_widths"]);
        art.check_per_token_writes(&art.per_dispatch_writes()).expect("every input is written");
    }

    #[test]
    fn a_non_causal_prefill_artifact_has_no_widths_buffer() {
        let art = load_causal_prefill(&generator_shaped_prefill_meta(2, serde_json::Value::Null))
            .expect("the --causal none control still loads");
        assert!(art.mask_widths.is_none());
        assert_eq!(art.per_dispatch_writes(), vec!["x", "rope"]);
    }

    #[test]
    fn causal_without_a_mask_widths_block_fails_loud() {
        let mut meta = causal_prefill_meta(2);
        meta["mask_widths"] = serde_json::Value::Null;
        let e = load_causal_prefill(&meta).expect_err("causal with nothing to write is not drivable");
        assert!(format!("{e}").contains("no `mask_widths` block"), "{e}");
    }

    #[test]
    fn a_widths_buffer_that_is_not_int32_fails_loud() {
        // The exact defect the dtype field exists to catch: AIERuntimeArgSpec defaults to bfloat16
        // and sizes the buffer off it, so a generator that forgot the dtype allocates HALF what the
        // host writes and the tail lands in the next buffer.
        let mut meta = causal_prefill_meta(2);
        meta["mask_widths"]["dtype"] = serde_json::json!("bfloat16");
        let e = load_causal_prefill(&meta).expect_err("only int32 widths are the ones we write");
        assert!(format!("{e}").contains("want \"int32\""), "{e}");
    }

    #[test]
    fn a_widths_buffer_sized_for_the_wrong_shape_fails_loud() {
        let mut meta = causal_prefill_meta(2);
        meta["layout"]["sm_widths"]["len"] = serde_json::json!(4); // one row, not q_heads*M
        let e = load_causal_prefill(&meta).expect_err("a short widths buffer is a truncated mask");
        assert!(format!("{e}").contains("dims.q_heads(1) * dims.M(2) * 4 = 8"), "{e}");
    }

    #[test]
    fn a_widths_buffer_the_artifact_does_not_declare_as_an_input_fails_loud() {
        let mut meta = causal_prefill_meta(2);
        meta["inputs"] = serde_json::json!(["x", "rope"]);
        let e = load_causal_prefill(&meta).expect_err("nothing would place it in the input arena");
        assert!(format!("{e}").contains("not a declared input"), "{e}");
    }

    #[test]
    fn declaring_both_a_scalar_width_and_a_widths_vector_fails_loud() {
        let mut meta = causal_prefill_meta(2);
        meta["scratchpad"]["mask_param"] = serde_json::json!("sm_mask");
        let e = load_causal_prefill(&meta).expect_err("one graph cannot have two masks");
        assert!(format!("{e}").contains("two disagreeing masks"), "{e}");
    }

    #[test]
    fn a_causal_artifact_that_cannot_say_how_many_heads_it_has_fails_loud() {
        let mut meta = causal_prefill_meta(2);
        meta["dims"].as_object_mut().unwrap().remove("q_heads");
        let e = load_causal_prefill(&meta).expect_err("the widths length is q_heads * M");
        assert!(format!("{e}").contains("dims.q_heads"), "{e}");
    }

    #[test]
    fn a_decode_artifact_is_not_read_as_carrying_widths() {
        // `causal` on a decode artifact would be describing its scalar `sm_mask`, a different
        // mechanism. Reading it as this one would demand a buffer decode does not have.
        let dir = tempfile::tempdir().unwrap();
        let mut meta = base_meta(8, 4, serde_json::json!({}));
        meta["causal"] = serde_json::json!(true);
        write_meta(dir.path(), &meta);
        let art = LlmArtifact::load(dir.path()).expect("decode still loads");
        assert!(art.mask_widths.is_none());
        assert!(art.sm_mask.is_some(), "decode masks with the scalar, and still does");
    }

    #[test]
    fn the_generators_own_prefill_meta_loads_and_pairs() {
        let (dec, _) = pair_metas(2);
        let pre = generator_shaped_prefill_meta(2, serde_json::json!("sm_mask"));
        let (_d, _p, da, pa) = load_pair(&dec, &pre);
        assert_eq!(pa.batch, 2);
        assert_eq!(pa.vocab, None, "prefill has no lm-head and declares no vocabulary");
        assert_eq!(pa.embed_scale, None, "and inherits the scale from the decode half");
        assert_eq!(pa.rope_theta_global, None);
        assert_eq!(pa.per_dispatch_writes(), vec!["x", "rope"], "its angle buffer is named `rope`");
        da.check_shared_layout_agrees(&pa).expect("shared scratch agrees");
        da.check_prefill_pairing(&pa).expect("same model, same window");
        // The inherited base is decode's, resolved once at load rather than looked up per dispatch.
        let writes = pa.rope_writes(&da).expect("resolve the angle bases against the decode half");
        assert_eq!(writes.len(), 1);
        assert_eq!(writes[0].0, *pa.loc("rope"));
        assert_eq!(writes[0].1, da.rope_theta_global.unwrap());
    }

    #[test]
    fn a_non_causal_prefill_build_declares_no_mask_param_and_still_loads() {
        // `--causal none` is a real bring-up configuration and its graph has no scalar causal
        // width, so there is nothing for the host to write. Decode has no such freedom.
        let pre = generator_shaped_prefill_meta(2, serde_json::Value::Null);
        let dir = tempfile::tempdir().unwrap();
        write_meta(dir.path(), &pre);
        assert!(LlmArtifact::load_prefill(dir.path()).unwrap().sm_mask.is_none());
    }

    #[test]
    fn a_prefill_declaring_a_contradicting_constant_still_fails_loud() {
        // Inheriting an UNDECLARED constant and accepting a DECLARED-but-different one are not the
        // same allowance, and only the first is one.
        let (dec, _) = pair_metas(2);
        let mut pre = generator_shaped_prefill_meta(2, serde_json::json!("sm_mask"));
        pre["host_protocol"]["rope_theta_global"] = serde_json::json!(10_000.0);
        let (_d, _p, da, pa) = load_pair(&dec, &pre);
        let err = da.check_prefill_pairing(&pa).unwrap_err().to_string();
        assert!(err.contains("RoPE base"), "{err}");
    }

    #[test]
    fn a_prefill_missing_a_rope_table_the_model_needs_fails_loud() {
        // Gemma-3 declares two angle tables; a prefill half that declares one would leave the local
        // layers rotating at whatever its buffer last held -- no error, wrong text.
        let (mut dec, _) = pair_metas(2);
        dec["host_protocol"]["rope_theta_local"] = serde_json::json!(10_000.0);
        dec["inputs"] = serde_json::json!(["x", "rope_global", "rope_local"]);
        dec["layout"]["rope_local"] =
            serde_json::json!({"type": "input", "offset": (D_MODEL + HEAD_DIM) * 2, "len": HEAD_DIM * 2});
        dec["input_size"] = serde_json::json!((D_MODEL + 2 * HEAD_DIM) * 2);
        let pre = generator_shaped_prefill_meta(2, serde_json::json!("sm_mask"));
        let (_d, _p, da, pa) = load_pair(&dec, &pre);
        let err = da.check_prefill_pairing(&pa).unwrap_err().to_string();
        assert!(err.contains("RoPE tables"), "{err}");
    }

    #[test]
    fn an_agreeing_pair_passes_the_model_level_check_too() {
        let (dec, pre) = pair_metas(2);
        let (_d, _p, da, pa) = load_pair(&dec, &pre);
        da.check_prefill_pairing(&pa).expect("same model, same window, M divides S");
    }

    #[test]
    fn a_prefill_built_for_a_different_window_fails_loud() {
        // S is the head stride of the [Hkv, S, HD] cache, so this is not a capacity difference --
        // it puts prefill's rows under decode's head boundaries, in-arena and past every check.
        let (dec, mut pre) = pair_metas(2);
        pre["dims"]["S"] = serde_json::json!(4);
        let (_d, _p, da, pa) = load_pair(&dec, &pre);
        let err = da.check_prefill_pairing(&pa).unwrap_err().to_string();
        assert!(err.contains("dims.S"), "{err}");
        assert!(err.contains('8') && err.contains('4'), "must name both windows: {err}");
    }

    #[test]
    fn a_prefill_rotating_at_a_different_rope_base_fails_loud() {
        let (dec, mut pre) = pair_metas(2);
        pre["host_protocol"]["rope_theta_global"] = serde_json::json!(10_000.0);
        let (_d, _p, da, pa) = load_pair(&dec, &pre);
        let err = da.check_prefill_pairing(&pa).unwrap_err().to_string();
        assert!(err.contains("RoPE base"), "{err}");
    }

    #[test]
    fn a_batch_that_does_not_divide_the_window_is_refused_at_load() {
        // The failure it prevents happens only for prompts NEAR the window, so it would pass every
        // short-prompt test and then write past the last row of the cache in production.
        let (dec, mut pre) = pair_metas(2);
        pre["dims"]["M"] = serde_json::json!(3);
        pre["dims"]["S"] = serde_json::json!(8);
        pre["layout"]["x"] = serde_json::json!({"type": "input", "offset": 0, "len": 3 * D_MODEL * 2});
        pre["layout"]["rope_global"] =
            serde_json::json!({"type": "input", "offset": 3 * D_MODEL * 2, "len": 3 * HEAD_DIM * 2});
        pre["input_size"] = serde_json::json!(3 * (D_MODEL + HEAD_DIM) * 2);
        let (_d, _p, da, pa) = load_pair(&dec, &pre);
        let err = da.check_prefill_pairing(&pa).unwrap_err().to_string();
        assert!(err.contains("M=3") && err.contains("S=8"), "must name both numbers: {err}");
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
