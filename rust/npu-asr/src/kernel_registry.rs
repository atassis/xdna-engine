//! Kernel artifact registry -- the single source of truth for the on-disk kernel
//! artifact naming convention, PLUS a content-hash manifest that makes a wrong/stale
//! artifact at the right path a loud error instead of a silent wrong load.
//!
//! Every compiled kernel ships as an xclbin + an instruction-stream file whose names
//! follow one convention under a build dir: `final_{stem}.xclbin` and `insts_{stem}.txt`,
//! where `stem` is a shape/mode descriptor (e.g. `512x800x3072_32x32x32_8c_silu`).
//!
//! Historically this convention was string-built inline via scattered
//! `format!("final_{stem}.xclbin")` call sites across the engine. Lifting it here gives a
//! data-driven compute graph one seam to query for `(stem) -> {xclbin, insts}` instead of
//! re-deriving the prefixes/extensions everywhere. Behavior-preserving: the produced paths
//! are byte-identical to the old inline `format!` (guarded by the parity test below).
//!
//! # Why `resolve()` alone is not enough (engine-op-manifest-and-dynamic-xclbin)
//!
//! `resolve()` only builds a PATH from a hand-typed `stem`. Nothing checks that the bytes at
//! that path are what the caller thinks they are. Measured on 13 shipped artifacts spanning
//! distinct op families (ctxln/affcast/deint/glu/accadd/resadd/dwconv/silu/lnaffcast/two matmul
//! shapes): every xclbin this project builds exposes the byte-identical `EMBEDDED_METADATA`
//! kernel signature -- name `MLIR_AIE`, instance `MLIRAIE`, args
//! `[opcode, instr, ninstr, bo0, bo1, bo2, bo3, bo4]` -- regardless of shape, tile or variant.
//! `xrt::xclbin`/`xrt::kernel` (see `npu-xrt/shim/xrt_shim.cpp:63-74`, `xb.get_kernels().front()`)
//! carry NO op-identity signal for this project's build convention, so a load can never fail (or
//! even warn) just because the wrong shape's xclbin sits at the expected path -- see the module
//! doctrine on lockfile-green-is-not-provenance: the resolved PATH is not evidence about what the
//! artifact under it actually IS. The only ground truth left is the artifact's own bytes.
//!
//! [`generate_manifest`] hashes every `final_*.xclbin` (+ its `insts_*.txt`, when present) under a
//! dir and writes `kernel_manifest.json` there. [`resolve_checked`] re-hashes the same files at
//! load time and refuses (loud [`ManifestError`]) if they no longer match what was recorded --
//! whether the artifact was overwritten, replaced from a different kernels_dir, or was never
//! recorded at all. The manifest is a build artifact, not a hand-maintained list: every entry's
//! hash comes from reading the real file, so it cannot assert anything the directory doesn't
//! currently contain (regenerate it -> it reflects reality again).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

/// The resolved on-disk artifacts for one kernel `stem` under a build dir.
#[derive(Debug)]
pub struct KernelArtifacts {
    pub xclbin: PathBuf,
    pub insts: PathBuf,
}

/// The xclbin path for a kernel identified by `stem` under `dir`: `dir/final_{stem}.xclbin`.
pub fn xclbin_path(dir: &Path, stem: &str) -> PathBuf {
    dir.join(format!("final_{stem}.xclbin"))
}

/// The instruction-stream path for a kernel `stem` under `dir`: `dir/insts_{stem}.txt`.
pub fn insts_path(dir: &Path, stem: &str) -> PathBuf {
    dir.join(format!("insts_{stem}.txt"))
}

/// Resolve both artifacts for one `stem` under `dir`. Does NOT check that anything exists or
/// that the bytes there are what `stem` implies -- see [`resolve_checked`] for that.
pub fn resolve(dir: &Path, stem: &str) -> KernelArtifacts {
    KernelArtifacts {
        xclbin: xclbin_path(dir, stem),
        insts: insts_path(dir, stem),
    }
}

// ---------------------------------------------------------------------------------------------
// The op manifest: content-hash-pinned, filename-token-annotated, generated FROM the artifacts.
// ---------------------------------------------------------------------------------------------

/// Best-effort tokens parsed out of a `stem` by the naming grammar the filenames already use:
/// optional `{name}_` prefix, then either `{M}x{K}x{N}_{mt}x{kt}x{nt}_{cols}c[_{variant}]` (the
/// matmul family) or a bare `{a}x{b}[...][_{variant}]` shape (the elementwise/LN family). Fields
/// that don't match are `None` -- this is DESCRIPTIVE metadata for humans/tooling, not a
/// verification input: nothing here is checked against the artifact's content (see module docs --
/// the xclbin container carries no op-identity signal in this project's convention). Content-hash
/// pinning ([`ManifestEntry`]) is the actual verification; this is what the task calls "the shape,
/// tile, column count and variant its FILENAME encodes" -- an inventory field, not a gate.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct StemTokens {
    pub raw: String,
    pub name: Option<String>,
    /// M x K x N (matmul family) or the 2-axis shape (elementwise/LN family).
    pub shape: Option<Vec<u64>>,
    /// mt x kt x nt tile, only present alongside a 3-axis `shape`.
    pub tile: Option<(u64, u64, u64)>,
    /// column count parsed from a trailing `{n}c` token (e.g. `8c` -> 8).
    pub cols: Option<u64>,
    /// leftover trailing tokens joined with `_` (e.g. `modalsilu`, `s050`, `p4cW`).
    pub variant: Option<String>,
}

fn parse_shape_token(tok: &str) -> Option<Vec<u64>> {
    let parts: Vec<&str> = tok.split('x').collect();
    if parts.len() < 2 || parts.len() > 3 {
        return None;
    }
    parts.iter().map(|p| p.parse::<u64>().ok()).collect()
}

fn parse_cols_token(tok: &str) -> Option<u64> {
    let digits = tok.strip_suffix('c')?;
    if digits.is_empty() {
        return None;
    }
    digits.parse::<u64>().ok()
}

/// Parse a `stem` into [`StemTokens`]. Never fails -- an unrecognized token just falls through to
/// `variant`, and a stem that matches none of the grammar comes back with everything `None` but
/// `raw` (so `raw` alone can always round-trip back to the original stem for a lookup).
pub fn parse_stem_tokens(stem: &str) -> StemTokens {
    let parts: Vec<&str> = stem.split('_').collect();
    let mut i = 0;
    let mut name_parts = Vec::new();
    // Leading non-shape-shaped tokens are the op name (e.g. "ctxln", "mha_decode", "dwconv_silu_t").
    while i < parts.len() && parse_shape_token(parts[i]).is_none() {
        name_parts.push(parts[i]);
        i += 1;
    }
    let name = if name_parts.is_empty() { None } else { Some(name_parts.join("_")) };

    let shape = parts.get(i).and_then(|p| parse_shape_token(p));
    if shape.is_some() {
        i += 1;
    }

    // A tile (mt x kt x nt) only follows a 3-axis shape, by this codebase's own grammar.
    let tile = if shape.as_ref().is_some_and(|s| s.len() == 3) {
        parts.get(i).and_then(|p| parse_shape_token(p)).and_then(|t| {
            if t.len() == 3 {
                Some((t[0], t[1], t[2]))
            } else {
                None
            }
        })
    } else {
        None
    };
    if tile.is_some() {
        i += 1;
    }

    let cols = parts.get(i).and_then(|p| parse_cols_token(p));
    if cols.is_some() {
        i += 1;
    }

    let variant = if i < parts.len() { Some(parts[i..].join("_")) } else { None };

    StemTokens { raw: stem.to_string(), name, shape, tile, cols, variant }
}

/// One manifest entry: the content-hash identity of one stem's on-disk artifacts, as they were
/// when the manifest was (re)generated -- see [`generate_manifest`].
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ManifestEntry {
    pub xclbin_sha256: String,
    pub xclbin_bytes: u64,
    /// `None` when no `insts_{stem}.txt` sits next to the xclbin (some stems don't have one).
    pub insts_sha256: Option<String>,
    pub insts_bytes: Option<u64>,
    pub tokens: StemTokens,
}

/// `dir/kernel_manifest.json` -- one manifest per artifact dir, matching `resolve()`'s
/// directory-scoped convention (so regenerating one model's kernels doesn't touch another's).
pub const MANIFEST_FILE: &str = "kernel_manifest.json";

pub fn manifest_path(dir: &Path) -> PathBuf {
    dir.join(MANIFEST_FILE)
}

pub type Manifest = BTreeMap<String, ManifestEntry>;

fn sha256_hex(path: &Path) -> std::io::Result<(String, u64)> {
    let mut f = std::fs::File::open(path)?;
    let mut h = Sha256::new();
    let bytes = std::io::copy(&mut f, &mut h)?;
    let digest = h.finalize();
    let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
    Ok((hex, bytes))
}

/// Regenerate the manifest for every `final_*.xclbin` found directly under `dir` (non-recursive,
/// matching `resolve()`'s directory scoping). Every entry's hash is computed by reading the real
/// file right now -- this is a build/inventory step meant to be re-run after kernels change, not a
/// hand-maintained list; a manifest that didn't come from reading the actual bytes would just be a
/// sixth instance of the declared-but-not-connected bug this task exists to close.
pub fn generate_manifest(dir: &Path) -> std::io::Result<Manifest> {
    let mut manifest = Manifest::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let Some(fname) = path.file_name().and_then(|s| s.to_str()) else { continue };
        let Some(stem) = fname.strip_prefix("final_").and_then(|s| s.strip_suffix(".xclbin")) else {
            continue;
        };
        let (xclbin_sha256, xclbin_bytes) = sha256_hex(&path)?;
        let insts = insts_path(dir, stem);
        let (insts_sha256, insts_bytes) = if insts.exists() {
            let (h, b) = sha256_hex(&insts)?;
            (Some(h), Some(b))
        } else {
            (None, None)
        };
        manifest.insert(
            stem.to_string(),
            ManifestEntry {
                xclbin_sha256,
                xclbin_bytes,
                insts_sha256,
                insts_bytes,
                tokens: parse_stem_tokens(stem),
            },
        );
    }
    Ok(manifest)
}

pub fn write_manifest(dir: &Path, manifest: &Manifest) -> std::io::Result<()> {
    let json = serde_json::to_string_pretty(manifest)?;
    std::fs::write(manifest_path(dir), json)
}

/// Errors from [`resolve_checked`] -- each variant names exactly what disagreed, so a mismatch is
/// diagnosable from the message alone (path, and both hashes where relevant), not just "load
/// failed somewhere downstream".
#[derive(Debug)]
pub enum ManifestError {
    Io(PathBuf, String),
    /// `dir/kernel_manifest.json` doesn't exist -- nothing has ever recorded what SHOULD be here.
    MissingManifest(PathBuf),
    /// The manifest exists but has no entry for this stem -- an artifact was added (or a stem
    /// renamed) without regenerating the manifest.
    UnknownStem { dir: PathBuf, stem: String },
    XclbinHashMismatch { path: PathBuf, expected: String, actual: String },
    InstsHashMismatch { path: PathBuf, expected: String, actual: String },
}

impl std::fmt::Display for ManifestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ManifestError::Io(p, e) => write!(f, "manifest io error at {}: {e}", p.display()),
            ManifestError::MissingManifest(dir) => write!(
                f,
                "no {MANIFEST_FILE} in {} -- run gen_kernel_manifest on this dir before resolve_checked",
                dir.display()
            ),
            ManifestError::UnknownStem { dir, stem } => write!(
                f,
                "stem '{stem}' has no entry in {}/{MANIFEST_FILE} -- artifact added (or the manifest \
                 is stale); regenerate before trusting it",
                dir.display()
            ),
            ManifestError::XclbinHashMismatch { path, expected, actual } => write!(
                f,
                "xclbin at {} does NOT match its manifest record (expected sha256={expected}, \
                 on-disk sha256={actual}) -- the file at this path is not the artifact the manifest \
                 last verified; do not load it as-is",
                path.display()
            ),
            ManifestError::InstsHashMismatch { path, expected, actual } => write!(
                f,
                "insts at {} does NOT match its manifest record (expected sha256={expected}, \
                 on-disk sha256={actual})",
                path.display()
            ),
        }
    }
}

impl std::error::Error for ManifestError {}

pub fn load_manifest(dir: &Path) -> Result<Manifest, ManifestError> {
    let path = manifest_path(dir);
    if !path.exists() {
        return Err(ManifestError::MissingManifest(dir.to_path_buf()));
    }
    let bytes = std::fs::read(&path).map_err(|e| ManifestError::Io(path.clone(), e.to_string()))?;
    serde_json::from_slice(&bytes).map_err(|e| ManifestError::Io(path.clone(), e.to_string()))
}

/// Like [`resolve`], but verifies the artifacts at `dir` for `stem` still hash to what
/// `dir/kernel_manifest.json` recorded when it was generated. Existing call sites keep using
/// `resolve()` unchanged; a caller opts into this to turn "wrong artifact silently loaded" into a
/// `Result::Err` naming the exact path and both hashes, before any XRT/device call happens.
pub fn resolve_checked(dir: &Path, stem: &str) -> Result<KernelArtifacts, ManifestError> {
    let manifest = load_manifest(dir)?;
    let entry = manifest
        .get(stem)
        .ok_or_else(|| ManifestError::UnknownStem { dir: dir.to_path_buf(), stem: stem.to_string() })?;
    let art = resolve(dir, stem);

    let (actual_xclbin, _) =
        sha256_hex(&art.xclbin).map_err(|e| ManifestError::Io(art.xclbin.clone(), e.to_string()))?;
    if actual_xclbin != entry.xclbin_sha256 {
        return Err(ManifestError::XclbinHashMismatch {
            path: art.xclbin.clone(),
            expected: entry.xclbin_sha256.clone(),
            actual: actual_xclbin,
        });
    }

    if let Some(expected_insts) = &entry.insts_sha256 {
        let (actual_insts, _) =
            sha256_hex(&art.insts).map_err(|e| ManifestError::Io(art.insts.clone(), e.to_string()))?;
        if &actual_insts != expected_insts {
            return Err(ManifestError::InstsHashMismatch {
                path: art.insts.clone(),
                expected: expected_insts.clone(),
                actual: actual_insts,
            });
        }
    }

    Ok(art)
}

// ---------------------------------------------------------------------------------------------
// Toolchain-pin freshness: catches a re-pin that wiped a kernel build dir and was never rebuilt.
// ---------------------------------------------------------------------------------------------
//
// `resolve_checked` above proves an artifact is INTERNALLY consistent (the bytes on disk match
// what the manifest recorded), but a manifest generated before a `toolchain.lock` bump is
// internally consistent and STALE at the same time -- content-hashing the artifact against its
// own manifest can never see that the pin underneath it moved. `scripts/kernel_sandbox.sh`'s
// `ensure_fresh_sandbox` already writes a `.toolchain-stamp` (12 hex chars of
// `sha256(toolchain.lock)`) into a build dir on every purge-or-keep decision, but until now
// nothing ever read it back: the file existed only to drive the NEXT build's purge, not to
// answer "is what's here still built against the CURRENT pin" before the engine loads from it.
// That gap is exactly how a re-pin went silent for 5 days: `ensure_fresh_sandbox` purged the
// dir on the pin change, the rebuild that should have followed never completed, and nothing
// downstream compared the (now-stale-or-missing) stamp against `toolchain.lock` again.

/// Filename `scripts/kernel_sandbox.sh::ensure_fresh_sandbox` writes at build time.
pub const TOOLCHAIN_STAMP_FILE: &str = ".toolchain-stamp";

pub fn toolchain_stamp_path(dir: &Path) -> PathBuf {
    dir.join(TOOLCHAIN_STAMP_FILE)
}

/// `sha256(toolchain.lock)` truncated to 12 hex chars -- byte-for-byte the same derivation as
/// `kernel_sandbox.sh`'s `cur="$(sha256sum "$repo/toolchain.lock" | cut -c1-12)"`. Kept here
/// rather than re-derived at each call site so the truncation length has exactly one definition.
pub fn current_toolchain_hash(repo_root: &Path) -> std::io::Result<String> {
    let (hex, _) = sha256_hex(&repo_root.join("toolchain.lock"))?;
    Ok(hex[..12].to_string())
}

/// Errors from [`check_toolchain_freshness`] -- each variant names exactly what is wrong so the
/// message is actionable without a second lookup.
#[derive(Debug)]
pub enum FreshnessError {
    /// The dir does not exist at all -- a re-pin purge (or a fresh checkout) with no build since.
    MissingDir(PathBuf),
    /// The dir exists but was never stamped -- built before this check existed, or by a path that
    /// does not call `ensure_fresh_sandbox`. Freshness is unknown, and unknown must fail, not pass.
    MissingStamp(PathBuf),
    /// The stamp does not match the CURRENT `toolchain.lock` hash: the dir was built against a
    /// different pin and nothing has rebuilt it since. This is the exact re-pin-wipe shape --
    /// worse than a missing dir when the stale build happens to still load.
    StampMismatch { dir: PathBuf, stamp: String, current: String },
    /// The stamp matches, but the dir holds no `final_*.xclbin` -- the purge ran, the rebuild did
    /// not (or died before producing anything), so a fresh stamp sits over an empty dir.
    NoArtifacts(PathBuf),
    Io(PathBuf, String),
}

impl std::fmt::Display for FreshnessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FreshnessError::MissingDir(dir) => {
                write!(f, "kernel build dir {} does not exist", dir.display())
            }
            FreshnessError::MissingStamp(dir) => write!(
                f,
                "{} has no {TOOLCHAIN_STAMP_FILE} -- freshness against toolchain.lock is unknown",
                dir.display()
            ),
            FreshnessError::StampMismatch { dir, stamp, current } => write!(
                f,
                "{} was built against toolchain.lock={stamp}, but toolchain.lock is now {current} \
                 -- it was re-pinned and this dir was never rebuilt",
                dir.display()
            ),
            FreshnessError::NoArtifacts(dir) => write!(
                f,
                "{} is stamped current but contains no final_*.xclbin -- the rebuild after the \
                 last purge did not finish",
                dir.display()
            ),
            FreshnessError::Io(p, e) => write!(f, "io error at {}: {e}", p.display()),
        }
    }
}

impl std::error::Error for FreshnessError {}

/// Verify `dir` was built against the toolchain pin currently in `toolchain.lock` (under
/// `repo_root`) and actually holds at least one xclbin. Pure filesystem -- no device, no XRT --
/// so it is safe to run before anything touches `/dev/accel/*`, and cheap enough to run on every
/// `npu serve` startup.
pub fn check_toolchain_freshness(dir: &Path, repo_root: &Path) -> Result<(), FreshnessError> {
    if !dir.is_dir() {
        return Err(FreshnessError::MissingDir(dir.to_path_buf()));
    }
    let stamp_path = toolchain_stamp_path(dir);
    let stamp = match std::fs::read_to_string(&stamp_path) {
        Ok(s) => s.trim().to_string(),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err(FreshnessError::MissingStamp(dir.to_path_buf()));
        }
        Err(e) => return Err(FreshnessError::Io(stamp_path, e.to_string())),
    };
    let current = current_toolchain_hash(repo_root)
        .map_err(|e| FreshnessError::Io(repo_root.join("toolchain.lock"), e.to_string()))?;
    if stamp != current {
        return Err(FreshnessError::StampMismatch { dir: dir.to_path_buf(), stamp, current });
    }
    let has_xclbin = std::fs::read_dir(dir)
        .map_err(|e| FreshnessError::Io(dir.to_path_buf(), e.to_string()))?
        .filter_map(|e| e.ok())
        .any(|e| e.file_name().to_str().is_some_and(|s| s.starts_with("final_") && s.ends_with(".xclbin")));
    if !has_xclbin {
        return Err(FreshnessError::NoArtifacts(dir.to_path_buf()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // String-level parity: the registry must reproduce the exact paths the pre-refactor
    // inline `format!("final_{stem}.xclbin")` / `format!("insts_{stem}.txt")` produced, for
    // the real stems across the current model set. Pins the naming convention so a future
    // edit to a prefix/extension breaks this test instead of silently missing every xclbin.
    #[test]
    fn registry_string_parity_matches_inline_format() {
        let dir = Path::new("/wa");
        // Representative real stems drawn from the routed call sites:
        //   ctx_decode GEMV, engines ffn (silu/bias), ctx_ln, conv_npu mstat band, mha_decode.
        let stems = [
            "512x800x3072_32x32x32_8c_silu",
            "512x1536x768_32x32x32_8c_bias",
            "ctxln_512x768",
            "mstat_512x768x1500_16x32x32_8c",
            "mha_decode_448",
        ];
        for stem in stems {
            // parity vs the historical inline expressions
            assert_eq!(xclbin_path(dir, stem), dir.join(format!("final_{stem}.xclbin")));
            assert_eq!(insts_path(dir, stem), dir.join(format!("insts_{stem}.txt")));
            let a = resolve(dir, stem);
            assert_eq!(a.xclbin, dir.join(format!("final_{stem}.xclbin")));
            assert_eq!(a.insts, dir.join(format!("insts_{stem}.txt")));
        }
        // absolute-literal pins (guard the convention constants themselves)
        assert_eq!(
            xclbin_path(dir, "512x768x3072_32x32x32_8c").to_str().unwrap(),
            "/wa/final_512x768x3072_32x32x32_8c.xclbin"
        );
        assert_eq!(
            insts_path(dir, "mha_decode_448").to_str().unwrap(),
            "/wa/insts_mha_decode_448.txt"
        );
    }

    // ------------------------------------------------------------------------------------
    // parse_stem_tokens: real stems drawn from the literal-site inventory (npu-asr/npu-parakeet
    // call sites), not invented ones. Matmul family (shape+tile+cols[+variant]), bare-shape
    // family (name_AxB[_variant]), and single-param family (name_N) all appear in production.
    // ------------------------------------------------------------------------------------

    #[test]
    fn parse_stem_tokens_matmul_family() {
        let t = parse_stem_tokens("512x1024x4096_64x32x128_8c_modalsilubf16outpanel1024");
        assert_eq!(t.shape, Some(vec![512, 1024, 4096]));
        assert_eq!(t.tile, Some((64, 32, 128)));
        assert_eq!(t.cols, Some(8));
        assert_eq!(t.variant.as_deref(), Some("modalsilubf16outpanel1024"));
        assert_eq!(t.name, None); // no leading op-name token in this family
    }

    #[test]
    fn parse_stem_tokens_named_matmul() {
        let t = parse_stem_tokens("mstat_512x768x1500_16x32x32_8c");
        assert_eq!(t.name.as_deref(), Some("mstat"));
        assert_eq!(t.shape, Some(vec![512, 768, 1500]));
        assert_eq!(t.tile, Some((16, 32, 32)));
        assert_eq!(t.cols, Some(8));
        assert_eq!(t.variant, None);
    }

    #[test]
    fn parse_stem_tokens_bare_shape_family() {
        let t = parse_stem_tokens("ctxln_512x1024");
        assert_eq!(t.name.as_deref(), Some("ctxln"));
        assert_eq!(t.shape, Some(vec![512, 1024]));
        assert_eq!(t.tile, None); // no tile follows a 2-axis shape
        assert_eq!(t.cols, None);
        assert_eq!(t.variant, None);
    }

    #[test]
    fn parse_stem_tokens_bare_shape_with_variant() {
        let t = parse_stem_tokens("resadd_512x1024_s050");
        assert_eq!(t.name.as_deref(), Some("resadd"));
        assert_eq!(t.shape, Some(vec![512, 1024]));
        assert_eq!(t.variant.as_deref(), Some("s050"));
    }

    #[test]
    fn parse_stem_tokens_single_param_family() {
        let t = parse_stem_tokens("mha_decode_448");
        // None of "mha"/"decode"/"448" parses as a shape token (parse_shape_token requires an
        // 'x'-joined 2-3-axis numeric token), so the whole stem folds into `name` and shape/tile/
        // cols/variant all stay None. This IS the honest behavior for a stem outside the grammar
        // the matmul/bare-shape families use -- parse_stem_tokens never fails, it just can't
        // recover axes that aren't there.
        assert_eq!(t.name.as_deref(), Some("mha_decode_448"));
        assert_eq!(t.shape, None);
        assert_eq!(t.variant, None);
        // raw always round-trips, regardless of how much of the grammar matched.
        assert_eq!(t.raw, "mha_decode_448");
    }

    // ------------------------------------------------------------------------------------
    // generate_manifest / resolve_checked: filesystem-only, no device. Proves both directions --
    // a manifest generated from real bytes lets resolve_checked SUCCEED, and any drift after
    // generation (the artifact at the path is no longer what was hashed) makes resolve_checked
    // FAIL loud, naming the path and both hashes.
    // ------------------------------------------------------------------------------------

    fn write_fake_kernel(dir: &Path, stem: &str, xclbin_bytes: &[u8], insts_bytes: Option<&[u8]>) {
        std::fs::write(xclbin_path(dir, stem), xclbin_bytes).unwrap();
        if let Some(b) = insts_bytes {
            std::fs::write(insts_path(dir, stem), b).unwrap();
        }
    }

    #[test]
    fn generate_manifest_hashes_real_bytes_on_disk() {
        let td = tempfile::tempdir().unwrap();
        write_fake_kernel(td.path(), "512x1024x4096_64x32x128_8c_modalsilu", b"fake-xclbin-bytes-A", Some(b"fake-insts-A"));
        write_fake_kernel(td.path(), "ctxln_512x1024", b"fake-xclbin-bytes-B", None);
        let manifest = generate_manifest(td.path()).unwrap();
        assert_eq!(manifest.len(), 2);
        let a = &manifest["512x1024x4096_64x32x128_8c_modalsilu"];
        let (expect_xclbin, _) = sha256_hex(&xclbin_path(td.path(), "512x1024x4096_64x32x128_8c_modalsilu")).unwrap();
        assert_eq!(a.xclbin_sha256, expect_xclbin);
        assert!(a.insts_sha256.is_some());
        assert_eq!(a.tokens.shape, Some(vec![512, 1024, 4096]));
        let b = &manifest["ctxln_512x1024"];
        assert_eq!(b.insts_sha256, None); // no insts file was written for this stem
    }

    #[test]
    fn resolve_checked_succeeds_when_artifact_matches_manifest() {
        let td = tempfile::tempdir().unwrap();
        let stem = "affcast_512x1024";
        write_fake_kernel(td.path(), stem, b"real-affcast-bytes", Some(b"real-affcast-insts"));
        let manifest = generate_manifest(td.path()).unwrap();
        write_manifest(td.path(), &manifest).unwrap();

        let art = resolve_checked(td.path(), stem).expect("matching artifact must resolve");
        assert_eq!(art.xclbin, xclbin_path(td.path(), stem));
    }

    #[test]
    fn resolve_checked_fails_loud_when_xclbin_drifts_after_manifest() {
        let td = tempfile::tempdir().unwrap();
        let stem = "512x1024x4096_64x32x128_8c_modalsilu";
        write_fake_kernel(td.path(), stem, b"the-correct-modalsilu-xclbin", Some(b"insts"));
        let manifest = generate_manifest(td.path()).unwrap();
        write_manifest(td.path(), &manifest).unwrap();

        // Simulate the kernels_dir-class bug: some other build (wrong source, stale copy, wrong
        // instance) overwrites the file at the SAME path the manifest verified.
        std::fs::write(xclbin_path(td.path(), stem), b"a-DIFFERENT-xclbin-ended-up-here").unwrap();

        let err = resolve_checked(td.path(), stem).expect_err("drifted xclbin must not resolve");
        match &err {
            ManifestError::XclbinHashMismatch { path, .. } => {
                assert_eq!(path, &xclbin_path(td.path(), stem));
            }
            other => panic!("expected XclbinHashMismatch, got {other}"),
        }
        // The message is specific enough to act on: names the path and doesn't just say "error".
        let msg = err.to_string();
        assert!(msg.contains("does NOT match"), "message: {msg}");
        assert!(msg.contains(stem), "message should name the artifact path: {msg}");
    }

    #[test]
    fn resolve_checked_fails_loud_when_insts_drifts() {
        let td = tempfile::tempdir().unwrap();
        let stem = "glu_512x1024";
        write_fake_kernel(td.path(), stem, b"glu-xclbin", Some(b"glu-insts-v1"));
        let manifest = generate_manifest(td.path()).unwrap();
        write_manifest(td.path(), &manifest).unwrap();

        std::fs::write(insts_path(td.path(), stem), b"glu-insts-v2-DIFFERENT").unwrap();

        let err = resolve_checked(td.path(), stem).expect_err("drifted insts must not resolve");
        assert!(matches!(err, ManifestError::InstsHashMismatch { .. }), "got {err:?}");
    }

    #[test]
    fn resolve_checked_fails_loud_on_missing_manifest() {
        let td = tempfile::tempdir().unwrap();
        let stem = "deint_512x4096";
        write_fake_kernel(td.path(), stem, b"deint-bytes", None);
        // No generate_manifest/write_manifest call: the dir has a real xclbin but was never
        // inventoried -- exactly the "declared but not connected" shape this task targets.
        let err = resolve_checked(td.path(), stem).expect_err("no manifest must not resolve");
        assert!(matches!(err, ManifestError::MissingManifest(_)), "got {err:?}");
    }

    #[test]
    fn resolve_checked_fails_loud_on_unknown_stem() {
        let td = tempfile::tempdir().unwrap();
        write_fake_kernel(td.path(), "known_stem_512x1024", b"known-bytes", None);
        let manifest = generate_manifest(td.path()).unwrap();
        write_manifest(td.path(), &manifest).unwrap();
        // A different stem's xclbin appears later without regenerating the manifest.
        write_fake_kernel(td.path(), "unrecorded_stem_512x2048", b"unrecorded-bytes", None);

        let err = resolve_checked(td.path(), "unrecorded_stem_512x2048").expect_err("must not resolve");
        assert!(matches!(err, ManifestError::UnknownStem { .. }), "got {err:?}");
    }

    #[test]
    fn manifest_round_trips_through_json() {
        let td = tempfile::tempdir().unwrap();
        let stem = "512x4096x1024_64x32x128_8c_modalid";
        write_fake_kernel(td.path(), stem, b"modalid-bytes", Some(b"modalid-insts"));
        let manifest = generate_manifest(td.path()).unwrap();
        write_manifest(td.path(), &manifest).unwrap();

        let loaded = load_manifest(td.path()).unwrap();
        assert_eq!(loaded, manifest);
    }

    // ------------------------------------------------------------------------------------
    // check_toolchain_freshness: constructs the exact re-pin-wipe shape (defect #3,
    // artifact-preflight-and-fail-loud) and proves the check fails loud on it, alongside its
    // sibling failure shapes and the one case that must pass.
    // ------------------------------------------------------------------------------------

    /// A fake repo root with a `toolchain.lock` whose hash is independently known, so tests do
    /// not depend on `sha256sum` being on PATH or on this repo's real lock file.
    fn fake_repo(lock_bytes: &[u8]) -> (tempfile::TempDir, String) {
        let repo = tempfile::tempdir().unwrap();
        std::fs::write(repo.path().join("toolchain.lock"), lock_bytes).unwrap();
        let hash = current_toolchain_hash(repo.path()).unwrap();
        (repo, hash)
    }

    #[test]
    fn freshness_passes_when_stamp_matches_and_xclbin_present() {
        let (repo, hash) = fake_repo(b"pin-a");
        let dir = repo.path().join("build");
        std::fs::create_dir(&dir).unwrap();
        std::fs::write(toolchain_stamp_path(&dir), &hash).unwrap();
        write_fake_kernel(&dir, "512x1024x4096_64x32x128_8c", b"xclbin-bytes", None);

        check_toolchain_freshness(&dir, repo.path()).expect("fresh dir must pass");
    }

    /// THE defect this check exists for: `ensure_fresh_sandbox` stamped the dir for an OLD pin
    /// (its xclbins are real, current, self-consistent artifacts -- `resolve_checked` would pass
    /// them), `toolchain.lock` was re-pinned since, and nothing rebuilt. Old resolve_checked-style
    /// checks see nothing wrong; this one must fail loud and name both hashes.
    #[test]
    fn freshness_fails_loud_when_repin_leaves_a_stale_stamp() {
        let (repo, _old_hash_unused) = fake_repo(b"pin-a");
        let dir = repo.path().join("build");
        std::fs::create_dir(&dir).unwrap();
        let stale_stamp = current_toolchain_hash(repo.path()).unwrap();
        std::fs::write(toolchain_stamp_path(&dir), &stale_stamp).unwrap();
        write_fake_kernel(&dir, "512x1024x4096_64x32x128_8c", b"xclbin-bytes", None);

        // The re-pin: toolchain.lock changes, nobody rebuilds `dir`.
        std::fs::write(repo.path().join("toolchain.lock"), b"pin-b").unwrap();
        let new_hash = current_toolchain_hash(repo.path()).unwrap();
        assert_ne!(stale_stamp, new_hash, "test setup must actually change the hash");

        let err = check_toolchain_freshness(&dir, repo.path()).expect_err("stale stamp must fail");
        match &err {
            FreshnessError::StampMismatch { dir: d, stamp, current } => {
                assert_eq!(d, &dir);
                assert_eq!(stamp, &stale_stamp);
                assert_eq!(current, &new_hash);
            }
            other => panic!("expected StampMismatch, got {other}"),
        }
        let msg = err.to_string();
        assert!(msg.contains(&stale_stamp) && msg.contains(&new_hash), "message: {msg}");
        assert!(msg.contains("re-pinned"), "message should say why: {msg}");
    }

    /// The other half of the same 5-day outage: `ensure_fresh_sandbox` purges the dir on the pin
    /// change and nothing rebuilds it AT ALL -- not stale, just gone.
    #[test]
    fn freshness_fails_loud_on_missing_dir() {
        let (repo, _hash) = fake_repo(b"pin-a");
        let dir = repo.path().join("build"); // never created
        let err = check_toolchain_freshness(&dir, repo.path()).expect_err("missing dir must fail");
        assert!(matches!(err, FreshnessError::MissingDir(d) if d == dir));
    }

    #[test]
    fn freshness_fails_loud_when_never_stamped() {
        let (repo, _hash) = fake_repo(b"pin-a");
        let dir = repo.path().join("build");
        std::fs::create_dir(&dir).unwrap();
        write_fake_kernel(&dir, "512x1024x4096_64x32x128_8c", b"xclbin-bytes", None);
        // No .toolchain-stamp written: a dir built by a path that predates this check, or that
        // never calls ensure_fresh_sandbox. Freshness is unknown, which must not read as "ok".
        let err = check_toolchain_freshness(&dir, repo.path()).expect_err("unstamped dir must fail");
        assert!(matches!(err, FreshnessError::MissingStamp(d) if d == dir));
    }

    /// The purge ran (stamp is current) but the rebuild after it died before producing anything --
    /// a fresh stamp over an empty dir, which a stamp-only check would wrongly call healthy.
    #[test]
    fn freshness_fails_loud_when_stamped_but_empty() {
        let (repo, hash) = fake_repo(b"pin-a");
        let dir = repo.path().join("build");
        std::fs::create_dir(&dir).unwrap();
        std::fs::write(toolchain_stamp_path(&dir), &hash).unwrap();
        // No xclbin written.
        let err = check_toolchain_freshness(&dir, repo.path()).expect_err("empty dir must fail");
        assert!(matches!(err, FreshnessError::NoArtifacts(d) if d == dir));
    }
}
