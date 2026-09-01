//! S2 TTS codec: load one exported streamed design (`final.xclbin` + `insts.bin` + `meta.json`)
//! and dispatch it, following the build-once/dispatch-many split `npu-whisper/src/mha_npu.rs`
//! establishes -- [`S2Design::open`] loads the xclbin, uploads insts and allocates every BO ONCE;
//! [`S2Design::dispatch`] only uploads/runs/downloads.
//!
//! The design ABI (Python side: `bricklib._build_streamed`, `route_b_kernels/bricks/_verify/
//! bricklib.py`) is `kern(in_tile, resident, out_tile)` per streamed tile when the design has a
//! resident operand, else `kern(in_tile, out_tile)`; `resident_len == 0` in `meta.json` IS the
//! "no resident" signal, matching that Python convention exactly. On the XRT side this is IRON's
//! fixed `(opcode, instr, count, data...)` kernel signature, so the data BOs land at arg indices
//! 3.. in the same order: `in_tiles[, resident], out`.
//!
//! [`S2Artifacts`] is the top-level `manifest.json` over every design a codec exports (directory +
//! role per design, plus the toolchain pin the designs were built against).

use std::path::{Path, PathBuf};
use std::rc::Rc;

use sha2::{Digest, Sha256};

use npu_xrt::{Bo, Device, Kernel, FLAG_CACHEABLE, FLAG_HOST_ONLY};

/// Fixed filenames inside one design's directory (the exporter's convention -- see the task
/// context this crate was built against: "a directory containing `final.xclbin`, `insts.bin`, and
/// a `meta.json`").
pub const XCLBIN_FILE: &str = "final.xclbin";
pub const INSTS_FILE: &str = "insts.bin";
pub const META_FILE: &str = "meta.json";
/// Top-level file listing every design a codec exports (directory + role) plus the toolchain pin.
pub const MANIFEST_FILE: &str = "manifest.json";

/// IRON's fixed dispatch opcode. Every design in this codebase (matmul8/dwconv6/mha7/bd8, see
/// `npu-xrt::Kernel::run_*`) hardcodes `opcode=3` regardless of shape -- it selects "run the
/// EMBEDDED_METADATA sequence", not a per-op variant, so the streamed codec designs use the same
/// constant rather than a new one.
const OPCODE: u32 = 3;
/// All codec-decoder buffers are f32 (per task scope); one element is 4 bytes.
const F32_BYTES: usize = 4;

#[derive(Debug)]
pub enum S2Error {
    Io(PathBuf, std::io::Error),
    Json(PathBuf, serde_json::Error),
    /// An `npu-xrt` call failed; the message already names the XRT-level op.
    Xrt(String),
    /// A shape/dtype/ABI mismatch between `meta.json`/`manifest.json` and what was actually asked
    /// for or found on disk.
    Shape(String),
    Toolchain(String),
}

impl std::fmt::Display for S2Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            S2Error::Io(p, e) => write!(f, "{}: {e}", p.display()),
            S2Error::Json(p, e) => write!(f, "{}: {e}", p.display()),
            S2Error::Xrt(e) => write!(f, "{e}"),
            S2Error::Shape(e) => write!(f, "{e}"),
            S2Error::Toolchain(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for S2Error {}

pub type Result<T> = std::result::Result<T, S2Error>;

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    let bytes = std::fs::read(path).map_err(|e| S2Error::Io(path.to_path_buf(), e))?;
    serde_json::from_slice(&bytes).map_err(|e| S2Error::Json(path.to_path_buf(), e))
}

fn sha256_hex(path: &Path) -> Result<String> {
    let mut f = std::fs::File::open(path).map_err(|e| S2Error::Io(path.to_path_buf(), e))?;
    let mut h = Sha256::new();
    std::io::copy(&mut f, &mut h).map_err(|e| S2Error::Io(path.to_path_buf(), e))?;
    Ok(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

fn is_f32(dtype: &str) -> bool {
    matches!(dtype, "f32" | "float32" | "F32" | "FLOAT32")
}

fn f32_bytes(v: &[f32]) -> &[u8] {
    // SAFETY: f32 has no padding/alignment requirement stricter than u8; the slice is read-only.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}
fn f32_bytes_mut(v: &mut [f32]) -> &mut [u8] {
    // SAFETY: same as f32_bytes; the resulting bytes are only ever overwritten in place.
    unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, std::mem::size_of_val(v)) }
}

/// Cross-check a `meta.json`-declared byte size against the one computed from the element counts.
/// `meta.json` may omit the field (`None`); a present-but-wrong value is a hanging-number bug
/// (schema drift, a stale export) and must fail loud rather than load a mis-sized BO.
fn check_bytes(label: &str, declared: Option<usize>, computed: usize, meta_path: &Path) -> Result<()> {
    match declared {
        Some(d) if d != computed => Err(S2Error::Shape(format!(
            "{}: {label}_bytes={d} in meta.json disagrees with n_tiles*{label}_elems*4={computed}",
            meta_path.display()
        ))),
        _ => Ok(()),
    }
}

/// One design's `meta.json`. Required fields are the ones [`S2Design::open`]/[`dispatch`] cannot
/// operate without; everything else is `Option`/defaulted so a field this schema doesn't (yet)
/// carry -- or spells differently -- degrades to "not cross-checked", not a parse failure.
/// Unrecognized fields (`ci_chunk`, `window`, ...) land in `extra` rather than being dropped, since
/// this crate was written ahead of the exporter and the exact key set is unconfirmed (see the
/// crate's delivery report for the list of fields guessed here).
#[derive(Debug, Clone, serde::Deserialize)]
pub struct S2Meta {
    pub symbol: String,
    pub role: String,
    pub n_tiles: usize,
    pub in_tile: usize,
    pub out_numel: usize,
    /// Elements in the resident operand; `0` means the design has none (mirrors
    /// `bricklib._build_streamed`'s own `has_resident = resident_len > 0`).
    #[serde(default)]
    pub resident_len: usize,
    pub in_dtype: String,
    #[serde(default)]
    pub resident_dtype: Option<String>,
    pub out_dtype: String,
    #[serde(default)]
    pub resident_depth: Option<usize>,
    #[serde(default)]
    pub compile_flags: Vec<String>,
    /// Instruction word count, if `meta.json` carries one -- cross-checked against
    /// `insts.bin`'s own length (`bytes/4`), which is what `open()` actually trusts.
    #[serde(default)]
    pub n_instr: Option<usize>,
    #[serde(default)]
    pub in_bytes: Option<usize>,
    #[serde(default)]
    pub resident_bytes: Option<usize>,
    #[serde(default)]
    pub out_bytes: Option<usize>,
    #[serde(default)]
    pub xclbin_sha256: Option<String>,
    #[serde(default)]
    pub shim_sha256: Option<String>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// One design loaded from an exported artifact directory (`final.xclbin` + `insts.bin` +
/// `meta.json`). BOs are allocated ONCE in [`open`](Self::open); [`dispatch`](Self::dispatch)
/// only uploads/runs/downloads -- see `npu-whisper/src/mha_npu.rs`, the template this mirrors.
pub struct S2Design {
    kern: Rc<Kernel>,
    instr: Bo,
    n_instr: usize,
    bo_in: Bo,
    bo_resident: Option<Bo>,
    bo_out: Bo,
    in_elems: usize,
    out_elems: usize,
    pub meta: S2Meta,
}

impl S2Design {
    /// Load `dir/{final.xclbin,insts.bin,meta.json}` onto an already-open Device (single-tenant;
    /// reuse the handle) and allocate every BO. No dispatch happens here.
    pub fn open(dev: &Rc<Device>, dir: &Path) -> Result<Self> {
        let meta_path = dir.join(META_FILE);
        let meta: S2Meta = read_json(&meta_path)?;

        if !is_f32(&meta.in_dtype) || !is_f32(&meta.out_dtype) {
            return Err(S2Error::Shape(format!(
                "{}: in_dtype={} out_dtype={} -- npu-s2 only handles the f32 codec-decoder ABI",
                meta_path.display(), meta.in_dtype, meta.out_dtype
            )));
        }
        if meta.resident_len > 0 {
            if let Some(rdt) = &meta.resident_dtype {
                if !is_f32(rdt) {
                    return Err(S2Error::Shape(format!(
                        "{}: resident_dtype={rdt} -- npu-s2 only handles the f32 codec-decoder ABI",
                        meta_path.display()
                    )));
                }
            }
        }

        let xclbin_path = dir.join(XCLBIN_FILE);
        if let Some(expected) = &meta.xclbin_sha256 {
            let actual = sha256_hex(&xclbin_path)?;
            if &actual != expected {
                return Err(S2Error::Shape(format!(
                    "{}: xclbin sha256 mismatch -- meta.json says {expected}, on-disk is {actual} \
                     (the artifact at this path is not the one meta.json describes)",
                    xclbin_path.display()
                )));
            }
        }
        let xclbin_str = xclbin_path
            .to_str()
            .ok_or_else(|| S2Error::Shape(format!("{} is not valid UTF-8", xclbin_path.display())))?;
        let kern = dev.load_kernel(xclbin_str, None).map_err(S2Error::Xrt)?;

        let insts_path = dir.join(INSTS_FILE);
        let ibytes = std::fs::read(&insts_path).map_err(|e| S2Error::Io(insts_path.clone(), e))?;
        let n_instr = ibytes.len() / 4;
        if let Some(declared) = meta.n_instr {
            if declared != n_instr {
                return Err(S2Error::Shape(format!(
                    "{}: n_instr={declared} in meta.json but insts.bin is {n_instr} words \
                     ({} bytes)",
                    meta_path.display(), ibytes.len()
                )));
            }
        }

        let g = |arg: i32| kern.group_id(arg).map_err(S2Error::Xrt);

        let instr = dev.alloc_bo(&kern, ibytes.len(), FLAG_CACHEABLE, g(1)?).map_err(S2Error::Xrt)?;
        instr.write_bytes(&ibytes).map_err(S2Error::Xrt)?;
        instr.sync_to_device().map_err(S2Error::Xrt)?;

        let in_elems = meta.n_tiles.checked_mul(meta.in_tile).ok_or_else(|| {
            S2Error::Shape(format!("{}: n_tiles*in_tile overflows usize", meta_path.display()))
        })?;
        let out_elems = meta.n_tiles.checked_mul(meta.out_numel).ok_or_else(|| {
            S2Error::Shape(format!("{}: n_tiles*out_numel overflows usize", meta_path.display()))
        })?;
        let in_bytes = in_elems * F32_BYTES;
        let out_bytes = out_elems * F32_BYTES;
        check_bytes("in", meta.in_bytes, in_bytes, &meta_path)?;
        check_bytes("out", meta.out_bytes, out_bytes, &meta_path)?;

        // Data BOs land at arg indices 3.. in ABI order (in[, resident], out) -- same convention
        // every other design in this codebase uses (`run_mha`'s Q@3 K@4 V@5 O@6, `run_dwconv6`'s
        // X@3 W@4 Y@5); group_id(1)=instr and arg 2 (count) is a scalar with no BO/group_id.
        let bo_in = dev.alloc_bo(&kern, in_bytes, FLAG_HOST_ONLY, g(3)?).map_err(S2Error::Xrt)?;
        let (bo_resident, bo_out) = if meta.resident_len > 0 {
            let resident_bytes = meta.resident_len * F32_BYTES;
            check_bytes("resident", meta.resident_bytes, resident_bytes, &meta_path)?;
            let bo_r =
                dev.alloc_bo(&kern, resident_bytes, FLAG_HOST_ONLY, g(4)?).map_err(S2Error::Xrt)?;
            let bo_o = dev.alloc_bo(&kern, out_bytes, FLAG_HOST_ONLY, g(5)?).map_err(S2Error::Xrt)?;
            (Some(bo_r), bo_o)
        } else {
            let bo_o = dev.alloc_bo(&kern, out_bytes, FLAG_HOST_ONLY, g(4)?).map_err(S2Error::Xrt)?;
            (None, bo_o)
        };

        if !npu_xrt::quiet() {
            eprintln!(
                "[S2Design] loaded {} (role={} symbol={} n_tiles={} in_tile={} out_numel={} \
                 resident_len={}, {n_instr} instr)",
                xclbin_path.display(), meta.role, meta.symbol, meta.n_tiles, meta.in_tile,
                meta.out_numel, meta.resident_len
            );
        }

        Ok(S2Design { kern, instr, n_instr, bo_in, bo_resident, bo_out, in_elems, out_elems, meta })
    }

    /// `in_tiles`: `n_tiles*in_tile` f32 elements. `resident`: `Some(resident_len elements)` iff
    /// this design has a resident operand, else `None` -- mismatching either way is an `Err`, not
    /// a silent zero-fill. Returns `n_tiles*out_numel` f32 elements. Upload/run/download only; no
    /// allocation, no loading (both happened once in [`open`](Self::open)).
    pub fn dispatch(&self, in_tiles: &[f32], resident: Option<&[f32]>) -> Result<Vec<f32>> {
        if in_tiles.len() != self.in_elems {
            return Err(S2Error::Shape(format!(
                "in_tiles: got {} elements, design ({}) wants {}",
                in_tiles.len(), self.meta.role, self.in_elems
            )));
        }
        match (resident, &self.bo_resident) {
            (Some(r), Some(bo)) => {
                if r.len() != self.meta.resident_len {
                    return Err(S2Error::Shape(format!(
                        "resident: got {} elements, design ({}) wants {}",
                        r.len(), self.meta.role, self.meta.resident_len
                    )));
                }
                bo.write_bytes(f32_bytes(r)).map_err(S2Error::Xrt)?;
                bo.sync_to_device().map_err(S2Error::Xrt)?;
            }
            (None, Some(_)) => {
                return Err(S2Error::Shape(format!(
                    "design ({}) requires a resident operand ({} elements), none given",
                    self.meta.role, self.meta.resident_len
                )))
            }
            (Some(r), None) => {
                return Err(S2Error::Shape(format!(
                    "design ({}) has no resident operand, {} elements given",
                    self.meta.role, r.len()
                )))
            }
            (None, None) => {}
        }

        self.bo_in.write_bytes(f32_bytes(in_tiles)).map_err(S2Error::Xrt)?;
        self.bo_in.sync_to_device().map_err(S2Error::Xrt)?;

        let data: Vec<&Bo> = match &self.bo_resident {
            Some(r) => vec![&self.bo_in, r, &self.bo_out],
            None => vec![&self.bo_in, &self.bo_out],
        };
        self.kern.run_kernel(OPCODE, &self.instr, self.n_instr, &data).map_err(S2Error::Xrt)?;

        self.bo_out.sync_from_device().map_err(S2Error::Xrt)?;
        let mut out = vec![0f32; self.out_elems];
        self.bo_out.read_bytes(f32_bytes_mut(&mut out)).map_err(S2Error::Xrt)?;
        Ok(out)
    }
}

/// One `manifest.json` row: a design's identity (`name`) and `role` (`head`/`stage1_up`/.../
/// `tail`), plus the directory (relative to the manifest's own directory) holding its
/// `final.xclbin`/`insts.bin`/`meta.json`.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct S2ManifestDesign {
    pub name: String,
    pub role: String,
    pub dir: String,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct S2ManifestFile {
    toolchain_pin: String,
    designs: Vec<S2ManifestDesign>,
}

/// The top-level `manifest.json` over every design one codec exports.
pub struct S2Artifacts {
    root: PathBuf,
    toolchain_pin: String,
    designs: Vec<S2ManifestDesign>,
}

impl S2Artifacts {
    /// Read `dir/manifest.json`. Does not touch the device or open any design.
    pub fn open(dir: &Path) -> Result<Self> {
        let path = dir.join(MANIFEST_FILE);
        let m: S2ManifestFile = read_json(&path)?;
        Ok(S2Artifacts { root: dir.to_path_buf(), toolchain_pin: m.toolchain_pin, designs: m.designs })
    }

    pub fn toolchain_pin(&self) -> &str {
        &self.toolchain_pin
    }

    pub fn designs(&self) -> &[S2ManifestDesign] {
        &self.designs
    }

    pub fn by_role(&self, role: &str) -> Option<&S2ManifestDesign> {
        self.designs.iter().find(|d| d.role == role)
    }

    pub fn by_name(&self, name: &str) -> Option<&S2ManifestDesign> {
        self.designs.iter().find(|d| d.name == name)
    }

    pub fn design_dir(&self, d: &S2ManifestDesign) -> PathBuf {
        self.root.join(&d.dir)
    }

    /// Compare `manifest.json`'s `toolchain_pin` against `toolchain.lock`'s
    /// `MLIR_AIE_FORK_COMMIT` -- the fork commit is this project's single source of truth for
    /// "one exact AIE toolchain" (`toolchain.lock`'s own header comment). Accepts either string
    /// being a prefix of the other, so a short-hash pin still matches a full-hash lock (or vice
    /// versa).
    pub fn validate_toolchain(&self, toolchain_lock: &Path) -> Result<()> {
        let pinned = parse_mlir_aie_fork_commit(toolchain_lock)?;
        let (a, b) = (self.toolchain_pin.trim(), pinned.trim());
        if a != b && !a.starts_with(b) && !b.starts_with(a) {
            return Err(S2Error::Toolchain(format!(
                "manifest.json toolchain_pin={a} != {}'s MLIR_AIE_FORK_COMMIT={b}",
                toolchain_lock.display()
            )));
        }
        Ok(())
    }

    pub fn open_by_role(&self, dev: &Rc<Device>, role: &str) -> Result<S2Design> {
        let d = self.by_role(role).ok_or_else(|| {
            S2Error::Shape(format!("no design with role '{role}' in {}", self.root.join(MANIFEST_FILE).display()))
        })?;
        S2Design::open(dev, &self.design_dir(d))
    }

    pub fn open_by_name(&self, dev: &Rc<Device>, name: &str) -> Result<S2Design> {
        let d = self.by_name(name).ok_or_else(|| {
            S2Error::Shape(format!("no design named '{name}' in {}", self.root.join(MANIFEST_FILE).display()))
        })?;
        S2Design::open(dev, &self.design_dir(d))
    }
}

/// Parse `MLIR_AIE_FORK_COMMIT=<sha>` out of `toolchain.lock` (`KEY=value   # comment` lines --
/// see `xdna-engine/toolchain.lock`'s own header). This is the fork instance's commit, the pin
/// this whole toolchain doctrine treats as authoritative ("FORK-ONLY, never the wheel").
fn parse_mlir_aie_fork_commit(toolchain_lock: &Path) -> Result<String> {
    let text = std::fs::read_to_string(toolchain_lock)
        .map_err(|e| S2Error::Io(toolchain_lock.to_path_buf(), e))?;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("MLIR_AIE_FORK_COMMIT=") {
            let commit = rest.split_whitespace().next().unwrap_or("");
            if commit.is_empty() {
                break;
            }
            return Ok(commit.to_string());
        }
    }
    Err(S2Error::Toolchain(format!("no MLIR_AIE_FORK_COMMIT= line in {}", toolchain_lock.display())))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(dir: &Path, name: &str, contents: &str) {
        std::fs::write(dir.join(name), contents).unwrap();
    }

    #[test]
    fn meta_parses_minimal_required_fields() {
        let td = tempfile::tempdir().unwrap();
        write(
            td.path(), META_FILE,
            r#"{"symbol":"stage1_up","role":"stage1_up","n_tiles":4,"in_tile":128,
                "out_numel":256,"resident_len":512,"in_dtype":"float32","out_dtype":"float32"}"#,
        );
        let meta: S2Meta = read_json(&td.path().join(META_FILE)).unwrap();
        assert_eq!(meta.n_tiles, 4);
        assert_eq!(meta.resident_len, 512);
        assert_eq!(meta.resident_depth, None);
        assert!(meta.compile_flags.is_empty());
    }

    #[test]
    fn meta_captures_unknown_fields_in_extra() {
        let td = tempfile::tempdir().unwrap();
        write(
            td.path(), META_FILE,
            r#"{"symbol":"head","role":"head","n_tiles":1,"in_tile":8,"out_numel":8,
                "in_dtype":"f32","out_dtype":"f32","ci_chunk":128,"window":{"t":64}}"#,
        );
        let meta: S2Meta = read_json(&td.path().join(META_FILE)).unwrap();
        assert_eq!(meta.extra.get("ci_chunk").unwrap(), 128);
        assert_eq!(meta.resident_len, 0);
    }

    #[test]
    fn check_bytes_rejects_mismatched_declared_size() {
        let p = Path::new("/meta.json");
        assert!(check_bytes("in", Some(999), 1000, p).is_err());
        assert!(check_bytes("in", Some(1000), 1000, p).is_ok());
        assert!(check_bytes("in", None, 1000, p).is_ok());
    }

    #[test]
    fn manifest_by_role_and_name() {
        let td = tempfile::tempdir().unwrap();
        write(
            td.path(), MANIFEST_FILE,
            r#"{"toolchain_pin":"deadbeef","designs":[
                {"name":"head","role":"head","dir":"head"},
                {"name":"stage1_up","role":"stage1_up","dir":"stage1"}
            ]}"#,
        );
        let art = S2Artifacts::open(td.path()).unwrap();
        assert_eq!(art.toolchain_pin(), "deadbeef");
        assert_eq!(art.by_role("stage1_up").unwrap().dir, "stage1");
        assert!(art.by_name("tail").is_none());
        assert_eq!(art.design_dir(art.by_name("head").unwrap()), td.path().join("head"));
    }

    #[test]
    fn toolchain_pin_matches_full_hash_lock() {
        let td = tempfile::tempdir().unwrap();
        write(td.path(), MANIFEST_FILE, r#"{"toolchain_pin":"035528f71cf1dd0","designs":[]}"#);
        let lock = td.path().join("toolchain.lock");
        std::fs::write(&lock, "MLIR_AIE_FORK_COMMIT=035528f71cf1dd067ff01562fda99088678ca2b0   # comment\n")
            .unwrap();
        let art = S2Artifacts::open(td.path()).unwrap();
        art.validate_toolchain(&lock).expect("short pin should prefix-match the full lock commit");
    }

    #[test]
    fn toolchain_pin_mismatch_is_an_error() {
        let td = tempfile::tempdir().unwrap();
        write(td.path(), MANIFEST_FILE, r#"{"toolchain_pin":"0000000000000000","designs":[]}"#);
        let lock = td.path().join("toolchain.lock");
        std::fs::write(&lock, "MLIR_AIE_FORK_COMMIT=035528f71cf1dd067ff01562fda99088678ca2b0\n").unwrap();
        let art = S2Artifacts::open(td.path()).unwrap();
        assert!(art.validate_toolchain(&lock).is_err());
    }
}
