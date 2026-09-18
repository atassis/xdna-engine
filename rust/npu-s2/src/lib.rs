//! S2 TTS codec: load one exported streamed design (`final.xclbin` + `insts.bin` + `meta.json`)
//! and dispatch it, following the build-once/dispatch-many split `npu-whisper/src/mha_npu.rs`
//! establishes -- [`S2Design::open`] loads the xclbin, uploads insts and allocates every BO ONCE;
//! [`S2Design::dispatch`] only uploads/runs/downloads.
//!
//! The design ABI (Python side: `bricklib._build_streamed`, `aie_kernels/_test/
//! bricklib.py`) is `kern(in_tile, resident, out_tile)` per streamed tile when the design has a
//! resident operand, else `kern(in_tile, out_tile)`; `resident_len == 0` in `meta.json` IS the
//! "no resident" signal, matching that Python convention exactly. On the XRT side this is IRON's
//! fixed `(opcode, instr, count, data...)` kernel signature, so the data BOs land at arg indices
//! 3.. in the same order: `in_tiles[, resident], out`.
//!
//! [`S2Artifacts`] is the top-level `manifest.json` over every design a codec exports (directory +
//! role per design, plus the toolchain pin the designs were built against).

use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::rc::{Rc, Weak};

use sha2::{Digest, Sha256};

use npu_xrt::{Bo, Device, Kernel, FLAG_CACHEABLE, FLAG_HOST_ONLY};

pub mod ar;
pub mod chain;
pub mod gguf;
pub mod weights;
pub mod window;

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

/// `meta.json`'s `dtypes` object. `resident` is null when the design has no resident operand.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct S2Dtypes {
    #[serde(rename = "in")]
    pub in_: String,
    pub out: String,
    #[serde(default)]
    pub resident: Option<String>,
}

/// `meta.json`'s `buffer_bytes` object: the exporter's own byte sizing for each operand.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct S2BufferBytes {
    #[serde(rename = "in")]
    pub in_: usize,
    pub out: usize,
    #[serde(default)]
    pub resident: usize,
}

/// `meta.json`'s `op_params` object -- the exporter's copy of window_driver.py's own `op_meta`
/// (see `export_codec_artifacts.py`'s `export_one`). This is what lets the driver derive every
/// windowing parameter (step, context, chunk width) from the artifact itself rather than
/// replicating `stage_shapes.py`'s chunk-size policy in Rust.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum S2OpParams {
    Snake {
        tag: String,
        #[serde(rename = "C")]
        c: usize,
        #[serde(rename = "T")]
        t: usize,
    },
    Conv {
        tag: String,
        k: usize,
        dilation: usize,
        ctx: usize,
        c_in: usize,
        c_in_total: usize,
        c_out: usize,
        has_add: bool,
        #[serde(rename = "T")]
        t: usize,
        step: usize,
        #[serde(default)]
        #[allow(dead_code)]
        vector: bool,
    },
    ConvTranspose {
        tag: String,
        k: usize,
        stride: usize,
        ctx: usize,
        c_in: usize,
        c_in_total: usize,
        c_out: usize,
        t: usize,
        step: usize,
    },
}

/// One design's `meta.json`. Required fields are the ones [`S2Design::open`]/[`dispatch`] cannot
/// operate without; everything else is `Option`/defaulted so a field this schema doesn't (yet)
/// carry -- or spells differently -- degrades to "not cross-checked", not a parse failure.
/// Unrecognized fields (`ci_chunk`, `window`, ...) land in `extra` rather than being dropped, since
/// this crate was written ahead of the exporter and the exact key set is not pinned.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct S2Meta {
    pub symbol: String,
    /// The exporter's `op` ("head_conv", "stage1_res0_1x1", "tail_conv", ...). Named `op` there,
    /// alongside a coarser `group` ("head"/"stage1"/.../"tail"); both are carried.
    pub op: String,
    #[serde(default)]
    pub group: Option<String>,
    #[serde(default)]
    pub stage: Option<u32>,
    pub n_tiles: usize,
    pub in_tile: usize,
    pub out_numel: usize,
    /// Elements in the resident operand; `0` means the design has none (mirrors
    /// `bricklib._build_streamed`'s own `has_resident = resident_len > 0`).
    #[serde(default)]
    pub resident_len: usize,
    /// Nested in the exporter's JSON as `dtypes: {in, out, resident}`, not flat per-operand keys.
    pub dtypes: S2Dtypes,
    #[serde(default)]
    pub resident_depth: Option<usize>,
    #[serde(default)]
    pub compile_flags: Vec<String>,
    /// Instruction word count, if `meta.json` carries one -- cross-checked against
    /// `insts.bin`'s own length (`bytes/4`), which is what `open()` actually trusts.
    /// Instruction word count. The exporter emits both `insts_words` and `insts_bytes`; `open()`
    /// trusts `insts.bin`'s own length and only cross-checks against this.
    #[serde(default)]
    pub insts_words: Option<usize>,
    #[serde(default)]
    pub insts_bytes: Option<usize>,
    /// Nested as `buffer_bytes: {in, out, resident}`, not flat per-operand keys.
    #[serde(default)]
    pub buffer_bytes: Option<S2BufferBytes>,
    #[serde(default)]
    pub xclbin_sha256: Option<String>,
    #[serde(default)]
    pub shim_sha256: Option<String>,
    /// See [`S2OpParams`]. Absent on a hand-written test fixture; every real exported design
    /// carries it.
    #[serde(default)]
    pub op_params: Option<S2OpParams>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// One design loaded from an exported artifact directory (`final.xclbin` + `insts.bin` +
/// `meta.json`). BOs are allocated ONCE in [`open`](Self::open); [`dispatch`](Self::dispatch)
/// only uploads/runs/downloads -- see `npu-whisper/src/mha_npu.rs`, the template this mirrors.
/// Cumulative dispatches and a periodic progress line, gated by `NPU_S2_HEARTBEAT=<n>` (dispatches
/// per line; unset = silent). The chain issues tens of thousands of dispatches over minutes with no
/// output between op boundaries, which makes "slow" and "hung waiting on a syncobj" look identical
/// from outside -- a single `/proc/<pid>/wchan` sample cannot tell them apart, since a healthy run
/// sits in the same DRM wait almost all of its life.
/// Times a re-read disagreed with the read before it under `NPU_S2_RESYNC` -- i.e. observed
/// instances of the host-only-BO read race, counted rather than inferred.
pub static RESYNC_HITS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

static DISPATCHES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn heartbeat(op: &str) {
    // Count FIRST: the dispatch total is the denominator for the stale-read rate, so it must not
    // depend on whether the progress log happens to be enabled.
    let n = DISPATCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
    note_op(op, false);
    static EVERY: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let every = *EVERY.get_or_init(|| {
        std::env::var("NPU_S2_HEARTBEAT").ok().and_then(|v| v.parse().ok()).unwrap_or(0)
    });
    if every == 0 {
        return;
    }
    static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    let start = *START.get_or_init(std::time::Instant::now);
    if n % every == 0 {
        let secs = start.elapsed().as_secs_f64();
        eprintln!("[S2Design] {n} dispatches, {secs:.1}s ({:.1}/s), in {op}", n as f64 / secs);
    }
}

/// Observed stale reads: times a re-read under `NPU_S2_RESYNC` disagreed with the read before it.
/// Measures the host-only-BO coherency race directly -- a host-only BO's mapped pages can serve a
/// dispatch's output from cache, so the first read after a run can return the previous run's
/// bytes. Counted per dispatch rather than inferred from a rate.
pub fn resync_hits() -> usize {
    RESYNC_HITS.load(std::sync::atomic::Ordering::Relaxed)
}

thread_local! {
    /// Per-op (dispatches, stale reads). The aggregate rate is an average over a heterogeneous mix
    /// -- designs differ by three orders of magnitude in per-dispatch output bytes -- so only the
    /// per-op split can say whether the race concentrates where the buffers are small.
    static PER_OP: std::cell::RefCell<std::collections::BTreeMap<String, (usize, usize)>> =
        std::cell::RefCell::new(std::collections::BTreeMap::new());
}

fn note_op(op: &str, stale: bool) {
    PER_OP.with(|m| {
        let mut m = m.borrow_mut();
        let e = m.entry(op.to_string()).or_insert((0, 0));
        if stale {
            e.1 += 1;
        } else {
            e.0 += 1;
        }
    });
}

/// `(op, dispatches, stale_reads)` sorted by stale count, for a caller that wants to report where
/// the race actually fires rather than an aggregate.
pub fn per_op_stats() -> Vec<(String, usize, usize)> {
    PER_OP.with(|m| {
        let mut v: Vec<(String, usize, usize)> =
            m.borrow().iter().map(|(k, (d, s))| (k.clone(), *d, *s)).collect();
        v.sort_by(|a, b| b.2.cmp(&a.2).then(b.1.cmp(&a.1)));
        v
    })
}

/// Dispatches issued through [`S2Design::dispatch`] so far, for a caller that wants the count
/// without the log.
pub fn dispatch_count() -> usize {
    DISPATCHES.load(std::sync::atomic::Ordering::Relaxed)
}

/// How many hw_contexts a pooled chain may hold at once. One below the driver's
/// [`npu_xrt::HWCTX_LIMIT`] so a chain does not consume the whole budget while an unrelated
/// context (a serving engine, a trace design) is open. Raising it buys nothing: the decoder runs
/// OP-MAJOR -- each op dispatches all of its chunks before the next op is touched -- so the
/// instantaneous working set is one design and the LRU never thrashes.
pub const DEFAULT_POOL_LIMIT: usize = npu_xrt::HWCTX_LIMIT - 1;

/// The device-side half of a design: the hw_context and every BO bound to it. Held behind
/// [`DesignSlot`] so a [`DesignPool`] can drop it -- and with it the context -- without dropping
/// the [`S2Design`] handle, which stays usable and reloads on next dispatch.
struct Loaded {
    // FIELD ORDER IS LOAD-BEARING. Rust drops fields in declaration order, and every BO here was
    // allocated against `kern`'s hw_context (`shim_bo_alloc` takes the kernel), so the BOs must go
    // first -- freeing one after its context is destroyed is a use-after-free in XRT. It never
    // mattered while designs lived until process exit; eviction is what made drop order reachable.
    instr: Bo,
    bo_in: Bo,
    bo_resident: Option<Bo>,
    bo_out: Bo,
    kern: Rc<Kernel>,
}

#[derive(Default)]
struct DesignSlot {
    loaded: RefCell<Option<Loaded>>,
}

/// Bounds how many designs hold a hw_context at once, evicting least-recently-used.
///
/// Needed because the chain has more designs than the driver has context slots: the S2 codec
/// decoder is 68 distinct xclbins against `HWCTX_LIMIT` = 16, and opening them eagerly fails the
/// 17th `CREATE_HWCTX` with EINVAL. Eviction only works on designs opened through
/// [`S2Design::open_pooled`], which take an uncached context they own outright
/// ([`Device::load_kernel_owned`]); [`S2Design::open`]'s cached context is never freed.
pub struct DesignPool {
    limit: usize,
    /// Loaded designs, oldest use first. Weak so a released [`S2Design`] leaves no entry behind.
    lru: RefCell<Vec<Weak<DesignSlot>>>,
}

impl DesignPool {
    pub fn new(limit: usize) -> Rc<Self> {
        Rc::new(DesignPool { limit: limit.max(1), lru: RefCell::new(Vec::new()) })
    }

    /// [`DEFAULT_POOL_LIMIT`], or `NPU_S2_POOL_LIMIT` when set. The override exists to make eviction
    /// pressure an independent variable: forcing a low limit at a known-good input separates
    /// "breaks because the run is longer" from "breaks because more designs were evicted".
    pub fn with_default_limit() -> Rc<Self> {
        let limit = std::env::var("NPU_S2_POOL_LIMIT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_POOL_LIMIT);
        Self::new(limit)
    }

    /// Designs currently holding a context through this pool.
    pub fn live(&self) -> usize {
        self.lru.borrow().iter().filter(|w| w.upgrade().is_some()).count()
    }

    fn touch(&self, slot: &Rc<DesignSlot>) {
        let mut lru = self.lru.borrow_mut();
        lru.retain(|w| w.upgrade().is_some_and(|s| !Rc::ptr_eq(&s, slot)));
        lru.push(Rc::downgrade(slot));
    }

    /// Drop loaded designs until one more context fits under [`Self::limit`], never touching
    /// `keep`. A slot that is already borrowed is in use further up the stack, so it is skipped
    /// rather than evicted.
    fn make_room(&self, keep: &Rc<DesignSlot>) {
        let mut lru = self.lru.borrow_mut();
        lru.retain(|w| w.upgrade().is_some());
        let mut i = 0;
        while lru.len() >= self.limit && i < lru.len() {
            let Some(s) = lru[i].upgrade() else {
                lru.remove(i);
                continue;
            };
            if Rc::ptr_eq(&s, keep) {
                i += 1;
                continue;
            }
            let evicted = match s.loaded.try_borrow_mut() {
                Ok(mut g) => {
                    *g = None;
                    true
                }
                Err(_) => false,
            };
            if evicted {
                lru.remove(i);
            } else {
                i += 1;
            }
        }
    }
}

/// One exported design. The handle is device-free -- [`open_pooled`](Self::open_pooled) validates
/// the artifact and reads its instruction stream, but the hw_context and BOs are built on first
/// dispatch and may be evicted by the [`DesignPool`] afterwards.
pub struct S2Design {
    dev: Rc<Device>,
    pool: Option<Rc<DesignPool>>,
    slot: Rc<DesignSlot>,
    xclbin: String,
    /// Kept in memory (a few hundred bytes per design) so an evict/reload cycle does not re-read
    /// and re-validate `insts.bin`.
    insts: Vec<u8>,
    in_elems: usize,
    out_elems: usize,
    pub meta: S2Meta,
}

impl S2Design {
    /// Load `dir/{final.xclbin,insts.bin,meta.json}` onto an already-open Device (single-tenant;
    /// reuse the handle) and allocate every BO up front, holding the hw_context for this
    /// `S2Design`'s whole life. Use [`open_pooled`](Self::open_pooled) for a chain whose design
    /// count exceeds [`npu_xrt::HWCTX_LIMIT`]. No dispatch happens here.
    pub fn open(dev: &Rc<Device>, dir: &Path) -> Result<Self> {
        let d = Self::prepare(dev, dir, None)?;
        d.ensure_loaded()?;
        Ok(d)
    }

    /// Like [`open`](Self::open) but the context is created on first dispatch and `pool` may evict
    /// it again, so N designs sharing one pool hold at most `pool`'s limit of the driver's
    /// budget. Touches no device state.
    pub fn open_pooled(dev: &Rc<Device>, dir: &Path, pool: &Rc<DesignPool>) -> Result<Self> {
        Self::prepare(dev, dir, Some(pool.clone()))
    }

    /// Everything that can be checked without the device: the ABI, the artifact's identity against
    /// `meta.json`, and every declared byte count.
    fn prepare(dev: &Rc<Device>, dir: &Path, pool: Option<Rc<DesignPool>>) -> Result<Self> {
        let meta_path = dir.join(META_FILE);
        let meta: S2Meta = read_json(&meta_path)?;

        if !is_f32(&meta.dtypes.in_) || !is_f32(&meta.dtypes.out) {
            return Err(S2Error::Shape(format!(
                "{}: dtypes.in={} dtypes.out={} -- npu-s2 only handles the f32 codec-decoder ABI",
                meta_path.display(), meta.dtypes.in_, meta.dtypes.out
            )));
        }
        if meta.resident_len > 0 {
            if let Some(rdt) = &meta.dtypes.resident {
                if !is_f32(rdt) {
                    return Err(S2Error::Shape(format!(
                        "{}: dtypes.resident={rdt} -- npu-s2 only handles the f32 codec-decoder ABI",
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
        let xclbin = xclbin_path
            .to_str()
            .ok_or_else(|| S2Error::Shape(format!("{} is not valid UTF-8", xclbin_path.display())))?
            .to_string();

        let insts_path = dir.join(INSTS_FILE);
        let insts = std::fs::read(&insts_path).map_err(|e| S2Error::Io(insts_path.clone(), e))?;
        let n_instr = insts.len() / 4;
        if let Some(declared) = meta.insts_words {
            if declared != n_instr {
                return Err(S2Error::Shape(format!(
                    "{}: n_instr={declared} in meta.json but insts.bin is {n_instr} words \
                     ({} bytes)",
                    meta_path.display(), insts.len()
                )));
            }
        }

        let in_elems = meta.n_tiles.checked_mul(meta.in_tile).ok_or_else(|| {
            S2Error::Shape(format!("{}: n_tiles*in_tile overflows usize", meta_path.display()))
        })?;
        let out_elems = meta.n_tiles.checked_mul(meta.out_numel).ok_or_else(|| {
            S2Error::Shape(format!("{}: n_tiles*out_numel overflows usize", meta_path.display()))
        })?;
        check_bytes("in", meta.buffer_bytes.as_ref().map(|b| b.in_), in_elems * F32_BYTES, &meta_path)?;
        check_bytes("out", meta.buffer_bytes.as_ref().map(|b| b.out), out_elems * F32_BYTES, &meta_path)?;
        if meta.resident_len > 0 {
            check_bytes(
                "resident",
                meta.buffer_bytes.as_ref().map(|b| b.resident),
                meta.resident_len * F32_BYTES,
                &meta_path,
            )?;
        }

        Ok(S2Design {
            dev: dev.clone(),
            pool,
            slot: Rc::new(DesignSlot::default()),
            xclbin,
            insts,
            in_elems,
            out_elems,
            meta,
        })
    }

    /// Instruction-stream length in 32-bit words.
    fn n_instr(&self) -> usize {
        self.insts.len() / 4
    }

    /// Build the hw_context and BOs if this design does not currently hold them, evicting other
    /// pooled designs first if the pool is full. Idempotent; a no-op on the hot path.
    fn ensure_loaded(&self) -> Result<()> {
        if self.slot.loaded.borrow().is_some() {
            if let Some(p) = &self.pool {
                p.touch(&self.slot);
            }
            return Ok(());
        }
        if let Some(p) = &self.pool {
            p.make_room(&self.slot);
        }

        // A pooled design owns its context so the pool can free it; an unpooled one takes the
        // Device's shared cached context, which is never released.
        let kern = match &self.pool {
            Some(_) => self.dev.load_kernel_owned(&self.xclbin, None).map_err(S2Error::Xrt)?,
            None => self.dev.load_kernel(&self.xclbin, None).map_err(S2Error::Xrt)?,
        };
        let g = |arg: i32| kern.group_id(arg).map_err(S2Error::Xrt);

        let instr = self
            .dev
            .alloc_bo(&kern, self.insts.len(), FLAG_CACHEABLE, g(1)?)
            .map_err(S2Error::Xrt)?;
        instr.write_bytes(&self.insts).map_err(S2Error::Xrt)?;
        instr.sync_to_device().map_err(S2Error::Xrt)?;

        // Data BOs land at arg indices 3.. in ABI order (in[, resident], out) -- same convention
        // every other design in this codebase uses (`run_mha`'s Q@3 K@4 V@5 O@6, `run_dwconv6`'s
        // X@3 W@4 Y@5); group_id(1)=instr and arg 2 (count) is a scalar with no BO/group_id.
        let in_bytes = self.in_elems * F32_BYTES;
        let out_bytes = self.out_elems * F32_BYTES;
        // A/B handle for the host-only-BO read race: this tree records an unfenced xdna-driver
        // CLFLUSH race on HOST_ONLY buffers, and the chain's output reads back byte-exactly equal to
        // the PREVIOUS dispatch's result on some windows. `NPU_S2_OUT_BO_FLAG=0` allocates the
        // output as a normal device BO instead, so `sync_from_device` is a real transfer rather
        // than a coherency assumption. Default is unchanged.
        let out_flag: i32 = std::env::var("NPU_S2_OUT_BO_FLAG")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(FLAG_HOST_ONLY);
        let bo_in = self.dev.alloc_bo(&kern, in_bytes, FLAG_HOST_ONLY, g(3)?).map_err(S2Error::Xrt)?;
        let (bo_resident, bo_out) = if self.meta.resident_len > 0 {
            let bo_r = self
                .dev
                .alloc_bo(&kern, self.meta.resident_len * F32_BYTES, FLAG_HOST_ONLY, g(4)?)
                .map_err(S2Error::Xrt)?;
            let bo_o = self.dev.alloc_bo(&kern, out_bytes, out_flag, g(5)?).map_err(S2Error::Xrt)?;
            (Some(bo_r), bo_o)
        } else {
            let bo_o = self.dev.alloc_bo(&kern, out_bytes, out_flag, g(4)?).map_err(S2Error::Xrt)?;
            (None, bo_o)
        };

        if !npu_xrt::quiet() {
            eprintln!(
                "[S2Design] loaded {} (role={} symbol={} n_tiles={} in_tile={} out_numel={} \
                 resident_len={}, {} instr)",
                self.xclbin, self.meta.op, self.meta.symbol, self.meta.n_tiles, self.meta.in_tile,
                self.meta.out_numel, self.meta.resident_len, self.n_instr()
            );
        }

        *self.slot.loaded.borrow_mut() = Some(Loaded { kern, instr, bo_in, bo_resident, bo_out });
        if let Some(p) = &self.pool {
            p.touch(&self.slot);
        }
        Ok(())
    }

    /// `in_tiles`: `n_tiles*in_tile` f32 elements. `resident`: `Some(resident_len elements)` iff
    /// this design has a resident operand, else `None` -- mismatching either way is an `Err`, not
    /// a silent zero-fill. Returns `n_tiles*out_numel` f32 elements. Loads the design first if the
    /// pool has evicted it since the last call.
    pub fn dispatch(&self, in_tiles: &[f32], resident: Option<&[f32]>) -> Result<Vec<f32>> {
        heartbeat(&self.meta.op);
        if in_tiles.len() != self.in_elems {
            return Err(S2Error::Shape(format!(
                "in_tiles: got {} elements, design ({}) wants {}",
                in_tiles.len(), self.meta.op, self.in_elems
            )));
        }
        match (resident, self.meta.resident_len) {
            (Some(r), n) if n > 0 => {
                if r.len() != n {
                    return Err(S2Error::Shape(format!(
                        "resident: got {} elements, design ({}) wants {n}",
                        r.len(), self.meta.op
                    )));
                }
            }
            (None, n) if n > 0 => {
                return Err(S2Error::Shape(format!(
                    "design ({}) requires a resident operand ({n} elements), none given",
                    self.meta.op
                )))
            }
            (Some(r), _) => {
                return Err(S2Error::Shape(format!(
                    "design ({}) has no resident operand, {} elements given",
                    self.meta.op, r.len()
                )))
            }
            (None, _) => {}
        }

        self.ensure_loaded()?;
        let guard = self.slot.loaded.borrow();
        let l = guard.as_ref().expect("ensure_loaded left the slot filled");

        if let (Some(r), Some(bo)) = (resident, &l.bo_resident) {
            bo.write_bytes(f32_bytes(r)).map_err(S2Error::Xrt)?;
            bo.sync_to_device().map_err(S2Error::Xrt)?;
        }

        l.bo_in.write_bytes(f32_bytes(in_tiles)).map_err(S2Error::Xrt)?;
        l.bo_in.sync_to_device().map_err(S2Error::Xrt)?;

        let data: Vec<&Bo> = match &l.bo_resident {
            Some(r) => vec![&l.bo_in, r, &l.bo_out],
            None => vec![&l.bo_in, &l.bo_out],
        };
        l.kern.run_kernel(OPCODE, &l.instr, self.n_instr(), &data).map_err(S2Error::Xrt)?;

        l.bo_out.sync_from_device().map_err(S2Error::Xrt)?;
        let mut out = vec![0f32; self.out_elems];
        l.bo_out.read_bytes(f32_bytes_mut(&mut out)).map_err(S2Error::Xrt)?;

        // `NPU_S2_RESYNC=<n>`: re-sync and re-read until two consecutive reads agree, up to n extra
        // attempts. Tests the workaround this tree records for the unfenced xdna-driver CLFLUSH
        // host-only-BO read race -- measured here as a window reading back byte-exactly equal to the
        // PREVIOUS dispatch's output. Off by default: a retry loop that cannot distinguish "stale"
        // from "legitimately identical" is a diagnostic, not a fix.
        let retries: u32 = std::env::var("NPU_S2_RESYNC").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        if retries > 0 {
            let mut prev = out;
            for _ in 0..retries {
                l.bo_out.sync_from_device().map_err(S2Error::Xrt)?;
                let mut again = vec![0f32; self.out_elems];
                l.bo_out.read_bytes(f32_bytes_mut(&mut again)).map_err(S2Error::Xrt)?;
                let same = prev.iter().zip(&again).all(|(a, b)| a.to_bits() == b.to_bits());
                prev = again;
                if same {
                    break;
                }
                RESYNC_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                note_op(&self.meta.op, true);
            }
            out = prev;
        }
        Ok(out)
    }
}

/// One `manifest.json` row: a design's identity (`name`) and `role` (`head`/`stage1_up`/.../
/// `tail`), plus the directory (relative to the manifest's own directory) holding its
/// `final.xclbin`/`insts.bin`/`meta.json`.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct S2ManifestDesign {
    pub name: String,
    pub op: String,
    pub dir: String,
    #[serde(default)]
    pub group: Option<String>,
    #[serde(default)]
    pub stage: Option<u32>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct S2ManifestFile {
    /// The exporter records the whole pin as an object, not one string: MLIR_AIE_FORK_COMMIT plus
    /// PEANO_FORK_COMMIT. The mlir-aie fork commit is the identity toolchain.lock calls "one exact
    /// AIE toolchain", so that is what a stale artifact set is detected against.
    toolchain: S2ToolchainPin,
    designs: Vec<S2ManifestDesign>,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct S2ToolchainPin {
    #[serde(rename = "MLIR_AIE_FORK_COMMIT")]
    mlir_aie_fork_commit: String,
    #[serde(rename = "PEANO_FORK_COMMIT", default)]
    #[allow(dead_code)]
    peano_fork_commit: Option<String>,
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
        Ok(S2Artifacts { root: dir.to_path_buf(), toolchain_pin: m.toolchain.mlir_aie_fork_commit, designs: m.designs })
    }

    pub fn toolchain_pin(&self) -> &str {
        &self.toolchain_pin
    }

    pub fn designs(&self) -> &[S2ManifestDesign] {
        &self.designs
    }

    pub fn by_op(&self, role: &str) -> Option<&S2ManifestDesign> {
        self.designs.iter().find(|d| d.op == role)
    }

    /// Every design with `op == role`. A chunked op that carries a residual add (the 1x1 conv in
    /// a residual unit, when `c_in_total > c_in`) exports TWO designs under the same `op` -- one
    /// with the add slot for chunk 0, one without for every later chunk (see `chain::ConvOp`,
    /// which classifies them by `op_params.has_add` rather than by name suffix). Every other op
    /// has exactly one.
    pub fn designs_by_op<'a>(&'a self, role: &str) -> Vec<&'a S2ManifestDesign> {
        self.designs.iter().filter(|d| d.op == role).collect()
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

    pub fn open_by_op(&self, dev: &Rc<Device>, role: &str) -> Result<S2Design> {
        let d = self.by_op(role).ok_or_else(|| {
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
            r#"{"symbol":"stage1_up","op":"stage1_up","n_tiles":4,"in_tile":128,
                "out_numel":256,"resident_len":512,
                "dtypes":{"in":"float32","out":"float32","resident":"float32"}}"#,
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
            r#"{"symbol":"head","op":"head","n_tiles":1,"in_tile":8,"out_numel":8,
                "dtypes":{"in":"float32","out":"float32","resident":null},
                "ci_chunk":128,"window":{"t":64}}"#,
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
    fn manifest_by_op_and_name() {
        let td = tempfile::tempdir().unwrap();
        write(
            td.path(), MANIFEST_FILE,
            r#"{"toolchain":{"MLIR_AIE_FORK_COMMIT":"deadbeef","PEANO_FORK_COMMIT":"cafe"},
                "designs":[
                {"name":"head","op":"head","dir":"head"},
                {"name":"stage1_up","op":"stage1_up","dir":"stage1"}
            ]}"#,
        );
        let art = S2Artifacts::open(td.path()).unwrap();
        assert_eq!(art.toolchain_pin(), "deadbeef");
        assert_eq!(art.by_op("stage1_up").unwrap().dir, "stage1");
        assert!(art.by_name("tail").is_none());
        assert_eq!(art.design_dir(art.by_name("head").unwrap()), td.path().join("head"));
    }

    #[test]
    fn toolchain_pin_matches_full_hash_lock() {
        let td = tempfile::tempdir().unwrap();
        write(td.path(), MANIFEST_FILE,
              r#"{"toolchain":{"MLIR_AIE_FORK_COMMIT":"035528f71cf1dd0"},"designs":[]}"#);
        let lock = td.path().join("toolchain.lock");
        std::fs::write(&lock, "MLIR_AIE_FORK_COMMIT=035528f71cf1dd067ff01562fda99088678ca2b0   # comment\n")
            .unwrap();
        let art = S2Artifacts::open(td.path()).unwrap();
        art.validate_toolchain(&lock).expect("short pin should prefix-match the full lock commit");
    }

    #[test]
    fn toolchain_pin_mismatch_is_an_error() {
        let td = tempfile::tempdir().unwrap();
        write(td.path(), MANIFEST_FILE,
              r#"{"toolchain":{"MLIR_AIE_FORK_COMMIT":"0000000000000000"},"designs":[]}"#);
        let lock = td.path().join("toolchain.lock");
        std::fs::write(&lock, "MLIR_AIE_FORK_COMMIT=035528f71cf1dd067ff01562fda99088678ca2b0\n").unwrap();
        let art = S2Artifacts::open(td.path()).unwrap();
        assert!(art.validate_toolchain(&lock).is_err());
    }

    /// Parse a meta.json produced by the REAL exporter, not a fixture. The first version of this
    /// crate was written before the exporter existed and guessed six field names wrong (`role` for
    /// `op`, flat `in_dtype`/`in_bytes` for nested `dtypes`/`buffer_bytes`, `n_instr` for
    /// `insts_words`, a `toolchain_pin` string for a `toolchain` object). Fixtures alone could not
    /// catch that, because they encoded the same guess the parser did. Skips when no export is
    /// present so the suite stays runnable on a clean checkout.
    #[test]
    fn meta_parses_a_real_exported_artifact() {
        // $S2_ARTIFACT_DIR points at one exported design directory. Env-driven rather than a
        // hardcoded path: an absolute home directory does not belong in the public tree, and this
        // way the test runs against whatever export the caller actually has.
        let Ok(dir) = std::env::var("S2_ARTIFACT_DIR") else {
            eprintln!("SKIP: set S2_ARTIFACT_DIR to an exported design directory");
            return;
        };
        let dir = Path::new(&dir);
        if !dir.join(META_FILE).is_file() {
            eprintln!("SKIP: no meta.json at {}", dir.display());
            return;
        }
        let meta: S2Meta = read_json(&dir.join(META_FILE)).unwrap();
        // Shape-agnostic: assert the INVARIANTS that must hold for any exported design, not one
        // design's literals, so this works against whatever S2_ARTIFACT_DIR points at.
        assert!(!meta.symbol.is_empty() && !meta.op.is_empty());
        assert!(meta.n_tiles > 0 && meta.in_tile > 0 && meta.out_numel > 0);
        assert_eq!(meta.dtypes.in_, "float32");
        assert_eq!(meta.dtypes.out, "float32");
        let bb = meta.buffer_bytes.as_ref().unwrap();
        assert_eq!(bb.in_, meta.n_tiles * meta.in_tile * F32_BYTES);
        assert_eq!(bb.out, meta.n_tiles * meta.out_numel * F32_BYTES);
        assert_eq!(bb.resident, meta.resident_len * F32_BYTES);
    }

    /// Parse `op_params` on EVERY design in a real exported manifest (not one hand-picked
    /// directory) and cross-check it against that same design's `n_tiles`/`in_tile`/`out_numel`/
    /// `resident_len` -- i.e. the exact formulas `window::SnakeOp/ConvOp/ConvTransposeOp::new`
    /// apply when opening a design for real (device-gated, so untestable here directly). This
    /// catches an `S2OpParams` schema mistake (a wrong field name silently landing in `extra`
    /// instead of failing to parse, or a formula that disagrees with what the exporter actually
    /// shipped) BEFORE it would only show up as a device-side dispatch error.
    #[test]
    fn op_params_matches_shapes_on_every_real_design() {
        let Ok(root) = std::env::var("S2_ARTIFACTS_ROOT") else {
            eprintln!("skip: set S2_ARTIFACTS_ROOT to a directory holding manifest.json + design dirs");
            return;
        };
        let root = Path::new(&root);
        if !root.join(MANIFEST_FILE).is_file() {
            eprintln!("skip: no manifest.json at {}", root.display());
            return;
        }
        let art = S2Artifacts::open(root).unwrap();
        assert!(!art.designs().is_empty(), "manifest at {} lists no designs", root.display());
        let mut checked = 0;
        for d in art.designs() {
            let dir = art.design_dir(d);
            let meta: S2Meta = read_json(&dir.join(META_FILE)).unwrap();
            match meta.op_params.as_ref().unwrap_or_else(|| panic!("{}: no op_params", dir.display())) {
                S2OpParams::Snake { c, t, .. } => {
                    assert_eq!(meta.n_tiles, *c, "{}: n_tiles", dir.display());
                    assert_eq!(meta.in_tile, t + 1, "{}: in_tile", dir.display());
                    assert_eq!(meta.out_numel, *t, "{}: out_numel", dir.display());
                    assert_eq!(meta.resident_len, 0, "{}: resident_len", dir.display());
                }
                S2OpParams::Conv { k, ctx, c_in, c_out, has_add, t, .. } => {
                    assert_eq!(meta.n_tiles, *c_out, "{}: n_tiles", dir.display());
                    let want_in = c_in * k + 1 + if *has_add { *t } else { 0 };
                    assert_eq!(meta.in_tile, want_in, "{}: in_tile", dir.display());
                    assert_eq!(meta.out_numel, *t, "{}: out_numel", dir.display());
                    assert_eq!(meta.resident_len, c_in * t, "{}: resident_len", dir.display());
                    assert!(*ctx > 0 || *k == 1, "{}: ctx=0 but k={k} != 1", dir.display());
                }
                S2OpParams::ConvTranspose { k, c_in, c_out, t, stride, .. } => {
                    assert_eq!(meta.n_tiles, *c_out, "{}: n_tiles", dir.display());
                    assert_eq!(meta.in_tile, c_in * k + 1, "{}: in_tile", dir.display());
                    assert_eq!(meta.out_numel, t * stride, "{}: out_numel", dir.display());
                    assert_eq!(meta.resident_len, c_in * t, "{}: resident_len", dir.display());
                }
            }
            checked += 1;
        }
        eprintln!("op_params_matches_shapes_on_every_real_design: checked {checked} designs under {}", root.display());
    }
}
