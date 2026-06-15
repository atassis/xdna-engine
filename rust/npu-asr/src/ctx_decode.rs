//! ctxDecode — resident thin-M GEMV primitive for the on-NPU Whisper decoder.
//!
//! Decode runs one token at a time (M=1). The whole_array GEMM design's smallest legal M is 64
//! (see scripts/build_decode_kernels.sh), so the single decode query is placed in row 0 of a
//! [64,K] activation and rows 1..63 are zero-padded; the result is read back from output row 0.
//! Native bf16-in / f32-out, tile 8x32x32, 8 cols.
//!
//! Pattern mirrors `ctx_ln.rs` (resident xclbin loaded once, reused BOs, dispatched per call) but
//! generalises to many GEMV shapes: one resident kernel per (K, N_pad), held in a map. Weights are
//! registered once (packed to bf16 into a resident `bo_b`, written+synced once); `gemv` then only
//! pays the per-call activation write/sync, dispatch, and readback.
//!
//! N is padded up to a multiple of 32 (the design requires N % 32 == 0); the extra columns are
//! zero in the weight and dropped from the output. Epilogues are MINIMAL: optional host-side bias
//! add after readback. GELU / on-chip fusion are intentionally out of scope.
//!
//! ABI (see decode_gemv_probe.rs / ctx2.rs:655-675): kernel args 1=instr (CACHEABLE),
//! 3=A=activation[64,K] bf16, 4=B=weight[K,N_pad] bf16, 5=C=output[64,N_pad] f32, 6=tmp, 7=trace;
//! run_matmul8(opcode=3, instr, n_instr, A, B, C, tmp, trace).
//!
//! NPU is single-tenant — stop npu-asr.service / voxd.service before any on-device run.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use ndarray::prelude::*;
use npu_xrt::{f32_to_bf16_bits, pack_f32_to_bf16, Bo, Device, Kernel, FLAG_CACHEABLE, FLAG_HOST_ONLY};

/// Smallest legal M for the native-bf16 8-col whole_array design (see build_decode_kernels.sh).
const M: usize = 64;
const TILE: &str = "8x32x32"; // m x k x n
const COLS: usize = 8;
const WA: &str = "mlir-aie/programming_examples/basic/matrix_multiplication/whole_array/build";

// --- on-chip single-query multi-head attention (mha_decode) ---
const ATTN_D: usize = 768;
const ATTN_NHEADS: usize = 12;
const ATTN_HD: usize = 64; // D / NHEADS
const ATTN_TKV: usize = 64; // keys per K/V tile; MUST match the kernel's MHA_TKV.
const ATTN_S_MAX: usize = 448; // fixed max cache length -> fixed unrolled tile count.
const ATTN_N_TILES: usize = (ATTN_S_MAX + ATTN_TKV - 1) / ATTN_TKV; // 7 (the ONE xclbin)
const ATTN_KV_TILE: usize = 2 * ATTN_TKV * ATTN_HD + 2; // K | V | int32 header (2 bf16)
const MHA_DIR: &str = "mlir-aie/programming_examples/ml/mha_decode/build";

/// Round `n` up to the next multiple of 32 (the whole_array N-tiling constraint, N % (n*cols)..
/// here N % 32 == 0 suffices for the GEMV shapes we register; cols=8,n=32 => N % 256 for full
/// 8-col packing, but the design accepts any N % 32 by leaving trailing AIE columns idle).
fn pad32(n: usize) -> usize {
    n.div_ceil(32) * 32
}

fn u16_bytes(v: &[u16]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

/// Host-side epilogue applied after readback. Minimal by design.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeEpi {
    None,
    Bias,
}

/// Pre-norm applied to the decode activation `x[K]` before the projection. The norm is FOLDED into
/// the resident weight (`W'' = diag(γ)·W`, LN also `bias' = β@W + bias`), making it SEPARABLE from
/// the matmul (see `internal notes` §3):
///   RMS: `out = inv_rms·(x @ W'') + bias`,  `inv_rms = 1/√(Σx²/K + eps)`
///   LN : `out = inv_std·((x−mean) @ W'') + bias'`,  `mean=Σx/K`, `inv_std=1/√(Σ(x−mean)²/K + eps)`
/// Because decode is M=1, the norm is a single K-vector reduction (free on host, not a dispatch);
/// the heavy `x @ W''` stays the one resident GEMV dispatch. `register_fused` precomputes `W''`/`bias'`;
/// `fused_norm_gemv` does the (trivial) input-scale on host, dispatches the GEMV, and host-adds `bias'`.
#[derive(Clone, Debug)]
pub enum Norm {
    /// RMSNorm(γ, eps): `inv_rms = 1/√(mean(x²)+eps)`; `γ` folded into `W''`.
    Rms { gamma: Vec<f32>, eps: f32 },
    /// LayerNorm(γ, β, eps): `mean`/`inv_std` over x; `γ` folded into `W''`, `β` folded into `bias'`.
    Ln { gamma: Vec<f32>, beta: Vec<f32>, eps: f32 },
}

/// A registered decoder weight: resident `[K, N_pad]` bf16 matrix, written to the device once.
pub struct DecodeWeight {
    /// Resident weight BO `[K, N_pad]` bf16, written+synced at registration.
    bo_b: Bo,
    k: usize,
    n: usize,
    n_pad: usize,
    epi: DecodeEpi,
    /// Length-`n` bias, used only when `epi == DecodeEpi::Bias`.
    bias: Vec<f32>,
}

impl DecodeWeight {
    pub fn k(&self) -> usize {
        self.k
    }
    pub fn n(&self) -> usize {
        self.n
    }
    pub fn n_pad(&self) -> usize {
        self.n_pad
    }
}

/// A resident GEMV kernel for one (K, N_pad) shape: the loaded xclbin + its instruction stream +
/// the reusable per-shape activation/output/scratch BOs.
struct ShapeKernel {
    kern: Rc<Kernel>,
    instr: Bo,
    n_instr: usize,
    bo_a: Bo,   // [M, K] bf16 activation (row 0 = the query, rows 1.. zero)
    bo_c: Bo,   // [M, N_pad] f32 output
    bo_tmp: Bo, // scratch (host ABI requires a live BO)
    bo_tr: Bo,  // trace (host ABI requires a live BO)
    /// Reused host-side activation staging buffer [M*K] f32 (row 0 holds x, rest stays 0).
    a_buf: RefCell<Vec<f32>>,
    /// Reused host-side activation bf16 buffer [M*K].
    a_bf16: RefCell<Vec<u16>>,
    /// Reused host-side readback buffer [M*N_pad] f32.
    c_buf: RefCell<Vec<f32>>,
}

/// Resident on-chip single-query MHA kernel (the ONE runtime-S mha_decode xclbin) plus its reusable
/// q/kv/ctx BOs. Loaded lazily on first `attn` call. The kernel streams a FIXED `ATTN_N_TILES` K/V
/// tiles per head and reads the real per-tile key count at runtime from each tile's int32 header, so
/// one xclbin serves every cache length S<=ATTN_S_MAX.
struct AttnKernel {
    kern: Rc<Kernel>,
    instr: Bo,
    n_instr: usize,
    bo_q: Bo,   // [NHEADS*HD] bf16 (head-major query)
    bo_kv: Bo,  // [NHEADS*N_TILES*KV_TILE] bf16 (per head/tile: K | V | int32 hdr)
    bo_ctx: Bo, // [NHEADS*HD] f32 (read back)
    bo_tmp: Bo,
    bo_tr: Bo,
    q_bf: RefCell<Vec<u16>>,
    kv_bf: RefCell<Vec<u16>>,
    ctx_buf: RefCell<Vec<f32>>,
}

/// Resident on-NPU GEMV primitive for the decoder. Holds one [`ShapeKernel`] per (K, N_pad) shape.
pub struct CtxDecode {
    dev: Rc<Device>,
    root: PathBuf,
    shapes: RefCell<HashMap<(usize, usize), Rc<ShapeKernel>>>,
    /// The lazily-loaded on-chip attention kernel (single runtime-S mha_decode xclbin).
    attn_kernel: RefCell<Option<Rc<AttnKernel>>>,
    /// Cheap dispatch counter: incremented once per `gemv` (== one NPU matmul dispatch). Used by the
    /// timing harness to report dispatches/token exactly. Free in the hot path (a single `Cell` add).
    dispatches: Cell<u64>,
}

impl CtxDecode {
    pub fn new(dev: &Rc<Device>, root: &Path) -> Self {
        CtxDecode {
            dev: Rc::clone(dev),
            root: root.to_path_buf(),
            shapes: RefCell::new(HashMap::new()),
            attn_kernel: RefCell::new(None),
            dispatches: Cell::new(0),
        }
    }

    /// Total number of `gemv` dispatches issued since construction (or last `reset_dispatches`).
    pub fn dispatches(&self) -> u64 {
        self.dispatches.get()
    }

    /// Reset the dispatch counter (called by the timing harness before each transcription).
    pub fn reset_dispatches(&self) {
        self.dispatches.set(0);
    }

    /// Ensure the resident GEMV kernel for shape (k, n_pad) is loaded; load it (and alloc its
    /// reusable BOs) on first use. `n_pad` must already be a multiple of 32. Returns a clear error
    /// naming the missing xclbin/insts file if the build step has not produced it.
    fn ensure_shape(&self, k: usize, n_pad: usize) -> Result<Rc<ShapeKernel>, String> {
        debug_assert_eq!(n_pad % 32, 0, "n_pad must be a multiple of 32");
        if let Some(sh) = self.shapes.borrow().get(&(k, n_pad)) {
            return Ok(Rc::clone(sh));
        }

        let wa = self.root.join(WA);
        let stem = format!("{M}x{k}x{n_pad}_{TILE}_{COLS}c");
        let xclbin = wa.join(format!("final_{stem}.xclbin"));
        let insts = wa.join(format!("insts_{stem}.txt"));

        if !xclbin.exists() {
            return Err(format!(
                "ctxDecode: missing GEMV xclbin {} — build it with: \
                 source scripts/iron_env.sh && bash scripts/build_decode_kernels.sh {k} {n_pad}",
                xclbin.display()
            ));
        }
        if !insts.exists() {
            return Err(format!(
                "ctxDecode: missing GEMV insts {} — build it with: \
                 source scripts/iron_env.sh && bash scripts/build_decode_kernels.sh {k} {n_pad}",
                insts.display()
            ));
        }

        let kern = self
            .dev
            .load_kernel(xclbin.to_str().unwrap(), None)
            .map_err(|e| format!("ctxDecode: load {}: {e}", xclbin.display()))?;
        let ibytes = std::fs::read(&insts)
            .map_err(|e| format!("ctxDecode: read insts {}: {e}", insts.display()))?;
        let n_instr = ibytes.len() / 4; // 4 bytes/instr
        let g = |i| kern.group_id(i).unwrap();

        let instr = self
            .dev
            .alloc_bo(&kern, ibytes.len(), FLAG_CACHEABLE, g(1))
            .map_err(|e| format!("ctxDecode: alloc instr BO: {e}"))?;
        instr.write_bytes(&ibytes).map_err(|e| format!("ctxDecode: write instr: {e}"))?;
        instr.sync_to_device().map_err(|e| format!("ctxDecode: sync instr: {e}"))?;

        let bo_a = self
            .dev
            .alloc_bo(&kern, M * k * 2, FLAG_HOST_ONLY, g(3))
            .map_err(|e| format!("ctxDecode: alloc A BO: {e}"))?;
        let bo_c = self
            .dev
            .alloc_bo(&kern, M * n_pad * 4, FLAG_HOST_ONLY, g(5))
            .map_err(|e| format!("ctxDecode: alloc C BO: {e}"))?;
        let bo_tmp = self
            .dev
            .alloc_bo(&kern, 1, FLAG_HOST_ONLY, g(6))
            .map_err(|e| format!("ctxDecode: alloc tmp BO: {e}"))?;
        let bo_tr = self
            .dev
            .alloc_bo(&kern, 4, FLAG_HOST_ONLY, g(7))
            .map_err(|e| format!("ctxDecode: alloc trace BO: {e}"))?;
        eprintln!(
            "[ctxDecode] resident GEMV xclbin loaded (M={M} K={k} N_pad={n_pad}, native bf16->f32, {COLS} cols)"
        );

        let sh = Rc::new(ShapeKernel {
            kern,
            instr,
            n_instr,
            bo_a,
            bo_c,
            bo_tmp,
            bo_tr,
            a_buf: RefCell::new(vec![0f32; M * k]),
            a_bf16: RefCell::new(vec![0u16; M * k]),
            c_buf: RefCell::new(vec![0f32; M * n_pad]),
        });
        self.shapes.borrow_mut().insert((k, n_pad), Rc::clone(&sh));
        Ok(sh)
    }

    /// Like [`ensure_shape`], but takes a raw N (pads it to a multiple of 32 internally). Useful to
    /// preload a shape before any `register_weight`/`gemv`.
    pub fn ensure_shape_n(&self, k: usize, n: usize) -> Result<(), String> {
        self.ensure_shape(k, pad32(n)).map(|_| ())
    }

    /// Register a decoder weight `[K, N]` (f32, row-major): pad N up to a multiple of 32 (zero-pad
    /// the extra weight columns), pack `[K, N_pad]` to bf16 into a resident `bo_b` written+synced
    /// once, and ensure the (K, N_pad) kernel is loaded. The returned [`DecodeWeight`] is reused
    /// across `gemv` calls (the weight stays resident on the device).
    ///
    /// Panics on shape/epilogue inconsistency (programmer error); device/build failures surface via
    /// [`ensure_shape`] which `gemv` re-runs lazily, but the weight BO alloc here may also error —
    /// it panics with a clear message (registration is a setup step, not a hot path).
    pub fn register_weight(
        &mut self,
        w: &Array2<f32>,
        epi: DecodeEpi,
        bias: &[f32],
    ) -> DecodeWeight {
        let k = w.nrows();
        let n = w.ncols();
        let n_pad = pad32(n);
        if epi == DecodeEpi::Bias {
            assert_eq!(bias.len(), n, "ctxDecode: bias len {} != N {n}", bias.len());
        }

        // Ensure the kernel/BOs for this shape exist (so a later gemv can't fail on a cold shape,
        // and so we surface a missing-xclbin error at registration time).
        let sh = self
            .ensure_shape(k, n_pad)
            .unwrap_or_else(|e| panic!("{e}"));

        // Pack [K, N_pad] bf16 weight: real columns from w, padding columns zero.
        let mut b_f32 = vec![0f32; k * n_pad];
        for (r, row) in w.outer_iter().enumerate() {
            let dst = &mut b_f32[r * n_pad..r * n_pad + n];
            for (c, &v) in row.iter().enumerate() {
                dst[c] = v;
            }
            // columns n..n_pad stay 0
        }
        let mut b_bf16 = vec![0u16; k * n_pad];
        pack_f32_to_bf16(&b_f32, &mut b_bf16);

        let bo_b = self
            .dev
            .alloc_bo(&sh.kern, k * n_pad * 2, FLAG_HOST_ONLY, sh.kern.group_id(4).unwrap())
            .unwrap_or_else(|e| panic!("ctxDecode: alloc weight BO: {e}"));
        bo_b.write_bytes(u16_bytes(&b_bf16))
            .unwrap_or_else(|e| panic!("ctxDecode: write weight: {e}"));
        bo_b.sync_to_device()
            .unwrap_or_else(|e| panic!("ctxDecode: sync weight: {e}"));

        DecodeWeight {
            bo_b,
            k,
            n,
            n_pad,
            epi,
            bias: if epi == DecodeEpi::Bias { bias.to_vec() } else { Vec::new() },
        }
    }

    /// Run the resident GEMV `x · W` for a registered weight. `x` has length K; it is placed in
    /// row 0 of the `[64, K]` activation (rows 1.. zero), dispatched on the (K, N_pad) kernel
    /// against the resident weight BO, and the first N values of output row 0 are returned (the
    /// N-padding columns are dropped). Applies the host bias if the weight's epilogue is `Bias`.
    ///
    /// Returns the length-N result, or an error if the shape's kernel cannot be (re)loaded.
    pub fn gemv(&self, w: &DecodeWeight, x: &[f32]) -> Result<Vec<f32>, String> {
        assert_eq!(x.len(), w.k, "ctxDecode: x len {} != weight K {}", x.len(), w.k);
        self.dispatches.set(self.dispatches.get() + 1);
        let sh = self.ensure_shape(w.k, w.n_pad)?;

        // Stage activation: row 0 = x, rows 1..M = 0; pack to bf16; write+sync.
        {
            let mut ab = sh.a_buf.borrow_mut();
            ab[..w.k].copy_from_slice(x);
            // rows 1..M stay zero (never written after init).
            let mut abf = sh.a_bf16.borrow_mut();
            pack_f32_to_bf16(&ab, &mut abf);
            sh.bo_a
                .write_bytes(u16_bytes(&abf))
                .map_err(|e| format!("ctxDecode: write A: {e}"))?;
        }
        sh.bo_a.sync_to_device().map_err(|e| format!("ctxDecode: sync A: {e}"))?;

        sh.kern
            .run_matmul8(3, &sh.instr, sh.n_instr, &sh.bo_a, &w.bo_b, &sh.bo_c, &sh.bo_tmp, &sh.bo_tr)
            .map_err(|e| format!("ctxDecode: dispatch: {e}"))?;

        sh.bo_c.sync_from_device().map_err(|e| format!("ctxDecode: sync C: {e}"))?;
        {
            let mut cb = sh.c_buf.borrow_mut();
            debug_assert_eq!(cb.len(), M * w.n_pad, "ctx_decode output buffer size mismatch");
            let dst = unsafe {
                std::slice::from_raw_parts_mut(cb.as_mut_ptr() as *mut u8, M * w.n_pad * 4)
            };
            sh.bo_c.read_bytes(dst).map_err(|e| format!("ctxDecode: read C: {e}"))?;
        }

        // Output row 0, first N values (drop the N-padding columns).
        let cb = sh.c_buf.borrow();
        let mut out = cb[..w.n].to_vec();
        if w.epi == DecodeEpi::Bias {
            for (o, &b) in out.iter_mut().zip(w.bias.iter()) {
                *o += b;
            }
        }
        Ok(out)
    }

    /// Ensure the on-chip MHA kernel (single runtime-S mha_decode xclbin) is loaded; load it and
    /// alloc its reusable BOs on first use. Returns a clear error naming the missing artifact if
    /// the build step (scripts/build_mha_decode.sh) has not produced it.
    fn ensure_attn(&self) -> Result<Rc<AttnKernel>, String> {
        if let Some(ak) = self.attn_kernel.borrow().as_ref() {
            return Ok(Rc::clone(ak));
        }
        let dir = self.root.join(MHA_DIR);
        let xclbin = dir.join(format!("final_mha_decode_{ATTN_S_MAX}.xclbin"));
        let insts = dir.join(format!("insts_mha_decode_{ATTN_S_MAX}.txt"));
        if !xclbin.exists() {
            return Err(format!(
                "ctxDecode::attn: missing MHA xclbin {} — build it with: \
                 source scripts/iron_env.sh && bash scripts/build_mha_decode.sh",
                xclbin.display()
            ));
        }
        if !insts.exists() {
            return Err(format!(
                "ctxDecode::attn: missing MHA insts {} — build it with: \
                 source scripts/iron_env.sh && bash scripts/build_mha_decode.sh",
                insts.display()
            ));
        }

        let kern = self
            .dev
            .load_kernel(xclbin.to_str().unwrap(), None)
            .map_err(|e| format!("ctxDecode::attn: load {}: {e}", xclbin.display()))?;
        let ibytes = std::fs::read(&insts)
            .map_err(|e| format!("ctxDecode::attn: read insts {}: {e}", insts.display()))?;
        let n_instr = ibytes.len() / 4;
        let g = |i| kern.group_id(i).unwrap();

        let instr = self
            .dev
            .alloc_bo(&kern, ibytes.len(), FLAG_CACHEABLE, g(1))
            .map_err(|e| format!("ctxDecode::attn: alloc instr BO: {e}"))?;
        instr.write_bytes(&ibytes).map_err(|e| format!("ctxDecode::attn: write instr: {e}"))?;
        instr.sync_to_device().map_err(|e| format!("ctxDecode::attn: sync instr: {e}"))?;

        let q_elems = ATTN_NHEADS * ATTN_HD;
        let kv_elems = ATTN_NHEADS * ATTN_N_TILES * ATTN_KV_TILE;
        let ctx_elems = ATTN_NHEADS * ATTN_HD;
        let bo_q = self
            .dev
            .alloc_bo(&kern, q_elems * 2, FLAG_HOST_ONLY, g(3))
            .map_err(|e| format!("ctxDecode::attn: alloc q BO: {e}"))?;
        let bo_kv = self
            .dev
            .alloc_bo(&kern, kv_elems * 2, FLAG_HOST_ONLY, g(4))
            .map_err(|e| format!("ctxDecode::attn: alloc kv BO: {e}"))?;
        let bo_ctx = self
            .dev
            .alloc_bo(&kern, ctx_elems * 4, FLAG_HOST_ONLY, g(5))
            .map_err(|e| format!("ctxDecode::attn: alloc ctx BO: {e}"))?;
        let bo_tmp = self
            .dev
            .alloc_bo(&kern, 8, FLAG_HOST_ONLY, g(6))
            .map_err(|e| format!("ctxDecode::attn: alloc tmp BO: {e}"))?;
        let bo_tr = self
            .dev
            .alloc_bo(&kern, 4, FLAG_HOST_ONLY, g(7))
            .map_err(|e| format!("ctxDecode::attn: alloc trace BO: {e}"))?;
        eprintln!(
            "[ctxDecode] resident MHA xclbin loaded (single-query, {ATTN_NHEADS} heads x {ATTN_HD}, \
             runtime-S streaming/flash, {ATTN_N_TILES} tiles, bf16 in / f32 ctx)"
        );

        let ak = Rc::new(AttnKernel {
            kern,
            instr,
            n_instr,
            bo_q,
            bo_kv,
            bo_ctx,
            bo_tmp,
            bo_tr,
            q_bf: RefCell::new(vec![0u16; q_elems]),
            kv_bf: RefCell::new(vec![0u16; kv_elems]),
            ctx_buf: RefCell::new(vec![0f32; ctx_elems]),
        });
        *self.attn_kernel.borrow_mut() = Some(Rc::clone(&ak));
        Ok(ak)
    }

    /// Eagerly load the on-chip MHA kernel + alloc its BOs (preload at decoder construction so a
    /// missing xclbin fails loudly up front rather than mid-decode). Idempotent.
    pub fn ensure_attn_loaded(&self) -> Result<(), String> {
        self.ensure_attn().map(|_| ())
    }

    /// On-chip single-query multi-head attention for the decoder's self-attention sublayer.
    /// `q` is the query row [768]; `k_flat`/`v_flat` are the self-KV cache `[s, 768]` row-major
    /// (f32, as held by the decoder). `s` is the real cache length (1..=448). Converts K/V to bf16,
    /// packs head-major into the FIXED `ATTN_N_TILES`-tile layout with per-tile runtime key counts
    /// (zero-pad-free: empty/partial tiles carry their real count in the int32 header), dispatches
    /// the resident MHA kernel (ONE dispatch, M=1), and returns the context row ctx[768] f32.
    /// Mirrors the BO management in `gemv` (reused resident BOs; per-call write/sync/dispatch/read).
    pub fn attn(
        &self,
        q: &[f32],
        k_flat: &[f32],
        v_flat: &[f32],
        s: usize,
    ) -> Result<Vec<f32>, String> {
        assert_eq!(q.len(), ATTN_D, "ctxDecode::attn: q len {} != D {ATTN_D}", q.len());
        assert!(s >= 1 && s <= ATTN_S_MAX, "ctxDecode::attn: s={s} out of 1..={ATTN_S_MAX}");
        assert_eq!(k_flat.len(), s * ATTN_D, "ctxDecode::attn: k_flat len {} != s*D", k_flat.len());
        assert_eq!(v_flat.len(), s * ATTN_D, "ctxDecode::attn: v_flat len {} != s*D", v_flat.len());
        self.dispatches.set(self.dispatches.get() + 1);
        let ak = self.ensure_attn()?;

        let n_real_tiles = s.div_ceil(ATTN_TKV); // tiles holding >=1 real key (<= N_TILES)
        let hdr_off = 2 * ATTN_TKV * ATTN_HD;

        // ---- pack q [12,64] bf16 (head-major) ----
        {
            let mut qb = ak.q_bf.borrow_mut();
            for h in 0..ATTN_NHEADS {
                for d in 0..ATTN_HD {
                    qb[h * ATTN_HD + d] = f32_to_bf16_bits(q[h * ATTN_HD + d]);
                }
            }
            ak.bo_q.write_bytes(u16_bytes(&qb)).map_err(|e| format!("ctxDecode::attn: write q: {e}"))?;
        }
        ak.bo_q.sync_to_device().map_err(|e| format!("ctxDecode::attn: sync q: {e}"))?;

        // ---- pack kv [12, N_TILES, KV_TILE] bf16 (K | V | int32 runtime count) ----
        {
            let mut kvb = ak.kv_bf.borrow_mut();
            for v in kvb.iter_mut() {
                *v = 0;
            }
            for h in 0..ATTN_NHEADS {
                let base = h * ATTN_HD;
                for t in 0..ATTN_N_TILES {
                    let off = (h * ATTN_N_TILES + t) * ATTN_KV_TILE;
                    let k_off = off;
                    let v_off = off + ATTN_TKV * ATTN_HD;
                    for r in 0..ATTN_TKV {
                        let key = t * ATTN_TKV + r;
                        if key >= s {
                            break; // remaining rows in this tile are empty (stay 0)
                        }
                        for d in 0..ATTN_HD {
                            kvb[k_off + r * ATTN_HD + d] = f32_to_bf16_bits(k_flat[key * ATTN_D + base + d]);
                            kvb[v_off + r * ATTN_HD + d] = f32_to_bf16_bits(v_flat[key * ATTN_D + base + d]);
                        }
                    }
                    // per-tile runtime count (int32, bit-exact into 2 bf16 lanes).
                    let s_in_tile: i32 = if t >= n_real_tiles {
                        0
                    } else {
                        let real = (s - t * ATTN_TKV).min(ATTN_TKV) as i32;
                        if t == n_real_tiles - 1 {
                            -real
                        } else {
                            real
                        }
                    };
                    let bytes = s_in_tile.to_le_bytes();
                    kvb[off + hdr_off] = u16::from_le_bytes([bytes[0], bytes[1]]);
                    kvb[off + hdr_off + 1] = u16::from_le_bytes([bytes[2], bytes[3]]);
                }
            }
            ak.bo_kv.write_bytes(u16_bytes(&kvb)).map_err(|e| format!("ctxDecode::attn: write kv: {e}"))?;
        }
        ak.bo_kv.sync_to_device().map_err(|e| format!("ctxDecode::attn: sync kv: {e}"))?;

        // ---- dispatch (same ABI as gemv: opcode 3, A=q, B=kv, C=ctx) ----
        ak.kern
            .run_matmul8(3, &ak.instr, ak.n_instr, &ak.bo_q, &ak.bo_kv, &ak.bo_ctx, &ak.bo_tmp, &ak.bo_tr)
            .map_err(|e| format!("ctxDecode::attn: dispatch: {e}"))?;

        // ---- read back ctx [12,64] f32 -> [768] ----
        ak.bo_ctx.sync_from_device().map_err(|e| format!("ctxDecode::attn: sync ctx: {e}"))?;
        {
            let mut cb = ak.ctx_buf.borrow_mut();
            let dst = unsafe {
                std::slice::from_raw_parts_mut(cb.as_mut_ptr() as *mut u8, ATTN_NHEADS * ATTN_HD * 4)
            };
            ak.bo_ctx.read_bytes(dst).map_err(|e| format!("ctxDecode::attn: read ctx: {e}"))?;
        }
        let ctx = ak.ctx_buf.borrow().clone();
        Ok(ctx)
    }

    /// Register a FUSED pre-norm + projection weight: precompute the folded weight `W'' = diag(γ)·W`
    /// (LN also `bias' = β@W + bias`), pack `W''` to a resident bf16 `[K, N_pad]` BO (same residency as
    /// [`register_weight`]), and ensure the matching `{ln|rms}` xclbin is loaded. The returned
    /// [`FusedWeight`] holds the folded `bias'` (LN) / `bias` (RMS) for the host post-add and the norm
    /// params for the (trivial, M=1) per-call input reduction.
    ///
    /// `w` is `[K, N]` f32 row-major (the ORIGINAL weight, pre-fold). `bias` is length-N (or empty).
    /// The fold reference is `rust/npu-asr/src/bin/norm_gemv_probe.rs`; correctness is the §3 separable
    /// identity proven to machine-ε by `norm_gemv_probe selftest`.
    pub fn register_fused(&mut self, w: &Array2<f32>, norm: Norm, bias: &[f32]) -> FusedWeight {
        let k = w.nrows();
        let n = w.ncols();
        if !bias.is_empty() {
            assert_eq!(bias.len(), n, "ctxDecode: bias len {} != N {n}", bias.len());
        }

        // --- fold: W''[k][n] = gamma[k] * W[k][n] ---
        let gamma = match &norm {
            Norm::Rms { gamma, .. } => gamma,
            Norm::Ln { gamma, .. } => gamma,
        };
        assert_eq!(gamma.len(), k, "ctxDecode: gamma len {} != K {k}", gamma.len());
        let mut wpp = Array2::<f32>::zeros((k, n));
        for kk in 0..k {
            let g = gamma[kk];
            for nn in 0..n {
                wpp[[kk, nn]] = g * w[[kk, nn]];
            }
        }

        // --- folded bias: LN bias'[n] = sum_k beta[k]*W[k][n] + bias[n]; RMS bias' = bias ---
        let bias_p: Vec<f32> = match &norm {
            Norm::Ln { beta, .. } => {
                assert_eq!(beta.len(), k, "ctxDecode: beta len {} != K {k}", beta.len());
                (0..n)
                    .map(|nn| {
                        let s: f32 = (0..k).map(|kk| beta[kk] * w[[kk, nn]]).sum();
                        s + bias.get(nn).copied().unwrap_or(0.0)
                    })
                    .collect()
            }
            Norm::Rms { .. } => {
                if bias.is_empty() {
                    vec![0.0; n]
                } else {
                    bias.to_vec()
                }
            }
        };

        // Reuse register_weight's residency for W'' (DecodeEpi::None — bias' is host-added in
        // fused_norm_gemv, not in gemv, since the folded bias may differ from a plain weight bias).
        let dw = self.register_weight(&wpp, DecodeEpi::None, &[]);

        FusedWeight { dw, norm, bias_p }
    }

    /// Run the fused pre-norm + projection for a registered [`FusedWeight`]: normalize `x` (the M=1
    /// reduction over K — `inv_rms·x` for RMS, `inv_std·(x−mean)` for LN), dispatch the resident GEMV
    /// on `W''` (one dispatch), and host-add the folded `bias'`. Returns the length-N result.
    pub fn fused_norm_gemv(&self, fw: &FusedWeight, x: &[f32]) -> Result<Vec<f32>, String> {
        let k = fw.dw.k;
        assert_eq!(x.len(), k, "ctxDecode: x len {} != weight K {k}", x.len());

        // Input normalization (f32; the bf16-long-reduction lesson: reduce in f32).
        let x_norm: Vec<f32> = match &fw.norm {
            Norm::Rms { eps, .. } => {
                let ms = x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / k as f64;
                let inv = 1.0 / (ms + *eps as f64).sqrt();
                x.iter().map(|&v| (v as f64 * inv) as f32).collect()
            }
            Norm::Ln { eps, .. } => {
                let mean = x.iter().map(|&v| v as f64).sum::<f64>() / k as f64;
                let var = x.iter().map(|&v| (v as f64 - mean) * (v as f64 - mean)).sum::<f64>()
                    / k as f64;
                let inv = 1.0 / (var + *eps as f64).sqrt();
                x.iter().map(|&v| ((v as f64 - mean) * inv) as f32).collect()
            }
        };

        // x_norm @ W'' on device (the resident GEMV), then host-add the folded bias'.
        let mut out = self.gemv(&fw.dw, &x_norm)?;
        for (o, &b) in out.iter_mut().zip(fw.bias_p.iter()) {
            *o += b;
        }
        Ok(out)
    }

    /// Collapsed decoder SELF-attention sublayer: fused LN+QKV → on-chip MHA → O projection, run as
    /// ONE host call with the q/ctx intermediates threaded buffer-to-buffer (no standalone host
    /// `Vec`s, no caller-visible round-trip between the three dispatches). Numerically BYTE-IDENTICAL
    /// to the M1.a sequence `fused_norm_gemv(qkv) → attn → gemv(self_out)` — same dispatches, same
    /// bf16 packing, same f32 epilogues; only the host marshaling between stages is removed.
    ///
    /// `qkv` is the fused LN+QKV weight `[768, 2304]` (q|k|v concat). `self_out` is the O projection
    /// `[768, 768]` (`DecodeEpi::Bias`). `x` is the residual-stream row `[768]`. `self_k`/`self_v` are
    /// the growing self-KV caches `[s, 768]` row-major; this method APPENDS the new step's k/v (so the
    /// caller must NOT extend them itself) and then attends over the full `s = n_self_before + 1` rows.
    /// Returns the self-attn output `[768]` to add to the residual.
    ///
    /// ## Why this is BO-chaining's limit on our stack (the collapse mechanism note)
    /// The three dispatches use THREE distinct xclbins (QKV gemv / mha_decode / O gemv), each its own
    /// hw-context with its own arg group-ids and — decisively — INCOMPATIBLE buffer LAYOUTS at every
    /// seam: QKV emits f32 row-0-of-`[64,2304]`; attn wants q as bf16 head-major `[12,64]`; attn emits
    /// f32 ctx`[768]`; O wants bf16 row-0-of-`[64,768]`. So the inter-stage transform (f32↔bf16 +
    /// re-layout) is INTRINSIC host compute, not removable by keeping a BO device-resident. Unlike the
    /// ctx2 FFN BO-chain (which keeps H resident across mm1→mm2 on the SAME resident kernel), there is
    /// no same-kernel residency to exploit across these three kernels. This method therefore collapses
    /// the *host marshaling* (intermediate allocations + the caller-visible Vec hops), which is the
    /// available win; true device-resident chaining would require a single stitched multi-launch ELF
    /// (M1.b, separate task) or an ERT_CMD_CHAIN/runlist shim (not present in npu-xrt).
    #[allow(clippy::too_many_arguments)]
    pub fn self_attn_chained(
        &self,
        qkv: &FusedWeight,
        self_out: &DecodeWeight,
        x: &[f32],
        self_k: &mut Vec<f32>,
        self_v: &mut Vec<f32>,
        n_self_before: usize,
    ) -> Result<Vec<f32>, String> {
        debug_assert_eq!(qkv.dw.n, 3 * ATTN_D, "self_attn_chained: qkv N must be 3*768");
        // Stage 1: fused LN+QKV (one dispatch). Identical to the M1.a `fused_norm_gemv` call.
        let qkv_out = self.fused_norm_gemv(qkv, x)?; // [2304] = q|k|v
        // Append this step's k/v to the caches IN PLACE (the caller no longer round-trips k/v through
        // its own Vecs). Byte-identical to `extend_from_slice(&k); extend_from_slice(&v)` in M1.a.
        self_k.extend_from_slice(&qkv_out[ATTN_D..2 * ATTN_D]);
        self_v.extend_from_slice(&qkv_out[2 * ATTN_D..3 * ATTN_D]);
        let s = n_self_before + 1;

        // Stage 2: on-chip MHA (one dispatch). q is passed as a SLICE of qkv_out — no standalone Vec.
        let ctx = self.attn(&qkv_out[0..ATTN_D], self_k, self_v, s)?; // [768] f32 ctx

        // Stage 3: O projection (one dispatch). ctx flows straight in as the gemv activation — no
        // caller-visible ctx_vec hop. `gemv` applies `self_out`'s bias epilogue (DecodeEpi::Bias).
        self.gemv(self_out, &ctx)
    }
}

/// A registered fused pre-norm + projection weight: the folded `W''` resident GEMV weight + the
/// folded `bias'` (host post-add) + the norm params (the M=1 input reduction). Built by
/// [`CtxDecode::register_fused`], consumed by [`CtxDecode::fused_norm_gemv`].
pub struct FusedWeight {
    dw: DecodeWeight,
    norm: Norm,
    /// Folded bias: LN `β@W + bias`; RMS `bias` (or zeros). Length N.
    bias_p: Vec<f32>,
}

impl FusedWeight {
    pub fn k(&self) -> usize {
        self.dw.k
    }
    pub fn n(&self) -> usize {
        self.dw.n
    }
    pub fn n_pad(&self) -> usize {
        self.dw.n_pad
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tiny deterministic LCG -> ~[-0.5, 0.5) values (matches decode_gemv_probe's generator).
    fn lcg_fill(buf: &mut [f32], seed: u32) {
        let mut s = seed;
        for v in buf.iter_mut() {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *v = ((s >> 8) as f32 / u32::MAX as f32) - 0.5;
        }
    }

    /// Device-gated parity: on-NPU GEMV vs host f32 matmul `x·W`, [768,768], rel-L2 <= 0.08.
    /// Requires the NPU (single-tenant: stop npu-asr/voxd first) and the 768x768 xclbin.
    /// Run with:  cargo test -p npu-asr ctx_decode -- --ignored --test-threads=1
    #[test]
    #[ignore]
    fn gemv_parity_768x768() {
        let k = 768usize;
        let n = 768usize;

        let mut w_data = vec![0f32; k * n];
        lcg_fill(&mut w_data, 0x1234_5678);
        let w = Array2::from_shape_vec((k, n), w_data).unwrap();

        let mut x = vec![0f32; k];
        lcg_fill(&mut x, 0x9E37_79B9);

        // host reference: y = x · W  (length n)
        let x_arr = Array1::from_vec(x.clone());
        let y_ref = x_arr.dot(&w); // [n]

        // repo root = two levels up from this crate (rust/npu-asr) — that's where the `mlir-aie`
        // symlink (and the whole_array build dir) lives.
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let root = root.as_path();
        let dev = Rc::new(Device::open(0).expect("open NPU (stop npu-asr.service/voxd.service first)"));
        let mut dec = CtxDecode::new(&dev, root);
        let dw = dec.register_weight(&w, DecodeEpi::None, &[]);
        let y_npu = dec.gemv(&dw, &x).expect("gemv");

        assert_eq!(y_npu.len(), n);
        let mut num = 0f64;
        let mut den = 0f64;
        for i in 0..n {
            let d = (y_npu[i] - y_ref[i]) as f64;
            num += d * d;
            den += (y_ref[i] as f64) * (y_ref[i] as f64);
        }
        let rel_l2 = (num / den).sqrt();
        eprintln!("[ctxDecode parity] rel-L2 = {rel_l2:.4e} (threshold 0.08), n={n}");
        assert!(rel_l2 <= 0.08, "rel-L2 {rel_l2:.4e} exceeds 0.08");
    }

    /// Device-gated parity for the FUSED LN path: `LayerNorm(x)@W + bias` (host golden) vs
    /// `register_fused(W, Ln{γ,β,eps}, bias)` + `fused_norm_gemv`, [768,768], rel-L2 <= 0.08.
    /// Requires the NPU (single-tenant) and the 768x768 xclbin.
    /// Run with:  cargo test -p npu-asr fused_ln_parity -- --ignored --test-threads=1
    #[test]
    #[ignore]
    fn fused_ln_parity_768x768() {
        let (k, n) = (768usize, 768usize);
        let eps = 1e-5f32;

        let mut w_data = vec![0f32; k * n];
        lcg_fill(&mut w_data, 0x1234_5678);
        let w = Array2::from_shape_vec((k, n), w_data).unwrap();
        let mut x = vec![0f32; k];
        lcg_fill(&mut x, 0x9E37_79B9);
        let mut gamma = vec![0f32; k];
        lcg_fill(&mut gamma, 0xABCD_1234);
        for g in gamma.iter_mut() {
            *g += 1.0;
        }
        let mut beta = vec![0f32; k];
        lcg_fill(&mut beta, 0x5555_AAAA);
        let mut bias = vec![0f32; n];
        lcg_fill(&mut bias, 0x0F0F_F0F0);

        // host golden: y = (LayerNorm(x; gamma, beta, eps)) @ W + bias
        let mean = x.iter().map(|&v| v as f64).sum::<f64>() / k as f64;
        let var =
            x.iter().map(|&v| (v as f64 - mean) * (v as f64 - mean)).sum::<f64>() / k as f64;
        let inv = 1.0 / (var + eps as f64).sqrt();
        let xn: Vec<f64> =
            (0..k).map(|kk| (x[kk] as f64 - mean) * inv * gamma[kk] as f64 + beta[kk] as f64).collect();
        let mut y_ref = vec![0f64; n];
        for nn in 0..n {
            let mut s = 0f64;
            for kk in 0..k {
                s += xn[kk] * w[[kk, nn]] as f64;
            }
            y_ref[nn] = s + bias[nn] as f64;
        }

        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let dev = Rc::new(Device::open(0).expect("open NPU (stop npu-asr/voxd first)"));
        let mut dec = CtxDecode::new(&dev, root.as_path());
        let fw = dec.register_fused(&w, Norm::Ln { gamma, beta, eps }, &bias);
        let y_npu = dec.fused_norm_gemv(&fw, &x).expect("fused_norm_gemv");

        assert_eq!(y_npu.len(), n);
        let mut num = 0f64;
        let mut den = 0f64;
        for i in 0..n {
            let d = y_npu[i] as f64 - y_ref[i];
            num += d * d;
            den += y_ref[i] * y_ref[i];
        }
        let rel_l2 = (num / den).sqrt();
        eprintln!("[ctxDecode fused-LN parity] rel-L2 = {rel_l2:.4e} (threshold 0.08), n={n}");
        assert!(rel_l2 <= 0.08, "fused-LN rel-L2 {rel_l2:.4e} exceeds 0.08");
    }
}
