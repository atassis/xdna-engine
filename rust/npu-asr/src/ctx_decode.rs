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
use npu_xrt::{pack_f32_to_bf16, Bo, Device, Kernel, FLAG_CACHEABLE, FLAG_HOST_ONLY};

/// Smallest legal M for the native-bf16 8-col whole_array design (see build_decode_kernels.sh).
const M: usize = 64;
const TILE: &str = "8x32x32"; // m x k x n
const COLS: usize = 8;
const WA: &str = "mlir-aie/programming_examples/basic/matrix_multiplication/whole_array/build";

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

/// Resident on-NPU GEMV primitive for the decoder. Holds one [`ShapeKernel`] per (K, N_pad) shape.
pub struct CtxDecode {
    dev: Rc<Device>,
    root: PathBuf,
    shapes: RefCell<HashMap<(usize, usize), Rc<ShapeKernel>>>,
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
}
