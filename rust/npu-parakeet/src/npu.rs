//! NPU matmul path (feature `npu`) — ZERO-SWITCH resident design (the production path).
//!
//! All encoder matmuls run on ONE resident fast-BFP16 whole_array xclbin (K=1024, tile 64x32x128,
//! the N=4096 build), dispatched with per-N instruction streams (N=1024/2048/4096) by swapping only
//! the instruction BO — never reloading the array program, so ZERO hw-context switches across the
//! whole encoder (mirrors GigaAM V2 / parakeet-npu-port-estimate). ff.l2's K=4096 is K-split into
//! 4× K=1024 N=1024 partials, host-accumulated (like GigaAM's mm2). Weights packed+synced once
//! (cached); resident A/C/instr BOs allocated once. Fast BFP16_IREE kernel (~2× native).
//!
//! Single-tenant NPU only. Dispatch ABI: run_matmul8(opcode=3, instr, count, A, B, C, tmp, trace).

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::Instant;

use ndarray::prelude::*;
use npu_xrt::{Bo, Device, Kernel, FLAG_CACHEABLE, FLAG_HOST_ONLY};

const PAD_M: usize = 512;
const KRES: usize = 1024; // resident kernel contraction dim
const WA_SUBDIR: &str =
    "mlir-aie/programming_examples/basic/matrix_multiplication/whole_array/build";

/// Per-N instruction stream + its output BO (on the resident kernel).
struct NStream {
    instr: Bo,
    n_instr: usize,
    bo_c: Bo, // [PAD_M, n] f32
}

// Resident relpos-MHA block (STEP=8). One xclbin per encoder frame count T (the kernel bakes
// RELPOS_T); loaded + cached per T from {root}/artifacts/relpos/T<T>/. TQ/KB match the build.
const RELPOS_TQ: usize = 8;
const RELPOS_KB: usize = 43;
const RELPOS_DK: usize = 128; // Parakeet head_dim (kernel bakes DK=128)

/// A loaded per-T resident relpos block: its own xclbin/kernel, instr stream, and reusable
/// QUV/KPV/CTX BOs sized for this T. Dispatched per head via run_dwconv6(3, instr, n, quv, kpv, ctx).
struct RelposK {
    kern: Rc<Kernel>,
    instr: Bo,
    n_instr: usize,
    bo_quv: Bo,
    bo_kpv: Bo,
    bo_ctx: Bo,
    n_qt: usize,   // ceil(T/TQ)
    tp: usize,     // n_kb*KB (k/V padded rows)
    pp: usize,     // n_pb*KB (p padded rows)
    ctx_rows: usize, // n_qt*TQ (CTX readback rows, take [:T])
}

#[derive(Default)]
pub struct NpuStats {
    pub pack_a_s: f64,
    pub dispatch_s: f64,
    pub read_s: f64,
    pub weight_load_s: f64,
    pub accum_s: f64,
    pub calls: usize,
    pub dispatches: usize,
}

/// One pipeline slot: own A/C/tmp/trace so a dispatch in flight isn't clobbered while the host
/// preps the next on the other slot (mirrors ctx2 PipeSlot). C sized for the K-split output N=1024.
struct PipeSlot {
    bo_a: Bo,
    bo_c: Bo,
    bo_tmp: Bo,
    bo_tr: Bo,
}

pub struct NpuMatmul {
    dev: Device,
    base: PathBuf,
    tile: String, // "64x32x128" (fast BFP16, default) or "32x32x32" (native bf16, accurate)
    kern: Rc<Kernel>,
    bo_a: Bo, // [PAD_M, KRES] bf16 (resident, single-dispatch path)
    bo_tmp: Bo,
    bo_tr: Bo,
    slots: Vec<PipeSlot>, // 2-slot ring for the K-split pipeline (output N=1024)
    modal: bool, // resident is the MODAL xclbin (fused f32-out silu/identity epilogue) vs plain matmul
    streams: RefCell<HashMap<(usize, bool), Rc<NStream>>>, // (N, silu) -> stream
    wcache: RefCell<HashMap<String, Rc<Bo>>>,      // packed weight BOs by id
    ncache: RefCell<HashMap<String, usize>>,       // weight N (ncols) by id, paired with wcache
    relpos_dir: PathBuf,                           // {root}/artifacts/relpos (per-T xclbin cache)
    relpos: RefCell<HashMap<usize, Rc<RelposK>>>,  // T -> loaded resident block
    pub stats: RefCell<NpuStats>,
}

impl NpuMatmul {
    pub fn open(root: &Path) -> Self {
        let dev = Device::open(0).expect("open NPU (single-tenant: stop npu-asr/voxd)");
        let base = root.join(WA_SUBDIR);
        // resident kernel tile: fast BFP16 64x32x128 (default) or native bf16 32x32x32 (NPU_NATIVE=1)
        let tile = if std::env::var("NPU_NATIVE").is_ok() { "32x32x32" } else { "64x32x128" }.to_string();
        // resident xclbin = a K=1024 whole_array kernel for this tile. The array program is
        // N-independent (per-N differs only in the runtime instruction stream, swapped per
        // dispatch), so ANY surviving N works as the resident. Prefer the largest N present;
        // fall back to a smaller surviving build (the N=4096/2048 twins were deleted by the
        // an earlier occupancy run; N=1024 survives). Env NPU_RESIDENT_XCLBIN overrides.
        let xclbin = if let Ok(p) = std::env::var("NPU_RESIDENT_XCLBIN") {
            PathBuf::from(p)
        } else {
            // A1 (ff_act on-chip): prefer the MODAL resident xclbin (fused f32-out epilogue; the
            // per-inst-stream RTP selects silu@N=4096 / identity elsewhere -> the FFN SiLU runs on
            // chip with zero extra hw-context switches). Fall back to the plain matmul xclbin if the
            // modal build is absent (then `modal=false` and the host keeps applying silu).
            let modal = base.join(format!("final_512x1024x4096_{tile}_8c_modalsilu.xclbin"));
            if modal.exists() {
                modal
            } else {
                let mut chosen = None;
                for n in ["4096", "2048", "1024"] {
                    let cand = base.join(format!("final_512x1024x{n}_{tile}_8c.xclbin"));
                    if cand.exists() {
                        chosen = Some(cand);
                        break;
                    }
                }
                chosen.unwrap_or_else(|| base.join(format!("final_512x1024x4096_{tile}_8c.xclbin")))
            }
        };
        // The modal resident bakes the silu/identity epilogue; the plain one does not (host silu).
        let modal = xclbin.file_name().and_then(|s| s.to_str()).is_some_and(|s| s.contains("modal"));
        eprintln!("[npu] resident xclbin = {} (modal={modal})", xclbin.display());
        let kern = dev
            .load_kernel(xclbin.to_str().unwrap(), None)
            .unwrap_or_else(|e| panic!("load resident {}: {e:?}", xclbin.display()));
        let g = |i| kern.group_id(i).unwrap();
        let bo_a = dev.alloc_bo(&kern, PAD_M * KRES * 2, FLAG_HOST_ONLY, g(3)).unwrap();
        let bo_tmp = dev.alloc_bo(&kern, 1, FLAG_HOST_ONLY, g(6)).unwrap();
        let bo_tr = dev.alloc_bo(&kern, 4, FLAG_HOST_ONLY, g(7)).unwrap();
        // 2-slot ring for the K-split pipeline (ff.l2 output N=1024)
        let slots = (0..2)
            .map(|_| PipeSlot {
                bo_a: dev.alloc_bo(&kern, PAD_M * KRES * 2, FLAG_HOST_ONLY, g(3)).unwrap(),
                bo_c: dev.alloc_bo(&kern, PAD_M * 1024 * 4, FLAG_HOST_ONLY, g(5)).unwrap(),
                bo_tmp: dev.alloc_bo(&kern, 1, FLAG_HOST_ONLY, g(6)).unwrap(),
                bo_tr: dev.alloc_bo(&kern, 4, FLAG_HOST_ONLY, g(7)).unwrap(),
            })
            .collect();
        NpuMatmul {
            dev,
            base,
            tile,
            kern,
            bo_a,
            bo_tmp,
            bo_tr,
            slots,
            modal,
            streams: RefCell::new(HashMap::new()),
            wcache: RefCell::new(HashMap::new()),
            ncache: RefCell::new(HashMap::new()),
            relpos_dir: root.join("artifacts/relpos"),
            relpos: RefCell::new(HashMap::new()),
            stats: RefCell::new(NpuStats::default()),
        }
    }

    /// Load (and cache) the resident relpos block for encoder frame count `t`. The xclbin bakes
    /// RELPOS_T, so there is one per T under {root}/artifacts/relpos/T<t>/ (pre-build with
    /// scripts/relpos_prebuild.sh). Panics with a build hint if the artifacts are missing.
    fn relpos_kern(&self, t: usize) -> Rc<RelposK> {
        if let Some(k) = self.relpos.borrow().get(&t) {
            return k.clone();
        }
        let p = 2 * t - 1;
        let cdiv = |a: usize, b: usize| (a + b - 1) / b;
        let n_qt = cdiv(t, RELPOS_TQ);
        let tp = cdiv(t, RELPOS_KB) * RELPOS_KB;
        let pp = cdiv(p, RELPOS_KB) * RELPOS_KB;
        let ctx_rows = n_qt * RELPOS_TQ;
        let dir = self.relpos_dir.join(format!("T{t}"));
        let xclbin = dir.join("final.xclbin");
        let insts = dir.join("insts.bin");
        let kern = self
            .dev
            .load_kernel(xclbin.to_str().unwrap(), None)
            .unwrap_or_else(|e| panic!("load relpos T={t} ({}): {e:?}\n  pre-build: scripts/relpos_prebuild.sh {t}", xclbin.display()));
        let ib = std::fs::read(&insts).unwrap_or_else(|e| panic!("read {}: {e}", insts.display()));
        let n_instr = ib.len() / 4;
        let g = |i| kern.group_id(i).unwrap();
        let instr = self.dev.alloc_bo(&kern, ib.len(), FLAG_CACHEABLE, g(1)).unwrap();
        instr.write_bytes(&ib).unwrap();
        instr.sync_to_device().unwrap();
        let bo_quv = self.dev.alloc_bo(&kern, 2 * n_qt * RELPOS_TQ * RELPOS_DK * 2, FLAG_HOST_ONLY, g(3)).unwrap();
        let bo_kpv = self.dev.alloc_bo(&kern, (tp + pp + tp) * RELPOS_DK * 2, FLAG_HOST_ONLY, g(4)).unwrap();
        let bo_ctx = self.dev.alloc_bo(&kern, ctx_rows * RELPOS_DK * 2, FLAG_HOST_ONLY, g(5)).unwrap();
        let rk = Rc::new(RelposK { kern, instr, n_instr, bo_quv, bo_kpv, bo_ctx, n_qt, tp, pp, ctx_rows });
        self.relpos.borrow_mut().insert(t, rk.clone());
        rk
    }

    /// Resident relpos-MHA block for ONE head. qu/qv/k [t,DK], p [2t-1,DK], v [t,DK] (f32) ->
    /// ctx [t,DK] (f32). Packs the STEP=8 stream layout (tile-interleaved QUV, padded KPV),
    /// dispatches via the 3-BO ABI, unpacks the bf16 CTX. Mirrors run_npu_relpos_rowtiled.py.
    pub fn relpos_mha(&self, qu: &Array2<f32>, qv: &Array2<f32>, k: &Array2<f32>, p: &Array2<f32>, v: &Array2<f32>) -> Array2<f32> {
        let t = qu.nrows();
        let rk = self.relpos_kern(t);
        let mut quv = Vec::<f32>::with_capacity(2 * rk.n_qt * RELPOS_TQ * RELPOS_DK);
        for q in 0..rk.n_qt {
            let q0 = q * RELPOS_TQ;
            let take = RELPOS_TQ.min(t - q0);
            push_pad_rows(&mut quv, qu, q0, take, RELPOS_TQ);
            push_pad_rows(&mut quv, qv, q0, take, RELPOS_TQ);
        }
        let mut kpv = Vec::<f32>::with_capacity((rk.tp + rk.pp + rk.tp) * RELPOS_DK);
        push_pad_rows(&mut kpv, k, 0, t, rk.tp);
        push_pad_rows(&mut kpv, p, 0, p.nrows(), rk.pp);
        push_pad_rows(&mut kpv, v, 0, t, rk.tp);
        let mut qb = vec![0u16; quv.len()];
        let mut kb = vec![0u16; kpv.len()];
        npu_xrt::pack_f32_to_bf16(&quv, &mut qb);
        npu_xrt::pack_f32_to_bf16(&kpv, &mut kb);
        let t0 = Instant::now();
        rk.bo_quv.write_bytes(u16_bytes(&qb)).unwrap();
        rk.bo_quv.sync_to_device().unwrap();
        rk.bo_kpv.write_bytes(u16_bytes(&kb)).unwrap();
        rk.bo_kpv.sync_to_device().unwrap();
        rk.kern.run_dwconv6(3, &rk.instr, rk.n_instr, &rk.bo_quv, &rk.bo_kpv, &rk.bo_ctx).unwrap();
        rk.bo_ctx.sync_from_device().unwrap();
        {
            let mut s = self.stats.borrow_mut();
            s.dispatch_s += t0.elapsed().as_secs_f64();
            s.dispatches += 1;
        }
        let mut cb = vec![0u8; rk.ctx_rows * RELPOS_DK * 2];
        rk.bo_ctx.read_bytes(&mut cb).unwrap();
        let mut ctx = Array2::<f32>::zeros((t, RELPOS_DK));
        for i in 0..t {
            for d in 0..RELPOS_DK {
                let off = (i * RELPOS_DK + d) * 2;
                let u = u16::from_le_bytes([cb[off], cb[off + 1]]);
                ctx[[i, d]] = f32::from_bits((u as u32) << 16);
            }
        }
        ctx
    }

    /// Per-N instruction stream. On the MODAL resident, `silu` picks the baked-RTP mode: the
    /// `modalsilu` stream (fc1 / ff.l1, N=4096) applies SiLU on chip, `modalid` is a numerically
    /// identity epilogue (every other GEMM). On the plain resident, `silu` is ignored (there is no
    /// on-chip epilogue; the host applies silu) and the classic insts_*_8c.txt stream is used.
    fn stream(&self, n: usize, silu: bool) -> Rc<NStream> {
        let key = (n, silu && self.modal);
        if let Some(s) = self.streams.borrow().get(&key) {
            return s.clone();
        }
        let g = |i| self.kern.group_id(i).unwrap();
        let insts = if self.modal {
            let mode = if silu { "modalsilu" } else { "modalid" };
            self.base.join(format!("insts_512x1024x{n}_{}_8c_{mode}.txt", self.tile))
        } else {
            self.base.join(format!("insts_512x1024x{n}_{}_8c.txt", self.tile))
        };
        let bytes = std::fs::read(&insts).unwrap_or_else(|e| panic!("read {}: {e}", insts.display()));
        let n_instr = bytes.len() / 4;
        let instr = self.dev.alloc_bo(&self.kern, bytes.len(), FLAG_CACHEABLE, g(1)).unwrap();
        instr.write_bytes(&bytes).unwrap();
        instr.sync_to_device().unwrap();
        let bo_c = self.dev.alloc_bo(&self.kern, PAD_M * n * 4, FLAG_HOST_ONLY, g(5)).unwrap();
        let s = Rc::new(NStream { instr, n_instr, bo_c });
        self.streams.borrow_mut().insert(key, s.clone());
        s
    }

    /// True when the resident is the modal xclbin (the NPU applies the FFN SiLU epilogue on chip,
    /// so the host must NOT re-apply it). False on the plain resident / host fallback.
    pub fn modal(&self) -> bool {
        self.modal
    }

    fn weight_bo(&self, id: &str, b_km: ArrayView2<f32>) -> Rc<Bo> {
        if let Some(bo) = self.wcache.borrow().get(id) {
            return bo.clone();
        }
        let t0 = Instant::now();
        let (k, n) = b_km.dim();
        let g4 = self.kern.group_id(4).unwrap();
        let b_std = b_km.as_standard_layout();
        let mut bits = vec![0u16; k * n];
        npu_xrt::pack_f32_to_bf16(b_std.as_slice().unwrap(), &mut bits);
        let bo = self.dev.alloc_bo(&self.kern, k * n * 2, FLAG_HOST_ONLY, g4).unwrap();
        bo.write_bytes(u16_bytes(&bits)).unwrap();
        bo.sync_to_device().unwrap();
        self.stats.borrow_mut().weight_load_s += t0.elapsed().as_secs_f64();
        let bo = Rc::new(bo);
        self.wcache.borrow_mut().insert(id.to_string(), bo.clone());
        self.ncache.borrow_mut().insert(id.to_string(), n);
        bo
    }

    /// One resident-kernel dispatch: A[m,KRES] (zero-padded) @ wbo[KRES,n] -> C[m,n].
    /// `silu=true` (only fc1 / ff.l1 on the modal resident) applies the on-chip SiLU epilogue.
    fn dispatch(&self, a_km: ArrayView2<f32>, wbo: &Bo, n: usize, silu: bool) -> Array2<f32> {
        let m = a_km.nrows();
        let st = self.stream(n, silu);
        let stage = crate::prof::phase::current_stage();
        // (a) input marshaling: pack A -> bf16 + upload (host->device, no math).
        // pack only the m REAL rows of A: matmul row i depends only on A row i, so the kernel's
        // padding rows (m..PAD_M) produce ignored C rows — their (stale) content is harmless.
        let t0 = Instant::now();
        {
            let _m = crate::prof::phase::PhaseScope::new(stage, crate::prof::phase::Bucket::Marshal);
            let a_std = a_km.as_standard_layout();
            let a_s = a_std.as_slice().unwrap();
            let mut a_bits = vec![0u16; m * KRES];
            npu_xrt::pack_f32_to_bf16(&a_s[..m * KRES], &mut a_bits);
            self.bo_a.write_bytes(u16_bytes(&a_bits)).unwrap(); // writes first m rows; rest stale (ignored)
            self.bo_a.sync_to_device().unwrap();
        }
        self.stats.borrow_mut().pack_a_s += t0.elapsed().as_secs_f64();

        // (b) NPU dispatch + wait for completion (run_matmul8 is blocking).
        let t1 = Instant::now();
        {
            let _d = crate::prof::phase::PhaseScope::new(stage, crate::prof::phase::Bucket::Npu);
            self.kern
                .run_matmul8(3, &st.instr, st.n_instr, &self.bo_a, wbo, &st.bo_c, &self.bo_tmp, &self.bo_tr)
                .unwrap();
        }
        {
            let mut s = self.stats.borrow_mut();
            s.dispatch_s += t1.elapsed().as_secs_f64();
            s.dispatches += 1;
        }

        // (c) output marshaling: download C + read rows back into an f32 ndarray (no math).
        let t2 = Instant::now();
        let out = {
            let _m2 = crate::prof::phase::PhaseScope::new(stage, crate::prof::phase::Bucket::Marshal);
            st.bo_c.sync_from_device().unwrap();
            // read only the first m rows (row-major); rows m..PAD_M are padding-row garbage
            let mut c_bytes = vec![0u8; m * n * 4];
            st.bo_c.read_bytes(&mut c_bytes).unwrap();
            let mut out = Array2::<f32>::zeros((m, n));
            for r in 0..m {
                for c in 0..n {
                    let off = (r * n + c) * 4;
                    out[[r, c]] = f32::from_le_bytes([
                        c_bytes[off], c_bytes[off + 1], c_bytes[off + 2], c_bytes[off + 3],
                    ]);
                }
            }
            out
        };
        self.stats.borrow_mut().read_s += t2.elapsed().as_secs_f64();
        out
    }

    /// C[m,n] = A[m,k] @ B[k,n] on the NPU; `id` keys the weight-BO cache. K=1024 dispatches
    /// directly on the resident kernel; K=4096 is K-split into 4× K=1024 partials (host-accumulated).
    pub fn matmul_id(&self, a: &Array2<f32>, b: &Array2<f32>, id: &str) -> Array2<f32> {
        let (m, k) = a.dim();
        let (kb, n) = b.dim();
        assert_eq!(k, kb);
        assert!(m <= PAD_M);
        self.stats.borrow_mut().calls += 1;

        if k == KRES {
            let wbo = self.weight_bo(id, b.view());
            return self.dispatch(a.view(), &wbo, n, false);
        }
        assert_eq!(k % KRES, 0, "K={k} not a multiple of {KRES}");
        assert_eq!(n, 1024, "K-split path assumes N=1024 (ff.l2)");
        let parts = k / KRES;
        // Per-partial weight comes straight from the passed `b` (packed/cached on first touch).
        self.ksplit_dispatch(a, n, parts, |i| {
            self.weight_bo(&format!("{id}.{i}"), b.slice(s![i * KRES..(i + 1) * KRES, ..]))
        })
    }

    /// Lazy variant of [`matmul_id`]: the host weight matrix is materialized by `make_b` ONLY on a
    /// cache miss. When the weight BO(s) are already cached (every warm pass for a constant encoder
    /// weight) `make_b` is never called, so the per-pass host reclone/transpose of the constant
    /// weight is skipped entirely. `id` keys the weight-BO cache (same keying as `matmul_id`, so the
    /// two are interchangeable per call site). `a`'s ncols selects the K path (K=weight nrows).
    pub fn matmul_id_lazy<F: FnOnce() -> Array2<f32>>(&self, a: &Array2<f32>, make_b: F, id: &str) -> Array2<f32> {
        let (m, k) = a.dim();
        assert!(m <= PAD_M);
        self.stats.borrow_mut().calls += 1;

        if k == KRES {
            // single-dispatch path: need the weight BO + its N. On a hit, read N from ncache and
            // never touch make_b; on a miss, build the weight, then pack+cache it.
            let cached = self.wcache.borrow().get(id).cloned();
            let (wbo, n) = if let Some(bo) = cached {
                let n = *self.ncache.borrow().get(id).expect("ncache miss on wcache hit");
                (bo, n)
            } else {
                let b = make_b();
                let n = b.ncols();
                assert_eq!(b.nrows(), KRES, "lazy K={k} weight nrows {} != {KRES}", b.nrows());
                (self.weight_bo(id, b.view()), n)
            };
            return self.dispatch(a.view(), &wbo, n, false);
        }
        // K-split: if ALL of {id}.0..parts-1 are cached, dispatch without make_b; else build `b`
        // once and pack the partials from it (identical to matmul_id's packing).
        assert_eq!(k % KRES, 0, "K={k} not a multiple of {KRES}");
        let parts = k / KRES;
        let all_cached = (0..parts).all(|i| self.wcache.borrow().contains_key(&format!("{id}.{i}")));
        let b_opt: Option<Array2<f32>> = if all_cached { None } else { Some(make_b()) };
        let n = if let Some(ref b) = b_opt {
            assert_eq!(b.nrows(), k, "lazy K-split weight nrows {} != {k}", b.nrows());
            b.ncols()
        } else {
            *self.ncache.borrow().get(&format!("{id}.0")).expect("ncache miss on wcache hit")
        };
        assert_eq!(n, 1024, "K-split path assumes N=1024 (ff.l2)");
        self.ksplit_dispatch(a, n, parts, |i| {
            let pid = format!("{id}.{i}");
            let cached = self.wcache.borrow().get(&pid).cloned();
            if let Some(bo) = cached {
                bo
            } else {
                let b = b_opt.as_ref().expect("b_opt present on cache miss");
                self.weight_bo(&pid, b.slice(s![i * KRES..(i + 1) * KRES, ..]))
            }
        })
    }

    /// Like [`matmul_id_lazy`] but applies the FFN SiLU activation as the on-chip GEMM epilogue
    /// (A1 / `ff_act` on-chip). Only the single-dispatch K=KRES path is supported (fc1 / ff.l1 is
    /// always K=1024, N=4096). On the MODAL resident this dispatches the `modalsilu` stream so
    /// `out = silu(A @ B)` comes back already activated -- the host must NOT re-apply silu. On the
    /// plain resident (`modal=false`) the epilogue is a no-op (`silu` flag ignored by `stream`), so
    /// the caller falls back to host silu; use [`Self::modal`] to branch.
    pub fn matmul_id_lazy_silu<F: FnOnce() -> Array2<f32>>(&self, a: &Array2<f32>, make_b: F, id: &str) -> Array2<f32> {
        let (m, k) = a.dim();
        assert!(m <= PAD_M);
        assert_eq!(k, KRES, "matmul_id_lazy_silu is single-dispatch only (fc1 K={KRES})");
        self.stats.borrow_mut().calls += 1;
        let cached = self.wcache.borrow().get(id).cloned();
        let (wbo, n) = if let Some(bo) = cached {
            let n = *self.ncache.borrow().get(id).expect("ncache miss on wcache hit");
            (bo, n)
        } else {
            let b = make_b();
            let n = b.ncols();
            assert_eq!(b.nrows(), KRES, "lazy-silu K={k} weight nrows {} != {KRES}", b.nrows());
            (self.weight_bo(id, b.view()), n)
        };
        self.dispatch(a.view(), &wbo, n, true)
    }

    /// K-split 2-slot pipeline (ff.l2: K=4096, N=1024): submit partial[i] while accumulating
    /// partial[i-1] (mirrors ctx2 forward_pipelined). Partials are independent (summed). `get_w(i)`
    /// yields the cached/packed weight BO for partial i (lazy per-partial so the first partial's
    /// pack overlaps nothing, matching the original). Numerics identical across callers.
    fn ksplit_dispatch<G: Fn(usize) -> Rc<Bo>>(&self, a: &Array2<f32>, n: usize, parts: usize, get_w: G) -> Array2<f32> {
        let m = a.nrows();
        let st = self.stream(n, false); // ff.l2 K-split output has no activation (identity epilogue)
        // Phase-timing stage label for this K-split op; each partial's pack/read is Marshal and
        // each dispatch-launch + wait is Npu (the pipeline overlaps them, so per-bucket wall
        // sums may exceed e2e — report() surfaces that as overlap_ms).
        let stage = crate::prof::phase::current_stage();

        let pack_into = |slot: &PipeSlot, a_p: ArrayView2<f32>| {
            let _m = crate::prof::phase::PhaseScope::new(stage, crate::prof::phase::Bucket::Marshal);
            let a_std = a_p.as_standard_layout();
            let mut bits = vec![0u16; m * KRES];
            npu_xrt::pack_f32_to_bf16(&a_std.as_slice().unwrap()[..m * KRES], &mut bits);
            slot.bo_a.write_bytes(u16_bytes(&bits)).unwrap();
            slot.bo_a.sync_to_device().unwrap();
        };
        let read_part = |slot: &PipeSlot| -> Array2<f32> {
            let _m = crate::prof::phase::PhaseScope::new(stage, crate::prof::phase::Bucket::Marshal);
            slot.bo_c.sync_from_device().unwrap();
            let mut cb = vec![0u8; m * n * 4];
            slot.bo_c.read_bytes(&mut cb).unwrap();
            let mut out = Array2::<f32>::zeros((m, n));
            for r in 0..m {
                for c in 0..n {
                    let off = (r * n + c) * 4;
                    out[[r, c]] = f32::from_le_bytes([cb[off], cb[off + 1], cb[off + 2], cb[off + 3]]);
                }
            }
            out
        };
        let submit = |slot: &PipeSlot, wbo: &Bo| {
            let _d = crate::prof::phase::PhaseScope::new(stage, crate::prof::phase::Bucket::Npu);
            self.stats.borrow_mut().dispatches += 1;
            self.kern
                .run_matmul8_start(3, &st.instr, st.n_instr, &slot.bo_a, wbo, &slot.bo_c, &slot.bo_tmp, &slot.bo_tr)
                .unwrap()
        };

        // submit partial 0
        let w0 = get_w(0);
        pack_into(&self.slots[0], a.slice(s![.., 0..KRES]));
        let t0 = Instant::now();
        let mut prev_run = submit(&self.slots[0], &w0);
        let mut prev_slot = 0usize;
        let mut acc = Array2::<f32>::zeros((m, n));
        for i in 1..parts {
            let slot = i % 2;
            let wi = get_w(i);
            pack_into(&self.slots[slot], a.slice(s![.., i * KRES..(i + 1) * KRES])); // overlaps prev NPU exec
            let cur_run = submit(&self.slots[slot], &wi);
            {
                let _d = crate::prof::phase::PhaseScope::new(stage, crate::prof::phase::Bucket::Npu);
                prev_run.wait().unwrap();
            }
            acc += &read_part(&self.slots[prev_slot]); // overlaps cur NPU exec
            prev_run = cur_run;
            prev_slot = slot;
        }
        {
            let _d = crate::prof::phase::PhaseScope::new(stage, crate::prof::phase::Bucket::Npu);
            prev_run.wait().unwrap();
        }
        acc += &read_part(&self.slots[prev_slot]);
        self.stats.borrow_mut().dispatch_s += t0.elapsed().as_secs_f64();
        acc
    }
}

fn u16_bytes(v: &[u16]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 2) }
}

/// Append `take` rows of `m` (starting at row `start`) to `dst`, then zero-pad to `n_total` rows
/// (each row `m.ncols()` wide). Used to build the STEP=8 QUV/KPV packing (ragged tiles + block pad).
fn push_pad_rows(dst: &mut Vec<f32>, m: &Array2<f32>, start: usize, take: usize, n_total: usize) {
    let dk = m.ncols();
    for r in 0..take {
        dst.extend(m.row(start + r).iter().copied());
    }
    dst.extend(std::iter::repeat(0.0f32).take((n_total - take) * dk));
}
