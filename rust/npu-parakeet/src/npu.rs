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
    streams: RefCell<HashMap<usize, Rc<NStream>>>, // N -> stream
    wcache: RefCell<HashMap<String, Rc<Bo>>>,      // packed weight BOs by id
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
            let mut chosen = None;
            for n in ["4096", "2048", "1024"] {
                let cand = base.join(format!("final_512x1024x{n}_{tile}_8c.xclbin"));
                if cand.exists() {
                    chosen = Some(cand);
                    break;
                }
            }
            chosen.unwrap_or_else(|| base.join(format!("final_512x1024x4096_{tile}_8c.xclbin")))
        };
        eprintln!("[npu] resident xclbin = {}", xclbin.display());
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
            streams: RefCell::new(HashMap::new()),
            wcache: RefCell::new(HashMap::new()),
            stats: RefCell::new(NpuStats::default()),
        }
    }

    fn stream(&self, n: usize) -> Rc<NStream> {
        if let Some(s) = self.streams.borrow().get(&n) {
            return s.clone();
        }
        let g = |i| self.kern.group_id(i).unwrap();
        let insts = self.base.join(format!("insts_512x1024x{n}_{}_8c.txt", self.tile));
        let bytes = std::fs::read(&insts).unwrap_or_else(|e| panic!("read {}: {e}", insts.display()));
        let n_instr = bytes.len() / 4;
        let instr = self.dev.alloc_bo(&self.kern, bytes.len(), FLAG_CACHEABLE, g(1)).unwrap();
        instr.write_bytes(&bytes).unwrap();
        instr.sync_to_device().unwrap();
        let bo_c = self.dev.alloc_bo(&self.kern, PAD_M * n * 4, FLAG_HOST_ONLY, g(5)).unwrap();
        let s = Rc::new(NStream { instr, n_instr, bo_c });
        self.streams.borrow_mut().insert(n, s.clone());
        s
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
        bo
    }

    /// One resident-kernel dispatch: A[m,KRES] (zero-padded) @ wbo[KRES,n] -> C[m,n].
    fn dispatch(&self, a_km: ArrayView2<f32>, wbo: &Bo, n: usize) -> Array2<f32> {
        let m = a_km.nrows();
        let st = self.stream(n);
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
            return self.dispatch(a.view(), &wbo, n);
        }
        // K-split (ff.l2: K=4096, N=1024): 2-slot pipeline — submit partial[i] while accumulating
        // partial[i-1] (mirrors ctx2 forward_pipelined). The partials are independent (summed).
        assert_eq!(k % KRES, 0, "K={k} not a multiple of {KRES}");
        assert_eq!(n, 1024, "K-split path assumes N=1024 (ff.l2)");
        let parts = k / KRES;
        let st = self.stream(n);
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
        let w0 = self.weight_bo(&format!("{id}.0"), b.slice(s![0..KRES, ..]));
        pack_into(&self.slots[0], a.slice(s![.., 0..KRES]));
        let t0 = Instant::now();
        let mut prev_run = submit(&self.slots[0], &w0);
        let mut prev_slot = 0usize;
        let mut acc = Array2::<f32>::zeros((m, n));
        for i in 1..parts {
            let slot = i % 2;
            let wi = self.weight_bo(&format!("{id}.{i}"), b.slice(s![i * KRES..(i + 1) * KRES, ..]));
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
