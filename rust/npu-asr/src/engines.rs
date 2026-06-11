//! NPU engines: the weight-bound whole-array fused matmul (`WAEpilogue`) and the depthwise-conv
//! (`DwconvEngine`). Mirrors `npu_asr/fused.py` (WAEpilogue) and `npu_asr/ops.py` (DwconvEngine).
//!
//! WAEpilogue is WEIGHT-BOUND: the K-augmented weight `B_aug` and the instruction buffer are built
//! and synced ONCE in `new`; the activation/output buffers are allocated once and reused, so
//! `forward` only writes the activation tile and dispatches (task-15 levers 1+2).

use std::cell::RefCell;
use std::path::Path;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use ndarray::prelude::*;
use npu_xrt::{bf16_bits_to_f32, f32_to_bf16_bits, Bo, Device, Kernel, FLAG_CACHEABLE, FLAG_HOST_ONLY};
use rayon::prelude::*;

pub const WA_SUBDIR: &str =
    "mlir-aie/programming_examples/basic/matrix_multiplication/whole_array/build";
pub const DW_SUBDIR: &str = "mlir-aie/programming_examples/ml/dwconv1d/build";

pub const PAD_M: usize = 512;
const TILE: usize = 32;

// --- dispatch profiling (cumulative NPU .wait() time + dispatch count) ---
static NPU_NS: AtomicU64 = AtomicU64::new(0);
static NPU_DISP: AtomicU64 = AtomicU64::new(0);

pub fn reset_prof() {
    NPU_NS.store(0, Ordering::Relaxed);
    NPU_DISP.store(0, Ordering::Relaxed);
}
/// (seconds spent in NPU dispatch, number of dispatches)
pub fn prof() -> (f64, u64) {
    (
        NPU_NS.load(Ordering::Relaxed) as f64 / 1e9,
        NPU_DISP.load(Ordering::Relaxed),
    )
}
fn record(dt: std::time::Duration) {
    NPU_NS.fetch_add(dt.as_nanos() as u64, Ordering::Relaxed);
    NPU_DISP.fetch_add(1, Ordering::Relaxed);
}

fn u16_bytes(v: &[u16]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 2) }
}

fn read_instr_words(path: &Path) -> (Vec<u8>, usize) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read instr {}: {e}", path.display()));
    let words = bytes.len() / 4;
    (bytes, words)
}

/// Weight-bound whole-array fused matmul with on-chip bias(+SiLU). bias rides a K-augmented
/// extra k-block. Output is bf16 (the epilogue narrows f32->bf16 on-chip).
pub struct WAEpilogue {
    dev: Rc<Device>,
    kern: Rc<Kernel>,
    k: usize,
    n: usize,
    kaug: usize,
    n_instr: usize,
    bo_instr: Bo,
    bo_b: Bo, // K-augmented weight (synced once)
    bo_a: Bo, // activation (reused)
    bo_c: Bo, // output bf16 (reused)
    bo_tmp: Bo,
    bo_tr: Bo,
    a_aug: RefCell<Vec<u16>>, // host activation buffer, col K preset to bf16(1.0)
    cbuf: RefCell<Vec<u8>>,   // reused output read buffer
    inited: std::cell::Cell<bool>,
}

impl WAEpilogue {
    /// `mode` = "silu" or "bias". `b_real` is [K, N]; `bias` is length N.
    pub fn new(
        dev: Rc<Device>,
        root: &Path,
        mode: &str,
        k: usize,
        n: usize,
        b_real: &Array2<f32>,
        bias: &[f32],
    ) -> Self {
        assert!(mode == "silu" || mode == "bias");
        let kaug = k + TILE;
        let suffix = format!("{PAD_M}x{kaug}x{n}_{TILE}x{TILE}x{TILE}_8c_{mode}");
        let wa = root.join(WA_SUBDIR);
        let xclbin = wa.join(format!("final_{suffix}.xclbin"));
        let insts = wa.join(format!("insts_{suffix}.txt"));
        let kern = dev
            .load_kernel(xclbin.to_str().unwrap(), None)
            .unwrap_or_else(|e| panic!("load {}: {e}", xclbin.display()));
        let (instr_bytes, n_instr) = read_instr_words(&insts);

        // constant K-augmented weight, built once: rows 0..K = weight, row K = bias, rest 0
        let mut b_bits = vec![0u16; kaug * n];
        for kk in 0..k {
            for nn in 0..n {
                b_bits[kk * n + nn] = f32_to_bf16_bits(b_real[[kk, nn]]);
            }
        }
        for nn in 0..n {
            b_bits[k * n + nn] = f32_to_bf16_bits(bias[nn]);
        }

        let g_instr = kern.group_id(1).unwrap();
        let g_a = kern.group_id(3).unwrap();
        let g_b = kern.group_id(4).unwrap();
        let g_c = kern.group_id(5).unwrap();
        let g_tmp = kern.group_id(6).unwrap();
        let g_tr = kern.group_id(7).unwrap();

        let bo_instr = dev.alloc_bo(&kern, instr_bytes.len(), FLAG_CACHEABLE, g_instr).unwrap();
        bo_instr.write_bytes(&instr_bytes).unwrap();
        bo_instr.sync_to_device().unwrap();
        let bo_b = dev.alloc_bo(&kern, b_bits.len() * 2, FLAG_HOST_ONLY, g_b).unwrap();
        bo_b.write_bytes(u16_bytes(&b_bits)).unwrap();
        bo_b.sync_to_device().unwrap();
        let bo_a = dev.alloc_bo(&kern, PAD_M * kaug * 2, FLAG_HOST_ONLY, g_a).unwrap();
        let bo_c = dev.alloc_bo(&kern, PAD_M * n * 2, FLAG_HOST_ONLY, g_c).unwrap();
        let bo_tmp = dev.alloc_bo(&kern, 1, FLAG_HOST_ONLY, g_tmp).unwrap();
        let bo_tr = dev.alloc_bo(&kern, 4, FLAG_HOST_ONLY, g_tr).unwrap();

        // host activation buffer: zeros, col K preset to bf16(1.0) for ALL rows (the bias rider)
        let mut a_aug = vec![0u16; PAD_M * kaug];
        let one = f32_to_bf16_bits(1.0);
        for r in 0..PAD_M {
            a_aug[r * kaug + k] = one;
        }

        WAEpilogue {
            dev,
            kern,
            k,
            n,
            kaug,
            n_instr,
            bo_instr,
            bo_b,
            bo_a,
            bo_c,
            bo_tmp,
            bo_tr,
            a_aug: RefCell::new(a_aug),
            cbuf: RefCell::new(vec![0u8; PAD_M * n * 2]),
            inited: std::cell::Cell::new(false),
        }
    }

    /// a_real is [Mp, K] (Mp <= PAD_M). Returns [Mp, N] f32.
    pub fn forward(&self, a_real: &Array2<f32>) -> Array2<f32> {
        let (mp, kk) = a_real.dim();
        assert_eq!(kk, self.k);
        assert!(mp <= PAD_M);
        {
            let mut a = self.a_aug.borrow_mut();
            let (k, kaug) = (self.k, self.kaug);
            // convert this call's `mp` real rows f32->bf16 in parallel (disjoint row chunks)
            a.par_chunks_mut(kaug).take(mp).enumerate().for_each(|(r, row)| {
                for c in 0..k {
                    row[c] = f32_to_bf16_bits(a_real[[r, c]]);
                }
            });
            // First call writes the full buffer (sets the constant padding rows + bias column);
            // later calls only need to refresh the `mp` real rows.
            if self.inited.get() {
                self.bo_a.write_bytes(&u16_bytes(&a)[..mp * self.kaug * 2]).unwrap();
            } else {
                self.bo_a.write_bytes(u16_bytes(&a)).unwrap();
                self.inited.set(true);
            }
        }
        self.bo_a.sync_to_device().unwrap();

        let t0 = Instant::now();
        self.kern
            .run_matmul8(
                3,
                &self.bo_instr,
                self.n_instr,
                &self.bo_a,
                &self.bo_b,
                &self.bo_c,
                &self.bo_tmp,
                &self.bo_tr,
            )
            .unwrap();
        record(t0.elapsed());

        self.bo_c.sync_from_device().unwrap();
        // output is bf16 row-major [M,N]; the first mp rows are the first mp*N elements
        let n = self.n;
        {
            let mut cb = self.cbuf.borrow_mut();
            self.bo_c.read_bytes(&mut cb[..mp * n * 2]).unwrap();
        }
        let cb = self.cbuf.borrow();
        let bytes: &[u8] = &cb[..mp * n * 2];
        // bf16 -> f32 in parallel over all mp*N elements
        let data: Vec<f32> = (0..mp * n)
            .into_par_iter()
            .map(|i| {
                let idx = i * 2;
                bf16_bits_to_f32(u16::from_le_bytes([bytes[idx], bytes[idx + 1]]))
            })
            .collect();
        Array2::from_shape_vec((mp, n), data).unwrap()
    }
}

/// depthwise conv1d k=5 'same' on [768,400]; one channel per ObjectFifo tile.
pub struct DwconvEngine {
    dev: Rc<Device>,
    kern: Rc<Kernel>,
    n_instr: usize,
    bo_instr: Bo,
    bo_x: Bo,
    bo_w: Bo,
    bo_y: Bo,
    ch: usize,
    t: usize,
}

impl DwconvEngine {
    pub fn new(dev: Rc<Device>, root: &Path, ch: usize, t: usize) -> Self {
        let dw = root.join(DW_SUBDIR);
        let xclbin = dw.join("final.xclbin");
        let insts = dw.join("insts.bin");
        let kern = dev
            .load_kernel(xclbin.to_str().unwrap(), None)
            .unwrap_or_else(|e| panic!("load {}: {e}", xclbin.display()));
        let (instr_bytes, n_instr) = read_instr_words(&insts);

        let g_instr = kern.group_id(1).unwrap();
        let g_x = kern.group_id(3).unwrap();
        let g_w = kern.group_id(4).unwrap();
        let g_y = kern.group_id(5).unwrap();

        let bo_instr = dev.alloc_bo(&kern, instr_bytes.len(), FLAG_CACHEABLE, g_instr).unwrap();
        bo_instr.write_bytes(&instr_bytes).unwrap();
        bo_instr.sync_to_device().unwrap();
        let bo_x = dev.alloc_bo(&kern, ch * t * 2, FLAG_HOST_ONLY, g_x).unwrap();
        let bo_w = dev.alloc_bo(&kern, ch * 16 * 2, FLAG_HOST_ONLY, g_w).unwrap();
        let bo_y = dev.alloc_bo(&kern, ch * t * 2, FLAG_HOST_ONLY, g_y).unwrap();

        DwconvEngine {
            dev,
            kern,
            n_instr,
            bo_instr,
            bo_x,
            bo_w,
            bo_y,
            ch,
            t,
        }
    }

    /// x is [ch, T] f32; taps is [ch, 5] f32. Returns [ch, T] f32.
    pub fn dwconv(&self, x: &Array2<f32>, taps: &Array2<f32>) -> Array2<f32> {
        let (ch, t) = x.dim();
        assert_eq!((ch, t), (self.ch, self.t));
        // weights padded to [ch,16]: cols 0..5 = taps, rest 0
        let mut w_bits = vec![0u16; ch * 16];
        for c in 0..ch {
            for ki in 0..5 {
                w_bits[c * 16 + ki] = f32_to_bf16_bits(taps[[c, ki]]);
            }
        }
        self.bo_w.write_bytes(u16_bytes(&w_bits)).unwrap();
        self.bo_w.sync_to_device().unwrap();
        // x bf16 row-major flat
        let mut x_bits = vec![0u16; ch * t];
        for c in 0..ch {
            for ti in 0..t {
                x_bits[c * t + ti] = f32_to_bf16_bits(x[[c, ti]]);
            }
        }
        self.bo_x.write_bytes(u16_bytes(&x_bits)).unwrap();
        self.bo_x.sync_to_device().unwrap();

        let t0 = Instant::now();
        self.kern
            .run_dwconv6(3, &self.bo_instr, self.n_instr, &self.bo_x, &self.bo_w, &self.bo_y)
            .unwrap();
        record(t0.elapsed());

        self.bo_y.sync_from_device().unwrap();
        let mut yb = vec![0u8; ch * t * 2];
        self.bo_y.read_bytes(&mut yb).unwrap();
        let mut out = Array2::<f32>::zeros((ch, t));
        for c in 0..ch {
            for ti in 0..t {
                let idx = (c * t + ti) * 2;
                let bits = u16::from_le_bytes([yb[idx], yb[idx + 1]]);
                out[[c, ti]] = bf16_bits_to_f32(bits);
            }
        }
        out
    }
}

// keep Rc<Device> field used even if some accessors are unused
impl WAEpilogue {
    pub fn device(&self) -> &Rc<Device> {
        &self.dev
    }
}
impl DwconvEngine {
    pub fn device(&self) -> &Rc<Device> {
        &self.dev
    }
}
