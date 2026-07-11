//! ctxLN — encoder LayerNorm on the NPU array (Step D, internal notes §4).
//!
//! A standalone resident xclbin (the `ml/layernorm` design rebuilt f32 + the `ln_2pass` f32
//! two-pass kernel — route_b_kernels/ctx_ln/), loaded ONCE, dispatched per LN call. It computes
//! the NORMALIZE-ONLY part `(x - mean) / sqrt(var + eps)` per row over the 768 channels, f32 in /
//! f32 out (docs/05 "never re-expand"); the affine γ,β is applied on the host by the caller for the
//! 4 affine LN sites (cheap, exact). This is design D-i: a SEPARATE small xclbin co-resident with
//! ctxA — functionally correct (two hwctx coexist), it pays a per-call hw-context switch which we
//! accept here (the Step-C goal as scoped is LN-on-NPU correctness, not latency).
//!
//! Verified: NPU output matches the host `layer_norm_normalize` to rel ~7.8e-7 (bin/ln_probe.rs).
//!
//! ABI (scripts/run_npu_layernorm.py): the 2-arg IRON sequence (in, out) yields kernel args
//! 1=instr, 3=in, 4=out, 5=tmp, 6=ctrl, 7=trace → run_matmul8(3, instr, n, in, out, c, tmp, tr),
//! output read from the b/out slot.

use std::cell::RefCell;
use std::path::Path;
use std::rc::Rc;

use ndarray::prelude::*;
use npu_xrt::{Bo, Device, Kernel, FLAG_CACHEABLE, FLAG_HOST_ONLY};

/// ctxLN is built for PAD_M rows (the matmul pads T up to this); shorter T is zero-padded and the
/// first T output rows are returned (padding rows normalize to 0 and are discarded).
const LN_ROWS: usize = 512;
const LN_COLS: usize = 768;
const LN_SUBDIR: &str = "mlir-aie/programming_examples/ml/layernorm/build";

fn f32_as_bytes(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

/// Resident on-NPU LayerNorm (normalize-only). One xclbin, loaded once; dispatched per LN call.
pub struct CtxLn {
    kern: Rc<Kernel>,
    instr: Bo,
    n_instr: usize,
    bo_in: Bo,
    bo_out: Bo,
    // dummy tmp/ctrl/trace placeholders required by the host ABI (must be live, non-zero size).
    bo_c: Bo,
    bo_tmp: Bo,
    bo_tr: Bo,
    in_buf: RefCell<Vec<f32>>,  // [LN_ROWS*LN_COLS], reused; padding rows stay 0
    out_buf: RefCell<Vec<f32>>, // [LN_ROWS*LN_COLS], reused readback
}

impl CtxLn {
    pub fn new(dev: &Rc<Device>, root: &Path) -> Rc<Self> {
        let dir = root.join(LN_SUBDIR);
        let stem = format!("ctxln_{LN_ROWS}x{LN_COLS}");
        let crate::kernel_registry::KernelArtifacts { xclbin, insts } =
            crate::kernel_registry::resolve(&dir, &stem);
        let kern = dev
            .load_kernel(xclbin.to_str().unwrap(), None)
            .unwrap_or_else(|e| panic!("load ctxLN {}: {e}", xclbin.display()));
        let ibytes = std::fs::read(&insts)
            .unwrap_or_else(|e| panic!("read ctxLN insts {}: {e}", insts.display()));
        let n_instr = ibytes.len() / 4;
        let g = |i| kern.group_id(i).unwrap();
        let instr = dev.alloc_bo(&kern, ibytes.len(), FLAG_CACHEABLE, g(1)).unwrap();
        instr.write_bytes(&ibytes).unwrap();
        instr.sync_to_device().unwrap();

        let bo_in = dev.alloc_bo(&kern, LN_ROWS * LN_COLS * 4, FLAG_HOST_ONLY, g(3)).unwrap();
        let bo_out = dev.alloc_bo(&kern, LN_ROWS * LN_COLS * 4, FLAG_HOST_ONLY, g(4)).unwrap();
        let bo_c = dev.alloc_bo(&kern, 64, FLAG_HOST_ONLY, g(5)).unwrap();
        let bo_tmp = dev.alloc_bo(&kern, 8, FLAG_HOST_ONLY, g(6)).unwrap();
        let bo_tr = dev.alloc_bo(&kern, 4, FLAG_HOST_ONLY, g(7)).unwrap();

        eprintln!("[ctxLN] resident LayerNorm xclbin loaded ({LN_ROWS}x{LN_COLS}, f32 two-pass, normalize-only)");
        Rc::new(CtxLn {
            kern,
            instr,
            n_instr,
            bo_in,
            bo_out,
            bo_c,
            bo_tmp,
            bo_tr,
            in_buf: RefCell::new(vec![0f32; LN_ROWS * LN_COLS]),
            out_buf: RefCell::new(vec![0f32; LN_ROWS * LN_COLS]),
        })
    }

    /// Normalize-only LayerNorm on the NPU: `[T,768] -> [T,768]`, `(x-mean)/sqrt(var+eps)` per row.
    /// T is zero-padded to LN_ROWS for the dispatch; the first T rows are returned.
    pub fn normalize(&self, x: &Array2<f32>) -> Array2<f32> {
        let t = x.nrows();
        assert_eq!(x.ncols(), LN_COLS, "ctxLN expects {LN_COLS} channels");
        assert!(t <= LN_ROWS, "ctxLN T={t} exceeds PAD rows {LN_ROWS}");

        {
            let mut ib = self.in_buf.borrow_mut();
            // write the T real rows; rows T..LN_ROWS remain 0 from the last init/zeroing.
            for (r, row) in x.outer_iter().enumerate() {
                let dst = &mut ib[r * LN_COLS..(r + 1) * LN_COLS];
                for (c, &v) in row.iter().enumerate() {
                    dst[c] = v;
                }
            }
            // zero any rows beyond T that a previous larger call may have written.
            for v in ib[t * LN_COLS..].iter_mut() {
                *v = 0.0;
            }
            self.bo_in.write_bytes(f32_as_bytes(&ib)).unwrap();
        }
        self.bo_in.sync_to_device().unwrap();

        self.kern
            .run_matmul8(3, &self.instr, self.n_instr, &self.bo_in, &self.bo_out, &self.bo_c, &self.bo_tmp, &self.bo_tr)
            .unwrap();

        self.bo_out.sync_from_device().unwrap();
        let mut ob = self.out_buf.borrow_mut();
        {
            let dst = unsafe {
                std::slice::from_raw_parts_mut(ob.as_mut_ptr() as *mut u8, LN_ROWS * LN_COLS * 4)
            };
            self.bo_out.read_bytes(dst).unwrap();
        }
        Array2::from_shape_fn((t, LN_COLS), |(r, c)| ob[r * LN_COLS + c])
    }
}
