//! Standalone NATIVE whole-array GEMM dispatcher for ESM (research/comparison path). Does NOT touch
//! the shipped, validated `SharedCtxA`/`FfnMm2` (which parakeet/gigaam/bge depend on). Loads a native
//! per-shape `final_MxKxN_..._8c.xclbin` (+ its `insts_..._8c.txt`) and runs `C[M,N] = A[M,K] @ B[K,N]`
//! at the kernel's REAL K (no zero-pad to 768) using the documented dispatch ABI:
//!   opcode 3, group_ids instr=1 a=3 b=4 c=5 tmp=6 trace=7; A bf16 [PAD_M,K] row-major (rows>mp zero),
//!   B bf16 [K,N] row-major, C f32 [PAD_M,N] row-major (no de-shuffle), bias added on host.
//! NOTE: whole-array tiling requires N % (tile_n * n_aie_cols) == 0; callers pad N accordingly.
//!
//! Design: `NativeKernel` = one loaded xclbin (one hw-context) + reusable scratch BOs, shared by all
//! ops of that (K,N) shape. `NativeWeight` = a resident per-op weight BO bound to a kernel. Multiple
//! weights (e.g. q/k/v/o, all 320x512) share ONE kernel/context — weights stay resident, so the
//! per-matmul latency excludes weight upload (fair comparison).
use std::path::Path;
use std::rc::Rc;

use ndarray::Array2;
use npu_xrt::{f32_to_bf16_bits, Bo, Device, Kernel, FLAG_CACHEABLE, FLAG_HOST_ONLY};

pub const PAD_M: usize = 512;

fn read_instr_words(path: &Path) -> (Vec<u8>, usize) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read instr {}: {e}", path.display()));
    let words = bytes.len() / 4;
    (bytes, words)
}

/// A loaded native xclbin of shape (K,N) + its instruction stream + reusable scratch BOs. One per
/// distinct (K,N); shared across all weights of that shape. Dispatch is sequential, so the shared
/// activation/output scratch is safe.
pub struct NativeKernel {
    dev: Rc<Device>,
    kern: Rc<Kernel>,
    instr: Bo,
    n_instr: usize,
    bo_a: Bo,
    bo_c: Bo,
    bo_tmp: Bo,
    bo_tr: Bo,
    g_b: i32,
    pub k: usize,
    pub n: usize,
}

/// A resident weight BO [K,N] (bf16) bound to a `NativeKernel`.
pub struct NativeWeight {
    bo_b: Bo,
}

impl NativeKernel {
    /// Load `final_{PAD_M}x{k}x{n}_{tile}_8c.xclbin` + insts from `wa`. `tile` e.g. "32x32x32".
    pub fn load(dev: &Rc<Device>, wa: &Path, k: usize, n: usize, tile: &str) -> Rc<Self> {
        let xclbin = wa.join(format!("final_{PAD_M}x{k}x{n}_{tile}_8c.xclbin"));
        let insts = wa.join(format!("insts_{PAD_M}x{k}x{n}_{tile}_8c.txt"));
        let kern = dev.load_kernel(xclbin.to_str().unwrap(), None).unwrap_or_else(|e| panic!("load {}: {e}", xclbin.display()));
        let g_instr = kern.group_id(1).unwrap();
        let g_a = kern.group_id(3).unwrap();
        let g_b = kern.group_id(4).unwrap();
        let g_c = kern.group_id(5).unwrap();
        let g_tmp = kern.group_id(6).unwrap();
        let g_tr = kern.group_id(7).unwrap();
        let (ibytes, n_instr) = read_instr_words(&insts);
        let instr = dev.alloc_bo(&kern, ibytes.len(), FLAG_CACHEABLE, g_instr).unwrap();
        instr.write_bytes(&ibytes).unwrap();
        instr.sync_to_device().unwrap();
        let bo_a = dev.alloc_bo(&kern, PAD_M * k * 2, FLAG_HOST_ONLY, g_a).unwrap();
        let bo_c = dev.alloc_bo(&kern, PAD_M * n * 4, FLAG_HOST_ONLY, g_c).unwrap();
        let bo_tmp = dev.alloc_bo(&kern, 1, FLAG_HOST_ONLY, g_tmp).unwrap();
        let bo_tr = dev.alloc_bo(&kern, 4, FLAG_HOST_ONLY, g_tr).unwrap();
        Rc::new(NativeKernel { dev: dev.clone(), kern, instr, n_instr, bo_a, bo_c, bo_tmp, bo_tr, g_b, k, n })
    }

    /// Build a resident weight from a real [k_real <= K, n_real <= N] matrix, zero-padded to [K,N].
    pub fn weight(&self, w_real: &Array2<f32>) -> NativeWeight {
        let (kr, nr) = w_real.dim();
        assert!(kr <= self.k && nr <= self.n, "weight {kr}x{nr} exceeds kernel {}x{}", self.k, self.n);
        let mut bits = vec![0u16; self.k * self.n];
        for kk in 0..kr {
            let base = kk * self.n;
            for nn in 0..nr {
                bits[base + nn] = f32_to_bf16_bits(w_real[[kk, nn]]);
            }
        }
        let bytes = unsafe { std::slice::from_raw_parts(bits.as_ptr() as *const u8, bits.len() * 2) };
        let bo_b = self.dev.alloc_bo(&self.kern, self.k * self.n * 2, FLAG_HOST_ONLY, self.g_b).unwrap();
        bo_b.write_bytes(bytes).unwrap();
        bo_b.sync_to_device().unwrap();
        NativeWeight { bo_b }
    }

    /// C[mp, n_out] = A[mp, k_real] @ W (+ optional bias[n_out]). A's K is zero-padded to self.k,
    /// output sliced to n_out (<= self.n). mp <= PAD_M.
    pub fn matmul(&self, w: &NativeWeight, a: &Array2<f32>, n_out: usize, bias: Option<&[f32]>) -> Array2<f32> {
        let (mp, kr) = a.dim();
        assert!(kr <= self.k && mp <= PAD_M && n_out <= self.n);
        let mut abuf = vec![0u16; PAD_M * self.k];
        for r in 0..mp {
            let base = r * self.k;
            for c in 0..kr {
                abuf[base + c] = f32_to_bf16_bits(a[[r, c]]);
            }
        }
        let abytes = unsafe { std::slice::from_raw_parts(abuf.as_ptr() as *const u8, abuf.len() * 2) };
        self.bo_a.write_bytes(abytes).unwrap();
        self.bo_a.sync_to_device().unwrap();
        self.kern.run_matmul8(3, &self.instr, self.n_instr, &self.bo_a, &w.bo_b, &self.bo_c, &self.bo_tmp, &self.bo_tr).unwrap();
        self.bo_c.sync_from_device().unwrap();
        let mut cf = vec![0f32; PAD_M * self.n];
        let dst = unsafe { std::slice::from_raw_parts_mut(cf.as_mut_ptr() as *mut u8, PAD_M * self.n * 4) };
        self.bo_c.read_bytes(dst).unwrap();
        let mut out = Array2::<f32>::zeros((mp, n_out));
        for r in 0..mp {
            for c in 0..n_out {
                let v = cf[r * self.n + c];
                out[[r, c]] = if let Some(b) = bias { v + b[c] } else { v };
            }
        }
        out
    }
}

/// Thin wrapper: one kernel + one weight (used by verify_native_gemm).
pub struct NativeGemm {
    kernel: Rc<NativeKernel>,
    weight: Option<NativeWeight>,
}
impl NativeGemm {
    pub fn load(dev: &Rc<Device>, wa: &Path, k: usize, n: usize, tile: &str) -> Self {
        NativeGemm { kernel: NativeKernel::load(dev, wa, k, n, tile), weight: None }
    }
    pub fn set_weight(&mut self, w: &Array2<f32>) {
        self.weight = Some(self.kernel.weight(w));
    }
    pub fn matmul(&self, a: &Array2<f32>, bias: Option<&[f32]>) -> Array2<f32> {
        let n = self.kernel.n;
        self.kernel.matmul(self.weight.as_ref().expect("set_weight first"), a, n, bias)
    }
}
