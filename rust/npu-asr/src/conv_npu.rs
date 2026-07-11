//! ResNet conv on the NPU: per-channel-band M-stationary GEMM dispatch. One xclbin per Cout band
//! (512x768xN). `gemm_tile` runs ONE [512,768]x[768,N] dispatch; `conv` lowers a full conv layer via
//! im2col2d + M-tile (512 rows) + host K-split (768 chunks) + accumulate + bias.
//! See internal notes (spike GO). Dispatch mirrors bin/mstat_probe.rs.
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::PathBuf;
use std::rc::Rc;

use ndarray::prelude::*;
use npu_asr_host::im2col2d;
use npu_xrt::{Bo, Device, Kernel, FLAG_CACHEABLE, FLAG_HOST_ONLY};

const MT: usize = 512; // M-tile = m*n_aie_rows*n_aie_cols = 16*4*8 (kernel-fixed)
const KT: usize = 768; // K-split chunk

fn f32_to_bf16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let r = 0x7fff + ((b >> 16) & 1);
    ((b.wrapping_add(r)) >> 16) as u16
}

struct Band {
    kern: Rc<Kernel>,
    instr: Bo,
    n_instr: usize,
}

pub struct ConvNpu {
    dev: Rc<Device>,
    wa: PathBuf, // dir holding final_mstat_*.xclbin + insts_*.txt
    bands: RefCell<HashMap<usize, Rc<Band>>>,
}

impl ConvNpu {
    pub fn new(dev: Rc<Device>, wa: PathBuf) -> Self {
        ConvNpu { dev, wa, bands: RefCell::new(HashMap::new()) }
    }

    fn band(&self, n: usize) -> Rc<Band> {
        if let Some(b) = self.bands.borrow().get(&n) {
            return b.clone();
        }
        let stem = format!("mstat_512x768x{n}_16x32x32_8c");
        let crate::kernel_registry::KernelArtifacts { xclbin, insts } =
            crate::kernel_registry::resolve(&self.wa, &stem);
        let kern = self
            .dev
            .load_kernel(xclbin.to_str().unwrap(), None)
            .unwrap_or_else(|e| panic!("load band xclbin {xclbin:?}: {e}"));
        let bytes = std::fs::read(&insts).unwrap_or_else(|e| panic!("read insts {insts:?}: {e}"));
        let n_instr = bytes.len() / 4;
        let instr = self
            .dev
            .alloc_bo(&kern, bytes.len(), FLAG_CACHEABLE, kern.group_id(1).unwrap())
            .unwrap();
        instr.write_bytes(&bytes).unwrap();
        instr.sync_to_device().unwrap();
        let b = Rc::new(Band { kern, instr, n_instr });
        self.bands.borrow_mut().insert(n, b.clone());
        b
    }

    /// One M-stationary dispatch: a[MT,KT] @ b[KT,N] -> [MT,N] f32. a,b row-major f32 (cast to bf16).
    fn gemm_tile(&self, a: &Array2<f32>, bmat: &Array2<f32>, n: usize) -> Array2<f32> {
        assert_eq!(a.dim(), (MT, KT));
        assert_eq!(bmat.dim(), (KT, n));
        let band = self.band(n);
        let k = &band.kern;
        let a_bits: Vec<u8> = a.iter().flat_map(|&v| f32_to_bf16_bits(v).to_le_bytes()).collect();
        let b_bits: Vec<u8> = bmat.iter().flat_map(|&v| f32_to_bf16_bits(v).to_le_bytes()).collect();
        let bo_a = self.dev.alloc_bo(k, MT * KT * 2, FLAG_HOST_ONLY, k.group_id(3).unwrap()).unwrap();
        let bo_b = self.dev.alloc_bo(k, KT * n * 2, FLAG_HOST_ONLY, k.group_id(4).unwrap()).unwrap();
        let bo_c = self.dev.alloc_bo(k, MT * n * 4, FLAG_HOST_ONLY, k.group_id(5).unwrap()).unwrap();
        let bo_tmp = self.dev.alloc_bo(k, 1, FLAG_HOST_ONLY, k.group_id(6).unwrap()).unwrap();
        let bo_tr = self.dev.alloc_bo(k, 4, FLAG_HOST_ONLY, k.group_id(7).unwrap()).unwrap();
        bo_a.write_bytes(&a_bits).unwrap();
        bo_a.sync_to_device().unwrap();
        bo_b.write_bytes(&b_bits).unwrap();
        bo_b.sync_to_device().unwrap();
        k.run_matmul8(3, &band.instr, band.n_instr, &bo_a, &bo_b, &bo_c, &bo_tmp, &bo_tr)
            .expect("dispatch");
        bo_c.sync_from_device().unwrap();
        let mut cb = vec![0u8; MT * n * 4];
        bo_c.read_bytes(&mut cb).unwrap();
        let c: Vec<f32> = cb
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        Array2::from_shape_vec((MT, n), c).unwrap()
    }

    /// Full conv layer on NPU (symmetric stride/pad). x[Cin,H,W], w[Cout,Cin,kh,kw], b[Cout].
    /// Returns y[Cout,Hout,Wout]. Thin wrapper over `conv_asym`.
    pub fn conv(
        &self,
        x: &Array3<f32>,
        w: &Array4<f32>,
        b: &Array1<f32>,
        kh: usize,
        kw: usize,
        stride: usize,
        pad: usize,
    ) -> Array3<f32> {
        self.conv_asym(x, w, b, kh, kw, stride, stride, pad, pad)
    }

    /// Full conv layer on NPU with ASYMMETRIC stride/pad per dim (sh/sw, ph/pw) — needed for 1D convs
    /// laid out as 2D (kh=1, ph=0). x[Cin,H,W], w[Cout,Cin,kh,kw], b[Cout]. Cout must be a built band.
    /// im2col2d -> M-tile(512) -> K-split(768) -> accumulate -> +bias.
    #[allow(clippy::too_many_arguments)]
    pub fn conv_asym(
        &self,
        x: &Array3<f32>,
        w: &Array4<f32>,
        b: &Array1<f32>,
        kh: usize,
        kw: usize,
        sh: usize,
        sw: usize,
        ph: usize,
        pw: usize,
    ) -> Array3<f32> {
        let (cin, h, wd) = x.dim();
        let cout = w.dim().0;
        let out_h = (h + 2 * ph - kh) / sh + 1;
        let out_w = (wd + 2 * pw - kw) / sw + 1;
        let m_real = out_h * out_w;
        let k_real = cin * kh * kw;
        let cols = im2col2d(x, kh, kw, sh, sw, ph, pw); // [m_real, k_real]
        let wmat = w.to_shape((cout, k_real)).unwrap().to_owned(); // [cout, k_real]
        let k_chunks = k_real.div_ceil(KT);
        let mut out = Array2::<f32>::zeros((m_real, cout));
        let mut a = Array2::<f32>::zeros((MT, KT));
        let mut bmat = Array2::<f32>::zeros((KT, cout));
        let mt_n = m_real.div_ceil(MT);
        for mi in 0..mt_n {
            let r0 = mi * MT;
            let rows = (m_real - r0).min(MT);
            for kc in 0..k_chunks {
                let c0 = kc * KT;
                let kk = (k_real - c0).min(KT);
                a.fill(0.0);
                for r in 0..rows {
                    for c in 0..kk {
                        a[[r, c]] = cols[[r0 + r, c0 + c]];
                    }
                }
                bmat.fill(0.0);
                for c in 0..kk {
                    for co in 0..cout {
                        bmat[[c, co]] = wmat[[co, c0 + c]];
                    }
                }
                let part = self.gemm_tile(&a, &bmat, cout); // [MT, cout]
                for r in 0..rows {
                    for co in 0..cout {
                        out[[r0 + r, co]] += part[[r, co]];
                    }
                }
            }
        }
        let mut y = Array3::<f32>::zeros((cout, out_h, out_w));
        for p in 0..m_real {
            let (oh, ow) = (p / out_w, p % out_w);
            for co in 0..cout {
                y[[co, oh, ow]] = out[[p, co]] + b[co];
            }
        }
        y
    }
}
