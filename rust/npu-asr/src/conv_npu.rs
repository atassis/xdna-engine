//! ResNet conv on the NPU: per-channel-band M-stationary GEMM dispatch. ONE xclbin serves every Cout
//! band; the band is selected by swapping the instruction stream, which carries the BD size/stride
//! fields. `gemm_tile` runs ONE [512,768]x[768,N] dispatch; `conv` lowers a full conv layer via
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

/// Bands this path can dispatch. The largest is the ANCHOR: its xclbin is the one actually loaded and
/// its shapes size the data BOs, so every smaller band reuses both. Measured safe -- one hw_context
/// runs both bands' streams clean in either order, and an N mismatch leaves A=[M,K] fixed while
/// B=[K,N] and C=[M,N] scale, which is the output-scales case that reuse permits.
const ANCHOR: usize = 512;

struct Ctx {
    kern: Rc<Kernel>,
    streams: RefCell<HashMap<usize, (Rc<Bo>, usize)>>, // band -> (instruction BO, word count)
    // Sized for ANCHOR and reused by every band. Reallocating these per dispatch cost two sessions to
    // a fake intermittency, so they are allocated exactly once.
    bo_a: Bo,
    bo_b: Bo,
    bo_c: Bo,
    bo_tmp: Bo,
    bo_tr: Bo,
}

pub struct ConvNpu {
    dev: Rc<Device>,
    wa: PathBuf, // dir holding final_mstat_*.xclbin + insts_*.txt
    ctx: RefCell<Option<Rc<Ctx>>>,
}

/// `dir/final_{stem}.xclbin` + `dir/insts_{stem}.txt`, with the opt-in manifest check
/// (engine-op-manifest-and-dynamic-xclbin): NPU_KERNEL_MANIFEST_VERIFY=1 re-hashes both against
/// dir/kernel_manifest.json first, so a stale artifact fails loud instead of loading silently.
fn artifacts(wa: &std::path::Path, n: usize) -> crate::kernel_registry::KernelArtifacts {
    let stem = format!("mstat_512x768x{n}_16x32x32_8c");
    if std::env::var("NPU_KERNEL_MANIFEST_VERIFY").is_ok() {
        crate::kernel_registry::resolve_checked(wa, &stem)
            .unwrap_or_else(|e| panic!("kernel manifest check failed for stem={stem}: {e}"))
    } else {
        crate::kernel_registry::resolve(wa, &stem)
    }
}

impl ConvNpu {
    pub fn new(dev: Rc<Device>, wa: PathBuf) -> Self {
        ConvNpu { dev, wa, ctx: RefCell::new(None) }
    }

    /// The single loaded context, built on first use from the ANCHOR band.
    fn ctx(&self) -> Rc<Ctx> {
        if let Some(c) = self.ctx.borrow().as_ref() {
            return c.clone();
        }
        let xclbin = artifacts(&self.wa, ANCHOR).xclbin;
        let kern = self
            .dev
            .load_kernel(xclbin.to_str().unwrap(), None)
            .unwrap_or_else(|e| panic!("load anchor xclbin {xclbin:?}: {e}"));
        let alloc = |nbytes: usize, arg: i32| {
            self.dev
                .alloc_bo(&kern, nbytes, FLAG_HOST_ONLY, kern.group_id(arg).unwrap())
                .unwrap()
        };
        let c = Rc::new(Ctx {
            streams: RefCell::new(HashMap::new()),
            bo_a: alloc(MT * KT * 2, 3),
            bo_b: alloc(KT * ANCHOR * 2, 4),
            bo_c: alloc(MT * ANCHOR * 4, 5),
            bo_tmp: alloc(1, 6),
            bo_tr: alloc(4, 7),
            kern,
        });
        *self.ctx.borrow_mut() = Some(c.clone());
        c
    }

    /// This band's instruction stream, uploaded once. Only the stream differs per band -- the loaded
    /// xclbin does not change.
    fn stream(&self, ctx: &Ctx, n: usize) -> (Rc<Bo>, usize) {
        if let Some((bo, count)) = ctx.streams.borrow().get(&n) {
            return (bo.clone(), *count);
        }
        let insts = artifacts(&self.wa, n).insts;
        let bytes = std::fs::read(&insts).unwrap_or_else(|e| panic!("read insts {insts:?}: {e}"));
        let count = bytes.len() / 4;
        let bo = Rc::new(
            self.dev
                .alloc_bo(&ctx.kern, bytes.len(), FLAG_CACHEABLE, ctx.kern.group_id(1).unwrap())
                .unwrap(),
        );
        bo.write_bytes(&bytes).unwrap();
        bo.sync_to_device().unwrap();
        ctx.streams.borrow_mut().insert(n, (bo.clone(), count));
        (bo, count)
    }

    /// One M-stationary dispatch: a[MT,KT] @ b[KT,N] -> [MT,N] f32. a,b row-major f32 (cast to bf16).
    fn gemm_tile(&self, a: &Array2<f32>, bmat: &Array2<f32>, n: usize) -> Array2<f32> {
        assert_eq!(a.dim(), (MT, KT));
        assert_eq!(bmat.dim(), (KT, n));
        assert!(n <= ANCHOR, "band N={n} exceeds anchor {ANCHOR}: rebuild with a larger anchor");
        let ctx = self.ctx();
        let (instr, n_instr) = self.stream(&ctx, n);
        let a_bits: Vec<u8> = a.iter().flat_map(|&v| f32_to_bf16_bits(v).to_le_bytes()).collect();
        let b_bits: Vec<u8> = bmat.iter().flat_map(|&v| f32_to_bf16_bits(v).to_le_bytes()).collect();
        ctx.bo_a.write_bytes(&a_bits).unwrap();
        ctx.bo_a.sync_to_device().unwrap();
        ctx.bo_b.write_bytes(&b_bits).unwrap();
        ctx.bo_b.sync_to_device().unwrap();
        ctx.kern
            .run_matmul8(3, &instr, n_instr, &ctx.bo_a, &ctx.bo_b, &ctx.bo_c, &ctx.bo_tmp, &ctx.bo_tr)
            .expect("dispatch");
        ctx.bo_c.sync_from_device().unwrap();
        let mut cb = vec![0u8; MT * n * 4];
        ctx.bo_c.read_bytes(&mut cb).unwrap();
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
