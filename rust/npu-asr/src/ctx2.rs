//! Two-context shared-kernel matmul engines (`two_ctx` feature).
//!
//! Motivation (measured on HW): each hw-context switch costs ~2.67 ms; a dispatch with NO context
//! switch costs ~0.7-1.0 ms. The encoder's per-block matmuls today rotate through 4 distinct
//! whole-array xclbin shapes (`512x800x3072_silu`, `512x3072x768`, `512x800x1536_bias`,
//! `512x800x768_bias`), and since a whole-array xclbin occupies all 8 columns, every shape change
//! reloads the array program (the 2.67 ms). Switching only the WEIGHT BO on the SAME xclbin does
//! NOT switch context (the program stays loaded).
//!
//! Insight: every encoder matmul is K=768 EXCEPT the two FFN-contraction mm2 (K=3072). So we route
//! EVERYTHING through ONE resident context (V2 — zero switches across the whole encoder):
//!   * ctxA = resident `512x768x3072` whole-array kernel, loaded ONCE. Handles ALL 7 K=768 ops per
//!     block (ffn1-mm1, qk, v, o, pw1, pw2, ffn2-mm1) via per-shape instruction streams (N=768/1536/
//!     3072) — each op dispatches at its REAL N by swapping only the instruction BO ([`CtxAOp`]).
//!   * mm2 (K=3072) runs on the SAME ctxA: K is split into 4× N=768 partials, host-accumulated in
//!     f32, bias2 added once ([`FfnMm2`]). No separate xclbin -> no context switch.
//!
//! Epilogues that used to ride the K-augmented xclbin move to the HOST (f32 SiLU / bias-add), which
//! is numerically equal-or-better than the on-chip bf16 tanh-approx. The LN-affine fold into mm1's
//! weight (`fold_ln_into_mm1`) is unchanged; it folds into the real (unpadded) weight here.

use std::cell::RefCell;
use std::path::Path;
use std::rc::Rc;
use std::time::Instant;

use ndarray::prelude::*;
use npu_xrt::{f32_to_bf16_bits, Bo, Device, Kernel, FLAG_CACHEABLE, FLAG_HOST_ONLY};
use rayon::prelude::*;

use crate::engines::{marsh, prof_record, read_instr_words, u16_bytes, PAD_M, WA_SUBDIR};

/// Matmul precision for the V2 encoder — a SELECTABLE, first-class choice (the general-purpose
/// multi-precision engine). Each precision = a kernel tile + device dtype + host pre/post, but ALL
/// run on the same resident-kernel + per-N instruction-stream V2 architecture. The tile's `n` must
/// divide every served width/8 (768/8=96, 1536/8=192, 3072/8=384) so the streams reuse on one
/// resident xclbin — n=96 satisfies all three for every precision here. Selected at runtime via
/// `NPU_PRECISION` (native|bf16|int8); default = `bf16` (the shipped fast default).
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Precision {
    /// native bf16, tile 32×32×32 — most precise (encoder rel ~1.8e-2 ≪ 0.08), ~8% util.
    NativeBf16,
    /// fast BFP16_IREE bf16 drop-in, tile 64×32×96 — ~2× native, calibration-free, WER-safe. DEFAULT.
    FastBf16,
    /// int8, tile 64×64×96 — ~3.6×, integer-EXACT kernel; needs host PTQ + per-model WER validation.
    Int8,
}
impl Precision {
    /// (m, k, n) kernel tile -> xclbin/insts filename suffix `_{m}x{k}x{n}_8c`.
    pub fn tile(self) -> (usize, usize, usize) {
        match self {
            Precision::NativeBf16 => (32, 32, 32),
            Precision::FastBf16 => (64, 32, 96),
            Precision::Int8 => (64, 64, 96),
        }
    }
    /// device input-element bytes (bf16 = 2, int8 = 1).
    fn in_bytes(self) -> usize {
        if self == Precision::Int8 { 1 } else { 2 }
    }
    fn is_int8(self) -> bool {
        self == Precision::Int8
    }
    /// Runtime selector: `NPU_PRECISION` = native|bf16|int8 (default FastBf16).
    pub fn from_env() -> Self {
        match std::env::var("NPU_PRECISION").ok().as_deref() {
            Some("native") => Precision::NativeBf16,
            Some("int8") => Precision::Int8,
            _ => Precision::FastBf16,
        }
    }
}
/// ctxA fixed contraction / padded output width.
pub const KA: usize = 768;
pub const NA: usize = 3072;

/// On-chip-epilogue replacement, applied on the HOST to ctxA's f32 output (first N columns only).
#[derive(Clone, Copy, PartialEq)]
pub enum Epi {
    /// SiLU(x + bias[col]) (replaces the `_silu` xclbin for the FFN mm1; bias rode the K-aug block
    /// there, applied BEFORE SiLU, so it's added here before the sigmoid). bias length = n.
    SiluBias,
    /// x + bias[col] (replaces the `_bias` xclbin for qk/v/o/pw1/pw2). bias length = n.
    Bias,
    /// raw matmul output, no bias (bias slice is empty). Used by the mm2 K-split partials, which
    /// accumulate on host and add bias2 once after the sum (see [`FfnMm2`]).
    None,
}

/// Shared ctxA (V2 — resident xclbin + per-shape instruction streams). The single plain
/// `512x768x3072` whole-array kernel is loaded ONCE (one hw-context). Unlike the old 2-context path
/// that padded every K=768 op's N up to 3072, V2 holds the THREE per-shape instruction streams
/// (N=768/1536/3072) and dispatches each op at its REAL N on the SAME resident kernel by swapping
/// only the instruction BO — proven cheap (~floor, not the 2.4ms reload) AND numerically exact
/// (cross-stream test max_rel 5.7e-7; see dispatch_spike EXP5/EXP6). So no padding-compute, no
/// padded readback, and still zero context switches. Each op (see [`CtxAOp`]) brings its own
/// real-sized `[KA, n]` weight BO + host epilogue.
pub struct SharedCtxA {
    dev: Rc<Device>,
    kern: Rc<Kernel>,
    /// per-shape instruction streams: (N, instr BO, n_instr). Same resident kernel; swap per op.
    streams: Vec<(usize, Bo, usize)>,
    bo_a: Bo, // activation bf16 [PAD_M, KA] (reused, written per dispatch)
    bo_c: Bo, // output f32 [PAD_M, NA] (reused; an N=n stream writes [PAD_M, n] contiguous = prefix)
    bo_tmp: Bo,
    bo_tr: Bo,
    /// group_id(4) so each op can allocate its weight BO against the shared kernel's B slot.
    g_b: i32,
    prec: Precision,
    a_buf: RefCell<Vec<u16>>,  // bf16 paths: host activation scratch (bf16 bits), zero-padded once
    a_buf_i8: RefCell<Vec<i8>>, // int8 path: host activation scratch (quantized), empty for bf16
    cbuf: RefCell<Vec<f32>>,  // reused output readback buffer (PAD_M*NA, 4B/elem; f32 for bf16, i32-bits for int8)
    // LE = native x86 f32, so we read straight into this aligned f32 buffer (no per-elem from_le_bytes)
    // first dispatch writes the FULL activation BO (zeroing device padding rows mp..PAD_M); later
    // dispatches write only the mp prefix (the padding rows stay zero on device thereafter).
    a_inited: std::cell::Cell<bool>,
}

/// The per-shape output widths the resident 768x3072 kernel serves via instruction streams.
const CTXA_STREAMS: [usize; 3] = [768, 1536, NA];

impl SharedCtxA {
    pub fn new(dev: &Rc<Device>, root: &Path) -> Rc<Self> {
        Self::with_precision(dev, root, Precision::from_env())
    }

    pub fn with_precision(dev: &Rc<Device>, root: &Path, prec: Precision) -> Rc<Self> {
        let wa = root.join(WA_SUBDIR);
        let (mt, kt, nt) = prec.tile();
        eprintln!("[ctx2] V2 encoder precision = {prec:?} (tile {mt}x{kt}x{nt})");
        // ONE resident kernel = the largest (N=3072) plain whole-array program; every K=768 op runs
        // on it via its own per-N instruction stream.
        let xclbin = wa.join(format!("final_{PAD_M}x{KA}x{NA}_{mt}x{kt}x{nt}_8c.xclbin"));
        let kern = dev
            .load_kernel(xclbin.to_str().unwrap(), None)
            .unwrap_or_else(|e| panic!("load {}: {e}", xclbin.display()));

        let g_instr = kern.group_id(1).unwrap();
        let g_a = kern.group_id(3).unwrap();
        let g_b = kern.group_id(4).unwrap();
        let g_c = kern.group_id(5).unwrap();
        let g_tmp = kern.group_id(6).unwrap();
        let g_tr = kern.group_id(7).unwrap();

        // load the per-shape streams (insts built for the 768x{n} xclbins; they run correctly on the
        // resident 768x3072 kernel — N is stream-driven, K is fixed at 768 for all of them).
        let mut streams = Vec::with_capacity(CTXA_STREAMS.len());
        for &n in CTXA_STREAMS.iter() {
            let insts = wa.join(format!("insts_{PAD_M}x{KA}x{n}_{mt}x{kt}x{nt}_8c.txt"));
            let (instr_bytes, n_instr) = read_instr_words(&insts);
            let bo = dev.alloc_bo(&kern, instr_bytes.len(), FLAG_CACHEABLE, g_instr).unwrap();
            bo.write_bytes(&instr_bytes).unwrap();
            bo.sync_to_device().unwrap();
            streams.push((n, bo, n_instr));
        }

        // activation BO: in_bytes/elem (bf16=2, int8=1). Output BO is 4B/elem either way (f32 / i32).
        let bo_a = dev.alloc_bo(&kern, PAD_M * KA * prec.in_bytes(), FLAG_HOST_ONLY, g_a).unwrap();
        let bo_c = dev.alloc_bo(&kern, PAD_M * NA * 4, FLAG_HOST_ONLY, g_c).unwrap();
        let bo_tmp = dev.alloc_bo(&kern, 1, FLAG_HOST_ONLY, g_tmp).unwrap();
        let bo_tr = dev.alloc_bo(&kern, 4, FLAG_HOST_ONLY, g_tr).unwrap();

        Rc::new(SharedCtxA {
            dev: dev.clone(),
            kern,
            streams,
            bo_a,
            bo_c,
            bo_tmp,
            bo_tr,
            g_b,
            prec,
            a_buf: RefCell::new(if prec.is_int8() { Vec::new() } else { vec![0u16; PAD_M * KA] }),
            a_buf_i8: RefCell::new(if prec.is_int8() { vec![0i8; PAD_M * KA] } else { Vec::new() }),
            cbuf: RefCell::new(vec![0f32; PAD_M * NA]),
            a_inited: std::cell::Cell::new(false),
        })
    }

    /// (instr BO, n_instr) for the per-shape stream that produces N=`n` output on the resident kernel.
    fn stream(&self, n: usize) -> (&Bo, usize) {
        self.streams
            .iter()
            .find(|(sn, _, _)| *sn == n)
            .map(|(_, bo, ni)| (bo, *ni))
            .unwrap_or_else(|| panic!("ctxA: no instruction stream for N={n} (have {CTXA_STREAMS:?})"))
    }
}

/// One K=768 op routed through [`SharedCtxA`]: owns its REAL `[KA, n]` weight BO (no padding) and its
/// host epilogue (SiLU or per-column bias). Dispatches on the resident kernel via the N=`n` stream.
pub struct CtxAOp {
    shared: Rc<SharedCtxA>,
    n: usize,       // real output width (one of CTXA_STREAMS)
    epi: Epi,
    bias: Vec<f32>, // length n
    bo_b: Bo,       // weight [KA, n] row-major, synced once (bf16 bits, or int8 for the int8 path)
    w_scale: Vec<f32>, // int8: per-output-channel symmetric scale (len n); empty for bf16
}

impl CtxAOp {
    /// `w_real` is `[KA, n]` (LN affine pre-folded for the FFN mm1; raw weight otherwise). `bias` is
    /// length `n`: applied on host after the matmul (before SiLU for `Epi::SiluBias`). `n` must be a
    /// served stream width (768/1536/3072).
    pub fn new(shared: Rc<SharedCtxA>, w_real: &Array2<f32>, n: usize, epi: Epi, bias: &[f32]) -> Self {
        assert_eq!(w_real.dim(), (KA, n));
        assert!(CTXA_STREAMS.contains(&n), "ctxA op N={n} not a served stream");
        // Epi::None carries no bias (an empty slice); the bias-applying epilogues need length n.
        assert_eq!(bias.len(), if epi == Epi::None { 0 } else { n });

        // Weight [KA, n] row-major (no NA padding — the N=n stream reads exactly [KA, n]).
        let (bo_b, w_scale) = if shared.prec.is_int8() {
            // int8: per-output-channel (per n column) symmetric quant. scale[nn] = max|W[:,nn]|/127.
            let mut w_scale = vec![0f32; n];
            for nn in 0..n {
                let mut amax = 0f32;
                for kk in 0..KA {
                    amax = amax.max(w_real[[kk, nn]].abs());
                }
                w_scale[nn] = if amax > 0.0 { amax / 127.0 } else { 1.0 };
            }
            let mut b_i8 = vec![0i8; KA * n];
            for kk in 0..KA {
                let base = kk * n;
                for nn in 0..n {
                    b_i8[base + nn] = quant_i8(w_real[[kk, nn]], w_scale[nn]);
                }
            }
            let bo_b = shared.dev_alloc_b(b_i8.len()).expect("alloc ctxA int8 weight BO");
            bo_b.write_bytes(i8_bytes(&b_i8)).unwrap();
            bo_b.sync_to_device().unwrap();
            (bo_b, w_scale)
        } else {
            let mut b_bits = vec![0u16; KA * n];
            for kk in 0..KA {
                let base = kk * n;
                for nn in 0..n {
                    b_bits[base + nn] = f32_to_bf16_bits(w_real[[kk, nn]]);
                }
            }
            let bo_b = shared.dev_alloc_b(b_bits.len() * 2).expect("alloc ctxA weight BO");
            bo_b.write_bytes(u16_bytes(&b_bits)).unwrap();
            bo_b.sync_to_device().unwrap();
            (bo_b, Vec::new())
        };

        CtxAOp {
            shared,
            n,
            epi,
            bias: bias.to_vec(),
            bo_b,
            w_scale,
        }
    }

    /// `a_real` is `[Mp, KA]` (Mp <= PAD_M). Returns `[Mp, n]` f32 with the host epilogue applied.
    pub fn forward(&self, a_real: &Array2<f32>) -> Array2<f32> {
        self.forward_view(a_real.view())
    }

    /// As [`forward`], but accepts a (possibly strided) view so the f32->bf16 conversion can read
    /// straight from a column-slice of a larger tensor — no intermediate contiguous copy. The mm2
    /// K-split uses this to convert each 768-col slice of H directly (one pass over H, not a
    /// gather-to-owned then a separate convert).
    pub fn forward_view(&self, a_real: ArrayView2<f32>) -> Array2<f32> {
        if self.shared.prec.is_int8() {
            return self.forward_view_int8(a_real);
        }
        let (mp, kk) = a_real.dim();
        assert_eq!(kk, KA);
        assert!(mp <= PAD_M);
        let sh = &self.shared;

        // --- write/convert activation into the shared bf16 buffer (rows beyond mp stay zero) ---
        let tc = Instant::now();
        {
            let mut a = sh.a_buf.borrow_mut();
            a.par_chunks_mut(KA).take(mp).enumerate().for_each(|(r, row)| {
                for c in 0..KA {
                    row[c] = f32_to_bf16_bits(a_real[[r, c]]);
                }
            });
            if sh.a_inited.get() {
                sh.bo_a.write_bytes(&u16_bytes(&a)[..mp * KA * 2]).unwrap();
            } else {
                sh.bo_a.write_bytes(u16_bytes(&a)).unwrap();
                sh.a_inited.set(true);
            }
        }
        marsh::add(marsh::CONV, tc.elapsed());
        let ts = Instant::now();
        sh.bo_a.sync_to_device().unwrap();
        marsh::add(marsh::SYNC_TO, ts.elapsed());

        // --- dispatch on the resident kernel via this op's N=n stream (swap instr BO only) ---
        let n = self.n;
        let (instr, n_instr) = sh.stream(n);
        let t0 = Instant::now();
        sh.kern
            .run_matmul8(3, instr, n_instr, &sh.bo_a, &self.bo_b, &sh.bo_c, &sh.bo_tmp, &sh.bo_tr)
            .unwrap();
        prof_record(t0.elapsed());

        let tf = Instant::now();
        sh.bo_c.sync_from_device().unwrap();
        marsh::add(marsh::SYNC_FROM, tf.elapsed());
        // an N=n stream writes [PAD_M, n] contiguous f32; the first mp rows are [mp, n] contiguous.
        // Read straight into the aligned f32 buffer (device f32 LE == native x86 f32 layout).
        let tr = Instant::now();
        {
            let mut cf = sh.cbuf.borrow_mut();
            let dst = unsafe { std::slice::from_raw_parts_mut(cf.as_mut_ptr() as *mut u8, mp * n * 4) };
            sh.bo_c.read_bytes(dst).unwrap();
        }
        marsh::add(marsh::READ, tr.elapsed());
        let te = Instant::now();
        let cf = sh.cbuf.borrow();
        let vals: &[f32] = &cf[..mp * n];
        let epi = self.epi;
        let bias = &self.bias;
        let data: Vec<f32> = match epi {
            Epi::None => vals.to_vec(),
            Epi::Bias => vals.par_iter().enumerate().map(|(i, &raw)| raw + bias[i % n]).collect(),
            Epi::SiluBias => vals
                .par_iter()
                .enumerate()
                .map(|(i, &raw)| {
                    let v = raw + bias[i % n];
                    v * fast_sigmoid(v)
                })
                .collect(),
        };
        let out = Array2::from_shape_vec((mp, n), data).unwrap();
        marsh::add(marsh::EPI, te.elapsed());
        out
    }

    /// int8 path (W8A8 dynamic): per-tensor dynamic activation quant (i8) × per-output-channel weight
    /// quant (i8) on the int8 xclbin → i32 accumulate → host dequant by `scale_a * w_scale[col]`, then
    /// the same host epilogue. Kernel ABI is unchanged (opcode 3, plain row-major); only the buffer
    /// dtypes differ (i8 in, i32 out). Per-matmul dual-precision: a WER-unsafe op just uses a bf16
    /// `CtxAOp` on a bf16 `SharedCtxA` instead.
    fn forward_view_int8(&self, a_real: ArrayView2<f32>) -> Array2<f32> {
        let (mp, kk) = a_real.dim();
        assert_eq!(kk, KA);
        assert!(mp <= PAD_M);
        let sh = &self.shared;
        let n = self.n;

        let tc = Instant::now();
        // dynamic per-tensor activation scale = max|A| / 127 over the real mp×KA elements
        let amax = a_real.iter().fold(0f32, |m, &v| m.max(v.abs()));
        let scale_a = if amax > 0.0 { amax / 127.0 } else { 1.0 };
        {
            let mut a = sh.a_buf_i8.borrow_mut();
            a.par_chunks_mut(KA).take(mp).enumerate().for_each(|(r, row)| {
                for c in 0..KA {
                    row[c] = quant_i8(a_real[[r, c]], scale_a);
                }
            });
            if sh.a_inited.get() {
                sh.bo_a.write_bytes(&i8_bytes(&a)[..mp * KA]).unwrap();
            } else {
                sh.bo_a.write_bytes(i8_bytes(&a)).unwrap();
                sh.a_inited.set(true);
            }
        }
        marsh::add(marsh::CONV, tc.elapsed());
        let ts = Instant::now();
        sh.bo_a.sync_to_device().unwrap();
        marsh::add(marsh::SYNC_TO, ts.elapsed());

        let (instr, n_instr) = sh.stream(n);
        let t0 = Instant::now();
        sh.kern
            .run_matmul8(3, instr, n_instr, &sh.bo_a, &self.bo_b, &sh.bo_c, &sh.bo_tmp, &sh.bo_tr)
            .unwrap();
        prof_record(t0.elapsed());

        let tf = Instant::now();
        sh.bo_c.sync_from_device().unwrap();
        marsh::add(marsh::SYNC_FROM, tf.elapsed());
        let tr = Instant::now();
        {
            let mut cf = sh.cbuf.borrow_mut();
            let dst = unsafe { std::slice::from_raw_parts_mut(cf.as_mut_ptr() as *mut u8, mp * n * 4) };
            sh.bo_c.read_bytes(dst).unwrap();
        }
        marsh::add(marsh::READ, tr.elapsed());
        let te = Instant::now();
        let cf = sh.cbuf.borrow();
        // the device wrote i32; reinterpret the 4B/elem readback buffer as i32 (same layout, no copy).
        let acc: &[i32] = unsafe { std::slice::from_raw_parts(cf.as_ptr() as *const i32, mp * n) };
        let epi = self.epi;
        let bias = &self.bias;
        let ws = &self.w_scale;
        let data: Vec<f32> = (0..mp * n)
            .into_par_iter()
            .map(|i| {
                let c = i % n;
                let v = acc[i] as f32 * scale_a * ws[c]; // dequant
                match epi {
                    Epi::None => v,
                    Epi::Bias => v + bias[c],
                    Epi::SiluBias => {
                        let z = v + bias[c];
                        z * fast_sigmoid(z)
                    }
                }
            })
            .collect();
        let out = Array2::from_shape_vec((mp, n), data).unwrap();
        marsh::add(marsh::EPI, te.elapsed());
        out
    }
}

/// The FFN mm2 (K=3072 -> N=768) on the SAME resident ctxA kernel — no separate xclbin, so ZERO
/// context switches across the whole encoder (this is what made the EXP6 137.7ms pool switch-free).
/// ctxA only contracts K=768, so mm2's K=3072 is split into 4 partials `h[:, i*768..] @ W2[i*768.., :]`
/// (each a plain N=768 ctxA dispatch, Epi::None), accumulated on the host in f32, then bias2 added
/// once. Host-side f32 accumulation across the 4 partials is numerically equal-or-better than one
/// on-chip K=3072 reduction. `MM2_OUT` (768) is a served stream; each partial reuses it.
pub const MM2_OUT: usize = 768;
const MM2_KSPLIT: usize = NA / KA; // 3072 / 768 = 4

pub struct FfnMm2 {
    parts: Vec<CtxAOp>, // MM2_KSPLIT ops on ctxA, each weight [KA, MM2_OUT], Epi::None
    bias2: Vec<f32>,    // length MM2_OUT, added on host once after the partial sum
}

impl FfnMm2 {
    /// `w2` is `[3072, 768]` (K-major), `b2` length 768. Split W2 along K into 4× `[768, 768]`.
    pub fn new(shared: Rc<SharedCtxA>, w2: &Array2<f32>, b2: &[f32]) -> Self {
        assert_eq!(w2.dim(), (NA, MM2_OUT));
        assert_eq!(b2.len(), MM2_OUT);
        let parts = (0..MM2_KSPLIT)
            .map(|i| {
                let wk = w2.slice(s![i * KA..(i + 1) * KA, ..]).to_owned(); // [KA, MM2_OUT]
                CtxAOp::new(shared.clone(), &wk, MM2_OUT, Epi::None, &[])
            })
            .collect();
        FfnMm2 {
            parts,
            bias2: b2.to_vec(),
        }
    }

    /// `h` is `[Mp, 3072]` (the SiLU'd FFN intermediate). Returns `[Mp, 768]` = h@W2 + b2.
    pub fn forward(&self, h: &Array2<f32>) -> Array2<f32> {
        let (mp, kk) = h.dim();
        assert_eq!(kk, NA);
        let mut acc = Array2::<f32>::zeros((mp, MM2_OUT));
        for (i, op) in self.parts.iter().enumerate() {
            // strided column-slice view of H -> converted directly into the kernel's bf16 buffer
            // (one pass over H; no per-partial gather-to-owned f32 copy).
            let hk = h.slice(s![.., i * KA..(i + 1) * KA]); // [mp, KA] view
            acc += &op.forward_view(hk);
        }
        let b2 = &self.bias2;
        acc.axis_iter_mut(Axis(0)).for_each(|mut row| {
            for c in 0..MM2_OUT {
                row[c] += b2[c];
            }
        });
        acc
    }
}

/// Numerically-stable sigmoid via the host crate's branch-free `fast_exp_nonpos` (~1e-7, ~5-8x
/// faster than libm `expf`). Feeds only `exp(<=0)`: for x>=0 use exp(-x); for x<0 use exp(x)/(1+exp(x)).
#[inline(always)]
fn fast_sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + npu_asr_host::fast_exp_nonpos(-x))
    } else {
        let e = npu_asr_host::fast_exp_nonpos(x);
        e / (1.0 + e)
    }
}

/// Symmetric int8 quantization: round(x / scale) clamped to [-127, 127] (−128 unused for symmetry).
#[inline(always)]
fn quant_i8(x: f32, scale: f32) -> i8 {
    (x / scale).round().clamp(-127.0, 127.0) as i8
}

/// Reinterpret an i8 slice as raw bytes for `Bo::write_bytes` (1 byte/elem).
fn i8_bytes(v: &[i8]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len()) }
}

// --- small alloc helper so each op allocates its weight BO against the shared kernel's B slot ---
impl SharedCtxA {
    fn dev_alloc_b(&self, nbytes: usize) -> Result<Bo, String> {
        self.dev.alloc_bo(&self.kern, nbytes, FLAG_HOST_ONLY, self.g_b)
    }
}
