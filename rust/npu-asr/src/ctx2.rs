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
use npu_xrt::{f32_to_bf16_bits, Bo, Device, Kernel, Run, FLAG_CACHEABLE, FLAG_HOST_ONLY};
use rayon::prelude::*;

use crate::engines::{marsh, prof_record, read_instr_words, u16_bytes, PAD_M, WA_SUBDIR};
use npu_asr_host::prof;

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
/// K-augmented contraction for the Step-A modal on-chip epilogue (`NPU_MODAL_EPI=1`): bias rides an
/// extra 32-wide k-block (`A_aug=[A|ones]`, `B_aug=[B;bias]` → `A@B+bias`), so the on-chip epilogue
/// needs no host bias-add and no 3rd DMA channel. KAUG = KA + 32 = 800.
pub const KAUG: usize = KA + 32;

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
    /// Goal-1 async overlap: double-buffer slots for pipelining the INDEPENDENT mm2 K-split partials
    /// (each partial reads a different column-slice of H and a different weight → no data dep). Empty
    /// unless `NPU_MM2_PIPELINE=1`. Two slots = a 2-deep pipeline (one dispatch in flight on the NPU
    /// while the host preps the next / post-processes the previous). See [`FfnMm2::forward`].
    pipeline: bool,
    pipe: Vec<PipeSlot>,
    /// Step-A modal on-chip epilogue (`NPU_MODAL_EPI=1`, bf16/native only). The resident xclbin is the
    /// K-augmented (K=800) f32-out modal design; bias is folded via K-aug and SiLU runs on-chip,
    /// selected per dispatch by the instruction-stream's baked RTP mode. Output stays f32 (no
    /// re-expand), so the host epilogue becomes a no-op. `ka_dev` = device-side K (768 normal, 800
    /// modal). `modal_streams` = (N, is_silu, instr BO, n_instr) — 6 streams (3 N × {silu,identity}).
    modal: bool,
    ka_dev: usize,
    modal_streams: Vec<(usize, bool, Bo, usize)>,
    /// int8 host fast-path (`NPU_INT8_FASTEPI`, default ON; `=0` reverts to the legacy path for A/B).
    /// Two byte-identical cuts to the int8 marshaling pools: (1) parallel exact `amax` reduction
    /// (replaces the serial `iter().fold` quant scan); (2) division-free row-chunked dequant epilogue
    /// (replaces the per-element `i % n` hardware divide, keeping the exact `(acc·scale_a)·ws[c]`
    /// multiply order). int8-only; no effect on the bf16 default.
    fast_int8: bool,
}

/// One double-buffer slot for the async mm2 pipeline: its own activation/output BOs (so a dispatch
/// in flight on the NPU isn't clobbered by the host prepping the next on the other slot) + own
/// tmp/trace (avoid sharing kernel scratch across concurrent runs) + own host scratch.
struct PipeSlot {
    bo_a: Bo,
    bo_c: Bo,
    bo_tmp: Bo,
    bo_tr: Bo,
    cbuf: RefCell<Vec<f32>>,
    a_buf: RefCell<Vec<u16>>,   // bf16 scratch (empty for int8)
    a_buf_i8: RefCell<Vec<i8>>, // int8 scratch (empty for bf16)
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
        // Step-A modal on-chip epilogue: K-aug bias + on-chip SiLU, f32 out, one resident xclbin with
        // RTP-selected mode per inst-stream. bf16/native only (the modal xclbin is the native 32³ tile).
        // modal on-chip epilogue (K-aug bias + on-chip SiLU, f32 out) — built for both bf16 tiles
        // (native 32³, fast 64×32×96). DEFAULT-ON for bf16 (measured: fast −40ms → sub-300ms idle,
        // WER 9.6% unchanged). int8 would need an i32-dequant epilogue (not built). Opt out:
        // `NPU_MODAL_EPI=0`.
        let modal = !prec.is_int8() && std::env::var("NPU_MODAL_EPI").as_deref() != Ok("0");
        let ka_dev = if modal { KAUG } else { KA };
        eprintln!(
            "[ctx2] V2 encoder precision = {prec:?} (tile {mt}x{kt}x{nt}){}",
            if modal { "  [modal on-chip epilogue: K-aug bias + on-chip SiLU, f32 out]" } else { "" }
        );
        // ONE resident kernel = the largest (N=3072) whole-array program; every op runs on it via its
        // per-N (and, modal, per-mode) instruction stream.
        let xclbin = if modal {
            wa.join(format!("final_{PAD_M}x{KAUG}x{NA}_{mt}x{kt}x{nt}_8c_modalsilu.xclbin"))
        } else {
            wa.join(format!("final_{PAD_M}x{KA}x{NA}_{mt}x{kt}x{nt}_8c.xclbin"))
        };
        let kern = dev
            .load_kernel(xclbin.to_str().unwrap(), None)
            .unwrap_or_else(|e| panic!("load {}: {e}", xclbin.display()));

        let g_instr = kern.group_id(1).unwrap();
        let g_a = kern.group_id(3).unwrap();
        let g_b = kern.group_id(4).unwrap();
        let g_c = kern.group_id(5).unwrap();
        let g_tmp = kern.group_id(6).unwrap();
        let g_tr = kern.group_id(7).unwrap();

        let load_stream = |insts: &Path| {
            let (instr_bytes, n_instr) = read_instr_words(insts);
            let bo = dev.alloc_bo(&kern, instr_bytes.len(), FLAG_CACHEABLE, g_instr).unwrap();
            bo.write_bytes(&instr_bytes).unwrap();
            bo.sync_to_device().unwrap();
            (bo, n_instr)
        };

        // plain (non-modal) per-N streams; modal loads its 6 (N × {silu,identity}) streams below.
        let mut streams = Vec::with_capacity(CTXA_STREAMS.len());
        let mut modal_streams: Vec<(usize, bool, Bo, usize)> = Vec::new();
        if modal {
            for &n in CTXA_STREAMS.iter() {
                for (is_silu, tag) in [(true, "modalsilu"), (false, "modalid")] {
                    let insts = wa.join(format!("insts_{PAD_M}x{KAUG}x{n}_{mt}x{kt}x{nt}_8c_{tag}.txt"));
                    let (bo, n_instr) = load_stream(&insts);
                    modal_streams.push((n, is_silu, bo, n_instr));
                }
            }
        } else {
            for &n in CTXA_STREAMS.iter() {
                let insts = wa.join(format!("insts_{PAD_M}x{KA}x{n}_{mt}x{kt}x{nt}_8c.txt"));
                let (bo, n_instr) = load_stream(&insts);
                streams.push((n, bo, n_instr));
            }
        }

        // activation BO: in_bytes/elem (bf16=2, int8=1), K = ka_dev (768 / 800 modal). Output BO is
        // 4B/elem (f32 / i32).
        let bo_a = dev.alloc_bo(&kern, PAD_M * ka_dev * prec.in_bytes(), FLAG_HOST_ONLY, g_a).unwrap();
        let bo_c = dev.alloc_bo(&kern, PAD_M * NA * 4, FLAG_HOST_ONLY, g_c).unwrap();
        let bo_tmp = dev.alloc_bo(&kern, 1, FLAG_HOST_ONLY, g_tmp).unwrap();
        let bo_tr = dev.alloc_bo(&kern, 4, FLAG_HOST_ONLY, g_tr).unwrap();

        // Goal-1 async overlap: 2-slot double-buffer for the mm2 pipeline. DEFAULT-ON (measured s10:
        // ~29ms/8% off the bf16 default, ~15-17ms native/int8, numerically byte-identical — same
        // kernel + same host f32 accumulation order). Opt out with `NPU_MM2_PIPELINE=0`.
        let pipeline = std::env::var("NPU_MM2_PIPELINE").as_deref() != Ok("0");
        let mut pipe = Vec::new();
        if pipeline {
            eprintln!("[ctx2] async mm2 pipeline ENABLED (default; set NPU_MM2_PIPELINE=0 to disable)");
            for _ in 0..2 {
                // modal: K-aug pipe activation (ones-column at KA) so the mm2 partials' identity stream
                // adds their (zero) bias correctly; sized ka_dev.
                let mut a_buf = vec![0u16; PAD_M * ka_dev];
                if modal {
                    let one = f32_to_bf16_bits(1.0);
                    for r in 0..PAD_M {
                        a_buf[r * ka_dev + KA] = one;
                    }
                }
                pipe.push(PipeSlot {
                    bo_a: dev.alloc_bo(&kern, PAD_M * ka_dev * prec.in_bytes(), FLAG_HOST_ONLY, g_a).unwrap(),
                    bo_c: dev.alloc_bo(&kern, PAD_M * NA * 4, FLAG_HOST_ONLY, g_c).unwrap(),
                    bo_tmp: dev.alloc_bo(&kern, 1, FLAG_HOST_ONLY, g_tmp).unwrap(),
                    bo_tr: dev.alloc_bo(&kern, 4, FLAG_HOST_ONLY, g_tr).unwrap(),
                    cbuf: RefCell::new(vec![0f32; PAD_M * NA]),
                    a_buf: RefCell::new(if prec.is_int8() { Vec::new() } else { a_buf }),
                    a_buf_i8: RefCell::new(if prec.is_int8() { vec![0i8; PAD_M * KA] } else { Vec::new() }),
                    a_inited: std::cell::Cell::new(false),
                });
            }
        }

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
            a_buf: RefCell::new(if prec.is_int8() {
                Vec::new()
            } else {
                // modal: K-aug activation [PAD_M, 800] with the ones-column at index KA (so the K-aug
                // matmul adds bias); the conversion loop writes only cols 0..KA, leaving this intact.
                let mut v = vec![0u16; PAD_M * ka_dev];
                if modal {
                    let one = f32_to_bf16_bits(1.0);
                    for r in 0..PAD_M {
                        v[r * ka_dev + KA] = one;
                    }
                }
                v
            }),
            a_buf_i8: RefCell::new(if prec.is_int8() { vec![0i8; PAD_M * KA] } else { Vec::new() }),
            cbuf: RefCell::new(vec![0f32; PAD_M * NA]),
            a_inited: std::cell::Cell::new(false),
            pipeline,
            pipe,
            modal,
            ka_dev,
            modal_streams,
            fast_int8: {
                let on = prec.is_int8() && std::env::var("NPU_INT8_FASTEPI").as_deref() != Ok("0");
                if prec.is_int8() {
                    eprintln!(
                        "[ctx2] int8 host fast-path {} (parallel amax + division-free dequant; NPU_INT8_FASTEPI=0 disables)",
                        if on { "ENABLED" } else { "DISABLED" }
                    );
                }
                on
            },
        })
    }

    /// (instr BO, n_instr) for the modal stream producing N=`n` in mode `is_silu` on the resident xclbin.
    fn modal_stream(&self, n: usize, is_silu: bool) -> (&Bo, usize) {
        self.modal_streams
            .iter()
            .find(|(sn, ss, _, _)| *sn == n && *ss == is_silu)
            .map(|(_, _, bo, ni)| (bo, *ni))
            .unwrap_or_else(|| panic!("ctxA modal: no stream N={n} silu={is_silu}"))
    }

    /// Async overlap, the prep half: convert+write+sync slot `s`'s activation from the strided view
    /// `a_real` `[mp, KA]`, then SUBMIT the dispatch (N=`n`, weight `bo_b`) WITHOUT waiting — the NPU
    /// runs while the caller does other host work. Returns the in-flight [`Run`] + (int8) the dynamic
    /// activation scale to dequant with in [`pipe_read`]. bf16 and int8 both supported.
    fn pipe_start(&self, s: usize, a_real: ArrayView2<f32>, bo_b: &Bo, n: usize) -> (Run, f32) {
        let (mp, kk) = a_real.dim();
        debug_assert_eq!(kk, KA);
        let slot = &self.pipe[s];
        let tc = Instant::now();
        let scale_a = if self.prec.is_int8() {
            let scale_a = quant_scale(a_real, mp, self.fast_int8);
            let mut a = slot.a_buf_i8.borrow_mut();
            a.par_chunks_mut(KA).take(mp).enumerate().for_each(|(r, row)| {
                for c in 0..KA {
                    row[c] = quant_i8(a_real[[r, c]], scale_a);
                }
            });
            if slot.a_inited.get() {
                slot.bo_a.write_bytes(&i8_bytes(&a)[..mp * KA]).unwrap();
            } else {
                slot.bo_a.write_bytes(i8_bytes(&a)).unwrap();
                slot.a_inited.set(true);
            }
            scale_a
        } else {
            let kd = self.ka_dev;
            let mut a = slot.a_buf.borrow_mut();
            a.par_chunks_mut(kd).take(mp).enumerate().for_each(|(r, row)| {
                for c in 0..KA {
                    row[c] = f32_to_bf16_bits(a_real[[r, c]]);
                }
            });
            if slot.a_inited.get() {
                slot.bo_a.write_bytes(&u16_bytes(&a)[..mp * kd * 2]).unwrap();
            } else {
                slot.bo_a.write_bytes(u16_bytes(&a)).unwrap();
                slot.a_inited.set(true);
            }
            1.0
        };
        marsh::add(marsh::CONV, tc.elapsed());
        let ts = Instant::now();
        slot.bo_a.sync_to_device().unwrap();
        marsh::add(marsh::SYNC_TO, ts.elapsed());

        // mm2 partials use the identity epilogue (Epi::None -> zero K-aug bias); modal selects it.
        let (instr, n_instr) = if self.modal {
            self.modal_stream(n, false)
        } else {
            self.stream(n)
        };
        let run = self
            .kern
            .run_matmul8_start(3, instr, n_instr, &slot.bo_a, bo_b, &slot.bo_c, &slot.bo_tmp, &slot.bo_tr)
            .unwrap();
        (run, scale_a)
    }

    /// Async overlap, the post half: read slot `s`'s output `[mp, n]` (the dispatch must already be
    /// waited) and apply the mm2 epilogue (Epi::None → raw, or int8 dequant by `scale_a * w_scale`).
    /// Returns an owned `[mp, n]` f32 to be host-accumulated.
    fn pipe_read(&self, s: usize, mp: usize, n: usize, scale_a: f32, w_scale: &[f32]) -> Array2<f32> {
        let slot = &self.pipe[s];
        let tf = Instant::now();
        slot.bo_c.sync_from_device().unwrap();
        marsh::add(marsh::SYNC_FROM, tf.elapsed());
        let tr = Instant::now();
        {
            let mut cf = slot.cbuf.borrow_mut();
            let dst = unsafe { std::slice::from_raw_parts_mut(cf.as_mut_ptr() as *mut u8, mp * n * 4) };
            slot.bo_c.read_bytes(dst).unwrap();
        }
        marsh::add(marsh::READ, tr.elapsed());
        let te = Instant::now();
        let cf = slot.cbuf.borrow();
        let data: Vec<f32> = if self.prec.is_int8() {
            let acc: &[i32] = unsafe { std::slice::from_raw_parts(cf.as_ptr() as *const i32, mp * n) };
            // raw dequant (Epi::None): the per-op bias/SiLU is applied later in `finish_slot`.
            dequant_epi(acc, mp, n, scale_a, w_scale, Epi::None, &[], self.fast_int8)
        } else {
            cf[..mp * n].to_vec()
        };
        let out = Array2::from_shape_vec((mp, n), data).unwrap();
        marsh::add(marsh::EPI, te.elapsed());
        out
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
    bo_b: Bo,       // weight [KA, n] row-major (modal: [KAUG, n] with bias K-aug'd into row KA)
    w_scale: Vec<f32>, // int8: per-output-channel symmetric scale (len n); empty for bf16
    is_silu: bool,  // modal: dispatch the silu-mode stream (Epi::SiluBias) vs identity (else)
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
            // modal: weight is [KAUG, n] with bias K-aug'd into row KA (rows KA+1..KAUG stay 0); the
            // on-chip epilogue then adds nothing for bias (it's in the matmul) and applies SiLU for
            // the silu-mode stream. Non-modal: plain [KA, n].
            let kd = shared.ka_dev;
            let mut b_bits = vec![0u16; kd * n];
            for kk in 0..KA {
                let base = kk * n;
                for nn in 0..n {
                    b_bits[base + nn] = f32_to_bf16_bits(w_real[[kk, nn]]);
                }
            }
            if shared.modal {
                let base = KA * n; // the K-aug bias row
                for nn in 0..n {
                    let bv = if epi == Epi::None { 0.0 } else { bias[nn] };
                    b_bits[base + nn] = f32_to_bf16_bits(bv);
                }
            }
            let bo_b = shared.dev_alloc_b(b_bits.len() * 2).expect("alloc ctxA weight BO");
            bo_b.write_bytes(u16_bytes(&b_bits)).unwrap();
            bo_b.sync_to_device().unwrap();
            (bo_b, Vec::new())
        };

        CtxAOp {
            is_silu: shared.modal && epi == Epi::SiluBias,
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
        // modal: buffer row width is ka_dev=800 with the ones-column at index KA preserved (set once
        // at init); we only write the real KA cols here, so the K-aug bias term stays intact.
        let kd = sh.ka_dev;
        let tc = Instant::now();
        {
            let mut a = sh.a_buf.borrow_mut();
            a.par_chunks_mut(kd).take(mp).enumerate().for_each(|(r, row)| {
                for c in 0..KA {
                    row[c] = f32_to_bf16_bits(a_real[[r, c]]);
                }
            });
            if sh.a_inited.get() {
                sh.bo_a.write_bytes(&u16_bytes(&a)[..mp * kd * 2]).unwrap();
            } else {
                sh.bo_a.write_bytes(u16_bytes(&a)).unwrap();
                sh.a_inited.set(true);
            }
        }
        marsh::add(marsh::CONV, tc.elapsed());
        let ts = Instant::now();
        sh.bo_a.sync_to_device().unwrap();
        marsh::add(marsh::SYNC_TO, ts.elapsed());

        // --- dispatch on the resident kernel via this op's stream (modal: N + epilogue-mode) ---
        let n = self.n;
        let (instr, n_instr) = if sh.modal {
            sh.modal_stream(n, self.is_silu)
        } else {
            sh.stream(n)
        };
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
        // modal: bias + SiLU already applied on-chip (f32 out) -> host epilogue is a no-op copy.
        let data: Vec<f32> = if sh.modal {
            vals.to_vec()
        } else {
            match epi {
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
            }
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
        let scale_a = quant_scale(a_real, mp, sh.fast_int8);
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
        let data = dequant_epi(acc, mp, n, scale_a, &self.w_scale, self.epi, &self.bias, sh.fast_int8);
        let out = Array2::from_shape_vec((mp, n), data).unwrap();
        marsh::add(marsh::EPI, te.elapsed());
        out
    }

    /// Goal-1 async overlap for an INDEPENDENT op pair (qk ∥ v): qk reads `rope(ln)`, v reads `ln`,
    /// neither depends on the other, both feed `mha`. Submit both — each into its own double-buffer
    /// slot — so `self`'s NPU compute overlaps `other`'s host prep (quant/convert+sync) and `self`'s
    /// readback overlaps `other`'s NPU compute. Reuses the 2 `PipeSlot`s (free during attention: ffn1's
    /// mm2 has finished and ffn2's hasn't started). Numerically IDENTICAL to two sequential `forward_view`
    /// calls (same kernel, same per-op epilogue) — only the scheduling differs. Falls back to sequential
    /// when the pipeline is off (`NPU_MM2_PIPELINE=0`).
    pub fn forward2_overlapped(
        &self,
        a_self: ArrayView2<f32>,
        other: &CtxAOp,
        a_other: ArrayView2<f32>,
    ) -> (Array2<f32>, Array2<f32>) {
        let sh = &self.shared;
        if !sh.pipeline {
            return (self.forward_view(a_self), other.forward_view(a_other));
        }
        let mp0 = a_self.dim().0;
        let mp1 = a_other.dim().0;
        // submit both (slot 0 = self, slot 1 = other); the 2nd submit's host prep overlaps the 1st's run.
        let (run0, sc0) = sh.pipe_start(0, a_self, &self.bo_b, self.n);
        let (run1, sc1) = sh.pipe_start(1, a_other, &other.bo_b, other.n);
        run0.wait().unwrap();
        let o0 = self.finish_slot(0, mp0, sc0); // self's readback overlaps other's NPU compute
        run1.wait().unwrap();
        let o1 = other.finish_slot(1, mp1, sc1);
        (o0, o1)
    }

    /// Read `[mp, n]` from the pipe slot this op was `pipe_start`ed into, applying this op's epilogue.
    /// modal: bias is K-aug'd + applied on-chip (and these ops are never SiLU) → `pipe_read`'s output is
    /// already complete. non-modal bf16 / int8: `pipe_read` returns the raw (int8: dequant'd) matmul, so
    /// add this op's host bias here — matching `forward_view`'s epilogue exactly.
    fn finish_slot(&self, slot: usize, mp: usize, scale_a: f32) -> Array2<f32> {
        let sh = &self.shared;
        let mut out = sh.pipe_read(slot, mp, self.n, scale_a, &self.w_scale);
        if sh.modal {
            return out;
        }
        let n = self.n;
        let bias = &self.bias;
        match self.epi {
            Epi::None => {}
            Epi::Bias => out.axis_iter_mut(Axis(0)).for_each(|mut row| {
                for c in 0..n {
                    row[c] += bias[c];
                }
            }),
            Epi::SiluBias => out.iter_mut().enumerate().for_each(|(i, v)| {
                let z = *v + bias[i % n];
                *v = z * fast_sigmoid(z);
            }),
        }
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
        if self.parts[0].shared.pipeline {
            return self.forward_pipelined(h, mp);
        }
        let mut acc = Array2::<f32>::zeros((mp, MM2_OUT));
        for (i, op) in self.parts.iter().enumerate() {
            // strided column-slice view of H -> converted directly into the kernel's bf16 buffer
            // (one pass over H; no per-partial gather-to-owned f32 copy).
            let hk = h.slice(s![.., i * KA..(i + 1) * KA]); // [mp, KA] view
            acc += &op.forward_view(hk);
        }
        self.add_bias2(&mut acc);
        acc
    }

    /// Goal-1 async overlap: the 4 K-split partials are mutually INDEPENDENT (different H column-slice
    /// + different weight), so run them as a 2-deep start/wait pipeline. At each step we SUBMIT the
    /// next partial (its host prep — quant/convert + sync — overlaps the previous partial's NPU
    /// compute) and then read+accumulate the previous partial (its host post — readback + dequant —
    /// overlaps the just-submitted partial's NPU compute). Two double-buffer slots keep one dispatch
    /// in flight without the host clobbering its activation/output BOs. Numerically identical to the
    /// sequential path (same kernel, same host f32 accumulate); only the scheduling differs.
    fn forward_pipelined(&self, h: &Array2<f32>, mp: usize) -> Array2<f32> {
        let shared = &self.parts[0].shared;
        let n = MM2_OUT;
        let np = self.parts.len();
        let mut acc = Array2::<f32>::zeros((mp, MM2_OUT));

        let h0 = h.slice(s![.., 0..KA]);
        let (mut prev_run, mut prev_scale) = shared.pipe_start(0, h0, &self.parts[0].bo_b, n);
        let (mut prev_slot, mut prev_i) = (0usize, 0usize);

        for i in 1..np {
            let slot = i % 2;
            let hi = h.slice(s![.., i * KA..(i + 1) * KA]);
            let (cur_run, cur_scale) = shared.pipe_start(slot, hi, &self.parts[i].bo_b, n);
            // P(i-1) is on a different slot than the just-submitted Pi → safe to finish it now; its
            // host post-processing overlaps Pi's NPU compute.
            prev_run.wait().unwrap();
            let part = shared.pipe_read(prev_slot, mp, n, prev_scale, &self.parts[prev_i].w_scale);
            prof::time("mm2_accum", || acc += &part);
            prev_run = cur_run;
            prev_scale = cur_scale;
            prev_slot = slot;
            prev_i = i;
        }
        prev_run.wait().unwrap();
        let part = shared.pipe_read(prev_slot, mp, n, prev_scale, &self.parts[prev_i].w_scale);
        prof::time("mm2_accum", || acc += &part);

        prof::time("mm2_accum", || self.add_bias2(&mut acc));
        acc
    }

    #[inline]
    fn add_bias2(&self, acc: &mut Array2<f32>) {
        let b2 = &self.bias2;
        acc.axis_iter_mut(Axis(0)).for_each(|mut row| {
            for c in 0..MM2_OUT {
                row[c] += b2[c];
            }
        });
    }
}

/// The subsample front-end's conv2 matmul on the resident ctxA (DEFAULT-ON, `NPU_SS_NPU=0` reverts).
/// conv2 is `cols[Lout, 3840] @ w2ᵀ -> [Lout, 768]` and K=3840 = 5×768, so it K-splits onto ctxA
/// EXACTLY like [`FfnMm2`] (5 partials, Epi::None, host f32 accumulate, +b2 once; ReLU stays on the
/// caller). Output is `[Lout, 768]` = the subsample result directly (no transpose). Runs at the
/// resident kernel's precision. MEASURED net-positive (e2e −20ms bf16) + WER-safe at every precision
/// (bf16 9.6→9.2%, int8 9.2→8.7%, native 9.2%). Offloads ~1.18 GMAC of host matmul.
pub const CONV2_KSPLIT: usize = 5; // 3840 / 768

pub struct Conv2Mm {
    parts: Vec<CtxAOp>, // CONV2_KSPLIT ops on ctxA, each weight [KA, 768], Epi::None
    bias: Vec<f32>,     // length 768 (cout), added on host once after the partial sum
}

impl Conv2Mm {
    /// `w2` is the conv2 weight `[cout=768, cin=768, k=5]`, `b2` length 768. Reshaped to `[768, 3840]`
    /// (Cin-major, j = ci*k + ki — matching the host `im2col` cols flatten) and split along K into 5×
    /// `[768, 768]` ctxA weights.
    pub fn new(shared: Rc<SharedCtxA>, w2: &Array3<f32>, b2: &[f32]) -> Self {
        let (cout, cin, k) = w2.dim();
        assert_eq!(cout, MM2_OUT, "conv2 cout must be {MM2_OUT}");
        let kk = cin * k;
        assert_eq!(kk, CONV2_KSPLIT * KA, "conv2 K={kk} must be {CONV2_KSPLIT}×{KA}");
        assert_eq!(b2.len(), MM2_OUT);
        // [cout, cin*k] in the same Cin-major flatten the host im2col uses for cols.
        let w2r = w2.to_shape((cout, kk)).expect("reshape conv2 weight").to_owned();
        let parts = (0..CONV2_KSPLIT)
            .map(|p| {
                // partial p: activation = cols[:, p*KA..(p+1)*KA] [Lout, KA]; weight W_p[j', co] =
                // w2r[co, p*KA + j'] → CtxAOp wants [KA, n]=[K, cout], so W_p = w2r[:, p-block].T.
                let wp = w2r.slice(s![.., p * KA..(p + 1) * KA]).t().to_owned(); // [KA, cout]
                CtxAOp::new(shared.clone(), &wp, MM2_OUT, Epi::None, &[])
            })
            .collect();
        Conv2Mm { parts, bias: b2.to_vec() }
    }

    /// `cols` is `[Lout, 3840]` (host im2col of the conv0 output). Returns `[Lout, 768]` = the conv2
    /// pre-activation `cols @ w2ᵀ + b2` (the caller applies ReLU). Lout (=400) ≤ PAD_M.
    pub fn forward(&self, cols: &Array2<f32>) -> Array2<f32> {
        let (mp, kk) = cols.dim();
        assert_eq!(kk, CONV2_KSPLIT * KA);
        let mut acc = Array2::<f32>::zeros((mp, MM2_OUT));
        for (p, op) in self.parts.iter().enumerate() {
            let ck = cols.slice(s![.., p * KA..(p + 1) * KA]); // [Lout, KA] view
            acc += &op.forward_view(ck);
        }
        acc.axis_iter_mut(Axis(0)).for_each(|mut row| {
            for c in 0..MM2_OUT {
                row[c] += self.bias[c];
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

/// Dynamic per-tensor activation scale = max|A|/127 over the real `[mp, KA]` view. `max` is exact and
/// associative, so the parallel row-reduction (`fast`) yields the BYTE-IDENTICAL amax to the serial
/// `iter().fold` — it just spreads the mp×KA scan across cores instead of one thread. Returns the
/// quant scale (1.0 for an all-zero tensor, matching the legacy guard).
#[inline]
fn quant_scale(a_real: ArrayView2<f32>, mp: usize, fast: bool) -> f32 {
    let amax = if fast {
        (0..mp)
            .into_par_iter()
            .map(|r| {
                let mut m = 0f32;
                for c in 0..KA {
                    m = m.max(a_real[[r, c]].abs());
                }
                m
            })
            .reduce(|| 0f32, f32::max)
    } else {
        a_real.iter().fold(0f32, |m, &v| m.max(v.abs()))
    };
    if amax > 0.0 { amax / 127.0 } else { 1.0 }
}

/// Dequant an i32 accumulator `[mp, n]` → f32 with the per-op epilogue. The `fast` path is row-chunked
/// so the column index `c` is an inner loop counter — no per-element `i % n` hardware divide (n is a
/// runtime non-power-of-2 divisor, so it can't be strength-reduced) — and `ws[c]`/`bias[c]` reads are
/// sequential. The per-element float ops keep the EXACT legacy order `(acc as f32 * scale_a) * ws[c]
/// [+ bias[c]]`, so the output is byte-identical to the `into_par_iter` path; only the loop shape and
/// the dropped divide differ. The legacy branch is retained for clean A/B.
#[inline]
fn dequant_epi(
    acc: &[i32],
    mp: usize,
    n: usize,
    scale_a: f32,
    ws: &[f32],
    epi: Epi,
    bias: &[f32],
    fast: bool,
) -> Vec<f32> {
    if !fast {
        return (0..mp * n)
            .into_par_iter()
            .map(|i| {
                let c = i % n;
                let v = acc[i] as f32 * scale_a * ws[c];
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
    }
    let mut data = vec![0f32; mp * n];
    data.par_chunks_mut(n).zip(acc.par_chunks(n)).for_each(|(orow, arow)| match epi {
        Epi::None => {
            for c in 0..n {
                orow[c] = arow[c] as f32 * scale_a * ws[c];
            }
        }
        Epi::Bias => {
            for c in 0..n {
                orow[c] = arow[c] as f32 * scale_a * ws[c] + bias[c];
            }
        }
        Epi::SiluBias => {
            for c in 0..n {
                let z = arow[c] as f32 * scale_a * ws[c] + bias[c];
                orow[c] = z * fast_sigmoid(z);
            }
        }
    });
    data
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
