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

// Resident relpos-MHA block (STEP=8, STEP-C runtime-t_active). ONE xclbin sized for the MAX
// frame count RELPOS_BUILT_T serves ANY clip T <= it: the softmax reads t_active from an RTP
// baked into the instruction stream at word RELPOS_TACTIVE_WORD, so per clip we PATCH that one
// word of a template insts (zero build) and pad k/p/V to RELPOS_BUILT_T. Loaded once, resident.
const RELPOS_TQ: usize = 8;
const RELPOS_KB: usize = 43;
const RELPOS_DK: usize = 128;   // Parakeet head_dim (kernel bakes DK=128)
const RELPOS_BUILT_T: usize = 172; // baked buffer/dataflow size of the single xclbin
const RELPOS_TACTIVE_WORD: usize = 8; // insts word holding t_active (verified device-side)

/// The single resident relpos block (built at RELPOS_BUILT_T). BOs are sized for BUILT_T; per
/// dispatch we patch the instr template's t_active word and pad data to BUILT_T. Dispatched per
/// head via run_dwconv6(3, instr, n, quv, kpv, ctx).
struct RelposK {
    kern: Rc<Kernel>,
    instr_template: Vec<u32>, // insts as u32 words; word[RELPOS_TACTIVE_WORD] = t_active (patched)
    n_instr: usize,
    bo_instr: Bo,
    bo_quv: Bo,
    bo_kpv: Bo,
    bo_ctx: Bo,
    n_qt: usize,     // ceil(BUILT_T/TQ)
    tp: usize,       // k/V padded rows (n_kb*KB for BUILT_T)
    pp: usize,       // p padded rows (n_pb*KB for BUILT_T)
    ctx_rows: usize, // n_qt*TQ (CTX readback rows, take [:active_t])
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
    ln_dir: PathBuf,                               // {root}/artifacts/parakeet/ln (ctxln + affcast xclbins)
    // Tri-state cache: None = untried; Some(None) = xclbins absent, FF stays host (no retry);
    // Some(Some) = co-resident on-chip LN + affine-cast chain loaded.
    resident_ln: RefCell<Option<Option<Rc<ResidentLn>>>>,
    pub stats: RefCell<NpuStats>,
}

/// Co-resident on-chip LayerNorm (normalize-only, f32) + AFFINE cast, chained device-side
/// (resident-rails LN->fc1 seam). x[512,1024] f32 -> ctxLN -> bo_ln[512,1024] f32 (device) ->
/// affine_cast(*gamma+beta) -> bo_bf16[512,1024] bf16 (device) = affine_LN(x), the modal fc1's A
/// input -- no host round-trip on the intermediate (feasibility: prototype_ln_cast_resident.py).
/// The affine folds into the cast so fc1 uses the EXISTING modalsilu xclbin (on-chip SiLU) with the
/// UNMODIFIED weight. gamma|beta packed in bo_gb[2*KRES] on ONE DMA channel. Built at PAD_M x KRES.
struct ResidentLn {
    ln_kern: Rc<Kernel>,
    ln_instr: Bo,
    ln_n: usize,
    ac_kern: Rc<Kernel>, // affine_cast
    ac_instr: Bo,
    ac_n: usize,
    bo_x: Bo,    // [PAD_M, KRES] f32   (ctxLN input,  ln g3)
    bo_ln: Bo,   // [PAD_M, KRES] f32   (ctxLN output = affine_cast input, ln g4 / ac g3)
    bo_gb: Bo,   // [2*KRES] f32        (gamma|beta params, ac g4)
    bo_bf16: Bo, // [PAD_M, KRES] bf16  (affine_cast output = modal fc1 A, ac g5)
    // fc1->fc2 device-side (full FFN, Variant B): deinterleave+cast the [PAD_M,DFF] fc1 output into a
    // CHUNK-MAJOR [n_chunks,PAD_M,KRES] bf16 buffer (one dispatch, 3D drain TAP), then the fc2 K-split
    // reads each K=KRES chunk as a device SUB-BUFFER (Bo::sub) into the K=KRES modal -- bit-identical
    // to the host 4xK=1024 K-split (WER-neutral), A fed device-side.
    deint_kern: Rc<Kernel>,
    deint_instr: Bo,
    deint_n: usize,
    bo_deint: Bo, // [n_chunks*PAD_M*KRES] bf16 chunk-major (deint output, deint g4)
    // conv-module GLU (step 2): a*sigmoid(g) over pw1's on-chip [PAD_M,2*KRES] f32 -> [PAD_M,KRES] f32,
    // device-side (the pw1 GEMM output stays resident; GLU reads it as its A/g3 input, no host). OPTIONAL:
    // absent when the glu xclbin isn't built, so the FFN LN->fc1 seam + step-1 resident pw1 still load.
    glu: Option<ConvGlu>,
    // conv-module depthwise conv1d (step 3), OPTIONAL like glu.
    dwconv: Option<ConvDw>,
    // per-kernel dummy placeholders (0-size segfaults)
    ln_c: Bo,
    ln_tmp: Bo,
    ln_tr: Bo,
    ac_tmp: Bo,
    ac_tr: Bo,
    deint_c: Bo,
    deint_tmp: Bo,
    deint_tr: Bo,
}

/// Device-side conv-module GLU kernel + its output/dummy BOs. Input (pw1's [PAD_M,2*KRES] f32) is fed
/// as the A/g3 slot from the modal stream's bo_c; `bo_out` is the [PAD_M,KRES] f32 output on B/g4.
struct ConvGlu {
    kern: Rc<Kernel>,
    instr: Bo,
    n: usize,
    bo_out: Bo, // [PAD_M, KRES] f32 (glu output, g4)
    dummy_c: Bo,
    dummy_tmp: Bo,
    dummy_tr: Bo,
}

// Conv-module depthwise conv1d (step 3): sliding_mul FIR along time, [C,T] channel-major bf16.
// T=400 is Parakeet's ~30s frame cap (>subsample); the brick bakes it. C=1024 = d_model.
const DW_C: usize = 1024; // channels (d_model)
const DW_T: usize = 400; // baked time steps (Parakeet frame cap)
const DW_KW: usize = 16; // weight tile: taps[0..8] + BN-folded bias[9]

/// Device-side depthwise conv1d brick (dwconv1d_k9_bf16). 3-buffer ABI: in[C,T] bf16 (g3), w[C,16]
/// bf16 (g4), out[C,T] bf16 (g5). Host-fed in step 3a (transposes still host); device-fed in 3b.
struct ConvDw {
    kern: Rc<Kernel>,
    instr: Bo,
    n: usize,
    bo_in: Bo,  // [C, T] bf16 (g3)
    bo_w: Bo,   // [C, 16] bf16 (g4)
    bo_out: Bo, // [C, T] bf16 (g5)
    dummy_tmp: Bo,
    dummy_tr: Bo,
}

const DFF: usize = 4096; // Parakeet FFN inner dim (fc1 N / fc2 K)
// Variant B fc2 = deinterleave -> 4x K=KRES modal (same tile as host) on device sub-buffers +
// host-accumulate = bit-identical to the host K-split (WER-neutral), A device-side.

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
            ln_dir: root.join("artifacts/parakeet/ln"),
            resident_ln: RefCell::new(None),
            stats: RefCell::new(NpuStats::default()),
        }
    }

    /// Load (once) the SINGLE resident relpos block built at RELPOS_BUILT_T. Reads the xclbin +
    /// template insts from {root}/artifacts/relpos/single/ (pre-build: scripts/relpos_prebuild.sh).
    /// The same xclbin serves any clip T <= BUILT_T; per dispatch we patch the insts t_active word.
    fn relpos_block(&self) -> Rc<RelposK> {
        if let Some(k) = self.relpos.borrow().get(&RELPOS_BUILT_T) {
            return k.clone();
        }
        let bt = RELPOS_BUILT_T;
        let p = 2 * bt - 1;
        let cdiv = |a: usize, b: usize| (a + b - 1) / b;
        let n_qt = cdiv(bt, RELPOS_TQ);
        let tp = cdiv(bt, RELPOS_KB) * RELPOS_KB;
        let pp = cdiv(p, RELPOS_KB) * RELPOS_KB;
        let ctx_rows = n_qt * RELPOS_TQ;
        let dir = self.relpos_dir.join("single");
        let xclbin = dir.join("final.xclbin");
        let insts = dir.join("insts.bin");
        let kern = self
            .dev
            .load_kernel(xclbin.to_str().unwrap(), None)
            .unwrap_or_else(|e| panic!("load relpos single ({}): {e:?}\n  pre-build: scripts/relpos_prebuild.sh", xclbin.display()));
        let ib = std::fs::read(&insts).unwrap_or_else(|e| panic!("read {}: {e}", insts.display()));
        let instr_template: Vec<u32> = ib
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let n_instr = instr_template.len();
        let g = |i| kern.group_id(i).unwrap();
        let bo_instr = self.dev.alloc_bo(&kern, ib.len(), FLAG_CACHEABLE, g(1)).unwrap();
        let bo_quv = self.dev.alloc_bo(&kern, 2 * n_qt * RELPOS_TQ * RELPOS_DK * 2, FLAG_HOST_ONLY, g(3)).unwrap();
        let bo_kpv = self.dev.alloc_bo(&kern, (tp + pp + tp) * RELPOS_DK * 2, FLAG_HOST_ONLY, g(4)).unwrap();
        let bo_ctx = self.dev.alloc_bo(&kern, ctx_rows * RELPOS_DK * 2, FLAG_HOST_ONLY, g(5)).unwrap();
        let rk = Rc::new(RelposK { kern, instr_template, n_instr, bo_instr, bo_quv, bo_kpv, bo_ctx, n_qt, tp, pp, ctx_rows });
        self.relpos.borrow_mut().insert(RELPOS_BUILT_T, rk.clone());
        rk
    }

    /// Resident relpos-MHA block for ONE head. qu/qv/k [t,DK], p [2t-1,DK], v [t,DK] (f32) ->
    /// ctx [t,DK] (f32), t <= RELPOS_BUILT_T. STEP-C: pad the stream layout to BUILT_T, PATCH the
    /// insts t_active word to `t`, dispatch the single resident block (3-BO ABI), unpack bf16 CTX.
    pub fn relpos_mha(&self, qu: &Array2<f32>, qv: &Array2<f32>, k: &Array2<f32>, p: &Array2<f32>, v: &Array2<f32>) -> Array2<f32> {
        let t = qu.nrows();
        assert!(t <= RELPOS_BUILT_T, "clip T={t} exceeds relpos BUILT_T={RELPOS_BUILT_T}");
        let rk = self.relpos_block();
        // QUV tile-interleaved over the BUILT_T tiles; real rows only where q0 < t (rest zero pad).
        let mut quv = Vec::<f32>::with_capacity(2 * rk.n_qt * RELPOS_TQ * RELPOS_DK);
        for q in 0..rk.n_qt {
            let q0 = q * RELPOS_TQ;
            let take = RELPOS_TQ.min(t.saturating_sub(q0));
            push_pad_rows(&mut quv, qu, q0, take, RELPOS_TQ);
            push_pad_rows(&mut quv, qv, q0, take, RELPOS_TQ);
        }
        // KPV = k(pad tp) | p(pad pp) | V(pad tp); pad rows are zero so ctx ignores pad keys.
        let mut kpv = Vec::<f32>::with_capacity((rk.tp + rk.pp + rk.tp) * RELPOS_DK);
        push_pad_rows(&mut kpv, k, 0, t, rk.tp);
        push_pad_rows(&mut kpv, p, 0, p.nrows(), rk.pp);
        push_pad_rows(&mut kpv, v, 0, t, rk.tp);
        let mut qb = vec![0u16; quv.len()];
        let mut kb = vec![0u16; kpv.len()];
        npu_xrt::pack_f32_to_bf16(&quv, &mut qb);
        npu_xrt::pack_f32_to_bf16(&kpv, &mut kb);
        let t0 = Instant::now();
        // STEP-C: patch the instruction stream's t_active word to this clip's T, then upload.
        let mut insts = rk.instr_template.clone();
        insts[RELPOS_TACTIVE_WORD] = t as u32;
        let instr_bytes: Vec<u8> = insts.iter().flat_map(|w| w.to_le_bytes()).collect();
        rk.bo_instr.write_bytes(&instr_bytes).unwrap();
        rk.bo_instr.sync_to_device().unwrap();
        rk.bo_quv.write_bytes(u16_bytes(&qb)).unwrap();
        rk.bo_quv.sync_to_device().unwrap();
        rk.bo_kpv.write_bytes(u16_bytes(&kb)).unwrap();
        rk.bo_kpv.sync_to_device().unwrap();
        rk.kern.run_dwconv6(3, &rk.bo_instr, rk.n_instr, &rk.bo_quv, &rk.bo_kpv, &rk.bo_ctx).unwrap();
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

    /// Lazy-load the co-resident ctxLN + cast xclbins from {root}/artifacts/parakeet/ln (built at
    /// PAD_M x KRES = 512 x 1024). Two extra hw-contexts alongside the modal matmul.
    fn resident_ln(&self) -> Option<Rc<ResidentLn>> {
        if let Some(cached) = self.resident_ln.borrow().as_ref() {
            return cached.clone();
        }
        // Graceful: if the ctxln+affcast xclbins aren't present, the FFN LN->fc1 stays on the host
        // path (no panic) -- so the resident seam can be the DEFAULT without breaking builds/branches
        // that haven't built these kernels.
        let seam = ["ctxln", "affcast"].iter().all(|n| {
            self.ln_dir.join(format!("final_{n}_{PAD_M}x{KRES}.xclbin")).exists()
                && self.ln_dir.join(format!("insts_{n}_{PAD_M}x{KRES}.txt")).exists()
        });
        // full FFN (Variant B) also needs the deinterleave xclbin
        let fc2ok = self.ln_dir.join(format!("final_deint_{PAD_M}x{DFF}.xclbin")).exists();
        let present = seam && fc2ok;
        let result = if present {
            Some(self.load_resident_ln())
        } else {
            eprintln!("[npu] resident-ln xclbins absent in {} -- FFN LN->fc1 stays host (build ctxln+affcast for the on-NPU seam)", self.ln_dir.display());
            None
        };
        *self.resident_ln.borrow_mut() = Some(result.clone());
        result
    }

    fn load_resident_ln(&self) -> Rc<ResidentLn> {
        let load = |name: &str| -> (Rc<Kernel>, Bo, usize) {
            let xcl = self.ln_dir.join(format!("final_{name}_{PAD_M}x{KRES}.xclbin"));
            let ins = self.ln_dir.join(format!("insts_{name}_{PAD_M}x{KRES}.txt"));
            let kern = self
                .dev
                .load_kernel(xcl.to_str().unwrap(), None)
                .unwrap_or_else(|e| panic!("load resident-ln {} : {e:?}\n  prebuild: build ctxln+cast at {PAD_M}x{KRES} and copy to artifacts/parakeet/ln", xcl.display()));
            let ib = std::fs::read(&ins).unwrap_or_else(|e| panic!("read {}: {e}", ins.display()));
            let n = ib.len() / 4;
            let bo = self.dev.alloc_bo(&kern, ib.len(), FLAG_CACHEABLE, kern.group_id(1).unwrap()).unwrap();
            bo.write_bytes(&ib).unwrap();
            bo.sync_to_device().unwrap();
            (kern, bo, n)
        };
        let (ln_kern, ln_instr, ln_n) = load("ctxln");
        let (ac_kern, ac_instr, ac_n) = load("affcast");
        // cast @ DFF + the K=DFF fc2 matmul (explicit filenames, not the {name}_PADxKRES pattern)
        let load_path = |xcl: PathBuf, ins: PathBuf| -> (Rc<Kernel>, Bo, usize) {
            let kern = self.dev.load_kernel(xcl.to_str().unwrap(), None).unwrap_or_else(|e| panic!("load {} : {e:?}", xcl.display()));
            let ib = std::fs::read(&ins).unwrap_or_else(|e| panic!("read {}: {e}", ins.display()));
            let n = ib.len() / 4;
            let bo = self.dev.alloc_bo(&kern, ib.len(), FLAG_CACHEABLE, kern.group_id(1).unwrap()).unwrap();
            bo.write_bytes(&ib).unwrap();
            bo.sync_to_device().unwrap();
            (kern, bo, n)
        };
        let (deint_kern, deint_instr, deint_n) = load_path(
            self.ln_dir.join(format!("final_deint_{PAD_M}x{DFF}.xclbin")),
            self.ln_dir.join(format!("insts_deint_{PAD_M}x{DFF}.txt")));
        // conv-module GLU (step 2), OPTIONAL: load only if the glu xclbin was built. A/g3 input is fed
        // from the modal stream's bo_c (pw1 output); bo_out (g4) is the [PAD_M,KRES] f32 GLU result.
        let glu = {
            let xcl = self.ln_dir.join(format!("final_glu_{PAD_M}x{KRES}.xclbin"));
            let ins = self.ln_dir.join(format!("insts_glu_{PAD_M}x{KRES}.txt"));
            if xcl.exists() && ins.exists() {
                let (kern, instr, n) = load_path(xcl, ins);
                let gg = |i| kern.group_id(i).unwrap();
                Some(ConvGlu {
                    bo_out: self.dev.alloc_bo(&kern, PAD_M * KRES * 4, FLAG_HOST_ONLY, gg(4)).unwrap(),
                    dummy_c: self.dev.alloc_bo(&kern, 1, FLAG_HOST_ONLY, gg(5)).unwrap(),
                    dummy_tmp: self.dev.alloc_bo(&kern, 8, FLAG_HOST_ONLY, gg(6)).unwrap(),
                    dummy_tr: self.dev.alloc_bo(&kern, 1, FLAG_HOST_ONLY, gg(7)).unwrap(),
                    kern, instr, n,
                })
            } else {
                eprintln!("[npu] glu xclbin absent in {} -- conv GLU stays host (build final_glu_{PAD_M}x{KRES})", self.ln_dir.display());
                None
            }
        };
        // conv-module depthwise conv1d (step 3), OPTIONAL. 3-buffer ABI in[C,T]/w[C,16]/out[C,T] bf16.
        let dwconv = {
            let xcl = self.ln_dir.join(format!("final_dwconv_{DW_C}x{DW_T}.xclbin"));
            let ins = self.ln_dir.join(format!("insts_dwconv_{DW_C}x{DW_T}.txt"));
            if xcl.exists() && ins.exists() {
                let (kern, instr, n) = load_path(xcl, ins);
                let gw = |i| kern.group_id(i).unwrap();
                Some(ConvDw {
                    bo_in: self.dev.alloc_bo(&kern, DW_C * DW_T * 2, FLAG_HOST_ONLY, gw(3)).unwrap(),
                    bo_w: self.dev.alloc_bo(&kern, DW_C * DW_KW * 2, FLAG_HOST_ONLY, gw(4)).unwrap(),
                    bo_out: self.dev.alloc_bo(&kern, DW_C * DW_T * 2, FLAG_HOST_ONLY, gw(5)).unwrap(),
                    dummy_tmp: self.dev.alloc_bo(&kern, 8, FLAG_HOST_ONLY, gw(6)).unwrap(),
                    dummy_tr: self.dev.alloc_bo(&kern, 1, FLAG_HOST_ONLY, gw(7)).unwrap(),
                    kern, instr, n,
                })
            } else {
                eprintln!("[npu] dwconv xclbin absent in {} -- conv dwconv stays host (build final_dwconv_{DW_C}x{DW_T})", self.ln_dir.display());
                None
            }
        };
        let gl = |i| ln_kern.group_id(i).unwrap();
        let ga = |i| ac_kern.group_id(i).unwrap();
        let gd = |i| deint_kern.group_id(i).unwrap();
        let rl = Rc::new(ResidentLn {
            bo_x: self.dev.alloc_bo(&ln_kern, PAD_M * KRES * 4, FLAG_HOST_ONLY, gl(3)).unwrap(),
            bo_ln: self.dev.alloc_bo(&ln_kern, PAD_M * KRES * 4, FLAG_HOST_ONLY, gl(4)).unwrap(),
            bo_gb: self.dev.alloc_bo(&ac_kern, 2 * KRES * 4, FLAG_HOST_ONLY, ga(4)).unwrap(),
            bo_bf16: self.dev.alloc_bo(&ac_kern, PAD_M * KRES * 2, FLAG_HOST_ONLY, ga(5)).unwrap(),
            bo_deint: self.dev.alloc_bo(&deint_kern, (DFF / KRES) * PAD_M * KRES * 2, FLAG_HOST_ONLY, gd(4)).unwrap(),
            ln_c: self.dev.alloc_bo(&ln_kern, 1, FLAG_HOST_ONLY, gl(5)).unwrap(),
            ln_tmp: self.dev.alloc_bo(&ln_kern, 8, FLAG_HOST_ONLY, gl(6)).unwrap(),
            ln_tr: self.dev.alloc_bo(&ln_kern, 1, FLAG_HOST_ONLY, gl(7)).unwrap(),
            ac_tmp: self.dev.alloc_bo(&ac_kern, 8, FLAG_HOST_ONLY, ga(6)).unwrap(),
            ac_tr: self.dev.alloc_bo(&ac_kern, 1, FLAG_HOST_ONLY, ga(7)).unwrap(),
            deint_c: self.dev.alloc_bo(&deint_kern, 1, FLAG_HOST_ONLY, gd(5)).unwrap(),
            deint_tmp: self.dev.alloc_bo(&deint_kern, 8, FLAG_HOST_ONLY, gd(6)).unwrap(),
            deint_tr: self.dev.alloc_bo(&deint_kern, 1, FLAG_HOST_ONLY, gd(7)).unwrap(),
            ln_kern, ln_instr, ln_n, ac_kern, ac_instr, ac_n,
            deint_kern, deint_instr, deint_n, glu, dwconv,
        });
        rl
    }

    /// True when the resident on-NPU LN->fc1 seam is usable (modal resident + ctxln/affcast xclbins
    /// present). Lets `feed_forward` default to the resident path and fall back to host otherwise.
    pub fn resident_ff_available(&self) -> bool {
        self.modal && self.resident_ln().is_some()
    }

    /// On-chip normalize-only LN then AFFINE cast (*gamma+beta), chained DEVICE-SIDE (the
    /// intermediate bo_ln never touches host). Pads x[t,KRES] to [PAD_M,KRES]; gamma/beta [KRES]
    /// packed into bo_gb. Returns the resident block whose bo_bf16 holds affine_LN(x) as bf16, ready
    /// as the modal fc1's A input.
    fn ln_affine_cast(&self, x: &Array2<f32>, gamma: &[f32], beta: &[f32]) -> Rc<ResidentLn> {
        let (t, d) = x.dim();
        assert_eq!(d, KRES, "resident LN needs D=KRES={KRES}");
        assert!(t <= PAD_M, "T={t} exceeds PAD_M={PAD_M}");
        assert_eq!(gamma.len(), KRES);
        assert_eq!(beta.len(), KRES);
        // Only called on the resident path (gated by resident_ff_available), so the load succeeded.
        let rl = self.resident_ln().expect("ln_affine_cast without resident_ff_available()");
        let x_std = x.as_standard_layout();
        let mut buf = vec![0f32; PAD_M * KRES];
        buf[..t * KRES].copy_from_slice(&x_std.as_slice().unwrap()[..t * KRES]);
        rl.bo_x.write_bytes(f32_bytes(&buf)).unwrap();
        rl.bo_x.sync_to_device().unwrap();
        // gamma|beta packed on one channel
        let mut gb = vec![0f32; 2 * KRES];
        gb[..KRES].copy_from_slice(gamma);
        gb[KRES..].copy_from_slice(beta);
        rl.bo_gb.write_bytes(f32_bytes(&gb)).unwrap();
        rl.bo_gb.sync_to_device().unwrap();
        // (1) ctxLN: bo_x -> bo_ln  (NO sync back -- stays device-resident)
        rl.ln_kern.run_matmul8(3, &rl.ln_instr, rl.ln_n, &rl.bo_x, &rl.bo_ln, &rl.ln_c, &rl.ln_tmp, &rl.ln_tr).unwrap();
        // (2) affine_cast: (bo_ln * gamma + beta) -> bo_bf16  (device-side, no host round-trip)
        rl.ac_kern.run_matmul8(3, &rl.ac_instr, rl.ac_n, &rl.bo_ln, &rl.bo_gb, &rl.bo_bf16, &rl.ac_tmp, &rl.ac_tr).unwrap();
        self.stats.borrow_mut().dispatches += 2;
        rl
    }

    /// One modal-resident matmul dispatch whose A input is an ALREADY-device-resident bf16 BO
    /// (a_bo), skipping dispatch()'s host pack+upload. Output read to host (C[m,n] f32).
    fn dispatch_with_a(&self, a_bo: &Bo, m: usize, wbo: &Bo, n: usize, silu: bool) -> Array2<f32> {
        let st = self.stream(n, silu);
        self.kern.run_matmul8(3, &st.instr, st.n_instr, a_bo, wbo, &st.bo_c, &self.bo_tmp, &self.bo_tr).unwrap();
        self.stats.borrow_mut().dispatches += 1;
        st.bo_c.sync_from_device().unwrap();
        let mut cb = vec![0u8; m * n * 4];
        st.bo_c.read_bytes(&mut cb).unwrap();
        let mut out = Array2::<f32>::zeros((m, n));
        for r in 0..m {
            for c in 0..n {
                let off = (r * n + c) * 4;
                out[[r, c]] = f32::from_le_bytes([cb[off], cb[off + 1], cb[off + 2], cb[off + 3]]);
            }
        }
        out
    }

    /// Resident FF1 fc1 (LN->fc1 seam, the first frontier advance): on-chip normalize-only LN +
    /// AFFINE cast (device-side) -> modal fc1 with ON-CHIP SiLU and the UNMODIFIED weight W1.
    /// Returns `silu(affine_LN(x) @ W1)` [t,n] f32 -- exactly the host feed_forward fc1 (bf16-class),
    /// fully on-chip, no host reduction / bias / silu on this seam. `id` keys the W1 BO cache.
    /// (On a non-modal resident the on-chip silu is absent; the caller must apply host silu -- use
    /// [`Self::modal`] to branch, mirroring feed_forward.)
    /// Resident LN -> GEMM: ctxLN -> affine_cast(gamma,beta) -> modal GEMM [m,n], with the on-chip
    /// SiLU epilogue applied iff `silu` (n=DFF fc1 wants silu; conv pw1 / plain GEMMs want identity).
    pub fn resident_ff1_fc1<F: FnOnce() -> Array2<f32>>(&self, x: &Array2<f32>, gamma: &[f32], beta: &[f32], make_w1: F, id: &str, n: usize, silu: bool) -> Array2<f32> {
        self.stats.borrow_mut().calls += 1;
        let m = x.nrows();
        let rl = self.ln_affine_cast(x, gamma, beta);
        let cached = self.wcache.borrow().get(id).cloned();
        let wbo = if let Some(bo) = cached {
            bo
        } else {
            let w = make_w1();
            assert_eq!(w.nrows(), KRES, "W1 nrows {} != {KRES}", w.nrows());
            assert_eq!(w.ncols(), n, "W1 ncols {} != {n}", w.ncols());
            self.weight_bo(id, w.view())
        };
        self.dispatch_with_a(&rl.bo_bf16, m, &wbo, n, silu && self.modal)
    }

    /// Resident conv-module front (LN -> pw1 -> GLU), the conv step-2 frontier advance: the activation
    /// never touches host across the three ops.
    ///   ctxLN -> affine_cast -> modal pw1 GEMM (N=2*KRES, identity, output STAYS device in the stream
    ///   bo_c) -> GLU brick (a*sigmoid(g) over [PAD_M,2*KRES] -> [PAD_M,KRES], device-side) -> read [t,KRES].
    /// `make_w1` = pw1 weight [KRES, 2*KRES]; `id` keys the pw1 W BO cache (shared with resident_ff1_fc1,
    /// so warm passes hit). Returns None (caller keeps the host GLU) when the resident seam or the glu
    /// xclbin is absent -- so a tree without the glu kernel still gets step-1's resident LN->pw1.
    pub fn resident_conv_pw1_glu<F: FnOnce() -> Array2<f32>>(&self, x: &Array2<f32>, gamma: &[f32], beta: &[f32], make_w1: F, id: &str) -> Option<Array2<f32>> {
        let rl = self.resident_ln()?;
        let glu = rl.glu.as_ref()?; // glu xclbin absent -> None, caller falls back
        self.stats.borrow_mut().calls += 1;
        let m = x.nrows();
        let n2 = 2 * KRES; // pw1 output width 2D
        // LN + affine cast -> bo_bf16 = affine_LN(x) bf16 [PAD_M, KRES] (device).
        let rlc = self.ln_affine_cast(x, gamma, beta);
        // pw1 GEMM: A=bo_bf16, W1=[KRES,2D] identity-modal -> st.bo_c [PAD_M,2D] f32, STAYS on device.
        let cached = self.wcache.borrow().get(id).cloned();
        let wbo = if let Some(bo) = cached {
            bo
        } else {
            let w = make_w1();
            assert_eq!(w.nrows(), KRES, "pw1 W nrows {} != {KRES}", w.nrows());
            assert_eq!(w.ncols(), n2, "pw1 W ncols {} != {n2}", w.ncols());
            self.weight_bo(id, w.view())
        };
        let st = self.stream(n2, false); // identity modal (no on-chip silu on pw1)
        self.kern.run_matmul8(3, &st.instr, st.n_instr, &rlc.bo_bf16, &wbo, &st.bo_c, &self.bo_tmp, &self.bo_tr).unwrap();
        // GLU: st.bo_c [PAD_M,2D] f32 (A/g3) -> glu.bo_out [PAD_M,D] f32 (B/g4), device-side.
        glu.kern.run_matmul8(3, &glu.instr, glu.n, &st.bo_c, &glu.bo_out, &glu.dummy_c, &glu.dummy_tmp, &glu.dummy_tr).unwrap();
        self.stats.borrow_mut().dispatches += 2; // pw1 + glu
        // read the D-wide GLU output for the m real rows (row-major, first m rows contiguous).
        glu.bo_out.sync_from_device().unwrap();
        let mut cb = vec![0u8; m * KRES * 4];
        glu.bo_out.read_bytes(&mut cb).unwrap();
        let mut out = Array2::<f32>::zeros((m, KRES));
        for r in 0..m {
            for c in 0..KRES {
                let off = (r * KRES + c) * 4;
                out[[r, c]] = f32::from_le_bytes([cb[off], cb[off + 1], cb[off + 2], cb[off + 3]]);
            }
        }
        Some(out)
    }

    /// Host-fed on-NPU depthwise conv1d (conv step 3a): the sliding_mul FIR brick. `x_ct` = [C=1024, T]
    /// channel-major f32 (T <= 400), `taps` [C,9], `bias` [C]. Packs to bf16, runs the brick (in->w->out
    /// 3-buffer ABI), returns [C, T] f32. Transposes stay on host (killed in 3b, when x_ct is fed
    /// device-to-device from the GLU output). None if the dwconv xclbin is absent or T exceeds the baked
    /// DW_T (caller keeps the host dwconv1d).
    pub fn npu_dwconv1d(&self, x_ct: &Array2<f32>, taps: &Array2<f32>, bias: &Array1<f32>) -> Option<Array2<f32>> {
        let rl = self.resident_ln()?;
        let dw = rl.dwconv.as_ref()?;
        let (c, t) = x_ct.dim();
        if c != DW_C || t > DW_T {
            return None; // shape outside the baked brick -> host fallback
        }
        self.stats.borrow_mut().calls += 1;
        // pack input [C, t] f32 -> bf16 [C, DW_T] channel-major, zero-padding the time tail (t..DW_T).
        // 'same' conv sees zeros past the sequence end == correct end-padding; the pad outputs are sliced off.
        let x_std = x_ct.as_standard_layout();
        let xs = x_std.as_slice().unwrap();
        let mut in_bits = vec![0u16; DW_C * DW_T];
        for ch in 0..c {
            npu_xrt::pack_f32_to_bf16(&xs[ch * t..ch * t + t], &mut in_bits[ch * DW_T..ch * DW_T + t]);
        }
        dw.bo_in.write_bytes(u16_bytes(&in_bits)).unwrap();
        dw.bo_in.sync_to_device().unwrap();
        // pack weights [C,9] + bias[C] -> [C,16] bf16 (taps in [0..8], BN-folded bias in [9]).
        let taps_std = taps.as_standard_layout();
        let tp = taps_std.as_slice().unwrap();
        let mut w_bits = vec![0u16; DW_C * DW_KW];
        for ch in 0..c {
            let mut row = [0f32; DW_KW];
            row[..9].copy_from_slice(&tp[ch * 9..ch * 9 + 9]);
            row[9] = bias[ch];
            npu_xrt::pack_f32_to_bf16(&row, &mut w_bits[ch * DW_KW..ch * DW_KW + DW_KW]);
        }
        dw.bo_w.write_bytes(u16_bytes(&w_bits)).unwrap();
        dw.bo_w.sync_to_device().unwrap();
        // dispatch + read [C, DW_T] bf16 -> f32, slice to [C, t].
        dw.kern.run_matmul8(3, &dw.instr, dw.n, &dw.bo_in, &dw.bo_w, &dw.bo_out, &dw.dummy_tmp, &dw.dummy_tr).unwrap();
        self.stats.borrow_mut().dispatches += 1;
        dw.bo_out.sync_from_device().unwrap();
        let mut ob = vec![0u8; DW_C * DW_T * 2];
        dw.bo_out.read_bytes(&mut ob).unwrap();
        let mut out = Array2::<f32>::zeros((c, t));
        for ch in 0..c {
            for ti in 0..t {
                let off = (ch * DW_T + ti) * 2;
                let u = u16::from_le_bytes([ob[off], ob[off + 1]]);
                out[[ch, ti]] = f32::from_bits((u as u32) << 16);
            }
        }
        Some(out)
    }

    /// Full FFN device-side (LN -> fc1 -> SiLU -> fc2), the fc1->fc2 frontier step. Everything on-NPU,
    /// the activation stream never touching host across the whole FFN:
    ///   ctxLN -> affine_cast -> modal fc1 (on-chip silu, [t,DFF]) -> cast@DFF (bf16) -> K=DFF fc2
    ///   (identity, on-chip K-reduce, [t,KRES]) -> read [t,KRES] f32.
    /// No host K-split / accumulate. `make_w1` = [KRES,DFF] fc1 weight; `make_w2` = [DFF,KRES] fc2.
    pub fn resident_ffn<F1: FnOnce() -> Array2<f32>, F2: FnOnce() -> Array2<f32>>(
        &self, x: &Array2<f32>, gamma: &[f32], beta: &[f32],
        make_w1: F1, id1: &str, make_w2: F2, id2: &str,
    ) -> Array2<f32> {
        self.stats.borrow_mut().calls += 1;
        let m = x.nrows();
        let rl = self.ln_affine_cast(x, gamma, beta); // bo_bf16 = affine_LN bf16 [PAD_M,KRES]
        // fc1: modal, A=bo_bf16, W1, on-chip SiLU -> st1.bo_c (f32 [PAD_M,DFF]) -- stays DEVICE
        let w1 = {
            let c = self.wcache.borrow().get(id1).cloned();
            c.unwrap_or_else(|| {
                let w = make_w1();
                assert_eq!(w.dim(), (KRES, DFF), "fc1 W1 dim");
                self.weight_bo(id1, w.view())
            })
        };
        let st1 = self.stream(DFF, self.modal);
        self.kern.run_matmul8(3, &st1.instr, st1.n_instr, &rl.bo_bf16, &w1, &st1.bo_c, &self.bo_tmp, &self.bo_tr).unwrap();
        // deinterleave+cast: st1.bo_c (f32 [PAD_M,DFF]) -> rl.bo_deint (bf16 [parts,PAD_M,KRES]
        // chunk-major), device-side. One dispatch (chunk-major drain TAP). NOTE: this n-D output DMA
        // HANGS ("run did not complete") when the deint is a co-resident hw-context alongside the
        // modal (it works standalone) -- a multi-context n-D-DMA toolchain issue; see the debug note.
        rl.deint_kern.run_matmul8(3, &rl.deint_instr, rl.deint_n, &st1.bo_c, &rl.bo_deint, &rl.deint_c, &rl.deint_tmp, &rl.deint_tr).unwrap();
        self.stats.borrow_mut().dispatches += 2; // fc1 + deint
        // fc2 K-split: each K=KRES chunk is a device SUB-BUFFER of bo_deint; K=KRES modal (identity),
        // host-accumulate the `parts` partials in f32 -- bit-identical to the host K-split (WER-neutral).
        let parts = DFF / KRES;
        let chunk_bytes = PAD_M * KRES * 2;
        let need_w2 = (0..parts).any(|c| !self.wcache.borrow().contains_key(&format!("{id2}.{c}")));
        let w2 = if need_w2 {
            let w = make_w2();
            assert_eq!(w.dim(), (DFF, KRES), "fc2 W2 dim");
            Some(w)
        } else {
            None
        };
        let mut acc = Array2::<f32>::zeros((m, KRES));
        for c in 0..parts {
            let chunk = rl.bo_deint.sub(c * chunk_bytes, chunk_bytes).unwrap();
            let sid = format!("{id2}.{c}");
            let w2c = {
                let cc = self.wcache.borrow().get(&sid).cloned();
                cc.unwrap_or_else(|| {
                    let w = w2.as_ref().expect("w2 present on cache miss");
                    self.weight_bo(&sid, w.slice(s![c * KRES..(c + 1) * KRES, ..]))
                })
            };
            acc += &self.dispatch_with_a(&chunk, m, &w2c, KRES, false);
        }
        acc
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

fn f32_bytes(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
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
