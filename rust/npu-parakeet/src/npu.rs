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

// 8-head relpos-MHA CONVEYOR (opt-in PARAKEET_CONVEYOR_MHA=1). Real Parakeet dims, must match the
// conveyor_attn_iron.py 8-head build: TQ=8, DK=128, T padded 172->176 (a VL(16) multiple), GJ=4
// heads per MemTile group (the validated 3-MemTile-op recipe: split q+k, v-direct, ctx-join).
const CONV_TQ: usize = 8;
const CONV_DK: usize = 128;
const CONV_BUILT_T: usize = 176; // 172 padded to a VL-multiple; the 8-head conveyor's baked T
const CONV_GJ: usize = 4;        // heads per MemTile group (must match the generator's join)
// Key-mask sentinel packed into the BD_shifted belt for pad keys kk >= t. The conveyor kernel has no
// t_active word, so it softmaxes over all CONV_BUILT_T keys; k[pad]=0 makes q.k[pad]=0, so a large
// negative BD makes scores[pad]=CONV_KEY_MASK*inv_scale ~= -884 -> exp2 clamps to ~0 (masked). Host-only
// fix (no kernel change): reproduces the shipped relpos t_active masking for variable-length clips.
const CONV_KEY_MASK: f32 = -1.0e4;

// BD-ON-CHIP conveyor (opt-in PARAKEET_CONVEYOR_MHA_BDONCHIP=1). The 4th BD stage computes
// BD = rel_shift((q+bias_v) @ p^T) ON-CHIP (deletes the host BD precompute = the +19% regression),
// so the belt carries qpv (qu||qv) only + p resident, dispatched via the 5-BO run_bd_conveyor ABI.
// All-direct head-major layout (NOT the group-major join of the host-BD conveyor): the bd_onchip
// generator packs H heads head-major and drains ctx head-major. Shipped as H_BD heads/xclbin,
// ceil(n_heads/H_BD) dispatches (H=4x2 = the spec's 8-head fallback until H=8-in-1 clears the
// .split() deadlock). Real dims: TQ=8 DK=128 BUILT_T=176 P=2*BUILT_T-1=351 N_QT=22.
const CONV_BD_HEADS: usize = 4;                  // heads baked per BD-onchip xclbin dispatch (H_BD)
const CONV_BD_P: usize = 2 * CONV_BUILT_T - 1;   // 351 = ATTN_P, the baked rel-pos table rows
// insts word(s) holding the per-head t_active RTP immediate (baked default = CONV_BUILT_T). Filled by
// the device-side probe (dump insts.bin, find the CONV_BUILT_T occurrences at the RTP-write sites);
// EMPTY = no patch = full-length passthrough (correct for T==BUILT_T + the standalone gate). Short
// clips (t<BUILT_T) need these patched per dispatch -- see the turnkey device doc.
const CONV_BD_TACTIVE_WORDS: &[usize] = &[];

/// BD-carriage precision for the conveyor query belt (open-item C / SPLITP). Default PLAIN per the
/// Deliverable-1 gate (scripts/conveyor_bd_precision_check.py). Env PARAKEET_CONVEYOR_BD=split flips
/// to two-bf16 (hi+lo, ~14 mantissa bits) if the device 17-clip WER ever regresses vs 8.5.
#[derive(Clone, Copy, PartialEq)]
pub enum BdCarry { Plain, Split }
impl BdCarry {
    fn from_env() -> Self {
        match std::env::var("PARAKEET_CONVEYOR_BD").as_deref() {
            Ok("split") => BdCarry::Split,
            _ => BdCarry::Plain, // Deliverable-1 verdict: plain sufficient, half the BD belt bytes
        }
    }
    fn factor(self) -> usize { match self { BdCarry::Plain => 1, BdCarry::Split => 2 } }
    fn name(self) -> &'static str { match self { BdCarry::Plain => "plain-bf16", BdCarry::Split => "split-bf16" } }
}
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

/// Loaded 8-head relpos CONVEYOR (scores(relpos) -> softmax -> ctx across 24 tiles, ONE dispatch).
/// 4-BO ABI (instr | q | k | v | ctx), mirroring run_conveyor_attn.py. H + belt layout are baked into
/// the xclbin, so this is a single cached instance (not a per-T map). NOTE: unlike RelposK there is NO
/// t_active word to patch -- the conveyor kernel has no key-mask, so it softmaxes over all CONV_BUILT_T
/// keys (correct only when the clip's real T == CONV_BUILT_T; short clips need a kernel key-mask).
struct ConveyorK {
    kern: Rc<Kernel>,
    n_instr: usize,
    bo_instr: Bo,
    bo_q: Bo,
    bo_k: Bo,
    bo_v: Bo,
    bo_ctx: Bo,
    n_qt: usize,    // CONV_BUILT_T / CONV_TQ (query tiles streamed)
    qelem: usize,   // per-tile query-belt elems (carriage-dependent; asserts the xclbin match)
    n_heads: usize, // baked head count (columns)
}

/// Loaded BD-ON-CHIP conveyor (BD->scores->softmax->ctx, H_BD heads/xclbin, all-direct head-major).
/// 5-BO ABI (instr | qpv | p | k | v | ctx), mirroring run_bd_onchip.py. Unlike ConveyorK the instr
/// stream MAY be patched per dispatch (the t_active RTP immediate) so a MAX-T xclbin serves any t.
struct ConveyorBdK {
    kern: Rc<Kernel>,
    instr_template: Vec<u32>, // insts as u32 words; words[CONV_BD_TACTIVE_WORDS] = t_active (patched)
    n_instr: usize,
    bo_instr: Bo,
    bo_qpv: Bo,
    bo_p: Bo,
    bo_k: Bo,
    bo_v: Bo,
    bo_ctx: Bo,
    n_qt: usize,     // CONV_BUILT_T / CONV_TQ
    n_heads: usize,  // baked heads per dispatch (H_BD)
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
    conveyor_dir: PathBuf,                         // {root}/artifacts/conveyor (8-head xclbin)
    conveyor: RefCell<Option<Rc<ConveyorK>>>,      // loaded 8-head conveyor (H baked, single instance)
    conveyor_bd_dir: PathBuf,                       // {root}/artifacts/conveyor_bd (BD-onchip xclbin)
    conveyor_bd: RefCell<Option<Rc<ConveyorBdK>>>,  // loaded BD-onchip conveyor (H_BD baked per dispatch)
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
    bo_bf16: Rc<Bo>, // [PAD_M, KRES] bf16  (affine_cast output = modal fc1 A / device-in satt, ac g5)
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
    // resident-FFN fc2 on-device K-split accumulate (out = a + b, f32), OPTIONAL like glu. When
    // present, resident_ffn_dev sums the fc2 partials on-chip into ONE device BO (no host acc).
    acc_add: Option<AccAdd>,
    // scaled residual-add (out = a + 0.5*b, f32), OPTIONAL. The Macaron FFN residual x+0.5*ff on-chip.
    resadd_s050: Option<ResidualAdd>,
    // scaled residual-add (out = a + 1.0*b, f32), OPTIONAL. The full MHSA/conv residual x+sublayer.
    resadd_s100: Option<ResidualAdd>,
    // one-dispatch K=4096 fc2 (cast@4096 -> K=4096 modal), OPTIONAL. Collapses the 4x K=1024 + acc_add.
    fc2_k4096: Option<Fc2K4096>,
    // conv-module depthwise conv1d (step 3), OPTIONAL like glu.
    dwconv: Option<ConvDw>,
    // conv-module post-dwconv SiLU (step 4), OPTIONAL like glu/dwconv. SEPARATE single-op-loop
    // brick (NOT a dwconv epilogue) -- immune to the fused-epilogue per-channel-loop miscompile.
    silu: Option<ConvSilu>,
    // FUSED dwconv->SiLU (step 3+4 in one xclbin), OPTIONAL. When present it replaces the
    // separate dwconv + silu dispatches (one hw-context, no host bridge); absent -> the two-brick path.
    dwconv_silu: Option<ConvDwSilu>,
    // TIME-MAJOR fused dwconv->SiLU (step 3b), OPTIONAL. When present the conv path prefers it: [T,D]
    // in/out DISSOLVES both host transposes (vs the channel-major dwconv_silu which keeps them).
    dwconv_silu_t: Option<ConvDwSiluT>,
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

/// Device-side f32 accumulate-add brick (resident-FFN fc2 on-device K-split accumulation).
/// out[g5] = a[g3] + b[g4] over [PAD_M,KRES] f32. Used to sum the DFF/KRES fc2 partials into
/// ONE device BO (ping-pong `acc0`/`acc1`) instead of a host `Array2` -- bit-identical to the
/// host sequential f32 K-split (WER-neutral), but the FFN output stays device-resident. `zero`
/// (a persistent zeroed BO) seeds the first partial (acc = partial0 + 0). OPTIONAL like glu.
struct AccAdd {
    kern: Rc<Kernel>,
    instr: Bo,
    n: usize,
    acc0: Rc<Bo>, // [PAD_M, KRES] f32 ping accumulator
    acc1: Rc<Bo>, // [PAD_M, KRES] f32 pong accumulator
    zero: Bo,     // [PAD_M, KRES] f32, zeroed once (seed for the first partial)
    dummy_tmp: Bo,
    dummy_tr: Bo,
}

/// One-dispatch fc2 (K=DFF=4096) brick: replaces the 4x K=1024 chunk GEMMs + acc_add (which cost
/// separate hw-context dispatches) with `cast@4096 (f32->bf16 row-major) -> K=4096 modal GEMM (internal
/// L1 K-accumulation over 4096) -> f32 [PAD_M,KRES] device BO`. NOT bit-identical to the 4-way split
/// (different L1 accumulation order + bfp16), so gated by the sound rel-L2 gate, not per-op bit-parity.
struct Fc2K4096 {
    cast_kern: Rc<Kernel>,
    cast_instr: Bo,
    cast_n: usize,
    cast_out: Bo, // bf16 [PAD_M, DFF] row-major (cast output = K=4096 modal A input)
    cast_dc: Bo,
    cast_dt: Bo,
    cast_dr: Bo,
    mm_kern: Rc<Kernel>, // K=4096 modal (identity epilogue)
    mm_instr: Bo,
    mm_n: usize,
    mm_c: Rc<Bo>, // f32 [PAD_M, KRES] fc2 output (device-resident)
}

/// Device-side f32 scaled residual-add brick (whole-block fusion residual). out[g5] = a[g3] +
/// scale*b[g4] over [PAD_M,KRES] f32, `scale` baked into the xclbin (one per value: s050 = 0.5).
/// Keeps `x = x + scale*sublayer` on-chip so the residual never round-trips. OPTIONAL like acc_add.
struct ResidualAdd {
    kern: Rc<Kernel>,
    instr: Bo,
    n: usize,
    scale: f32,   // baked scale this xclbin applies (asserted against the caller's requested scale)
    bo_out: Rc<Bo>, // [PAD_M, KRES] f32 result (scratch; overwritten by the next call)
    dummy_tmp: Bo,
    dummy_tr: Bo,
}

// Conv-module depthwise conv1d (step 3): sliding_mul FIR along time, [C,T] channel-major bf16.
// T=400 is Parakeet's ~30s frame cap (>subsample); the brick bakes it. C=1024 = d_model.
const DW_C: usize = 1024; // channels (d_model)
const DW_T: usize = 400; // baked time steps (Parakeet frame cap)
const DW_KW: usize = 16; // weight tile: taps[0..8] + BN-folded bias[9]
// TIME-MAJOR fused dwconv+silu (conv step 3b): [T,D] layout. Input host-padded to [T+2P, D] (P=4 halo
// rows top+bottom); weights repacked TAP-MAJOR [K+1, D] (rows 0..8 per-channel taps, row 9 BN bias).
const DW_K: usize = 9; // depthwise kernel width
const DW_P: usize = 4; // 'same' pad = (K-1)/2
const DW_TPAD: usize = DW_T + 2 * DW_P; // padded input rows (=408)

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

// Conv-module post-dwconv SiLU brick (step 4): out[c,t] = silu(in[c,t]), [C,T] f32 -> f32, per-row
// (one channel's T-row per core loop). A SEPARATE single-op-loop kernel (silu_row), fed the dwconv
// output host-side (device-to-device in a later step). Same [C=1024,T=400] shape as the dwconv brick.
// 2-buffer ABI: in[C,T] f32 (g3), out[C,T] f32 (g4); tmp/ctrl/trace dummies (g5/g6/g7) -- like ctx_ln/glu.
struct ConvSilu {
    kern: Rc<Kernel>,
    instr: Bo,
    n: usize,
    bo_in: Bo,      // [C, T] f32 (g3)
    bo_out: Bo,     // [C, T] f32 (g4)
    dummy_tmp: Bo,  // g5
    dummy_ctrl: Bo, // g6
    dummy_tr: Bo,   // g7
}

// FUSED conv-module dwconv->SiLU brick (step 3+4 in ONE xclbin). A two-stage on-chip
// pipeline (dwconv core -> f32 ObjectFifo -> silu core, per column): the post-dwconv SiLU runs
// device-to-device with NO second hw-context switch and NO host round-trip -- collapsing the two
// separate ConvDw + ConvSilu xclbins (which each cost a ~1.9 ms switch) into one resident dispatch.
// Same 3-buffer ABI as ConvDw (in[C,T] bf16 g3, w[C,16] bf16 g4) but out[C,T] is f32 (g5). Both cores
// stay simple single-op loops, so it is immune to the alt-channel per-tile-loop miscompile. OPTIONAL.
struct ConvDwSilu {
    kern: Rc<Kernel>,
    instr: Bo,
    n: usize,
    bo_in: Bo,  // [C, T] bf16 (g3)
    bo_w: Bo,   // [C, 16] bf16 (g4)
    bo_out: Bo, // [C, T] f32 (g5)
    dummy_tmp: Bo,
    dummy_tr: Bo,
}

// TIME-MAJOR fused dwconv->SiLU brick (conv step 3b -- the transpose-DISSOLVING layout). Same two-stage
// on-chip pipeline as ConvDwSilu but in [T,D] instead of [C,T]: it consumes GLU's [T,D] directly and
// emits pw2's [T,D] directly, so BOTH host transposes (GLU[T,D]->[D,T] and [D,T]->[T,D]) are gone. The
// FIR vectorizes along D with the k=9 halo along TIME (consecutive row loads, NO shuffle / cross-column
// DMA -> immune to the n-D-DMA co-residency hang). 3-buffer ABI: in [T+2P, D] bf16 (g3, host-padded),
// w [K+1, D] bf16 TAP-MAJOR (g4), out [T, D] f32 (g5). OPTIONAL; present -> the Rust conv path prefers it.
struct ConvDwSiluT {
    kern: Rc<Kernel>,
    instr: Bo,
    n: usize,
    bo_in: Bo,  // [T+2P, D] bf16 (g3, host-padded)
    bo_w: Bo,   // [K+1, D] bf16 tap-major (g4)
    bo_out: Bo, // [T, D] f32 (g5)
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
            conveyor_dir: root.join("artifacts/conveyor"),
            conveyor: RefCell::new(None),
            conveyor_bd_dir: root.join("artifacts/conveyor_bd"),
            conveyor_bd: RefCell::new(None),
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

    /// Load (once) the 8-head relpos CONVEYOR built at CONV_BUILT_T by scripts/conveyor_prebuild.sh
    /// into {root}/artifacts/conveyor/single/. Static insts (no per-clip t_active patch), 4-BO ABI.
    fn conveyor_block(&self, n_heads: usize, qelem: usize) -> Rc<ConveyorK> {
        if let Some(k) = self.conveyor.borrow().as_ref() {
            assert_eq!(k.n_heads, n_heads, "conveyor xclbin baked for H={}, got {n_heads}", k.n_heads);
            assert_eq!(k.qelem, qelem, "conveyor belt qelem mismatch (carriage changed since load?)");
            return k.clone();
        }
        let dk = CONV_DK;
        let n_qt = CONV_BUILT_T / CONV_TQ;
        let dir = self.conveyor_dir.join("single");
        let xclbin = dir.join("final.xclbin");
        let insts = dir.join("insts.bin");
        let kern = self
            .dev
            .load_kernel(xclbin.to_str().unwrap(), None)
            .unwrap_or_else(|e| panic!("load conveyor single ({}): {e:?}\n  pre-build: scripts/conveyor_prebuild.sh", xclbin.display()));
        let ib = std::fs::read(&insts).unwrap_or_else(|e| panic!("read {}: {e}", insts.display()));
        let n_instr = ib.len() / 4;
        let g = |i| kern.group_id(i).unwrap();
        let bo_instr = self.dev.alloc_bo(&kern, ib.len(), FLAG_CACHEABLE, g(1)).unwrap();
        let bo_q = self.dev.alloc_bo(&kern, n_heads * n_qt * qelem * 2, FLAG_HOST_ONLY, g(3)).unwrap();
        let bo_k = self.dev.alloc_bo(&kern, n_heads * CONV_BUILT_T * dk * 2, FLAG_HOST_ONLY, g(4)).unwrap();
        let bo_v = self.dev.alloc_bo(&kern, n_heads * CONV_BUILT_T * dk * 2, FLAG_HOST_ONLY, g(5)).unwrap();
        let bo_ctx = self.dev.alloc_bo(&kern, n_heads * n_qt * CONV_TQ * dk * 2, FLAG_HOST_ONLY, g(6)).unwrap();
        bo_instr.write_bytes(&ib).unwrap(); // static instr stream -> upload once
        bo_instr.sync_to_device().unwrap();
        let ck = Rc::new(ConveyorK { kern, n_instr, bo_instr, bo_q, bo_k, bo_v, bo_ctx, n_qt, qelem, n_heads });
        *self.conveyor.borrow_mut() = Some(ck.clone());
        ck
    }

    /// Load (once) the BD-ONCHIP conveyor built at CONV_BUILT_T for H_BD heads by
    /// scripts/conveyor_bd_prebuild.sh into {root}/artifacts/conveyor_bd/single/. 5-BO ABI
    /// (instr | qpv | p | k | v | ctx). The insts template is cached so the t_active RTP word(s)
    /// can be patched per dispatch (short clips); default-baked t_active = CONV_BUILT_T.
    fn conveyor_bd_block(&self, n_heads: usize) -> Rc<ConveyorBdK> {
        if let Some(k) = self.conveyor_bd.borrow().as_ref() {
            assert_eq!(k.n_heads, n_heads, "bd-onchip xclbin baked for H_BD={}, got {n_heads}", k.n_heads);
            return k.clone();
        }
        let dk = CONV_DK;
        let n_qt = CONV_BUILT_T / CONV_TQ;
        let dir = self.conveyor_bd_dir.join("single");
        let xclbin = dir.join("final.xclbin");
        let insts = dir.join("insts.bin");
        let kern = self
            .dev
            .load_kernel(xclbin.to_str().unwrap(), None)
            .unwrap_or_else(|e| panic!("load conveyor_bd single ({}): {e:?}\n  pre-build: scripts/conveyor_bd_prebuild.sh", xclbin.display()));
        let ib = std::fs::read(&insts).unwrap_or_else(|e| panic!("read {}: {e}", insts.display()));
        let n_instr = ib.len() / 4;
        let instr_template: Vec<u32> = ib
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let g = |i| kern.group_id(i).unwrap();
        // 5 data BOs: qpv (g3) | p (g4) | k (g5) | v (g6) | ctx (g7). Sizes head-major over H_BD.
        let bo_instr = self.dev.alloc_bo(&kern, ib.len(), FLAG_CACHEABLE, g(1)).unwrap();
        let bo_qpv = self.dev.alloc_bo(&kern, n_heads * n_qt * 2 * CONV_TQ * dk * 2, FLAG_HOST_ONLY, g(3)).unwrap();
        let bo_p = self.dev.alloc_bo(&kern, n_heads * CONV_BD_P * dk * 2, FLAG_HOST_ONLY, g(4)).unwrap();
        let bo_k = self.dev.alloc_bo(&kern, n_heads * CONV_BUILT_T * dk * 2, FLAG_HOST_ONLY, g(5)).unwrap();
        let bo_v = self.dev.alloc_bo(&kern, n_heads * CONV_BUILT_T * dk * 2, FLAG_HOST_ONLY, g(6)).unwrap();
        let bo_ctx = self.dev.alloc_bo(&kern, n_heads * n_qt * CONV_TQ * dk * 2, FLAG_HOST_ONLY, g(7)).unwrap();
        let ck = Rc::new(ConveyorBdK {
            kern, instr_template, n_instr, bo_instr, bo_qpv, bo_p, bo_k, bo_v, bo_ctx, n_qt, n_heads,
        });
        *self.conveyor_bd.borrow_mut() = Some(ck.clone());
        ck
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

    /// 8-head relpos-MHA CONVEYOR (opt-in PARAKEET_CONVEYOR_MHA=1). Replaces the per-head
    /// `relpos_mha` LOOP (8 dispatches) with ONE 8-head conveyor dispatch (scores(relpos) ->
    /// softmax -> ctx, 8 heads x 3 tiles = 24 tiles, device-validated H=8 rel-L2 4.69e-3).
    ///
    /// This method owns the HOST-SIDE belt packing (the reviewable part):
    ///   * qu_h        = q_h + pos_bias_u[h]                              -> AC query, packed bf16.
    ///   * BD_shifted_h = rel_shift( (q_h + pos_bias_v[h]) @ p_h^T )      -> host-precomputed, packed
    ///                    into the belt AFTER qu_h (the conveyor's BD-in-belt design; no p resident).
    /// Carriage precision = `BdCarry` (env PARAKEET_CONVEYOR_BD, default PLAIN). Deliverable-1 gate
    /// (scripts/conveyor_bd_precision_check.py, block-0/T=32) found PLAIN sufficient: total ctx
    /// rel-L2 2.43e-3 == split, ~2x under the 5e-3 bf16 gate; the plain-vs-split carriage delta
    /// (2.4e-4 vs 3.8e-5) sits ~10x below the bf16 pipeline floor, so it never reaches ctx. SPLIT
    /// DOUBLES the BD belt bytes (BD is already why the relpos q belt runs depth-1) for no measured
    /// gain -> default PLAIN; flip to split only if the device 17-clip WER regresses vs 8.5.
    ///
    /// Inputs (host f32, as encoder.rs already has them): q/k/v [T, H*DK], pm [P, H*DK],
    /// ubias/vbias [H, DK]. Returns merged ctx [T, H*DK] (pre-linear_out; caller applies linear_out).
    ///
    /// NOTE: the actual 8-head xclbin LOAD + DISPATCH + output de-interleave is a TODO STUB below
    /// (needs artifacts/conveyor/single/{final.xclbin,insts.bin} from scripts/conveyor_prebuild.sh
    /// and the group-major join ABI from conveyor_attn_iron.py -- see CONVEYOR_INTEGRATION_RUNBOOK.md).
    pub fn relpos_mha_conveyor(
        &self,
        q: &Array2<f32>, k: &Array2<f32>, v: &Array2<f32>, pm: &Array2<f32>,
        ubias: &Array2<f32>, vbias: &Array2<f32>, n_heads: usize,
    ) -> Array2<f32> {
        let carry = BdCarry::from_env();
        let t = q.nrows();
        let p = pm.nrows(); // 2T-1
        let dk = CONV_DK;
        assert_eq!(dk, RELPOS_DK, "conveyor DK must match the baked head_dim");
        assert!(t <= CONV_BUILT_T, "clip T={t} exceeds conveyor BUILT_T={CONV_BUILT_T}");
        let n_qt = CONV_BUILT_T / CONV_TQ;                       // query tiles streamed (176/8 = 22)
        // per-tile query-belt element count: qu [TQ*DK] then BD_shifted [carry_factor * TQ*BUILT_T].
        let qelem = CONV_TQ * dk + carry.factor() * CONV_TQ * CONV_BUILT_T;

        // ---- host-side belt inputs: qu_all [H,T,DK] and BD (pre-shift) [H,T,P] ----
        let mut qu_all = Array3::<f32>::zeros((n_heads, t, dk));
        let mut bd_all = Array3::<f32>::zeros((n_heads, t, p));
        for h in 0..n_heads {
            let col = h * dk;
            let mut qv = Array2::<f32>::zeros((t, dk));
            for i in 0..t {
                for c in 0..dk {
                    let qi = q[[i, col + c]];
                    qu_all[[h, i, c]] = qi + ubias[[h, c]];
                    qv[[i, c]] = qi + vbias[[h, c]];
                }
            }
            let ph = pm.slice(s![.., col..col + dk]); // [P, DK]
            bd_all.slice_mut(s![h, .., ..]).assign(&qv.dot(&ph.t())); // [T, P]
        }
        // rel_shift the whole [H,T,P] -> [H,T,T] (reuses the shipped host brick). BD_shifted covers the
        // REAL t keys; pad keys kk >= t get CONV_KEY_MASK in the belt so the mask-free conveyor softmax
        // drives them to ~0 (see CONV_KEY_MASK). This makes the conveyor correct for variable-length T.
        let bd_sh = crate::ops::rel_shift(&bd_all, t); // [H,T,T]

        // ---- per-head belt: [N_QT * QELEM] f32, tile-major (q0 = qt*TQ) ----
        // per tile: qu rows [TQ,DK] (zero past t) then BD_shifted rows. PLAIN packs the bf16 value;
        // SPLIT packs hi(BUILT_T) then lo(BUILT_T) so the split kernel reconstructs (float)hi+(float)lo.
        let build_head_belt = |h: usize| -> Vec<f32> {
            let mut belt = Vec::<f32>::with_capacity(n_qt * qelem);
            for qt in 0..n_qt {
                let q0 = qt * CONV_TQ;
                // qu block
                for r in 0..CONV_TQ {
                    let i = q0 + r;
                    if i < t { belt.extend(qu_all.slice(s![h, i, ..]).iter().copied()); }
                    else { belt.extend(std::iter::repeat(0.0f32).take(dk)); }
                }
                // BD_shifted block(s): width BUILT_T, real keys in [0,t), zero pad beyond.
                let mut push_bd = |transform: &dyn Fn(f32) -> f32| {
                    for r in 0..CONV_TQ {
                        let i = q0 + r;
                        for kk in 0..CONV_BUILT_T {
                            // pad keys (kk >= t) get the mask sentinel; real query rows get real BD;
                            // pad query rows (i >= t) are discarded on de-interleave so 0.0 is fine.
                            let val = if kk >= t { CONV_KEY_MASK }
                                      else if i < t { bd_sh[[h, i, kk]] }
                                      else { 0.0 };
                            belt.push(transform(val));
                        }
                    }
                };
                match carry {
                    BdCarry::Plain => push_bd(&|x| x),                       // pack rounds to bf16
                    BdCarry::Split => {                                       // hi then lo
                        push_bd(&|x| bf16_round_f32(x));                     // hi (already bf16-valued)
                        push_bd(&|x| x - bf16_round_f32(x));                 // lo residual (pack rounds)
                    }
                }
            }
            debug_assert_eq!(belt.len(), n_qt * qelem);
            belt
        };

        // ---- group-major (GJ heads/MemTile group) step-interleave: per group, per tile, per head ----
        // matches conveyor_attn_iron.py's split q fill (stack heads-in-group on axis 1). k/v head-major.
        let head_belts: Vec<Vec<f32>> = (0..n_heads).map(build_head_belt).collect();
        let mut q_belt = Vec::<f32>::with_capacity(n_heads * n_qt * qelem);
        for g in (0..n_heads).step_by(CONV_GJ) {
            let gsz = CONV_GJ.min(n_heads - g);
            for qt in 0..n_qt {
                for i in 0..gsz {
                    let off = qt * qelem;
                    q_belt.extend_from_slice(&head_belts[g + i][off..off + qelem]);
                }
            }
        }
        // k / v head-major, each [H * BUILT_T * DK] with real rows in [0,t), zero pad (acquire-once).
        let mut k_pack = Vec::<f32>::with_capacity(n_heads * CONV_BUILT_T * dk);
        let mut v_pack = Vec::<f32>::with_capacity(n_heads * CONV_BUILT_T * dk);
        for h in 0..n_heads {
            let col = h * dk;
            push_pad_rows(&mut k_pack, &k.slice(s![.., col..col + dk]).to_owned(), 0, t, CONV_BUILT_T);
            push_pad_rows(&mut v_pack, &v.slice(s![.., col..col + dk]).to_owned(), 0, t, CONV_BUILT_T);
        }
        // pack f32 -> bf16 (device belt dtype).
        let mut qb = vec![0u16; q_belt.len()];
        let mut kb = vec![0u16; k_pack.len()];
        let mut vb = vec![0u16; v_pack.len()];
        npu_xrt::pack_f32_to_bf16(&q_belt, &mut qb);
        npu_xrt::pack_f32_to_bf16(&k_pack, &mut kb);
        npu_xrt::pack_f32_to_bf16(&v_pack, &mut vb);

        // ---- device dispatch: 4-BO conveyor ABI (instr | q | k | v | ctx), ONE run ----
        let ck = self.conveyor_block(n_heads, qelem);
        debug_assert_eq!(qb.len(), n_heads * n_qt * qelem);
        let t0 = Instant::now();
        ck.bo_q.write_bytes(u16_bytes(&qb)).unwrap();
        ck.bo_q.sync_to_device().unwrap();
        ck.bo_k.write_bytes(u16_bytes(&kb)).unwrap();
        ck.bo_k.sync_to_device().unwrap();
        ck.bo_v.write_bytes(u16_bytes(&vb)).unwrap();
        ck.bo_v.sync_to_device().unwrap();
        ck.kern.run_mha(3, &ck.bo_instr, ck.n_instr, &ck.bo_q, &ck.bo_k, &ck.bo_v, &ck.bo_ctx).unwrap();
        ck.bo_ctx.sync_from_device().unwrap();
        {
            let mut s = self.stats.borrow_mut();
            s.dispatch_s += t0.elapsed().as_secs_f64();
            s.dispatches += 1;
        }
        // ---- de-interleave bo_ctx -> merged ctx [t, H*DK] (run_conveyor_attn.py 88-96) ----
        // Heads group by CONV_GJ; each group drains contiguously as [N_QT, gsz, TQ, DK]. Per group,
        // element (qt,i,r,d) lives at group_base + (((qt*gsz + i)*TQ + r)*DK + d); it maps to head
        // h=g+i, ctx row (qt*TQ + r). Take the first t rows (pad rows qt*TQ+r >= t are dropped).
        let mut cb = vec![0u8; n_heads * n_qt * CONV_TQ * dk * 2];
        ck.bo_ctx.read_bytes(&mut cb).unwrap();
        let rd = |e: usize| -> f32 {
            let o = e * 2;
            f32::from_bits((u16::from_le_bytes([cb[o], cb[o + 1]]) as u32) << 16)
        };
        let mut ctx = Array2::<f32>::zeros((t, n_heads * dk));
        let mut base = 0usize;
        for g in (0..n_heads).step_by(CONV_GJ) {
            let gsz = CONV_GJ.min(n_heads - g);
            for i in 0..gsz {
                let h = g + i;
                for qt in 0..n_qt {
                    for r in 0..CONV_TQ {
                        let row = qt * CONV_TQ + r;
                        if row >= t { continue; }
                        for d in 0..dk {
                            ctx[[row, h * dk + d]] = rd(base + (((qt * gsz + i) * CONV_TQ + r) * dk + d));
                        }
                    }
                }
            }
            base += n_qt * gsz * CONV_TQ * dk;
        }
        ctx
    }

    /// BD-ON-CHIP 8-head MHSA conveyor (opt-in PARAKEET_CONVEYOR_MHA_BDONCHIP=1). This is the WIN
    /// variant of `relpos_mha_conveyor`: it DELETES the host BD precompute (the qv@p^T matmuls +
    /// rel_shift that doubled the mhsa host bucket = the measured +19% regression) and computes
    /// BD = rel_shift((q+bias_v) @ p^T) ON-CHIP as the 4th conveyor stage. The host now packs only:
    ///   * qu_h = q_h + pos_bias_u[h]   (the AC query, = q_pass in the belt head)
    ///   * qv_h = q_h + pos_bias_v[h]   (the BD query; the kernel dots it against p on-chip)
    ///   * p_h  = pm[:,h] real [2t-1,DK] table, zero-padded to the baked P=CONV_BD_P (rel_shift is a
    ///           function of key distance j-i only, so the real table + t_active base is correct).
    /// Belt = qpv (qu||qv per tile), head-major; p/k/v resident per head; 5-BO run_bd_conveyor.
    /// Dispatched CONV_BD_HEADS heads per xclbin (ceil(n_heads/H_BD) dispatches; H=4x2 fallback).
    ///
    /// t_active: the BD-onchip scores stage has NO host belt-sentinel (BD is in-kernel), so pad keys
    /// j>=t are masked in-kernel via the t_active RTP; the BD emit ALSO uses t_active for the
    /// rel_shift base (short clips t<BUILT_T). Host sets t_active by patching CONV_BD_TACTIVE_WORDS in
    /// the insts template per dispatch (empty until the device probe fills the offsets -> unpatched =
    /// full-length passthrough, correct for t==BUILT_T). See the turnkey device doc.
    ///
    /// Needs artifacts/conveyor_bd/single/{final.xclbin,insts.bin} (scripts/conveyor_bd_prebuild.sh
    /// built with --tactive-mask). Returns merged ctx [t, H*DK]; caller applies linear_out.
    pub fn relpos_mha_conveyor_bdonchip(
        &self,
        q: &Array2<f32>, k: &Array2<f32>, v: &Array2<f32>, pm: &Array2<f32>,
        ubias: &Array2<f32>, vbias: &Array2<f32>, n_heads: usize,
    ) -> Array2<f32> {
        let t = q.nrows();
        let p = pm.nrows(); // 2t-1
        let dk = CONV_DK;
        assert_eq!(dk, RELPOS_DK, "conveyor DK must match the baked head_dim");
        assert!(t <= CONV_BUILT_T, "clip T={t} exceeds conveyor BUILT_T={CONV_BUILT_T}");
        let n_qt = CONV_BUILT_T / CONV_TQ; // 22

        // ---- host-side belt inputs: qu_all [H,T,DK] and qv_all [H,T,DK] (NO BD precompute) ----
        let mut qu_all = Array3::<f32>::zeros((n_heads, t, dk));
        let mut qv_all = Array3::<f32>::zeros((n_heads, t, dk));
        for h in 0..n_heads {
            let col = h * dk;
            for i in 0..t {
                for c in 0..dk {
                    let qi = q[[i, col + c]];
                    qu_all[[h, i, c]] = qi + ubias[[h, c]];
                    qv_all[[h, i, c]] = qi + vbias[[h, c]];
                }
            }
        }

        let ck = self.conveyor_bd_block(CONV_BD_HEADS);
        // t_active: patch the insts template once (same t for every group), upload once, reuse.
        let mut insts = ck.instr_template.clone();
        for &w in CONV_BD_TACTIVE_WORDS {
            insts[w] = t as u32;
        }
        let instr_bytes: Vec<u8> = insts.iter().flat_map(|w| w.to_le_bytes()).collect();
        ck.bo_instr.write_bytes(&instr_bytes).unwrap();
        ck.bo_instr.sync_to_device().unwrap();

        // per-tile query-belt = qu block [TQ,DK] then qv block [TQ,DK] (pad rows past t are zero).
        let build_head_qpv = |h: usize, out: &mut Vec<f32>| {
            for qt in 0..n_qt {
                let q0 = qt * CONV_TQ;
                for r in 0..CONV_TQ {
                    let i = q0 + r;
                    if i < t { out.extend(qu_all.slice(s![h, i, ..]).iter().copied()); }
                    else { out.extend(std::iter::repeat(0.0f32).take(dk)); }
                }
                for r in 0..CONV_TQ {
                    let i = q0 + r;
                    if i < t { out.extend(qv_all.slice(s![h, i, ..]).iter().copied()); }
                    else { out.extend(std::iter::repeat(0.0f32).take(dk)); }
                }
            }
        };

        let mut ctx = Array2::<f32>::zeros((t, n_heads * dk));
        // ---- per head-group (H_BD heads/xclbin) pack -> dispatch -> de-interleave ----
        for g in (0..n_heads).step_by(CONV_BD_HEADS) {
            let gsz = CONV_BD_HEADS.min(n_heads - g);
            // buffers are baked for exactly H_BD heads; a ragged final group (gsz < H_BD) pads the
            // trailing head slots with zeros (their ctx is discarded below).
            let hb = CONV_BD_HEADS;
            let mut qpv_pack = Vec::<f32>::with_capacity(hb * n_qt * 2 * CONV_TQ * dk);
            let mut p_pack = Vec::<f32>::with_capacity(hb * CONV_BD_P * dk);
            let mut k_pack = Vec::<f32>::with_capacity(hb * CONV_BUILT_T * dk);
            let mut v_pack = Vec::<f32>::with_capacity(hb * CONV_BUILT_T * dk);
            for slot in 0..hb {
                let h = g + slot;
                if h < n_heads {
                    build_head_qpv(h, &mut qpv_pack);
                    let col = h * dk;
                    push_pad_rows(&mut p_pack, &pm.slice(s![.., col..col + dk]).to_owned(), 0, p, CONV_BD_P);
                    push_pad_rows(&mut k_pack, &k.slice(s![.., col..col + dk]).to_owned(), 0, t, CONV_BUILT_T);
                    push_pad_rows(&mut v_pack, &v.slice(s![.., col..col + dk]).to_owned(), 0, t, CONV_BUILT_T);
                } else {
                    qpv_pack.extend(std::iter::repeat(0.0f32).take(n_qt * 2 * CONV_TQ * dk));
                    p_pack.extend(std::iter::repeat(0.0f32).take(CONV_BD_P * dk));
                    k_pack.extend(std::iter::repeat(0.0f32).take(CONV_BUILT_T * dk));
                    v_pack.extend(std::iter::repeat(0.0f32).take(CONV_BUILT_T * dk));
                }
            }
            let mut qb = vec![0u16; qpv_pack.len()];
            let mut pb = vec![0u16; p_pack.len()];
            let mut kb = vec![0u16; k_pack.len()];
            let mut vb = vec![0u16; v_pack.len()];
            npu_xrt::pack_f32_to_bf16(&qpv_pack, &mut qb);
            npu_xrt::pack_f32_to_bf16(&p_pack, &mut pb);
            npu_xrt::pack_f32_to_bf16(&k_pack, &mut kb);
            npu_xrt::pack_f32_to_bf16(&v_pack, &mut vb);

            let t0 = Instant::now();
            ck.bo_qpv.write_bytes(u16_bytes(&qb)).unwrap();
            ck.bo_qpv.sync_to_device().unwrap();
            ck.bo_p.write_bytes(u16_bytes(&pb)).unwrap();
            ck.bo_p.sync_to_device().unwrap();
            ck.bo_k.write_bytes(u16_bytes(&kb)).unwrap();
            ck.bo_k.sync_to_device().unwrap();
            ck.bo_v.write_bytes(u16_bytes(&vb)).unwrap();
            ck.bo_v.sync_to_device().unwrap();
            ck.kern.run_bd_conveyor(3, &ck.bo_instr, ck.n_instr, &ck.bo_qpv, &ck.bo_p, &ck.bo_k, &ck.bo_v, &ck.bo_ctx).unwrap();
            ck.bo_ctx.sync_from_device().unwrap();
            {
                let mut s = self.stats.borrow_mut();
                s.dispatch_s += t0.elapsed().as_secs_f64();
                s.dispatches += 1;
            }
            // de-interleave: head-major ctx, head slot at slot*n_qt*TQ*DK, row = qt*TQ+r (take [0,t)).
            let mut cb = vec![0u8; hb * n_qt * CONV_TQ * dk * 2];
            ck.bo_ctx.read_bytes(&mut cb).unwrap();
            let rd = |e: usize| -> f32 {
                let o = e * 2;
                f32::from_bits((u16::from_le_bytes([cb[o], cb[o + 1]]) as u32) << 16)
            };
            for slot in 0..gsz {
                let h = g + slot;
                let base = slot * n_qt * CONV_TQ * dk;
                for qt in 0..n_qt {
                    for r in 0..CONV_TQ {
                        let row = qt * CONV_TQ + r;
                        if row >= t { continue; }
                        for d in 0..dk {
                            ctx[[row, h * dk + d]] = rd(base + ((qt * CONV_TQ + r) * dk + d));
                        }
                    }
                }
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
        // resident-FFN fc2 on-device accumulate (out=a+b f32), OPTIONAL: load only if built. acc0/acc1
        // ping-pong the running sum; `zero` (zeroed once) seeds the first partial (acc = partial0 + 0).
        let acc_add = {
            let xcl = self.ln_dir.join(format!("final_accadd_{PAD_M}x{KRES}.xclbin"));
            let ins = self.ln_dir.join(format!("insts_accadd_{PAD_M}x{KRES}.txt"));
            if xcl.exists() && ins.exists() {
                let (kern, instr, n) = load_path(xcl, ins);
                let gaa = |i| kern.group_id(i).unwrap();
                let zero = self.dev.alloc_bo(&kern, PAD_M * KRES * 4, FLAG_HOST_ONLY, gaa(4)).unwrap();
                zero.write_bytes(&vec![0u8; PAD_M * KRES * 4]).unwrap();
                zero.sync_to_device().unwrap();
                Some(AccAdd {
                    acc0: Rc::new(self.dev.alloc_bo(&kern, PAD_M * KRES * 4, FLAG_HOST_ONLY, gaa(5)).unwrap()),
                    acc1: Rc::new(self.dev.alloc_bo(&kern, PAD_M * KRES * 4, FLAG_HOST_ONLY, gaa(5)).unwrap()),
                    zero,
                    dummy_tmp: self.dev.alloc_bo(&kern, 8, FLAG_HOST_ONLY, gaa(6)).unwrap(),
                    dummy_tr: self.dev.alloc_bo(&kern, 1, FLAG_HOST_ONLY, gaa(7)).unwrap(),
                    kern, instr, n,
                })
            } else {
                eprintln!("[npu] acc_add xclbin absent in {} -- resident_ffn_dev unavailable (build final_accadd_{PAD_M}x{KRES})", self.ln_dir.display());
                None
            }
        };
        // scaled residual-add s050 (out = a + 0.5*b, f32), OPTIONAL: the Macaron FFN residual on-chip.
        let resadd_s050 = {
            let xcl = self.ln_dir.join(format!("final_resadd_{PAD_M}x{KRES}_s050.xclbin"));
            let ins = self.ln_dir.join(format!("insts_resadd_{PAD_M}x{KRES}_s050.txt"));
            if xcl.exists() && ins.exists() {
                let (kern, instr, n) = load_path(xcl, ins);
                let gr = |i| kern.group_id(i).unwrap();
                Some(ResidualAdd {
                    scale: 0.5,
                    bo_out: Rc::new(self.dev.alloc_bo(&kern, PAD_M * KRES * 4, FLAG_HOST_ONLY, gr(5)).unwrap()),
                    dummy_tmp: self.dev.alloc_bo(&kern, 8, FLAG_HOST_ONLY, gr(6)).unwrap(),
                    dummy_tr: self.dev.alloc_bo(&kern, 1, FLAG_HOST_ONLY, gr(7)).unwrap(),
                    kern, instr, n,
                })
            } else {
                eprintln!("[npu] resadd_s050 xclbin absent in {} -- residual_add_dev(0.5) unavailable (build final_resadd_{PAD_M}x{KRES}_s050)", self.ln_dir.display());
                None
            }
        };
        // scaled residual-add s100 (out=a+1.0*b f32), OPTIONAL: the full MHSA/conv residual x+sublayer.
        let resadd_s100 = {
            let xcl = self.ln_dir.join(format!("final_resadd_{PAD_M}x{KRES}_s100.xclbin"));
            let ins = self.ln_dir.join(format!("insts_resadd_{PAD_M}x{KRES}_s100.txt"));
            if xcl.exists() && ins.exists() {
                let (kern, instr, n) = load_path(xcl, ins);
                let gr = |i| kern.group_id(i).unwrap();
                Some(ResidualAdd {
                    scale: 1.0,
                    bo_out: Rc::new(self.dev.alloc_bo(&kern, PAD_M * KRES * 4, FLAG_HOST_ONLY, gr(5)).unwrap()),
                    dummy_tmp: self.dev.alloc_bo(&kern, 8, FLAG_HOST_ONLY, gr(6)).unwrap(),
                    dummy_tr: self.dev.alloc_bo(&kern, 1, FLAG_HOST_ONLY, gr(7)).unwrap(),
                    kern, instr, n,
                })
            } else {
                eprintln!("[npu] resadd_s100 xclbin absent in {} -- residual_add_dev(1.0) unavailable (build final_resadd_{PAD_M}x{KRES}_s100)", self.ln_dir.display());
                None
            }
        };
        // one-dispatch K=DFF fc2 (cast@DFF row-major bf16 -> K=DFF modal), OPTIONAL: collapses the
        // deint + 4x K=1024 chunk GEMMs + 4x acc_add into cast + 1 K=4096 modal. Both xclbins are
        // built+staged by build_parakeet_modal_kernels.sh (cast_512x4096, 512x4096x1024 modalid).
        let fc2_k4096 = {
            let cast_x = self.ln_dir.join(format!("final_cast_{PAD_M}x{DFF}.xclbin"));
            let cast_i = self.ln_dir.join(format!("insts_cast_{PAD_M}x{DFF}.txt"));
            let mm_x = self.ln_dir.join(format!("final_{PAD_M}x{DFF}x{KRES}_{}_8c_modalid.xclbin", self.tile));
            let mm_i = self.ln_dir.join(format!("insts_{PAD_M}x{DFF}x{KRES}_{}_8c_modalid.txt", self.tile));
            if cast_x.exists() && cast_i.exists() && mm_x.exists() && mm_i.exists() {
                let (cast_kern, cast_instr, cast_n) = load_path(cast_x, cast_i);
                let (mm_kern, mm_instr, mm_n) = load_path(mm_x, mm_i);
                let gc = |i| cast_kern.group_id(i).unwrap();
                let gm = |i| mm_kern.group_id(i).unwrap();
                Some(Fc2K4096 {
                    cast_out: self.dev.alloc_bo(&cast_kern, PAD_M * DFF * 2, FLAG_HOST_ONLY, gc(4)).unwrap(),
                    cast_dc: self.dev.alloc_bo(&cast_kern, 1, FLAG_HOST_ONLY, gc(5)).unwrap(),
                    cast_dt: self.dev.alloc_bo(&cast_kern, 8, FLAG_HOST_ONLY, gc(6)).unwrap(),
                    cast_dr: self.dev.alloc_bo(&cast_kern, 1, FLAG_HOST_ONLY, gc(7)).unwrap(),
                    mm_c: Rc::new(self.dev.alloc_bo(&mm_kern, PAD_M * KRES * 4, FLAG_HOST_ONLY, gm(5)).unwrap()),
                    cast_kern, cast_instr, cast_n, mm_kern, mm_instr, mm_n,
                })
            } else {
                eprintln!("[npu] fc2_k4096 xclbins absent in {} -- one-dispatch fc2 unavailable (build cast_{PAD_M}x{DFF} + {PAD_M}x{DFF}x{KRES} modal)", self.ln_dir.display());
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
        // conv-module post-dwconv SiLU (step 4), OPTIONAL. 2-buffer ABI in[C,T]/out[C,T] f32.
        let silu = {
            let xcl = self.ln_dir.join(format!("final_silu_{DW_C}x{DW_T}.xclbin"));
            let ins = self.ln_dir.join(format!("insts_silu_{DW_C}x{DW_T}.txt"));
            if xcl.exists() && ins.exists() {
                let (kern, instr, n) = load_path(xcl, ins);
                let gs = |i| kern.group_id(i).unwrap();
                Some(ConvSilu {
                    bo_in: self.dev.alloc_bo(&kern, DW_C * DW_T * 4, FLAG_HOST_ONLY, gs(3)).unwrap(),
                    bo_out: self.dev.alloc_bo(&kern, DW_C * DW_T * 4, FLAG_HOST_ONLY, gs(4)).unwrap(),
                    dummy_tmp: self.dev.alloc_bo(&kern, 8, FLAG_HOST_ONLY, gs(5)).unwrap(),
                    dummy_ctrl: self.dev.alloc_bo(&kern, 8, FLAG_HOST_ONLY, gs(6)).unwrap(),
                    dummy_tr: self.dev.alloc_bo(&kern, 1, FLAG_HOST_ONLY, gs(7)).unwrap(),
                    kern, instr, n,
                })
            } else {
                eprintln!("[npu] silu xclbin absent in {} -- conv SiLU stays host (build final_silu_{DW_C}x{DW_T})", self.ln_dir.display());
                None
            }
        };
        // FUSED dwconv->SiLU (step 3+4, one xclbin), OPTIONAL. 3-buffer ABI in[C,T] bf16 / w[C,16] bf16 /
        // out[C,T] f32 (== ConvDw ABI, f32 out). Present -> replaces the separate dwconv+silu dispatches.
        let dwconv_silu = {
            let xcl = self.ln_dir.join(format!("final_dwconv_silu_{DW_C}x{DW_T}.xclbin"));
            let ins = self.ln_dir.join(format!("insts_dwconv_silu_{DW_C}x{DW_T}.txt"));
            if xcl.exists() && ins.exists() {
                let (kern, instr, n) = load_path(xcl, ins);
                let gw = |i| kern.group_id(i).unwrap();
                Some(ConvDwSilu {
                    bo_in: self.dev.alloc_bo(&kern, DW_C * DW_T * 2, FLAG_HOST_ONLY, gw(3)).unwrap(),
                    bo_w: self.dev.alloc_bo(&kern, DW_C * DW_KW * 2, FLAG_HOST_ONLY, gw(4)).unwrap(),
                    bo_out: self.dev.alloc_bo(&kern, DW_C * DW_T * 4, FLAG_HOST_ONLY, gw(5)).unwrap(),
                    dummy_tmp: self.dev.alloc_bo(&kern, 8, FLAG_HOST_ONLY, gw(6)).unwrap(),
                    dummy_tr: self.dev.alloc_bo(&kern, 1, FLAG_HOST_ONLY, gw(7)).unwrap(),
                    kern, instr, n,
                })
            } else {
                eprintln!("[npu] fused dwconv+silu xclbin absent in {} -- separate dwconv+silu path (build final_dwconv_silu_{DW_C}x{DW_T})", self.ln_dir.display());
                None
            }
        };
        // TIME-MAJOR fused dwconv->SiLU (step 3b), OPTIONAL. 3-buffer ABI: in [T+2P,D] bf16 (g3,
        // host-padded), w [K+1,D] bf16 tap-major (g4), out [T,D] f32 (g5). Present -> conv path prefers
        // it (dissolves both host transposes); absent -> channel-major dwconv_silu / separate bricks.
        let dwconv_silu_t = {
            let xcl = self.ln_dir.join(format!("final_dwconv_silu_t_{DW_C}x{DW_T}.xclbin"));
            let ins = self.ln_dir.join(format!("insts_dwconv_silu_t_{DW_C}x{DW_T}.txt"));
            if xcl.exists() && ins.exists() {
                let (kern, instr, n) = load_path(xcl, ins);
                let gw = |i| kern.group_id(i).unwrap();
                Some(ConvDwSiluT {
                    bo_in: self.dev.alloc_bo(&kern, DW_TPAD * DW_C * 2, FLAG_HOST_ONLY, gw(3)).unwrap(),
                    bo_w: self.dev.alloc_bo(&kern, (DW_K + 1) * DW_C * 2, FLAG_HOST_ONLY, gw(4)).unwrap(),
                    bo_out: self.dev.alloc_bo(&kern, DW_T * DW_C * 4, FLAG_HOST_ONLY, gw(5)).unwrap(),
                    dummy_tmp: self.dev.alloc_bo(&kern, 8, FLAG_HOST_ONLY, gw(6)).unwrap(),
                    dummy_tr: self.dev.alloc_bo(&kern, 1, FLAG_HOST_ONLY, gw(7)).unwrap(),
                    kern, instr, n,
                })
            } else {
                eprintln!("[npu] time-major fused dwconv+silu xclbin absent in {} -- channel-major path w/ host transposes (build final_dwconv_silu_t_{DW_C}x{DW_T})", self.ln_dir.display());
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
            bo_bf16: Rc::new(self.dev.alloc_bo(&ac_kern, PAD_M * KRES * 2, FLAG_HOST_ONLY, ga(5)).unwrap()),
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
            deint_kern, deint_instr, deint_n, glu, acc_add, resadd_s050, resadd_s100, fc2_k4096, dwconv, silu, dwconv_silu, dwconv_silu_t,
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

    /// Device-in variant of [`Self::ln_affine_cast`]: the ctxLN input is an ALREADY-device-resident
    /// f32 [PAD_M,KRES] BO `a_bo` (the previous brick's output = FFN/residual output), so the host
    /// `bo_x` write+`sync_to` is SKIPPED -- the LN never round-trips to host. gamma/beta stay
    /// host-written (small per-block const). Returns the shared ResidentLn whose `bo_bf16` holds
    /// affine_LN(a) bf16 [PAD_M,KRES], ready as the next modal GEMM's device A input.
    fn ln_affine_cast_dev(&self, a_bo: &Bo, gamma: &[f32], beta: &[f32]) -> Rc<ResidentLn> {
        assert_eq!(gamma.len(), KRES);
        assert_eq!(beta.len(), KRES);
        let rl = self.resident_ln().expect("ln_affine_cast_dev without resident_ff_available()");
        // gamma|beta packed on one channel (host-written, small per-block const)
        let mut gb = vec![0f32; 2 * KRES];
        gb[..KRES].copy_from_slice(gamma);
        gb[KRES..].copy_from_slice(beta);
        rl.bo_gb.write_bytes(f32_bytes(&gb)).unwrap();
        rl.bo_gb.sync_to_device().unwrap();
        // (1) ctxLN: a_bo -> bo_ln  (DEVICE-IN: no host write of x; stays device-resident)
        rl.ln_kern.run_matmul8(3, &rl.ln_instr, rl.ln_n, a_bo, &rl.bo_ln, &rl.ln_c, &rl.ln_tmp, &rl.ln_tr).unwrap();
        // (2) affine_cast: (bo_ln * gamma + beta) -> bo_bf16  (device-side)
        rl.ac_kern.run_matmul8(3, &rl.ac_instr, rl.ac_n, &rl.bo_ln, &rl.bo_gb, &rl.bo_bf16, &rl.ac_tmp, &rl.ac_tr).unwrap();
        self.stats.borrow_mut().dispatches += 2;
        rl
    }

    /// Device-parity self-test for Task 3 (device-in LN). Uploads synthetic x to a device BO, runs
    /// [`Self::ln_affine_cast_dev`], reads `bo_bf16` back, compares to host `ops::layernorm(x,g,b)`.
    /// bf16 output -> rel-L2 <= 5e-3. `None` when the resident-ln (ctxln/affcast) xclbins are absent.
    pub fn ln_affine_cast_dev_selftest(&self, t: usize, seed: u64) -> Option<(Array2<f32>, Array2<f32>)> {
        let rl = self.resident_ln()?;
        let fill = |rows: usize, cols: usize, sd: u64, sc: f32| -> Array2<f32> {
            let mut s = sd.wrapping_add(0x9E37_79B9_7F4A_7C15);
            Array2::from_shape_fn((rows, cols), |_| {
                s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = s;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^= z >> 31;
                let u = (z >> 40) as f32 / (1u32 << 24) as f32;
                (u * 2.0 - 1.0) * sc
            })
        };
        let x = fill(t, KRES, seed, 1.0);
        let gv: Vec<f32> = fill(1, KRES, seed ^ 0x1A, 1.0).iter().copied().collect(); // affine scale ~1
        let bv: Vec<f32> = fill(1, KRES, seed ^ 0x2B, 0.1).iter().copied().collect();
        // Upload x into a device f32 [PAD_M,KRES] BO (first t rows real; the rest zero -> ignored).
        let a_bo = self.dev.alloc_bo(&rl.ln_kern, PAD_M * KRES * 4, FLAG_HOST_ONLY, rl.ln_kern.group_id(3).unwrap()).unwrap();
        let mut buf = vec![0f32; PAD_M * KRES];
        let xs = x.as_standard_layout();
        buf[..t * KRES].copy_from_slice(&xs.as_slice().unwrap()[..t * KRES]);
        a_bo.write_bytes(f32_bytes(&buf)).unwrap();
        a_bo.sync_to_device().unwrap();
        let rl2 = self.ln_affine_cast_dev(&a_bo, &gv, &bv);
        rl2.bo_bf16.sync_from_device().unwrap();
        let mut cb = vec![0u8; t * KRES * 2]; // bf16, first t rows (row-major)
        rl2.bo_bf16.read_bytes(&mut cb).unwrap();
        let mut dev = Array2::<f32>::zeros((t, KRES));
        for r in 0..t {
            for c in 0..KRES {
                let off = (r * KRES + c) * 2;
                let bits = u16::from_le_bytes([cb[off], cb[off + 1]]);
                dev[[r, c]] = f32::from_bits((bits as u32) << 16);
            }
        }
        let g1 = Array1::from(gv);
        let b1 = Array1::from(bv);
        let host = crate::ops::layernorm(&x, &g1, &b1);
        Some((host, dev))
    }

    /// True when the whole-block fused seam (PARAKEET_FUSED_BLOCK) can run: modal resident + the
    /// resident-ln seam + the fc2-accumulate (acc_add) + the Macaron residual (resadd_s050) bricks.
    pub fn resident_fused_available(&self) -> bool {
        if !self.modal {
            return false;
        }
        match self.resident_ln() {
            Some(rl) => rl.acc_add.is_some() && rl.resadd_s050.is_some(),
            None => false,
        }
    }

    /// Upload a host activation `x` [m, KRES] into a fresh device f32 [PAD_M,KRES] BO (the resident
    /// stream head): the block uploads x ONCE here, then every brick reads/writes device BOs.
    pub fn upload_stream(&self, x: &Array2<f32>) -> Rc<Bo> {
        let m = x.nrows();
        assert!(m <= PAD_M, "T={m} exceeds PAD_M={PAD_M}");
        assert_eq!(x.ncols(), KRES, "stream needs D=KRES={KRES}");
        let bo = self.dev.alloc_bo(&self.kern, PAD_M * KRES * 4, FLAG_HOST_ONLY, self.kern.group_id(3).unwrap()).unwrap();
        let xs = x.as_standard_layout();
        let mut buf = vec![0f32; PAD_M * KRES];
        buf[..m * KRES].copy_from_slice(&xs.as_slice().unwrap()[..m * KRES]);
        bo.write_bytes(f32_bytes(&buf)).unwrap();
        bo.sync_to_device().unwrap();
        Rc::new(bo)
    }

    /// Read a device f32 [PAD_M,KRES] BO back to a host [m, KRES] array (the block/encoder boundary).
    pub fn readback_stream(&self, bo: &Bo, m: usize) -> Array2<f32> {
        bo.sync_from_device().unwrap();
        let mut cb = vec![0u8; m * KRES * 4];
        bo.read_bytes(&mut cb).unwrap();
        let mut out = Array2::<f32>::zeros((m, KRES));
        for r in 0..m {
            for c in 0..KRES {
                let off = (r * KRES + c) * 4;
                out[[r, c]] = f32::from_le_bytes([cb[off], cb[off + 1], cb[off + 2], cb[off + 3]]);
            }
        }
        out
    }

    /// Device-in LN for the seam: run [`Self::ln_affine_cast_dev`] and hand back the shared `bo_bf16`
    /// (affine_LN(a_bo) bf16 [PAD_M,KRES]) as an owned handle, ready to feed the MHSA projections via
    /// [`Self::proj_from_bf16`]. `None` when the resident-ln xclbins are absent.
    pub fn ln_affine_cast_dev_bf16(&self, a_bo: &Bo, gamma: &[f32], beta: &[f32]) -> Option<Rc<Bo>> {
        self.resident_ln()?;
        let rl = self.ln_affine_cast_dev(a_bo, gamma, beta);
        Some(rl.bo_bf16.clone())
    }

    /// Device-in projection: A[m,KRES] bf16 device BO `a_bo` @ W[KRES,n] -> C[m,n] f32 (read to host).
    /// The device-in twin of `matmul_id_lazy` (k=KRES path): the input is ALREADY device-resident bf16
    /// (a resident-stream LN output), so the host pack+upload of A is SKIPPED. `id` shares the weight-BO
    /// cache with the host path, so warm passes hit. Identity modal (no silu) -- for q/k/v/out projections.
    pub fn proj_from_bf16<F: FnOnce() -> Array2<f32>>(&self, a_bo: &Bo, m: usize, make_b: F, id: &str, n: usize) -> Array2<f32> {
        self.stats.borrow_mut().calls += 1;
        let cached = self.wcache.borrow().get(id).cloned();
        let wbo = if let Some(bo) = cached {
            bo
        } else {
            let b = make_b();
            assert_eq!(b.nrows(), KRES, "proj weight nrows {} != {KRES}", b.nrows());
            assert_eq!(b.ncols(), n, "proj weight ncols {} != {n}", b.ncols());
            self.weight_bo(id, b.view())
        };
        self.dispatch_with_a(a_bo, m, &wbo, n, false)
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

    /// Device-in variant of [`Self::resident_conv_pw1_glu`]: the conv-module LN input is the ALREADY-
    /// device f32 [PAD_M,KRES] BO `a_bo` (the MHSA-residual result), so the conv front's own input never
    /// round-trips to host. Returns the host GLU output [m, KRES] (the rest of the conv module continues
    /// host-fed for this seam). `None` when the resident-ln / glu xclbins are absent.
    pub fn resident_conv_pw1_glu_dev<F: FnOnce() -> Array2<f32>>(&self, a_bo: &Bo, m: usize, gamma: &[f32], beta: &[f32], make_w1: F, id: &str) -> Option<Array2<f32>> {
        let rl = self.resident_ln()?;
        let glu = rl.glu.as_ref()?;
        self.stats.borrow_mut().calls += 1;
        let n2 = 2 * KRES; // pw1 output width 2D
        let rlc = self.ln_affine_cast_dev(a_bo, gamma, beta); // device-in LN
        let cached = self.wcache.borrow().get(id).cloned();
        let wbo = if let Some(bo) = cached {
            bo
        } else {
            let w = make_w1();
            assert_eq!(w.nrows(), KRES, "pw1 W nrows {} != {KRES}", w.nrows());
            assert_eq!(w.ncols(), n2, "pw1 W ncols {} != {n2}", w.ncols());
            self.weight_bo(id, w.view())
        };
        let st = self.stream(n2, false);
        self.kern.run_matmul8(3, &st.instr, st.n_instr, &rlc.bo_bf16, &wbo, &st.bo_c, &self.bo_tmp, &self.bo_tr).unwrap();
        glu.kern.run_matmul8(3, &glu.instr, glu.n, &st.bo_c, &glu.bo_out, &glu.dummy_c, &glu.dummy_tmp, &glu.dummy_tr).unwrap();
        self.stats.borrow_mut().dispatches += 2; // pw1 + glu
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

    /// Host-in -> DEVICE-OUT matmul: A[m,KRES] @ W[KRES,n] -> C[m,n] f32 left in a FRESH device BO (no
    /// read). The device-out twin of the k=KRES `matmul_id_lazy` path: packs+uploads A, GEMMs into a new
    /// BO, returns it -- so a projection result (e.g. MHSA linear_out) stays resident for the next seam.
    pub fn matmul_id_to_bo<F: FnOnce() -> Array2<f32>>(&self, a: &Array2<f32>, make_w: F, id: &str, n: usize) -> Rc<Bo> {
        let m = a.nrows();
        assert_eq!(a.ncols(), KRES, "matmul_id_to_bo needs K=KRES={KRES}");
        self.stats.borrow_mut().calls += 1;
        let cached = self.wcache.borrow().get(id).cloned();
        let wbo = if let Some(bo) = cached {
            bo
        } else {
            let w = make_w();
            assert_eq!(w.nrows(), KRES, "weight nrows {} != {KRES}", w.nrows());
            assert_eq!(w.ncols(), n, "weight ncols {} != {n}", w.ncols());
            self.weight_bo(id, w.view())
        };
        // pack A -> bf16 -> bo_a. ZERO-PAD rows m..PAD_M: unlike dispatch() (whose stale padding rows
        // are harmless because the result is read back at m rows), the output here stays DEVICE-resident
        // and flows into the next seam (residual -> conv front) which processes all PAD_M rows -- garbage
        // padding then corrupts valid rows sharing a partial m-tile. Zero input padding -> zero output
        // padding (0 @ W = 0), matching the host path's clean zero-padding invariant.
        let a_std = a.as_standard_layout();
        let a_s = a_std.as_slice().unwrap();
        let mut a_bits = vec![0u16; PAD_M * KRES]; // rows m..PAD_M stay zero
        npu_xrt::pack_f32_to_bf16(&a_s[..m * KRES], &mut a_bits[..m * KRES]);
        self.bo_a.write_bytes(u16_bytes(&a_bits)).unwrap();
        self.bo_a.sync_to_device().unwrap();
        // GEMM into a FRESH device f32 BO (identity modal, NO read).
        let out = self.dev.alloc_bo(&self.kern, PAD_M * n * 4, FLAG_HOST_ONLY, self.kern.group_id(5).unwrap()).unwrap();
        let st = self.stream(n, false);
        self.kern.run_matmul8(3, &st.instr, st.n_instr, &self.bo_a, &wbo, &out, &self.bo_tmp, &self.bo_tr).unwrap();
        self.stats.borrow_mut().dispatches += 1;
        Rc::new(out)
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

    /// Host-fed on-NPU SiLU (conv step 4): the post-dwconv activation as a SEPARATE brick (silu_row).
    /// `x_ct` = [C=1024, T] channel-major f32 (T <= 400). Packs f32 -> device (zero-padding the time
    /// tail; silu(0)=0 so pad rows are 0 and sliced off), runs the 2-buffer brick, returns [C, T] f32.
    /// This replaces the host `silu_inplace` on the dwconv output -- advancing the single-hardware graph
    /// WITHOUT fusing silu into dwconv (which miscompiles alternate channels; see the KB log). None if
    /// the silu xclbin is absent or T exceeds the baked DW_T (caller keeps the host silu).
    pub fn npu_silu(&self, x_ct: &Array2<f32>) -> Option<Array2<f32>> {
        let rl = self.resident_ln()?;
        let s = rl.silu.as_ref()?;
        let (c, t) = x_ct.dim();
        if c != DW_C || t > DW_T {
            return None; // shape outside the baked brick -> host fallback
        }
        self.stats.borrow_mut().calls += 1;
        let x_std = x_ct.as_standard_layout();
        let xs = x_std.as_slice().unwrap();
        let mut in_f = vec![0f32; DW_C * DW_T];
        for ch in 0..c {
            in_f[ch * DW_T..ch * DW_T + t].copy_from_slice(&xs[ch * t..ch * t + t]);
        }
        s.bo_in.write_bytes(f32_bytes(&in_f)).unwrap();
        s.bo_in.sync_to_device().unwrap();
        // 2-buffer ABI: in(g3) -> out(g4); tmp/ctrl/trace dummies (g5/g6/g7).
        s.kern.run_matmul8(3, &s.instr, s.n, &s.bo_in, &s.bo_out, &s.dummy_tmp, &s.dummy_ctrl, &s.dummy_tr).unwrap();
        self.stats.borrow_mut().dispatches += 1;
        s.bo_out.sync_from_device().unwrap();
        let mut ob = vec![0u8; DW_C * DW_T * 4];
        s.bo_out.read_bytes(&mut ob).unwrap();
        let mut out = Array2::<f32>::zeros((c, t));
        for ch in 0..c {
            for ti in 0..t {
                let off = (ch * DW_T + ti) * 4;
                out[[ch, ti]] = f32::from_le_bytes([ob[off], ob[off + 1], ob[off + 2], ob[off + 3]]);
            }
        }
        Some(out)
    }

    /// FUSED on-NPU dwconv->SiLU (conv steps 3+4 in ONE xclbin). Replaces the two
    /// separate npu_dwconv1d + npu_silu dispatches: one hw-context, the post-dwconv SiLU runs
    /// device-to-device (dwconv core -> on-chip f32 fifo -> silu core), so the on-NPU SiLU costs NO
    /// extra hw-context switch and no host round-trip (the ~1 ms/block the separate silu xclbin added).
    /// `x_ct` = [C=1024, T] channel-major f32 (T <= 400, the transposed GLU output), taps [C,9], bias
    /// [C]. Returns silu(dwconv(x)) as [C, T] f32. None if the fused xclbin is absent or T > DW_T
    /// (caller falls back to the separate dwconv+silu path, or host).
    pub fn npu_dwconv_silu(&self, x_ct: &Array2<f32>, taps: &Array2<f32>, bias: &Array1<f32>) -> Option<Array2<f32>> {
        let rl = self.resident_ln()?;
        let ds = rl.dwconv_silu.as_ref()?;
        let (c, t) = x_ct.dim();
        if c != DW_C || t > DW_T {
            return None; // shape outside the baked brick -> fallback
        }
        self.stats.borrow_mut().calls += 1;
        // pack input [C,t] f32 -> bf16 [C,DW_T] channel-major, zero-padding the time tail (== 'same' end pad).
        let x_std = x_ct.as_standard_layout();
        let xs = x_std.as_slice().unwrap();
        let mut in_bits = vec![0u16; DW_C * DW_T];
        for ch in 0..c {
            npu_xrt::pack_f32_to_bf16(&xs[ch * t..ch * t + t], &mut in_bits[ch * DW_T..ch * DW_T + t]);
        }
        ds.bo_in.write_bytes(u16_bytes(&in_bits)).unwrap();
        ds.bo_in.sync_to_device().unwrap();
        // pack weights [C,9] + bias[C] -> [C,16] bf16 (taps [0..8], BN-folded bias [9]).
        let taps_std = taps.as_standard_layout();
        let tp = taps_std.as_slice().unwrap();
        let mut w_bits = vec![0u16; DW_C * DW_KW];
        for ch in 0..c {
            let mut row = [0f32; DW_KW];
            row[..9].copy_from_slice(&tp[ch * 9..ch * 9 + 9]);
            row[9] = bias[ch];
            npu_xrt::pack_f32_to_bf16(&row, &mut w_bits[ch * DW_KW..ch * DW_KW + DW_KW]);
        }
        ds.bo_w.write_bytes(u16_bytes(&w_bits)).unwrap();
        ds.bo_w.sync_to_device().unwrap();
        // 3-buffer ABI (== dwconv): in(g3), w(g4), out(g5) f32; tmp/trace dummies (g6/g7).
        ds.kern.run_matmul8(3, &ds.instr, ds.n, &ds.bo_in, &ds.bo_w, &ds.bo_out, &ds.dummy_tmp, &ds.dummy_tr).unwrap();
        self.stats.borrow_mut().dispatches += 1;
        ds.bo_out.sync_from_device().unwrap();
        let mut ob = vec![0u8; DW_C * DW_T * 4];
        ds.bo_out.read_bytes(&mut ob).unwrap();
        let mut out = Array2::<f32>::zeros((c, t));
        for ch in 0..c {
            for ti in 0..t {
                let off = (ch * DW_T + ti) * 4;
                out[[ch, ti]] = f32::from_le_bytes([ob[off], ob[off + 1], ob[off + 2], ob[off + 3]]);
            }
        }
        Some(out)
    }

    /// TIME-MAJOR fused on-NPU dwconv->SiLU (conv step 3b -- the transpose-DISSOLVING path). Unlike
    /// `npu_dwconv_silu` ([C,T], bracketed by two host transposes), this takes the GLU output `x_td`
    /// [T,D] DIRECTLY and returns silu(dwconv(x)) as [T,D] DIRECTLY -- so `conv_module` feeds pw2 the
    /// result with NO transpose on either side. The FIR vectorizes along D with the k=9 halo along time
    /// (consecutive row loads, no shuffle / cross-column DMA), so it dodges the n-D-DMA co-residency
    /// hang. Precision recipe IDENTICAL to the channel-major fused brick (bf16 in, f32 on-chip mid to
    /// silu, bf16-tanh silu). `taps` [D,9], `bias` [D]. None if the time-major xclbin is absent or
    /// t > DW_T (caller falls back to the channel-major path, then host).
    pub fn npu_dwconv_silu_tmajor(&self, x_td: &Array2<f32>, taps: &Array2<f32>, bias: &Array1<f32>) -> Option<Array2<f32>> {
        let rl = self.resident_ln()?;
        let ds = rl.dwconv_silu_t.as_ref()?;
        let (t, d) = x_td.dim();
        if d != DW_C || t > DW_T {
            return None; // shape outside the baked brick -> fallback
        }
        self.stats.borrow_mut().calls += 1;
        // pad input [t,D] -> [T+2P, D] f32: real rows land at [P, P+t) (4 zero rows top; zeros below,
        // == 'same' end pad past the sequence), then pack to bf16 in one shot.
        let x_std = x_td.as_standard_layout();
        let xs = x_std.as_slice().unwrap();
        let mut in_f = vec![0f32; DW_TPAD * DW_C];
        for r in 0..t {
            in_f[(DW_P + r) * DW_C..(DW_P + r) * DW_C + DW_C].copy_from_slice(&xs[r * d..r * d + d]);
        }
        let mut in_bits = vec![0u16; DW_TPAD * DW_C];
        npu_xrt::pack_f32_to_bf16(&in_f, &mut in_bits);
        ds.bo_in.write_bytes(u16_bytes(&in_bits)).unwrap();
        ds.bo_in.sync_to_device().unwrap();
        // repack weights TAP-MAJOR [K+1, D]: row p (0..8) = tap p across all D; row 9 = BN bias.
        let taps_std = taps.as_standard_layout();
        let tp = taps_std.as_slice().unwrap(); // [D, 9] row-major
        let mut w_f = vec![0f32; (DW_K + 1) * DW_C];
        for ch in 0..d {
            for p in 0..DW_K {
                w_f[p * DW_C + ch] = tp[ch * DW_K + p];
            }
            w_f[DW_K * DW_C + ch] = bias[ch];
        }
        let mut w_bits = vec![0u16; (DW_K + 1) * DW_C];
        npu_xrt::pack_f32_to_bf16(&w_f, &mut w_bits);
        ds.bo_w.write_bytes(u16_bytes(&w_bits)).unwrap();
        ds.bo_w.sync_to_device().unwrap();
        // 3-buffer ABI (== dwconv): in(g3), w(g4), out(g5) f32; tmp/trace dummies (g6/g7).
        ds.kern.run_matmul8(3, &ds.instr, ds.n, &ds.bo_in, &ds.bo_w, &ds.bo_out, &ds.dummy_tmp, &ds.dummy_tr).unwrap();
        self.stats.borrow_mut().dispatches += 1;
        ds.bo_out.sync_from_device().unwrap();
        let mut ob = vec![0u8; DW_T * DW_C * 4];
        ds.bo_out.read_bytes(&mut ob).unwrap();
        // read [T,D] f32, slice to the t real rows.
        let mut out = Array2::<f32>::zeros((t, d));
        for r in 0..t {
            for ch in 0..d {
                let off = (r * DW_C + ch) * 4;
                out[[r, ch]] = f32::from_le_bytes([ob[off], ob[off + 1], ob[off + 2], ob[off + 3]]);
            }
        }
        Some(out)
    }

    /// Full FFN device-side (LN -> fc1 -> SiLU -> fc2), the fc1->fc2 frontier step. Everything on-NPU,
    /// the activation stream never touching host across the whole FFN:
    ///   ctxLN -> affine_cast -> modal fc1 (on-chip silu, [t,DFF]) -> cast@DFF (bf16) -> K=DFF fc2
    ///   (identity, on-chip K-reduce, [t,KRES]) -> read [t,KRES] f32.
    /// No host K-split / accumulate. `make_w1` = [KRES,DFF] fc1 weight; `make_w2` = [DFF,KRES] fc2.
    /// True when the one-dispatch K=DFF fc2 collapse is enabled (opt-in `PARAKEET_FC2_K4096`).
    fn fc2_k4096_on(&self) -> bool {
        std::env::var("PARAKEET_FC2_K4096").map(|v| v != "0").unwrap_or(false)
    }

    /// Shared one-dispatch K=DFF fc2: cast the fc1 output (`fc1_out` f32 [PAD_M,DFF]) to bf16 row-major,
    /// then ONE K=DFF modal GEMM (internal L1 K-accum over DFF) with the full fc2 weight -> f32
    /// [PAD_M,KRES] device BO. Counts 2 dispatches (cast + modal); the caller counts fc1. Full fc2
    /// weight cached under "{id2}.full". Collapses the deint + 4x K=1024 GEMM + 4x acc_add.
    fn fc2_k4096_dev<F2: FnOnce() -> Array2<f32>>(&self, k4: &Fc2K4096, fc1_out: &Bo, make_w2: F2, id2: &str) -> Rc<Bo> {
        k4.cast_kern.run_matmul8(3, &k4.cast_instr, k4.cast_n, fc1_out, &k4.cast_out, &k4.cast_dc, &k4.cast_dt, &k4.cast_dr).unwrap();
        let wid = format!("{id2}.full");
        let cached = self.wcache.borrow().get(&wid).cloned();
        let w2f = if let Some(bo) = cached {
            bo
        } else {
            let w = make_w2();
            assert_eq!(w.dim(), (DFF, KRES), "fc2 W2 dim");
            self.weight_bo(&wid, w.view())
        };
        k4.mm_kern.run_matmul8(3, &k4.mm_instr, k4.mm_n, &k4.cast_out, &w2f, &k4.mm_c, &self.bo_tmp, &self.bo_tr).unwrap();
        self.stats.borrow_mut().dispatches += 2; // cast + K=DFF modal
        k4.mm_c.clone()
    }

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
        // ONE-DISPATCH K=DFF fc2 (opt-in): cast@DFF -> K=DFF modal -> readback to host [m,KRES].
        if self.fc2_k4096_on() {
            if let Some(k4) = rl.fc2_k4096.as_ref() {
                self.stats.borrow_mut().dispatches += 1; // fc1
                let bo = self.fc2_k4096_dev(k4, &st1.bo_c, make_w2, id2);
                bo.sync_from_device().unwrap();
                let mut cb = vec![0u8; m * KRES * 4];
                bo.read_bytes(&mut cb).unwrap();
                let mut out = Array2::<f32>::zeros((m, KRES));
                for r in 0..m {
                    for c in 0..KRES {
                        let off = (r * KRES + c) * 4;
                        out[[r, c]] = f32::from_le_bytes([cb[off], cb[off + 1], cb[off + 2], cb[off + 3]]);
                    }
                }
                return out;
            }
        }
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

    /// Shared fc1 -> deint -> fc2 ON-DEVICE-accumulate core for the resident FFN device path. `rl`
    /// must already hold `bo_bf16 = affine_LN(input)` (from `ln_affine_cast` host-in or
    /// `ln_affine_cast_dev` device-in) AND have the acc_add brick loaded. Returns the device BO
    /// [PAD_M,KRES] f32 = sum of the DFF/KRES fc2 partials (acc=0, +partial0, +partial1, ...).
    fn ffn_dev_accum<F1: FnOnce() -> Array2<f32>, F2: FnOnce() -> Array2<f32>>(
        &self, rl: &Rc<ResidentLn>, make_w1: F1, id1: &str, make_w2: F2, id2: &str,
    ) -> Rc<Bo> {
        let aa = rl.acc_add.as_ref().expect("ffn_dev_accum without acc_add");
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
        // ONE-DISPATCH fc2 (K=DFF): cast fc1's f32 [PAD_M,DFF] -> bf16 row-major, then a SINGLE K=DFF
        // modal GEMM that accumulates all DFF K internally in L1 -> f32 [PAD_M,KRES] device. Collapses
        // deint + 4x K=1024 GEMM + 4x acc_add (8 dispatches) into cast + 1 modal (2). NOT bit-identical
        // to the 4-way split (different L1 accum + bfp16) -> validated by the sound rel-L2 gate.
        if self.fc2_k4096_on() {
            if let Some(k4) = rl.fc2_k4096.as_ref() {
                self.stats.borrow_mut().dispatches += 1; // fc1
                return self.fc2_k4096_dev(k4, &st1.bo_c, make_w2, id2);
            }
        }
        // deinterleave+cast: st1.bo_c (f32 [PAD_M,DFF]) -> rl.bo_deint (bf16 chunk-major), device-side.
        rl.deint_kern.run_matmul8(3, &rl.deint_instr, rl.deint_n, &st1.bo_c, &rl.bo_deint, &rl.deint_c, &rl.deint_tmp, &rl.deint_tr).unwrap();
        self.stats.borrow_mut().dispatches += 2; // fc1 + deint
        // fc2 K-split with ON-DEVICE accumulate: each partial modal GEMM -> st.bo_c (device); acc_add
        // sums it into the acc0/acc1 ping-pong (seed acc=0 for partial0). Result stays device-resident.
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
        let st = self.stream(KRES, false);
        let mut cur = aa.acc0.clone();
        let mut nxt = aa.acc1.clone();
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
            // modal identity GEMM: partial c -> st.bo_c (device, NO sync_from/read).
            self.kern.run_matmul8(3, &st.instr, st.n_instr, &chunk, &w2c, &st.bo_c, &self.bo_tmp, &self.bo_tr).unwrap();
            // accumulate on-chip: nxt = (c==0 ? zero : cur) + st.bo_c, then ping-pong.
            let a_in: &Bo = if c == 0 { &aa.zero } else { &cur };
            aa.kern.run_matmul8(3, &aa.instr, aa.n, a_in, &st.bo_c, &nxt, &aa.dummy_tmp, &aa.dummy_tr).unwrap();
            self.stats.borrow_mut().dispatches += 2; // partial GEMM + acc_add
            std::mem::swap(&mut cur, &mut nxt);
        }
        cur // device BO [PAD_M, KRES] f32 holding sum of all `parts` partials
    }

    /// Same as [`Self::resident_ffn`] but the fc2 K-split partials are accumulated ON-DEVICE (the
    /// acc_add brick) so the FFN output lands in ONE device BO `[PAD_M, KRES]` f32 -- no host `acc`,
    /// no `sync_from`/`read`. Returns the device accumulator (the fused seam's resident-stream handle).
    /// `None` when the acc_add xclbin is absent, so callers fall back to the host-accum resident_ffn.
    /// Bit-identical to resident_ffn: SAME partials (same modal GEMM into st.bo_c) summed in the SAME
    /// sequential f32 order. The returned Rc is AccAdd scratch, overwritten by the next call.
    pub fn resident_ffn_dev<F1: FnOnce() -> Array2<f32>, F2: FnOnce() -> Array2<f32>>(
        &self, x: &Array2<f32>, gamma: &[f32], beta: &[f32],
        make_w1: F1, id1: &str, make_w2: F2, id2: &str,
    ) -> Option<Rc<Bo>> {
        let rl = self.resident_ln()?;
        rl.acc_add.as_ref()?;
        self.stats.borrow_mut().calls += 1;
        let rl2 = self.ln_affine_cast(x, gamma, beta); // host-in LN: bo_bf16 = affine_LN(x)
        Some(self.ffn_dev_accum(&rl2, make_w1, id1, make_w2, id2))
    }

    /// Device-in FFN for the fused seam: like [`Self::resident_ffn_dev`] but the LN input is the
    /// ALREADY-device-resident f32 [PAD_M,KRES] BO `a_bo` (previous op's output) -- the LN uses
    /// `ln_affine_cast_dev`, so the FFN's own input never round-trips to host either. Returns the
    /// device fc2 accumulator BO. `None` when the resident/acc_add xclbins are absent.
    pub fn resident_ffn_dev_bo<F1: FnOnce() -> Array2<f32>, F2: FnOnce() -> Array2<f32>>(
        &self, a_bo: &Bo, gamma: &[f32], beta: &[f32],
        make_w1: F1, id1: &str, make_w2: F2, id2: &str,
    ) -> Option<Rc<Bo>> {
        let rl = self.resident_ln()?;
        rl.acc_add.as_ref()?;
        self.stats.borrow_mut().calls += 1;
        let rl2 = self.ln_affine_cast_dev(a_bo, gamma, beta); // device-in LN: bo_bf16 = affine_LN(a_bo)
        Some(self.ffn_dev_accum(&rl2, make_w1, id1, make_w2, id2))
    }

    /// Host-readback wrapper over [`Self::resident_ffn_dev`] for the FFN-boundary gate
    /// (`PARAKEET_FFN_DEVACC`): device-accumulate the FFN, then `sync_from`+read the first `m` rows.
    /// So ONLY the accumulation moved on-device vs resident_ffn; the block dataflow is unchanged.
    pub fn resident_ffn_devacc_readback<F1: FnOnce() -> Array2<f32>, F2: FnOnce() -> Array2<f32>>(
        &self, x: &Array2<f32>, gamma: &[f32], beta: &[f32],
        make_w1: F1, id1: &str, make_w2: F2, id2: &str,
    ) -> Option<Array2<f32>> {
        let m = x.nrows();
        let acc_bo = self.resident_ffn_dev(x, gamma, beta, make_w1, id1, make_w2, id2)?;
        acc_bo.sync_from_device().unwrap();
        let mut cb = vec![0u8; m * KRES * 4];
        acc_bo.read_bytes(&mut cb).unwrap();
        let mut out = Array2::<f32>::zeros((m, KRES));
        for r in 0..m {
            for c in 0..KRES {
                let off = (r * KRES + c) * 4;
                out[[r, c]] = f32::from_le_bytes([cb[off], cb[off + 1], cb[off + 2], cb[off + 3]]);
            }
        }
        Some(out)
    }

    /// Device-parity self-test for Task 1 (on-device fc2 accumulation). Runs [`Self::resident_ffn`]
    /// (host-accum reference) and [`Self::resident_ffn_dev`] (device-accum) on the SAME synthetic
    /// input + weights, returns both `[t, KRES]` host arrays. The accumulation is the ONLY difference,
    /// so rel-L2 must be ~0. `None` when the modal/resident/acc_add xclbins are absent. No encoder
    /// weights needed -- synthetic weights fully exercise the K-split accumulate path.
    pub fn ffn_devacc_selftest(&self, t: usize, seed: u64) -> Option<(Array2<f32>, Array2<f32>)> {
        if !self.modal || self.resident_ln()?.acc_add.is_none() {
            return None;
        }
        // Deterministic splitmix64 fill in [-scale, scale].
        let fill = |rows: usize, cols: usize, sd: u64, scale: f32| -> Array2<f32> {
            let mut s = sd.wrapping_add(0x9E37_79B9_7F4A_7C15);
            Array2::from_shape_fn((rows, cols), |_| {
                s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = s;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^= z >> 31;
                let u = (z >> 40) as f32 / (1u32 << 24) as f32;
                (u * 2.0 - 1.0) * scale
            })
        };
        let x = fill(t, KRES, seed, 1.0);
        let gamma: Vec<f32> = fill(1, KRES, seed ^ 0xA1, 0.1).iter().copied().collect();
        let beta: Vec<f32> = fill(1, KRES, seed ^ 0xB2, 0.1).iter().copied().collect();
        let w1 = fill(KRES, DFF, seed ^ 0xC3, 0.05);
        let w2 = fill(DFF, KRES, seed ^ 0xD4, 0.05);
        // Same ids -> the host path caches w1/w2c on first touch; the dev path hits the cache, so both
        // paths use bit-identical partials (only host-sum vs device-sum differs).
        let (w1a, w2a) = (w1.clone(), w2.clone());
        let host = self.resident_ffn(&x, &gamma, &beta,
            move || w1a, "selftest.ffn.l1", move || w2a, "selftest.ffn.l2");
        let dev = self.resident_ffn_devacc_readback(&x, &gamma, &beta,
            move || w1, "selftest.ffn.l1", move || w2, "selftest.ffn.l2")?;
        Some((host, dev))
    }

    /// Task-5 debug parity: `matmul_id` (host-read reference) vs `matmul_id_to_bo` (device-out) on the
    /// SAME synthetic ctx + weight (shared id -> same weight BO). Must be bit-identical (same GEMM).
    pub fn linout_selftest(&self, t: usize, seed: u64) -> Option<(Array2<f32>, Array2<f32>)> {
        if !self.modal {
            return None;
        }
        let fill = |rows: usize, cols: usize, sd: u64, sc: f32| -> Array2<f32> {
            let mut s = sd.wrapping_add(0x9E37_79B9_7F4A_7C15);
            Array2::from_shape_fn((rows, cols), |_| {
                s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = s; z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB); z ^= z >> 31;
                ((z >> 40) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0) * sc
            })
        };
        let ctx = fill(t, KRES, seed, 1.0);
        let w = fill(KRES, KRES, seed ^ 0x7E, 0.05);
        let host = self.matmul_id(&ctx, &w, "selftest.linout");
        let wc = w.clone();
        let dev_bo = self.matmul_id_to_bo(&ctx, move || wc, "selftest.linout", KRES);
        dev_bo.sync_from_device().unwrap();
        let mut cb = vec![0u8; t * KRES * 4];
        dev_bo.read_bytes(&mut cb).unwrap();
        let mut dev = Array2::<f32>::zeros((t, KRES));
        for r in 0..t { for c in 0..KRES {
            let off = (r * KRES + c) * 4;
            dev[[r, c]] = f32::from_le_bytes([cb[off], cb[off + 1], cb[off + 2], cb[off + 3]]);
        }}
        Some((host, dev))
    }

    /// Task-5 debug parity: `resident_conv_pw1_glu` (host-in) vs `resident_conv_pw1_glu_dev` (device-in,
    /// input uploaded) on the SAME synthetic x + weights (shared id). Must be bit-identical.
    pub fn conv_front_selftest(&self, t: usize, seed: u64) -> Option<(Array2<f32>, Array2<f32>)> {
        let rl = self.resident_ln()?;
        rl.glu.as_ref()?;
        let fill = |rows: usize, cols: usize, sd: u64, sc: f32| -> Array2<f32> {
            let mut s = sd.wrapping_add(0x9E37_79B9_7F4A_7C15);
            Array2::from_shape_fn((rows, cols), |_| {
                s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = s; z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB); z ^= z >> 31;
                ((z >> 40) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0) * sc
            })
        };
        let x = fill(t, KRES, seed, 1.0);
        let gv: Vec<f32> = fill(1, KRES, seed ^ 0x1A, 1.0).iter().copied().collect();
        let bv: Vec<f32> = fill(1, KRES, seed ^ 0x2B, 0.1).iter().copied().collect();
        let pw1 = fill(KRES, 2 * KRES, seed ^ 0x3C, 0.05);
        let pw1a = pw1.clone();
        let host = self.resident_conv_pw1_glu(&x, &gv, &bv, move || pw1a, "selftest.convpw1")?;
        let a_bo = self.upload_stream(&x);
        let dev = self.resident_conv_pw1_glu_dev(&a_bo, t, &gv, &bv, move || pw1, "selftest.convpw1")?;
        Some((host, dev))
    }

    /// On-chip scaled residual add: out = a + scale*b, f32 [PAD_M,KRES], device-resident. Selects
    /// the baked-scale xclbin (only s050 = 0.5 built so far). `a_bo`/`b_bo` are device f32 [PAD_M,KRES]
    /// BOs; returns the device result (ResidualAdd scratch, overwritten by the next call). `None` when
    /// the resident-ln xclbins are absent; PANICS if the requested `scale` has no built xclbin.
    pub fn residual_add_dev(&self, a_bo: &Bo, b_bo: &Bo, scale: f32, _m: usize) -> Option<Rc<Bo>> {
        let rl = self.resident_ln()?;
        let ra = if (scale - 0.5).abs() < 1e-6 {
            rl.resadd_s050.as_ref()?
        } else if (scale - 1.0).abs() < 1e-6 {
            rl.resadd_s100.as_ref()?
        } else {
            panic!("residual_add_dev: scale {scale} has no built xclbin (only s050=0.5, s100=1.0); build final_resadd_{PAD_M}x{KRES}_s<stag>");
        };
        debug_assert!((ra.scale - scale).abs() < 1e-6);
        ra.kern.run_matmul8(3, &ra.instr, ra.n, a_bo, b_bo, &ra.bo_out, &ra.dummy_tmp, &ra.dummy_tr).unwrap();
        self.stats.borrow_mut().dispatches += 1;
        Some(ra.bo_out.clone())
    }

    /// Device-parity self-test for Task 2 (on-chip residual add). Uploads synthetic a,b to device BOs,
    /// runs [`Self::residual_add_dev`], returns (host `a + scale*b`, device out) as `[t, KRES]`. f32
    /// mul+add is near-exact, so rel-L2 must be ~0. `None` when the resadd xclbin is absent.
    pub fn residual_add_selftest(&self, t: usize, seed: u64, scale: f32) -> Option<(Array2<f32>, Array2<f32>)> {
        let rl = self.resident_ln()?;
        let ra = rl.resadd_s050.as_ref()?;
        let fill = |rows: usize, cols: usize, sd: u64, sc: f32| -> Array2<f32> {
            let mut s = sd.wrapping_add(0x9E37_79B9_7F4A_7C15);
            Array2::from_shape_fn((rows, cols), |_| {
                s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = s;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^= z >> 31;
                let u = (z >> 40) as f32 / (1u32 << 24) as f32;
                (u * 2.0 - 1.0) * sc
            })
        };
        let a = fill(t, KRES, seed, 1.0);
        let b = fill(t, KRES, seed ^ 0x51, 1.0);
        // Upload a,b into device BOs [PAD_M,KRES] f32 (first t rows real; the rest stale -> ignored).
        let mkbo = |arr: &Array2<f32>, gid: i32| -> Bo {
            let bo = self.dev.alloc_bo(&ra.kern, PAD_M * KRES * 4, FLAG_HOST_ONLY, ra.kern.group_id(gid).unwrap()).unwrap();
            let mut buf = vec![0f32; PAD_M * KRES];
            let s = arr.as_standard_layout();
            buf[..t * KRES].copy_from_slice(&s.as_slice().unwrap()[..t * KRES]);
            bo.write_bytes(f32_bytes(&buf)).unwrap();
            bo.sync_to_device().unwrap();
            bo
        };
        let a_bo = mkbo(&a, 3);
        let b_bo = mkbo(&b, 4);
        let out_bo = self.residual_add_dev(&a_bo, &b_bo, scale, t)?;
        out_bo.sync_from_device().unwrap();
        let mut cb = vec![0u8; t * KRES * 4];
        out_bo.read_bytes(&mut cb).unwrap();
        let mut dev = Array2::<f32>::zeros((t, KRES));
        for r in 0..t {
            for c in 0..KRES {
                let off = (r * KRES + c) * 4;
                dev[[r, c]] = f32::from_le_bytes([cb[off], cb[off + 1], cb[off + 2], cb[off + 3]]);
            }
        }
        // Host ref in the kernel's op order (scale*b, then a + that).
        let sb = b.mapv(|x| scale * x);
        let host = &a + &sb;
        Some((host, dev))
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

/// Round an f32 to the nearest bf16 value, returned as f32 (round-to-nearest-even, top 16 bits).
/// Used to split BD into hi+lo bf16 halves (hi = bf16_round_f32(x); lo = x - hi). Mirrors the
/// device pack_f32_to_bf16 rounding so the split reconstruction matches on-device arithmetic.
fn bf16_round_f32(x: f32) -> f32 {
    if !x.is_finite() { return x; }
    let bits = x.to_bits();
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
    f32::from_bits(rounded & 0xffff_0000)
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
