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
use npu_asr::kernel_registry;
use npu_xrt::{Bo, Device, Kernel, FLAG_CACHEABLE, FLAG_HOST_ONLY};

const PAD_M: usize = 512;
const KRES: usize = 1024; // resident kernel contraction dim
const WA_SUBDIR: &str =
    "mlir-aie/programming_examples/basic/matrix_multiplication/whole_array/build";

/// Activation epilogue baked into the modal resident's per-N instruction stream (RTP-selected at
/// generate time -- `whole_array_modal_iron.py` sets `mode_val` and `set_modes` bakes it into that
/// stream's RTP). The modal FFN epilogue kernel compiles all three branches; the host selects the
/// mode by dispatching the matching insts stream. `Silu` -> `modalsilu` (Parakeet fc1 / ff.l1),
/// `Identity` -> `modalid` (every plain GEMM, conv pw1, K-collapse fc2), `Gelu` -> `modalgelu`
/// (the K=768 GELU FFN rail: BERT/Whisper/ESM). On the PLAIN (non-modal) resident there is no
/// on-chip epilogue, so the activation is a no-op there (the host applies it) -- `stream()`
/// normalizes to `Identity` so the stream cache stays 1:1 with the single plain insts file.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Act {
    Identity,
    Silu,
    Gelu,
    /// conv-module pw1 with GLU folded in: out = a * sigmoid(g) over the tile's [value|gate]
    /// halves. Only valid against a W1 whose columns were permuted at weight load.
    Glu,
}

impl Act {
    /// The mode tag baked into the modal insts filename (`insts_..._8c_{tag}.txt`).
    fn mode_tag(self) -> &'static str {
        match self {
            Act::Identity => "modalid",
            Act::Silu => "modalsilu",
            Act::Gelu => "modalgelu",
            Act::Glu => "modalglu",
        }
    }
}

/// Per-N instruction stream + its output BO (on the resident kernel).
struct NStream {
    instr: Bo,
    n_instr: usize,
    bo_c: Bo, // [PAD_M, n] f32
}

// Resident relpos-MHA block (STEP=8, STEP-C runtime-t_active). An xclbin sized for a bucket's
// BUILT_T serves ANY clip T <= it: the softmax reads t_active from RTP words baked into the
// instruction stream, so per clip we PATCH those words of a template insts (zero build) and pad
// k/p/V to that bucket's BUILT_T. Loaded once per bucket, resident. See RELPOS_BUCKETS below --
// it carries the per-bucket (BUILT_T, KB, subdir); the old single-block RELPOS_BUILT_T/RELPOS_KB/
// RELPOS_TACTIVE_WORD constants went away with the per-head `relpos_mha` the buckets replaced.
const RELPOS_TQ: usize = 8;
const RELPOS_DK: usize = 128;   // Parakeet head_dim (kernel bakes DK=128)

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

// --- Phase-2 spatial-parallel relpos (opt-in PARAKEET_RESIDENT_MHA) -----------------------------
// Independent of the CONVEYOR above: that path loads via `conveyor_block` behind
// PARAKEET_CONVEYOR_MHA, this one via `relpos_block` behind PARAKEET_RESIDENT_MHA. Both are kept.
// This path carries the positional operand as SPLIT bf16 (p_hi + p_lo), preserving the WER-critical
// precision the shipped relpos block uses; the conveyor's BD-in-belt defaults to plain bf16.
//
// T-bucketing: dispatch the SMALLEST bucket whose BUILT_T >= this clip's T, so short clips run a
// smaller padded dataflow.
const RELPOS_BUCKETS: &[(usize, usize, &str)] = &[
    (100, 25, "bucket_100"), // short clips (T<=100): ~29% less relpos padding compute (measured)
    (152, 38, "bucket_152"), // mid clips (100<T<=152): ~12% less padding vs the 172 ceiling
    (172, 43, "single"),     // ceiling bucket; serves any T in (152, 172]
];
// The H=8 attention heads run on H PARALLEL cores (one head/core), ONE dispatch/block. The BOs
// concatenate all H heads; each head's Worker reads/writes its own slice via a per-head shim OFFSET
// tap. The insts template holds H t_active words (one per head's RTP write), all == the bucket's BUILT_T
// at build; per clip we patch EVERY word == BUILT_T to t.
const RELPOS_HEADS: usize = 8; // = Parakeet n_heads; must match the xclbin's --heads build

/// The single resident relpos block (built at its bucket's BUILT_T -- see RELPOS_BUCKETS). BOs are sized for BUILT_T; per
/// dispatch we patch the instr template's t_active word and pad data to BUILT_T. Dispatched per
/// head via run_dwconv6(3, instr, n, quv, kpv, ctx).
struct RelposK {
    kern: Rc<Kernel>,
    instr_template: Vec<u32>, // insts as u32 words; every word == BUILT_T is a per-head t_active (patched)
    n_instr: usize,
    bo_instr: Bo,
    bo_quv: Bo,
    bo_kpv: Bo,
    bo_ctx: Bo,
    built_t: usize,  // this bucket's baked BUILT_T (t_active words in the template == this; patched per clip)
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

#[derive(Default)]
pub struct NpuStats {
    pub pack_a_s: f64,
    pub dispatch_s: f64,
    pub read_s: f64,
    pub weight_load_s: f64,
    pub accum_s: f64,
    pub calls: usize,
    pub dispatches: usize,
    /// Per-stage LEAF breakdown of `resident_ffn`, which is ~90% of encoder NPU time and was only
    /// ever visible as one aggregate. Disjoint in time, but READ THE UNITS: these are WALL-CLOCK
    /// spans, not subsets of `dispatch_s`. `ffn_fc1_s` and `ffn_deint_s` wrap a single
    /// `run_matmul8` and so are dispatch to within a few instructions, but `ffn_ln_s` spans
    /// `ln_affine_cast` and `ffn_fc2_s` spans the whole fc2 K-split loop -- both include host work
    /// (weight-cache lookups, sub-buffer setup, the f32 `acc +=`). Measured 2026-08-03: the four
    /// sum to 8.20 s against a `dispatch_s` of 8.07 s, and that 0.13 s excess IS the host work
    /// inside them. Do not treat the leaves as a partition of `dispatch_s`.
    ///
    /// Plain `Instant` rather than the `DispatchTimer` RAII on purpose: the guard borrows
    /// `self.stats` at drop, and these regions straddle other `self.stats.borrow_mut()` calls,
    /// which would panic with a double borrow.
    pub ffn_ln_s: f64,
    pub ffn_fc1_s: f64,
    pub ffn_deint_s: f64,
    pub ffn_fc2_s: f64,
    /// HOST, not device: weight cache-miss packing, and the f32 readback + decode loop. Both sit
    /// inside the ff_resident phase scope and were charged to `Bucket::Npu`.
    pub ffn_weight_prep_s: f64,
    pub ffn_readback_s: f64,
    /// The `dispatch_with_a` return path, split. Every GEMM that hands its result back as an
    /// `Array2` pays these three after the dispatch, and none of them were timed: `ffn_readback_s`
    /// above sits on the fc2_k4096 branch, which the default path does not take, so it reads 0.00
    /// and reads as "readback is free". These are HOST time inside the `ff_resident` phase scope,
    /// charged to `Bucket::Npu` like the other two host leaves.
    ///
    /// `rb_decode_elems` is carried so the decode cost can be quoted per element -- the loop is
    /// scalar `f32::from_le_bytes` over m*n, and at fc2's [512,1024] that is 524288 elements per
    /// partial, four partials per FFN.
    pub rb_sync_s: f64,
    pub rb_read_s: f64,
    pub rb_decode_s: f64,
    pub rb_decode_elems: usize,
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
    // Resident was built with k_loop_rtp, so its cores read the k-loop bound from rtp[1] and EVERY
    // stream dispatched on it must carry that write. Derived from the loaded xclbin's own name
    // rather than an env knob: the resident is the authority on what its streams must look like,
    // and a separate knob could be set to disagree with it. (PARAKEET_MODAL_EPI_SUFFIX cannot serve
    // here -- it is also appended to fc1's panel stem, which knocks fc1 onto its fallback path.)
    krtp: bool,
    // Resident's epilogue was compiled with the GLU branch (`rtp[0]==3`). Derived from the xclbin
    // name for the same reason as `krtp`. This one has teeth: a modalglu STREAM dispatched on a
    // resident built before that branch existed takes the `else` arm and returns pw1's raw [a|g]
    // -- a wrong answer, not a failure. The shipped `modalsilu` resident is such a build.
    glu_epi: bool,
    // (K, N, activation, a_panel) -> stream. `a_panel` is part of the KEY, not just the filename:
    // at k!=KRES the same (k,n,act) has both a panel-major and a row-major stream, and letting them
    // collide would hand one consumer the other's A tap and be silently wrong.
    streams: RefCell<HashMap<(usize, usize, Act, bool), Rc<NStream>>>,
    // fc2 output for the one-dispatch K=DFF collapse. Its own BO against self.kern's C group --
    // NOT the stream's shared bo_c (every N=1024 identity dispatch would alias it) and NOT the
    // accadd scratch (that is another kernel's group). Lazily allocated so the default path,
    // which never takes this branch, does not spend a buffer on it.
    fc2_out: RefCell<Option<Rc<Bo>>>,
    // A buffer for the K=DFF host-fed GEMM (subsample out). PAD_M x DFF bf16 = 4 MB, 4x bo_a's
    // width, so it is separate and lazily allocated -- a run that never takes that path pays nothing.
    bo_a4: RefCell<Option<Rc<Bo>>>,
    wcache: RefCell<HashMap<String, Rc<Bo>>>,      // packed weight BOs by id
    ncache: RefCell<HashMap<String, usize>>,       // weight N (ncols) by id, paired with wcache
    relpos_dir: PathBuf,                           // {root}/artifacts/relpos (per-T xclbin cache)
    relpos: RefCell<HashMap<usize, Rc<RelposK>>>,  // T -> loaded resident block
    conveyor_dir: PathBuf,                         // {root}/artifacts/conveyor (8-head xclbin)
    conveyor: RefCell<Option<Rc<ConveyorK>>>,      // loaded 8-head conveyor (H baked, single instance)
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
    // FUSED ctxLN->affine_cast in ONE dispatch (x, gamma|beta) -> bf16, skipping the f32 bo_ln
    // intermediate entirely. OPTIONAL like glu: absent -> the two-dispatch chain.
    //
    // This is a TRANSITION lever, not a dispatch-count one. Measured in place on the shipped path
    // (`NPU_DISPATCH_LOG=1`): a dispatch that FOLLOWS a different xclbin costs 1.669 ms mean against
    // 0.267 ms for one on the same xclbin, and 239 of 552 dispatches/clip follow a switch. Collapsing
    // this pair removes 48 dispatches AND the 48 ctxln->affcast transitions with them.
    //
    // NOT usable by `resident_mha_affine_ln_f32`, which reads the f32 `bo_ln` the fused kernel never
    // materializes -- that caller goes through `ln_affine_cast_chained`. `PARAKEET_LN_FUSED=0`
    // restores the chain everywhere.
    lnaffcast: Option<LnFused>,
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
    // fc1 that drains chunk-major bf16 itself, deleting the deint dispatch. OPTIONAL + opt-in.
    fc1_panel_bf16: Option<Fc1PanelBf16>,
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

/// Fused ctxLN -> affine_cast brick: `(x f32[PAD_M,KRES], gamma|beta f32[2*KRES]) -> bf16[PAD_M,KRES]`
/// in one dispatch. Same 8-arg host ABI as the bricks it replaces (x g3, gb g4, out g5), so it drops
/// straight into either call site. Reuses the chain's `bo_gb`/`bo_bf16`, owning only its dummies.
struct LnFused {
    kern: Rc<Kernel>,
    instr: Bo,
    n: usize,
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

/// fc1 with the K-PANEL PACKING FOLDED INTO ITS OWN C DRAIN. DEFAULT ON; `PARAKEET_FC1_PACK_IN_DRAIN=0` opts out.
///
/// The shipped seam is two dispatches on two xclbins: the modal fc1 writes C row-major f32
/// [PAD_M,DFF], then `deint` casts it to bf16 and reorders it chunk-major so the fc2 K-split can
/// take each chunk as a sub-buffer. This variant makes the GEMM write that layout directly, so the
/// deint dispatch -- and one hw-context transition per FFN -- disappear. `bo_out` is bit-compatible
/// with `bo_deint`: same [n_chunks,PAD_M,KRES] bf16 chunk-major buffer, so nothing downstream moves.
///
/// TWO things make this a different xclbin rather than a different instruction stream:
///   * chunk-major drain alone IS insts-only (a pure re-stride of the same 4-D drain TAP; the PDI is
///     byte-identical). That half is free.
///   * bf16 C is not. The modal matmul reduces f32 IN-PLACE into the C tile, which only works while
///     dtype_out == dtype_acc; a bf16 C tile needs the per-core f32 accumulator back -- the buffer
///     the modal design deleted to make the m=64 fast tile fit. At m=64 L1 overflows by ~11.6 KB, so
///     this is built at m=32 and pays ~+0.24 ms/dispatch for the smaller tile.
///
/// DEFAULT ON since 2026-07-28. It is still a genuine trade, not a free win -- it adds a SECOND
/// full-array design to the resident set and regresses the fc1 tile from m=64 to m=32 -- but the
/// trade was measured end to end and it wins:
///
///   * -3.4% whole-clip (0.698 -> 0.674 s/clip, min-of-3 over 17 clips), -48 commands and -48
///     hwctx switches (504 -> 456, 191 -> 143). Note this is SMALLER than an earlier uncommitted
///     measurement claimed (-4.6%); the modeled ~3.5% was the accurate one.
///   * Accuracy: the burst-aware parity gate FAILS `new-burst` (+0.42 at ru_02[77]), and that
///     failure is cosmetic. On that exact clip the fold takes burst frames from 8 to 34 and the
///     transcript stays IDENTICAL to f32 truth. Across 17 clips the folded path differs from f32
///     on FEWER clips than the shipped one (2 vs 4) with fewer word edits (3 vs 6), and differing
///     tokens sit at or below chance on burst frames. Zero frames reach the 1.0 sensitivity knee.
///
/// A modal<->modal transition does cost more than the modal<->deint pair it replaces (deint is
/// cheap precisely because it is a small design), which is why the win is much smaller than the
/// -48/-48 suggests. Model the whole per-FFN sequence, not the command count.
///
/// Revert with `PARAKEET_FC1_PACK_IN_DRAIN=0`. First thing to reconsider if the resident set turns
/// out to be design-constrained by the LN-into-GEMM-epilogue work, which is worth ~20% against this
/// 3.4%. See the deint-fold-into-gemm-drain task.
struct Fc1PanelBf16 {
    kern: Rc<Kernel>,
    instr: Bo,
    n: usize,
    bo_out: Bo, // [n_chunks*PAD_M*KRES] bf16 chunk-major -- same layout deint used to produce
    dummy_tmp: Bo,
    dummy_tr: Bo,
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

/// Per-call-site census of dispatches on the resident modal kernel, printed with the stats.
///
/// Exists because the aggregate count cannot answer a COVERAGE question. The bf16-narrowing
/// simulation reached one site, and the dispatch log shows 360 modal dispatches per clip against
/// fc2's 192 -- so the remaining 168 had to be attributed to a site before the gate could be
/// believed. Always on; a BTreeMap increment per dispatch is free next to a device command.
static MODAL_SITES: std::sync::Mutex<Option<std::collections::BTreeMap<&'static str, usize>>> =
    std::sync::Mutex::new(None);

fn modal_site(site: &'static str) {
    let mut g = MODAL_SITES.lock().unwrap();
    *g.get_or_insert_with(Default::default).entry(site).or_insert(0) += 1;
}

pub fn modal_site_report() -> String {
    let g = MODAL_SITES.lock().unwrap();
    match g.as_ref() {
        None => "modal sites: none".into(),
        Some(m) => {
            let tot: usize = m.values().sum();
            let per: Vec<String> = m.iter().map(|(k, v)| format!("{k}={v}")).collect();
            format!("modal dispatch sites ({tot} total): {}", per.join(" "))
        }
    }
}

const DFF: usize = 4096; // Parakeet FFN inner dim (fc1 N / fc2 K)
// Tile of the bf16-out fc1 (Fc1PanelBf16). NOT `self.tile`: bf16 out needs a per-core f32 accumulator that
// does not fit alongside the m=64 C tile (L1 overflows by ~11.6 KB), so this variant only exists at m=32.
const FC1_PANEL_BF16_TILE: &str = "32x32x128";

/// `PARAKEET_FOLD_FC1=1`: make fc1's bf16-out xclbin the RESIDENT one, so fc1 and every other modal
/// GEMM share a single hardware context.
///
/// The lever: fc1 is the only op in the 3-xclbin rotation that could move, and moving it takes the
/// rotation from 3 visits per FFN to 2 -- 47-48 transitions/clip at the controlled 1.78 ms each. It
/// works without any dispatch-site change because `Device::load_kernel` caches by PATH: point the
/// resident at fc1's own xclbin and both handles resolve to the same `Kernel`, so no switch can occur
/// between them.
///
/// The cost, and why this is opt-in rather than default. `split_acc` bakes `dtype_out` into the array
/// program, so one xclbin cannot serve an f32-out and a bf16-out GEMM; folding therefore moves EVERY
/// modal GEMM to bf16 out, and the only bf16-out build that fits L1 is m=32 (at m=64 it overflows by
/// exactly 11596 B, verified by build). Measured tile penalty: 1.13x at N=1024, 1.27x at N=4096.
/// Projected net ~-67 ms/clip.
///
/// **TIMING-ONLY as it stands.** The host readback still decodes C as f32 while the folded xclbin
/// writes bf16, so encoder OUTPUT IS WRONG under this flag -- it exists to measure the dispatch
/// sequence, the transition count and the wall clock end-to-end. Shipping it additionally needs the
/// readback converted in both crates that dispatch modal GEMMs (here and `npu_asr::ctx2`). The
/// accuracy side is already priced separately: zero WER cost on 200 clips.
fn fold_fc1() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("PARAKEET_FOLD_FC1").map(|v| v != "0").unwrap_or(false))
}

/// `PARAKEET_FOLD_GLU=1`: apply the conv-module gate in pw1's OWN epilogue (`rtp[0]==3`), taking the
/// resident conv front from two dispatches (pw1-identity + the standalone GLU brick) to one -- 24/clip
/// on the 24-block encoder. It also deletes a narrowing: the epilogue still holds pw1's f32
/// accumulator, so it never rounds to bf16 to hand GLU its input the way `glu.cc` had to.
///
/// Opt-in, and it carries a pairing requirement rather than just a flag. The mode is selected by the
/// instruction stream, but the BRANCH lives in the resident's compiled epilogue, so it must be paired
/// with `NPU_RESIDENT_XCLBIN=<...modalglu...>`. On a resident built before that branch existed --
/// which the shipped `modalsilu` one is -- `rtp[0]=3` falls through to identity and returns pw1's raw
/// `[a|g]`: a wrong answer, not a failed run. `conv_pw1_glu_folded` asserts the pairing.
fn fold_glu() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("PARAKEET_FOLD_GLU").map(|v| v != "0").unwrap_or(false))
}
// Variant B fc2 = deinterleave -> 4x K=KRES modal (same tile as host) on device sub-buffers +
// host-accumulate = bit-identical to the host K-split (WER-neutral), A device-side.

/// Resolve one kernel artifact pair by `stem` under `dir`, routed through the shared
/// `kernel_registry` naming convention (`engine-op-manifest-and-dynamic-xclbin`) instead of a
/// hand-built `format!("final_{stem}.xclbin")` literal. This fn does not decide "built or not" --
/// callers that need graceful degradation on a missing artifact keep their own `.exists()` gate
/// before calling it (unchanged from before this routing pass); this only replaces how the PATH
/// is built for a stem already known to be present (or about to be loaded unconditionally, same
/// as the pre-existing `dev.load_kernel(...).unwrap_or_else(|e| panic!(...))` callers already do
/// on a missing file). When `NPU_KERNEL_MANIFEST_VERIFY=1`, the artifact bytes are additionally
/// re-hashed against `dir/kernel_manifest.json` (`kernel_registry::resolve_checked`) and a
/// MISMATCH panics here, loud, instead of loading a silently-wrong xclbin -- unlike "missing"
/// (still handled by each call site's own gate/panic), "wrong content at the expected path" is
/// exactly the silent-wrong-load bug class the manifest exists to catch (see `kernel_registry`
/// module docs: this project's xclbin containers carry no independent op-identity signal to
/// check tokens against otherwise). Mirrors `npu_asr::conv_npu::ConvNpu::band()`'s existing wiring.
fn resolve_verified(dir: &Path, stem: &str) -> kernel_registry::KernelArtifacts {
    if std::env::var("NPU_KERNEL_MANIFEST_VERIFY").is_ok() {
        kernel_registry::resolve_checked(dir, stem)
            .unwrap_or_else(|e| panic!("kernel manifest check failed for stem={stem}: {e}"))
    } else {
        kernel_registry::resolve(dir, stem)
    }
}


/// Times a device dispatch into [`NpuStats::dispatch_s`] for as long as it is alive.
///
/// WHY AN RAII GUARD AND NOT A LINE AT EACH CALL SITE: a census on 2026-08-02 found 25 of 32
/// `run_matmul8` call sites incremented `dispatches` (the count) and never `dispatch_s` (the time).
/// Only ONE site was timed, so `dispatch_s` reported a single path and read as if it were the whole
/// device total -- which is how the phase profiler and NpuStats came to disagree ~4x about where
/// encode time goes. An instrument that has to be remembered at 32 sites will be forgotten at 25;
/// this makes the timing a scoped object so a dispatch cannot be added without one being visible.
struct DispatchTimer<'a> {
    stats: &'a std::cell::RefCell<NpuStats>,
    t: Instant,
}

impl Drop for DispatchTimer<'_> {
    fn drop(&mut self) {
        self.stats.borrow_mut().dispatch_s += self.t.elapsed().as_secs_f64();
    }
}

impl NpuMatmul {
    /// Scope a device dispatch so its wall time lands in `dispatch_s`. Hold it in a `{ }` block
    /// around the `run_matmul8` call; it must not straddle another `self.stats.borrow_mut()`.
    #[inline]
    fn dtimer(&self) -> DispatchTimer<'_> {
        DispatchTimer { stats: &self.stats, t: Instant::now() }
    }

    pub fn open(root: &Path) -> Self {
        let dev = Device::open(0).expect("open NPU (single-tenant: stop npu-asr/voxd)");
        let base = root.join(WA_SUBDIR);
        // resident kernel tile: fast BFP16 64x32x128 (default) or native bf16 32x32x32 (NPU_NATIVE=1),
        // or the FOLD's 32x32x128 (see `fold_fc1`).
        let tile = if fold_fc1() {
            FC1_PANEL_BF16_TILE.to_string()
        } else if std::env::var("NPU_NATIVE").is_ok() {
            "32x32x32".to_string()
        } else {
            "64x32x128".to_string()
        };
        // resident xclbin = a K=1024 whole_array kernel for this tile. What is N-independent is the
        // BD-CHAIN SHAPE, not the array program: all three modal insts are 1436 words and N lives in
        // the BDs' size/stride fields (see `stream()`), while the device region does differ across N
        // by 32 cores x 4 lines -- each core's tile-loop bound, baked at 2/4/8 for N=1024/2048/4096.
        // A stream whose N is smaller than the build's therefore spans several dispatches inside one
        // core-body iteration, which the free-running consumer + order-preserving objectFIFOs absorb.
        // Anything read from RTP outside that tile loop does NOT survive it -- see the k_trip note in
        // whole_array_modal_iron.py's core_fn. Prefer the largest N present;
        // fall back to a smaller surviving build (the N=4096/2048 twins were deleted by the
        // an earlier occupancy run; N=1024 survives). Env NPU_RESIDENT_XCLBIN overrides.
        let (xclbin, modal) = if fold_fc1() {
            // The resident IS fc1's bf16-out xclbin. Same path as `Fc1PanelBf16` resolves, so
            // load_kernel's path cache hands both the same Kernel and the fc1<->fc2 transition
            // disappears without touching a dispatch site.
            let stem = format!("{PAD_M}x{KRES}x{DFF}_{FC1_PANEL_BF16_TILE}_8c_modalsilubf16outpanel{KRES}");
            (resolve_verified(&base, &stem).xclbin, true)
        } else if let Ok(p) = std::env::var("NPU_RESIDENT_XCLBIN") {
            let path = PathBuf::from(p);
            // Arbitrary override path (a manual/debug knob): no guaranteed `final_{stem}.xclbin`
            // convention to recover a stem from, so this branch keeps the raw filename check it
            // always used rather than force-fitting kernel_registry's stem grammar onto it.
            let modal = path.file_name().and_then(|s| s.to_str()).is_some_and(|s| s.contains("modal"));
            (path, modal)
        } else {
            // A1 (ff_act on-chip): prefer the MODAL resident xclbin (fused f32-out epilogue; the
            // per-inst-stream RTP selects silu@N=4096 / identity elsewhere -> the FFN SiLU runs on
            // chip with zero extra hw-context switches). Fall back to the plain matmul xclbin if the
            // modal build is absent (then `modal=false` and the host keeps applying silu).
            let modal_stem = format!("512x1024x4096_{tile}_8c_modalsilu");
            let stem = if kernel_registry::xclbin_path(&base, &modal_stem).exists() {
                modal_stem
            } else {
                let mut chosen = None;
                for n in ["4096", "2048", "1024"] {
                    let cand_stem = format!("512x1024x{n}_{tile}_8c");
                    if kernel_registry::xclbin_path(&base, &cand_stem).exists() {
                        chosen = Some(cand_stem);
                        break;
                    }
                }
                chosen.unwrap_or_else(|| format!("512x1024x4096_{tile}_8c"))
            };
            // Opt-in manifest verification (engine-op-manifest-and-dynamic-xclbin): this xclbin is
            // REQUIRED (no host fallback exists for a missing resident matmul kernel -- a missing
            // file already panics below at `load_kernel`), so a content mismatch under
            // NPU_KERNEL_MANIFEST_VERIFY=1 panics too rather than degrading, mirroring
            // conv_npu.rs::band()'s existing wiring (see `resolve_verified`).
            let path = resolve_verified(&base, &stem).xclbin;
            // The modal resident bakes the silu/identity epilogue; the plain one does not (host
            // silu). Derived from the stem's parsed tokens (kernel_registry::parse_stem_tokens),
            // not a raw filename substring match -- the stem is what actually carries this; the
            // filename was always just `final_{stem}.xclbin` (engine-op-manifest-and-dynamic-xclbin
            // item 4).
            let modal =
                kernel_registry::parse_stem_tokens(&stem).variant.is_some_and(|v| v.contains("modal"));
            (path, modal)
        };
        let krtp = xclbin.file_name().and_then(|s| s.to_str()).is_some_and(|s| s.contains("krtp"));
        let glu_epi = xclbin.file_name().and_then(|s| s.to_str()).is_some_and(|s| s.contains("modalglu"));
        if !npu_xrt::quiet() {
            eprintln!("[npu] resident xclbin = {} (modal={modal} krtp={krtp} glu_epi={glu_epi})", xclbin.display());
        }
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
            krtp,
            glu_epi,
            streams: RefCell::new(HashMap::new()),
            fc2_out: RefCell::new(None),
            bo_a4: RefCell::new(None),
            wcache: RefCell::new(HashMap::new()),
            ncache: RefCell::new(HashMap::new()),
            relpos_dir: root.join("artifacts/relpos"),
            relpos: RefCell::new(HashMap::new()),
            conveyor_dir: root.join("artifacts/conveyor"),
            conveyor: RefCell::new(None),
            ln_dir: root.join("artifacts/parakeet/ln"),
            resident_ln: RefCell::new(None),
            stats: RefCell::new(NpuStats::default()),
        }
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

    /// Max clip length T the resident relpos block can serve (the largest bucket's baked BUILT_T).
    /// Callers MUST gate the resident MHA path on `t <= relpos_max_t()` and fall back to the host
    /// attention for longer clips -- the resident BOs/dataflow are sized for BUILT_T and cannot serve
    /// T beyond it.
    pub fn relpos_max_t(&self) -> usize { RELPOS_BUCKETS.last().unwrap().0 }

    /// Pick the SMALLEST bucket whose BUILT_T >= t (RELPOS_BUCKETS is ascending). Panics only if t
    /// exceeds the ceiling bucket -- callers gate on relpos_max_t() first.
    /// A/B toggle: PARAKEET_RELPOS_NO_BUCKET=1 forces the ceiling bucket for EVERY clip (the pre-
    /// T-bucketing baseline), so one binary runs the rigorous same-session A/B (no rebuild drift).
    fn relpos_bucket_for(t: usize) -> (usize, usize, &'static str) {
        if std::env::var_os("PARAKEET_RELPOS_NO_BUCKET").is_some() {
            return *RELPOS_BUCKETS.last().unwrap();
        }
        // A/B toggle: PARAKEET_RELPOS_2BUCKET=1 uses only the first + ceiling bucket (skips the mids),
        // so one binary A/Bs {100,172} vs the full {100,152,172} same-session (measure the mid bucket).
        let two = std::env::var_os("PARAKEET_RELPOS_2BUCKET").is_some();
        let last = RELPOS_BUCKETS.len() - 1;
        RELPOS_BUCKETS.iter().enumerate()
            .filter(|(i, _)| !two || *i == 0 || *i == last)
            .map(|(_, b)| b)
            .find(|(bt, _, _)| *bt >= t)
            .copied()
            .unwrap_or_else(|| panic!("clip T={t} exceeds relpos ceiling BUILT_T={}", RELPOS_BUCKETS.last().unwrap().0))
    }

    /// Load (once) the resident relpos block for one T-bucket (BUILT_T, KB, subdir). Reads the xclbin +
    /// template insts from {root}/artifacts/relpos/{subdir}/ (pre-build: scripts/relpos_prebuild.sh).
    /// The bucket serves any clip T <= its BUILT_T; per dispatch we patch the insts t_active word.
    /// Cached in `self.relpos` keyed by BUILT_T so each bucket loads once and stays co-resident.
    fn relpos_block(&self, bt: usize, kb: usize, subdir: &str) -> Rc<RelposK> {
        if let Some(k) = self.relpos.borrow().get(&bt) {
            return k.clone();
        }
        let p = 2 * bt - 1;
        let cdiv = |a: usize, b: usize| (a + b - 1) / b;
        let n_qt = cdiv(bt, RELPOS_TQ);
        let tp = cdiv(bt, kb) * kb;
        let pp = cdiv(p, kb) * kb;
        let ctx_rows = n_qt * RELPOS_TQ;
        let dir = self.relpos_dir.join(subdir);
        let xclbin = dir.join("final.xclbin");
        let insts = dir.join("insts.bin");
        let kern = self
            .dev
            .load_kernel(xclbin.to_str().unwrap(), None)
            .unwrap_or_else(|e| panic!("load relpos bucket {bt} ({}): {e:?}\n  pre-build: scripts/relpos_prebuild.sh", xclbin.display()));
        let ib = std::fs::read(&insts).unwrap_or_else(|e| panic!("read {}: {e}", insts.display()));
        let instr_template: Vec<u32> = ib
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let n_instr = instr_template.len();
        let g = |i| kern.group_id(i).unwrap();
        let bo_instr = self.dev.alloc_bo(&kern, ib.len(), FLAG_CACHEABLE, g(1)).unwrap();
        // BOs concatenate all RELPOS_HEADS heads (head h at slice h*head_len). One dispatch fills
        // all H heads; the generator scatters each head's slice to its own core via an offset tap.
        let bo_quv = self.dev.alloc_bo(&kern, RELPOS_HEADS * 2 * n_qt * RELPOS_TQ * RELPOS_DK * 2, FLAG_HOST_ONLY, g(3)).unwrap();
        // SPLITP: kpv is k | p_hi | p_lo | V (the positional operand p split-bf16 for ~f32
        // BD precision -- the resident-MHA WER fix). Two padded p-sections (2*pp). Per-head, x H.
        let bo_kpv = self.dev.alloc_bo(&kern, RELPOS_HEADS * (tp + pp + pp + tp) * RELPOS_DK * 2, FLAG_HOST_ONLY, g(4)).unwrap();
        let bo_ctx = self.dev.alloc_bo(&kern, RELPOS_HEADS * ctx_rows * RELPOS_DK * 2, FLAG_HOST_ONLY, g(5)).unwrap();
        let rk = Rc::new(RelposK { kern, instr_template, n_instr, bo_instr, bo_quv, bo_kpv, bo_ctx, built_t: bt, n_qt, tp, pp, ctx_rows });
        self.relpos.borrow_mut().insert(bt, rk.clone());
        rk
    }

    /// Resident relpos-MHA block for ALL RELPOS_HEADS heads in ONE dispatch (H parallel cores, one
    /// head/core). q/k/v are [t, D=H*DK], pm is [P=2t-1, D], ubias/vbias are [H, DK]. Returns ctx
    /// [t, D] with head h in columns [h*DK..(h+1)*DK]. t <= relpos_max_t(). STEP-C: pad each head's
    /// stream to BUILT_T, PATCH every t_active word of the insts template (one per head's RTP) to t,
    /// dispatch the single resident block (3-BO ABI, all H heads concatenated), unpack bf16 CTX.
    /// This REPLACES the old per-head loop (H sequential dispatches on 1 core) with 1 dispatch that
    /// runs the H heads in parallel -- the Phase-2 perf rework.
    pub fn relpos_mha_batched(&self, q: &Array2<f32>, k: &Array2<f32>, pm: &Array2<f32>,
                              v: &Array2<f32>, ubias: &Array2<f32>, vbias: &Array2<f32>) -> Array2<f32> {
        let t = q.nrows();
        let d = q.ncols();
        let dk = RELPOS_DK;
        let h = RELPOS_HEADS;
        assert_eq!(d, h * dk, "hidden D must be RELPOS_HEADS*RELPOS_DK");
        // T-bucketing: dispatch the SMALLEST bucket whose BUILT_T >= this clip's T, so short clips run
        // a smaller padded dataflow (less wasted relpos padding compute). Callers gate on relpos_max_t().
        let (bt, kb, subdir) = Self::relpos_bucket_for(t);
        let rk = self.relpos_block(bt, kb, subdir);
        assert!(t <= rk.built_t, "clip T={t} exceeds relpos bucket BUILT_T={}", rk.built_t);
        let quv_head = 2 * rk.n_qt * RELPOS_TQ * RELPOS_DK;
        let kpv_head = (rk.tp + rk.pp + rk.pp + rk.tp) * RELPOS_DK;
        // Concatenate all H heads' quv / kpv (head hh at slice hh*head_len -- the generator scatters
        // each slice to its own core). Per head: QUV tile-interleaved [qu_t0,qv_t0,...]; KPV = k(pad
        // tp) | p_hi(pad pp) | p_lo(pad pp) | V(pad tp) with p split-bf16 (SPLITP: p_hi=bf16(p),
        // p_lo=bf16(p-p_hi); BD=qv.p_hi+qv.p_lo recovers ~f32 p -- the probe-localized ~1% sink).
        // HOST work, inside a scope (`mhsa_resident`) that is bucketed Npu -- so until this leaf
        // existed, ~57 ms/clip of host data shuffling was counted as NPU time.
        let (quv, kpv) = {
        let _p_concat = crate::prof::phase::PhaseScope::new("mh_concat", crate::prof::phase::Bucket::Host);
        let mut quv = Vec::<f32>::with_capacity(h * quv_head);
        let mut kpv = Vec::<f32>::with_capacity(h * kpv_head);
        for hh in 0..h {
            let col = hh * dk;
            let qh = q.slice(s![.., col..col + dk]);
            let kh = k.slice(s![.., col..col + dk]).to_owned();
            let ph = pm.slice(s![.., col..col + dk]).to_owned();
            let vh = v.slice(s![.., col..col + dk]).to_owned();
            let mut qu = qh.to_owned();
            let mut qv = qh.to_owned();
            for i in 0..t {
                for c in 0..dk {
                    qu[[i, c]] += ubias[[hh, c]];
                    qv[[i, c]] += vbias[[hh, c]];
                }
            }
            for qi in 0..rk.n_qt {
                let q0 = qi * RELPOS_TQ;
                let take = RELPOS_TQ.min(t.saturating_sub(q0));
                push_pad_rows(&mut quv, &qu, q0, take, RELPOS_TQ);
                push_pad_rows(&mut quv, &qv, q0, take, RELPOS_TQ);
            }
            let p_lo = ph.mapv(|x| x - npu_xrt::bf16_bits_to_f32(npu_xrt::f32_to_bf16_bits(x)));
            push_pad_rows(&mut kpv, &kh, 0, t, rk.tp);
            push_pad_rows(&mut kpv, &ph, 0, ph.nrows(), rk.pp);
            push_pad_rows(&mut kpv, &p_lo, 0, ph.nrows(), rk.pp);
            push_pad_rows(&mut kpv, &vh, 0, t, rk.tp);
        }
        (quv, kpv)
        };
        let (qb, kb) = {
            let _p = crate::prof::phase::PhaseScope::new("mh_pack", crate::prof::phase::Bucket::Host);
            let mut qb = vec![0u16; quv.len()];
            let mut kb = vec![0u16; kpv.len()];
            npu_xrt::pack_f32_to_bf16(&quv, &mut qb);
            npu_xrt::pack_f32_to_bf16(&kpv, &mut kb);
            (qb, kb)
        };
        let t0 = Instant::now();
        // STEP-C: patch EVERY t_active word (one per head's RTP, all == BUILT_T in the template) to
        // this clip's t. All H heads share the clip's single t_active. (prebuild asserts count==H.)
        {
            let _p = crate::prof::phase::PhaseScope::new("mh_instpatch", crate::prof::phase::Bucket::Marshal);
            let mut insts = rk.instr_template.clone();
            for w in insts.iter_mut() {
                if *w == rk.built_t as u32 {
                    *w = t as u32;
                }
            }
            let instr_bytes: Vec<u8> = insts.iter().flat_map(|w| w.to_le_bytes()).collect();
            rk.bo_instr.write_bytes(&instr_bytes).unwrap();
            rk.bo_instr.sync_to_device().unwrap();
        }
        {
            let _p = crate::prof::phase::PhaseScope::new("mh_upload", crate::prof::phase::Bucket::Marshal);
            rk.bo_quv.write_bytes(u16_bytes(&qb)).unwrap();
            rk.bo_quv.sync_to_device().unwrap();
            rk.bo_kpv.write_bytes(u16_bytes(&kb)).unwrap();
            rk.bo_kpv.sync_to_device().unwrap();
        }
        {
            let _p = crate::prof::phase::PhaseScope::new("mh_kernel", crate::prof::phase::Bucket::Npu);
            rk.kern.run_dwconv6(3, &rk.bo_instr, rk.n_instr, &rk.bo_quv, &rk.bo_kpv, &rk.bo_ctx).unwrap();
            rk.bo_ctx.sync_from_device().unwrap();
        }
        {
            let mut s = self.stats.borrow_mut();
            s.dispatch_s += t0.elapsed().as_secs_f64();
            s.dispatches += 1;
        }
        let _p_unpack = crate::prof::phase::PhaseScope::new("mh_unpack", crate::prof::phase::Bucket::Host);
        let mut cb = vec![0u8; h * rk.ctx_rows * RELPOS_DK * 2];
        rk.bo_ctx.read_bytes(&mut cb).unwrap();
        // Unpack: head hh's ctx (first t of ctx_rows) -> columns [hh*DK..(hh+1)*DK] of [t, D].
        let mut ctx = Array2::<f32>::zeros((t, d));
        for hh in 0..h {
            let base = hh * rk.ctx_rows * RELPOS_DK;
            let col = hh * dk;
            for i in 0..t {
                for dd in 0..RELPOS_DK {
                    let off = (base + i * RELPOS_DK + dd) * 2;
                    let u = u16::from_le_bytes([cb[off], cb[off + 1]]);
                    ctx[[i, col + dd]] = f32::from_bits((u as u32) << 16);
                }
            }
        }
        ctx
    }

    /// bf16x2 device-A gate helper (opt-in `PARAKEET_MHA_SPLITA`): run the resident ctxLN+affine and
    /// return the DEVICE affine_LN(x) in f32 [m, KRES] (read `bo_ln` back + apply the affine on host).
    /// The caller splits it into A_hi + A_lo (near-f32) and feeds the QKV GEMM twice, testing whether a
    /// near-f32 device A closes the LN->QKV seam WER gap (host golden `HOSTQKV=1` 8.5 vs full-resident
    /// bf16-A 8.9). Uses the SAME device ctxLN as `resident_mha_ln_qkv` (so it is faithful to the 8.9
    /// path's LN), reads no shared default infra, and mutates nothing on the FFN/conv paths.
    pub fn resident_mha_affine_ln_f32(&self, x: &Array2<f32>, gamma: &[f32], beta: &[f32]) -> Array2<f32> {
        self.stats.borrow_mut().calls += 1;
        let m = x.nrows();
        // CHAINED on purpose: this path reads the f32 `bo_ln` back, and the fused ctxLN->affine_cast
        // kernel never materializes it.
        let rl = self.ln_affine_cast_chained(x, gamma, beta);
        rl.bo_ln.sync_from_device().unwrap();
        let mut cb = vec![0u8; PAD_M * KRES * 4];
        rl.bo_ln.read_bytes(&mut cb).unwrap();
        let mut out = Array2::<f32>::zeros((m, KRES));
        for r in 0..m {
            for c in 0..KRES {
                let off = (r * KRES + c) * 4;
                let ln = f32::from_le_bytes([cb[off], cb[off + 1], cb[off + 2], cb[off + 3]]);
                out[[r, c]] = ln * gamma[c] + beta[c];
            }
        }
        out
    }

    /// Resident MHA LN->QKV seam (norm_self_att on-NPU), the MHSA head frontier advance: ctxLN ->
    /// affine_cast(gamma,beta) -> bf16 affine_LN(x) BO (device-side), then the q/k/v modal GEMMs
    /// (N=KRES=D identity, no silu) ALL run off that ONE resident bf16 A -- so the host norm_self_att
    /// LN is off the MHSA frontier. Returns (q,k,v) [t,D] f32, exactly host layernorm(x)@Wq/k/v
    /// (bf16-class). `pos` is NOT normalized, so it stays a plain mm_lazy in the caller. Requires the
    /// resident seam (caller gates on resident_ff_available); `idq/idk/idv` key the per-weight BO cache.
    pub fn resident_mha_ln_qkv<Fq, Fk, Fv>(
        &self, x: &Array2<f32>, gamma: &[f32], beta: &[f32],
        make_wq: Fq, idq: &str, make_wk: Fk, idk: &str, make_wv: Fv, idv: &str,
    ) -> (Array2<f32>, Array2<f32>, Array2<f32>)
    where
        Fq: FnOnce() -> Array2<f32>,
        Fk: FnOnce() -> Array2<f32>,
        Fv: FnOnce() -> Array2<f32>,
    {
        self.stats.borrow_mut().calls += 1;
        let m = x.nrows();
        let n = KRES; // q/k/v projections are [D,D]; N = D = KRES
        // LN + affine cast ONCE -> bo_bf16 = affine_LN(x) bf16 [PAD_M, KRES] (device-resident).
        let rl = self.ln_affine_cast(x, gamma, beta);
        let q = self.qkv_proj(&rl.bo_bf16, m, n, make_wq, idq);
        let k = self.qkv_proj(&rl.bo_bf16, m, n, make_wk, idk);
        let v = self.qkv_proj(&rl.bo_bf16, m, n, make_wv, idv);
        (q, k, v)
    }

    /// One q/k/v modal projection off an ALREADY-resident bf16 A (the shared affine_LN(x)): fetch/build
    /// the cached [KRES,n] weight BO, then a single identity (no-silu) modal dispatch -> C[m,n] f32.
    fn qkv_proj<F: FnOnce() -> Array2<f32>>(&self, a_bo: &Bo, m: usize, n: usize, make_w: F, id: &str) -> Array2<f32> {
        let cached = self.wcache.borrow().get(id).cloned();
        let wbo = if let Some(bo) = cached {
            bo
        } else {
            let w = make_w();
            assert_eq!(w.nrows(), KRES, "qkv W nrows {} != {KRES}", w.nrows());
            assert_eq!(w.ncols(), n, "qkv W ncols {} != {n}", w.ncols());
            self.weight_bo(id, w.view())
        };
        self.dispatch_with_a(a_bo, m, &wbo, n, false)
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
            let stem = format!("{n}_{PAD_M}x{KRES}");
            kernel_registry::xclbin_path(&self.ln_dir, &stem).exists()
                && kernel_registry::insts_path(&self.ln_dir, &stem).exists()
        });
        // full FFN (Variant B) also needs the deinterleave xclbin. Pre-existing asymmetry,
        // preserved as-is: this checks only the xclbin, not its insts counterpart (unlike every
        // other presence gate in this function) -- a behavior-preserving routing pass is not the
        // place to also change what "present" means here.
        let fc2ok =
            kernel_registry::xclbin_path(&self.ln_dir, &format!("deint_{PAD_M}x{DFF}")).exists();
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
            let art = resolve_verified(&self.ln_dir, &format!("{name}_{PAD_M}x{KRES}"));
            let kern = self
                .dev
                .load_kernel(art.xclbin.to_str().unwrap(), None)
                .unwrap_or_else(|e| panic!("load resident-ln {} : {e:?}\n  prebuild: build ctxln+cast at {PAD_M}x{KRES} and copy to artifacts/parakeet/ln", art.xclbin.display()));
            let ib = std::fs::read(&art.insts).unwrap_or_else(|e| panic!("read {}: {e}", art.insts.display()));
            let n = ib.len() / 4;
            let bo = self.dev.alloc_bo(&kern, ib.len(), FLAG_CACHEABLE, kern.group_id(1).unwrap()).unwrap();
            bo.write_bytes(&ib).unwrap();
            bo.sync_to_device().unwrap();
            (kern, bo, n)
        };
        let (ln_kern, ln_instr, ln_n) = load("ctxln");
        let (ac_kern, ac_instr, ac_n) = load("affcast");
        // cast @ DFF + the K=DFF fc2 matmul (explicit stems, not the {name}_PADxKRES pattern)
        let load_path = |dir: &Path, stem: &str| -> (Rc<Kernel>, Bo, usize) {
            let art = resolve_verified(dir, stem);
            let kern = self.dev.load_kernel(art.xclbin.to_str().unwrap(), None).unwrap_or_else(|e| panic!("load {} : {e:?}", art.xclbin.display()));
            let ib = std::fs::read(&art.insts).unwrap_or_else(|e| panic!("read {}: {e}", art.insts.display()));
            let n = ib.len() / 4;
            let bo = self.dev.alloc_bo(&kern, ib.len(), FLAG_CACHEABLE, kern.group_id(1).unwrap()).unwrap();
            bo.write_bytes(&ib).unwrap();
            bo.sync_to_device().unwrap();
            (kern, bo, n)
        };
        let (deint_kern, deint_instr, deint_n) =
            load_path(&self.ln_dir, &format!("deint_{PAD_M}x{DFF}"));
        // FUSED ctxLN->affine_cast, OPTIONAL like glu. Default ON when built; PARAKEET_LN_FUSED=0
        // forces the two-dispatch chain back (two-way, not a one-way flip).
        let lnaffcast = {
            let want = std::env::var("PARAKEET_LN_FUSED").map(|v| v != "0").unwrap_or(true);
            let stem = format!("lnaffcast_{PAD_M}x{KRES}");
            let present = kernel_registry::xclbin_path(&self.ln_dir, &stem).exists()
                && kernel_registry::insts_path(&self.ln_dir, &stem).exists();
            if want && present {
                let (kern, instr, n) = load_path(&self.ln_dir, &stem);
                let gg = |i| kern.group_id(i).unwrap();
                Some(LnFused {
                    dummy_tmp: self.dev.alloc_bo(&kern, 8, FLAG_HOST_ONLY, gg(6)).unwrap(),
                    dummy_tr: self.dev.alloc_bo(&kern, 1, FLAG_HOST_ONLY, gg(7)).unwrap(),
                    kern, instr, n,
                })
            } else {
                // NOT quiet-gated: NPU_QUIET suppresses BANNERS (which precision, which xclbin),
                // and this is a DEGRADATION -- the run is slower than the one that was asked for.
                // `npu transcribe`/`embed` set NPU_QUIET=1 by default, which is exactly where a
                // silently-missing artifact would otherwise go unreported.
                if want {
                    eprintln!("[npu] lnaffcast xclbin absent in {} -- LN seam stays a 2-dispatch chain", self.ln_dir.display());
                }
                None
            }
        };
        // conv-module GLU (step 2), OPTIONAL: load only if the glu xclbin was built. A/g3 input is fed
        // from the modal stream's bo_c (pw1 output); bo_out (g4) is the [PAD_M,KRES] f32 GLU result.
        let glu = {
            let stem = format!("glu_{PAD_M}x{KRES}");
            let present = kernel_registry::xclbin_path(&self.ln_dir, &stem).exists()
                && kernel_registry::insts_path(&self.ln_dir, &stem).exists();
            if present {
                let (kern, instr, n) = load_path(&self.ln_dir, &stem);
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
            let stem = format!("accadd_{PAD_M}x{KRES}");
            let present = kernel_registry::xclbin_path(&self.ln_dir, &stem).exists()
                && kernel_registry::insts_path(&self.ln_dir, &stem).exists();
            if present {
                let (kern, instr, n) = load_path(&self.ln_dir, &stem);
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
            let stem = format!("resadd_{PAD_M}x{KRES}_s050");
            let present = kernel_registry::xclbin_path(&self.ln_dir, &stem).exists()
                && kernel_registry::insts_path(&self.ln_dir, &stem).exists();
            if present {
                let (kern, instr, n) = load_path(&self.ln_dir, &stem);
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
            let stem = format!("resadd_{PAD_M}x{KRES}_s100");
            let present = kernel_registry::xclbin_path(&self.ln_dir, &stem).exists()
                && kernel_registry::insts_path(&self.ln_dir, &stem).exists();
            if present {
                let (kern, instr, n) = load_path(&self.ln_dir, &stem);
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
            let cast_stem = format!("cast_{PAD_M}x{DFF}");
            let mm_stem = format!("{PAD_M}x{DFF}x{KRES}_{}_8c_modalid", self.tile);
            // TWO hw_context slots (the cast and the K=DFF GEMM) for an OPT-IN path. Loading them
            // when the flag is off spent 2 of the driver's 16 on programs that can never dispatch;
            // that budget is what blocked attention-on-NPU. Gate on the flag, not on the artifacts
            // existing.
            let present = self.fc2_k4096_on()
                && kernel_registry::xclbin_path(&self.ln_dir, &cast_stem).exists()
                && kernel_registry::insts_path(&self.ln_dir, &cast_stem).exists()
                && kernel_registry::xclbin_path(&self.ln_dir, &mm_stem).exists()
                && kernel_registry::insts_path(&self.ln_dir, &mm_stem).exists();
            if present {
                let (cast_kern, cast_instr, cast_n) = load_path(&self.ln_dir, &cast_stem);
                let (mm_kern, mm_instr, mm_n) = load_path(&self.ln_dir, &mm_stem);
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
        // fc1 with the K-panel packing folded into its C drain. DEFAULT ON (PARAKEET_FC1_PACK_IN_DRAIN=0
        // opts out). Loaded whenever the artifact is present so the flag alone selects it; the m=32 tile
        // is baked into the name because bf16-out does not FIT at the m=64 fast tile (L1 overflow, see
        // Fc1PanelBf16). Absent artifact still falls back cleanly, which is what makes default-on safe
        // for a tree that has not rebuilt the modal kernels.
        let fc1_panel_bf16 = {
            // Same PARAKEET_MODAL_EPI_SUFFIX hook as the modal streams. The panel fc1 carries its
            // OWN copy of the epilogue, so an epilogue variant has to be built and selected here
            // too -- otherwise the variant reaches only the identity-mode GEMMs and misses fc1,
            // which is the one dispatch whose SiLU branch the variant actually changes.
            let sfx = std::env::var("PARAKEET_MODAL_EPI_SUFFIX").unwrap_or_default();
            let tag = format!("{PAD_M}x{KRES}x{DFF}_{FC1_PANEL_BF16_TILE}_8c_modalsilubf16outpanel{KRES}{sfx}");
            let present = kernel_registry::xclbin_path(&self.ln_dir, &tag).exists()
                && kernel_registry::insts_path(&self.ln_dir, &tag).exists();
            if present {
                let (kern, instr, n) = load_path(&self.ln_dir, &tag);
                let gg = |i| kern.group_id(i).unwrap();
                Some(Fc1PanelBf16 {
                    bo_out: self.dev.alloc_bo(&kern, (DFF / KRES) * PAD_M * KRES * 2, FLAG_HOST_ONLY, gg(5)).unwrap(),
                    dummy_tmp: self.dev.alloc_bo(&kern, 1, FLAG_HOST_ONLY, gg(6)).unwrap(),
                    dummy_tr: self.dev.alloc_bo(&kern, 4, FLAG_HOST_ONLY, gg(7)).unwrap(),
                    kern, instr, n,
                })
            } else {
                // Say why the DEFAULT path is not running. The old wording blamed the env var, which
                // is now wrong in the common case: the flag defaults on, so an operator who set
                // nothing would be told they had set something.
                // NOT quiet-gated, same reason as the lnaffcast warning above: a missing artifact
                // that silently costs ~3.4%/clip is a degradation, not a banner.
                if self.fc1_pack_in_drain_on() {
                    eprintln!("[npu] final_{tag}.xclbin absent in {} -- falling back to fc1+deint (slower by ~3.4%/clip). \
                               Build it with scripts/build_parakeet_modal_kernels.sh, or set PARAKEET_FC1_PACK_IN_DRAIN=0 to silence this.",
                              self.ln_dir.display());
                }
                None
            }
        };
        // conv-module depthwise conv1d (step 3), OPTIONAL. 3-buffer ABI in[C,T]/w[C,16]/out[C,T] bf16.
        // The dwconv/SiLU bricks are FOUR builds of ONE op, and the conv path has a strict
        // preference order (time-major fused > channel-major fused > separate dwconv+silu). Loading
        // all four spends four of the driver's 16 hw_context slots to dispatch at most two of them,
        // which is what left no slot for attention-on-NPU (EINVAL at CREATE_HWCTX). Decide the
        // variant ONCE here and load only that one.
        let have = |stem: &str| {
            kernel_registry::xclbin_path(&self.ln_dir, stem).exists()
                && kernel_registry::insts_path(&self.ln_dir, stem).exists()
        };
        let want_dws_t = have(&format!("dwconv_silu_t_{DW_C}x{DW_T}"));
        let want_dws = !want_dws_t && have(&format!("dwconv_silu_{DW_C}x{DW_T}"));
        // separate dwconv + silu only when neither fused variant exists
        let want_split = !want_dws_t && !want_dws;

        let dwconv = {
            let stem = format!("dwconv_{DW_C}x{DW_T}");
            let present = want_split && have(&stem);
            if present {
                let (kern, instr, n) = load_path(&self.ln_dir, &stem);
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
            let stem = format!("silu_{DW_C}x{DW_T}");
            let present = want_split && have(&stem);
            if present {
                let (kern, instr, n) = load_path(&self.ln_dir, &stem);
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
            let stem = format!("dwconv_silu_{DW_C}x{DW_T}");
            let present = want_dws && have(&stem);
            if present {
                let (kern, instr, n) = load_path(&self.ln_dir, &stem);
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
            let stem = format!("dwconv_silu_t_{DW_C}x{DW_T}");
            let present = want_dws_t && have(&stem);
            if present {
                let (kern, instr, n) = load_path(&self.ln_dir, &stem);
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
            ln_kern, ln_instr, ln_n, ac_kern, ac_instr, ac_n, lnaffcast,
            deint_kern, deint_instr, deint_n, glu, acc_add, resadd_s050, resadd_s100, fc2_k4096, fc1_panel_bf16, dwconv, silu, dwconv_silu, dwconv_silu_t,
        });
        rl
    }

    /// True when the resident on-NPU LN->fc1 seam is usable (modal resident + ctxln/affcast xclbins
    /// present). Lets `feed_forward` default to the resident path and fall back to host otherwise.
    pub fn resident_ff_available(&self) -> bool {
        self.modal && self.resident_ln().is_some()
    }

    /// Capability accessors: the resident rail's baked contraction/padding/inner dims. A K=768
    /// consumer (BERT / Whisper / ESM post-norm FFN) capability-GATES on these before dispatching --
    /// every resident brick asserts `KRES`, so a mismatched hidden dim panics; the consumer must
    /// confirm `resident_kres()` matches its `D` first (see `resident_ffn_nonorm`).
    ///
    /// These return the compile-time Parakeet defaults (KRES=1024, PAD_M=512, DFF=4096). Turning the
    /// K=768 rail on is a device-session step that makes KRES/PAD_M/DFF real fields set from the
    /// loaded xclbin name; keeping them consts here preserves Parakeet's path byte-for-byte (see the
    /// DEVICE-GATED notes on `resident_ffn_nonorm`).
    pub fn resident_kres(&self) -> usize { KRES }
    pub fn resident_pad_m(&self) -> usize { PAD_M }
    pub fn resident_dff(&self) -> usize { DFF }

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
        match rl.lnaffcast.as_ref() {
            // ONE dispatch: (bo_x, gamma|beta) -> bo_bf16. bo_ln is never materialized.
            Some(f) => {
                modal_site("f.kern#1");
                { let _dt = self.dtimer(); f.kern.run_matmul8(3, &f.instr, f.n, &rl.bo_x, &rl.bo_gb, &rl.bo_bf16, &f.dummy_tmp, &f.dummy_tr).unwrap(); }
                self.stats.borrow_mut().dispatches += 1;
            }
            None => {
                // (1) ctxLN: bo_x -> bo_ln  (NO sync back -- stays device-resident)
                modal_site("rl.ln_kern#1");
                { let _dt = self.dtimer(); rl.ln_kern.run_matmul8(3, &rl.ln_instr, rl.ln_n, &rl.bo_x, &rl.bo_ln, &rl.ln_c, &rl.ln_tmp, &rl.ln_tr).unwrap(); }
                // (2) affine_cast: (bo_ln * gamma + beta) -> bo_bf16  (device-side, no host round-trip)
                modal_site("rl.ac_kern#1");
                { let _dt = self.dtimer(); rl.ac_kern.run_matmul8(3, &rl.ac_instr, rl.ac_n, &rl.bo_ln, &rl.bo_gb, &rl.bo_bf16, &rl.ac_tmp, &rl.ac_tr).unwrap(); }
                self.stats.borrow_mut().dispatches += 2;
            }
        }
        rl
    }

    /// [`Self::ln_affine_cast`] forced onto the two-dispatch chain, so `bo_ln` (the f32 LN output)
    /// IS materialized. Only for callers that read it back -- the fused kernel skips it.
    fn ln_affine_cast_chained(&self, x: &Array2<f32>, gamma: &[f32], beta: &[f32]) -> Rc<ResidentLn> {
        let rl = self.resident_ln().expect("ln_affine_cast_chained without resident_ff_available()");
        let x_std = x.as_standard_layout();
        let t = x.nrows();
        let mut buf = vec![0f32; PAD_M * KRES];
        buf[..t * KRES].copy_from_slice(&x_std.as_slice().unwrap()[..t * KRES]);
        rl.bo_x.write_bytes(f32_bytes(&buf)).unwrap();
        rl.bo_x.sync_to_device().unwrap();
        let mut gb = vec![0f32; 2 * KRES];
        gb[..KRES].copy_from_slice(gamma);
        gb[KRES..].copy_from_slice(beta);
        rl.bo_gb.write_bytes(f32_bytes(&gb)).unwrap();
        rl.bo_gb.sync_to_device().unwrap();
        modal_site("rl.ln_kern#2");
        { let _dt = self.dtimer(); rl.ln_kern.run_matmul8(3, &rl.ln_instr, rl.ln_n, &rl.bo_x, &rl.bo_ln, &rl.ln_c, &rl.ln_tmp, &rl.ln_tr).unwrap(); }
        modal_site("rl.ac_kern#2");
        { let _dt = self.dtimer(); rl.ac_kern.run_matmul8(3, &rl.ac_instr, rl.ac_n, &rl.bo_ln, &rl.bo_gb, &rl.bo_bf16, &rl.ac_tmp, &rl.ac_tr).unwrap(); }
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
        match rl.lnaffcast.as_ref() {
            // ONE dispatch, DEVICE-IN: (a_bo, gamma|beta) -> bo_bf16.
            Some(f) => {
                modal_site("f.kern#2");
                { let _dt = self.dtimer(); f.kern.run_matmul8(3, &f.instr, f.n, a_bo, &rl.bo_gb, &rl.bo_bf16, &f.dummy_tmp, &f.dummy_tr).unwrap(); }
                self.stats.borrow_mut().dispatches += 1;
            }
            None => {
                // (1) ctxLN: a_bo -> bo_ln  (DEVICE-IN: no host write of x; stays device-resident)
                modal_site("rl.ln_kern#3");
                { let _dt = self.dtimer(); rl.ln_kern.run_matmul8(3, &rl.ln_instr, rl.ln_n, a_bo, &rl.bo_ln, &rl.ln_c, &rl.ln_tmp, &rl.ln_tr).unwrap(); }
                // (2) affine_cast: (bo_ln * gamma + beta) -> bo_bf16  (device-side)
                modal_site("rl.ac_kern#3");
                { let _dt = self.dtimer(); rl.ac_kern.run_matmul8(3, &rl.ac_instr, rl.ac_n, &rl.bo_ln, &rl.bo_gb, &rl.bo_bf16, &rl.ac_tmp, &rl.ac_tr).unwrap(); }
                self.stats.borrow_mut().dispatches += 2;
            }
        }
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
    /// resident-ln seam + the fc2-accumulate (acc_add) + BOTH residual scales (resadd_s050 for the
    /// Macaron residual AND resadd_s100 for the MHSA residual). block()'s FUSED_BLOCK branch
    /// hard-`.expect()`s the s100 residual mid-block, so it must be gated here or a tree with only
    /// s050 built would green-light the fused path and then panic after FF1 already dispatched.
    pub fn resident_fused_available(&self) -> bool {
        if !self.modal {
            return false;
        }
        match self.resident_ln() {
            Some(rl) => rl.acc_add.is_some() && rl.resadd_s050.is_some() && rl.resadd_s100.is_some(),
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
        assert!(m <= PAD_M, "readback_stream: m={m} exceeds PAD_M={PAD_M}");
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
        assert!(m <= PAD_M, "dispatch_with_a: m={m} exceeds PAD_M={PAD_M}");
        let st = self.stream(n, if silu { Act::Silu } else { Act::Identity });
        modal_site("kern#1");
        { let _dt = self.dtimer(); self.kern.run_matmul8(3, &st.instr, st.n_instr, a_bo, wbo, &st.bo_c, &self.bo_tmp, &self.bo_tr).unwrap(); }
        self.stats.borrow_mut().dispatches += 1;
        let t_sync = Instant::now();
        st.bo_c.sync_from_device().unwrap();
        let sync_s = t_sync.elapsed().as_secs_f64();
        let mut cb = vec![0u8; m * n * 4];
        let t_read = Instant::now();
        st.bo_c.read_bytes(&mut cb).unwrap();
        let read_s = t_read.elapsed().as_secs_f64();
        let t_dec = Instant::now();
        let mut out = Array2::<f32>::zeros((m, n));
        for r in 0..m {
            for c in 0..n {
                let off = (r * n + c) * 4;
                out[[r, c]] = f32::from_le_bytes([cb[off], cb[off + 1], cb[off + 2], cb[off + 3]]);
            }
        }
        {
            let mut s = self.stats.borrow_mut();
            s.rb_sync_s += sync_s;
            s.rb_read_s += read_s;
            s.rb_decode_s += t_dec.elapsed().as_secs_f64();
            s.rb_decode_elems += m * n;
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

    /// Width of one modal C tile (`EPI_N`), parsed from the resident's tile spec ("64x32x128" -> 128).
    /// The GLU value/gate split is defined in units of it, so a tile change with a stale literal here
    /// would pair the wrong elements silently -- derive it, do not write 128.
    fn epi_n(&self) -> usize {
        self.tile
            .rsplit('x')
            .next()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| panic!("resident tile {:?} has no parsable N", self.tile))
    }

    /// The W1 column permutation the GLU epilogue's pairing requires. pw1's natural
    /// `[a(0..D) | g(0..D)]` becomes, per C tile, `[64 values | their 64 gate partners]` -- which is
    /// what puts each element's partner at a constant half-row-block offset in mm.cc's blocked C
    /// order, so the epilogue can walk flat runs instead of being layout-aware.
    ///
    /// Value `c` lands in tile `c/half` at position `c%half`; its gate lands `half` further on. This
    /// is the exact inverse of `glu_strided_read` -- the two only make sense as a pair.
    fn permute_w1_glu(&self, w: ArrayView2<f32>) -> Array2<f32> {
        let (k, n2) = w.dim();
        let tile_n = self.epi_n();
        assert_eq!(n2 % tile_n, 0, "pw1 N={n2} is not a whole number of {tile_n}-wide C tiles");
        let half = tile_n / 2;
        let d = n2 / 2;
        assert_eq!(d % half, 0, "pw1 value width {d} does not split into {half}-wide tile halves");
        let mut p = Array2::<f32>::zeros((k, n2));
        for c in 0..d {
            let base = (c / half) * tile_n + (c % half);
            for r in 0..k {
                p[[r, base]] = w[[r, c]]; // value column
                p[[r, base + half]] = w[[r, d + c]]; // its gate partner
            }
        }
        p
    }

    /// Read the folded GLU result out of the FULL-width C drain. The epilogue writes the gated value
    /// into the value half of each tile and leaves the gate half holding the raw accumulator, and the
    /// drain tap was deliberately left full width (narrowing it would resize `C_l1_ty` into a new
    /// array program, costing the insts-only property). So the live output is `half` of every
    /// `tile_n` columns. Inverse of `permute_w1_glu`.
    fn glu_strided_read(&self, bo: &Bo, m: usize, d: usize) -> Array2<f32> {
        let tile_n = self.epi_n();
        let half = tile_n / 2;
        let n2 = 2 * d;
        bo.sync_from_device().unwrap();
        let mut cb = vec![0u8; m * n2 * 4]; // first m rows are contiguous, as in the two-dispatch read
        bo.read_bytes(&mut cb).unwrap();
        let mut out = Array2::<f32>::zeros((m, d));
        for r in 0..m {
            for c in 0..d {
                let off = (r * n2 + (c / half) * tile_n + (c % half)) * 4;
                out[[r, c]] = f32::from_le_bytes([cb[off], cb[off + 1], cb[off + 2], cb[off + 3]]);
            }
        }
        out
    }

    /// `PARAKEET_FOLD_GLU=1` body, shared by the host-in and device-in conv fronts (they differ only
    /// in how the LN'd bf16 A was produced). One modal GEMM whose epilogue applies `a * sigmoid(g)`.
    fn conv_pw1_glu_folded<F: FnOnce() -> Array2<f32>>(
        &self, a_bf16: &Bo, m: usize, make_w1: F, id: &str,
    ) -> Option<Array2<f32>> {
        assert!(
            self.glu_epi,
            "PARAKEET_FOLD_GLU=1 needs a resident whose epilogue carries the GLU branch, and this \
             one does not. Pair it with NPU_RESIDENT_XCLBIN=<...modalglu...xclbin>: on a pre-GLU \
             resident rtp[0]=3 falls through to identity and silently returns pw1's raw [a|g]."
        );
        let n2 = 2 * KRES;
        // The permuted weight gets its OWN cache id: `{blk}.pw1` is shared with resident_ff1_fc1's
        // unpermuted path, and handing that path a permuted W1 would corrupt the fallback arm.
        let pid = format!("{id}.gluperm");
        let cached = self.wcache.borrow().get(&pid).cloned();
        let wbo = if let Some(bo) = cached {
            bo
        } else {
            let w = make_w1();
            assert_eq!(w.nrows(), KRES, "pw1 W nrows {} != {KRES}", w.nrows());
            assert_eq!(w.ncols(), n2, "pw1 W ncols {} != {n2}", w.ncols());
            let wp = self.permute_w1_glu(w.view());
            self.weight_bo(&pid, wp.view())
        };
        let st = self.stream(n2, Act::Glu);
        modal_site("glu.fold");
        {
            let _dt = self.dtimer();
            self.kern
                .run_matmul8(3, &st.instr, st.n_instr, a_bf16, &wbo, &st.bo_c, &self.bo_tmp, &self.bo_tr)
                .unwrap();
        }
        self.stats.borrow_mut().dispatches += 1; // pw1 + gate in one
        Some(self.glu_strided_read(&st.bo_c, m, KRES))
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
        // Only the two-dispatch arm needs the standalone GLU xclbin -- the fold applies the gate in
        // pw1's own epilogue. Checked before the LN so a missing xclbin still costs no dispatch.
        if !fold_glu() && rl.glu.is_none() {
            return None; // caller falls back to resident LN->pw1 + host GLU
        }
        self.stats.borrow_mut().calls += 1;
        let m = x.nrows();
        let n2 = 2 * KRES; // pw1 output width 2D
        // LN + affine cast -> bo_bf16 = affine_LN(x) bf16 [PAD_M, KRES] (device).
        let rlc = self.ln_affine_cast(x, gamma, beta);
        if fold_glu() {
            return self.conv_pw1_glu_folded(&rlc.bo_bf16, m, make_w1, id);
        }
        let glu = rl.glu.as_ref().unwrap(); // presence checked above
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
        let st = self.stream(n2, Act::Identity); // identity modal (no on-chip silu on pw1)
        modal_site("kern#2");
        { let _dt = self.dtimer(); self.kern.run_matmul8(3, &st.instr, st.n_instr, &rlc.bo_bf16, &wbo, &st.bo_c, &self.bo_tmp, &self.bo_tr).unwrap(); }
        // GLU: st.bo_c [PAD_M,2D] f32 (A/g3) -> glu.bo_out [PAD_M,D] f32 (B/g4), device-side.
        modal_site("glu.kern#1");
        { let _dt = self.dtimer(); glu.kern.run_matmul8(3, &glu.instr, glu.n, &st.bo_c, &glu.bo_out, &glu.dummy_c, &glu.dummy_tmp, &glu.dummy_tr).unwrap(); }
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
        if !fold_glu() && rl.glu.is_none() {
            return None; // see resident_conv_pw1_glu
        }
        self.stats.borrow_mut().calls += 1;
        let n2 = 2 * KRES; // pw1 output width 2D
        let rlc = self.ln_affine_cast_dev(a_bo, gamma, beta); // device-in LN
        if fold_glu() {
            return self.conv_pw1_glu_folded(&rlc.bo_bf16, m, make_w1, id);
        }
        let glu = rl.glu.as_ref().unwrap(); // presence checked above
        let cached = self.wcache.borrow().get(id).cloned();
        let wbo = if let Some(bo) = cached {
            bo
        } else {
            let w = make_w1();
            assert_eq!(w.nrows(), KRES, "pw1 W nrows {} != {KRES}", w.nrows());
            assert_eq!(w.ncols(), n2, "pw1 W ncols {} != {n2}", w.ncols());
            self.weight_bo(id, w.view())
        };
        let st = self.stream(n2, Act::Identity);
        modal_site("kern#3");
        {
            let _p = crate::prof::phase::PhaseScope::new("cf_pw1", crate::prof::phase::Bucket::Npu);
            let _dt = self.dtimer();
            self.kern.run_matmul8(3, &st.instr, st.n_instr, &rlc.bo_bf16, &wbo, &st.bo_c, &self.bo_tmp, &self.bo_tr).unwrap();
        }
        modal_site("glu.kern#2");
        {
            let _p = crate::prof::phase::PhaseScope::new("cf_glu", crate::prof::phase::Bucket::Npu);
            let _dt = self.dtimer();
            glu.kern.run_matmul8(3, &glu.instr, glu.n, &st.bo_c, &glu.bo_out, &glu.dummy_c, &glu.dummy_tmp, &glu.dummy_tr).unwrap();
        }
        self.stats.borrow_mut().dispatches += 2; // pw1 + glu
        // Device->host readback of the gated [m,KRES] stream, plus a SCALAR f32 decode loop. The
        // decode is host CPU, not DMA, and at m*KRES elements it is not obviously small -- scope it
        // separately from the sync so the two do not hide in each other.
        let _p = crate::prof::phase::PhaseScope::new("cf_readback", crate::prof::phase::Bucket::Marshal);
        glu.bo_out.sync_from_device().unwrap();
        let mut cb = vec![0u8; m * KRES * 4];
        glu.bo_out.read_bytes(&mut cb).unwrap();
        drop(_p);
        let _p2 = crate::prof::phase::PhaseScope::new("cf_decode", crate::prof::phase::Bucket::Host);
        let mut out = Array2::<f32>::zeros((m, KRES));
        for r in 0..m {
            for c in 0..KRES {
                let off = (r * KRES + c) * 4;
                out[[r, c]] = f32::from_le_bytes([cb[off], cb[off + 1], cb[off + 2], cb[off + 3]]);
            }
        }
        Some(out)
    }

    /// Host-in -> host-out `A[m,DFF] @ W[DFF,KRES]` on the RESIDENT xclbin, for the subsample stem's
    /// output projection. Same K=4096 contraction the one-dispatch fc2 already runs, so it needs no
    /// new kernel and -- because it dispatches on the resident -- no new hw_context, which matters at
    /// 14/16 used.
    ///
    /// Requires the krtp resident: at `k != KRES` the k-loop bound comes from rtp[1], and a BAKED
    /// resident would silently contract over its own K instead. Returns `None` rather than dispatching
    /// wrong, so the caller keeps the host path.
    ///
    /// A is HOST-fed and therefore row-major, which is why this asks `stream_k_ex` for the row-major
    /// tap explicitly instead of letting it derive fc1's panel-major layout.
    pub fn matmul_k4096<F: FnOnce() -> Array2<f32>>(&self, a: &Array2<f32>, make_w: F, id: &str) -> Option<Array2<f32>> {
        if !self.krtp || !self.modal {
            return None;
        }
        let m = a.nrows();
        assert_eq!(a.ncols(), DFF, "matmul_k4096 needs K=DFF={DFF}");
        assert!(m <= PAD_M, "m={m} exceeds PAD_M={PAD_M}");
        self.stats.borrow_mut().calls += 1;
        let cached = self.wcache.borrow().get(id).cloned();
        let wbo = if let Some(bo) = cached {
            bo
        } else {
            let w = make_w();
            assert_eq!(w.dim(), (DFF, KRES), "matmul_k4096 W dim");
            self.weight_bo(id, w.view())
        };
        // A needs PAD_M x DFF bf16 = 4 MB, four times `bo_a`'s KRES width, so it gets its own buffer.
        // Lazily, so a run that never takes this path does not pay for it.
        let bo_a4 = {
            let mut slot = self.bo_a4.borrow_mut();
            if slot.is_none() {
                let g3 = self.kern.group_id(3).unwrap();
                *slot = Some(Rc::new(self.dev.alloc_bo(&self.kern, PAD_M * DFF * 2, FLAG_HOST_ONLY, g3).unwrap()));
            }
            slot.as_ref().unwrap().clone()
        };
        let a_std = a.as_standard_layout();
        let mut bits = vec![0u16; PAD_M * DFF]; // rows m..PAD_M stay zero
        npu_xrt::pack_f32_to_bf16(&a_std.as_slice().unwrap()[..m * DFF], &mut bits[..m * DFF]);
        bo_a4.write_bytes(u16_bytes(&bits)).unwrap();
        bo_a4.sync_to_device().unwrap();
        let st = self.stream_k_ex(DFF, KRES, Act::Identity, false);
        modal_site("kern#ss_out");
        {
            let _dt = self.dtimer();
            self.kern
                .run_matmul8(3, &st.instr, st.n_instr, &bo_a4, &wbo, &st.bo_c, &self.bo_tmp, &self.bo_tr)
                .unwrap();
        }
        self.stats.borrow_mut().dispatches += 1;
        st.bo_c.sync_from_device().unwrap();
        let mut cb = vec![0u8; m * KRES * 4];
        st.bo_c.read_bytes(&mut cb).unwrap();
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
        let st = self.stream(n, Act::Identity);
        modal_site("kern#4");
        { let _dt = self.dtimer(); self.kern.run_matmul8(3, &st.instr, st.n_instr, &self.bo_a, &wbo, &out, &self.bo_tmp, &self.bo_tr).unwrap(); }
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
        modal_site("dw.kern#1");
        { let _dt = self.dtimer(); dw.kern.run_matmul8(3, &dw.instr, dw.n, &dw.bo_in, &dw.bo_w, &dw.bo_out, &dw.dummy_tmp, &dw.dummy_tr).unwrap(); }
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
        modal_site("s.kern#1");
        { let _dt = self.dtimer(); s.kern.run_matmul8(3, &s.instr, s.n, &s.bo_in, &s.bo_out, &s.dummy_tmp, &s.dummy_ctrl, &s.dummy_tr).unwrap(); }
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
        modal_site("ds.kern#1");
        { let _dt = self.dtimer(); ds.kern.run_matmul8(3, &ds.instr, ds.n, &ds.bo_in, &ds.bo_w, &ds.bo_out, &ds.dummy_tmp, &ds.dummy_tr).unwrap(); }
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
        modal_site("ds.kern#2");
        { let _dt = self.dtimer(); ds.kern.run_matmul8(3, &ds.instr, ds.n, &ds.bo_in, &ds.bo_w, &ds.bo_out, &ds.dummy_tmp, &ds.dummy_tr).unwrap(); }
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
    /// True when the one-dispatch K=DFF fc2 collapse is enabled (opt-in `PARAKEET_FC2_ONEDISPATCH`).
    /// Unlike `fc2_k4096_on` this is tested on the DEFAULT path, not inside a `None` arm the default
    /// never takes -- see the warning in `ffn_dev_accum`.
    fn fc2_onedispatch_on(&self) -> bool {
        std::env::var("PARAKEET_FC2_ONEDISPATCH").map(|v| v != "0").unwrap_or(false)
    }

    /// Output BO for the one-dispatch fc2, lazily allocated against the RESIDENT kernel's C group.
    /// Deliberately not the stream's `bo_c` (shared by every dispatch on that key) and not the
    /// accadd scratch (allocated against another kernel's group, whose bank is what makes a
    /// mis-homed BO corrupt silently).
    fn fc2_out_bo(&self) -> Rc<Bo> {
        if let Some(bo) = self.fc2_out.borrow().as_ref() {
            return bo.clone();
        }
        let g5 = self.kern.group_id(5).unwrap();
        let bo = Rc::new(self.dev.alloc_bo(&self.kern, PAD_M * KRES * 4, FLAG_HOST_ONLY, g5).unwrap());
        *self.fc2_out.borrow_mut() = Some(bo.clone());
        bo
    }

    /// True when the one-dispatch K=DFF fc2 collapse is enabled (opt-in `PARAKEET_FC2_K4096`).
    fn fc2_k4096_on(&self) -> bool {
        std::env::var("PARAKEET_FC2_K4096").map(|v| v != "0").unwrap_or(false)
    }

    /// DEFAULT ON since 2026-07-28. `PARAKEET_FC1_PACK_IN_DRAIN=0` returns to the fc1+deint pair.
    fn fc1_pack_in_drain_on(&self) -> bool {
        std::env::var("PARAKEET_FC1_PACK_IN_DRAIN").map(|v| v != "0").unwrap_or(true)
    }

    /// fc1 -> the chunk-major bf16 buffer the fc2 K-split reads, as ONE dispatch instead of two.
    ///
    /// `Some(bo)` means the fold ran: the GEMM drained chunk-major bf16 itself and no deint was
    /// dispatched. `None` means the caller must run the shipped fc1 + deint pair -- either the flag
    /// is off or the xclbin was not built. The returned BO holds exactly what `bo_deint` would have,
    /// so callers only need to swap which buffer they sub-slice.
    fn fc1_pack_in_drain<'a>(&self, rl: &'a ResidentLn, w1: &Bo) -> Option<&'a Bo> {
        if !self.fc1_pack_in_drain_on() {
            return None;
        }
        let o = rl.fc1_panel_bf16.as_ref()?;
        // This is the DEFAULT fc1 path, so the leaf breakdown has to charge it here -- the
        // fc1/deint timers on the fallback branch never run while the fold is on, and reported 0.
        // Timed outside the dtimer scope: that guard borrows self.stats at drop.
        let t_fc1 = Instant::now();
        {
            let _p = crate::prof::phase::PhaseScope::new("ffn_fc1", crate::prof::phase::Bucket::Npu);
            let _dt = self.dtimer();
            o.kern
                .run_matmul8(3, &o.instr, o.n, &rl.bo_bf16, w1, &o.bo_out, &o.dummy_tmp, &o.dummy_tr)
                .unwrap();
        }
        let fc1_s = t_fc1.elapsed().as_secs_f64();
        {
            let mut s = self.stats.borrow_mut();
            s.dispatches += 1; // fc1 only -- the deint is folded into its drain
            s.ffn_fc1_s += fc1_s; // includes the folded deint, which no longer has its own dispatch
        }
        Some(&o.bo_out)
    }

    /// Shared one-dispatch K=DFF fc2: cast the fc1 output (`fc1_out` f32 [PAD_M,DFF]) to bf16 row-major,
    /// then ONE K=DFF modal GEMM (internal L1 K-accum over DFF) with the full fc2 weight -> f32
    /// [PAD_M,KRES] device BO. Counts 2 dispatches (cast + modal); the caller counts fc1. Full fc2
    /// weight cached under "{id2}.full". Collapses the deint + 4x K=1024 GEMM + 4x acc_add.
    fn fc2_k4096_dev<F2: FnOnce() -> Array2<f32>>(&self, k4: &Fc2K4096, fc1_out: &Bo, make_w2: F2, id2: &str) -> Rc<Bo> {
        modal_site("k4.cast_kern#1");
        { let _dt = self.dtimer(); k4.cast_kern.run_matmul8(3, &k4.cast_instr, k4.cast_n, fc1_out, &k4.cast_out, &k4.cast_dc, &k4.cast_dt, &k4.cast_dr).unwrap(); }
        let wid = format!("{id2}.full");
        let cached = self.wcache.borrow().get(&wid).cloned();
        let w2f = if let Some(bo) = cached {
            bo
        } else {
            let w = make_w2();
            assert_eq!(w.dim(), (DFF, KRES), "fc2 W2 dim");
            self.weight_bo(&wid, w.view())
        };
        modal_site("k4.mm_kern#1");
        { let _dt = self.dtimer(); k4.mm_kern.run_matmul8(3, &k4.mm_instr, k4.mm_n, &k4.cast_out, &w2f, &k4.mm_c, &self.bo_tmp, &self.bo_tr).unwrap(); }
        self.stats.borrow_mut().dispatches += 2; // cast + K=DFF modal
        k4.mm_c.clone()
    }

    pub fn resident_ffn<F1: FnOnce() -> Array2<f32>, F2: FnOnce() -> Array2<f32>>(
        &self, x: &Array2<f32>, gamma: &[f32], beta: &[f32],
        make_w1: F1, id1: &str, make_w2: F2, id2: &str,
    ) -> Array2<f32> {
        self.stats.borrow_mut().calls += 1;
        let m = x.nrows();
        let t_ln = Instant::now();
        let rl = self.ln_affine_cast(x, gamma, beta); // bo_bf16 = affine_LN bf16 [PAD_M,KRES]
        let ln_s = t_ln.elapsed().as_secs_f64();
        self.stats.borrow_mut().ffn_ln_s += ln_s;
        // fc1: modal, A=bo_bf16, W1, on-chip SiLU -> st1.bo_c (f32 [PAD_M,DFF]) -- stays DEVICE
        let t_wp = Instant::now();
        let w1 = {
            let c = self.wcache.borrow().get(id1).cloned();
            c.unwrap_or_else(|| {
                let w = make_w1();
                assert_eq!(w.dim(), (KRES, DFF), "fc1 W1 dim");
                self.weight_bo(id1, w.view())
            })
        };
        self.stats.borrow_mut().ffn_weight_prep_s += t_wp.elapsed().as_secs_f64();
        // fc1 -> chunk-major bf16. The fold (opt-in) does it in ONE dispatch; otherwise the shipped
        // fc1 + deint pair. `a_chunks` is the buffer the fc2 K-split sub-slices either way.
        let a_chunks: &Bo = match self.fc1_pack_in_drain(&rl, &w1) {
            Some(bo) => bo,
            None => {
                let st1 = self.stream(DFF, Act::Silu); // fc1 on-chip SiLU iff modal (plain resident ignores it)
                let t_fc1 = Instant::now();
                modal_site("kern#5");
                self.kern.run_matmul8(3, &st1.instr, st1.n_instr, &rl.bo_bf16, &w1, &st1.bo_c, &self.bo_tmp, &self.bo_tr).unwrap();
                let fc1_s = t_fc1.elapsed().as_secs_f64();
                {
                    let mut s = self.stats.borrow_mut();
                    s.dispatch_s += fc1_s;
                    s.ffn_fc1_s += fc1_s;
                }
                // ONE-DISPATCH K=DFF fc2 (opt-in): cast@DFF -> K=DFF modal -> readback to host [m,KRES].
                if self.fc2_k4096_on() {
                    if let Some(k4) = rl.fc2_k4096.as_ref() {
                        self.stats.borrow_mut().dispatches += 1; // fc1
                        let bo = self.fc2_k4096_dev(k4, &st1.bo_c, make_w2, id2);
                        // HOST from here: sync, copy out, decode f32. Inside the ff_resident phase
                        // scope this was charged to Bucket::Npu.
                        let t_rb = Instant::now();
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
                        self.stats.borrow_mut().ffn_readback_s += t_rb.elapsed().as_secs_f64();
                        return out;
                    }
                }
                // deinterleave+cast: st1.bo_c (f32 [PAD_M,DFF]) -> rl.bo_deint (bf16 [parts,PAD_M,KRES]
                // chunk-major), device-side. One dispatch (chunk-major drain TAP). NOTE: this n-D output DMA
                // HANGS ("run did not complete") when the deint is a co-resident hw-context alongside the
                // modal (it works standalone) -- a multi-context n-D-DMA toolchain issue; see the debug note.
                let t_deint = Instant::now();
                modal_site("rl.deint_kern#1");
                rl.deint_kern.run_matmul8(3, &rl.deint_instr, rl.deint_n, &st1.bo_c, &rl.bo_deint, &rl.deint_c, &rl.deint_tmp, &rl.deint_tr).unwrap();
                let deint_s = t_deint.elapsed().as_secs_f64();
                {
                    let mut s = self.stats.borrow_mut();
                    s.dispatch_s += deint_s;
                    s.ffn_deint_s += deint_s;
                }
                self.stats.borrow_mut().dispatches += 2; // fc1 + deint
                &rl.bo_deint
            }
        };
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
        let t_fc2 = Instant::now();
        for c in 0..parts {
            let chunk = a_chunks.sub(c * chunk_bytes, chunk_bytes).unwrap();
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
        self.stats.borrow_mut().ffn_fc2_s += t_fc2.elapsed().as_secs_f64();
        acc
    }

    /// bf16-checkpoint sibling of [`Self::resident_ffn`]: `bits1`/`bits2` are pre-packed bf16 straight
    /// from a bf16-baked `NPU_WEIGHTS_CHECKPOINT` (fc1 `[KRES,DFF]`, fc2 `[DFF,KRES]`, both verbatim
    /// layout), so every weight-BO build on a cache miss skips the host f32->bf16 pack entirely.
    /// Same device-side LN->fc1->deint->fc2(K-split) dataflow as `resident_ffn`; only the weight
    /// source differs.
    pub fn resident_ffn_bf16(
        &self, x: &Array2<f32>, gamma: &[f32], beta: &[f32],
        id1: &str, k1: usize, n1: usize, bits1: &[u16],
        id2: &str, k2: usize, n2: usize, bits2: &[u16],
    ) -> Array2<f32> {
        self.stats.borrow_mut().calls += 1;
        let m = x.nrows();
        assert_eq!((k1, n1), (KRES, DFF), "fc1 W1 dim");
        assert_eq!((k2, n2), (DFF, KRES), "fc2 W2 dim");
        let rl = self.ln_affine_cast(x, gamma, beta); // bo_bf16 = affine_LN bf16 [PAD_M,KRES]
        let w1 = {
            let c = self.wcache.borrow().get(id1).cloned();
            c.unwrap_or_else(|| self.weight_bo_bf16(id1, k1, n1, bits1))
        };
        // Ported to the Act enum that `stream()` took on in the k768 rail merge; matches the f32
        // sibling `resident_ffn` exactly (was `self.modal` under the old bool API).
        let a_chunks: &Bo = match self.fc1_pack_in_drain(&rl, &w1) {
            Some(bo) => bo,
            None => {
                let st1 = self.stream(DFF, Act::Silu); // fc1 on-chip SiLU iff modal (plain resident ignores it)
                let t_fc1 = Instant::now();
                modal_site("kern#6");
                self.kern.run_matmul8(3, &st1.instr, st1.n_instr, &rl.bo_bf16, &w1, &st1.bo_c, &self.bo_tmp, &self.bo_tr).unwrap();
                let fc1_s = t_fc1.elapsed().as_secs_f64();
                {
                    let mut s = self.stats.borrow_mut();
                    s.dispatch_s += fc1_s;
                    s.ffn_fc1_s += fc1_s;
                }
                let t_deint = Instant::now();
                modal_site("rl.deint_kern#2");
                rl.deint_kern.run_matmul8(3, &rl.deint_instr, rl.deint_n, &st1.bo_c, &rl.bo_deint, &rl.deint_c, &rl.deint_tmp, &rl.deint_tr).unwrap();
                let deint_s = t_deint.elapsed().as_secs_f64();
                {
                    let mut s = self.stats.borrow_mut();
                    s.dispatch_s += deint_s;
                    s.ffn_deint_s += deint_s;
                }
                self.stats.borrow_mut().dispatches += 2; // fc1 + deint
                &rl.bo_deint
            }
        };
        let parts = DFF / KRES;
        let chunk_bytes = PAD_M * KRES * 2;
        let mut acc = Array2::<f32>::zeros((m, KRES));
        for c in 0..parts {
            let chunk = a_chunks.sub(c * chunk_bytes, chunk_bytes).unwrap();
            let sid = format!("{id2}.{c}");
            let w2c = {
                let cc = self.wcache.borrow().get(&sid).cloned();
                cc.unwrap_or_else(|| {
                    // bits2 is row-major [DFF, KRES]; a KRES-row chunk is a contiguous slice.
                    let part_bits = &bits2[c * KRES * KRES..(c + 1) * KRES * KRES];
                    self.weight_bo_bf16(&sid, KRES, KRES, part_bits)
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
        let a_chunks: &Bo = match self.fc1_pack_in_drain(rl, &w1) {
            Some(bo) => bo,
            None => {
                let st1 = self.stream(DFF, Act::Silu); // fc1 on-chip SiLU iff modal (plain resident ignores it)
                let t_fc1 = Instant::now();
                modal_site("kern#7");
                self.kern.run_matmul8(3, &st1.instr, st1.n_instr, &rl.bo_bf16, &w1, &st1.bo_c, &self.bo_tmp, &self.bo_tr).unwrap();
                let fc1_s = t_fc1.elapsed().as_secs_f64();
                {
                    let mut s = self.stats.borrow_mut();
                    s.dispatch_s += fc1_s;
                    s.ffn_fc1_s += fc1_s;
                }
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
                let t_deint = Instant::now();
                modal_site("rl.deint_kern#3");
                rl.deint_kern.run_matmul8(3, &rl.deint_instr, rl.deint_n, &st1.bo_c, &rl.bo_deint, &rl.deint_c, &rl.deint_tmp, &rl.deint_tr).unwrap();
                let deint_s = t_deint.elapsed().as_secs_f64();
                {
                    let mut s = self.stats.borrow_mut();
                    s.dispatch_s += deint_s;
                    s.ffn_deint_s += deint_s;
                }
                self.stats.borrow_mut().dispatches += 2; // fc1 + deint
                &rl.bo_deint
            }
        };
        // ONE-DISPATCH fc2 (opt-in `PARAKEET_FC2_ONEDISPATCH`): contract all DFF of K in a SINGLE
        // modal dispatch instead of 4 K=KRES partials + 4 acc_add. The accumulator already lives
        // on-core across the k-loop; the K-split is what threw it away and made acc_add necessary.
        // Worth -336 of 816 dispatches/clip and deletes the accadd program (one hw_context back).
        //
        // Needs the krtp resident AND krtp streams together -- see `stream_k`. A is read straight out
        // of fc1's panel-major bf16 drain by the stream's own A tap (`--a-panel-width`), so nothing
        // repacks; that tap fits the shim's 4 dims only because N=KRES makes its outer repeat
        // degenerate, which is asserted in the generator rather than left to hold by luck.
        //
        // NOT bit-identical to the 4-partial path, and that is expected: on-core accumulation rounds
        // the same products in a different order than four separately-rounded host-order partials
        // (`acc_add.cc` -- f32 add is deterministic, not exact). Gate on the transcript, not bits.
        //
        // Tested HERE, on the path the default actually takes -- unlike `fc2_k4096_on` below, which
        // sits inside the `None` arm of the `fc1_pack_in_drain` match and is therefore inert under
        // the default fold. Warn rather than let that stay silent:
        if self.fc2_k4096_on() && self.fc1_pack_in_drain_on() {
            eprintln!("[npu] PARAKEET_FC2_K4096 is set but has NO EFFECT: it is only reachable with \
                       PARAKEET_FC1_PACK_IN_DRAIN=0. Did you mean PARAKEET_FC2_ONEDISPATCH=1?");
        }
        if self.fc2_onedispatch_on() {
            let wid = format!("{id2}.full");
            let cached = self.wcache.borrow().get(&wid).cloned();
            let w2f = cached.unwrap_or_else(|| {
                let w = make_w2();
                assert_eq!(w.dim(), (DFF, KRES), "fc2 W2 dim");
                self.weight_bo(&wid, w.view())
            });
            let out = self.fc2_out_bo();
            let st = self.stream_k(DFF, KRES, Act::Identity);
            modal_site("kern#8.onedispatch");
            {
                let _dt = self.dtimer();
                self.kern
                    .run_matmul8(3, &st.instr, st.n_instr, a_chunks, &w2f, &out, &self.bo_tmp, &self.bo_tr)
                    .unwrap();
            }
            self.stats.borrow_mut().dispatches += 1;
            return out;
        }
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
        let st = self.stream(KRES, Act::Identity);
        let mut cur = aa.acc0.clone();
        let mut nxt = aa.acc1.clone();
        for c in 0..parts {
            let chunk = a_chunks.sub(c * chunk_bytes, chunk_bytes).unwrap();
            let sid = format!("{id2}.{c}");
            let w2c = {
                let cc = self.wcache.borrow().get(&sid).cloned();
                cc.unwrap_or_else(|| {
                    let w = w2.as_ref().expect("w2 present on cache miss");
                    self.weight_bo(&sid, w.slice(s![c * KRES..(c + 1) * KRES, ..]))
                })
            };
            // modal identity GEMM: partial c -> st.bo_c (device, NO sync_from/read).
            modal_site("kern#8");
            {
                let _p = crate::prof::phase::PhaseScope::new("ffn_fc2_gemm", crate::prof::phase::Bucket::Npu);
                let _dt = self.dtimer();
                self.kern.run_matmul8(3, &st.instr, st.n_instr, &chunk, &w2c, &st.bo_c, &self.bo_tmp, &self.bo_tr).unwrap();
            }
            // accumulate on-chip: nxt = (c==0 ? zero : cur) + st.bo_c, then ping-pong.
            let a_in: &Bo = if c == 0 { &aa.zero } else { &cur };
            modal_site("aa.kern#1");
            {
                let _p = crate::prof::phase::PhaseScope::new("ffn_fc2_accadd", crate::prof::phase::Bucket::Npu);
                let _dt = self.dtimer();
                aa.kern.run_matmul8(3, &aa.instr, aa.n, a_in, &st.bo_c, &nxt, &aa.dummy_tmp, &aa.dummy_tr).unwrap();
            }
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

    /// K=768 post-norm resident FFN with NO leading LayerNorm (BERT / Whisper-encoder / ESM-2 share
    /// this shape family). Runs `x + fc2(act(fc1(x)))` on-chip and returns the device BO
    /// [PAD_M,768] f32 (the residual is added on-device via `resadd s100`; only the trailing
    /// post-norm LN stays host). The runbook Step-3 schedule:
    ///   fc1: A = cast(x) bf16 [T,768]; W1 K_aug=800 (768 real + one k=32 block that folds `b1` INTO
    ///        the matmul, since GELU needs the bias inside the activation); N=3072; `modalgelu`
    ///        -> h f32 [T,3072] device.
    ///   cast: h f32 [T,3072] -> bf16 (cast_512x3072).
    ///   fc2: K=3072 collapse; N=768; `modalid` -> y f32 [T,768] device; `b2` host-added (bias
    ///        outside the identity epilogue is exact -> no K-aug on fc2).
    ///   resadd_s100: x + y (scale=1.0 full residual) -> device [T,768].
    /// `make_w1`/`make_w2` lazily materialize W1 [768,3072] / W2 [3072,768] (id-cached); `b1`/`b2`
    /// are the fc1/fc2 biases; `act` = `Act::Gelu` for the shipping rail. Returns None on any
    /// non-K=768 rail, so the caller falls back to the host FFN (the shipped default).
    ///
    /// DEVICE-GATED (returns None today on the Parakeet KRES=1024 instance). Lighting this up is a
    /// DEVICE session, not more CPU work:
    ///   1. KRES/PAD_M/DFF become RailCfg FIELDS set from the loaded xclbin name so `resident_kres()`
    ///      can report 768 (kept as consts here to preserve Parakeet byte-for-byte -- runbook Step 2).
    ///   2. `open()` loads the K=768 rail (built by scripts/build_k768_gelu_rail.sh): fc1
    ///      512x800x3072 `modalgelu`, fc2 512x3072x768 `modalid`, cast_512x768 + cast_512x3072,
    ///      resadd_512x768_s100.
    ///   3. `stream()`'s `insts_512x1024x{n}` literals parameterize by pad_m/kres.
    ///   4. the K_aug=800 bias-fold packing of `b1` (one k=32 block appended to W1) + the N=768 fc2
    ///      tile n=96 (768 = 96*8, satisfies the epilogue `(m*n)%16==0`) -- shapes the device session
    ///      validates on rel-L2 vs host truth.
    /// With those in place the schedule above dispatches here; until then the capability gate short-
    /// circuits to None (host FFN) and the `resident_kres()==768` arm is `unimplemented!` so a future
    /// K=768 build cannot SILENTLY fall through to host (which would look like the rail ran but didn't).
    pub fn resident_ffn_nonorm<F1, F2>(
        &self, x_bo: &Bo, m: usize,
        make_w1: F1, b1: &[f32], id1: &str,
        make_w2: F2, b2: &[f32], id2: &str,
        act: Act,
    ) -> Option<Rc<Bo>>
    where
        F1: FnOnce() -> Array2<f32>,
        F2: FnOnce() -> Array2<f32>,
    {
        // Bind the args so the signature the device session wires against is fixed, WITHOUT invoking
        // the lazy weight closures (no host weight materialization on the fall-through-to-host path).
        let _ = (x_bo, m, b1, id1, b2, id2, act);
        drop((make_w1, make_w2));
        if self.resident_kres() == 768 {
            // DEVICE-GATED: the K=768 dispatch chain (cast -> fc1 modalgelu -> cast -> fc2 modalid ->
            // resadd_s100) lands here once (1)-(4) above are in place; not reachable on Parakeet.
            unimplemented!("resident_ffn_nonorm K=768 dispatch is device-gated (see the doc notes)")
        }
        None
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
    /// the baked-scale xclbin: s050 = 0.5 (Macaron residual) and s100 = 1.0 (MHSA residual) are both
    /// built; the FUSED_BLOCK path requires BOTH (gated by `resident_fused_available`). `a_bo`/`b_bo`
    /// are device f32 [PAD_M,KRES] BOs; returns the device result (ResidualAdd scratch, overwritten by
    /// the next call). `None` when the selected-scale xclbin is absent; PANICS on an unbuilt scale.
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
        modal_site("ra.kern#1");
        { let _dt = self.dtimer(); ra.kern.run_matmul8(3, &ra.instr, ra.n, a_bo, b_bo, &ra.bo_out, &ra.dummy_tmp, &ra.dummy_tr).unwrap(); }
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

    /// Per-N instruction stream. On the MODAL resident, `act` picks the baked-RTP mode
    /// (`Silu`->`modalsilu` = fc1/ff.l1 N=4096 on-chip SiLU, `Identity`->`modalid` = numerically
    /// identity epilogue for every other GEMM, `Gelu`->`modalgelu` = the K=768 GELU FFN rail). On
    /// the plain resident there is no on-chip epilogue (the host applies the activation), so `act`
    /// is normalized to Identity and the classic insts_*_8c.txt stream is used.
    fn stream(&self, n: usize, act: Act) -> Rc<NStream> {
        // Every shipped stream contracts over K=KRES; only the one-dispatch fc2 collapse varies K.
        self.stream_k(KRES, n, act)
    }

    /// `stream()` with the contraction length explicit. K reaches the array ONLY through the
    /// instruction stream (the A/B tap counts and the rtp[1] trip count), which is why one resident
    /// xclbin can serve several K -- but that is true only of a `k_loop_rtp` build, where the core
    /// reads its k-loop bound from rtp[1] instead of baking it. Dispatching a K != the build's K on a
    /// BAKED resident silently contracts over the wrong length, so the krtp streams and the krtp
    /// resident must be selected together (`PARAKEET_MODAL_EPI_SUFFIX=krtp` + `NPU_RESIDENT_XCLBIN`).
    /// A missing stream file panics in the read below rather than falling back, which is what makes
    /// that pairing fail loud instead of quietly wrong.
    fn stream_k(&self, k: usize, n: usize, act: Act) -> Rc<NStream> {
        // Derived, because for a DEVICE-fed A the layout is a property of the producer (fc1's drain).
        let a_panel = k != KRES && self.fc1_pack_in_drain_on();
        self.stream_k_ex(k, n, act, a_panel)
    }

    /// `stream_k` with the A-tap layout stated rather than derived -- for a HOST-fed A, which is
    /// row-major regardless of what fc1's drain does.
    fn stream_k_ex(&self, k: usize, n: usize, act: Act, a_panel: bool) -> Rc<NStream> {
        // The plain (non-modal) resident has no epilogue, so the activation is a no-op there; collapse
        // to Identity so the cache stays 1:1 with the single plain insts file (byte-identical to the
        // old `silu && self.modal` key).
        let act = if self.modal { act } else { Act::Identity };
        let key = (k, n, act, a_panel);
        if let Some(s) = self.streams.borrow().get(&key) {
            return s.clone();
        }
        let g = |i| self.kern.group_id(i).unwrap();
        let insts = if self.modal {
            // PARAKEET_MODAL_EPI_SUFFIX appends the Makefile.modal epilogue-variant tag (`re`,
            // `ft`, `fh`, `refh`, ...) to the mode tag, so an epilogue A/B can select its own
            // instruction streams. Pair it with NPU_RESIDENT_XCLBIN, which already overrides the
            // xclbin: the variant is a different EPI_DEFINES build, so both the array program and
            // every per-N stream carry the suffix and the two must not be mixed.
            let mode = act.mode_tag();
            // Under the fold every stream is a bf16-out build, so it carries `bf16out`. NOT via
            // PARAKEET_MODAL_EPI_SUFFIX: that suffix is also appended to fc1's own panel stem, which
            // would name a stream that does not exist (`...panel1024bf16out`).
            let sfx = if fold_fc1() {
                "bf16out".to_string()
            } else {
                std::env::var("PARAKEET_MODAL_EPI_SUFFIX").unwrap_or_default()
            };
            // insts-only stem (engine-op-manifest-and-dynamic-xclbin): most (n, mode) combos here
            // have NO co-resident `final_*.xclbin` at this same stem -- they all dispatch on the
            // ONE resident kernel loaded in `open()`, only the instruction stream differs per call
            // -- so this stays `insts_path` only, never `resolve`/`resolve_checked` (those assume
            // a paired xclbin at the same stem, which does not exist for most of these).
            // A WIDE-K stream contracts over fc1's output, and fc1 drains PANEL-major by default,
            // so such a stream must carry the matching panel A tap or it reads A row-major and is
            // silently wrong. That is a property of the PRODUCER, not a free choice, so derive it
            // here rather than add a second env knob that could disagree with the fc1 path.
            let krtp = if self.krtp { "krtp" } else { "" };
            let apanel = if a_panel { format!("apanel{KRES}") } else { String::new() };
            kernel_registry::insts_path(&self.base, &format!("{PAD_M}x{k}x{n}_{}_8c_{mode}{sfx}{krtp}{apanel}", self.tile))
        } else {
            kernel_registry::insts_path(&self.base, &format!("{PAD_M}x{k}x{n}_{}_8c", self.tile))
        };
        let bytes = std::fs::read(&insts).unwrap_or_else(|e| panic!("read {}: {e}", insts.display()));
        let n_instr = bytes.len() / 4;
        let instr = self.dev.alloc_bo(&self.kern, bytes.len(), FLAG_CACHEABLE, g(1)).unwrap();
        instr.write_bytes(&bytes).unwrap();
        instr.sync_to_device().unwrap();
        let bo_c = self.dev.alloc_bo(&self.kern, PAD_M * n * 4, FLAG_HOST_ONLY, g(5)).unwrap();
        let s = Rc::new(NStream { instr, n_instr, bo_c });
        // The dispatch report identifies streams by BO address only -- it cannot know what one
        // MEANS. Name it here, where (n, act) is in hand, so the two outputs can be joined.
        // Note n_instr does NOT identify N: all three modal N stream files are 1436 words, since
        // they differ in the BDs' size/stride FIELDS, not in the chain's shape.
        if npu_xrt::dispatch_log::enabled() {
            println!(
                "[dispatch_log] stream k={k} n={n} act={} -> {} insts words, bo 0x{:x}",
                act.mode_tag(), s.n_instr, s.instr.addr()
            );
        }
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

    /// bf16-native sibling of [`Self::weight_bo`]: `bits` are ALREADY-packed bf16 (row-major
    /// `[k, n]`), straight from a bf16-baked `NPU_WEIGHTS_CHECKPOINT` checkpoint tensor, so this writes them
    /// to the device BO directly and skips `pack_f32_to_bf16` entirely. Same wcache/ncache keying
    /// as `weight_bo`, so the two are interchangeable per `id` (a cache hit on either serves both).
    fn weight_bo_bf16(&self, id: &str, k: usize, n: usize, bits: &[u16]) -> Rc<Bo> {
        if let Some(bo) = self.wcache.borrow().get(id) {
            return bo.clone();
        }
        debug_assert_eq!(bits.len(), k * n, "weight_bo_bf16: bits.len() != k*n for id={id}");
        let t0 = Instant::now();
        let g4 = self.kern.group_id(4).unwrap();
        let bo = self.dev.alloc_bo(&self.kern, k * n * 2, FLAG_HOST_ONLY, g4).unwrap();
        bo.write_bytes(u16_bytes(bits)).unwrap();
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
        let st = self.stream(n, if silu { Act::Silu } else { Act::Identity });
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

    /// bf16-checkpoint sibling of [`Self::matmul_id`]: single-dispatch (K=KRES) only -- the shape every
    /// mhsa q/k/v/pos/out projection uses. `bits` are pre-packed bf16 `[k, n]` row-major straight
    /// from a bf16-baked `NPU_WEIGHTS_CHECKPOINT` tensor (verbatim layout, no transpose), so this skips
    /// the host f32->bf16 pack entirely on a cache miss.
    pub fn matmul_id_bf16(&self, a: &Array2<f32>, id: &str, k: usize, n: usize, bits: &[u16]) -> Array2<f32> {
        let (m, ka) = a.dim();
        assert_eq!(ka, k);
        assert!(m <= PAD_M);
        assert_eq!(k, KRES, "matmul_id_bf16 is single-dispatch only (K=KRES)");
        self.stats.borrow_mut().calls += 1;
        let cached = self.wcache.borrow().get(id).cloned();
        let wbo = cached.unwrap_or_else(|| self.weight_bo_bf16(id, k, n, bits));
        self.dispatch(a.view(), &wbo, n, false)
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
        let st = self.stream(n, Act::Identity); // ff.l2 K-split output has no activation (identity epilogue)
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

#[cfg(test)]
mod resolve_verified_tests {
    use super::*;

    // Deterministic (no filesystem, no device): with NPU_KERNEL_MANIFEST_VERIFY unset -- the
    // default for every existing build/CI invocation -- `resolve_verified` must be an exact
    // passthrough to `kernel_registry::resolve`, not just "close to it". This is the guarantee the
    // whole routing pass rests on: every call site converted in this pass keeps its pre-existing
    // default-path behavior byte-for-byte, only gaining opt-in verification on top.
    #[test]
    fn default_is_unchecked_passthrough_matching_kernel_registry_resolve() {
        std::env::remove_var("NPU_KERNEL_MANIFEST_VERIFY");
        let dir = Path::new("/does/not/need/to/exist");
        for stem in [
            "512x1024x4096_64x32x128_8c_modalsilu",
            "ctxln_512x1024",
            "resadd_512x1024_s050",
            "dwconv_silu_t_1024x448",
        ] {
            let got = resolve_verified(dir, stem);
            let want = kernel_registry::resolve(dir, stem);
            assert_eq!(got.xclbin, want.xclbin, "stem={stem}");
            assert_eq!(got.insts, want.insts, "stem={stem}");
        }
    }
}
