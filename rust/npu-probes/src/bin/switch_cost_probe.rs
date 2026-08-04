//! Switch-cost probe: is the encoder's wall clock explained by WHAT we dispatch, or by the ORDER
//! we dispatch it in?
//!
//! Every encoder brick is built for `devicename = npu2` = the full 8-column partition
//! (`partition_main.json`: `column_width: 8`), even the elementwise ones that occupy 2 columns. Two
//! 8-wide contexts cannot co-reside, so an xclbin-to-xclbin transition costs a full array reprogram.
//! A warm same-context dispatch is ~0.05-0.1 ms (`dispatch-overhead-is-47us-not-2670us`); a
//! modal<->relpos switch measured 0.99 ms (`modal-relpos-per-switch-cost`). This probe measures the
//! switch cost for the bricks the SHIPPED encoder actually alternates among, rather than borrowing
//! that number from a different pair.
//!
//! Three arms over the SAME multiset of dispatches (identical bytes, identical compute -- only the
//! transition count differs, so the delta is pure switch cost):
//!   SOLO     each kernel looped alone         -> per-dispatch cost at ZERO transitions
//!   GROUPED  the real multiset, sorted        -> ~one transition per distinct kernel
//!   REPLAY   the real per-clip sequence       -> the encoder's actual transition count
//!
//! Feed REPLAY the sequence dumped by `NPU_DISPATCH_LOG=1 NPU_DISPATCH_SEQ=<file>
//! parakeet_encode_npu`. NPU is single-tenant: stop npu-asr/flm-asr/voxd first.
//!
//! Usage: switch_cost_probe <repo_root> <sequence_file>

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::Instant;

use npu_xrt::{Bo, Device, Kernel, FLAG_CACHEABLE, FLAG_HOST_ONLY};

const WA: &str = "mlir-aie/programming_examples/basic/matrix_multiplication/whole_array/build";
const LN: &str = "artifacts/parakeet/ln";

const PAD_M: usize = 512;
const KRES: usize = 1024;
const DFF: usize = 4096;

// dwconv_silu_t, mirroring npu.rs's DW_* block.
const DW_C: usize = 1024;
const DW_T: usize = 400;
const DW_K: usize = 9;
const DW_TPAD: usize = DW_T + 2 * 4; // 'same' pad = (K-1)/2 = 4

// relpos bucket_152, derived exactly as `relpos_block` does rather than written as three magic
// byte counts: RELPOS_BUCKETS says (BUILT_T, KB) = (152, 38).
const RELPOS_TQ: usize = 8;
const RELPOS_DK: usize = 128;
const RELPOS_HEADS: usize = 8;
const RELPOS_BT: usize = 152;
const RELPOS_KB: usize = 38;
const RELPOS_NQT: usize = RELPOS_BT.div_ceil(RELPOS_TQ);
const RELPOS_TP: usize = RELPOS_BT.div_ceil(RELPOS_KB) * RELPOS_KB;
const RELPOS_PP: usize = (2 * RELPOS_BT - 1).div_ceil(RELPOS_KB) * RELPOS_KB;

/// One dispatchable brick, with the BO sizes the shipped encoder gives it (args 3,4,5,6,7 of the
/// matmul8 ABI). Sizes are copied from `npu-parakeet/src/npu.rs` so per-dispatch bytes are real.
struct Brick {
    label: String,
    kern: Rc<Kernel>,
    instr: Bo,
    n_instr: usize,
    a: Bo,
    b: Bo,
    c: Bo,
    tmp: Bo,
    tr: Bo,
}

impl Brick {
    fn load(dev: &Device, xclbin: &Path, insts: &Path, sizes: [usize; 5]) -> Brick {
        let kern = dev
            .load_kernel(xclbin.to_str().unwrap(), None)
            .unwrap_or_else(|e| panic!("load {}: {e}", xclbin.display()));
        let ib = std::fs::read(insts).unwrap_or_else(|e| panic!("read {}: {e}", insts.display()));
        let n_instr = ib.len() / 4;
        let g = |i| kern.group_id(i).unwrap();
        let instr = dev.alloc_bo(&kern, ib.len(), FLAG_CACHEABLE, g(1)).unwrap();
        instr.write_bytes(&ib).unwrap();
        instr.sync_to_device().unwrap();
        let label = xclbin.file_stem().unwrap().to_string_lossy().to_string();
        Brick {
            a: dev.alloc_bo(&kern, sizes[0], FLAG_HOST_ONLY, g(3)).unwrap(),
            b: dev.alloc_bo(&kern, sizes[1], FLAG_HOST_ONLY, g(4)).unwrap(),
            c: dev.alloc_bo(&kern, sizes[2], FLAG_HOST_ONLY, g(5)).unwrap(),
            tmp: dev.alloc_bo(&kern, sizes[3], FLAG_HOST_ONLY, g(6)).unwrap(),
            tr: dev.alloc_bo(&kern, sizes[4], FLAG_HOST_ONLY, g(7)).unwrap(),
            label, kern, instr, n_instr,
        }
    }

    fn dispatch(&self) {
        self.kern
            .run_matmul8(3, &self.instr, self.n_instr, &self.a, &self.b, &self.c, &self.tmp, &self.tr)
            .unwrap();
    }
}

fn transitions(seq: &[usize]) -> usize {
    seq.windows(2).filter(|w| w[0] != w[1]).count()
}

/// Time one ordered run of `seq` (indices into `bricks`), min over `reps`.
fn time_seq(bricks: &[Brick], seq: &[usize], reps: usize) -> f64 {
    let mut best = f64::MAX;
    for _ in 0..reps {
        let t = Instant::now();
        for &i in seq {
            bricks[i].dispatch();
        }
        best = best.min(t.elapsed().as_secs_f64());
    }
    best
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let root = PathBuf::from(&args[1]);
    let seq_file = &args[2];
    let reps: usize = std::env::var("PROBE_REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(3);

    let dev = Device::open(0).expect("open NPU");
    let wa = root.join(WA);
    let ln = root.join(LN);

    // The kernels the shipped encoder dispatches TODAY. Re-derived 2026-07-30 from a fresh
    // `NPU_DISPATCH_LOG=1` dump: 456 dispatches/clip over THREE xclbins (360 fc-modal + 48
    // lnaffcast + 48 bf16-outpanel fc1), not the four this probe was first written against.
    // ctxln/affcast were fused into lnaffcast and deint was folded away, so loading those three
    // made the REPLAY arm panic on an unknown kernel name. BO sizes copied from deint_fold_probe.
    let c_bytes_f32 = PAD_M * DFF * 4;
    let c_bytes_bf16 = PAD_M * DFF * 2;
    let bricks = vec![
        Brick::load(
            &dev,
            &wa.join("final_512x1024x4096_64x32x128_8c_modalsilu.xclbin"),
            &wa.join("insts_512x1024x4096_64x32x128_8c_modalsilu.txt"),
            [PAD_M * KRES * 2, KRES * DFF * 2, c_bytes_f32, 1, 4],
        ),
        Brick::load(
            &dev,
            &ln.join("final_lnaffcast_512x1024.xclbin"),
            &ln.join("insts_lnaffcast_512x1024.txt"),
            [PAD_M * KRES * 4, 2 * KRES * 4, PAD_M * KRES * 2, 8, 1],
        ),
        Brick::load(
            &dev,
            &wa.join("final_512x1024x4096_32x32x128_8c_modalsilubf16outpanel1024.xclbin"),
            &wa.join("insts_512x1024x4096_32x32x128_8c_modalsilubf16outpanel1024.txt"),
            [PAD_M * KRES * 2, KRES * DFF * 2, c_bytes_bf16, 1, 4],
        ),
        // The six bricks the fc2/conv/MHSA path adds. Without them REPLAY panics on an unknown
        // name, so the probe could only ever see 3 of the 9 kernels the encoder alternates among
        // -- and the three it saw are the CHEAP ones. Sizes copied from npu.rs's own alloc sites.
        Brick::load(
            &dev,
            &ln.join("final_accadd_512x1024.xclbin"),
            &ln.join("insts_accadd_512x1024.txt"),
            [PAD_M * KRES * 4, PAD_M * KRES * 4, PAD_M * KRES * 4, 8, 1],
        ),
        Brick::load(
            &dev,
            &ln.join("final_glu_512x1024.xclbin"),
            &ln.join("insts_glu_512x1024.txt"),
            [PAD_M * 2 * KRES * 4, PAD_M * KRES * 4, 1, 8, 1],
        ),
        Brick::load(
            &dev,
            &ln.join("final_resadd_512x1024_s050.xclbin"),
            &ln.join("insts_resadd_512x1024_s050.txt"),
            [PAD_M * KRES * 4, PAD_M * KRES * 4, PAD_M * KRES * 4, 8, 1],
        ),
        Brick::load(
            &dev,
            &ln.join("final_resadd_512x1024_s100.xclbin"),
            &ln.join("insts_resadd_512x1024_s100.txt"),
            [PAD_M * KRES * 4, PAD_M * KRES * 4, PAD_M * KRES * 4, 8, 1],
        ),
        Brick::load(
            &dev,
            &ln.join("final_dwconv_silu_t_1024x400.xclbin"),
            &ln.join("insts_dwconv_silu_t_1024x400.txt"),
            [DW_TPAD * DW_C * 2, (DW_K + 1) * DW_C * 2, DW_T * DW_C * 4, 8, 1],
        ),
        // relpos: the priciest dispatch in the encoder (9.9 ms in place) and the one this probe
        // has never covered. Its artifact is literally named `final.xclbin`, which is why it shows
        // up as the bare label `final` in every dispatch report.
        Brick::load(
            &dev,
            &root.join("artifacts/relpos/bucket_152/final.xclbin"),
            &root.join("artifacts/relpos/bucket_152/insts.bin"),
            [
                RELPOS_HEADS * 2 * RELPOS_NQT * RELPOS_TQ * RELPOS_DK * 2,
                RELPOS_HEADS * (RELPOS_TP + RELPOS_PP + RELPOS_PP + RELPOS_TP) * RELPOS_DK * 2,
                RELPOS_HEADS * RELPOS_NQT * RELPOS_TQ * RELPOS_DK * 2,
                1,
                4,
            ],
        ),
    ];
    let idx: BTreeMap<&str, usize> =
        bricks.iter().enumerate().map(|(i, b)| (b.label.as_str(), i)).collect();

    // --- ARM 1: SOLO -- per-dispatch cost with zero transitions -------------------------------
    println!("== SOLO (looped alone, zero transitions) ==");
    let mut solo_ms = vec![0f64; bricks.len()];
    const SOLO_N: usize = 64;
    for (i, b) in bricks.iter().enumerate() {
        let s = vec![i; SOLO_N];
        let t = time_seq(&bricks, &s, reps);
        solo_ms[i] = t * 1e3 / SOLO_N as f64;
        println!("  {:<44} {:.3} ms/dispatch", b.label, solo_ms[i]);
    }

    // --- The real sequence, and the same multiset grouped ---------------------------------------
    let raw = std::fs::read_to_string(seq_file).unwrap_or_else(|e| panic!("read {seq_file}: {e}"));
    let replay: Vec<usize> = raw
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| *idx.get(l.trim()).unwrap_or_else(|| panic!("sequence names unknown kernel {l:?}")))
        .collect();
    let mut grouped = replay.clone();
    grouped.sort_unstable();

    let t_replay = time_seq(&bricks, &replay, reps);
    let t_grouped = time_seq(&bricks, &grouped, reps);
    let (n_r, n_g) = (transitions(&replay), transitions(&grouped));
    // Cost the same multiset would take at the solo rate = the zero-transition floor.
    let floor: f64 = replay.iter().map(|&i| solo_ms[i] / 1e3).sum();

    println!("\n== SAME {} dispatches, two orders ==", replay.len());
    println!("  REPLAY  (real order)   {:>8.3} s   transitions {n_r}", t_replay);
    println!("  GROUPED (sorted)       {:>8.3} s   transitions {n_g}", t_grouped);
    println!("  solo-rate floor        {:>8.3} s   transitions 0", floor);
    println!("\n  delta replay-grouped   {:>8.3} s  over {} extra transitions", t_replay - t_grouped, n_r - n_g);
    if n_r > n_g {
        println!("  => measured switch cost {:.3} ms/transition", (t_replay - t_grouped) * 1e3 / (n_r - n_g) as f64);
    }
    println!("  delta grouped-floor    {:>8.3} s  (residual: not order, not solo compute)", t_grouped - floor);

    // --- ARM 4: RESTREAM -- swapping instruction streams WITHIN one hw_context ------------------
    // The arms above vary the xclbin, so they price a context switch. This one holds the xclbin
    // fixed and varies only the instruction stream, which is the other thing the encoder already
    // does: the modal resident serves every N off one hw_context via a per-N stream, and those
    // streams differ in their BDs' size/stride fields. `load_kernel` caches by path, so both
    // bricks share ONE Kernel and no switch can occur.
    //
    // Same multiset in both orders, so bytes and compute are identical and only the stream-change
    // count differs -- the delta is the reconfiguration alone. Comparing the two streams against
    // EACH OTHER would instead measure N=1024 against N=2048.
    let restream = vec![
        Brick::load(
            &dev,
            &wa.join("final_512x1024x4096_64x32x128_8c_modalsilu.xclbin"),
            &wa.join("insts_512x1024x1024_64x32x128_8c_modalid.txt"),
            [PAD_M * KRES * 2, KRES * 1024 * 2, PAD_M * 1024 * 4, 1, 4],
        ),
        Brick::load(
            &dev,
            &wa.join("final_512x1024x4096_64x32x128_8c_modalsilu.xclbin"),
            &wa.join("insts_512x1024x2048_64x32x128_8c_modalid.txt"),
            [PAD_M * KRES * 2, KRES * 2048 * 2, PAD_M * 2048 * 4, 1, 4],
        ),
    ];
    const RS_N: usize = 128; // per stream, so 256 dispatches per arm
    let alt: Vec<usize> = (0..2 * RS_N).map(|i| i % 2).collect();
    let mut grp = alt.clone();
    grp.sort_unstable();
    let t_alt = time_seq(&restream, &alt, reps);
    let t_grp = time_seq(&restream, &grp, reps);
    let (n_a, n_g2) = (transitions(&alt), transitions(&grp));
    println!("\n== RESTREAM: same xclbin, {} dispatches, two orders ==", alt.len());
    println!("  ALTERNATING (n=1024/2048)  {:>8.4} s   stream changes {n_a}", t_alt);
    println!("  GROUPED     (same multiset) {:>7.4} s   stream changes {n_g2}", t_grp);
    if n_a > n_g2 {
        println!(
            "  => restream cost {:.4} ms/change   (vs {:.3} ms/xclbin-transition above)",
            (t_alt - t_grp) * 1e3 / (n_a - n_g2) as f64,
            (t_replay - t_grouped) * 1e3 / (n_r - n_g).max(1) as f64,
        );
    }
}
