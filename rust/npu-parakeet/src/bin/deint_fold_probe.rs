//! Ladder probe for the fc1 seam: layout, output dtype, rounding mode and the bfp16 emulation.
//!
//! Started as "does folding the fc1->fc2 K-panel packing into the GEMM's own C drain pay" and grew
//! into the instrument for several follow-on questions. Arms, each isolating ONE variable:
//!   panel-major drain   same shipped xclbin, C drained panel-major. Control-code only: the PDI is
//!                       byte-identical, and the device output is a BIT-EXACT permutation.
//!   bf16 out + panel    the packing folded away entirely. New xclbin (a bf16 C tile cannot hold the
//!                       f32 accumulator), so m=32 -- at m=64 L1 overflows by ~11.6 KB.
//!   tile / C-depth      controls proving the accuracy delta is neither the tile nor the buffering.
//!   identity epilogue   no tanh, no narrowing: isolates the MATMUL from the epilogue.
//!   rounding arms       conv_even in the epilogue vs inside mm.cc itself.
//!   NATIVE              emulation off. CURRENTLY PRODUCES GARBAGE (rel-L2 1.367) -- that build
//!                       configuration is unvalidated upstream, see the native-bf16 task.
//!
//! TWO TRAPS THIS PROBE WALKED INTO, both fixed, both worth knowing before trusting any arm:
//!   1. `fill()` had a shift bug that emitted 99.96% POSITIVE values with mean ~1018 instead of
//!      zero-mean [-1,1]. A same-signed reduction flatters nearest-even against a floor bias, which
//!      inflated a real 1.3x result into a reported 40x. VALIDATE THE GENERATOR, not its comment.
//!   2. The epilogue and matmul objects were named by TILE while their content depends on -D flags,
//!      so make relinked a stale object into a differently-named xclbin and an A/B compared two
//!      identical binaries. Both Makefile rules now carry a defines stamp. md5sum the .o, always.
//!
//! NPU is single-tenant: stop xdna-engine/npu-vox first.
//!
//! Usage: deint_fold_probe <repo_root>

use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::Instant;

use npu_xrt::{Bo, Device, Kernel, FLAG_CACHEABLE, FLAG_HOST_ONLY};

const WA: &str = "mlir-aie/programming_examples/basic/matrix_multiplication/whole_array/build";
const LN: &str = "artifacts/parakeet/ln";

const PAD_M: usize = 512;
const KRES: usize = 1024;
const DFF: usize = 4096;
const N_CHUNKS: usize = DFF / KRES;

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
    fn load(dev: &Device, xclbin: &Path, insts: &Path, sizes: [usize; 5], label: &str) -> Brick {
        let kern = dev
            .load_kernel(xclbin.to_str().unwrap(), None)
            .unwrap_or_else(|e| panic!("load {}: {e}", xclbin.display()));
        let ib = std::fs::read(insts).unwrap_or_else(|e| panic!("read {}: {e}", insts.display()));
        let n_instr = ib.len() / 4;
        let g = |i| kern.group_id(i).unwrap();
        let instr = dev.alloc_bo(&kern, ib.len(), FLAG_CACHEABLE, g(1)).unwrap();
        instr.write_bytes(&ib).unwrap();
        instr.sync_to_device().unwrap();
        Brick {
            a: dev.alloc_bo(&kern, sizes[0], FLAG_HOST_ONLY, g(3)).unwrap(),
            b: dev.alloc_bo(&kern, sizes[1], FLAG_HOST_ONLY, g(4)).unwrap(),
            c: dev.alloc_bo(&kern, sizes[2], FLAG_HOST_ONLY, g(5)).unwrap(),
            tmp: dev.alloc_bo(&kern, sizes[3], FLAG_HOST_ONLY, g(6)).unwrap(),
            tr: dev.alloc_bo(&kern, sizes[4], FLAG_HOST_ONLY, g(7)).unwrap(),
            label: label.to_string(),
            kern,
            instr,
            n_instr,
        }
    }

    fn dispatch(&self) {
        self.kern
            .run_matmul8(3, &self.instr, self.n_instr, &self.a, &self.b, &self.c, &self.tmp, &self.tr)
            .unwrap();
    }
}

fn bf16_bits(x: f32) -> u16 {
    // round-to-nearest-even, matching cast_f32_bf16_row and the bf16-out epilogue's conv_even.
    let b = x.to_bits();
    let lsb = (b >> 16) & 1;
    (((b + 0x7fff + lsb) >> 16) & 0xffff) as u16
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Deterministic pseudo-random fill; avoids pulling a RNG crate into a probe binary.
fn fill(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            // 14-bit window. `s >> 40` alone leaves the top 24 bits, which made this generate
            // values in [-1, 2047] with mean ~1018 and 99.96% of them POSITIVE -- a same-sign
            // reduction, which is precisely the case that flatters round-to-nearest against a
            // floor bias and inflated an earlier measurement by ~30x. Mask to the intended range.
            ((((s >> 40) & 0x3FFF) as i32 - 8192) as f32 / 8192.0) * scale
        })
        .collect()
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// rel-L2 of `got` against `want`.
fn rel_l2(got: &[f32], want: &[f32]) -> f64 {
    let (mut num, mut den) = (0f64, 0f64);
    for (g, w) in got.iter().zip(want) {
        num += ((*g - *w) as f64).powi(2);
        den += (*w as f64).powi(2);
    }
    (num / den.max(1e-30)).sqrt()
}

fn write_bf16(bo: &Bo, vals: &[f32]) {
    let mut bytes = vec![0u8; vals.len() * 2];
    for (i, v) in vals.iter().enumerate() {
        bytes[i * 2..i * 2 + 2].copy_from_slice(&bf16_bits(*v).to_le_bytes());
    }
    bo.write_bytes(&bytes).unwrap();
    bo.sync_to_device().unwrap();
}

fn read_f32(bo: &Bo, n: usize) -> Vec<f32> {
    bo.sync_from_device().unwrap();
    let mut b = vec![0u8; n * 4];
    bo.read_bytes(&mut b).unwrap();
    (0..n)
        .map(|i| f32::from_le_bytes([b[i * 4], b[i * 4 + 1], b[i * 4 + 2], b[i * 4 + 3]]))
        .collect()
}

fn read_bf16(bo: &Bo, n: usize) -> Vec<f32> {
    bo.sync_from_device().unwrap();
    let mut b = vec![0u8; n * 2];
    bo.read_bytes(&mut b).unwrap();
    (0..n)
        .map(|i| bf16_to_f32(u16::from_le_bytes([b[i * 2], b[i * 2 + 1]])))
        .collect()
}

/// min-of-reps per-dispatch ms for a looped single brick.
fn solo_ms(b: &Brick, reps: usize, n: usize) -> f64 {
    let mut best = f64::MAX;
    for _ in 0..reps {
        let t = Instant::now();
        for _ in 0..n {
            b.dispatch();
        }
        best = best.min(t.elapsed().as_secs_f64());
    }
    best * 1e3 / n as f64
}

/// min-of-reps ms for one pass of an alternating sequence.
fn seq_ms(bricks: &[&Brick], seq: &[usize], reps: usize) -> f64 {
    let mut best = f64::MAX;
    for _ in 0..reps {
        let t = Instant::now();
        for &i in seq {
            bricks[i].dispatch();
        }
        best = best.min(t.elapsed().as_secs_f64());
    }
    best * 1e3
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let root = PathBuf::from(args.get(1).map(String::as_str).unwrap_or("."));
    let reps: usize = std::env::var("PROBE_REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(5);

    let dev = Device::open(0).expect("open NPU");
    let wa = root.join(WA);
    let ln = root.join(LN);

    // A = affine_LN output [PAD_M, KRES] bf16; B = fc1 weight [KRES, DFF] bf16.
    let a_host = fill(0x1234_5678, PAD_M * KRES, 1.0);
    let b_host = fill(0x9abc_def0, KRES * DFF, 0.05);

    // --- host reference: silu(A@B), then the two layouts we compare against -------------------
    // f32 row-major [PAD_M, DFF] -- what the shipped fc1 writes today.
    let mut ref_row = vec![0f32; PAD_M * DFF];
    for i in 0..PAD_M {
        for kk in 0..KRES {
            let av = bf16_to_f32(bf16_bits(a_host[i * KRES + kk]));
            if av == 0.0 {
                continue;
            }
            for j in 0..DFF {
                ref_row[i * DFF + j] += av * bf16_to_f32(bf16_bits(b_host[kk * DFF + j]));
            }
        }
    }
    for v in ref_row.iter_mut() {
        *v = silu(*v);
    }
    // chunk-major [N_CHUNKS, PAD_M, KRES] -- what deint produces, and what both new arms must write.
    let mut ref_chunk = vec![0f32; PAD_M * DFF];
    for c in 0..N_CHUNKS {
        for i in 0..PAD_M {
            for j in 0..KRES {
                ref_chunk[c * PAD_M * KRES + i * KRES + j] = ref_row[i * DFF + c * KRES + j];
            }
        }
    }

    let c_bytes_f32 = PAD_M * DFF * 4;
    let c_bytes_bf16 = PAD_M * DFF * 2;

    // ARM 0: shipped fc1 -- row-major f32 out, m=64.
    let base = Brick::load(
        &dev,
        &wa.join("final_512x1024x4096_64x32x128_8c_modalsilu.xclbin"),
        &wa.join("insts_512x1024x4096_64x32x128_8c_modalsilu.txt"),
        [PAD_M * KRES * 2, KRES * DFF * 2, c_bytes_f32, 1, 4],
        "fc1 row-major f32 m=64 (shipped)",
    );
    // ARM 1: STEP 1 -- same xclbin, chunk-major drain. Instruction stream only.
    let cm = Brick::load(
        &dev,
        &wa.join("final_512x1024x4096_64x32x128_8c_modalsilu.xclbin"),
        &wa.join("insts_512x1024x4096_64x32x128_8c_modalsilupanel1024.txt"),
        [PAD_M * KRES * 2, KRES * DFF * 2, c_bytes_f32, 1, 4],
        "fc1 chunk-major f32 m=64 (step 1, insts-only)",
    );
    // ARM 2: STEP 2 -- bf16 out + chunk-major, m=32. New xclbin; deint is deleted.
    let o16 = Brick::load(
        &dev,
        &wa.join("final_512x1024x4096_32x32x128_8c_modalsilubf16outpanel1024.xclbin"),
        &wa.join("insts_512x1024x4096_32x32x128_8c_modalsilubf16outpanel1024.txt"),
        [PAD_M * KRES * 2, KRES * DFF * 2, c_bytes_bf16, 1, 4],
        "fc1 chunk-major bf16 m=32 (step 2, deint folded)",
    );
    // ACCURACY DISCRIMINATOR. The folded arm measured CLOSER to f32 truth than the shipped one, which
    // is backwards for a bf16-stored path. Two candidate causes: the m=64 -> m=32 TILE change, or the
    // f32 -> bf16 OUTPUT change. This arm is m=32 with f32 out and a row-major drain, so it differs
    // from the shipped build in the tile ALONE. If it matches the shipped error the tile is innocent
    // and the epilogue/output path is implicated; if it matches the folded error the tile is the cause
    // and the shipped m=64 fast tile is losing accuracy on every GEMM in the encoder.
    let m32f32 = Brick::load(
        &dev,
        &wa.join("final_512x1024x4096_32x32x128_8c_modalsilu.xclbin"),
        &wa.join("insts_512x1024x4096_32x32x128_8c_modalsilu.txt"),
        [PAD_M * KRES * 2, KRES * DFF * 2, c_bytes_f32, 1, 4],
        "fc1 row-major f32 m=32 (tile control)",
    );
    // SECOND DISCRIMINATOR. The tile turned out to be innocent (m=32 f32 == m=64 f32 == 0.01205), so
    // the remaining difference between the shipped arm and the folded one is that the folded build got
    // WA_C_DEPTH=2. At depth 1 the C objectFIFO has a single buffer AND the matmul reduces IN PLACE
    // into it, so the same memory is both the running accumulator and the buffer the shim drains. This
    // arm is m=32 f32 -- identical to the tile control -- with C double-buffered and nothing else
    // changed. If its error drops to the folded arm's, single-buffered in-place accumulation is
    // costing the SHIPPED encoder accuracy on every GEMM, which is a correctness finding, not a tuning
    // one. m=64 cannot test this: an f32 C does not fit twice in L1, which is why depth is 1 there.
    let m32d2 = Brick::load(
        &dev,
        &wa.join("final_512x1024x4096_32x32x128_8c_modalsilud2.xclbin"),
        &wa.join("insts_512x1024x4096_32x32x128_8c_modalsilud2.txt"),
        [PAD_M * KRES * 2, KRES * DFF * 2, c_bytes_f32, 1, 4],
        "fc1 row-major f32 m=32 C-depth 2 (buffering control)",
    );
    // THE HYPOTHESIS UNDER TEST. Tile and buffering both came back innocent, and every f32-out arm
    // returns the SAME error to five decimals -- which points away from the GEMM's arithmetic and at
    // a global. The one thing the bf16-out epilogue does that no f32-out epilogue does is
    // `aie::set_rounding(conv_even)`. Rounding is a CORE CONTROL REGISTER, so it does not stay inside
    // the epilogue: it persists into the NEXT tile's matmul, where `emulate_bfloat16_mmul_with_bfp16`
    // converts every A/B tile bf16 -> bfp16. If truncation there is the real error source, then this
    // arm -- the SHIPPED m=64 f32 build with nothing changed but that one register write -- should
    // collapse to the folded arm's error, and the shipped encoder has been leaving accuracy on the
    // table on every GEMM for free.
    let m64re = Brick::load(
        &dev,
        &wa.join("final_512x1024x4096_64x32x128_8c_modalsilure.xclbin"),
        &wa.join("insts_512x1024x4096_64x32x128_8c_modalsilure.txt"),
        [PAD_M * KRES * 2, KRES * DFF * 2, c_bytes_f32, 1, 4],
        "fc1 row-major f32 m=64 + round-nearest-even",
    );
    // ISOLATION. The conv_even write in the `re` arm lands before BOTH the next tile's bfp16
    // conversion AND this epilogue's own `tanh<bfloat16>` narrowing, so the silu arms cannot say
    // which one the 2.8x came from -- and the answer decides whether this is an aie_api emulation
    // issue or purely our epilogue's. These two arms run the IDENTITY epilogue, which is a plain f32
    // copy with no tanh and no narrowing anywhere, so any difference between them is attributable to
    // the bfp16 conversion inside aie::mmul and nothing else.
    let id_def = Brick::load(
        &dev,
        &wa.join("final_512x1024x4096_64x32x128_8c_modalid.xclbin"),
        &wa.join("insts_512x1024x4096_64x32x128_8c_modalid.txt"),
        [PAD_M * KRES * 2, KRES * DFF * 2, c_bytes_f32, 1, 4],
        "identity epilogue, default rounding",
    );
    let id_re = Brick::load(
        &dev,
        &wa.join("final_512x1024x4096_64x32x128_8c_modalidre.xclbin"),
        &wa.join("insts_512x1024x4096_64x32x128_8c_modalidre.txt"),
        [PAD_M * KRES * 2, KRES * DFF * 2, c_bytes_f32, 1, 4],
        "identity epilogue, round-nearest-even",
    );
    // THE SILU PRECISION LADDER. The bias enters at the bf16 narrows inside the sigmoid, so there
    // are two ways out: DE-BIAS them (round-nearest-even, the `re` arm) or DELETE them.
    //   ft  keep tanh's bf16 output, do (t+1)*0.5 in f32 -- removes the second narrow.
    //   fh  aie::tanh<float> -- removes the narrow entirely, so the sigmoid is all-f32 and there is
    //       nothing left for a rounding mode to bias. This is the "annihilate" arm.
    let ft = Brick::load(
        &dev,
        &wa.join("final_512x1024x4096_64x32x128_8c_modalsiluft.xclbin"),
        &wa.join("insts_512x1024x4096_64x32x128_8c_modalsiluft.txt"),
        [PAD_M * KRES * 2, KRES * DFF * 2, c_bytes_f32, 1, 4],
        "silu: f32 (t+1)*0.5 tail, default rounding",
    );
    let fh = Brick::load(
        &dev,
        &wa.join("final_512x1024x4096_64x32x128_8c_modalsilufh.xclbin"),
        &wa.join("insts_512x1024x4096_64x32x128_8c_modalsilufh.txt"),
        [PAD_M * KRES * 2, KRES * DFF * 2, c_bytes_f32, 1, 4],
        "silu: all-f32 sigmoid (tanh<float>), default rounding",
    );
    // BOTH FIXES TOGETHER: round-nearest-even (kills the matmul's bfp16 bias) AND the all-f32
    // sigmoid (which measured 30% FASTER than the bf16 tanh path, refuting the "full-f32 tanh blows
    // the cycle budget" note in the kernel). They are orthogonal -- one is the matmul, one is the
    // epilogue -- so this arm should be both the most accurate and the fastest.
    let refh = Brick::load(
        &dev,
        &wa.join("final_512x1024x4096_64x32x128_8c_modalsilurefh.xclbin"),
        &wa.join("insts_512x1024x4096_64x32x128_8c_modalsilurefh.txt"),
        [PAD_M * KRES * 2, KRES * DFF * 2, c_bytes_f32, 1, 4],
        "silu: all-f32 sigmoid + round-nearest-even",
    );
    // THE ACTUAL UPSTREAM PATCH, measured as proposed rather than via an epilogue proxy: the
    // rounding swap lives inside AMD's own aie_kernels/aie2p/mm.cc (matmul_vectorized_2x2_mmul),
    // scoped to the bf16 shapes and restoring the caller's mode on exit. The epilogue is the
    // UNMODIFIED shipped one, so anything this arm gains comes from the matmul alone.
    let mmrne = Brick::load(
        &dev,
        &wa.join("final_512x1024x4096_64x32x128_8c_modalsilummrne.xclbin"),
        &wa.join("insts_512x1024x4096_64x32x128_8c_modalsilummrne.txt"),
        [PAD_M * KRES * 2, KRES * DFF * 2, c_bytes_f32, 1, 4],
        "PATCH: rounding swap inside mm.cc (stock epilogue)",
    );
    // NATIVE bf16: the emulation turned OFF entirely. No bf16->bfp16 conversion, so no shared-exponent
    // block quantization and no rounding-mode dependency at all. The MAC shape drops 8x8x8 -> 4x8x8,
    // i.e. the compute PEAK drops 4x (512 -> 128 MAC/cyc/core). The question this arm answers: we
    // measure 46.8 MAC/cyc/core, which is 9% of the emulated peak but still only 37% of the NATIVE
    // peak -- so if the GEMM is limited by data movement rather than MAC rate, turning the emulation
    // off should cost little or nothing, and buys back the precision for free.
    let nat = Brick::load(
        &dev,
        &wa.join("final_512x1024x4096_64x32x128_8c_modalsilunat.xclbin"),
        &wa.join("insts_512x1024x4096_64x32x128_8c_modalsilunat.txt"),
        [PAD_M * KRES * 2, KRES * DFF * 2, c_bytes_f32, 1, 4],
        "NATIVE bf16 (emulation OFF)",
    );
    // The fused ctxLN->affine_cast the encoder runs immediately before every fc1. It matters to the
    // verdict: it is a SMALL design, so it is what the encoder transitions back through between one
    // FFN's fc2 and the next FFN's fc1. Without it in the sequence the folded arm would be charged
    // two GEMM<->GEMM transitions per FFN instead of the one it actually pays.
    let lnaff = Brick::load(
        &dev,
        &ln.join("final_lnaffcast_512x1024.xclbin"),
        &ln.join("insts_lnaffcast_512x1024.txt"),
        [PAD_M * KRES * 4, 2 * KRES * 4, PAD_M * KRES * 2, 8, 1],
        "lnaffcast (fused ctxLN->affine_cast)",
    );
    // The deint dispatch step 2 deletes. npu.rs warns its n-D output DMA has hung when co-resident
    // with the modal, and a wedged hw_context needs a reboot on this driver, so it is loaded last
    // and can be skipped (PROBE_NO_DEINT=1) to get the correctness + solo numbers at no risk first.
    let with_deint = std::env::var("PROBE_NO_DEINT").map(|v| v == "0").unwrap_or(true);
    let deint = with_deint.then(|| {
        Brick::load(
            &dev,
            &ln.join("final_deint_512x4096.xclbin"),
            &ln.join("insts_deint_512x4096.txt"),
            [c_bytes_f32, c_bytes_bf16, 1, 8, 1],
            "deint (the dispatch step 2 deletes)",
        )
    });

    for b in [&base, &cm, &o16, &m32f32, &m32d2, &m64re, &id_def, &id_re, &ft, &fh, &refh, &mmrne, &nat] {
        write_bf16(&b.a, &a_host);
        write_bf16(&b.b, &b_host);
    }

    // --- correctness ---------------------------------------------------------------------------
    println!("== correctness (rel-L2 vs host silu(A@B) reference) ==");
    base.dispatch();
    let got_row = read_f32(&base.c, PAD_M * DFF);
    println!("  {:<48} {:.5}", base.label, rel_l2(&got_row, &ref_row));

    m32f32.dispatch();
    let got_m32 = read_f32(&m32f32.c, PAD_M * DFF);
    println!("  {:<48} {:.5}", m32f32.label, rel_l2(&got_m32, &ref_row));

    m32d2.dispatch();
    let got_d2 = read_f32(&m32d2.c, PAD_M * DFF);
    println!("  {:<48} {:.5}", m32d2.label, rel_l2(&got_d2, &ref_row));

    m64re.dispatch();
    let got_re = read_f32(&m64re.c, PAD_M * DFF);
    println!("  {:<48} {:.5}", m64re.label, rel_l2(&got_re, &ref_row));

    // identity arms compare against the PRE-silu reference
    let mut ref_lin = vec![0f32; PAD_M * DFF];
    for i in 0..PAD_M {
        for kk in 0..KRES {
            let av = bf16_to_f32(bf16_bits(a_host[i * KRES + kk]));
            if av == 0.0 { continue; }
            for j in 0..DFF {
                ref_lin[i * DFF + j] += av * bf16_to_f32(bf16_bits(b_host[kk * DFF + j]));
            }
        }
    }
    id_def.dispatch();
    let e_def = rel_l2(&read_f32(&id_def.c, PAD_M * DFF), &ref_lin);
    id_re.dispatch();
    let e_re = rel_l2(&read_f32(&id_re.c, PAD_M * DFF), &ref_lin);
    println!("  {:<48} {:.5}", id_def.label, e_def);
    println!("  {:<48} {:.5}", id_re.label, e_re);
    println!("  -> pure bfp16-conversion effect (no tanh in this path): {:.2}x", e_def / e_re.max(1e-12));

    ft.dispatch();
    println!("  {:<48} {:.5}", ft.label, rel_l2(&read_f32(&ft.c, PAD_M * DFF), &ref_row));
    fh.dispatch();
    println!("  {:<48} {:.5}", fh.label, rel_l2(&read_f32(&fh.c, PAD_M * DFF), &ref_row));

    refh.dispatch();
    println!("  {:<48} {:.5}", refh.label, rel_l2(&read_f32(&refh.c, PAD_M * DFF), &ref_row));

    mmrne.dispatch();
    println!("  {:<48} {:.5}", mmrne.label, rel_l2(&read_f32(&mmrne.c, PAD_M * DFF), &ref_row));

    nat.dispatch();
    println!("  {:<48} {:.5}", nat.label, rel_l2(&read_f32(&nat.c, PAD_M * DFF), &ref_row));

    cm.dispatch();
    let got_cm = read_f32(&cm.c, PAD_M * DFF);
    let e_cm = rel_l2(&got_cm, &ref_chunk);
    println!("  {:<48} {:.5}", cm.label, e_cm);
    // The strong statement: the chunk-major drain is EXACTLY a permutation of the row-major one.
    let mut perm_exact = true;
    for c in 0..N_CHUNKS {
        for i in 0..PAD_M {
            for j in 0..KRES {
                if got_cm[c * PAD_M * KRES + i * KRES + j] != got_row[i * DFF + c * KRES + j] {
                    perm_exact = false;
                }
            }
        }
    }
    println!("  step 1 is a bit-exact permutation of the shipped drain: {perm_exact}");

    o16.dispatch();
    let got_o16 = read_bf16(&o16.c, PAD_M * DFF);
    println!("  {:<48} {:.5}", o16.label, rel_l2(&got_o16, &ref_chunk));
    // ...and against what the shipped pair (fc1 f32 -> deint bf16 cast) would have produced.
    let want_o16: Vec<f32> = ref_chunk.iter().map(|v| bf16_to_f32(bf16_bits(*v))).collect();
    println!(
        "  step 2 vs shipped fc1+deint numerics (rel-L2):   {:.5}",
        rel_l2(&got_o16, &want_o16)
    );

    // --- cost ----------------------------------------------------------------------------------
    println!("\n== per-dispatch cost, looped alone (no transitions) ==");
    for b in [Some(&base), Some(&cm), Some(&o16), Some(&m32f32), Some(&m32d2), Some(&m64re), Some(&ft), Some(&fh), Some(&refh), Some(&mmrne), Some(&nat), Some(&lnaff), deint.as_ref()].into_iter().flatten() {
        println!("  {:<48} {:.3} ms", b.label, solo_ms(b, reps, 32));
    }
    let Some(deint) = deint.as_ref() else {
        println!("\n(PROBE_NO_DEINT=1: skipping the alternation arms)");
        return;
    };

    // --- the comparison that decides it --------------------------------------------------------
    // Model the WHOLE per-FFN sequence, not just the fc1 neighbourhood. That distinction decides
    // the sign: the encoder re-enters fc1 through lnaffcast, so a bf16 fc1 on its own xclbin pays
    // the expensive GEMM<->GEMM transition ONCE per FFN (fc1o16 -> fc2), not twice. An arm that
    // alternates fc1o16 <-> modal directly charges it twice and wrongly condemns the fold.
    //
    //   today   lnaffcast -> fc1(modal) -> deint -> fc2(modal) x4
    //   folded  lnaffcast -> fc1(o16)   ->          fc2(modal) x4
    //
    // `base` is the shipped modal xclbin, so it serves as both fc1-today and the fc2 partials.
    println!("\n== the encoder's real per-FFN sequence, 24 FFNs' worth ==");
    const FFNS: usize = 24;
    const FC2_PARTS: usize = 4;

    // 0 = lnaffcast, 1 = modal (fc1 today / fc2 in both arms), 2 = deint, 3 = fc1 bf16 chunk-major
    let all: Vec<&Brick> = vec![&lnaff, &base, deint, &o16];
    let mut seq_today = Vec::new();
    let mut seq_folded = Vec::new();
    for _ in 0..FFNS {
        seq_today.push(0);
        seq_today.push(1);
        seq_today.push(2);
        seq_today.extend(std::iter::repeat(1).take(FC2_PARTS));

        seq_folded.push(0);
        seq_folded.push(3);
        seq_folded.extend(std::iter::repeat(1).take(FC2_PARTS));
    }
    let t_today = seq_ms(&all, &seq_today, reps);
    let t_folded = seq_ms(&all, &seq_folded, reps);

    let tr = |s: &[usize]| s.windows(2).filter(|w| w[0] != w[1]).count();
    println!(
        "  today   {} dispatches, {} transitions   {:.2} ms  ({:.3} ms/FFN)",
        seq_today.len(), tr(&seq_today), t_today, t_today / FFNS as f64
    );
    println!(
        "  folded  {} dispatches, {} transitions   {:.2} ms  ({:.3} ms/FFN)",
        seq_folded.len(), tr(&seq_folded), t_folded, t_folded / FFNS as f64
    );
    let d = t_today - t_folded;
    println!(
        "  delta   {:+.2} ms over {FFNS} FFNs = {:+.3} ms/FFN  -> {}",
        d,
        d / FFNS as f64,
        if d > 0.0 { "FOLDING WINS" } else { "folding loses" }
    );
    println!(
        "  extrapolated to the 48 FFNs/clip: {:+.3} s/clip",
        d / FFNS as f64 * 48.0 / 1e3
    );
}
