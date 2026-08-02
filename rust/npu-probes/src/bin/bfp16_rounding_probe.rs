//! Is the local `round_even` epilogue trick still a win, or did upstream #3442 subsume it?
//!
//! Two identity-epilogue arms at the fc1 shape (512x1024x4096). The identity epilogue is a plain f32
//! copy -- no tanh, no bf16 narrowing -- so any difference between the arms is attributable to the
//! bfp16 conversion inside `aie::mmul` and nothing else.
//!   id_def  round_even=0, the shipped configuration
//!   id_re   round_even=1, our epilogue writes conv_even, which leaks into the next tile's conversion
//!
//! #3442 makes `mm.cc` save/set conv_even/restore around the emulated-bf16 mmul itself, so once
//! kernels_dir resolves against the pinned toolchain instance both arms should round the same way and
//! the ratio should collapse to ~1.00x. It measured 1.324x while kernels_dir still resolved mm.cc from
//! the drifted mlir-aie submodule -- that number is a FALSE NEGATIVE, not a result.
//!
//! Extracted from deint_fold_probe's id_def/id_re pair so this question is answerable without the
//! other 11 xclbins that probe wants. Tracked rather than scratchpad: the previous run of this
//! measurement lived in a scratchpad crate and was lost, which is why it had to be re-derived.
//!
//! Build the two arms first (CPU-only), from the repo root with scripts/iron_env.sh sourced:
//!   WA=mlir-aie/programming_examples/basic/matrix_multiplication/whole_array
//!   for re in 0 1; do
//!     tag=modalid$([ $re = 1 ] && echo re)
//!     WA_C_DEPTH=1 make -f Makefile.modal -C $WA NPU2=1 M=512 K=1024 N=4096 m=64 k=32 n=128 \
//!       dtype_in=bf16 dtype_out=f32 n_aie_cols=8 use_iron=1 emulate_bfloat16_mmul_with_bfp16=1 \
//!       bfp16_iree=1 no_silu=1 round_even=$re build/final_512x1024x4096_64x32x128_8c_$tag.xclbin
//!   done
//!
//! NPU is single-tenant: run under scripts/npu_lock.sh.
//!
//! Usage: bfp16_rounding_probe <repo_root>

use std::path::{Path, PathBuf};
use std::rc::Rc;

use npu_xrt::{Bo, Device, Kernel, FLAG_CACHEABLE, FLAG_HOST_ONLY};

const WA: &str = "mlir-aie/programming_examples/basic/matrix_multiplication/whole_array/build";

const PAD_M: usize = 512;
const KRES: usize = 1024;
const DFF: usize = 4096;

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
    fn load(dev: &Device, xclbin: &Path, insts: &Path, label: &str) -> Brick {
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
            a: dev.alloc_bo(&kern, PAD_M * KRES * 2, FLAG_HOST_ONLY, g(3)).unwrap(),
            b: dev.alloc_bo(&kern, KRES * DFF * 2, FLAG_HOST_ONLY, g(4)).unwrap(),
            c: dev.alloc_bo(&kern, PAD_M * DFF * 4, FLAG_HOST_ONLY, g(5)).unwrap(),
            tmp: dev.alloc_bo(&kern, 1, FLAG_HOST_ONLY, g(6)).unwrap(),
            tr: dev.alloc_bo(&kern, 4, FLAG_HOST_ONLY, g(7)).unwrap(),
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
    let b = x.to_bits();
    let lsb = (b >> 16) & 1;
    (((b + 0x7fff + lsb) >> 16) & 0xffff) as u16
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Deterministic pseudo-random fill. The 14-bit mask is load-bearing: an earlier version left the top
/// 24 bits, emitting 99.96% positive values, and a same-signed reduction flatters nearest-even against
/// a floor bias -- that inflated a real 1.3x into a reported 40x.
fn fill(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((((s >> 40) & 0x3FFF) as i32 - 8192) as f32 / 8192.0) * scale
        })
        .collect()
}

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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let root = PathBuf::from(args.get(1).map(String::as_str).unwrap_or("."));
    let wa = root.join(WA);

    let dev = Device::open(0).expect("open NPU");

    let a_host = fill(0x1234_5678, PAD_M * KRES, 1.0);
    let b_host = fill(0x9abc_def0, KRES * DFF, 0.05);

    // Host reference: plain A@B in f32, from the bf16-rounded inputs the device actually sees. No
    // silu -- the identity epilogue is linear.
    let mut ref_lin = vec![0f32; PAD_M * DFF];
    for i in 0..PAD_M {
        for kk in 0..KRES {
            let av = bf16_to_f32(bf16_bits(a_host[i * KRES + kk]));
            if av == 0.0 {
                continue;
            }
            for j in 0..DFF {
                ref_lin[i * DFF + j] += av * bf16_to_f32(bf16_bits(b_host[kk * DFF + j]));
            }
        }
    }

    let arms = [
        ("final_512x1024x4096_64x32x128_8c_modalid.xclbin", "identity epilogue, default rounding"),
        ("final_512x1024x4096_64x32x128_8c_modalidre.xclbin", "identity epilogue, round-nearest-even"),
    ];

    let mut errs = Vec::new();
    for (xclbin, label) in arms {
        let insts = xclbin.replace("final_", "insts_").replace(".xclbin", ".txt");
        let b = Brick::load(&dev, &wa.join(xclbin), &wa.join(&insts), label);
        write_bf16(&b.a, &a_host);
        write_bf16(&b.b, &b_host);
        b.dispatch();
        let e = rel_l2(&read_f32(&b.c, PAD_M * DFF), &ref_lin);
        println!("  {:<48} rel-L2 = {:.6e}", b.label, e);
        errs.push(e);
    }

    println!(
        "  -> default/round_even ratio: {:.3}x",
        errs[0] / errs[1].max(1e-12)
    );
    println!("     (1.324x = kernels_dir still drifted; ~1.00x = #3442 subsumes the local trick)");
}
