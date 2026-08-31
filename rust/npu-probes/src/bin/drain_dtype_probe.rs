//! Does a host readback decode the resident's C drain at the drain's own width?
//!
//! `PARAKEET_FOLD_FC1` makes fc1's bf16-out build the resident, which halves the C drain of EVERY
//! modal GEMM on it, not just fc1's. The readbacks were written against f32, which is why the fold
//! has always been TIMING-ONLY. This is the smallest thing that separates "the reader is
//! dtype-aware" from "the fold happens to be off": one host-fed K=1024 GEMM against an in-Rust
//! bf16-in/f32-accumulate reference, read back BOTH ways out of one device buffer -- at the drain's
//! own width, and through a hand-rolled f32 decode, which is exactly what every readback did before.
//! The f32 decode IS the control: on an f32 drain the two agree element-for-element, on a bf16 drain
//! it reads adjacent output pairs as one f32.
//!
//! It also prices what no host reader reaches: the modal C is handed straight to `residual_add_dev`
//! on the encoder's own MHSA seam, whose addend slot must follow the drain while the residual
//! stream it adds to stays f32.
//!
//! Read the two arms together, one process each:
//!   drain_dtype_probe                      -> drain 4 B/elem, both decodes identical
//!   PARAKEET_FOLD_FC1=1 drain_dtype_probe  -> drain 2 B/elem, dtype-aware holds, f32 decode diverges
//! The fold arm cannot beat the f32 arm: its drain rounds every output to bf16, so the correct
//! answer there is the f32 arm's error and the bf16 floor added in quadrature.

use std::path::Path;

use ndarray::prelude::*;
use npu_parakeet::npu::NpuMatmul;

/// splitmix64 -> f32 in [-1, 1]. Deterministic and dependency-free, so both arms see one input.
fn fill(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
    let mut s = seed;
    Array2::from_shape_fn((rows, cols), |_| {
        s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        ((z >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    })
}

fn bf16(x: f32) -> f32 {
    npu_xrt::bf16_bits_to_f32(npu_xrt::f32_to_bf16_bits(x))
}

fn rel_l2(got: &Array2<f32>, want: &Array2<f32>) -> f64 {
    let (mut num, mut den) = (0f64, 0f64);
    for (g, r) in got.iter().zip(want.iter()) {
        num += ((*g - *r) as f64).powi(2);
        den += (*r as f64).powi(2);
    }
    (num / den).sqrt()
}

fn main() {
    let root = std::env::var("XDNA_ROOT").unwrap_or_else(|_| ".".to_string());
    let (m, k, n) = (128usize, 1024usize, 1024usize);
    let a = fill(m, k, 1);
    let b = fill(k, n, 2);

    // Reference: the inputs the device actually sees (bf16) accumulated in f32.
    let want = a.mapv(bf16).dot(&b.mapv(bf16));
    let bf16_floor = rel_l2(&want.mapv(bf16), &want);

    let npu = NpuMatmul::open(Path::new(&root)).expect("open NpuMatmul");
    let w = npu.c_elem_bytes();

    // Device-out, so the SAME buffer can be decoded twice.
    let b2 = b.clone();
    let bo = npu.matmul_id_to_bo(&a, move || b2, "drain.probe.bo", n);
    bo.sync_from_device().unwrap();
    let mut cb = vec![0u8; m * n * 4];
    bo.read_bytes(&mut cb).unwrap();
    let aware = Array2::from_shape_fn((m, n), |(r, c)| {
        let i = r * n + c;
        if w == 4 {
            f32::from_le_bytes([cb[i * 4], cb[i * 4 + 1], cb[i * 4 + 2], cb[i * 4 + 3]])
        } else {
            npu_xrt::bf16_bits_to_f32(u16::from_le_bytes([cb[i * 2], cb[i * 2 + 1]]))
        }
    });
    let assumed = Array2::from_shape_fn((m, n), |(r, c)| {
        let o = (r * n + c) * 4;
        f32::from_le_bytes([cb[o], cb[o + 1], cb[o + 2], cb[o + 3]])
    });
    // The host-fed readback path (`dispatch`), which has its own decode loop.
    let dispatched = npu.matmul_id(&a, &b, "drain.probe.dispatch");

    println!("drain            = {w} bytes/element");
    println!("bf16 floor       = {bf16_floor:.3e}  (drain rounding alone)");
    println!("drain-width read = {:.3e}", rel_l2(&aware, &want));
    println!("dispatch         = {:.3e}", rel_l2(&dispatched, &want));
    println!("f32-assumed      = {:.3e}  (control: the decode every readback used to do)", rel_l2(&assumed, &want));

    // The device-side consumer, in the SEAM'S OWN SHAPE: the encoder's s100 residual adds the
    // modal C (linear_out) to the f32 residual stream the previous residual left resident, and
    // reads the sum back as f32. Both slots fed from a modal C would price an arrangement no
    // encoder path builds, and would hide which slot the drain width actually reaches.
    let x = fill(m, n, 3);
    let x_bo = npu.upload_stream(&x);
    let resadd = npu.residual_add_dev(&x_bo, &bo, 1.0, m).map(|sum_bo| {
        let host_sum = &x + &want;
        rel_l2(&npu.readback_stream(&sum_bo, m), &host_sum)
    });
    match resadd {
        Some(e) => println!("resadd(dev)      = {e:.3e}  (device consumer of the same C)"),
        None => println!("resadd(dev)      = SKIPPED (resadd xclbin absent)"),
    }

    let gate = 1e-2; // the ln-cheap numeric gate the FF1 parity harness uses
    // The HOST decodes are what this gate covers. On an f32 drain the control is not a control --
    // it is the same decode, so it must agree; on a bf16 drain it must diverge or nothing was fixed.
    let ok = rel_l2(&aware, &want) <= gate && rel_l2(&dispatched, &want) <= gate;
    let control_ok = (rel_l2(&assumed, &want) <= gate) == (w == 4);
    println!("host decodes {gate:.0e} -> {}, control -> {}",
        if ok { "PASS" } else { "FAIL" },
        if control_ok { "PASS" } else { "FAIL" });
    // Reported, never gated: a bf16 drain feeding an f32-in brick is a KERNEL gap, so a green exit
    // here means the readbacks are right, NOT that the fold is shippable.
    if let Some(e) = resadd {
        if e > gate {
            println!("OPEN: the device consumer is still wrong at {e:.3e} -- no host reader reaches it");
        }
    }
    if !ok || !control_ok {
        std::process::exit(1);
    }
}
