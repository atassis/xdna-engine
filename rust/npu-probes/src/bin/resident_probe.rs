//! Track B gate: does same-RESIDENT-KERNEL `bo_h` reuse (mm1 writes C=bo_h, mm2 reads A=bo_h on the
//! SAME modal V2 xclbin) work without the H-reuse deadlock / a BO-bank mismatch?
//!
//! ChainedFFN (engines.rs:454-484) already proves the bo_h-sharing DATAFLOW works across TWO xclbins
//! (kern1 mm1 -> bo_h -> kern2 mm2). V2 wants it on ONE resident kernel (no 2.67ms switch). The only
//! untested piece is whether reusing the SAME kernel's C buffer as the next dispatch's A — with no
//! context switch flushing state between them — hangs or errors. This probe isolates exactly that.
//!
//! Data is GARBAGE here (mm1 is the shipped f32-out modal stream, mm2 reads bo_h as bf16) — correctness
//! needs a bf16-out mm1 mode, which is the NEXT step. This probe answers ONLY the data-independent
//! question: does the same-kernel C->A reuse complete? (run under `timeout` — a hang = the gate FAILS).
//!
//! NPU single-tenant — stop npu-asr/voxd first. Run from repo root.

use std::path::Path;
use std::rc::Rc;
use std::time::Instant;

use npu_xrt::{Bo, Device, Kernel, FLAG_CACHEABLE, FLAG_HOST_ONLY};

const WA: &str = "mlir-aie/programming_examples/basic/matrix_multiplication/whole_array/build";
const PAD_M: usize = 512;
const KAUG: usize = 800; // K-augmented contraction of the modal resident kernel
const NA: usize = 3072; // mm1 output width (FFN linear1)
const N2: usize = 768; // mm2 partial output width

fn load_stream(dev: &Device, kern: &Kernel, wa: &Path, ib: &str) -> (Bo, usize) {
    let bytes = std::fs::read(wa.join(ib)).unwrap_or_else(|e| panic!("read {ib}: {e}"));
    let n = bytes.len() / 4;
    let bo = dev.alloc_bo(kern, bytes.len(), FLAG_CACHEABLE, kern.group_id(1).unwrap()).unwrap();
    bo.write_bytes(&bytes).unwrap();
    bo.sync_to_device().unwrap();
    (bo, n)
}

fn main() {
    let root = Path::new(".");
    let wa = root.join(WA);
    let dev = Device::open(0).expect("open NPU (stop npu-asr/voxd first)");

    // ONE resident modal kernel (the shipped fast-bf16 default xclbin) serves both mm1 and mm2.
    let xb = wa.join("final_512x800x3072_64x32x96_8c_modalsilu.xclbin");
    let kern = dev.load_kernel(xb.to_str().unwrap(), None).expect("load modal xclbin");
    println!("[resident_probe] loaded ONE resident modal kernel: {}", xb.display());

    // mm1 = N=3072 silu stream (writes f32 H); mm2 = N=768 identity stream (reads H as A).
    let (instr1, n1) = load_stream(&dev, &kern, &wa, "insts_512x800x3072_64x32x96_8c_modalsilu.txt");
    let (instr2, n2) = load_stream(&dev, &kern, &wa, "insts_512x800x768_64x32x96_8c_modalid.txt");

    let g = |i| kern.group_id(i).unwrap();
    // mm1 buffers
    let bo_a1 = dev.alloc_bo(&kern, PAD_M * KAUG * 2, FLAG_HOST_ONLY, g(3)).unwrap();
    let bo_b1 = dev.alloc_bo(&kern, KAUG * NA * 2, FLAG_HOST_ONLY, g(4)).unwrap();
    // THE SHARED BO: mm1's C output AND mm2's A input. Allocated against mm1's C bank (group 5),
    // exactly as ChainedFFN does (engines.rs: bo_h on group_id(5), reused as kern2's A).
    let bo_h = dev.alloc_bo(&kern, PAD_M * NA * 4, FLAG_HOST_ONLY, g(5)).unwrap();
    let bo_tmp = dev.alloc_bo(&kern, 1, FLAG_HOST_ONLY, g(6)).unwrap();
    let bo_tr = dev.alloc_bo(&kern, 4, FLAG_HOST_ONLY, g(7)).unwrap();
    // mm2 buffers (its A is bo_h)
    let bo_b2 = dev.alloc_bo(&kern, KAUG * N2 * 2, FLAG_HOST_ONLY, g(4)).unwrap();
    let bo_c2 = dev.alloc_bo(&kern, PAD_M * N2 * 4, FLAG_HOST_ONLY, g(5)).unwrap();

    // zero-init inputs (garbage-data deadlock test; correctness comes with the bf16-out mode)
    let zeros_a = vec![0u8; PAD_M * KAUG * 2];
    bo_a1.write_bytes(&zeros_a).unwrap(); bo_a1.sync_to_device().unwrap();
    let zeros_b1 = vec![0u8; KAUG * NA * 2];
    bo_b1.write_bytes(&zeros_b1).unwrap(); bo_b1.sync_to_device().unwrap();
    let zeros_b2 = vec![0u8; KAUG * N2 * 2];
    bo_b2.write_bytes(&zeros_b2).unwrap(); bo_b2.sync_to_device().unwrap();

    println!("[resident_probe] dispatching mm1 (writes C=bo_h) on the resident kernel...");
    let t0 = Instant::now();
    kern.run_matmul8(3, &instr1, n1, &bo_a1, &bo_b1, &bo_h, &bo_tmp, &bo_tr)
        .expect("mm1 dispatch failed");
    println!("  mm1 done in {:.3} ms", t0.elapsed().as_secs_f64() * 1e3);

    println!("[resident_probe] dispatching mm2 (reads A=bo_h) on the SAME resident kernel...");
    let t1 = Instant::now();
    kern.run_matmul8(3, &instr2, n2, &bo_h, &bo_b2, &bo_c2, &bo_tmp, &bo_tr)
        .expect("mm2 dispatch failed");
    println!("  mm2 done in {:.3} ms", t1.elapsed().as_secs_f64() * 1e3);

    // read mm2 output back (confirms the dispatch chain completed end-to-end)
    bo_c2.sync_from_device().unwrap();
    let mut buf = vec![0u8; 16];
    bo_c2.read_bytes(&mut buf).unwrap();

    // repeat 50× to be sure it's not a one-off (a deadlock would hang here under timeout)
    for _ in 0..50 {
        kern.run_matmul8(3, &instr1, n1, &bo_a1, &bo_b1, &bo_h, &bo_tmp, &bo_tr).unwrap();
        kern.run_matmul8(3, &instr2, n2, &bo_h, &bo_b2, &bo_c2, &bo_tmp, &bo_tr).unwrap();
    }

    println!("\n[resident_probe] ✅ GATE PASS: same-resident-kernel bo_h reuse (mm1 C=bo_h -> mm2 A=bo_h)");
    println!("  completed 51× with NO hang and NO BO-bank error. The H-reuse hazard does NOT bite on");
    println!("  one resident kernel. Resident FFN-H is expressible in V2 -> next: bf16-out mm1 mode +");
    println!("  correctness + marshaling-delta measurement.");
}
