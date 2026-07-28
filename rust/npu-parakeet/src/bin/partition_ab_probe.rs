//! Partition-width A/B: does claiming FEWER columns change a brick's cost, and does it buy
//! co-residency?
//!
//! `ctxln` is the one encoder brick whose shim-DMA footprint fits inside a narrower partition
//! (16 DDR-facing channels -> 4 columns). Built twice from IDENTICAL source -- same 8 workers, same
//! kernel, same instruction stream length -- differing only in `partition.column_width` (8 vs 4).
//! So any solo-cost difference is the cost of the narrower claim itself, and any alternating-cost
//! difference is co-residency.
//!
//! Arms:
//!   SOLO ctxln8 / SOLO ctxln4      -> is narrowing free?
//!   ALT ctxln8 <-> affcast (8-col) -> today's switch cost
//!   ALT ctxln4 <-> affcast (8-col) -> affcast still claims 8, so this should NOT improve
//!   ALT ctxln4 <-> ctxln4          -> same partition, sanity floor
//!
//! NPU is single-tenant: stop npu-asr/flm-asr/voxd first.
//! Usage: partition_ab_probe <repo_root>

use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::Instant;

use npu_xrt::{Bo, Device, Kernel, FLAG_CACHEABLE, FLAG_HOST_ONLY};

const LN: &str = "artifacts/parakeet/ln";
const PAD_M: usize = 512;
const KRES: usize = 1024;

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
        Brick {
            label: xclbin.file_stem().unwrap().to_string_lossy().to_string(),
            a: dev.alloc_bo(&kern, sizes[0], FLAG_HOST_ONLY, g(3)).unwrap(),
            b: dev.alloc_bo(&kern, sizes[1], FLAG_HOST_ONLY, g(4)).unwrap(),
            c: dev.alloc_bo(&kern, sizes[2], FLAG_HOST_ONLY, g(5)).unwrap(),
            tmp: dev.alloc_bo(&kern, sizes[3], FLAG_HOST_ONLY, g(6)).unwrap(),
            tr: dev.alloc_bo(&kern, sizes[4], FLAG_HOST_ONLY, g(7)).unwrap(),
            kern, instr, n_instr,
        }
    }
    fn dispatch(&self) {
        self.kern
            .run_matmul8(3, &self.instr, self.n_instr, &self.a, &self.b, &self.c, &self.tmp, &self.tr)
            .unwrap();
    }
}

fn time_alt(bricks: &[&Brick], n: usize, reps: usize) -> f64 {
    let mut best = f64::MAX;
    for _ in 0..reps {
        let t = Instant::now();
        for i in 0..n {
            bricks[i % bricks.len()].dispatch();
        }
        best = best.min(t.elapsed().as_secs_f64());
    }
    best * 1e3 / n as f64
}

fn main() {
    let root = PathBuf::from(std::env::args().nth(1).expect("usage: partition_ab_probe <root>"));
    let ln = root.join(LN);
    let reps: usize = std::env::var("PROBE_REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(3);
    const N: usize = 64;

    let dev = Device::open(0).expect("open NPU");
    let ln_sizes = [PAD_M * KRES * 4, PAD_M * KRES * 4, 1, 8, 1];
    let ctxln8 = Brick::load(&dev, &ln.join("final_ctxln_512x1024.xclbin"), &ln.join("insts_ctxln_512x1024.txt"), ln_sizes);
    let ctxln4 = Brick::load(&dev, &ln.join("final_ctxln_512x1024_p4c.xclbin"), &ln.join("insts_ctxln_512x1024_p4c.txt"), ln_sizes);
    let affcast = Brick::load(
        &dev,
        &ln.join("final_affcast_512x1024.xclbin"),
        &ln.join("insts_affcast_512x1024.txt"),
        [PAD_M * KRES * 4, 2 * KRES * 4, PAD_M * KRES * 2, 8, 1],
    );

    println!("== SOLO (zero transitions) ==");
    for b in [&ctxln8, &ctxln4, &affcast] {
        println!("  {:<30} {:.3} ms/dispatch", b.label, time_alt(&[b], N, reps));
    }
    println!("\n== ALTERNATING (one transition per dispatch) ==");
    let cast8 = Brick::load(&dev, &ln.join("final_cast_512x1024.xclbin"), &ln.join("insts_cast_512x1024.txt"), ln_sizes);
    let cast4 = Brick::load(&dev, &ln.join("final_cast_512x1024_p4c.xclbin"), &ln.join("insts_cast_512x1024_p4c.txt"), ln_sizes);
    let ctxln4w = Brick::load(&dev, &ln.join("final_ctxln_512x1024_p4cW.xclbin"), &ln.join("insts_ctxln_512x1024_p4cW.txt"), ln_sizes);
    let cast4w = Brick::load(&dev, &ln.join("final_cast_512x1024_p4cW.xclbin"), &ln.join("insts_cast_512x1024_p4cW.txt"), ln_sizes);
    let cases: [(&str, Vec<&Brick>); 6] = [
        ("ctxln(8col) <-> affcast(8col)", vec![&ctxln8, &affcast]),
        ("ctxln(4col) <-> affcast(8col)", vec![&ctxln4, &affcast]),
        ("ctxln(8col) <-> ctxln(4col)  ", vec![&ctxln8, &ctxln4]),
        ("ctxln(8col) <-> cast(8col)   ", vec![&ctxln8, &cast8]),
        // THE TEST: two 4-col designs. 4+4 fits in 8 columns, so IF the solver could place them
        // side by side there would be no reprogram. Both declare start_columns=[1], so it cannot.
        ("ctxln(4col) <-> cast(4col)   ", vec![&ctxln4, &cast4]),
        // Same two designs, start_columns widened 0..4 so the driver's column solver can place
        // them DISJOINTLY (4+4 = 8). If co-residency is real, the switch cost collapses here.
        ("ctxln(4colW) <-> cast(4colW) ", vec![&ctxln4w, &cast4w]),
    ];
    for (name, bs) in cases {
        let alt = time_alt(&bs, N, reps);
        let solo: f64 = bs.iter().map(|b| time_alt(&[b], N, reps)).sum::<f64>() / bs.len() as f64;
        println!("  {name}  {alt:.3} ms/dispatch   solo-mean {solo:.3}   => switch {:.3} ms", alt - solo);
    }
}
