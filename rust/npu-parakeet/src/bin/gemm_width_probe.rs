//! GEMM column-width tradeoff: what does narrowing the whole_array GEMM cost?
//!
//! The modal GEMM is the ONE encoder design that genuinely uses all 8 columns, and it is on 80% of
//! the measured per-clip hw-context transitions. Narrowing it is the only way the elementwise bricks
//! could ever sit beside it -- so the question is what the narrower GEMM costs on its own.
//!
//! Same M/K/N (512x1024x4096), same tile (64x32x128), same kernel and epilogue; only `n_aie_cols`
//! differs, so cores go 32 -> 16 -> 8. Reports solo ms/dispatch and the implied MAC/cycle/core, plus
//! the alternating cost against a brick (which should NOT move, since all three still claim an
//! 8-column PARTITION -- narrowing compute does not narrow the partition).
//!
//! NPU is single-tenant: stop npu-asr/flm-asr/voxd first.
//! Usage: gemm_width_probe <repo_root>

use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::Instant;

use npu_xrt::{Bo, Device, Kernel, FLAG_CACHEABLE, FLAG_HOST_ONLY};

const WA: &str = "mlir-aie/programming_examples/basic/matrix_multiplication/whole_array/build";
const LN: &str = "artifacts/parakeet/ln";
const M: usize = 512;
const K: usize = 1024;
const N: usize = 4096;
/// AIE core clock, resolved (see HARDWARE CONSTANTS): ~1.53 GHz.
const CLK_GHZ: f64 = 1.53;

struct Brick {
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
    fn load(dev: &Device, xclbin: &Path, insts: &Path, s: [usize; 5]) -> Brick {
        let kern = dev
            .load_kernel(xclbin.to_str().unwrap(), None)
            .unwrap_or_else(|e| panic!("load {}: {e}", xclbin.display()));
        let ib = std::fs::read(insts).unwrap_or_else(|e| panic!("read {}: {e}", insts.display()));
        let g = |i| kern.group_id(i).unwrap();
        let instr = dev.alloc_bo(&kern, ib.len(), FLAG_CACHEABLE, g(1)).unwrap();
        instr.write_bytes(&ib).unwrap();
        instr.sync_to_device().unwrap();
        Brick {
            n_instr: ib.len() / 4,
            a: dev.alloc_bo(&kern, s[0], FLAG_HOST_ONLY, g(3)).unwrap(),
            b: dev.alloc_bo(&kern, s[1], FLAG_HOST_ONLY, g(4)).unwrap(),
            c: dev.alloc_bo(&kern, s[2], FLAG_HOST_ONLY, g(5)).unwrap(),
            tmp: dev.alloc_bo(&kern, s[3], FLAG_HOST_ONLY, g(6)).unwrap(),
            tr: dev.alloc_bo(&kern, s[4], FLAG_HOST_ONLY, g(7)).unwrap(),
            kern, instr,
        }
    }
    fn dispatch(&self) {
        self.kern
            .run_matmul8(3, &self.instr, self.n_instr, &self.a, &self.b, &self.c, &self.tmp, &self.tr)
            .unwrap();
    }
}

fn time_seq(bs: &[&Brick], n: usize, reps: usize) -> f64 {
    let mut best = f64::MAX;
    for _ in 0..reps {
        let t = Instant::now();
        for i in 0..n {
            bs[i % bs.len()].dispatch();
        }
        best = best.min(t.elapsed().as_secs_f64());
    }
    best * 1e3 / n as f64
}

fn main() {
    let root = PathBuf::from(std::env::args().nth(1).expect("usage: gemm_width_probe <root>"));
    let (wa, ln) = (root.join(WA), root.join(LN));
    let reps: usize = std::env::var("PROBE_REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(3);
    const REP: usize = 32;

    let dev = Device::open(0).expect("open NPU");
    let gemm_sizes = [M * K * 2, K * N * 2, M * N * 4, 1, 4];
    let gemms: Vec<(usize, Brick)> = [2usize, 4, 8]
        .iter()
        .map(|&c| {
            let t = format!("512x1024x4096_64x32x128_{c}c_modalsilu");
            (c, Brick::load(&dev, &wa.join(format!("final_{t}.xclbin")), &wa.join(format!("insts_{t}.txt")), gemm_sizes))
        })
        .collect();
    let ctxln = Brick::load(
        &dev,
        &ln.join("final_ctxln_512x1024.xclbin"),
        &ln.join("insts_ctxln_512x1024.txt"),
        [M * K * 4, M * K * 4, 1, 8, 1],
    );

    // 512x1024x4096 = 2.147 GMAC per dispatch.
    let gmac = (M * K * N) as f64 / 1e9;
    println!("== modal GEMM 512x1024x4096, solo (zero transitions) ==");
    println!("  {:<6} {:>6} {:>12} {:>10} {:>14} {:>12}", "cols", "cores", "ms/dispatch", "vs 8col", "GMAC/s", "MAC/cyc/core");
    let base = time_seq(&[&gemms[2].1], REP, reps);
    for (c, g) in &gemms {
        let ms = time_seq(&[g], REP, reps);
        let cores = c * 4;
        let mac_cyc = gmac * 1e9 / (ms / 1e3) / (CLK_GHZ * 1e9) / cores as f64;
        println!("  {:<6} {:>6} {:>12.3} {:>9.2}x {:>14.1} {:>12.1}", c, cores, ms, ms / base, gmac / (ms / 1e3), mac_cyc);
    }

    println!("\n== alternating GEMM <-> ctxln (all still claim an 8-column PARTITION) ==");
    for (c, g) in &gemms {
        let alt = time_seq(&[g, &ctxln], REP * 2, reps);
        let solo = (time_seq(&[g], REP, reps) + time_seq(&[&ctxln], REP, reps)) / 2.0;
        println!("  GEMM({c}col) <-> ctxln   alt {alt:.3} ms   solo-mean {solo:.3}   => switch {:.3} ms", alt - solo);
    }
}
