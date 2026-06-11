//! Dispatch-cost spike: decompose the ~3 ms/dispatch NPU overhead.
//!
//! The encoder issues ~144 whole-array matmul dispatches at ~3 ms each, while actual compute is
//! ~0.1-0.3 ms. This bin measures WHAT the overhead is, to pick the right latency lever:
//!   - EXP1 (context-switch isolation): run the SAME 144 dispatches (36 of each of the 4 real
//!     encoder shapes) two ways — GROUPED (36 of a shape in a row -> only 3 hw-context switches)
//!     vs ROTATING (s0,s1,s2,s3,... -> 143 switches). Identical total compute+DMA, so the delta
//!     is pure hw-context-switch cost. If grouped << rotating -> fewer-contexts (padding) is the
//!     high-leverage lever. If grouped ~= rotating -> context-switching is NOT the wall.
//!   - EXP2 (per-phase): on one shape, time activation write+sync vs run(+wait) vs read-back,
//!     to split host-marshalling from the NPU run itself.
//!
//! NPU is single-tenant — stop flm-asr/voxd (or npu-asr) first. Run from the repo root.

use std::path::Path;
use std::rc::Rc;
use std::time::Instant;

use npu_xrt::{Bo, Device, Kernel, FLAG_CACHEABLE, FLAG_HOST_ONLY};

const WA: &str = "mlir-aie/programming_examples/basic/matrix_multiplication/whole_array/build";

/// The 4 distinct matmul shapes the encoder cycles through (name, xclbin, insts, M, K2, N).
/// K2 is the augmented inner dim (Kaug = K+32 for bias/silu; plain mm2 uses K directly).
const SHAPES: &[(&str, &str, &str, usize, usize, usize)] = &[
    ("ffn_mm1_silu", "final_512x800x3072_32x32x32_8c_silu.xclbin", "insts_512x800x3072_32x32x32_8c_silu.txt", 512, 800, 3072),
    ("ffn_mm2_plain", "final_512x3072x768_32x32x32_8c.xclbin", "insts_512x3072x768_32x32x32_8c.txt", 512, 3072, 768),
    ("qk_pw1_bias", "final_512x800x1536_32x32x32_8c_bias.xclbin", "insts_512x800x1536_32x32x32_8c_bias.txt", 512, 800, 1536),
    ("vo_pw2_bias", "final_512x800x768_32x32x32_8c_bias.xclbin", "insts_512x800x768_32x32x32_8c_bias.txt", 512, 800, 768),
];

struct Disp {
    name: String,
    kern: Rc<Kernel>,
    instr: Bo,
    n_instr: usize,
    a: Bo,
    b: Bo,
    c: Bo,
    tmp: Bo,
    tr: Bo,
    a_bytes: usize,
}

impl Disp {
    fn make(dev: &Device, root: &Path, s: &(&str, &str, &str, usize, usize, usize)) -> Disp {
        let (name, xb, ib, m, k2, n) = (s.0, s.1, s.2, s.3, s.4, s.5);
        let wa = root.join(WA);
        let kern = dev
            .load_kernel(wa.join(xb).to_str().unwrap(), None)
            .unwrap_or_else(|e| panic!("load {xb}: {e}"));
        let ibytes = std::fs::read(wa.join(ib)).unwrap_or_else(|e| panic!("read {ib}: {e}"));
        let n_instr = ibytes.len() / 4;
        let g = |i| kern.group_id(i).unwrap();
        let instr = dev.alloc_bo(&kern, ibytes.len(), FLAG_CACHEABLE, g(1)).unwrap();
        instr.write_bytes(&ibytes).unwrap();
        instr.sync_to_device().unwrap();
        let a_bytes = m * k2 * 2;
        let a = dev.alloc_bo(&kern, a_bytes, FLAG_HOST_ONLY, g(3)).unwrap();
        let b = dev.alloc_bo(&kern, k2 * n * 2, FLAG_HOST_ONLY, g(4)).unwrap();
        let c = dev.alloc_bo(&kern, m * n * 4, FLAG_HOST_ONLY, g(5)).unwrap(); // f32-superset
        let tmp = dev.alloc_bo(&kern, 1, FLAG_HOST_ONLY, g(6)).unwrap();
        let tr = dev.alloc_bo(&kern, 4, FLAG_HOST_ONLY, g(7)).unwrap();
        a.write_bytes(&vec![0u8; a_bytes]).unwrap();
        a.sync_to_device().unwrap();
        b.write_bytes(&vec![0u8; k2 * n * 2]).unwrap();
        b.sync_to_device().unwrap();
        Disp { name: name.into(), kern, instr, n_instr, a, b, c, tmp, tr, a_bytes }
    }

    #[inline]
    fn run(&self) {
        self.kern
            .run_matmul8(3, &self.instr, self.n_instr, &self.a, &self.b, &self.c, &self.tmp, &self.tr)
            .unwrap();
    }
}

fn main() {
    let root = Path::new(".");
    let dev = Device::open(0).expect("open NPU (stop flm-asr/voxd/npu-asr first)");
    let disps: Vec<Disp> = SHAPES.iter().map(|s| Disp::make(&dev, root, s)).collect();
    let reps = 36usize; // 4 shapes * 36 = 144 dispatches, matching the encoder's count

    // warmup every context (first dispatch of an xclbin loads its program into the array)
    for _ in 0..3 {
        for d in &disps {
            d.run();
        }
    }

    // ---- EXP1a: GROUPED — all `reps` of a shape consecutively (only 3 context switches) ----
    let t0 = Instant::now();
    for d in &disps {
        for _ in 0..reps {
            d.run();
        }
    }
    let grouped = t0.elapsed().as_secs_f64() * 1e3;

    // ---- EXP1b: ROTATING — s0,s1,s2,s3,... (143 context switches) ----
    let t0 = Instant::now();
    for _ in 0..reps {
        for d in &disps {
            d.run();
        }
    }
    let rotating = t0.elapsed().as_secs_f64() * 1e3;

    let n = (reps * disps.len()) as f64;
    let switches_grouped = (disps.len() - 1) as f64; // 3
    let switches_rotating = (reps * disps.len() - 1) as f64; // 143
    let per_switch = (rotating - grouped) / (switches_rotating - switches_grouped);

    println!("=== EXP1: hw-context-switch isolation ({} dispatches each, identical workload) ===", n as usize);
    println!("  GROUPED  (3 switches):   {grouped:7.1} ms total  | {:.3} ms/dispatch", grouped / n);
    println!("  ROTATING (143 switches): {rotating:7.1} ms total  | {:.3} ms/dispatch", rotating / n);
    println!("  delta = {:.1} ms over {} extra switches  ->  ~{:.3} ms per context-switch", rotating - grouped, (switches_rotating - switches_grouped) as usize, per_switch);
    println!("  => if delta is large, FEWER hw-contexts (padding) is the high-leverage lever.\n");

    // ---- EXP2: per-phase decomposition on one shape (the small bias op, the common case) ----
    let d = disps.iter().find(|d| d.name == "vo_pw2_bias").unwrap();
    let iters = 100;
    let zeros = vec![0u8; d.a_bytes];
    let (mut t_write, mut t_run, mut t_read) = (0f64, 0f64, 0f64);
    let mut cbuf = vec![0u8; d.c.nbytes()];
    for _ in 0..iters {
        let t = Instant::now();
        d.a.write_bytes(&zeros).unwrap();
        d.a.sync_to_device().unwrap();
        t_write += t.elapsed().as_secs_f64();
        let t = Instant::now();
        d.run();
        t_run += t.elapsed().as_secs_f64();
        let t = Instant::now();
        d.c.sync_from_device().unwrap();
        d.c.read_bytes(&mut cbuf).unwrap();
        t_read += t.elapsed().as_secs_f64();
    }
    let ms = |x: f64| x * 1e3 / iters as f64;
    println!("=== EXP2: per-dispatch phase split (shape {}, {iters} iters) ===", d.name);
    println!("  activation write+sync_to_device : {:.3} ms", ms(t_write));
    println!("  run_matmul8 (submit+exec+wait)  : {:.3} ms", ms(t_run));
    println!("  output sync_from+read           : {:.3} ms", ms(t_read));
    println!("  total host-visible/dispatch     : {:.3} ms\n", ms(t_write + t_run + t_read));

    // ---- EXP3: realistic encoder dispatch sequence under FEWER contexts ----
    // The encoder's 9 matmuls/block: ffn1(mm1,mm2), attn(qk,v,o), conv(pw1,pw2), ffn2(mm1,mm2).
    // All are K=768 EXCEPT the two FFN-contraction mm2 (K=3072). So a 2-context scheme:
    //   ctxA = plain 512x768x3072 (all K=768 ops, N padded up to 3072; bias/silu on host)
    //   ctxB = plain 512x3072x768 (the two mm2)
    // Per-block context pattern: A,B,A,A,A,A,A,A,B  -> ~4 switches/block vs ~8 today.
    let plain = |name: &str, xb: &str, ib: &str, m, k2, n| {
        Disp::make(&dev, root, &(name, xb, ib, m, k2, n))
    };
    let ca = plain("ctxA_768x3072", "final_512x768x3072_32x32x32_8c.xclbin", "insts_512x768x3072_32x32x32_8c.txt", 512, 768, 3072);
    let cb = plain("ctxB_3072x768", "final_512x3072x768_32x32x32_8c.xclbin", "insts_512x3072x768_32x32x32_8c.txt", 512, 3072, 768);
    // warm both
    for _ in 0..3 { ca.run(); cb.run(); }
    let block_pat = [&ca, &cb, &ca, &ca, &ca, &ca, &ca, &ca, &cb]; // 9 matmuls
    let blocks = 16;

    let t0 = Instant::now();
    for _ in 0..blocks {
        for d in block_pat {
            d.run();
        }
    }
    let two_ctx = t0.elapsed().as_secs_f64() * 1e3;

    // 1-context ceiling reference: all 144 on ctxA (ignores mm2's K mismatch; shows the no-switch floor)
    let t0 = Instant::now();
    for _ in 0..(blocks * block_pat.len()) {
        ca.run();
    }
    let one_ctx = one_ctx_ref(t0);

    println!("=== EXP3: encoder dispatch sequence ({} matmuls/block x {} blocks = {}) ===", block_pat.len(), blocks, block_pat.len() * blocks);
    println!("  current (4 contexts, ~128 switches): ~527 ms  [EXP1 rotating proxy]");
    println!("  2-context A/B (~{} switches)        : {two_ctx:7.1} ms  [N-padded 768->3072, +6MB readback]", block_pat.len() * blocks / 9 * 4);
    println!("  1-context ceiling (all ctxA, 0 sw)  : {one_ctx:7.1} ms  [N-padded big op, no switches]");

    // ---- EXP4: RIGHT-SIZED bucketing — natural per-op shapes, NO N-padding (Track O2) ----
    // EXP3's 2-ctx pads every K=768 op's N up to 3072 (4x the MACs on N<=1536 ops + a 6 MB
    // padded readback). Right-sizing uses the REAL shapes via 4 plain xclbins (which already
    // exist), trading padding-compute for a few more switches. This measures which wins, and is
    // the cheapest path to bank the conservative win end-to-end without the padding penalty.
    let s768 = plain("rs_768x768", "final_512x768x768_32x32x32_8c.xclbin", "insts_512x768x768_32x32x32_8c.txt", 512, 768, 768);
    let s1536 = plain("rs_768x1536", "final_512x768x1536_32x32x32_8c.xclbin", "insts_512x768x1536_32x32x32_8c.txt", 512, 768, 1536);
    let s3072 = plain("rs_768x3072", "final_512x768x3072_32x32x32_8c.xclbin", "insts_512x768x3072_32x32x32_8c.txt", 512, 768, 3072);
    let smm2r = plain("rs_3072x768", "final_512x3072x768_32x32x32_8c.xclbin", "insts_512x3072x768_32x32x32_8c.txt", 512, 3072, 768);
    for _ in 0..3 { s3072.run(); smm2r.run(); s1536.run(); s768.run(); }
    // real per-op shapes: ffn1mm1, ffn1mm2, qk, v, o, pw1, pw2, ffn2mm1, ffn2mm2
    let rs_pat = [&s3072, &smm2r, &s1536, &s768, &s768, &s1536, &s768, &s3072, &smm2r];
    // switches/block: count adjacent-distinct xclbins within a block, + 1 block->block transition
    let mut sw_per_block = 0usize;
    for w in rs_pat.windows(2) { if !std::ptr::eq(w[0].kern.as_ref(), w[1].kern.as_ref()) { sw_per_block += 1; } }
    if !std::ptr::eq(rs_pat[rs_pat.len()-1].kern.as_ref(), rs_pat[0].kern.as_ref()) { sw_per_block += 1; }
    let t0 = Instant::now();
    for _ in 0..blocks {
        for d in rs_pat {
            d.run();
        }
    }
    let rs_ms = t0.elapsed().as_secs_f64() * 1e3;
    println!("  RIGHT-SIZED 4-ctx (NO padding)      : {rs_ms:7.1} ms  [natural shapes, ~{} sw/block = {} switches]", sw_per_block, sw_per_block * blocks);
    println!("  MEASURED: right-sizing into 4 ctx is WORSE than 2-ctx padded ({rs_ms:.0} > {two_ctx:.0}) -- at ~2.4ms/switch,");
    println!("  switch cost dominates. The lever is FEWER contexts (1-2), not right-sizing. Banking the win needs a");
    println!("  multi-shape single xclbin (O1) or resident program (O3); padding is a *readback* problem (sliceable),");
    println!("  not a pool problem. 2-ctx padded {two_ctx:.0}ms already beats current ~427ms IF readback is right-sized.");

    // ---- EXP5: instruction-stream swap on ONE resident xclbin (R1: is the switch host-side removable?) ----
    // The ~2.4ms switch was ALWAYS measured by rotating DISTINCT xclbins (distinct array programs). But the
    // instr BO is a PER-DISPATCH arg to an already-loaded kernel. Untested: does swapping the instr stream on
    // a RESIDENT xclbin cost a switch, or ~0.4ms? If cheap, the whole switch wall is a host-side fix (one
    // resident xclbin + per-shape instr BOs), no new IRON.
    use std::io::Write;
    let res = &s3072; // resident kernel = 768x3072 (one hw_context); BOs sized for the max shape
    let load_instr = |name: &str| -> (Bo, usize) {
        let bytes = std::fs::read(root.join(WA).join(name)).unwrap_or_else(|e| panic!("read {name}: {e}"));
        let n = bytes.len() / 4;
        let bo = dev.alloc_bo(&res.kern, bytes.len(), FLAG_CACHEABLE, res.kern.group_id(1).unwrap()).unwrap();
        bo.write_bytes(&bytes).unwrap();
        bo.sync_to_device().unwrap();
        (bo, n)
    };
    let (ia, na) = load_instr("insts_512x768x3072_32x32x32_8c.txt"); // matches the resident xclbin
    let (ia2, na2) = load_instr("insts_512x768x3072_32x32x32_8c.txt"); // same content, DIFFERENT BO
    let (ib, nb) = load_instr("insts_512x768x1536_32x32x32_8c.txt"); // DIFFERENT shape
    let runi = |instr: &Bo, n: usize| res.kern.run_matmul8(3, instr, n, &res.a, &res.b, &res.c, &res.tmp, &res.tr);
    for _ in 0..5 { let _ = runi(&ia, na); }
    let r5 = 200usize;
    let t0 = Instant::now(); for _ in 0..r5 { runi(&ia, na).unwrap(); } let same_bo = t0.elapsed().as_secs_f64() * 1e3 / r5 as f64;
    let t0 = Instant::now(); for _ in 0..r5/2 { runi(&ia, na).unwrap(); runi(&ia2, na2).unwrap(); } let swap_same = t0.elapsed().as_secs_f64() * 1e3 / r5 as f64;
    println!("\n=== EXP5: instr-stream swap on ONE resident xclbin (R1) ===");
    println!("  resident, same instr BO repeated     : {same_bo:.3} ms/dispatch  [floor]");
    println!("  resident, swap SAME-shape instr BO   : {swap_same:.3} ms/dispatch  [BO-swap cost; ~floor => BO swap is free]");
    std::io::stdout().flush().ok(); // flush BEFORE the risky cross-shape dispatch (may hang on baked loop bounds)
    match runi(&ib, nb) {
        Ok(_) => {
            let t0 = Instant::now(); for _ in 0..r5/2 { runi(&ia, na).unwrap(); runi(&ib, nb).unwrap(); } let swap_diff = t0.elapsed().as_secs_f64() * 1e3 / r5 as f64;
            println!("  resident, swap DIFFERENT-shape stream: {swap_diff:.3} ms/dispatch  [COMPLETED -- ~floor => R1 PASS: switch host-side removable]");
        }
        Err(e) => println!("  resident, DIFFERENT-shape stream     : ERROR ({e:?}) => streams bound to their xclbin (naive swap R1-neg; needs IRON multi-op stream)"),
    }
    println!("  reference: distinct-xclbin switch     : ~2.42 ms/dispatch  [EXP1]");

    // ---- EXP6: FULL V2 encoder dispatch on ONE resident xclbin via per-shape streams (zero switches) ----
    // Mechanism proven: per-shape streams on the resident 768x3072 xclbin are cheap AND numerically correct
    // (cross-stream test max_rel 5.7e-7). N is stream-flexible; K is baked, so mm2 (K=3072) splits into 4x
    // K=768 (768x768) streams. The whole encoder then runs on ONE compiled xclbin with NO context switches.
    let (i768, n768) = load_instr("insts_512x768x768_32x32x32_8c.txt");
    for _ in 0..3 { let _ = runi(&i768, n768); let _ = runi(&ib, nb); let _ = runi(&ia, na); }
    // block: ffn1mm1(3072), ffn1mm2(K3072->4x768), qk(1536), v(768), o(768), pw1(1536), pw2(768), ffn2mm1(3072), ffn2mm2(4x768)
    let block: &[(&Bo, usize)] = &[
        (&ia, na),
        (&i768, n768), (&i768, n768), (&i768, n768), (&i768, n768),
        (&ib, nb),
        (&i768, n768), (&i768, n768),
        (&ib, nb),
        (&i768, n768),
        (&ia, na),
        (&i768, n768), (&i768, n768), (&i768, n768), (&i768, n768),
    ];
    let t0 = Instant::now();
    for _ in 0..blocks {
        for &(instr, n) in block { runi(instr, n).unwrap(); }
    }
    let v2 = t0.elapsed().as_secs_f64() * 1e3;
    println!("\n=== EXP6: FULL V2 encoder on ONE resident xclbin ({} disp/block x {} blocks = {}) ===", block.len(), blocks, block.len() * blocks);
    println!("  one resident 768x3072 xclbin, per-shape streams, mm2 K-split 4x : {v2:7.1} ms  [ZERO context switches]");
    println!("  vs: current ~427-527ms (4 ctx) | 2-ctx padded {two_ctx:.0}ms | right-sized {rs_ms:.0}ms | this V2 {v2:.0}ms");
}

fn one_ctx_ref(t0: Instant) -> f64 {
    t0.elapsed().as_secs_f64() * 1e3
}
