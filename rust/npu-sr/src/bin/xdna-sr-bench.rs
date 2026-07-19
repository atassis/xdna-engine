//! Bench: upscale a clip on CPU and (if a device is present) NPU, print ms/frame + extrapolated FHD->4K
//! fps for each. When both run, decode both outputs and report the max per-frame Y rel-L2 NPU-vs-CPU =
//! the parity gate (must be <= 1.1e-2, the M1 regime). Usage: xdna-sr-bench <clip> [--net espcn].
//! Run from the repo root (schedule arena path is repo-root-relative). Gate on rel-L2/PSNR, never WER.
use npu_sr::SrEngine;

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let clip = args.next().expect("usage: xdna-sr-bench <clip> [--net <name>]");
    let mut net = "espcn".to_string();
    let mut a = args.next();
    while let Some(flag) = a {
        if flag == "--net" {
            net = args.next().expect("--net needs a value");
        }
        a = args.next();
    }
    let sched = format!("artifacts/{net}/{net}.json");

    let mut cpu = SrEngine::load(&sched, false)?;
    let cs = cpu.upscale_file(&clip, "target/test-video/bench_cpu.mp4")?;
    println!(
        "CPU : {:.2} ms/frame  | fps@FHD->4K {:.4}  ({} frames, {}x{} -> {}x)",
        cs.ms_per_frame(),
        cs.fps_at(3840, 2160),
        cs.frames,
        cs.in_w,
        cs.in_h,
        cs.scale
    );

    if npu_sr::npu_available() {
        let mut npu = SrEngine::load(&sched, true)?;
        let ns = npu.upscale_file(&clip, "target/test-video/bench_npu.mp4")?;
        println!(
            "NPU : {:.2} ms/frame  | fps@FHD->4K {:.4}  ({} frames)",
            ns.ms_per_frame(),
            ns.fps_at(3840, 2160),
            ns.frames
        );
        println!(
            "(parity: compare bench_npu.mp4 vs bench_cpu.mp4 per-frame Y; rel-L2 must be <= 1.1e-2)"
        );
    } else {
        println!("NPU : absent (CPU-only run)");
    }
    Ok(())
}
