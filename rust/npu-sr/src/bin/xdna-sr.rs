//! xdna-sr: offline video upscaler. `xdna-sr in.mp4 out.mp4 [--net espcn] [--cpu] [--bench]`.
//! A thin adapter over the npu-sr engine library (SrEngine::upscale_file). Run from the repo root so the
//! schedule's relative arena path resolves.
use clap::Parser;
use npu_sr::SrEngine;

#[derive(Parser)]
#[command(about = "NPU video upscaler (offline file->file)")]
struct Args {
    input: String,
    output: String,
    /// Schedule name under artifacts/<net>/<net>.json
    #[arg(long, default_value = "espcn")]
    net: String,
    /// Force the CPU frontier (default: use the NPU if a device is present)
    #[arg(long)]
    cpu: bool,
    /// Print ms/frame + extrapolated FHD->4K fps
    #[arg(long)]
    bench: bool,
}

fn main() -> anyhow::Result<()> {
    let a = Args::parse();
    let sched = format!("artifacts/{}/{}.json", a.net, a.net);
    let use_npu = !a.cpu && npu_sr::npu_available();
    let mut eng = SrEngine::load(&sched, use_npu)?;
    let stats = eng.upscale_file(&a.input, &a.output)?;
    eprintln!(
        "upscaled {} frames -> {} ({}x, backend: {})",
        stats.frames,
        a.output,
        eng.scale(),
        if use_npu { "npu" } else { "cpu" }
    );
    if a.bench {
        eprintln!(
            "ms/frame = {:.2}  |  extrapolated fps @ FHD->4K = {:.2}",
            stats.ms_per_frame(),
            stats.fps_at(3840, 2160)
        );
    }
    Ok(())
}
