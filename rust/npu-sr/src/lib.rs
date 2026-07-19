//! NPU super-resolution engine. Stable frame-in/frame-out ABI over the XDNA2 NPU.
//! The durable interface (`SrEngine`) is designed once; the CLI + ffmpeg filter are thin adapters.

pub mod schedule;
pub mod color;
pub mod frontier;
pub mod pipeline;

use std::path::Path;

/// Engine error surface (mirrors `npu_engine::api::EngineError` style).
#[derive(thiserror::Error, Debug)]
pub enum SrError {
    #[error("no XDNA2 NPU device available")]
    NotAvailable,
    #[error("load failed: {0}")]
    Load(String),
    #[error("device error: {0}")]
    Device(String),
    #[error("bad frame: {0}")]
    Frame(String),
}

/// A planar single-channel image (the luma plane the net upscales). Row-major, f32 in [0,1].
#[derive(Clone)]
pub struct Plane {
    pub w: usize,
    pub h: usize,
    pub data: Vec<f32>,
}

/// The SR engine. Holds device resources -> NOT Send/Sync; the caller serializes (NPU single-tenant).
pub struct SrEngine {
    sched: schedule::Schedule,
    frontier: frontier::Frontier,
}

impl SrEngine {
    /// Load a schedule (espcn.json) + its baked weights arena. `use_npu`=false forces the CPU frontier.
    pub fn load(schedule_path: impl AsRef<Path>, use_npu: bool) -> Result<SrEngine, SrError> {
        let sched = schedule::Schedule::load(schedule_path.as_ref())?;
        let frontier = frontier::Frontier::build(&sched, use_npu)?;
        Ok(SrEngine { sched, frontier })
    }

    /// Upscale one luma plane by the schedule's scale factor. The per-frame ABI (the ffmpeg filter uses this).
    pub fn upscale_plane(&mut self, y: &Plane) -> Result<Plane, SrError> {
        self.frontier.run(&self.sched, y)
    }

    /// Upscale an interleaved RGB8 image (Y-only SR + bicubic chroma). Returns (rgb8, out_w, out_h).
    pub fn upscale_rgb8(&mut self, rgb: &[u8], w: usize, h: usize)
        -> Result<(Vec<u8>, usize, usize), SrError> {
        if rgb.len() != w * h * 3 {
            return Err(SrError::Frame(format!("rgb len {} != {}*{}*3", rgb.len(), w, h)));
        }
        let (y, cb, cr) = color::rgb8_to_ycbcr(rgb, w, h);
        let sr_y = self.upscale_plane(&y)?;
        let (ow, oh) = (sr_y.w, sr_y.h);
        let sr_cb = color::bicubic(&cb, ow, oh);
        let sr_cr = color::bicubic(&cr, ow, oh);
        Ok((color::ycbcr_to_rgb8(&sr_y, &sr_cb, &sr_cr), ow, oh))
    }

    /// The schedule's integer scale factor (e.g. 3 for ESPCN x3).
    pub fn scale(&self) -> usize {
        self.sched.scale
    }

    /// Upscale a whole video file (the CLI path): decode -> upscale -> encode via ffmpeg. Returns timing.
    pub fn upscale_file(
        &mut self,
        input: impl AsRef<Path>,
        output: impl AsRef<Path>,
    ) -> Result<pipeline::Stats, SrError> {
        let start = std::time::Instant::now();
        pipeline::upscale_file(self, input.as_ref(), output.as_ref(), move || {
            start.elapsed().as_secs_f64() * 1000.0
        })
    }
}

/// True if an XDNA2 NPU device node is present (cheap file check; mirrors `npu_engine::Engine::available`).
pub fn npu_available() -> bool {
    std::path::Path::new("/dev/accel/accel0").exists()
}
