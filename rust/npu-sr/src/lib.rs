//! NPU super-resolution engine. Stable frame-in/frame-out ABI over the XDNA2 NPU.
//! The durable interface (`SrEngine`) is designed once; the CLI + ffmpeg filter are thin adapters.

pub mod schedule;
pub mod color;
pub mod frontier;
pub mod pipeline;

use std::path::Path;

use npu_engine::EngineError;

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

/// Paths [`SrEngine::load_with`] resolves itself, replacing the schedule's own CWD-relative
/// defaults (which assume the dev checkout layout -- `artifacts/`, `mlir-aie/` beside the process's
/// cwd -- and have no meaning for an installed service). Both default to `load`'s existing
/// behavior when left `None`: the whole_array dir from `frontier`'s hardcoded dev path, the
/// checkpoint from the schedule JSON's own `checkpoint` field.
#[derive(Default, Clone, Copy)]
pub struct LoadOverrides<'a> {
    /// The whole_array kernel build dir the NPU frontier loads its xclbin from.
    pub wa_dir: Option<&'a Path>,
    /// Overrides the schedule's own `checkpoint` field. The schedule's field is a path relative to
    /// the repo root the schedule was authored against, not the engine root a service runs with --
    /// the caller resolves the real path (e.g. against the engine root) and hands it here.
    pub checkpoint: Option<&'a Path>,
}

impl SrEngine {
    /// Load a schedule (espcn.json) + its baked weights checkpoint. `use_npu`=false forces the CPU frontier.
    pub fn load(schedule_path: impl AsRef<Path>, use_npu: bool) -> Result<SrEngine, SrError> {
        Self::load_with(schedule_path, use_npu, LoadOverrides::default())
    }

    /// Like `load`, with explicit overrides a caller resolved itself instead of the schedule's own
    /// (CWD-relative, dev-checkout-only) paths. See [`LoadOverrides`].
    pub fn load_with(schedule_path: impl AsRef<Path>, use_npu: bool, overrides: LoadOverrides)
        -> Result<SrEngine, SrError> {
        let mut sched = schedule::Schedule::load(schedule_path.as_ref())?;
        if let Some(ckpt) = overrides.checkpoint {
            sched.checkpoint = ckpt.to_string_lossy().into_owned();
        }
        let frontier = frontier::Frontier::build(&sched, use_npu, overrides.wa_dir)?;
        Ok(SrEngine { sched, frontier })
    }

    /// Upscale one luma plane by the schedule's scale factor. The per-frame ABI (the ffmpeg filter uses this).
    pub fn upscale_plane(&mut self, y: &Plane) -> Result<Plane, SrError> {
        self.frontier.run(&self.sched, y)
    }

    /// Upscale an interleaved RGB8 image. Y-only nets (ESPCN): SR the luma + bicubic chroma. RGB nets
    /// (EDSR): all three planar channels through the net. Returns (rgb8, out_w, out_h).
    pub fn upscale_rgb8(&mut self, rgb: &[u8], w: usize, h: usize)
        -> Result<(Vec<u8>, usize, usize), SrError> {
        if rgb.len() != w * h * 3 {
            return Err(SrError::Frame(format!("rgb len {} != {}*{}*3", rgb.len(), w, h)));
        }
        match self.sched.input {
            schedule::InputMode::Y => {
                let (y, cb, cr) = color::rgb8_to_ycbcr(rgb, w, h);
                let sr_y = self.upscale_plane(&y)?;
                let (ow, oh) = (sr_y.w, sr_y.h);
                let sr_cb = color::bicubic(&cb, ow, oh);
                let sr_cr = color::bicubic(&cr, ow, oh);
                Ok((color::ycbcr_to_rgb8(&sr_y, &sr_cb, &sr_cr), ow, oh))
            }
            schedule::InputMode::Rgb => {
                // Interleaved RGB8 -> planar [3,H,W] f32 in [0,1] -> net -> planar [3,oH,oW] -> RGB8.
                let mut planar = vec![0f32; 3 * w * h];
                for i in 0..w * h {
                    planar[i] = rgb[3 * i] as f32 / 255.0;
                    planar[w * h + i] = rgb[3 * i + 1] as f32 / 255.0;
                    planar[2 * w * h + i] = rgb[3 * i + 2] as f32 / 255.0;
                }
                let out = self.frontier.run_feat_planar(&self.sched, planar, 3, h, w)?;
                let (oc, oh, ow, data) = out;
                if oc != 3 {
                    return Err(SrError::Frame(format!("rgb net produced {oc} channels, want 3")));
                }
                let mut rgb_out = vec![0u8; ow * oh * 3];
                for i in 0..ow * oh {
                    for c in 0..3 {
                        rgb_out[3 * i + c] =
                            (data[c * ow * oh + i] * 255.0).round().clamp(0.0, 255.0) as u8;
                    }
                }
                Ok((rgb_out, ow, oh))
            }
        }
    }

    /// The schedule's integer scale factor (e.g. 3 for ESPCN x3).
    pub fn scale(&self) -> usize {
        self.sched.scale
    }

    /// RGB f32 planar path: [3,H,W] row-major in [0,1] -> ([3,oH,oW], oW, oH). For RGB nets (EDSR);
    /// used by the parity gate to feed the exact f32 oracle input (no u8 round-trip).
    pub fn upscale_planar_rgb(&mut self, planar: &[f32], w: usize, h: usize)
        -> Result<(Vec<f32>, usize, usize), SrError> {
        if planar.len() != 3 * w * h {
            return Err(SrError::Frame(format!("planar len {} != 3*{}*{}", planar.len(), w, h)));
        }
        let (oc, oh, ow, data) = self.frontier.run_feat_planar(&self.sched, planar.to_vec(), 3, h, w)?;
        if oc != 3 {
            return Err(SrError::Frame(format!("rgb net produced {oc} channels, want 3")));
        }
        Ok((data, ow, oh))
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

// image in, image out, genuinely &mut self. Routed through npu-runtime's `EngineLoader` for
// scenario kind `image-sr`, not npu_engine's own `ModelKind`/`Scenario` (npu_engine cannot depend
// on npu_sr). `npu_sr` already depends on `npu_engine` (for `esm::native`'s conv-as-GEMM rail,
// `frontier.rs`), so this adapter costs no new Cargo dependency.
impl npu_engine::capability::Servable for SrEngine {
    fn capabilities(&self) -> npu_engine::capability::Capability {
        npu_engine::capability::Capability::IMAGE_SR
    }
    fn run(&mut self, req: npu_engine::capability::Request) -> Result<npu_engine::capability::Response, EngineError> {
        use npu_engine::capability::{Request, Response};
        match req {
            Request::Image { rgb, w, h } => {
                // SrError -> EngineError: no `From` impl (orphan rule -- neither type is local to
                // this crate for that trait), so the mapping is inline here. `to_string()` loses
                // SrError's variant identity (NotAvailable/Load/Device/Frame all fold to `Device`);
                // acceptable for a probe, a real integration would want EngineError variants that
                // actually distinguish them.
                let (rgb, w, h) = self.upscale_rgb8(&rgb, w, h).map_err(|e| EngineError::Device(e.to_string()))?;
                Ok(Response::Image { rgb, w, h })
            }
            other => Err(EngineError::Unsupported(
                format!("SrEngine: expected an image request, got {}", other.shape()))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A schedule with no conv ops: `Frontier::build` still reads the checkpoint unconditionally
    /// (before touching any op or any device), so this fails on the checkpoint alone -- no device,
    /// no real checkpoint, no `set_current_dir` needed.
    fn write_schedule(dir: &std::path::Path, checkpoint: &str) -> std::path::PathBuf {
        let p = dir.join("sched.json");
        std::fs::write(&p, format!(r#"{{"name":"t","scale":1,"checkpoint":"{checkpoint}","ops":[]}}"#)).unwrap();
        p
    }

    /// `load_with`'s `checkpoint` override REPLACES the schedule's own field rather than
    /// supplementing it -- proven with two different missing paths so the error names the one
    /// actually attempted.
    #[test]
    fn load_with_checkpoint_override_replaces_the_schedules_own_field() {
        let dir = tempfile::tempdir().unwrap();
        let sched = write_schedule(dir.path(), "schedule-owns-this.safetensors");

        // `unwrap_err` needs `T: Debug`; `SrEngine` has none (it holds device-frontier state), so
        // `.err()` instead.
        let err = SrEngine::load(&sched, false).err().expect("a missing checkpoint must fail to load");
        assert!(err.to_string().contains("schedule-owns-this.safetensors"), "{err}");

        let override_path = dir.path().join("caller-resolved.safetensors");
        let overrides = LoadOverrides { checkpoint: Some(override_path.as_path()), ..Default::default() };
        let err = SrEngine::load_with(&sched, false, overrides)
            .err().expect("a missing checkpoint must fail to load");
        let msg = err.to_string();
        assert!(msg.contains("caller-resolved.safetensors"), "{msg}");
        assert!(!msg.contains("schedule-owns-this.safetensors"), "{msg}");
    }

    /// No override (`load`'s own path, and `load_with` given the default overrides) keeps the
    /// schedule's own `checkpoint` field untouched.
    #[test]
    fn no_checkpoint_override_keeps_the_schedules_own_field() {
        let dir = tempfile::tempdir().unwrap();
        let sched = write_schedule(dir.path(), "schedule-owns-this.safetensors");
        let err = SrEngine::load_with(&sched, false, LoadOverrides::default())
            .err().expect("a missing checkpoint must fail to load");
        assert!(err.to_string().contains("schedule-owns-this.safetensors"), "{err}");
    }
}
