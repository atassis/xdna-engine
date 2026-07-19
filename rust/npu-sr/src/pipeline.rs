//! Whole-file decode -> upscale -> encode. Drives the system `ffmpeg` for codec I/O over rawvideo pipes:
//! a decoder subprocess emits RGB24 frames on its stdout, we upscale each via `SrEngine`, and an encoder
//! subprocess consumes the enlarged RGB24 frames on its stdin. This is the double-buffer pipeline (spec
//! §2); the SR engine owns it. (We shell to ffmpeg rather than link libav because the box runs ffmpeg 8.x
//! / libavcodec 62, ahead of the `ffmpeg-next` binding -- a substrate choice; `upscale_file` is unchanged.)
//! The ffmpeg *filter* adapter does NOT use this path (ffmpeg owns decode/encode there).
use crate::{SrEngine, SrError};
use std::io::{Read, Write};
use std::path::Path;
use std::process::{Command, Stdio};

#[derive(Debug, Default, Clone)]
pub struct Stats {
    pub frames: usize,
    pub total_ms: f64, // SR compute only (excludes container I/O)
    pub in_w: usize,
    pub in_h: usize,
    pub scale: usize,
}
impl Stats {
    pub fn ms_per_frame(&self) -> f64 {
        if self.frames == 0 {
            0.0
        } else {
            self.total_ms / self.frames as f64
        }
    }
    /// Extrapolated fps at a target resolution (SR compute scales ~ pixel count).
    pub fn fps_at(&self, target_w: usize, target_h: usize) -> f64 {
        let src_px = (self.in_w * self.in_h).max(1) as f64;
        let tgt_px = (target_w * target_h) as f64;
        let ms = self.ms_per_frame() * (tgt_px / src_px);
        if ms > 0.0 {
            1000.0 / ms
        } else {
            0.0
        }
    }
}

/// Probe (width, height, frame_rate_str) of the first video stream via ffprobe.
fn probe(input: &Path) -> Result<(usize, usize, String), SrError> {
    let out = Command::new("ffprobe")
        .args([
            "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate",
            "-of", "csv=p=0",
        ])
        .arg(input)
        .output()
        .map_err(|e| SrError::Load(format!("ffprobe spawn: {e}")))?;
    if !out.status.success() {
        return Err(SrError::Load(format!(
            "ffprobe failed: {}",
            String::from_utf8_lossy(&out.stderr)
        )));
    }
    let s = String::from_utf8_lossy(&out.stdout);
    let line = s.lines().next().unwrap_or("").trim();
    let parts: Vec<&str> = line.split(',').collect();
    if parts.len() < 2 {
        return Err(SrError::Load(format!("ffprobe bad output: {line:?}")));
    }
    let w: usize = parts[0].trim().parse().map_err(|_| SrError::Load(format!("width {:?}", parts[0])))?;
    let h: usize = parts[1].trim().parse().map_err(|_| SrError::Load(format!("height {:?}", parts[1])))?;
    let fps = parts.get(2).map(|s| s.trim().to_string()).unwrap_or_else(|| "25/1".into());
    Ok((w, h, fps))
}

/// Decode `input`, upscale every frame, encode to `output`. `now_ms` supplies monotonic ms (injected so
/// timing is testable). Times only the SR compute, not the container I/O.
pub fn upscale_file(
    eng: &mut SrEngine,
    input: &Path,
    output: &Path,
    mut now_ms: impl FnMut() -> f64,
) -> Result<Stats, SrError> {
    let (w, h, fps) = probe(input)?;
    let scale = eng.scale();
    let (ow, oh) = (w * scale, h * scale);
    let in_frame_bytes = w * h * 3;
    let out_frame_bytes = ow * oh * 3;

    // Decoder: emit RGB24 frames on stdout.
    let mut dec = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(input)
        .args(["-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|e| SrError::Load(format!("ffmpeg decode spawn: {e}")))?;

    // Encoder: consume RGB24 frames on stdin, write `output`.
    let mut enc = Command::new("ffmpeg")
        .args(["-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24"])
        .args(["-s", &format!("{ow}x{oh}"), "-r", &fps, "-i", "-"])
        .args(["-pix_fmt", "yuv420p"])
        .arg(output)
        .stdin(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|e| SrError::Load(format!("ffmpeg encode spawn: {e}")))?;

    let mut dec_out = dec.stdout.take().unwrap();
    let mut enc_in = enc.stdin.take().unwrap();

    let mut frames = 0usize;
    let mut total = 0.0f64;
    let mut buf = vec![0u8; in_frame_bytes];
    loop {
        match dec_out.read_exact(&mut buf) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(SrError::Frame(format!("decode read: {e}"))),
        }
        let t0 = now_ms();
        let (out_rgb, gw, gh) = eng.upscale_rgb8(&buf, w, h)?;
        total += now_ms() - t0;
        debug_assert_eq!((gw, gh), (ow, oh));
        debug_assert_eq!(out_rgb.len(), out_frame_bytes);
        enc_in
            .write_all(&out_rgb)
            .map_err(|e| SrError::Frame(format!("encode write: {e}")))?;
        frames += 1;
    }
    drop(enc_in); // signal EOF to the encoder
    let dstat = dec.wait().map_err(|e| SrError::Load(format!("decode wait: {e}")))?;
    let estat = enc.wait().map_err(|e| SrError::Load(format!("encode wait: {e}")))?;
    if !dstat.success() {
        return Err(SrError::Load("ffmpeg decode exited non-zero".into()));
    }
    if !estat.success() {
        return Err(SrError::Load("ffmpeg encode exited non-zero".into()));
    }
    Ok(Stats {
        frames,
        total_ms: total,
        in_w: w,
        in_h: h,
        scale,
    })
}
