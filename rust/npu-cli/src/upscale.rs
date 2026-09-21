//! `npu upscale`: decode a video with ffmpeg, upscale each frame over the control socket's
//! `POST /v1/images/upscale`, re-encode with ffmpeg. One request per frame -- the NPU is
//! single-tenant and the actor must not be held for a whole video (`npu_sr::pipeline::upscale_file`
//! does the same decode/encode shape in-process; this is its client-side, one-frame-at-a-time twin).
use std::io::{Read, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, Command, Stdio};

use anyhow::{bail, Context, Result};

use crate::socket_client;

pub struct Stats {
    pub frames: usize,
    /// The container's own `nb_frames`, when it states one -- not every container does.
    pub total_frames: Option<usize>,
}

/// (width, height, frame-rate string, total frame count if the container states one) of the first
/// video stream, via ffprobe.
fn probe(input: &Path) -> Result<(usize, usize, String, Option<usize>)> {
    let out = Command::new("ffprobe")
        .args([
            "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
            "-of", "csv=p=0",
        ])
        .arg(input)
        .output()
        .context("run ffprobe (is ffmpeg installed?)")?;
    if !out.status.success() {
        bail!("ffprobe failed on {}: {}", input.display(), String::from_utf8_lossy(&out.stderr));
    }
    let s = String::from_utf8_lossy(&out.stdout);
    let line = s.lines().next().unwrap_or("").trim();
    let parts: Vec<&str> = line.split(',').collect();
    if parts.len() < 2 {
        bail!("ffprobe produced no usable stream info for {}: {line:?}", input.display());
    }
    let w: usize = parts[0].trim().parse().with_context(|| format!("width {:?}", parts[0]))?;
    let h: usize = parts[1].trim().parse().with_context(|| format!("height {:?}", parts[1]))?;
    let fps = parts.get(2).map(|s| s.trim().to_string()).unwrap_or_else(|| "25/1".into());
    let total = parts.get(3).and_then(|s| s.trim().parse::<usize>().ok());
    Ok((w, h, fps, total))
}

/// `/v1/images/upscale`'s response body: 4 bytes LE width, 4 bytes LE height, then raw RGB8.
fn decode_image_body(b: &[u8]) -> Result<(usize, usize, &[u8])> {
    if b.len() < 8 {
        bail!("image response too short: {} bytes", b.len());
    }
    let w = u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize;
    let h = u32::from_le_bytes([b[4], b[5], b[6], b[7]]) as usize;
    let px = &b[8..];
    if px.len() != w * h * 3 {
        bail!("image response {w}x{h} disagrees with {} pixel bytes (want {})", px.len(), w * h * 3);
    }
    Ok((w, h, px))
}

/// Decode -> per-frame `/v1/images/upscale` -> encode. The encoder is spawned lazily, sized off
/// the FIRST response's dims -- the server, not this client, knows the model's scale factor.
///
/// Any error mid-loop (a socket call, a malformed response, a broken pipe) is reaped through
/// `reap_after_error` before returning -- `dec`/`enc` stay live captures of this function's own
/// locals, never moved into the loop, so the tail code can still kill/wait them either way.
fn run(input: &Path, out: &Path, model: &str, mut progress: impl FnMut(usize, Option<usize>)) -> Result<Stats> {
    let (w, h, fps, total) = probe(input)?;
    let in_frame_bytes = w * h * 3;

    let mut dec = Command::new("ffmpeg")
        .args(["-v", "error", "-i"]).arg(input)
        .args(["-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
        .stdout(Stdio::piped()).stderr(Stdio::inherit())
        .spawn().context("spawn ffmpeg decode")?;
    let mut dec_out = dec.stdout.take().expect("piped stdout");
    let mut enc: Option<Child> = None;
    let mut enc_in: Option<ChildStdin> = None;

    let loop_result: Result<usize> = (|| {
        let mut frames = 0usize;
        let mut buf = vec![0u8; in_frame_bytes];
        loop {
            match dec_out.read_exact(&mut buf) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e).context("decode read"),
            }
            let resp = socket_client::call_multipart_bytes(
                "/v1/images/upscale",
                &[("model", model), ("w", &w.to_string()), ("h", &h.to_string())],
                "image", "frame.rgb", &buf,
            )?;
            let (ow, oh, out_rgb) = decode_image_body(&resp)?;
            if enc_in.is_none() {
                let mut child = Command::new("ffmpeg")
                    .args(["-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24"])
                    .args(["-s", &format!("{ow}x{oh}"), "-r", &fps, "-i", "-"])
                    .args(["-pix_fmt", "yuv420p"]).arg(out)
                    .stdin(Stdio::piped()).stderr(Stdio::inherit())
                    .spawn().context("spawn ffmpeg encode")?;
                enc_in = child.stdin.take();
                enc = Some(child);
            }
            enc_in.as_mut().expect("just set").write_all(out_rgb).context("encode write")?;
            frames += 1;
            progress(frames, total);
        }
        Ok(frames)
    })();

    let frames = match loop_result {
        Ok(frames) => frames,
        Err(e) => return Err(reap_after_error(e, &mut dec, enc, enc_in)),
    };

    drop(enc_in);
    let dstat = dec.wait().context("decode wait")?;
    let Some(mut enc) = enc else { bail!("{} has no frames", input.display()) };
    let estat = enc.wait().context("encode wait")?;
    if !dstat.success() {
        bail!("ffmpeg decode exited non-zero");
    }
    if !estat.success() {
        bail!("ffmpeg encode exited non-zero");
    }
    Ok(Stats { frames, total_frames: total })
}

/// Reaps the ffmpeg children after a mid-loop error, so a failed upscale never leaks a running (or
/// zombie) process. The decoder is no longer wanted -- killed, then waited; its resulting exit is
/// never reported, because WE caused it (always non-zero/signalled, never diagnostic). The encoder
/// gets its stdin dropped (EOF) instead of being killed, so it finalizes whatever partial output it
/// already has; if IT then exits non-zero on its own, that is real information -- e.g. a
/// `write_all` "broken pipe" is far less useful on its own than knowing the encoder had already
/// exited non-zero -- so that status is folded into the returned error.
fn reap_after_error(
    err: anyhow::Error,
    dec: &mut Child,
    enc: Option<Child>,
    enc_in: Option<ChildStdin>,
) -> anyhow::Error {
    let _ = dec.kill();
    let _ = dec.wait();
    drop(enc_in);
    match enc.map(|mut c| c.wait()) {
        Some(Ok(st)) if !st.success() => err.context(format!("ffmpeg encode also exited non-zero: {st}")),
        _ => err,
    }
}

/// `npu upscale <in> <out> [--net espcn] [--model <name>]`: `--net` names the schedule/model this
/// install serves it under and doubles as the request's `model` field when `--model` is absent;
/// `--model` overrides that when the served name differs.
pub fn run_cli(input: &Path, out: &Path, net: &str, model: Option<&str>) -> Result<()> {
    crate::quiet_one_shot();
    let model = model.unwrap_or(net);
    let stats = run(input, out, model, |done, total| {
        if done % 30 == 0 {
            match total {
                Some(t) => eprintln!("[npu] upscale: {done}/{t} frames"),
                None => eprintln!("[npu] upscale: {done} frames"),
            }
        }
    })?;
    match stats.total_frames {
        Some(t) => println!("{} ({}/{} frames)", out.display(), stats.frames, t),
        None => println!("{} ({} frames)", out.display(), stats.frames),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Real (non-ffmpeg) child processes stand in for the decoder/encoder -- no device, and no
    /// dependency on ffmpeg's own exit-status shape. `sleep` is still running when reaped (so
    /// `kill` has something to do); `false` has already exited non-zero.
    #[test]
    fn reap_after_error_kills_the_decoder_and_folds_in_the_encoders_exit_status() {
        let mut dec = Command::new("sleep").arg("5").spawn().expect("spawn sleep");
        let enc = Command::new("false").spawn().expect("spawn false");
        let orig = anyhow::anyhow!("encode write: broken pipe");
        let err = reap_after_error(orig, &mut dec, Some(enc), None);
        // `{:?}` (what `main.rs`'s top-level error printer uses) renders the whole chain; `context`
        // makes the new layer the OUTERMOST message, so `{}`/`.to_string()` alone would hide the
        // original cause.
        let msg = format!("{err:?}");
        assert!(msg.contains("broken pipe"), "{msg}");
        assert!(msg.contains("encode also exited non-zero"), "{msg}");
        // The decoder was actually killed, not left running: `try_wait` on an already-reaped
        // child returns its cached exit status instead of hanging or erroring.
        assert!(dec.try_wait().expect("decoder was reaped").is_some());
    }

    /// A cleanly-exiting encoder adds no extra context -- the original error passes through
    /// unchanged, not padded with a status nobody needs. The decoder's own status is never
    /// checked (it is always force-killed by this function, so it is never informative) -- `sleep`
    /// here proves that: it would report non-zero/signalled if this function looked at it.
    #[test]
    fn reap_after_error_adds_nothing_when_the_encoder_exits_cleanly() {
        let mut dec = Command::new("sleep").arg("5").spawn().expect("spawn sleep");
        let enc = Command::new("true").spawn().expect("spawn true");
        let orig = anyhow::anyhow!("some other failure");
        let err = reap_after_error(orig, &mut dec, Some(enc), None);
        assert_eq!(err.to_string(), "some other failure");
    }
}
