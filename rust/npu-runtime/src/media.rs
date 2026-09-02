//! Media -> 16 kHz mono PCM, the one shape every ASR model in this engine consumes.
//!
//! The transcription surfaces used to take a 16 kHz mono 16-bit WAV and nothing else: any other
//! sample rate, a stereo file, or a video container was a 400 with no way to proceed. Everything
//! that is not already that exact WAV goes through the system `ffmpeg`, the same subprocess
//! dependency `npu-sr` already carries for video codec I/O.
//!
//! An already-correct WAV never touches ffmpeg -- it is parsed in-process, so the shipped path keeps
//! working on a box with no ffmpeg installed.

use std::path::Path;
use std::process::Command;

use crate::http::parse::parse_wav_i16;

/// `ffmpeg` binary, overridable for a box where it is not on PATH.
fn ffmpeg_bin() -> String {
    std::env::var("FFMPEG").unwrap_or_else(|_| "ffmpeg".into())
}

/// Decode any media file to 16 kHz mono `i16`.
///
/// Tries the in-process WAV parser first (exact-format WAVs skip the subprocess entirely), then
/// hands the file to ffmpeg. Errors name the file and, when ffmpeg is missing, say so directly
/// rather than surfacing a bare ENOENT.
pub fn decode_file(path: &Path) -> Result<Vec<i16>, String> {
    if let Ok(bytes) = std::fs::read(path) {
        if let Some(pcm) = parse_wav_i16(&bytes) {
            return Ok(pcm);
        }
    }
    ffmpeg_decode(path)
}

/// Decode uploaded bytes to 16 kHz mono `i16`.
///
/// ffmpeg needs a seekable input for most containers (an mp4 whose `moov` atom trails the media
/// cannot be read from a pipe), so anything that is not already a correct WAV is spilled to a temp
/// file rather than piped.
pub fn decode_bytes(bytes: &[u8]) -> Result<Vec<i16>, String> {
    if let Some(pcm) = parse_wav_i16(bytes) {
        return Ok(pcm);
    }
    let tmp = std::env::temp_dir().join(format!("npu-transcribe-{}.media", std::process::id()));
    std::fs::write(&tmp, bytes).map_err(|e| format!("spill upload to {}: {e}", tmp.display()))?;
    let out = ffmpeg_decode(&tmp);
    let _ = std::fs::remove_file(&tmp);
    out
}

fn ffmpeg_decode(path: &Path) -> Result<Vec<i16>, String> {
    let bin = ffmpeg_bin();
    let out = Command::new(&bin)
        .args(["-nostdin", "-loglevel", "error", "-i"])
        .arg(path)
        // No video, one channel, 16 kHz, signed 16-bit LE, raw to stdout.
        .args(["-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"])
        .output()
        .map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                format!("{bin} not found: install ffmpeg, or set FFMPEG to its path, to transcribe \
                         anything other than a 16 kHz mono 16-bit WAV")
            } else {
                format!("{bin}: {e}")
            }
        })?;
    if !out.status.success() {
        return Err(format!(
            "{bin} could not decode {}: {}",
            path.display(),
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    if out.stdout.is_empty() {
        return Err(format!("{} decoded to no audio (is there an audio stream?)", path.display()));
    }
    Ok(out.stdout.chunks_exact(2).map(|b| i16::from_le_bytes([b[0], b[1]])).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::http::parse::wav_from_i16;

    fn have_ffmpeg() -> bool {
        Command::new(ffmpeg_bin()).arg("-version").output().is_ok()
    }

    /// A correct WAV is parsed in-process. The assertion that matters is the round trip, but the
    /// path it proves is "no subprocess": this is what keeps an ffmpeg-less box working.
    #[test]
    fn an_exact_wav_needs_no_ffmpeg() {
        let pcm: Vec<i16> = (0..1600).map(|i| (i % 300) as i16 * 100).collect();
        let bytes = wav_from_i16(&pcm, 16_000);
        assert_eq!(decode_bytes(&bytes).unwrap(), pcm);
    }

    /// The case the old surface rejected outright: right container, wrong rate and channel count.
    #[test]
    fn a_44k_stereo_wav_is_resampled_rather_than_refused() {
        if !have_ffmpeg() {
            eprintln!("skipped: no ffmpeg");
            return;
        }
        let dir = std::env::temp_dir().join(format!("npu-media-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("stereo.wav");
        let st = Command::new(ffmpeg_bin())
            .args(["-nostdin", "-loglevel", "error", "-y", "-f", "lavfi",
                   "-i", "sine=frequency=440:duration=2:sample_rate=44100",
                   "-ac", "2"])
            .arg(&src)
            .status()
            .unwrap();
        assert!(st.success());
        assert!(parse_wav_i16(&std::fs::read(&src).unwrap()).is_none(),
            "44.1k stereo must NOT satisfy the strict parser -- otherwise this proves nothing");
        let pcm = decode_file(&src).unwrap();
        // 2 s at 16 kHz, mono. ffmpeg's resampler can land a frame either side of exact.
        assert!((pcm.len() as i64 - 32_000).abs() < 1_000, "got {} samples", pcm.len());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The headline case: a video container transcribes because its audio track is extracted.
    #[test]
    fn a_video_file_yields_its_audio_track() {
        if !have_ffmpeg() {
            eprintln!("skipped: no ffmpeg");
            return;
        }
        let dir = std::env::temp_dir().join(format!("npu-media-vid-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("clip.mp4");
        let st = Command::new(ffmpeg_bin())
            .args(["-nostdin", "-loglevel", "error", "-y",
                   "-f", "lavfi", "-i", "testsrc=size=160x120:rate=10:duration=3",
                   "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
                   "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest"])
            .arg(&src)
            .status()
            .unwrap();
        assert!(st.success(), "could not build the test clip");
        let pcm = decode_file(&src).unwrap();
        assert!((pcm.len() as i64 - 48_000).abs() < 3_000, "3 s of 16 kHz mono, got {}", pcm.len());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_file_with_no_audio_is_an_error_that_says_so() {
        if !have_ffmpeg() {
            eprintln!("skipped: no ffmpeg");
            return;
        }
        let dir = std::env::temp_dir().join(format!("npu-media-silent-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("mute.mp4");
        let st = Command::new(ffmpeg_bin())
            .args(["-nostdin", "-loglevel", "error", "-y",
                   "-f", "lavfi", "-i", "testsrc=size=160x120:rate=10:duration=1",
                   "-c:v", "libx264", "-pix_fmt", "yuv420p"])
            .arg(&src)
            .status()
            .unwrap();
        assert!(st.success());
        let e = decode_file(&src).unwrap_err();
        assert!(e.contains("audio stream") || e.contains("could not decode"), "{e}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
