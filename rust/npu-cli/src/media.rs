//! Media transcription: a container in, a speaker-attributed transcript file out.
//!
//! The hard part is not ASR, it is that a recording's audio TRACKS have no declared meaning. A file
//! may carry one mixed track, or one track per participant, or a microphone track plus a
//! system-audio track, or something else entirely. Nothing in the container says which.
//!
//! So this does not try to guess. Every audio track is treated as an independent source: diarized
//! on its own, transcribed on its own, and labelled from its own metadata. That degrades correctly
//! across all of those cases without needing to know which one it is --
//!
//!   * one track per person  -> each track yields one speaker, and the track's title names them
//!   * mic + system audio    -> two labelled sources interleaved on one timeline
//!   * a single mixed track  -> diarization splits it into speakers as usual
//!   * anything unlabelled   -> falls back to `track0`, `track1`, ... and still works
//!
//! All tracks share the recording's clock, so the per-track results merge into ONE chronological
//! transcript rather than separate per-track sections.

use std::path::Path;
use std::process::Command;

use anyhow::{anyhow, bail, Context, Result};

/// One audio stream in the container.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioTrack {
    /// Position among AUDIO streams (what `-map 0:a:N` takes), not the container stream index.
    pub ord: usize,
    pub title: Option<String>,
    pub language: Option<String>,
    pub channels: u32,
}

impl AudioTrack {
    /// How this track is named in the transcript. Prefers a human title, then a language tag, then
    /// an ordinal -- so a well-tagged per-person recording reads as names, and an untagged one
    /// still reads unambiguously.
    pub fn label(&self) -> String {
        match (&self.title, &self.language) {
            (Some(t), _) if !t.trim().is_empty() => t.trim().to_string(),
            (_, Some(l)) if !l.trim().is_empty() && l != "und" => format!("track{} ({})", self.ord, l),
            _ => format!("track{}", self.ord),
        }
    }
}

/// One transcribed span, already attributed.
#[derive(Debug, Clone, PartialEq)]
pub struct Utterance {
    pub start_s: f32,
    pub end_s: f32,
    pub speaker: String,
    pub text: String,
}

/// Parse `ffprobe -show_entries ... -of json` output into tracks.
///
/// Split out from the subprocess call so the shape handling -- missing tags, absent `channels`,
/// a file with no audio at all -- is testable without ffprobe on the path.
pub fn parse_ffprobe(json: &str) -> Result<Vec<AudioTrack>> {
    let v: serde_json::Value = serde_json::from_str(json).context("ffprobe json")?;
    let streams = v.get("streams").and_then(|s| s.as_array())
        .ok_or_else(|| anyhow!("ffprobe json has no streams array"))?;
    Ok(streams.iter().enumerate().map(|(ord, s)| {
        let tags = s.get("tags");
        let tag = |k: &str| tags.and_then(|t| t.get(k)).and_then(|x| x.as_str()).map(String::from);
        AudioTrack {
            ord,
            title: tag("title"),
            language: tag("language"),
            channels: s.get("channels").and_then(|c| c.as_u64()).unwrap_or(1) as u32,
        }
    }).collect())
}

/// Enumerate the container's audio tracks.
pub fn probe_audio_tracks(input: &Path) -> Result<Vec<AudioTrack>> {
    let out = Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", "a", "-show_entries",
               "stream=index,channels:stream_tags=title,language", "-of", "json"])
        .arg(input)
        .output()
        .context("run ffprobe (is ffmpeg installed?)")?;
    if !out.status.success() {
        bail!("ffprobe failed on {}: {}", input.display(), String::from_utf8_lossy(&out.stderr));
    }
    parse_ffprobe(&String::from_utf8_lossy(&out.stdout))
}

/// Decode one audio track to a 16 kHz mono 16-bit WAV -- the only shape the engine accepts.
/// Downmixes multi-channel tracks rather than refusing them: a stereo system-audio capture is
/// still one source.
pub fn extract_track(input: &Path, ord: usize, out_wav: &Path) -> Result<()> {
    let out = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"])
        .arg(input)
        .args(["-map", &format!("0:a:{ord}"), "-ac", "1", "-ar", "16000",
               "-c:a", "pcm_s16le", "-f", "wav"])
        .arg(out_wav)
        .output()
        .context("run ffmpeg (is ffmpeg installed?)")?;
    if !out.status.success() {
        bail!("ffmpeg failed extracting audio track {ord} of {}: {}",
              input.display(), String::from_utf8_lossy(&out.stderr));
    }
    Ok(())
}

/// Name a speaker within a track.
///
/// When a track yielded exactly ONE speaker the track label IS the speaker -- that is the
/// per-participant-track case, and appending `SPEAKER_00` to it would be noise. With several
/// speakers the track is a shared source and both parts are needed.
pub fn speaker_label(track: &str, speaker: u32, n_speakers_in_track: usize) -> String {
    if n_speakers_in_track <= 1 { track.to_string() }
    else { format!("{track} / SPEAKER_{speaker:02}") }
}

fn ts(t: f32) -> String {
    let total_ms = (t.max(0.0) * 1000.0).round() as u64;
    let (h, m, s, ms) = (total_ms / 3_600_000, (total_ms / 60_000) % 60,
                         (total_ms / 1000) % 60, total_ms % 1000);
    format!("{h:02}:{m:02}:{s:02},{ms:03}")
}

/// Render the merged transcript. `md` is the default because the output is meant to be read.
pub fn render(utts: &[Utterance], format: &str, source: &str) -> Result<String> {
    let mut s = String::new();
    match format {
        "srt" => {
            for (i, u) in utts.iter().enumerate() {
                s.push_str(&format!("{}\n{} --> {}\n{}: {}\n\n",
                    i + 1, ts(u.start_s), ts(u.end_s), u.speaker, u.text));
            }
        }
        "txt" => {
            for u in utts { s.push_str(&format!("{}: {}\n", u.speaker, u.text)); }
        }
        "json" => {
            let items: Vec<serde_json::Value> = utts.iter().map(|u| serde_json::json!({
                "start": u.start_s, "end": u.end_s, "speaker": u.speaker, "text": u.text,
            })).collect();
            s = serde_json::to_string_pretty(&serde_json::json!({
                "source": source, "utterances": items }))?;
            s.push('\n');
        }
        "md" => {
            s.push_str(&format!("# Transcript — {source}\n\n"));
            let mut last: Option<&str> = None;
            for u in utts {
                // Group consecutive lines from one speaker under a single heading, the way a
                // dialogue reads, instead of repeating the name on every line.
                if last != Some(u.speaker.as_str()) {
                    s.push_str(&format!("\n**{}**\n\n", u.speaker));
                    last = Some(u.speaker.as_str());
                }
                s.push_str(&format!("- `[{:.2} - {:.2}]` {}\n", u.start_s, u.end_s, u.text));
            }
        }
        other => bail!("unknown format {other:?} (one of: md|srt|txt|json)"),
    }
    Ok(s)
}

/// Split a diarized turn into ASR-sized pieces.
///
/// The Parakeet path silently truncates its input at `WIN_MEL` = 2040 mel frames = 20.4 s
/// (`npu-engine/src/asr/parakeet.rs:160`, `t.min(WIN_MEL)`): a longer request returns 200 OK with
/// the tail simply missing, no error and no log. Conversational turns routinely exceed that -- a
/// 240 s recording measured here contained a single 124 s span -- so transcribing a turn whole
/// loses most of it.
///
/// This is a WORKAROUND at the call site, not a fix: the truncation belongs to the ASR backend and
/// is tracked separately. It is done here because silently inheriting a silent truncation is how
/// the original defect survived unnoticed in the first place.
///
/// Splits on an even division so pieces are similar length rather than leaving a runt at the end,
/// and never emits a piece longer than `max_s`.
pub fn split_turn(start_s: f32, end_s: f32, max_s: f32) -> Vec<(f32, f32)> {
    let dur = end_s - start_s;
    if dur <= max_s || max_s <= 0.0 {
        return vec![(start_s, end_s)];
    }
    let n = (dur / max_s).ceil() as usize;
    let step = dur / n as f32;
    (0..n).map(|i| {
        let a = start_s + step * i as f32;
        let b = if i + 1 == n { end_s } else { start_s + step * (i + 1) as f32 };
        (a, b)
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ffprobe_tracks_carry_their_titles_and_languages() {
        let j = r#"{"streams":[
          {"index":1,"channels":1,"tags":{"language":"eng","title":"Alice-mic"}},
          {"index":2,"channels":2,"tags":{"language":"rus","title":"Bob-mic"}}]}"#;
        let t = parse_ffprobe(j).unwrap();
        assert_eq!(t.len(), 2);
        assert_eq!(t[0].ord, 0, "ord is the AUDIO ordinal, not the container index");
        assert_eq!(t[1].ord, 1);
        assert_eq!(t[0].label(), "Alice-mic");
        assert_eq!(t[1].channels, 2);
    }

    #[test]
    fn untagged_tracks_still_get_unambiguous_labels() {
        let j = r#"{"streams":[{"index":0,"channels":2},{"index":1,"channels":1,
                   "tags":{"language":"und"}},{"index":2,"tags":{"language":"deu"}}]}"#;
        let t = parse_ffprobe(j).unwrap();
        assert_eq!(t[0].label(), "track0", "no tags at all");
        assert_eq!(t[1].label(), "track1", "'und' is not a language");
        assert_eq!(t[2].label(), "track2 (deu)");
        assert_eq!(t[2].channels, 1, "absent channels defaults to mono, never zero");
    }

    #[test]
    fn a_file_with_no_audio_is_an_empty_list_not_an_error() {
        assert!(parse_ffprobe(r#"{"streams":[]}"#).unwrap().is_empty());
        assert!(parse_ffprobe(r#"{"nope":1}"#).is_err(), "malformed json must still error");
    }

    #[test]
    fn a_single_speaker_track_is_named_by_the_track_alone() {
        // The per-participant-track case: "Alice-mic / SPEAKER_00" would be noise.
        assert_eq!(speaker_label("Alice-mic", 0, 1), "Alice-mic");
        // A shared track needs both halves.
        assert_eq!(speaker_label("room-mic", 0, 3), "room-mic / SPEAKER_00");
        assert_eq!(speaker_label("room-mic", 2, 3), "room-mic / SPEAKER_02");
    }

    fn utts() -> Vec<Utterance> {
        vec![
            Utterance { start_s: 0.5, end_s: 2.0, speaker: "Alice".into(), text: "hello".into() },
            Utterance { start_s: 2.1, end_s: 3.0, speaker: "Alice".into(), text: "again".into() },
            Utterance { start_s: 3.5, end_s: 5.0, speaker: "Bob".into(), text: "hi".into() },
        ]
    }

    #[test]
    fn srt_is_numbered_and_uses_comma_milliseconds() {
        let s = render(&utts(), "srt", "x.mkv").unwrap();
        assert!(s.starts_with("1\n00:00:00,500 --> 00:00:02,000\nAlice: hello\n"), "{s}");
        assert!(s.contains("3\n00:00:03,500 --> 00:00:05,000\nBob: hi"), "{s}");
    }

    #[test]
    fn md_groups_consecutive_lines_under_one_speaker_heading() {
        let s = render(&utts(), "md", "x.mkv").unwrap();
        assert_eq!(s.matches("**Alice**").count(), 1, "two consecutive lines, one heading: {s}");
        assert_eq!(s.matches("**Bob**").count(), 1);
        assert!(s.contains("- `[0.50 - 2.00]` hello"), "{s}");
    }

    #[test]
    fn json_round_trips_and_names_its_source() {
        let s = render(&utts(), "json", "x.mkv").unwrap();
        let v: serde_json::Value = serde_json::from_str(&s).unwrap();
        assert_eq!(v["source"], "x.mkv");
        assert_eq!(v["utterances"].as_array().unwrap().len(), 3);
        assert_eq!(v["utterances"][2]["speaker"], "Bob");
    }

    #[test]
    fn an_unknown_format_is_a_named_error() {
        let e = render(&utts(), "docx", "x").unwrap_err().to_string();
        assert!(e.contains("docx") && e.contains("md|srt|txt|json"), "{e}");
    }

    #[test]
    fn a_turn_within_the_asr_window_is_left_alone() {
        assert_eq!(split_turn(1.0, 15.0, 18.0), vec![(1.0, 15.0)]);
        assert_eq!(split_turn(0.0, 18.0, 18.0), vec![(0.0, 18.0)], "exactly at the limit");
    }

    #[test]
    fn a_long_turn_splits_into_pieces_the_asr_can_actually_hold() {
        // The measured case: one 124s span in a 240s recording. Whole, Parakeet would keep 20.4s
        // of it and silently drop the rest.
        let parts = split_turn(116.11, 240.39, 18.0);
        assert!(parts.len() >= 7, "124s must not stay one piece: {parts:?}");
        for (a, b) in &parts {
            assert!(b - a <= 18.0 + 1e-3, "piece {a}-{b} exceeds the ASR window");
            assert!(b > a, "no empty piece");
        }
        assert!((parts[0].0 - 116.11).abs() < 1e-3, "must start where the turn starts");
        assert!((parts.last().unwrap().1 - 240.39).abs() < 1e-3, "and end where it ends");
    }

    #[test]
    fn splitting_loses_no_audio_and_leaves_no_gaps() {
        let parts = split_turn(5.0, 100.0, 18.0);
        let covered: f32 = parts.iter().map(|(a, b)| b - a).sum();
        assert!((covered - 95.0).abs() < 1e-2, "pieces must tile the turn: {covered}");
        for w in parts.windows(2) {
            assert!((w[1].0 - w[0].1).abs() < 1e-3, "gap between {:?} and {:?}", w[0], w[1]);
        }
    }

    #[test]
    fn timestamps_cross_the_hour_and_clamp_negatives() {
        assert_eq!(ts(3661.5), "01:01:01,500");
        assert_eq!(ts(-1.0), "00:00:00,000");
    }
}
