//! The CLI's shape: subcommands, flags, and the enums their values come from.
//!
//! Split out of `main.rs` so the completion-coverage test can walk the same command tree the
//! binary uses. A test that rebuilt its own copy of the tree would pass while the real CLI grew a
//! subcommand nobody could tab-complete, which is the failure this is meant to catch.

use std::path::PathBuf;

use clap::builder::PossibleValuesParser;
use clap::{Parser, Subcommand, ValueEnum, ValueHint};
use clap_complete::Shell;
use npu_engine::capability::Capability;

#[derive(Parser)]
#[command(name = "npu", about = "XDNA2 NPU engine multitool")]
pub struct Cli {
    /// Config path (default: $NPU_CONFIG or ~/.config/npu/engine.toml)
    #[arg(long, global = true, value_hint = ValueHint::FilePath)]
    pub config: Option<PathBuf>,
    #[command(subcommand)]
    pub cmd: Cmd,
}

#[derive(Subcommand)]
pub enum Cmd {
    /// Run the HTTP service (single device owner).
    Serve {
        #[arg(long)] port: Option<u16>,
        /// Bind even when a configured model failed to load. `/healthz` still reports 503.
        #[arg(long)] allow_degraded: bool,
    },
    /// One-shot transcription of an audio or video file, printed to stdout.
    ///
    /// A 16 kHz mono 16-bit WAV is read directly; anything else (other rates, stereo, mp3, a video
    /// container) is decoded through ffmpeg. For a speaker-attributed transcript file, see
    /// `transcribe-media`.
    Transcribe {
        #[arg(value_hint = ValueHint::FilePath)] input: PathBuf,
        #[arg(long)] model: Option<String>,
    },
    /// Transcribe a media file (video or audio) to a speaker-attributed transcript FILE.
    ///
    /// Every audio track is handled independently -- diarized, transcribed and labelled from its
    /// own metadata -- then merged onto one timeline. That covers a mixed track, one track per
    /// participant, and mic-plus-system-audio without having to know which it is.
    TranscribeMedia {
        #[arg(value_hint = ValueHint::FilePath)] input: PathBuf,
        /// Output file. Defaults to <input> with the format's extension.
        #[arg(long, short, value_hint = ValueHint::FilePath)] out: Option<PathBuf>,
        /// Transcript output format.
        // A ValueEnum rather than a String so clap both validates it at parse time and emits the
        // choices into shell completion; as a String it generated `--format=[]`.
        #[arg(long, value_enum, default_value_t = OutFormat::Md)] format: OutFormat,
        /// ASR model name; omit to use the configured asr default.
        #[arg(long)] asr: Option<String>,
        /// Diarization model name; omit to use the configured diarize default.
        #[arg(long)] diarize: Option<String>,
        /// Only this audio track (0-based among audio streams). Default: all of them.
        #[arg(long)] track: Option<usize>,
        /// Skip diarization; label every utterance by its track. Faster, no speaker split.
        #[arg(long)] no_diarize: bool,
    },
    /// Speaker diarization of a 16 kHz mono 16-bit WAV: who spoke when.
    Diarize {
        #[arg(value_hint = ValueHint::FilePath)] wav: PathBuf,
        #[arg(long)] model: Option<String>,
        /// Emit the same JSON body the HTTP route returns, instead of readable lines.
        #[arg(long)] json: bool,
    },
    /// One-shot embedding of a text string.
    /// `allow_hyphen_values`: the text to embed is prose, and prose begins with `-` all the time
    /// (every Markdown bullet). Without it clap read a bullet as an unknown flag and failed with a
    /// usage error, so the CLI rejected inputs the HTTP route accepted.
    Embed { #[arg(allow_hyphen_values = true)] text: String, #[arg(long)] model: Option<String> },
    /// List models on a running server.
    Models { #[arg(long)] port: Option<u16> },
    /// Ask a running server to re-read the config and reconcile.
    Reload { #[arg(long)] port: Option<u16> },
    /// Pre-bake a model's weight checkpoint (host-only, no device).
    Bake { name: String },
    /// Weight-checkpoint tooling: bake, inspect, and parity-check.
    // Folded in from the separate `npu-weights` binary: `npu` is documented as the single
    // entrypoint, `npu bake` already overlapped `npu-weights bake`, and a second binary was a
    // second completion surface with none of this one's coverage guarantees.
    Weights {
        #[command(subcommand)]
        action: WeightsCmd,
    },
    /// Print a shell completion script (zsh, bash, fish, elvish, powershell).
    ///
    /// Generated from the clap command tree, so it covers every subcommand and flag and cannot
    /// drift from them the way a hand-written script would.
    Completions { shell: Shell },
    /// Inspect / edit the desired-state config.
    Config { #[command(subcommand)] action: ConfigCmd },
}

/// Transcript output formats.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum OutFormat { Md, Srt, Txt, Json }

impl OutFormat {
    pub fn as_str(self) -> &'static str {
        match self { OutFormat::Md => "md", OutFormat::Srt => "srt",
                     OutFormat::Txt => "txt", OutFormat::Json => "json" }
    }
    /// File extension for the default output path.
    pub fn ext(self) -> &'static str { self.as_str() }
}

#[derive(Subcommand)]
pub enum WeightsCmd {
    /// Bake source weights into a bf16 checkpoint (skips if fresh, unless --force).
    Bake {
        /// `hf:<repo>[@rev]` or `path:/abs`.
        #[arg(long)] source: String,
        /// npu-weights arch transform: bert|esm|vit|opt|whisper|fastconformer|gigaam|...
        #[arg(long)] arch: String,
        #[arg(long, value_hint = ValueHint::FilePath)] checkpoint: Option<PathBuf>,
        #[arg(long)] force: bool,
    },
    /// mmap-load a checkpoint and print tensor stats.
    Load {
        #[arg(long, value_hint = ValueHint::FilePath)] checkpoint: PathBuf,
        #[arg(long)] arch: String,
    },
    /// Verify checkpoint tensors match a directory of reference .npy within tolerance.
    Verify {
        #[arg(long, value_hint = ValueHint::FilePath)] checkpoint: PathBuf,
        #[arg(long)] arch: String,
        #[arg(long, value_hint = ValueHint::DirPath)] refs: PathBuf,
    },
}

#[derive(Subcommand)]
pub enum ConfigCmd {
    Show,
    AddModel { name: String, #[arg(value_hint = ValueHint::FilePath)] scenario: String },
    RemoveModel { name: String },
    /// Set the default model for a capability.
    // Values come from Capability::ALL, so completion cannot offer a capability this binary does
    // not implement, nor fall behind when one is added.
    SetDefault {
        #[arg(value_parser = PossibleValuesParser::new(
            Capability::ALL.iter().map(|c| c.0).collect::<Vec<_>>()))]
        capability: String,
        model: String,
    },
}

