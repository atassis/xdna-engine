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
#[command(name = "npu", about = "XDNA2 NPU engine multitool",
          subcommand_required = true, arg_required_else_help = true)]
pub struct Cli {
    /// Config path (default: $NPU_CONFIG or ~/.config/npu/engine.toml)
    #[arg(long, global = true, value_hint = ValueHint::FilePath)]
    pub config: Option<PathBuf>,
    /// Output format for commands that have a machine-readable form.
    // No `short = 'o'`, though the design asked for `-o`: `transcribe-media` already spells its
    // output FILE `-o`, and a global short collides with it -- clap panics there with "Short option
    // names must be unique". Freeing `-o` means renaming that one, which is a user-visible break and
    // a separate decision.
    #[arg(long = "output", global = true, value_enum, default_value_t = OutputFormat::Table)]
    pub output: OutputFormat,
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
    ///
    /// `allow_hyphen_values`: the text to embed is prose, and prose begins with `-` all the time
    /// (every Markdown bullet). Without it clap read a bullet as an unknown flag and failed with a
    /// usage error, so the CLI rejected inputs the HTTP route accepted.
    Embed { #[arg(allow_hyphen_values = true)] text: String, #[arg(long)] model: Option<String> },
    /// One-shot text generation, streamed to stdout by default.
    ///
    /// The prompt goes through the model's chat template, so an instruction-tuned model answers it
    /// and stops. `--raw` sends the bytes verbatim instead, which is `/v1/completions` semantics:
    /// pure continuation, and on a chat-tuned model that means it rambles until max_tokens because
    /// nothing in the prompt ever gives it a turn to end.
    Generate {
        #[arg(allow_hyphen_values = true)] prompt: String,
        #[arg(long)] model: Option<String>,
        #[command(flatten)] sampling: SamplingArgs,
        /// Print the whole completion at once instead of streaming it token by token.
        #[arg(long)] no_stream: bool,
        /// Send the prompt verbatim, with no chat template -- raw continuation.
        #[arg(long)] raw: bool,
    },
    /// Interactive chat REPL: reads a line from stdin, streams the reply, keeps history across
    /// turns. Ctrl-D exits.
    ///
    /// An opening turn on the command line is answered before the first prompt, so `npu chat "hi"`
    /// starts talking instead of waiting, and the session continues from there -- which is the part
    /// `npu generate "hi"` does not do. A turn may begin with `-`; it is not read as a flag.
    Chat {
        /// Opening turn, answered immediately. Omit it to start at an empty prompt.
        #[arg(allow_hyphen_values = true)] prompt: Option<String>,
        #[arg(long)] model: Option<String>,
        #[command(flatten)] sampling: SamplingArgs,
        #[arg(long)] no_stream: bool,
    },
    /// The configured models, plus what the service currently has resident.
    ///
    /// Answers from the config, so it works with the service down. When the service IS up it merges
    /// the state it publishes to a file -- no socket, no probe, nothing to hang on -- and prints how
    /// old that snapshot is. A `*` in PIN means the running server's pin disagrees with the config,
    /// which is what `npu reload` fixes.
    Models {
        /// Machine-readable output. Carries both `pinned` (config) and `live_pinned` (server), which
        /// the table collapses into one PIN cell, so a script can act on the drift the `*` only flags.
        #[arg(long)] json: bool,
        /// Read the status published for this port instead of the config's, to inspect a second
        /// instance. The port is the key the service files its status under, not something dialled.
        #[arg(long)] port: Option<u16>,
    },
    /// Ask a running server to re-read the config and reconcile.
    Reload { #[arg(long)] port: Option<u16> },
    /// Make a model resident on the running server, now.
    ///
    /// Fails rather than evicting when the server is already at `max_resident` -- an explicit load
    /// is a statement about capacity, so honouring it by dropping someone else's model would answer
    /// a different question. The refusal names what is holding the slots. Serving a request still
    /// evicts as before; this is the operator path, not the request path.
    ///
    /// Runtime state, not config: it does not edit `engine.toml` and does not survive a restart.
    /// For that, pin the model (`npu config pin`).
    Load {
        model: String,
        #[arg(long)] port: Option<u16>,
    },
    /// Give a model's device memory back now, without stopping the service.
    ///
    /// The same release the idle sweep performs, fired by hand: the config entry stays, routing
    /// still knows what the model is, and the next request that needs it loads it again. This is
    /// what frees the NPU for another process without `systemctl stop`.
    Unload {
        model: String,
        #[arg(long)] port: Option<u16>,
    },
    /// Pre-bake a model's weight checkpoint (host-only, no device).
    Bake { name: String },
    /// Weight-checkpoint tooling: bake, inspect, and parity-check.
    // Folded in from the separate `npu-weights` binary: `npu` is documented as the single
    // entrypoint, `npu bake` already overlapped `npu-weights bake`, and a second binary was a
    // second completion surface with none of this one's coverage guarantees.
    #[command(subcommand_required = true, arg_required_else_help = true)]
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
    #[command(subcommand_required = true, arg_required_else_help = true)]
    Config { #[command(subcommand)] action: ConfigCmd },
    /// Read-only self-test: device/driver versions, power mode, who holds the device, which
    /// config is in effect and why, whether configured models' artifacts resolve, service status.
    ///
    /// Answers "is my install actually working, and what state is the device in" without touching
    /// the device -- no `Device::open`, no dispatch, no hardware context taken. Shells out to
    /// `xrt-smi examine` (unprivileged, read-only) and reads files only.
    Doctor {
        /// Machine-readable output.
        #[arg(long)] json: bool,
    },
    /// List every `NPU_*`/related env var the engine reads, with its LIVE value and source.
    ///
    /// A third configuration plane alongside `engine.toml` and these CLI flags: vars read directly
    /// by `env::var`/`var_os` across the shipped crates, none of them visible in `engine.toml` or
    /// `--help`. This is that registry (`npu_runtime::env_flags::FLAGS`) rendered against the
    /// current process environment -- report-only, changes nothing.
    Flags {
        /// Machine-readable output.
        #[arg(long)] json: bool,
    },
}

/// Sampling flags shared by `generate` and `chat`. `None` means "the caller did not ask", which is
/// what lets the lower tiers (the scenario's `[generation]` block, then the checkpoint's own
/// `generation_config.json`) supply a value -- so a bare `npu generate "..."` behaves identically to
/// an HTTP request with no sampling fields at all. The two surfaces accept the same set on purpose;
/// a flag here without a JSON field there (or the reverse) is a parity bug.
#[derive(clap::Args)]
pub struct SamplingArgs {
    #[arg(long)] pub temperature: Option<f32>,
    #[arg(long)] pub top_p: Option<f32>,
    #[arg(long)] pub top_k: Option<u32>,
    #[arg(long)] pub max_tokens: Option<u32>,
    /// OpenAI's current spelling for `--max-tokens`. Setting both to different values is an error
    /// rather than a silent pick, matching the HTTP surface.
    #[arg(long)] pub max_completion_tokens: Option<u32>,
    /// OpenAI range -2.0..2.0. Accepted over HTTP since the beginning; the CLI could not send it.
    #[arg(long)] pub presence_penalty: Option<f32>,
    /// OpenAI range -2.0..2.0.
    #[arg(long)] pub frequency_penalty: Option<f32>,
    /// Not an OpenAI field, but universal in local servers. 1.0 = no penalty.
    #[arg(long)] pub repetition_penalty: Option<f32>,
    /// May be repeated: `--stop A --stop B`.
    #[arg(long)] pub stop: Vec<String>,
    #[arg(long)] pub seed: Option<u64>,
    /// Reasoning models (Qwen3) default to emitting a `<think>` block. `--no-think` suppresses it;
    /// without this flag the model's own template default applies, and at --max-tokens 256 that
    /// default spends the entire budget reasoning and never reaches an answer.
    #[arg(long, overrides_with = "think")] pub no_think: bool,
    /// Force the `<think>` block on even if the model's template would omit it.
    #[arg(long, overrides_with = "no_think")] pub think: bool,
}

/// Transcript output formats.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum OutFormat { Md, Srt, Txt, Json }

/// How a command RENDERS its response. Distinct from `OutFormat`, which is the document format
/// `transcribe-media` writes to a file.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default, ValueEnum)]
pub enum OutputFormat { #[default] Table, Json }

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
        /// npu-weights arch transform.
        // Values from ARCH_NAMES, so completion cannot offer an arch npu-weights does not
        // implement, nor fall behind when `arch/` grows a module.
        #[arg(long, value_parser = PossibleValuesParser::new(npu_weights::arch::ARCH_NAMES.to_vec()))]
        arch: String,
        #[arg(long, value_hint = ValueHint::FilePath)] checkpoint: Option<PathBuf>,
        #[arg(long)] force: bool,
    },
    /// mmap-load a checkpoint and print tensor stats.
    Load {
        #[arg(long, value_hint = ValueHint::FilePath)] checkpoint: PathBuf,
        /// npu-weights arch transform.
        #[arg(long, value_parser = PossibleValuesParser::new(npu_weights::arch::ARCH_NAMES.to_vec()))]
        arch: String,
    },
    /// Verify checkpoint tensors match a directory of reference .npy within tolerance.
    Verify {
        #[arg(long, value_hint = ValueHint::FilePath)] checkpoint: PathBuf,
        /// npu-weights arch transform.
        #[arg(long, value_parser = PossibleValuesParser::new(npu_weights::arch::ARCH_NAMES.to_vec()))]
        arch: String,
        #[arg(long, value_hint = ValueHint::DirPath)] refs: PathBuf,
    },
}

#[derive(Subcommand)]
pub enum ConfigCmd {
    /// Print the config's own view: `[server]`, defaults, and every `[[model]]` with its pin state.
    ///
    /// Reads the FILE, not the running server -- unlike `npu models`, nothing here is merged with
    /// live state, so it works with the service down. This is also where a pin overcommit or a
    /// pin the admission order will not honour gets surfaced, on purpose: before the write that
    /// would trip it, not after.
    Show,
    /// Add a model, or repoint an existing one's scenario.
    ///
    /// Updates the entry IN PLACE when `name` is already in the config: only `scenario` changes,
    /// residency and every other key on that model stay as they were. The old writer instead
    /// dropped the entry and pushed a fresh one, which silently unpinned a model the moment its
    /// scenario path was corrected.
    AddModel { name: String, #[arg(value_hint = ValueHint::FilePath)] scenario: String },
    /// Delete a model's `[[model]]` entry entirely.
    ///
    /// Unlike `unpin`, nothing of the model is left behind -- no scenario, no pin state, nothing
    /// for `npu load`/`npu models` to resolve. Fails on a name the config does not have, the same
    /// refusal `pin`/`unpin` make: there is nothing to act on, so silently doing nothing would only
    /// hide the typo.
    RemoveModel { name: String },
    /// Pin a model resident: exempt from idle unload, never chosen as an eviction victim.
    ///
    /// What a pin does NOT do is win a slot it would not otherwise have had. Admission is still
    /// first-N-in-config-order against `max_resident`, so pinning a model listed after enough
    /// others leaves it loading on demand as before -- the server says so on startup rather than
    /// declining the intent silently. Takes effect on `npu reload`; no restart, no device churn.
    Pin { model: String },
    /// Drop a model's residency pin: it becomes swept when idle and evictable again.
    Unpin { model: String },
    /// Set one `[server]` key. `npu config set --help` lists them.
    ///
    /// The key list is closed on purpose: an unrecognised key would produce a file that still
    /// parses and silently does nothing, which is the one failure a config typo must never have.
    // Values from SERVER_KEYS, so completion cannot offer a knob the binary does not read, nor
    // fall behind when one is added.
    #[command(after_long_help = npu_runtime::config_doc::server_key_help_text())]
    Set {
        #[arg(value_parser = PossibleValuesParser::new(
            npu_runtime::config_doc::SERVER_KEYS.iter().map(|(k, _)| *k).collect::<Vec<_>>()))]
        key: String,
        value: String,
    },
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

