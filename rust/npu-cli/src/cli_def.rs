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
    ///
    /// For `generate` and `chat`, `json` follows the stream flag the way `/v1/chat/completions`
    /// does: streaming (the default) writes NDJSON to stdout -- a conditions header, one
    /// `chat.completion.chunk` per decoded token carrying that token's own timing under `x_npu`,
    /// then a summary -- flushed per line, so `> run.jsonl` produces a file `npu stats` and
    /// `npu replay` read. `--no-stream` writes the single `chat.completion` object instead.
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
    ///
    /// The HTTP bind and the control socket path are both `$NPU_HTTP_ENDPOINT`/
    /// `$NPU_SOCKET_ENDPOINT`-overridable (unset, they default to `engine.toml`'s `server.port`
    /// and the systemd `RuntimeDirectory`) -- see `npu flags`.
    Serve {
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
        /// ASR model name; omit to use the configured asr default.
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
        /// Diarization model name; omit to use the configured diarize default.
        #[arg(long)] model: Option<String>,
        /// Emit the same JSON body the HTTP route returns, instead of readable lines.
        #[arg(long)] json: bool,
    },
    /// One-shot embedding of a text string.
    ///
    /// `allow_hyphen_values`: the text to embed is prose, and prose begins with `-` all the time
    /// (every Markdown bullet). Without it clap read a bullet as an unknown flag and failed with a
    /// usage error, so the CLI rejected inputs the HTTP route accepted.
    Embed {
        #[arg(allow_hyphen_values = true)] text: String,
        /// Embedding model name; omit to use the configured embed default.
        #[arg(long)] model: Option<String>,
    },
    /// One-shot text generation, streamed to stdout by default.
    ///
    /// The prompt goes through the model's chat template, so an instruction-tuned model answers it
    /// and stops. `--raw` sends the bytes verbatim instead, which is `/v1/completions` semantics:
    /// pure continuation, and on a chat-tuned model that means it rambles until max_tokens because
    /// nothing in the prompt ever gives it a turn to end.
    Generate {
        #[arg(allow_hyphen_values = true)] prompt: String,
        /// Generation model name; omit to use the configured generate default.
        #[arg(long)] model: Option<String>,
        #[command(flatten)] sampling: SamplingArgs,
        /// Print the full per-token measurement breakdown after the answer.
        ///
        /// The one-line form is printed after every generation anyway, on stderr -- measuring is
        /// free, so it always happens, and stderr keeps a pipe's stdout clean. This asks for the
        /// whole table: the phase split, the latency tail, and the conditions the run happened
        /// under.
        #[arg(long)]
        stats: bool,
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
        /// Generation model name; omit to use the configured generate default.
        #[arg(long)] model: Option<String>,
        #[command(flatten)] sampling: SamplingArgs,
        #[arg(long)] no_stream: bool,
    },
    /// Model lifecycle and inventory: what's configured, what's resident, start/stop it now,
    /// enable/disable it always-on, add/remove it from config, set capability defaults.
    #[command(subcommand_required = true, arg_required_else_help = true)]
    Model {
        #[command(subcommand)]
        action: ModelCmd,
    },
    /// Weight-checkpoint tooling: bake, inspect, and parity-check.
    // Folded in from the separate `npu-weights` binary AND the top-level `npu bake <name>`, which
    // fully overlapped `npu weights bake --name`: one namespace, one completion surface.
    #[command(subcommand_required = true, arg_required_else_help = true)]
    Weights {
        #[command(subcommand)]
        action: WeightsCmd,
    },
    /// Live view of the device: who is resident, who is serving, and where the time went.
    ///
    /// `docker stats` for the NPU. Reads the control socket's status snapshot -- answered
    /// out-of-band, so nothing here can hang on a busy device -- and refreshes in place. With the
    /// service down it says so rather than showing an empty table.
    Top {
        /// Seconds between refreshes.
        #[arg(long, default_value_t = 1.0)]
        interval: f64,
        /// Print one snapshot and exit. The default when stdout is not a terminal, so
        /// `npu top | ...` behaves like every other command here.
        #[arg(long)]
        once: bool,
    },
    /// Read a JSONL run log written by `--output json` or `NPU_TELEMETRY_LOG`.
    ///
    /// Renders the same overlay a live generation prints, from a file -- so a run from another day
    /// or another machine reads the same way, and the renderer has exactly one input type.
    Stats {
        /// The run log to read.
        #[arg(value_hint = ValueHint::FilePath)]
        log: PathBuf,
        /// Compare against a second run: token divergence first, then the timing deltas, with the
        /// two conditions stamps side by side.
        #[arg(long, value_name = "OTHER", value_hint = ValueHint::FilePath)]
        diff: Option<PathBuf>,
    },
    /// Re-emit a run log's completion, with no device and no model.
    ///
    /// The chunk lines in a log ARE the stream frames that were served, so replaying is reading
    /// them back out. Useful to drive a client against a recorded run, and to reproduce a bad
    /// answer without needing the NPU free.
    Replay {
        /// The run log to replay.
        #[arg(value_hint = ValueHint::FilePath)]
        log: PathBuf,
        /// Reproduce the original inter-token timing instead of emitting as fast as possible.
        #[arg(long)]
        realtime: bool,
        /// Emit the raw SSE frames as recorded, rather than just the text.
        #[arg(long)]
        frames: bool,
    },
    /// Print a shell completion script (zsh, bash, fish, elvish, powershell).
    ///
    /// Generated from the clap command tree, so it covers every subcommand and flag and cannot
    /// drift from them the way a hand-written script would.
    Completions { shell: Shell },
    /// Inspect / edit the desired-state config.
    #[command(subcommand_required = true, arg_required_else_help = true)]
    Config {
        #[command(subcommand)] action: ConfigCmd,
    },
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
    /// Force per-dispatch/hw-context-transition accounting on for THIS generation, regardless of
    /// `NPU_DISPATCH_LOG` on the service. The service is long-lived and that env var latches at its
    /// first read, so this is the only way to turn accounting on for one run without a restart.
    #[arg(long, overrides_with = "no_dispatch_log")] pub dispatch_log: bool,
    /// Force it off for this one generation even if `NPU_DISPATCH_LOG=1` is set on the service.
    #[arg(long, overrides_with = "dispatch_log")] pub no_dispatch_log: bool,
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
        /// Bake a CONFIGURED model by name instead: resolves its scenario's declarative spec.
        /// Talks to the running service when one is up (a resident model's checkpoint may be
        /// mmap'd by that same process), falling back to running in-process otherwise. Mutually
        /// exclusive with --source/--arch, which take a spec directly with no configured model.
        #[arg(long, conflicts_with_all = ["source", "arch"])]
        name: Option<String>,
        /// `hf:<repo>[@rev]` or `path:/abs`. Required unless --name is given.
        #[arg(long, required_unless_present = "name")]
        source: Option<String>,
        /// npu-weights arch transform. Required unless --name is given.
        // Values from ARCH_NAMES, so completion cannot offer an arch npu-weights does not
        // implement, nor fall behind when `arch/` grows a module.
        #[arg(long, required_unless_present = "name",
              value_parser = PossibleValuesParser::new(npu_weights::arch::ARCH_NAMES.to_vec()))]
        arch: Option<String>,
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
    /// Reads the FILE, not the running server -- unlike `npu model ls`, nothing here is merged with
    /// live state, so it works with the service down. This is also where a pin overcommit or a
    /// pin the admission order will not honour gets surfaced, on purpose: before the write that
    /// would trip it, not after.
    Show,
    /// Set one `[server]` key. `npu config set --help` lists them.
    ///
    /// The key list is closed on purpose: an unrecognised key would produce a file that still
    /// parses and silently does nothing, which is the one failure a config typo must never have.
    #[command(after_long_help = npu_runtime::config_doc::server_key_help_text())]
    Set {
        #[arg(value_parser = PossibleValuesParser::new(
            npu_runtime::config_doc::SERVER_KEYS.iter().map(|(k, _)| *k).collect::<Vec<_>>()))]
        key: String,
        value: String,
    },
}

#[derive(Subcommand)]
pub enum ModelCmd {
    /// The configured models, plus what the service currently has resident.
    ///
    /// Answers from the config, so it works with the service down. When the service IS up it merges
    /// the state answered over the control socket's out-of-band snapshot -- nothing to hang on --
    /// and prints how old that snapshot is. A `*` in PIN means the running server's pin disagrees
    /// with the config; `npu model enable`/`disable` reconcile a running server automatically, and a
    /// config edited by hand needs `systemctl --user restart xdna-engine` to take effect.
    Ls {
        /// Machine-readable output. Carries both `pinned` (config) and `live_pinned` (server), which
        /// the table collapses into one PIN cell, so a script can act on the drift the `*` only flags.
        #[arg(long)] json: bool,
        /// Add columns beyond what an operator needs day to day: KV block size, attention-window
        /// rungs, toolchain freshness, and the full weight-quant breakdown. Developer detail, not
        /// hidden -- just not printed by default.
        #[arg(long)] verbose: bool,
    },
    /// Full detail for one model: everything `ls --verbose` shows, for a single name.
    Show {
        model: String,
        /// Machine-readable output.
        #[arg(long)] json: bool,
    },
    /// Make a model resident on the running server, now.
    ///
    /// Fails rather than evicting when the server is already over `memory_ceiling_mb` -- an explicit
    /// start is a statement about capacity, so honouring it by dropping someone else's model would
    /// answer a different question. The refusal names what is holding the budget. Serving a request
    /// still evicts as before; this is the operator path, not the request path.
    ///
    /// Runtime state, not config: it does not edit `engine.toml` and does not survive a restart.
    /// For that, enable the model (`npu model enable`).
    Start {
        /// The configured model to make resident.
        model: String,
    },
    /// Give a model's device memory back now, without stopping the service.
    ///
    /// The config entry stays, routing still knows what the model is, and a real request that needs
    /// it loads it again on demand. If the model is ENABLED (always-on), this still sticks: it will
    /// NOT be reloaded by an unrelated config edit or reconcile pass, only by an explicit `npu model
    /// start`, a re-`enable`, or an actual service restart -- matching Docker's `restart: always`
    /// semantics, where a manual stop is respected until the daemon itself restarts.
    Stop {
        /// The resident model whose device memory to release.
        model: String,
    },
    /// Enable a model: always on. Admitted before any on-demand model at boot/reload, exempt from
    /// idle unload, never chosen as an eviction victim.
    ///
    /// The one exception is the invariant itself: `sum(enabled bytes) <= memory_ceiling_mb`. An
    /// enable that would push the sum over the ceiling is not silently granted -- it is refused at
    /// admission (nothing loaded yet), or demoted (an already-resident enable the ceiling was
    /// lowered under, or whose own footprint grew), reported either way rather than declined in
    /// silence. Takes effect immediately on a running server (this command reconciles it
    /// automatically unless `--no-reload` is given); no restart, no device churn.
    Enable { model: String },
    /// Disable a model: it becomes swept when idle and evictable again. Does NOT force it off the
    /// device right now -- it drains via the ordinary idle sweep/LRU, same as any unpinned model.
    Disable { model: String },
    /// Add a model, or repoint an existing one's scenario.
    ///
    /// Updates the entry IN PLACE when `name` is already in the config: only `scenario` changes,
    /// residency and every other key on that model stay as they were. The old writer instead
    /// dropped the entry and pushed a fresh one, which silently disabled a model the moment its
    /// scenario path was corrected.
    Add { name: String, #[arg(value_hint = ValueHint::FilePath)] scenario: String },
    /// Delete a model's `[[model]]` entry entirely.
    ///
    /// Unlike `disable`, nothing of the model is left behind -- no scenario, no enable state, nothing
    /// for `npu model start`/`ls` to resolve. Fails on a name the config does not have, the same
    /// refusal `enable`/`disable` make: there is nothing to act on, so silently doing nothing would
    /// only hide the typo.
    Rm { name: String },
    /// Set the default model for a capability.
    Default {
        #[arg(value_parser = PossibleValuesParser::new(
            Capability::ALL.iter().map(|c| c.0).collect::<Vec<_>>()))]
        capability: String,
        model: String,
    },
}

