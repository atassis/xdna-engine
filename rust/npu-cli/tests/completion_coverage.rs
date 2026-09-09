//! Proof that the shell completions cover the whole CLI.
//!
//! The generated script is only as good as the command tree it is generated from, and the failure
//! mode is silent: a subcommand or flag added without a thought reaches users as a command that
//! simply does not tab-complete, which nobody reports as a bug. So this walks clap's own tree and
//! asserts every name appears in the emitted zsh script.
//!
//! It also checks VALUES, not just names. `--format` shipped as a bare `String` and generated
//! `--format=[]` -- flag completion present, choices absent -- which is exactly the kind of
//! half-coverage a names-only check would call passing.

use clap::CommandFactory;
use clap_complete::Shell;

/// Rebuild the CLI definition the binary uses. Kept in one place so the test cannot drift from it.
#[path = "../src/cli_def.rs"]
mod cli_def;

fn zsh_script() -> String {
    let mut cmd = cli_def::Cli::command();
    let mut buf: Vec<u8> = Vec::new();
    clap_complete::generate(Shell::Zsh, &mut cmd, "npu", &mut buf);
    String::from_utf8(buf).expect("completion script is utf8")
}

/// Every (subcommand, flag) pair clap knows about.
fn walk(cmd: &clap::Command, path: &str, out: &mut Vec<(String, String)>) {
    for a in cmd.get_arguments() {
        if let Some(l) = a.get_long() {
            out.push((path.to_string(), format!("--{l}")));
        }
    }
    for sub in cmd.get_subcommands() {
        let name = sub.get_name();
        if name == "help" { continue }
        let child = if path.is_empty() { name.to_string() } else { format!("{path} {name}") };
        out.push((child.clone(), String::new()));
        walk(sub, &child, out);
    }
}

#[test]
fn every_subcommand_and_flag_appears_in_the_zsh_completion() {
    let script = zsh_script();
    let mut pairs = Vec::new();
    walk(&cli_def::Cli::command(), "", &mut pairs);

    let mut missing = Vec::new();
    for (path, flag) in &pairs {
        let needle = if flag.is_empty() {
            path.rsplit(' ').next().unwrap().to_string()
        } else {
            flag.clone()
        };
        if !script.contains(&needle) {
            missing.push(format!("{path} {flag}").trim().to_string());
        }
    }
    assert!(missing.is_empty(), "not tab-completable: {missing:?}");

    // Guard against the check passing vacuously. Not a magic count -- an arbitrary threshold is
    // itself a bug waiting to fire (this one was written as `> 30` against a real surface of 29).
    // Instead: every top-level subcommand clap reports must have been walked.
    let walked: std::collections::BTreeSet<&str> =
        pairs.iter().map(|(p, _)| p.split(' ').next().unwrap()).collect();
    for sub in cli_def::Cli::command().get_subcommands() {
        let name = sub.get_name();
        if name == "help" { continue }
        assert!(walked.contains(name), "walk() never reached subcommand {name:?}");
    }
    assert!(pairs.iter().any(|(_, f)| !f.is_empty()), "walk() found no flags at all");
}

#[test]
fn enumerated_flags_offer_their_values_not_an_empty_set() {
    let script = zsh_script();
    // `--format` was a bare String and generated `--format=[]`: the flag completed, the choices
    // did not. Every value-enum flag must carry its choices into the script.
    for (flag, values) in [("--format", ["md", "srt", "txt", "json"].as_slice())] {
        assert!(!script.contains(&format!("{flag}=[]")),
            "{flag} generates an EMPTY value set -- it is not a ValueEnum");
        for v in values {
            assert!(script.contains(v), "{flag} does not offer {v:?}");
        }
    }
}

#[test]
fn capability_values_come_from_the_engine_not_a_hardcoded_list() {
    let script = zsh_script();
    // `config set-default <capability>` must offer exactly what this binary can serve.
    for cap in npu_engine::capability::Capability::ALL {
        assert!(script.contains(cap.0),
            "capability {:?} is implemented but not offered by completion", cap.0);
    }
}

/// Every subcommand `Command` node reachable from the root, keyed by its full path (`"weights
/// bake"`). Unlike `walk()` above, this keeps the `&Command` itself, not just its name/flags, so a
/// test can inspect `get_about()` / `get_arguments()` -- what `--help` actually renders.
fn walk_commands<'a>(cmd: &'a clap::Command, path: &str, out: &mut Vec<(String, &'a clap::Command)>) {
    for sub in cmd.get_subcommands() {
        if sub.get_name() == "help" { continue }
        let child = if path.is_empty() { sub.get_name().to_string() } else { format!("{path} {}", sub.get_name()) };
        out.push((child.clone(), sub));
        walk_commands(sub, &child, out);
    }
}

#[test]
fn every_subcommand_has_a_description() {
    let cmd = cli_def::Cli::command();
    let mut nodes = Vec::new();
    walk_commands(&cmd, "", &mut nodes);
    let missing: Vec<&str> = nodes.iter()
        .filter(|(_, c)| c.get_about().is_none())
        .map(|(p, _)| p.as_str())
        .collect();
    assert!(missing.is_empty(), "`--help` prints a blank description for: {missing:?}");
}

/// Value-bearing arguments with no enumerable possible-values, allow-listed because the value
/// names something outside this process (a config-file model, a filesystem path, a network port)
/// or is arbitrary operator-supplied text/number -- nothing a `PossibleValuesParser` could offer.
/// Keyed by clap's argument id, which is shared by a flag and same-named positional across
/// subcommands (e.g. `model` means "model name" everywhere it appears).
const FREE_FORM_ARGS: &[(&str, &str)] = &[
    ("config", "engine.toml path -- arbitrary filesystem location"),
    ("port", "TCP port number -- not a finite set"),
    ("input", "input audio/video file path"),
    ("out", "output file path"),
    ("model", "model name -- from the operator's [[model]] list, not a fixed set"),
    ("asr", "ASR model name, same reason as `model`"),
    ("diarize", "diarization model name, same reason as `model`"),
    ("track", "0-based audio track index -- arbitrary integer"),
    ("wav", "input WAV file path"),
    ("text", "prose to embed -- arbitrary string"),
    ("prompt", "generation prompt -- arbitrary string"),
    ("temperature", "sampling float -- arbitrary number"),
    ("top_p", "sampling float -- arbitrary number"),
    ("top_k", "sampling integer -- arbitrary number"),
    ("max_tokens", "sampling integer -- arbitrary number"),
    ("max_completion_tokens", "sampling integer -- arbitrary number, alias of max_tokens"),
    ("presence_penalty", "sampling float -- arbitrary number in -2.0..2.0"),
    ("frequency_penalty", "sampling float -- arbitrary number in -2.0..2.0"),
    ("repetition_penalty", "sampling float -- arbitrary number in 0.01..2.0"),
    ("stop", "stop sequences -- arbitrary, repeatable strings"),
    ("seed", "RNG seed -- arbitrary integer"),
    ("name", "model name being registered/baked -- not a fixed set"),
    ("source", "`hf:<repo>[@rev]` or `path:/abs` -- arbitrary source spec"),
    ("checkpoint", "checkpoint file path"),
    ("refs", "reference .npy directory path"),
    ("scenario", "scenario TOML file path"),
    ("value", "a `config set` value -- shape depends on which key, checked at write time"),
    ("log", "JSONL run-log path to read"),
    ("diff", "second JSONL run log to compare against"),
];

#[test]
fn no_value_bearing_argument_completes_to_an_empty_set() {
    // `--arch` was exactly this bug: a bare String, no possible-values, so it offered nothing to
    // complete -- and its help text was a hand-written list that had already fallen behind (7 of
    // the 13 real archs named, with a trailing "...").
    let cmd = cli_def::Cli::command();
    let mut nodes = vec![(String::new(), &cmd)];
    walk_commands(&cmd, "", &mut nodes);
    let allow: std::collections::BTreeSet<&str> = FREE_FORM_ARGS.iter().map(|(id, _)| *id).collect();

    let mut bad = Vec::new();
    for (path, c) in &nodes {
        for a in c.get_arguments() {
            if !a.get_action().takes_values() { continue }
            if !a.get_possible_values().is_empty() { continue }
            let id = a.get_id().as_str();
            if !allow.contains(id) { bad.push(format!("{path} --{id}")); }
        }
    }
    assert!(bad.is_empty(),
        "value-bearing, no possible-values, not in FREE_FORM_ARGS -- completes to an empty set: {bad:?}");
}

#[test]
fn every_supported_shell_generates_a_non_trivial_script() {
    for sh in [Shell::Zsh, Shell::Bash, Shell::Fish] {
        let mut cmd = cli_def::Cli::command();
        let mut buf: Vec<u8> = Vec::new();
        clap_complete::generate(sh, &mut cmd, "npu", &mut buf);
        let s = String::from_utf8(buf).unwrap();
        assert!(s.len() > 1000, "{sh:?} script suspiciously short ({} bytes)", s.len());
        assert!(s.contains("transcribe-media"), "{sh:?} missing a subcommand");
        assert!(s.contains("diarize"), "{sh:?} missing a subcommand");
    }
}

/// `--q-len` and `--preemption` are the two knobs the multi-tenant queue work will need on `serve`:
/// how deep the request queue runs, and whether a higher-criticality request may preempt a running
/// one. Neither exists yet, and this test asserts they still do not parse.
///
/// It is here to hold the NAMES. This tree has a specific allergy to declared-but-unimplemented
/// surface -- `Capability::TTS` and `IMAGE_SR` sat in `ALL` and were served over HTTP with no CLI
/// verb for long enough that nobody noticed. A flag that parses and then errors is the same thing
/// with a friendlier face: shell completion offers it, so it reads as a capability. Better that the
/// name is unclaimed until it works, and that whoever repurposes either spelling for something else
/// fails here rather than shipping a `--q-len` that means something unrelated.
#[test]
fn reserved_queue_names_do_not_parse() {
    use clap::Parser;
    for flag in ["--q-len", "--preemption"] {
        let r = cli_def::Cli::try_parse_from(["npu", "serve", flag, "10"]);
        assert!(
            r.is_err(),
            "{flag} parses, but nothing implements it. Either it now does something -- in which \
             case document it and delete it from this test -- or the name has been taken for an \
             unrelated purpose, which is what this test exists to catch."
        );
    }
}

/// clap validates the command tree (duplicate long names, colliding shorts, bad defaults) only in
/// `debug_assert`s that fire when a command is BUILT, so an inconsistency reaches a debug user as a
/// panic at the moment they run the offending subcommand and a release user as silence.
///
/// A global `-o` for `--output` was written and did exactly that: it collided with
/// `transcribe-media -o/--out`, built clean, ran clean on every other subcommand, and panicked only
/// once `transcribe-media` itself was invoked.
///
/// The completion tests above do catch it, because generating a script builds the tree -- but they
/// report it as a panic inside an unrelated assertion. This one names the failure.
#[test]
fn the_command_tree_is_internally_consistent() {
    cli_def::Cli::command().debug_assert();
}
