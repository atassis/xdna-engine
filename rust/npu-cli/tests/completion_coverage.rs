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
