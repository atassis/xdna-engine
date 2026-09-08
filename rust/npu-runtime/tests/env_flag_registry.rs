//! The registry has to stay true, and only a test can keep it that way.
//!
//! `env_flags::FLAGS` records a `file:line` per flag. A line number is a hanging number -- it
//! carries no way to verify itself -- and this one rotted within hours of being written: the
//! 2026-09-08 CLI merge moved `npu-cli/src/main.rs`'s body into `run()` and added a subcommand,
//! shifting every line below and silently invalidating five entries. Nothing noticed, because
//! nothing was looking.
//!
//! So: assert every entry still points at its flag. The check is deliberately fuzzy about the exact
//! line (edits above a site shift it by a few) and strict about the file and the name.
//!
//! It matches the name as a QUOTED STRING LITERAL, not as a bare substring, and that detail is the
//! whole test. The first version searched for the bare name and passed against a deliberately wrong
//! line, because `main.rs` carries a doc comment reading "`--config` beats `$NPU_CONFIG` beats the
//! default path" six lines away -- prose about the flag satisfied a check meant to find the read of
//! it. A read is spelled `env::var("NAME")`, `var_os("NAME")` or a closure literal
//! `resident_on("NAME")`; all three quote it, and comments generally do not.

use npu_runtime::env_flags::FLAGS;

/// How far from the recorded line the name may have drifted before this is a stale entry rather
/// than ordinary churn. Wide enough that unrelated edits nearby do not fail the build, narrow
/// enough that a moved flag does.
const DRIFT: usize = 6;

#[test]
fn every_registry_site_still_points_at_its_flag() {
    let rust_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("npu-runtime sits under rust/");

    let mut stale = Vec::new();
    for f in FLAGS {
        let Some((rel, line)) = f.site.rsplit_once(':') else {
            stale.push(format!("{}: site {:?} is not file:line", f.name, f.site));
            continue;
        };
        let Ok(line) = line.parse::<usize>() else {
            stale.push(format!("{}: site {:?} has no line number", f.name, f.site));
            continue;
        };
        let path = rust_root.join(rel);
        let Ok(text) = std::fs::read_to_string(&path) else {
            stale.push(format!("{}: {} does not exist", f.name, path.display()));
            continue;
        };
        let lines: Vec<&str> = text.lines().collect();
        let lo = line.saturating_sub(DRIFT + 1);
        let hi = (line + DRIFT).min(lines.len());
        let quoted = format!("\"{}\"", f.name);
        if !lines[lo..hi].iter().any(|l| l.contains(&quoted)) {
            stale.push(format!(
                "{}: {} line {} no longer mentions it (searched +-{} lines)",
                f.name, rel, line, DRIFT
            ));
        }
    }

    assert!(
        stale.is_empty(),
        "registry entries have gone stale -- update `site` in env_flags.rs:\n  {}",
        stale.join("\n  ")
    );
}

/// A flag read by the engine but absent from `FLAGS` is invisible to `npu flags`, to the
/// measurement stamp, and to every rule in the env-flag contract that keys off the declaration.
/// This is the weaker half of that gate: it catches a flag whose registry entry names a file that
/// no longer reads it at all.
#[test]
fn every_registry_entry_names_a_file_that_reads_it() {
    let rust_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let mut orphaned = Vec::new();
    for f in FLAGS {
        let Some((rel, _)) = f.site.rsplit_once(':') else { continue };
        if let Ok(text) = std::fs::read_to_string(rust_root.join(rel)) {
            if !text.contains(&format!("\"{}\"", f.name)) {
                orphaned.push(format!("{} is not read anywhere in {}", f.name, rel));
            }
        }
    }
    assert!(orphaned.is_empty(), "registry names a stale owner:\n  {}", orphaned.join("\n  "));
}
