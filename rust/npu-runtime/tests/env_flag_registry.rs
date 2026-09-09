//! The registry has to stay true, and only a test can keep it that way.
//!
//! `env_flags::FLAGS` records the FILE that reads each flag, and this test asserts it still does.
//!
//! It matches the name as a QUOTED STRING LITERAL, not as a bare substring, and that detail is the
//! whole test. The first version searched for the bare name and passed against a deliberately wrong
//! site, because `main.rs` carries a doc comment reading "`--config` beats `$NPU_CONFIG` beats the
//! default path" -- prose about the flag satisfying a check meant to find the read of it. A read is
//! spelled `env::var("NAME")`, `var_os("NAME")` or a closure literal `resident_on("NAME")`; all
//! three quote it, and comments generally do not.
//!
//! `site` USED TO carry a `:line`, and the line was checked with a +-6 tolerance. That number could
//! not be kept true: it rotted within hours of being written (the 2026-09-08 CLI merge moved
//! `main.rs`'s body into `run()` and five entries immediately stopped pointing at their flag), and
//! then failed this gate three more times on 2026-09-09 for edits that moved a function without
//! touching a flag -- four failures, zero real defects. It was a second guard for a hole the quoted
//! literal had already closed, so it is gone rather than tolerated: the invariant worth asserting is
//! "this file reads this flag", and `grep` finds the line in the time it takes to read it.

use npu_runtime::env_flags::FLAGS;

#[test]
fn every_registry_site_still_points_at_its_flag() {
    let rust_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("npu-runtime sits under rust/");

    let mut stale = Vec::new();
    for f in FLAGS {
        let rel = f.site;
        let path = rust_root.join(rel);
        let Ok(text) = std::fs::read_to_string(&path) else {
            stale.push(format!("{}: {} does not exist", f.name, path.display()));
            continue;
        };
        let quoted = format!("\"{}\"", f.name);
        if !text.lines().any(|l| l.contains(&quoted)) {
            stale.push(format!("{}: {} no longer reads it", f.name, rel));
        }
    }

    assert!(
        stale.is_empty(),
        "registry entries name a file that no longer reads their flag -- fix `site` in env_flags.rs:\n  {}",
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
        let rel = f.site;
        if let Ok(text) = std::fs::read_to_string(rust_root.join(rel)) {
            if !text.contains(&format!("\"{}\"", f.name)) {
                orphaned.push(format!("{} is not read anywhere in {}", f.name, rel));
            }
        }
    }
    assert!(orphaned.is_empty(), "registry names a stale owner:\n  {}", orphaned.join("\n  "));
}

// =================================================================================================
// The third gate: a flag-shaped `env::var`/`var_os` read outside `FLAGS` is a build failure.
//
// SCOPE, matched to `env_flags.rs`'s own module doc rather than invented here: crates in
// `rust/Cargo.toml`'s `[workspace] default-members` (derived at test time by parsing that array,
// so a new workspace member cannot slip past this gate unscoped -- `npu-probes` is excluded simply
// because it is not in that list), scanning only each crate's `src/` tree (so `build.rs`, which
// runs at build time and never ships, and `tests/`, which is test-only, are out of scope by
// directory, not by a name list). Within `src/`, a read inside a `#[cfg(test)]` region is excluded
// (never compiled into a shipped binary). And a small fixed set of names is out of scope regardless
// of where it is read, copied VERBATIM from `env_flags.rs`'s own module doc (`is_out_of_scope_by_name`
// below) rather than reinvented: build-time (`CARGO_*`, `OUT_DIR`) and pure OS/toolchain environment
// (`HOME`, `PATH`, `LD_LIBRARY_PATH`, `XRT_*`). `XDG_*` is deliberately NOT in that set --
// `XDG_DATA_HOME` and `XDG_RUNTIME_DIR` are both live registry entries, so a blanket `XDG_*`
// exclusion would have hidden a real gap.
//
// INDIRECT READS. `E005`'s own text says a literal-string grep undercounted the original census by
// 30, blind to a flag read through a closure parameterized by name -- `resident_on("PARAKEET_...")`
// in `npu-parakeet/src/encoder.rs`, `not_zero("NPU_...", ..)` / `is_one(..)` in
// `npu-asr/src/tuning.rs`. This scan resolves exactly that shape: it finds every `env::var`/`var_os`
// call whose argument is a bare identifier rather than a quoted literal (evidence the call sits
// inside a small wrapper parameterized by that identifier), looks at the same and immediately
// preceding source line for a `fn IDENT(` or `let IDENT = |` binding to name the wrapper, and then
// treats any call `IDENT("SOME_FLAG")` anywhere else in scope as a read of `SOME_FLAG` -- exactly
// the two idioms the codebase actually uses today (`resident_on`, `not_zero`, `is_one`; verified by
// grep against the whole scoped tree before writing this).
//
// DOCUMENTED BLIND SPOT: wrapper-name resolution is a two-line textual proximity heuristic, not
// scope-aware parsing. It will not follow a SECOND level of indirection (a wrapper that calls
// another wrapper), will not resolve a wrapper whose parameterized `env::var` call sits more than
// one line below its `fn`/`let` signature, and will not resolve a name built at runtime
// (`format!("NPU_{stem}_MODE")`, a `const` fed to `env::var`, or an identifier read from a loop
// variable such as `npu-cli`'s own `flags_cmd`, which does `env::var_os(f.name)` over the registry
// itself -- correctly invisible to this scan, since there is no new literal name to check there in
// the first place). Any indirect read this heuristic cannot resolve to a wrapper is printed as a
// non-fatal note, not silently dropped and not a failure -- the gate's job is to catch names it CAN
// see and be honest about the ones it cannot.
//
// `#[cfg(test)]` EXCLUSION uses brace-depth tracking like `env_flags.rs`'s own module doc says the
// registry's author used, over a comment/string/char-literal-aware tokenizer (needed because this
// tree's kernels and parsers contain both multi-line/raw string literals and at least one char
// literal brace, `'{'` in `npu-cli/src/main.rs`, either of which corrupts a naive brace count).

/// True for a Rust source char at `chars[i]` that is real code -- not inside a string, a char
/// literal, or a comment. Built once per file by [`code_mask`] and consulted by every scan below so
/// none of them can mistake a comment or a string's contents for syntax.
type Mask = Vec<bool>;

/// Classifies every char in `chars` as code (`true`) or comment/string/char-literal content
/// (`false`). Newlines are always left `true` (never blanked) so downstream line-number bookkeeping
/// stays in sync with the original file regardless of what state a `\n` interrupts.
///
/// Handles: `//` line comments, `/* */` block comments (not nested -- none exist in the scoped
/// tree, checked by grep before writing this), `"..."` strings with `\`-escapes (including a
/// literal newline, since this tree uses `\`-continued multi-line string literals), `b"..."` byte
/// strings, `r"..."`/`r#"..."#`/`br#"..."#` raw strings of any hash count, and `'x'`/`'\n'`/`'\u{..}'`
/// char literals distinguished from a lifetime/generic tick by lookahead (a real char literal is
/// exactly one char or one escape followed immediately by a closing `'`; a lifetime like `'a` is
/// not, so it is left as ordinary code and never toggles string-like state).
fn code_mask(chars: &[char]) -> Mask {
    #[derive(PartialEq)]
    enum St {
        Normal,
        LineComment,
        BlockComment,
        Str,
        RawStr(usize),
    }
    let n = chars.len();
    let mut mask = vec![true; n];
    let mut st = St::Normal;
    let mut i = 0;
    while i < n {
        let c = chars[i];
        if c == '\n' {
            if st == St::LineComment {
                st = St::Normal;
            }
            i += 1;
            continue;
        }
        match st {
            St::Normal => {
                if c == '/' && chars.get(i + 1) == Some(&'/') {
                    mask[i] = false;
                    st = St::LineComment;
                    i += 1;
                    continue;
                }
                if c == '/' && chars.get(i + 1) == Some(&'*') {
                    mask[i] = false;
                    st = St::BlockComment;
                    i += 1;
                    continue;
                }
                // Raw string: optional `b`, then `r`, then N `#`, then `"`.
                let mut j = i;
                if chars.get(j) == Some(&'b') {
                    j += 1;
                }
                if chars.get(j) == Some(&'r') {
                    let mut k = j + 1;
                    let mut hashes = 0usize;
                    while chars.get(k) == Some(&'#') {
                        hashes += 1;
                        k += 1;
                    }
                    if chars.get(k) == Some(&'"') {
                        for p in i..=k {
                            mask[p] = false;
                        }
                        i = k + 1;
                        st = St::RawStr(hashes);
                        continue;
                    }
                }
                if c == 'b' && chars.get(i + 1) == Some(&'"') {
                    mask[i] = false;
                    mask[i + 1] = false;
                    i += 2;
                    st = St::Str;
                    continue;
                }
                if c == '"' {
                    mask[i] = false;
                    i += 1;
                    st = St::Str;
                    continue;
                }
                if c == '\'' {
                    if let Some(end) = char_literal_end(chars, i) {
                        for p in i..=end {
                            mask[p] = false;
                        }
                        i = end + 1;
                        continue;
                    }
                    // Not a char literal (lifetime, generic tick): ordinary code, fall through.
                }
                i += 1;
            }
            St::LineComment => {
                mask[i] = false;
                i += 1;
            }
            St::BlockComment => {
                mask[i] = false;
                if c == '*' && chars.get(i + 1) == Some(&'/') {
                    mask[i + 1] = false;
                    i += 2;
                    st = St::Normal;
                    continue;
                }
                i += 1;
            }
            St::Str => {
                mask[i] = false;
                if c == '\\' {
                    if i + 1 < n && chars[i + 1] != '\n' {
                        mask[i + 1] = false;
                        i += 2;
                        continue;
                    }
                }
                if c == '"' {
                    st = St::Normal;
                }
                i += 1;
            }
            St::RawStr(hashes) => {
                mask[i] = false;
                if c == '"' {
                    let mut k = i + 1;
                    let mut got = 0usize;
                    while got < hashes && chars.get(k) == Some(&'#') {
                        mask[k] = false;
                        k += 1;
                        got += 1;
                    }
                    if got == hashes {
                        i = k;
                        st = St::Normal;
                        continue;
                    }
                }
                i += 1;
            }
        }
    }
    mask
}

/// If `chars[q] == '\''` opens a char literal (`'x'`, `'\n'`, `'\u{1F600}'`, ...), returns the
/// index of its closing `'`. Returns `None` for a lifetime/generic tick (`'a`), which is left
/// untouched by the caller.
fn char_literal_end(chars: &[char], q: usize) -> Option<usize> {
    if chars.get(q) != Some(&'\'') {
        return None;
    }
    let mut k = q + 1;
    if chars.get(k) == Some(&'\\') {
        k += 1;
        match chars.get(k)? {
            'u' => {
                k += 1;
                if chars.get(k) == Some(&'{') {
                    k += 1;
                    while let Some(c) = chars.get(k) {
                        k += 1;
                        if *c == '}' {
                            break;
                        }
                    }
                }
            }
            'x' => k += 3,
            _ => k += 1,
        }
    } else {
        chars.get(k)?;
        k += 1;
    }
    if chars.get(k) == Some(&'\'') {
        Some(k)
    } else {
        None
    }
}

/// True iff `pat` occurs at `chars[i..]` with every matched char marked as code -- i.e. the pattern
/// is real syntax, not text inside a string or comment that happens to spell the same characters.
fn match_code(chars: &[char], mask: &[bool], i: usize, pat: &str) -> bool {
    let pc: Vec<char> = pat.chars().collect();
    if i + pc.len() > chars.len() {
        return false;
    }
    (0..pc.len()).all(|k| mask[i + k] && chars[i + k] == pc[k])
}

fn skip_ws(chars: &[char], mut i: usize) -> usize {
    while i < chars.len() && chars[i].is_whitespace() {
        i += 1;
    }
    i
}

/// An identifier (`[A-Za-z_][A-Za-z0-9_]*`) starting at `chars[i]`, if `i` is real code.
fn ident_at(chars: &[char], mask: &[bool], i: usize) -> Option<(String, usize)> {
    if i >= chars.len() || !mask[i] {
        return None;
    }
    let c = chars[i];
    if !(c.is_ascii_alphabetic() || c == '_') {
        return None;
    }
    let mut end = i + 1;
    while end < chars.len() && mask[end] && (chars[end].is_ascii_alphanumeric() || chars[end] == '_') {
        end += 1;
    }
    Some((chars[i..end].iter().collect(), end))
}

/// Per-file precomputed view: raw chars, the code/non-code mask, and which positions sit inside a
/// `#[cfg(test)]` region.
struct FileView {
    rel: String,
    chars: Vec<char>,
    mask: Mask,
    in_test: Vec<bool>,
    line_of: Vec<usize>,
}

/// Marks every position inside a `#[cfg(test)]`-decorated item as excluded, by brace-depth
/// tracking: on seeing the (code-only) substring `cfg(test)`, remember the current depth; the next
/// `{` at that depth opens the item's body, excluded until depth returns to the same level; a `;`
/// at that depth with no intervening `{` (a brace-less item, e.g. `#[cfg(test)] use x;`) closes the
/// exclusion immediately. Nested/nearby occurrences are handled the same way one after another.
fn compute_in_test(chars: &[char], mask: &[bool]) -> Vec<bool> {
    enum St {
        None,
        Pending(i64),
        Active(i64),
    }
    let n = chars.len();
    let mut in_test = vec![false; n];
    let mut depth: i64 = 0;
    let mut st = St::None;
    let mut region_start = 0usize;
    let mut i = 0;
    while i < n {
        if matches!(st, St::None) && match_code(chars, mask, i, "cfg(test)") {
            st = St::Pending(depth);
            region_start = i;
        }
        if mask[i] {
            match chars[i] {
                '{' => {
                    if let St::Pending(d) = st {
                        if depth == d {
                            st = St::Active(d);
                        }
                    }
                    depth += 1;
                }
                '}' => {
                    depth -= 1;
                    if let St::Active(d) = st {
                        if depth == d {
                            for p in region_start..=i {
                                in_test[p] = true;
                            }
                            st = St::None;
                        }
                    }
                }
                ';' => {
                    if let St::Pending(d) = st {
                        if depth == d {
                            for p in region_start..=i {
                                in_test[p] = true;
                            }
                            st = St::None;
                        }
                    }
                }
                _ => {}
            }
        }
        i += 1;
    }
    assert_eq!(depth, 0, "unbalanced braces in the code view -- tokenizer bug, not a real gap");
    in_test
}

fn is_flag_shaped(name: &str) -> bool {
    !name.is_empty()
        && name.chars().next().unwrap().is_ascii_uppercase()
        && name.chars().all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
}

/// Copied VERBATIM from `env_flags.rs`'s own module doc ("Build-time vars (`CARGO_*`, `OUT_DIR`)
/// and pure environment (`HOME`, `PATH`, `LD_LIBRARY_PATH`, `XRT_*`) are out of scope too") -- not
/// invented here. `XDG_*` is deliberately absent: `XDG_DATA_HOME`/`XDG_RUNTIME_DIR` are registered.
fn is_out_of_scope_by_name(name: &str) -> bool {
    matches!(name, "HOME" | "PATH" | "LD_LIBRARY_PATH" | "OUT_DIR")
        || name.starts_with("CARGO_")
        || name.starts_with("XRT_")
}

/// `rust/Cargo.toml`'s `[workspace] default-members` array, parsed at test time (not hardcoded) so
/// a new workspace member is scoped automatically. `npu-probes` is excluded simply by not being
/// listed there, matching `env_flags.rs`'s own scoping -- not by a name check here.
fn shipped_crate_dirs(rust_root: &std::path::Path) -> Vec<std::path::PathBuf> {
    let text = std::fs::read_to_string(rust_root.join("Cargo.toml")).expect("rust/Cargo.toml must exist");
    let key_pos = text.find("default-members").expect("Cargo.toml has no default-members");
    let eq = text[key_pos..].find('=').expect("default-members has no '='") + key_pos;
    let open = text[eq..].find('[').expect("default-members value is not an array") + eq;
    let close = text[open..].find(']').expect("default-members array is unterminated") + open;
    text[open + 1..close]
        .split(',')
        .map(|s| s.trim().trim_matches('"').to_string())
        .filter(|s| !s.is_empty())
        .map(|name| rust_root.join(name))
        .collect()
}

fn walk_rs_files(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            walk_rs_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

fn load_scoped_files(rust_root: &std::path::Path) -> Vec<FileView> {
    let mut paths = Vec::new();
    for crate_dir in shipped_crate_dirs(rust_root) {
        walk_rs_files(&crate_dir.join("src"), &mut paths);
    }
    paths.sort();
    paths
        .into_iter()
        .map(|p| {
            let text = std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
            let chars: Vec<char> = text.chars().collect();
            let mask = code_mask(&chars);
            let in_test = compute_in_test(&chars, &mask);
            let mut line_of = vec![1usize; chars.len()];
            let mut line = 1usize;
            for (i, &c) in chars.iter().enumerate() {
                line_of[i] = line;
                if c == '\n' {
                    line += 1;
                }
            }
            let rel = p.strip_prefix(rust_root).unwrap_or(&p).to_string_lossy().into_owned();
            FileView { rel, chars, mask, in_test, line_of }
        })
        .collect()
}

/// One flag-shaped literal name found flowing into `env::var`/`var_os`, direct or through a
/// resolved wrapper.
struct Found {
    name: String,
    file: String,
    line: usize,
}

/// Extracts the `"..."` content starting at `chars[quote]` (which must be the opening `"`).
/// Flag names never contain a backslash or quote, so this does not need general escape handling.
fn string_literal_at(chars: &[char], quote: usize) -> String {
    let mut q = quote + 1;
    let mut s = String::new();
    while q < chars.len() && chars[q] != '"' {
        s.push(chars[q]);
        q += 1;
    }
    s
}

#[test]
fn every_flag_shaped_env_read_is_registered() {
    let rust_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let files = load_scoped_files(rust_root);

    let mut direct: Vec<Found> = Vec::new();
    let mut wrapper_names: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    let mut unresolved_indirect: Vec<(String, usize)> = Vec::new(); // (file, line)

    // Pass 1: direct literal reads, and indirect (bare-identifier) reads -- resolving each
    // indirect site's wrapper name from the same/preceding source line.
    for fv in &files {
        let n = fv.chars.len();
        let mut i = 0;
        while i < n {
            let anchor = if match_code(&fv.chars, &fv.mask, i, "::var_os(") {
                Some("::var_os(")
            } else if match_code(&fv.chars, &fv.mask, i, "::var(") {
                Some("::var(")
            } else {
                None
            };
            let Some(anchor) = anchor else {
                i += 1;
                continue;
            };
            let arg = skip_ws(&fv.chars, i + anchor.chars().count());
            if arg < n && fv.chars[arg] == '"' {
                if !fv.in_test[i] {
                    let name = string_literal_at(&fv.chars, arg);
                    if is_flag_shaped(&name) && !is_out_of_scope_by_name(&name) {
                        direct.push(Found { name, file: fv.rel.clone(), line: fv.line_of[i] });
                    }
                }
            } else if arg < n && fv.mask[arg] && (fv.chars[arg].is_ascii_alphabetic() || fv.chars[arg] == '_') {
                if !fv.in_test[i] {
                    let line = fv.line_of[i];
                    let line_start = fv.chars[..i].iter().rposition(|&c| c == '\n').map(|p| p + 1).unwrap_or(0);
                    let prev_line_start = fv.chars[..line_start.saturating_sub(1)]
                        .iter()
                        .rposition(|&c| c == '\n')
                        .map(|p| p + 1)
                        .unwrap_or(0);
                    let search_start = if line_start == 0 { 0 } else { prev_line_start };
                    match find_wrapper_decl(&fv.chars, &fv.mask, search_start..i) {
                        Some(name) => {
                            wrapper_names.insert(name);
                        }
                        None => unresolved_indirect.push((fv.rel.clone(), line)),
                    }
                }
            }
            i += 1;
        }
    }

    // Pass 2: every call `wrapper_name("LITERAL")` anywhere in scope, for every wrapper resolved
    // above -- this is what turns `resident_on("PARAKEET_FFN_DEVACC")` into a read of
    // `PARAKEET_FFN_DEVACC` without ever grepping for that literal directly.
    let mut indirect: Vec<Found> = Vec::new();
    for wrapper in &wrapper_names {
        for fv in &files {
            let n = fv.chars.len();
            let mut i = 0;
            while i < n {
                if match_code(&fv.chars, &fv.mask, i, wrapper) {
                    let before_ok = i == 0 || !(fv.chars[i - 1].is_ascii_alphanumeric() || fv.chars[i - 1] == '_');
                    let after = i + wrapper.chars().count();
                    let after_ok = after >= n || !(fv.chars[after].is_ascii_alphanumeric() || fv.chars[after] == '_');
                    if before_ok && after_ok {
                        let paren = skip_ws(&fv.chars, after);
                        if paren < n && fv.mask[paren] && fv.chars[paren] == '(' {
                            let arg = skip_ws(&fv.chars, paren + 1);
                            if arg < n && fv.chars[arg] == '"' && !fv.in_test[i] {
                                let name = string_literal_at(&fv.chars, arg);
                                if is_flag_shaped(&name) && !is_out_of_scope_by_name(&name) {
                                    indirect.push(Found { name, file: fv.rel.clone(), line: fv.line_of[i] });
                                }
                            }
                        }
                    }
                }
                i += 1;
            }
        }
    }

    if !unresolved_indirect.is_empty() {
        eprintln!(
            "note: {} indirect env::var/var_os read(s) with a bare-identifier argument could not be \
             resolved to a wrapper name (fn/let-closure not found within one line) -- documented blind \
             spot, not a failure:",
            unresolved_indirect.len()
        );
        for (file, line) in &unresolved_indirect {
            eprintln!("  {file}:{line}");
        }
    }

    let registered: std::collections::BTreeSet<&str> = FLAGS.iter().map(|f| f.name).collect();
    let mut missing: std::collections::BTreeMap<&str, (&str, usize)> = std::collections::BTreeMap::new();
    for f in direct.iter().chain(indirect.iter()) {
        if !registered.contains(f.name.as_str()) {
            missing.entry(f.name.as_str()).or_insert((f.file.as_str(), f.line));
        }
    }

    assert!(
        missing.is_empty(),
        "flag-shaped env read(s) not in npu_runtime::env_flags::FLAGS -- add an entry for each:\n  {}",
        missing.iter().map(|(name, (file, line))| format!("{name} at {file}:{line}")).collect::<Vec<_>>().join("\n  ")
    );
}

/// Looks in `chars[range]` (expected to span the line before, and up to, an indirect
/// `env::var`/`var_os` call) for the nearest `fn IDENT(` or `let IDENT = |` and returns `IDENT`.
/// This is the whole of the wrapper-name heuristic: proximity on the source line, not scope
/// tracking -- see the blind-spot note on the module-level comment above.
fn find_wrapper_decl(chars: &[char], mask: &[bool], range: std::ops::Range<usize>) -> Option<String> {
    let mut last = None;
    let mut i = range.start;
    while i < range.end {
        if match_code(chars, mask, i, "fn ") {
            let after = skip_ws(chars, i + 3);
            if let Some((name, _)) = ident_at(chars, mask, after) {
                last = Some(name);
            }
        } else if match_code(chars, mask, i, "let ") {
            let after = skip_ws(chars, i + 4);
            if let Some((name, end)) = ident_at(chars, mask, after) {
                let eq = skip_ws(chars, end);
                if eq < chars.len() && mask[eq] && chars[eq] == '=' {
                    let bar = skip_ws(chars, eq + 1);
                    if bar < chars.len() && mask[bar] && chars[bar] == '|' {
                        last = Some(name);
                    }
                }
            }
        }
        i += 1;
    }
    last
}
