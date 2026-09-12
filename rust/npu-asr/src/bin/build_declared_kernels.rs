//! Build every declared kernel that `verify_declared_kernels` reports Missing, then publish and
//! re-verify. This is the "then go build it" half `declared_kernels.json` was missing when it
//! landed.
//!
//! For each Missing declared stem, dispatches to its family's `recipe` adapter script
//! (`<recipe> build <stem> <dest-dir>`) -- see `kernel_registry::build_missing_declared_kernels`.
//! One family's failure does not stop another's attempt. After every dispatch (successful or
//! not), runs `scripts/publish_kernels.sh` so a rebuilt kernel gets hashed into
//! `kernel_manifest.json` the same way any other publish does, then re-verifies the WHOLE
//! declared set (not just what was just attempted) so the final report reflects reality rather
//! than trusting an adapter's exit code.
//!
//!   cargo run -p npu-asr --bin build_declared_kernels -- [<repo-root>] [<kernels-root>] [<mlir-aie-root>]
//!
//! repo-root defaults to ".". kernels-root defaults to <repo-root>/kernels. mlir-aie-root
//! defaults to <repo-root>/mlir-aie (passed through to publish_kernels.sh unchanged).
//!
//! Exit code: 0 if the final re-verify shows everything declared as Present; 1 otherwise. Nothing
//! upstream reads this exit code yet -- same observability-first posture as verify_declared_kernels.

use std::path::PathBuf;

use npu_asr::kernel_registry::{
    build_missing_declared_kernels, load_declared_kernel_set, verify_declared_kernel_set, BuildOutcome,
    DeclaredStatus, PUBLISHED_KERNELS_DIR,
};

fn main() {
    let mut args = std::env::args().skip(1);
    let repo_root = args.next().map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."));
    // Recipe/script paths built from repo_root get passed to Command::current_dir(repo_root) --
    // if repo_root were still relative, the child resolves them against ITS new cwd, not ours.
    let repo_root = repo_root.canonicalize().unwrap_or(repo_root);
    let kernels_root = args.next().map(PathBuf::from).unwrap_or_else(|| repo_root.join(PUBLISHED_KERNELS_DIR));
    let mlir_aie_root = args.next().map(PathBuf::from).unwrap_or_else(|| repo_root.join("mlir-aie"));

    let declared = match load_declared_kernel_set(&repo_root) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("[build-declared-kernels] could not load declared_kernels.json under {}: {e}", repo_root.display());
            std::process::exit(2);
        }
    };

    let results = build_missing_declared_kernels(&declared, &repo_root, &kernels_root);
    let attempted =
        results.iter().filter(|r| matches!(r.outcome, BuildOutcome::Built | BuildOutcome::BuildFailed(_))).count();
    let no_recipe = results.iter().filter(|r| matches!(r.outcome, BuildOutcome::NoRecipe(_))).count();

    for r in &results {
        match &r.outcome {
            BuildOutcome::AlreadyPresent => {}
            BuildOutcome::Built => println!("[build-declared-kernels]      BUILT  {}/{}", r.family, r.stem),
            BuildOutcome::BuildFailed(msg) => {
                println!("[build-declared-kernels]    FAILED  {}/{}  ({msg})", r.family, r.stem)
            }
            BuildOutcome::NoRecipe(path) => {
                println!("[build-declared-kernels]  NO-RECIPE  {}/{}  (looked for {})", r.family, r.stem, path.display())
            }
        }
    }

    if attempted > 0 {
        println!("[build-declared-kernels] {attempted} build(s) attempted -- publishing");
        let publish = std::process::Command::new("bash")
            .arg(repo_root.join("scripts/publish_kernels.sh"))
            .arg(&kernels_root)
            .arg(&mlir_aie_root)
            .current_dir(&repo_root)
            .status();
        match publish {
            Ok(s) if s.success() => {}
            Ok(s) => println!("[build-declared-kernels] WARNING: publish_kernels.sh exited {s}"),
            Err(e) => println!("[build-declared-kernels] WARNING: could not run publish_kernels.sh: {e}"),
        }
    } else if no_recipe > 0 {
        println!(
            "[build-declared-kernels] {no_recipe} Missing stem(s) had no usable recipe -- publish skipped"
        );
    } else {
        println!("[build-declared-kernels] nothing was Missing -- publish skipped");
    }

    let final_report = verify_declared_kernel_set(&declared, &kernels_root);
    let mut missing = 0usize;
    let mut mismatched = 0usize;
    for entry in &final_report {
        match &entry.status {
            DeclaredStatus::Missing => missing += 1,
            DeclaredStatus::HashMismatch(_) => mismatched += 1,
            _ => {}
        }
    }
    println!(
        "[build-declared-kernels] final: {} declared, {missing} still missing, {mismatched} hash-mismatched",
        final_report.len()
    );

    if missing > 0 || mismatched > 0 {
        std::process::exit(1);
    }
}
