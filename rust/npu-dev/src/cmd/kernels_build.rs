//! Build every kernel missing from the selected engine configuration's scenario closure, then
//! publish and re-verify.
//!
//! For each Missing declared stem, dispatches to its family's `recipe` adapter script
//! (`<recipe> build <stem> <mlir-aie-root>`) -- see `kernel_registry::build_missing_declared_kernels`.
//! An adapter's job is to build and stamp its own build dir under `mlir-aie-root`; it never
//! writes into `kernels_root` itself. One family's failure does not stop another's attempt.
//! After every dispatch (successful or not), runs `scripts/publish_kernels.sh` -- the ONLY thing
//! that copies artifacts into `kernels_root` -- so a rebuilt kernel gets hashed into
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn publication_receives_the_selected_engine_root() {
        let root = PathBuf::from("/checkout");
        let kernels = PathBuf::from("/artifacts/kernels");
        let mlir = PathBuf::from("/toolchain");
        let config = PathBuf::from("/configuration/engine.toml");
        let engine = PathBuf::from("/installed/engine");
        let command = publish_command(&root, &kernels, &mlir, &config, &engine);
        let args: Vec<_> = command.get_args().map(PathBuf::from).collect();
        assert_eq!(args, vec![root.join("scripts/publish_kernels.sh"), kernels, mlir, config, engine]);
        assert_eq!(command.get_current_dir(), Some(root.as_path()));
    }
}

use npu_asr::kernel_registry::{
    build_missing_declared_kernels, default_engine_config_path, load_declared_kernel_set_from_engine_config,
    verify_declared_kernel_set_from, BuildOutcome, DeclaredStatus, PUBLISHED_KERNELS_DIR,
};

fn publish_command(
    repo_root: &std::path::Path,
    kernels_root: &std::path::Path,
    mlir_aie_root: &std::path::Path,
    config: &std::path::Path,
    engine_root: &std::path::Path,
) -> std::process::Command {
    let mut command = std::process::Command::new("bash");
    command.arg(repo_root.join("scripts/publish_kernels.sh"))
        .arg(kernels_root)
        .arg(mlir_aie_root)
        .arg(config)
        .arg(engine_root)
        .current_dir(repo_root);
    command
}

pub fn run(argv: Vec<String>) {
    let mut config = None;
    let mut engine_root = None;
    let mut positional = Vec::new();
    let mut args = argv.into_iter().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--config" {
            config = args.next().map(PathBuf::from);
            if config.is_none() {
                eprintln!("kernels-build: --config needs an engine.toml path");
                std::process::exit(2);
            }
        } else if arg == "--engine-root" {
            engine_root = args.next().map(PathBuf::from);
            if engine_root.is_none() {
                eprintln!("kernels-build: --engine-root needs a directory");
                std::process::exit(2);
            }
        } else {
            positional.push(arg);
        }
    }
    let mut positional = positional.into_iter();
    let repo_root = positional.next().map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."));
    // Recipe/script paths built from repo_root get passed to Command::current_dir(repo_root) --
    // if repo_root were still relative, the child resolves them against ITS new cwd, not ours.
    let repo_root = repo_root.canonicalize().unwrap_or(repo_root);
    let kernels_root = positional.next().map(PathBuf::from).unwrap_or_else(|| repo_root.join(PUBLISHED_KERNELS_DIR));
    let mlir_aie_root = positional.next().map(PathBuf::from).unwrap_or_else(|| repo_root.join("mlir-aie"));
    let config = config.unwrap_or_else(default_engine_config_path);
    let engine_root = engine_root.unwrap_or_else(|| repo_root.clone());

    let declared = match load_declared_kernel_set_from_engine_config(&repo_root, &engine_root, &config) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("[kernels-build] could not resolve served kernel declarations from {}: {e}", config.display());
            std::process::exit(2);
        }
    };

    let results = build_missing_declared_kernels(&declared, &repo_root, &kernels_root, &mlir_aie_root);
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
        let publish = publish_command(&repo_root, &kernels_root, &mlir_aie_root, &config, &engine_root).status();
        match publish {
            Ok(s) if s.success() => {}
            Ok(s) => {
                eprintln!("[kernels-build] publish_kernels.sh exited {s}");
                std::process::exit(1);
            }
            Err(e) => {
                eprintln!("[kernels-build] could not run publish_kernels.sh: {e}");
                std::process::exit(1);
            }
        }
    } else if no_recipe > 0 {
        println!(
            "[build-declared-kernels] {no_recipe} Missing stem(s) had no usable recipe -- publish skipped"
        );
    } else {
        println!("[build-declared-kernels] nothing was Missing -- publish skipped");
    }

    let final_report = verify_declared_kernel_set_from(&declared, &kernels_root, Some(&repo_root));
    let mut missing = 0usize;
    let mut mismatched = 0usize;
    for entry in &final_report {
        match &entry.status {
            DeclaredStatus::Missing => missing += 1,
            DeclaredStatus::HashMismatch(_) | DeclaredStatus::PresentUnverified | DeclaredStatus::StaleSource { .. } => mismatched += 1,
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
