//! Regenerate `kernel_manifest.json` for one or more artifact directories, from the xclbin/insts
//! files actually sitting there right now (engine-op-manifest-and-dynamic-xclbin).
//!
//! Every entry's hash comes from reading the real file -- run this after building or copying
//! kernels into a directory `kernel_registry::resolve()` reads from. A manifest produced any other
//! way (hand-typed, copied from a different dir) would be exactly the kind of declared-but-
//! unverified state this tool exists to eliminate; there is deliberately no way to construct a
//! `ManifestEntry` except by hashing a file that is actually there.
//!
//! Non-recursive per directory, matching `kernel_registry::resolve()`'s directory-scoped
//! convention -- pass each artifact directory you want covered, e.g.:
//!   cargo run -p npu-asr --bin gen_kernel_manifest -- artifacts/parakeet/ln artifacts/asr
//!
//! With `--repo-root <path>` each directory's manifest also records the digest of the kernel
//! SOURCE its scenario-owned declaration lists, so a later verify can tell a
//! stale-but-intact artifact from a fresh one. The family is the directory's own basename, which
//! is how `publish_kernels.sh` lays `<dest>/<family>` out. Without the flag the digest is left
//! absent, which reports as unverified rather than as fresh.
//!
//! Device-free: this only reads files from disk and writes `kernel_manifest.json`. It never opens
//! the NPU.

use std::path::PathBuf;

use npu_asr::kernel_registry::{
    default_engine_config_path, load_declared_kernel_set_from_engine_config, source_digest,
};

pub fn run(argv: Vec<String>) {
    let mut repo_root: Option<PathBuf> = None;
    let mut config: Option<PathBuf> = None;
    let mut engine_root: Option<PathBuf> = None;
    let mut dirs: Vec<PathBuf> = Vec::new();
    let mut args = argv.into_iter().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--repo-root" => match args.next() {
                Some(v) => repo_root = Some(PathBuf::from(v)),
                None => {
                    eprintln!("gen_kernel_manifest: --repo-root needs a path");
                    std::process::exit(2);
                }
            },
            "--config" => match args.next() {
                Some(v) => config = Some(PathBuf::from(v)),
                None => {
                    eprintln!("kernels-manifest: --config needs an engine.toml path");
                    std::process::exit(2);
                }
            },
            "--engine-root" => match args.next() {
                Some(v) => engine_root = Some(PathBuf::from(v)),
                None => {
                    eprintln!("kernels-manifest: --engine-root needs a directory");
                    std::process::exit(2);
                }
            },
            _ => dirs.push(PathBuf::from(a)),
        }
    }
    if dirs.is_empty() {
        eprintln!("usage: kernels-manifest --repo-root <path> [--config <engine.toml>] <artifact-dir> [<artifact-dir> ...]");
        std::process::exit(2);
    }
    let repo_root = match repo_root {
        Some(root) => root,
        None => {
            eprintln!("kernels-manifest: --repo-root is required to derive source digests");
            std::process::exit(2);
        }
    };
    let config = config.unwrap_or_else(default_engine_config_path);
    let engine_root = engine_root.unwrap_or_else(|| repo_root.clone());
    let declared = match load_declared_kernel_set_from_engine_config(&repo_root, &engine_root, &config) {
        Ok(declared) => declared,
        Err(e) => {
            eprintln!("kernels-manifest: could not resolve served kernel declarations from {}: {e}", config.display());
            std::process::exit(2);
        }
    };

    let mut had_error = false;
    for dir in dirs {
        let family = match dir.file_name().and_then(|f| f.to_str()) {
            Some(family) => family,
            None => {
                eprintln!("kernels-manifest: cannot derive family from {}", dir.display());
                had_error = true;
                continue;
            }
        };
        let Some(declaration) = declared.get(family) else {
            println!("[kernels-manifest] skip undeclared family {}", dir.display());
            continue;
        };
        let source = match source_digest(&repo_root, &declaration.sources) {
            Ok(source) => source,
            Err(e) => {
                eprintln!("kernels-manifest: digest {}: {e}", dir.display());
                had_error = true;
                continue;
            }
        };
        match npu_asr::kernel_registry::generate_manifest_with_source(&dir, source.as_deref()) {
            Ok(manifest) => {
                let n = manifest.len();
                match npu_asr::kernel_registry::write_manifest(&dir, &manifest) {
                    Ok(()) => println!(
                        "[gen_kernel_manifest] {}: {n} stem(s) -> {}",
                        dir.display(),
                        npu_asr::kernel_registry::manifest_path(&dir).display()
                    ),
                    Err(e) => {
                        eprintln!("[gen_kernel_manifest] write manifest for {}: {e}", dir.display());
                        had_error = true;
                    }
                }
            }
            Err(e) => {
                eprintln!("[gen_kernel_manifest] scan {}: {e}", dir.display());
                had_error = true;
            }
        }
    }
    if had_error {
        std::process::exit(1);
    }
}
