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
//! SOURCE its family declares (`declared_kernels.json`'s `sources`), so a later verify can tell a
//! stale-but-intact artifact from a fresh one. The family is the directory's own basename, which
//! is how `publish_kernels.sh` lays `<dest>/<family>` out. Without the flag the digest is left
//! absent, which reports as unverified rather than as fresh.
//!
//! Device-free: this only reads files from disk and writes `kernel_manifest.json`. It never opens
//! the NPU.

use std::path::PathBuf;

fn main() {
    let mut repo_root: Option<PathBuf> = None;
    let mut dirs: Vec<PathBuf> = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--repo-root" => match args.next() {
                Some(v) => repo_root = Some(PathBuf::from(v)),
                None => {
                    eprintln!("gen_kernel_manifest: --repo-root needs a path");
                    std::process::exit(2);
                }
            },
            _ => dirs.push(PathBuf::from(a)),
        }
    }
    if dirs.is_empty() {
        eprintln!("usage: gen_kernel_manifest [--repo-root <path>] <artifact-dir> [<artifact-dir> ...]");
        std::process::exit(2);
    }
    let declared = repo_root
        .as_deref()
        .and_then(|r| npu_asr::kernel_registry::load_declared_kernel_set(r).ok());

    let mut had_error = false;
    for dir in dirs {
        // The family is the directory's basename; a dir that is not a declared family simply
        // gets no digest rather than a wrong one.
        let src = match (&repo_root, &declared) {
            (Some(root), Some(decl)) => dir
                .file_name()
                .and_then(|f| f.to_str())
                .and_then(|fam| decl.get(fam))
                .and_then(|d| {
                    npu_asr::kernel_registry::source_digest(root, &d.sources).ok().flatten()
                }),
            _ => None,
        };
        match npu_asr::kernel_registry::generate_manifest_with_source(&dir, src.as_deref()) {
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
