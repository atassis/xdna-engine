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
//! Device-free: this only reads files from disk and writes `kernel_manifest.json`. It never opens
//! the NPU.

use std::path::PathBuf;

fn main() {
    let dirs: Vec<PathBuf> = std::env::args().skip(1).map(PathBuf::from).collect();
    if dirs.is_empty() {
        eprintln!("usage: gen_kernel_manifest <artifact-dir> [<artifact-dir> ...]");
        std::process::exit(2);
    }

    let mut had_error = false;
    for dir in dirs {
        match npu_asr::kernel_registry::generate_manifest(&dir) {
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
