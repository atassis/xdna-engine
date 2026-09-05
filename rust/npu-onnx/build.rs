// Compile the onnxruntime C-shim and link the system libonnxruntime. The python-package .so has
// no unversioned `libonnxruntime.so` symlink, so we make one in OUT_DIR to link against; at RUNTIME
// the loader finds libonnxruntime.so.* via LD_LIBRARY_PATH (set by install.sh / the systemd unit /
// dev runs). ORT_LIB_DIR overrides the .so directory directly.
use std::path::{Path, PathBuf};

// The versioned libonnxruntime.so.* under `dir`, if any (no unversioned name is shipped).
fn find_versioned_so(dir: &Path) -> Option<PathBuf> {
    std::fs::read_dir(dir).ok()?.filter_map(|e| e.ok()).map(|e| e.path()).find(|p| {
        p.file_name()
            .and_then(|n| n.to_str())
            .map(|n| n.starts_with("libonnxruntime.so."))
            .unwrap_or(false)
    })
}

// A venv's onnxruntime capi dir, if it has one with a usable .so. The python minor version is
// not assumed (lib/python3.<minor>/...), so this works across interpreter versions.
fn onnxruntime_capi_in_venv(venv: &Path) -> Option<PathBuf> {
    std::fs::read_dir(venv.join("lib")).ok()?.filter_map(|e| e.ok()).find_map(|e| {
        if !e.file_name().to_str()?.starts_with("python3.") {
            return None;
        }
        let capi = e.path().join("site-packages/onnxruntime/capi");
        find_versioned_so(&capi).is_some().then_some(capi)
    })
}

fn main() {
    let home = std::env::var("HOME").unwrap_or_default();
    // rust/npu-onnx -> rust -> repo root.
    let repo_root = Path::new(&std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("../..");

    // ORT_LIB_DIR names the onnxruntime capi dir directly. Otherwise there is no universal
    // default (this used to hardcode the author's own throwaway
    // ~/npuvox-asr-bench/.venv/.../python3.12/...), so search the same venv locations
    // install.sh's ONNX_ASR_VENV preflight does, in the same order: an explicit ONNX_ASR_VENV,
    // then ./.venv, then the documented conventional path.
    let ort_dir = std::env::var("ORT_LIB_DIR").ok().or_else(|| {
        [
            std::env::var("ONNX_ASR_VENV").ok().map(PathBuf::from),
            Some(repo_root.join(".venv")),
            Some(Path::new(&home).join(".local/share/xdna-engine/onnx-asr-venv")),
            // Legacy last resort: the venv this project was developed against. Kept as the
            // LAST candidate, never the default -- dropping it entirely broke `cargo check`
            // on the one machine where it does exist.
            Some(Path::new(&home).join("npuvox-asr-bench/.venv")),
        ]
        .into_iter()
        .flatten()
        .find_map(|v| onnxruntime_capi_in_venv(&v))
        .map(|p| p.to_string_lossy().into_owned())
    })
    .unwrap_or_else(|| {
        panic!(
            "no onnxruntime found. Set ORT_LIB_DIR=/path/to/onnxruntime/capi directly, or \
             ONNX_ASR_VENV=/path/to/venv (needs \
             lib/python3.*/site-packages/onnxruntime/capi/libonnxruntime.so.* present)."
        )
    });

    // find the real versioned .so
    let real = find_versioned_so(Path::new(&ort_dir))
        .unwrap_or_else(|| panic!("no libonnxruntime.so.* in {ort_dir}"));

    // In OUT_DIR make: `libonnxruntime.so` (so `-lonnxruntime` resolves at LINK time) and
    // `libonnxruntime.so.1` (the SONAME the loader needs at RUNTIME — the .so dir has only the
    // versioned file, no SONAME symlink). rpath OUT_DIR so the runtime finds it.
    let out = std::env::var("OUT_DIR").unwrap();
    for name in ["libonnxruntime.so", "libonnxruntime.so.1"] {
        let link = Path::new(&out).join(name);
        let _ = std::fs::remove_file(&link);
        std::os::unix::fs::symlink(&real, &link).unwrap_or_else(|e| panic!("symlink {name}: {e}"));
    }

    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .file("shim/onnx_shim.cpp")
        .include("shim")
        .warnings(false)
        .compile("onnx_shim");

    println!("cargo:rustc-link-search=native={out}");
    println!("cargo:rustc-link-lib=dylib=onnxruntime");
    // bake rpath to OUT_DIR (has the SONAME symlink) so the binary finds libonnxruntime.so.1 at
    // runtime without LD_LIBRARY_PATH. rustc-link-arg propagates to dependent bins via DEP info.
    println!("cargo:rustc-link-arg=-Wl,-rpath,{out}");
    let _ = &ort_dir;
    println!("cargo:rerun-if-changed=shim/onnx_shim.cpp");
    println!("cargo:rerun-if-changed=shim/onnx_shim.h");
    println!("cargo:rerun-if-env-changed=ORT_LIB_DIR");
    println!("cargo:rerun-if-env-changed=ONNX_ASR_VENV");
    // export the SONAME-symlink dir to dependents (as DEP_ONNXRUNTIME_RPATH) so the final binary
    // can bake the same rpath and find libonnxruntime.so.1 at runtime.
    println!("cargo:rpath={out}");
}
