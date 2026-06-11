// Compile the onnxruntime C-shim and link the system libonnxruntime. The python-package .so has
// no unversioned `libonnxruntime.so` symlink, so we make one in OUT_DIR to link against; at RUNTIME
// the loader finds libonnxruntime.so.* via LD_LIBRARY_PATH (set by install.sh / the systemd unit /
// dev runs). ORT_LIB_DIR overrides the .so directory.
use std::path::Path;

fn main() {
    let home = std::env::var("HOME").unwrap_or_default();
    let default_dir =
        format!("{home}/npuvox-asr-bench/.venv/lib/python3.12/site-packages/onnxruntime/capi");
    let ort_dir = std::env::var("ORT_LIB_DIR").unwrap_or(default_dir);

    // find the real versioned .so
    let real = std::fs::read_dir(&ort_dir)
        .unwrap_or_else(|e| panic!("ORT_LIB_DIR {ort_dir}: {e}"))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.starts_with("libonnxruntime.so."))
                .unwrap_or(false)
        })
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
    // export the SONAME-symlink dir to dependents (as DEP_ONNXRUNTIME_RPATH) so the final binary
    // can bake the same rpath and find libonnxruntime.so.1 at runtime.
    println!("cargo:rpath={out}");
}
