// Compile the C++ XRT shim and link it against the system XRT runtime (the same
// libxrt_coreutil.so that pyxrt links). XRT headers live under /usr/include/xrt on this box.
fn main() {
    let xrt_inc = std::env::var("XRT_INC_DIR").unwrap_or_else(|_| "/usr/include".to_string());
    let xrt_lib = std::env::var("XRT_LIB_DIR").unwrap_or_else(|_| "/usr/lib".to_string());

    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .file("shim/xrt_shim.cpp")
        .include("shim")
        .include(&xrt_inc)
        .warnings(false)
        .compile("xrt_shim");

    println!("cargo:rustc-link-search=native={xrt_lib}");
    println!("cargo:rustc-link-lib=dylib=xrt_coreutil");
    println!("cargo:rerun-if-changed=shim/xrt_shim.cpp");
    println!("cargo:rerun-if-changed=shim/xrt_shim.h");
    println!("cargo:rerun-if-env-changed=XRT_INC_DIR");
    println!("cargo:rerun-if-env-changed=XRT_LIB_DIR");
}
