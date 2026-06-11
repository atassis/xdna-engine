// Bake the rpath to onnxruntime's SONAME-symlink dir (exported by npu-onnx via DEP_ONNXRUNTIME_RPATH)
// into the final binary, so `asr_serve` finds libonnxruntime.so.1 at runtime without LD_LIBRARY_PATH.
fn main() {
    if let Ok(dir) = std::env::var("DEP_ONNXRUNTIME_RPATH") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{dir}");
    }
}
