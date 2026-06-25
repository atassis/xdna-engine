// Compiles a tiny C program against the generated header + the cdylib and runs it. Calls only
// npu_available() + npu_last_error() so it needs NO NPU device (host-safe). Proves the C ABI links
// and is callable from C. Links the cdylib (not the staticlib) so XRT/native deps are already
// resolved in the .so. Skips cleanly if cc or the cdylib are absent (run `cargo build -p npu-capi`
// first to produce target/<profile>/libnpu_engine.so).

use std::path::PathBuf;
use std::process::Command;

fn target_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().join("target")
}

fn cdylib_dir() -> Option<PathBuf> {
    for profile in ["debug", "release"] {
        let dir = target_dir().join(profile);
        if dir.join("libnpu_engine.so").exists() { return Some(dir); }
    }
    None
}

/// The npu-onnx build copies libonnxruntime.so.1 into its OUT_DIR; the cdylib needs it on
/// LD_LIBRARY_PATH at runtime (same requirement as every binary in this repo). Find one.
fn onnxruntime_dir() -> Option<PathBuf> {
    for profile in ["debug", "release"] {
        let build = target_dir().join(profile).join("build");
        let Ok(entries) = std::fs::read_dir(&build) else { continue };
        for e in entries.flatten() {
            let out = e.path().join("out");
            if out.join("libonnxruntime.so.1").exists() { return Some(out); }
        }
    }
    None
}

#[test]
fn c_program_links_and_calls_npu_available() {
    if Command::new("cc").arg("--version").output().is_err() {
        eprintln!("SKIP c_smoke: no `cc` compiler");
        return;
    }
    let Some(libdir) = cdylib_dir() else {
        eprintln!("SKIP c_smoke: libnpu_engine.so not built - run `cargo build -p npu-capi` first");
        return;
    };
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let out = PathBuf::from(env!("OUT_DIR"));
    let src = out.join("smoke.c");
    std::fs::write(&src, r#"
#include "npu_engine.h"
#include <stdio.h>
int main(void) {
    int a = npu_available();
    const char* e = npu_last_error();
    printf("available=%d err=%s\n", a, e ? e : "(null)");
    return 0;
}
"#).unwrap();
    let exe = out.join("smoke");
    let status = Command::new("cc")
        .arg(&src)
        .arg(format!("-I{}", manifest.join("include").display()))
        .arg(format!("-L{}", libdir.display()))
        .arg(format!("-Wl,-rpath,{}", libdir.display()))
        .arg("-lnpu_engine")
        .arg("-o").arg(&exe)
        .status().expect("cc invocation");
    assert!(status.success(), "C program failed to compile/link against libnpu_engine.so");
    let mut ldpath = libdir.display().to_string();
    if let Some(ort) = onnxruntime_dir() { ldpath = format!("{}:{ldpath}", ort.display()); }
    if let Ok(existing) = std::env::var("LD_LIBRARY_PATH") { ldpath = format!("{ldpath}:{existing}"); }
    let run = Command::new(&exe)
        .env("LD_LIBRARY_PATH", ldpath)
        .output().expect("run smoke");
    let stdout = String::from_utf8_lossy(&run.stdout);
    assert!(run.status.success(), "smoke exited nonzero: {}", String::from_utf8_lossy(&run.stderr));
    assert!(stdout.contains("available="), "unexpected output: {stdout}");
}
