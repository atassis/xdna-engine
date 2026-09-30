//! `npu_sr_fsr1` gates the closed FSR1 backends (`src/fsr1_frame.rs`, `src/fsr1_rt.rs`), which are
//! withheld from the public repo. Set only when both files are present in this checkout.
use std::path::Path;

fn main() {
    println!("cargo::rustc-check-cfg=cfg(npu_sr_fsr1)");
    println!("cargo:rerun-if-changed=src/fsr1_frame.rs");
    println!("cargo:rerun-if-changed=src/fsr1_rt.rs");
    println!("cargo:rerun-if-changed=src");
    if Path::new("src/fsr1_frame.rs").exists() && Path::new("src/fsr1_rt.rs").exists() {
        println!("cargo::rustc-cfg=npu_sr_fsr1");
    }
}
