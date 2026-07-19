fn main() {
    let crate_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let out = std::path::Path::new(&crate_dir).join("include/xdna_sr.h");
    std::fs::create_dir_all(std::path::Path::new(&crate_dir).join("include")).ok();
    if let Ok(cfg) = cbindgen::Config::from_file(format!("{crate_dir}/cbindgen.toml")) {
        if let Ok(b) = cbindgen::Builder::new()
            .with_crate(&crate_dir)
            .with_config(cfg)
            .generate()
        {
            b.write_to_file(&out);
        }
    }
    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=cbindgen.toml");
}
