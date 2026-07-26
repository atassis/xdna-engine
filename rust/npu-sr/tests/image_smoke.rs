//! End-to-end CPU upscale of a small synthetic image -> correct output dims + no panic. Proves the whole
//! API path (RGB8 -> YCbCr -> Y-frontier + bicubic chroma -> RGB8) composes. Real-image PSNR lives in
//! the frontier gate + bench. Runs from repo root (checkpoint path is repo-root-relative).
use npu_sr::SrEngine;

#[test]
fn upscales_synthetic_image_cpu() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    if !root.join("target/test-checkpoints/espcn.safetensors").exists() {
        eprintln!("SKIP: checkpoint missing - run `cargo test -p npu-weights parity_espcn` first");
        return;
    }
    std::env::set_current_dir(&root).unwrap();
    let mut eng = SrEngine::load("artifacts/espcn/espcn.json", false).unwrap();
    let (w, h) = (16, 16);
    let rgb: Vec<u8> = (0..w * h * 3).map(|i| (i % 256) as u8).collect();
    let (out, ow, oh) = eng.upscale_rgb8(&rgb, w, h).unwrap();
    assert_eq!((ow, oh), (w * eng.scale(), h * eng.scale()));
    assert_eq!(out.len(), ow * oh * 3);
}
