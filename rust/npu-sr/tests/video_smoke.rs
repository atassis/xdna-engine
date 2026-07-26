//! Generate a 5-frame test clip with ffmpeg, upscale it via the CLI pipeline, assert the output opens
//! with the right dims + frame count. Runs from repo root (checkpoint path is repo-root-relative).
use std::path::Path;
use std::process::Command;

#[test]
fn upscales_generated_clip_cpu() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    if !root.join("target/test-checkpoints/espcn.safetensors").exists() {
        eprintln!("SKIP: checkpoint missing");
        return;
    }
    std::env::set_current_dir(&root).unwrap();
    let dir = root.join("target/test-video");
    std::fs::create_dir_all(&dir).unwrap();
    let inp = dir.join("in.mp4");
    let outp = dir.join("out.mp4");
    // 5 frames, 32x32, testsrc.
    let st = Command::new("ffmpeg")
        .args([
            "-y", "-v", "error", "-f", "lavfi", "-i",
            "testsrc=size=32x32:rate=5:duration=1", "-frames:v", "5",
            inp.to_str().unwrap(),
        ])
        .status()
        .unwrap();
    assert!(st.success(), "ffmpeg gen failed");

    let mut eng = npu_sr::SrEngine::load("artifacts/espcn/espcn.json", false).unwrap();
    let stats = eng.upscale_file(&inp, &outp).unwrap();
    assert_eq!(stats.frames, 5, "expected 5 frames, got {}", stats.frames);
    assert!(outp.exists());

    // Probe output dims = 96x96 (32*3).
    let out = Command::new("ffprobe")
        .args([
            "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "csv=p=0",
            outp.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    let dims = String::from_utf8_lossy(&out.stdout);
    assert_eq!(dims.trim(), "96,96", "output dims = {dims:?}, want 96,96");
}
