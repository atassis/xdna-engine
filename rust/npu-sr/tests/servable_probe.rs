//! engine-open-capability-contract probe instance #2, functional gate (not just a compile check):
//! `SrEngine` driven ONLY through `npu_engine::capability::Servable` (never `upscale_rgb8` directly)
//! must produce byte-identical output to the direct call, on the real CPU frontier -- no NPU touched
//! (`use_npu=false`), so this runs on any lane, device or not. Checkpoint prereq: bake the schedule's
//! checkpoint first (`target/test-checkpoints/espcn.safetensors`, matching `espcn.json`'s `checkpoint`
//! field). It is the SAME baked copy the other npu-sr tests guard on -- `target/test-arenas/` was the
//! pre-rename path and nothing produces it, which is what made this probe and edsr_gate disagree.
use npu_engine::capability::{Request, Response, Servable};
use npu_sr::SrEngine;

#[test]
fn sr_engine_through_servable_matches_the_direct_call() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    if !root.join("target/test-checkpoints/espcn.safetensors").exists() {
        eprintln!("SKIP: checkpoint missing at target/test-checkpoints/espcn.safetensors -- bake via \
                   `npu-weights bake --source path:artifacts/espcn/espcn_x3_dyn.onnx --arch espcn \
                   --checkpoint target/test-checkpoints/espcn.safetensors` from the repo root");
        return;
    }
    std::env::set_current_dir(&root).unwrap();

    let (w, h) = (16, 16);
    let rgb: Vec<u8> = (0..w * h * 3).map(|i| (i % 256) as u8).collect();

    // The direct call -- today's real ABI, what npu-sr-capi actually invokes.
    let mut direct = SrEngine::load("artifacts/espcn/espcn.json", false).unwrap();
    let (direct_rgb, direct_w, direct_h) = direct.upscale_rgb8(&rgb, w, h).unwrap();

    // The SAME model, driven only through the trait object -- proves `Box<dyn Servable>` is a real
    // substitute for the direct ABI, not just a type that happens to compile.
    let mut boxed: Box<dyn Servable> = Box::new(SrEngine::load("artifacts/espcn/espcn.json", false).unwrap());
    assert_eq!(boxed.capabilities().0, "image-sr");
    let resp = boxed.run(Request::Image { rgb: rgb.clone(), w, h }).unwrap();
    let (via_rgb, via_w, via_h) = match resp {
        Response::Image { rgb, w, h } => (rgb, w, h),
        other => panic!("SrEngine::run returned a {} response for an Image request", other.shape()),
    };

    assert_eq!((via_w, via_h), (direct_w, direct_h));
    assert_eq!(via_rgb, direct_rgb, "Servable::run must be byte-identical to the direct upscale_rgb8 call");

    // Capability mismatch is a real Err, not a panic or a silently-wrong result.
    let err = boxed.run(Request::Text("wrong shape".into())).unwrap_err();
    // Matches on the SHAPE words, not the Rust variant path: the message is built from
    // `Request::shape()` so it stays readable when a request carries megabytes of PCM.
    let msg = err.to_string();
    assert!(msg.contains("image") && msg.contains("text"), "{err}");
}
