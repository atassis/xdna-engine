//! Device smoke test for the full window/chunk driver loop: build [`S2DecoderChain`] against a
//! real exported artifact set + the real GGUF, run `run_head` on a deterministic all-zero latent,
//! and check the output shape/finiteness (no golden comparison -- `s2_chain_probe` covers rel-L2
//! against a Python-generated expected.bin; this just checks the loop doesn't panic or dispatch
//! garbage). `#[ignore]`d: opens the NPU device. NPU is single-tenant -- run only where the caller
//! has already quiesced it (announce + `fuser` per project convention).
//!
//!   S2_ARTIFACTS_ROOT=<dir with manifest.json> S2_GGUF=<path> \
//!     cargo test -p npu-s2 --test chain_device_smoke -- --ignored --nocapture

use std::rc::Rc;

use ndarray::Array2;
use npu_s2::chain::S2DecoderChain;
use npu_s2::weights::S2Weights;
use npu_s2::S2Artifacts;
use npu_xrt::Device;

#[test]
#[ignore = "opens the NPU device (Device::open + S2DecoderChain::open load every design's xclbin); \
            needs $S2_ARTIFACTS_ROOT (a real 68-design export) and $S2_GGUF (the real checkpoint)"]
fn run_head_on_device_produces_finite_output_of_the_expected_shape() {
    let artifacts_root = std::env::var("S2_ARTIFACTS_ROOT").expect("set S2_ARTIFACTS_ROOT");
    let gguf = std::env::var("S2_GGUF").expect("set S2_GGUF");

    let dev = Rc::new(Device::open(0).expect("Device::open(0)"));
    let artifacts = S2Artifacts::open(std::path::Path::new(&artifacts_root)).expect("S2Artifacts::open");
    let chain = S2DecoderChain::open(&dev, &artifacts).expect("S2DecoderChain::open");
    let weights = S2Weights::open(std::path::Path::new(&gguf)).expect("S2Weights::open");

    // Smallest latent length that survives every stage's window assertion (mirrors
    // export_codec_artifacts.py's `_min_latent_len`), fed as all-zero -- head alone only needs to
    // clear its own CTX_HEAD, so any L > CTX_HEAD works; 128 is comfortably over.
    let l = 128;
    let z = Array2::<f32>::zeros((1024, l));

    let out = chain.run_head(&z, &weights).expect("run_head");
    let (audio_start, _) = chain.chain_offset(0, l as i64).expect("chain_offset");
    assert!(audio_start > 0, "sanity: chain_offset should report a positive audio_start");
    assert_eq!(out.dim().0, 1536, "head output channel count");
    assert_eq!(out.dim().1, l - 6, "head output length (L - CTX_HEAD)");
    assert!(out.iter().all(|v| v.is_finite()), "head output contains a NaN/inf");
}
