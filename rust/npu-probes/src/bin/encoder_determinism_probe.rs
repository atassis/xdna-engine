//! Run-to-run determinism probe for the shipped Whisper NPU encoder (`WhisperEncoder::new_npu`,
//! the `two_ctx` default path -- `npu_asr::ctx2` for the block matmuls, `npu_asr::ctx_ln` for
//! LayerNorm). Dispatches `forward_last` TWICE on the SAME mel input in one process and checks the
//! two device outputs bit-for-bit.
//!
//! Exists because every device-gated check on this path (`verify_whisper --npu`,
//! `verify_whisper_decode`, the `npu-sr` gates) compares against a rel-L2/argmax oracle, never
//! against a second live run -- so a host-only-BO stale readback (see KB
//! `npu-hostonly-bo-coherency-race`, second manifestation 2026-09-02 in `rust/npu-s2`: the STALE
//! UNIT was the whole output buffer one dispatch behind, and 2 of 4 corrupted runs still passed a
//! 3e-2 rel-L2 gate) would need a run-to-run check to be caught at all. Reporting mirrors
//! `npu-s2/src/bin/s2_chain_probe.rs`'s run2run block.
//!
//!   encoder_determinism_probe [--turbo] [artifacts_dir]
//!
//! Run from the worktree root (the `artifacts/` symlink and `mlir-aie/.../whole_array/build` are
//! resolved relative to cwd, same convention as `verify_whisper.rs --npu`). NPU is single-tenant --
//! quiesce it first (stop flm-asr.service/voxd.service, `fuser` check) before running this.

use std::path::Path;

use ndarray::Axis;
use npu_whisper::config::WhisperCfg;
use npu_whisper::encoder::WhisperEncoder;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let turbo = args.iter().any(|a| a == "--turbo");
    let artifacts = args
        .iter()
        .find(|a| !a.starts_with("--"))
        .cloned()
        .unwrap_or_else(|| if turbo { "artifacts/whisper-turbo".into() } else { "artifacts/whisper-small".into() });
    let cfg = if turbo { WhisperCfg::TURBO } else { WhisperCfg::SMALL };

    let enc = WhisperEncoder::new_npu(Path::new(&artifacts), cfg, Path::new("."));
    let mel = enc
        .weights()
        .ref_tensor("input_features")
        .index_axis(Axis(0), 0)
        .to_owned()
        .into_dimensionality::<ndarray::Ix2>()
        .expect("input_features not [1, n_mels, T]");

    println!("encoder_determinism_probe: {}-layer Whisper NPU encoder, dispatching run 1...", cfg.n_layers);
    let out1 = enc.forward_last(&mel);
    println!("encoder_determinism_probe: dispatching run 2 (identical mel input)...");
    let out2 = enc.forward_last(&mel);

    assert_eq!(out1.dim(), out2.dim(), "output shape changed between runs: {:?} vs {:?}", out1.dim(), out2.dim());
    let (o1, _) = out1.into_raw_vec_and_offset();
    let (o2, _) = out2.into_raw_vec_and_offset();

    let deterministic = o1.iter().zip(&o2).all(|(a, b)| a.to_bits() == b.to_bits());
    if !deterministic {
        let diff: Vec<usize> = (0..o1.len()).filter(|&i| o1[i].to_bits() != o2[i].to_bits()).collect();
        let maxd = diff.iter().map(|&i| (o1[i] - o2[i]).abs()).fold(0f32, f32::max);
        println!(
            "encoder_determinism_probe: run2run differs in {}/{} elements ({:.2}%), first {:?} last {:?}, max |delta| {maxd:.6e}",
            diff.len(), o1.len(), 100.0 * diff.len() as f64 / o1.len() as f64,
            diff.first(), diff.last()
        );
    }
    println!("encoder_determinism_probe: run2run={}", if deterministic { "bit-identical" } else { "MISMATCH" });
    println!("encoder_determinism_probe: {}", npu_xrt::context_report());
    if !deterministic {
        eprintln!("encoder_determinism_probe: FAIL: run-to-run determinism check FAILED");
        std::process::exit(1);
    }
}
