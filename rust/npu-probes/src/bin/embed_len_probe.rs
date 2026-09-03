//! How the WeSpeaker embedder's cost scales with the LENGTH of the crop it is given.
//!
//! The pipeline hands it a whole 10 s window plus a per-frame weights mask, because that is what
//! pyannote does. If the graph accepts a shorter waveform, the frames outside a speaker's active
//! span carry weight 0 and contribute nothing to the pooled statistic -- so their cost is paid for
//! nothing, and this measures how much of the cost that is.
//!
//!   rust/target/release/embed_len_probe [model]
use std::path::Path;
use std::time::Instant;

use npu_engine::diarize::types::Manifest;
use npu_onnx::{Env, Session, Tensor};

fn main() {
    let model = std::env::args().nth(1)
        .unwrap_or_else(|| "speaker-diarization-3.1".to_string());
    let dir = Path::new("artifacts/pyannote").join(&model);
    let m: Manifest = serde_json::from_str(
        &std::fs::read_to_string(dir.join("diarize.json")).expect("diarize.json")).expect("manifest");
    let env = Env::new().expect("env");
    let sess = Session::load(&env, dir.join(&m.embedding.onnx).to_str().unwrap()).expect("session");

    // Frame count for a given sample count, under kaldi snip_edges framing -- the same rule the
    // manifest's n_frames records for the full window. Getting this wrong is not a slow path:
    // onnxruntime rejects a weights mask whose length disagrees with the graph's frame count.
    let flen = (m.embedding.frame_length_ms / 1000.0 * m.sample_rate as f32) as usize;
    let fshift = (m.embedding.frame_shift_ms / 1000.0 * m.sample_rate as f32) as usize;
    let frames_for = |n: usize| if n < flen { 0 } else { (n - flen) / fshift + 1 };

    println!("model {model}  full window {:.1}s -> {} frames (manifest)",
             m.segmentation.duration_s, m.embedding.n_frames);
    println!("{:>8} {:>8} {:>10} {:>10} {:>12}", "crop_s", "frames", "run_ms", "vs_10s", "ms_per_100f");

    let full = (m.segmentation.duration_s * m.sample_rate as f32) as usize;
    let mut base_ms = 0.0f64;
    for &secs in &[10.0f32, 8.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.5, 1.0, 0.5] {
        let n = ((secs * m.sample_rate as f32) as usize).min(full);
        let nf = frames_for(n);
        if nf == 0 { continue }
        let wav = vec![0.01f32; n];
        let msk = vec![1.0f32; nf];
        let wt = Tensor::F32(&wav, vec![1, n as i64]);
        let mt = Tensor::F32(&msk, vec![1, nf as i64]);
        // One warm run before timing: the first call through a fresh shape allocates the arena
        // for it, which is a one-off the steady-state cost does not carry.
        match sess.run(&[("waveform", wt), ("weights", mt)], &["embedding"]) {
            Ok(_) => {}
            Err(e) => { println!("{secs:>8.1} {nf:>8}   REJECTED: {e}"); continue }
        }
        let reps = 3;
        let t0 = Instant::now();
        for _ in 0..reps {
            let wt = Tensor::F32(&wav, vec![1, n as i64]);
            let mt = Tensor::F32(&msk, vec![1, nf as i64]);
            sess.run(&[("waveform", wt), ("weights", mt)], &["embedding"]).expect("run");
        }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / reps as f64;
        if base_ms == 0.0 { base_ms = ms; }
        println!("{secs:>8.1} {nf:>8} {ms:>10.1} {:>10.2}x {:>12.2}",
                 base_ms / ms, ms / (nf as f64 / 100.0));
    }
}
