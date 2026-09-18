//! Device probe for the FULL window/chunk driver loop (`npu_s2::chain::S2DecoderChain`), as
//! opposed to `s2_design_probe`'s single-design probe. Runs one piece of the decoder chain --
//! head / one stage / tail / the whole chain -- against externally supplied golden `.bin` files
//! (raw little-endian f32, no header), gated on rel-L2 + run-to-run determinism, same convention
//! `s2_design_probe` already uses. Generate the goldens with the matching Python call
//! (`decoder_chain.run_head`/`run_stage`/`run_tail`/`run_chain`) on the SAME input.
//!
//!   s2_chain_probe <artifact_dir> <gguf_path> <part> <input.bin> <expected.bin> [out.wav]
//!
//! For `chain`, `expected.bin` is the FULL-clip audio and the comparison window is computed with
//! `S2DecoderChain::chain_offset` -- the chain convolves with valid (unpadded) windows, so its
//! output is shorter than the reference rail's and starts later in it. `NPU_S2_LATENT_FRAMES=N`
//! truncates the latent to its first N frames; because every op is a local sliding window, the
//! result is bit-identical to the corresponding slice of a full-length run, which makes a short
//! segment a real end-to-end gate rather than a different computation.
//!
//! `part` is one of `head`, `stage1`..`stage4`, `tail`, `chain`. `input.bin`'s channel count is
//! implied by `part` (1024 for head/chain, the stage's own c_in from the GGUF for stageN, 96 for
//! tail); its length is inferred from the file size. NPU is single-tenant -- this touches the
//! device, run it only where the caller has already quiesced it.

use std::path::Path;
use std::rc::Rc;

use ndarray::Array2;
use npu_s2::chain::{S2DecoderChain, StageWeights};
use npu_s2::weights::S2Weights;
use npu_s2::S2Artifacts;
use npu_xrt::Device;

const REL_L2_GATE: f64 = 3.0e-02;

/// Used only if the GGUF does not declare `fish_speech.codec.sample_rate`, matching the fallback
/// `s2_codec.cpp:701` applies before `:826` overwrites it from that same key.
const FALLBACK_SAMPLE_RATE: u32 = 44_100;

/// Mono 16-bit PCM. The chain's output is already tanh'd into [-1, 1], so the only clamping this
/// does is against a sample landing exactly on the boundary.
fn write_wav(path: &Path, samples: &[f32], sample_rate: u32) -> Result<(), String> {
    let bytes_per_sample = 2u32;
    let data_len = samples.len() as u32 * bytes_per_sample;
    let mut w: Vec<u8> = Vec::with_capacity(44 + data_len as usize);
    w.extend_from_slice(b"RIFF");
    w.extend_from_slice(&(36 + data_len).to_le_bytes());
    w.extend_from_slice(b"WAVEfmt ");
    w.extend_from_slice(&16u32.to_le_bytes()); // PCM fmt chunk size
    w.extend_from_slice(&1u16.to_le_bytes()); // format = PCM
    w.extend_from_slice(&1u16.to_le_bytes()); // channels
    w.extend_from_slice(&sample_rate.to_le_bytes());
    w.extend_from_slice(&(sample_rate * bytes_per_sample).to_le_bytes()); // byte rate
    w.extend_from_slice(&(bytes_per_sample as u16).to_le_bytes()); // block align
    w.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    w.extend_from_slice(b"data");
    w.extend_from_slice(&data_len.to_le_bytes());
    for v in samples {
        let q = (v.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16;
        w.extend_from_slice(&q.to_le_bytes());
    }
    std::fs::write(path, &w).map_err(|e| format!("write {}: {e}", path.display()))
}

fn read_f32_file(path: &Path) -> Result<Vec<f32>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    if bytes.len() % 4 != 0 {
        return Err(format!("{}: {} bytes is not a whole number of f32 elements", path.display(), bytes.len()));
    }
    Ok(bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

fn rel_l2(got: &[f32], want: &[f32]) -> f64 {
    let mut num = 0f64;
    let mut den = 0f64;
    for (g, w) in got.iter().zip(want) {
        let d = *g as f64 - *w as f64;
        num += d * d;
        den += (*w as f64) * (*w as f64);
    }
    let num = num.sqrt();
    if den > 0.0 {
        num / den.sqrt()
    } else {
        num
    }
}

fn input_channels(part: &str, weights: &S2Weights) -> Result<usize, String> {
    Ok(match part {
        "head" | "chain" => 1024,
        "tail" => 96,
        s if s.starts_with("stage") => {
            let stage: u32 = s[5..].parse().map_err(|_| format!("bad stage part `{s}`"))?;
            weights.stage_conv_transpose(stage).map_err(|e| e.to_string())?.0.dim().0
        }
        other => return Err(format!("unknown part `{other}` (want head/stageN/tail/chain)")),
    })
}

fn run(part: &str, chain: &S2DecoderChain, weights: &S2Weights, x: &Array2<f32>) -> Result<Vec<f32>, String> {
    match part {
        "head" => chain.run_head(x, weights).map(|a| a.into_raw_vec_and_offset().0).map_err(|e| e.to_string()),
        "tail" => chain.run_tail(x, weights).map(|a| a.into_raw_vec_and_offset().0).map_err(|e| e.to_string()),
        "chain" => chain.run_chain(x, weights).map(|a| a.into_raw_vec_and_offset().0).map_err(|e| e.to_string()),
        s if s.starts_with("stage") => {
            let stage: u32 = s[5..].parse().map_err(|_| format!("bad stage part `{s}`"))?;
            let sw = StageWeights::load(weights, stage).map_err(|e| e.to_string())?;
            chain.run_stage(x, stage, &sw).map(|a| a.into_raw_vec_and_offset().0).map_err(|e| e.to_string())
        }
        other => Err(format!("unknown part `{other}`")),
    }
}

fn main_inner() -> Result<(), String> {
    let argv: Vec<String> = std::env::args().collect();
    if argv.len() != 6 && argv.len() != 7 {
        return Err(format!(
            "usage: {} <artifact_dir> <gguf_path> <head|stageN|tail|chain> <input.bin> \
             <expected.bin> [out.wav]",
            argv.first().map(String::as_str).unwrap_or("s2_chain_probe")
        ));
    }
    let artifact_dir = Path::new(&argv[1]);
    let gguf_path = Path::new(&argv[2]);
    let part = argv[3].as_str();
    let input_flat = read_f32_file(Path::new(&argv[4]))?;
    let expected = read_f32_file(Path::new(&argv[5]))?;

    let weights = S2Weights::open(gguf_path).map_err(|e| format!("S2Weights::open: {e}"))?;
    let c_in = input_channels(part, &weights)?;
    if input_flat.len() % c_in != 0 {
        return Err(format!("input.bin has {} f32 elements, not a multiple of c_in={c_in}", input_flat.len()));
    }
    // LAYOUT, and it is not the obvious one. An oracle dump carries a `.shape` sidecar in ggml `ne`
    // order (ne[0] is the FASTEST axis), so "1024 264" means 264 frames of 1024 contiguous channels
    // -- i.e. [frames, channels] on disk, which must be transposed to the [channel, time] the graph
    // works in. `scripts/codec_decoder_ref.py::main` does exactly this. Reading it as [1024, 264]
    // row-major instead silently scrambles the latent and still decodes to plausible-looking audio.
    // Files WE dump (`NPU_S2_DUMP`) carry no sidecar and are already [channel, time].
    let shape_path = Path::new(&argv[4]).with_extension("shape");
    let mut x = if shape_path.exists() {
        let txt = std::fs::read_to_string(&shape_path)
            .map_err(|e| format!("read {}: {e}", shape_path.display()))?;
        let dims: Vec<usize> = txt
            .split_whitespace()
            .map(|v| v.parse::<usize>().map_err(|e| format!("{}: {e}", shape_path.display())))
            .collect::<Result<_, _>>()?;
        if dims.len() != 2 {
            return Err(format!("{}: expected 2 dims, got {dims:?}", shape_path.display()));
        }
        let (ne0, ne1) = (dims[0], dims[1]);
        if ne0 != c_in {
            return Err(format!(
                "{}: ne[0]={ne0} but part `{part}` wants {c_in} channels",
                shape_path.display()
            ));
        }
        println!("s2_chain_probe: {} is ggml ne order {ne0}x{ne1}; transposing to [channel, time]", shape_path.display());
        Array2::from_shape_vec((ne1, ne0), input_flat).map_err(|e| e.to_string())?.reversed_axes().as_standard_layout().to_owned()
    } else {
        let l = input_flat.len() / c_in;
        Array2::from_shape_vec((c_in, l), input_flat).map_err(|e| e.to_string())?
    };
    if let Ok(n) = std::env::var("NPU_S2_LATENT_FRAMES") {
        let n: usize = n.parse().map_err(|_| format!("NPU_S2_LATENT_FRAMES={n} is not a number"))?;
        if n == 0 || n > x.dim().1 {
            return Err(format!("NPU_S2_LATENT_FRAMES={n} outside 1..={}", x.dim().1));
        }
        x = x.slice(ndarray::s![.., ..n]).to_owned();
        println!("s2_chain_probe: truncated latent to {n} frames");
    }
    let l = x.dim().1;

    let dev = Rc::new(Device::open(0).map_err(|e| format!("Device::open(0): {e}"))?);
    let artifacts = S2Artifacts::open(artifact_dir).map_err(|e| format!("S2Artifacts::open: {e}"))?;
    let chain = S2DecoderChain::open(&dev, &artifacts).map_err(|e| format!("S2DecoderChain::open: {e}"))?;

    // The chain runs valid windows, so its output is a strict interior slice of the reference
    // rail's -- compare against that window, not against the whole file.
    let expected = if part == "chain" {
        let (start, len) = chain.chain_offset(0, l as i64).map_err(|e| e.to_string())?;
        let (start, len) = (start as usize, len as usize);
        let end = start.checked_add(len).filter(|&e| e <= expected.len()).ok_or_else(|| {
            format!("chain_offset window {start}..{} exceeds expected.bin ({})", start + len, expected.len())
        })?;
        println!("s2_chain_probe: comparing audio[{start}..{end}] of {}", expected.len());
        expected[start..end].to_vec()
    } else {
        expected
    };

    let out1 = run(part, &chain, &weights, &x)?;
    // Dumped before any gate, so a failing part can be aligned against the reference rail's own
    // per-stage tensors offline instead of being re-run once per hypothesis.
    if let Ok(path) = std::env::var("NPU_S2_DUMP") {
        let mut raw = Vec::with_capacity(out1.len() * 4);
        for v in &out1 {
            raw.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(&path, &raw).map_err(|e| format!("write {path}: {e}"))?;
        println!("s2_chain_probe: dumped {} f32 to {path}", out1.len());
    }
    let out2 = run(part, &chain, &weights, &x)?;
    if out1.len() != expected.len() {
        return Err(format!("{part}: output has {} elements, expected.bin has {}", out1.len(), expected.len()));
    }
    let deterministic = out1.iter().zip(&out2).all(|(a, b)| a.to_bits() == b.to_bits());
    if !deterministic {
        // Quantify rather than just flag it: WHERE and HOW MUCH two runs diverge separates a
        // whole-output corruption from a bounded region, which is the difference between a wrong
        // computation and stale state in one buffer.
        let diff: Vec<usize> = (0..out1.len()).filter(|&i| out1[i].to_bits() != out2[i].to_bits()).collect();
        let maxd = diff.iter().map(|&i| (out1[i] - out2[i]).abs()).fold(0f32, f32::max);
        println!(
            "s2_chain_probe: run2run differs in {}/{} elements ({:.2}%), first {:?} last {:?}, max |delta| {maxd:.6e}",
            diff.len(), out1.len(), 100.0 * diff.len() as f64 / out1.len() as f64,
            diff.first(), diff.last()
        );
        if let Ok(path) = std::env::var("NPU_S2_DUMP2") {
            let mut raw = Vec::with_capacity(out2.len() * 4);
            for v in &out2 {
                raw.extend_from_slice(&v.to_le_bytes());
            }
            std::fs::write(&path, &raw).map_err(|e| format!("write {path}: {e}"))?;
            println!("s2_chain_probe: dumped run 2 to {path}");
        }
    }
    let rel = rel_l2(&out1, &expected);
    println!(
        "s2_chain_probe: part={part} rel-L2={rel:.6e} (gate {REL_L2_GATE:.1e}) run2run={}",
        if deterministic { "bit-identical" } else { "MISMATCH" }
    );
    // Written before the gates decide, so a FAILING run is still listenable -- which of the two
    // conventions a chain got wrong is usually audible long before it is visible in a rel-L2.
    if let Some(wav) = argv.get(6) {
        let (rate, src) = match weights.sample_rate() {
            Some(r) => (r, "gguf"),
            None => (FALLBACK_SAMPLE_RATE, "fallback"),
        };
        write_wav(Path::new(wav), &out1, rate)?;
        println!("s2_chain_probe: wrote {wav} ({} samples @ {rate} Hz [{src}])", out1.len());
    }
    println!("s2_chain_probe: {}", npu_xrt::context_report());
    let hits = npu_s2::resync_hits();
    let disp = npu_s2::dispatch_count();
    if hits > 0 || disp > 0 {
        println!(
            "s2_chain_probe: stale-read re-reads: {hits} over {disp} dispatches ({:.3}%)",
            if disp > 0 { 100.0 * hits as f64 / disp as f64 } else { 0.0 }
        );
        for (op, d, st) in npu_s2::per_op_stats().into_iter().filter(|(_, _, s)| *s > 0) {
            println!("s2_chain_probe:   {op}: {st} stale over {d} dispatches ({:.2}%)",
                     if d > 0 { 100.0 * st as f64 / d as f64 } else { 0.0 });
        }
    }
    if !deterministic {
        return Err("run-to-run determinism check FAILED".into());
    }
    if rel.is_nan() || rel > REL_L2_GATE {
        return Err(format!("rel-L2 {rel:.6e} exceeds gate {REL_L2_GATE:.1e}"));
    }
    Ok(())
}

fn main() {
    if let Err(e) = main_inner() {
        eprintln!("s2_chain_probe: FAIL: {e}");
        std::process::exit(1);
    }
}
