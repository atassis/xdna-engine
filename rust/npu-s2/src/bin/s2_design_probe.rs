//! Device probe for one exported S2 codec design: loads it, dispatches twice, and gates on
//! rel-L2 vs a host-computed reference plus run-to-run determinism.
//!
//!   s2_design_probe <artifact_dir> <input.bin> <resident.bin|-> <expected.bin>
//!
//! `artifact_dir` holds `final.xclbin`/`insts.bin`/`meta.json` for ONE design (see
//! `npu_s2::S2Design`). `resident.bin` is `-` when the design has no resident operand. All three
//! data files are raw little-endian f32, no header. Exit 0 = both gates pass; non-zero + a message
//! on stderr otherwise. NPU is single-tenant -- this touches the device, run it only where the
//! caller has already quiesced it.

use std::path::Path;
use std::rc::Rc;

use npu_s2::S2Design;
use npu_xrt::Device;

const REL_L2_GATE: f64 = 3.0e-02;

fn read_f32_file(path: &Path) -> Result<Vec<f32>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    if bytes.len() % 4 != 0 {
        return Err(format!("{}: {} bytes is not a whole number of f32 elements", path.display(), bytes.len()));
    }
    Ok(bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

/// `||got-want|| / ||want||`. `want` all-zero (den 0) falls back to the raw L2 distance rather
/// than dividing by zero, so a zero-reference case still gets a meaningful, finite number.
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

fn run() -> Result<(), String> {
    let argv: Vec<String> = std::env::args().collect();
    if argv.len() != 5 {
        return Err(format!(
            "usage: {} <artifact_dir> <input.bin> <resident.bin|-> <expected.bin>",
            argv.first().map(String::as_str).unwrap_or("s2_design_probe")
        ));
    }
    let artifact_dir = Path::new(&argv[1]);
    let input = read_f32_file(Path::new(&argv[2]))?;
    let resident = if argv[3] == "-" { None } else { Some(read_f32_file(Path::new(&argv[3]))?) };
    let expected = read_f32_file(Path::new(&argv[4]))?;

    let dev = Rc::new(Device::open(0).map_err(|e| format!("Device::open(0): {e}"))?);
    let design = S2Design::open(&dev, artifact_dir).map_err(|e| format!("S2Design::open: {e}"))?;

    let out1 = design.dispatch(&input, resident.as_deref()).map_err(|e| format!("dispatch 1: {e}"))?;
    let out2 = design.dispatch(&input, resident.as_deref()).map_err(|e| format!("dispatch 2: {e}"))?;

    if out1.len() != expected.len() {
        return Err(format!("output has {} elements, expected.bin has {}", out1.len(), expected.len()));
    }

    let deterministic = out1.iter().zip(&out2).all(|(a, b)| a.to_bits() == b.to_bits());
    let rel = rel_l2(&out1, &expected);
    println!("s2_design_probe: {} rel-L2={rel:.6e} (gate {REL_L2_GATE:.1e}) run2run={}",
        artifact_dir.display(), if deterministic { "bit-identical" } else { "MISMATCH" });

    if !deterministic {
        return Err("run-to-run determinism check FAILED (two dispatches of the same input diverged)".into());
    }
    if rel.is_nan() || rel > REL_L2_GATE {
        return Err(format!("rel-L2 {rel:.6e} exceeds gate {REL_L2_GATE:.1e}"));
    }
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("s2_design_probe: FAIL: {e}");
        std::process::exit(1);
    }
}
