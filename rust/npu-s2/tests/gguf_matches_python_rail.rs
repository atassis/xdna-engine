//! Gates `npu_s2::gguf::GgufFile` against `scripts/gguf_extract.py`'s (the python rail's own
//! GGUF reader) live output on the REAL S2 decoder checkpoint. Nothing from the checkpoint is
//! committed anywhere -- both sides read the same on-disk GGUF at test time, and only a handful
//! of scalar float comparisons cross the process boundary (via python3 on stdout), never persisted.
//! `gguf_extract.py`/`codec_paths.py` have zero `aie.iron` dependency (by their own docstrings), so
//! this needs no toolchain env, no device.
//!
//! Skips (does not fail) when python3, numpy, or the GGUF file itself aren't available, matching
//! this crate's existing `S2_ARTIFACT_DIR`-gated test convention.

use std::path::{Path, PathBuf};
use std::process::Command;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Resolve the GGUF path via the python rail's OWN resolution order (`$S2_GGUF`, then the
/// sibling-checkout layout) rather than re-implementing `codec_paths.py`'s candidate search here.
fn python_gguf_path() -> Option<PathBuf> {
    let out = Command::new("python3")
        .arg("-c")
        .arg("import sys; sys.path.insert(0, 'scripts'); import codec_paths; print(codec_paths.gguf())")
        .current_dir(repo_root())
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8(out.stdout).ok()?;
    let p = PathBuf::from(s.trim());
    p.is_file().then_some(p)
}

struct PyCase {
    name: &'static str,
    /// Multi-index into the tensor's numpy (row-major) shape, or `&[]` for a 1-element tensor.
    index: &'static [usize],
}

/// Ask python (gguf_extract.load) for `shape` + `value at index` for each case, as
/// `name\tshape_csv\tvalue` lines on stdout.
fn python_values(gguf: &Path, cases: &[PyCase]) -> Option<Vec<(Vec<usize>, f64)>> {
    let spec: String = cases
        .iter()
        .map(|c| format!("({:?},{:?})", c.name, c.index))
        .collect::<Vec<_>>()
        .join(",");
    let script = format!(
        "import sys; sys.path.insert(0, 'scripts'); import gguf_extract as gx\n\
         cases = [{spec}]\n\
         for name, idx in cases:\n\
         \ta = gx.load({gguf:?}, name)\n\
         \tprint(','.join(str(d) for d in a.shape), float(a[tuple(idx)] if idx else a.reshape(-1)[0]), sep='\\t')\n",
    );
    let out = Command::new("python3").arg("-c").arg(&script).current_dir(repo_root()).output().ok()?;
    if !out.status.success() {
        eprintln!("python3 gguf_extract helper failed: {}", String::from_utf8_lossy(&out.stderr));
        return None;
    }
    let text = String::from_utf8(out.stdout).ok()?;
    let mut results = Vec::new();
    for line in text.lines() {
        let (shape_s, val_s) = line.split_once('\t')?;
        let shape: Vec<usize> = if shape_s.is_empty() {
            vec![]
        } else {
            shape_s.split(',').map(|d| d.parse().unwrap()).collect()
        };
        results.push((shape, val_s.parse().ok()?));
    }
    Some(results)
}

#[test]
fn gguf_reader_matches_python_rail_on_real_decoder_tensors() {
    let Some(gguf) = python_gguf_path() else {
        eprintln!("skip: no python3/codec_paths/GGUF available (set $S2_GGUF to force)");
        return;
    };

    let cases = [
        PyCase { name: "c.decoder.model.0.conv.bias", index: &[3] },
        PyCase { name: "c.decoder.model.0.conv.weight", index: &[700, 511, 3] },
        PyCase { name: "c.decoder.model.1.block.0.alpha", index: &[10, 0] },
        PyCase { name: "c.decoder.model.6.conv.weight", index: &[5, 2] },
        PyCase { name: "c.decoder.model.6.conv.bias", index: &[0] },
    ];
    let Some(py) = python_values(&gguf, &cases) else {
        eprintln!("skip: python3 cross-check helper unavailable");
        return;
    };
    assert_eq!(py.len(), cases.len());

    let g = npu_s2::gguf::GgufFile::open(&gguf).expect("GgufFile::open");
    for (case, (py_shape, py_val)) in cases.iter().zip(py) {
        let rust_shape = g.shape(case.name).unwrap();
        assert_eq!(rust_shape, py_shape, "{}: shape", case.name);
        let flat = g.tensor_f32(case.name).unwrap();
        // Row-major flat index from `case.index` against `rust_shape`.
        let mut idx = 0usize;
        let mut stride = 1usize;
        for (&d, &s) in case.index.iter().rev().zip(rust_shape.iter().rev()) {
            idx += d * stride;
            stride *= s;
        }
        let rust_val = flat[idx] as f64;
        assert!(
            (rust_val - py_val).abs() < 1e-4,
            "{}: rust={rust_val} python={py_val} (index {:?})",
            case.name, case.index
        );
    }
}
