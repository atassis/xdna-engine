//! Gates `npu_s2::gguf`'s q6_k decode against `scripts/s2_ar_ref.py`'s `read_tensor` (itself a
//! port of `ggml/src/ggml-quants.c:dequantize_row_q6_K`, verified there byte-exact against a
//! scalar transliteration of that C function -- see that module's self-test) on real AR tensors
//! from the S2-Pro checkpoint. Full tensors, not sampled indices: a block-boundary bug would hide
//! from a handful of probed values, so python dumps the WHOLE dequantized tensor to a temp file
//! and every element is compared bit-for-bit. Companion to `gguf_matches_python_rail.rs`, which
//! gates the F16 decoder side against `gguf_extract.py` -- this is a separate rail
//! (`s2_ar_ref.py` is the one module that knows q6_k) on the AR side of the same checkpoint.
//!
//! Skips (does not fail) when python3/numpy or the GGUF file aren't available, same convention
//! as the sibling test.

use std::path::{Path, PathBuf};
use std::process::Command;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

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
    let p = PathBuf::from(String::from_utf8(out.stdout).ok()?.trim());
    p.is_file().then_some(p)
}

/// Ask `s2_ar_ref.read_tensor` to dequantize `name` and dump it as raw little-endian f32, flat
/// in the same row-major order `GgufFile::tensor_f32` produces (both start from ggml's `ne`,
/// fastest-dim-first, reversed to numpy shape -- see `gguf.rs`'s module doc). Returns the numpy
/// shape python reports, so the test can check shape agreement independently of the byte compare.
fn dump_python_tensor(gguf: &Path, name: &str, out_path: &Path) -> Option<Vec<usize>> {
    let script = format!(
        "import sys; sys.path.insert(0, 'scripts')\n\
         import s2_ar_ref as ar\n\
         gg = ar.open_gguf({gguf:?})\n\
         arr = ar.read_tensor(gg, {name:?})\n\
         print(','.join(str(d) for d in arr.shape))\n\
         arr.astype('<f4').tofile({out_path:?})\n",
    );
    let out = Command::new("python3").arg("-c").arg(&script).current_dir(repo_root()).output().ok()?;
    if !out.status.success() {
        eprintln!("python3 s2_ar_ref dump of `{name}` failed: {}", String::from_utf8_lossy(&out.stderr));
        return None;
    }
    let text = String::from_utf8(out.stdout).ok()?;
    let shape_s = text.trim();
    Some(if shape_s.is_empty() { vec![] } else { shape_s.split(',').map(|d| d.parse().unwrap()).collect() })
}

/// One tensor's exact-match report against the python dump. Panics on any mismatch, printing the
/// first differing index and both values -- there is no tolerance here, q6_k dequant is
/// deterministic integer/scale arithmetic and a mismatch of any size is a bug, not noise.
fn check_tensor(g: &npu_s2::gguf::GgufFile, gguf: &Path, name: &str) {
    let td = tempfile::tempdir().unwrap();
    let dump_path = td.path().join("t.bin");
    let Some(py_shape) = dump_python_tensor(gguf, name, &dump_path) else {
        eprintln!("skip: python3 s2_ar_ref dump unavailable for `{name}`");
        return;
    };

    let rust_shape = g.shape(name).unwrap();
    assert_eq!(rust_shape, py_shape, "{name}: shape");

    let py_bytes = std::fs::read(&dump_path).unwrap();
    let py_vals: Vec<f32> =
        py_bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    let rust_vals = g.tensor_f32(name).unwrap();

    assert_eq!(rust_vals.len(), py_vals.len(), "{name}: element count");
    let n = rust_vals.len();
    let mismatches: Vec<usize> =
        (0..n).filter(|&i| rust_vals[i].to_bits() != py_vals[i].to_bits()).collect();
    println!(
        "{name}: shape={rust_shape:?} n={n} bit-exact matches={}/{n}",
        n - mismatches.len()
    );
    if let Some(&i) = mismatches.first() {
        panic!(
            "{name}: {} / {n} elements mismatch, first at index {i}: rust={} ({:#010x}) python={} ({:#010x})",
            mismatches.len(),
            rust_vals[i],
            rust_vals[i].to_bits(),
            py_vals[i],
            py_vals[i].to_bits(),
        );
    }
}

#[test]
fn q6k_decode_matches_python_rail_on_real_ar_tensors() {
    let Some(gguf) = python_gguf_path() else {
        eprintln!("skip: no python3/codec_paths/GGUF available (set $S2_GGUF to force)");
        return;
    };
    let g = npu_s2::gguf::GgufFile::open(&gguf).expect("GgufFile::open");

    // Three different shapes/orientations, per the task: an attention weight, an FFN weight
    // (w2's ne is transposed relative to w1/w3 -- (9728,2560) vs (2560,9728) file order), and
    // the vocab embedding table (398,786,560 elements, 1,557,760 blocks) as the big one. Every
    // q6_k tensor in this checkpoint has numel % 256 == 0 (2560/9728/40960/155776 are all
    // multiples of 256) -- there is no non-block-aligned tensor to add here.
    for name in [
        "layers.0.attention.wqkv.weight",
        "layers.0.feed_forward.w2.weight",
        "embeddings.weight",
    ] {
        check_tensor(&g, &gguf, name);
    }
}
