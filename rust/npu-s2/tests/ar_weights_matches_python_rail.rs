//! Gates `npu_s2::ar::ArWeights` against `scripts/s2_ar_ref.py` (the one module that knows q6_k
//! and is this checkpoint's tensor-name authority) on real tensors from `s2-pro-q6_k.gguf`:
//! multiple slow layers, multiple fast layers, the two f16 top-level RMSNorm gammas (the AR set's
//! only non-q6_k tensors), and row-range reads on both embedding tables. Full tensors compared
//! bit-for-bit, not sampled indices -- matches the sibling `gguf_q6k_matches_python_rail.rs`
//! convention, and for the same reason: a block-boundary bug would hide from a handful of probed
//! values.
//!
//! Skips (does not fail) when python3/numpy or the GGUF file aren't available.

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

/// Dump `name` (optionally row-range `[lo,hi)`) via `s2_ar_ref.read_tensor`, raw little-endian f32,
/// flat in the row-major order both sides produce. Returns the numpy shape python reports.
fn dump_python_tensor(
    gguf: &Path,
    name: &str,
    row_range: Option<(usize, usize)>,
    out_path: &Path,
) -> Option<Vec<usize>> {
    let rr = match row_range {
        Some((lo, hi)) => format!("({lo},{hi})"),
        None => "None".to_string(),
    };
    let script = format!(
        "import sys; sys.path.insert(0, 'scripts')\n\
         import s2_ar_ref as ar\n\
         gg = ar.open_gguf({gguf:?})\n\
         arr = ar.read_tensor(gg, {name:?}, row_range={rr})\n\
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

/// Bit-exact check of one already-decoded Rust array against python's dump of the same tensor
/// (optionally row-ranged). `label` is only for the panic/println, `py_name` is the GGUF tensor
/// name python reads.
fn check(gguf: &Path, py_name: &str, row_range: Option<(usize, usize)>, label: &str, rust_shape: &[usize], rust_flat: &[f32]) {
    let td = tempfile::tempdir().unwrap();
    let dump_path = td.path().join("t.bin");
    let Some(py_shape) = dump_python_tensor(gguf, py_name, row_range, &dump_path) else {
        eprintln!("skip: python3 s2_ar_ref dump unavailable for `{py_name}`");
        return;
    };
    assert_eq!(rust_shape, py_shape, "{label}: shape");

    let py_bytes = std::fs::read(&dump_path).unwrap();
    let py_vals: Vec<f32> = py_bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    assert_eq!(rust_flat.len(), py_vals.len(), "{label}: element count");

    let n = rust_flat.len();
    let mismatches: Vec<usize> = (0..n).filter(|&i| rust_flat[i].to_bits() != py_vals[i].to_bits()).collect();
    println!("{label} ({py_name}): shape={rust_shape:?} n={n} bit-exact matches={}/{n}", n - mismatches.len());
    if let Some(&i) = mismatches.first() {
        panic!(
            "{label} ({py_name}): {}/{n} elements mismatch, first at index {i}: rust={} python={}",
            mismatches.len(), rust_flat[i], py_vals[i]
        );
    }
}

#[test]
fn ar_weights_matches_python_rail_on_real_tensors() {
    let Some(gguf) = python_gguf_path() else {
        eprintln!("skip: no python3/codec_paths/GGUF available (set $S2_GGUF to force)");
        return;
    };
    let w = npu_s2::ar::ArWeights::open(&gguf).expect("ArWeights::open");

    // Layer counts read off the checkpoint's own KV, cross-checked against s2_ar_ref.py's
    // read_ar_hparams() on this same GGUF (block_count=36, fast_block_count=4).
    assert_eq!(w.slow_layer_count(), Some(36));
    assert_eq!(w.fast_layer_count(), Some(4));

    // Slow layer 0: every field, including q_norm/k_norm (attention_qk_norm=true for the slow
    // stack on this checkpoint) and the f16 norms.
    let l0 = w.slow_layer(0).expect("slow_layer(0)");
    check(&gguf, "layers.0.attention_norm.weight", None, "slow[0].attention_norm", &[l0.attention_norm.len()], l0.attention_norm.as_slice().unwrap());
    check(&gguf, "layers.0.ffn_norm.weight", None, "slow[0].ffn_norm", &[l0.ffn_norm.len()], l0.ffn_norm.as_slice().unwrap());
    check(&gguf, "layers.0.attention.wqkv.weight", None, "slow[0].wqkv", l0.wqkv.shape(), l0.wqkv.as_slice().unwrap());
    check(&gguf, "layers.0.attention.wo.weight", None, "slow[0].wo", l0.wo.shape(), l0.wo.as_slice().unwrap());
    check(&gguf, "layers.0.feed_forward.w1.weight", None, "slow[0].w1", l0.w1.shape(), l0.w1.as_slice().unwrap());
    check(&gguf, "layers.0.feed_forward.w2.weight", None, "slow[0].w2", l0.w2.shape(), l0.w2.as_slice().unwrap());
    check(&gguf, "layers.0.feed_forward.w3.weight", None, "slow[0].w3", l0.w3.shape(), l0.w3.as_slice().unwrap());
    let q0 = l0.q_norm.as_ref().expect("slow layer 0 has q_norm (attention_qk_norm=true)");
    let k0 = l0.k_norm.as_ref().expect("slow layer 0 has k_norm (attention_qk_norm=true)");
    check(&gguf, "layers.0.attention.q_norm.weight", None, "slow[0].q_norm", &[q0.len()], q0.as_slice().unwrap());
    check(&gguf, "layers.0.attention.k_norm.weight", None, "slow[0].k_norm", &[k0.len()], k0.as_slice().unwrap());

    // Slow layer 35 (last of 36): spot-check one big q6_k weight at the far end of the stack.
    let l35 = w.slow_layer(35).expect("slow_layer(35)");
    check(&gguf, "layers.35.attention.wqkv.weight", None, "slow[35].wqkv", l35.wqkv.shape(), l35.wqkv.as_slice().unwrap());

    // Fast layer 0: attention_qk_norm=false for the fast stack -- q_norm/k_norm must be absent.
    let f0 = w.fast_layer(0).expect("fast_layer(0)");
    assert!(f0.q_norm.is_none(), "fast layer 0 must have no q_norm (fast_attention_qk_norm=false)");
    assert!(f0.k_norm.is_none(), "fast layer 0 must have no k_norm (fast_attention_qk_norm=false)");
    check(&gguf, "fast_layers.0.attention.wqkv.weight", None, "fast[0].wqkv", f0.wqkv.shape(), f0.wqkv.as_slice().unwrap());
    check(&gguf, "fast_layers.0.feed_forward.w2.weight", None, "fast[0].w2", f0.w2.shape(), f0.w2.as_slice().unwrap());

    // Fast layer 3 (last of 4).
    let f3 = w.fast_layer(3).expect("fast_layer(3)");
    check(&gguf, "fast_layers.3.attention.wo.weight", None, "fast[3].wo", f3.wo.shape(), f3.wo.as_slice().unwrap());

    // Top-level tensors: two more f16 norms, one q6_k projection.
    let norm = w.norm().expect("norm()");
    check(&gguf, "norm.weight", None, "norm", &[norm.len()], norm.as_slice().unwrap());
    let fast_norm = w.fast_norm().expect("fast_norm()");
    check(&gguf, "fast_norm.weight", None, "fast_norm", &[fast_norm.len()], fast_norm.as_slice().unwrap());
    let fast_output = w.fast_output().expect("fast_output()");
    check(&gguf, "fast_output.weight", None, "fast_output", fast_output.shape(), fast_output.as_slice().unwrap());

    // Row-range reads on both big embedding tables -- the mechanism that keeps a real AR step from
    // ever decoding the full 155776/40960-row tables.
    let emb_rows = w.embedding_rows(100, 108).expect("embedding_rows");
    check(&gguf, "embeddings.weight", Some((100, 108)), "embedding_rows[100:108]", emb_rows.shape(), emb_rows.as_slice().unwrap());
    let cb_rows = w.codebook_embedding_rows(0, 16).expect("codebook_embedding_rows");
    check(&gguf, "codebook_embeddings.weight", Some((0, 16)), "codebook_embedding_rows[0:16]", cb_rows.shape(), cb_rows.as_slice().unwrap());
}
