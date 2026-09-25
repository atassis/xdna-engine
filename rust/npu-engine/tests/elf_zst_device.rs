//! Device check for the `.elf.zst` loader path: a decode dir with ONLY `decode.elf.zst` (no plain
//! `decode.elf`) must generate token-for-token identical output, at the same `artifact_hash`, as
//! the real served dir it was compressed from.
//!
//! NPU is single-tenant. Gate manually before running:
//!   fuser -v /dev/accel/accel0   # must be empty
//!   curl -s 127.0.0.1:11434/v1/models  # every model state must be "unloaded"
//!
//!   ELF_ZST_DEVICE=1 cargo test -p npu-engine --test elf_zst_device -- --nocapture --test-threads=1
use std::path::Path;
use std::rc::Rc;

use npu_engine::llm::generator::DecodeStep;
use npu_engine::llm::npu_decode::NpuDecodeStep;
use npu_xrt::Device;

#[test]
fn zst_only_decode_dir_matches_plain_elf_dir() {
    if std::env::var("ELF_ZST_DEVICE").as_deref() != Ok("1") {
        eprintln!("skip: set ELF_ZST_DEVICE=1 (opens the NPU; single-tenant)");
        return;
    }
    let served = Path::new("/mnt/data/xdna/artifacts/gemma3-270m/decode_p8c684");
    let zst_only = Path::new(env!("CARGO_TARGET_TMPDIR")).join("gemma3-270m-zst-only");

    // Build the zst-only copy fresh every run: everything but decode.elf (which becomes
    // decode.elf.zst) is hardlinked, so this costs one compression pass, not a weight copy.
    let _ = std::fs::remove_dir_all(&zst_only);
    copy_tree_hardlinked(served, &zst_only, "decode.elf");
    let status = std::process::Command::new("zstd")
        .args(["-q", "-f", "-3", "--long=27", "-o"])
        .arg(zst_only.join("decode.elf.zst"))
        .arg(served.join("decode.elf"))
        .status()
        .expect("run zstd");
    assert!(status.success(), "zstd compress failed");
    assert!(!zst_only.join("decode.elf").exists(), "zst-only dir must NOT carry the plain ELF");

    let dev = Rc::new(Device::open(0).expect("open NPU (single-tenant -- check fuser first)"));
    let plain = NpuDecodeStep::new(&dev, served).expect("load plain-ELF dir");
    let plain_hash = plain.provenance().artifact_hash.clone();
    drop(plain);
    // Re-open device-side state cleanly for the second artifact rather than reuse `dev`'s prior
    // hardware context -- matches how two separate loads behave in the served path.
    let dev2 = Rc::new(Device::open(0).expect("re-open NPU for the zst-only dir"));
    let mut zst = NpuDecodeStep::new(&dev2, &zst_only).expect("load zst-only dir");
    let zst_hash = zst.provenance().artifact_hash.clone();

    assert_eq!(plain_hash, zst_hash, "artifact_hash must be the hash of the UNCOMPRESSED ELF");
    assert!(plain_hash.is_some());

    // Re-open the plain dir again (its NpuDecodeStep was dropped above) so both arms run the
    // SAME token loop for a real generation comparison, not just a load-time hash check.
    let dev3 = Rc::new(Device::open(0).expect("re-open NPU for the plain dir generation"));
    let mut plain = NpuDecodeStep::new(&dev3, served).expect("reload plain-ELF dir");
    let prompt: Vec<u32> = vec![10, 200, 3000, 40000, 5];
    let vocab = 262_144u32;

    let toks_plain = run_greedy(&mut plain, &prompt, vocab, 8);
    let toks_zst = run_greedy(&mut zst, &prompt, vocab, 8);
    eprintln!("plain tokens: {toks_plain:?}");
    eprintln!("zst   tokens: {toks_zst:?}");
    assert_eq!(toks_plain, toks_zst, "greedy decode must be token-for-token identical");
}

fn run_greedy(step: &mut NpuDecodeStep, prompt: &[u32], vocab: u32, n_new: usize) -> Vec<u32> {
    step.reset().expect("reset KV");
    let mut pos = 0usize;
    let mut last = prompt[0];
    for (i, &tok) in prompt.iter().enumerate() {
        let logits = step.step(tok, i).expect("prompt step");
        pos = i + 1;
        last = argmax(&logits, vocab);
    }
    let mut out = vec![last];
    for _ in 1..n_new {
        let logits = step.step(last, pos).expect("decode step");
        pos += 1;
        last = argmax(&logits, vocab);
        out.push(last);
    }
    out
}

fn argmax(logits: &[f32], vocab: u32) -> u32 {
    let n = (vocab as usize).min(logits.len());
    (0..n)
        .max_by(|&a, &b| logits[a].partial_cmp(&logits[b]).unwrap())
        .expect("non-empty logits") as u32
}

/// Hardlink every file under `src` into `dst` except `skip_name`, which the caller writes itself
/// (here, replaced by a compressed sibling). Hardlinks share the weight/blob bytes rather than
/// copying gigabytes for a load-path test.
fn copy_tree_hardlinked(src: &Path, dst: &Path, skip_name: &str) {
    std::fs::create_dir_all(dst).expect("mkdir zst-only dir");
    for entry in walkdir(src) {
        let rel = entry.strip_prefix(src).unwrap();
        let out = dst.join(rel);
        if entry.is_dir() {
            std::fs::create_dir_all(&out).expect("mkdir");
            continue;
        }
        if entry.file_name().and_then(|n| n.to_str()) == Some(skip_name) {
            continue;
        }
        std::fs::create_dir_all(out.parent().unwrap()).ok();
        std::fs::hard_link(&entry, &out).expect("hardlink");
    }
}

fn walkdir(root: &Path) -> Vec<std::path::PathBuf> {
    let mut out = vec![];
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for e in std::fs::read_dir(&dir).expect("read_dir") {
            let e = e.expect("dirent");
            let p = e.path();
            if p.is_dir() {
                stack.push(p.clone());
            }
            out.push(p);
        }
    }
    out
}
