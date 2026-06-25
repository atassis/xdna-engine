// rust/npu-weights/src/fingerprint.rs
use sha2::{Digest, Sha256};
use std::path::Path;

/// sha256 hex of one file's bytes.
pub fn file_sha256(path: &Path) -> anyhow::Result<String> {
    let mut f = std::fs::File::open(path)?;
    let mut h = Sha256::new();
    std::io::copy(&mut f, &mut h)?;
    Ok(hex(&h.finalize()))
}

/// Stable digest of a multi-file source: sha256 over sorted "relname:filehash\n" lines.
/// `files` = (display-name, path) pairs. Order-independent.
pub fn multi_sha256(files: &[(String, &Path)]) -> anyhow::Result<String> {
    let mut lines: Vec<String> = files.iter()
        .map(|(name, p)| Ok(format!("{name}:{}", file_sha256(p)?)))
        .collect::<anyhow::Result<_>>()?;
    lines.sort();
    let mut h = Sha256::new();
    for l in &lines {
        h.update(l.as_bytes());
        h.update(b"\n");
    }
    Ok(hex(&h.finalize()))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    #[test]
    fn single_file_stable_and_known() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(b"hello").unwrap();
        let h = file_sha256(f.path()).unwrap();
        assert_eq!(h, "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824");
    }
    #[test]
    fn multi_is_order_independent() {
        let mut a = tempfile::NamedTempFile::new().unwrap();
        a.write_all(b"A").unwrap();
        let mut b = tempfile::NamedTempFile::new().unwrap();
        b.write_all(b"B").unwrap();
        let h1 = multi_sha256(&[("a".into(), a.path()), ("b".into(), b.path())]).unwrap();
        let h2 = multi_sha256(&[("b".into(), b.path()), ("a".into(), a.path())]).unwrap();
        assert_eq!(h1, h2);
    }
}
