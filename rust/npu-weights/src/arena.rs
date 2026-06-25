// rust/npu-weights/src/arena.rs
use crate::arch::OutTensor;
use half::bf16;
use memmap2::Mmap;
use safetensors::tensor::{Dtype, TensorView};
use std::collections::BTreeMap;
use std::path::Path;

pub struct ArenaMeta { pub format_version: u32, pub arch: String,
                       pub source_ref: String, pub source_fingerprint: String }

/// Bake OutTensors -> a safetensors file (bf16/f32 per OutTensor.bf16), metadata embedded,
/// written atomically (unique tmp + rename).
pub fn bake(path: &Path, tensors: &BTreeMap<String, OutTensor>, meta: &ArenaMeta) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() { std::fs::create_dir_all(parent)?; }
    // Build owned byte buffers (bf16 or f32 LE) so TensorViews can borrow them.
    let mut bufs: Vec<(String, Vec<usize>, Dtype, Vec<u8>)> = Vec::new();
    for (name, t) in tensors {
        let (dtype, bytes) = if t.bf16 {
            (Dtype::BF16, t.data.iter().flat_map(|x| bf16::from_f32(*x).to_le_bytes()).collect::<Vec<u8>>())
        } else {
            (Dtype::F32, t.data.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())
        };
        bufs.push((name.clone(), t.shape.clone(), dtype, bytes));
    }
    let views: Vec<(String, TensorView)> = bufs.iter()
        .map(|(n, sh, dt, by)| Ok((n.clone(), TensorView::new(*dt, sh.clone(), by)?)))
        .collect::<anyhow::Result<_>>()?;
    let mut md = std::collections::HashMap::new();
    md.insert("format_version".to_string(), meta.format_version.to_string());
    md.insert("arch".to_string(), meta.arch.clone());
    md.insert("source_ref".to_string(), meta.source_ref.clone());
    md.insert("source_fingerprint".to_string(), meta.source_fingerprint.clone());
    let out = safetensors::serialize(views.iter().map(|(n, v)| (n.as_str(), v)), &Some(md))?;
    // atomic: unique tmp + rename
    let pid = std::process::id();
    let tmp = path.with_extension(format!("tmp.{pid}"));
    std::fs::write(&tmp, &out)?;
    { let f = std::fs::File::open(&tmp)?; f.sync_all()?; }
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// A zero-copy view into a baked tensor.
pub struct Loaded { _mmap: memmap2::Mmap, pub meta_version: u32, pub arch: String,
                    pub names: Vec<String> }

/// mmap-load an arena; verifies magic/header via safetensors and format_version.
pub fn load(path: &Path, expect_arch: &str) -> anyhow::Result<Loaded> {
    let file = std::fs::File::open(path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let (_n, md) = safetensors::SafeTensors::read_metadata(&mmap)?;
    let meta = md.metadata().clone().unwrap_or_default();
    let arch = meta.get("arch").cloned().unwrap_or_default();
    anyhow::ensure!(arch == expect_arch, "arena arch {arch:?} != expected {expect_arch:?}");
    let version: u32 = meta.get("format_version").and_then(|s| s.parse().ok()).unwrap_or(0);
    anyhow::ensure!(version == crate::FORMAT_VERSION,
        "arena format_version {version} != {} (rebake)", crate::FORMAT_VERSION);
    let st = safetensors::SafeTensors::deserialize(&mmap)?;
    let names = st.names().into_iter().map(|s| s.to_string()).collect();
    Ok(Loaded { _mmap: mmap, meta_version: version, arch, names })
}

impl Loaded {
    /// f32 values of a tensor (bf16 upcast on read; for parity checks/host use).
    pub fn tensor_f32(&self, name: &str) -> anyhow::Result<(Vec<usize>, Vec<f32>)> {
        let st = safetensors::SafeTensors::deserialize(&self._mmap)?;
        let v = st.tensor(name)?;
        let shape = v.shape().to_vec();
        let raw = v.data();
        let data: Vec<f32> = match v.dtype() {
            Dtype::F32 => raw.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect(),
            Dtype::BF16 => raw.chunks_exact(2).map(|b| bf16::from_le_bytes(b.try_into().unwrap()).to_f32()).collect(),
            d => anyhow::bail!("unexpected arena dtype {d:?}"),
        };
        Ok((shape, data))
    }

    /// Raw bf16 bits of a tensor, in the arena's stored byte order (no f32 upcast). Returns the
    /// shape plus a `Vec<u16>` of the bf16 bit patterns, ready to write straight to a device BO that
    /// expects bf16 weights -- this is the fast-restart path that skips re-packing f32->bf16 every
    /// startup. Only valid for tensors baked as BF16; an F32 tensor returns `Ok(None)` so the caller
    /// can fall back to the f32 path. (Tensors that were not baked bf16 cannot serve this path
    /// losslessly without a pack, which is exactly the work we are trying to avoid.)
    pub fn tensor_bf16(&self, name: &str) -> anyhow::Result<Option<(Vec<usize>, Vec<u16>)>> {
        let st = safetensors::SafeTensors::deserialize(&self._mmap)?;
        let v = st.tensor(name)?;
        match v.dtype() {
            Dtype::BF16 => {
                let shape = v.shape().to_vec();
                let raw = v.data();
                let bits: Vec<u16> =
                    raw.chunks_exact(2).map(|b| u16::from_le_bytes(b.try_into().unwrap())).collect();
                Ok(Some((shape, bits)))
            }
            // F32 (or anything else) has no bf16 representation to hand off cheaply.
            _ => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bake_then_load_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("a.safetensors");
        let mut t = BTreeMap::new();
        t.insert("L0/q_w".to_string(), OutTensor { shape: vec![2,2], data: vec![1.,2.,3.,4.], bf16: true });
        t.insert("L0/q_b".to_string(), OutTensor { shape: vec![2], data: vec![0.5,-0.5], bf16: false });
        let meta = ArenaMeta { format_version: crate::FORMAT_VERSION, arch: "bert".into(),
                               source_ref: "test".into(), source_fingerprint: "deadbeef".into() };
        bake(&p, &t, &meta).unwrap();
        let l = load(&p, "bert").unwrap();
        assert_eq!(l.meta_version, crate::FORMAT_VERSION);
        let (sh, v) = l.tensor_f32("L0/q_b").unwrap();
        assert_eq!(sh, vec![2]); assert_eq!(v, vec![0.5,-0.5]);   // f32 exact
        let (_sh, w) = l.tensor_f32("L0/q_w").unwrap();
        // bf16 round-trip: 1,2,3,4 are exactly representable
        assert_eq!(w, vec![1.,2.,3.,4.]);
    }
    #[test]
    fn tensor_bf16_returns_raw_bits_for_bf16_and_none_for_f32() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("a.safetensors");
        let mut t = BTreeMap::new();
        // bf16 weight (exactly representable values) + an f32 bias.
        t.insert("L0/q_w".to_string(), OutTensor { shape: vec![2, 2], data: vec![1., 2., 3., 4.], bf16: true });
        t.insert("L0/q_b".to_string(), OutTensor { shape: vec![2], data: vec![0.5, -0.5], bf16: false });
        let meta = ArenaMeta { format_version: crate::FORMAT_VERSION, arch: "bert".into(),
                               source_ref: "test".into(), source_fingerprint: "deadbeef".into() };
        bake(&p, &t, &meta).unwrap();
        let l = load(&p, "bert").unwrap();
        // bf16 tensor: raw bits handed back, no upcast; match what from_f32 produced at bake.
        let (sh, bits) = l.tensor_bf16("L0/q_w").unwrap().expect("bf16 tensor must yield raw bits");
        assert_eq!(sh, vec![2, 2]);
        let want: Vec<u16> = [1.0f32, 2., 3., 4.].iter().map(|x| bf16::from_f32(*x).to_bits()).collect();
        assert_eq!(bits, want);
        // f32 tensor: no cheap bf16 handoff -> None (caller falls back to f32 path).
        assert!(l.tensor_bf16("L0/q_b").unwrap().is_none());
    }

    #[test]
    fn load_rejects_wrong_arch() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("a.safetensors");
        let mut t = BTreeMap::new();
        t.insert("x".to_string(), OutTensor { shape: vec![1], data: vec![1.], bf16: false });
        let meta = ArenaMeta { format_version: crate::FORMAT_VERSION, arch: "bert".into(),
                               source_ref: "t".into(), source_fingerprint: "d".into() };
        bake(&p, &t, &meta).unwrap();
        assert!(load(&p, "whisper").is_err());
    }
}
