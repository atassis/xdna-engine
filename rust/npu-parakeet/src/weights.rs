//! Parakeet weight loader — own loader per the general-engine contract (GigaAM's WeightStore
//! is RoPE-specific). Two sources, selected at load time:
//!   1. (default) fp32 `.npy` under artifacts/parakeet/encoder/{L0..L{N-1}, pre_encode, refs}.
//!   2. (opt-in, env `NPU_WEIGHTS_ARENA=<path to .safetensors>`) the bf16-baked `npu-weights`
//!      arena (`arch=fastconformer`), mmapped via `npu_weights::arena`. Matmul-weight tensors are
//!      kept as raw bf16 bits ONLY (never upcast to a permanently-resident f32 host array) — this
//!      is the ~2.5x host-RSS win over the npy path (2.3 GB f32 vs ~1.16 GB bf16). LayerNorm
//!      weight/bias and other small f32-native tensors are upcast once and kept (negligible size).
//! The npy path is byte-for-byte unchanged when `NPU_WEIGHTS_ARENA` is unset.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use ndarray::prelude::*;
use ndarray_npy::read_npy;

/// One stored tensor: either a plain f32 array (npy path, or an arena tensor the arch baked
/// f32 — norms/bias/pos_bias), or raw bf16 bits (arena path, matmul weights). The bf16 variant
/// is never eagerly upcast to a resident f32 array; `to_f32()` builds a TRANSIENT f32 copy only
/// when a caller actually needs one (host math, or the one-time NPU weight-BO pack on a cache
/// miss), and that copy is dropped immediately after use — it is never stored back into `self`.
enum WVal {
    F32(ArrayD<f32>),
    Bf16 { shape: Vec<usize>, bits: Vec<u16> },
}

impl WVal {
    fn to_f32(&self) -> ArrayD<f32> {
        match self {
            WVal::F32(a) => a.clone(),
            WVal::Bf16 { shape, bits } => {
                // bf16 -> f32 is an exact widen (zero-extend the mantissa/exponent into the low
                // 16 bits) — no rounding, so this recovers precisely what the bake wrote.
                let data: Vec<f32> = bits.iter().map(|&b| f32::from_bits((b as u32) << 16)).collect();
                ArrayD::from_shape_vec(shape.clone(), data)
                    .unwrap_or_else(|e| panic!("arena tensor bad shape: {e}"))
            }
        }
    }
    /// Raw bf16 bits + shape, iff this value is arena-bf16-native (matmul weight). `None` for an
    /// f32-native value (npy path always; arena norms/bias/pos_bias) — caller falls back to
    /// `to_f32()` + the existing host f32->bf16 pack, unchanged.
    fn bf16_bits(&self) -> Option<(&[usize], &[u16])> {
        match self {
            WVal::Bf16 { shape, bits } => Some((shape, bits)),
            WVal::F32(_) => None,
        }
    }
}

pub struct BlockWeights {
    map: HashMap<String, WVal>,
}

impl BlockWeights {
    fn get(&self, key: &str) -> &WVal {
        self.map.get(key).unwrap_or_else(|| panic!("missing weight `{key}`"))
    }
    pub fn v(&self, key: &str) -> Array1<f32> {
        self.get(key).to_f32().into_dimensionality::<Ix1>().unwrap_or_else(|_| panic!("`{key}` not 1-D"))
    }
    pub fn m(&self, key: &str) -> Array2<f32> {
        self.get(key).to_f32().into_dimensionality::<Ix2>().unwrap_or_else(|_| panic!("`{key}` not 2-D"))
    }
    pub fn m3(&self, key: &str) -> Array3<f32> {
        self.get(key).to_f32().into_dimensionality::<Ix3>().unwrap_or_else(|_| panic!("`{key}` not 3-D"))
    }
    /// Raw bf16 bits for a 2-D matmul weight `[k, n]` row-major, iff `key` is arena-bf16-native.
    /// `None` on the npy path (always) or for a non-bf16 / non-2D arena tensor — the caller then
    /// takes the existing `.m(key)` + host f32->bf16 pack path, unchanged. This is the accessor
    /// the NPU weight-BO register fast path uses to skip the pack entirely on a cache miss.
    pub fn bf16_m(&self, key: &str) -> Option<(usize, usize, &[u16])> {
        let (shape, bits) = self.get(key).bf16_bits()?;
        if shape.len() != 2 {
            return None;
        }
        Some((shape[0], shape[1], bits))
    }
}

pub struct ParakeetWeights {
    root: PathBuf,
    blocks: Vec<BlockWeights>,
    pre_encode: HashMap<String, WVal>,
}

fn load_dir(dir: &Path) -> std::io::Result<HashMap<String, WVal>> {
    let mut map = HashMap::new();
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.extension().and_then(|e| e.to_str()) == Some("npy") {
            let stem = path.file_stem().unwrap().to_string_lossy().into_owned();
            let arr: ArrayD<f32> = read_npy(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
            map.insert(stem, WVal::F32(arr));
        }
    }
    Ok(map)
}

impl ParakeetWeights {
    /// Load weights: npy dir at `artifacts` by default; when env `NPU_WEIGHTS_ARENA` is set (path
    /// to a baked `arch=fastconformer` `.safetensors` arena), load from the arena instead and
    /// ignore `artifacts` entirely. The npy path below is byte-for-byte unchanged when the env var
    /// is unset — this is an additive, opt-in branch, never a default-path edit.
    pub fn load(artifacts: &Path) -> std::io::Result<Self> {
        if let Some(arena) = std::env::var_os("NPU_WEIGHTS_ARENA") {
            return Self::load_arena(Path::new(&arena))
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()));
        }
        let mut nblocks = 0;
        while artifacts.join(format!("L{nblocks}")).is_dir() {
            nblocks += 1;
        }
        let blocks = (0..nblocks)
            .map(|i| BlockWeights { map: load_dir(&artifacts.join(format!("L{i}"))).expect("load block dir") })
            .collect();
        let pre_encode = load_dir(&artifacts.join("pre_encode"))?;
        Ok(ParakeetWeights { root: artifacts.to_path_buf(), blocks, pre_encode })
    }

    /// Load from a baked `npu-weights` arena (`arch=fastconformer`). Tensor names in the arena
    /// mirror the npy tree 1:1 (`L{i}/<key>`, `pre_encode/<key>` — see
    /// `npu-weights/src/arch/fastconformer.rs`), so this maps straight onto the same
    /// `BlockWeights`/`pre_encode` shape the npy loader builds. Matmul weights (baked bf16) are
    /// kept ONLY as raw bf16 bits (`WVal::Bf16`) — never upcast to a resident f32 array — which is
    /// the host-RSS win over the npy path. Block count is inferred from the tensor names (max
    /// `L{i}` index + 1), mirroring the npy loader's directory-count probe.
    pub fn load_arena(arena_path: &Path) -> anyhow::Result<Self> {
        let loaded = npu_weights::arena::load(arena_path, "fastconformer")?;
        let mut nblocks = 0usize;
        for name in &loaded.names {
            if let Some(rest) = name.strip_prefix('L') {
                if let Some((idx_s, _)) = rest.split_once('/') {
                    if let Ok(i) = idx_s.parse::<usize>() {
                        nblocks = nblocks.max(i + 1);
                    }
                }
            }
        }
        let mut block_maps: Vec<HashMap<String, WVal>> = (0..nblocks).map(|_| HashMap::new()).collect();
        let mut pre_encode: HashMap<String, WVal> = HashMap::new();
        for name in &loaded.names {
            let wval = match loaded.tensor_bf16(name)? {
                Some((shape, bits)) => WVal::Bf16 { shape, bits },
                None => {
                    let (shape, data) = loaded.tensor_f32(name)?;
                    WVal::F32(
                        ArrayD::from_shape_vec(shape, data)
                            .map_err(|e| anyhow::anyhow!("arena tensor {name}: bad shape: {e}"))?,
                    )
                }
            };
            if let Some(rest) = name.strip_prefix('L') {
                let (idx_s, key) = rest
                    .split_once('/')
                    .ok_or_else(|| anyhow::anyhow!("arena tensor {name}: missing layer key"))?;
                let i: usize = idx_s
                    .parse()
                    .map_err(|_| anyhow::anyhow!("arena tensor {name}: bad layer index"))?;
                if i < nblocks {
                    block_maps[i].insert(key.to_string(), wval);
                }
            } else if let Some(key) = name.strip_prefix("pre_encode/") {
                pre_encode.insert(key.to_string(), wval);
            }
            // any other name is ignored (none expected for fastconformer)
        }
        let blocks = block_maps.into_iter().map(|map| BlockWeights { map }).collect();
        Ok(ParakeetWeights { root: arena_path.to_path_buf(), blocks, pre_encode })
    }

    pub fn nblocks(&self) -> usize {
        self.blocks.len()
    }
    pub fn block(&self, i: usize) -> &BlockWeights {
        &self.blocks[i]
    }
    pub fn pre(&self, key: &str) -> ArrayD<f32> {
        self.pre_encode.get(key).unwrap_or_else(|| panic!("missing pre_encode `{key}`")).to_f32()
    }
    /// Reference activations for parity checks — npy-path only (reads `<root>/refs/<name>.npy`).
    /// Not meaningful with an arena `root` (no `refs/` dir is baked into the arena); callers that
    /// need refs under `NPU_WEIGHTS_ARENA` should still point `artifacts` at the npy tree for that
    /// harness, or keep using the npy path for verification.
    pub fn ref_tensor(&self, name: &str) -> ArrayD<f32> {
        let p = self.root.join("refs").join(format!("{name}.npy"));
        read_npy(&p).unwrap_or_else(|e| panic!("read ref {}: {e}", p.display()))
    }
}
