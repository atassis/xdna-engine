//! Load encoder weights + reference tensors from `artifacts/encoder/` (produced by
//! `scripts/extract_encoder.py`). fp32 `.npy` on disk; engines bf16-quantize at use.
//! Mirrors `npu_asr/weights.py`. We load every `.npy` in each block dir keyed by file stem
//! (no manifest parse needed).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use ndarray::prelude::*;
use ndarray_npy::read_npy;

/// Weights for one Conformer block, keyed by the dotted name (e.g. `self_attn.linear_q.weight`).
pub struct BlockWeights {
    map: HashMap<String, ArrayD<f32>>,
}

impl BlockWeights {
    pub fn get(&self, key: &str) -> &ArrayD<f32> {
        self.map
            .get(key)
            .unwrap_or_else(|| panic!("missing weight `{key}`"))
    }
    /// 1-D weight as a plain Vec<f32>.
    pub fn v(&self, key: &str) -> Vec<f32> {
        self.get(key).iter().copied().collect()
    }
    /// 2-D weight (clone).
    pub fn m(&self, key: &str) -> Array2<f32> {
        self.get(key)
            .clone()
            .into_dimensionality::<Ix2>()
            .unwrap_or_else(|_| panic!("weight `{key}` is not 2-D"))
    }
    /// 3-D weight (clone).
    pub fn m3(&self, key: &str) -> Array3<f32> {
        self.get(key)
            .clone()
            .into_dimensionality::<Ix3>()
            .unwrap_or_else(|_| panic!("weight `{key}` is not 3-D"))
    }
}

pub struct WeightStore {
    root: PathBuf,
    blocks: Vec<BlockWeights>,
    pub pre_encode: HashMap<String, ArrayD<f32>>,
    /// RoPE tables, squeezed to [T, head_dim].
    pub cos: Array2<f32>,
    pub sin: Array2<f32>,
}

fn load_dir(dir: &Path) -> std::io::Result<HashMap<String, ArrayD<f32>>> {
    let mut map = HashMap::new();
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.extension().and_then(|e| e.to_str()) == Some("npy") {
            let stem = path.file_stem().unwrap().to_string_lossy().into_owned();
            let arr: ArrayD<f32> =
                read_npy(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
            map.insert(stem, arr);
        }
    }
    Ok(map)
}

impl WeightStore {
    pub fn load(artifacts: &Path) -> std::io::Result<Self> {
        // count blocks by present L{i} dirs
        let mut nblocks = 0;
        while artifacts.join(format!("L{nblocks}")).is_dir() {
            nblocks += 1;
        }
        let blocks = (0..nblocks)
            .map(|i| {
                let map = load_dir(&artifacts.join(format!("L{i}"))).expect("load block dir");
                BlockWeights { map }
            })
            .collect();
        let pre_encode = load_dir(&artifacts.join("pre_encode"))?;

        // cos/sin: [T,1,1,HD] -> [T,HD]
        let cos4: ArrayD<f32> = read_npy(artifacts.join("refs/pos_cos.npy")).expect("pos_cos");
        let sin4: ArrayD<f32> = read_npy(artifacts.join("refs/pos_sin.npy")).expect("pos_sin");
        let (t, hd) = (cos4.shape()[0], cos4.shape()[3]);
        let cos = cos4.into_shape_with_order((t, hd)).unwrap();
        let sin = sin4.into_shape_with_order((t, hd)).unwrap();

        Ok(WeightStore {
            root: artifacts.to_path_buf(),
            blocks,
            pre_encode,
            cos,
            sin,
        })
    }

    pub fn nblocks(&self) -> usize {
        self.blocks.len()
    }

    pub fn block(&self, i: usize) -> &BlockWeights {
        &self.blocks[i]
    }

    pub fn pre(&self, key: &str) -> &ArrayD<f32> {
        self.pre_encode
            .get(key)
            .unwrap_or_else(|| panic!("missing pre_encode `{key}`"))
    }

    /// Load a reference tensor by name (e.g. "encoded", "out_L0", "audio_signal").
    pub fn ref_tensor(&self, name: &str) -> ArrayD<f32> {
        let p = self.root.join("refs").join(format!("{name}.npy"));
        read_npy(&p).unwrap_or_else(|e| panic!("read ref {}: {e}", p.display()))
    }
}
