//! Whisper weight loader — fp32 `.npy` under artifacts/whisper-small/{conv, L0..L{N-1}, refs}.
//! Mirrors npu-parakeet's loader: per-block dirs keyed by file stem, with v/m/m3 accessors plus
//! a `conv` dir and a `ref_tensor` reader for the golden activations.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use ndarray::prelude::*;
use ndarray_npy::read_npy;

/// A keyed bag of fp32 tensors (one directory's worth of `.npy` files, keyed by file stem).
pub struct TensorMap {
    map: HashMap<String, ArrayD<f32>>,
}

impl TensorMap {
    pub fn get(&self, key: &str) -> &ArrayD<f32> {
        self.map.get(key).unwrap_or_else(|| panic!("missing weight `{key}`"))
    }
    pub fn v(&self, key: &str) -> Array1<f32> {
        self.get(key).clone().into_dimensionality::<Ix1>().unwrap_or_else(|_| panic!("`{key}` not 1-D"))
    }
    pub fn m(&self, key: &str) -> Array2<f32> {
        self.get(key).clone().into_dimensionality::<Ix2>().unwrap_or_else(|_| panic!("`{key}` not 2-D"))
    }
    pub fn m3(&self, key: &str) -> Array3<f32> {
        self.get(key).clone().into_dimensionality::<Ix3>().unwrap_or_else(|_| panic!("`{key}` not 3-D"))
    }
}

pub struct WhisperWeights {
    root: PathBuf,
    blocks: Vec<TensorMap>,
    conv: TensorMap,
}

fn load_dir(dir: &Path) -> std::io::Result<HashMap<String, ArrayD<f32>>> {
    let mut map = HashMap::new();
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.extension().and_then(|e| e.to_str()) == Some("npy") {
            let stem = path.file_stem().unwrap().to_string_lossy().into_owned();
            let arr: ArrayD<f32> = read_npy(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
            map.insert(stem, arr);
        }
    }
    Ok(map)
}

impl WhisperWeights {
    pub fn load(artifacts: &Path) -> std::io::Result<Self> {
        let mut nblocks = 0;
        while artifacts.join(format!("L{nblocks}")).is_dir() {
            nblocks += 1;
        }
        let blocks = (0..nblocks)
            .map(|i| TensorMap { map: load_dir(&artifacts.join(format!("L{i}"))).expect("load block dir") })
            .collect();
        let conv = TensorMap { map: load_dir(&artifacts.join("conv"))? };
        Ok(WhisperWeights { root: artifacts.to_path_buf(), blocks, conv })
    }

    pub fn nblocks(&self) -> usize {
        self.blocks.len()
    }
    pub fn block(&self, i: usize) -> &TensorMap {
        &self.blocks[i]
    }
    pub fn conv(&self) -> &TensorMap {
        &self.conv
    }
    /// Read a golden activation / post-LN param from `refs/{name}.npy` as a dynamic-rank array.
    pub fn ref_tensor(&self, name: &str) -> ArrayD<f32> {
        let p = self.root.join("refs").join(format!("{name}.npy"));
        read_npy(&p).unwrap_or_else(|e| panic!("read ref {}: {e}", p.display()))
    }
}
