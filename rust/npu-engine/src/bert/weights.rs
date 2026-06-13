//! BERT weight store: flat npy files by name (no RoPE tables, unlike npu_asr::WeightStore).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use ndarray::{Array1, Array2, ArrayD};
use ndarray_npy::read_npy;

pub struct BertLayer {
    map: HashMap<String, ArrayD<f32>>,
}
impl BertLayer {
    pub fn m(&self, k: &str) -> Array2<f32> {
        self.map[k].clone().into_dimensionality().unwrap_or_else(|_| panic!("bert weight {k} not 2-D"))
    }
    pub fn v(&self, k: &str) -> Vec<f32> {
        self.map[k].iter().copied().collect()
    }
}

pub struct BertWeights {
    pub emb: HashMap<String, ArrayD<f32>>,
    pub layers: Vec<BertLayer>,
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

impl BertWeights {
    pub fn load(weights: &Path, n_layers: usize) -> std::io::Result<Self> {
        let emb = load_dir(&weights.join("emb"))?;
        let layers = (0..n_layers)
            .map(|i| {
                let dir: PathBuf = weights.join(format!("L{i}"));
                BertLayer { map: load_dir(&dir).expect("load bert layer dir") }
            })
            .collect();
        Ok(BertWeights { emb, layers })
    }

    /// Embedding matrices.
    pub fn word_emb(&self) -> Array2<f32> { self.emb["word_emb"].clone().into_dimensionality().unwrap() }
    pub fn pos_emb(&self) -> Array2<f32> { self.emb["pos_emb"].clone().into_dimensionality().unwrap() }
    pub fn type_emb(&self) -> Array2<f32> { self.emb["type_emb"].clone().into_dimensionality().unwrap() }
    pub fn emb_ln(&self) -> (Vec<f32>, Vec<f32>) {
        (self.emb["emb_ln_w"].iter().copied().collect(), self.emb["emb_ln_b"].iter().copied().collect())
    }
    pub fn n_layers(&self) -> usize { self.layers.len() }
    #[allow(dead_code)]
    fn _unused(&self) -> Array1<f32> { Array1::zeros(0) }
}
