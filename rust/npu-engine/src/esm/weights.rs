//! ESM weight store: per-layer npy by name + word-emb + final LN.
use std::collections::HashMap;
use std::path::Path;
use ndarray::{Array2, ArrayD};
use ndarray_npy::read_npy;

pub struct EsmLayer {
    map: HashMap<String, ArrayD<f32>>,
}
impl EsmLayer {
    pub fn m(&self, k: &str) -> Array2<f32> {
        self.map[k].clone().into_dimensionality().unwrap_or_else(|_| panic!("esm weight {k} not 2-D"))
    }
    pub fn v(&self, k: &str) -> Vec<f32> {
        self.map[k].iter().copied().collect()
    }
}

pub struct EsmWeights {
    pub emb: HashMap<String, ArrayD<f32>>,
    pub layers: Vec<EsmLayer>,
    pub final_ln_w: Vec<f32>,
    pub final_ln_b: Vec<f32>,
}

fn load_dir(dir: &Path) -> std::io::Result<HashMap<String, ArrayD<f32>>> {
    let mut map = HashMap::new();
    for e in std::fs::read_dir(dir)? {
        let p = e?.path();
        if p.extension().and_then(|x| x.to_str()) == Some("npy") {
            let stem = p.file_stem().unwrap().to_string_lossy().into_owned();
            map.insert(stem, read_npy(&p).unwrap_or_else(|e| panic!("read {}: {e}", p.display())));
        }
    }
    Ok(map)
}

impl EsmWeights {
    pub fn load(weights: &Path, n_layers: usize) -> std::io::Result<Self> {
        let emb = load_dir(&weights.join("emb"))?;
        let layers = (0..n_layers)
            .map(|i| EsmLayer { map: load_dir(&weights.join(format!("L{i}"))).expect("esm layer dir") })
            .collect();
        let f = load_dir(weights)?; // final_ln_{w,b}.npy live at the encoder root
        let v = |k: &str| f[k].iter().copied().collect::<Vec<f32>>();
        Ok(EsmWeights { emb, layers, final_ln_w: v("final_ln_w"), final_ln_b: v("final_ln_b") })
    }
    pub fn word_emb(&self) -> Array2<f32> {
        self.emb["word_emb"].clone().into_dimensionality().unwrap()
    }
    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }
}
