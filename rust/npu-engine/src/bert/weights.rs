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
    /// Uniform declarative entry point (host-only, no NPU): resolve weights for a scenario.
    /// 1. `artifacts.source` set -> ensure (bake-on-missing) the `npu-weights` arena via the
    ///    declarative `ModelSpec` and load from it ("add a model" = config + arch fn).
    /// 2. else the legacy npy `artifacts.weights` dir (default; byte-identical to before).
    pub fn load_for(
        artifacts: &crate::config::Artifacts,
        root: &Path,
        n_layers: usize,
    ) -> anyhow::Result<Self> {
        if let Some(spec) = artifacts.model_spec()? {
            let arena = spec.ensure_arena(root, false)?;
            return Self::load_arena(&arena, &spec.arch, n_layers);
        }
        let wpath = root.join(&artifacts.weights);
        Self::load(&wpath, n_layers)
            .map_err(|e| anyhow::anyhow!("bert weights load {}: {e}", wpath.display()))
    }

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

    /// Load BertWeights from a baked safetensors arena (mmap, bf16 upcast to f32). Tensor names use
    /// the same keying as the npy path: `emb/<k>` -> emb map, `L{i}/<k>` -> layers[i] map. Produces
    /// the same in-memory BertWeights as the npy loader for a matching export.
    pub fn load_arena(arena_path: &Path, arch: &str, n_layers: usize) -> anyhow::Result<Self> {
        let loaded = npu_weights::arena::load(arena_path, arch)?;
        let mut emb: HashMap<String, ArrayD<f32>> = HashMap::new();
        let mut layer_maps: Vec<HashMap<String, ArrayD<f32>>> =
            (0..n_layers).map(|_| HashMap::new()).collect();
        for name in &loaded.names {
            let (shape, data) = loaded.tensor_f32(name)?;
            let arr = ArrayD::<f32>::from_shape_vec(shape, data)
                .map_err(|e| anyhow::anyhow!("arena tensor {name}: bad shape: {e}"))?;
            if let Some(k) = name.strip_prefix("emb/") {
                emb.insert(k.to_string(), arr);
            } else if let Some(rest) = name.strip_prefix('L') {
                let (idx_s, k) = rest
                    .split_once('/')
                    .ok_or_else(|| anyhow::anyhow!("arena tensor {name}: missing layer key"))?;
                let i: usize = idx_s
                    .parse()
                    .map_err(|_| anyhow::anyhow!("arena tensor {name}: bad layer index"))?;
                if i < n_layers {
                    layer_maps[i].insert(k.to_string(), arr);
                }
            }
            // any other name is ignored
        }
        let layers = layer_maps.into_iter().map(|map| BertLayer { map }).collect();
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
