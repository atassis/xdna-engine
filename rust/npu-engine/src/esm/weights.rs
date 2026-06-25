//! ESM weight store: per-layer npy by name + word-emb + final LN.
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use ndarray::{Array2, ArrayD};
use ndarray_npy::read_npy;

pub struct EsmLayer {
    map: HashMap<String, ArrayD<f32>>,
    /// Optional parallel bf16 representation, populated ONLY when this layer was loaded from a
    /// bf16-baked `NPU_WEIGHTS_ARENA` (key -> (shape, bf16 bits)). Lets the device-side consumer
    /// hand pre-packed bf16 straight to the BO without re-packing f32->bf16 every startup. Empty
    /// (so every accessor returns `None`) on the default npy path, keeping that path byte-identical.
    bf16: HashMap<String, (Vec<usize>, Vec<u16>)>,
}
impl EsmLayer {
    pub fn m(&self, k: &str) -> Array2<f32> {
        self.map[k].clone().into_dimensionality().unwrap_or_else(|_| panic!("esm weight {k} not 2-D"))
    }
    pub fn v(&self, k: &str) -> Vec<f32> {
        self.map[k].iter().copied().collect()
    }
    /// Tensor keys present in this layer (for parity checks / introspection).
    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.map.keys()
    }
    /// Pre-packed bf16 bits for tensor `k` as (shape, bits), iff this layer was loaded from a
    /// bf16-baked arena AND `k` was stored bf16. `None` on the npy path or for an f32 tensor -- the
    /// caller then takes the existing f32 pack path unchanged.
    pub fn bf16_bits(&self, k: &str) -> Option<&(Vec<usize>, Vec<u16>)> {
        self.bf16.get(k)
    }
}

pub struct EsmWeights {
    pub emb: HashMap<String, ArrayD<f32>>,
    pub layers: Vec<EsmLayer>,
    /// Final LayerNorm weight/bias, present only for models that apply a final LN (e.g. esm2).
    /// `None` for no-final-LN models (e.g. bge).
    pub final_ln: Option<(Vec<f32>, Vec<f32>)>,
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
    /// Uniform declarative entry point: load weights for a scenario from the best available source.
    ///
    /// Resolution order (host-only, no NPU):
    /// 1. `artifacts.source` set -> ensure (bake-on-missing) the `npu-weights` arena via the
    ///    declarative `ModelSpec` and load from it. This is how "add a model" becomes config + an
    ///    arch fn rather than a code fork.
    /// 2. else `NPU_WEIGHTS_ARENA` env points at a baked arena -> load it (legacy staged path).
    /// 3. else the legacy npy `artifacts.weights` dir (default; byte-identical to before).
    ///
    pub fn load_for(
        artifacts: &crate::config::Artifacts,
        root: &Path,
        n_layers: usize,
    ) -> anyhow::Result<Self> {
        if let Some(spec) = artifacts.model_spec()? {
            let arena = spec.ensure_arena(root, false)?;
            return Self::load_arena(&arena, &spec.arch, n_layers);
        }
        // No declarative source: legacy path. `load()` still honors NPU_WEIGHTS_ARENA internally
        // (env-arena path, arch="bert"), else reads the npy `weights` dir -- both unchanged.
        let wpath = root.join(&artifacts.weights);
        Self::load(&wpath, n_layers)
            .map_err(|e| anyhow::anyhow!("esm weights load {}: {e}", wpath.display()))
    }

    pub fn load(weights: &Path, n_layers: usize) -> std::io::Result<Self> {
        // Staged arena loader (default OFF): if NPU_WEIGHTS_ARENA points at a baked .safetensors
        // arena, load from it instead of the npy dirs. When the var is unset the npy path below is
        // taken UNCHANGED, so default behavior is byte-identical.
        if let Some(arena) = std::env::var_os("NPU_WEIGHTS_ARENA") {
            return Self::load_arena(&PathBuf::from(arena), "bert", n_layers)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()));
        }
        let emb = load_dir(&weights.join("emb"))?;
        let layers = (0..n_layers)
            .map(|i| EsmLayer {
                map: load_dir(&weights.join(format!("L{i}"))).expect("esm layer dir"),
                bf16: HashMap::new(), // npy path carries no pre-packed bf16
            })
            .collect();
        // final_ln_{w,b}.npy live at the encoder root, but only for models with a final LN.
        let final_ln = if weights.join("final_ln_w.npy").exists() && weights.join("final_ln_b.npy").exists() {
            let f = load_dir(weights)?;
            let v = |k: &str| f[k].iter().copied().collect::<Vec<f32>>();
            Some((v("final_ln_w"), v("final_ln_b")))
        } else {
            None
        };
        Ok(EsmWeights { emb, layers, final_ln })
    }

    /// Load EsmWeights from a baked safetensors arena (mmap, bf16 upcast to f32). Tensor names use
    /// the same keying as the npy path: `emb/<k>` -> emb map, `L{i}/<k>` -> layers[i] map, and
    /// `final_ln_w`/`final_ln_b` at the root -> Option. Produces the same in-memory EsmWeights as
    /// the npy loader for a matching export.
    pub fn load_arena(arena_path: &Path, arch: &str, n_layers: usize) -> anyhow::Result<Self> {
        let loaded = npu_weights::arena::load(arena_path, arch)?;
        let mut emb: HashMap<String, ArrayD<f32>> = HashMap::new();
        let mut layer_maps: Vec<HashMap<String, ArrayD<f32>>> =
            (0..n_layers).map(|_| HashMap::new()).collect();
        // Parallel bf16 bits per layer, kept for any tensor the arena baked as BF16. Used by the
        // device consumer to skip the per-start f32->bf16 weight pack; host-reference paths keep
        // reading the f32 `layer_maps` above.
        let mut layer_bf16: Vec<HashMap<String, (Vec<usize>, Vec<u16>)>> =
            (0..n_layers).map(|_| HashMap::new()).collect();
        let mut final_ln_w: Option<Vec<f32>> = None;
        let mut final_ln_b: Option<Vec<f32>> = None;

        for name in &loaded.names {
            let (shape, data) = loaded.tensor_f32(name)?;
            let arr = ArrayD::<f32>::from_shape_vec(shape, data)
                .map_err(|e| anyhow::anyhow!("arena tensor {name}: bad shape: {e}"))?;
            if let Some(k) = name.strip_prefix("emb/") {
                emb.insert(k.to_string(), arr);
            } else if name == "final_ln_w" {
                final_ln_w = Some(arr.iter().copied().collect());
            } else if name == "final_ln_b" {
                final_ln_b = Some(arr.iter().copied().collect());
            } else if let Some(rest) = name.strip_prefix('L') {
                // "L{i}/{k}"
                let (idx_s, k) = rest
                    .split_once('/')
                    .ok_or_else(|| anyhow::anyhow!("arena tensor {name}: missing layer key"))?;
                let i: usize = idx_s
                    .parse()
                    .map_err(|_| anyhow::anyhow!("arena tensor {name}: bad layer index"))?;
                if i < n_layers {
                    layer_maps[i].insert(k.to_string(), arr);
                    // If this tensor was baked bf16, stash its raw bits alongside the f32 copy.
                    if let Some((sh, bits)) = loaded.tensor_bf16(name)? {
                        layer_bf16[i].insert(k.to_string(), (sh, bits));
                    }
                }
                // layers beyond n_layers are ignored (caller asked for fewer)
            }
            // any other name (e.g. unrelated metadata tensors) is ignored
        }

        let layers = layer_maps
            .into_iter()
            .zip(layer_bf16)
            .map(|(map, bf16)| EsmLayer { map, bf16 })
            .collect();
        let final_ln = match (final_ln_w, final_ln_b) {
            (Some(w), Some(b)) => Some((w, b)),
            _ => None,
        };
        Ok(EsmWeights { emb, layers, final_ln })
    }

    pub fn word_emb(&self) -> Array2<f32> {
        self.emb["word_emb"].clone().into_dimensionality().unwrap()
    }
    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;
    use ndarray_npy::write_npy;

    #[test]
    fn load_without_final_ln_succeeds() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        // emb/word_emb.npy (minimal)
        std::fs::create_dir_all(root.join("emb")).unwrap();
        write_npy(root.join("emb").join("word_emb.npy"), &Array1::<f32>::zeros(4)).unwrap();
        // L0/ minimal (one tensor is enough for load_dir; readers index by key lazily)
        std::fs::create_dir_all(root.join("L0")).unwrap();
        write_npy(root.join("L0").join("q_w.npy"), &Array1::<f32>::zeros(2)).unwrap();
        // NO final_ln_w.npy / final_ln_b.npy at root.
        let w = EsmWeights::load(root, 1).expect("load must not panic when final_ln is absent");
        assert!(w.final_ln.is_none(), "final_ln must be None when files are absent");
        assert_eq!(w.n_layers(), 1);
    }
}
