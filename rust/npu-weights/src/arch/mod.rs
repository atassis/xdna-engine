// rust/npu-weights/src/arch/mod.rs
pub mod bert;
pub mod clip;
pub mod dinov2;
pub mod esm;
pub mod fastconformer;
pub mod gigaam;
pub mod modernbert;
pub mod opt;
pub mod resnet;
pub mod vit;
pub mod whisper;
use std::collections::BTreeMap;

/// A source tensor: row-major f32 values + shape. (We upcast source to f32 at read; bake decides
/// the stored dtype.)
#[derive(Clone)]
pub struct RawTensor { pub shape: Vec<usize>, pub data: Vec<f32> }

/// A baked output tensor in final engine layout, with target dtype.
#[derive(Clone)]
pub struct OutTensor { pub shape: Vec<usize>, pub data: Vec<f32>, pub bf16: bool }

pub trait Arch {
    fn name(&self) -> &'static str;
    /// Source tensor names this arch requires (hard error if any missing).
    fn required_tensors(&self, n_layers: usize) -> Vec<String>;
    /// Transform source bag -> baked bag (name -> OutTensor). Names use '/' separators.
    fn transform(&self, src: &BTreeMap<String, RawTensor>) -> anyhow::Result<BTreeMap<String, OutTensor>>;
}

pub fn get(name: &str) -> anyhow::Result<Box<dyn Arch>> {
    match name {
        "bert" => Ok(Box::new(bert::Bert)),
        "clip" => Ok(Box::new(clip::Clip)),
        "dinov2" => Ok(Box::new(dinov2::Dinov2)),
        "esm" => Ok(Box::new(esm::Esm)),
        "fastconformer" => Ok(Box::new(fastconformer::FastConformer)),
        "gigaam" => Ok(Box::new(gigaam::Gigaam)),
        "modernbert" => Ok(Box::new(modernbert::ModernBert)),
        "opt" => Ok(Box::new(opt::Opt)),
        "resnet" => Ok(Box::new(resnet::Resnet)),
        "vit" => Ok(Box::new(vit::Vit)),
        "whisper" => Ok(Box::new(whisper::Whisper)),
        other => anyhow::bail!("unknown arch {other:?}"),
    }
}

/// row-major 2D transpose [r,c] -> [c,r].
pub fn transpose2d(t: &RawTensor) -> RawTensor {
    assert_eq!(t.shape.len(), 2, "transpose2d needs 2D");
    let (r, c) = (t.shape[0], t.shape[1]);
    let mut out = vec![0f32; r * c];
    for i in 0..r {
        for j in 0..c {
            out[j * r + i] = t.data[i * c + j];
        }
    }
    RawTensor { shape: vec![c, r], data: out }
}
