// rust/npu-weights/src/arch/mod.rs
pub mod bert;
pub mod clip;
pub mod dinov2;
pub mod edsr;
pub mod espcn;
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

/// Constructor for one arch. A plain fn pointer, not `Box<dyn Fn>`, so [`REGISTRY`] is a `const`
/// table rather than something built fresh on every `get()` call.
type Ctor = fn() -> Box<dyn Arch>;

/// (name, constructor), alphabetical, one entry per `arch/*.rs` module. The single source for both
/// [`get`]'s dispatch and [`ARCH_NAMES`] -- a new module is wired in by adding one line here, and
/// `npu weights --arch`'s clap possible-values + help text (`npu-cli/src/cli_def.rs`) read
/// `ARCH_NAMES`, so a module that forgets this line cannot be dispatched to either, and one that
/// remembers it cannot be missing from the CLI's help.
const REGISTRY: &[(&str, Ctor)] = &[
    ("bert", || Box::new(bert::Bert)),
    ("clip", || Box::new(clip::Clip)),
    ("dinov2", || Box::new(dinov2::Dinov2)),
    ("edsr", || Box::new(edsr::Edsr)),
    ("espcn", || Box::new(espcn::Espcn)),
    ("esm", || Box::new(esm::Esm)),
    ("fastconformer", || Box::new(fastconformer::FastConformer)),
    ("gigaam", || Box::new(gigaam::Gigaam)),
    ("modernbert", || Box::new(modernbert::ModernBert)),
    ("opt", || Box::new(opt::Opt)),
    ("resnet", || Box::new(resnet::Resnet)),
    ("vit", || Box::new(vit::Vit)),
    ("whisper", || Box::new(whisper::Whisper)),
];

const ARCH_NAMES_ARR: [&str; REGISTRY.len()] = {
    let mut out = [""; REGISTRY.len()];
    let mut i = 0;
    while i < REGISTRY.len() {
        out[i] = REGISTRY[i].0;
        i += 1;
    }
    out
};

/// Every `--arch` value `get` accepts, derived from [`REGISTRY`] at compile time -- same shape as
/// `npu_runtime::config_doc::SERVER_KEYS`, a `pub const` slice single-sourced with the code that
/// consumes the names.
pub const ARCH_NAMES: &[&str] = &ARCH_NAMES_ARR;

pub fn get(name: &str) -> anyhow::Result<Box<dyn Arch>> {
    match REGISTRY.iter().find(|(n, _)| *n == name) {
        Some((_, ctor)) => Ok(ctor()),
        None => anyhow::bail!("unknown arch {name:?} (one of: {})", ARCH_NAMES.join(", ")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `ARCH_NAMES` and `get`'s dispatch are the same table by construction, but this is the
    /// assertion that grades any future change to that construction: every name the CLI can offer
    /// must actually build, and must build the arch it names.
    #[test]
    fn every_arch_name_dispatches_to_the_arch_it_names() {
        assert_eq!(ARCH_NAMES.len(), REGISTRY.len());
        for name in ARCH_NAMES {
            let a = get(name).unwrap_or_else(|e| panic!("ARCH_NAMES has {name:?} but get() does not: {e}"));
            assert_eq!(a.name(), *name);
        }
    }

    #[test]
    fn an_unknown_arch_is_rejected() {
        assert!(get("no-such-arch").is_err());
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
