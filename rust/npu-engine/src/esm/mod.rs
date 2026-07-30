pub mod frontend;
pub mod rope;
pub mod weights;
pub mod encoder;
pub mod native;

pub use frontend::EsmFrontend;

use std::path::Path;
use std::rc::Rc;

use crate::api::EngineError;
use crate::bert::head::{EmbedHead, Pooling};
use crate::config::ScenarioConfig;
use crate::esm::{encoder::{EsmEncoder, EsmEncoderNative}, frontend::EsmFrontend as Frontend_, weights::EsmWeights};
use crate::pipeline::{Encoder, Frontend, Head};
use npu_xrt::Device;

/// ESM-2 embedding pipeline: residue frontend -> pre-norm NPU encoder -> mean-pool head.
/// `cfg.model.kernel` selects the encoder: "zeropad" (default, reuse 768/3072 kernels) or "native"
/// (real-K xclbins via the standalone NativeKernel dispatcher).
pub struct EsmEmbedPipeline {
    frontend: Frontend_,
    encoder: Box<dyn Encoder>,
    head: EmbedHead,
}
impl EsmEmbedPipeline {
    pub fn build(cfg: &ScenarioConfig, root: &Path, dev: Rc<Device>) -> Result<Self, EngineError> {
        // Uniform declarative entry point: checkpoint (bake-on-missing) when artifacts.source is set,
        // else NPU_WEIGHTS_CHECKPOINT env, else the legacy npy dir -- all behind one call.
        let w = Rc::new(
            EsmWeights::load_for(&cfg.artifacts, root, cfg.model.n_layers)
                .map_err(|e| EngineError::Load(format!("esm weights: {e}")))?,
        );
        let frontend = Frontend_::new(w.clone(), cfg.model.max_seq);
        let m = &cfg.model;
        let encoder: Box<dyn Encoder> = if m.kernel == "native" {
            Box::new(EsmEncoderNative::new(dev, root, &w, m.hidden, m.ff, m.n_heads, m.head_dim))
        } else {
            Box::new(EsmEncoder::new(dev, root, &w, m.hidden, m.ff, m.n_heads, m.head_dim))
        };
        let head = EmbedHead {
            pooling: Pooling::parse(&cfg.embeddings.pooling),
            normalize: cfg.embeddings.normalize,
        };
        Ok(EsmEmbedPipeline { frontend, encoder, head })
    }
    /// Full pipeline: protein string -> embedding vector.
    pub fn embed(&self, seq: String) -> Vec<f32> {
        let (x, valid) = self.frontend.run(seq);
        let enc = self.encoder.forward_last(&x, valid);
        self.head.run(&enc, valid)
    }
}

impl crate::pipeline::Embedder for EsmEmbedPipeline {
    fn embed_one(&self, text: String) -> Result<Vec<f32>, EngineError> {
        Ok(self.embed(text))
    }
}
