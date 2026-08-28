pub mod weights;
pub mod frontend;
pub mod encoder;
pub mod head;

use std::path::Path;
use std::rc::Rc;

use crate::api::EngineError;
use crate::config::ScenarioConfig;
use crate::pipeline::{Frontend, Head};
use encoder::BertEncoder;
use frontend::EmbedFrontend;
use head::{EmbedHead, Pooling};
use weights::BertWeights;
use npu_xrt::Device;

pub struct EmbedPipeline {
    frontend: EmbedFrontend,
    encoder: BertEncoder,
    head: EmbedHead,
}

impl EmbedPipeline {
    pub fn build(cfg: &ScenarioConfig, root: &Path, dev: Rc<Device>) -> Result<Self, EngineError> {
        let m = cfg.model_or_err().map_err(EngineError::Load)?;
        // Uniform declarative entry point: checkpoint (bake-on-missing) when artifacts.source is set,
        // else the legacy npy dir -- all behind one call.
        let weights = Rc::new(
            BertWeights::load_for(&cfg.artifacts, root, m.n_layers)
                .map_err(|e| EngineError::Load(format!("bert weights: {e}")))?,
        );
        let frontend = EmbedFrontend::new(
            &root.join(&cfg.artifacts.tokenizer), weights.clone(), m.max_seq);
        let encoder = BertEncoder::new(dev, root, &weights, m.n_heads, m.head_dim);
        let head = EmbedHead {
            pooling: Pooling::parse(&cfg.embeddings.pooling),
            normalize: cfg.embeddings.normalize,
        };
        Ok(EmbedPipeline { frontend, encoder, head })
    }

    /// Full pipeline: text -> embedding vector.
    pub fn embed(&self, text: String) -> Vec<f32> {
        let (x, valid) = self.frontend.run(text);
        let enc = crate::pipeline::Encoder::forward_last(&self.encoder, &x, valid);
        self.head.run(&enc, valid)
    }
}

impl crate::pipeline::Embedder for EmbedPipeline {
    fn embed_one(&self, text: String) -> Result<Vec<f32>, EngineError> {
        Ok(self.embed(text))
    }
}

// engine-open-capability-contract probe instance #1: text in, vector out, genuinely &self (no
// interior mutability laundering needed -- `&mut self` on `run` just reborrows).
impl crate::capability::Servable for EmbedPipeline {
    fn capabilities(&self) -> crate::capability::Capability {
        crate::capability::Capability::EMBED
    }
    fn run(&mut self, req: crate::capability::Request) -> Result<crate::capability::Response, EngineError> {
        match req {
            crate::capability::Request::Text(text) => Ok(crate::capability::Response::Vector(self.embed(text))),
            other => Err(EngineError::Unsupported(
                format!("EmbedPipeline: expected a text request, got {}", other.shape()))),
        }
    }
}
