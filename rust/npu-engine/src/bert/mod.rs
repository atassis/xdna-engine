pub mod weights;
pub mod frontend;
pub mod encoder;
pub mod head;

use std::path::Path;
use std::rc::Rc;

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
    pub fn build(cfg: &ScenarioConfig, root: &Path, dev: Rc<Device>) -> Self {
        // Uniform declarative entry point: arena (bake-on-missing) when artifacts.source is set,
        // else the legacy npy dir -- all behind one call.
        let weights = Rc::new(
            BertWeights::load_for(&cfg.artifacts, root, cfg.model.n_layers).expect("bert weights"),
        );
        let frontend = EmbedFrontend::new(
            &root.join(&cfg.artifacts.tokenizer), weights.clone(), cfg.model.max_seq);
        let encoder = BertEncoder::new(dev, root, &weights, cfg.model.n_heads, cfg.model.head_dim);
        let head = EmbedHead {
            pooling: Pooling::parse(&cfg.embeddings.pooling),
            normalize: cfg.embeddings.normalize,
        };
        EmbedPipeline { frontend, encoder, head }
    }

    /// Full pipeline: text -> embedding vector.
    pub fn embed(&self, text: String) -> Vec<f32> {
        let (x, valid) = self.frontend.run(text);
        let enc = crate::pipeline::Encoder::forward_last(&self.encoder, &x, valid);
        self.head.run(&enc, valid)
    }
}

impl crate::pipeline::Embedder for EmbedPipeline {
    fn embed_one(&self, text: String) -> Vec<f32> {
        self.embed(text)
    }
}
