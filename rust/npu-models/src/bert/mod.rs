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
use npu_asr::ctx2::{CtxAShape, Precision};
use npu_asr::tuning::TuningConfig;
use npu_asr_host::batch::passage_batches;

/// Packed CLS is an instance-local native-bf16 contract; process-global precision/tile selectors
/// apply only to the single-text instance. No passage padding is needed on this native rail.
pub(crate) fn batch_plan(cfg: &ScenarioConfig, root: &Path) -> Result<Option<(CtxAShape, TuningConfig)>, EngineError> {
    let Some(rows) = cfg.embeddings.batch_rows else { return Ok(None); };
    let m = cfg.model_or_err().map_err(EngineError::Load)?;
    if m.precision != "native" {
        return Err(EngineError::Load("batch_rows supports only explicit [model] precision = native; bf16/Int8 packed numerics are not supported".into()));
    }
    if cfg.embeddings.pooling != "cls" || !cfg.embeddings.normalize {
        return Err(EngineError::Load("batch_rows requires pooling = cls and normalize = true".into()));
    }
    if m.hidden != 768 || m.ff != 3072 || m.n_heads.checked_mul(m.head_dim) != Some(768)
        || m.n_heads == 0 || m.max_seq == 0 || m.max_seq > 512 || rows < m.max_seq {
        return Err(EngineError::Load("batch_rows requires BERT 768/3072, valid heads, and 1..=512 passage positions fitting the row capacity".into()));
    }
    let shape = CtxAShape { rows, streams: vec![768, 3072], tile: Some((32, 32, 32)), ..CtxAShape::whisper_small() };
    let tuning = crate::tuning_profile::resolve(root, Precision::NativeBf16);
    shape.validate(&tuning).map_err(EngineError::Load)?;
    Ok(Some((shape, tuning)))
}

pub(crate) fn preflight_batch(cfg: &ScenarioConfig, root: &Path) -> Result<(), EngineError> {
    if let Some((shape, tuning)) = batch_plan(cfg, root)? {
        shape.preflight_native(root, &tuning).map_err(EngineError::Load)?;
    }
    Ok(())
}

fn checked_single_request(
    batch_rows: Option<usize>, text: &str,
    packed: impl FnOnce(&[&str]) -> Result<Vec<Vec<f32>>, EngineError>,
    legacy: impl FnOnce() -> Vec<f32>,
) -> Result<Vec<f32>, EngineError> {
    if batch_rows.is_some() {
        let mut rows = packed(&[text])?;
        if rows.len() != 1 {
            return Err(EngineError::Unsupported("singleton packed embedding returned wrong row count".into()));
        }
        Ok(rows.remove(0))
    } else {
        Ok(legacy())
    }
}

pub struct EmbedPipeline {
    frontend: EmbedFrontend,
    encoder: BertEncoder,
    head: EmbedHead,
    /// Kept alongside `encoder`'s own clone so `bo_bytes()` can read the device's resident counter
    /// without reaching into the encoder's private fields.
    dev: Rc<Device>,
    batch_rows: Option<usize>,
}

impl EmbedPipeline {
    pub fn build(cfg: &ScenarioConfig, root: &Path, dev: Rc<Device>) -> Result<Self, EngineError> {
        preflight_batch(cfg, root)?;
        let plan = batch_plan(cfg, root)?;
        let m = cfg.model_or_err().map_err(EngineError::Load)?;
        // Uniform declarative entry point: checkpoint (bake-on-missing) when artifacts.source is set,
        // else the legacy npy dir -- all behind one call.
        let weights = Rc::new(
            BertWeights::load_for(&cfg.artifacts, root, m.n_layers)
                .map_err(|e| EngineError::Load(format!("bert weights: {e}")))?,
        );
        let frontend = EmbedFrontend::new(
            &root.join(&cfg.artifacts.tokenizer), weights.clone(), m.max_seq);
        let encoder = match plan {
            Some((shape, tuning)) => {
                eprintln!("[bert] packed CLS: native bf16 32x32x32, row capacity {}", shape.rows);
                BertEncoder::with_shape(dev.clone(), root, &weights, m.n_heads, m.head_dim, &tuning, shape)
            }
            None => BertEncoder::new(dev.clone(), root, &weights, m.n_heads, m.head_dim),
        };
        let head = EmbedHead {
            pooling: Pooling::parse(&cfg.embeddings.pooling),
            normalize: cfg.embeddings.normalize,
        };
        Ok(EmbedPipeline { frontend, encoder, head, dev, batch_rows: cfg.embeddings.batch_rows })
    }

    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EngineError> {
        let Some(capacity) = self.batch_rows else {
            // Unconfigured BERT arrays retain single-text tokenization/truncation and precision.
            return texts.iter().map(|text| Ok(self.embed((*text).to_string()))).collect();
        };
        let ids = self.frontend.batch_token_ids(texts).map_err(EngineError::Unsupported)?;
        let lengths: Vec<_> = ids.iter().map(Vec::len).collect();
        let batches = passage_batches(&lengths, self.frontend.max_seq(), capacity)
            .map_err(EngineError::Unsupported)?;
        let mut out = Vec::with_capacity(texts.len());
        for batch in batches {
            let (x, spans) = self.frontend.run_packed(&ids[batch], capacity).map_err(EngineError::Unsupported)?;
            let enc = self.encoder.forward_packed(&x, &spans).map_err(EngineError::Unsupported)?;
            out.extend(self.head.run_cls_packed(&enc, &spans).map_err(EngineError::Unsupported)?);
        }
        Ok(out)
    }

    /// Strict fallible entry point; configured CLS uses the same validation as arrays.
    pub fn embed_checked(&self, text: &str) -> Result<Vec<f32>, EngineError> {
        checked_single_request(self.batch_rows, text, |texts| self.embed_batch(texts), || self.embed(text.to_owned()))
    }

    /// Legacy/internal infallible control, retaining stored tokenizer truncation.
    /// Serving callers should use the fallible checked entry point.
    pub fn embed(&self, text: String) -> Vec<f32> {
        let (x, valid) = self.frontend.run(text);
        let enc = crate::pipeline::Encoder::forward_last(&self.encoder, &x, valid);
        self.head.run(&enc, valid)
    }
}

impl crate::pipeline::Embedder for EmbedPipeline {
    fn embed_one(&self, text: String) -> Result<Vec<f32>, EngineError> {
        self.embed_checked(&text)
    }
    fn embed_many(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EngineError> {
        self.embed_batch(texts)
    }
    fn bo_bytes(&self) -> u64 {
        self.dev.resident_bo_bytes()
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
            crate::capability::Request::Text(text) => self.embed_checked(&text).map(crate::capability::Response::Vector),
            crate::capability::Request::Texts(texts) => {
                let refs: Vec<_> = texts.iter().map(String::as_str).collect();
                self.embed_batch(&refs).map(crate::capability::Response::Vectors)
            },
            other => Err(EngineError::Unsupported(
                format!("EmbedPipeline: expected a text request, got {}", other.shape()))),
        }
    }
}

#[cfg(test)]
mod batch_plan_tests {
    use super::*;

    fn cfg() -> ScenarioConfig {
        ScenarioConfig::from_toml_str(include_str!("../../../../scenarios/bge-base.toml")).unwrap()
    }

    #[test]
    fn cls_staged_root_preflight_uses_published_contracts_only() {
        use sha2::{Digest, Sha256};
        let root = tempfile::tempdir().unwrap();
        let c = ScenarioConfig::from_toml_str(include_str!("../../../../scenarios/bge-base-cls.toml")).unwrap();
        let dir = root.path().join("kernels/whole_array");
        std::fs::create_dir_all(&dir).unwrap();
        let (shape, tuning) = batch_plan(&c, root.path()).unwrap().unwrap();
        let hash = format!("{:x}", Sha256::digest(b"fixture"));
        for (stem, n) in [("1024x800x3072_32x32x32_8c_modalsilunat".into(), 3072),
            (shape.stream_stem(&tuning, 768, 0), 768), (shape.stream_stem(&tuning, 3072, 0), 3072)] {
            std::fs::write(npu_asr::kernel_registry::xclbin_path(&dir, &stem), b"fixture").unwrap();
            std::fs::write(dir.join(format!("insts_{stem}.bin")), b"fixture").unwrap();
            let receipt = serde_json::json!({"rows":1024,"k":800,"n":n,"tile":[32,32,32],"precision":"native", "xclbin_sha256":hash,"insts_sha256":hash});
            std::fs::write(dir.join(format!("contract_{stem}.json")), receipt.to_string()).unwrap();
        }
        assert!(!root.path().join("mlir-aie").exists());
        preflight_batch(&c, root.path()).unwrap();
        let stem = shape.stream_stem(&tuning, 768, 0);
        let receipt = dir.join(format!("contract_{stem}.json"));
        std::fs::copy(&receipt, root.path().join("source-only.json")).unwrap();
        std::fs::remove_file(receipt).unwrap();
        assert!(preflight_batch(&c, root.path()).is_err());
    }

    #[test]
    fn cls_scenario_declares_exact_native_resident_and_stream_closure() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scenarios/bge-base-cls.toml");
        let c = ScenarioConfig::from_toml_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let (shape, tuning) = batch_plan(&c, Path::new(".")).unwrap().unwrap();
        assert_eq!(shape.rows, 1024);
        assert_eq!(shape.streams, vec![768,3072]);
        assert_eq!(shape.tile, Some((32,32,32)));
        assert_eq!(tuning.precision, Precision::NativeBf16);
        let mut expected = vec!["1024x800x3072_32x32x32_8c_modalsilunat".to_string()];
        expected.extend(shape.streams.iter().map(|&n| shape.stream_stem(&tuning, n, 0)));
        let text = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scenarios/bge-base-cls.toml")).unwrap();
        let value: toml::Value = toml::from_str(&text).unwrap();
        let actual: Vec<_> = value["kernels"]["whole_array"]["required"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_owned()).collect();
        assert_eq!(actual, expected);
        assert_eq!(cfg().embeddings.pooling, "mean");
        assert!(batch_plan(&cfg(), Path::new(".")).unwrap().is_none());
    }

    #[test]
    fn checked_single_request_uses_packed_errors_only_when_configured() {
        let packed = |texts: &[&str]| -> Result<Vec<Vec<f32>>, EngineError> {
            assert_eq!(texts, &["overlength"]);
            Err(EngineError::Unsupported("513 tokens".into()))
        };
        let legacy = || vec![7.];
        assert!(checked_single_request(Some(1024), "overlength", packed, legacy).unwrap_err().to_string().contains("513 tokens"));
        assert_eq!(checked_single_request(None, "overlength", packed, legacy).unwrap(), vec![7.]);
        assert_eq!(checked_single_request(Some(1024), "x", |texts| {
            assert_eq!(texts, &["x"]);
            Ok(vec![vec![1.,0.]])
        }, || panic!("configured request reached legacy control")).unwrap(), vec![1.,0.]);
        assert!(checked_single_request(Some(1024), "x", |_| Ok(vec![]), legacy).is_err());
    }

    #[test]
    fn packed_instance_is_local_native_and_defaults_are_serial() {
        let root = Path::new(".");
        let mut c = cfg();
        assert!(batch_plan(&c, root).unwrap().is_none());
        c.embeddings.batch_rows = Some(1024);
        assert!(batch_plan(&c, root).unwrap_err().to_string().contains("precision = native"));
        c.model.as_mut().unwrap().precision = "int8".into();
        assert!(batch_plan(&c, root).is_err());
        c.model.as_mut().unwrap().precision = "native".into();
        assert!(batch_plan(&c, root).unwrap_err().to_string().contains("pooling"));
        c.embeddings.pooling = "cls".into();
        let (shape, tuning) = batch_plan(&c, root).unwrap().unwrap();
        assert_eq!(shape.rows, 1024);
        assert_eq!(shape.tile, Some((32,32,32)));
        assert_eq!(tuning.precision, Precision::NativeBf16);
        c.embeddings.normalize = false;
        assert!(batch_plan(&c, root).is_err());
        c.embeddings.normalize = true;
        for rows in [0, 511, 1025] {
            c.embeddings.batch_rows = Some(rows);
            assert!(batch_plan(&c, root).is_err());
        }
    }
}
