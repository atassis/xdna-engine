//! Model config — mirrors the `feat/general-engine` `ModelCfg` shape (dims are fields, not
//! consts) so Parakeet plugs into the generalized pipeline without per-model constants.

#[derive(Clone, Copy, Debug)]
pub struct ModelCfg {
    pub hidden: usize,   // d_model
    pub ff: usize,       // d_ff
    pub n_heads: usize,
    pub head_dim: usize,
    pub n_layers: usize,
}

impl ModelCfg {
    /// Parakeet-tdt-0.6b-v3 FastConformer (verified from the ONNX in parakeet-phase1-weights).
    pub const PARAKEET_V3: ModelCfg = ModelCfg {
        hidden: 1024,
        ff: 4096,
        n_heads: 8,
        head_dim: 128,
        n_layers: 24,
    };
}
