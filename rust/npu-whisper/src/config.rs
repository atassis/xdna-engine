//! Whisper encoder config. Dims are fields (not consts) so other Whisper sizes plug in later.

#[derive(Clone, Copy, Debug)]
pub struct WhisperCfg {
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub ffn: usize,
    pub n_mels: usize,
}

impl WhisperCfg {
    /// whisper-small encoder (d_model 768, 12 layers, 12 heads, head_dim 64, ffn 3072, n_mels 80).
    pub const SMALL: WhisperCfg = WhisperCfg {
        d_model: 768,
        n_layers: 12,
        n_heads: 12,
        head_dim: 64,
        ffn: 3072,
        n_mels: 80,
    };

    /// whisper-large-v3-turbo encoder (d_model 1280, 32 layers, 20 heads, head_dim 64 -- unchanged
    /// from SMALL, 1280/20 == 768/12 -- ffn 5120, n_mels 128). Host-path only today: this cfg has
    /// no matching NPU-resident kernel (see whisper-v3-turbo-on-npu).
    pub const TURBO: WhisperCfg = WhisperCfg {
        d_model: 1280,
        n_layers: 32,
        n_heads: 20,
        head_dim: 64,
        ffn: 5120,
        n_mels: 128,
    };
}
