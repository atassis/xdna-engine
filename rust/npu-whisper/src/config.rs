//! Whisper encoder config. Dims are fields (not consts) so other Whisper sizes plug in later.

#[derive(Clone, Copy, Debug)]
pub struct WhisperCfg {
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub ffn: usize,
    pub n_mels: usize,
    /// `verify_whisper --npu`'s `encoded` rel-L2 gate (NPU bf16/bfp16 vs the ONNX f32 golden). A
    /// PER-CONFIG bound, not a per-binary one: bf16/bfp16 error accumulates over `n_layers`, so a
    /// bound measured at one depth does not transfer to another (measured 2026-09-08: turbo's
    /// shipped-default `encoded` rel is 0.1917 at 32 layers against SMALL's bar of 0.08 for 12 --
    /// see each variant's own derivation below).
    pub tol_npu: f32,
}

impl WhisperCfg {
    /// whisper-small encoder (d_model 768, 12 layers, 12 heads, head_dim 64, ffn 3072, n_mels 80).
    ///
    /// `tol_npu = 0.08` is UNCALIBRATED: introduced with this cfg (2026-06-15, faadc80) with no
    /// comment and no derivation, and unchanged since. The |alpha| 2.069e-3 / 3.426e-3 "~2x the
    /// larger" derivation sometimes cited for it belongs to the sibling `TOL_SCALE` gate in
    /// `verify_whisper.rs` (the `[scale]`/alpha check), not here -- a `///` comment documents the
    /// item BELOW it, and `git log -p` shows 0.08 arriving bare.
    ///
    /// Left as-is because there is nothing to calibrate against: `verify_whisper --npu` times out
    /// in ctx2's first matmul (ERT_CMD_STATE_TIMEOUT) at this pin, so the whisper-small NPU path
    /// produces no device evidence.
    pub const SMALL: WhisperCfg = WhisperCfg {
        d_model: 768,
        n_layers: 12,
        n_heads: 12,
        head_dim: 64,
        ffn: 3072,
        n_mels: 80,
        tol_npu: 0.08,
    };

    /// whisper-large-v3-turbo encoder (d_model 1280, 32 layers, 20 heads, head_dim 64 -- unchanged
    /// from SMALL, 1280/20 == 768/12 -- ffn 5120, n_mels 128).
    ///
    /// `tol_npu = 0.40`: derived 2026-09-08 from ONE measured arm, not two -- weaker evidence than
    /// SMALL's intended two-arm style. Measured 2026-09-03 (device, shipped default = host MHA):
    /// `encoded` rel-L2 = 0.1917, and that exact configuration transcribes both test clips correctly
    /// (`encoder-mha-e2e-gate-is-blocked-on-two-unrelated-defects`). Error starts at block_0
    /// (8.335e-3 vs a host f32 reference's 2.295e-6) and grows ~8x over 32 layers, near the sqrt(32)
    /// = 5.7x an accumulating bfp16 rounding error predicts -- no step change, so 0.1917 looks like
    /// the FORMAT's accumulated error at this depth, not a defect. Set at 2x that one known-good
    /// measurement (2 * 0.1917 = 0.3834, rounded up): still 2x above the non-causal NPU-MHA
    /// experimental arm (0.2602, itself not a "known good" baseline), and still catches a >5x
    /// regression. NEEDS A DEVICE SESSION to reach SMALL's confidence: a second independent
    /// known-good arm (e.g. once turbo's causal on-chip MHA lands) to set this the way TOL_SCALE
    /// was actually set, not just doubled from one point.
    pub const TURBO: WhisperCfg = WhisperCfg {
        d_model: 1280,
        n_layers: 32,
        n_heads: 20,
        head_dim: 64,
        ffn: 5120,
        n_mels: 128,
        tol_npu: 0.40,
    };
}
