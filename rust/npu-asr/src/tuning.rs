//! The single source of the engine's performance knobs. Replaces scattered `NPU_*` env reads:
//! defaults are BAKED here (the measured Class-0 winners); env vars survive only as overrides
//! (`with_env_overrides`), parsed in exactly one place.

use crate::ctx2::Precision;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TuningConfig {
    pub precision: Precision,
    pub modal_epilogue: bool,      // NPU_MODAL_EPI (gated to non-int8 at construction)
    pub subsample_on_npu: bool,    // NPU_SS_NPU
    pub layernorm_on_npu: bool,    // NPU_LN_NPU
    pub glu_fused: bool,           // NPU_GLU_FUSED
    pub qkv_overlap: bool,         // NPU_QKV_OVERLAP
    pub mm2_pipeline: bool,        // NPU_MM2_PIPELINE
    pub int8_fast_epi: bool,       // NPU_INT8_FASTEPI
    pub int8_onchip_dequant: bool, // NPU_INT8_ONCHIP
}

impl TuningConfig {
    /// The committed Class-0/1 winners — exactly today's effective defaults. `qkv_overlap` is `false`
    /// for all precisions (behavior-preserving); a profile may later set it true on native once the
    /// probe harness confirms.
    pub fn baked_default(precision: Precision) -> Self {
        TuningConfig {
            precision,
            modal_epilogue: true,
            subsample_on_npu: true,
            layernorm_on_npu: false,
            glu_fused: true,
            qkv_overlap: false,
            mm2_pipeline: true,
            int8_fast_epi: true,
            int8_onchip_dequant: false,
        }
    }

    /// Apply `NPU_*` env vars ON TOP, with the SAME semantics the old scattered reads used. The one
    /// and only place env is parsed. Precision is NOT re-read here (already chosen by the caller).
    pub fn with_env_overrides(mut self) -> Self {
        // "!= 0" knobs: any value except the string "0" enables.
        let not_zero = |k: &str, dflt: bool| match std::env::var(k).ok().as_deref() {
            Some("0") => false,
            Some(_) => true,
            None => dflt,
        };
        self.modal_epilogue = not_zero("NPU_MODAL_EPI", self.modal_epilogue);
        self.subsample_on_npu = not_zero("NPU_SS_NPU", self.subsample_on_npu);
        self.glu_fused = not_zero("NPU_GLU_FUSED", self.glu_fused);
        self.mm2_pipeline = not_zero("NPU_MM2_PIPELINE", self.mm2_pipeline);
        self.int8_fast_epi = not_zero("NPU_INT8_FASTEPI", self.int8_fast_epi);
        // "== 1" knobs: only the exact value "1" enables; any other set value disables.
        let is_one = |k: &str, dflt: bool| match std::env::var(k).ok().as_deref() {
            Some("1") => true,
            Some(_) => false,
            None => dflt,
        };
        self.layernorm_on_npu = is_one("NPU_LN_NPU", self.layernorm_on_npu);
        self.qkv_overlap = is_one("NPU_QKV_OVERLAP", self.qkv_overlap);
        self.int8_onchip_dequant = is_one("NPU_INT8_ONCHIP", self.int8_onchip_dequant);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ctx2::Precision;

    #[test]
    fn baked_default_matches_legacy_defaults() {
        let bf16 = TuningConfig::baked_default(Precision::FastBf16);
        assert!(bf16.modal_epilogue);
        assert!(bf16.subsample_on_npu);
        assert!(!bf16.layernorm_on_npu);
        assert!(bf16.glu_fused);
        assert!(!bf16.qkv_overlap);     // legacy default false for ALL precisions
        assert!(bf16.mm2_pipeline);
        assert!(!bf16.int8_onchip_dequant);
        let i8 = TuningConfig::baked_default(Precision::Int8);
        assert!(i8.int8_fast_epi);
        assert!(!i8.int8_onchip_dequant);
    }

    #[test]
    fn env_override_beats_default() {
        // mutates process env -> run single-threaded (cargo test -- --test-threads=1)
        std::env::set_var("NPU_GLU_FUSED", "0");
        let c = TuningConfig::baked_default(Precision::FastBf16).with_env_overrides();
        assert!(!c.glu_fused, "NPU_GLU_FUSED=0 must override the baked true");
        std::env::remove_var("NPU_GLU_FUSED");
    }
}
