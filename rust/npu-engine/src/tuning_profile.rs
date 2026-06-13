//! Resolve a TuningConfig: detect the hardware class, load the committed per-class profile, apply it
//! over baked defaults, then env overrides. (The Class-2 local probe cache is a reserved future slot.)

use npu_asr::ctx2::Precision;
use npu_asr::tuning::TuningConfig;
use serde::Deserialize;

/// Returns the hardware-class id used to pick `profiles/<class>.toml`. Single supported class today;
/// TODO: derive from xrt-smi / device id when a second class is targeted.
pub fn detect_hw_class() -> String {
    "xdna2".to_string()
}

/// The committed profile file. Every field is optional → omitted keys keep the baked default.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct TuningProfile {
    pub modal_epilogue: Option<bool>,
    pub subsample_on_npu: Option<bool>,
    pub layernorm_on_npu: Option<bool>,
    pub glu_fused: Option<bool>,
    pub qkv_overlap: Option<bool>,
    pub mm2_pipeline: Option<bool>,
    pub int8_fast_epi: Option<bool>,
    pub int8_onchip_dequant: Option<bool>,
    /// Class-3: measured CPU↔NPU crossover (seconds of audio). Recorded; dispatch wired in spec B.
    pub cpu_npu_crossover_s: Option<f32>,
}

impl TuningProfile {
    pub fn from_toml_str(s: &str) -> Result<Self, toml::de::Error> {
        toml::from_str(s)
    }
    /// Apply this profile over the baked default for `precision`, then env overrides on top.
    pub fn resolve(&self, precision: Precision) -> TuningConfig {
        let mut c = TuningConfig::baked_default(precision);
        if let Some(v) = self.modal_epilogue { c.modal_epilogue = v; }
        if let Some(v) = self.subsample_on_npu { c.subsample_on_npu = v; }
        if let Some(v) = self.layernorm_on_npu { c.layernorm_on_npu = v; }
        if let Some(v) = self.glu_fused { c.glu_fused = v; }
        if let Some(v) = self.qkv_overlap { c.qkv_overlap = v; }
        if let Some(v) = self.mm2_pipeline { c.mm2_pipeline = v; }
        if let Some(v) = self.int8_fast_epi { c.int8_fast_epi = v; }
        if let Some(v) = self.int8_onchip_dequant { c.int8_onchip_dequant = v; }
        c.with_env_overrides()
    }
}

/// Load `profiles/<detect()>.toml` under `root`; if absent, baked default (+ env). Never fails hard
/// on a missing profile (baked default is a valid resolution).
pub fn resolve(root: &std::path::Path, precision: Precision) -> TuningConfig {
    let path = root.join("profiles").join(format!("{}.toml", detect_hw_class()));
    match std::fs::read_to_string(&path) {
        Ok(s) => TuningProfile::from_toml_str(&s)
            .unwrap_or_else(|e| panic!("parse {}: {e}", path.display()))
            .resolve(precision),
        Err(_) => TuningConfig::baked_default(precision).with_env_overrides(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_overrides_default_then_env() {
        let prof = TuningProfile::from_toml_str("qkv_overlap = true\ncpu_npu_crossover_s = 20.0\n").unwrap();
        let c = prof.resolve(Precision::FastBf16);
        assert!(c.qkv_overlap, "profile sets qkv_overlap true over baked false");
        assert!(c.glu_fused, "unset key keeps baked default true");
    }

    #[test]
    fn detect_returns_xdna2() {
        assert_eq!(detect_hw_class(), "xdna2");
    }
}
