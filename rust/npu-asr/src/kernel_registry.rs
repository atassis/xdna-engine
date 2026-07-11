//! Kernel artifact registry -- the single source of truth for the on-disk kernel
//! artifact naming convention.
//!
//! Every compiled kernel ships as an xclbin + an instruction-stream file whose names
//! follow one convention under a build dir: `final_{stem}.xclbin` and `insts_{stem}.txt`,
//! where `stem` is a shape/mode descriptor (e.g. `512x800x3072_32x32x32_8c_silu`).
//!
//! Historically this convention was string-built inline via scattered
//! `format!("final_{stem}.xclbin")` call sites across the engine. Lifting it here gives a
//! data-driven compute graph one seam to query for `(stem) -> {xclbin, insts}` instead of
//! re-deriving the prefixes/extensions everywhere. Behavior-preserving: the produced paths
//! are byte-identical to the old inline `format!` (guarded by the parity test below).

use std::path::{Path, PathBuf};

/// The resolved on-disk artifacts for one kernel `stem` under a build dir.
pub struct KernelArtifacts {
    pub xclbin: PathBuf,
    pub insts: PathBuf,
}

/// The xclbin path for a kernel identified by `stem` under `dir`: `dir/final_{stem}.xclbin`.
pub fn xclbin_path(dir: &Path, stem: &str) -> PathBuf {
    dir.join(format!("final_{stem}.xclbin"))
}

/// The instruction-stream path for a kernel `stem` under `dir`: `dir/insts_{stem}.txt`.
pub fn insts_path(dir: &Path, stem: &str) -> PathBuf {
    dir.join(format!("insts_{stem}.txt"))
}

/// Resolve both artifacts for one `stem` under `dir`.
pub fn resolve(dir: &Path, stem: &str) -> KernelArtifacts {
    KernelArtifacts {
        xclbin: xclbin_path(dir, stem),
        insts: insts_path(dir, stem),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // String-level parity: the registry must reproduce the exact paths the pre-refactor
    // inline `format!("final_{stem}.xclbin")` / `format!("insts_{stem}.txt")` produced, for
    // the real stems across the current model set. Pins the naming convention so a future
    // edit to a prefix/extension breaks this test instead of silently missing every xclbin.
    #[test]
    fn registry_string_parity_matches_inline_format() {
        let dir = Path::new("/wa");
        // Representative real stems drawn from the routed call sites:
        //   ctx_decode GEMV, engines ffn (silu/bias), ctx_ln, conv_npu mstat band, mha_decode.
        let stems = [
            "512x800x3072_32x32x32_8c_silu",
            "512x1536x768_32x32x32_8c_bias",
            "ctxln_512x768",
            "mstat_512x768x1500_16x32x32_8c",
            "mha_decode_448",
        ];
        for stem in stems {
            // parity vs the historical inline expressions
            assert_eq!(xclbin_path(dir, stem), dir.join(format!("final_{stem}.xclbin")));
            assert_eq!(insts_path(dir, stem), dir.join(format!("insts_{stem}.txt")));
            let a = resolve(dir, stem);
            assert_eq!(a.xclbin, dir.join(format!("final_{stem}.xclbin")));
            assert_eq!(a.insts, dir.join(format!("insts_{stem}.txt")));
        }
        // absolute-literal pins (guard the convention constants themselves)
        assert_eq!(
            xclbin_path(dir, "512x768x3072_32x32x32_8c").to_str().unwrap(),
            "/wa/final_512x768x3072_32x32x32_8c.xclbin"
        );
        assert_eq!(
            insts_path(dir, "mha_decode_448").to_str().unwrap(),
            "/wa/insts_mha_decode_448.txt"
        );
    }
}
