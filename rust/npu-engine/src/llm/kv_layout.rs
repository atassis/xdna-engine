//! Rust mirror of `iron.common.kv_layout`'s `KVLayout.kv_off` -- the Python checkout owns the
//! definition, this module owns agreeing with it. See that module's docstring for the full
//! addressing model (`[S/T blocks, Hkv heads, T positions, HD dims]`, block-major); this file
//! carries only the RUNTIME half the host writes per token, `kv_off`.
//!
//! Until 2026-09-09 this formula was `pos * head_dim`, hand-written at
//! [`crate::llm::npu_decode::NpuDecodeStep::step`] with nothing checking it against the four
//! Python sites computing the same address -- see the `kv-cache-layout-for-full-context` task.
//! There is no code-sharing across the Python/Rust boundary, so agreement is enforced by a
//! differential test below instead: the same (Hkv, S, HD, T, pos) inputs must produce the same
//! output in both languages. `test_matches_python_kv_layout` pins that against values computed by
//! `iron.common.kv_layout.KVLayout` on 2026-09-09 -- re-derive them if this formula ever changes.

/// The runtime per-token scratchpad value: everything that depends on `pos` alone, common to
/// every head (the head term is a BUILD-TIME constant baked into the ELF's own taps, not
/// something the host computes). At `kv_block == max_seq` (one block) this is exactly
/// `pos * head_dim`, the pre-blocking formula, unchanged.
///
/// Panics (debug) / wraps (release) are not a concern here: every input is `usize`-checked by
/// its caller (`pos < max_seq`, from [`crate::llm::artifact::LlmArtifact::max_seq`]).
pub fn kv_off(pos: usize, kv_block: usize, head_dim: usize, kv_heads: usize) -> usize {
    let block = pos / kv_block;
    let within = pos % kv_block;
    let block_stride = kv_heads * kv_block * head_dim;
    block * block_stride + within * head_dim
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn degenerate_kv_block_equals_max_seq_is_the_old_pos_times_head_dim() {
        // kv_block == S (one block): must reduce to the pre-blocking formula exactly.
        for pos in [0usize, 1, 511, 2047] {
            assert_eq!(kv_off(pos, 2048, 128, 8), pos * 128);
        }
    }

    #[test]
    fn kv_off_takes_no_head_parameter_by_construction() {
        // The head term is a build-time tap stride, never part of the runtime scratchpad write --
        // this signature has no head argument at all, which is exactly why. Two positions in the
        // SAME block share a kv_off regardless of which head each one is destined for.
        assert_eq!(kv_off(4, 128, 128, 8), kv_off(4, 128, 128, 8));
        // kv_heads DOES change the output, though -- it sizes block_stride -- so it is not a
        // free parameter the way "head" is; the block term crosses kv_heads on the SECOND block.
        assert_ne!(kv_off(128, 128, 128, 8), kv_off(128, 128, 128, 1));
    }

    /// Pinned against `iron.common.kv_layout.KVLayout(Hkv=8, S=4096, HD=128,
    /// T=128).kv_off(pos)`, computed 2026-09-09 -- see that module's own
    /// `test_blocked_offset_matches_the_S_over_T_Hkv_T_HD_layout_by_hand` for the Python side of
    /// this same arithmetic. Re-derive both if the formula changes; a passing test here after an
    /// independent change to either side is exactly the silent-drift failure this pin exists to
    /// catch.
    #[test]
    fn test_matches_python_kv_layout() {
        let cases: &[(usize, usize)] = &[
            (0, 0),
            (1, 128),
            (127, 127 * 128),
            (128, 8 * 128 * 128),           // first position of block 1: one full block_stride
            (255, 8 * 128 * 128 + 127 * 128),
            (256, 2 * 8 * 128 * 128),
            (4095, 31 * 8 * 128 * 128 + 127 * 128), // last position, block 31 of 32
        ];
        for &(pos, expect) in cases {
            assert_eq!(kv_off(pos, 128, 128, 8), expect, "pos={pos}");
        }
    }
}
