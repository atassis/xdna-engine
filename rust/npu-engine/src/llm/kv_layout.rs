//! Rust mirror of `iron.common.kv_layout`'s `KVLayout.kv_off` -- the Python checkout owns the
//! definition, this module owns agreeing with it. See that module's docstring for the full
//! addressing model (`[S/T blocks, Hkv heads, T positions, HD dims]`, block-major). `kv_off` is
//! the RUNTIME half the host writes per token; `head_base`/`head_runs` are the build-time half,
//! here because a host that READS the cache back (the debug probes) needs what a host that only
//! drives it does not.
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

/// The BUILD-TIME per-head term `kv_off` deliberately omits: where head `head`'s positions start
/// inside each block. A host that only drives the cache never needs it; a host that READS the
/// cache back -- the debug probes, which compare a slab against a CPU golden -- does.
pub fn head_base(head: usize, kv_block: usize, head_dim: usize) -> usize {
    head * kv_block * head_dim
}

/// The contiguous runs making up `cache[head, 0..positions, :]`, as `(element offset, positions)`
/// in position order.
///
/// Blocked, a head's positions are contiguous only INSIDE a block -- across blocks the other
/// heads sit in between -- so a reader takes one run per block where the flat layout gave it one
/// slice. At `kv_block >= positions` that is a single run at `head * kv_block * head_dim`, which
/// is exactly the `head * max_seq * head_dim` the probes hand-wrote before blocking.
pub fn head_runs(
    head: usize, positions: usize, kv_block: usize, head_dim: usize, kv_heads: usize,
) -> Vec<(usize, usize)> {
    let mut runs = Vec::new();
    let mut pos = 0;
    while pos < positions {
        let take = (kv_block - pos % kv_block).min(positions - pos);
        runs.push((kv_off(pos, kv_block, head_dim, kv_heads) + head_base(head, kv_block, head_dim),
                   take));
        pos += take;
    }
    runs
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

    #[test]
    fn head_runs_is_one_flat_slice_when_the_window_is_one_block() {
        // The pre-blocking read the probes hand-wrote: head h at h * max_seq * head_dim, one run.
        for head in 0..8 {
            assert_eq!(head_runs(head, 256, 2048, 128, 8), vec![(head * 2048 * 128, 256)]);
        }
    }

    #[test]
    fn head_runs_covers_every_position_exactly_where_kv_off_puts_it() {
        // Element-by-element against the formula the DEVICE writes through, for a slab that spans
        // several blocks -- the case where a single flat slice reads seven other heads' bytes.
        let (kv_block, head_dim, kv_heads, n) = (128usize, 128usize, 8usize, 300usize);
        for head in 0..kv_heads {
            let mut pos = 0;
            for (off, take) in head_runs(head, n, kv_block, head_dim, kv_heads) {
                for i in 0..take {
                    assert_eq!(off + i * head_dim,
                               kv_off(pos, kv_block, head_dim, kv_heads)
                                   + head_base(head, kv_block, head_dim),
                               "head={head} pos={pos}");
                    pos += 1;
                }
            }
            assert_eq!(pos, n);
        }
    }

    #[test]
    fn head_runs_splits_at_block_boundaries_not_at_the_slab_start() {
        // 300 positions over 128-position blocks: 128 + 128 + 44, not three equal runs.
        let runs = head_runs(0, 300, 128, 128, 8);
        assert_eq!(runs.iter().map(|(_, n)| *n).collect::<Vec<_>>(), vec![128, 128, 44]);
    }
}
