//! Stage 0 of a resident-forward `DecodeStep` backend: an artifact that keeps every layer's
//! weights MemTile-resident across a whole generation instead of streaming one layer's weights
//! per dispatch. This module holds the `ResidentForward` skeleton plus every per-piece value
//! builder, the K/V ring rule and the TDR segment planner, as CPU-only pure functions. No device
//! access -- opening an [`npu_xrt::Arena`]/[`crate::llm::npu_decode::ElfResident`], the piece
//! dispatch loop and the canary check land in a later stage.
//!
//! Land-first note (owner, 2026-09-29): the differential tests against the reference Python
//! functions these port, and the negative-test suite for every loader check, are DEFERRED. What is
//! here compiles and the loader reads the real store manifest and a seeded `meta.json` fixture
//! (`resident_artifact.rs` tests).

use crate::api::EngineError;
use crate::llm::generator::{CacheState, DecodeStep};
use crate::llm::resident_artifact::{ResidentArtifact, StoreManifest};

/// A piece's key-window widths for one 16-row pass of the resident attention: `[lo, hi)` relative
/// to the pass's own rows, one pair per row of the pass's 32-row (2 query heads x 16 rows) layout.
/// Port of `prototypes/resident-forward/rat_run.py::widths`, one pass at a time (the Python builds
/// all `nt` passes in one call; here the caller loops over `tb` itself, since S0 has no piece loop
/// yet to drive it).
///
/// `n_past` is the position of the piece's first row; `win` is the attention window (`None` for
/// global, unbounded); `p_in_piece` is this pass's row-within-piece count `P - 16*tb` clamped so
/// callers can pass the piece's overall `P` unchanged across passes. Rows at or past the piece's
/// real length (`row >= p_in_piece`) are pad rows and get `[0, 1)`, matching the Python's "rows at
/// or past P see key 0 only".
pub fn attention_widths(n_past: usize, win: Option<usize>, p_in_piece: i64, pass_row: usize) -> (i64, i64) {
    let p = n_past + pass_row;
    if (pass_row as i64) >= p_in_piece {
        return (1, 0);
    }
    let first = match win {
        Some(w) => (n_past.saturating_sub(w.saturating_sub(1))) / 64 * 64,
        None => 0,
    };
    let hi = p as i64 - first as i64 + 1;
    let lo = match win {
        Some(w) => (p as i64 - (w as i64 - 1) - first as i64).max(0),
        None => 0,
    };
    (hi, lo)
}

/// Ring geometry for one piece's K/V write, split at the wrap point (spec §3.1/§4.2). A piece
/// writes `p_len` positions starting at `s`; `slot` is the ring offset of the first row, `len0` is
/// how many rows land before the ring wraps, and `len1` is the rest (0 unless the piece itself
/// crosses the wrap).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RingWrite {
    pub slot: usize,
    pub len0: usize,
    pub len1: usize,
}

/// `kv_slot_s`/`kv_len_s0`/`kv_len_s1` for a piece of `p_len` positions starting at `s`, against a
/// ring of `capacity` slots (spec table, row 3: "ring slot of the piece's first row" /
/// "rows before and after the ring wrap").
pub fn ring_write(s: usize, p_len: usize, capacity: usize) -> RingWrite {
    let slot = s % capacity;
    let len0 = p_len.min(capacity - slot);
    let len1 = p_len - len0;
    RingWrite { slot, len0, len1 }
}

/// The read-side counterpart: the ring offset of the oldest position a piece's rows may still
/// attend to (spec table, row 5: `att_s_first`). `w` is the sliding window; `s` is the piece's
/// first row's position.
pub fn ring_read_first(s: usize, w: usize) -> usize {
    s.saturating_sub(w.saturating_sub(1)) / 64 * 64
}

/// The resume-safety rule for reusing a cached prefix against a circular sliding-window K/V
/// cache, generalised from a fixed-window cache's single check to this backend's `c_s - w` slack:
/// a resume at `r` after the ring last held position `n` is sound only if the ring still holds
/// every slot `r` still needs, i.e. `n - r <= c_s - w`. Shared with a plain circular cache (no
/// slack) by passing `c_s - w == 0`, which is the shape a resumed request must satisfy on any
/// backend whose ring capacity equals its window.
///
/// Returns the resume point to actually use: `r` unchanged when safe, `0` (a full reset) when not.
/// `n` is the last position written to the ring; `r <= n` is the caller's own invariant (a resume
/// point is never ahead of what was primed) and is asserted in debug builds only, matching this
/// module's other pure functions.
pub fn ring_safe_resume(r: usize, n: usize, c_s: usize, w: usize) -> usize {
    debug_assert!(r <= n, "resume point {r} must not be ahead of the last written position {n}");
    let slack = c_s.saturating_sub(w);
    if n.saturating_sub(r) <= slack {
        r
    } else {
        0
    }
}

/// The TDR segment planner (spec §3.3): the largest number of layers `L_d` that fits half the TDR
/// budget at this piece's row count, from the meta.json-measured per-layer time fit
/// `t_layer(p) = a_ms + b_ms_per_row * p`. Half the budget, not the whole of it, is deliberate
/// headroom against the fit's own error before the runtime re-check (K042) refuses the dispatch.
///
/// Panics if `a_ms + b_ms_per_row * p_len <= 0.0` (a non-positive fit is not a time, and dividing
/// by it would silently produce a nonsensical layer count rather than a caught bug).
pub fn tdr_segment_layers(a_ms: f64, b_ms_per_row: f64, p_len: usize, tdr_budget_ms: f64) -> usize {
    let t_layer = a_ms + b_ms_per_row * p_len as f64;
    assert!(t_layer > 0.0, "non-positive per-layer time fit: a_ms={a_ms} b_ms_per_row={b_ms_per_row} p_len={p_len}");
    ((0.5 * tdr_budget_ms) / t_layer).floor() as usize
}

/// Predicted time for one segment of `layers` layers at `p_len` rows, against the 1000 ms per-
/// segment refusal rule (spec §3.3 / §5, "predicted segment time over 1000 ms"). Kept separate
/// from [`tdr_segment_layers`] (the BUILD-time planner) because this is the RUN-time re-check: it
/// takes the segment size the ELF was actually compiled for, not a freshly derived one.
pub fn predicted_segment_ms(a_ms: f64, b_ms_per_row: f64, p_len: usize, layers: usize) -> f64 {
    (a_ms + b_ms_per_row * p_len as f64) * layers as f64
}

/// `nt` for a piece of `p_len` rows (spec table, last row): `ceil(p_len / row_block)`, the count
/// that selects which class code (`p{nt}`) carries the piece.
pub fn piece_nt(p_len: usize, row_block: usize) -> usize {
    p_len.div_ceil(row_block)
}

/// Skeleton for the resident-forward `DecodeStep` backend (spec §2/§3). S0 holds the loaded,
/// checked artifact and store manifest; it does not yet open a device context or dispatch anything
/// -- every `DecodeStep` method below returns `Unsupported` until S1 wires `FusedArena`/
/// `ElfResident` through the value builders above.
pub struct ResidentForward {
    pub artifact: ResidentArtifact,
    pub store: StoreManifest,
    /// The last K/V ring position written, for [`ring_safe_resume`]. `None` before the first
    /// piece of a generation.
    last_written: Option<usize>,
}

impl ResidentForward {
    /// Load and cross-check the artifact against its store manifest (§1.3), but do not touch a
    /// device. Fails loud on every S0 loader check (`ResidentArtifact::load`,
    /// `check_weights_against_store`).
    pub fn load(dir: &std::path::Path) -> Result<ResidentForward, EngineError> {
        let artifact = ResidentArtifact::load(dir)?;
        let store = StoreManifest::load(&artifact.store.path)?;
        artifact.check_weights_against_store(&store)?;
        Ok(ResidentForward { artifact, store, last_written: None })
    }
}

impl DecodeStep for ResidentForward {
    fn step(&mut self, _token: u32, _pos: usize) -> Result<Vec<f32>, EngineError> {
        Err(EngineError::Unsupported("ResidentForward: device dispatch is not built yet (S0 is host-side only)".into()))
    }

    fn reset(&mut self) -> Result<CacheState, EngineError> {
        self.last_written = None;
        Ok(CacheState::Retained)
    }

    fn max_context(&self) -> Option<usize> {
        // The widest capacity this backend can address at all -- the global-layer cache, sized by
        // `derived.global_capacity`. The narrower sliding-ring window is what `ring_safe_resume`
        // (not this bound) protects: total-length bounds and resume-point bounds are different
        // checks, and a backend that only enforces the former can still resume into stale slots.
        self.artifact.derived.get("global_capacity").map(|d| d.value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attention_widths_pad_row_is_key_zero_only() {
        assert_eq!(attention_widths(100, Some(1023), 21, 21), (1, 0));
    }

    #[test]
    fn attention_widths_real_row_matches_the_hand_worked_case() {
        // n_past=100, win=1023: first = max(0, 100-1022)//64*64 = 0. Row 0: p=100, hi=101, lo=0.
        assert_eq!(attention_widths(100, Some(1023), 21, 0), (101, 0));
    }

    #[test]
    fn ring_write_splits_at_the_wrap() {
        let rw = ring_write(1150, 16, 1152);
        assert_eq!(rw, RingWrite { slot: 1150, len0: 2, len1: 14 });
    }

    #[test]
    fn ring_write_no_wrap_when_it_fits() {
        let rw = ring_write(0, 16, 1152);
        assert_eq!(rw, RingWrite { slot: 0, len0: 16, len1: 0 });
    }

    #[test]
    fn ring_safe_resume_allows_a_resume_inside_the_slack() {
        // c_s=1152, w=1024, slack=128: n-r=100 <= 128, resume kept.
        assert_eq!(ring_safe_resume(900, 1000, 1152, 1024), 900);
    }

    #[test]
    fn ring_safe_resume_refuses_past_the_slack() {
        // n-r=500 > 128: overwritten slots, forced reset to 0.
        assert_eq!(ring_safe_resume(500, 1000, 1152, 1024), 0);
    }

    #[test]
    fn ring_safe_resume_has_zero_slack_when_c_s_equals_w() {
        // A plain circular cache with no slack is c_s == w, so this formula's slack is 0 and only
        // an exact, full-ring resume is safe. A related, looser boundary condition
        // (`reused >= pos_old - 1`) is used elsewhere for the same cache shape; reconciling the
        // two is deferred -- this test pins only this module's `N - r <= C_s - W` formula.
        assert_eq!(ring_safe_resume(500, 1025, 1024, 1024), 0);
        assert_eq!(ring_safe_resume(1025, 1025, 1024, 1024), 1025);
    }

    #[test]
    fn tdr_segment_layers_matches_a_hand_worked_fit() {
        // t_layer = 0.5 + 0.01*112 = 1.62 ms; 0.5*1000/1.62 = 308.6 -> 308.
        assert_eq!(tdr_segment_layers(0.5, 0.01, 112, 1000.0), 308);
    }

    #[test]
    fn piece_nt_rounds_up() {
        assert_eq!(piece_nt(1, 16), 1);
        assert_eq!(piece_nt(16, 16), 1);
        assert_eq!(piece_nt(17, 16), 2);
        assert_eq!(piece_nt(112, 16), 7);
    }
}
