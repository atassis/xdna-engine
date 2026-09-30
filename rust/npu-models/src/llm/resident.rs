//! A resident-forward `DecodeStep` backend: an artifact that keeps every layer's weights
//! MemTile-resident across a whole generation instead of streaming one layer's weights per
//! dispatch. This module holds every per-piece value builder, the K/V ring rule and the TDR
//! segment planner as CPU-only pure functions, plus [`ResidentForward`]: a real driver over
//! `ElfResident`/`FusedArena` (`open` loads the artifact, fills the scratch arena from the weight
//! store, boots the context and opens one control code per class; `run_piece` writes one piece's
//! values and dispatches).
//!
//! Land-first note (owner, 2026-09-29): the differential tests against the reference Python
//! functions the pure builders port, and the negative-test suite for every loader check, are
//! DEFERRED. `open`/`run_piece` compile and are unit-testable in isolation, but nothing in this
//! tree calls `open` against a live `Device` yet -- untested on hardware until the device lane
//! names a build whose generator emits arena-order arguments (`[input, output, scratch]`).

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::rc::Rc;

use npu_xrt::{Device, ElfResident, FusedArena};

use crate::api::EngineError;
use crate::llm::generator::{CacheState, DecodeStep};
use crate::llm::npu_decode::unpack_bf16_bytes;
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
/// `n` and `r` are both POSITION INDICES (0-based), not counts: `n` is the index of the last
/// position written, `r` is the index of the first position this resume will (re)write -- the same
/// value a caller usually holds as "how many prefix positions matched" (a count), since a count of
/// `k` trusted positions `[0, k)` and "first new index `k`" are numerically identical.
///
/// Derivation, in index terms: a resumed piece's first new row at index `r` reads the window
/// `[r-w+1, r-1]` (its own row is freshly written before it is read: write-first, one ring slot per
/// position, `slot = pos % capacity`). A sequentially-written ring holds position `p` correctly
/// only while `p > n - c_s` (a later position at the same slot, `p + c_s <= n`, overwrote it
/// otherwise); requiring that for the window's oldest row, `r - w + 1 > n - c_s`, rearranges to
/// `n - r <= c_s - w`.
///
/// Empirically checked, not just derived, against a host-side ring oracle (float64 stepwise
/// decode vs. a resumed run, teacher-forced): with `w = 1024` and `c_s == w` (a plain circular
/// cache, no slack), a ring primed to `n = 1099` (1100 positions written) resumed at `r = 1090`
/// (`n - r = 9 > 0`) measures a real divergence between the two runs -- that resume point IS wrong,
/// independent of this function. At the same `n`, `r = 1099` (`n - r = 0`) is the boundary this
/// function accepts.
///
/// Returns the resume point to actually use: `r` unchanged when safe, `0` (a full reset) when not.
/// `r <= n` is the caller's own invariant (a resume point is never ahead of what was primed) and is
/// asserted in debug builds only, matching this module's other pure functions.
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

/// The resident-forward `DecodeStep` backend (spec §2/§3): one hardware context (`boot` + a named
/// control code per class), one `FusedArena` holding the weight store and the K/V cache, driven by
/// [`ResidentForward::run_piece`] using the value builders above.
pub struct ResidentForward {
    pub artifact: ResidentArtifact,
    pub store: StoreManifest,
    arena: Rc<FusedArena>,
    /// One `ElfResident` per class code (`main:p{nt}`), each bound to `arena` -- §1.1's "each
    /// variant owns its OWN run and therefore its OWN ctrl scratchpad" (`npu_xrt::ElfResident`'s
    /// own doc). Keyed by `nt`, not the code string, since [`piece_nt`] is what a caller has in
    /// hand at dispatch time.
    classes: HashMap<usize, ElfResident>,
    /// The last K/V ring position written, for [`ring_safe_resume`]. `None` before the first
    /// piece of a generation.
    last_written: Option<usize>,
}

impl ResidentForward {
    /// Load and cross-check the artifact against its store manifest (§1.3, unchanged from S0), but
    /// do not touch a device. Fails loud on every S0 loader check (`ResidentArtifact::load`,
    /// `check_weights_against_store`).
    pub fn load(dir: &Path) -> Result<(ResidentArtifact, StoreManifest), EngineError> {
        let artifact = ResidentArtifact::load(dir)?;
        let store = StoreManifest::load(&artifact.store.path)?;
        artifact.check_weights_against_store(&store)?;
        Ok((artifact, store))
    }

    /// Load, fill the scratch arena from the store, and open the hardware context: `boot` (the
    /// only `load_pdi`, dispatched once here per §2 step 4), then `open_named` + `bind_resident`
    /// for every declared class code. Mirrors `NpuDecodeStep::build`'s shape (load -> arena ->
    /// upload weights -> zero caches -> `open_elf_resident` -> bind -> per-variant `open_named`).
    pub fn open(dev: &Rc<Device>, dir: &Path) -> Result<ResidentForward, EngineError> {
        let (artifact, store) = Self::load(dir)?;

        let arena = Rc::new(
            FusedArena::new(dev, artifact.input_size, artifact.output_size, artifact.scratch_size)
                .map_err(|e| EngineError::Load(format!("alloc fused arenas: {e}")))?,
        );

        // Every weight region the artifact declares, read from the store's blob file at the
        // region's own (blob, offset, length) and written to the artifact's own scratch offset --
        // two independent claims (§1.3), already cross-checked in `load`.
        for (name, region) in &artifact.weights {
            let entry = store.resolve(&region.manifest_key).ok_or_else(|| {
                EngineError::Load(format!("weight region `{name}`: no store entry for `{}` (should have failed at load)", region.manifest_key))
            })?;
            let blob_path = store.blob_path(entry);
            let bytes = read_blob_slice(&blob_path, entry.offset, entry.length)?;
            arena
                .write_at(npu_xrt::Arena::Scratch, region.scratch_off, &bytes)
                .map_err(|e| EngineError::Load(format!("write weight region `{name}`: {e}")))?;
        }
        arena.sync_to_device().map_err(|e| EngineError::Load(format!("sync weights to device: {e}")))?;

        let elf = artifact.read_elf_bytes()?;
        let boot = dev
            .open_elf_resident(&elf, Some(&format!("main:{}", artifact.boot)))
            .map_err(|e| EngineError::Load(format!("open_elf_resident (boot): {e}")))?;
        arena.bind_resident(&boot).map_err(|e| EngineError::Load(format!("bind boot arena BOs: {e}")))?;
        // "dispatched once after the context opens" (spec §1.1) -- the boot code carries only
        // `aiex.npu.load_pdi`, nothing to write beforehand.
        boot.dispatch().map_err(EngineError::Device)?;

        let mut classes = HashMap::new();
        for c in &artifact.classes {
            let r = boot
                .open_named(&format!("main:{}", c.code))
                .map_err(|e| EngineError::Load(format!("open class code {} (nt={}): {e}", c.code, c.nt)))?;
            arena.bind_resident(&r).map_err(|e| EngineError::Load(format!("bind class {} to the shared arena: {e}", c.code)))?;
            classes.insert(c.nt, r);
        }

        Ok(ResidentForward { artifact, store, arena, classes, last_written: None })
    }

    /// Run one piece of `p_len` positions starting at `s`, and read back whatever the artifact's
    /// `io["logits"]` names (spec §3.1's per-piece loop, minus the embedding gather and RoPE table
    /// writes -- those need an `EmbedTable`/`rope_row` wiring this stage does not yet do; `x_bytes`
    /// is the caller's already-embedded input row(s)).
    ///
    /// Writes, in order: `x_bytes` into `io["x_in"]`; the K/V ring split (`ring_write`) into
    /// `kv_slot_s`/`kv_len_s0`/`kv_len_s1` when the artifact declares them (a fixture or an early
    /// build may not yet); `nt` into its own scratchpad param when declared. Then `sync_input`,
    /// dispatch the class code [`piece_nt`] selects, `sync_from_device`, and return the unpacked
    /// bf16 logits.
    pub fn run_piece(&mut self, s: usize, p_len: usize, x_bytes: &[u8]) -> Result<Vec<f32>, EngineError> {
        let nt = piece_nt(p_len, self.artifact.dims.row_block);
        let res = self.classes.get(&nt).ok_or_else(|| {
            EngineError::Unsupported(format!("no class code carries nt={nt} (piece of {p_len} rows)"))
        })?;

        let x_in = self.artifact.io.get("x_in").ok_or_else(|| EngineError::Load("artifact declares no `io.x_in`".to_string()))?;
        self.arena.write_at(x_in.arena, x_in.off, x_bytes).map_err(|e| EngineError::Device(format!("write x_in: {e}")))?;

        let sliding_capacity = self
            .artifact
            .derived
            .get("sliding_capacity")
            .ok_or_else(|| EngineError::Load("artifact declares no `derived.sliding_capacity`".to_string()))?
            .value;
        let rw = ring_write(s, p_len, sliding_capacity);
        write_param_if_present(res, &self.artifact.scratchpad_params, "kv_slot_s", rw.slot as u32)?;
        write_param_if_present(res, &self.artifact.scratchpad_params, "kv_len_s0", rw.len0 as u32)?;
        write_param_if_present(res, &self.artifact.scratchpad_params, "kv_len_s1", rw.len1 as u32)?;
        write_param_if_present(res, &self.artifact.scratchpad_params, "nt", nt as u32)?;

        self.arena.sync_input().map_err(|e| EngineError::Device(format!("sync input: {e}")))?;
        res.dispatch().map_err(EngineError::Device)?;
        self.arena.sync_from_device().map_err(|e| EngineError::Device(format!("sync output: {e}")))?;

        self.last_written = Some(s + p_len - 1);

        let logits_loc = self.artifact.io.get("logits").ok_or_else(|| EngineError::Load("artifact declares no `io.logits`".to_string()))?;
        let mut bytes = vec![0u8; logits_loc.len];
        self.arena
            .read_at(logits_loc.arena, logits_loc.off, &mut bytes)
            .map_err(|e| EngineError::Device(format!("read logits: {e}")))?;
        Ok(unpack_bf16_bytes(&bytes))
    }
}

/// `store/blobs/<blob>.bin[offset..offset+length]` -- reads the whole file (blobs are one matrix
/// each, not multi-GB) and slices rather than seeking, since S0/S1 have no mmap loader yet (tracked
/// as a follow-up: `StoreManifest`'s own doc).
fn read_blob_slice(path: &Path, offset: usize, length: usize) -> Result<Vec<u8>, EngineError> {
    let bytes = fs::read(path).map_err(|e| EngineError::Load(format!("read blob {}: {e}", path.display())))?;
    bytes
        .get(offset..offset + length)
        .map(|s| s.to_vec())
        .ok_or_else(|| EngineError::Load(format!("blob {} is {} bytes, region wants [{offset}, {})", path.display(), bytes.len(), offset + length)))
}

/// Write `value` (little-endian u32) to `name`'s scratchpad slot if the artifact declares one,
/// no-op otherwise. Scratchpad params are OPTIONAL per-artifact (an early or seeded build may carry
/// only a subset), unlike the shipped decode's fixed set -- see `ResidentArtifact`'s own doc.
fn write_param_if_present(
    res: &ElfResident, params: &HashMap<String, crate::llm::resident_artifact::ScratchpadParam>, name: &str,
    value: u32,
) -> Result<(), EngineError> {
    if let Some(p) = params.get(name) {
        res.write_scratchpad(p.byte_offset, &value.to_le_bytes())
            .map_err(|e| EngineError::Device(format!("write {name} scratchpad: {e}")))?;
    }
    Ok(())
}

impl DecodeStep for ResidentForward {
    fn step(&mut self, _token: u32, _pos: usize) -> Result<Vec<f32>, EngineError> {
        // `run_piece` is the real driver; `step` needs an embedding gather and the RoPE/attention-
        // width writes §3.1 also lists, which land with the embed table wiring in a later stage.
        Err(EngineError::Unsupported("ResidentForward::step: embedding + full per-piece writes not wired yet (use run_piece)".into()))
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
    fn ring_safe_resume_reproduces_the_oracles_proven_negative_control() {
        // A host-side ring oracle: W=1024, ring primed to n=1099 (1100 positions), resume at
        // r=1090 is proven wrong end to end (measured divergence). c_s == w here (a plain circular
        // cache, no slack), so this function must refuse it too.
        assert_eq!(ring_safe_resume(1090, 1099, 1024, 1024), 0);
        // The boundary the oracle's own derivation implies (n - r == 0) is exactly where this
        // function switches back to accepting the resume.
        assert_eq!(ring_safe_resume(1099, 1099, 1024, 1024), 1099);
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
