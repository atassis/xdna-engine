//! Generic streamed-op bricks: window/chunk one causal op (snake / conv / conv_transpose) across
//! an arbitrarily long `[C, L]` stream, over one or more pre-built [`S2Design`]s. Ports
//! `designs/codec_block/window_driver.py`'s `snake`/`conv`/`conv_transpose` functions
//! 1:1 -- see that file's docstring for the windowing contract this implements: OP-MAJOR (one op
//! runs to completion over the whole segment, not pipelined per window), overlap-save causal
//! windowing (a window carries `ctx` samples of left context, output crops the first `ctx`
//! samples of each window's raw result), and driver-supplied lead-in rather than fabricated zero
//! padding (a caller that wants the true unblocked stream's answer must pass unwindowed context).
//!
//! Every design's streamed operand carries the WEIGHTS (one row per output channel); the
//! resident operand carries the activation window. A dispatch re-uploads the same weight values
//! on every window -- wasteful, but it is `bricklib._build_streamed`'s fixed ABI, and
//! window_driver.py pays the same cost.

use ndarray::{s, Array1, Array2, Array3};

use crate::{S2Design, S2Error, S2OpParams};

fn kind_err(want: &str, meta: &crate::S2Meta) -> S2Error {
    S2Error::Shape(format!(
        "design `{}` (op={}): op_params.kind is {:?}, expected {want}",
        meta.symbol, meta.op, meta.op_params.as_ref().map(|_| "present-but-wrong-variant").unwrap_or("absent")
    ))
}

fn shape_err(what: &str, symbol: &str, got: usize, want: usize) -> S2Error {
    S2Error::Shape(format!("design `{symbol}`: {what} is {got}, op_params says {want}"))
}

/// `(offset, len)` for every window a windowed op walks over a segment of length `m`, stepping by
/// `step` -- window_driver.py's `for o in range(0, m, step): n = min(step, m - o)`, ported as a
/// standalone, device-free function so the loop bounds themselves are unit-testable (see
/// `tests` below). Shared by snake (`step == t`, `m == l`), conv and conv_transpose
/// (`step == t - ctx`, `m == l - ctx`).
fn window_offsets(m: usize, step: usize) -> Vec<(usize, usize)> {
    let mut v = Vec::new();
    let mut o = 0usize;
    while o < m {
        v.push((o, step.min(m - o)));
        o += step;
    }
    v
}

#[cfg(test)]
mod tests {
    use super::window_offsets;

    /// Self-authored: no python-rail fixture exists for this (it never runs a real design), so
    /// this checks the arithmetic's OWN documented invariants, not an external oracle.
    #[test]
    fn window_offsets_covers_the_segment_exactly_once() {
        for &(m, step) in &[(0, 64), (1, 64), (64, 64), (65, 64), (128, 64), (100, 58), (58, 58), (57, 58)] {
            let offs = window_offsets(m, step);
            let mut covered = 0;
            for (i, &(o, n)) in offs.iter().enumerate() {
                assert_eq!(o, covered, "m={m} step={step}: window {i} starts at {o}, expected {covered}");
                assert!(n > 0 && n <= step, "m={m} step={step}: window {i} len {n} out of (0, {step}]");
                covered += n;
            }
            assert_eq!(covered, m, "m={m} step={step}: windows cover {covered}, segment is {m}");
        }
    }

    /// window_driver.py's own loop is `for o in range(0, m, step)`. Cross-checked here against
    /// Python's real `range()` builtin (a `python3 -c` subprocess, stdlib only -- no project code,
    /// no toolchain) rather than a second Rust reimplementation of the same arithmetic, which
    /// would just be this test agreeing with itself. Skips if python3 isn't on PATH.
    #[test]
    fn window_offsets_matches_pythons_range_builtin() {
        let Ok(out) = std::process::Command::new("python3")
            .arg("-c")
            .arg(
                "for m, step in [(100,58),(116,58),(30,58),(0,58),(1,64),(64,64)]:\n\
                 \tprint(','.join(str(o) for o in range(0, m, step)))",
            )
            .output()
        else {
            eprintln!("skip: python3 not on PATH");
            return;
        };
        assert!(out.status.success());
        let text = String::from_utf8(out.stdout).unwrap();
        for (line, &(m, step)) in text.lines().zip(&[(100, 58), (116, 58), (30, 58), (0, 58), (1, 64), (64, 64)]) {
            let want: Vec<usize> = if line.is_empty() { vec![] } else { line.split(',').map(|s| s.parse().unwrap()).collect() };
            let got: Vec<usize> = window_offsets(m, step).into_iter().map(|(o, _)| o).collect();
            assert_eq!(got, want, "m={m} step={step}");
        }
    }
}

// ---- snake --------------------------------------------------------------------------------

/// Pointwise `[C, L] -> [C, L]`: no context, no chunking. `alpha` rides in the tile as an extra
/// per-channel column (the pure-buffer shim ABI carries no scalars).
pub struct SnakeOp {
    design: S2Design,
    c: usize,
    t: usize,
}

impl SnakeOp {
    pub fn new(design: S2Design) -> crate::Result<Self> {
        let Some(S2OpParams::Snake { c, t, .. }) = &design.meta.op_params else {
            return Err(kind_err("snake", &design.meta));
        };
        let (c, t) = (*c, *t);
        if design.meta.n_tiles != c {
            return Err(shape_err("n_tiles", &design.meta.symbol, design.meta.n_tiles, c));
        }
        if design.meta.in_tile != t + 1 {
            return Err(shape_err("in_tile", &design.meta.symbol, design.meta.in_tile, t + 1));
        }
        if design.meta.out_numel != t {
            return Err(shape_err("out_numel", &design.meta.symbol, design.meta.out_numel, t));
        }
        Ok(SnakeOp { design, c, t })
    }

    /// `x`: `[C, L]`. `alpha`: `[C]` (one value per channel). -> `[C, L]`.
    pub fn run(&self, x: &Array2<f32>, alpha: &Array1<f32>) -> crate::Result<Array2<f32>> {
        let (c, l) = x.dim();
        if c != self.c {
            return Err(S2Error::Shape(format!("snake: x has {c} channels, design wants {}", self.c)));
        }
        if alpha.len() != c {
            return Err(S2Error::Shape(format!("snake: alpha has {} elements, x has {c} channels", alpha.len())));
        }
        let t = self.t;
        let mut out = Array2::<f32>::zeros((c, l));
        for (o, n) in window_offsets(l, t) {
            let mut tiles = Array2::<f32>::zeros((c, t + 1));
            tiles.slice_mut(s![.., 0..n]).assign(&x.slice(s![.., o..o + n]));
            tiles.column_mut(t).assign(alpha);
            let got = self.design.dispatch(tiles.as_slice().expect("standard layout"), None)?;
            let got = Array2::from_shape_vec((c, t), got).map_err(|e| S2Error::Shape(e.to_string()))?;
            out.slice_mut(s![.., o..o + n]).assign(&got.slice(s![.., 0..n]));
        }
        Ok(out)
    }
}

// ---- conv -----------------------------------------------------------------------------------

/// Causal dilated conv `[c_in_total, L] -> [c_out, L - ctx]`, `ctx = (k-1)*dilation`, optionally
/// chunked over input channels (partials accumulated on the host) and optionally carrying a
/// residual add on chunk 0 only. Ports `window_driver.conv`/`_conv_chunk`.
pub struct ConvOp {
    /// Used for chunk 0, and for every later chunk when `rest` is `None` (the common case: an op
    /// that never carries `add` reuses ONE compiled design across every chunk, because its
    /// tag/symbol never varies with chunk index -- see window_driver.py's `_get_design` docstring).
    first: S2Design,
    /// Only `Some` for a chunked op that carries `add` on chunk 0: a later chunk's tile has no add
    /// slot, which is a different symbol/design (see [`crate::S2Artifacts::designs_by_op`]).
    rest: Option<S2Design>,
    k: usize,
    ctx: usize,
    c_in_chunk: usize,
    c_in_total: usize,
    c_out: usize,
    has_add: bool,
    t: usize,
    step: usize,
}

fn conv_params(design: &S2Design) -> crate::Result<(usize, usize, usize, usize, usize, usize, bool, usize, usize)> {
    let Some(S2OpParams::Conv { k, dilation, ctx, c_in, c_in_total, c_out, has_add, t, step, .. }) =
        &design.meta.op_params
    else {
        return Err(kind_err("conv", &design.meta));
    };
    Ok((*k, *dilation, *ctx, *c_in, *c_in_total, *c_out, *has_add, *t, *step))
}

impl ConvOp {
    /// `rest` (when given) must be the sibling chunk-1.. design for the SAME op (same
    /// k/dilation/ctx/c_in/c_in_total/c_out/t/step, `has_add=false` where `first` has
    /// `has_add=true`). Which of two designs sharing an `op` role is `first` is decided by
    /// `has_add`, not by name.
    pub fn new(first: S2Design, rest: Option<S2Design>) -> crate::Result<Self> {
        let (k, dilation, ctx, c_in_chunk, c_in_total, c_out, has_add, t, step) = conv_params(&first)?;
        if first.meta.n_tiles != c_out {
            return Err(shape_err("n_tiles", &first.meta.symbol, first.meta.n_tiles, c_out));
        }
        let want_in_tile = c_in_chunk * k + 1 + if has_add { t } else { 0 };
        if first.meta.in_tile != want_in_tile {
            return Err(shape_err("in_tile", &first.meta.symbol, first.meta.in_tile, want_in_tile));
        }
        if first.meta.resident_len != c_in_chunk * t {
            return Err(shape_err("resident_len", &first.meta.symbol, first.meta.resident_len, c_in_chunk * t));
        }
        if let Some(r) = &rest {
            let (rk, rd, rctx, rcin, rcintot, rcout, rhas_add, rt, rstep) = conv_params(r)?;
            if (rk, rd, rctx, rcin, rcintot, rcout, rt, rstep) != (k, dilation, ctx, c_in_chunk, c_in_total, c_out, t, step)
            {
                return Err(S2Error::Shape(format!(
                    "ConvOp::new: `rest` design `{}` shape params don't match `first` `{}`",
                    r.meta.symbol, first.meta.symbol
                )));
            }
            if !has_add || rhas_add {
                return Err(S2Error::Shape(format!(
                    "ConvOp::new: expected first.has_add=true rest.has_add=false, got {has_add}/{rhas_add}"
                )));
            }
        }
        Ok(ConvOp { first, rest, k, ctx, c_in_chunk, c_in_total, c_out, has_add, t, step })
    }

    pub fn ctx(&self) -> usize {
        self.ctx
    }

    fn design_for(&self, chunk_idx: usize) -> &S2Design {
        if chunk_idx == 0 {
            &self.first
        } else {
            self.rest.as_ref().unwrap_or(&self.first)
        }
    }

    /// `x`: `[c_in_total, L]`. `w`: `[c_out, c_in_total, k]`. `bias`: `[c_out]`. `add`: `Some([c_out,
    /// M])` iff this op carries a residual (`has_add`), applied only at chunk 0's positions and
    /// only once per output position -- Python's convention, ported unchanged. -> `[c_out, M]`,
    /// `M = L - ctx`.
    pub fn run(
        &self, x: &Array2<f32>, w: &Array3<f32>, bias: &Array1<f32>, add: Option<&Array2<f32>>,
    ) -> crate::Result<Array2<f32>> {
        let (c_in_total, l) = x.dim();
        if c_in_total != self.c_in_total {
            return Err(S2Error::Shape(format!("conv: x has {c_in_total} input channels, op wants {}", self.c_in_total)));
        }
        if w.dim() != (self.c_out, self.c_in_total, self.k) {
            return Err(S2Error::Shape(format!("conv: w.dim()={:?}, want {:?}", w.dim(), (self.c_out, self.c_in_total, self.k))));
        }
        if bias.len() != self.c_out {
            return Err(S2Error::Shape(format!("conv: bias has {} elements, want {}", bias.len(), self.c_out)));
        }
        if add.is_some() != self.has_add {
            return Err(S2Error::Shape(format!("conv: add.is_some()={}, op has_add={}", add.is_some(), self.has_add)));
        }
        let m = l.checked_sub(self.ctx).filter(|&m| m > 0).ok_or_else(|| {
            S2Error::Shape(format!("conv: segment {l} shorter than context {}", self.ctx))
        })?;

        let mut out = Array2::<f32>::zeros((self.c_out, m));
        let zero_bias = Array1::<f32>::zeros(self.c_out);
        let mut c0 = 0usize;
        let mut chunk_idx = 0usize;
        while c0 < self.c_in_total {
            let cs = self.c_in_chunk.min(self.c_in_total - c0);
            let is_first = chunk_idx == 0;
            let bias_eff = if is_first { bias } else { &zero_bias };
            let add_eff = if is_first { add } else { None };
            let design = self.design_for(chunk_idx);
            let partial = self.run_chunk(design, x, w, bias_eff, add_eff, c0, cs, m)?;
            out += &partial;
            c0 += self.c_in_chunk;
            chunk_idx += 1;
        }
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_chunk(
        &self, design: &S2Design, x: &Array2<f32>, w: &Array3<f32>, bias: &Array1<f32>,
        add: Option<&Array2<f32>>, c0: usize, cs: usize, m: usize,
    ) -> crate::Result<Array2<f32>> {
        let (k, ctx, t, step, c_in_chunk, c_out) = (self.k, self.ctx, self.t, self.step, self.c_in_chunk, self.c_out);
        let l = x.dim().1;

        let mut xc = Array2::<f32>::zeros((c_in_chunk, l));
        xc.slice_mut(s![0..cs, ..]).assign(&x.slice(s![c0..c0 + cs, ..]));
        let mut wc = Array3::<f32>::zeros((c_out, c_in_chunk, k));
        wc.slice_mut(s![.., 0..cs, ..]).assign(&w.slice(s![.., c0..c0 + cs, ..]));
        let wc_flat = wc.to_shape((c_out, c_in_chunk * k)).map_err(|e| S2Error::Shape(e.to_string()))?;

        let wide = c_in_chunk * k + 1 + if add.is_some() { t } else { 0 };
        let mut out = Array2::<f32>::zeros((c_out, m));
        for (o, n) in window_offsets(m, step) {
            let take = t.min(l - o);
            let mut win = Array2::<f32>::zeros((c_in_chunk, t));
            win.slice_mut(s![.., 0..take]).assign(&xc.slice(s![.., o..o + take]));

            let mut tiles = Array2::<f32>::zeros((c_out, wide));
            tiles.slice_mut(s![.., 0..c_in_chunk * k]).assign(&wc_flat);
            tiles.column_mut(c_in_chunk * k).assign(bias);
            if let Some(add) = add {
                let seg = add.slice(s![.., o..o + n]);
                tiles.slice_mut(s![.., c_in_chunk * k + 1 + ctx..c_in_chunk * k + 1 + ctx + n]).assign(&seg);
            }

            let got = design.dispatch(tiles.as_slice().expect("standard layout"), Some(win.as_slice().expect("standard layout")))?;
            let got = Array2::from_shape_vec((c_out, t), got).map_err(|e| S2Error::Shape(e.to_string()))?;
            out.slice_mut(s![.., o..o + n]).assign(&got.slice(s![.., ctx..ctx + n]));
        }
        Ok(out)
    }
}

// ---- conv_transpose --------------------------------------------------------------------------

/// Transposed (upsampling) conv `[c_in_total, L] -> [c_out, (L - ctx) * stride]`, `k = 2*stride`,
/// `ctx = ceil((k-1)/stride) = 2` for every codec rate. Never carries a residual add (unlike
/// [`ConvOp`]), so exactly one design is reused across every chunk. Ports
/// `window_driver.conv_transpose`/`_conv_transpose_chunk`.
pub struct ConvTransposeOp {
    design: S2Design,
    k: usize,
    stride: usize,
    ctx: usize,
    c_in_chunk: usize,
    c_in_total: usize,
    c_out: usize,
    t: usize,
    step: usize,
}

impl ConvTransposeOp {
    pub fn new(design: S2Design) -> crate::Result<Self> {
        let Some(S2OpParams::ConvTranspose { k, stride, ctx, c_in, c_in_total, c_out, t, step, .. }) =
            &design.meta.op_params
        else {
            return Err(kind_err("conv_transpose", &design.meta));
        };
        let (k, stride, ctx, c_in_chunk, c_in_total, c_out, t, step) =
            (*k, *stride, *ctx, *c_in, *c_in_total, *c_out, *t, *step);
        if design.meta.n_tiles != c_out {
            return Err(shape_err("n_tiles", &design.meta.symbol, design.meta.n_tiles, c_out));
        }
        let want_in_tile = c_in_chunk * k + 1;
        if design.meta.in_tile != want_in_tile {
            return Err(shape_err("in_tile", &design.meta.symbol, design.meta.in_tile, want_in_tile));
        }
        if design.meta.out_numel != t * stride {
            return Err(shape_err("out_numel", &design.meta.symbol, design.meta.out_numel, t * stride));
        }
        Ok(ConvTransposeOp { design, k, stride, ctx, c_in_chunk, c_in_total, c_out, t, step })
    }

    pub fn ctx(&self) -> usize {
        self.ctx
    }
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// `x`: `[c_in_total, L]`. `w`: `[c_in_total, c_out, k]` (conv_transpose layout, NOT
    /// `[c_out,c_in,k]`). `bias`: `[c_out]`. -> `[c_out, M*stride]`, `M = L - ctx`.
    pub fn run(&self, x: &Array2<f32>, w: &Array3<f32>, bias: &Array1<f32>) -> crate::Result<Array2<f32>> {
        let (c_in_total, l) = x.dim();
        if c_in_total != self.c_in_total {
            return Err(S2Error::Shape(format!("conv_transpose: x has {c_in_total} input channels, op wants {}", self.c_in_total)));
        }
        if w.dim() != (self.c_in_total, self.c_out, self.k) {
            return Err(S2Error::Shape(format!("conv_transpose: w.dim()={:?}, want {:?}", w.dim(), (self.c_in_total, self.c_out, self.k))));
        }
        if bias.len() != self.c_out {
            return Err(S2Error::Shape(format!("conv_transpose: bias has {} elements, want {}", bias.len(), self.c_out)));
        }
        let m = l.checked_sub(self.ctx).filter(|&m| m > 0).ok_or_else(|| {
            S2Error::Shape(format!("conv_transpose: segment {l} shorter than context {}", self.ctx))
        })?;

        let mut out = Array2::<f32>::zeros((self.c_out, m * self.stride));
        let zero_bias = Array1::<f32>::zeros(self.c_out);
        let mut c0 = 0usize;
        let mut chunk_idx = 0usize;
        while c0 < self.c_in_total {
            let cs = self.c_in_chunk.min(self.c_in_total - c0);
            let bias_eff = if chunk_idx == 0 { bias } else { &zero_bias };
            let partial = self.run_chunk(x, w, bias_eff, c0, cs, m)?;
            out += &partial;
            c0 += self.c_in_chunk;
            chunk_idx += 1;
        }
        Ok(out)
    }

    fn run_chunk(&self, x: &Array2<f32>, w: &Array3<f32>, bias: &Array1<f32>, c0: usize, cs: usize, m: usize) -> crate::Result<Array2<f32>> {
        let (k, ctx, t, step, stride, c_in_chunk, c_out) =
            (self.k, self.ctx, self.t, self.step, self.stride, self.c_in_chunk, self.c_out);
        let l = x.dim().1;

        let mut xc = Array2::<f32>::zeros((c_in_chunk, l));
        xc.slice_mut(s![0..cs, ..]).assign(&x.slice(s![c0..c0 + cs, ..]));
        // w is [c_in_total, c_out, k]; per-output-channel row needs [c_in_chunk, k] (ci-major,
        // k-minor), so slice+permute into [c_out, c_in_chunk, k] before flattening -- same target
        // layout ConvOp builds, just from a differently-axis-ordered source.
        let mut wc = Array3::<f32>::zeros((c_out, c_in_chunk, k));
        wc.slice_mut(s![.., 0..cs, ..]).assign(&w.slice(s![c0..c0 + cs, .., ..]).permuted_axes([1, 0, 2]));
        let wc_flat = wc.to_shape((c_out, c_in_chunk * k)).map_err(|e| S2Error::Shape(e.to_string()))?;

        let tile_w = c_in_chunk * k;
        let mut out = Array2::<f32>::zeros((c_out, m * stride));
        for (o, n) in window_offsets(m, step) {
            let take = t.min(l - o);
            let mut win = Array2::<f32>::zeros((c_in_chunk, t));
            win.slice_mut(s![.., 0..take]).assign(&xc.slice(s![.., o..o + take]));

            let mut tiles = Array2::<f32>::zeros((c_out, tile_w + 1));
            tiles.slice_mut(s![.., 0..tile_w]).assign(&wc_flat);
            tiles.column_mut(tile_w).assign(bias);

            let got = self.design.dispatch(tiles.as_slice().expect("standard layout"), Some(win.as_slice().expect("standard layout")))?;
            let got = Array2::from_shape_vec((c_out, t * stride), got).map_err(|e| S2Error::Shape(e.to_string()))?;
            out.slice_mut(s![.., o * stride..(o + n) * stride]).assign(&got.slice(s![.., ctx * stride..(ctx + n) * stride]));
        }
        Ok(out)
    }
}
