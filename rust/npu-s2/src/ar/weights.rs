//! Lazy, per-tensor weight resolver for the S2 AR checkpoint (`s2.cpp/models/s2-pro-q6_k.gguf`,
//! resolved by `scripts/codec_paths.py::gguf()`). Names and shapes mirror `scripts/s2_ar_ref.py`
//! exactly (`slow_layer_names`/`fast_layer_names`/`SLOW_TOP_LEVEL`/`FAST_TOP_LEVEL`), which is the
//! authority this module was checked against, not this file's own judgment.
//!
//! WHY LAZY, MEASURED: the AR half needs 358 tensors -- 204 q6_k (4,561,633,280 elements) + 154
//! f16 norms (219,136 elements) = 4,561,852,416 elements total, 17.0 GiB dequantized to f32. This
//! box has ~18 GiB available (`free -h`, 2026-09). Decoding every tensor at load time would come
//! within a few GiB of the box's own budget before a single token is produced, on top of the
//! ~4.2 GiB the checkpoint's raw bytes already occupy once opened (`GgufFile::open` reads the
//! whole file, decoder weights included -- unchanged pre-existing behavior, not this module's
//! doing). One slow OR fast transformer block, by contrast, is `wqkv` 15.7M + `wo` 10.5M +
//! `w1`/`w3`/`w2` 24.9M each = 100.9M elements = 403 MiB f32 -- a workable per-layer working set.
//! So every accessor here decodes only the tensor(s) it names, on the caller's schedule, with no
//! caching in this type: caching is a residency POLICY (per-dispatch re-upload vs register-once),
//! and that decision is explicitly out of scope here -- see `ArWeights`' own doc.
//!
//! Embedding tables get the same treatment one level deeper: `embeddings.weight` alone is
//! [155776, 2560] q6_k (~1.66 GiB decoded), so [`ArWeights::embedding_rows`] and
//! [`ArWeights::codebook_embedding_rows`] read only the requested numpy-shape row range
//! (`GgufFile::tensor_f32_rows`, this module's one addition to `gguf.rs`) rather than the whole
//! table -- an AR step gathers a handful of rows, never all of them. `fast_embeddings.weight` is
//! small enough (~42 MiB decoded) that a full read is fine.

use ndarray::{Array1, Array2};

use crate::gguf::GgufFile;
use crate::S2Error;

/// One transformer block's weights, in the order `s2_ar_ref.py`'s `slow_layer_names`/
/// `fast_layer_names` list them. `q_norm`/`k_norm` are `Some` only when the checkpoint carries
/// that pair of tensors for this stack -- measured on `s2-pro-q6_k.gguf`: present for all 36 slow
/// layers, absent for all 4 fast layers (`fish_speech.attention_qk_norm`=true,
/// `fish_speech.fast_attention_qk_norm`=false). Both keys are BOOL-typed GGUF KVs, a wire type
/// `GgufFile::meta_u32` doesn't decode, so presence is detected per-tensor instead of read from
/// the KV directly (see [`ArWeights::layer`]).
pub struct ArLayerWeights {
    pub attention_norm: Array1<f32>,
    pub ffn_norm: Array1<f32>,
    pub wqkv: Array2<f32>,
    pub wo: Array2<f32>,
    pub w1: Array2<f32>,
    pub w2: Array2<f32>,
    pub w3: Array2<f32>,
    pub q_norm: Option<Array1<f32>>,
    pub k_norm: Option<Array1<f32>>,
}

/// Resolves AR tensors by name, decoding each one lazily on the accessor call that asks for it.
/// Holds no dequantized state and no cache -- repeated calls re-decode, which is deliberate: this
/// type answers "what are the weights", not "how should they be staged for dispatch" (re-upload
/// per call vs. register-once vs. an LRU of hot layers). That residency question is a separate,
/// still-open decision; building it in here would foreclose whichever strategy gets picked.
pub struct ArWeights {
    gguf: GgufFile,
}

impl ArWeights {
    pub fn open(path: &std::path::Path) -> crate::Result<Self> {
        Ok(ArWeights { gguf: GgufFile::open(path)? })
    }

    fn v(&self, name: &str) -> crate::Result<Array1<f32>> {
        Ok(Array1::from_vec(self.gguf.tensor_f32(name)?))
    }

    /// Requires an on-disk 2-D shape; every AR weight matrix is 2-D (`s2_ar_ref.py`'s
    /// `tensor_numpy_shape`, ggml `ne` reversed).
    fn m2(&self, name: &str) -> crate::Result<Array2<f32>> {
        let shape = self.gguf.shape(name)?;
        let [d0, d1]: [usize; 2] = shape.clone().try_into().map_err(|_| {
            S2Error::Shape(format!("{name}: expected a 2-D tensor, got shape {shape:?}"))
        })?;
        let data = self.gguf.tensor_f32(name)?;
        Array2::from_shape_vec((d0, d1), data).map_err(|e| S2Error::Shape(format!("{name}: {e}")))
    }

    /// Rows `[row_lo, row_hi)` of a 2-D tensor -- see `GgufFile::tensor_f32_rows`.
    fn m2_rows(&self, name: &str, row_lo: usize, row_hi: usize) -> crate::Result<Array2<f32>> {
        let shape = self.gguf.shape(name)?;
        let [_, d1]: [usize; 2] = shape.clone().try_into().map_err(|_| {
            S2Error::Shape(format!("{name}: expected a 2-D tensor, got shape {shape:?}"))
        })?;
        let data = self.gguf.tensor_f32_rows(name, row_lo, row_hi)?;
        Array2::from_shape_vec((row_hi - row_lo, d1), data)
            .map_err(|e| S2Error::Shape(format!("{name}: {e}")))
    }

    fn has_tensor(&self, name: &str) -> bool {
        self.gguf.shape(name).is_ok()
    }

    /// `fish-speech.block_count` (hyphenated, arch-prefixed key). 36 on `s2-pro-q6_k.gguf`. `None`
    /// if the checkpoint doesn't carry it as a UINT32 KV -- every AR hparam this module cares
    /// about is UINT32 in the real checkpoint (confirmed by inspecting its KV wire types) except
    /// the two `*_qk_norm` BOOLs, which `layer()` handles by tensor presence instead.
    pub fn slow_layer_count(&self) -> Option<u32> {
        self.gguf.meta_u32("fish-speech.block_count")
    }

    /// `fish_speech.fast_block_count` (underscored, NOT arch-prefixed -- `s2_ar_ref.py`'s
    /// `read_ar_hparams` reads slow and fast hparams under two different GGUF key-namespace
    /// conventions; this mirrors that exactly). 4 on `s2-pro-q6_k.gguf`.
    pub fn fast_layer_count(&self) -> Option<u32> {
        self.gguf.meta_u32("fish_speech.fast_block_count")
    }

    fn layer(&self, stack: &str, il: usize) -> crate::Result<ArLayerWeights> {
        let p = format!("{stack}.{il}.");
        let q_norm_name = format!("{p}attention.q_norm.weight");
        let k_norm_name = format!("{p}attention.k_norm.weight");
        Ok(ArLayerWeights {
            attention_norm: self.v(&format!("{p}attention_norm.weight"))?,
            ffn_norm: self.v(&format!("{p}ffn_norm.weight"))?,
            wqkv: self.m2(&format!("{p}attention.wqkv.weight"))?,
            wo: self.m2(&format!("{p}attention.wo.weight"))?,
            w1: self.m2(&format!("{p}feed_forward.w1.weight"))?,
            w2: self.m2(&format!("{p}feed_forward.w2.weight"))?,
            w3: self.m2(&format!("{p}feed_forward.w3.weight"))?,
            q_norm: self.has_tensor(&q_norm_name).then(|| self.v(&q_norm_name)).transpose()?,
            k_norm: self.has_tensor(&k_norm_name).then(|| self.v(&k_norm_name)).transpose()?,
        })
    }

    /// Slow (main) transformer block `il`, `layers.{il}.*` (`s2_ar_ref.py::slow_layer_names`).
    pub fn slow_layer(&self, il: usize) -> crate::Result<ArLayerWeights> {
        self.layer("layers", il)
    }

    /// Fast (residual-codebook) transformer block `il`, `fast_layers.{il}.*`
    /// (`s2_ar_ref.py::fast_layer_names`).
    pub fn fast_layer(&self, il: usize) -> crate::Result<ArLayerWeights> {
        self.layer("fast_layers", il)
    }

    /// Slow transformer's final RMSNorm gamma, `norm.weight`, `[dim]`.
    pub fn norm(&self) -> crate::Result<Array1<f32>> {
        self.v("norm.weight")
    }

    /// Fast transformer's final RMSNorm gamma, `fast_norm.weight`, `[dim]`.
    pub fn fast_norm(&self) -> crate::Result<Array1<f32>> {
        self.v("fast_norm.weight")
    }

    /// Fast transformer's norm-to-logits projection, `fast_output.weight`, `[codebook_size, dim]`
    /// = `[4096, 2560]` on this checkpoint.
    pub fn fast_output(&self) -> crate::Result<Array2<f32>> {
        self.m2("fast_output.weight")
    }

    /// Full tied input/output embedding table, `embeddings.weight`, `[vocab_size, dim]` =
    /// `[155776, 2560]` -- q6_k, ~1.66 GiB decoded. Prefer [`Self::embedding_rows`] unless the
    /// whole table is genuinely needed.
    pub fn embeddings(&self) -> crate::Result<Array2<f32>> {
        self.m2("embeddings.weight")
    }

    /// Numpy-shape rows `[row_lo, row_hi)` of `embeddings.weight`. Also serves the tied output
    /// projection (`s2_ar_ref.py::logits_rows`/`logits_full` read the same table).
    pub fn embedding_rows(&self, row_lo: usize, row_hi: usize) -> crate::Result<Array2<f32>> {
        self.m2_rows("embeddings.weight", row_lo, row_hi)
    }

    /// Full RVQ codebook-token embedding table, `codebook_embeddings.weight`,
    /// `[num_codebooks*codebook_size, dim]` = `[40960, 2560]` -- q6_k, ~420 MiB decoded. Prefer
    /// [`Self::codebook_embedding_rows`] unless the whole table is genuinely needed.
    pub fn codebook_embeddings(&self) -> crate::Result<Array2<f32>> {
        self.m2("codebook_embeddings.weight")
    }

    /// Numpy-shape rows `[row_lo, row_hi)` of `codebook_embeddings.weight`.
    pub fn codebook_embedding_rows(&self, row_lo: usize, row_hi: usize) -> crate::Result<Array2<f32>> {
        self.m2_rows("codebook_embeddings.weight", row_lo, row_hi)
    }

    /// Full fast-transformer prefix-token embedding table, `fast_embeddings.weight`,
    /// `[codebook_size, dim]` = `[4096, 2560]` -- q6_k, ~42 MiB decoded; small enough that a
    /// row-range accessor buys little, so this is the only form offered.
    pub fn fast_embeddings(&self) -> crate::Result<Array2<f32>> {
        self.m2("fast_embeddings.weight")
    }
}
