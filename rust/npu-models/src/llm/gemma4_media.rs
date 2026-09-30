//! Host-side Gemma-4-12B vision/audio tower forward -- the Rust port of
//! `scripts/gemma4_towers_host_ref.py`, wired to real request bytes instead of `.npy` dumps.
//! Both towers are encoder-free (see `multimodal.rs`'s module doc): vision is `patch_ln1 ->
//! patch_dense -> patch_ln2 -> +factorized-position-gather -> pos_norm -> RMSNorm(no-scale) ->
//! embedding_projection`, audio is `reshape(n,640) -> RMSNorm(no-scale) -> embedding_projection`.
//!
//! Weights load from the served checkpoint's OWN `model.safetensors`
//! (`gemma4-12b-qat/checkpoint`, not the decode artifact's baked int4 blobs and not
//! `artifacts/gemma4-12b/towers/*.npy`, which is a DIFFERENT checkpoint -- see
//! `multimodal-prompt-join-embedtable-scatter`'s worklog), via a direct mmap + selective
//! safetensors read of only these 11 names (see [`read_named_tensors`] -- this checkpoint's
//! `model.safetensors` is the full 24 GB 12B-param file).
//!
//! Resize is the one stage `gemma4_towers_host_ref.py` never reproduced (its oracle picks an image
//! size where `get_aspect_ratio_preserving_size` is a no-op specifically to keep the torchvision
//! resize kernel out of the traced path). This module DOES resize, because a served image is not
//! guaranteed to already fit the patch budget -- a bit-exact port of ATen's uint8 antialiased
//! bicubic kernel (fixed-point weights, not a float convolution: see
//! [`resize_bicubic_antialias_u8`]'s doc for why a float port measured 6.79e-3 rel-L2 off). Parity
//! against `tvF.resize(..., antialias=True)` is measured, not assumed; see `tests/gemma4_media_gate.rs`.

use std::path::Path;

use crate::api::EngineError;
use crate::llm::multimodal::{AUDIO_TOKEN_ID, IMAGE_TOKEN_ID};

pub const PATCH_SIZE: usize = 16;
pub const POOLING_KERNEL_SIZE: usize = 3;
pub const DEFAULT_MAX_SOFT_TOKENS: usize = 280;
pub const AUDIO_SAMPLES_PER_TOKEN: usize = 640;
pub const AUDIO_SAMPLE_RATE: u32 = 16_000;

const VISION_WEIGHT_NAMES: &[&str] = &[
    "model.vision_embedder.patch_ln1.weight",
    "model.vision_embedder.patch_ln1.bias",
    "model.vision_embedder.patch_dense.weight",
    "model.vision_embedder.patch_dense.bias",
    "model.vision_embedder.patch_ln2.weight",
    "model.vision_embedder.patch_ln2.bias",
    "model.vision_embedder.pos_embedding",
    "model.vision_embedder.pos_norm.weight",
    "model.vision_embedder.pos_norm.bias",
    "model.embed_vision.embedding_projection.weight",
];
const AUDIO_WEIGHT_NAMES: &[&str] = &["model.embed_audio.embedding_projection.weight"];

/// One attachment's decoded soft-token rows, ready for [`crate::llm::multimodal::scatter_media_rows`].
pub struct TowerOutput {
    pub token_id: u32,
    pub rows: Vec<Vec<f32>>,
}

/// The 11 tower tensors, held as flat row-major f32 (checkpoint dtype upcast on read, matching
/// `gemma4_towers_host_ref.py`'s `q()` identity path -- no bf16 rounding is applied here, the same
/// choice the numpy reference makes when `--bf16` is not passed).
pub struct Gemma4Towers {
    d_model: usize,
    patch_dense_in: usize,
    // vision
    patch_ln1_w: Vec<f32>, patch_ln1_b: Vec<f32>,
    patch_dense_w: Vec<f32>, patch_dense_b: Vec<f32>, // [d_model, patch_dense_in]
    patch_ln2_w: Vec<f32>, patch_ln2_b: Vec<f32>,
    pos_embedding: Vec<f32>, // [num_positions, 2, d_model]
    pos_norm_w: Vec<f32>, pos_norm_b: Vec<f32>,
    vision_proj_w: Vec<f32>, vision_proj_out: usize, // [out, d_model]
    // audio
    audio_proj_w: Vec<f32>, audio_proj_out: usize, audio_proj_in: usize, // [out, in]
}

struct Tensor { shape: Vec<usize>, data: Vec<f32> }

/// Read exactly the named tensors out of a (possibly huge) safetensors file, upcasting only THOSE
/// to f32 -- never the whole file. `safetensors::SafeTensors::deserialize` parses the header and
/// hands back zero-copy views into the mmap; only `tensor(name).data()` for a name we actually ask
/// for touches its bytes.
fn read_named_tensors(path: &Path, names: &[&str])
    -> Result<std::collections::HashMap<String, Tensor>, EngineError> {
    let file = std::fs::File::open(path).map_err(|e| EngineError::Load(format!("open {}: {e}", path.display())))?;
    let mmap = unsafe { memmap2::Mmap::map(&file) }
        .map_err(|e| EngineError::Load(format!("mmap {}: {e}", path.display())))?;
    let st = safetensors::SafeTensors::deserialize(&mmap)
        .map_err(|e| EngineError::Load(format!("parse {}: {e}", path.display())))?;
    let mut out = std::collections::HashMap::new();
    for &name in names {
        let view = st.tensor(name).map_err(|_| EngineError::Load(format!(
            "checkpoint {} missing tower tensor {name}", path.display())))?;
        let shape = view.shape().to_vec();
        let raw = view.data();
        let data: Vec<f32> = match view.dtype() {
            safetensors::Dtype::F32 => raw.chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect(),
            safetensors::Dtype::BF16 => raw.chunks_exact(2)
                .map(|b| half::bf16::from_le_bytes(b.try_into().unwrap()).to_f32()).collect(),
            safetensors::Dtype::F16 => raw.chunks_exact(2)
                .map(|b| half::f16::from_le_bytes(b.try_into().unwrap()).to_f32()).collect(),
            d => return Err(EngineError::Load(format!("tower tensor {name} has unexpected dtype {d:?}"))),
        };
        out.insert(name.to_string(), Tensor { shape, data });
    }
    Ok(out)
}

fn get<'a>(bag: &'a std::collections::HashMap<String, Tensor>, name: &str)
    -> Result<&'a Tensor, EngineError> {
    bag.get(name).ok_or_else(|| EngineError::Load(format!("checkpoint missing tower tensor {name}")))
}

impl Gemma4Towers {
    /// Load the 11 tower tensors from `<checkpoint_dir>/model.safetensors`. Fails loud (never
    /// silently substitutes random-init params) if any name is absent -- the exact failure mode
    /// `from_pretrained` hides on this checkpoint (see the module doc). Reads ONLY these 11 names
    /// out of the (multi-GB) checkpoint file -- see [`read_named_tensors`]'s doc for why that
    /// matters here specifically.
    pub fn load(checkpoint_dir: &Path) -> Result<Self, EngineError> {
        let path = checkpoint_dir.join("model.safetensors");
        let names: Vec<&str> = VISION_WEIGHT_NAMES.iter().chain(AUDIO_WEIGHT_NAMES).copied().collect();
        let bag = read_named_tensors(&path, &names)?;
        let patch_dense = get(&bag, "model.vision_embedder.patch_dense.weight")?;
        let (d_model, patch_dense_in) = (patch_dense.shape[0], patch_dense.shape[1]);
        let pos_embedding = get(&bag, "model.vision_embedder.pos_embedding")?;
        let vision_proj = get(&bag, "model.embed_vision.embedding_projection.weight")?;
        let audio_proj = get(&bag, "model.embed_audio.embedding_projection.weight")?;
        Ok(Gemma4Towers {
            d_model,
            patch_dense_in,
            patch_ln1_w: get(&bag, "model.vision_embedder.patch_ln1.weight")?.data.clone(),
            patch_ln1_b: get(&bag, "model.vision_embedder.patch_ln1.bias")?.data.clone(),
            patch_dense_w: patch_dense.data.clone(),
            patch_dense_b: get(&bag, "model.vision_embedder.patch_dense.bias")?.data.clone(),
            patch_ln2_w: get(&bag, "model.vision_embedder.patch_ln2.weight")?.data.clone(),
            patch_ln2_b: get(&bag, "model.vision_embedder.patch_ln2.bias")?.data.clone(),
            pos_embedding: pos_embedding.data.clone(),
            pos_norm_w: get(&bag, "model.vision_embedder.pos_norm.weight")?.data.clone(),
            pos_norm_b: get(&bag, "model.vision_embedder.pos_norm.bias")?.data.clone(),
            vision_proj_w: vision_proj.data.clone(),
            vision_proj_out: vision_proj.shape[0],
            audio_proj_w: audio_proj.data.clone(),
            audio_proj_out: audio_proj.shape[0],
            audio_proj_in: audio_proj.shape[1],
        })
    }

    pub fn d_model(&self) -> usize { self.d_model }

    // ---------------------------------------------------------------- vision ----

    /// Full vision pipeline for one image: decode -> aspect-ratio-preserving resize -> rescale ->
    /// patchify -> teacher positions -> patches_merge -> pad -> tower forward -> drop the padding
    /// tail. Returns real soft-token rows only, in prompt order.
    pub fn vision_forward(&self, image_bytes: &[u8], max_soft_tokens: usize)
        -> Result<Vec<Vec<f32>>, EngineError> {
        let img = image::load_from_memory(image_bytes)
            .map_err(|e| EngineError::Unsupported(format!("image decode: {e}")))?
            .to_rgb8();
        let (w0, h0) = (img.width() as usize, img.height() as usize);
        let max_patches = max_soft_tokens * POOLING_KERNEL_SIZE * POOLING_KERNEL_SIZE;
        let (target_h, target_w) = aspect_ratio_preserving_size(
            h0, w0, PATCH_SIZE, max_patches, POOLING_KERNEL_SIZE)?;

        // Resize in u8 (a no-op resize is skipped, matching the HF processor exactly): the
        // processor's own tensor is uint8 end to end, and torchvision's `resize(..., antialias=True)`
        // on a uint8 input runs the fixed-point ATen kernel below, not a float convolution --
        // see [`resize_bicubic_antialias_u8`]'s doc.
        let resized: Vec<Vec<Vec<u8>>> = if (target_h, target_w) == (h0, w0) {
            to_chw_u8(&img)
        } else {
            resize_bicubic_antialias_u8(&to_chw_u8(&img), h0, w0, target_h, target_w)
        };

        // rescale: 1/255, no normalize (image_mean=0, image_std=1 in processor_config.json).
        let rescaled: Vec<Vec<Vec<f32>>> = resized.iter()
            .map(|plane| plane.iter().map(|row| row.iter().map(|&v| v as f32 / 255.0).collect()).collect())
            .collect();

        let patch_h = target_h / PATCH_SIZE;
        let patch_w = target_w / PATCH_SIZE;
        let teacher_patches = convert_image_to_patches(&rescaled, patch_h, patch_w, PATCH_SIZE);
        let teacher_positions = teacher_positions_xy(patch_h, patch_w);

        let num_model_patches = teacher_patches.len() / (POOLING_KERNEL_SIZE * POOLING_KERNEL_SIZE);
        let (merged_patches, merged_positions) = patches_merge(
            &teacher_patches, &teacher_positions, num_model_patches, PATCH_SIZE);

        let n_real = merged_patches.len();
        if n_real > max_soft_tokens {
            return Err(EngineError::Unsupported(format!(
                "image needs {n_real} soft tokens, budget is max_soft_tokens={max_soft_tokens}")));
        }
        let (padded_patches, padded_positions) =
            pad_along_first_dim(&merged_patches, &merged_positions, max_soft_tokens);

        let rows = self.vision_tower_forward(&padded_patches, &padded_positions);
        Ok(rows.into_iter().take(n_real).collect())
    }

    fn vision_tower_forward(&self, patches: &[Vec<f32>], positions: &[(i64, i64)]) -> Vec<Vec<f32>> {
        patches.iter().zip(positions).map(|(patch, &(px, py))| {
            let s1 = layer_norm(patch, &self.patch_ln1_w, &self.patch_ln1_b);
            let s2 = linear(&s1, &self.patch_dense_w, Some(&self.patch_dense_b),
                             self.d_model, self.patch_dense_in);
            let s3 = layer_norm(&s2, &self.patch_ln2_w, &self.patch_ln2_b);
            let pos_emb = self.gather_pos_embedding(px, py);
            let s4: Vec<f32> = s3.iter().zip(&pos_emb).map(|(a, b)| a + b).collect();
            let s5 = layer_norm(&s4, &self.pos_norm_w, &self.pos_norm_b);
            let s6 = rms_norm_no_scale(&s5);
            linear(&s6, &self.vision_proj_w, None, self.vision_proj_out, self.d_model)
        }).collect()
    }

    /// `pos_embedding[clamp(p,0),axis] * (p != -1)`, summed over the two axes (x then y) --
    /// `Gemma4UnifiedVisionEmbedder`'s factorized position embedding, ported from
    /// `gemma4_towers_host_ref.py::vision_tower_forward`'s `pos_embedding[clamped, axes]` gather.
    fn gather_pos_embedding(&self, x: i64, y: i64) -> Vec<f32> {
        let d = self.d_model;
        let mut out = vec![0.0f32; d];
        for (axis, coord) in [(0usize, x), (1usize, y)] {
            if coord == -1 { continue; }
            let idx = coord.max(0) as usize;
            let base = (idx * 2 + axis) * d;
            for i in 0..d { out[i] += self.pos_embedding[base + i]; }
        }
        out
    }

    // ---------------------------------------------------------------- audio ----

    /// Decode a WAV, require 16 kHz mono (this tower has no resampler; a different rate is
    /// rejected loud rather than silently mis-timed), zero-pad to a multiple of
    /// [`AUDIO_SAMPLES_PER_TOKEN`], and run the tower over every resulting frame (no padding is
    /// dropped -- the feature extractor's mask is all-ones, matching
    /// `_compute_audio_num_tokens`'s `ceil(num_samples / 640)`).
    pub fn audio_forward(&self, wav_bytes: &[u8]) -> Result<Vec<Vec<f32>>, EngineError> {
        let samples = decode_wav_mono16k(wav_bytes)?;
        let pad = (AUDIO_SAMPLES_PER_TOKEN - samples.len() % AUDIO_SAMPLES_PER_TOKEN)
            % AUDIO_SAMPLES_PER_TOKEN;
        let mut padded = samples;
        padded.resize(padded.len() + pad, 0.0);
        let frames: Vec<&[f32]> = padded.chunks(AUDIO_SAMPLES_PER_TOKEN).collect();
        Ok(frames.iter().map(|f| {
            let a1 = rms_norm_no_scale(f);
            linear(&a1, &self.audio_proj_w, None, self.audio_proj_out, self.audio_proj_in)
        }).collect())
    }
}

/// `ceil(num_samples / 640)` -- the processor's own token-count formula
/// (`Gemma4UnifiedProcessor._compute_audio_num_tokens`), needed by the caller BEFORE the tower
/// runs, to size the `<|audio|>` placeholder expansion in the rendered prompt.
pub fn audio_num_tokens(num_samples: usize) -> usize {
    num_samples.div_ceil(AUDIO_SAMPLES_PER_TOKEN)
}

/// `get_aspect_ratio_preserving_size`, ported field-for-field from
/// `image_processing_gemma4_unified.py` (float arithmetic order preserved: the floor happens on
/// `ideal / side_mult`, not on `ideal` itself).
pub fn aspect_ratio_preserving_size(
    height: usize, width: usize, patch_size: usize, max_patches: usize, pooling: usize,
) -> Result<(usize, usize), EngineError> {
    let total_px = (height * width) as f64;
    let target_px = (max_patches * patch_size * patch_size) as f64;
    let factor = (target_px / total_px).sqrt();
    let ideal_h = factor * height as f64;
    let ideal_w = factor * width as f64;
    let side_mult = pooling * patch_size;

    let mut target_h = ((ideal_h / side_mult as f64).floor() as i64).max(0) as usize * side_mult;
    let mut target_w = ((ideal_w / side_mult as f64).floor() as i64).max(0) as usize * side_mult;

    if target_h == 0 && target_w == 0 {
        return Err(EngineError::Unsupported(format!(
            "image resizes to 0x0 (height/width must be divisible by pooling_kernel_size*patch_size={side_mult})")));
    }
    let max_side_length = (max_patches / (pooling * pooling)) * side_mult;
    if target_h == 0 {
        target_h = side_mult;
        target_w = ((width / height) * side_mult).min(max_side_length);
    } else if target_w == 0 {
        target_w = side_mult;
        target_h = ((height / width) * side_mult).min(max_side_length);
    }
    if target_h * target_w > max_patches * patch_size * patch_size {
        return Err(EngineError::Unsupported(format!(
            "resizing {height}x{width} to {target_h}x{target_w} exceeds the {max_patches}-patch budget")));
    }
    Ok((target_h, target_w))
}

// ---------------------------------------------------------------- math primitives ----

fn layer_norm(x: &[f32], weight: &[f32], bias: &[f32]) -> Vec<f32> {
    let n = x.len() as f32;
    let mu = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|v| (v - mu) * (v - mu)).sum::<f32>() / n;
    let inv_std = 1.0 / (var + 1e-5).sqrt();
    x.iter().zip(weight).zip(bias).map(|((v, w), b)| (v - mu) * inv_std * w + b).collect()
}

fn rms_norm_no_scale(x: &[f32]) -> Vec<f32> {
    let n = x.len() as f32;
    let ms = x.iter().map(|v| v * v).sum::<f32>() / n + 1e-6;
    let inv = ms.powf(-0.5);
    x.iter().map(|v| v * inv).collect()
}

/// HF `nn.Linear` convention: `weight` is `[out, in]` row-major, `y = x @ weight.T (+ bias)`.
fn linear(x: &[f32], weight: &[f32], bias: Option<&[f32]>, out: usize, inp: usize) -> Vec<f32> {
    debug_assert_eq!(x.len(), inp);
    (0..out).map(|o| {
        let row = &weight[o * inp..(o + 1) * inp];
        let acc: f32 = row.iter().zip(x).map(|(w, v)| w * v).sum();
        bias.map(|b| acc + b[o]).unwrap_or(acc)
    }).collect()
}

/// Test-only entry point into the resize stage alone, for `tests/gemma4_media_gate.rs` to measure
/// its rel-L2 against the oracle independent of patchify/tower error. `chw` is the uint8-domain
/// tensor upcast to f32 (as the oracle's `.npy` dump stores it) -- rounded back to u8 here.
#[doc(hidden)]
pub fn resize_for_gate(chw: &[Vec<Vec<f32>>], in_h: usize, in_w: usize, out_h: usize, out_w: usize)
    -> Vec<Vec<Vec<f32>>> {
    let u8_chw: Vec<Vec<Vec<u8>>> = chw.iter()
        .map(|p| p.iter().map(|r| r.iter().map(|&v| v.round() as u8).collect()).collect()).collect();
    resize_bicubic_antialias_u8(&u8_chw, in_h, in_w, out_h, out_w).iter()
        .map(|p| p.iter().map(|r| r.iter().map(|&v| v as f32).collect()).collect()).collect()
}

// ---------------------------------------------------------------- image preprocessing ----

fn to_chw_u8(img: &image::RgbImage) -> Vec<Vec<Vec<u8>>> {
    let (w, h) = (img.width() as usize, img.height() as usize);
    (0..3).map(|c| (0..h).map(|y| (0..w).map(|x| img.get_pixel(x as u32, y as u32).0[c])
        .collect()).collect()).collect()
}

/// `convert_image_to_patches`: `(C,H,W) -> reshape(C,nph,ps,npw,ps) -> permute(1,3,2,4,0) ->
/// reshape(nph*npw, ps*ps*C)`, ported directly rather than via a generic tensor permute since this
/// is the only shape it is ever called at.
fn convert_image_to_patches(chw: &[Vec<Vec<f32>>], patch_h: usize, patch_w: usize, ps: usize)
    -> Vec<Vec<f32>> {
    let c = chw.len();
    (0..patch_h).flat_map(|ph| (0..patch_w).map(move |pw| (ph, pw))).map(|(ph, pw)| {
        let mut out = Vec::with_capacity(ps * ps * c);
        for i in 0..ps {
            for j in 0..ps {
                for ch in 0..c {
                    out.push(chw[ch][ph * ps + i][pw * ps + j]);
                }
            }
        }
        out
    }).collect()
}

/// `torch.meshgrid(arange(w), arange(h), indexing="xy")` stacked as `(x,y)` pairs, row-major over
/// (patch_h, patch_w) -- i.e. position `i = ph*patch_w + pw` gets `(pw, ph)`.
fn teacher_positions_xy(patch_h: usize, patch_w: usize) -> Vec<(i64, i64)> {
    (0..patch_h).flat_map(|ph| (0..patch_w).map(move |pw| (pw as i64, ph as i64))).collect()
}

/// `patches_merge`, ported for the single-image (no leading batch dim, no pre-existing -1 padding
/// in the input) case this call site always has -- `gemma4_towers_host_ref.py::patches_merge`'s
/// general form specialized to B=1.
fn patches_merge(patches: &[Vec<f32>], positions: &[(i64, i64)], length: usize, patch_size: usize)
    -> (Vec<Vec<f32>>, Vec<(i64, i64)>) {
    let length_l = patches.len();
    let k = ((length_l / length) as f64).sqrt().round() as i64;
    let max_x = positions.iter().map(|p| p.0).max().unwrap_or(0) + 1;

    let mut order: Vec<usize> = (0..length_l).collect();
    let target_ordering: Vec<i64> = positions.iter().map(|&(x, y)| {
        let (kx, ky) = (x / k, y / k);
        let num_from_top_left = k * k * kx + k * max_x * ky;
        let (wx, wy) = (x % k, y % k);
        let num_from_top_left_of_kernel = wx + wy * k;
        num_from_top_left_of_kernel + num_from_top_left
    }).collect();
    order.sort_by_key(|&i| target_ordering[i]);

    let d = patches[0].len();
    let c = d / (patch_size * patch_size);
    let mut merged_patches = Vec::with_capacity(length);
    let mut merged_positions = Vec::with_capacity(length);
    for group in order.chunks(k as usize * k as usize) {
        // group is already in (kk-row-major over ky,kx) order because `order` sorts by
        // `num_from_top_left_of_kernel` within a kernel, which enumerates (wy,wx) row-major --
        // matching numpy's `kop.reshape(length,k,k,ps,ps,c)` before the ky/kx/py/px/c transpose.
        let mut merged = vec![0.0f32; (k * k) as usize * patch_size * patch_size * c];
        // Target layout (ky, py, kx, px, c): numpy reshapes the k*k group-slot axis into (ky,kx)
        // row-major, splits d into (py,px,c) row-major, then transposes to (ky,py,kx,px,c) before
        // flattening -- so a merged patch is spatially (ky*ps+py) rows by (kx*ps+px) cols, which
        // is the natural interleave of the k x k grid of ps x ps teacher patches.
        for (slot, &src_idx) in group.iter().enumerate() {
            let ky = slot / k as usize;
            let kx = slot % k as usize;
            let src = &patches[src_idx];
            for py in 0..patch_size {
                for px in 0..patch_size {
                    for ch in 0..c {
                        let src_off = (py * patch_size + px) * c + ch;
                        let dst_off = ((((ky * patch_size + py) * k as usize + kx) * patch_size + px) * c) + ch;
                        merged[dst_off] = src[src_off];
                    }
                }
            }
        }
        merged_patches.push(merged);
        let min_pos = group.iter().map(|&i| (positions[i].0 / k, positions[i].1 / k)).min().unwrap();
        merged_positions.push(min_pos);
    }
    (merged_patches, merged_positions)
}

fn pad_along_first_dim(patches: &[Vec<f32>], positions: &[(i64, i64)], target_len: usize)
    -> (Vec<Vec<f32>>, Vec<(i64, i64)>) {
    let mut p = patches.to_vec();
    let mut pos = positions.to_vec();
    let width = p.first().map(|r| r.len()).unwrap_or(0);
    while p.len() < target_len {
        p.push(vec![0.0f32; width]);
        pos.push((-1, -1));
    }
    (p, pos)
}

/// Separable bicubic resize with antialiasing, bit-exact port of `upsample_avx_bilinear_bicubic_uint8`
/// (`aten/src/ATen/native/cpu/UpSampleKernel.cpp`) -- the kernel torchvision's
/// `resize(..., antialias=True)` actually runs on a uint8 tensor, which is what the HF processor's
/// `process_image` produces. It is NOT the plain float separable-cubic convolution: weights are
/// quantized to int16 at a precision chosen so the largest weight stays under 2^15, each pass
/// accumulates in fixed-point and rounds+clamps to u8 BEFORE the next pass runs (horizontal, then
/// vertical) -- confirmed bit-exact against the oracle (rel-L2 0.0), where a plain f32 port
/// measured 6.79e-3 off. See [`quantized_resize_taps`], [`convolve_u8`].
fn resize_bicubic_antialias_u8(
    chw: &[Vec<Vec<u8>>], in_h: usize, in_w: usize, out_h: usize, out_w: usize,
) -> Vec<Vec<Vec<u8>>> {
    let horiz = (in_w != out_w).then(|| quantized_resize_taps(in_w, out_w));
    let vert = (in_h != out_h).then(|| quantized_resize_taps(in_h, out_h));
    chw.iter().map(|plane| {
        let mid: Vec<Vec<u8>> = match &horiz {
            Some((precision, taps)) => plane.iter()
                .map(|row| taps.iter().map(|t| convolve_u8(t, |i| row[i], *precision)).collect())
                .collect(),
            None => plane.clone(),
        };
        match &vert {
            Some((precision, taps)) => taps.iter()
                .map(|t| (0..out_w).map(|ox| convolve_u8(t, |iy| mid[iy][ox], *precision)).collect())
                .collect(),
            None => mid,
        }
    }).collect()
}

fn cubic_weight(x: f64, a: f64) -> f64 {
    let x = x.abs();
    if x <= 1.0 { (a + 2.0) * x.powi(3) - (a + 3.0) * x.powi(2) + 1.0 }
    else if x < 2.0 { a * x.powi(3) - 5.0 * a * x.powi(2) + 8.0 * a * x - 4.0 * a }
    else { 0.0 }
}

/// One dimension's per-output-index `(input_index, int16_weight)` tap lists plus the shared
/// fixed-point `precision` (bits), ported field-for-field from
/// `HelperInterpBase::_compute_index_ranges_int16_weights` / `HelperInterpCubic::aa_filter`
/// (`a=-0.5`, matching PIL). `scale` here is ATen's convention (input/output, NOT output/input).
fn quantized_resize_taps(in_size: usize, out_size: usize) -> (u32, Vec<Vec<(usize, i32)>>) {
    let scale = in_size as f64 / out_size as f64;
    let (support, invscale) = if scale >= 1.0 { (2.0 * scale, 1.0 / scale) } else { (2.0, 1.0) };
    let max_interp = support.ceil() as i64 * 2 + 1;

    let mut raw: Vec<Vec<(usize, f64)>> = Vec::with_capacity(out_size);
    let mut wt_max = 0.0f64;
    for o in 0..out_size {
        let center = scale * (o as f64 + 0.5);
        let xmin = ((center - support + 0.5).floor() as i64).max(0);
        let xmax_excl = ((center + support + 0.5).floor() as i64).min(in_size as i64);
        let xsize = (xmax_excl - xmin).clamp(0, max_interp) as usize;
        let mut taps: Vec<(usize, f64)> = Vec::with_capacity(xsize);
        let mut total = 0.0;
        for j in 0..xsize {
            let idx = xmin as usize + j;
            let w = cubic_weight((idx as f64 - center + 0.5) * invscale, -0.5);
            taps.push((idx, w));
            total += w;
        }
        if total != 0.0 {
            for t in &mut taps { t.1 /= total; }
        }
        wt_max = taps.iter().fold(wt_max, |m, &(_, w)| m.max(w));
        raw.push(taps);
    }

    // Largest bit count such that round(wt_max * 2^(precision+1)) still fits an int16.
    let mut precision = 0u32;
    while precision < 22 {
        let next = (0.5 + wt_max * (1i64 << (precision + 1)) as f64) as i64;
        if next >= 1 << 15 { break; }
        precision += 1;
    }

    let quant = raw.iter().map(|taps| taps.iter().map(|&(idx, w)| {
        let v = w * (1i64 << precision) as f64;
        (idx, if v < 0.0 { (v - 0.5) as i32 } else { (v + 0.5) as i32 })
    }).collect()).collect();
    (precision, quant)
}

/// The uint8 fixed-point tap sum: `NOTE [ Weights computation for uint8_t and multiplication
/// trick ]` in `UpSampleKernel.cpp` -- round via a `1 << (precision-1)` bias then arithmetic
/// shift, clamped to `[0, 255]`.
fn convolve_u8(taps: &[(usize, i32)], src: impl Fn(usize) -> u8, precision: u32) -> u8 {
    let mut acc: i64 = if precision == 0 { 0 } else { 1i64 << (precision - 1) };
    for &(idx, w) in taps { acc += src(idx) as i64 * w as i64; }
    (acc >> precision).clamp(0, 255) as u8
}

// ---------------------------------------------------------------- audio: WAV decode ----

/// Minimal PCM WAV reader: `RIFF/WAVE` with a `fmt ` chunk (PCM 16-bit or IEEE float 32-bit) and a
/// `data` chunk. Requires mono, [`AUDIO_SAMPLE_RATE`] -- this tower has no resampler, so any other
/// rate/channel count is rejected loud (never silently mixed down or resampled, which would move
/// every downstream soft token's content).
fn decode_wav_mono16k(bytes: &[u8]) -> Result<Vec<f32>, EngineError> {
    let err = |m: &str| EngineError::Unsupported(format!("wav decode: {m}"));
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(err("not a RIFF/WAVE file"));
    }
    let mut pos = 12;
    let (mut channels, mut sample_rate, mut bits_per_sample, mut fmt_tag) = (0u16, 0u32, 0u16, 0u16);
    let mut data: Option<&[u8]> = None;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as usize;
        let body_start = pos + 8;
        let body_end = (body_start + size).min(bytes.len());
        let body = &bytes[body_start..body_end];
        match id {
            b"fmt " => {
                if body.len() < 16 { return Err(err("fmt chunk too short")); }
                fmt_tag = u16::from_le_bytes(body[0..2].try_into().unwrap());
                channels = u16::from_le_bytes(body[2..4].try_into().unwrap());
                sample_rate = u32::from_le_bytes(body[4..8].try_into().unwrap());
                bits_per_sample = u16::from_le_bytes(body[14..16].try_into().unwrap());
            }
            b"data" => data = Some(body),
            _ => {}
        }
        pos = body_start + size + (size % 2); // chunks are word-aligned
    }
    if channels != 1 {
        return Err(err(&format!("{channels}-channel audio is not supported (mono only, no downmix)")));
    }
    if sample_rate != AUDIO_SAMPLE_RATE {
        return Err(err(&format!(
            "{sample_rate} Hz is not supported (this tower has no resampler; send {AUDIO_SAMPLE_RATE} Hz)")));
    }
    let data = data.ok_or_else(|| err("no data chunk"))?;
    let samples: Vec<f32> = match (fmt_tag, bits_per_sample) {
        (1, 16) => data.chunks_exact(2)
            .map(|b| i16::from_le_bytes(b.try_into().unwrap()) as f32 / 32768.0).collect(),
        (3, 32) => data.chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect(),
        (fmt, bits) => return Err(err(&format!("unsupported wav format tag={fmt} bits={bits}"))),
    };
    Ok(samples)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aspect_ratio_preserving_size_is_a_noop_when_already_under_budget() {
        // 672x960 at max_soft_tokens=280 (max_patches=2520=42x60) is EXACTLY the fixed point
        // gemma4_towers_oracle_gen.py's docstring claims -- reproduced here as a Rust-side check
        // of the same arithmetic, independent of the python oracle.
        let (h, w) = aspect_ratio_preserving_size(672, 960, PATCH_SIZE, 280 * 9, POOLING_KERNEL_SIZE).unwrap();
        assert_eq!((h, w), (672, 960));
    }

    #[test]
    fn aspect_ratio_preserving_size_downscales_and_stays_in_budget() {
        let (h, w) = aspect_ratio_preserving_size(1200, 1600, PATCH_SIZE, 280 * 9, POOLING_KERNEL_SIZE).unwrap();
        assert!(h % (PATCH_SIZE * POOLING_KERNEL_SIZE) == 0);
        assert!(w % (PATCH_SIZE * POOLING_KERNEL_SIZE) == 0);
        assert!((h / PATCH_SIZE) * (w / PATCH_SIZE) <= 280 * 9);
    }

    #[test]
    fn quantized_resize_taps_int16_weights_sum_to_the_precision_unit() {
        // Quantized taps sum to ~2^precision (the float taps sum to 1 before quantization; each
        // is independently rounded, so the reconstructed sum is within a few ULPs of the unit).
        for (in_size, out_size) in [(37, 16), (16, 37)] {
            let (precision, taps) = quantized_resize_taps(in_size, out_size);
            let unit = 1i64 << precision;
            for t in &taps {
                let s: i64 = t.iter().map(|(_, w)| *w as i64).sum();
                assert!((s - unit).abs() <= 4, "sum={s} unit={unit}");
            }
        }
    }

    #[test]
    fn resize_bicubic_antialias_u8_is_a_noop_when_sizes_match() {
        let plane = vec![vec![10u8, 20, 30], vec![40, 50, 60]];
        let chw = vec![plane.clone()];
        let out = resize_bicubic_antialias_u8(&chw, 2, 3, 2, 3);
        assert_eq!(out[0], plane);
    }

    #[test]
    fn audio_num_tokens_matches_ceil_division() {
        assert_eq!(audio_num_tokens(640), 1);
        assert_eq!(audio_num_tokens(641), 2);
        assert_eq!(audio_num_tokens(0), 0);
        assert_eq!(audio_num_tokens(639), 1);
    }

    #[test]
    fn decode_wav_mono16k_rejects_a_wrong_sample_rate_rather_than_resampling() {
        // 44 byte header, 8kHz mono 16-bit PCM, zero samples.
        let mut w = Vec::new();
        w.extend_from_slice(b"RIFF");
        w.extend_from_slice(&(36u32).to_le_bytes());
        w.extend_from_slice(b"WAVE");
        w.extend_from_slice(b"fmt ");
        w.extend_from_slice(&(16u32).to_le_bytes());
        w.extend_from_slice(&(1u16).to_le_bytes()); // PCM
        w.extend_from_slice(&(1u16).to_le_bytes()); // mono
        w.extend_from_slice(&(8000u32).to_le_bytes());
        w.extend_from_slice(&(16000u32).to_le_bytes()); // byte rate (unused by the reader)
        w.extend_from_slice(&(2u16).to_le_bytes()); // block align
        w.extend_from_slice(&(16u16).to_le_bytes());
        w.extend_from_slice(b"data");
        w.extend_from_slice(&(0u32).to_le_bytes());
        let err = decode_wav_mono16k(&w).unwrap_err();
        assert!(format!("{err}").contains("8000"), "{err}");
    }

    #[test]
    fn decode_wav_mono16k_round_trips_pcm16_samples() {
        let samples_in: Vec<i16> = vec![0, 100, -100, i16::MAX, i16::MIN];
        let mut w = Vec::new();
        w.extend_from_slice(b"RIFF");
        w.extend_from_slice(&(36u32 + samples_in.len() as u32 * 2).to_le_bytes());
        w.extend_from_slice(b"WAVE");
        w.extend_from_slice(b"fmt ");
        w.extend_from_slice(&(16u32).to_le_bytes());
        w.extend_from_slice(&(1u16).to_le_bytes());
        w.extend_from_slice(&(1u16).to_le_bytes());
        w.extend_from_slice(&AUDIO_SAMPLE_RATE.to_le_bytes());
        w.extend_from_slice(&(32000u32).to_le_bytes());
        w.extend_from_slice(&(2u16).to_le_bytes());
        w.extend_from_slice(&(16u16).to_le_bytes());
        w.extend_from_slice(b"data");
        w.extend_from_slice(&(samples_in.len() as u32 * 2).to_le_bytes());
        for s in &samples_in { w.extend_from_slice(&s.to_le_bytes()); }
        let out = decode_wav_mono16k(&w).unwrap();
        assert_eq!(out.len(), samples_in.len());
        assert!((out[3] - 1.0).abs() < 1e-3, "{}", out[3]);
    }
}

#[allow(dead_code)]
fn assert_token_ids_are_media() { let _ = (IMAGE_TOKEN_ID, AUDIO_TOKEN_ID); }
