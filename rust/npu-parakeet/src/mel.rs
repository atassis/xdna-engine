//! Pure-Rust port of `preprocessor.onnx` (NeMo `AudioToMelSpectrogramPreprocessor`,
//! `nemo128`): pre-emphasis, centered STFT, power spectrogram, mel filterbank, log, per-utterance
//! (mean/std) normalization. Ported node-for-node from the graph (`onnx.load` dump, opset 17,
//! see the task's own investigation) rather than from textbook defaults, and every constant below
//! is that graph's own initializer value, not a guess.
//!
//! Scope: batch=1, full-length input only (`waveforms_lens == waveforms.len()`), matching the
//! single call site (`npu-models/src/asr/parakeet.rs`, which always passes the whole clip's own
//! length). The graph's batch/length masking (`timemask`, the `mask` Where nodes) is dead code
//! under that constraint and is not implemented -- a shorter `waveforms_lens` would need it back.
//!
//! One graph quirk this preserves exactly, confirmed against the real onnxruntime golden
//! (`tests/mel_parity.rs`): `features_lens = waveforms_lens / hop` (floor) is ONE LESS than the
//! actual number of STFT frames, so the graph always emits one extra trailing frame column that
//! is excluded from the mean/std stats and then zeroed in the output. `compute_features` does the
//! same: `valid_len = wav.len() / HOP`, output has `valid_len + 1` columns, and the last is zero.

use ndarray::{Array1, Array2};

const N_FFT: usize = 512; // STFT frame length = len(hann_window) initializer
const HOP: usize = 160; // hop_length_23_cast initializer
const PAD: usize = N_FFT / 2; // Pad node's [0,256,0,256]: centered STFT, half the frame each side
const N_MELS: usize = 128;
const N_FREQ: usize = N_FFT / 2 + 1; // onesided STFT bin count (ONNX STFT-17 default onesided=1)
const PREEMPH: f32 = 0.97; // preemph_7_cast initializer
const LOG_GUARD: f32 = 5.9604645e-08; // log_zero_guard_value_cast initializer (== 2^-24)
const NORM_EPS: f32 = 1e-5; // const_14_cast_2, added to std (not var) before dividing

const HANN_BYTES: &[u8] = include_bytes!("assets/mel_hann_window_512.f32le");
const MEL_FB_BYTES: &[u8] = include_bytes!("assets/mel_filterbank_257x128.f32le");

fn parse_f32le(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

fn hann_window() -> Array1<f32> {
    let v = parse_f32le(HANN_BYTES);
    assert_eq!(v.len(), N_FFT);
    Array1::from_vec(v)
}

fn mel_filterbank() -> Array2<f32> {
    let v = parse_f32le(MEL_FB_BYTES);
    Array2::from_shape_vec((N_FREQ, N_MELS), v).unwrap()
}

fn preemphasis(wav: &[f32]) -> Vec<f32> {
    if wav.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(wav.len());
    out.push(wav[0]);
    for n in 1..wav.len() {
        out.push(wav[n] - PREEMPH * wav[n - 1]);
    }
    out
}

/// Onesided power spectrum of one windowed 512-sample frame: naive O(N_FFT*N_FREQ) real DFT, not
/// an FFT -- this runs a handful of times per clip (CPU pre/post-processing, not the NPU hot
/// path), so the O(N^2) cost buys avoiding an FFT crate dependency for a one-shot host op.
fn frame_power_spectrum(frame: &[f32], window: &Array1<f32>, out: &mut [f32]) {
    debug_assert_eq!(frame.len(), N_FFT);
    let windowed: Vec<f32> = frame.iter().zip(window.iter()).map(|(&x, &w)| x * w).collect();
    for k in 0..N_FREQ {
        let mut re = 0f64;
        let mut im = 0f64;
        let w = -2.0 * std::f64::consts::PI * (k as f64) / (N_FFT as f64);
        for (n, &x) in windowed.iter().enumerate() {
            let theta = w * (n as f64);
            re += (x as f64) * theta.cos();
            im += (x as f64) * theta.sin();
        }
        out[k] = (re * re + im * im) as f32; // ReduceSumSquare over the [real,imag] pair
    }
}

/// `waveforms` -> `features` (`[N_MELS, T]`), matching `preprocessor.onnx` run with
/// `waveforms_lens == [waveforms.len()]`. See the module doc for the trailing-zero-column quirk.
pub fn compute_features(wav: &[f32]) -> Array2<f32> {
    let pre = preemphasis(wav);
    let mut padded = vec![0f32; PAD];
    padded.extend_from_slice(&pre);
    padded.extend(std::iter::repeat(0f32).take(PAD));

    let n_frames = 1 + (padded.len() - N_FFT) / HOP; // ONNX STFT-17: 1 + (len - frame_length) // frame_step
    let window = hann_window();
    let fb = mel_filterbank();

    let mut power = Array2::<f32>::zeros((n_frames, N_FREQ));
    let mut bin = vec![0f32; N_FREQ];
    for t in 0..n_frames {
        let start = t * HOP;
        frame_power_spectrum(&padded[start..start + N_FFT], &window, &mut bin);
        power.row_mut(t).assign(&Array1::from_vec(bin.clone()));
    }

    let mel_power = power.dot(&fb); // [n_frames, N_MELS]
    let log_mel = mel_power.mapv(|v| (v + LOG_GUARD).ln());

    let valid_len = (wav.len() / HOP).min(n_frames); // features_lens = waveforms_lens // hop
    let mut out = Array2::<f32>::zeros((N_MELS, n_frames));
    for c in 0..N_MELS {
        let col: Vec<f32> = (0..valid_len).map(|t| log_mel[[t, c]]).collect();
        if valid_len == 0 {
            continue;
        }
        let mean = col.iter().sum::<f32>() / valid_len as f32;
        let var = if valid_len > 1 {
            col.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / (valid_len - 1) as f32
        } else {
            0.0
        };
        let std = var.sqrt();
        for t in 0..valid_len {
            out[[c, t]] = (log_mel[[t, c]] - mean) / (std + NORM_EPS);
        }
        // t in [valid_len, n_frames) stays 0, matching the graph's mask Where.
    }
    out
}
