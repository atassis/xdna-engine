//! Verify the Rust TDT decoder host reference vs the NumPy golden.
//!
//! Loads the decoder weights + the saved encoder output for en_01.wav (produced by
//! `scripts/parakeet_tdt_decoder_ref.py --dump-weights` and the enc_en01.npy save), runs
//! the greedy TDT decode, and checks the emitted token sequence matches the golden tokens
//! in artifacts/parakeet/decoder/refs/tdt_decode_golden.npz (exported alongside as a .npy).
//!
//! Run from repo root:  rust/target/release/verify_tdt_decoder

use std::path::Path;

use ndarray::prelude::*;
use ndarray_npy::read_npy;
use npu_parakeet::decoder::TdtDecoder;

fn main() {
    let wdir = Path::new("artifacts/parakeet/decoder/weights");
    let enc_path = Path::new("artifacts/parakeet/decoder/refs/enc_en01.npy");
    let gold_path = Path::new("artifacts/parakeet/decoder/refs/tokens_en01.npy");

    let dec = TdtDecoder::load(wdir);
    let enc: Array2<f32> = read_npy(enc_path).expect("read enc_en01.npy");
    let enc_len = enc.shape()[0];
    println!("[tdt] enc T={} d={}", enc_len, enc.shape()[1]);

    let tokens = dec.greedy_decode(&enc, enc_len);
    println!("[tdt] rust tokens ({}): {:?}", tokens.len(), tokens);

    let gold: Array1<i64> = read_npy(gold_path).expect("read tokens_en01.npy golden");
    let gold: Vec<usize> = gold.iter().map(|&x| x as usize).collect();
    println!("[tdt] gold tokens ({}): {:?}", gold.len(), gold);

    let pass = tokens == gold;
    println!("RESULT: {}", if pass { "PASS" } else { "FAIL" });
    std::process::exit(if pass { 0 } else { 1 });
}
