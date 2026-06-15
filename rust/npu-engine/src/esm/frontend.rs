//! ESM-2 frontend: hand-ported 33-symbol residue tokenizer -> word-emb -> token_dropout x0.88.

use std::rc::Rc;
use ndarray::Array2;
use crate::esm::weights::EsmWeights;
use crate::pipeline::Frontend;

/// (1 - 0.15*0.8) / (1 - mask_ratio_observed); for inference with no <mask> tokens this is 0.88.
const TOKEN_DROPOUT_SCALE: f32 = 0.88;

pub struct EsmFrontend {
    weights: Rc<EsmWeights>,
    max_seq: usize,
}
impl EsmFrontend {
    pub fn new(weights: Rc<EsmWeights>, max_seq: usize) -> Self {
        EsmFrontend { weights, max_seq }
    }
    /// Token ids for one protein string (<cls> + residues + <eos>, truncated to max_seq).
    pub fn token_ids(&self, seq: &str) -> Vec<u32> {
        let mut ids = tokenize(seq);
        ids.truncate(self.max_seq);
        ids
    }
}
impl Frontend for EsmFrontend {
    type Input = String;
    /// Returns ([seq, H] embeddings (token_dropout-scaled), valid_len = seq). No pos/type/emb-LN.
    fn run(&self, seq: String) -> (Array2<f32>, usize) {
        let ids = self.token_ids(&seq);
        let word = self.weights.word_emb(); // [V, H]
        let h = word.ncols();
        let n = ids.len();
        let mut emb = Array2::<f32>::zeros((n, h));
        for (t, &id) in ids.iter().enumerate() {
            let wr = word.row(id as usize);
            for j in 0..h {
                emb[[t, j]] = wr[j] * TOKEN_DROPOUT_SCALE;
            }
        }
        (emb, n)
    }
}

/// Residue alphabet in id order (id == index). <null_1> at 31, <mask> at 32.
const VOCAB: [&str; 33] = [
    "<cls>", "<pad>", "<eos>", "<unk>", "L", "A", "G", "V", "S", "E", "R", "T", "I", "D", "P", "K",
    "Q", "N", "F", "Y", "M", "H", "W", "C", "X", "B", "U", "Z", "O", ".", "-", "<null_1>", "<mask>",
];
const CLS: u32 = 0;
const EOS: u32 = 2;
const UNK: u32 = 3;

/// Encode a protein string to ids: <cls> + per-residue + <eos>. Unknown residue -> <unk>.
pub fn tokenize(seq: &str) -> Vec<u32> {
    let mut ids = vec![CLS];
    for ch in seq.chars() {
        let s = ch.to_ascii_uppercase().to_string();
        let id = VOCAB.iter().position(|v| *v == s).map(|p| p as u32).unwrap_or(UNK);
        ids.push(id);
    }
    ids.push(EOS);
    ids
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mktv_matches_hf() {
        assert_eq!(tokenize("MKTV"), vec![0, 20, 15, 11, 7, 2]);
    }
    #[test]
    fn unknown_residue_is_unk() {
        assert_eq!(tokenize("J"), vec![0, 3, 2]); // J not in alphabet -> <unk>=3
    }
}
