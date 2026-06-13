//! BERT embedding frontend: WordPiece tokenize (HF tokenizers) -> summed embeddings -> LayerNorm.

use std::path::Path;
use std::rc::Rc;

use ndarray::{Array2, Axis};
use npu_asr_host::layer_norm;
use tokenizers::Tokenizer;

use crate::bert::weights::BertWeights;
use crate::pipeline::Frontend;

const LN_EPS: f32 = 1e-12; // BERT LayerNorm eps

pub struct EmbedFrontend {
    tok: Tokenizer,
    weights: Rc<BertWeights>,
    max_seq: usize,
}

impl EmbedFrontend {
    pub fn new(tokenizer_path: &Path, weights: Rc<BertWeights>, max_seq: usize) -> Self {
        let tok = Tokenizer::from_file(tokenizer_path)
            .unwrap_or_else(|e| panic!("load tokenizer {}: {e}", tokenizer_path.display()));
        EmbedFrontend { tok, weights, max_seq }
    }

    /// Token ids for one string (special tokens [CLS]/[SEP] added by the tokenizer config).
    pub fn token_ids(&self, text: &str) -> Vec<u32> {
        let enc = self.tok.encode(text, true).expect("tokenize");
        let mut ids = enc.get_ids().to_vec();
        ids.truncate(self.max_seq);
        ids
    }
}

impl Frontend for EmbedFrontend {
    type Input = String;
    /// Returns ([seq, 768] post-LN embeddings, valid_len = seq).
    fn run(&self, text: String) -> (Array2<f32>, usize) {
        let ids = self.token_ids(&text);
        let seq = ids.len();
        let w = &self.weights;
        let word = w.word_emb(); // [V, 768]
        let pos = w.pos_emb();   // [512, 768]
        let typ = w.type_emb();  // [2, 768]
        let d = word.ncols();
        let mut emb = Array2::<f32>::zeros((seq, d));
        for (t, &id) in ids.iter().enumerate() {
            let wr = word.row(id as usize);
            let pr = pos.row(t);
            let tr = typ.row(0); // single-segment
            for j in 0..d {
                emb[[t, j]] = wr[j] + pr[j] + tr[j];
            }
        }
        let (g, b) = w.emb_ln();
        let out = layer_norm(&emb, &g, &b, LN_EPS);
        let _ = Axis(0);
        (out, seq)
    }
}
