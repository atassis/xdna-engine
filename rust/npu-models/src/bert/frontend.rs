//! BERT embedding frontend: WordPiece tokenize (HF tokenizers) -> summed embeddings -> LayerNorm.

use std::path::Path;
use std::rc::Rc;

use ndarray::{Array2, Axis};
use npu_asr_host::batch::PassageSpans;
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

    pub fn max_seq(&self) -> usize { self.max_seq }

    /// Full passage lengths are checked without tokenizer-config truncation or padding.
    pub fn batch_token_ids(&self, texts: &[&str]) -> Result<Vec<Vec<u32>>, String> {
        let mut tok = self.tok.clone();
        tok.with_truncation(None).map_err(|e| format!("disable tokenizer truncation: {e}"))?;
        tok.with_padding(None);
        texts.iter().enumerate().map(|(i, text)| {
            let enc = tok.encode(*text, true).map_err(|e| format!("tokenize passage {i}: {e}"))?;
            let ids = enc.get_ids().to_vec();
            if ids.is_empty() || ids.len() > self.max_seq {
                return Err(format!("passage {i} has {} tokens; expected 1..={} (including special tokens)", ids.len(), self.max_seq));
            }
            Ok(ids)
        }).collect()
    }

    pub fn run_packed(&self, ids: &[Vec<u32>], capacity: usize) -> Result<(Array2<f32>, PassageSpans), String> {
        let lengths: Vec<_> = ids.iter().map(Vec::len).collect();
        let spans = PassageSpans::from_lengths(&lengths, self.max_seq, capacity)?;
        let word = self.weights.word_emb();
        let pos = self.weights.pos_emb();
        let typ = self.weights.type_emb();
        if pos.nrows() < self.max_seq || pos.ncols() != word.ncols() || typ.ncols() != word.ncols() || typ.nrows() == 0 {
            return Err("BERT embedding table dimensions do not cover the configured position limit".into());
        }
        let mut emb = Array2::<f32>::zeros((spans.rows(), word.ncols()));
        for (passage, span) in ids.iter().zip(spans.ranges()) {
            for (position, &id) in passage.iter().enumerate() {
                if id as usize >= word.nrows() { return Err(format!("token id {id} exceeds embedding vocabulary")); }
                for j in 0..word.ncols() {
                    emb[[span.start + position, j]] = word[[id as usize, j]] + pos[[position, j]] + typ[[0, j]];
                }
            }
        }
        let (g, b) = self.weights.emb_ln();
        let out = layer_norm(&emb, &g, &b, LN_EPS);
        Ok((out, spans))
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use ndarray::{array, s};
    use npu_asr_host::{batch::segmented_mha, gelu};
    use crate::bert::head::{EmbedHead, Pooling};
    use tokenizers::{models::wordlevel::WordLevel, pre_tokenizers::whitespace::Whitespace,
                     processors::template::TemplateProcessing, utils::truncation::TruncationParams};

    fn fixture() -> EmbedFrontend {
        let vocab: HashMap<String, u32> = [("[UNK]", 0), ("[CLS]", 1), ("[SEP]", 2), ("a", 3), ("b", 4), ("c", 5)]
            .into_iter().map(|(s, i)| (s.into(), i)).collect();
        let mut tok = Tokenizer::new(WordLevel::builder().vocab(vocab).unk_token("[UNK]".into()).build().unwrap());
        tok.with_pre_tokenizer(Some(Whitespace));
        tok.with_post_processor(Some(TemplateProcessing::builder().try_single("[CLS] $A [SEP]").unwrap()
            .special_tokens(vec![("[CLS]", 1), ("[SEP]", 2)]).build().unwrap()));
        tok.with_truncation(Some(TruncationParams { max_length: 4, ..Default::default() })).unwrap();
        let emb = HashMap::from([
            ("word_emb".into(), array![[0.,0.,0.,0.], [0.3,-0.4,0.2,0.8], [0.2,0.1,-0.3,0.6],
                [0.7,-0.2,0.9,0.1], [-0.4,0.8,0.2,0.5], [0.9,0.3,-0.6,0.2]].into_dyn()),
            ("pos_emb".into(), array![[0.1,0.2,0.3,0.4], [0.2,-0.1,0.4,0.5], [-0.1,0.6,0.2,0.3],
                [0.3,0.1,-0.5,0.4], [0.7,-0.2,0.1,0.6], [-0.1,0.2,0.3,0.5]].into_dyn()),
            ("type_emb".into(), Array2::<f32>::zeros((2,4)).into_dyn()),
            ("emb_ln_w".into(), ndarray::Array1::ones(4).into_dyn()),
            ("emb_ln_b".into(), ndarray::Array1::zeros(4).into_dyn()),
        ]);
        EmbedFrontend { tok, weights: Rc::new(BertWeights { emb, layers: vec![] }), max_seq: 6 }
    }

    #[test]
    fn strict_512_includes_specials_and_validates_tail_before_execution() {
        let mut f = fixture();
        f.max_seq = 512;
        let max = vec!["a"; 510].join(" ");
        let over = vec!["a"; 511].join(" ");
        assert_eq!(f.batch_token_ids(&[&max]).unwrap()[0].len(), 512);
        assert_eq!(f.batch_token_ids(&[""]).unwrap()[0], vec![1,2]);
        assert!(f.batch_token_ids(&[]).unwrap().is_empty());
        let calls = std::cell::Cell::new(0);
        let execute = |texts: &[&str]| -> Result<(), String> {
            let ids = f.batch_token_ids(texts)?;
            for _ in ids { calls.set(calls.get() + 1); }
            Ok(())
        };
        assert!(execute(&["a", &over]).unwrap_err().contains("passage 1 has 513 tokens"));
        assert_eq!(calls.get(), 0);
        assert_eq!(f.token_ids(&over).len(), 4, "legacy stored truncation unchanged");
        f.tok.with_post_processor(None::<TemplateProcessing>);
        assert!(f.batch_token_ids(&[""]).unwrap_err().contains("0 tokens"));
    }

    #[test]
    fn batch_tokenizer_limits_include_special_tokens_and_ignore_config_truncation() {
        let f = fixture();
        assert_eq!(f.batch_token_ids(&[""]).unwrap(), vec![vec![1,2]]);
        assert_eq!(f.batch_token_ids(&["a a a a"]).unwrap()[0].len(), 6);
        assert_eq!(f.token_ids("a a a a a").len(), 4, "single tokenizer behavior");
        assert!(f.batch_token_ids(&["a a a a a"]).unwrap_err().contains("7 tokens"));
        assert!(f.batch_token_ids(&[]).unwrap().is_empty());
    }

    #[test]
    fn packed_embeddings_reset_positions_and_reject_invalid_ids() {
        let f = fixture();
        let ids = f.batch_token_ids(&["a", "b c", "a", ""]).unwrap();
        let (packed, spans) = f.run_packed(&ids, 32).unwrap();
        assert_eq!(spans.ranges(), &[0..3, 3..7, 7..10, 10..12]);
        for (i, text) in ["a", "b c", "a", ""].iter().enumerate() {
            let (single, _) = f.run(text.to_string());
            assert_eq!(packed.slice(s![spans.ranges()[i].clone(), ..]), single);
        }
        assert!(f.run_packed(&[vec![1,99,2]], 32).is_err());
        assert!(f.run_packed(&[vec![]], 32).is_err());
        assert!(f.run_packed(&[vec![1; 7]], 32).is_err());
    }

    // CPU fixture uses shared packed projections/FFNs and the production frontend/attention/head.
    // It is not a model-weight/device oracle.
    fn cpu_encoder(f: &EmbedFrontend, texts: &[&str]) -> Vec<Vec<f32>> {
        let ids = f.batch_token_ids(texts).unwrap();
        let (mut x, spans) = f.run_packed(&ids, 32).unwrap();
        let w = array![[0.7,-0.2,0.1,0.3], [0.1,0.8,-0.4,0.2], [-0.3,0.4,0.6,0.1], [0.2,-0.1,0.3,0.9]];
        for _ in 0..2 {
            let q = x.dot(&w);
            let k = x.dot(&w.t());
            let v = x.dot(&w);
            let ctx = segmented_mha(&q, &k, &v, 2, 2, &spans).unwrap();
            x = layer_norm(&(&x + &ctx.dot(&w)), &[1.;4], &[0.;4], LN_EPS);
            x = layer_norm(&(&x + &gelu(&x.dot(&w)).dot(&w)), &[1.;4], &[0.;4], LN_EPS);
        }
        EmbedHead { pooling: Pooling::Cls, normalize: true }.run_cls_packed(&x, &spans).unwrap()
    }

    fn close(a: &[f32], b: &[f32]) {
        assert!(a.iter().zip(b).all(|(a,b)| (a-b).abs() < 1e-5), "{a:?} != {b:?}");
    }

    #[test]
    fn numerical_packed_encoder_matches_b1_order_repeats_and_neighbor_mutation() {
        let f = fixture();
        let texts = ["a", "b c", "a", ""];
        let batch = cpu_encoder(&f, &texts);
        assert_eq!(batch, cpu_encoder(&f, &texts), "repeated CPU control");
        for (i, text) in texts.iter().enumerate() {
            close(&batch[i], &cpu_encoder(&f, &[*text])[0]);
            assert!((batch[i].iter().map(|x| x*x).sum::<f32>() - 1.).abs() < 1e-5);
        }
        let changed = cpu_encoder(&f, &["a", "c c c", "a", ""]);
        close(&batch[0], &changed[0]);
        close(&batch[2], &changed[2]);
        let reordered = cpu_encoder(&f, &["", "a", "b c", "a"]);
        for (i, old) in [3,0,1,2].iter().enumerate() { close(&reordered[i], &batch[*old]); }
        assert!(cpu_encoder(&f, &[]).is_empty());
    }
}
