// rust/npu-weights/src/arch/bert.rs
use std::collections::BTreeMap;
use super::{Arch, RawTensor, OutTensor, transpose2d};

pub struct Bert;

/// Return a view of `src` with any leading `bert.` prefix stripped from tensor names. If no key
/// carries the prefix the map is cloned unchanged (cheap relative to the bake's I/O).
fn strip_bert_prefix(src: &BTreeMap<String, RawTensor>) -> BTreeMap<String, RawTensor> {
    if !src.keys().any(|k| k.starts_with("bert.")) { return src.clone(); }
    src.iter()
        .map(|(k, v)| (k.strip_prefix("bert.").unwrap_or(k).to_string(), v.clone()))
        .collect()
}

impl Bert {
    /// Infer encoder depth from the source bag (bert-base/bge-base=12, all-MiniLM-L6=6). Counts the
    /// highest `encoder.layer.{i}.` index present (same scheme as esm.rs).
    fn n_layers(src: &BTreeMap<String, RawTensor>) -> usize {
        let mut n = 0usize;
        for k in src.keys() {
            if let Some(rest) = k.strip_prefix("encoder.layer.") {
                if let Some(idx) = rest.split('.').next() {
                    if let Ok(i) = idx.parse::<usize>() { n = n.max(i + 1); }
                }
            }
        }
        n
    }

    // helpers ---------------------------------------------------------------
    fn w(src: &BTreeMap<String, RawTensor>, k: &str) -> anyhow::Result<RawTensor> {
        src.get(k).cloned().ok_or_else(|| anyhow::anyhow!("missing source tensor {k:?}"))
    }
    fn lin(src: &BTreeMap<String, RawTensor>, k: &str) -> anyhow::Result<OutTensor> {
        let t = transpose2d(&Self::w(src, k)?);
        Ok(OutTensor { shape: t.shape, data: t.data, bf16: true })
    }
    fn keep_f32(src: &BTreeMap<String, RawTensor>, k: &str) -> anyhow::Result<OutTensor> {
        let t = Self::w(src, k)?;
        Ok(OutTensor { shape: t.shape, data: t.data, bf16: false })
    }
    fn keep_bf16(src: &BTreeMap<String, RawTensor>, k: &str) -> anyhow::Result<OutTensor> {
        let t = Self::w(src, k)?;
        Ok(OutTensor { shape: t.shape, data: t.data, bf16: true })
    }

    /// One layer (used by tests + the full transform). Source tensors that are absent are
    /// skipped here; the full `transform` hard-errors on any missing required tensor up front,
    /// so completeness is guaranteed there while unit tests can exercise a single tensor.
    pub fn transform_subset(&self, src: &BTreeMap<String, RawTensor>, i: usize)
        -> anyhow::Result<BTreeMap<String, OutTensor>> {
        let p = format!("encoder.layer.{i}.");
        let l = format!("L{i}");
        let mut o = BTreeMap::new();
        let mut linear = |dst: &str, key: &str, o: &mut BTreeMap<String, OutTensor>| -> anyhow::Result<()> {
            let k = format!("{p}{key}");
            if src.contains_key(&k) { o.insert(format!("{l}/{dst}"), Self::lin(src, &k)?); }
            Ok(())
        };
        let mut f32v = |dst: &str, key: &str, o: &mut BTreeMap<String, OutTensor>| -> anyhow::Result<()> {
            let k = format!("{p}{key}");
            if src.contains_key(&k) { o.insert(format!("{l}/{dst}"), Self::keep_f32(src, &k)?); }
            Ok(())
        };
        linear("q_w", "attention.self.query.weight", &mut o)?;
        f32v ("q_b", "attention.self.query.bias", &mut o)?;
        linear("k_w", "attention.self.key.weight", &mut o)?;
        f32v ("k_b", "attention.self.key.bias", &mut o)?;
        linear("v_w", "attention.self.value.weight", &mut o)?;
        f32v ("v_b", "attention.self.value.bias", &mut o)?;
        linear("attn_out_w", "attention.output.dense.weight", &mut o)?;
        f32v ("attn_out_b", "attention.output.dense.bias", &mut o)?;
        f32v ("attn_ln_w", "attention.output.LayerNorm.weight", &mut o)?;
        f32v ("attn_ln_b", "attention.output.LayerNorm.bias", &mut o)?;
        linear("ffn1_w", "intermediate.dense.weight", &mut o)?;
        f32v ("ffn1_b", "intermediate.dense.bias", &mut o)?;
        linear("ffn2_w", "output.dense.weight", &mut o)?;
        f32v ("ffn2_b", "output.dense.bias", &mut o)?;
        f32v ("out_ln_w", "output.LayerNorm.weight", &mut o)?;
        f32v ("out_ln_b", "output.LayerNorm.bias", &mut o)?;
        Ok(o)
    }
}

impl Arch for Bert {
    fn name(&self) -> &'static str { "bert" }
    fn required_tensors(&self, n_layers: usize) -> Vec<String> {
        let mut v = vec![
            "embeddings.word_embeddings.weight", "embeddings.position_embeddings.weight",
            "embeddings.token_type_embeddings.weight", "embeddings.LayerNorm.weight",
            "embeddings.LayerNorm.bias",
        ].into_iter().map(String::from).collect::<Vec<_>>();
        for i in 0..n_layers {
            for s in ["attention.self.query","attention.self.key","attention.self.value",
                      "attention.output.dense","intermediate.dense","output.dense"] {
                v.push(format!("encoder.layer.{i}.{s}.weight"));
                v.push(format!("encoder.layer.{i}.{s}.bias"));
            }
            for s in ["attention.output.LayerNorm","output.LayerNorm"] {
                v.push(format!("encoder.layer.{i}.{s}.weight"));
                v.push(format!("encoder.layer.{i}.{s}.bias"));
            }
        }
        v
    }
    fn transform(&self, src: &BTreeMap<String, RawTensor>) -> anyhow::Result<BTreeMap<String, OutTensor>> {
        // Some checkpoints (BertForMaskedLM/SequenceClassification) prefix encoder tensors with
        // `bert.`; AutoModel(BertModel).state_dict() - the names export_*.py mirror - strips it.
        // Normalize so this arch matches the oracle regardless of checkpoint flavor.
        let src = strip_bert_prefix(src);
        let src = &src;
        // Infer encoder depth from the bag (bge-base/bert-base=12, all-MiniLM-L6-v2=6), like esm.
        let n_layers = Self::n_layers(src);
        anyhow::ensure!(n_layers > 0, "bert: no encoder.layer.* tensors found in source");
        // hard error on any missing required tensor
        for k in self.required_tensors(n_layers) {
            anyhow::ensure!(src.contains_key(&k), "bert: missing required source tensor {k:?}");
        }
        let mut o = BTreeMap::new();
        o.insert("emb/word_emb".into(), Self::keep_bf16(src, "embeddings.word_embeddings.weight")?);
        o.insert("emb/pos_emb".into(), Self::keep_bf16(src, "embeddings.position_embeddings.weight")?);
        o.insert("emb/type_emb".into(), Self::keep_bf16(src, "embeddings.token_type_embeddings.weight")?);
        o.insert("emb/emb_ln_w".into(), Self::keep_f32(src, "embeddings.LayerNorm.weight")?);
        o.insert("emb/emb_ln_b".into(), Self::keep_f32(src, "embeddings.LayerNorm.bias")?);
        for i in 0..n_layers { o.extend(self.transform_subset(src, i)?); }
        Ok(o)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn rt(shape: Vec<usize>, data: Vec<f32>) -> RawTensor { RawTensor { shape, data } }
    #[test]
    fn transpose_is_correct() {
        let t = rt(vec![2,3], vec![1.,2.,3., 4.,5.,6.]);
        let x = transpose2d(&t);
        assert_eq!(x.shape, vec![3,2]);
        assert_eq!(x.data, vec![1.,4., 2.,5., 3.,6.]);
    }
    #[test]
    fn transform_transposes_linears_and_keeps_norms() {
        let mut src = BTreeMap::new();
        // minimal: one layer's query weight [out=2,in=3] + bias + a LayerNorm
        src.insert("encoder.layer.0.attention.self.query.weight".into(), rt(vec![2,3], vec![1.,2.,3.,4.,5.,6.]));
        src.insert("encoder.layer.0.attention.self.query.bias".into(), rt(vec![2], vec![7.,8.]));
        let out = Bert.transform_subset(&src, 0).unwrap();
        let qw = &out["L0/q_w"];
        assert_eq!(qw.shape, vec![3,2]);          // transposed
        assert!(qw.bf16);                          // weight -> bf16
        let qb = &out["L0/q_b"];
        assert_eq!(qb.data, vec![7.,8.]);          // bias verbatim
        assert!(!qb.bf16);                         // bias -> f32
    }
    #[test]
    fn infers_layer_count() {
        let mut src = BTreeMap::new();
        src.insert("encoder.layer.0.attention.self.query.weight".into(), rt(vec![1,1], vec![1.]));
        src.insert("encoder.layer.5.output.dense.bias".into(), rt(vec![1], vec![1.]));
        assert_eq!(Bert::n_layers(&src), 6);   // all-MiniLM-L6-v2 = 6 layers
    }
}
