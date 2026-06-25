// rust/npu-weights/src/arch/esm.rs
//
// ESM-2 encoder arch (facebook/esm2_*). Mirrors scripts/export_esm.py EXACTLY:
//   - emb/word_emb               <- embeddings.word_embeddings.weight   (bf16, verbatim)
//   - L{i}/ln_attn_{w,b}         <- attention.LayerNorm.{weight,bias}    (f32, verbatim)  [pre-LN]
//   - L{i}/{q,k,v}_{w,b}         <- attention.self.{query,key,value}     (w TRANSPOSED -> [in,out] bf16; b f32)
//   - L{i}/attn_out_{w,b}        <- attention.output.dense               (w TRANSPOSED bf16; b f32)
//   - L{i}/ln_ffn_{w,b}          <- LayerNorm.{weight,bias}              (f32, verbatim)  [pre-FFN]
//   - L{i}/ffn1_{w,b}            <- intermediate.dense                   (w TRANSPOSED bf16; b f32)
//   - L{i}/ffn2_{w,b}            <- output.dense                         (w TRANSPOSED bf16; b f32)
//   - final_ln_{w,b}             <- encoder.emb_layer_norm_after.{w,b}   (f32, verbatim)
//
// RoPE is NOT baked: the script exports no rotary_embeddings.inv_freq (it is derived/applied at
// runtime), so this arch likewise emits no rotary tensor. ESM-2 attention/FFN linears all carry
// biases; LayerNorm is pre-norm (attention.LayerNorm before attn, LayerNorm before FFN) plus a
// final emb_layer_norm_after. No position/token-type embeddings (RoPE replaces them).
use std::collections::BTreeMap;
use super::{Arch, RawTensor, OutTensor, transpose2d};

pub struct Esm;

/// Return a view of `src` with any leading `esm.` prefix stripped from tensor names. If no key
/// carries the prefix the map is cloned unchanged (cheap relative to the bake's I/O).
fn strip_esm_prefix(src: &BTreeMap<String, RawTensor>) -> BTreeMap<String, RawTensor> {
    if !src.keys().any(|k| k.starts_with("esm.")) { return src.clone(); }
    src.iter()
        .map(|(k, v)| (k.strip_prefix("esm.").unwrap_or(k).to_string(), v.clone()))
        .collect()
}

impl Esm {
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

    /// Infer encoder depth from the source bag (esm2-8m=6 layers, esm2-35m=12). Counts the
    /// highest `encoder.layer.{i}.` index present.
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

    /// One layer (used by tests + the full transform). Absent source tensors are skipped here;
    /// the full `transform` hard-errors on any missing required tensor up front.
    pub fn transform_subset(&self, src: &BTreeMap<String, RawTensor>, i: usize)
        -> anyhow::Result<BTreeMap<String, OutTensor>> {
        let p = format!("encoder.layer.{i}.");
        let l = format!("L{i}");
        let mut o = BTreeMap::new();
        let linear = |dst: &str, key: &str, o: &mut BTreeMap<String, OutTensor>| -> anyhow::Result<()> {
            let k = format!("{p}{key}");
            if src.contains_key(&k) { o.insert(format!("{l}/{dst}"), Self::lin(src, &k)?); }
            Ok(())
        };
        let f32v = |dst: &str, key: &str, o: &mut BTreeMap<String, OutTensor>| -> anyhow::Result<()> {
            let k = format!("{p}{key}");
            if src.contains_key(&k) { o.insert(format!("{l}/{dst}"), Self::keep_f32(src, &k)?); }
            Ok(())
        };
        // pre-attention LayerNorm
        f32v ("ln_attn_w", "attention.LayerNorm.weight", &mut o)?;
        f32v ("ln_attn_b", "attention.LayerNorm.bias", &mut o)?;
        // attention projections
        linear("q_w", "attention.self.query.weight", &mut o)?;
        f32v ("q_b", "attention.self.query.bias", &mut o)?;
        linear("k_w", "attention.self.key.weight", &mut o)?;
        f32v ("k_b", "attention.self.key.bias", &mut o)?;
        linear("v_w", "attention.self.value.weight", &mut o)?;
        f32v ("v_b", "attention.self.value.bias", &mut o)?;
        linear("attn_out_w", "attention.output.dense.weight", &mut o)?;
        f32v ("attn_out_b", "attention.output.dense.bias", &mut o)?;
        // pre-FFN LayerNorm
        f32v ("ln_ffn_w", "LayerNorm.weight", &mut o)?;
        f32v ("ln_ffn_b", "LayerNorm.bias", &mut o)?;
        // FFN
        linear("ffn1_w", "intermediate.dense.weight", &mut o)?;
        f32v ("ffn1_b", "intermediate.dense.bias", &mut o)?;
        linear("ffn2_w", "output.dense.weight", &mut o)?;
        f32v ("ffn2_b", "output.dense.bias", &mut o)?;
        Ok(o)
    }
}

impl Arch for Esm {
    fn name(&self) -> &'static str { "esm" }
    fn required_tensors(&self, n_layers: usize) -> Vec<String> {
        let mut v = vec![
            "embeddings.word_embeddings.weight",
            "encoder.emb_layer_norm_after.weight",
            "encoder.emb_layer_norm_after.bias",
        ].into_iter().map(String::from).collect::<Vec<_>>();
        for i in 0..n_layers {
            for s in ["attention.self.query","attention.self.key","attention.self.value",
                      "attention.output.dense","intermediate.dense","output.dense"] {
                v.push(format!("encoder.layer.{i}.{s}.weight"));
                v.push(format!("encoder.layer.{i}.{s}.bias"));
            }
            for s in ["attention.LayerNorm","LayerNorm"] {
                v.push(format!("encoder.layer.{i}.{s}.weight"));
                v.push(format!("encoder.layer.{i}.{s}.bias"));
            }
        }
        v
    }
    fn transform(&self, src: &BTreeMap<String, RawTensor>) -> anyhow::Result<BTreeMap<String, OutTensor>> {
        // The on-disk EsmForMaskedLM checkpoint prefixes every encoder tensor with `esm.`
        // (e.g. `esm.encoder.layer.0...`), while AutoModel(EsmModel).state_dict() - the names
        // export_esm.py mirrors - strips it. Normalize to the stripped form so this arch matches
        // the Python oracle regardless of which checkpoint flavor the source file carries.
        let src = strip_esm_prefix(src);
        let src = &src;
        let n_layers = Self::n_layers(src);
        anyhow::ensure!(n_layers > 0, "esm: no encoder.layer.* tensors found in source");
        // hard error on any missing required tensor
        for k in self.required_tensors(n_layers) {
            anyhow::ensure!(src.contains_key(&k), "esm: missing required source tensor {k:?}");
        }
        let mut o = BTreeMap::new();
        o.insert("emb/word_emb".into(), Self::keep_bf16(src, "embeddings.word_embeddings.weight")?);
        for i in 0..n_layers { o.extend(self.transform_subset(src, i)?); }
        o.insert("final_ln_w".into(), Self::keep_f32(src, "encoder.emb_layer_norm_after.weight")?);
        o.insert("final_ln_b".into(), Self::keep_f32(src, "encoder.emb_layer_norm_after.bias")?);
        Ok(o)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn rt(shape: Vec<usize>, data: Vec<f32>) -> RawTensor { RawTensor { shape, data } }
    #[test]
    fn transform_transposes_linears_and_keeps_norms() {
        let mut src = BTreeMap::new();
        src.insert("encoder.layer.0.attention.self.query.weight".into(), rt(vec![2,3], vec![1.,2.,3.,4.,5.,6.]));
        src.insert("encoder.layer.0.attention.self.query.bias".into(), rt(vec![2], vec![7.,8.]));
        src.insert("encoder.layer.0.attention.LayerNorm.weight".into(), rt(vec![3], vec![1.,1.,1.]));
        let out = Esm.transform_subset(&src, 0).unwrap();
        let qw = &out["L0/q_w"];
        assert_eq!(qw.shape, vec![3,2]);          // transposed [out,in] -> [in,out]
        assert!(qw.bf16);                          // weight -> bf16
        let qb = &out["L0/q_b"];
        assert_eq!(qb.data, vec![7.,8.]);          // bias verbatim
        assert!(!qb.bf16);                         // bias -> f32
        let ln = &out["L0/ln_attn_w"];
        assert!(!ln.bf16);                         // LayerNorm -> f32
    }
    #[test]
    fn infers_layer_count() {
        let mut src = BTreeMap::new();
        src.insert("encoder.layer.0.attention.self.query.weight".into(), rt(vec![1,1], vec![1.]));
        src.insert("encoder.layer.5.output.dense.bias".into(), rt(vec![1], vec![1.]));
        assert_eq!(Esm::n_layers(&src), 6);
    }
}
