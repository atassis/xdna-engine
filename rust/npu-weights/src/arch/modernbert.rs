// rust/npu-weights/src/arch/modernbert.rs
//
// ModernBERT arch (answerdotai/ModernBERT-{base,large}). Mirrors scripts/convert_modernbert.py
// EXACTLY. Source = raw HF safetensors. ModernBERT is a modern encoder: BIAS-FREE, RoPE (NO learned
// position embeddings), FUSED QKV (attn.Wqkv -> 3*hidden), GeGLU MLP (mlp.Wi -> 2*intermediate
// gate+value, mlp.Wo back to hidden), pre-norm with weight-only LayerNorm (no bias). Layer 0's
// attn_norm is nn.Identity (absent in the checkpoint) since the embedding norm already normalizes.
// Only the encoder BACKBONE is baked (not the MaskedLM head/decoder), like the bert/vit arches.
// Reference npy layout (refs dir = model root `artifacts/modernbert-base`, arena names = npy paths):
//   - emb/tok_emb        <- model.embeddings.tok_embeddings.weight  ([vocab,hidden] VERBATIM, bf16)
//   - emb/norm_w         <- model.embeddings.norm.weight            (f32, weight-only LayerNorm)
//   - final_norm_w       <- model.final_norm.weight                 (f32)
//   - L{i}/attn_norm_w   <- model.layers.{i}.attn_norm.weight       (f32; ABSENT for layer 0)
//   - L{i}/qkv_w         <- model.layers.{i}.attn.Wqkv.weight       (TRANSPOSED [hidden,3*hidden] bf16)
//   - L{i}/attn_out_w    <- model.layers.{i}.attn.Wo.weight         (TRANSPOSED bf16)
//   - L{i}/mlp_norm_w    <- model.layers.{i}.mlp_norm.weight        (f32)
//   - L{i}/wi_w          <- model.layers.{i}.mlp.Wi.weight          (TRANSPOSED [hidden,2*inter] bf16)
//   - L{i}/wo_w          <- model.layers.{i}.mlp.Wo.weight          (TRANSPOSED [inter,hidden] bf16)
//
// Linears transposed to [K,N] (the x@W form the engine expects), like bert/vit. RoPE has no learned
// weights (a runtime op), so nothing rotary is baked. Layer count is inferred from the source bag.
use std::collections::BTreeMap;
use super::{Arch, RawTensor, OutTensor, transpose2d};

pub struct ModernBert;

impl ModernBert {
    /// Infer encoder depth from the source bag (highest `model.layers.{i}.` index + 1), like esm/bert.
    fn n_layers(src: &BTreeMap<String, RawTensor>) -> usize {
        let mut n = 0usize;
        for k in src.keys() {
            if let Some(rest) = k.strip_prefix("model.layers.") {
                if let Some(idx) = rest.split('.').next() {
                    if let Ok(i) = idx.parse::<usize>() { n = n.max(i + 1); }
                }
            }
        }
        n
    }

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

    /// One transformer block. attn_norm is skipped here when absent (layer 0 = Identity); `transform`
    /// hard-errors on any *required* tensor up front, so completeness is still guaranteed there.
    pub fn transform_subset(&self, src: &BTreeMap<String, RawTensor>, i: usize)
        -> anyhow::Result<BTreeMap<String, OutTensor>> {
        let p = format!("model.layers.{i}.");
        let l = format!("L{i}");
        let mut o = BTreeMap::new();
        // attn_norm: present for layers >= 1 (layer 0 is nn.Identity, no param)
        let an = format!("{p}attn_norm.weight");
        if src.contains_key(&an) { o.insert(format!("{l}/attn_norm_w"), Self::keep_f32(src, &an)?); }
        o.insert(format!("{l}/qkv_w"), Self::lin(src, &format!("{p}attn.Wqkv.weight"))?);
        o.insert(format!("{l}/attn_out_w"), Self::lin(src, &format!("{p}attn.Wo.weight"))?);
        o.insert(format!("{l}/mlp_norm_w"), Self::keep_f32(src, &format!("{p}mlp_norm.weight"))?);
        o.insert(format!("{l}/wi_w"), Self::lin(src, &format!("{p}mlp.Wi.weight"))?);
        o.insert(format!("{l}/wo_w"), Self::lin(src, &format!("{p}mlp.Wo.weight"))?);
        Ok(o)
    }
}

impl Arch for ModernBert {
    fn name(&self) -> &'static str { "modernbert" }
    fn required_tensors(&self, n_layers: usize) -> Vec<String> {
        let mut v = vec![
            "model.embeddings.tok_embeddings.weight",
            "model.embeddings.norm.weight",
            "model.final_norm.weight",
        ].into_iter().map(String::from).collect::<Vec<_>>();
        for i in 0..n_layers {
            let p = format!("model.layers.{i}.");
            // attn_norm is intentionally NOT required (absent for layer 0 = Identity).
            for s in ["attn.Wqkv.weight", "attn.Wo.weight", "mlp_norm.weight",
                      "mlp.Wi.weight", "mlp.Wo.weight"] {
                v.push(format!("{p}{s}"));
            }
        }
        v
    }
    fn transform(&self, src: &BTreeMap<String, RawTensor>) -> anyhow::Result<BTreeMap<String, OutTensor>> {
        let n_layers = Self::n_layers(src);
        anyhow::ensure!(n_layers > 0, "modernbert: no model.layers.* tensors found in source");
        for k in self.required_tensors(n_layers) {
            anyhow::ensure!(src.contains_key(&k), "modernbert: missing required source tensor {k:?}");
        }
        let mut o = BTreeMap::new();
        o.insert("emb/tok_emb".into(), Self::keep_bf16(src, "model.embeddings.tok_embeddings.weight")?);
        o.insert("emb/norm_w".into(), Self::keep_f32(src, "model.embeddings.norm.weight")?);
        o.insert("final_norm_w".into(), Self::keep_f32(src, "model.final_norm.weight")?);
        for i in 0..n_layers { o.extend(self.transform_subset(src, i)?); }
        Ok(o)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn rt(shape: Vec<usize>, data: Vec<f32>) -> RawTensor { RawTensor { shape, data } }
    #[test]
    fn block_transposes_fused_linears_keeps_norm_f32() {
        let mut src = BTreeMap::new();
        // layer 1 has attn_norm; Wqkv [out=2,in=3] transposes to [3,2]
        src.insert("model.layers.1.attn.Wqkv.weight".into(), rt(vec![2,3], vec![1.,2.,3.,4.,5.,6.]));
        src.insert("model.layers.1.attn.Wo.weight".into(), rt(vec![1,2], vec![1.,2.]));
        src.insert("model.layers.1.mlp.Wi.weight".into(), rt(vec![2,1], vec![1.,2.]));
        src.insert("model.layers.1.mlp.Wo.weight".into(), rt(vec![1,2], vec![3.,4.]));
        src.insert("model.layers.1.mlp_norm.weight".into(), rt(vec![3], vec![1.,1.,1.]));
        src.insert("model.layers.1.attn_norm.weight".into(), rt(vec![3], vec![2.,2.,2.]));
        let out = ModernBert.transform_subset(&src, 1).unwrap();
        assert_eq!(out["L1/qkv_w"].shape, vec![3,2]);   // transposed [in,out]
        assert!(out["L1/qkv_w"].bf16);
        assert!(!out["L1/mlp_norm_w"].bf16);            // norm f32
        assert!(out.contains_key("L1/attn_norm_w"));
    }
    #[test]
    fn layer0_attn_norm_skipped_when_absent() {
        let mut src = BTreeMap::new();
        src.insert("model.layers.0.attn.Wqkv.weight".into(), rt(vec![2,3], vec![1.,2.,3.,4.,5.,6.]));
        src.insert("model.layers.0.attn.Wo.weight".into(), rt(vec![1,2], vec![1.,2.]));
        src.insert("model.layers.0.mlp.Wi.weight".into(), rt(vec![2,1], vec![1.,2.]));
        src.insert("model.layers.0.mlp.Wo.weight".into(), rt(vec![1,2], vec![3.,4.]));
        src.insert("model.layers.0.mlp_norm.weight".into(), rt(vec![3], vec![1.,1.,1.]));
        // NO model.layers.0.attn_norm.weight (Identity)
        let out = ModernBert.transform_subset(&src, 0).unwrap();
        assert!(!out.contains_key("L0/attn_norm_w"));   // skipped, not an error
        assert!(out.contains_key("L0/qkv_w"));
    }
    #[test]
    fn infers_layer_count() {
        let mut src = BTreeMap::new();
        src.insert("model.layers.0.attn.Wo.weight".into(), rt(vec![1,1], vec![1.]));
        src.insert("model.layers.21.mlp.Wo.weight".into(), rt(vec![1,1], vec![1.]));
        assert_eq!(ModernBert::n_layers(&src), 22);   // ModernBERT-base = 22 layers
    }
}
