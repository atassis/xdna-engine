// rust/npu-weights/src/arch/dinov2.rs
//
// DINOv2 backbone arch (facebook/dinov2-{small,base,large}). Mirrors scripts/convert_dinov2.py
// EXACTLY. Source = raw HF safetensors (NO `vit.` prefix). Reference npy layout (refs dir is the
// model root `artifacts/dinov2-base`, so arena names equal npy paths relative to it):
//   - patch_proj.weight   <- embeddings.patch_embeddings.projection.weight (Conv2d [768,3,14,14])
//                            FAITHFUL im2col-flatten: reshape row-major to [768, 3*14*14] then
//                            TRANSPOSE -> [K,N] bf16 (the oracle does .reshape(out,-1).T).
//   - patch_proj.bias     <- ...projection.bias                            (f32, verbatim)
//   - cls_token           <- embeddings.cls_token  ([1,1,768] -> [768] flat, bf16)
//   - mask_token          <- embeddings.mask_token ([1,768]  -> [768] flat, bf16) [learnable param]
//   - pos_emb             <- embeddings.position_embeddings ([1,1370,768] -> [1370,768], bf16)
//   - ln_final.{weight,bias} <- layernorm.{weight,bias}                    (f32, verbatim)
//   - L{i}/{q,k,v}.weight <- attention.attention.{query,key,value}.weight  (TRANSPOSED bf16)
//   - L{i}/{q,k,v}.bias   <- ...{query,key,value}.bias                     (f32)
//   - L{i}/attn_out.weight<- attention.output.dense.weight                 (TRANSPOSED bf16)
//   - L{i}/attn_out.bias  <- attention.output.dense.bias                   (f32)
//   - L{i}/norm1.{weight,bias} <- norm1.{w,b}                              (f32) [pre-attn]
//   - L{i}/norm2.{weight,bias} <- norm2.{w,b}                              (f32) [pre-FFN]
//   - L{i}/ls1            <- layer_scale1.lambda1                          (f32) [per-channel scale]
//   - L{i}/ls2            <- layer_scale2.lambda1                          (f32) [per-channel scale]
//   - L{i}/fc1.weight     <- mlp.fc1.weight                               (TRANSPOSED bf16)
//   - L{i}/fc1.bias       <- mlp.fc1.bias                                  (f32)
//   - L{i}/fc2.weight     <- mlp.fc2.weight                               (TRANSPOSED bf16)
//   - L{i}/fc2.bias       <- mlp.fc2.bias                                  (f32)
//
// DINOv2 differs from vanilla ViT: no `vit.` prefix; per-block LayerScale (ls1/ls2, baked verbatim
// f32); norm1/norm2 instead of layernorm_before/after; mlp.fc1/fc2 instead of intermediate/output;
// a learnable mask_token; and NO classifier (backbone Dinov2Model). Layer count is inferred from the
// source bag (small=6, base=12, large=24), like esm/bert. The patch-embed conv2d flatten/transpose
// is a faithful deterministic mirror (baked here, not parked).
use std::collections::BTreeMap;
use super::{Arch, RawTensor, OutTensor, transpose2d};

pub struct Dinov2;

impl Dinov2 {
    /// Infer encoder depth from the source bag (highest `encoder.layer.{i}.` index + 1), like esm/bert.
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
    /// Reshape an N-D tensor to the given shape (row-major, total length preserved), bf16.
    fn reshape_bf16(src: &BTreeMap<String, RawTensor>, k: &str, shape: Vec<usize>) -> anyhow::Result<OutTensor> {
        let t = Self::w(src, k)?;
        let n: usize = shape.iter().product();
        anyhow::ensure!(t.data.len() == n, "dinov2 reshape {k:?}: {} elems != target {n}", t.data.len());
        Ok(OutTensor { shape, data: t.data, bf16: true })
    }
    /// Conv2d patch-embed weight [out,c,ph,pw] -> im2col-flatten [out, c*ph*pw] -> TRANSPOSE [K,N], bf16.
    fn patch_proj(src: &BTreeMap<String, RawTensor>, k: &str) -> anyhow::Result<OutTensor> {
        let t = Self::w(src, k)?;
        anyhow::ensure!(t.shape.len() == 4, "dinov2 patch_proj {k:?}: expected 4D conv weight, got {:?}", t.shape);
        let out = t.shape[0];
        let flat = t.shape[1] * t.shape[2] * t.shape[3];
        let reshaped = RawTensor { shape: vec![out, flat], data: t.data };
        let tr = transpose2d(&reshaped);          // [flat, out] = [K, N]
        Ok(OutTensor { shape: tr.shape, data: tr.data, bf16: true })
    }

    /// One transformer block (used by tests + the full transform). Absent source tensors are skipped
    /// here; `transform` hard-errors on any missing required tensor up front.
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
        // attention projections
        for (dst, src_name) in [("q","query"),("k","key"),("v","value")] {
            linear(&format!("{dst}.weight"), &format!("attention.attention.{src_name}.weight"), &mut o)?;
            f32v(&format!("{dst}.bias"), &format!("attention.attention.{src_name}.bias"), &mut o)?;
        }
        linear("attn_out.weight", "attention.output.dense.weight", &mut o)?;
        f32v ("attn_out.bias", "attention.output.dense.bias", &mut o)?;
        // pre-attn / pre-FFN LayerNorms
        f32v ("norm1.weight", "norm1.weight", &mut o)?;
        f32v ("norm1.bias", "norm1.bias", &mut o)?;
        f32v ("norm2.weight", "norm2.weight", &mut o)?;
        f32v ("norm2.bias", "norm2.bias", &mut o)?;
        // LayerScale (per-channel learnable scale applied after attn / after MLP) - verbatim f32
        f32v ("ls1", "layer_scale1.lambda1", &mut o)?;
        f32v ("ls2", "layer_scale2.lambda1", &mut o)?;
        // MLP (gelu)
        linear("fc1.weight", "mlp.fc1.weight", &mut o)?;
        f32v ("fc1.bias", "mlp.fc1.bias", &mut o)?;
        linear("fc2.weight", "mlp.fc2.weight", &mut o)?;
        f32v ("fc2.bias", "mlp.fc2.bias", &mut o)?;
        Ok(o)
    }
}

impl Arch for Dinov2 {
    fn name(&self) -> &'static str { "dinov2" }
    fn required_tensors(&self, n_layers: usize) -> Vec<String> {
        let mut v = vec![
            "embeddings.patch_embeddings.projection.weight",
            "embeddings.patch_embeddings.projection.bias",
            "embeddings.cls_token",
            "embeddings.mask_token",
            "embeddings.position_embeddings",
            "layernorm.weight", "layernorm.bias",
        ].into_iter().map(String::from).collect::<Vec<_>>();
        for i in 0..n_layers {
            let p = format!("encoder.layer.{i}.");
            for s in ["attention.attention.query","attention.attention.key","attention.attention.value",
                      "attention.output.dense","mlp.fc1","mlp.fc2"] {
                v.push(format!("{p}{s}.weight"));
                v.push(format!("{p}{s}.bias"));
            }
            for s in ["norm1","norm2"] {
                v.push(format!("{p}{s}.weight"));
                v.push(format!("{p}{s}.bias"));
            }
            v.push(format!("{p}layer_scale1.lambda1"));
            v.push(format!("{p}layer_scale2.lambda1"));
        }
        v
    }
    fn transform(&self, src: &BTreeMap<String, RawTensor>) -> anyhow::Result<BTreeMap<String, OutTensor>> {
        let n_layers = Self::n_layers(src);
        anyhow::ensure!(n_layers > 0, "dinov2: no encoder.layer.* tensors found in source");
        for k in self.required_tensors(n_layers) {
            anyhow::ensure!(src.contains_key(&k), "dinov2: missing required source tensor {k:?}");
        }
        let mut o = BTreeMap::new();
        // patch embed: Conv2d -> im2col-flatten -> transposed GEMM weight [K,N]
        o.insert("patch_proj.weight".into(),
                 Self::patch_proj(src, "embeddings.patch_embeddings.projection.weight")?);
        o.insert("patch_proj.bias".into(),
                 Self::keep_f32(src, "embeddings.patch_embeddings.projection.bias")?);
        // cls_token [1,1,768] -> [768]; mask_token [1,768] -> [768]; pos_emb [1,1370,768] -> [1370,768]
        let cls = Self::w(src, "embeddings.cls_token")?;
        o.insert("cls_token".into(), Self::reshape_bf16(src, "embeddings.cls_token", vec![cls.data.len()])?);
        let mask = Self::w(src, "embeddings.mask_token")?;
        o.insert("mask_token".into(), Self::reshape_bf16(src, "embeddings.mask_token", vec![mask.data.len()])?);
        let pos = Self::w(src, "embeddings.position_embeddings")?;
        let (np, hp) = (pos.shape[pos.shape.len()-2], pos.shape[pos.shape.len()-1]);
        o.insert("pos_emb".into(), Self::reshape_bf16(src, "embeddings.position_embeddings", vec![np, hp])?);
        o.insert("ln_final.weight".into(), Self::keep_f32(src, "layernorm.weight")?);
        o.insert("ln_final.bias".into(), Self::keep_f32(src, "layernorm.bias")?);
        for i in 0..n_layers { o.extend(self.transform_subset(src, i)?); }
        Ok(o)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn rt(shape: Vec<usize>, data: Vec<f32>) -> RawTensor { RawTensor { shape, data } }
    #[test]
    fn patch_proj_flatten_transpose() {
        // [out=2, c=1, ph=2, pw=2] -> flatten [2,4] -> transpose [4,2]
        let mut src = BTreeMap::new();
        src.insert("w".into(), rt(vec![2,1,2,2], vec![1.,2.,3.,4.,  5.,6.,7.,8.]));
        let o = Dinov2::patch_proj(&src, "w").unwrap();
        assert_eq!(o.shape, vec![4,2]);           // [K=c*ph*pw, N=out]
        assert!(o.bf16);
        assert_eq!(o.data, vec![1.,5., 2.,6., 3.,7., 4.,8.]);
    }
    #[test]
    fn block_transposes_linears_keeps_scales_f32() {
        let mut src = BTreeMap::new();
        src.insert("encoder.layer.0.attention.attention.query.weight".into(), rt(vec![2,3], vec![1.,2.,3.,4.,5.,6.]));
        src.insert("encoder.layer.0.attention.attention.query.bias".into(), rt(vec![2], vec![7.,8.]));
        src.insert("encoder.layer.0.layer_scale1.lambda1".into(), rt(vec![2], vec![0.1,0.2]));
        src.insert("encoder.layer.0.norm1.weight".into(), rt(vec![3], vec![1.,1.,1.]));
        let out = Dinov2.transform_subset(&src, 0).unwrap();
        assert_eq!(out["L0/q.weight"].shape, vec![3,2]);   // transposed
        assert!(out["L0/q.weight"].bf16);
        assert_eq!(out["L0/q.bias"].data, vec![7.,8.]);
        assert!(!out["L0/q.bias"].bf16);
        assert_eq!(out["L0/ls1"].data, vec![0.1,0.2]);     // LayerScale verbatim
        assert!(!out["L0/ls1"].bf16);                       // and f32
        assert!(!out["L0/norm1.weight"].bf16);
    }
    #[test]
    fn infers_layer_count() {
        let mut src = BTreeMap::new();
        src.insert("encoder.layer.0.attention.attention.query.weight".into(), rt(vec![1,1], vec![1.]));
        src.insert("encoder.layer.11.mlp.fc2.bias".into(), rt(vec![1], vec![1.]));
        assert_eq!(Dinov2::n_layers(&src), 12);   // dinov2-base = 12 layers
    }
}
