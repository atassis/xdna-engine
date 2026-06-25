// rust/npu-weights/src/arch/vit.rs
//
// ViT-base arch (google/vit-base-patch16-224). Mirrors scripts/convert_vit.py EXACTLY. Source = raw HF
// safetensors (vit.*). Reference npy layout (verify --refs dir is `artifacts/vit-base`, so arena names
// equal npy paths relative to it):
//   - patch_proj.weight   <- vit.embeddings.patch_embeddings.projection.weight  (Conv2d [768,3,16,16])
//                            FAITHFUL im2col-flatten: reshape row-major to [768, 3*16*16=768] then
//                            TRANSPOSE -> [K=768, N=768] bf16. (The oracle does .reshape(768,-1).T.)
//   - patch_proj.bias     <- ...projection.bias                                 (f32, verbatim)
//   - cls_token           <- vit.embeddings.cls_token  ([1,1,768] -> [768] flat, bf16)
//   - pos_emb             <- vit.embeddings.position_embeddings ([1,197,768] -> [197,768], bf16)
//   - ln_final.{weight,bias} <- vit.layernorm.{weight,bias}                     (f32, verbatim)
//   - classifier.weight   <- classifier.weight  (TRANSPOSED [768,1000] bf16)
//   - classifier.bias     <- classifier.bias                                   (f32, verbatim)
//   - L{i}/{q,k,v}.weight <- attention.attention.{query,key,value}.weight       (TRANSPOSED bf16)
//   - L{i}/{q,k,v}.bias   <- ...{query,key,value}.bias                          (f32)
//   - L{i}/attn_out.weight<- attention.output.dense.weight                      (TRANSPOSED bf16)
//   - L{i}/attn_out.bias  <- attention.output.dense.bias                        (f32)
//   - L{i}/ln_before.{weight,bias} <- layernorm_before.{w,b}                    (f32) [pre-attn]
//   - L{i}/inter.weight   <- intermediate.dense.weight                          (TRANSPOSED bf16)
//   - L{i}/inter.bias     <- intermediate.dense.bias                            (f32)
//   - L{i}/out.weight     <- output.dense.weight                               (TRANSPOSED bf16)
//   - L{i}/out.bias       <- output.dense.bias                                  (f32)
//   - L{i}/ln_after.{weight,bias} <- layernorm_after.{w,b}                      (f32) [pre-FFN]
//
// ViT-base: 768/12/12/3072, pre-norm, gelu, patch16. The patch-embed conv2d is flattened/transposed
// into a GEMM weight exactly as the oracle does (a faithful, deterministic mirror - NOT an undecided
// packing choice), so it is baked here rather than parked.
use std::collections::BTreeMap;
use super::{Arch, RawTensor, OutTensor, transpose2d};

pub struct Vit;

const N_LAYERS: usize = 12;

impl Vit {
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
    /// Reshape an N-D tensor to the given 2-D shape (row-major, total length preserved), bf16.
    fn reshape_bf16(src: &BTreeMap<String, RawTensor>, k: &str, shape: Vec<usize>) -> anyhow::Result<OutTensor> {
        let t = Self::w(src, k)?;
        let n: usize = shape.iter().product();
        anyhow::ensure!(t.data.len() == n, "vit reshape {k:?}: {} elems != target {n}", t.data.len());
        Ok(OutTensor { shape, data: t.data, bf16: true })
    }
    /// Conv2d patch-embed weight [out,c,ph,pw] -> im2col-flatten [out, c*ph*pw] -> TRANSPOSE [K,N], bf16.
    fn patch_proj(src: &BTreeMap<String, RawTensor>, k: &str) -> anyhow::Result<OutTensor> {
        let t = Self::w(src, k)?;
        anyhow::ensure!(t.shape.len() == 4, "vit patch_proj {k:?}: expected 4D conv weight, got {:?}", t.shape);
        let out = t.shape[0];
        let flat = t.shape[1] * t.shape[2] * t.shape[3];
        // row-major [out, flat] is already the memory layout; just relabel the shape, then transpose.
        let reshaped = RawTensor { shape: vec![out, flat], data: t.data };
        let tr = transpose2d(&reshaped);          // [flat, out] = [K, N]
        Ok(OutTensor { shape: tr.shape, data: tr.data, bf16: true })
    }

    /// One transformer block (used by tests + the full transform). Absent source tensors are skipped
    /// here; `transform` hard-errors on any missing required tensor up front.
    pub fn transform_subset(&self, src: &BTreeMap<String, RawTensor>, i: usize)
        -> anyhow::Result<BTreeMap<String, OutTensor>> {
        let p = format!("vit.encoder.layer.{i}.");
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
        f32v ("ln_before.weight", "layernorm_before.weight", &mut o)?;
        f32v ("ln_before.bias", "layernorm_before.bias", &mut o)?;
        f32v ("ln_after.weight", "layernorm_after.weight", &mut o)?;
        f32v ("ln_after.bias", "layernorm_after.bias", &mut o)?;
        // FFN (gelu)
        linear("inter.weight", "intermediate.dense.weight", &mut o)?;
        f32v ("inter.bias", "intermediate.dense.bias", &mut o)?;
        linear("out.weight", "output.dense.weight", &mut o)?;
        f32v ("out.bias", "output.dense.bias", &mut o)?;
        Ok(o)
    }
}

impl Arch for Vit {
    fn name(&self) -> &'static str { "vit" }
    fn required_tensors(&self, n_layers: usize) -> Vec<String> {
        let mut v = vec![
            "vit.embeddings.patch_embeddings.projection.weight",
            "vit.embeddings.patch_embeddings.projection.bias",
            "vit.embeddings.cls_token",
            "vit.embeddings.position_embeddings",
            "vit.layernorm.weight", "vit.layernorm.bias",
            "classifier.weight", "classifier.bias",
        ].into_iter().map(String::from).collect::<Vec<_>>();
        for i in 0..n_layers {
            let p = format!("vit.encoder.layer.{i}.");
            for s in ["attention.attention.query","attention.attention.key","attention.attention.value",
                      "attention.output.dense","intermediate.dense","output.dense"] {
                v.push(format!("{p}{s}.weight"));
                v.push(format!("{p}{s}.bias"));
            }
            for s in ["layernorm_before","layernorm_after"] {
                v.push(format!("{p}{s}.weight"));
                v.push(format!("{p}{s}.bias"));
            }
        }
        v
    }
    fn transform(&self, src: &BTreeMap<String, RawTensor>) -> anyhow::Result<BTreeMap<String, OutTensor>> {
        for k in self.required_tensors(N_LAYERS) {
            anyhow::ensure!(src.contains_key(&k), "vit: missing required source tensor {k:?}");
        }
        let mut o = BTreeMap::new();
        // patch embed: Conv2d -> im2col-flatten -> transposed GEMM weight [K,N]
        o.insert("patch_proj.weight".into(),
                 Self::patch_proj(src, "vit.embeddings.patch_embeddings.projection.weight")?);
        o.insert("patch_proj.bias".into(),
                 Self::keep_f32(src, "vit.embeddings.patch_embeddings.projection.bias")?);
        // cls_token [1,1,768] -> [768]; pos_emb [1,197,768] -> [197,768]
        let cls = Self::w(src, "vit.embeddings.cls_token")?;
        o.insert("cls_token".into(), Self::reshape_bf16(src, "vit.embeddings.cls_token", vec![cls.data.len()])?);
        let pos = Self::w(src, "vit.embeddings.position_embeddings")?;
        let (np, hp) = (pos.shape[pos.shape.len()-2], pos.shape[pos.shape.len()-1]);
        o.insert("pos_emb".into(), Self::reshape_bf16(src, "vit.embeddings.position_embeddings", vec![np, hp])?);
        o.insert("ln_final.weight".into(), Self::keep_f32(src, "vit.layernorm.weight")?);
        o.insert("ln_final.bias".into(), Self::keep_f32(src, "vit.layernorm.bias")?);
        o.insert("classifier.weight".into(), Self::lin(src, "classifier.weight")?);   // [768,1000]
        o.insert("classifier.bias".into(), Self::keep_f32(src, "classifier.bias")?);
        for i in 0..N_LAYERS { o.extend(self.transform_subset(src, i)?); }
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
        let o = Vit::patch_proj(&src, "w").unwrap();
        assert_eq!(o.shape, vec![4,2]);           // [K=c*ph*pw, N=out]
        assert!(o.bf16);
        // column 0 = first out-channel's 4 weights, column 1 = second
        assert_eq!(o.data, vec![1.,5., 2.,6., 3.,7., 4.,8.]);
    }
    #[test]
    fn block_transposes_linears_keeps_norms() {
        let mut src = BTreeMap::new();
        src.insert("vit.encoder.layer.0.attention.attention.query.weight".into(), rt(vec![2,3], vec![1.,2.,3.,4.,5.,6.]));
        src.insert("vit.encoder.layer.0.attention.attention.query.bias".into(), rt(vec![2], vec![7.,8.]));
        src.insert("vit.encoder.layer.0.layernorm_before.weight".into(), rt(vec![3], vec![1.,1.,1.]));
        let out = Vit.transform_subset(&src, 0).unwrap();
        assert_eq!(out["L0/q.weight"].shape, vec![3,2]);   // transposed
        assert!(out["L0/q.weight"].bf16);
        assert_eq!(out["L0/q.bias"].data, vec![7.,8.]);
        assert!(!out["L0/q.bias"].bf16);
        assert!(!out["L0/ln_before.weight"].bf16);
    }
}
