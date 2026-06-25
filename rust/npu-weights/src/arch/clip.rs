// rust/npu-weights/src/arch/clip.rs
//
// CLIP arch (transformers CLIPModel; baked from laion/CLIP-ViT-B-32-laion2B-s34B-b79K, the
// openai-architecture CLIP that ships model.safetensors - openai/clip-vit-base-patch32 ships only
// pytorch_model.bin, which the loader does not read). Mirrors scripts/convert_clip.py EXACTLY.
//
// Two towers, each a stack of the SAME CLIPEncoderLayer (q/k/v/out_proj WITH bias, layer_norm1/2,
// mlp.fc1/fc2 with bias, gelu), plus joint projections + a logit_scale scalar. Reference npy layout
// (refs dir = model root `artifacts/clip-vit-b32`, arena names = npy paths):
//   TEXT (hidden 512, 12 layers, ffn 2048):
//     text/tok_emb            <- text_model.embeddings.token_embedding.weight ([vocab,512] VERBATIM bf16)
//     text/pos_emb            <- text_model.embeddings.position_embedding.weight ([77,512] bf16)
//     text/final_ln.{weight,bias} <- text_model.final_layer_norm.{w,b}        (f32)
//     text/L{i}/{q,k,v,out}.weight <- self_attn.{q,k,v,out}_proj.weight       (TRANSPOSED bf16)
//     text/L{i}/{q,k,v,out}.bias   <- ...bias                                 (f32)
//     text/L{i}/ln1.{weight,bias}  <- layer_norm1.{w,b}                       (f32)
//     text/L{i}/ln2.{weight,bias}  <- layer_norm2.{w,b}                       (f32)
//     text/L{i}/fc1.weight <- mlp.fc1.weight (TR bf16); fc1.bias (f32); fc2.weight/bias likewise
//   VISION (hidden 768, 12 layers, ffn 3072, patch32):
//     vision/cls_emb          <- vision_model.embeddings.class_embedding ([768] bf16)
//     vision/patch_proj.weight<- vision_model.embeddings.patch_embedding.weight (Conv2d [768,3,32,32]
//                                im2col-flatten + TRANSPOSE -> [K,N] bf16, like vit)
//     vision/pos_emb          <- vision_model.embeddings.position_embedding.weight ([50,768] bf16)
//     vision/pre_ln.{weight,bias}  <- vision_model.pre_layrnorm.{w,b}         (f32) [note HF typo]
//     vision/post_ln.{weight,bias} <- vision_model.post_layernorm.{w,b}       (f32)
//     vision/L{i}/...         <- same CLIPEncoderLayer layout as text
//   JOINT:
//     text_projection         <- text_projection.weight   (Linear bias-free, TRANSPOSED bf16 [512,512])
//     visual_projection       <- visual_projection.weight (Linear bias-free, TRANSPOSED bf16 [768,512])
//     logit_scale             <- logit_scale              (scalar -> [1] f32)
//
// Per-tower layer counts are inferred from the source bag (so other CLIP sizes ride this arch). The
// vision patch-embed conv flatten/transpose is a faithful deterministic mirror (baked, not parked).
use std::collections::BTreeMap;
use super::{Arch, RawTensor, OutTensor, transpose2d};

pub struct Clip;

impl Clip {
    /// Highest `{prefix}encoder.layers.{i}.` index + 1 in the source bag.
    fn tower_layers(src: &BTreeMap<String, RawTensor>, prefix: &str) -> usize {
        let head = format!("{prefix}encoder.layers.");
        let mut n = 0usize;
        for k in src.keys() {
            if let Some(rest) = k.strip_prefix(&head) {
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
    /// Flatten an N-D / scalar tensor to a 1-D bf16 vector of its element count.
    fn flat_bf16(src: &BTreeMap<String, RawTensor>, k: &str) -> anyhow::Result<OutTensor> {
        let t = Self::w(src, k)?;
        Ok(OutTensor { shape: vec![t.data.len()], data: t.data, bf16: true })
    }
    /// 2-D embedding table kept VERBATIM (row-major), bf16.
    fn keep2d_bf16(src: &BTreeMap<String, RawTensor>, k: &str) -> anyhow::Result<OutTensor> {
        let t = Self::w(src, k)?;
        Ok(OutTensor { shape: t.shape, data: t.data, bf16: true })
    }
    /// Conv2d patch-embed weight [out,c,ph,pw] -> im2col-flatten -> TRANSPOSE [K,N], bf16 (like vit).
    fn patch_proj(src: &BTreeMap<String, RawTensor>, k: &str) -> anyhow::Result<OutTensor> {
        let t = Self::w(src, k)?;
        anyhow::ensure!(t.shape.len() == 4, "clip patch_proj {k:?}: expected 4D, got {:?}", t.shape);
        let out = t.shape[0];
        let flat = t.shape[1] * t.shape[2] * t.shape[3];
        let tr = transpose2d(&RawTensor { shape: vec![out, flat], data: t.data });
        Ok(OutTensor { shape: tr.shape, data: tr.data, bf16: true })
    }

    /// One CLIPEncoderLayer (shared by both towers). `tower` is the output namespace ("text"/"vision").
    fn emit_layer(&self, src: &BTreeMap<String, RawTensor>, tower: &str, prefix: &str, i: usize,
                  o: &mut BTreeMap<String, OutTensor>) -> anyhow::Result<()> {
        let p = format!("{prefix}encoder.layers.{i}.");
        let d = format!("{tower}/L{i}");
        for (dst, s) in [("q","q_proj"),("k","k_proj"),("v","v_proj"),("out","out_proj")] {
            o.insert(format!("{d}/{dst}.weight"), Self::lin(src, &format!("{p}self_attn.{s}.weight"))?);
            o.insert(format!("{d}/{dst}.bias"), Self::keep_f32(src, &format!("{p}self_attn.{s}.bias"))?);
        }
        for (dst, s) in [("ln1","layer_norm1"),("ln2","layer_norm2")] {
            o.insert(format!("{d}/{dst}.weight"), Self::keep_f32(src, &format!("{p}{s}.weight"))?);
            o.insert(format!("{d}/{dst}.bias"), Self::keep_f32(src, &format!("{p}{s}.bias"))?);
        }
        for (dst, s) in [("fc1","mlp.fc1"),("fc2","mlp.fc2")] {
            o.insert(format!("{d}/{dst}.weight"), Self::lin(src, &format!("{p}{s}.weight"))?);
            o.insert(format!("{d}/{dst}.bias"), Self::keep_f32(src, &format!("{p}{s}.bias"))?);
        }
        Ok(())
    }

    fn layer_required(prefix: &str, n: usize) -> Vec<String> {
        let mut v = Vec::new();
        for i in 0..n {
            let p = format!("{prefix}encoder.layers.{i}.");
            for s in ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.out_proj",
                      "mlp.fc1","mlp.fc2"] {
                v.push(format!("{p}{s}.weight")); v.push(format!("{p}{s}.bias"));
            }
            for s in ["layer_norm1","layer_norm2"] {
                v.push(format!("{p}{s}.weight")); v.push(format!("{p}{s}.bias"));
            }
        }
        v
    }
}

impl Arch for Clip {
    fn name(&self) -> &'static str { "clip" }
    fn required_tensors(&self, _n_layers: usize) -> Vec<String> {
        // Tower anchors + the per-tower layer counts are resolved in `transform`; here we list the
        // always-present non-layer anchors so the guard catches a wrong/empty checkpoint early.
        ["text_model.embeddings.token_embedding.weight",
         "text_model.embeddings.position_embedding.weight",
         "text_model.final_layer_norm.weight",
         "vision_model.embeddings.class_embedding",
         "vision_model.embeddings.patch_embedding.weight",
         "vision_model.embeddings.position_embedding.weight",
         "vision_model.pre_layrnorm.weight", "vision_model.post_layernorm.weight",
         "text_projection.weight", "visual_projection.weight", "logit_scale"]
            .into_iter().map(String::from).collect()
    }
    fn transform(&self, src: &BTreeMap<String, RawTensor>) -> anyhow::Result<BTreeMap<String, OutTensor>> {
        let tl = Self::tower_layers(src, "text_model.");
        let vl = Self::tower_layers(src, "vision_model.");
        anyhow::ensure!(tl > 0 && vl > 0, "clip: missing text/vision encoder layers (text={tl}, vision={vl})");
        let mut req = self.required_tensors(0);
        req.extend(Self::layer_required("text_model.", tl));
        req.extend(Self::layer_required("vision_model.", vl));
        for k in &req {
            anyhow::ensure!(src.contains_key(k), "clip: missing required source tensor {k:?}");
        }
        let mut o = BTreeMap::new();
        // text tower
        o.insert("text/tok_emb".into(), Self::keep2d_bf16(src, "text_model.embeddings.token_embedding.weight")?);
        o.insert("text/pos_emb".into(), Self::keep2d_bf16(src, "text_model.embeddings.position_embedding.weight")?);
        o.insert("text/final_ln.weight".into(), Self::keep_f32(src, "text_model.final_layer_norm.weight")?);
        o.insert("text/final_ln.bias".into(), Self::keep_f32(src, "text_model.final_layer_norm.bias")?);
        for i in 0..tl { self.emit_layer(src, "text", "text_model.", i, &mut o)?; }
        // vision tower
        o.insert("vision/cls_emb".into(), Self::flat_bf16(src, "vision_model.embeddings.class_embedding")?);
        o.insert("vision/patch_proj.weight".into(), Self::patch_proj(src, "vision_model.embeddings.patch_embedding.weight")?);
        o.insert("vision/pos_emb".into(), Self::keep2d_bf16(src, "vision_model.embeddings.position_embedding.weight")?);
        o.insert("vision/pre_ln.weight".into(), Self::keep_f32(src, "vision_model.pre_layrnorm.weight")?);
        o.insert("vision/pre_ln.bias".into(), Self::keep_f32(src, "vision_model.pre_layrnorm.bias")?);
        o.insert("vision/post_ln.weight".into(), Self::keep_f32(src, "vision_model.post_layernorm.weight")?);
        o.insert("vision/post_ln.bias".into(), Self::keep_f32(src, "vision_model.post_layernorm.bias")?);
        for i in 0..vl { self.emit_layer(src, "vision", "vision_model.", i, &mut o)?; }
        // joint projections (bias-free) + logit scale
        o.insert("text_projection".into(), Self::lin(src, "text_projection.weight")?);
        o.insert("visual_projection".into(), Self::lin(src, "visual_projection.weight")?);
        o.insert("logit_scale".into(),
                 OutTensor { shape: vec![1], data: Self::w(src, "logit_scale")?.data, bf16: false });
        Ok(o)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn rt(shape: Vec<usize>, data: Vec<f32>) -> RawTensor { RawTensor { shape, data } }
    #[test]
    fn infers_per_tower_layer_counts() {
        let mut src = BTreeMap::new();
        src.insert("text_model.encoder.layers.11.x".into(), rt(vec![1], vec![1.]));
        src.insert("vision_model.encoder.layers.11.y".into(), rt(vec![1], vec![1.]));
        assert_eq!(Clip::tower_layers(&src, "text_model."), 12);
        assert_eq!(Clip::tower_layers(&src, "vision_model."), 12);
    }
    #[test]
    fn emit_layer_transposes_linears_keeps_norms_and_bias_f32() {
        let mut src = BTreeMap::new();
        let p = "text_model.encoder.layers.0.";
        src.insert(format!("{p}self_attn.q_proj.weight"), rt(vec![2,3], vec![1.,2.,3.,4.,5.,6.]));
        src.insert(format!("{p}self_attn.q_proj.bias"), rt(vec![2], vec![7.,8.]));
        src.insert(format!("{p}self_attn.k_proj.weight"), rt(vec![2,3], vec![1.;6].to_vec()));
        src.insert(format!("{p}self_attn.k_proj.bias"), rt(vec![2], vec![0.,0.]));
        src.insert(format!("{p}self_attn.v_proj.weight"), rt(vec![2,3], vec![1.;6].to_vec()));
        src.insert(format!("{p}self_attn.v_proj.bias"), rt(vec![2], vec![0.,0.]));
        src.insert(format!("{p}self_attn.out_proj.weight"), rt(vec![2,2], vec![1.;4].to_vec()));
        src.insert(format!("{p}self_attn.out_proj.bias"), rt(vec![2], vec![0.,0.]));
        src.insert(format!("{p}layer_norm1.weight"), rt(vec![3], vec![1.,1.,1.]));
        src.insert(format!("{p}layer_norm1.bias"), rt(vec![3], vec![0.,0.,0.]));
        src.insert(format!("{p}layer_norm2.weight"), rt(vec![3], vec![1.,1.,1.]));
        src.insert(format!("{p}layer_norm2.bias"), rt(vec![3], vec![0.,0.,0.]));
        src.insert(format!("{p}mlp.fc1.weight"), rt(vec![4,3], vec![1.;12].to_vec()));
        src.insert(format!("{p}mlp.fc1.bias"), rt(vec![4], vec![0.;4].to_vec()));
        src.insert(format!("{p}mlp.fc2.weight"), rt(vec![3,4], vec![1.;12].to_vec()));
        src.insert(format!("{p}mlp.fc2.bias"), rt(vec![3], vec![0.,0.,0.]));
        let mut o = BTreeMap::new();
        Clip.emit_layer(&src, "text", "text_model.", 0, &mut o).unwrap();
        assert_eq!(o["text/L0/q.weight"].shape, vec![3,2]);  // transposed
        assert!(o["text/L0/q.weight"].bf16);
        assert_eq!(o["text/L0/q.bias"].data, vec![7.,8.]);
        assert!(!o["text/L0/q.bias"].bf16);                   // bias f32
        assert!(!o["text/L0/ln1.weight"].bf16);               // norm f32
        assert_eq!(o["text/L0/fc1.weight"].shape, vec![3,4]); // transposed [in,out]
    }
    #[test]
    fn patch_proj_flatten_transpose() {
        let mut src = BTreeMap::new();
        src.insert("w".into(), rt(vec![2,1,2,2], vec![1.,2.,3.,4., 5.,6.,7.,8.]));
        let o = Clip::patch_proj(&src, "w").unwrap();
        assert_eq!(o.shape, vec![4,2]);
        assert_eq!(o.data, vec![1.,5., 2.,6., 3.,7., 4.,8.]);
    }
}
