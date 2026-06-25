// rust/npu-weights/src/arch/whisper.rs
//
// Whisper encoder arch (openai/whisper-small). Mirrors scripts/extract_whisper_encoder.py EXACTLY.
// Source = raw HF safetensors names (model.encoder.*). Reference npy layout (the verify --refs dir is
// `artifacts/whisper-small`, so arena names equal the npy paths relative to it):
//   - conv/conv1.weight          <- model.encoder.conv1.weight     (3D [768,80,3], VERBATIM, bf16)
//   - conv/conv1.bias            <- model.encoder.conv1.bias        (f32, verbatim)
//   - conv/conv2.weight          <- model.encoder.conv2.weight     (3D [768,768,3], VERBATIM, bf16)
//   - conv/conv2.bias            <- model.encoder.conv2.bias        (f32, verbatim)
//   - conv/embed_positions       <- model.encoder.embed_positions.weight ([1500,768], VERBATIM, bf16)
//   - L{i}/{q,k,v,out}.weight    <- self_attn.{q,k,v,out}_proj.weight (TRANSPOSED [in,out] bf16)
//   - L{i}/{q,k,v,out}.bias      <- self_attn.{q,k,v,out}_proj.bias   (f32; k_proj has NO bias -> zeros[768])
//   - L{i}/fc1.weight, fc2.weight<- fc1/fc2.weight                  (TRANSPOSED bf16)
//   - L{i}/fc1.bias, fc2.bias    <- fc1/fc2.bias                    (f32)
//   - L{i}/ln1.{weight,bias}     <- self_attn_layer_norm.{weight,bias}  (f32, verbatim) [pre-attn]
//   - L{i}/ln2.{weight,bias}     <- final_layer_norm.{weight,bias}      (f32, verbatim) [pre-FFN]
//   - refs/ln_post.{weight,bias} <- model.encoder.layer_norm.{weight,bias}  (f32, verbatim) [post-stack]
//
// Whisper is pre-norm with a conv stem (two Conv1d + GELU + learned positional embedding) feeding 12
// pre-norm transformer blocks, then a post-LayerNorm. The conv-stem weights are stored in their native
// PyTorch [out,in,k] layout (no transpose) - the oracle does NOT flatten them, so neither do we; any
// im2col/packing decision belongs to the conv kernel consumer, not the bake.
use std::collections::BTreeMap;
use super::{Arch, RawTensor, OutTensor, transpose2d};

pub struct Whisper;

const N_LAYERS: usize = 12;
const D_MODEL: usize = 768;

impl Whisper {
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
    /// A zero bias [D_MODEL] kept as f32 (Whisper's k_proj carries no bias; oracle stores zeros).
    fn zero_bias() -> OutTensor {
        OutTensor { shape: vec![D_MODEL], data: vec![0f32; D_MODEL], bf16: false }
    }

    /// One transformer block (used by tests + the full transform). Absent source tensors are skipped
    /// here; `transform` hard-errors on any missing required tensor up front.
    pub fn transform_subset(&self, src: &BTreeMap<String, RawTensor>, i: usize)
        -> anyhow::Result<BTreeMap<String, OutTensor>> {
        let p = format!("model.encoder.layers.{i}.");
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
        // attention projections (q,v,out carry bias; k_proj has none -> zeros)
        linear("q.weight", "self_attn.q_proj.weight", &mut o)?;
        f32v ("q.bias", "self_attn.q_proj.bias", &mut o)?;
        linear("k.weight", "self_attn.k_proj.weight", &mut o)?;
        if src.contains_key(&format!("{p}self_attn.k_proj.weight")) {
            o.insert(format!("{l}/k.bias"), Self::zero_bias());
        }
        linear("v.weight", "self_attn.v_proj.weight", &mut o)?;
        f32v ("v.bias", "self_attn.v_proj.bias", &mut o)?;
        linear("out.weight", "self_attn.out_proj.weight", &mut o)?;
        f32v ("out.bias", "self_attn.out_proj.bias", &mut o)?;
        // pre-attn / pre-FFN LayerNorms
        f32v ("ln1.weight", "self_attn_layer_norm.weight", &mut o)?;
        f32v ("ln1.bias", "self_attn_layer_norm.bias", &mut o)?;
        f32v ("ln2.weight", "final_layer_norm.weight", &mut o)?;
        f32v ("ln2.bias", "final_layer_norm.bias", &mut o)?;
        // FFN
        linear("fc1.weight", "fc1.weight", &mut o)?;
        f32v ("fc1.bias", "fc1.bias", &mut o)?;
        linear("fc2.weight", "fc2.weight", &mut o)?;
        f32v ("fc2.bias", "fc2.bias", &mut o)?;
        Ok(o)
    }
}

impl Arch for Whisper {
    fn name(&self) -> &'static str { "whisper" }
    fn required_tensors(&self, n_layers: usize) -> Vec<String> {
        let mut v = vec![
            "model.encoder.conv1.weight", "model.encoder.conv1.bias",
            "model.encoder.conv2.weight", "model.encoder.conv2.bias",
            "model.encoder.embed_positions.weight",
            "model.encoder.layer_norm.weight", "model.encoder.layer_norm.bias",
        ].into_iter().map(String::from).collect::<Vec<_>>();
        for i in 0..n_layers {
            let p = format!("model.encoder.layers.{i}.");
            for s in ["self_attn.q_proj","self_attn.v_proj","self_attn.out_proj","fc1","fc2"] {
                v.push(format!("{p}{s}.weight"));
                v.push(format!("{p}{s}.bias"));
            }
            // k_proj weight exists but has no bias
            v.push(format!("{p}self_attn.k_proj.weight"));
            for s in ["self_attn_layer_norm","final_layer_norm"] {
                v.push(format!("{p}{s}.weight"));
                v.push(format!("{p}{s}.bias"));
            }
        }
        v
    }
    fn transform(&self, src: &BTreeMap<String, RawTensor>) -> anyhow::Result<BTreeMap<String, OutTensor>> {
        for k in self.required_tensors(N_LAYERS) {
            anyhow::ensure!(src.contains_key(&k), "whisper: missing required source tensor {k:?}");
        }
        let mut o = BTreeMap::new();
        // conv stem: native [out,in,k] layout, VERBATIM (no transpose)
        o.insert("conv/conv1.weight".into(), Self::keep_bf16(src, "model.encoder.conv1.weight")?);
        o.insert("conv/conv1.bias".into(), Self::keep_f32(src, "model.encoder.conv1.bias")?);
        o.insert("conv/conv2.weight".into(), Self::keep_bf16(src, "model.encoder.conv2.weight")?);
        o.insert("conv/conv2.bias".into(), Self::keep_f32(src, "model.encoder.conv2.bias")?);
        o.insert("conv/embed_positions".into(), Self::keep_bf16(src, "model.encoder.embed_positions.weight")?);
        for i in 0..N_LAYERS { o.extend(self.transform_subset(src, i)?); }
        // post-stack LayerNorm
        o.insert("refs/ln_post.weight".into(), Self::keep_f32(src, "model.encoder.layer_norm.weight")?);
        o.insert("refs/ln_post.bias".into(), Self::keep_f32(src, "model.encoder.layer_norm.bias")?);
        Ok(o)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn rt(shape: Vec<usize>, data: Vec<f32>) -> RawTensor { RawTensor { shape, data } }
    #[test]
    fn transforms_block_with_zero_k_bias() {
        let mut src = BTreeMap::new();
        src.insert("model.encoder.layers.0.self_attn.q_proj.weight".into(), rt(vec![2,3], vec![1.,2.,3.,4.,5.,6.]));
        src.insert("model.encoder.layers.0.self_attn.q_proj.bias".into(), rt(vec![2], vec![7.,8.]));
        src.insert("model.encoder.layers.0.self_attn.k_proj.weight".into(), rt(vec![2,3], vec![1.,1.,1.,1.,1.,1.]));
        let out = Whisper.transform_subset(&src, 0).unwrap();
        let qw = &out["L0/q.weight"];
        assert_eq!(qw.shape, vec![3,2]);          // transposed [out,in] -> [in,out]
        assert!(qw.bf16);                          // weight -> bf16
        assert_eq!(out["L0/q.bias"].data, vec![7.,8.]);
        assert!(!out["L0/q.bias"].bf16);
        // k_proj has no bias -> zeros, f32, length D_MODEL
        let kb = &out["L0/k.bias"];
        assert_eq!(kb.shape, vec![D_MODEL]);
        assert!(kb.data.iter().all(|&x| x == 0.0));
        assert!(!kb.bf16);
    }
}
