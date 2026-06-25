// rust/npu-weights/src/arch/opt.rs
//
// OPT-125m arch (facebook/opt-125m, decoder weights used as an encoder-style weight bag). Mirrors
// scripts/convert_opt125m.py EXACTLY. Source = raw HF safetensors (model.decoder.*). Reference npy
// layout (verify --refs dir is `artifacts/opt-125m`, so arena names equal npy paths relative to it):
//   - embed_tokens              <- model.decoder.embed_tokens.weight   ([vocab,d] VERBATIM bf16)
//   - embed_positions           <- model.decoder.embed_positions.weight ([2050,d] VERBATIM bf16; offset +2)
//   - ln_final.{weight,bias}    <- model.decoder.final_layer_norm.{w,b}  (f32, verbatim)
//   - lm_head.weight            <- lm_head.weight                       (TRANSPOSED [d,vocab] bf16)
//   - L{i}/{q,k,v,out}.weight   <- self_attn.{q,k,v,out}_proj.weight    (TRANSPOSED [in,out] bf16)
//   - L{i}/{q,k,v,out}.bias     <- self_attn.{q,k,v,out}_proj.bias      (f32, verbatim)
//   - L{i}/ln_self.{weight,bias}<- self_attn_layer_norm.{w,b}           (f32) [pre-attn]
//   - L{i}/fc1.weight, fc2.weight<- fc1/fc2.weight                      (TRANSPOSED bf16)
//   - L{i}/fc1.bias, fc2.bias   <- fc1/fc2.bias                         (f32)
//   - L{i}/ln_ffn.{weight,bias} <- final_layer_norm.{w,b}  (OPT's PER-LAYER "final_layer_norm" = pre-FFN LN, f32)
//
// OPT-125m is dimension-identical to whisper-small (768/12/12/3072), decoder-only, relu FFN, learned
// positions (offset +2), pre-norm.
//
// TIED EMBEDDING: the safetensors checkpoint omits `model.decoder.embed_tokens.weight` (it is tied to
// `lm_head.weight`; both [vocab,d] and bit-identical - verified). The oracle reads pytorch_model.bin
// which carries embed_tokens explicitly. We reconstruct embed_tokens from lm_head.weight (verbatim,
// untransposed) so the bake matches the oracle from the safetensors source the hf: backend resolves.
use std::collections::BTreeMap;
use super::{Arch, RawTensor, OutTensor, transpose2d};

pub struct Opt;

const N_LAYERS: usize = 12;

impl Opt {
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

    /// One decoder block (used by tests + the full transform). Absent source tensors are skipped here;
    /// `transform` hard-errors on any missing required tensor up front.
    pub fn transform_subset(&self, src: &BTreeMap<String, RawTensor>, i: usize)
        -> anyhow::Result<BTreeMap<String, OutTensor>> {
        let p = format!("model.decoder.layers.{i}.");
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
        // attention projections (all carry bias in OPT)
        for (dst, src_name) in [("q","q_proj"),("k","k_proj"),("v","v_proj"),("out","out_proj")] {
            linear(&format!("{dst}.weight"), &format!("self_attn.{src_name}.weight"), &mut o)?;
            f32v(&format!("{dst}.bias"), &format!("self_attn.{src_name}.bias"), &mut o)?;
        }
        // pre-attention LayerNorm
        f32v ("ln_self.weight", "self_attn_layer_norm.weight", &mut o)?;
        f32v ("ln_self.bias", "self_attn_layer_norm.bias", &mut o)?;
        // FFN (relu)
        linear("fc1.weight", "fc1.weight", &mut o)?;
        f32v ("fc1.bias", "fc1.bias", &mut o)?;
        linear("fc2.weight", "fc2.weight", &mut o)?;
        f32v ("fc2.bias", "fc2.bias", &mut o)?;
        // OPT's per-layer "final_layer_norm" is the PRE-FFN LayerNorm
        f32v ("ln_ffn.weight", "final_layer_norm.weight", &mut o)?;
        f32v ("ln_ffn.bias", "final_layer_norm.bias", &mut o)?;
        Ok(o)
    }
}

impl Arch for Opt {
    fn name(&self) -> &'static str { "opt" }
    fn required_tensors(&self, n_layers: usize) -> Vec<String> {
        // embed_tokens is reconstructed from lm_head (tied) so it is NOT required from source.
        let mut v = vec![
            "lm_head.weight",
            "model.decoder.embed_positions.weight",
            "model.decoder.final_layer_norm.weight",
            "model.decoder.final_layer_norm.bias",
        ].into_iter().map(String::from).collect::<Vec<_>>();
        for i in 0..n_layers {
            let p = format!("model.decoder.layers.{i}.");
            for s in ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.out_proj","fc1","fc2"] {
                v.push(format!("{p}{s}.weight"));
                v.push(format!("{p}{s}.bias"));
            }
            for s in ["self_attn_layer_norm","final_layer_norm"] {
                v.push(format!("{p}{s}.weight"));
                v.push(format!("{p}{s}.bias"));
            }
        }
        v
    }
    fn transform(&self, src: &BTreeMap<String, RawTensor>) -> anyhow::Result<BTreeMap<String, OutTensor>> {
        for k in self.required_tensors(N_LAYERS) {
            anyhow::ensure!(src.contains_key(&k), "opt: missing required source tensor {k:?}");
        }
        let mut o = BTreeMap::new();
        // embed_tokens: tied to lm_head ([vocab,d]); store verbatim like the oracle.
        o.insert("embed_tokens".into(), Self::keep_bf16(src, "lm_head.weight")?);
        o.insert("embed_positions".into(), Self::keep_bf16(src, "model.decoder.embed_positions.weight")?);
        o.insert("ln_final.weight".into(), Self::keep_f32(src, "model.decoder.final_layer_norm.weight")?);
        o.insert("ln_final.bias".into(), Self::keep_f32(src, "model.decoder.final_layer_norm.bias")?);
        // lm_head: TRANSPOSED to [d,vocab] (the oracle does .T)
        o.insert("lm_head.weight".into(), Self::lin(src, "lm_head.weight")?);
        for i in 0..N_LAYERS { o.extend(self.transform_subset(src, i)?); }
        Ok(o)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn rt(shape: Vec<usize>, data: Vec<f32>) -> RawTensor { RawTensor { shape, data } }
    #[test]
    fn transforms_block_transposes_linears_keeps_norms() {
        let mut src = BTreeMap::new();
        src.insert("model.decoder.layers.0.self_attn.q_proj.weight".into(), rt(vec![2,3], vec![1.,2.,3.,4.,5.,6.]));
        src.insert("model.decoder.layers.0.self_attn.q_proj.bias".into(), rt(vec![2], vec![7.,8.]));
        src.insert("model.decoder.layers.0.self_attn_layer_norm.weight".into(), rt(vec![3], vec![1.,1.,1.]));
        let out = Opt.transform_subset(&src, 0).unwrap();
        let qw = &out["L0/q.weight"];
        assert_eq!(qw.shape, vec![3,2]);          // transposed [out,in] -> [in,out]
        assert!(qw.bf16);
        assert_eq!(out["L0/q.bias"].data, vec![7.,8.]);
        assert!(!out["L0/q.bias"].bf16);
        assert!(!out["L0/ln_self.weight"].bf16);  // LayerNorm -> f32
    }
    #[test]
    fn embed_tokens_tied_to_lm_head_verbatim() {
        // minimal full-ish bag would be large; just exercise the tied-embedding path directly.
        let mut src = BTreeMap::new();
        src.insert("lm_head.weight".into(), rt(vec![2,3], vec![1.,2.,3.,4.,5.,6.]));
        let et = Opt::keep_bf16(&src, "lm_head.weight").unwrap();
        assert_eq!(et.shape, vec![2,3]);          // verbatim [vocab,d]
        assert!(et.bf16);
        let lm = Opt::lin(&src, "lm_head.weight").unwrap();
        assert_eq!(lm.shape, vec![3,2]);          // transposed [d,vocab]
    }
}
