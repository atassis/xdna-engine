// rust/npu-weights/src/arch/gigaam.rs
//
// GigaAM-v3 Conformer encoder arch. Mirrors scripts/extract_encoder.py (full 16-block encoder)
// EXACTLY. Source = the ONNX initializers + node-name aliases from source::read_onnx
// (models/gigaam_v3_encoder_static.onnx; weights inline, no external-data sidecar). Like the NeMo
// export, GigaAM anonymises every MatMul WEIGHT initializer (onnx::MatMul_*), so linear weights
// are reached via their consuming node's name; linear BIASES + norms + convs are NAMED inits.
//
// Per-block layout (16 blocks, d_model 768, d_ff 3072, conv kernel 5). Arena names equal the
// oracle npy paths relative to artifacts/encoder, so verify maps name -> <refs>/<name>.npy:
//   - L{i}/norm_feed_forward1.{weight,bias}   <- LayerNormalization (f32 verbatim)
//   - L{i}/norm_self_att.{weight,bias}         <- LayerNormalization (f32 verbatim)
//   - L{i}/norm_conv.{weight,bias}             <- LayerNormalization (f32 verbatim)
//   - L{i}/norm_feed_forward2.{weight,bias}    <- LayerNormalization (f32 verbatim)
//   - L{i}/norm_out.{weight,bias}              <- LayerNormalization (f32 verbatim)
//   - L{i}/self_attn.linear_{q,k,v,out}.weight <- MatMul weight, VERBATIM [K,N] (bf16)
//   - L{i}/self_attn.linear_{q,k,v,out}.bias   <- named init (f32 verbatim) [GigaAM linears HAVE bias]
//   - L{i}/feed_forward{1,2}.linear{1,2}.weight<- MatMul weight, VERBATIM [K,N] (bf16)
//   - L{i}/feed_forward{1,2}.linear{1,2}.bias  <- named init (f32 verbatim)
//   - L{i}/conv.pointwise_conv1.{weight,bias}  <- Conv [1536,768,1] (w bf16, bias f32) verbatim
//   - L{i}/conv.depthwise_conv.{weight,bias}   <- Conv [768,1,5] depthwise (w bf16, bias f32) verbatim
//   - L{i}/conv.batch_norm.{weight,bias}       <- LayerNormalization (folded BN, f32 verbatim)
//   - L{i}/conv.pointwise_conv2.{weight,bias}  <- Conv [768,768,1] (w bf16, bias f32) verbatim
// pre_encode (the /4 conv1d subsample) - names relative to pre_encode/ (NOTE the oracle prefixes
// these npy files with `pre_encode.`):
//   - pre_encode/pre_encode.conv.{0,2}.{weight,bias}  <- named Conv inits (w bf16, bias f32) verbatim
//
// MatMul weights kept VERBATIM in ONNX [K_in, N_out] (NO transpose); conv weights kept native
// [out,in,k]. Differs from FastConformer: GigaAM linears carry biases, the conv module adds a
// folded batch_norm, and pre_encode is a 2-conv stack (no final projection MatMul).
use std::collections::BTreeMap;
use super::{Arch, RawTensor, OutTensor};

pub struct Gigaam;

const N_BLOCKS: usize = 16;

impl Gigaam {
    fn w(src: &BTreeMap<String, RawTensor>, k: &str) -> anyhow::Result<RawTensor> {
        src.get(k).cloned().ok_or_else(|| anyhow::anyhow!("missing source tensor {k:?}"))
    }
    fn keep(src: &BTreeMap<String, RawTensor>, k: &str, bf16: bool) -> anyhow::Result<OutTensor> {
        let t = Self::w(src, k)?;
        Ok(OutTensor { shape: t.shape, data: t.data, bf16 })
    }

    /// (arena_dst, source_key, bf16) for block `i`.
    fn block_keys(i: usize) -> Vec<(String, String, bool)> {
        let mut v = Vec::new();
        // LayerNorms incl. the folded conv batch_norm (all f32 verbatim)
        for nm in ["norm_feed_forward1", "norm_self_att", "norm_conv", "norm_feed_forward2",
                   "norm_out", "conv.batch_norm"] {
            v.push((format!("L{i}/{nm}.weight"), format!("layers.{i}.{nm}.weight"), false));
            v.push((format!("L{i}/{nm}.bias"),   format!("layers.{i}.{nm}.bias"),   false));
        }
        // attention linears: weight = anon MatMul (node alias, bf16 verbatim); bias = named init (f32)
        for lin in ["linear_q", "linear_k", "linear_v", "linear_out"] {
            v.push((format!("L{i}/self_attn.{lin}.weight"),
                    format!("/layers.{i}/self_attn/{lin}/MatMul"), true));
            v.push((format!("L{i}/self_attn.{lin}.bias"),
                    format!("layers.{i}.self_attn.{lin}.bias"), false));
        }
        // FFN linears: weight = anon MatMul (node alias); bias = named init
        for ff in ["feed_forward1", "feed_forward2"] {
            for lin in ["linear1", "linear2"] {
                v.push((format!("L{i}/{ff}.{lin}.weight"),
                        format!("/layers.{i}/{ff}/{lin}/MatMul"), true));
                v.push((format!("L{i}/{ff}.{lin}.bias"),
                        format!("layers.{i}.{ff}.{lin}.bias"), false));
            }
        }
        // conv module: all three convs are named inits with weight+bias (verbatim)
        for c in ["pointwise_conv1", "depthwise_conv", "pointwise_conv2"] {
            v.push((format!("L{i}/conv.{c}.weight"), format!("layers.{i}.conv.{c}.weight"), true));
            v.push((format!("L{i}/conv.{c}.bias"),   format!("layers.{i}.conv.{c}.bias"),   false));
        }
        v
    }

    /// pre_encode (/4 conv1d subsample): two named Conv inits (conv.0, conv.2), each weight+bias.
    /// The oracle writes them under the `pre_encode.` npy prefix, so the arena dst carries it too.
    fn pre_encode_keys() -> Vec<(String, String, bool)> {
        let mut v = Vec::new();
        for idx in ["0", "2"] {
            v.push((format!("pre_encode/pre_encode.conv.{idx}.weight"),
                    format!("pre_encode.conv.{idx}.weight"), true));
            v.push((format!("pre_encode/pre_encode.conv.{idx}.bias"),
                    format!("pre_encode.conv.{idx}.bias"), false));
        }
        v
    }
}

impl Arch for Gigaam {
    fn name(&self) -> &'static str { "gigaam" }

    fn required_tensors(&self, n_layers: usize) -> Vec<String> {
        let mut v: Vec<String> = Self::pre_encode_keys().into_iter().map(|(_, k, _)| k).collect();
        for i in 0..n_layers {
            for (_, k, _) in Self::block_keys(i) { v.push(k); }
        }
        v
    }

    fn transform(&self, src: &BTreeMap<String, RawTensor>) -> anyhow::Result<BTreeMap<String, OutTensor>> {
        for k in self.required_tensors(N_BLOCKS) {
            anyhow::ensure!(src.contains_key(&k), "gigaam: missing required source tensor {k:?}");
        }
        let mut o = BTreeMap::new();
        for (dst, key, bf16) in Self::pre_encode_keys() {
            o.insert(dst, Self::keep(src, &key, bf16)?);
        }
        for i in 0..N_BLOCKS {
            for (dst, key, bf16) in Self::block_keys(i) {
                o.insert(dst, Self::keep(src, &key, bf16)?);
            }
        }
        Ok(o)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn rt(shape: Vec<usize>, data: Vec<f32>) -> RawTensor { RawTensor { shape, data } }

    #[test]
    fn linears_have_bias_and_weight_is_verbatim() {
        let keys = Gigaam::block_keys(0);
        // weight via node-name alias, bias via named init - both present (GigaAM linears carry bias)
        assert!(keys.iter().any(|(d, k, b)| d == "L0/self_attn.linear_q.weight"
            && k == "/layers.0/self_attn/linear_q/MatMul" && *b));
        assert!(keys.iter().any(|(d, k, b)| d == "L0/self_attn.linear_q.bias"
            && k == "layers.0.self_attn.linear_q.bias" && !*b));
        // conv batch_norm present (folded BN as LayerNorm)
        assert!(keys.iter().any(|(d, _, _)| d == "L0/conv.batch_norm.weight"));
    }

    #[test]
    fn keep_is_verbatim_no_transpose() {
        let mut src = BTreeMap::new();
        src.insert("/layers.0/self_attn/linear_q/MatMul".into(), rt(vec![2, 3], vec![1., 2., 3., 4., 5., 6.]));
        let qw = Gigaam::keep(&src, "/layers.0/self_attn/linear_q/MatMul", true).unwrap();
        assert_eq!(qw.shape, vec![2, 3]);   // VERBATIM
        assert_eq!(qw.data, vec![1., 2., 3., 4., 5., 6.]);
        assert!(qw.bf16);
    }
}
