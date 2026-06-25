// rust/npu-weights/src/arch/fastconformer.rs
//
// Parakeet-tdt-0.6b-v3 FastConformer encoder arch. Mirrors scripts/extract_parakeet_encoder.py
// EXACTLY. Source = the ONNX initializers + node-name aliases produced by source::read_onnx
// (the NeMo export anonymises every MatMul weight, so the linear weights are reached via their
// consuming node's name, e.g. `/layers.0/self_attn/linear_q/MatMul`).
//
// Per-block layout (24 blocks, d_model 1024, d_ff 4096, 8 heads x 128). Arena names equal the
// oracle npy paths relative to artifacts/parakeet/encoder, so verify maps name -> <refs>/<name>.npy:
//   - L{i}/norm_feed_forward1.{weight,bias}  <- LayerNormalization (f32 verbatim)
//   - L{i}/norm_self_att.{weight,bias}        <- LayerNormalization (f32 verbatim)
//   - L{i}/norm_conv.{weight,bias}            <- LayerNormalization (f32 verbatim)
//   - L{i}/norm_feed_forward2.{weight,bias}   <- LayerNormalization (f32 verbatim)
//   - L{i}/norm_out.{weight,bias}             <- LayerNormalization (f32 verbatim)
//   - L{i}/self_attn.linear_{q,k,v,pos,out}.weight  <- MatMul weight, VERBATIM [K,N] (bf16, NO bias)
//   - L{i}/self_attn.pos_bias_{u,v}           <- named init [8,128] (f32 verbatim, rel-pos)
//   - L{i}/feed_forward{1,2}.linear{1,2}.weight     <- MatMul weight, VERBATIM [K,N] (bf16, NO bias)
//   - L{i}/conv.pointwise_conv1.weight        <- Conv [2048,1024,1] (bf16 verbatim, NO bias)
//   - L{i}/conv.depthwise_conv.{weight,bias}  <- Conv [1024,1,9] depthwise (w bf16, bias f32) verbatim
//   - L{i}/conv.pointwise_conv2.weight        <- Conv [1024,1024,1] (bf16 verbatim, NO bias)
// pre_encode (the /8 conv2d subsample stack) - names relative to pre_encode/:
//   - pre_encode/conv.{0,2,3,5,6}.{weight,bias}  <- named Conv inits (w bf16, bias f32) verbatim
//   - pre_encode/out.weight                    <- MatMul weight [4096,1024] VERBATIM (bf16)
//   - pre_encode/out.bias                      <- named init [1024] (f32 verbatim)
//
// NeMo FastConformer linears are BIAS-FREE (verified - the oracle stores no matmul bias). MatMul
// weights are kept VERBATIM in their ONNX [K_in, N_out] layout (NO transpose) - that is the x@W
// convention the engine's matvec expects; conv weights stay in native [out,in,k] layout. The
// engine's runtime is responsible for any packing; the bake only mirrors the oracle.
use std::collections::BTreeMap;
use super::{Arch, RawTensor, OutTensor};

pub struct FastConformer;

const N_BLOCKS: usize = 24;

impl FastConformer {
    fn w(src: &BTreeMap<String, RawTensor>, k: &str) -> anyhow::Result<RawTensor> {
        src.get(k).cloned().ok_or_else(|| anyhow::anyhow!("missing source tensor {k:?}"))
    }
    /// Keep a tensor VERBATIM (no transpose) at the given target dtype.
    fn keep(src: &BTreeMap<String, RawTensor>, k: &str, bf16: bool) -> anyhow::Result<OutTensor> {
        let t = Self::w(src, k)?;
        Ok(OutTensor { shape: t.shape, data: t.data, bf16 })
    }

    /// All source keys the arch reads for block `i`. The five LayerNorms + pos_bias are NAMED
    /// initializers (`layers.{i}....`); the linear/ffn weights are node-name ALIASES
    /// (`/layers.{i}/.../MatMul`); the convs are named initializers (`layers.{i}.conv...`).
    fn block_keys(i: usize) -> Vec<(String, String, bool)> {
        // (arena_dst, source_key, bf16)
        let mut v = Vec::new();
        // LayerNorms (f32 verbatim)
        for nm in ["norm_feed_forward1", "norm_self_att", "norm_conv", "norm_feed_forward2", "norm_out"] {
            v.push((format!("L{i}/{nm}.weight"), format!("layers.{i}.{nm}.weight"), false));
            v.push((format!("L{i}/{nm}.bias"),   format!("layers.{i}.{nm}.bias"),   false));
        }
        // rel-pos biases (named inits, f32 verbatim)
        for pb in ["pos_bias_u", "pos_bias_v"] {
            v.push((format!("L{i}/self_attn.{pb}"), format!("layers.{i}.self_attn.{pb}"), false));
        }
        // attention linears (anon MatMul -> node-name alias, bf16 verbatim, bias-free)
        for lin in ["linear_q", "linear_k", "linear_v", "linear_pos", "linear_out"] {
            v.push((format!("L{i}/self_attn.{lin}.weight"),
                    format!("/layers.{i}/self_attn/{lin}/MatMul"), true));
        }
        // FFN linears (anon MatMul -> node-name alias, bf16 verbatim, bias-free)
        for ff in ["feed_forward1", "feed_forward2"] {
            for lin in ["linear1", "linear2"] {
                v.push((format!("L{i}/{ff}.{lin}.weight"),
                        format!("/layers.{i}/{ff}/{lin}/MatMul"), true));
            }
        }
        // conv module: pointwise weights are NAMED inits (bias-free); the depthwise conv weight
        // AND bias are ANONYMOUS (onnx::Conv_*) so they are reached via the node-name alias
        // (`<node>` = weight, `<node>.bias` = bias).
        v.push((format!("L{i}/conv.pointwise_conv1.weight"),
                format!("layers.{i}.conv.pointwise_conv1.weight"), true));
        v.push((format!("L{i}/conv.depthwise_conv.weight"),
                format!("/layers.{i}/conv/depthwise_conv/Conv"), true));
        v.push((format!("L{i}/conv.depthwise_conv.bias"),
                format!("/layers.{i}/conv/depthwise_conv/Conv.bias"), false));
        v.push((format!("L{i}/conv.pointwise_conv2.weight"),
                format!("layers.{i}.conv.pointwise_conv2.weight"), true));
        v
    }

    /// pre_encode (/8 conv2d subsample) source keys. The five convs are named inits; the final
    /// projection is an anon MatMul -> node-name alias; out.bias is a named init.
    fn pre_encode_keys() -> Vec<(String, String, bool)> {
        let mut v = Vec::new();
        for idx in ["0", "2", "3", "5", "6"] {
            v.push((format!("pre_encode/conv.{idx}.weight"),
                    format!("pre_encode.conv.{idx}.weight"), true));
            v.push((format!("pre_encode/conv.{idx}.bias"),
                    format!("pre_encode.conv.{idx}.bias"), false));
        }
        v.push(("pre_encode/out.weight".into(), "/pre_encode/out/MatMul".into(), true));
        v.push(("pre_encode/out.bias".into(), "pre_encode.out.bias".into(), false));
        v
    }
}

impl Arch for FastConformer {
    fn name(&self) -> &'static str { "fastconformer" }

    fn required_tensors(&self, n_layers: usize) -> Vec<String> {
        let mut v: Vec<String> = Self::pre_encode_keys().into_iter().map(|(_, k, _)| k).collect();
        for i in 0..n_layers {
            for (_, k, _) in Self::block_keys(i) { v.push(k); }
        }
        v
    }

    fn transform(&self, src: &BTreeMap<String, RawTensor>) -> anyhow::Result<BTreeMap<String, OutTensor>> {
        for k in self.required_tensors(N_BLOCKS) {
            anyhow::ensure!(src.contains_key(&k), "fastconformer: missing required source tensor {k:?}");
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
    fn keeps_matmul_weight_verbatim_bf16_and_norm_f32() {
        let mut src = BTreeMap::new();
        // anon MatMul reached via node-name alias, verbatim [K,N]
        src.insert("/layers.0/self_attn/linear_q/MatMul".into(), rt(vec![2, 3], vec![1., 2., 3., 4., 5., 6.]));
        src.insert("layers.0.norm_out.weight".into(), rt(vec![3], vec![1., 1., 1.]));
        let qw = FastConformer::keep(&src, "/layers.0/self_attn/linear_q/MatMul", true).unwrap();
        assert_eq!(qw.shape, vec![2, 3]);   // VERBATIM - no transpose
        assert_eq!(qw.data, vec![1., 2., 3., 4., 5., 6.]);
        assert!(qw.bf16);
        let ln = FastConformer::keep(&src, "layers.0.norm_out.weight", false).unwrap();
        assert!(!ln.bf16);
    }

    #[test]
    fn block_keys_are_bias_free_for_linears() {
        let keys = FastConformer::block_keys(0);
        // no linear_*.bias / feed_forward*.linear*.bias entries (NeMo linears are bias-free)
        assert!(!keys.iter().any(|(d, _, _)| d.contains("linear_q.bias")));
        assert!(!keys.iter().any(|(d, _, _)| d.contains("feed_forward1.linear1.bias")));
        // depthwise conv DOES carry a bias
        assert!(keys.iter().any(|(d, _, _)| d == "L0/conv.depthwise_conv.bias"));
        // linear weights are reached via node-name aliases
        assert!(keys.iter().any(|(_, k, _)| k == "/layers.0/self_attn/linear_q/MatMul"));
    }
}
