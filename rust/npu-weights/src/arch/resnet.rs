// rust/npu-weights/src/arch/resnet.rs
//
// ResNet arch (microsoft/resnet-{18,34,50,...}, HF ResNetForImageClassification). Mirrors
// scripts/export_resnet.py EXACTLY. Source = raw HF safetensors. Conv layers carry a folded-away
// BatchNorm: the oracle folds BN INTO the preceding conv (conv has no bias of its own), so this arch
// does the identical fold and emits a conv weight [Cout,Cin,kh,kw] + a folded bias [Cout]:
//     scale = gamma / sqrt(var + eps);  w' = w * scale[:,None,None,None];  b' = beta - mean*scale
// BatchNorm2d eps is the PyTorch default 1e-5 (HF ResNetConvLayer uses nn.BatchNorm2d(out) with the
// default), matching the oracle's `bn.eps`. Reference npy layout (refs dir is the model root
// `artifacts/resnet18`, flat names = npy paths):
//   - stem_w / stem_b          <- resnet.embedder.embedder.{convolution,normalization} (7x7 s2 p3 + relu)
//   - s{S}l{L}c0_w / _b        <- ...stages.{S}.layers.{L}.layer.0.{conv,norm} (3x3, relu)
//   - s{S}l{L}c1_w / _b        <- ...stages.{S}.layers.{L}.layer.1.{conv,norm} (3x3, no act pre-residual)
//   - s{S}l{L}sc_w / _b        <- ...stages.{S}.layers.{L}.shortcut.{conv,norm}  (1x1 downsample; only
//                                 present when the block downsamples - emitted only when in source)
//   - fc_w / fc_b              <- classifier.1.{weight,bias}  (Linear 512->1000, kept VERBATIM [out,in]
//                                 to mirror the oracle, NOT transposed)
//
// Conv weights stay in native [Cout,Cin,kh,kw] layout (the conv2d kernel consumes them directly, NOT
// im2col-flattened like ViT's patch-embed). Weights -> bf16, folded biases / fc bias -> f32. Stage /
// layer / shortcut structure is discovered from the source bag (ResNet-18 = 4 stages x 2 layers; a
// downsampling shortcut at the first layer of stages 1..3), so this arch handles other ResNet depths.
// The BN fold is a deterministic numeric mirror of the oracle - baked here, not parked.
use std::collections::BTreeMap;
use super::{Arch, RawTensor, OutTensor};

pub struct Resnet;

/// PyTorch BatchNorm2d default eps (HF ResNetConvLayer uses the default), matching the oracle.
const BN_EPS: f32 = 1e-5;

impl Resnet {
    fn w(src: &BTreeMap<String, RawTensor>, k: &str) -> anyhow::Result<RawTensor> {
        src.get(k).cloned().ok_or_else(|| anyhow::anyhow!("missing source tensor {k:?}"))
    }

    /// Fold a conv+BN pair at `prefix` (`<prefix>.convolution.weight` + `<prefix>.normalization.*`)
    /// into a conv weight [Cout,Cin,kh,kw] (bf16) and a folded bias [Cout] (f32), matching the oracle.
    fn fold(src: &BTreeMap<String, RawTensor>, prefix: &str)
        -> anyhow::Result<(OutTensor, OutTensor)> {
        let conv = Self::w(src, &format!("{prefix}.convolution.weight"))?;
        let gamma = Self::w(src, &format!("{prefix}.normalization.weight"))?;
        let beta  = Self::w(src, &format!("{prefix}.normalization.bias"))?;
        let mean  = Self::w(src, &format!("{prefix}.normalization.running_mean"))?;
        let var   = Self::w(src, &format!("{prefix}.normalization.running_var"))?;
        let cout = conv.shape[0];
        anyhow::ensure!(conv.shape.len() == 4, "resnet conv {prefix:?}: expected 4D, got {:?}", conv.shape);
        anyhow::ensure!([gamma.data.len(), beta.data.len(), mean.data.len(), var.data.len()]
            .iter().all(|&n| n == cout), "resnet BN {prefix:?}: channel mismatch vs Cout={cout}");
        let per_c = conv.data.len() / cout;           // Cin*kh*kw
        let scale: Vec<f32> = (0..cout).map(|c| gamma.data[c] / (var.data[c] + BN_EPS).sqrt()).collect();
        let mut wf = conv.data.clone();
        for c in 0..cout {
            let s = scale[c];
            for j in 0..per_c { wf[c * per_c + j] *= s; }
        }
        let bf: Vec<f32> = (0..cout).map(|c| beta.data[c] - mean.data[c] * scale[c]).collect();
        Ok((OutTensor { shape: conv.shape, data: wf, bf16: true },
            OutTensor { shape: vec![cout], data: bf, bf16: false }))
    }
}

impl Arch for Resnet {
    fn name(&self) -> &'static str { "resnet" }
    /// Anchors that must always be present (the stem conv+BN and the classifier). Stage/layer convs
    /// are discovered dynamically in `transform`, which hard-errors on any missing conv/BN within a
    /// discovered block. `n_layers` is unused (ResNet depth is not a single transformer-block count).
    fn required_tensors(&self, _n_layers: usize) -> Vec<String> {
        ["resnet.embedder.embedder.convolution.weight",
         "resnet.embedder.embedder.normalization.weight",
         "classifier.1.weight", "classifier.1.bias"]
            .into_iter().map(String::from).collect()
    }
    fn transform(&self, src: &BTreeMap<String, RawTensor>) -> anyhow::Result<BTreeMap<String, OutTensor>> {
        for k in self.required_tensors(0) {
            anyhow::ensure!(src.contains_key(&k), "resnet: missing required source tensor {k:?}");
        }
        let mut o = BTreeMap::new();
        // stem (7x7 s2 conv + BN, folded)
        let (sw, sb) = Self::fold(src, "resnet.embedder.embedder")?;
        o.insert("stem_w".into(), sw);
        o.insert("stem_b".into(), sb);
        // stages -> layers -> {c0, c1, optional shortcut}, discovered from the bag.
        let mut s = 0usize;
        loop {
            let stage_anchor = format!("resnet.encoder.stages.{s}.layers.0.layer.0.convolution.weight");
            if !src.contains_key(&stage_anchor) { break; }
            let mut l = 0usize;
            loop {
                let base = format!("resnet.encoder.stages.{s}.layers.{l}");
                if !src.contains_key(&format!("{base}.layer.0.convolution.weight")) { break; }
                let (c0w, c0b) = Self::fold(src, &format!("{base}.layer.0"))?;
                o.insert(format!("s{s}l{l}c0_w"), c0w);
                o.insert(format!("s{s}l{l}c0_b"), c0b);
                let (c1w, c1b) = Self::fold(src, &format!("{base}.layer.1"))?;
                o.insert(format!("s{s}l{l}c1_w"), c1w);
                o.insert(format!("s{s}l{l}c1_b"), c1b);
                if src.contains_key(&format!("{base}.shortcut.convolution.weight")) {
                    let (scw, scb) = Self::fold(src, &format!("{base}.shortcut"))?;
                    o.insert(format!("s{s}l{l}sc_w"), scw);
                    o.insert(format!("s{s}l{l}sc_b"), scb);
                }
                l += 1;
            }
            anyhow::ensure!(l > 0, "resnet: stage {s} has no layers");
            s += 1;
        }
        anyhow::ensure!(s > 0, "resnet: no encoder stages found in source");
        // classifier (Linear, kept verbatim [out,in] to mirror the oracle; weight bf16, bias f32)
        let fcw = Self::w(src, "classifier.1.weight")?;
        o.insert("fc_w".into(), OutTensor { shape: fcw.shape, data: fcw.data, bf16: true });
        let fcb = Self::w(src, "classifier.1.bias")?;
        o.insert("fc_b".into(), OutTensor { shape: fcb.shape, data: fcb.data, bf16: false });
        Ok(o)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn rt(shape: Vec<usize>, data: Vec<f32>) -> RawTensor { RawTensor { shape, data } }
    #[test]
    fn bn_fold_matches_formula() {
        // 1 out-channel, 1 in-channel, 1x1 conv. w=2, gamma=3, var=(3^2/?)... pick easy numbers.
        let mut src = BTreeMap::new();
        let p = "resnet.embedder.embedder";
        src.insert(format!("{p}.convolution.weight"), rt(vec![1,1,1,1], vec![2.0]));
        src.insert(format!("{p}.normalization.weight"), rt(vec![1], vec![3.0]));       // gamma
        src.insert(format!("{p}.normalization.bias"), rt(vec![1], vec![0.5]));         // beta
        src.insert(format!("{p}.normalization.running_mean"), rt(vec![1], vec![1.0])); // mean
        src.insert(format!("{p}.normalization.running_var"), rt(vec![1], vec![4.0]));  // var
        let (wf, bf) = Resnet::fold(&src, p).unwrap();
        let scale = 3.0f32 / (4.0f32 + BN_EPS).sqrt();   // ~= 1.4999...
        assert!((wf.data[0] - 2.0 * scale).abs() < 1e-5, "w' = w*scale");
        assert!((bf.data[0] - (0.5 - 1.0 * scale)).abs() < 1e-5, "b' = beta - mean*scale");
        assert!(wf.bf16, "conv weight -> bf16");
        assert!(!bf.bf16, "folded bias -> f32");
        assert_eq!(wf.shape, vec![1,1,1,1]);
    }
    #[test]
    fn fold_errors_on_channel_mismatch() {
        let mut src = BTreeMap::new();
        let p = "x";
        src.insert(format!("{p}.convolution.weight"), rt(vec![2,1,1,1], vec![1.,2.]));
        src.insert(format!("{p}.normalization.weight"), rt(vec![1], vec![1.]));  // wrong len
        src.insert(format!("{p}.normalization.bias"), rt(vec![1], vec![0.]));
        src.insert(format!("{p}.normalization.running_mean"), rt(vec![1], vec![0.]));
        src.insert(format!("{p}.normalization.running_var"), rt(vec![1], vec![1.]));
        assert!(Resnet::fold(&src, p).is_err());
    }
}
