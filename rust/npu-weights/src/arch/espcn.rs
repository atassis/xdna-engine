// rust/npu-weights/src/arch/espcn.rs
//
// ESPCN sub-pixel CNN (sub_pixel_cnn_2016): 4 conv layers + a pixel-shuffle upsample. Source = the
// pretrained ONNX (onnx: source). Conv weights stay native [Cout,Cin,kh,kw] (the im2col->GEMM frontier
// flattens them per-layer, like resnet.rs keeps conv-native). Weights -> bf16, biases -> f32. Flat arena
// names match the oracle npy (scripts/export_espcn.py): conv{1..4}_w / conv{1..4}_b.
//
// ONNX initializer names are model-specific, so we discover the 4 conv weights (4-D) and 4 biases (1-D)
// in NAME-SORTED order and map them conv1..conv4 -- the pretrained model names them conv1.w/.b .. conv4.w/.b
// which sort into layer order. Hard-error if the count differs (guards against a wrong model).
use super::{Arch, OutTensor, RawTensor};
use std::collections::BTreeMap;

pub struct Espcn;

impl Arch for Espcn {
    fn name(&self) -> &'static str {
        "espcn"
    }

    fn required_tensors(&self, _n_layers: usize) -> Vec<String> {
        // Names are discovered from the source bag (ONNX initializer names vary), so no fixed anchors.
        vec![]
    }

    fn transform(
        &self,
        src: &BTreeMap<String, RawTensor>,
    ) -> anyhow::Result<BTreeMap<String, OutTensor>> {
        // Partition source tensors into 4-D conv weights and 1-D biases. BTreeMap already iterates by
        // name, so `ws`/`bs` are name-sorted -> layer order for conv{1..4}.
        let ws: Vec<(&String, &RawTensor)> =
            src.iter().filter(|(_, t)| t.shape.len() == 4).collect();
        let bs: Vec<(&String, &RawTensor)> =
            src.iter().filter(|(_, t)| t.shape.len() == 1).collect();
        anyhow::ensure!(
            ws.len() == 4 && bs.len() == 4,
            "espcn: expected 4 conv weights + 4 biases, got {} / {} (names: {:?} / {:?})",
            ws.len(),
            bs.len(),
            ws.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>(),
            bs.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>(),
        );
        let mut out = BTreeMap::new();
        for (i, ((_, w), (_, b))) in ws.iter().zip(bs.iter()).enumerate() {
            let n = i + 1;
            anyhow::ensure!(w.shape.len() == 4, "espcn conv{n}: weight not 4-D");
            anyhow::ensure!(
                b.shape.len() == 1 && b.shape[0] == w.shape[0],
                "espcn conv{n}: bias len {:?} != Cout {}",
                b.shape,
                w.shape[0]
            );
            out.insert(
                format!("conv{n}_w"),
                OutTensor {
                    shape: w.shape.clone(),
                    data: w.data.clone(),
                    bf16: true,
                },
            );
            out.insert(
                format!("conv{n}_b"),
                OutTensor {
                    shape: b.shape.clone(),
                    data: b.data.clone(),
                    bf16: false,
                },
            );
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn maps_four_convs_in_order() {
        let mut src = BTreeMap::new();
        src.insert("conv1.w".into(), RawTensor { shape: vec![64, 1, 5, 5], data: vec![0.0; 64 * 25] });
        src.insert("conv2.w".into(), RawTensor { shape: vec![64, 64, 3, 3], data: vec![0.0; 64 * 64 * 9] });
        src.insert("conv3.w".into(), RawTensor { shape: vec![32, 64, 3, 3], data: vec![0.0; 32 * 64 * 9] });
        src.insert("conv4.w".into(), RawTensor { shape: vec![9, 32, 3, 3], data: vec![0.0; 9 * 32 * 9] });
        src.insert("conv1.b".into(), RawTensor { shape: vec![64], data: vec![0.0; 64] });
        src.insert("conv2.b".into(), RawTensor { shape: vec![64], data: vec![0.0; 64] });
        src.insert("conv3.b".into(), RawTensor { shape: vec![32], data: vec![0.0; 32] });
        src.insert("conv4.b".into(), RawTensor { shape: vec![9], data: vec![0.0; 9] });
        let out = Espcn.transform(&src).unwrap();
        assert_eq!(out["conv1_w"].shape, vec![64, 1, 5, 5]);
        assert!(out["conv1_w"].bf16 && !out["conv1_b"].bf16);
        assert_eq!(out["conv4_w"].shape, vec![9, 32, 3, 3]);
        assert_eq!(out.len(), 8);
    }
}
