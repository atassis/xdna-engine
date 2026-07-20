// rust/npu-weights/src/arch/edsr.rs
//
// EDSR-base (eugenesiow/edsr-base) -- the M3 super-resolution ship net. Source = the torch state_dict as
// safetensors (module. stripped; see scripts/export_edsr.py). Conv weights stay native [Cout,Cin,kh,kw]
// (the im2col->GEMM frontier flattens per-layer). Weights -> bf16, biases -> f32. Flat arena names match
// the oracle npy: sub_mean_{w,b}, head_{w,b}, b{i}_c{0,1}_{w,b} (16 residual blocks), btail_{w,b},
// tail0_{w,b} (conv 64->Cout*r^2 before pixel_shuffle), tail1_{w,b}, add_mean_{w,b}.
//
// Architecture (net = the edsr.json schedule; this arch only bakes weights):
//   sub_mean(1x1) -> head -> [16x resblock: c0,relu,c1,+skip] -> btail -> +global_skip
//   -> tail0 -> pixel_shuffle(r) -> tail1 -> add_mean.
use super::{Arch, OutTensor, RawTensor};
use std::collections::BTreeMap;

pub struct Edsr;

/// Map a torch state_dict key to the flat arena name, or None to skip.
fn arena_name(key: &str) -> Option<String> {
    let p: Vec<&str> = key.split('.').collect();
    let suffix = |s: &str| match s {
        "weight" => Some("w"),
        "bias" => Some("b"),
        _ => None,
    };
    match p.as_slice() {
        ["sub_mean", s] => suffix(s).map(|x| format!("sub_mean_{x}")),
        ["add_mean", s] => suffix(s).map(|x| format!("add_mean_{x}")),
        ["head", "0", s] => suffix(s).map(|x| format!("head_{x}")),
        // residual block conv0 / conv1: body.{i}.body.{0|2}.{weight|bias}
        ["body", i, "body", j, s] => {
            let c = match *j {
                "0" => "c0",
                "2" => "c1",
                _ => return None,
            };
            suffix(s).map(|x| format!("b{i}_{c}_{x}"))
        }
        // body tail conv: body.{N}.{weight|bias} (no inner ".body.")
        ["body", _n, s] => suffix(s).map(|x| format!("btail_{x}")),
        ["tail", "0", "0", s] => suffix(s).map(|x| format!("tail0_{x}")),
        ["tail", "1", s] => suffix(s).map(|x| format!("tail1_{x}")),
        _ => None,
    }
}

impl Arch for Edsr {
    fn name(&self) -> &'static str {
        "edsr"
    }

    fn required_tensors(&self, _n_layers: usize) -> Vec<String> {
        vec![] // discovered from the source bag
    }

    fn transform(
        &self,
        src: &BTreeMap<String, RawTensor>,
    ) -> anyhow::Result<BTreeMap<String, OutTensor>> {
        let mut out = BTreeMap::new();
        for (k, t) in src {
            let Some(name) = arena_name(k) else { continue };
            let bf16 = t.shape.len() == 4; // conv weights bf16; biases (1-D) + 1x1 mean convs' bias f32
            out.insert(
                name,
                OutTensor {
                    shape: t.shape.clone(),
                    data: t.data.clone(),
                    bf16,
                },
            );
        }
        // sanity: at least head + 1 resblock + tail present
        anyhow::ensure!(
            out.contains_key("head_w") && out.contains_key("b0_c0_w") && out.contains_key("tail0_w"),
            "edsr: missing core tensors (got {} baked; keys e.g. {:?})",
            out.len(),
            out.keys().take(6).collect::<Vec<_>>()
        );
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn maps_edsr_keys() {
        assert_eq!(arena_name("head.0.weight").as_deref(), Some("head_w"));
        assert_eq!(arena_name("body.3.body.0.weight").as_deref(), Some("b3_c0_w"));
        assert_eq!(arena_name("body.7.body.2.bias").as_deref(), Some("b7_c1_b"));
        assert_eq!(arena_name("body.16.weight").as_deref(), Some("btail_w"));
        assert_eq!(arena_name("tail.0.0.weight").as_deref(), Some("tail0_w"));
        assert_eq!(arena_name("tail.1.bias").as_deref(), Some("tail1_b"));
        assert_eq!(arena_name("sub_mean.weight").as_deref(), Some("sub_mean_w"));
        assert_eq!(arena_name("add_mean.bias").as_deref(), Some("add_mean_b"));
    }

    #[test]
    fn transform_bakes_conv_bf16_bias_f32() {
        let mut src = BTreeMap::new();
        src.insert("head.0.weight".into(), RawTensor { shape: vec![64, 3, 3, 3], data: vec![0.0; 64 * 3 * 9] });
        src.insert("head.0.bias".into(), RawTensor { shape: vec![64], data: vec![0.0; 64] });
        src.insert("body.0.body.0.weight".into(), RawTensor { shape: vec![64, 64, 3, 3], data: vec![0.0; 64 * 64 * 9] });
        src.insert("tail.0.0.weight".into(), RawTensor { shape: vec![576, 64, 3, 3], data: vec![0.0; 576 * 64 * 9] });
        let out = Edsr.transform(&src).unwrap();
        assert!(out["head_w"].bf16 && !out["head_b"].bf16);
        assert_eq!(out["tail0_w"].shape, vec![576, 64, 3, 3]);
    }
}
