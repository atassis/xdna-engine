// rust/npu-weights/src/source.rs
use crate::arch::RawTensor;
use crate::onnx;
use crate::spec::Source;
use half::{bf16, f16};
use memmap2::Mmap;
use safetensors::tensor::Dtype;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// True if the resolved file list is an ONNX model (the first file is `.onnx`; any further
/// entries are external-data sidecars used only for fingerprinting). The bake pipeline
/// dispatches on this to pick the ONNX initializer reader vs the safetensors reader.
pub fn is_onnx(files: &[PathBuf]) -> bool {
    !files.is_empty() && files[0].extension().map(|e| e == "onnx").unwrap_or(false)
}

/// Collect the distinct external-data sidecar paths an ONNX model references (relative to the
/// .onnx dir), by decoding its initializers' `external_data.location`. Returns [] for a
/// fully-inline model. Used by resolve_files so the fingerprint covers the weight bytes.
fn onnx_external_sidecars(onnx_path: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let bytes = std::fs::read(onnx_path)?;
    let model = onnx::decode_model(&bytes)?;
    let dir = onnx_path.parent().unwrap_or(Path::new("."));
    let mut locs: Vec<String> = Vec::new();
    if let Some(g) = model.graph {
        for init in &g.initializer {
            if init.data_location == Some(onnx::LOC_EXTERNAL) {
                for e in &init.external_data {
                    if e.key.as_deref() == Some("location") {
                        if let Some(v) = e.value.as_deref() {
                            if !locs.iter().any(|x| x == v) { locs.push(v.to_string()); }
                        }
                    }
                }
            }
        }
    }
    Ok(locs.into_iter().map(|l| dir.join(l)).collect())
}

/// Read all weight INITIALIZERS from an ONNX model file as f32 RawTensors (NOT inference).
/// Decodes ModelProto.graph.initializer via prost; for each TensorProto returns name + dims +
/// f32 data. Handles raw_data for FLOAT/FLOAT16/BFLOAT16/INT64, the typed *_data fallback when
/// raw_data is empty, and data_location=EXTERNAL by reading the sidecar at the given
/// location/offset/length (offset/length default to whole-file/0 when absent, per onnx spec).
pub fn read_onnx(files: &[PathBuf]) -> anyhow::Result<BTreeMap<String, RawTensor>> {
    // files[0] is the .onnx graph; any further entries are external-data sidecars (resolved here
    // from the initializers' own `location`, not from this list, so order/extra entries are fine).
    anyhow::ensure!(is_onnx(files), "read_onnx expects a .onnx file first, got {files:?}");
    let onnx_path = &files[0];
    let bytes = std::fs::read(onnx_path)?;
    let model = onnx::decode_model(&bytes)?;
    let graph = model.graph.ok_or_else(|| anyhow::anyhow!("onnx model has no graph"))?;
    let dir = onnx_path.parent().unwrap_or(Path::new("."));

    // Lazily mmap external-data sidecars (keyed by the `location` string), so a model whose 600+
    // initializers all live in one .data file mmaps it exactly once.
    let mut sidecars: BTreeMap<String, Mmap> = BTreeMap::new();

    let mut bag = BTreeMap::new();
    for init in &graph.initializer {
        let name = match &init.name {
            Some(n) if !n.is_empty() => n.clone(),
            _ => continue, // unnamed initializer: nothing an arch can key on
        };
        let dt = init.data_type.unwrap_or(onnx::DT_FLOAT);
        let shape: Vec<usize> = init.dims.iter().map(|&d| d.max(0) as usize).collect();

        // Obtain the raw little-endian bytes for this tensor: either the EXTERNAL sidecar slice,
        // the inline raw_data, or (when both are empty) the typed *_data path below.
        let external = init.data_location == Some(onnx::LOC_EXTERNAL);
        let raw_owned: Option<Vec<u8>> = if external {
            let mut location = String::new();
            let mut offset: usize = 0;
            let mut length: Option<usize> = None;
            for e in &init.external_data {
                match (e.key.as_deref(), e.value.as_deref()) {
                    (Some("location"), Some(v)) => location = v.to_string(),
                    (Some("offset"), Some(v)) => offset = v.parse().unwrap_or(0),
                    (Some("length"), Some(v)) => length = v.parse().ok(),
                    _ => {}
                }
            }
            anyhow::ensure!(!location.is_empty(), "onnx external initializer {name:?} has no location");
            if !sidecars.contains_key(&location) {
                let p = dir.join(&location);
                let f = std::fs::File::open(&p)
                    .map_err(|e| anyhow::anyhow!("onnx external-data sidecar {p:?}: {e}"))?;
                sidecars.insert(location.clone(), unsafe { Mmap::map(&f)? });
            }
            let mm = &sidecars[&location];
            let end = length.map(|l| offset + l).unwrap_or(mm.len());
            anyhow::ensure!(end <= mm.len(),
                "onnx external slice [{offset}..{end}] out of range ({}) for {name:?}", mm.len());
            Some(mm[offset..end].to_vec())
        } else {
            init.raw_data.clone().filter(|r| !r.is_empty())
        };

        let data: Vec<f32> = if let Some(raw) = raw_owned {
            match dt {
                onnx::DT_FLOAT => raw.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect(),
                onnx::DT_FLOAT16 => raw.chunks_exact(2).map(|b| f16::from_le_bytes(b.try_into().unwrap()).to_f32()).collect(),
                onnx::DT_BFLOAT16 => raw.chunks_exact(2).map(|b| bf16::from_le_bytes(b.try_into().unwrap()).to_f32()).collect(),
                onnx::DT_INT64 => raw.chunks_exact(8).map(|b| i64::from_le_bytes(b.try_into().unwrap()) as f32).collect(),
                _ => continue, // unhandled dtype: skip (arch required_tensors guard still catches a real gap)
            }
        } else {
            // Typed-field fallback (no raw_data): float_data for FLOAT, int64_data for INT64.
            match dt {
                onnx::DT_FLOAT if !init.float_data.is_empty() => init.float_data.clone(),
                onnx::DT_INT64 if !init.int64_data.is_empty() =>
                    init.int64_data.iter().map(|&v| v as f32).collect(),
                _ => continue, // empty / unhandled: nothing to read
            }
        };
        bag.insert(name, RawTensor { shape, data });
    }

    // Node-name aliases: NeMo/GigaAM exports anonymise MatMul (and some Conv) weight initializers
    // (`onnx::MatMul_*`), so the only handle on them is the consuming node's name. For every
    // weight-bearing node (MatMul/Conv) that reads an initializer, alias that tensor under the
    // NODE NAME (e.g. `/layers.0/self_attn/linear_q/MatMul`). The arch then keys off node names -
    // exactly the lookup the Python oracle performs - while named initializers stay reachable by
    // their own name. Aliases never overwrite a real initializer entry.
    let init_names: std::collections::BTreeSet<&str> =
        graph.initializer.iter().filter_map(|i| i.name.as_deref()).collect();
    for node in &graph.node {
        let (Some(node_name), Some(op)) = (node.name.as_deref(), node.op_type.as_deref()) else { continue };
        if op != "MatMul" && op != "Conv" { continue; }
        // A MatMul/Conv reads its weight as the FIRST initializer input and (for Conv) an optional
        // bias as the SECOND. Alias the WEIGHT under the node name and the BIAS under
        // `<node>.bias`, so an arch can reach an anonymised conv weight+bias pair positionally
        // (NeMo depthwise convs carry an anonymous `onnx::Conv_*` weight AND bias).
        let init_inputs: Vec<&String> = node.input.iter().filter(|i| init_names.contains(i.as_str())).collect();
        if let Some(w) = init_inputs.first() {
            if let Some(t) = bag.get(*w).cloned() { bag.entry(node_name.to_string()).or_insert(t); }
        }
        if let Some(b) = init_inputs.get(1) {
            if let Some(t) = bag.get(*b).cloned() {
                bag.entry(format!("{node_name}.bias")).or_insert(t);
            }
        }
    }
    Ok(bag)
}

/// Read source weights from whatever the resolved file list is (safetensors shards or one .onnx).
pub fn read_weights(files: &[PathBuf]) -> anyhow::Result<BTreeMap<String, RawTensor>> {
    if is_onnx(files) { read_onnx(files) } else { read_safetensors(files) }
}

/// Read all tensors from safetensors file(s) as f32 RawTensors (handles fp32/fp16/bf16 source).
pub fn read_safetensors(files: &[PathBuf]) -> anyhow::Result<BTreeMap<String, RawTensor>> {
    let mut bag = BTreeMap::new();
    for path in files {
        let file = std::fs::File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let st = safetensors::SafeTensors::deserialize(&mmap)?;
        for (name, view) in st.tensors() {
            let shape = view.shape().to_vec();
            let raw = view.data();
            let data: Vec<f32> = match view.dtype() {
                Dtype::F32 => raw.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect(),
                Dtype::F16 => raw.chunks_exact(2).map(|b| f16::from_le_bytes(b.try_into().unwrap()).to_f32()).collect(),
                Dtype::BF16 => raw.chunks_exact(2).map(|b| bf16::from_le_bytes(b.try_into().unwrap()).to_f32()).collect(),
                // Non-float source tensors (e.g. int64 `position_ids` index buffers) are not
                // weights any arch consumes; skip them rather than fail the whole read. The
                // arch `required_tensors` guard still hard-errors if a needed weight is absent.
                _ => continue,
            };
            bag.insert(name.to_string(), RawTensor { shape, data });
        }
    }
    Ok(bag)
}

/// Resolve a Source to the local weight file(s) to read (download via hf-hub if needed).
pub fn resolve_files(src: &Source) -> anyhow::Result<Vec<PathBuf>> {
    match src {
        Source::Path(p) => {
            if p.is_dir() {
                let idx = p.join("model.safetensors.index.json");
                if idx.exists() { return shard_files(&idx); }
                let single = p.join("model.safetensors");
                anyhow::ensure!(single.exists(), "no model.safetensors[.index.json] in {p:?}");
                Ok(vec![single])
            } else if p.extension().map(|e| e == "onnx").unwrap_or(false) {
                // ONNX model: return the .onnx graph file plus any external-data sidecar(s) it
                // references, so the source fingerprint covers the (large) weight bytes too. The
                // reader (read_onnx) still keys off files[0]; the sidecars here are only for
                // fingerprinting/staleness. Sidecar paths come from the initializers' `location`.
                let mut out = vec![p.clone()];
                for side in onnx_external_sidecars(p)? {
                    if side.exists() && !out.contains(&side) { out.push(side); }
                }
                Ok(out)
            } else {
                Ok(vec![p.clone()])
            }
        }
        Source::Hf { repo, rev } => {
            use hf_hub::api::sync::ApiBuilder;
            let api = ApiBuilder::new().build()?;
            let r = match rev {
                Some(v) => api.repo(hf_hub::Repo::with_revision(repo.clone(), hf_hub::RepoType::Model, v.clone())),
                None => api.model(repo.clone()),
            };
            // try sharded first, else single
            match r.get("model.safetensors.index.json") {
                Ok(idx) => {
                    let mut out = Vec::new();
                    #[derive(serde::Deserialize)]
                    struct Idx { weight_map: std::collections::BTreeMap<String, String> }
                    let parsed: Idx = serde_json::from_slice(&std::fs::read(&idx)?)?;
                    let mut names: Vec<String> = parsed.weight_map.values().cloned().collect();
                    names.sort(); names.dedup();
                    for n in names { out.push(r.get(&n)?); }
                    Ok(out)
                }
                Err(_) => Ok(vec![r.get("model.safetensors")?]),
            }
        }
    }
}

/// Parse a HF index.json weight_map, return the unique shard files (resolved beside the index).
fn shard_files(index_json: &Path) -> anyhow::Result<Vec<PathBuf>> {
    #[derive(serde::Deserialize)]
    struct Idx { weight_map: std::collections::BTreeMap<String, String> }
    let idx: Idx = serde_json::from_slice(&std::fs::read(index_json)?)?;
    let dir = index_json.parent().unwrap_or(Path::new("."));
    let mut shards: Vec<String> = idx.weight_map.values().cloned().collect();
    shards.sort(); shards.dedup();
    Ok(shards.into_iter().map(|s| dir.join(s)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::{TensorView, Dtype};
    // ---- ONNX reader tests (round-trip via prost encode of our own message set) ----
    use crate::onnx::{GraphProto, ModelProto, StringStringEntryProto, TensorProto, DT_FLOAT, DT_INT64, LOC_EXTERNAL};
    use prost::Message;

    fn ent(k: &str, v: &str) -> StringStringEntryProto {
        StringStringEntryProto { key: Some(k.into()), value: Some(v.into()) }
    }

    #[test]
    fn reads_inline_onnx_f32_and_int64() {
        // f32 weight stored in raw_data; int64 stored in raw_data too.
        let w: Vec<u8> = [1f32, 2., 3., 4.].iter().flat_map(|x| x.to_le_bytes()).collect();
        let ids: Vec<u8> = [10i64, 20].iter().flat_map(|x| x.to_le_bytes()).collect();
        let model = ModelProto { graph: Some(GraphProto { node: vec![], initializer: vec![
            TensorProto { dims: vec![2, 2], data_type: Some(DT_FLOAT), name: Some("w".into()),
                          raw_data: Some(w), ..Default::default() },
            TensorProto { dims: vec![2], data_type: Some(DT_INT64), name: Some("ids".into()),
                          raw_data: Some(ids), ..Default::default() },
        ] }) };
        let bytes = model.encode_to_vec();
        let f = tempfile::Builder::new().suffix(".onnx").tempfile().unwrap();
        std::fs::write(f.path(), &bytes).unwrap();
        let bag = read_onnx(&[f.path().to_path_buf()]).unwrap();
        assert_eq!(bag["w"].shape, vec![2, 2]);
        assert_eq!(bag["w"].data, vec![1., 2., 3., 4.]);
        assert_eq!(bag["ids"].data, vec![10., 20.]);  // int64 -> f32
    }

    #[test]
    fn reads_onnx_float_data_fallback() {
        // no raw_data: typed float_data field is used instead.
        let model = ModelProto { graph: Some(GraphProto { node: vec![], initializer: vec![
            TensorProto { dims: vec![3], data_type: Some(DT_FLOAT), name: Some("b".into()),
                          float_data: vec![5., 6., 7.], ..Default::default() },
        ] }) };
        let f = tempfile::Builder::new().suffix(".onnx").tempfile().unwrap();
        std::fs::write(f.path(), &model.encode_to_vec()).unwrap();
        let bag = read_onnx(&[f.path().to_path_buf()]).unwrap();
        assert_eq!(bag["b"].data, vec![5., 6., 7.]);
    }

    #[test]
    fn reads_onnx_external_data_sidecar() {
        // weight bytes live in a sidecar; the initializer points at location/offset/length.
        let dir = tempfile::tempdir().unwrap();
        let side = dir.path().join("w.onnx.data");
        let pad = [0u8; 8];                          // 8-byte prefix to exercise a nonzero offset
        let payload: Vec<u8> = [9f32, 8., 7.].iter().flat_map(|x| x.to_le_bytes()).collect();
        let mut filebytes = pad.to_vec();
        filebytes.extend_from_slice(&payload);
        std::fs::write(&side, &filebytes).unwrap();
        let model = ModelProto { graph: Some(GraphProto { node: vec![], initializer: vec![
            TensorProto { dims: vec![3], data_type: Some(DT_FLOAT), name: Some("ext".into()),
                          data_location: Some(LOC_EXTERNAL),
                          external_data: vec![ent("location", "w.onnx.data"),
                                              ent("offset", "8"), ent("length", "12")],
                          ..Default::default() },
        ] }) };
        let onnx = dir.path().join("w.onnx");
        std::fs::write(&onnx, &model.encode_to_vec()).unwrap();
        // resolve_files should surface the sidecar (for fingerprinting) + is_onnx stays true.
        let files = resolve_files(&Source::Path(onnx.clone())).unwrap();
        assert!(is_onnx(&files));
        assert!(files.iter().any(|p| p.ends_with("w.onnx.data")), "sidecar not discovered: {files:?}");
        let bag = read_onnx(&files).unwrap();
        assert_eq!(bag["ext"].data, vec![9., 8., 7.]);
    }

    #[test]
    fn reads_f32_safetensors() {
        // build a tiny safetensors in a temp file
        let data: Vec<u8> = [1f32,2.,3.,4.].iter().flat_map(|x| x.to_le_bytes()).collect();
        let view = TensorView::new(Dtype::F32, vec![2,2], &data).unwrap();
        let bytes = safetensors::serialize([("w", &view)], &None).unwrap();
        let f = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(f.path(), &bytes).unwrap();
        let bag = read_safetensors(&[f.path().to_path_buf()]).unwrap();
        assert_eq!(bag["w"].shape, vec![2,2]);
        assert_eq!(bag["w"].data, vec![1.,2.,3.,4.]);
    }
}
