//! Header-only safetensors views. Integer tensors remain encoded rather than being skipped. I64
//! is admitted as container metadata only: `DenseTensor` and every codec still refuse it as weight
//! storage, so accepting its eight-byte elements cannot decode integers as floating weights.
use super::{FileSource, FileSpan, Limits, ScalarType};
use anyhow::{bail, ensure, Context, Result};
use serde::de::{MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::{Component, Path};

#[derive(Clone, Debug)]
pub struct TensorEntry {
    pub dtype: ScalarType,
    pub shape: Vec<usize>,
    pub span: FileSpan,
}
#[derive(Debug)]
pub struct SafetensorsFile {
    tensors: BTreeMap<String, TensorEntry>,
}

// Value's normal deserializer overwrites duplicate fields before RawEntry can reject them.
// Check every object, including nested tensor fields and metadata, at the parsing boundary.
struct UniqueValue(Value);
impl<'de> Deserialize<'de> for UniqueValue {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        struct JsonVisitor;
        impl<'de> Visitor<'de> for JsonVisitor {
            type Value = UniqueValue;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("JSON without duplicate keys")
            }
            fn visit_map<M: MapAccess<'de>>(
                self,
                mut map: M,
            ) -> std::result::Result<Self::Value, M::Error> {
                let mut entries = serde_json::Map::new();
                while let Some((name, value)) = map.next_entry::<String, UniqueValue>()? {
                    if entries.insert(name.clone(), value.0).is_some() {
                        return Err(serde::de::Error::custom(format!(
                            "duplicate safetensors key {name:?}"
                        )));
                    }
                }
                Ok(UniqueValue(Value::Object(entries)))
            }
            fn visit_seq<S: SeqAccess<'de>>(
                self,
                mut seq: S,
            ) -> std::result::Result<Self::Value, S::Error> {
                let mut values = Vec::new();
                while let Some(value) = seq.next_element::<UniqueValue>()? {
                    values.push(value.0);
                }
                Ok(UniqueValue(Value::Array(values)))
            }
            fn visit_str<E: serde::de::Error>(
                self,
                s: &str,
            ) -> std::result::Result<Self::Value, E> {
                Ok(UniqueValue(Value::String(s.into())))
            }
            fn visit_string<E: serde::de::Error>(
                self,
                s: String,
            ) -> std::result::Result<Self::Value, E> {
                Ok(UniqueValue(Value::String(s)))
            }
            fn visit_bool<E: serde::de::Error>(
                self,
                n: bool,
            ) -> std::result::Result<Self::Value, E> {
                Ok(UniqueValue(Value::Bool(n)))
            }
            fn visit_i64<E: serde::de::Error>(self, n: i64) -> std::result::Result<Self::Value, E> {
                Ok(UniqueValue(Value::Number(n.into())))
            }
            fn visit_u64<E: serde::de::Error>(self, n: u64) -> std::result::Result<Self::Value, E> {
                Ok(UniqueValue(Value::Number(n.into())))
            }
            fn visit_f64<E: serde::de::Error>(self, n: f64) -> std::result::Result<Self::Value, E> {
                serde_json::Number::from_f64(n)
                    .map(|n| UniqueValue(Value::Number(n)))
                    .ok_or_else(|| E::custom("non-finite JSON number"))
            }
            fn visit_unit<E: serde::de::Error>(self) -> std::result::Result<Self::Value, E> {
                Ok(UniqueValue(Value::Null))
            }
        }
        d.deserialize_any(JsonVisitor)
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawEntry {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}

pub(super) fn parse_unique(raw: &[u8]) -> Result<Value> {
    let mut de = serde_json::Deserializer::from_slice(raw);
    let UniqueValue(value) =
        UniqueValue::deserialize(&mut de).context("invalid weight source JSON")?;
    de.end()?;
    Ok(value)
}

/// A local sharded checkpoint index. Shard paths cannot escape the index directory, and the
/// index must account for every tensor in each referenced shard. No tensor payload is expanded.
#[derive(Debug)]
pub struct SafetensorsSet {
    tensors: BTreeMap<String, TensorEntry>,
}
impl SafetensorsSet {
    pub fn open_index(path: impl AsRef<Path>, limits: Limits) -> Result<Self> {
        let path = path.as_ref();
        let source = FileSource::open(path)?;
        ensure!(
            source.len() <= limits.max_header_bytes as u64,
            "safetensors index limit exceeded"
        );
        let mut raw = vec![0; usize::try_from(source.len())?];
        source.read_exact_at(0, &mut raw)?;
        let value = parse_unique(&raw)?;
        let map = value
            .get("weight_map")
            .and_then(Value::as_object)
            .context("missing index weight_map")?;
        ensure!(
            !map.is_empty() && map.len() <= limits.max_tensors,
            "invalid index tensor count"
        );
        let root = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."))
            .canonicalize()?;
        let mut shards = BTreeSet::new();
        for (name, file) in map {
            ensure!(
                !name.is_empty() && name.len() <= limits.max_string_bytes,
                "invalid indexed tensor name"
            );
            let file = file.as_str().context("index shard path must be a string")?;
            ensure!(
                !file.is_empty() && file.len() <= limits.max_string_bytes,
                "invalid shard path length"
            );
            let relative = Path::new(file);
            ensure!(
                relative
                    .components()
                    .all(|c| matches!(c, Component::Normal(_))),
                "index shard path must stay within source directory"
            );
            let resolved = root.join(relative).canonicalize()?;
            ensure!(
                resolved.starts_with(&root),
                "index shard symlink escapes source directory"
            );
            shards.insert(file.to_string());
        }
        let mut tensors = BTreeMap::new();
        for file in shards {
            let parsed = SafetensorsFile::open(root.join(&file), limits)?;
            for (name, entry) in parsed.tensors {
                ensure!(
                    map.get(&name).and_then(Value::as_str) == Some(file.as_str()),
                    "index/shard mismatch for {name}"
                );
                ensure!(
                    tensors.insert(name.clone(), entry).is_none(),
                    "duplicate tensor {name} across shards"
                );
                ensure!(
                    tensors.len() <= limits.max_tensors,
                    "aggregate tensor limit exceeded"
                );
            }
        }
        ensure!(
            tensors.len() == map.len(),
            "index names a tensor absent from its shards"
        );
        Ok(Self { tensors })
    }
    pub fn tensors(&self) -> &BTreeMap<String, TensorEntry> {
        &self.tensors
    }
    pub fn get(&self, name: &str) -> Result<&TensorEntry> {
        self.tensors
            .get(name)
            .with_context(|| format!("missing indexed tensor {name:?}"))
    }
}

impl SafetensorsFile {
    pub fn open(path: impl AsRef<Path>, limits: Limits) -> Result<Self> {
        let source = FileSource::open(path)?;
        let mut prefix = [0; 8];
        source.read_exact_at(0, &mut prefix)?;
        let n =
            usize::try_from(u64::from_le_bytes(prefix)).context("header length not addressable")?;
        ensure!(
            n > 0 && n <= limits.max_header_bytes,
            "safetensors header exceeds configured limit"
        );
        let base = 8u64
            .checked_add(n as u64)
            .context("header extent overflow")?;
        ensure!(base <= source.len(), "truncated safetensors header");
        let mut raw = vec![0; n];
        source.read_exact_at(8, &mut raw)?;
        let mut entries = match parse_unique(&raw)? {
            Value::Object(entries) => entries,
            _ => bail!("safetensors header must be an object"),
        };
        if let Some(metadata) = entries.remove("__metadata__") {
            let map = metadata
                .as_object()
                .context("safetensors metadata must be a string map")?;
            ensure!(
                map.len() <= limits.max_metadata_entries,
                "metadata entry limit exceeded"
            );
            for (key, value) in map {
                let value = value
                    .as_str()
                    .context("safetensors metadata value must be a string")?;
                ensure!(
                    key.len() <= limits.max_string_bytes && value.len() <= limits.max_string_bytes,
                    "safetensors metadata string limit exceeded"
                );
            }
        }
        ensure!(
            entries.len() <= limits.max_tensors,
            "safetensors tensor limit exceeded"
        );
        let mut tensors = BTreeMap::new();
        let mut ranges = Vec::with_capacity(entries.len());
        for (name, value) in entries {
            ensure!(
                !name.is_empty() && name.len() <= limits.max_string_bytes,
                "invalid tensor name length"
            );
            let entry: RawEntry =
                serde_json::from_value(value).with_context(|| format!("tensor {name}"))?;
            let dtype = match entry.dtype.as_str() {
                "F32" => ScalarType::F32,
                "F16" => ScalarType::F16,
                "BF16" => ScalarType::Bf16,
                "I32" => ScalarType::I32,
                "I64" => ScalarType::I64,
                other => bail!("{name}: unsupported safetensors dtype {other}"),
            };
            ensure!(
                entry.shape.len() <= 8,
                "{name}: tensor rank exceeds supported limit"
            );
            let count = entry
                .shape
                .iter()
                .try_fold(1usize, |n, d| n.checked_mul(*d))
                .context("tensor shape overflow")?;
            let bytes = count
                .checked_mul(dtype.byte_width())
                .context("tensor byte size overflow")?;
            let [lo, hi] = entry.data_offsets;
            ensure!(
                hi.checked_sub(lo) == Some(u64::try_from(bytes)?),
                "{name}: tensor byte length mismatch"
            );
            let absolute = base.checked_add(lo).context("tensor offset overflow")?;
            let span = source
                .span(absolute, hi - lo)
                .with_context(|| format!("{name}: tensor span"))?;
            ranges.push((lo, hi));
            tensors.insert(
                name,
                TensorEntry {
                    dtype,
                    shape: entry.shape,
                    span,
                },
            );
        }
        ranges.sort_unstable();
        let mut end = 0;
        for (lo, hi) in ranges {
            ensure!(lo == end, "safetensors spans overlap or contain a gap");
            end = hi;
        }
        ensure!(
            base.checked_add(end) == Some(source.len()),
            "unindexed or truncated safetensors data"
        );
        Ok(Self { tensors })
    }
    pub fn tensors(&self) -> &BTreeMap<String, TensorEntry> {
        &self.tensors
    }
    pub fn get(&self, name: &str) -> Result<&TensorEntry> {
        self.tensors
            .get(name)
            .with_context(|| format!("missing safetensors tensor {name:?}"))
    }
}
