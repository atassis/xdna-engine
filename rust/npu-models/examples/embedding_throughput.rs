//! JSON-lines adapter for controlled embedding comparisons. No device work occurs until load.
use anyhow::{bail, ensure, Context, Result};
use npu_asr::{
    ctx2::{xclbin_stem, CtxAShape, Precision},
    kernel_registry as kr,
};
use npu_asr_host::batch::passage_batches;
use npu_models::{
    bert::{
        encoder::BertEncoder,
        frontend::EmbedFrontend,
        head::{EmbedHead, Pooling},
        weights::BertWeights,
    },
    config::ScenarioConfig,
    pipeline::{Encoder, Frontend, Head},
    tuning_profile, Model,
};
use npu_xrt::Device;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::{self, BufRead, Read, Write},
    path::Path,
    rc::Rc,
    time::Instant,
};
use tokenizers::Tokenizer;

const MAX_LINE: usize = 8 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Arm {
    FastSerial,
    NativeSerial,
    NativePacked,
}
impl Arm {
    fn parse(s: &str) -> Result<Self> {
        match s {
            "fast_serial" => Ok(Self::FastSerial),
            "native_serial" => Ok(Self::NativeSerial),
            "native_packed" => Ok(Self::NativePacked),
            _ => bail!("unknown arm {s:?}"),
        }
    }
    fn name(self) -> &'static str {
        match self {
            Self::FastSerial => "fast_serial",
            Self::NativeSerial => "native_serial",
            Self::NativePacked => "native_packed",
        }
    }
    fn precision(self) -> Precision {
        if self == Self::FastSerial {
            Precision::FastBf16
        } else {
            Precision::NativeBf16
        }
    }
    fn precision_name(self) -> &'static str {
        if self == Self::FastSerial {
            "bf16"
        } else {
            "native"
        }
    }
    fn tile(self) -> (usize, usize, usize) {
        if self == Self::FastSerial {
            (64, 32, 96)
        } else {
            (32, 32, 32)
        }
    }
    fn rows(self) -> usize {
        if self == Self::NativePacked {
            1024
        } else {
            512
        }
    }
    fn shape(self) -> CtxAShape {
        CtxAShape {
            rows: self.rows(),
            streams: vec![768, 3072],
            tile: Some(self.tile()),
            ..CtxAShape::whisper_small()
        }
    }
}
fn field<'a>(v: &'a Value, key: &str) -> Result<&'a str> {
    v.get(key)
        .and_then(Value::as_str)
        .with_context(|| format!("missing/string field {key}"))
}
fn parse_request(line: &[u8]) -> Result<Value> {
    ensure!(
        line.len() <= MAX_LINE,
        "request line exceeds {MAX_LINE} bytes"
    );
    let v: Value = serde_json::from_slice(line)?;
    ensure!(
        v.is_object() && v["schema_version"].as_u64() == Some(1),
        "expected schema_version=1 object"
    );
    field(&v, "id")?;
    match field(&v, "op")? {
        "load" => {
            Arm::parse(field(&v, "arm")?)?;
            for k in ["root", "scenario"] {
                ensure!(
                    Path::new(field(&v, k)?).is_absolute(),
                    "{k} must be absolute"
                );
            }
        }
        "embed" => {
            texts(&v)?;
        }
        "unload" | "quit" => (),
        op => bail!("unknown op {op:?}"),
    }
    Ok(v)
}
fn texts(v: &Value) -> Result<Vec<&str>> {
    v["texts"]
        .as_array()
        .context("texts must be an array")?
        .iter()
        .map(|t| t.as_str().context("every text must be a string"))
        .collect()
}
fn vector_bits(vectors: &[Vec<f32>]) -> Vec<Vec<u32>> {
    vectors
        .iter()
        .map(|v| v.iter().map(|x| x.to_bits()).collect())
        .collect()
}
fn ambient_is_safe(vars: impl IntoIterator<Item = (String, String)>) -> Result<()> {
    for (key, _) in vars {
        // Reject even apparently matching selectors: the three arms must share an unambiguous setup.
        if matches!(
            key.as_str(),
            "NPU_PRECISION"
                | "NPU_TILE"
                | "NPU_SS_NPU"
                | "NPU_LN_NPU"
                | "NPU_GLU_FUSED"
                | "NPU_QKV_OVERLAP"
                | "NPU_MM2_PIPELINE"
                | "NPU_INT8_FASTEPI"
                | "NPU_ENC_FFN_RESIDENT"
        ) || key.starts_with("BERT_")
            || key.starts_with("PARAKEET_SIM_")
        {
            bail!("conflicting ambient selector {key}; unset it before launching the benchmark");
        }
    }
    Ok(())
}
fn file_identity(path: &Path) -> Result<Value> {
    let mut f = fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    ensure!(
        f.metadata()?.is_file() && f.metadata()?.len() > 0,
        "not a nonempty regular file: {}",
        path.display()
    );
    let mut hash = Sha256::new();
    let mut buf = [0u8; 65536];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hash.update(&buf[..n]);
    }
    Ok(json!({"path":fs::canonicalize(path)?, "sha256":format!("{:x}", hash.finalize())}))
}
fn check_contract(dir: &Path, stem: &str, n: usize, arm: Arm) -> Result<Vec<Value>> {
    if arm == Arm::FastSerial {
        // Legacy Fast artifacts have no extent attestation; report only their actual hashes.
        return Ok(vec![
            file_identity(&kr::xclbin_path(dir, stem))?,
            file_identity(&kr::insts_path(dir, stem))?,
        ]);
    }
    let sidecar = dir.join(format!("contract_{stem}.json"));
    let c: Value = serde_json::from_slice(
        &fs::read(&sidecar).with_context(|| format!("read {}", sidecar.display()))?,
    )?;
    let (m, k, t) = arm.tile();
    ensure!(
        c["rows"] == arm.rows()
            && c["k"] == 800
            && c["n"] == n
            && c["tile"] == json!([m, k, t])
            && c["precision"] == arm.precision_name(),
        "artifact extent/precision mismatch: {}",
        sidecar.display()
    );
    let x = file_identity(&kr::xclbin_path(dir, stem))?;
    let i = file_identity(&kr::insts_path(dir, stem))?;
    ensure!(
        x["sha256"] == c["xclbin_sha256"] && i["sha256"] == c["insts_sha256"],
        "artifact hash mismatch: {stem}"
    );
    Ok(vec![x, i, file_identity(&sidecar)?])
}
struct Preflight {
    cfg: ScenarioConfig,
    tok: Tokenizer,
    identity: Value,
}
fn preflight(arm: Arm, root: &Path, scenario: &Path) -> Result<Preflight> {
    ensure!(
        root.is_absolute() && root.is_dir() && scenario.is_absolute(),
        "root must be an absolute directory; scenario must be absolute"
    );
    ambient_is_safe(std::env::vars())?;
    let cfg = ScenarioConfig::load(scenario)?;
    let m = cfg.model_or_err().map_err(anyhow::Error::msg)?;
    ensure!(
        cfg.scenario.kind == "embeddings" && !cfg.scenario.name.to_lowercase().starts_with("esm"),
        "requires a BERT embedding scenario"
    );
    ensure!(
        cfg.artifacts.source.is_empty(),
        "artifacts.source baking is forbidden"
    );
    ensure!(
        !cfg.artifacts.weights.is_empty() && !cfg.artifacts.tokenizer.is_empty(),
        "explicit weights and tokenizer required"
    );
    ensure!(
        [
            &cfg.artifacts.decode,
            &cfg.artifacts.prefill,
            &cfg.artifacts.tokenizer_dir,
            &cfg.artifacts.resident,
            &cfg.artifacts.nli_head
        ]
        .iter()
        .all(|s| s.is_empty()),
        "unsupported non-BERT artifact declaration"
    );
    ensure!(
        m.hidden == 768
            && m.ff == 3072
            && m.n_heads == 12
            && m.head_dim == 64
            && m.n_layers == 12
            && m.max_seq == 512,
        "requires BERT-base 768/3072, 12x64 heads, 12 layers, max_seq512"
    );
    ensure!(
        m.precision == arm.precision_name()
            && cfg.embeddings.pooling == "cls"
            && cfg.embeddings.normalize,
        "scenario must explicitly match selected precision and normalized CLS"
    );
    ensure!(
        cfg.embeddings.batch_rows
            == if arm == Arm::NativePacked {
                Some(1024)
            } else {
                None
            },
        "scenario batch_rows must match selected construction"
    );
    let mut artifacts = vec![file_identity(scenario)?];
    let profile = root.join("config/profiles/xdna2.toml");
    if profile.exists() {
        let p = tuning_profile::TuningProfile::from_toml_str(&fs::read_to_string(&profile)?)?;
        ensure!(
            !p.resolve(arm.precision()).ffn_resident,
            "resident FFN is not permitted"
        );
        artifacts.push(file_identity(&profile)?);
    }
    let tokenizer_path = root.join(&cfg.artifacts.tokenizer);
    artifacts.push(file_identity(&tokenizer_path)?);
    let tok =
        Tokenizer::from_file(&tokenizer_path).map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
    // Legacy Frontend::run honors these settings, unlike the packed boundary. Reject rather than
    // silently changing the serial numerical/tokenization path or comparing different token IDs.
    ensure!(
        tok.get_truncation().is_none() && tok.get_padding().is_none(),
        "tokenizer truncation/padding must be disabled in the benchmark asset"
    );
    let weights = root.join(&cfg.artifacts.weights);
    for key in ["word_emb", "pos_emb", "type_emb", "emb_ln_w", "emb_ln_b"] {
        artifacts.push(file_identity(
            &weights.join("emb").join(format!("{key}.npy")),
        )?);
    }
    for layer in 0..m.n_layers {
        for key in [
            "q_w",
            "q_b",
            "k_w",
            "k_b",
            "v_w",
            "v_b",
            "attn_out_w",
            "attn_out_b",
            "ffn1_w",
            "ffn1_b",
            "ffn2_w",
            "ffn2_b",
            "attn_ln_w",
            "attn_ln_b",
            "out_ln_w",
            "out_ln_b",
        ] {
            artifacts.push(file_identity(&weights.join(format!("L{layer}/{key}.npy")))?);
        }
    }
    for declared in [&cfg.artifacts.onnx_ref, &cfg.artifacts.checkpoint] {
        if !declared.is_empty() {
            artifacts.push(file_identity(&root.join(declared))?);
        }
    }
    let tuning = tuning_profile::resolve(root, arm.precision());
    let shape = arm.shape();
    shape.validate(&tuning).map_err(anyhow::Error::msg)?;
    if arm != Arm::FastSerial {
        shape
            .preflight_native(root, &tuning)
            .map_err(anyhow::Error::msg)?;
    }
    let dir = kr::resolve_kernel_dir(root, npu_asr::engines::WA_SUBDIR);
    let mut stems = vec![(xclbin_stem(&shape, &tuning), 3072)];
    for n in [768, 3072] {
        stems.push((shape.stream_stem(&tuning, n, 0), n));
        let optional = shape.stream_stem(&tuning, n, 1);
        if kr::insts_path(&dir, &optional).exists() {
            stems.push((optional, n));
        }
    }
    stems.sort();
    stems.dedup();
    if let Some(declared) = &cfg.kernels {
        for (family, set) in declared {
            ensure!(
                family == "whole_array" && set.one_of.is_empty(),
                "unsupported declared kernel family/alternatives"
            );
            for stem in &set.required {
                ensure!(
                    stems.iter().any(|(s, _)| s == stem),
                    "declared kernel {stem} is not part of the selected construction"
                );
            }
        }
    }
    for (stem, n) in stems {
        artifacts.extend(check_contract(&dir, &stem, n, arm)?);
    }
    artifacts.sort_by(|a, b| a["path"].as_str().cmp(&b["path"].as_str()));
    artifacts.dedup_by(|a, b| a["path"] == b["path"]);
    let (tm, tk, tn) = arm.tile();
    let identity = json!({"precision":arm.precision_name(),"tile":[tm,tk,tn],"rows":arm.rows(),"artifacts":artifacts,
        "attestation": if arm == Arm::FastSerial { "hashed_legacy_only" } else { "native_contract" },
        "constructor": if arm == Arm::NativePacked { "Model::load_in/BERT batch_rows1024/Model::embed_batch" } else { "BertWeights::load/EmbedFrontend::new/BertEncoder::with_shape/EmbedHead::Cls+normalize" }});
    Ok(Preflight { cfg, tok, identity })
}
struct Serial {
    frontend: EmbedFrontend,
    encoder: BertEncoder,
    head: EmbedHead,
    dev: Rc<Device>,
}
enum Backend {
    Serial(Serial),
    Packed(Model),
}
trait Execute {
    fn serial(&self, text: &str) -> Result<Vec<f32>>;
    fn packed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;
}
impl Execute for Backend {
    fn serial(&self, text: &str) -> Result<Vec<f32>> {
        let Self::Serial(s) = self else {
            bail!("not a serial backend")
        };
        let (x, valid) = s.frontend.run(text.to_owned());
        Ok(s.head.run(&s.encoder.forward_last(&x, valid), valid))
    }
    fn packed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let Self::Packed(m) = self else {
            bail!("not a packed backend")
        };
        Ok(m.embed_batch(texts)?)
    }
}
fn dispatch(backend: &impl Execute, arm: Arm, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
    if texts.is_empty() {
        return Ok(vec![]);
    }
    if arm == Arm::NativePacked {
        backend.packed(texts)
    } else {
        texts.iter().map(|t| backend.serial(t)).collect()
    }
}
struct Loaded {
    arm: Arm,
    backend: Backend,
    tok: Tokenizer,
}
impl Loaded {
    fn load(arm: Arm, root: &Path, scenario: &Path) -> Result<(Self, Value)> {
        let start = Instant::now();
        let p = preflight(arm, root, scenario)?;
        let backend = if arm == Arm::NativePacked {
            Backend::Packed(Model::load_in(scenario, root)?)
        } else {
            let m = p.cfg.model_or_err().map_err(anyhow::Error::msg)?;
            let weights = Rc::new(BertWeights::load(
                &root.join(&p.cfg.artifacts.weights),
                m.n_layers,
            )?);
            let frontend = EmbedFrontend::new(
                &root.join(&p.cfg.artifacts.tokenizer),
                weights.clone(),
                m.max_seq,
            );
            let tuning = tuning_profile::resolve(root, arm.precision());
            let dev = Rc::new(Device::open(0).map_err(anyhow::Error::msg)?);
            let encoder = BertEncoder::with_shape(
                dev.clone(),
                root,
                &weights,
                m.n_heads,
                m.head_dim,
                &tuning,
                arm.shape(),
            );
            Backend::Serial(Serial {
                frontend,
                encoder,
                head: EmbedHead {
                    pooling: Pooling::Cls,
                    normalize: true,
                },
                dev,
            })
        };
        let loaded = Self {
            arm,
            backend,
            tok: p.tok,
        };
        let result = json!({"arm":arm.name(),"setup_ns":start.elapsed().as_nanos() as u64,"identity":p.identity,"bo_bytes":loaded.bo_bytes()});
        Ok((loaded, result))
    }
    fn bo_bytes(&self) -> u64 {
        match &self.backend {
            Backend::Packed(m) => m.bo_bytes(),
            Backend::Serial(s) => s.dev.resident_bo_bytes(),
        }
    }
    fn embed(&self, texts: &[&str]) -> Result<Value> {
        let cpu_start = process_cpu_ns()?;
        let start = Instant::now();
        let (vectors, mut ids, mut groups) = compute(&self.backend, self.arm, &self.tok, texts)?;
        let elapsed_ns = start.elapsed().as_nanos() as u64;
        let cpu_ns = process_cpu_ns()?
            .checked_sub(cpu_start)
            .context("CPU clock regressed")?;
        if self.arm == Arm::NativePacked {
            // Model::embed_batch already tokenizes and validates the full request. Duplicate
            // adapter diagnostics are outside both clocks, unlike necessary serial validation.
            ids = token_ids(&self.tok, texts)?;
            groups = group_count(self.arm, &ids.iter().map(Vec::len).collect::<Vec<_>>())?;
        }
        validate_vectors(&vectors, texts.len())?;
        Ok(
            json!({"arm":self.arm.name(),"elapsed_ns":elapsed_ns,"vectors_bits":vector_bits(&vectors),"token_ids":ids,
            "bo_bytes":self.bo_bytes(),"process_cpu_ns":cpu_ns,"peak_rss_bytes":peak_rss_bytes()?,
            "reached":{"mode":if self.arm == Arm::NativePacked {"packed"} else {"serial"},
            "serial_calls":if self.arm == Arm::NativePacked {0} else {texts.len()},
            "packed_api_calls":usize::from(self.arm == Arm::NativePacked && !texts.is_empty()),
            "groups":groups,"useful_rows":ids.iter().map(Vec::len).sum::<usize>(),"physical_rows":groups*self.arm.rows(),"command_count":null}}),
        )
    }
}
type Computed = (Vec<Vec<f32>>, Vec<Vec<u32>>, usize);
fn compute(backend: &impl Execute, arm: Arm, tok: &Tokenizer, texts: &[&str]) -> Result<Computed> {
    if arm == Arm::NativePacked {
        return Ok((dispatch(backend, arm, texts)?, vec![], 0));
    }
    // Serial Frontend::run retokenizes. This first pass is required to reject an invalid
    // whole request before executing any prefix, and intentionally remains in the timer.
    let ids = token_ids(tok, texts)?;
    let groups = group_count(arm, &ids.iter().map(Vec::len).collect::<Vec<_>>())?;
    let vectors = dispatch(backend, arm, texts)?;
    Ok((vectors, ids, groups))
}
fn token_ids(tok: &Tokenizer, texts: &[&str]) -> Result<Vec<Vec<u32>>> {
    let mut tok = tok.clone();
    tok.with_truncation(None)
        .map_err(|e| anyhow::anyhow!("disable truncation: {e}"))?;
    tok.with_padding(None);
    let ids: Vec<Vec<u32>> = texts
        .iter()
        .map(|t| {
            tok.encode(*t, true)
                .map(|e| e.get_ids().to_vec())
                .map_err(|e| anyhow::anyhow!("tokenize: {e}"))
        })
        .collect::<Result<_>>()?;
    group_count(
        Arm::NativePacked,
        &ids.iter().map(Vec::len).collect::<Vec<_>>(),
    )?;
    Ok(ids)
}
fn group_count(arm: Arm, lengths: &[usize]) -> Result<usize> {
    let groups = passage_batches(lengths, 512, arm.rows()).map_err(anyhow::Error::msg)?;
    Ok(if arm == Arm::NativePacked {
        groups.len()
    } else {
        lengths.len()
    })
}
fn validate_vectors(vectors: &[Vec<f32>], count: usize) -> Result<()> {
    ensure!(vectors.len() == count, "wrong output count");
    for v in vectors {
        ensure!(
            v.len() == 768 && v.iter().all(|x| x.is_finite()),
            "wrong output dimension or nonfinite output"
        );
        let norm: f32 = v.iter().map(|x| x * x).sum();
        ensure!(
            norm.is_finite() && (norm - 1.).abs() < 1e-4,
            "output is not normalized/nonzero"
        );
    }
    Ok(())
}
// Linux process CPU clock, not wall time or child CPU usage. RSS below is lifetime VmHWM.
fn process_cpu_ns() -> Result<u64> {
    #[repr(C)]
    struct Timespec {
        sec: std::os::raw::c_long,
        ns: std::os::raw::c_long,
    }
    unsafe extern "C" {
        fn clock_gettime(clock: i32, tp: *mut Timespec) -> i32;
    }
    let mut ts = Timespec { sec: 0, ns: 0 };
    ensure!(
        unsafe { clock_gettime(2, &mut ts) } == 0,
        "CLOCK_PROCESS_CPUTIME_ID failed"
    );
    Ok(ts.sec as u64 * 1_000_000_000 + ts.ns as u64)
}
fn peak_rss_bytes() -> Result<u64> {
    let status = fs::read_to_string("/proc/self/status")?;
    let hwm = status
        .lines()
        .find(|s| s.starts_with("VmHWM:"))
        .context("VmHWM unavailable")?;
    Ok(hwm
        .split_whitespace()
        .nth(1)
        .context("bad VmHWM")?
        .parse::<u64>()?
        * 1024)
}
fn handle(v: &Value, loaded: &mut Option<Loaded>) -> Result<Value> {
    match field(v, "op")? {
        "load" => {
            ensure!(loaded.is_none(), "already loaded; unload first");
            let (model, result) = Loaded::load(
                Arm::parse(field(v, "arm")?)?,
                Path::new(field(v, "root")?),
                Path::new(field(v, "scenario")?),
            )?;
            *loaded = Some(model);
            Ok(result)
        }
        "embed" => loaded
            .as_ref()
            .context("no loaded model")?
            .embed(&texts(v)?),
        "unload" | "quit" => {
            *loaded = None;
            Ok(json!({}))
        }
        _ => bail!("unknown op"),
    }
}
// Read at most MAX_LINE+1 bytes; an oversized line is fatal, so no unbounded drain is needed.
fn read_line_bounded(reader: &mut impl BufRead) -> Result<Option<Vec<u8>>> {
    let mut bytes = Vec::new();
    let count = reader
        .take((MAX_LINE + 1) as u64)
        .read_until(b'\n', &mut bytes)?;
    if count == 0 {
        return Ok(None);
    }
    ensure!(
        bytes.len() <= MAX_LINE,
        "request line exceeds {MAX_LINE} bytes"
    );
    Ok(Some(bytes))
}
fn serve(input: &mut impl BufRead, output: &mut impl Write) -> Result<()> {
    let mut loaded = None;
    loop {
        let line = match read_line_bounded(input) {
            Ok(Some(line)) => line,
            Ok(None) => break,
            Err(e) => {
                writeln!(
                    output,
                    "{}",
                    json!({"schema_version":1,"id":null,"ok":false,"error":format!("{e:#}")})
                )?;
                output.flush()?;
                break;
            }
        };
        let parsed = parse_request(&line);
        let id = parsed
            .as_ref()
            .ok()
            .map(|v| v["id"].clone())
            .unwrap_or_else(|| {
                serde_json::from_slice::<Value>(&line)
                    .ok()
                    .and_then(|v| v.get("id").filter(|id| id.is_string()).cloned())
                    .unwrap_or(Value::Null)
            });
        let quit = parsed.as_ref().is_ok_and(|v| v["op"] == "quit");
        let result = parsed.and_then(|v| handle(&v, &mut loaded));
        let mut response = match result {
            Ok(v) => v,
            Err(e) => json!({"error":format!("{e:#}")}),
        };
        response["schema_version"] = json!(1);
        response["id"] = id;
        response["ok"] = json!(response.get("error").is_none());
        writeln!(output, "{response}")?;
        output.flush()?;
        if quit {
            break;
        }
    }
    drop(loaded);
    Ok(())
}
fn main() -> Result<()> {
    serve(&mut io::stdin().lock(), &mut io::stdout().lock())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn protocol_accepts_commands_and_rejects_bad_inputs() {
        for op in ["load", "embed", "unload", "quit"] {
            let mut req = json!({"schema_version":1,"id":"a","op":op});
            if op == "load" {
                req["arm"] = json!("native_serial");
                req["root"] = json!("/tmp");
                req["scenario"] = json!("/tmp/missing.toml");
            }
            if op == "embed" {
                req["texts"] = json!(["", "a"]);
            }
            assert!(parse_request(req.to_string().as_bytes()).is_ok(), "{op}");
        }
        for req in [
            json!({"schema_version":2,"id":"a","op":"quit"}),
            json!({"schema_version":1,"id":"a","op":"unknown"}),
            json!({"schema_version":1,"id":"a","op":"embed","texts":[1]}),
            json!({"schema_version":1,"id":"a","op":"load","arm":"unknown","root":"/tmp","scenario":"/tmp/x"}),
            json!({"schema_version":1,"id":"a","op":"load","arm":"native_serial","root":"/tmp"}),
        ] {
            assert!(parse_request(req.to_string().as_bytes()).is_err());
        }
        assert!(parse_request(b"not json").is_err());
    }
    #[test]
    fn output_bits_preserve_signed_zero_and_exact_values() {
        assert_eq!(
            vector_bits(&[vec![0., -0., 1., f32::from_bits(1)]]),
            vec![vec![0, 0x80000000, 0x3f800000, 1]]
        );
    }
}

#[cfg(test)]
mod controls {
    use super::*;
    use std::{cell::RefCell, collections::HashMap};
    use tokenizers::{
        models::wordlevel::WordLevel, pre_tokenizers::whitespace::Whitespace,
        processors::template::TemplateProcessing,
    };

    fn tokenizer() -> Tokenizer {
        let vocab: HashMap<String, u32> =
            [("[UNK]", 0), ("[CLS]", 1), ("[SEP]", 2), ("a", 3), ("b", 4)]
                .into_iter()
                .map(|(s, i)| (s.into(), i))
                .collect();
        let mut tok = Tokenizer::new(
            WordLevel::builder()
                .vocab(vocab)
                .unk_token("[UNK]".into())
                .build()
                .unwrap(),
        );
        tok.with_pre_tokenizer(Some(Whitespace));
        tok.with_post_processor(Some(
            TemplateProcessing::builder()
                .try_single("[CLS] $A [SEP]")
                .unwrap()
                .special_tokens(vec![("[CLS]", 1), ("[SEP]", 2)])
                .build()
                .unwrap(),
        ));
        tok
    }
    #[derive(Default)]
    struct Fake {
        serial: RefCell<Vec<String>>,
        packed: RefCell<Vec<Vec<String>>>,
    }
    impl Execute for Fake {
        fn serial(&self, text: &str) -> Result<Vec<f32>> {
            self.serial.borrow_mut().push(text.into());
            Ok(vec![text.len() as f32])
        }
        fn packed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
            self.packed
                .borrow_mut()
                .push(texts.iter().map(|s| s.to_string()).collect());
            Ok(texts.iter().map(|s| vec![s.len() as f32]).collect())
        }
    }
    #[test]
    fn call_site_packed_once_serial_per_passage_and_empty_never_calls() {
        for arm in [Arm::NativePacked, Arm::NativeSerial, Arm::FastSerial] {
            let fake = Fake::default();
            let (vectors, ids, groups) =
                compute(&fake, arm, &tokenizer(), &["a", "b a", "a", ""]).unwrap();
            let expected_ids = vec![vec![1, 3, 2], vec![1, 4, 3, 2], vec![1, 3, 2], vec![1, 2]];
            if arm == Arm::NativePacked {
                assert!(
                    ids.is_empty(),
                    "packed diagnostics are deferred until after the timer"
                );
                assert_eq!(
                    token_ids(&tokenizer(), &["a", "b a", "a", ""]).unwrap(),
                    expected_ids
                );
            } else {
                assert_eq!(ids, expected_ids);
            }
            assert_eq!(vectors, vec![vec![1.], vec![3.], vec![1.], vec![0.]]);
            if arm == Arm::NativePacked {
                assert!(fake.serial.borrow().is_empty());
                assert_eq!(&*fake.packed.borrow(), &[vec!["a", "b a", "a", ""]]);
                assert_eq!(groups, 0, "packed grouping diagnostics are deferred");
            } else {
                assert!(fake.packed.borrow().is_empty());
                assert_eq!(&*fake.serial.borrow(), &["a", "b a", "a", ""]);
                assert_eq!(groups, 4);
            }
            let empty = Fake::default();
            let (v, ids, g) = compute(&empty, arm, &tokenizer(), &[]).unwrap();
            assert!(v.is_empty() && ids.is_empty() && g == 0);
            assert!(empty.serial.borrow().is_empty() && empty.packed.borrow().is_empty());
        }
    }
    #[test]
    fn full_request_overlength_rejected_before_any_prefix_execution() {
        let too_long = "a ".repeat(511);
        let exact = "a ".repeat(510);
        assert_eq!(token_ids(&tokenizer(), &[&exact]).unwrap()[0].len(), 512);
        // Packed whole-request rejection is owned by Model::embed_batch, not this serial adapter.
        for arm in [Arm::NativeSerial, Arm::FastSerial] {
            let fake = Fake::default();
            assert!(compute(&fake, arm, &tokenizer(), &["a", &too_long]).is_err());
            assert!(fake.serial.borrow().is_empty() && fake.packed.borrow().is_empty());
        }
        assert_eq!(
            group_count(Arm::NativePacked, &[512, 512, 2, 512, 512, 2]).unwrap(),
            3
        );
        assert!(group_count(Arm::NativePacked, &[0]).is_err());
        assert!(group_count(Arm::NativeSerial, &[513]).is_err());
    }
    #[test]
    fn ambient_selectors_rejected_without_environment_mutation() {
        for key in [
            "NPU_PRECISION",
            "NPU_TILE",
            "NPU_MM2_PIPELINE",
            "NPU_ENC_FFN_RESIDENT",
            "BERT_RESIDENT_FFN",
            "PARAKEET_SIM_BF16_GEMM_OUT",
        ] {
            assert!(
                ambient_is_safe([(key.into(), "0".into())]).is_err(),
                "{key}"
            );
        }
        assert!(ambient_is_safe([("RAYON_NUM_THREADS".into(), "2".into())]).is_ok());
    }
    fn fake_legacy_pair(dir: &Path, stem: &str) {
        fs::write(kr::xclbin_path(dir, stem), b"fake non-executable artifact").unwrap();
        fs::write(kr::insts_path(dir, stem), b"00000000\n").unwrap();
    }
    fn fake_contract(dir: &Path, stem: &str, n: usize, arm: Arm) {
        fake_legacy_pair(dir, stem);
        let (m, k, t) = arm.tile();
        let c = json!({"rows":arm.rows(),"k":800,"n":n,"tile":[m,k,t],"precision":arm.precision_name(),
            "xclbin_sha256":file_identity(&kr::xclbin_path(dir,stem)).unwrap()["sha256"],
            "insts_sha256":file_identity(&kr::insts_path(dir,stem)).unwrap()["sha256"]});
        fs::write(dir.join(format!("contract_{stem}.json")), c.to_string()).unwrap();
    }
    #[test]
    fn fast_legacy_pairs_need_hashes_but_no_native_sidecar() {
        let dir = tempfile::tempdir().unwrap();
        fake_legacy_pair(dir.path(), "fast");
        let artifacts = check_contract(dir.path(), "fast", 3072, Arm::FastSerial).unwrap();
        assert_eq!(
            artifacts,
            vec![
                file_identity(&kr::xclbin_path(dir.path(), "fast")).unwrap(),
                file_identity(&kr::insts_path(dir.path(), "fast")).unwrap()
            ]
        );
        for arm in [Arm::NativeSerial, Arm::NativePacked] {
            assert!(check_contract(dir.path(), "fast", 3072, arm).is_err());
        }
        fs::remove_file(kr::insts_path(dir.path(), "fast")).unwrap();
        assert!(check_contract(dir.path(), "fast", 3072, Arm::FastSerial).is_err());
    }
    #[test]
    fn contracts_reject_missing_wrong_capacity_and_replaced_artifacts_host_only() {
        let dir = tempfile::tempdir().unwrap();
        let arm = Arm::NativePacked;
        assert!(check_contract(dir.path(), "stub", 768, arm).is_err());
        fake_contract(dir.path(), "stub", 768, arm);
        assert_eq!(
            check_contract(dir.path(), "stub", 768, arm).unwrap().len(),
            3
        );
        assert!(check_contract(dir.path(), "stub", 768, Arm::NativeSerial).is_err());
        assert!(check_contract(dir.path(), "stub", 3072, arm).is_err());
        fs::write(kr::insts_path(dir.path(), "stub"), b"changed").unwrap();
        assert!(check_contract(dir.path(), "stub", 768, arm)
            .unwrap_err()
            .to_string()
            .contains("hash mismatch"));
        fake_contract(dir.path(), "wrong_precision", 3072, arm);
        let path = dir.path().join("contract_wrong_precision.json");
        let mut c: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        c["precision"] = json!("bf16");
        fs::write(path, c.to_string()).unwrap();
        assert!(check_contract(dir.path(), "wrong_precision", 3072, arm).is_err());
    }
    #[test]
    fn missing_scenario_and_source_baking_fail_in_host_preflight() {
        let dir = tempfile::tempdir().unwrap();
        assert!(preflight(Arm::NativePacked, dir.path(), &dir.path().join("missing")).is_err());
        let cfg = include_str!("../../../scenarios/bge-base.toml");
        // The source rejection precedes tokenizer/weights/kernel requirements.
        let cfg = cfg.replace(
            "[artifacts]",
            "[artifacts]\nsource = 'path:/must-not-be-baked'",
        );
        let scenario = dir.path().join("scenario.toml");
        fs::write(&scenario, cfg).unwrap();
        assert!(preflight(Arm::NativePacked, dir.path(), &scenario)
            .err()
            .unwrap()
            .to_string()
            .contains("baking"));
    }
    #[test]
    fn all_constructions_preflight_fake_files_without_loading_weights_or_device() {
        for arm in [Arm::NativePacked, Arm::NativeSerial, Arm::FastSerial] {
            let dir = tempfile::tempdir().unwrap();
            let root = dir.path();
            let scenario = root.join("scenario.toml");
            fs::write(
                &scenario,
                format!(
                    r#"
[scenario]
kind = "embeddings"
name = "fake-bert"
[model]
hidden = 768
ff = 3072
n_heads = 12
head_dim = 64
n_layers = 12
max_seq = 512
precision = "{}"
[artifacts]
weights = "weights"
tokenizer = "tokenizer.json"
[embeddings]
pooling = "cls"
normalize = true
{}
"#,
                    arm.precision_name(),
                    if arm == Arm::NativePacked {
                        "batch_rows = 1024"
                    } else {
                        ""
                    }
                ),
            )
            .unwrap();
            tokenizer()
                .save(root.join("tokenizer.json"), false)
                .unwrap();
            let emb = root.join("weights/emb");
            fs::create_dir_all(&emb).unwrap();
            // Deliberately not npy tensors: preflight checks presence/hash, never loads real weights.
            for k in ["word_emb", "pos_emb", "type_emb", "emb_ln_w", "emb_ln_b"] {
                fs::write(emb.join(format!("{k}.npy")), b"fake fixture only").unwrap();
            }
            for l in 0..12 {
                let layer = root.join(format!("weights/L{l}"));
                fs::create_dir_all(&layer).unwrap();
                for k in [
                    "q_w",
                    "q_b",
                    "k_w",
                    "k_b",
                    "v_w",
                    "v_b",
                    "attn_out_w",
                    "attn_out_b",
                    "ffn1_w",
                    "ffn1_b",
                    "ffn2_w",
                    "ffn2_b",
                    "attn_ln_w",
                    "attn_ln_b",
                    "out_ln_w",
                    "out_ln_b",
                ] {
                    fs::write(layer.join(format!("{k}.npy")), b"fake fixture only").unwrap();
                }
            }
            let kernels = root.join("kernels/whole_array");
            fs::create_dir_all(&kernels).unwrap();
            let tuning = npu_asr::tuning::TuningConfig::baked_default(arm.precision());
            let sh = arm.shape();
            let mut required = vec![(xclbin_stem(&sh, &tuning), 3072)];
            required.extend([768, 3072].map(|n| (sh.stream_stem(&tuning, n, 0), n)));
            for (stem, n) in &required {
                if arm == Arm::FastSerial {
                    fake_legacy_pair(&kernels, stem);
                } else {
                    fake_contract(&kernels, stem, *n, arm);
                }
            }
            if arm != Arm::FastSerial {
                let (stem, n) = &required[0];
                fs::remove_file(kernels.join(format!("contract_{stem}.json"))).unwrap();
                assert!(
                    preflight(arm, root, &scenario).is_err(),
                    "native requires an actual sidecar"
                );
                fake_contract(&kernels, stem, *n, arm);
            }
            let p = preflight(arm, root, &scenario).unwrap();
            assert_eq!(p.identity["rows"], arm.rows());
            assert_eq!(p.identity["precision"], arm.precision_name());
            assert_eq!(
                p.identity["attestation"],
                if arm == Arm::FastSerial {
                    "hashed_legacy_only"
                } else {
                    "native_contract"
                }
            );
            let artifacts = p.identity["artifacts"].as_array().unwrap();
            assert!(artifacts.len() > 190);
            if arm == Arm::FastSerial {
                assert!(artifacts
                    .iter()
                    .all(|a| !a["path"].as_str().unwrap().contains("contract_")));
                for (stem, _) in &required {
                    for path in [
                        kr::xclbin_path(&kernels, stem),
                        kr::insts_path(&kernels, stem),
                    ] {
                        assert!(artifacts.contains(&file_identity(&path).unwrap()));
                    }
                }
            }
            fs::write(
                kr::insts_path(&kernels, &sh.stream_stem(&tuning, 768, 0)),
                b"replaced",
            )
            .unwrap();
            if arm == Arm::FastSerial {
                let changed = preflight(arm, root, &scenario).unwrap();
                assert_ne!(
                    changed.identity["artifacts"], p.identity["artifacts"],
                    "legacy hashes must reflect the actual files"
                );
            } else {
                assert!(preflight(arm, root, &scenario).is_err());
                fake_contract(&kernels, &sh.stream_stem(&tuning, 768, 0), 768, arm);
            }
            fs::remove_file(emb.join("word_emb.npy")).unwrap();
            assert!(preflight(arm, root, &scenario).is_err());
        }
    }
    #[test]
    fn malformed_responses_cannot_be_successful_outputs() {
        assert!(validate_vectors(&[], 1).is_err());
        assert!(validate_vectors(&[vec![1.]], 1).is_err());
        for bad in [0., f32::NAN, f32::INFINITY] {
            assert!(validate_vectors(&[vec![bad; 768]], 1).is_err());
        }
        let mut v = vec![0.; 768];
        v[0] = 1.;
        assert!(validate_vectors(&[v], 1).is_ok());
    }
    #[test]
    fn protocol_loop_errors_echo_ids_emit_no_timing_and_quit_stops() {
        let input = b"not json\n{\"schema_version\":1,\"id\":\"bad\",\"op\":\"unknown\"}\n{\"schema_version\":1,\"id\":\"empty\",\"op\":\"embed\",\"texts\":[]}\n{\"schema_version\":1,\"id\":\"u\",\"op\":\"unload\"}\n{\"schema_version\":1,\"id\":\"q\",\"op\":\"quit\"}\nnot processed\n";
        let mut output = Vec::new();
        serve(&mut io::Cursor::new(input), &mut output).unwrap();
        let responses: Vec<Value> = std::str::from_utf8(&output)
            .unwrap()
            .lines()
            .map(|s| serde_json::from_str(s).unwrap())
            .collect();
        assert_eq!(responses.len(), 5);
        for (i, id) in [
            Value::Null,
            json!("bad"),
            json!("empty"),
            json!("u"),
            json!("q"),
        ]
        .iter()
        .enumerate()
        {
            assert_eq!(&responses[i]["id"], id);
            assert_eq!(responses[i]["schema_version"], 1);
            assert_eq!(responses[i]["ok"], i >= 3);
            assert!(responses[i].get("elapsed_ns").is_none());
            if i < 3 {
                assert!(responses[i]["error"].is_string());
            }
        }
    }
    #[test]
    fn bounded_lines_and_unloaded_state_are_host_only() {
        assert!(read_line_bounded(&mut io::Cursor::new(vec![b'x'; MAX_LINE + 1])).is_err());
        assert!(read_line_bounded(&mut io::Cursor::new(vec![]))
            .unwrap()
            .is_none());
        let mut loaded = None;
        assert!(handle(&json!({"op":"embed","texts":[]}), &mut loaded).is_err());
        for op in ["unload", "quit"] {
            assert!(handle(&json!({"op":op}), &mut loaded).is_ok());
        }
        assert!(process_cpu_ns().unwrap() > 0);
        assert!(peak_rss_bytes().unwrap() > 0);
    }
}
