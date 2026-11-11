//! Capture-only Qwen embedding evidence. This caller never qualifies parity or promotes an index.
use anyhow::{bail, ensure, Context, Result};
use npu_models::{
    api::{EmbeddingCapture, EmbeddingInputType, EmbeddingOptions},
    Model,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{env, fs, fs::OpenOptions, io::Write, path::PathBuf, time::Instant};

const SOURCE_BYTES: &[u8] = include_bytes!("qwen_embedding_capture.rs");

struct Cli {
    scenario: PathBuf,
    root: PathBuf,
    input: PathBuf,
    output: PathBuf,
    input_type: EmbeddingInputType,
}

fn usage() -> &'static str {
    "usage: qwen_embedding_capture --scenario PATH --root PATH --input PATH --output PATH [--input-type query|document]"
}

fn parse_cli(args: impl IntoIterator<Item = String>) -> Result<Cli> {
    let mut scenario = None;
    let mut root = None;
    let mut input = None;
    let mut output = None;
    let mut input_type = None;
    let mut args = args.into_iter();
    let _program = args.next();
    while let Some(flag) = args.next() {
        let value = args
            .next()
            .with_context(|| format!("missing value for {flag}; {}", usage()))?;
        match flag.as_str() {
            "--scenario" if scenario.is_none() => scenario = Some(PathBuf::from(value)),
            "--root" if root.is_none() => root = Some(PathBuf::from(value)),
            "--input" if input.is_none() => input = Some(PathBuf::from(value)),
            "--output" if output.is_none() => output = Some(PathBuf::from(value)),
            "--input-type" if input_type.is_none() => {
                input_type = Some(match value.as_str() {
                    "query" => EmbeddingInputType::Query,
                    "document" => EmbeddingInputType::Document,
                    _ => bail!("--input-type must be query or document"),
                });
            }
            _ => bail!("unknown or duplicate argument {flag}; {}", usage()),
        }
    }
    Ok(Cli {
        scenario: scenario.context("missing --scenario")?,
        root: root.context("missing --root")?,
        input: input.context("missing --input")?,
        output: output.context("missing --output")?,
        input_type: input_type.unwrap_or_default(),
    })
}

fn input_type_name(input_type: EmbeddingInputType) -> &'static str {
    match input_type {
        EmbeddingInputType::Query => "query",
        EmbeddingInputType::Document => "document",
    }
}

fn read_texts(input: &PathBuf) -> Result<(Vec<u8>, Vec<String>)> {
    let bytes = fs::read(input).with_context(|| format!("read input {}", input.display()))?;
    let value: Value = serde_json::from_slice(&bytes).context("input must be JSON")?;
    let array = value
        .as_array()
        .context("input must be a nonempty string array")?;
    ensure!(!array.is_empty(), "input must be a nonempty string array");
    let texts = array
        .iter()
        .enumerate()
        .map(|(i, value)| {
            let text = value
                .as_str()
                .with_context(|| format!("input item {i} must be a string"))?;
            ensure!(!text.is_empty(), "input item {i} must be a nonempty string");
            Ok(text.to_owned())
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((bytes, texts))
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Matches Python's json.dumps(texts, ensure_ascii=False, separators=(',', ':')).encode().
fn text_identity(texts: &[String]) -> Result<String> {
    Ok(sha256(serde_json::to_string(texts)?.as_bytes()))
}

fn full_dimension(capture: &EmbeddingCapture) -> Result<usize> {
    let dimension = capture
        .metadata
        .profile
        .get("dimension")
        .and_then(Value::as_u64)
        .context("capture profile lacks a positive integer full dimension")?;
    usize::try_from(dimension)
        .context("capture profile dimension does not fit usize")
        .and_then(|d| {
            if d == 0 {
                bail!("capture profile dimension must be positive")
            } else {
                Ok(d)
            }
        })
}

fn validate_capture(capture: &EmbeddingCapture, text_count: usize) -> Result<usize> {
    let dimension = full_dimension(capture)?;
    ensure!(
        capture
            .metadata
            .runtime
            .get("qualification")
            .and_then(Value::as_str)
            == Some("unqualified"),
        "capture runtime must remain unqualified"
    );
    ensure!(
        capture
            .metadata
            .runtime
            .get("index_compatibility")
            .and_then(Value::as_str)
            == Some("not-qualified"),
        "capture runtime must not promote index compatibility"
    );
    ensure!(
        capture.vectors.len() == text_count
            && capture.input_ids.len() == text_count
            && capture.attention_mask.len() == text_count,
        "capture row counts disagree with input"
    );
    for row in 0..text_count {
        let vector = &capture.vectors[row];
        ensure!(
            vector.len() == dimension,
            "vector row {row} is not full dimension {dimension}"
        );
        ensure!(
            vector.iter().all(|v| v.is_finite()),
            "vector row {row} contains nonfinite values"
        );
        let norm = vector
            .iter()
            .map(|v| (*v as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        ensure!(
            norm.is_finite() && norm > 0.0 && (norm - 1.0).abs() <= 1e-4,
            "vector row {row} is not finite L2-normalized"
        );
        let ids = &capture.input_ids[row];
        let mask = &capture.attention_mask[row];
        ensure!(
            !ids.is_empty() && ids.len() == mask.len(),
            "token/mask shape invalid for row {row}"
        );
        ensure!(
            mask.iter().all(|&v| v == 0 || v == 1),
            "attention mask row {row} is not binary"
        );
        let first = mask
            .iter()
            .position(|&v| v == 1)
            .context("attention mask has no tokens")?;
        let last = mask
            .iter()
            .rposition(|&v| v == 1)
            .expect("first token establishes last token");
        ensure!(
            mask[first..=last].iter().all(|&v| v == 1),
            "attention mask row {row} has a gap"
        );
    }
    Ok(dimension)
}

fn repeat_max_abs(
    first: &EmbeddingCapture,
    warm: &EmbeddingCapture,
    text_count: usize,
) -> Result<f32> {
    let dimension = validate_capture(first, text_count)?;
    ensure!(
        validate_capture(warm, text_count)? == dimension,
        "repeat capture dimension changed"
    );
    ensure!(
        first.metadata == warm.metadata,
        "repeat capture metadata changed"
    );
    ensure!(
        first.input_ids == warm.input_ids,
        "repeat capture token IDs changed"
    );
    ensure!(
        first.attention_mask == warm.attention_mask,
        "repeat capture attention masks changed"
    );
    let mut maximum = 0.0f32;
    for row in 0..text_count {
        ensure!(
            first.vectors[row].len() == warm.vectors[row].len(),
            "repeat vector shape changed at row {row}"
        );
        for column in 0..dimension {
            maximum = maximum.max((first.vectors[row][column] - warm.vectors[row][column]).abs());
        }
    }
    ensure!(maximum.is_finite(), "repeat difference is nonfinite");
    Ok(maximum)
}

fn record(
    first: &EmbeddingCapture,
    input_type: EmbeddingInputType,
    input_bytes: &[u8],
    texts: &[String],
    repeat_max_abs: f32,
    load_s: f64,
    first_s: f64,
    warm_s: f64,
) -> Result<Value> {
    validate_capture(first, texts.len())?;
    ensure!(repeat_max_abs.is_finite(), "repeat difference is nonfinite");
    Ok(json!({
        "qualification": "capture-only-unqualified",
        "profile": first.metadata.profile,
        "runtime": first.metadata.runtime,
        "code_sha256": { "qwen_embedding_capture.rs": sha256(SOURCE_BYTES) },
        "input_type": input_type_name(input_type),
        "input_sha256": sha256(input_bytes),
        "texts_sha256": text_identity(texts)?,
        "input_ids": first.input_ids,
        "attention_mask": first.attention_mask,
        "vectors": first.vectors,
        "repeat_max_abs": repeat_max_abs,
        "timings": { "load_s": load_s, "first_capture_s": first_s, "warm_capture_s": warm_s }
    }))
}

fn write_new(path: &PathBuf, value: &Value) -> Result<()> {
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| format!("create new output {}", path.display()))?;
    serde_json::to_writer_pretty(&mut output, value)?;
    output.write_all(b"\n")?;
    output.sync_all()?;
    Ok(())
}

fn run(cli: Cli) -> Result<()> {
    match fs::symlink_metadata(&cli.output) {
        Ok(_) => bail!("output already exists: {}", cli.output.display()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error).context("inspect output path"),
    }
    let (input_bytes, texts) = read_texts(&cli.input)?;
    let refs = texts.iter().map(String::as_str).collect::<Vec<_>>();
    let options = EmbeddingOptions {
        input_type: cli.input_type,
        dimensions: None,
    };

    let started = Instant::now();
    let model = Model::load_in(&cli.scenario, &cli.root)?;
    let load_s = started.elapsed().as_secs_f64();
    let started = Instant::now();
    let first = model.embed_batch_with_capture(&refs, &options)?;
    let first_s = started.elapsed().as_secs_f64();
    let started = Instant::now();
    let warm = model.embed_batch_with_capture(&refs, &options)?;
    let warm_s = started.elapsed().as_secs_f64();
    let max_abs = repeat_max_abs(&first, &warm, texts.len())?;
    let value = record(
        &first,
        cli.input_type,
        &input_bytes,
        &texts,
        max_abs,
        load_s,
        first_s,
        warm_s,
    )?;
    write_new(&cli.output, &value)
}

fn main() {
    if let Err(error) = parse_cli(env::args()).and_then(run) {
        eprintln!("qwen_embedding_capture: {error}");
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use npu_models::api::EmbeddingMetadata;
    use std::process::Command;

    #[test]
    #[ignore = "requires an explicitly prepared isolated NPU fixture"]
    fn real_engine_contract_probe() -> Result<()> {
        use npu_models::EngineError;
        use std::{fs::File, path::Path};

        fn trace(output: &mut File, value: Value) -> Result<()> {
            serde_json::to_writer(&mut *output, &value)?;
            output.write_all(b"\n")?;
            output.flush()?;
            Ok(())
        }

        fn capture_rows(
            model: &Model,
            output: &mut File,
            label: &str,
            texts: &[&str],
            options: &EmbeddingOptions,
        ) -> Result<EmbeddingCapture> {
            let result = model.embed_batch_with_capture(texts, options);
            match &result {
                Ok(capture) => trace(output, json!({
                    "event": "capture", "label": label,
                    "input_type": input_type_name(options.input_type),
                    "dimensions": options.dimensions, "texts": texts, "capture": capture
                }))?,
                Err(error) => trace(output, json!({
                    "event": "capture-error", "label": label, "error": format!("{error:?}")
                }))?,
            }
            let capture = result?;
            ensure!(validate_capture(&capture, texts.len())? == 1024, "{label}: dimension");
            for (ids, mask) in capture.input_ids.iter().zip(&capture.attention_mask) {
                ensure!(ids.len() <= 512 && mask.iter().all(|&v| v == 1),
                    "{label}: expected serial unpadded inputs within capacity");
            }
            Ok(capture)
        }

        fn same_vector(actual: &[f32], expected: &[f32]) -> Result<()> {
            ensure!(actual.len() == 1024 && expected.len() == 1024, "vector dimension");
            ensure!(actual.iter().all(|v| v.is_finite()), "nonfinite vector");
            let norm = actual.iter().map(|&v| (v as f64).powi(2)).sum::<f64>().sqrt();
            ensure!(norm > 0.0 && (norm - 1.0).abs() <= 1e-4, "vector normalization");
            let maximum = actual.iter().zip(expected)
                .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            ensure!(maximum <= 1e-6, "repeat max abs {maximum} exceeds 1e-6");
            Ok(())
        }

        fn same_row(actual: &EmbeddingCapture, row: usize, baseline: &EmbeddingCapture) -> Result<()> {
            ensure!(actual.metadata == baseline.metadata, "loaded identity changed");
            ensure!(actual.input_ids[row] == baseline.input_ids[0], "token IDs changed");
            ensure!(actual.attention_mask[row] == baseline.attention_mask[0], "mask changed");
            same_vector(&actual.vectors[row], &baseline.vectors[0])
        }

        let mut output = OpenOptions::new().write(true).create_new(true)
            .open("embedding-contract-output.jsonl")?;
        trace(&mut output, json!({
            "event": "start", "qualification": "integration-proof-only-unqualified",
            "code_sha256": {"qwen_embedding_capture.rs": sha256(SOURCE_BYTES)},
            "coverage_boundary": "invalid enum roles are parser-host coverage; live HTTP invalid roles require separate lifecycle coverage"
        }))?;
        ensure!(env::var("NPU_LLM_REUSE_KV").ok().as_deref() == Some("1"),
            "fixture must explicitly set NPU_LLM_REUSE_KV=1; independent reset must override generation reuse");
        let bytes = fs::read("embedding-contract-input.json")?;
        let fixture: Value = serde_json::from_slice(&bytes)?;
        ensure!(fixture.is_object(), "fixture must be an object");
        let field = |name: &str| -> Result<&str> {
            let value = fixture.get(name).and_then(Value::as_str)
                .with_context(|| format!("fixture lacks string {name}"))?;
            ensure!(!value.is_empty(), "fixture {name} must be nonempty");
            Ok(value)
        };
        let (short, neighbor, long, over, query) =
            (field("short")?, field("neighbor")?, field("long512")?, field("over513")?, field("query")?);
        ensure!(short != neighbor, "neighbor must be unrelated input, not duplicate short");
        trace(&mut output, json!({"event": "fixture", "input_sha256": sha256(&bytes)}))?;
        let loaded = Model::load_in(Path::new("scenario.toml"), Path::new("."));
        match &loaded {
            Ok(model) => trace(&mut output, json!({"event": "loaded", "metadata": model.embedding_metadata()}))?,
            Err(error) => trace(&mut output, json!({"event": "load-error", "error": format!("{error:?}")}))?,
        }
        let model = loaded?;
        let doc = EmbeddingOptions::default();
        let baseline = capture_rows(&model, &mut output, "baseline", &[short], &doc)?;
        let repeated = capture_rows(&model, &mut output, "repeat", &[short], &doc)?;
        same_row(&repeated, 0, &baseline)?;
        let ordered = capture_rows(&model, &mut output, "ordered-duplicates-long-short",
            &[short, neighbor, short, long, short], &doc)?;
        ensure!(ordered.input_ids[3].len() == 512, "long512 must encode to exactly 512 tokens");
        for row in [0, 2, 4] { same_row(&ordered, row, &baseline)?; }
        let reversed = capture_rows(&model, &mut output, "reversed-neighbor", &[neighbor, short], &doc)?;
        same_row(&reversed, 1, &baseline)?;
        let standalone = capture_rows(&model, &mut output, "standalone-after-neighbor", &[short], &doc)?;
        same_row(&standalone, 0, &baseline)?;
        let normal = model.embed_batch(&[short]);
        trace(&mut output, json!({"event": "normal-batch", "result": format!("{normal:?}"),
            "vectors": normal.as_ref().ok()}))?;
        let normal = normal?;
        ensure!(normal.len() == 1, "normal API row count");
        same_vector(&normal[0], &baseline.vectors[0])?;

        let query_options = EmbeddingOptions { input_type: EmbeddingInputType::Query, dimensions: None };
        let query_capture = capture_rows(&model, &mut output, "query", &[query], &query_options)?;
        let doc_query = capture_rows(&model, &mut output, "document-same-query", &[query], &doc)?;
        ensure!(query_capture.input_ids != doc_query.input_ids, "query instruction missing");
        let prefix = baseline.metadata.profile.get("query_prefix").and_then(Value::as_str)
            .context("loaded profile query_prefix missing")?;
        ensure!(!prefix.is_empty(), "query prefix is empty");
        let prepared_query = format!("{prefix}{query}");
        let prepared = capture_rows(&model, &mut output, "document-prepared-query", &[&prepared_query], &doc)?;
        same_row(&prepared, 0, &query_capture)?;

        // Empty batches are rejected by this Qwen API, unlike the generic batch spy contract.
        // Match Unsupported and its validation message, never accept a load/device failure.
        for (label, texts, options, expected) in [
            ("empty-batch", vec![], doc.clone(), "Qwen embedding input is empty"),
            ("empty-string", vec![""], doc.clone(), "Qwen embedding item 0 is empty"),
            ("mixed-empty", vec![short, ""], doc.clone(), "Qwen embedding item 1 is empty"),
            ("over513", vec![over], doc.clone(), "Qwen embedding item 0 has 513 tokens; expected 1..=512, including instruction"),
            ("dimension-one", vec![short], EmbeddingOptions { dimensions: Some(1), ..doc.clone() },
                "Qwen embedding supports only full dimension 1024"),
        ] {
            let result = model.embed_batch_with_options(&texts, &options);
            trace(&mut output, json!({"event": "validation", "label": label,
                "texts": texts, "dimensions": options.dimensions,
                "result": format!("{result:?}"), "vectors": result.as_ref().ok()}))?;
            ensure!(matches!(&result, Err(EngineError::Unsupported(message)) if message == expected),
                "{label}: expected specific validation rejection, got {result:?}");
            let recovered = capture_rows(&model, &mut output, &format!("recovery-{label}"), &[short], &doc)?;
            same_row(&recovered, 0, &baseline)?;
        }
        let full = EmbeddingOptions { dimensions: Some(1024), ..doc };
        let accepted = model.embed_batch_with_options(&[short], &full);
        trace(&mut output, json!({"event": "full-dimension", "result": format!("{accepted:?}"),
            "vectors": accepted.as_ref().ok()}))?;
        let accepted = accepted?;
        ensure!(accepted.len() == 1, "full dimension row count");
        same_vector(&accepted[0], &baseline.vectors[0])?;
        trace(&mut output, json!({"event": "success", "test": "real_engine_contract_probe",
            "qualification": "integration-proof-only-unqualified"}))?;
        Ok(())
    }

    fn capture() -> EmbeddingCapture {
        EmbeddingCapture {
            vectors: vec![vec![0.6, 0.8], vec![1.0, 0.0]],
            input_ids: vec![vec![11, 12], vec![13]],
            attention_mask: vec![vec![1, 1], vec![1]],
            metadata: EmbeddingMetadata {
                profile: json!({"dimension": 2, "format": 1}),
                runtime: json!({"qualification":"unqualified", "index_compatibility":"not-qualified"}),
            },
        }
    }

    #[test]
    fn canonical_text_hash_matches_python_for_unicode_quotes_newlines_and_controls() {
        let texts = vec!["é \"quote\"\n\u{0001}".to_owned(), "漢字".to_owned()];
        let encoded = serde_json::to_string(&texts).unwrap();
        let output = Command::new("python3").args(["-c",
            "import hashlib,json,sys; print(hashlib.sha256(json.dumps(json.loads(sys.argv[1]),ensure_ascii=False,separators=(',',':')).encode()).hexdigest())",
            &encoded]).output().expect("TEST only: existing Python environment");
        assert!(output.status.success());
        assert_eq!(
            text_identity(&texts).unwrap(),
            String::from_utf8(output.stdout).unwrap().trim()
        );
    }

    #[test]
    fn record_has_evaluator_contract_and_capture_only_provenance() {
        let texts = vec!["one".to_owned(), "two".to_owned()];
        let value = record(
            &capture(),
            EmbeddingInputType::Document,
            b"[\"one\",\"two\"]",
            &texts,
            0.0,
            1.0,
            2.0,
            3.0,
        )
        .unwrap();
        for key in [
            "profile",
            "runtime",
            "vectors",
            "input_ids",
            "attention_mask",
            "input_type",
            "input_sha256",
            "texts_sha256",
            "repeat_max_abs",
            "timings",
        ] {
            assert!(value.get(key).is_some(), "missing {key}");
        }
        assert_eq!(value["qualification"], "capture-only-unqualified");
        assert_eq!(value["runtime"]["qualification"], "unqualified");
        assert_eq!(value["input_type"], "document");
        assert_eq!(value["profile"]["dimension"], 2);
    }

    #[test]
    fn query_and_document_labels_are_explicit() {
        assert_eq!(input_type_name(EmbeddingInputType::Query), "query");
        assert_eq!(input_type_name(EmbeddingInputType::Document), "document");
    }

    #[test]
    fn repeat_refuses_changed_metadata_tokens_order_and_shape() {
        let first = capture();
        let mut changed = capture();
        changed.metadata.profile["format"] = json!(2);
        assert!(repeat_max_abs(&first, &changed, 2).is_err());
        let mut changed = capture();
        changed.input_ids[0][0] = 99;
        assert!(repeat_max_abs(&first, &changed, 2).is_err());
        let mut changed = capture();
        changed.vectors[0].pop();
        assert!(repeat_max_abs(&first, &changed, 2).is_err());
        let mut changed = capture();
        changed.input_ids.swap(0, 1);
        assert!(repeat_max_abs(&first, &changed, 2).is_err());
        let mut changed = capture();
        changed.vectors.swap(0, 1);
        assert!(repeat_max_abs(&first, &changed, 2).unwrap() > 0.0);
    }

    #[test]
    fn capture_validation_refuses_nonfinite_zero_norm_and_invalid_masks() {
        let mut bad = capture();
        bad.vectors[0][0] = f32::NAN;
        assert!(validate_capture(&bad, 2).is_err());
        let mut bad = capture();
        bad.vectors[0] = vec![0.0, 0.0];
        assert!(validate_capture(&bad, 2).is_err());
        let mut bad = capture();
        bad.attention_mask[0] = vec![1, 0, 1];
        assert!(validate_capture(&bad, 2).is_err());
        let mut bad = capture();
        bad.attention_mask[0] = vec![1, 2];
        assert!(validate_capture(&bad, 2).is_err());
    }

    #[test]
    fn write_new_refuses_overwrite_and_preserves_existing_bytes() {
        let path = env::temp_dir().join(format!("qwen-capture-test-{}", std::process::id()));
        fs::write(&path, b"preserve me").unwrap();
        assert!(write_new(&path, &json!({"test": true})).is_err());
        assert_eq!(fs::read(&path).unwrap(), b"preserve me");
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn run_rejects_existing_output_and_invalid_input_before_model_load() {
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("existing.json");
        fs::write(&output, b"preserve me").unwrap();
        assert!(run(Cli {
            scenario: directory.path().join("not-loaded.toml"),
            root: directory.path().to_owned(),
            input: directory.path().join("also-not-read.json"),
            output: output.clone(),
            input_type: EmbeddingInputType::Document,
        })
        .is_err());
        assert_eq!(fs::read(&output).unwrap(), b"preserve me");

        let input = directory.path().join("invalid.json");
        fs::write(&input, b"[1]").unwrap();
        let error = run(Cli {
            scenario: directory.path().join("not-loaded.toml"),
            root: directory.path().to_owned(),
            input,
            output: directory.path().join("new.json"),
            input_type: EmbeddingInputType::Document,
        })
        .unwrap_err();
        assert!(error.to_string().starts_with("input item"), "{error}");
    }

    #[test]
    fn duplicate_input_type_is_not_silently_overridden() {
        let args = [
            "capture",
            "--scenario",
            "s",
            "--root",
            "r",
            "--input",
            "i",
            "--output",
            "o",
            "--input-type",
            "query",
            "--input-type",
            "document",
        ];
        assert!(parse_cli(args.into_iter().map(str::to_owned)).is_err());
    }

    #[test]
    fn dangling_output_link_is_rejected_before_model_load() {
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("dangling.json");
        let target = directory.path().join("missing-target");
        std::os::unix::fs::symlink(&target, &output).unwrap();
        let error = run(Cli {
            scenario: directory.path().join("must-not-load.toml"),
            root: directory.path().to_owned(),
            input: directory.path().join("must-not-read.json"),
            output: output.clone(),
            input_type: EmbeddingInputType::Document,
        })
        .unwrap_err();
        assert!(
            error.to_string().contains("output already exists"),
            "{error}"
        );
        assert_eq!(fs::read_link(output).unwrap(), target);
    }

    #[test]
    fn cli_requires_all_paths_and_rejects_invalid_role() {
        assert!(parse_cli(vec!["capture".into(), "--scenario".into(), "s".into()]).is_err());
        assert!(parse_cli(vec![
            "capture".into(),
            "--scenario".into(),
            "s".into(),
            "--root".into(),
            "r".into(),
            "--input".into(),
            "i".into(),
            "--output".into(),
            "o".into(),
            "--input-type".into(),
            "bad".into()
        ])
        .is_err());
    }
}
