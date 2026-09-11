//! Model-directory config: tokenizer, chat template, and stop-token authorities, read as DATA from
//! the checkpoint's own files (`tokenizer.json`, `tokenizer_config.json`, `generation_config.json`)
//! rather than hardcoded per model -- the rail must take a second model without a rewrite.

use std::path::Path;

use tokenizers::Tokenizer;

use crate::api::EngineError;
use crate::llm::chat_template::ChatTemplate;
use crate::llm::tool_syntax::ToolSyntax;
use crate::pipeline::GenerationDefaults;

/// Stop-token ids, each resolved from its OWN authority rather than picked. Qwen-family checkpoints
/// carry two EOS-role tokens that are genuinely different: `<|im_end|>` ends a chat turn (the
/// tokenizer's own `eos_token`) while `<|endoftext|>`, the base-LM's original EOS, shows up as an
/// EXTRA member of `generation_config.json`'s `eos_token_id` list. They are tracked separately
/// rather than merged into one guess, because of the failure that merging causes: two authorities
/// computing different answers for "the" stop token, with the difference noticed only after a
/// mismatched decode length.
#[derive(Debug, Clone)]
pub struct StopTokens {
    /// `tokenizer_config.json`'s `eos_token`, resolved to an id via the tokenizer's own vocab.
    pub chat_eos: u32,
    /// `generation_config.json`'s `eos_token_id` (int or array), verbatim, if that file exists.
    pub generation_eos: Option<Vec<u32>>,
}

impl StopTokens {
    /// Every id that ends generation: the union of `chat_eos` and `generation_eos`.
    pub fn all(&self) -> Vec<u32> {
        let mut v = vec![self.chat_eos];
        if let Some(g) = &self.generation_eos {
            for &id in g {
                if !v.contains(&id) {
                    v.push(id);
                }
            }
        }
        v
    }

    pub fn is_stop(&self, tok: u32) -> bool {
        tok == self.chat_eos || self.generation_eos.as_ref().is_some_and(|g| g.contains(&tok))
    }
}

/// `tokenizer_config.json`'s `eos_token`/`pad_token` fields are either a bare string or an
/// `{"content": "...", ...}` `AddedToken` object, depending on HF version. Handle both rather than
/// assume one shape.
fn token_string(v: &serde_json::Value) -> Option<String> {
    match v {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Object(o) => o.get("content").and_then(|c| c.as_str()).map(String::from),
        _ => None,
    }
}

/// `generation_config.json`'s `eos_token_id` is a single int OR an array of ints across the
/// ecosystem; normalise to a `Vec`.
fn generation_eos_ids(v: &serde_json::Value) -> Option<Vec<u32>> {
    let e = v.get("eos_token_id")?;
    match e {
        serde_json::Value::Number(n) => n.as_u64().map(|x| vec![x as u32]),
        serde_json::Value::Array(a) => Some(a.iter().filter_map(|x| x.as_u64().map(|v| v as u32)).collect()),
        _ => None,
    }
}

/// Resolve [`StopTokens`], failing loud (never picking) if `tokenizer_config.json`'s `eos_token` and
/// `generation_config.json`'s `eos_token_id` disagree about the primary chat EOS.
pub fn resolve_stop_tokens(
    tok: &Tokenizer,
    tokenizer_config: &serde_json::Value,
    generation_config: Option<&serde_json::Value>,
) -> Result<StopTokens, EngineError> {
    let eos_str = tokenizer_config
        .get("eos_token")
        .and_then(token_string)
        .ok_or_else(|| EngineError::Load("tokenizer_config.json has no eos_token".to_string()))?;
    let chat_eos = tok
        .token_to_id(&eos_str)
        .ok_or_else(|| EngineError::Load(format!("tokenizer vocab has no id for eos_token {eos_str:?}")))?;

    let generation_eos = match generation_config.and_then(generation_eos_ids) {
        Some(ids) => {
            if !ids.contains(&chat_eos) {
                return Err(EngineError::Load(format!(
                    "stop-token authorities disagree: tokenizer_config.json eos_token {eos_str:?} = {chat_eos}, \
                     generation_config.json eos_token_id = {ids:?} does not contain it"
                )));
            }
            Some(ids)
        }
        None => None,
    };
    Ok(StopTokens { chat_eos, generation_eos })
}

/// Everything the generation loop needs about one checkpoint: how to tokenize, how to render a
/// chat prompt, and how to recognise the end of generation.
pub struct ModelConfig {
    pub tokenizer: Tokenizer,
    pub chat_template: Option<ChatTemplate>,
    pub stop: StopTokens,
    /// The sampling settings the CHECKPOINT asks for, from `generation_config.json`. It is a
    /// serialized `GenerationConfig` and stating defaults is what it is for, so this is the only
    /// source that knows what the model was tuned for: qwen3-0.6b says `temperature 0.6,
    /// top_p 0.95, top_k 20`, gemma3-270m says `top_p 0.95, top_k 64`. We read that file already
    /// and used to take `eos_token_id` out of it and drop the rest, which answered every
    /// sampling-silent request at the engine's 1.0/1.0/0 -- unfiltered, full entropy, from models
    /// that asked for a narrowed nucleus.
    ///
    /// Empty when the file is absent or names none of them; a request or a scenario still wins.
    pub checkpoint_defaults: GenerationDefaults,
    /// How this model writes a tool call, probed from its own `chat_template`. `None` means the
    /// template has no tool branch, or renders one we cannot scan for -- either way the model is
    /// tool-incapable and `tools` stays a 400 for it.
    ///
    /// Probed ONCE, here, because it costs two template renders and the alternative is Jinja on
    /// the per-request path.
    pub tool_syntax: Option<ToolSyntax>,
}

impl ModelConfig {
    /// Direct construction, e.g. from a tokenizer already loaded elsewhere, or from a test fixture.
    pub fn new(tokenizer: Tokenizer, chat_template: Option<ChatTemplate>, stop: StopTokens) -> Self {
        let tool_syntax = chat_template.as_ref().and_then(ToolSyntax::probe);
        ModelConfig {
            tokenizer,
            chat_template,
            stop,
            checkpoint_defaults: GenerationDefaults::default(),
            tool_syntax,
        }
    }

    /// Same, with the checkpoint's own sampling settings attached.
    pub fn with_checkpoint_defaults(mut self, d: GenerationDefaults) -> Self {
        self.checkpoint_defaults = d;
        self
    }

    /// Load `tokenizer.json` + `tokenizer_config.json` (+ `generation_config.json` if present) from
    /// a model directory.
    pub fn load(dir: &Path) -> Result<ModelConfig, EngineError> {
        let tok_path = dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tok_path)
            .map_err(|e| EngineError::Load(format!("load tokenizer {}: {e}", tok_path.display())))?;

        let tokenizer_config = read_json(&dir.join("tokenizer_config.json"))?;
        let gen_path = dir.join("generation_config.json");
        let generation_config = if gen_path.exists() { Some(read_json(&gen_path)?) } else { None };

        let stop = resolve_stop_tokens(&tokenizer, &tokenizer_config, generation_config.as_ref())?;
        let chat_template =
            tokenizer_config.get("chat_template").and_then(|v| v.as_str()).map(|s| ChatTemplate::new(s.to_string()));

        let checkpoint_defaults =
            generation_config.as_ref().map(generation_sampling).unwrap_or_default();

        let tool_syntax = chat_template.as_ref().and_then(ToolSyntax::probe);
        Ok(ModelConfig { tokenizer, chat_template, stop, checkpoint_defaults, tool_syntax })
    }
}

/// The sampling fields of a `generation_config.json`, ignoring everything else in it.
///
/// `do_sample` is deliberately NOT read. It is a transformers-side switch between two code paths,
/// and the equivalent here is `temperature == 0` (greedy) versus anything else -- honouring
/// `do_sample: false` by forcing greedy would override a caller's explicit temperature from a file,
/// which inverts the precedence this whole chain exists to establish.
fn generation_sampling(v: &serde_json::Value) -> GenerationDefaults {
    let f32_of = |k: &str| v.get(k).and_then(|x| x.as_f64()).map(|x| x as f32);
    GenerationDefaults {
        temperature: f32_of("temperature"),
        top_p: f32_of("top_p"),
        top_k: v.get("top_k").and_then(|x| x.as_u64()).map(|x| x as u32),
        // `max_length` is prompt+completion in transformers, not a completion budget, so it is NOT
        // max_tokens and is not read here. `max_new_tokens` is the comparable one.
        max_tokens: v.get("max_new_tokens").and_then(|x| x.as_u64()).map(|x| x as u32),
        presence_penalty: None,
        frequency_penalty: None,
        repetition_penalty: f32_of("repetition_penalty"),
    }
}

fn read_json(path: &Path) -> Result<serde_json::Value, EngineError> {
    let txt = std::fs::read_to_string(path).map_err(|e| EngineError::Load(format!("read {}: {e}", path.display())))?;
    serde_json::from_str(&txt).map_err(|e| EngineError::Load(format!("parse {}: {e}", path.display())))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::path::PathBuf;
    use tokenizers::models::wordlevel::WordLevelBuilder;

    /// A tiny real `Tokenizer` (WordLevel, no BPE) covering just the ids needed by these tests.
    /// `token_to_id` looks up the model's vocab directly, unaffected by pre-tokenization -- the
    /// same property `asr::whisper::WhisperTokens::resolve` already relies on.
    fn tiny_tokenizer(vocab: &[(&str, u32)]) -> Tokenizer {
        let map: HashMap<String, u32> = vocab.iter().map(|(k, v)| (k.to_string(), *v)).collect();
        let model = WordLevelBuilder::new().vocab(map).unk_token("<unk>".to_string()).build().unwrap();
        Tokenizer::new(model)
    }

    #[test]
    fn generation_config_confirming_the_chat_eos_is_not_a_disagreement() {
        // Mirrors the REAL Qwen3-0.6B files: tokenizer_config eos_token = im_end (151645),
        // generation_config eos_token_id = [151645, 151643]. The chat eos IS a member -> no error,
        // and both ids end up in the union.
        let tok = tiny_tokenizer(&[("<unk>", 0), ("<|im_end|>", 151645), ("<|endoftext|>", 151643)]);
        let tokenizer_config = serde_json::json!({ "eos_token": "<|im_end|>" });
        let generation_config = serde_json::json!({ "eos_token_id": [151645, 151643] });
        let stop = resolve_stop_tokens(&tok, &tokenizer_config, Some(&generation_config)).unwrap();
        assert_eq!(stop.chat_eos, 151645);
        assert_eq!(stop.generation_eos, Some(vec![151645, 151643]));
        let mut all = stop.all();
        all.sort_unstable();
        assert_eq!(all, vec![151643, 151645]);
        assert!(stop.is_stop(151645));
        assert!(stop.is_stop(151643));
        assert!(!stop.is_stop(1));
    }

    #[test]
    fn generation_config_int_form_is_accepted() {
        let tok = tiny_tokenizer(&[("<unk>", 0), ("<|endoftext|>", 50256)]);
        let tokenizer_config = serde_json::json!({ "eos_token": "<|endoftext|>" });
        let generation_config = serde_json::json!({ "eos_token_id": 50256 });
        let stop = resolve_stop_tokens(&tok, &tokenizer_config, Some(&generation_config)).unwrap();
        assert_eq!(stop.chat_eos, 50256);
        assert_eq!(stop.generation_eos, Some(vec![50256]));
    }

    #[test]
    fn missing_generation_config_leaves_generation_eos_none() {
        let tok = tiny_tokenizer(&[("<unk>", 0), ("<|im_end|>", 5)]);
        let tokenizer_config = serde_json::json!({ "eos_token": "<|im_end|>" });
        let stop = resolve_stop_tokens(&tok, &tokenizer_config, None).unwrap();
        assert_eq!(stop.chat_eos, 5);
        assert_eq!(stop.generation_eos, None);
        assert_eq!(stop.all(), vec![5]);
    }

    /// The load-bearing case: the two authorities give genuinely different answers for the primary
    /// EOS. Must fail loud, naming both values, never pick one.
    #[test]
    fn disagreeing_authorities_fail_loud_naming_both_values() {
        let tok = tiny_tokenizer(&[("<unk>", 0), ("<|im_end|>", 151645), ("<|weird|>", 999)]);
        let tokenizer_config = serde_json::json!({ "eos_token": "<|im_end|>" });
        let generation_config = serde_json::json!({ "eos_token_id": [999] }); // does NOT contain 151645
        let err = resolve_stop_tokens(&tok, &tokenizer_config, Some(&generation_config)).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("disagree"), "{msg}");
        assert!(msg.contains("151645"), "must name the tokenizer_config value: {msg}");
        assert!(msg.contains("999"), "must name the generation_config value: {msg}");
    }

    #[test]
    fn eos_token_missing_from_vocab_fails_loud() {
        let tok = tiny_tokenizer(&[("<unk>", 0)]);
        let tokenizer_config = serde_json::json!({ "eos_token": "<|im_end|>" });
        let err = resolve_stop_tokens(&tok, &tokenizer_config, None).unwrap_err();
        assert!(err.to_string().contains("<|im_end|>"));
    }

    /// Real Qwen3-0.6B files: `eos_token` is `<|im_end|>` (151645), `eos_token_id` is
    /// `[151645, 151643]` -- confirms the real checkpoint takes the non-error path.
    /// The whole point of the checkpoint tier, against the file we actually ship. If this ever
    /// reads `None`, every sampling-silent request has silently gone back to the engine's
    /// 1.0/1.0/0 -- unfiltered, full entropy, from a model that asked for a narrowed nucleus.
    #[test]
    fn the_real_qwen3_checkpoint_supplies_its_own_sampling_defaults() {
        let dir = PathBuf::from("../../artifacts/qwen3-0.6b/tokenizer");
        if !dir.join("generation_config.json").exists() {
            return; // artifacts are not always present in a bare checkout
        }
        let cfg = ModelConfig::load(&dir).expect("load real Qwen3 config");
        let d = &cfg.checkpoint_defaults;
        assert_eq!(d.temperature, Some(0.6));
        assert_eq!(d.top_p, Some(0.95));
        assert_eq!(d.top_k, Some(20));
        // `max_length` is prompt+completion, not a completion budget, so it must NOT become one.
        assert_eq!(d.max_tokens, None);
    }

    /// A checkpoint that names none of them must not invent any -- the engine tier decides.
    #[test]
    fn a_generation_config_without_sampling_fields_yields_nothing() {
        let v = serde_json::json!({"eos_token_id": 2, "bos_token_id": 1});
        assert_eq!(generation_sampling(&v), GenerationDefaults::default());
    }

    /// `do_sample` is deliberately not read: honouring `false` by forcing greedy would let a file
    /// override a caller's explicit temperature, inverting the precedence.
    #[test]
    fn do_sample_is_not_read_as_a_temperature() {
        let v = serde_json::json!({"do_sample": false, "temperature": 0.7});
        let d = generation_sampling(&v);
        assert_eq!(d.temperature, Some(0.7));
    }

    #[test]
    fn real_qwen3_config_files_resolve_without_disagreement() {
        let dir = qwen3_snapshot_dir();
        let Some(dir) = dir else {
            eprintln!("SKIP: Qwen3-0.6B HF cache missing -- `huggingface-cli download Qwen/Qwen3-0.6B`");
            return;
        };
        let cfg = ModelConfig::load(&dir).expect("load real Qwen3 config");
        assert_eq!(cfg.stop.chat_eos, 151645);
        assert_eq!(cfg.stop.generation_eos.as_deref(), Some(&[151645u32, 151643][..]));
        assert!(cfg.chat_template.is_some());
    }

    fn qwen3_snapshot_dir() -> Option<PathBuf> {
        let dir = std::env::var("QWEN3_TOKENIZER_DIR").map(PathBuf::from).unwrap_or_else(|_| {
            PathBuf::from(
                "/mnt/data/cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/\
                 c1899de289a04d12100db370d81485cdf75e47ca",
            )
        });
        dir.join("tokenizer.json").exists().then_some(dir)
    }
}
