//! Model-directory config: tokenizer, chat template, and stop-token authorities, read as DATA from
//! the checkpoint's own files (`tokenizer.json`, `tokenizer_config.json`, `generation_config.json`)
//! rather than hardcoded per model -- the rail must take a second model without a rewrite.

use std::path::Path;

use tokenizers::Tokenizer;

use crate::api::EngineError;
use crate::llm::chat_template::ChatTemplate;

/// Stop-token ids, each resolved from its OWN authority rather than picked. Qwen-family checkpoints
/// carry two EOS-role tokens that are genuinely different: `<|im_end|>` ends a chat turn (the
/// tokenizer's own `eos_token`) while `<|endoftext|>`, the base-LM's original EOS, shows up as an
/// EXTRA member of `generation_config.json`'s `eos_token_id` list. They are tracked separately
/// rather than merged into one guess -- see
/// `docs/kb/an-id-indexed-array-api-degrades-silently-on-a-subset.md` and
/// `docs/kb/the-two-s2-authorities-disagree-on-the-generation-stop-token.md` for the failure this
/// avoids: two authorities computing different answers for "the" stop token, with the difference
/// noticed only after a mismatched decode length.
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
}

impl ModelConfig {
    /// Direct construction, e.g. from a tokenizer already loaded elsewhere, or from a test fixture.
    pub fn new(tokenizer: Tokenizer, chat_template: Option<ChatTemplate>, stop: StopTokens) -> Self {
        ModelConfig { tokenizer, chat_template, stop }
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

        Ok(ModelConfig { tokenizer, chat_template, stop })
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
    /// EOS (the S2 shape -- `docs/kb/the-two-s2-authorities-disagree-on-the-generation-stop-token.md`).
    /// Must fail loud, naming both values, never pick one.
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
