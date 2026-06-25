// rust/npu-weights/src/spec.rs
use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq)]
pub enum Source {
    Hf { repo: String, rev: Option<String> },
    Path(PathBuf),
}

#[derive(Debug, Clone)]
pub struct ModelSpec {
    pub source: Source,
    pub arch: String,
    pub arena: Option<PathBuf>,
}

impl Source {
    /// Parse "hf:<repo>[@rev]" or "path:/abs".
    pub fn parse(s: &str) -> anyhow::Result<Source> {
        if let Some(rest) = s.strip_prefix("hf:") {
            let (repo, rev) = match rest.split_once('@') {
                Some((r, v)) => (r.to_string(), Some(v.to_string())),
                None => (rest.to_string(), None),
            };
            anyhow::ensure!(!repo.is_empty(), "empty hf repo in {s:?}");
            Ok(Source::Hf { repo, rev })
        } else if let Some(p) = s.strip_prefix("path:") {
            anyhow::ensure!(!p.is_empty(), "empty path in {s:?}");
            Ok(Source::Path(PathBuf::from(p)))
        } else {
            anyhow::bail!("source must start with 'hf:' or 'path:': {s:?}")
        }
    }
    /// Filesystem-safe token: '/',':','@' -> '_'.
    pub fn sanitized(&self) -> String {
        let raw = match self {
            Source::Hf { repo, rev } => match rev {
                Some(v) => format!("hf:{repo}@{v}"),
                None => format!("hf:{repo}"),
            },
            Source::Path(p) => format!("path:{}", p.display()),
        };
        raw.chars().map(|c| if matches!(c, '/' | ':' | '@') { '_' } else { c }).collect()
    }
}

impl ModelSpec {
    /// Derived arena path when `arena` is None:
    /// ${XDNA_ARENA_DIR:-<root>/artifacts/arenas}/<arch>__<sanitized>__<disc>.safetensors
    pub fn arena_path(&self, root: &std::path::Path, disc: &str) -> PathBuf {
        if let Some(a) = &self.arena {
            return a.clone();
        }
        let base = std::env::var("XDNA_ARENA_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| root.join("artifacts/arenas"));
        base.join(format!("{}__{}__{}.safetensors", self.arch, self.source.sanitized(), disc))
    }

    /// Resolve this spec to a usable arena on disk: resolve source files, fingerprint them, pick the
    /// arena path, and BAKE if it is missing/stale (unless a fresh one is already present and
    /// `force` is false). Returns the path to a verified-loadable arena.
    ///
    /// This is the single uniform entry point a consumer (engine) calls to turn a declarative
    /// `{source, arch, arena?}` spec into a baked arena, with no shelling out to the `npu-weights`
    /// binary. It is host-only (no NPU): it reads source weights, runs the arch transform, and
    /// writes a `.safetensors` arena atomically.
    pub fn ensure_arena(&self, root: &std::path::Path, force: bool) -> anyhow::Result<PathBuf> {
        let files = crate::source::resolve_files(&self.source)?;
        let fref: Vec<(String, &std::path::Path)> = files
            .iter()
            .map(|p| (p.file_name().unwrap().to_string_lossy().into_owned(), p.as_path()))
            .collect();
        let fp = crate::fingerprint::multi_sha256(&fref)?;
        let path = self.arena_path(root, &fp[..12]);
        if path.exists() && !force {
            if crate::arena::load(&path, &self.arch).is_ok() {
                return Ok(path);
            }
        }
        let bag = crate::source::read_weights(&files)?;
        let out = crate::arch::get(&self.arch)?.transform(&bag)?;
        let meta = crate::arena::ArenaMeta {
            format_version: crate::FORMAT_VERSION,
            arch: self.arch.clone(),
            source_ref: self.source.sanitized(),
            source_fingerprint: fp,
        };
        crate::arena::bake(&path, &out, &meta)?;
        Ok(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parse_hf_with_rev() {
        assert_eq!(Source::parse("hf:BAAI/bge-base-en-v1.5@abc123").unwrap(),
                   Source::Hf { repo: "BAAI/bge-base-en-v1.5".into(), rev: Some("abc123".into()) });
    }
    #[test]
    fn parse_hf_no_rev() {
        assert_eq!(Source::parse("hf:facebook/opt-125m").unwrap(),
                   Source::Hf { repo: "facebook/opt-125m".into(), rev: None });
    }
    #[test]
    fn parse_path() {
        assert_eq!(Source::parse("path:/models/m.onnx").unwrap(),
                   Source::Path("/models/m.onnx".into()));
    }
    #[test]
    fn derived_path_includes_arch_and_disc() {
        let spec = ModelSpec { source: Source::parse("hf:BAAI/bge-base-en-v1.5@abc").unwrap(),
                               arch: "bert".into(), arena: None };
        let p = spec.arena_path(std::path::Path::new("/repo"), "abc123def456");
        assert_eq!(p, PathBuf::from(
            "/repo/artifacts/arenas/bert__hf_BAAI_bge-base-en-v1.5_abc__abc123def456.safetensors"));
    }
}
