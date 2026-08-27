// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Early rejection of model references mlxcel structurally cannot load
//! (issue #1434).
//!
//! `llama-server` loads GGUF files: `-m model.gguf`, `-m shard-00001-of-00003.gguf`,
//! `--model-url https://…/model.gguf`, `--docker-repo ai/gemma3`. mlxcel is an
//! MLX runtime and loads a SafeTensors checkpoint *directory* (or a HuggingFace
//! repo id naming one). There is no GGML backend behind it and no GGUF reader,
//! so none of those references can ever resolve.
//!
//! Accepting them and failing later is the failure mode this module exists to
//! remove. A `.gguf` path reaches [`super::resolver::resolve_model_source`]'s
//! step 1 (`value.exists()`), is returned verbatim as "a local path", and dies
//! several seconds later inside the loader with a message about a missing
//! `config.json` that says nothing about the format. A `https://…` value is
//! neither an existing path nor a repo-id shape, so it dies in step 4 with the
//! generic "not a model name" error and no mention of what to pass instead.
//!
//! [`ensure_mlx_model_reference`] runs *before* every resolution step, so the
//! diagnostic names the requested value, the reason (mlxcel has no GGUF/GGML
//! backend), and a concrete MLX replacement for that exact reference. It is
//! wired into the shared resolver, so `mlxcel generate`, `mlxcel chat`,
//! `mlxcel serve` and `mlxcel-server` all reject identically.
//!
//! The b10621 side of the boundary is recorded in
//! `compat/llama-server/b10621/model-source.json`.

use std::borrow::Cow;
use std::path::Path;

use anyhow::{Result, anyhow};

/// A model reference shape that mlxcel structurally cannot load.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnsupportedModelReference {
    /// A single GGUF file, for example `models/gemma-3-4b-it-Q4_K_M.gguf`.
    Gguf,
    /// One shard of a split GGUF, for example `…-00001-of-00003.gguf`.
    GgufSplit,
    /// A HuggingFace web URL (`https://huggingface.co/<owner>/<name>[/…]`).
    /// Carries the repo id mlxcel would accept instead.
    HuggingFaceUrl,
    /// Any other URL, including `http(s)://`, `hf://`, `docker://`, `oci://`.
    Url,
}

/// Reject `value` when it names a model artifact mlxcel cannot load.
///
/// Returns `Ok(())` for every reference that could plausibly resolve to an MLX
/// SafeTensors checkpoint: local paths, `owner/name` repo ids, and bare model
/// names. The check is purely syntactic and touches no filesystem or network,
/// so it is safe to run before any resolution step and cheap enough to run on
/// every invocation.
///
/// # Errors
///
/// Returns an actionable error naming the requested value, why mlxcel cannot
/// load it, and a concrete replacement command for that exact reference shape.
pub fn ensure_mlx_model_reference(value: &Path) -> Result<()> {
    // A non-UTF-8 value can only be a local path; it cannot carry a URL scheme
    // or a `.gguf` suffix we would recognize, so let the resolver's own
    // path handling report it.
    let Some(text) = value.to_str() else {
        return Ok(());
    };
    // A GGUF artifact is a file, never a directory. Nothing else in this gate
    // touches the filesystem, but skipping this one probe would reject
    // `-m models/mymodel.gguf` when that is a real MLX checkpoint *directory*
    // someone happened to name that way, which used to load.
    if value.is_dir() {
        return Ok(());
    }
    match classify_model_reference(text) {
        None => Ok(()),
        Some(kind) => Err(unsupported_reference_error(text, kind)),
    }
}

/// Classify `value` as an unsupported reference shape, or `None` when mlxcel
/// may attempt to resolve it.
///
/// [`ensure_mlx_model_reference`] is the gate; this is the classification
/// behind it, exposed so a caller that wants to branch on the shape (rather
/// than take the gate's diagnostic) does not re-derive it. Purely syntactic:
/// unlike the gate it never touches the filesystem, so a directory named
/// `foo.gguf` classifies as [`UnsupportedModelReference::Gguf`] here.
#[must_use]
pub fn classify_model_reference(value: &str) -> Option<UnsupportedModelReference> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return None;
    }
    // A Windows drive letter (`C:\models\…`) is not a URL: `url_scheme_end`
    // requires `://`, which a drive path never has, so a match here is a real
    // scheme.
    if url_scheme_end(trimmed).is_some() {
        if huggingface_repo_from_url(trimmed).is_some() {
            return Some(UnsupportedModelReference::HuggingFaceUrl);
        }
        return Some(UnsupportedModelReference::Url);
    }
    if has_gguf_extension(trimmed) {
        if is_split_shard_name(trimmed) {
            return Some(UnsupportedModelReference::GgufSplit);
        }
        return Some(UnsupportedModelReference::Gguf);
    }
    None
}

/// Byte offset of the `:` ending a URL scheme in `value`, when `value` starts
/// with `<scheme>://` and the scheme matches RFC 3986's production
/// (`ALPHA *( ALPHA / DIGIT / "+" / "-" / "." )`).
fn url_scheme_end(value: &str) -> Option<usize> {
    let bytes = value.as_bytes();
    let first = *bytes.first()?;
    if !first.is_ascii_alphabetic() {
        return None;
    }
    let mut i = 1;
    while i < bytes.len() {
        let b = bytes[i];
        if b.is_ascii_alphanumeric() || b == b'+' || b == b'-' || b == b'.' {
            i += 1;
            continue;
        }
        break;
    }
    if value[i..].starts_with("://") {
        Some(i)
    } else {
        None
    }
}

/// The `<owner>/<name>` repo id inside a HuggingFace web URL, if `value` is
/// one. Accepts `https://huggingface.co/<owner>/<name>` with or without a
/// trailing path (`/resolve/main/model.gguf`, `/tree/main`, a query, …), and
/// the `hf://<owner>/<name>` shorthand.
#[must_use]
pub fn huggingface_repo_from_url(value: &str) -> Option<String> {
    let trimmed = value.trim();
    let scheme_end = url_scheme_end(trimmed)?;
    let scheme = trimmed[..scheme_end].to_ascii_lowercase();
    let rest = &trimmed[scheme_end + "://".len()..];
    let rest = rest.split(['?', '#']).next().unwrap_or(rest);

    let path = match scheme.as_str() {
        "hf" | "huggingface" => rest,
        "http" | "https" => {
            let (host, path) = rest.split_once('/')?;
            let host = host.rsplit('@').next().unwrap_or(host);
            let host = host.split(':').next().unwrap_or(host);
            if !matches!(
                host.to_ascii_lowercase().as_str(),
                "huggingface.co" | "www.huggingface.co" | "hf.co"
            ) {
                return None;
            }
            path
        }
        _ => return None,
    };

    // Model pages may be prefixed with a repo type (`/models/<owner>/<name>`).
    let path = path.strip_prefix("models/").unwrap_or(path);
    let mut segments = path.split('/').filter(|s| !s.is_empty());
    let owner = segments.next()?;
    let name = segments.next()?;
    if !is_repo_url_segment(owner) || !is_repo_url_segment(name) {
        return None;
    }
    Some(format!("{owner}/{name}"))
}

/// True for a HuggingFace path segment: non-empty ASCII alphanumerics plus
/// `.`, `_`, `-`. Mirrors `resolver::is_repo_segment` so a URL only ever
/// suggests a repo id the resolver would actually accept.
fn is_repo_url_segment(segment: &str) -> bool {
    !segment.is_empty()
        && segment
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'.' || b == b'_' || b == b'-')
}

/// True when the final path component ends in `.gguf`, case-insensitively.
fn has_gguf_extension(value: &str) -> bool {
    file_name(value).is_some_and(|name| {
        name.len() > ".gguf".len() && name.to_ascii_lowercase().ends_with(".gguf")
    })
}

/// True when the final path component is one shard of a split GGUF, that is
/// `<base>-<NNNNN>-of-<NNNNN>.gguf` (llama.cpp's `llama_split_path` format).
fn is_split_shard_name(value: &str) -> bool {
    let Some(name) = file_name(value) else {
        return false;
    };
    let lowered = name.to_ascii_lowercase();
    let Some(stem) = lowered.strip_suffix(".gguf") else {
        return false;
    };
    let Some((head, total)) = stem.rsplit_once("-of-") else {
        return false;
    };
    let Some((_, index)) = head.rsplit_once('-') else {
        return false;
    };
    !index.is_empty()
        && !total.is_empty()
        && index.bytes().all(|b| b.is_ascii_digit())
        && total.bytes().all(|b| b.is_ascii_digit())
}

/// Final path component of `value`, handling both `/` and `\` separators so a
/// Windows-style GGUF path is classified the same way.
fn file_name(value: &str) -> Option<&str> {
    let name = value.rsplit(['/', '\\']).next()?;
    if name.is_empty() { None } else { Some(name) }
}

/// Replace a URL's `userinfo` component with `***` so a diagnostic never
/// echoes an embedded credential.
///
/// `LLAMA_ARG_MODEL_URL=https://user:token@host/model.gguf` inherited from a
/// llama.cpp deployment would otherwise print the token to stderr, and into
/// whatever captures it, at startup. Non-URL values are returned unchanged.
#[must_use]
pub fn redact_url_userinfo(value: &str) -> Cow<'_, str> {
    let Some(scheme_end) = url_scheme_end(value) else {
        return Cow::Borrowed(value);
    };
    let authority_start = scheme_end + "://".len();
    let rest = &value[authority_start..];
    // The authority ends at the first `/`, `?` or `#`; userinfo is what sits
    // before the last `@` inside it.
    let authority_end = rest
        .find(['/', '?', '#'])
        .map_or(rest.len(), |i| i + authority_start);
    let authority = &value[authority_start..authority_end];
    match authority.rfind('@') {
        None => Cow::Borrowed(value),
        Some(at) => Cow::Owned(format!(
            "{}***@{}",
            &value[..authority_start],
            &value[authority_start + at + 1..]
        )),
    }
}

/// Build the actionable diagnostic for an unsupported reference.
fn unsupported_reference_error(value: &str, kind: UnsupportedModelReference) -> anyhow::Error {
    let example = mlx_replacement_example(value, kind);
    let value = redact_url_userinfo(value);
    match kind {
        UnsupportedModelReference::Gguf => anyhow!(
            "'{value}' is a GGUF file. mlxcel is an MLX runtime with no GGML \
             backend and no GGUF reader, so it cannot load one; it loads an MLX \
             SafeTensors checkpoint directory or the HuggingFace repo id of one.\n\
             Use an MLX build of the same model, for example:\n  {example}\n\
             `mlxcel list` reports the architectures this binary can load."
        ),
        UnsupportedModelReference::GgufSplit => anyhow!(
            "'{value}' is one shard of a split GGUF file. mlxcel is an MLX \
             runtime with no GGML backend and no GGUF reader, and it does not \
             reassemble GGUF splits; it loads an MLX SafeTensors checkpoint \
             directory (whose shards are described by \
             `model.safetensors.index.json`) or the HuggingFace repo id of one.\n\
             Use an MLX build of the same model, for example:\n  {example}\n\
             `mlxcel list` reports the architectures this binary can load."
        ),
        UnsupportedModelReference::HuggingFaceUrl => anyhow!(
            "'{value}' is a HuggingFace web URL. mlxcel does not download from \
             arbitrary URLs; it takes the repo id itself and resolves it through \
             the mlxcel model store and the HuggingFace cache.\n\
             Pass the repo id instead, for example:\n  {example}\n\
             The repository must be an MLX SafeTensors build; GGUF repositories \
             cannot be loaded."
        ),
        UnsupportedModelReference::Url => anyhow!(
            "'{value}' is a URL. mlxcel does not download models from arbitrary \
             URLs, Docker registries, or OCI registries; it loads an MLX \
             SafeTensors checkpoint directory or a HuggingFace repo id.\n\
             Use one of those instead, for example:\n  {example}"
        ),
    }
}

/// A concrete replacement command for the requested reference.
///
/// For a HuggingFace URL this is the exact repo id the user meant. For every
/// other shape mlxcel cannot know the MLX equivalent of a GGUF artifact, so
/// the example names the `mlx-community` convention and keeps the model's own
/// basename where one can be recovered.
fn mlx_replacement_example(value: &str, kind: UnsupportedModelReference) -> String {
    if kind == UnsupportedModelReference::HuggingFaceUrl
        && let Some(repo) = huggingface_repo_from_url(value)
    {
        return format!("-m {repo}");
    }
    match gguf_model_basename(value) {
        Some(base) => format!("-m mlx-community/{base}-4bit"),
        None => "-m mlx-community/Qwen3-4B-4bit".to_owned(),
    }
}

/// Best-effort model name recovered from a GGUF file name: the basename with
/// the `.gguf` suffix, any split suffix, and any trailing GGML quant tag
/// removed. Returns `None` when nothing usable is left.
fn gguf_model_basename(value: &str) -> Option<String> {
    let name = file_name(value)?;
    let stem = name.strip_suffix(".gguf").or_else(|| {
        let lowered = name.to_ascii_lowercase();
        lowered
            .ends_with(".gguf")
            .then(|| &name[..name.len() - ".gguf".len()])
    })?;

    // Drop a `-00001-of-00003` split suffix.
    let stem = match stem.rsplit_once("-of-") {
        Some((head, total)) if total.bytes().all(|b| b.is_ascii_digit()) => {
            match head.rsplit_once('-') {
                Some((base, index)) if index.bytes().all(|b| b.is_ascii_digit()) => base,
                _ => stem,
            }
        }
        _ => stem,
    };

    // Drop a trailing GGML quant tag (`-Q4_K_M`, `-IQ4_NL`, `-f16`, `-BF16`).
    let stem = match stem.rsplit_once('-') {
        Some((base, tag)) if !base.is_empty() && is_ggml_quant_tag(tag) => base,
        _ => stem,
    };
    // Repositories are conventionally named `<Model>-GGUF`.
    let stem = stem.strip_suffix("-GGUF").unwrap_or(stem);
    let stem = stem.strip_suffix("-gguf").unwrap_or(stem);

    (!stem.is_empty()).then(|| stem.to_owned())
}

/// True for a GGML quantization tag such as `Q4_K_M`, `IQ4_NL`, `f16`, `bf16`.
fn is_ggml_quant_tag(tag: &str) -> bool {
    let lowered = tag.to_ascii_lowercase();
    if matches!(lowered.as_str(), "f16" | "f32" | "bf16") {
        return true;
    }
    let rest = lowered
        .strip_prefix("iq")
        .or_else(|| lowered.strip_prefix('q'));
    rest.is_some_and(|rest| rest.starts_with(|c: char| c.is_ascii_digit()))
}

#[cfg(test)]
#[path = "model_format_tests.rs"]
mod tests;
