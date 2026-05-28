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

//! File-level allow-list for the HuggingFace snapshot downloader.
//!
//! The decision tree intentionally errs on the side of fetching anything that
//! mlxcel might plausibly load (configs, tokenizers, safetensors shards) and
//! skipping anything we know we cannot consume (`*.bin` PyTorch weights,
//! `*.gguf` quantizations, training artifacts). This way new model families
//! work without code changes; only the deny-list needs touching when a new
//! useless artifact convention shows up upstream.

/// Compute the basename of a HuggingFace `<owner>/<name>` repo id.
///
/// Returns the slice after the last `/`. For ids without a `/` the input is
/// returned unchanged.
pub fn repo_basename(repo_id: &str) -> &str {
    repo_id.rsplit('/').next().unwrap_or(repo_id)
}

/// True when `path` (a sibling `rfilename` from the HuggingFace API) should be
/// fetched as part of an mlxcel model snapshot.
///
/// The check is case-insensitive on the file extension and base name suffixes
/// to match the mixed conventions used across the Hub.
///
/// **Security:** A path-safety check runs first. Any `rfilename` that is
/// absolute, contains parent traversal (`..`), uses backslash separators, or
/// has empty/`.` components is rejected outright. This prevents a malicious
/// repository (or MitM attacker) from coercing the downloader into writing
/// outside the user's `--local-dir`. See `download_repo` for the
/// canonicalized destination guard that backs this filter as defense in
/// depth.
pub fn is_wanted_file(path: &str) -> bool {
    if !is_safe_relative_path(path) {
        return false;
    }

    let lower = path.to_ascii_lowercase();
    let base = lower.rsplit('/').next().unwrap_or(lower.as_str());

    // Hard deny-list. Match before the allow-list so that, e.g., a
    // `.bin.index.json` does not slip in via the broad `*.json` allow.
    if is_explicitly_denied(base) {
        return false;
    }

    // Hidden files / VCS metadata.
    if base.starts_with('.') {
        return false;
    }

    // Known-good extensions and exact filenames.
    if has_extension(base, "safetensors") || has_extension(base, "json") {
        // *.bin.index.json was filtered above.
        return true;
    }
    if has_extension(base, "tiktoken") {
        return true;
    }
    // HuggingFace ships chat templates as chat_template.jinja, not inlined JSON.
    if has_extension(base, "jinja") {
        return true;
    }
    if has_extension(base, "model") {
        // sentencepiece tokenizer.model and friends.
        return true;
    }
    if has_extension(base, "txt") {
        // merges.txt, vocab*.txt
        return base.starts_with("vocab") || base == "merges.txt" || base.contains("token");
    }

    // Exact-name allow-list for files without a recognizable extension.
    matches!(
        base,
        "vocab"
            | "merges"
            | "added_tokens"
            | "special_tokens_map"
            | "tokenizer_config"
            | "tokenizer"
            | "generation_config"
            | "preprocessor_config"
            | "processor_config"
            | "chat_template"
    )
}

/// True when `path` is a safe **relative** path that stays inside the
/// destination directory.
///
/// Rejects:
/// - empty strings (would later resolve to the destination root itself),
/// - absolute paths (`/etc/passwd`, `C:\Windows`...),
/// - paths containing backslashes (some platforms treat `\` as a separator),
/// - any component that is not [`std::path::Component::Normal`] — that
///   excludes `..` (parent traversal), `.` (current dir), `//` (empty
///   segments), and any platform-specific prefix or root component.
///
/// This is the primary line of defense against malicious `rfilename` values
/// served by a compromised HuggingFace API response. The downloader also
/// applies a canonicalized `starts_with` guard at write time as defense in
/// depth.
fn is_safe_relative_path(path: &str) -> bool {
    if path.is_empty() {
        return false;
    }
    let p = std::path::Path::new(path);
    if p.is_absolute() {
        return false;
    }
    // Reject backslash explicitly — some filesystems treat it as a separator.
    if path.contains('\\') {
        return false;
    }
    // `Path::components()` collapses consecutive slashes (`foo//bar` → two
    // `Normal` components) and trailing slashes, so an explicit string-level
    // check is needed for those shapes too. We require: no doubled `/`, no
    // leading `/` (already caught by `is_absolute` on Unix but cheap to
    // assert), and no trailing `/`.
    if path.contains("//") || path.starts_with('/') || path.ends_with('/') {
        return false;
    }
    use std::path::Component;
    for c in p.components() {
        match c {
            Component::Normal(s) if !s.is_empty() => {}
            _ => return false,
        }
    }
    true
}

fn is_explicitly_denied(base: &str) -> bool {
    // `*.bin.index.json` and similar PyTorch-only index files.
    if base.ends_with(".bin.index.json") || base.ends_with(".pt.index.json") {
        return true;
    }
    // Doc / repo metadata and license files we do not need at runtime.
    if base.starts_with("readme")
        || base.starts_with("license")
        || base.starts_with("notice")
        || base.starts_with("usage")
        || base == "citation.cff"
        || base.starts_with(".git")
        || base.starts_with(".huggingface")
    {
        return true;
    }
    // Image / media assets shipped in some repos for marketing.
    if has_extension(base, "png")
        || has_extension(base, "jpg")
        || has_extension(base, "jpeg")
        || has_extension(base, "gif")
        || has_extension(base, "webp")
        || has_extension(base, "svg")
        || has_extension(base, "mp4")
        || has_extension(base, "mp3")
        || has_extension(base, "wav")
    {
        return true;
    }
    // Other ML formats mlxcel cannot load.
    if has_extension(base, "bin")
        || has_extension(base, "pt")
        || has_extension(base, "pth")
        || has_extension(base, "ckpt")
        || has_extension(base, "h5")
        || has_extension(base, "msgpack")
        || has_extension(base, "ot")
        || has_extension(base, "onnx")
        || has_extension(base, "tflite")
        || has_extension(base, "gguf")
        || has_extension(base, "ggml")
    {
        return true;
    }
    // Source / build artifacts occasionally checked in.
    if has_extension(base, "py")
        || has_extension(base, "ipynb")
        || has_extension(base, "md")
        || has_extension(base, "rst")
        || has_extension(base, "yaml")
        || has_extension(base, "yml")
        || has_extension(base, "toml")
        || has_extension(base, "lock")
        || has_extension(base, "log")
    {
        return true;
    }
    false
}

fn has_extension(name: &str, ext: &str) -> bool {
    let needle = format!(".{}", ext.trim_start_matches('.').to_ascii_lowercase());
    name.ends_with(&needle)
}
