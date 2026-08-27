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

//! Translation of the `llama-server` b10621 model-source flags onto mlxcel's
//! MLX checkpoint resolution (issue #1434).
//!
//! b10621 selects a model through six flags, all of which end up naming a GGUF
//! artifact: `--model`, `--model-url`, `--docker-repo`, `--hf-repo`,
//! `--hf-file` and `--hf-token`, plus `--offline` gating whether any of them
//! may hit the network. mlxcel loads MLX SafeTensors checkpoints, so only some
//! of that vocabulary can select the same thing.
//!
//! This module is the whole decision. It runs before any weight is read, so a
//! command line that cannot possibly work fails at startup with a diagnostic
//! naming the flag, the reason, and the mlxcel replacement, rather than being
//! accepted and silently resolving to something else:
//!
//! | b10621 flag | mlxcel |
//! |---|---|
//! | `--model` | Same role, different value domain: an MLX checkpoint directory or a HuggingFace repo id. A GGUF path is rejected by [`crate::downloader::ensure_mlx_model_reference`]. |
//! | `--hf-repo` | Translated to the repo id `-m` accepts. A `:quant` suffix names a GGUF quantization and is rejected. |
//! | `--hf-token` | Forwarded to the mlxcel downloader as the explicit token. |
//! | `--offline` | Forwarded to the resolver, which then refuses to download. |
//! | `--hf-file` | Rejected: selects one GGUF file inside a repo; MLX loads the whole snapshot. |
//! | `--model-url` | Rejected: mlxcel downloads by repo id, not by URL. A HuggingFace URL names the repo id to use instead. |
//! | `--docker-repo` | Rejected: Docker Hub model repositories ship GGUF. |
//!
//! # Precedence
//!
//! b10621 resolves `--hf-repo` into a cache path and assigns it to the model
//! path after `-m` has been applied, so the repo wins when both are given.
//! [`resolve_llama_model_source`] reproduces that, and reports the superseded
//! `-m` value so the caller can log it instead of leaving the operator to
//! wonder which of the two was used.
//!
//! The compatibility boundary these rules implement is recorded in
//! `compat/llama-server/b10621/model-source.json`.

use std::path::{Path, PathBuf};

use anyhow::{Result, anyhow};

use crate::downloader::huggingface_repo_from_url;

/// The b10621 model-source flags, as parsed from either server binary.
#[derive(Debug, Default, Clone)]
pub struct LlamaModelSourceArgs {
    /// `-m` / `--model` / `LLAMA_ARG_MODEL`.
    pub model: Option<PathBuf>,
    /// `--hf-repo` / `LLAMA_ARG_HF_REPO`.
    pub hf_repo: Option<String>,
    /// `--hf-file` / `LLAMA_ARG_HF_FILE`.
    pub hf_file: Option<String>,
    /// `--model-url` / `LLAMA_ARG_MODEL_URL`.
    pub model_url: Option<String>,
    /// `--docker-repo` / `LLAMA_ARG_DOCKER_REPO`.
    pub docker_repo: Option<String>,
}

/// The model reference to resolve, plus what the choice superseded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedModelSource {
    /// The value to hand to [`crate::downloader::resolve_model_source_with_options`].
    pub reference: PathBuf,
    /// Name of the flag `reference` came from, for logging.
    pub origin: &'static str,
    /// A `-m` value that `--hf-repo` took precedence over, if any.
    pub superseded_model: Option<PathBuf>,
}

/// Pick the model reference from the b10621 model-source flags.
///
/// # Errors
///
/// Returns an actionable error when a flag names an artifact mlxcel cannot
/// load (`--docker-repo`, `--model-url`, `--hf-file`, a `:quant`-qualified
/// `--hf-repo`), or when no source flag was given at all.
pub fn resolve_llama_model_source(args: &LlamaModelSourceArgs) -> Result<ResolvedModelSource> {
    // Rejections come first and in a fixed order, so a command line carrying
    // several unsupported flags always reports the same one. Ordered by how
    // specific the diagnostic is: the more the flag pins mlxcel down to GGUF,
    // the earlier it is reported.
    if let Some(repo) = non_empty(args.docker_repo.as_deref()) {
        return Err(docker_repo_error(repo));
    }
    if let Some(url) = non_empty(args.model_url.as_deref()) {
        return Err(model_url_error(url));
    }
    if let Some(file) = non_empty(args.hf_file.as_deref()) {
        return Err(hf_file_error(file));
    }

    if let Some(repo) = non_empty(args.hf_repo.as_deref()) {
        let repo_id = parse_hf_repo(repo)?;
        return Ok(ResolvedModelSource {
            reference: PathBuf::from(repo_id),
            origin: "--hf-repo",
            // b10621 assigns the resolved HF path to `params.model.path` after
            // `-m` has been applied, so the repo wins. Reported rather than
            // dropped silently.
            superseded_model: args.model.clone(),
        });
    }

    match args.model.clone() {
        Some(model) if !model.as_os_str().is_empty() => Ok(ResolvedModelSource {
            reference: model,
            origin: "--model",
            superseded_model: None,
        }),
        _ => Err(anyhow!(
            "--model/-m is required to start the server (set the LLAMA_ARG_MODEL \
             environment variable, pass -m <PATH_OR_REPO_ID>, or pass --hf-repo \
             <owner>/<name>)"
        )),
    }
}

/// Validate a `--hf-repo` value and return the `owner/name` repo id.
///
/// b10621 accepts `<user>/<model>[:quant]`, where `quant` names a GGUF
/// quantization (`Q4_K_M`, `IQ4_NL`, …) and selects which file inside the
/// repository to download. MLX checkpoints are whole snapshots and carry their
/// quantization in the repository name itself
/// (`mlx-community/Qwen3-4B-4bit`), so there is nothing for a `:quant`
/// suffix to select and honoring it would be guesswork.
///
/// # Errors
///
/// Returns an error when the value carries a `:quant` suffix, or when it is
/// not a plain `owner/name` pair.
pub fn parse_hf_repo(value: &str) -> Result<String> {
    let trimmed = value.trim();
    if let Some((repo, quant)) = trimmed.split_once(':') {
        return Err(anyhow!(
            "--hf-repo '{trimmed}' selects the GGUF quantization '{quant}'. \
             mlxcel loads MLX SafeTensors snapshots, whose quantization is part \
             of the repository name, so there is no file inside '{repo}' for \
             ':{quant}' to select.\n\
             Name the MLX repository directly, for example:\n  \
             --hf-repo mlx-community/Qwen3-4B-4bit"
        ));
    }
    let mut parts = trimmed.split('/');
    let owner = parts.next().unwrap_or_default();
    let name = parts.next().unwrap_or_default();
    if owner.is_empty() || name.is_empty() || parts.next().is_some() {
        return Err(anyhow!(
            "--hf-repo '{trimmed}' is not a HuggingFace repository identifier. \
             Expected exactly '<owner>/<name>', for example:\n  \
             --hf-repo mlx-community/Qwen3-4B-4bit"
        ));
    }
    Ok(format!("{owner}/{name}"))
}

/// Split a b10621 `--alias` value into its aliases.
///
/// b10621 takes a comma-separated list and keeps every entry as an API-visible
/// name for the model. mlxcel serves one model under one id, so the first
/// non-empty entry becomes the served id and the rest are carried for the
/// `/v1/models` `aliases` array (issue #1438). Entries are stripped of
/// surrounding whitespace and empty entries are dropped, matching
/// `string_strip` in b10621's `--alias` handler.
///
/// Returns an empty vector when the value contains no non-empty entry, which
/// the caller treats exactly like an absent `--alias`.
#[must_use]
pub fn parse_model_aliases(value: &str) -> Vec<String> {
    let mut seen: Vec<String> = Vec::new();
    for part in value.split(',') {
        let trimmed = part.trim();
        if trimmed.is_empty() || seen.iter().any(|existing| existing == trimmed) {
            continue;
        }
        seen.push(trimmed.to_owned());
    }
    seen
}

/// Environment variable bound to `--offline` by b10621.
pub const LLAMA_ARG_OFFLINE: &str = "LLAMA_ARG_OFFLINE";

/// Apply `LLAMA_ARG_OFFLINE` to an already-parsed `--offline` flag.
///
/// Deliberately not a clap `env = "LLAMA_ARG_OFFLINE"` binding. `--offline` is
/// a value-less flag, and b10621 fires a value-less option from the
/// environment only when the value is truthy by
/// `common_arg_utils::is_truthy`: exactly `on`, `enabled`, `true` or `1`,
/// compared case-sensitively
/// (<https://github.com/ggml-org/llama.cpp/blob/c1d0e7a004015f23bc0233470b747b596f29b264/common/arg.cpp>).
/// clap's own boolish env parser accepts a wider vocabulary (`y`, `yes`, `t`,
/// …) and *errors* on a value outside it, so binding the variable through clap
/// would both enable offline mode on spellings b10621 ignores and refuse to
/// start on values b10621 silently drops. Everything outside the truthy set,
/// an empty value and `0` included, leaves the flag alone.
///
/// An explicit `--offline` on the command line always wins: it can only turn
/// the flag on, and the environment can only turn it on as well, so the two
/// never disagree.
pub fn env_fallback_offline(value: &mut bool) {
    if *value {
        return;
    }
    if let Ok(raw) = std::env::var(LLAMA_ARG_OFFLINE) {
        *value = matches!(raw.as_str(), "on" | "enabled" | "true" | "1");
    }
}

/// `Some(value)` when `value` is present and not whitespace-only.
///
/// Environment-bound flags routinely arrive as an empty string from a shell
/// variable that was never set (`LLAMA_ARG_DOCKER_REPO=`), and refusing to
/// start over one would make the compatibility surface worse than ignoring
/// the flag. b10621 tests these fields with `.empty()` for the same reason.
fn non_empty(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|v| !v.is_empty())
}

fn docker_repo_error(value: &str) -> anyhow::Error {
    anyhow!(
        "--docker-repo '{value}' is not supported. Docker Hub model \
         repositories distribute GGUF artifacts for the GGML backend, and \
         mlxcel is an MLX runtime that loads SafeTensors checkpoints; there is \
         no equivalent registry to translate this to.\n\
         Name an MLX checkpoint instead, for example:\n  \
         --hf-repo mlx-community/gemma-3-4b-it-4bit\n\
         `mlxcel list` reports the architectures this binary can load."
    )
}

fn model_url_error(value: &str) -> anyhow::Error {
    match huggingface_repo_from_url(value) {
        Some(repo) => anyhow!(
            "--model-url '{value}' is not supported. mlxcel resolves models by \
             HuggingFace repository identifier through the mlxcel model store \
             and the HuggingFace cache, not by URL.\n\
             That URL names a repository, so pass it directly:\n  \
             --hf-repo {repo}\n\
             The repository must be an MLX SafeTensors build; a GGUF \
             repository cannot be loaded."
        ),
        None => anyhow!(
            "--model-url '{value}' is not supported. It names a file to \
             download for the GGML backend, and mlxcel is an MLX runtime that \
             loads SafeTensors checkpoints resolved by HuggingFace repository \
             identifier.\n\
             Name an MLX checkpoint instead, for example:\n  \
             --hf-repo mlx-community/Qwen3-4B-4bit"
        ),
    }
}

fn hf_file_error(value: &str) -> anyhow::Error {
    anyhow!(
        "--hf-file '{value}' is not supported. It selects one GGUF file inside \
         a repository, and mlxcel loads an MLX SafeTensors snapshot as a whole \
         (its shards are described by `model.safetensors.index.json`), so \
         there is no single file to select.\n\
         Name the MLX repository instead, for example:\n  \
         --hf-repo mlx-community/Qwen3-4B-4bit"
    )
}

/// Log line describing a `--hf-repo` that superseded a `-m` value.
///
/// Separate from [`resolve_llama_model_source`] so the decision stays pure and
/// testable while both binaries emit the same wording.
#[must_use]
pub fn superseded_model_notice(reference: &Path, superseded: &Path) -> String {
    format!(
        "--hf-repo {} takes precedence over --model {}; serving the repository \
         (matches llama-server, which assigns the resolved repository path over \
         --model)",
        reference.display(),
        superseded.display()
    )
}

#[cfg(test)]
#[path = "model_source_tests.rs"]
mod tests;
