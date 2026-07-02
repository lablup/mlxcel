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

//! Repo-id-aware `-m/--model` resolver (epic #92, issue #94).
//!
//! Lets every subcommand that takes `-m/--model` (`generate`, `serve`,
//! `inspect`) accept either a local path *or* a HuggingFace repo-id, auto-
//! downloading the snapshot on a cache miss. The model is then runnable from
//! any directory — the mlx-lm / ollama / LM Studio convenience UX.
//!
//! # Resolution order (locked design, epic #92, extended by issue #112)
//!
//! [`resolve_model_source`] applies exactly this precedence:
//!
//! 1. **Existing on-disk path** — if the `-m` value already names an existing
//!    path (file or directory), it is used verbatim. This is byte-identical to
//!    the pre-#94 behavior, so every existing `-m models/foo` / `-m
//!    /abs/path` invocation keeps working with no observable difference, even
//!    when the path happens to look like an `owner/name` repo-id.
//! 2. **`owner/name` repo-id shape** — when the value is *not* an existing path
//!    but matches `^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$` (exactly one slash), it is
//!    treated as a HuggingFace repo-id and resolved in this sub-order: (a)
//!    legacy per-CWD `./models/<basename>` snapshot if complete (bridges the
//!    pre-#93 default download location); (b) an existing HuggingFace Hub cache
//!    snapshot ([`store::hf_cache_snapshot`], read-only reuse); (c) the mlxcel
//!    global store ([`store::model_dir`]) if complete; (d) on a miss, download
//!    the snapshot into the mlxcel global store via the shared hardened
//!    downloader ([`download_repo`]) and use it. Steps 1 and 2 take precedence
//!    over step 3, so an explicit `owner/name` and an existing local path are
//!    always resolved exactly as provided.
//! 3. **Bare single segment (issue #112)** — a single valid segment with no `/`
//!    (passes `is_repo_segment`, fails `is_repo_id_shape`) is resolved as
//!    `<DEFAULT_ORG>/<segment>`, where `DEFAULT_ORG` is read from
//!    `$MLXCEL_DEFAULT_ORG` (default `mlx-community`). This covers the common
//!    case where `mlx-community` is the source of MLX-format checkpoints, so
//!    `mlxcel run Qwen3-4B-4bit` resolves to
//!    `mlx-community/Qwen3-4B-4bit` without requiring the user to type the full
//!    `owner/name`. The resolved repo-id is printed before download/load so the
//!    expansion is never a silent surprise.
//! 4. **Neither** — a clear, actionable error (not an existing path, not a
//!    valid `owner/name` repo-id, and not a bare single segment).
//!
//! The "completeness" gate for the legacy and store directories verifies the
//! full weight set, not just `config.json` (issue #465): every shard named by a
//! local `model.safetensors.index.json` (or, for a single-file / repackaged
//! layout, at least one non-zero `*.safetensors`) must be present and non-zero.
//! An interrupted download that fetched `config.json` and only some shards is
//! therefore treated as a miss, and the resolver re-fetches it — resuming the
//! partial snapshot through the shared downloader — instead of handing the
//! loader a path that dies with `Weight not found`. See [`super::completeness`]
//! for the classifier. The read-only HuggingFace cache reuse
//! ([`store::hf_cache_snapshot`]) keeps its own `config.json` gate since mlxcel
//! never writes into that externally-managed layout.

use std::path::{Path, PathBuf};

use anyhow::{Result, anyhow};

use super::completeness::{SnapshotState, classify_snapshot};
use super::filters::repo_basename;
use super::store;
use super::{DownloadOptions, download_repo};

/// Legacy per-CWD download root used by mlxcel before the global store
/// (epic #92, issue #93). A repo-id whose basename already lives under
/// `./models/<basename>` is reused from there for back-compat.
const LEGACY_MODELS_DIR: &str = "models";

/// Default HuggingFace org prepended to a bare, prefix-less model name
/// (issue #112) when `MLXCEL_DEFAULT_ORG` is unset or empty.
const DEFAULT_ORG: &str = "mlx-community";

/// Environment variable overriding [`DEFAULT_ORG`] for bare-name resolution.
const DEFAULT_ORG_ENV: &str = "MLXCEL_DEFAULT_ORG";

/// Resolve a `-m/--model` value into a concrete on-disk model directory,
/// auto-downloading a HuggingFace repo-id on a cache miss.
///
/// See the [module docs](self) for the full precedence. On success the
/// returned [`PathBuf`] is guaranteed to name an existing path; for the
/// repo-id branch it is the reused or freshly-downloaded snapshot directory.
///
/// # Errors
///
/// Returns an actionable [`anyhow::Error`] when the value is neither an
/// existing path nor a valid `owner/name` repo-id, or when an auto-download was
/// required but failed (network / auth / disk — the underlying
/// [`download_repo`] error is propagated with context).
pub fn resolve_model_source(value: &Path) -> Result<PathBuf> {
    resolve_model_source_with_override(value, None)
}

/// Override-aware variant of [`resolve_model_source`] (issue #107).
///
/// `models_dir` is the inline `--models-dir <path>` flag (or `None`). It is
/// threaded into the store-probe and download steps so a repo-id is reused
/// from / downloaded into the override-aware models root (see
/// [`store::models_root`]). The existing-path and legacy-CWD / HF-cache reuse
/// steps are unaffected. [`resolve_model_source`] delegates here with `None`.
pub fn resolve_model_source_with_override(
    value: &Path,
    models_dir: Option<&Path>,
) -> Result<PathBuf> {
    // 1. Existing on-disk path wins unconditionally — byte-identical to the
    //    pre-#94 local-path behavior. Checked before any repo-id shape test so
    //    a local directory literally named `owner/name` is still used as-is.
    if value.exists() {
        return Ok(value.to_path_buf());
    }

    // The repo-id branch requires a UTF-8 string. A non-UTF-8 `-m` value can
    // only be a (non-existent) local path, so fall through to the error arm.
    let Some(value_str) = value.to_str() else {
        return Err(not_a_model_error(value));
    };

    // 2. `owner/name` repo-id shape → reuse-or-download. An explicit
    //    `owner/name` always wins over the bare-name default org below.
    if is_repo_id_shape(value_str) {
        return resolve_repo_id(value_str, None, models_dir);
    }

    // 3. Bare, prefix-less model name (issue #112): a single valid segment with
    //    no `/`. Prepend the default org ($MLXCEL_DEFAULT_ORG, else
    //    `mlx-community`) and resolve the result as a repo-id, so
    //    `mlxcel run gemma-4-e4b-it-4bit` means
    //    `mlx-community/gemma-4-e4b-it-4bit`. Steps 1 and 2 take precedence, so
    //    an existing local path and an explicit `owner/name` are unaffected.
    if is_repo_segment(value_str) {
        let repo_id = expand_bare_name(value_str)?;
        return resolve_repo_id(&repo_id, None, models_dir);
    }

    // 4. Neither an existing path, a valid repo-id, nor a bare model name.
    Err(not_a_model_error(value))
}

/// Resolve a value already known to have `owner/name` repo-id shape: reuse an
/// existing snapshot (legacy CWD → HF cache → mlxcel store) or download into
/// the mlxcel global store on a miss.
///
/// `revision` selects the HF-cache snapshot revision (branch / tag / commit);
/// `None` means `main`. The CLI subcommands do not currently expose a
/// `--revision` flag, so they pass `None`, matching `mlxcel download`'s default.
///
/// `models_dir` is the inline `--models-dir <path>` override (issue #107),
/// threaded into the store-probe (step 2c) and the download destination
/// (step 2d) so reuse and writes target the override-aware models root.
fn resolve_repo_id(
    repo_id: &str,
    revision: Option<&str>,
    models_dir: Option<&Path>,
) -> Result<PathBuf> {
    let cwd_models = PathBuf::from(LEGACY_MODELS_DIR);

    // 2a–2c: reuse an existing COMPLETE snapshot without re-downloading.
    if let Some(hit) = locate_cached_snapshot(repo_id, revision, &cwd_models, models_dir) {
        return Ok(hit);
    }

    // 2d: no complete snapshot anywhere. A miss is one of two things: nothing on
    // disk, or a PARTIAL mlxcel snapshot left by an interrupted download
    // (config.json + only some shards). We name the latter explicitly — instead
    // of letting the loader die later with a bare `Weight not found` (issue
    // #465) — then recover through the shared hardened downloader either way.
    // Routing the destination back through `download_repo` RESUMES cheaply: it
    // skips files already present and non-zero and re-fetches only what is
    // missing. Reusing it (rather than forking) keeps allow-list filtering,
    // token handling, progress UX, and HF-cache reuse in lock-step with
    // `mlxcel download`.
    let store_dest = store::model_dir_with_override(repo_id, models_dir);
    match store_dest.as_deref().map(classify_snapshot) {
        Some(SnapshotState::Incomplete { missing }) => {
            // `store_dest` is Some in this arm, so the unwrap cannot panic.
            report_incomplete_snapshot(store_dest.as_deref().unwrap_or(&cwd_models), &missing);
        }
        _ => announce_fresh_download(repo_id),
    }

    download_repo(download_options(repo_id, revision, models_dir, false))
        .map_err(|err| anyhow!("failed to download model '{repo_id}': {err}"))?;

    // After a successful download/resume the snapshot is reachable via either the
    // HF cache (download_repo reuses an existing HF snapshot read-only) or the
    // mlxcel store. Re-run the same completeness-gated lookup to return the real
    // landing path.
    if let Some(hit) = locate_cached_snapshot(repo_id, revision, &cwd_models, models_dir) {
        return Ok(hit);
    }

    // The resume did not yield a loadable snapshot (e.g. a present-but-corrupt,
    // non-zero file the resume skipped). Fall back to the "clean re-download"
    // recovery from issue #465: force a full re-fetch of every file, then load.
    eprintln!(
        "[mlxcel] snapshot for '{repo_id}' still incomplete after resume; \
         re-fetching every file..."
    );
    download_repo(download_options(repo_id, revision, models_dir, true))
        .map_err(|err| anyhow!("failed to re-download model '{repo_id}': {err}"))?;

    locate_cached_snapshot(repo_id, revision, &cwd_models, models_dir).ok_or_else(|| {
        anyhow!(
            "downloaded model '{repo_id}' but its snapshot is still incomplete \
             afterwards (expected under the mlxcel store or HuggingFace cache); \
             remove the partial snapshot and retry"
        )
    })
}

/// Probe every reuse location for a complete snapshot of `repo_id`, in the
/// locked precedence order: legacy per-CWD `./models/<basename>`, then the
/// HuggingFace Hub cache, then the mlxcel global store.
///
/// `cwd_models` is the legacy models root (normally `./models`); it is a
/// parameter so unit tests can point it at a temp dir. `models_dir` is the
/// inline `--models-dir <path>` override (issue #107) used for the mlxcel-store
/// probe in step 2c. Returns the first complete snapshot found, or `None` when
/// every location misses.
fn locate_cached_snapshot(
    repo_id: &str,
    revision: Option<&str>,
    cwd_models: &Path,
    models_dir: Option<&Path>,
) -> Option<PathBuf> {
    // 2a. Legacy per-CWD `./models/<basename>` (pre-#93 default location).
    //     Only a fully-materialized snapshot is a hit; an interrupted partial
    //     (config.json + only some shards) is skipped so 2d re-fetches it.
    let legacy = cwd_models.join(repo_basename(repo_id));
    if matches!(classify_snapshot(&legacy), SnapshotState::Complete) {
        return Some(legacy);
    }

    // 2b. Existing HuggingFace Hub cache snapshot (read-only reuse). Its own
    //     completeness gate keys on `config.json`; mlxcel never writes into that
    //     externally-managed layout, so it is left to HuggingFace tooling.
    if let Some(hf) = store::hf_cache_snapshot(repo_id, revision) {
        return Some(hf);
    }

    // 2c. mlxcel global store under the override-aware models root: the
    //     `--models-dir` / `MLXCEL_MODELS_DIR` root directly, or the legacy
    //     `${MLXCEL_CACHE_DIR}/models/<owner>/<name>`. Same full-weight gate as
    //     2a so an interrupted store download is re-fetched, not loaded.
    if let Some(store_dir) = store::model_dir_with_override(repo_id, models_dir)
        && matches!(classify_snapshot(&store_dir), SnapshotState::Complete)
    {
        return Some(store_dir);
    }

    None
}

/// Emit the issue #465 "incomplete download detected" line naming the condition
/// and the re-fetch action (a bounded preview of the missing files).
fn report_incomplete_snapshot(dir: &Path, missing: &[String]) {
    const PREVIEW: usize = 6;
    let shown: Vec<&str> = missing.iter().take(PREVIEW).map(String::as_str).collect();
    let extra = missing.len().saturating_sub(shown.len());
    let more = if extra > 0 {
        format!(", +{extra} more")
    } else {
        String::new()
    };
    println!(
        "[mlxcel] incomplete download detected at {}: re-fetching {} missing file(s): {}{}",
        dir.display(),
        missing.len(),
        shown.join(", "),
        more,
    );
}

/// Print the "not found locally; downloading" line for a genuine cache miss.
fn announce_fresh_download(repo_id: &str) {
    println!("[mlxcel] model '{repo_id}' not found locally; downloading into the mlxcel store...");
}

/// Build [`DownloadOptions`] for the resolver's store-destination download,
/// threading the `--models-dir` override and the resume/clean `force` flag.
fn download_options(
    repo_id: &str,
    revision: Option<&str>,
    models_dir: Option<&Path>,
    force: bool,
) -> DownloadOptions {
    DownloadOptions {
        repo_id: repo_id.to_string(),
        local_dir: None,
        models_dir: models_dir.map(Path::to_path_buf),
        revision: revision.map(str::to_string),
        token: None,
        force,
    }
}

/// True when `value` has HuggingFace `owner/name` repo-id shape:
/// `^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$` — exactly one `/`, with both the owner
/// and name segments non-empty and composed only of ASCII alphanumerics, `.`,
/// `_`, or `-`.
///
/// Implemented with direct char checks (no regex dependency). The single-slash
/// constraint is what distinguishes a repo-id from a multi-segment relative
/// path like `models/foo/bar`, which — if it does not exist on disk — is a
/// user error rather than a repo-id.
fn is_repo_id_shape(value: &str) -> bool {
    let mut parts = value.split('/');
    let (Some(owner), Some(name), None) = (parts.next(), parts.next(), parts.next()) else {
        return false;
    };
    is_repo_segment(owner) && is_repo_segment(name)
}

/// True when `segment` is non-empty and every byte is an ASCII alphanumeric or
/// one of `.`, `_`, `-`.
fn is_repo_segment(segment: &str) -> bool {
    !segment.is_empty()
        && segment
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
}

/// Build the "neither a path nor a repo-id" error for an unresolvable `-m`
/// value.
fn not_a_model_error(value: &Path) -> anyhow::Error {
    anyhow!(
        "model '{}' is neither an existing path nor a valid HuggingFace \
         repo-id (expected `owner/name`, e.g. `mlx-community/Qwen3-4B-4bit`). \
         Pass a local model directory or a repo-id to auto-download.",
        value.display()
    )
}

/// The org to prepend to a bare model name: the trimmed value of
/// `$MLXCEL_DEFAULT_ORG` when set and non-empty, else [`DEFAULT_ORG`]
/// (`mlx-community`). A blank/whitespace value falls back to the default.
fn default_org() -> String {
    std::env::var(DEFAULT_ORG_ENV)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| DEFAULT_ORG.to_string())
}

/// Build the error for when `$MLXCEL_DEFAULT_ORG` expands a bare name into a
/// value that is not a valid `owner/name` repo-id (e.g. the org contains a `/`
/// or an illegal character).
fn bad_default_org_error(org: &str, name: &str) -> anyhow::Error {
    anyhow!(
        "MLXCEL_DEFAULT_ORG='{org}' expands the bare model name '{name}' to an \
         invalid repo-id '{org}/{name}'; the org must be a single path segment \
         ([A-Za-z0-9._-]). Pass a full `owner/name` repo-id instead."
    )
}

/// Expand a bare, prefix-less model name into a canonical `<org>/<name>`
/// repo-id (issue #112), where `org` is [`default_org`].
///
/// This is the single source of truth for the bare-name expansion:
/// [`resolve_model_source_with_override`]'s step 3 and [`normalize_repo_id`]
/// (the `download`-verb entry point, issue #171) both funnel through it, so the
/// resolver-backed commands and the `download` verb cannot drift.
///
/// Emits the `'name' -> owner/name` info line so the expansion is never a
/// silent surprise, matching the resolver UX.
///
/// # Errors
///
/// Returns [`bad_default_org_error`] when `$MLXCEL_DEFAULT_ORG` expands the
/// name into something that is not a valid `owner/name` repo-id (e.g. the org
/// contains a `/` or an illegal character).
fn expand_bare_name(name: &str) -> Result<String> {
    let org = default_org();
    let repo_id = format!("{org}/{name}");
    if !is_repo_id_shape(&repo_id) {
        return Err(bad_default_org_error(&org, name));
    }
    println!("[mlxcel] '{name}' -> {repo_id}");
    Ok(repo_id)
}

/// Normalize a user-supplied model identifier into a canonical HuggingFace
/// repo-id, applying the bare-name → default-org expansion (issue #112) shared
/// with the `-m`/run resolver.
///
/// This is the shared funnel the `download` verb uses (issue #171) so
/// `mlxcel download <bare-name>` and `mlx-server download <bare-name>` expand a
/// prefix-less name to `<default-org>/<name>` exactly like the resolver-backed
/// commands, instead of 404ing on a slashless repo-id. It is applied at the top
/// of [`download_repo`] so both the download URL and the store destination use
/// the canonical id.
///
/// Conservative and idempotent:
///
/// - A bare single segment (no `/`, passes [`is_repo_segment`]) is expanded via
///   [`expand_bare_name`] to `<default-org>/<name>`.
/// - Anything else — an `owner/name` id, a multi-segment value, or a string
///   with characters illegal in a segment — has no expandable bare segment
///   (`is_repo_segment` is false), so it is returned verbatim for the caller /
///   HuggingFace to validate as before. An already-canonical `owner/name` id
///   therefore round-trips unchanged (no double expansion, no duplicate info
///   line), which is what makes the [`resolve_repo_id`] → [`download_repo`] path
///   a no-op here.
///
/// # Errors
///
/// Propagates [`expand_bare_name`]'s error when a bare name expands to an
/// invalid repo-id under a malformed `$MLXCEL_DEFAULT_ORG`.
pub fn normalize_repo_id(value: &str) -> Result<String> {
    if is_repo_segment(value) {
        expand_bare_name(value)
    } else {
        Ok(value.to_string())
    }
}

#[cfg(test)]
#[path = "resolver_tests.rs"]
mod tests;
