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

//! Location-independent global model store (epic #92, issue #93).
//!
//! Today's per-CWD `models/<basename>` default ties a downloaded snapshot to
//! the directory it was fetched from. This module introduces a single shared
//! store so a model downloaded once can be run from anywhere, mirroring the
//! mlx-lm / ollama / LM Studio convenience UX.
//!
//! # Layout
//!
//! - **Store root** — [`store_root`] = `${MLXCEL_CACHE_DIR:-$HOME/.cache/mlxcel}`.
//!   This reuses the exact same root and env-var semantics as the tokenizer
//!   language-analysis disk cache via [`mlxcel_core::cache_root`]; the model
//!   store lives in a sibling `models/` subtree next to `tokenizer-scripts/`.
//! - **Per-model directory** — [`model_dir`] = `store_root()/models/<owner>/<name>`.
//!   The `<owner>/<name>` namespacing matches HuggingFace / LM Studio and
//!   prevents same-name collisions across owners (e.g.
//!   `mlx-community/Qwen3-4B-4bit` vs `Qwen/Qwen3-4B-4bit`).
//!
//! # HuggingFace cache read-reuse
//!
//! [`hf_cache_snapshot`] probes an existing HuggingFace Hub cache
//! (`HF_HUB_CACHE`, then `HF_HOME/hub`, then `~/.cache/huggingface/hub`) for a
//! complete snapshot of the requested repo + revision. When found, the caller
//! reuses it directly so users who already pulled a model with mlx-lm /
//! transformers do not re-fetch gigabytes. This is **read-only**: mlxcel never
//! writes into the HF content-addressed layout.

use std::path::{Path, PathBuf};

/// Sub-directory under the mlxcel cache root that holds downloaded model
/// snapshots, e.g. `${MLXCEL_CACHE_DIR}/models/<owner>/<name>`.
pub const MODELS_SUBDIR: &str = "models";

/// Dedicated environment variable naming the model-store root directly
/// (issue #107). When set to a non-empty value, snapshots live at
/// `$MLXCEL_MODELS_DIR/<owner>/<name>` with **no** intervening `models/`
/// subdir — unlike the legacy `${MLXCEL_CACHE_DIR}/models` path. Read in
/// exactly one place ([`models_root`]) so the precedence stays in one spot.
pub const MODELS_DIR_ENV: &str = "MLXCEL_MODELS_DIR";

/// HuggingFace Hub cache sub-directory (under `HF_HOME`) and the trailing
/// segment of the default `~/.cache/huggingface/hub` path.
const HF_HUB_SUBDIR: &str = "hub";

/// Resolve the mlxcel store root directory
/// (`${MLXCEL_CACHE_DIR:-$HOME/.cache/mlxcel}`).
///
/// Delegates to [`mlxcel_core::cache_root`] so the model store and the
/// tokenizer language-analysis disk cache always agree on the root and on the
/// `MLXCEL_CACHE_DIR` override. Returns `None` only when neither
/// `MLXCEL_CACHE_DIR` nor a home directory can be determined.
pub fn store_root() -> Option<PathBuf> {
    mlxcel_core::cache_root()
}

/// Resolve the **models root** — the directory that directly contains
/// `<owner>/<name>` snapshots — honoring the issue #107 precedence
/// (highest priority first):
///
/// 1. `override_dir` (an inline `--models-dir <path>` CLI flag) — used
///    verbatim as the models root.
/// 2. `MLXCEL_MODELS_DIR` env var — used verbatim as the models root.
/// 3. `${MLXCEL_CACHE_DIR:-$HOME/.cache/mlxcel}/models` — the legacy cache-root
///    path, kept for backward compatibility.
///
/// The dedicated knobs (1, 2) place snapshots directly at
/// `<root>/<owner>/<name>` with no extra `models/` subdir; only the legacy
/// cache-root path (3) appends the `MODELS_SUBDIR` (`models/`). This is the single source of
/// truth for the env var — no other function reads `MLXCEL_MODELS_DIR`.
///
/// Returns `None` only when no override / env var is set **and** [`store_root`]
/// cannot be resolved (no `MLXCEL_CACHE_DIR` and no home directory).
pub fn models_root(override_dir: Option<&Path>) -> Option<PathBuf> {
    if let Some(dir) = override_dir {
        return Some(dir.to_path_buf());
    }
    if let Some(env_dir) = non_empty_env(MODELS_DIR_ENV) {
        return Some(PathBuf::from(env_dir));
    }
    store_root().map(|r| r.join(MODELS_SUBDIR))
}

/// Resolve the global store directory for a given HuggingFace `repo_id`,
/// using the default (cache-root) models root.
///
/// With neither `--models-dir` nor `MLXCEL_MODELS_DIR` set this returns
/// `store_root()/models/<owner>/<name>`; when `MLXCEL_MODELS_DIR` is set it
/// returns `$MLXCEL_MODELS_DIR/<owner>/<name>` (no `models/` subdir). This is a
/// thin `None`-delegating wrapper over [`model_dir_with_override`], which
/// resolves the override-aware [`models_root`]. When `repo_id` has no `/`
/// separator (e.g. a bare `gpt2`), the whole id is the final path component.
///
/// Returns `None` when [`models_root`] cannot be resolved (no
/// `MLXCEL_MODELS_DIR`/`MLXCEL_CACHE_DIR` and no home directory).
pub fn model_dir(repo_id: &str) -> Option<PathBuf> {
    model_dir_with_override(repo_id, None)
}

/// Resolve the per-model directory for `repo_id` under the override-aware
/// [`models_root`] (issue #107).
///
/// `override_dir` is the inline `--models-dir <path>` flag (or `None`). The
/// resolved path is `<models_root>/<owner>/<name>` — the dedicated knobs
/// (`--models-dir` / `MLXCEL_MODELS_DIR`) add no `models/` subdir, while the
/// legacy cache-root path keeps the `models/` segment via [`models_root`].
///
/// Returns `None` when [`models_root`] cannot be resolved.
pub fn model_dir_with_override(repo_id: &str, override_dir: Option<&Path>) -> Option<PathBuf> {
    models_root(override_dir).map(|root| model_dir_under(&root, repo_id))
}

/// Pure helper: compose `<root>/models/<owner>/<name>` from an explicit
/// **cache** root. Split out so unit tests can assert the legacy cache-root
/// layout without depending on process-wide env state. Reimplemented on top of
/// [`model_dir_under`] so the sanitization posture stays in one place.
///
/// `repo_id` segments are sanitized to stay inside the per-model directory:
/// only the `<owner>` and `<name>` components are honored, and any path
/// traversal (`.`, `..`, empty segments) or absolute/backslash shapes collapse
/// to a single safe `<name>` component. This mirrors the downloader's
/// `is_safe_relative_path` posture so an adversarial repo id cannot escape the
/// store root.
///
/// Test-only since #107: production paths go through [`model_dir_under`] via
/// the override-aware [`model_dir_with_override`]; this wrapper is retained so
/// the pre-#107 cache-root layout tests keep asserting the `models/`-subdir
/// semantics directly.
#[cfg(test)]
fn model_dir_in(root: &Path, repo_id: &str) -> PathBuf {
    model_dir_under(&root.join(MODELS_SUBDIR), repo_id)
}

/// Pure helper: compose `<models_root>/<owner>/<name>` from an explicit
/// **models** root (the directory that directly holds snapshots — no
/// `models/` subdir is appended here). The single sanitized-segment join used
/// by both the cache-root layout (`model_dir_in`) and the override-aware
/// layout ([`model_dir_with_override`]).
///
/// `repo_id` segments are sanitized via [`sanitize_repo_id_segments`] so an
/// adversarial id (`../../etc`, an absolute path, backslash shapes) collapses
/// to safe `Normal` components and can never escape `models_root`.
fn model_dir_under(models_root: &Path, repo_id: &str) -> PathBuf {
    let mut dir = models_root.to_path_buf();
    for segment in sanitize_repo_id_segments(repo_id) {
        dir.push(segment);
    }
    dir
}

/// Split a HuggingFace `repo_id` into the path segments used under
/// `models/`, rejecting any unsafe component.
///
/// HuggingFace repo ids are `<owner>/<name>` (exactly one slash) or a bare
/// `<name>`. We keep `Normal` components only; if sanitization leaves nothing
/// usable (e.g. `repo_id` was `..` or `/`), we fall back to a single
/// `"_unknown_"` segment so the path is still well-formed and contained.
fn sanitize_repo_id_segments(repo_id: &str) -> Vec<String> {
    use std::path::Component;
    let trimmed = repo_id.trim().replace('\\', "/");
    let mut out: Vec<String> = Vec::new();
    for raw in trimmed.split('/') {
        if raw.is_empty() || raw == "." || raw == ".." {
            continue;
        }
        // Defensive: collapse any residual path semantics in a single segment.
        let seg = Path::new(raw);
        let mut ok = true;
        for c in seg.components() {
            match c {
                Component::Normal(s) if !s.is_empty() => {}
                _ => {
                    ok = false;
                    break;
                }
            }
        }
        if ok {
            out.push(raw.to_string());
        }
    }
    if out.is_empty() {
        out.push("_unknown_".to_string());
    }
    out
}

/// Resolve the HuggingFace Hub cache directory, honoring the standard env vars.
///
/// Precedence (matches `huggingface_hub`):
/// 1. `HF_HUB_CACHE` — used verbatim as the hub directory.
/// 2. `HF_HOME/hub` — when `HF_HOME` is set.
/// 3. `$HOME/.cache/huggingface/hub` — the default.
///
/// Empty / whitespace-only env values are ignored so an exported-but-empty
/// `HF_HOME=""` does not short-circuit the fallback chain.
fn hf_hub_cache_dir() -> Option<PathBuf> {
    if let Some(dir) = non_empty_env("HF_HUB_CACHE") {
        return Some(PathBuf::from(dir));
    }
    if let Some(home) = non_empty_env("HF_HOME") {
        return Some(PathBuf::from(home).join(HF_HUB_SUBDIR));
    }
    dirs::home_dir().map(|h| h.join(".cache").join("huggingface").join(HF_HUB_SUBDIR))
}

/// Read an env var, returning `Some(trimmed-as-owned)` only for a non-empty,
/// non-whitespace value.
fn non_empty_env(key: &str) -> Option<String> {
    std::env::var(key)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

/// HuggingFace's on-disk repo-folder name for a model: `models--<owner>--<name>`,
/// with every `/` in the repo id replaced by `--`.
fn hf_repo_folder(repo_id: &str) -> String {
    format!("models--{}", repo_id.replace('/', "--"))
}

/// Probe an existing HuggingFace Hub cache for a complete snapshot of
/// `repo_id` at `revision` (defaulting to `main`).
///
/// Returns the snapshot directory (`.../models--<owner>--<name>/snapshots/<sha>`)
/// only when it exists and looks complete (contains a `config.json`). The
/// lookup is **read-only** — nothing is written into the HF cache.
///
/// # Revision resolution
/// HuggingFace names snapshot directories by commit SHA, not by branch/tag.
/// We therefore try, in order:
/// 1. A snapshot directory literally named `revision` (works when the caller
///    passes a commit hash).
/// 2. The SHA recorded in `refs/<revision>` (works for branch/tag names like
///    `main`), then the snapshot directory named by that SHA.
pub fn hf_cache_snapshot(repo_id: &str, revision: Option<&str>) -> Option<PathBuf> {
    let hub = hf_hub_cache_dir()?;
    let repo_dir = hub.join(hf_repo_folder(repo_id));
    if !repo_dir.is_dir() {
        return None;
    }
    let revision = revision.unwrap_or("main");
    let snapshots = repo_dir.join("snapshots");

    // 1. Direct: revision is already a snapshot dir name (commit hash).
    let direct = snapshots.join(revision);
    if snapshot_is_complete(&direct) {
        return Some(direct);
    }

    // 2. Indirect: resolve refs/<revision> -> commit SHA, then snapshots/<sha>.
    let ref_path = repo_dir.join("refs").join(revision);
    if let Ok(sha) = std::fs::read_to_string(&ref_path) {
        let sha = sha.trim();
        if !sha.is_empty() {
            let by_sha = snapshots.join(sha);
            if snapshot_is_complete(&by_sha) {
                return Some(by_sha);
            }
        }
    }

    None
}

/// True when `dir` is an existing directory that contains a `config.json`.
///
/// Mirrors the downloader's own completeness gate (`snapshot_complete`), which
/// also keys on `config.json` presence. Snapshot entries in the HF cache are
/// symlinks into the content-addressed `blobs/` store; `Path::exists` follows
/// symlinks, so a present-and-non-dangling `config.json` is a good signal that
/// the snapshot was fully materialized.
fn snapshot_is_complete(dir: &Path) -> bool {
    dir.is_dir() && dir.join("config.json").exists()
}

// ── Local model management (issue #97) ──────────────────────────────────────

/// A model snapshot found in the mlxcel global store.
///
/// Produced by [`list_models`] for the `mlxcel list` surface. Holds
/// the reconstructed HuggingFace `repo_id` (`<owner>/<name>` or a bare
/// `<name>`), the absolute on-disk directory, and the recursively-summed
/// byte size of that directory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoredModel {
    /// Reconstructed repo id: `<owner>/<name>`, or a bare `<name>` for
    /// snapshots stored directly under `models/` (e.g. a download of `gpt2`).
    pub repo_id: String,
    /// Absolute path to the snapshot directory under the store root.
    pub path: PathBuf,
    /// Recursive on-disk size of `path`, in bytes.
    pub size_bytes: u64,
    /// Last-modified time of the snapshot directory (its own mtime, taken from
    /// `std::fs::metadata(path).modified()`). Surfaced as the `MODIFIED` column
    /// of `mlxcel list`. `None` when the directory could not be `stat`'d or the
    /// platform does not expose a modification time; callers render it as `-`.
    pub modified: Option<std::time::SystemTime>,
}

/// Enumerate complete model snapshots in the mlxcel global store.
///
/// Walks `store_root()/models/` and returns one [`StoredModel`] per directory
/// that looks like a materialized snapshot (contains a `config.json`, the same
/// completeness gate used by the downloader and [`hf_cache_snapshot`]). Both
/// layouts written by issue #93 are recognized:
///
/// - `models/<owner>/<name>/` — the standard HuggingFace namespacing; the
///   reconstructed `repo_id` is `<owner>/<name>`.
/// - `models/<name>/` — a bare-id download (e.g. `gpt2`); the reconstructed
///   `repo_id` is `<name>`.
///
/// Results are sorted by `repo_id` for stable, golden-test-friendly output.
/// Returns an empty vector when the store root cannot be resolved or the
/// `models/` subtree does not exist yet. I/O errors on individual entries are
/// skipped rather than aborting the whole listing, so one unreadable directory
/// does not hide every other model.
///
/// Thin `None`-delegating wrapper over [`list_models_with_override`] for
/// back-compat with pre-#107 callers (default cache-root models root).
pub fn list_models() -> Vec<StoredModel> {
    list_models_with_override(None)
}

/// Enumerate complete model snapshots under the override-aware
/// [`models_root`] (issue #107).
///
/// `override_dir` is the inline `--models-dir <path>` flag (or `None`). The
/// scan walks whichever models root [`models_root`] resolves — the dedicated
/// `--models-dir` / `MLXCEL_MODELS_DIR` root directly, or the legacy
/// `${MLXCEL_CACHE_DIR}/models` subtree. Results are sorted by `repo_id`.
/// Returns an empty vector when the models root cannot be resolved or does not
/// exist yet.
pub fn list_models_with_override(override_dir: Option<&Path>) -> Vec<StoredModel> {
    let Some(models_root) = models_root(override_dir) else {
        return Vec::new();
    };
    let mut out = list_models_under(&models_root);
    out.sort_by(|a, b| a.repo_id.cmp(&b.repo_id));
    out
}

/// Pure helper behind [`list_models`]: enumerate snapshots under an explicit
/// **cache** root. Split out so unit tests can assert against a temp directory
/// without touching process-wide env state. Delegates to
/// [`list_models_under`] after appending the legacy `models/` subdir.
///
/// Test-only since #107: production listing goes through [`list_models_under`]
/// via the override-aware [`list_models_with_override`]; this wrapper preserves
/// the pre-#107 cache-root listing tests.
#[cfg(test)]
fn list_models_in(root: &Path) -> Vec<StoredModel> {
    list_models_under(&root.join(MODELS_SUBDIR))
}

/// Pure helper: enumerate snapshots under an explicit **models** root (the
/// directory that directly holds `<owner>/<name>` snapshots — no `models/`
/// subdir is appended here). Shared by the cache-root listing
/// ([`list_models_in`]) and the override-aware listing
/// ([`list_models_with_override`]). The returned order is filesystem order;
/// callers sort the public result.
fn list_models_under(models_root: &Path) -> Vec<StoredModel> {
    let mut out: Vec<StoredModel> = Vec::new();

    let Ok(top_entries) = std::fs::read_dir(models_root) else {
        return out;
    };

    for top in top_entries.flatten() {
        let top_path = top.path();
        // `read_dir` -> `DirEntry::path` does not follow the entry itself, but
        // an owner directory could in principle be a symlink; only descend
        // into real directories so a symlinked `models/<owner>` cannot make us
        // walk outside the store.
        if !top_path.is_dir() || top_path.symlink_metadata().is_ok_and(|m| m.is_symlink()) {
            continue;
        }
        let Some(top_name) = top.file_name().to_str().map(str::to_owned) else {
            continue;
        };

        // Case 1: a bare-id snapshot stored directly at models/<name>.
        if snapshot_is_complete(&top_path) {
            out.push(StoredModel {
                repo_id: top_name.clone(),
                modified: std::fs::metadata(&top_path).and_then(|m| m.modified()).ok(),
                size_bytes: dir_size(&top_path),
                path: top_path.clone(),
            });
            // A bare-id snapshot directory is terminal; do not also treat it
            // as an owner directory.
            continue;
        }

        // Case 2: an owner directory holding models/<owner>/<name> snapshots.
        let Ok(inner_entries) = std::fs::read_dir(&top_path) else {
            continue;
        };
        for inner in inner_entries.flatten() {
            let inner_path = inner.path();
            if !inner_path.is_dir() || inner_path.symlink_metadata().is_ok_and(|m| m.is_symlink()) {
                continue;
            }
            if !snapshot_is_complete(&inner_path) {
                continue;
            }
            let Some(inner_name) = inner.file_name().to_str().map(str::to_owned) else {
                continue;
            };
            out.push(StoredModel {
                repo_id: format!("{top_name}/{inner_name}"),
                modified: std::fs::metadata(&inner_path)
                    .and_then(|m| m.modified())
                    .ok(),
                size_bytes: dir_size(&inner_path),
                path: inner_path.clone(),
            });
        }
    }

    out
}

/// Recursively sum the byte sizes of every regular file under `dir`.
///
/// Symlinks are not followed (sizes are taken from `symlink_metadata`), so a
/// snapshot containing symlinks counts the link entry itself rather than its
/// (possibly out-of-tree) target. I/O errors on individual entries contribute
/// zero rather than aborting, keeping the size best-effort but never panicking.
///
/// Public so the CLI can size a single snapshot directory (e.g. the `mlxcel rm`
/// confirmation prompt) without re-listing or re-walking the whole store.
pub fn dir_size(dir: &Path) -> u64 {
    let mut total: u64 = 0;
    let mut stack: Vec<PathBuf> = vec![dir.to_path_buf()];
    while let Some(current) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&current) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let Ok(meta) = path.symlink_metadata() else {
                continue;
            };
            if meta.is_dir() {
                stack.push(path);
            } else {
                // Regular files and symlink entries both contribute their own
                // on-disk length; we intentionally do not follow symlinks.
                total = total.saturating_add(meta.len());
            }
        }
    }
    total
}

/// Outcome of a [`remove_model`] request.
#[derive(Debug, PartialEq, Eq)]
pub enum RemoveOutcome {
    /// The store directory existed and was deleted. Carries the freed size in
    /// bytes (measured before deletion) and the removed path.
    Removed { path: PathBuf, size_bytes: u64 },
    /// No snapshot for this repo id exists in the mlxcel store, but a complete
    /// snapshot was found in the read-only HuggingFace cache. mlxcel refuses to
    /// touch the HF cache (it is not ours to manage). Carries the HF snapshot
    /// path for the caller's warning message.
    HfCacheOnly { hf_path: PathBuf },
    /// No snapshot for this repo id exists anywhere mlxcel manages or reads.
    NotFound,
}

/// Errors that can abort a [`remove_model`] request before any deletion.
#[derive(Debug)]
pub enum RemoveError {
    /// The store root could not be resolved (no `MLXCEL_CACHE_DIR` and no home
    /// directory).
    NoStoreRoot,
    /// The resolved target escaped the active models root (whichever
    /// [`models_root`] resolved). This is a safety stop: it should be
    /// unreachable given the path sanitization in `model_dir_under`, but is
    /// enforced defensively so a future regression can never delete outside
    /// the store.
    OutsideStore(PathBuf),
    /// Deleting the store directory failed (I/O error), with the offending
    /// path and underlying error.
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
}

impl std::fmt::Display for RemoveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RemoveError::NoStoreRoot => write!(
                f,
                "cannot resolve the mlxcel model store root \
                 (pass --models-dir, or set MLXCEL_MODELS_DIR or MLXCEL_CACHE_DIR, \
                 or ensure a home directory is available)"
            ),
            RemoveError::OutsideStore(p) => write!(
                f,
                "refusing to remove {}: resolved path is outside the model store",
                p.display()
            ),
            RemoveError::Io { path, source } => {
                write!(f, "failed to remove {}: {source}", path.display())
            }
        }
    }
}

impl std::error::Error for RemoveError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            RemoveError::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Remove a model snapshot from the mlxcel global store.
///
/// Resolves the target via [`model_dir`] (which already sanitizes the repo id
/// against path traversal) and, before deleting, re-verifies that the target
/// is contained inside `store_root()/models/`. The deletion therefore can only
/// ever touch a directory under the store — never the HF cache, never a path
/// reached via `..`, never an absolute escape.
///
/// Behavior:
/// - **Store hit** → recursively deletes the directory and returns
///   [`RemoveOutcome::Removed`] with the freed size.
/// - **Store miss but HF-cache hit** → returns [`RemoveOutcome::HfCacheOnly`]
///   without deleting anything (the HF cache is read-only to mlxcel).
/// - **Miss everywhere** → returns [`RemoveOutcome::NotFound`].
///
/// `revision` is only used for the HF-cache probe (the mlxcel store is not
/// revision-namespaced); pass `None` for the default `main`.
///
/// Thin `None`-delegating wrapper over [`remove_model_with_override`] for
/// back-compat with pre-#107 callers (default cache-root models root).
pub fn remove_model(repo_id: &str, revision: Option<&str>) -> Result<RemoveOutcome, RemoveError> {
    remove_model_with_override(repo_id, revision, None)
}

/// Remove a model snapshot from the override-aware [`models_root`]
/// (issue #107).
///
/// `override_dir` is the inline `--models-dir <path>` flag (or `None`). The
/// containment safety check (`is_within`) uses the resolved models root as its
/// base, so a deletion can only ever touch a directory under whichever root
/// won the precedence — never the HF cache, never a `..` escape. Behavior is
/// otherwise identical to [`remove_model`].
pub fn remove_model_with_override(
    repo_id: &str,
    revision: Option<&str>,
    override_dir: Option<&Path>,
) -> Result<RemoveOutcome, RemoveError> {
    let models_root = models_root(override_dir).ok_or(RemoveError::NoStoreRoot)?;
    remove_model_under(&models_root, repo_id, revision)
}

/// Pure-ish helper behind [`remove_model`] operating against an explicit
/// **cache** root. Split out so unit tests can drive deletion against a temp
/// directory. Delegates to [`remove_model_under`] after appending the legacy
/// `models/` subdir.
///
/// Test-only since #107: production removal goes through [`remove_model_under`]
/// via the override-aware [`remove_model_with_override`]; this wrapper preserves
/// the pre-#107 cache-root removal tests.
#[cfg(test)]
fn remove_model_in(
    root: &Path,
    repo_id: &str,
    revision: Option<&str>,
) -> Result<RemoveOutcome, RemoveError> {
    remove_model_under(&root.join(MODELS_SUBDIR), repo_id, revision)
}

/// Pure-ish helper operating against an explicit **models** root (the
/// directory that directly holds snapshots — no `models/` subdir is appended
/// here). Shared by the cache-root removal ([`remove_model_in`]) and the
/// override-aware removal ([`remove_model_with_override`]). The HF-cache probe
/// still consults the process env via [`hf_cache_snapshot`] (only reached on a
/// store miss).
fn remove_model_under(
    models_root: &Path,
    repo_id: &str,
    revision: Option<&str>,
) -> Result<RemoveOutcome, RemoveError> {
    let target = model_dir_under(models_root, repo_id);

    // Defense-in-depth containment check. `model_dir_under` already strips `..`
    // and absolute components, but we re-assert that the composed path stays
    // under the resolved models root so a deletion can never escape the store.
    // Use a lexical/canonical comparison that does not require the target to
    // exist.
    if !is_within(models_root, &target) {
        return Err(RemoveError::OutsideStore(target));
    }

    if target.is_dir() {
        let size = dir_size(&target);
        std::fs::remove_dir_all(&target).map_err(|source| RemoveError::Io {
            path: target.clone(),
            source,
        })?;
        return Ok(RemoveOutcome::Removed {
            path: target,
            size_bytes: size,
        });
    }

    // Store miss: is it sitting read-only in the HuggingFace cache?
    if let Some(hf_path) = hf_cache_snapshot(repo_id, revision) {
        return Ok(RemoveOutcome::HfCacheOnly { hf_path });
    }

    Ok(RemoveOutcome::NotFound)
}

/// True when `child` is `base` itself or lexically nested under it.
///
/// Both paths are first canonicalized when they exist; for the (common) case
/// where `child` does not exist yet, we canonicalize `base` and compare the
/// canonical base against `child` after stripping any `.`/`..` we can resolve
/// lexically. Because `model_dir_under` never emits `..` components, a plain
/// `starts_with` against the canonical (or, if canonicalization fails, the raw)
/// base is sound here; the explicit check is the defensive backstop.
fn is_within(base: &Path, child: &Path) -> bool {
    let base_c = base.canonicalize().unwrap_or_else(|_| base.to_path_buf());
    // If the child exists, compare canonical forms (resolves symlinks too).
    if let Ok(child_c) = child.canonicalize() {
        return child_c.starts_with(&base_c) || child_c == base_c;
    }
    // Child does not exist yet (typical for a not-downloaded repo): compare the
    // raw child against the canonical base, and also against the raw base, so a
    // symlinked store root does not produce a false negative.
    child.starts_with(&base_c) || child.starts_with(base)
}

#[cfg(test)]
mod tests {
    use super::*;
    // Env-var-sensitive tests must serialize through the crate-wide `ENV_LOCK`
    // (issue #573): Rust 2024's `set_var`/`remove_var` are `unsafe` because
    // libc's env block has no internal lock and concurrent mutation is UB.
    use crate::test_support::env_lock::env_lock;

    /// Restore an env var to a prior captured state.
    fn restore_env(key: &str, prev: Option<String>) {
        unsafe {
            match prev {
                Some(value) => std::env::set_var(key, value),
                None => std::env::remove_var(key),
            }
        }
    }

    // ── store_root / model_dir ───────────────────────────────────────────────

    #[test]
    fn store_root_honors_mlxcel_cache_dir_override() {
        let _guard = env_lock();
        let prev = std::env::var("MLXCEL_CACHE_DIR").ok();
        unsafe {
            std::env::set_var("MLXCEL_CACHE_DIR", "/tmp/mlxcel-store-test");
        }
        let root = store_root();
        restore_env("MLXCEL_CACHE_DIR", prev);

        assert_eq!(root, Some(PathBuf::from("/tmp/mlxcel-store-test")));
    }

    #[test]
    fn model_dir_uses_owner_name_layout_under_override() {
        let _guard = env_lock();
        let prev = std::env::var("MLXCEL_CACHE_DIR").ok();
        unsafe {
            std::env::set_var("MLXCEL_CACHE_DIR", "/tmp/mlxcel-store-test");
        }
        let dir = model_dir("mlx-community/Qwen3-4B-4bit");
        restore_env("MLXCEL_CACHE_DIR", prev);

        assert_eq!(
            dir,
            Some(
                PathBuf::from("/tmp/mlxcel-store-test")
                    .join("models")
                    .join("mlx-community")
                    .join("Qwen3-4B-4bit")
            )
        );
    }

    #[test]
    fn model_dir_in_separates_owners_with_same_name() {
        let root = PathBuf::from("/store");
        let a = model_dir_in(&root, "mlx-community/Qwen3-4B-4bit");
        let b = model_dir_in(&root, "Qwen/Qwen3-4B-4bit");
        assert_ne!(a, b);
        assert_eq!(
            a,
            PathBuf::from("/store/models/mlx-community/Qwen3-4B-4bit")
        );
        assert_eq!(b, PathBuf::from("/store/models/Qwen/Qwen3-4B-4bit"));
    }

    #[test]
    fn model_dir_in_handles_bare_repo_id() {
        let root = PathBuf::from("/store");
        assert_eq!(
            model_dir_in(&root, "gpt2"),
            PathBuf::from("/store/models/gpt2")
        );
    }

    #[test]
    fn model_dir_in_rejects_path_traversal() {
        let root = PathBuf::from("/store");
        // A malicious repo id must never escape the models/ subtree.
        let escaped = model_dir_in(&root, "../../etc/passwd");
        assert!(
            escaped.starts_with(PathBuf::from("/store").join("models")),
            "sanitized path {escaped:?} escaped the models/ subtree"
        );
        assert_eq!(
            escaped,
            PathBuf::from("/store/models/etc/passwd"),
            "only Normal components should survive sanitization"
        );

        // A repo id that sanitizes to nothing collapses to a contained marker.
        let only_dots = model_dir_in(&root, "..");
        assert_eq!(only_dots, PathBuf::from("/store/models/_unknown_"));
    }

    // ── models_root / model_dir_with_override precedence ladder (issue #107) ──

    /// Capture, then clear, both env vars that feed [`models_root`] so each
    /// precedence case starts from a known state. Returns the prior values for
    /// restoration. Must be called while holding the env lock.
    fn clear_models_env() -> (Option<String>, Option<String>) {
        let prev_models = std::env::var("MLXCEL_MODELS_DIR").ok();
        let prev_cache = std::env::var("MLXCEL_CACHE_DIR").ok();
        unsafe {
            std::env::remove_var("MLXCEL_MODELS_DIR");
            std::env::remove_var("MLXCEL_CACHE_DIR");
        }
        (prev_models, prev_cache)
    }

    fn restore_models_env(prev: (Option<String>, Option<String>)) {
        restore_env("MLXCEL_MODELS_DIR", prev.0);
        restore_env("MLXCEL_CACHE_DIR", prev.1);
    }

    #[test]
    fn models_root_inline_override_beats_env_and_cache_dir() {
        let _guard = env_lock();
        let prev = clear_models_env();
        // Set BOTH env knobs so we prove the inline override wins over both.
        unsafe {
            std::env::set_var("MLXCEL_MODELS_DIR", "/env/models");
            std::env::set_var("MLXCEL_CACHE_DIR", "/cache");
        }
        let root = models_root(Some(Path::new("/inline/models")));
        let dir = model_dir_with_override("owner/name", Some(Path::new("/inline/models")));
        restore_models_env(prev);

        // Inline override is the models root verbatim — no `models/` subdir.
        assert_eq!(root, Some(PathBuf::from("/inline/models")));
        assert_eq!(dir, Some(PathBuf::from("/inline/models/owner/name")));
    }

    #[test]
    fn models_root_env_var_beats_cache_dir_with_no_models_subdir() {
        let _guard = env_lock();
        let prev = clear_models_env();
        unsafe {
            std::env::set_var("MLXCEL_MODELS_DIR", "/env/models");
            std::env::set_var("MLXCEL_CACHE_DIR", "/cache");
        }
        let root = models_root(None);
        let dir = model_dir_with_override("owner/name", None);
        restore_models_env(prev);

        // The dedicated env var is the models root verbatim: snapshots live at
        // `$MLXCEL_MODELS_DIR/<owner>/<name>`, NOT `.../models/<owner>/<name>`.
        assert_eq!(root, Some(PathBuf::from("/env/models")));
        assert_eq!(dir, Some(PathBuf::from("/env/models/owner/name")));
    }

    #[test]
    fn models_root_falls_back_to_cache_dir_models_subdir() {
        let _guard = env_lock();
        let prev = clear_models_env();
        // Only the legacy cache-dir knob is set: the `models/` subdir IS
        // appended (backward-compat semantics).
        unsafe {
            std::env::set_var("MLXCEL_CACHE_DIR", "/cache");
        }
        let root = models_root(None);
        let dir = model_dir_with_override("owner/name", None);
        restore_models_env(prev);

        assert_eq!(root, Some(PathBuf::from("/cache/models")));
        assert_eq!(dir, Some(PathBuf::from("/cache/models/owner/name")));
    }

    #[test]
    fn models_root_empty_env_var_is_ignored() {
        let _guard = env_lock();
        let prev = clear_models_env();
        // An exported-but-empty MLXCEL_MODELS_DIR must NOT short-circuit the
        // fallback to the cache-dir path.
        unsafe {
            std::env::set_var("MLXCEL_MODELS_DIR", "   ");
            std::env::set_var("MLXCEL_CACHE_DIR", "/cache");
        }
        let root = models_root(None);
        restore_models_env(prev);

        assert_eq!(root, Some(PathBuf::from("/cache/models")));
    }

    #[test]
    fn model_dir_with_override_rejects_path_traversal_under_override_root() {
        // An adversarial repo id must never escape the override models root.
        let override_root = Path::new("/inline/store");
        let escaped = model_dir_under(override_root, "../../etc/passwd");
        assert!(
            escaped.starts_with(override_root),
            "sanitized path {escaped:?} escaped the override models root"
        );
        assert_eq!(
            escaped,
            PathBuf::from("/inline/store/etc/passwd"),
            "only Normal components should survive sanitization"
        );

        // And via the public override entry point (inline override given).
        let _guard = env_lock();
        let prev = clear_models_env();
        let dir = model_dir_with_override("../../etc/passwd", Some(override_root));
        restore_models_env(prev);
        assert_eq!(dir, Some(PathBuf::from("/inline/store/etc/passwd")));

        // A repo id that sanitizes to nothing collapses to a contained marker.
        assert_eq!(
            model_dir_under(override_root, ".."),
            PathBuf::from("/inline/store/_unknown_")
        );
    }

    #[test]
    fn hf_repo_folder_matches_hf_layout() {
        assert_eq!(
            hf_repo_folder("mlx-community/Qwen3-4B-4bit"),
            "models--mlx-community--Qwen3-4B-4bit"
        );
        assert_eq!(hf_repo_folder("gpt2"), "models--gpt2");
    }

    // ── hf_cache_snapshot ────────────────────────────────────────────────────

    /// Build a fake HF hub cache for `repo_id` with a single snapshot named by
    /// `sha`, a `refs/<branch>` pointer, and a `config.json` inside the
    /// snapshot. Returns the hub root.
    fn make_hf_cache(
        hub: &Path,
        repo_id: &str,
        sha: &str,
        branch: &str,
        complete: bool,
    ) -> PathBuf {
        let repo_dir = hub.join(hf_repo_folder(repo_id));
        let snap = repo_dir.join("snapshots").join(sha);
        std::fs::create_dir_all(&snap).unwrap();
        if complete {
            std::fs::write(snap.join("config.json"), b"{}").unwrap();
        }
        let refs = repo_dir.join("refs");
        std::fs::create_dir_all(&refs).unwrap();
        std::fs::write(refs.join(branch), sha).unwrap();
        repo_dir
    }

    #[test]
    fn hf_cache_snapshot_resolves_branch_via_refs() {
        let tmp = tempfile::tempdir().unwrap();
        let hub = tmp.path().join("hub");
        let sha = "0123456789abcdef0123456789abcdef01234567";
        make_hf_cache(&hub, "mlx-community/Qwen3-4B-4bit", sha, "main", true);

        let _guard = env_lock();
        let prev_cache = std::env::var("HF_HUB_CACHE").ok();
        let prev_home = std::env::var("HF_HOME").ok();
        unsafe {
            std::env::set_var("HF_HUB_CACHE", &hub);
            std::env::remove_var("HF_HOME");
        }
        let found = hf_cache_snapshot("mlx-community/Qwen3-4B-4bit", None);
        restore_env("HF_HUB_CACHE", prev_cache);
        restore_env("HF_HOME", prev_home);

        assert_eq!(
            found,
            Some(
                hub.join(hf_repo_folder("mlx-community/Qwen3-4B-4bit"))
                    .join("snapshots")
                    .join(sha)
            )
        );
    }

    #[test]
    fn hf_cache_snapshot_resolves_direct_commit_hash() {
        let tmp = tempfile::tempdir().unwrap();
        let hub = tmp.path().join("hub");
        let sha = "abcdefabcdefabcdefabcdefabcdefabcdefabcd";
        make_hf_cache(&hub, "owner/model", sha, "main", true);

        let _guard = env_lock();
        let prev_cache = std::env::var("HF_HUB_CACHE").ok();
        let prev_home = std::env::var("HF_HOME").ok();
        unsafe {
            std::env::set_var("HF_HUB_CACHE", &hub);
            std::env::remove_var("HF_HOME");
        }
        // Pass the SHA directly as the revision.
        let found = hf_cache_snapshot("owner/model", Some(sha));
        restore_env("HF_HUB_CACHE", prev_cache);
        restore_env("HF_HOME", prev_home);

        assert_eq!(
            found,
            Some(
                hub.join(hf_repo_folder("owner/model"))
                    .join("snapshots")
                    .join(sha)
            )
        );
    }

    #[test]
    fn hf_cache_snapshot_uses_hf_home_hub_subdir() {
        let tmp = tempfile::tempdir().unwrap();
        // Here we set HF_HOME (not HF_HUB_CACHE); the hub lives under HF_HOME/hub.
        let hf_home = tmp.path();
        let hub = hf_home.join("hub");
        let sha = "1111111111111111111111111111111111111111";
        make_hf_cache(&hub, "owner/model", sha, "main", true);

        let _guard = env_lock();
        let prev_cache = std::env::var("HF_HUB_CACHE").ok();
        let prev_home = std::env::var("HF_HOME").ok();
        unsafe {
            std::env::remove_var("HF_HUB_CACHE");
            std::env::set_var("HF_HOME", hf_home);
        }
        let found = hf_cache_snapshot("owner/model", None);
        restore_env("HF_HUB_CACHE", prev_cache);
        restore_env("HF_HOME", prev_home);

        assert_eq!(
            found,
            Some(
                hub.join(hf_repo_folder("owner/model"))
                    .join("snapshots")
                    .join(sha)
            )
        );
    }

    #[test]
    fn hf_cache_snapshot_none_when_incomplete() {
        let tmp = tempfile::tempdir().unwrap();
        let hub = tmp.path().join("hub");
        let sha = "2222222222222222222222222222222222222222";
        // complete=false → no config.json inside the snapshot.
        make_hf_cache(&hub, "owner/model", sha, "main", false);

        let _guard = env_lock();
        let prev_cache = std::env::var("HF_HUB_CACHE").ok();
        let prev_home = std::env::var("HF_HOME").ok();
        unsafe {
            std::env::set_var("HF_HUB_CACHE", &hub);
            std::env::remove_var("HF_HOME");
        }
        let found = hf_cache_snapshot("owner/model", None);
        restore_env("HF_HUB_CACHE", prev_cache);
        restore_env("HF_HOME", prev_home);

        assert_eq!(found, None);
    }

    #[test]
    fn hf_cache_snapshot_none_when_repo_absent() {
        let tmp = tempfile::tempdir().unwrap();
        let hub = tmp.path().join("hub");
        std::fs::create_dir_all(&hub).unwrap();

        let _guard = env_lock();
        let prev_cache = std::env::var("HF_HUB_CACHE").ok();
        let prev_home = std::env::var("HF_HOME").ok();
        unsafe {
            std::env::set_var("HF_HUB_CACHE", &hub);
            std::env::remove_var("HF_HOME");
        }
        let found = hf_cache_snapshot("owner/never-downloaded", None);
        restore_env("HF_HUB_CACHE", prev_cache);
        restore_env("HF_HOME", prev_home);

        assert_eq!(found, None);
    }

    // ── list_models_in / dir_size (issue #97) ────────────────────────────────

    /// Materialize a complete snapshot at `<root>/models/<repo_id>` with a
    /// `config.json` plus an extra payload file of `extra_bytes` so the
    /// recursive size is non-trivial. Returns the snapshot directory.
    fn make_store_snapshot(root: &Path, repo_id: &str, extra_bytes: usize) -> PathBuf {
        let dir = model_dir_in(root, repo_id);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), b"{}").unwrap();
        std::fs::write(dir.join("model.safetensors"), vec![0u8; extra_bytes]).unwrap();
        dir
    }

    #[test]
    fn list_models_in_finds_owner_name_and_bare() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        make_store_snapshot(root, "mlx-community/Qwen3-4B-4bit", 100);
        make_store_snapshot(root, "Qwen/Qwen3-4B-4bit", 200);
        make_store_snapshot(root, "gpt2", 300); // bare id

        let mut found = list_models_in(root);
        found.sort_by(|a, b| a.repo_id.cmp(&b.repo_id));

        let ids: Vec<&str> = found.iter().map(|m| m.repo_id.as_str()).collect();
        assert_eq!(
            ids,
            vec!["Qwen/Qwen3-4B-4bit", "gpt2", "mlx-community/Qwen3-4B-4bit"]
        );

        // Sizes include config.json (2 bytes) + payload.
        let gpt2 = found.iter().find(|m| m.repo_id == "gpt2").unwrap();
        assert_eq!(gpt2.size_bytes, 2 + 300);
        assert_eq!(gpt2.path, root.join("models").join("gpt2"));
    }

    #[test]
    fn list_models_in_skips_incomplete_dirs() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        // Complete model.
        make_store_snapshot(root, "owner/good", 10);
        // Incomplete: owner dir with a child lacking config.json.
        let bad = root.join("models").join("owner").join("partial");
        std::fs::create_dir_all(&bad).unwrap();
        std::fs::write(bad.join("model.safetensors"), b"xxxx").unwrap();
        // A stray non-model directory directly under models/.
        std::fs::create_dir_all(root.join("models").join("scratch")).unwrap();

        let found = list_models_in(root);
        let ids: Vec<&str> = found.iter().map(|m| m.repo_id.as_str()).collect();
        assert_eq!(ids, vec!["owner/good"]);
    }

    #[test]
    fn list_models_in_empty_when_no_models_dir() {
        let tmp = tempfile::tempdir().unwrap();
        // No models/ subdir created at all.
        assert!(list_models_in(tmp.path()).is_empty());
    }

    #[test]
    fn dir_size_sums_nested_files() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        std::fs::write(root.join("a.bin"), vec![0u8; 1000]).unwrap();
        let sub = root.join("sub");
        std::fs::create_dir_all(&sub).unwrap();
        std::fs::write(sub.join("b.bin"), vec![0u8; 2345]).unwrap();
        assert_eq!(dir_size(root), 1000 + 2345);
    }

    // ── remove_model_in (issue #97) ──────────────────────────────────────────

    #[test]
    fn remove_model_in_deletes_store_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let dir = make_store_snapshot(root, "owner/model", 500);
        assert!(dir.is_dir());

        let outcome = remove_model_in(root, "owner/model", None).unwrap();
        match outcome {
            RemoveOutcome::Removed { path, size_bytes } => {
                assert_eq!(path, dir);
                assert_eq!(size_bytes, 2 + 500);
            }
            other => panic!("expected Removed, got {other:?}"),
        }
        assert!(!dir.exists(), "directory should be gone after removal");
    }

    #[test]
    fn remove_model_in_not_found_when_absent_everywhere() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        // Point the HF cache at an empty dir so the probe also misses.
        let hub = tmp.path().join("hub");
        std::fs::create_dir_all(&hub).unwrap();

        let _guard = env_lock();
        let prev_cache = std::env::var("HF_HUB_CACHE").ok();
        let prev_home = std::env::var("HF_HOME").ok();
        unsafe {
            std::env::set_var("HF_HUB_CACHE", &hub);
            std::env::remove_var("HF_HOME");
        }
        let outcome = remove_model_in(root, "owner/never", None);
        restore_env("HF_HUB_CACHE", prev_cache);
        restore_env("HF_HOME", prev_home);

        assert_eq!(outcome.unwrap(), RemoveOutcome::NotFound);
    }

    #[test]
    fn remove_model_in_reports_hf_cache_only_without_deleting() {
        let store_tmp = tempfile::tempdir().unwrap();
        let root = store_tmp.path();
        // Not in the mlxcel store; only in the HF cache.
        let hf_tmp = tempfile::tempdir().unwrap();
        let hub = hf_tmp.path().join("hub");
        let sha = "3333333333333333333333333333333333333333";
        make_hf_cache(&hub, "owner/cached", sha, "main", true);
        let expected_hf = hub
            .join(hf_repo_folder("owner/cached"))
            .join("snapshots")
            .join(sha);

        let _guard = env_lock();
        let prev_cache = std::env::var("HF_HUB_CACHE").ok();
        let prev_home = std::env::var("HF_HOME").ok();
        unsafe {
            std::env::set_var("HF_HUB_CACHE", &hub);
            std::env::remove_var("HF_HOME");
        }
        let outcome = remove_model_in(root, "owner/cached", None);
        restore_env("HF_HUB_CACHE", prev_cache);
        restore_env("HF_HOME", prev_home);

        assert_eq!(
            outcome.unwrap(),
            RemoveOutcome::HfCacheOnly {
                hf_path: expected_hf.clone()
            }
        );
        // The HF snapshot must remain untouched.
        assert!(expected_hf.join("config.json").exists());
    }

    #[test]
    fn remove_model_in_contains_traversal_to_store() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        // Stage a file OUTSIDE the models/ subtree that a naive join+remove
        // could have reached via `..`. It must survive.
        let outside = root.join("victim.txt");
        std::fs::write(&outside, b"do not delete").unwrap();
        // Point HF cache at an empty dir so the store-miss probe is
        // deterministic and never consults the host's real HF cache.
        let hub = tmp.path().join("hub");
        std::fs::create_dir_all(&hub).unwrap();

        // A traversal repo id sanitizes (via model_dir_in) to a path contained
        // under models/, never escaping to the sibling file. Removal of the
        // (non-existent) sanitized path is NotFound, and the outside file is
        // untouched.
        let _guard = env_lock();
        let prev_cache = std::env::var("HF_HUB_CACHE").ok();
        let prev_home = std::env::var("HF_HOME").ok();
        unsafe {
            std::env::set_var("HF_HUB_CACHE", &hub);
            std::env::remove_var("HF_HOME");
        }
        let outcome = remove_model_in(root, "../../victim.txt", None);
        restore_env("HF_HUB_CACHE", prev_cache);
        restore_env("HF_HOME", prev_home);

        assert_eq!(outcome.unwrap(), RemoveOutcome::NotFound);
        assert!(
            outside.exists(),
            "file outside the store must not be removed"
        );
    }

    // ── list / remove under an override models root (issue #107) ─────────────

    /// Materialize a complete snapshot directly under a **models root** (no
    /// `models/` subdir): `<models_root>/<repo_id>` with a `config.json`.
    fn make_snapshot_under(models_root: &Path, repo_id: &str) -> PathBuf {
        let dir = model_dir_under(models_root, repo_id);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), b"{}").unwrap();
        dir
    }

    #[test]
    fn list_models_with_override_walks_override_root_without_models_subdir() {
        let tmp = tempfile::tempdir().unwrap();
        let override_root = tmp.path().join("custom-store");
        // Snapshots land directly under the override root: no `models/` segment.
        make_snapshot_under(&override_root, "owner/model");
        make_snapshot_under(&override_root, "gpt2");

        let _guard = env_lock();
        let prev = clear_models_env();
        // Set decoy env knobs that must be ignored when an inline override is
        // passed. The cache-dir decoy points at an empty temp dir.
        let decoy_cache = tmp.path().join("decoy-cache");
        std::fs::create_dir_all(&decoy_cache).unwrap();
        unsafe {
            std::env::set_var("MLXCEL_MODELS_DIR", tmp.path().join("decoy-env"));
            std::env::set_var("MLXCEL_CACHE_DIR", &decoy_cache);
        }
        let found = list_models_with_override(Some(&override_root));
        restore_models_env(prev);

        let ids: Vec<&str> = found.iter().map(|m| m.repo_id.as_str()).collect();
        assert_eq!(ids, vec!["gpt2", "owner/model"]);
        // Path is under the override root directly, not under `<root>/models`.
        let model = found.iter().find(|m| m.repo_id == "owner/model").unwrap();
        assert_eq!(model.path, override_root.join("owner").join("model"));
    }

    #[test]
    fn remove_model_with_override_deletes_from_override_root() {
        let tmp = tempfile::tempdir().unwrap();
        let override_root = tmp.path().join("custom-store");
        let dir = make_snapshot_under(&override_root, "owner/model");
        assert!(dir.is_dir());

        let _guard = env_lock();
        let prev = clear_models_env();
        // Point HF cache at an empty dir so the store-miss probe is deterministic.
        let prev_hf_cache = std::env::var("HF_HUB_CACHE").ok();
        let prev_hf_home = std::env::var("HF_HOME").ok();
        let empty_hf = tmp.path().join("hf");
        std::fs::create_dir_all(&empty_hf).unwrap();
        unsafe {
            std::env::set_var("HF_HUB_CACHE", &empty_hf);
            std::env::remove_var("HF_HOME");
        }
        let outcome = remove_model_with_override("owner/model", None, Some(&override_root));
        restore_env("HF_HUB_CACHE", prev_hf_cache);
        restore_env("HF_HOME", prev_hf_home);
        restore_models_env(prev);

        match outcome.unwrap() {
            RemoveOutcome::Removed { path, .. } => assert_eq!(path, dir),
            other => panic!("expected Removed, got {other:?}"),
        }
        assert!(!dir.exists(), "directory should be gone after removal");
    }

    #[test]
    fn remove_model_with_override_contains_traversal_to_override_root() {
        let tmp = tempfile::tempdir().unwrap();
        let override_root = tmp.path().join("custom-store");
        std::fs::create_dir_all(&override_root).unwrap();
        // Stage a victim OUTSIDE the override root that `..` must never reach.
        let outside = tmp.path().join("victim.txt");
        std::fs::write(&outside, b"do not delete").unwrap();

        let _guard = env_lock();
        let prev = clear_models_env();
        let prev_hf_cache = std::env::var("HF_HUB_CACHE").ok();
        let prev_hf_home = std::env::var("HF_HOME").ok();
        let empty_hf = tmp.path().join("hf");
        std::fs::create_dir_all(&empty_hf).unwrap();
        unsafe {
            std::env::set_var("HF_HUB_CACHE", &empty_hf);
            std::env::remove_var("HF_HOME");
        }
        let outcome = remove_model_with_override("../../victim.txt", None, Some(&override_root));
        restore_env("HF_HUB_CACHE", prev_hf_cache);
        restore_env("HF_HOME", prev_hf_home);
        restore_models_env(prev);

        assert_eq!(outcome.unwrap(), RemoveOutcome::NotFound);
        assert!(
            outside.exists(),
            "file outside the override root must not be removed"
        );
    }
}
