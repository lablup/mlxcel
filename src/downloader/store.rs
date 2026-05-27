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

/// Resolve the global store directory for a given HuggingFace `repo_id`.
///
/// Returns `store_root()/models/<owner>/<name>`. When `repo_id` has no `/`
/// separator (e.g. a bare `gpt2`), the whole id is used as the final path
/// component so the layout stays well-formed.
///
/// Returns `None` when [`store_root`] cannot be resolved (no
/// `MLXCEL_CACHE_DIR` and no home directory).
pub fn model_dir(repo_id: &str) -> Option<PathBuf> {
    let root = store_root()?;
    Some(model_dir_in(&root, repo_id))
}

/// Pure helper: compose `<root>/models/<owner>/<name>` from an explicit store
/// root. Split out so unit tests can assert the layout without depending on
/// process-wide env state.
///
/// `repo_id` segments are sanitized to stay inside the per-model directory:
/// only the `<owner>` and `<name>` components are honored, and any path
/// traversal (`.`, `..`, empty segments) or absolute/backslash shapes collapse
/// to a single safe `<name>` component. This mirrors the downloader's
/// `is_safe_relative_path` posture so an adversarial repo id cannot escape the
/// store root.
fn model_dir_in(root: &Path, repo_id: &str) -> PathBuf {
    let mut dir = root.join(MODELS_SUBDIR);
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
}
