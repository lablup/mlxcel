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

//! Binary-side handlers for local model management (epic #92, issue #97).
//!
//! Two surfaces are implemented here, both operating on the location-
//! independent global store introduced by issue #93
//! (`${MLXCEL_CACHE_DIR:-$HOME/.cache/mlxcel}/models/<owner>/<name>`):
//!
//! - **`mlxcel list --local`** — enumerates downloaded snapshots with repo-id,
//!   on-disk size, and path (mirrors `ollama list` / `lms ls`). The bare
//!   `mlxcel list` (architecture summary) is unchanged; `--local` switches the
//!   command to this store listing instead.
//! - **`mlxcel rm <repo-id>`** — removes a snapshot directory from the store
//!   (confirms unless `--yes`). It never touches the read-only HuggingFace
//!   cache: a repo that exists only there is reported, not deleted.
//!
//! The enumeration/deletion logic lives in [`mlxcel::downloader`] next to the
//! store-layout helpers it depends on; these handlers are thin I/O + prompting
//! shims so the store semantics stay in one place.

use std::io::{IsTerminal, Write};
use std::path::Path;

use anyhow::{Result, anyhow};

use mlxcel::downloader::{
    RemoveOutcome, StoredModel, dir_size, list_models_with_override, models_root,
    remove_model_with_override,
};

/// Run `mlxcel list --local`: print downloaded models from the global store.
///
/// `models_dir` is the inline `--models-dir <path>` override (issue #107):
/// when `Some`, the listing operates against that models root directly; when
/// `None`, it resolves `MLXCEL_MODELS_DIR` then the cache-root `models/` path.
pub(crate) fn run_list_local(models_dir: Option<&Path>) -> Result<()> {
    let models = list_models_with_override(models_dir);
    let mut out = String::new();
    render_local_models(&mut out, &models, store_root_display(models_dir).as_deref());
    print!("{out}");
    Ok(())
}

/// Resolve the active models root as a display string for the listing header,
/// or `None` when it cannot be resolved (no override, no `MLXCEL_MODELS_DIR`,
/// no `MLXCEL_CACHE_DIR` / home dir). Honors the `--models-dir` override.
fn store_root_display(models_dir: Option<&Path>) -> Option<String> {
    models_root(models_dir).map(|p| p.display().to_string())
}

/// Render the `mlxcel list --local` output into `out`.
///
/// Separated from [`run_list_local`] so unit tests can capture the exact bytes
/// without filesystem state. The format is intentionally simple and stable
/// (golden-test friendly): a header line, then one aligned row per model with
/// repo-id, human-readable size, and absolute path. An empty store prints a
/// short, actionable hint instead of an empty table.
fn render_local_models<W: std::fmt::Write>(
    out: &mut W,
    models: &[StoredModel],
    store_models_dir: Option<&str>,
) {
    if models.is_empty() {
        // Infallible: writing to a String / fmt buffer does not fail in
        // practice for our callers; ignore the Result to keep the signature
        // ergonomic.
        let _ = match store_models_dir {
            Some(dir) => writeln!(
                out,
                "No models downloaded in the mlxcel store ({dir}).\n\
                 Download one with: mlxcel download <owner>/<name>"
            ),
            None => writeln!(
                out,
                "No models downloaded (mlxcel store root is unavailable; \
                 set MLXCEL_MODELS_DIR or MLXCEL_CACHE_DIR, or pass --models-dir).\n\
                 Download one with: mlxcel download <owner>/<name>"
            ),
        };
        return;
    }

    let total: u64 = models.iter().map(|m| m.size_bytes).sum();
    let _ = match store_models_dir {
        Some(dir) => writeln!(
            out,
            "Downloaded models ({}, {} total) in {dir}:",
            models.len(),
            compact_size(total)
        ),
        None => writeln!(
            out,
            "Downloaded models ({}, {} total):",
            models.len(),
            compact_size(total)
        ),
    };
    let _ = writeln!(out);

    // Width of the repo-id column = longest id, so SIZE/PATH align. The size
    // column is right-aligned to a fixed width for tidy scanning.
    let id_width = models
        .iter()
        .map(|m| m.repo_id.len())
        .max()
        .unwrap_or(0)
        .max("REPO ID".len());
    let sizes: Vec<String> = models.iter().map(|m| compact_size(m.size_bytes)).collect();
    let size_width = sizes
        .iter()
        .map(String::len)
        .max()
        .unwrap_or(0)
        .max("SIZE".len());

    let _ = writeln!(
        out,
        "  {:<id_width$}  {:>size_width$}  PATH",
        "REPO ID", "SIZE"
    );
    for (model, size) in models.iter().zip(&sizes) {
        let _ = writeln!(
            out,
            "  {:<id_width$}  {:>size_width$}  {}",
            model.repo_id,
            size,
            model.path.display()
        );
    }
}

/// Run `mlxcel rm <repo-id>`: remove a model from the global store.
///
/// Confirms interactively unless `yes` is set. Refuses to delete anything in
/// the read-only HuggingFace cache (reports it instead). When stdin is not a
/// TTY and `--yes` was not passed, the command errors rather than silently
/// deleting or silently skipping — the operator must opt in explicitly.
pub(crate) fn run_remove(
    repo_id: &str,
    yes: bool,
    revision: Option<&str>,
    models_dir: Option<&Path>,
) -> Result<()> {
    // Probe the store first so we can show what will be removed (and its size)
    // before asking for confirmation, and so a not-found / HF-cache-only repo
    // never reaches a confirmation prompt. Honors the `--models-dir` override
    // (issue #107) so the probe and deletion target the same models root.
    let target =
        mlxcel::downloader::model_dir_with_override(repo_id, models_dir).ok_or_else(|| {
            anyhow!(
                "cannot resolve the mlxcel model store root \
             (set MLXCEL_MODELS_DIR or MLXCEL_CACHE_DIR, pass --models-dir, \
             or ensure a home directory is available)"
            )
        })?;

    if !target.is_dir() {
        // Not in the store. Distinguish HF-cache-only from truly absent by
        // letting the store helper do the probe (it is read-only).
        match remove_model_with_override(repo_id, revision, models_dir)? {
            RemoveOutcome::HfCacheOnly { hf_path } => {
                return Err(anyhow!(
                    "'{repo_id}' is not in the mlxcel store; it exists only in the \
                     read-only HuggingFace cache at {}.\nmlxcel does not manage the \
                     HuggingFace cache and will not delete it. Use the huggingface_hub \
                     tooling (e.g. `huggingface-cli delete-cache`) if you want to remove it.",
                    hf_path.display()
                ));
            }
            RemoveOutcome::NotFound => {
                return Err(anyhow!(
                    "'{repo_id}' is not in the mlxcel store (looked in {}).\n\
                     Run `mlxcel list --local` to see downloaded models.",
                    target.display()
                ));
            }
            // Unreachable: target.is_dir() was false, so the store branch in
            // remove_model cannot return Removed. Handle defensively.
            RemoveOutcome::Removed { path, size_bytes } => {
                println!(
                    "Removed '{repo_id}' ({}) from {}",
                    mlxcel::memory_estimate::format_bytes(size_bytes),
                    path.display()
                );
                return Ok(());
            }
        }
    }

    // Store hit. Confirm unless --yes.
    if !yes {
        let size = dir_size_for_prompt(&target);
        if !confirm_removal(repo_id, &target.display().to_string(), &size)? {
            println!("Aborted; nothing was removed.");
            return Ok(());
        }
    }

    match remove_model_with_override(repo_id, revision, models_dir)? {
        RemoveOutcome::Removed { path, size_bytes } => {
            println!(
                "Removed '{repo_id}' ({}) from {}",
                mlxcel::memory_estimate::format_bytes(size_bytes),
                path.display()
            );
            Ok(())
        }
        // The directory existed a moment ago; a concurrent delete is the only
        // way these arise. Report rather than pretend success.
        RemoveOutcome::HfCacheOnly { .. } | RemoveOutcome::NotFound => Err(anyhow!(
            "'{repo_id}' disappeared from the store before it could be removed \
             (concurrent deletion?)"
        )),
    }
}

/// Best-effort size string for the `rm` confirmation prompt. Sizes the single
/// target directory directly via the library's shared walk, rather than listing
/// and summing every model in the store: pointing `--models-dir` /
/// `MLXCEL_MODELS_DIR` at a large volume with many snapshots should not make the
/// prompt pay an O(whole-store) stat cost just to show one model's size.
fn dir_size_for_prompt(dir: &Path) -> String {
    mlxcel::memory_estimate::format_bytes(dir_size(dir))
}

/// Prompt on the controlling TTY for a yes/no confirmation. Returns `Ok(true)`
/// only on an explicit affirmative. Errors (rather than defaulting either way)
/// when stdin is not interactive, so scripted callers must pass `--yes`.
fn confirm_removal(repo_id: &str, path: &str, size: &str) -> Result<bool> {
    if !std::io::stdin().is_terminal() {
        return Err(anyhow!(
            "refusing to remove '{repo_id}' ({size}) at {path} without confirmation: \
             stdin is not a TTY. Re-run with --yes to confirm non-interactively."
        ));
    }
    print!("Remove '{repo_id}' ({size}) at {path}? [y/N] ");
    std::io::stdout().flush().ok();
    let mut answer = String::new();
    std::io::stdin()
        .read_line(&mut answer)
        .map_err(|e| anyhow!("failed to read confirmation: {e}"))?;
    let answer = answer.trim().to_ascii_lowercase();
    Ok(answer == "y" || answer == "yes")
}

/// Compact human-readable size for table cells, e.g. `2.34 GiB`, `512.0 MiB`,
/// `48.0 KiB`, `12 B`. Distinct from [`mlxcel::memory_estimate::format_bytes`]
/// (which appends the exact byte count) so the listing columns stay narrow.
fn compact_size(bytes: u64) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = 1024.0 * 1024.0;
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    let b = bytes as f64;
    if b >= GIB {
        format!("{:.2} GiB", b / GIB)
    } else if b >= MIB {
        format!("{:.1} MiB", b / MIB)
    } else if b >= KIB {
        format!("{:.1} KiB", b / KIB)
    } else {
        format!("{bytes} B")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn model(repo_id: &str, path: &str, size_bytes: u64) -> StoredModel {
        StoredModel {
            repo_id: repo_id.to_string(),
            path: PathBuf::from(path),
            size_bytes,
        }
    }

    #[test]
    fn compact_size_picks_units() {
        assert_eq!(compact_size(0), "0 B");
        assert_eq!(compact_size(512), "512 B");
        assert_eq!(compact_size(2 * 1024), "2.0 KiB");
        assert_eq!(compact_size(5 * 1024 * 1024), "5.0 MiB");
        assert_eq!(compact_size(3 * 1024 * 1024 * 1024), "3.00 GiB");
    }

    #[test]
    fn render_empty_store_prints_hint() {
        let mut out = String::new();
        render_local_models(&mut out, &[], Some("/store/models"));
        assert!(out.contains("No models downloaded"));
        assert!(out.contains("/store/models"));
        assert!(out.contains("mlxcel download"));
    }

    #[test]
    fn render_empty_store_without_root() {
        let mut out = String::new();
        render_local_models(&mut out, &[], None);
        assert!(out.contains("No models downloaded"));
        assert!(out.contains("MLXCEL_CACHE_DIR"));
    }

    #[test]
    fn render_lists_models_with_size_and_path() {
        let models = vec![
            model(
                "mlx-community/Qwen3-4B-4bit",
                "/store/models/mlx-community/Qwen3-4B-4bit",
                3 * 1024 * 1024 * 1024,
            ),
            model("gpt2", "/store/models/gpt2", 500 * 1024 * 1024),
        ];
        let mut out = String::new();
        render_local_models(&mut out, &models, Some("/store/models"));

        // Header reports count and total.
        assert!(out.contains("Downloaded models (2,"));
        assert!(out.contains("/store/models"));
        // Column header present.
        assert!(out.contains("REPO ID"));
        assert!(out.contains("SIZE"));
        assert!(out.contains("PATH"));
        // Each model's id, a size cell, and its path appear.
        assert!(out.contains("mlx-community/Qwen3-4B-4bit"));
        assert!(out.contains("3.00 GiB"));
        assert!(out.contains("/store/models/mlx-community/Qwen3-4B-4bit"));
        assert!(out.contains("gpt2"));
        assert!(out.contains("500.0 MiB"));
    }

    #[test]
    fn render_aligns_repo_id_column() {
        // The shorter id row must be left-padded to the longest id width so the
        // SIZE column lines up. We assert the long-id and short-id rows share
        // the same byte offset for their size cell start.
        let models = vec![
            model("a/b", "/s/models/a/b", 1024),
            model(
                "very-long-owner/very-long-model-name",
                "/s/models/very-long-owner/very-long-model-name",
                2048,
            ),
        ];
        let mut out = String::new();
        render_local_models(&mut out, &models, Some("/s/models"));
        // Find the data rows (skip header/title/blank). Both should contain a
        // double-space-separated SIZE column at the same column index.
        let rows: Vec<&str> = out.lines().filter(|l| l.contains("/s/models/")).collect();
        assert_eq!(rows.len(), 2, "expected two data rows, got: {out:?}");
        // The path substring should start at the same column in both rows.
        let col_a = rows[0].find("/s/models/").unwrap();
        let col_b = rows[1].find("/s/models/").unwrap();
        assert_eq!(col_a, col_b, "PATH column misaligned:\n{out}");
    }
}
