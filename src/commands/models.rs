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
//! - **`mlxcel list`**: enumerates downloaded snapshots with repo-id,
//!   on-disk size, and path (mirrors `ollama list` / `lms ls`). The supported
//!   model-architecture catalog lives under the separate `mlxcel arch` verb.
//! - **`mlxcel rm <repo-id>`**: removes a snapshot directory from the store
//!   (confirms unless `--yes`). It never touches the read-only HuggingFace
//!   cache: a repo that exists only there is reported, not deleted.
//!
//! The enumeration/deletion logic lives in [`mlxcel::downloader`] next to the
//! store-layout helpers it depends on; these handlers are thin I/O + prompting
//! shims so the store semantics stay in one place.

use std::io::{IsTerminal, Write};
use std::path::Path;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Result, anyhow};
use serde::Serialize;

use mlxcel::downloader::{
    RemoveOutcome, StoredModel, dir_size, list_models_with_override, models_root,
    remove_model_with_override,
};

/// Maximum rendered width of the `NAME` column in the default/verbose tables.
///
/// Repo-ids longer than this are truncated with a `…` so a single pathological
/// id (e.g. a very long fine-tune name) cannot blow out column alignment for
/// every other row. Chosen to comfortably fit common `<owner>/<name>` ids.
const NAME_COL_MAX: usize = 40;

/// ANSI "dim" SGR prefix and reset, applied only to secondary cells (MODIFIED
/// column, header store root) and only when color is enabled.
const DIM: &str = "\x1b[2m";
const RESET: &str = "\x1b[0m";

/// Sort order for `mlxcel list` (`--sort`).
///
/// Applies to both the table and `--json` output, before rendering.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum SortKey {
    /// Repo-id ascending (the historical default).
    #[default]
    Name,
    /// Largest on-disk size first.
    Size,
    /// Most-recently-modified first; entries with no mtime sort last.
    Modified,
}

/// Resolved options for a single `mlxcel list` invocation.
///
/// Built by [`run_list_local`] from the parsed CLI flags. `now` and `use_color`
/// are injected here (not read inside the renderers) so the rendering functions
/// stay deterministic and filesystem-free for golden tests.
pub(crate) struct ListOptions {
    /// `--json`: emit a stable JSON array instead of the table.
    pub json: bool,
    /// `-q/--quiet`: emit only repo-ids, one per line.
    pub quiet: bool,
    /// `-v/--verbose`: append the absolute `PATH` column to the table.
    pub verbose: bool,
    /// Reference instant for relative `MODIFIED` rendering (injected so tests
    /// are deterministic). Ignored by `--json` (which emits absolute epochs).
    pub now: SystemTime,
    /// Whether to emit ANSI dim styling on secondary table cells. Never set for
    /// `--json` / `-q`.
    pub use_color: bool,
}

/// Run `mlxcel list`: print downloaded models from the global store.
///
/// `models_dir` is the inline `--models-dir <path>` override (issue #107):
/// when `Some`, the listing operates against that models root directly; when
/// `None`, it resolves `MLXCEL_MODELS_DIR` then the cache-root `models/` path.
///
/// `now` (`SystemTime::now()`) and `use_color` are computed here and threaded
/// into the renderer through [`ListOptions`], keeping the rendering functions
/// pure and testable.
pub(crate) fn run_list_local(models_dir: Option<&Path>, opts: &crate::ListArgs) -> Result<()> {
    let mut models = list_models_with_override(models_dir);
    sort_models(&mut models, opts.sort);

    // Styling is irrelevant to machine/quiet output and must never be emitted
    // there; only the human table consults `use_color`.
    let use_color = !opts.json && !opts.quiet && should_color_stdout();
    let options = ListOptions {
        json: opts.json,
        quiet: opts.quiet,
        verbose: opts.verbose,
        now: SystemTime::now(),
        use_color,
    };

    let mut out = String::new();
    if options.json {
        render_json(&mut out, &models);
    } else if options.quiet {
        render_quiet(&mut out, &models);
    } else {
        render_local_models(
            &mut out,
            &models,
            store_root_display(models_dir).as_deref(),
            &options,
        );
    }
    print!("{out}");
    Ok(())
}

/// Sort `models` in place per the requested [`SortKey`].
///
/// - `Name`: repo-id ascending (stable, matches the pre-existing default order).
/// - `Size`: largest on-disk size first; ties fall back to repo-id ascending.
/// - `Modified`: most-recent first, with `None` (unknown mtime) sorted last;
///   ties fall back to repo-id ascending.
fn sort_models(models: &mut [StoredModel], sort: SortKey) {
    match sort {
        SortKey::Name => models.sort_by(|a, b| a.repo_id.cmp(&b.repo_id)),
        SortKey::Size => models.sort_by(|a, b| {
            b.size_bytes
                .cmp(&a.size_bytes)
                .then_with(|| a.repo_id.cmp(&b.repo_id))
        }),
        SortKey::Modified => models.sort_by(|a, b| {
            // Most-recent first => reverse the time comparison. `None` must sort
            // last regardless of direction, so map to a key where present beats
            // absent and a later instant beats an earlier one.
            match (a.modified, b.modified) {
                (Some(ta), Some(tb)) => tb.cmp(&ta).then_with(|| a.repo_id.cmp(&b.repo_id)),
                (Some(_), None) => std::cmp::Ordering::Less,
                (None, Some(_)) => std::cmp::Ordering::Greater,
                (None, None) => a.repo_id.cmp(&b.repo_id),
            }
        }),
    }
}

/// Resolve the active models root as a display string for the listing header,
/// or `None` when it cannot be resolved (no override, no `MLXCEL_MODELS_DIR`,
/// no `MLXCEL_CACHE_DIR` / home dir). Honors the `--models-dir` override.
///
/// The returned string has any `$HOME` prefix contracted to `~`
/// (see [`contract_home`]) for a compact header.
fn store_root_display(models_dir: Option<&Path>) -> Option<String> {
    models_root(models_dir).map(|p| contract_home(&p.display().to_string()))
}

/// Contract a leading `$HOME` in `path` to `~` (e.g.
/// `/Users/me/.cache/mlxcel/models` → `~/.cache/mlxcel/models`).
///
/// Only an exact home-dir prefix immediately followed by a path separator (or
/// the end of the string) is contracted, so `/home/meadow` is not mangled into
/// `~adow` for a `/home/me` home. Falls back to the unchanged path when the
/// home directory cannot be resolved or is not a prefix. Reuses the same
/// [`dirs::home_dir`] resolution the downloader uses; adds no new dependency.
fn contract_home(path: &str) -> String {
    let Some(home) = dirs::home_dir() else {
        return path.to_string();
    };
    let home = home.display().to_string();
    if home.is_empty() {
        return path.to_string();
    }
    if let Some(rest) = path.strip_prefix(&home) {
        if rest.is_empty() {
            return "~".to_string();
        }
        if rest.starts_with(std::path::MAIN_SEPARATOR) {
            return format!("~{rest}");
        }
    }
    path.to_string()
}

/// Mirror of [`mlxcel::downloader::should_show_progress`] but gated on
/// **stdout** (the listing's output stream) rather than stderr.
///
/// Color is suppressed when ANY of the following holds (checked in order):
/// 1. `NO_COLOR` is set to a non-empty value (de-facto color-suppression std).
/// 2. `stdout` is not a TTY (piped, redirected).
///
/// Unlike the progress gate this intentionally omits the `MLXCEL_NO_PROGRESS`
/// and `CI` checks: those concern progress bars on stderr, not list styling.
fn should_color_stdout() -> bool {
    if matches!(std::env::var("NO_COLOR"), Ok(val) if !val.trim().is_empty()) {
        return false;
    }
    std::io::stdout().is_terminal()
}

/// Map an elapsed [`Duration`] to a compact, human-friendly relative label
/// (`just now`, `5 min ago`, `2 days ago`, `3 weeks ago`, …).
///
/// Pure (takes the already-computed elapsed duration, never reads the clock) so
/// callers (and unit tests) control the reference instant. The caller passes
/// `now.duration_since(modified)`; a future `modified` (clock skew → `Err`)
/// should be rendered by the caller as "just now" (see [`format_modified`]).
///
/// Thresholds (each unit floored): `< 60s` → `just now`; `< 60 min` → minutes;
/// `< 24 h` → hours; `< 7 d` → days; `< 28 d` (4 weeks) → weeks; `< 365 d` →
/// months (30-day months); otherwise years (365-day years). Units are
/// pluralized (`1 day ago` vs `2 days ago`).
fn humanize_relative(elapsed: Duration) -> String {
    const MIN: u64 = 60;
    const HOUR: u64 = 60 * MIN;
    const DAY: u64 = 24 * HOUR;
    const WEEK: u64 = 7 * DAY;
    const MONTH: u64 = 30 * DAY;
    const YEAR: u64 = 365 * DAY;

    let secs = elapsed.as_secs();
    if secs < MIN {
        return "just now".to_string();
    }
    let (value, unit) = if secs < HOUR {
        (secs / MIN, "min")
    } else if secs < DAY {
        (secs / HOUR, "hour")
    } else if secs < WEEK {
        (secs / DAY, "day")
    } else if secs < 4 * WEEK {
        (secs / WEEK, "week")
    } else if secs < YEAR {
        // The weeks cutoff (28d / 4 weeks) sits just below one 30-day month, so
        // `secs / MONTH` floors to 0 between 28d and 30d. Clamp to at least 1 so
        // the bucket never reads "0 months ago".
        ((secs / MONTH).max(1), "month")
    } else {
        (secs / YEAR, "year")
    };
    // "min" is rendered as a fixed abbreviation (no plural); the rest pluralize.
    if unit == "min" || value == 1 {
        format!("{value} {unit} ago")
    } else {
        format!("{value} {unit}s ago")
    }
}

/// Render a single model's `MODIFIED` cell from its stored mtime relative to
/// `now`. `None` → `-`; a future mtime (negative elapsed) → `just now`.
fn format_modified(modified: Option<SystemTime>, now: SystemTime) -> String {
    match modified {
        Some(t) => match now.duration_since(t) {
            Ok(elapsed) => humanize_relative(elapsed),
            // `modified` is in the future relative to `now` (clock skew): treat
            // as just-modified rather than rendering a negative duration.
            Err(_) => "just now".to_string(),
        },
        None => "-".to_string(),
    }
}

/// Truncate `name` to [`NAME_COL_MAX`] display chars, appending `…` when it
/// overflows. Counts `char`s (not bytes) so multibyte ids are not split mid-
/// codepoint. Returns the original string when it already fits.
fn ellipsize_name(name: &str) -> String {
    if name.chars().count() <= NAME_COL_MAX {
        return name.to_string();
    }
    // Reserve one column for the ellipsis.
    let keep = NAME_COL_MAX.saturating_sub(1);
    let truncated: String = name.chars().take(keep).collect();
    format!("{truncated}…")
}

/// Wrap `cell` in ANSI dim when `use_color`, else return it unstyled.
fn dim(cell: &str, use_color: bool) -> String {
    if use_color {
        format!("{DIM}{cell}{RESET}")
    } else {
        cell.to_string()
    }
}

/// Render the default/verbose `mlxcel list` table into `out`.
///
/// Columns: `NAME` / `SIZE` / `MODIFIED`, plus `PATH` when `opts.verbose`.
/// NAME is left-aligned and capped at [`NAME_COL_MAX`] (ellipsized); SIZE is
/// right-aligned; MODIFIED is the relative-time label. The header is
/// `N model(s) · <total> · <root>` with the store root `$HOME`-contracted.
///
/// Pure aside from the injected `opts.now` / `opts.use_color`: it reads no clock
/// and no environment, so golden tests pin both. An empty store prints the
/// short actionable hint (preserved from #138) instead of an empty table.
fn render_local_models<W: std::fmt::Write>(
    out: &mut W,
    models: &[StoredModel],
    store_models_dir: Option<&str>,
    opts: &ListOptions,
) {
    if models.is_empty() {
        // Infallible: writing to a String / fmt buffer does not fail in
        // practice for our callers; ignore the Result to keep the signature
        // ergonomic.
        let _ = match store_models_dir {
            Some(dir) => writeln!(
                out,
                "No models downloaded in the mlxcel store ({dir}).\n\
                 Download one with: mlxcel download <owner>/<name>\n\
                 To see supported architectures, run `mlxcel arch`."
            ),
            None => writeln!(
                out,
                "No models downloaded (mlxcel store root is unavailable; \
                 set MLXCEL_MODELS_DIR or MLXCEL_CACHE_DIR, or pass --models-dir).\n\
                 Download one with: mlxcel download <owner>/<name>\n\
                 To see supported architectures, run `mlxcel arch`."
            ),
        };
        return;
    }

    // Header: `N model(s) · <total> · <root>` (root is already $HOME-contracted
    // by the caller and is dimmed when styling is on).
    let total: u64 = models.iter().map(|m| m.size_bytes).sum();
    let noun = if models.len() == 1 { "model" } else { "models" };
    let count_size = format!("{} {noun} · {}", models.len(), compact_size(total));
    let _ = match store_models_dir {
        Some(dir) => writeln!(out, "{count_size} · {}", dim(dir, opts.use_color)),
        None => writeln!(out, "{count_size}"),
    };
    let _ = writeln!(out);

    // Pre-render every cell so column widths account for ellipsized names and
    // formatted sizes/times. NAME width is capped at NAME_COL_MAX.
    let names: Vec<String> = models.iter().map(|m| ellipsize_name(&m.repo_id)).collect();
    let sizes: Vec<String> = models.iter().map(|m| compact_size(m.size_bytes)).collect();
    let modifieds: Vec<String> = models
        .iter()
        .map(|m| format_modified(m.modified, opts.now))
        .collect();

    let name_width = names
        .iter()
        .map(|n| n.chars().count())
        .max()
        .unwrap_or(0)
        .max("NAME".len())
        .min(NAME_COL_MAX);
    let size_width = sizes
        .iter()
        .map(String::len)
        .max()
        .unwrap_or(0)
        .max("SIZE".len());
    let modified_width = modifieds
        .iter()
        .map(String::len)
        .max()
        .unwrap_or(0)
        .max("MODIFIED".len());

    // Header row. MODIFIED is dimmed when styling is on; PATH is appended only
    // in verbose mode. The MODIFIED header is padded *before* dimming so the
    // ANSI codes do not count toward column width.
    if opts.verbose {
        let _ = writeln!(
            out,
            "  {:<name_width$}  {:>size_width$}  {}  PATH",
            "NAME",
            "SIZE",
            dim(&format!("{:<modified_width$}", "MODIFIED"), opts.use_color),
        );
        for (((name, size), modified), model) in
            names.iter().zip(&sizes).zip(&modifieds).zip(models)
        {
            let _ = writeln!(
                out,
                "  {:<name_width$}  {:>size_width$}  {}  {}",
                name,
                size,
                dim(&format!("{modified:<modified_width$}"), opts.use_color),
                model.path.display(),
            );
        }
    } else {
        let _ = writeln!(
            out,
            "  {:<name_width$}  {:>size_width$}  {}",
            "NAME",
            "SIZE",
            dim("MODIFIED", opts.use_color),
        );
        for ((name, size), modified) in names.iter().zip(&sizes).zip(&modifieds) {
            let _ = writeln!(
                out,
                "  {:<name_width$}  {:>size_width$}  {}",
                name,
                size,
                dim(modified, opts.use_color),
            );
        }
    }
}

/// One `--json` array element. Dedicated `Serialize` struct (rather than
/// serializing [`StoredModel`] directly) so the wire field names stay stable
/// regardless of internal renames: `{repo_id, size_bytes, path, modified}`.
#[derive(Serialize)]
struct JsonModel<'a> {
    repo_id: &'a str,
    size_bytes: u64,
    path: String,
    /// Unix epoch **seconds** of the snapshot mtime, or `null` when unavailable.
    modified: Option<u64>,
}

/// Render `models` as a pretty-printed JSON array (no header, no ANSI) for
/// scripting. `modified` is emitted as `Option<u64>` Unix epoch seconds
/// (`null` when absent or pre-epoch). An empty store yields `[]`.
fn render_json<W: std::fmt::Write>(out: &mut W, models: &[StoredModel]) {
    let rows: Vec<JsonModel<'_>> = models
        .iter()
        .map(|m| JsonModel {
            repo_id: &m.repo_id,
            size_bytes: m.size_bytes,
            path: m.path.display().to_string(),
            modified: m
                .modified
                .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                .map(|d| d.as_secs()),
        })
        .collect();
    // serde_json over a Vec of owned/borrowed scalars cannot fail here; fall
    // back to an empty array on the impossible error rather than panicking.
    let json = serde_json::to_string_pretty(&rows).unwrap_or_else(|_| "[]".to_string());
    let _ = writeln!(out, "{json}");
}

/// Render just the repo-ids, one per line (`-q/--quiet`): no header, no
/// columns, no ANSI, so `mlxcel list -q | xargs -n1 mlxcel rm` works. An empty
/// store emits nothing.
fn render_quiet<W: std::fmt::Write>(out: &mut W, models: &[StoredModel]) {
    for m in models {
        let _ = writeln!(out, "{}", m.repo_id);
    }
}

/// Run `mlxcel rm <repo-id>`: remove a model from the global store.
///
/// Confirms interactively unless `yes` is set. Refuses to delete anything in
/// the read-only HuggingFace cache (reports it instead). When stdin is not a
/// TTY and `--yes` was not passed, the command errors rather than silently
/// deleting or silently skipping: the operator must opt in explicitly.
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
                     Run `mlxcel list` to see downloaded models.",
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

    /// A fixed reference instant for deterministic relative-time rendering:
    /// 1_700_000_000s after the Unix epoch (2023-11-14T22:13:20Z). Tests derive
    /// `modified` instants relative to this so output never depends on the wall
    /// clock or the filesystem.
    fn fixed_now() -> SystemTime {
        UNIX_EPOCH + Duration::from_secs(1_700_000_000)
    }

    /// Build a `StoredModel` with no known mtime (renders MODIFIED as `-`).
    fn model(repo_id: &str, path: &str, size_bytes: u64) -> StoredModel {
        StoredModel {
            repo_id: repo_id.to_string(),
            path: PathBuf::from(path),
            size_bytes,
            modified: None,
        }
    }

    /// Build a `StoredModel` modified `secs_ago` seconds before [`fixed_now`].
    fn model_at(repo_id: &str, path: &str, size_bytes: u64, secs_ago: u64) -> StoredModel {
        StoredModel {
            repo_id: repo_id.to_string(),
            path: PathBuf::from(path),
            size_bytes,
            modified: Some(fixed_now() - Duration::from_secs(secs_ago)),
        }
    }

    /// Default-table options (no json/quiet/verbose, no color, fixed `now`).
    fn opts_default() -> ListOptions {
        ListOptions {
            json: false,
            quiet: false,
            verbose: false,
            now: fixed_now(),
            use_color: false,
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
        render_local_models(&mut out, &[], Some("/store/models"), &opts_default());
        assert!(out.contains("No models downloaded"));
        assert!(out.contains("/store/models"));
        assert!(out.contains("mlxcel download"));
        assert!(out.contains("mlxcel arch"));
    }

    #[test]
    fn render_empty_store_without_root() {
        let mut out = String::new();
        render_local_models(&mut out, &[], None, &opts_default());
        assert!(out.contains("No models downloaded"));
        assert!(out.contains("MLXCEL_CACHE_DIR"));
        assert!(out.contains("mlxcel arch"));
    }

    #[test]
    fn render_default_shows_name_size_modified_no_path() {
        // Default table: NAME / SIZE / MODIFIED, no PATH column or paths.
        let models = vec![
            model_at(
                "mlx-community/Qwen3-4B-4bit",
                "/store/models/mlx-community/Qwen3-4B-4bit",
                3 * 1024 * 1024 * 1024,
                2 * 24 * 3600, // 2 days ago
            ),
            model_at(
                "gpt2",
                "/store/models/gpt2",
                500 * 1024 * 1024,
                21 * 24 * 3600,
            ), // 3 weeks
        ];
        let mut out = String::new();
        render_local_models(
            &mut out,
            &models,
            Some("~/.cache/mlxcel/models"),
            &opts_default(),
        );

        // Header: `N models · <total> · <root>`; PATH absent everywhere.
        assert!(out.contains("2 models · "), "header missing count: {out}");
        assert!(
            out.contains("~/.cache/mlxcel/models"),
            "header missing root: {out}"
        );
        assert!(
            !out.contains("PATH"),
            "default table must not show PATH: {out}"
        );
        assert!(
            !out.contains("/store/models/"),
            "default table must not show absolute paths: {out}"
        );
        // New column headers.
        assert!(out.contains("NAME"));
        assert!(out.contains("SIZE"));
        assert!(out.contains("MODIFIED"));
        // Cells: ids, sizes, relative times.
        assert!(out.contains("mlx-community/Qwen3-4B-4bit"));
        assert!(out.contains("3.00 GiB"));
        assert!(out.contains("2 days ago"));
        assert!(out.contains("gpt2"));
        assert!(out.contains("500.0 MiB"));
        assert!(out.contains("3 weeks ago"));
    }

    #[test]
    fn render_singular_header_noun() {
        let models = vec![model("only/one", "/s/models/only/one", 1024)];
        let mut out = String::new();
        render_local_models(&mut out, &models, Some("/s/models"), &opts_default());
        assert!(out.contains("1 model · "), "expected singular noun: {out}");
        assert!(
            !out.contains("1 models"),
            "should not pluralize for one: {out}"
        );
    }

    #[test]
    fn render_missing_mtime_is_dash() {
        let models = vec![model("a/b", "/s/models/a/b", 1024)];
        let mut out = String::new();
        render_local_models(&mut out, &models, Some("/s/models"), &opts_default());
        // MODIFIED cell renders `-` for a None mtime.
        let row = out
            .lines()
            .find(|l| l.contains("a/b"))
            .expect("data row present");
        assert!(
            row.trim_end().ends_with('-'),
            "expected dash mtime: {row:?}"
        );
    }

    #[test]
    fn render_verbose_restores_path_column() {
        let models = vec![model_at(
            "mlx-community/Qwen3-4B-4bit",
            "/store/models/mlx-community/Qwen3-4B-4bit",
            3 * 1024 * 1024 * 1024,
            3600,
        )];
        let mut opts = opts_default();
        opts.verbose = true;
        let mut out = String::new();
        render_local_models(&mut out, &models, Some("/store/models"), &opts);

        assert!(out.contains("PATH"), "verbose must show PATH header: {out}");
        assert!(
            out.contains("/store/models/mlx-community/Qwen3-4B-4bit"),
            "verbose must show absolute path: {out}"
        );
        // Still has the other columns.
        assert!(out.contains("NAME"));
        assert!(out.contains("MODIFIED"));
        assert!(out.contains("1 hour ago"));
    }

    #[test]
    fn render_aligns_name_column() {
        // The shorter id row must be padded to the longest id width so SIZE
        // lines up. We assert both rows' SIZE cell starts at the same column.
        let models = vec![
            model_at("a/b", "/s/models/a/b", 1024, 60),
            model_at(
                "very-long-owner/very-long-model-name",
                "/s/models/very-long-owner/very-long-model-name",
                2048,
                60,
            ),
        ];
        let mut out = String::new();
        render_local_models(&mut out, &models, Some("/s/models"), &opts_default());
        let rows: Vec<&str> = out
            .lines()
            .filter(|l| l.starts_with("  ") && (l.contains("a/b") || l.contains("very-long")))
            .collect();
        assert_eq!(rows.len(), 2, "expected two data rows, got: {out:?}");
        // The size token "1.0 KiB" / "2.0 KiB" should start at the same column.
        let col_a = rows[0].find("KiB").unwrap();
        let col_b = rows[1].find("KiB").unwrap();
        assert_eq!(col_a, col_b, "SIZE column misaligned:\n{out}");
    }

    #[test]
    fn render_ellipsizes_long_name() {
        let long_id = format!("owner/{}", "x".repeat(80));
        let models = vec![model_at(&long_id, "/s/models/owner/x", 1024, 60)];
        let mut out = String::new();
        render_local_models(&mut out, &models, Some("/s/models"), &opts_default());
        assert!(out.contains('…'), "long id should be ellipsized: {out}");
        // The full 86-char id must NOT appear verbatim.
        assert!(!out.contains(&long_id), "long id should be capped: {out}");
    }

    #[test]
    fn ellipsize_name_caps_width() {
        let short = "mlx-community/Qwen3-4B-4bit";
        assert_eq!(ellipsize_name(short), short, "short ids pass through");
        let long = "a".repeat(100);
        let out = ellipsize_name(&long);
        assert_eq!(out.chars().count(), NAME_COL_MAX);
        assert!(out.ends_with('…'));
    }

    // ── --json ───────────────────────────────────────────────────────────────

    #[test]
    fn render_json_is_parseable_and_stable() {
        let models = vec![
            model_at("a/b", "/s/models/a/b", 1024, 0),
            model("c/d", "/s/models/c/d", 2048), // no mtime
        ];
        let mut out = String::new();
        render_json(&mut out, &models);

        // No table furniture or ANSI.
        assert!(!out.contains("NAME"));
        assert!(!out.contains('\u{1b}'));

        let parsed: serde_json::Value = serde_json::from_str(&out).expect("valid json");
        let arr = parsed.as_array().expect("array");
        assert_eq!(arr.len(), 2);

        // Stable field names.
        assert_eq!(arr[0]["repo_id"], "a/b");
        assert_eq!(arr[0]["size_bytes"], 1024);
        assert_eq!(arr[0]["path"], "/s/models/a/b");
        // modified at fixed_now() - 0s => epoch seconds present (numeric).
        assert!(arr[0]["modified"].is_number(), "modified should be numeric");
        assert_eq!(arr[0]["modified"], 1_700_000_000u64);

        // None mtime serializes as JSON null.
        assert_eq!(arr[1]["repo_id"], "c/d");
        assert!(arr[1]["modified"].is_null(), "missing mtime should be null");
    }

    #[test]
    fn render_json_empty_store_is_empty_array() {
        let mut out = String::new();
        render_json(&mut out, &[]);
        let parsed: serde_json::Value = serde_json::from_str(&out).expect("valid json");
        assert_eq!(parsed.as_array().expect("array").len(), 0);
    }

    // ── -q / --quiet ───────────────────────────────────────────────────────────

    #[test]
    fn render_quiet_emits_repo_ids_only() {
        let models = vec![
            model_at("mlx-community/Qwen3-4B-4bit", "/s/m/q", 1024, 60),
            model("gpt2", "/s/m/gpt2", 2048),
        ];
        let mut out = String::new();
        render_quiet(&mut out, &models);
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines, vec!["mlx-community/Qwen3-4B-4bit", "gpt2"]);
        // No header, no sizes, no ANSI.
        assert!(!out.contains("NAME"));
        assert!(!out.contains("KiB"));
        assert!(!out.contains('\u{1b}'));
    }

    #[test]
    fn render_quiet_empty_store_emits_nothing() {
        let mut out = String::new();
        render_quiet(&mut out, &[]);
        assert!(out.is_empty());
    }

    // ── styling ────────────────────────────────────────────────────────────────

    #[test]
    fn render_without_color_has_no_ansi() {
        let models = vec![model_at("a/b", "/s/models/a/b", 1024, 60)];
        let mut out = String::new();
        // use_color defaults to false in opts_default().
        render_local_models(&mut out, &models, Some("/s/models"), &opts_default());
        assert!(
            !out.contains('\u{1b}'),
            "no ANSI escape with use_color=false: {out:?}"
        );
    }

    #[test]
    fn render_with_color_dims_modified_and_root() {
        let models = vec![model_at("a/b", "/s/models/a/b", 1024, 60)];
        let mut opts = opts_default();
        opts.use_color = true;
        let mut out = String::new();
        render_local_models(&mut out, &models, Some("/s/models"), &opts);
        // Dim sequence appears (around the root and MODIFIED cells); NAME plain.
        assert!(
            out.contains(DIM),
            "expected dim escape with color on: {out:?}"
        );
        assert!(out.contains(RESET));
        // The repo-id itself should not be wrapped in a dim/reset pair adjacent
        // to it (NAME stays plain): the id appears without an immediately
        // preceding DIM.
        let id_pos = out.find("a/b").unwrap();
        assert!(
            !out[..id_pos].ends_with(DIM),
            "NAME column must not be dimmed: {out:?}"
        );
    }

    // ── sorting ────────────────────────────────────────────────────────────────

    #[test]
    fn sort_by_name_ascending() {
        let mut models = vec![
            model("c/c", "/s/c", 10),
            model("a/a", "/s/a", 30),
            model("b/b", "/s/b", 20),
        ];
        sort_models(&mut models, SortKey::Name);
        let ids: Vec<&str> = models.iter().map(|m| m.repo_id.as_str()).collect();
        assert_eq!(ids, vec!["a/a", "b/b", "c/c"]);
    }

    #[test]
    fn sort_by_size_largest_first() {
        let mut models = vec![
            model("a/a", "/s/a", 10),
            model("b/b", "/s/b", 30),
            model("c/c", "/s/c", 20),
        ];
        sort_models(&mut models, SortKey::Size);
        let ids: Vec<&str> = models.iter().map(|m| m.repo_id.as_str()).collect();
        assert_eq!(ids, vec!["b/b", "c/c", "a/a"]);
    }

    #[test]
    fn sort_by_modified_recent_first_none_last() {
        // `model_at` sets modified = fixed_now() - secs_ago, so a larger
        // `secs_ago` is older. `model(...)` (no mtime) must sort last.
        let mut models = vec![
            model_at("old", "/s/old", 1, 1000),
            model("none", "/s/none", 1),
            model_at("new", "/s/new", 1, 10),
        ];
        sort_models(&mut models, SortKey::Modified);
        let ids: Vec<&str> = models.iter().map(|m| m.repo_id.as_str()).collect();
        assert_eq!(ids, vec!["new", "old", "none"], "recent first, None last");
    }

    // ── humanize_relative boundaries ─────────────────────────────────────────────

    #[test]
    fn humanize_just_now_under_a_minute() {
        assert_eq!(humanize_relative(Duration::from_secs(0)), "just now");
        assert_eq!(humanize_relative(Duration::from_secs(59)), "just now");
    }

    #[test]
    fn humanize_minutes() {
        assert_eq!(humanize_relative(Duration::from_secs(60)), "1 min ago");
        assert_eq!(humanize_relative(Duration::from_secs(5 * 60)), "5 min ago");
        assert_eq!(
            humanize_relative(Duration::from_secs(59 * 60 + 59)),
            "59 min ago"
        );
    }

    #[test]
    fn humanize_hours() {
        assert_eq!(humanize_relative(Duration::from_secs(3600)), "1 hour ago");
        assert_eq!(
            humanize_relative(Duration::from_secs(2 * 3600)),
            "2 hours ago"
        );
        assert_eq!(
            humanize_relative(Duration::from_secs(23 * 3600 + 59 * 60)),
            "23 hours ago"
        );
    }

    #[test]
    fn humanize_days() {
        assert_eq!(
            humanize_relative(Duration::from_secs(24 * 3600)),
            "1 day ago"
        );
        assert_eq!(
            humanize_relative(Duration::from_secs(2 * 24 * 3600)),
            "2 days ago"
        );
        assert_eq!(
            humanize_relative(Duration::from_secs(6 * 24 * 3600)),
            "6 days ago"
        );
    }

    #[test]
    fn humanize_weeks() {
        assert_eq!(
            humanize_relative(Duration::from_secs(7 * 24 * 3600)),
            "1 week ago"
        );
        assert_eq!(
            humanize_relative(Duration::from_secs(21 * 24 * 3600)),
            "3 weeks ago"
        );
        // 27 days is still < 4 weeks (28d) => 3 weeks.
        assert_eq!(
            humanize_relative(Duration::from_secs(27 * 24 * 3600)),
            "3 weeks ago"
        );
    }

    #[test]
    fn humanize_months_and_years() {
        // 28 days (the weeks cutoff) lands in the months bucket; the clamp keeps
        // it at "1 month ago" rather than "0 months ago".
        assert_eq!(
            humanize_relative(Duration::from_secs(28 * 24 * 3600)),
            "1 month ago"
        );
        assert_eq!(
            humanize_relative(Duration::from_secs(60 * 24 * 3600)),
            "2 months ago"
        );
        assert_eq!(
            humanize_relative(Duration::from_secs(365 * 24 * 3600)),
            "1 year ago"
        );
        assert_eq!(
            humanize_relative(Duration::from_secs(800 * 24 * 3600)),
            "2 years ago"
        );
    }

    #[test]
    fn format_modified_future_clamps_to_just_now() {
        // modified is AFTER now (clock skew) => duration_since errs => "just now".
        let now = fixed_now();
        let future = now + Duration::from_secs(120);
        assert_eq!(format_modified(Some(future), now), "just now");
    }

    #[test]
    fn format_modified_none_is_dash() {
        assert_eq!(format_modified(None, fixed_now()), "-");
    }

    // ── home contraction ─────────────────────────────────────────────────────────

    #[test]
    fn contract_home_replaces_prefix() {
        if let Some(home) = dirs::home_dir() {
            let home_s = home.display().to_string();
            let under = format!("{home_s}/.cache/mlxcel/models");
            let out = contract_home(&under);
            assert!(out.starts_with('~'), "expected ~ prefix: {out}");
            assert!(
                out.ends_with(".cache/mlxcel/models"),
                "tail preserved: {out}"
            );
            assert!(!out.contains(&home_s), "home should be contracted: {out}");
            // Exact home dir contracts to bare `~`.
            assert_eq!(contract_home(&home_s), "~");
        }
    }

    #[test]
    fn contract_home_leaves_non_prefix_untouched() {
        // A path that is not under $HOME is returned unchanged.
        let p = "/var/lib/mlxcel/models";
        assert_eq!(contract_home(p), p);
    }
}
