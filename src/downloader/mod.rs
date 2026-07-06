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

//! HuggingFace model repository downloader.
//!
//! Provides a single source of truth for downloading model snapshots from the
//! HuggingFace Hub. Both the `mlxcel` CLI and the `mlxcel-server` binary call
//! into the same [`download_repo`] entry point so the supported file set and
//! flag semantics stay in lock-step.
//!
//! # Design
//!
//! - **Allow-list filtering** — files are kept based on extension/name patterns
//!   (`config.json`, `*.safetensors`, tokenizers, processor configs, ...). New
//!   model families work without code changes; non-MLX artifacts (`*.bin`,
//!   `*.gguf`, ...) are skipped to save bandwidth and disk.
//! - **Token resolution order** — explicit `--token` > `HF_TOKEN` env >
//!   `HUGGING_FACE_HUB_TOKEN` env > anonymous. Tokens are validated to be
//!   pure printable-ASCII (no control chars) before they are used in an
//!   `Authorization` header (L3).
//! - **Default destination** — the location-independent global store at
//!   `${MLXCEL_CACHE_DIR:-$HOME/.cache/mlxcel}/models/<owner>/<name>` (issue
//!   #93), so a model downloaded once runs from any directory. `--local-dir`
//!   is the explicit opt-out. Before downloading, an existing HuggingFace Hub
//!   cache snapshot of the same repo is reused read-only (no re-fetch). See
//!   [`store`] for the path resolution and HF-cache probing.
//! - **Caching** — without `--force`, an existing snapshot with all expected
//!   files at the right size is treated as a no-op. With `--force`, every file
//!   is re-fetched and overwritten.
//! - **Progress** — when stderr is a tty and progress is not suppressed via
//!   env vars, per-file and aggregate `indicatif` progress bars render during
//!   the actual byte stream (Path B2 direct reqwest streaming).
//!   When bars are suppressed (CI, piped output, `MLXCEL_NO_PROGRESS=1`,
//!   `NO_COLOR=1`), one stdout line per file is emitted instead so CI logs
//!   remain golden-text-stable.
//!
//! # Hardening
//!
//! - **Plaintext-endpoint refusal** — if `HF_ENDPOINT` is set to a non-HTTPS
//!   URL *and* a token is resolved, [`download_repo`] aborts with a clear
//!   error so the bearer token is never leaked over plaintext HTTP. The
//!   reqwest client is additionally built with `https_only(true)` when a
//!   token is in use, so a same-host HTTPS→HTTP redirect cannot smuggle the
//!   bearer header onto plaintext either. Set `MLXCEL_ALLOW_INSECURE_ENDPOINT=1`
//!   to opt back out (intended for internal mirrors fronted by an
//!   HTTPS-terminated reverse proxy on a trusted network).
//! - **Network timeouts** — the shared `reqwest::Client` is built with
//!   `connect_timeout(10s)` and `read_timeout(30s)`. A stalled mirror or
//!   half-closed TCP connection therefore fails fast instead of hanging the
//!   CLI/server indefinitely. Total elapsed download time is intentionally
//!   unbounded (large files take time); only inactivity is bounded.
//! - **URL segment encoding** — `repo_id`, `revision`, and `filename` are
//!   percent-encoded per-segment when composing the GET/HEAD URL so that
//!   adversarial repo metadata containing `?`, `#`, or other reserved
//!   characters cannot smuggle a query string or fragment past the request.
//! - **Symlink-safe tempfiles** — on Unix the partial-download tempfile is
//!   opened with `O_CREAT|O_EXCL|O_NOFOLLOW`, so an attacker who pre-stages
//!   a symlink at the predicted tempfile path cannot redirect our writes.
//! - **Stale tempfile cleanup** — at the start of every download, partial
//!   files named `.mlxcel-partial.*` older than one hour are removed
//!   best-effort. Younger partials are left alone to avoid racing with a
//!   concurrent `mlxcel` process targeting the same directory.
//! - **Parallel HEAD prefetch** — per-file size discovery uses
//!   `futures::stream::iter(...).buffer_unordered(8)` so progress bars and
//!   aggregate totals are accurate without paying N sequential HEAD RTTs
//!   before the first byte streams.

mod cli;
mod completeness;
mod errors;
mod filters;
mod progress;
mod resolver;
mod store;

pub use cli::DownloadArgs;
pub use errors::map_hf_error;
pub use filters::{is_wanted_file, repo_basename};
pub use progress::should_show_progress;
pub use resolver::normalize_repo_id;
pub use resolver::resolve_model_source;
pub use resolver::resolve_model_source_with_override;
pub use store::{
    RemoveError, RemoveOutcome, StoredModel, dir_size, hf_cache_snapshot, list_models,
    list_models_with_override, model_dir, model_dir_with_override, models_root, remove_model,
    remove_model_with_override, store_root,
};

use anyhow::{Context, Result, anyhow};
use futures::StreamExt;
use hf_hub::api::sync::{Api, ApiBuilder};
use hf_hub::{Repo, RepoType};
use percent_encoding::{AsciiSet, CONTROLS, utf8_percent_encode};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime};
use tokio::io::AsyncWriteExt;

/// Characters NOT allowed in a single URL path segment.
///
/// We start from `CONTROLS` (RFC 3986 reserves all control chars in path
/// components) and add every byte that is reserved or unsafe within a single
/// path segment, namely the gen-delims `?`, `#`, `/`, `:`, `@`, `[`, `]`, the
/// sub-delims `!`, `$`, `&`, `'`, `(`, `)`, `*`, `+`, `,`, `;`, `=`, plus
/// `%`, `\`, `"`, `<`, `>`, ` `, `^`, `\``, `{`, `|`, `}`. The unreserved set
/// per RFC 3986 (alphanumerics plus `-`, `.`, `_`, `~`) is preserved.
///
/// Used by [`file_url`] to encode each `/`-separated segment of `repo_id`,
/// `revision`, and `filename` so adversarial metadata cannot smuggle a query
/// string or fragment past the URL composition step.
const SEGMENT_ENCODE_SET: &AsciiSet = &CONTROLS
    .add(b' ')
    .add(b'"')
    .add(b'#')
    .add(b'%')
    .add(b'&')
    .add(b'\'')
    .add(b'(')
    .add(b')')
    .add(b'*')
    .add(b'+')
    .add(b',')
    .add(b'/')
    .add(b':')
    .add(b';')
    .add(b'<')
    .add(b'=')
    .add(b'>')
    .add(b'?')
    .add(b'@')
    .add(b'[')
    .add(b'\\')
    .add(b']')
    .add(b'^')
    .add(b'`')
    .add(b'{')
    .add(b'|')
    .add(b'}')
    .add(b'!')
    .add(b'$');

/// Age threshold for `.mlxcel-partial.*` orphan cleanup (L5).
///
/// Younger partial files are left in place to avoid racing against a concurrent
/// `mlxcel` process that is mid-download in the same destination directory.
const PARTIAL_TEMPFILE_STALE_AGE: Duration = Duration::from_secs(60 * 60);

/// Resolved options for a download invocation.
///
/// Constructed from CLI arguments via [`DownloadOptions::from_args`] (the
/// shared adapter both binaries use). The struct exists so that programmatic
/// callers (and unit tests) can drive [`download_repo`] without going through
/// clap parsing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DownloadOptions {
    /// HuggingFace repository identifier, e.g. `mlx-community/Qwen3-4B-4bit`.
    pub repo_id: String,
    /// Local destination directory. When `None`, defaults to the global store
    /// at `${MLXCEL_CACHE_DIR:-$HOME/.cache/mlxcel}/models/<owner>/<name>`
    /// (issue #93). `Some(path)` is the explicit opt-out and writes the
    /// snapshot at `path` verbatim.
    pub local_dir: Option<PathBuf>,
    /// Override for the model-store ROOT (issue #107), set from `--models-dir`.
    /// When `Some(root)` and `local_dir` is `None`, the snapshot lands at
    /// `<root>/<owner>/<name>` (no `models/` subdir). `None` keeps the
    /// `MLXCEL_MODELS_DIR`-then-cache-root resolution in [`store::models_root`].
    /// Ignored when `local_dir` is `Some` (the verbatim path wins).
    pub models_dir: Option<PathBuf>,
    /// Repository revision (branch, tag, or commit). Defaults to `main` when
    /// `None`.
    pub revision: Option<String>,
    /// Authentication token override. When `None`, falls back to environment
    /// variables (`HF_TOKEN`, then `HUGGING_FACE_HUB_TOKEN`).
    pub token: Option<String>,
    /// Re-download every file even when a complete snapshot is already
    /// present locally.
    pub force: bool,
}

impl DownloadOptions {
    /// Convert the binary-side clap struct into a runtime options bundle.
    pub fn from_args(args: &DownloadArgs) -> Self {
        Self {
            repo_id: args.repo_id.clone(),
            local_dir: args.local_dir.clone(),
            models_dir: args.models_dir.clone(),
            revision: args.revision.clone(),
            token: args.token.clone(),
            force: args.force,
        }
    }

    /// Resolve the destination directory for a fresh download.
    ///
    /// - An explicit `--local-dir PATH` is honored verbatim (the opt-out) and
    ///   retains ultimate precedence over `--models-dir` / `MLXCEL_MODELS_DIR`.
    /// - Otherwise the destination is the location-independent global store
    ///   under the override-aware models root (issue #107): `--models-dir
    ///   <root>` or `MLXCEL_MODELS_DIR` place the snapshot directly at
    ///   `<root>/<owner>/<name>`, falling back to
    ///   `${MLXCEL_CACHE_DIR:-$HOME/.cache/mlxcel}/models/<owner>/<name>`
    ///   (issue #93) so a model downloaded once is runnable from any directory.
    /// - As a last-resort fallback (no override, no `MLXCEL_MODELS_DIR`, no
    ///   home directory *and* `MLXCEL_CACHE_DIR` unset — practically never on a
    ///   supported platform), we degrade to the legacy per-CWD
    ///   `models/<repo_basename>` so the downloader still produces a usable
    ///   path instead of panicking.
    ///
    /// Note: this returns the *write* destination only. HuggingFace-cache
    /// read-reuse (skipping the download entirely when a snapshot already
    /// exists under `$HF_HUB_CACHE` / `$HF_HOME`) is handled separately in
    /// [`download_repo`] so that `--local-dir` continues to mean "write here".
    pub fn resolve_local_dir(&self) -> PathBuf {
        match &self.local_dir {
            Some(path) => path.clone(),
            None => store::model_dir_with_override(&self.repo_id, self.models_dir.as_deref())
                .unwrap_or_else(|| PathBuf::from("models").join(repo_basename(&self.repo_id))),
        }
    }
}

/// Resolve the effective HuggingFace token using the documented precedence:
/// explicit `--token` flag, then `HF_TOKEN`, then `HUGGING_FACE_HUB_TOKEN`,
/// then anonymous (`None`).
///
/// Empty values from environment variables are treated as anonymous so that
/// `HF_TOKEN=""` does not poison the request with a malformed `Authorization`
/// header. Tokens containing non-ASCII bytes or ASCII control characters
/// (L3) are still returned here so the caller can produce a
/// targeted error message that names the env var or flag — see
/// [`validate_token`] which is invoked at HTTP-client construction time.
pub fn resolve_token(explicit: Option<&str>) -> Option<String> {
    if let Some(t) = explicit {
        let trimmed = t.trim();
        if !trimmed.is_empty() {
            return Some(trimmed.to_string());
        }
    }
    for env_key in ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"] {
        if let Ok(value) = std::env::var(env_key) {
            let trimmed = value.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }
    }
    None
}

/// Reject HF tokens containing non-ASCII bytes or ASCII control characters
/// (L3). Returns the original token slice on success.
///
/// `HeaderValue::from_str` would also reject these values, but we want a
/// domain-specific error message (mentions HF token, env var, control chars)
/// instead of reqwest's generic `InvalidHeaderValue`.
fn validate_token(token: &str) -> Result<&str> {
    if let Some((idx, ch)) = token
        .chars()
        .enumerate()
        .find(|(_, c)| !c.is_ascii() || c.is_ascii_control())
    {
        return Err(anyhow!(
            "HF token contains invalid characters (must be ASCII, no control chars): \
             byte index {idx} is U+{:04X}",
            ch as u32
        ));
    }
    Ok(token)
}

/// Build a configured `hf-hub` [`Api`] honoring the resolved auth token.
///
/// We keep `with_progress(false)` because progress is driven by our own
/// indicatif bars. hf-hub is used only for `info()` (manifest
/// fetch) — the actual file bytes come from direct reqwest streaming.
fn build_api(token: Option<String>) -> Result<Api> {
    let mut builder = ApiBuilder::from_env().with_progress(false);
    if let Some(tok) = token {
        builder = builder.with_token(Some(tok));
    }
    builder
        .build()
        .map_err(|err| anyhow!("Failed to initialize Hugging Face API client: {err}"))
}

/// Open the [`Repo`] handle for the requested model + revision.
fn build_repo_handle(repo_id: &str, revision: Option<&str>) -> Repo {
    match revision {
        Some(rev) => Repo::with_revision(repo_id.to_string(), RepoType::Model, rev.to_string()),
        None => Repo::new(repo_id.to_string(), RepoType::Model),
    }
}

/// Resolve the HuggingFace endpoint base URL.
///
/// Respects `HF_ENDPOINT` env var (allows using a mirror), otherwise defaults
/// to `https://huggingface.co`.
fn hf_endpoint() -> String {
    std::env::var("HF_ENDPOINT")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| "https://huggingface.co".to_string())
}

/// Env var that opts out of the M1 plaintext-endpoint refusal.
///
/// When set to any non-empty value, [`require_secure_endpoint_for_token`]
/// allows a `Bearer` token to be sent over a non-HTTPS endpoint. Intended
/// for internal mirrors fronted by an HTTPS-terminated reverse proxy on a
/// trusted network where the operator has audited the path.
const INSECURE_ENDPOINT_OPT_OUT: &str = "MLXCEL_ALLOW_INSECURE_ENDPOINT";

/// Return `true` when `MLXCEL_ALLOW_INSECURE_ENDPOINT` is set to a non-empty,
/// non-whitespace value.
///
/// Shared between [`require_secure_endpoint_for_token`] (initial-scheme guard)
/// and the reqwest client builder (`.https_only(true)` redirect guard) so both
/// honor the same operator escape hatch.
fn is_insecure_endpoint_opt_out() -> bool {
    matches!(std::env::var(INSECURE_ENDPOINT_OPT_OUT), Ok(val) if !val.trim().is_empty())
}

/// Refuse plaintext endpoints when a token would be transmitted (M1).
///
/// Returns `Ok(())` for anonymous downloads regardless of scheme, and for
/// authenticated downloads only when `endpoint` starts with `https://`
/// (case-insensitive) or the operator has explicitly set
/// `MLXCEL_ALLOW_INSECURE_ENDPOINT=<non-empty>`.
fn require_secure_endpoint_for_token(endpoint: &str, token: Option<&str>) -> Result<()> {
    if token.is_none() {
        return Ok(());
    }
    let lower = endpoint.trim().to_ascii_lowercase();
    if lower.starts_with("https://") {
        return Ok(());
    }
    if is_insecure_endpoint_opt_out() {
        eprintln!(
            "[mlxcel download] warning: {INSECURE_ENDPOINT_OPT_OUT} is set; sending HF token over \
             plaintext endpoint '{endpoint}'. The token can be intercepted on the network path."
        );
        return Ok(());
    }
    Err(anyhow!(
        "HF_ENDPOINT '{endpoint}' must use HTTPS when an auth token is set. \
         Set {INSECURE_ENDPOINT_OPT_OUT}=1 to override at your own risk."
    ))
}

/// Best-effort cleanup of stale `.mlxcel-partial.*` orphans (L5).
///
/// Walks `local_dir` (non-recursive) and removes regular files whose basename
/// starts with `.mlxcel-partial.` and whose last-modified timestamp is older
/// than [`PARTIAL_TEMPFILE_STALE_AGE`]. Any I/O error (including failure to
/// read the directory) is logged to stderr and otherwise ignored — this is
/// disk-hygiene, not a security boundary.
fn cleanup_stale_partials(local_dir: &Path) {
    let now = SystemTime::now();
    let read_dir = match fs::read_dir(local_dir) {
        Ok(d) => d,
        Err(err) => {
            eprintln!(
                "[mlxcel download] warning: could not scan {} for stale partials: {err}",
                local_dir.display()
            );
            return;
        }
    };
    for entry in read_dir.flatten() {
        let name = entry.file_name();
        let name_str = match name.to_str() {
            Some(s) => s,
            None => continue,
        };
        if !name_str.starts_with(".mlxcel-partial.") {
            continue;
        }
        let path = entry.path();
        let metadata = match fs::metadata(&path) {
            Ok(m) => m,
            Err(_) => continue,
        };
        if !metadata.is_file() {
            continue;
        }
        let modified = match metadata.modified() {
            Ok(t) => t,
            Err(_) => continue,
        };
        let age = match now.duration_since(modified) {
            Ok(d) => d,
            Err(_) => {
                // Future-mtime — assume it is racing in-flight; skip.
                continue;
            }
        };
        if age < PARTIAL_TEMPFILE_STALE_AGE {
            continue;
        }
        if let Err(err) = fs::remove_file(&path) {
            eprintln!(
                "[mlxcel download] warning: failed to remove stale partial {}: {err}",
                path.display()
            );
        }
    }
}

/// Percent-encode every `/`-separated segment of `path` using
/// [`SEGMENT_ENCODE_SET`] and reassemble them with `/`.
///
/// Empty segments (e.g. from a leading or duplicate `/`) are preserved verbatim
/// so the caller still gets back exactly the same number of segments.
fn encode_path_segments(path: &str) -> String {
    path.split('/')
        .map(|seg| utf8_percent_encode(seg, SEGMENT_ENCODE_SET).to_string())
        .collect::<Vec<_>>()
        .join("/")
}

/// Build the download URL for a single file in a HuggingFace repository.
///
/// Every path segment of `repo_id`, `revision`, and `filename` is
/// percent-encoded (L1) so adversarial metadata containing `?`,
/// `#`, or other reserved characters cannot smuggle a query string or
/// fragment past the URL composition step. `endpoint` is treated as a
/// trusted base URL (env-controlled by the operator) and is not re-encoded.
fn file_url(endpoint: &str, repo_id: &str, revision: &str, filename: &str) -> String {
    let repo_enc = encode_path_segments(repo_id);
    let rev_enc = encode_path_segments(revision);
    let file_enc = encode_path_segments(filename);
    format!("{endpoint}/{repo_enc}/resolve/{rev_enc}/{file_enc}")
}

/// Download a single file via reqwest streaming, ticking the per-file and
/// aggregate progress bars as each chunk arrives.
///
/// Writes to a sibling tempfile first, then atomically renames to `dest`.
/// On error, the tempfile is removed and both progress bars are abandoned.
async fn stream_file(
    client: &reqwest::Client,
    url: &str,
    dest: &Path,
    filename: &str,
    file_pb: &indicatif::ProgressBar,
    aggregate_pb: &indicatif::ProgressBar,
) -> Result<u64> {
    let tmp_name = format!(
        ".mlxcel-partial.{}.{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0),
    );
    let tmp = dest.with_file_name(tmp_name);

    let result = stream_to_tempfile(client, url, &tmp, dest, filename, file_pb, aggregate_pb).await;
    if result.is_err() {
        // Best-effort cleanup — ignore errors from remove_file (file may not
        // exist yet if `File::create` itself failed). The original error from
        // streaming is the actionable one for the user.
        let _ = tokio::fs::remove_file(&tmp).await;
    }
    result
}

/// Open the partial-download tempfile in a symlink-safe way (L2).
///
/// On Unix we open with `O_CREAT | O_EXCL | O_NOFOLLOW` (translated by
/// `OpenOptions`: `create_new(true)` provides `O_CREAT|O_EXCL`, and the
/// explicit `custom_flags(libc::O_NOFOLLOW)` adds belt-and-suspenders so that
/// even if an attacker wins the EEXIST race by hardlinking, the open still
/// refuses to traverse a symlink. `create_new(true)` is itself sufficient
/// against the symlink case because `O_EXCL` fails on any existing path
/// (including a symlink), but `O_NOFOLLOW` makes the intent explicit and
/// closes any narrow window between metadata stat and open syscall.
///
/// On non-Unix targets mlxcel is not officially supported, so we fall back to
/// the existing `create(truncate=true)` semantics with a comment.
async fn open_tempfile_no_symlink(tmp: &Path) -> Result<tokio::fs::File> {
    #[cfg(unix)]
    {
        // `tokio::fs::OpenOptions::custom_flags` is an inherent method (not the
        // trait extension `std::os::unix::fs::OpenOptionsExt`), so it does not
        // need an explicit `use` import.
        tokio::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .custom_flags(libc::O_NOFOLLOW)
            .open(tmp)
            .await
            .with_context(|| {
                format!(
                    "Failed to create tempfile at {} (O_CREAT|O_EXCL|O_NOFOLLOW). \
                     If the path already exists as a symlink or regular file, \
                     remove it manually and retry.",
                    tmp.display()
                )
            })
    }
    #[cfg(not(unix))]
    {
        // mlxcel only targets macOS + Linux; this branch exists so the crate
        // still compiles on Windows / WASM if someone tries. The hardening is
        // a no-op there.
        tokio::fs::File::create(tmp)
            .await
            .with_context(|| format!("Failed to create tempfile at {}", tmp.display()))
    }
}

/// Inner implementation of [`stream_file`]: stream bytes into `tmp`, then
/// atomically rename to `dest`. Callers are responsible for cleaning up `tmp`
/// on error.
async fn stream_to_tempfile(
    client: &reqwest::Client,
    url: &str,
    tmp: &Path,
    dest: &Path,
    filename: &str,
    file_pb: &indicatif::ProgressBar,
    aggregate_pb: &indicatif::ProgressBar,
) -> Result<u64> {
    // Open the tempfile FIRST so that an adversary cannot pre-stage a symlink
    // at `tmp` between the HTTP response and the actual write. `O_NOFOLLOW`
    // + `O_EXCL` fail closed on any pre-existing path (L2).
    let mut out = open_tempfile_no_symlink(tmp)
        .await
        .with_context(|| format!("Failed to create tempfile for {filename}"))?;

    let response = client
        .get(url)
        .send()
        .await
        .with_context(|| format!("HTTP request failed for {filename}"))?;

    let status = response.status();
    if !status.is_success() {
        let code = status.as_u16();
        return Err(anyhow!(
            "HTTP {code} downloading '{filename}'. \
             Check authentication (--token / HF_TOKEN) or that the repository exists."
        ));
    }

    let mut stream = response.bytes_stream();
    let mut bytes_written: u64 = 0;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.with_context(|| format!("Stream error while downloading {filename}"))?;
        out.write_all(&chunk)
            .await
            .with_context(|| format!("Write error while downloading {filename}"))?;
        let chunk_len = chunk.len() as u64;
        bytes_written += chunk_len;
        file_pb.inc(chunk_len);
        aggregate_pb.inc(chunk_len);
    }

    out.flush()
        .await
        .with_context(|| format!("Flush error for {filename}"))?;
    drop(out);

    tokio::fs::rename(tmp, dest)
        .await
        .with_context(|| format!("Failed to atomically install {filename}"))?;

    Ok(bytes_written)
}

/// Download a HuggingFace model repository snapshot into a local directory.
///
/// On success, every allow-listed file from the upstream repository is present
/// inside `local_dir` (resolved per [`DownloadOptions::resolve_local_dir`]).
///
/// # Errors
///
/// Returns actionable [`anyhow::Error`] messages for the common failure modes:
/// invalid repo id, missing authentication on a gated repo, missing revision,
/// network failure, and on-disk I/O errors.
pub fn download_repo(opts: DownloadOptions) -> Result<()> {
    // Issue #463: this function creates its own Tokio runtime and calls
    // `block_on`, which Tokio forbids on a thread that is already driving a
    // runtime ("Cannot start a runtime from within a runtime", an abort). The
    // async `mlxcel serve` / `mlxcel-server` startup paths hit exactly that
    // when a model must be auto-downloaded. When a runtime is detected on the
    // current thread, run the blocking download body on a dedicated OS thread
    // instead; the progress UX is unchanged (the bars write to the same
    // stderr) and every caller, sync or async, becomes safe.
    if tokio::runtime::Handle::try_current().is_ok() {
        let handle = std::thread::Builder::new()
            .name("mlxcel-download".to_string())
            .spawn(move || download_repo_blocking(opts))
            .context("Failed to spawn the download worker thread")?;
        return match handle.join() {
            Ok(result) => result,
            Err(panic) => std::panic::resume_unwind(panic),
        };
    }
    download_repo_blocking(opts)
}

fn download_repo_blocking(opts: DownloadOptions) -> Result<()> {
    // Issue #171: expand a bare, prefix-less model name (e.g. `Qwen3-4B-4bit`)
    // to `<default-org>/<name>` BEFORE anything is derived from `opts.repo_id` —
    // the HF-cache reuse probe below, the store destination
    // (`resolve_local_dir`), the repo handle, and every per-file download URL
    // all key off it. The bare-name → default-org expansion was wired only into
    // the `-m`/run resolver (issue #112); the `download` verb bypassed it and
    // 404'd on a slashless repo-id. `normalize_repo_id` is the shared funnel, so
    // `mlxcel download` and `mlx-server download` now match the resolver-backed
    // commands. An `owner/name` id (anything containing `/`) is returned
    // unchanged, so resolver-driven calls — which already pass a full
    // `owner/name` — are a no-op here (no double expansion, no duplicate info
    // line).
    let mut opts = opts;
    opts.repo_id = normalize_repo_id(&opts.repo_id)?;

    // HF-cache read-reuse (issue #93): when the caller did not pin an explicit
    // `--local-dir` and is not forcing a refresh, reuse a complete snapshot
    // already present in the HuggingFace Hub cache (`$HF_HUB_CACHE` /
    // `$HF_HOME` / `~/.cache/huggingface/hub`). This lets users who already
    // pulled a model with mlx-lm / transformers skip re-fetching gigabytes.
    // The reuse is strictly read-only — we never write into the HF
    // content-addressed layout. An explicit `--local-dir` keeps "write here"
    // semantics and bypasses this short-circuit entirely.
    if opts.local_dir.is_none()
        && !opts.force
        && let Some(hf_snapshot) = store::hf_cache_snapshot(&opts.repo_id, opts.revision.as_deref())
    {
        println!(
            "[mlxcel download] repo={} revision={} already present in HuggingFace cache; \
             reusing without re-download: {}",
            opts.repo_id,
            opts.revision.as_deref().unwrap_or("main"),
            hf_snapshot.display(),
        );
        return Ok(());
    }

    let local_dir = opts.resolve_local_dir();
    let token = resolve_token(opts.token.as_deref());
    let endpoint = hf_endpoint();

    // M1 — refuse plaintext endpoints when a token would be sent over the
    // wire. A bearer token sent over `http://` exposes the
    // long-lived credential to anyone on-path. The opt-out env var exists
    // for operators who genuinely run an HTTPS-terminated reverse proxy
    // in front of an internal HTTP mirror on a trusted network.
    require_secure_endpoint_for_token(&endpoint, token.as_deref())?;

    let api = build_api(token.clone())?;
    let repo = build_repo_handle(&opts.repo_id, opts.revision.as_deref());
    let api_repo = api.repo(repo);

    println!(
        "[mlxcel download] repo={} revision={} dest={}",
        opts.repo_id,
        opts.revision.as_deref().unwrap_or("main"),
        local_dir.display(),
    );

    let info = api_repo
        .info()
        .map_err(|err| map_hf_error(err, &opts.repo_id, opts.revision.as_deref(), None))?;
    let wanted: Vec<String> = info
        .siblings
        .iter()
        .map(|s| s.rfilename.clone())
        .filter(|name| is_wanted_file(name))
        .collect();

    if wanted.is_empty() {
        return Err(anyhow!(
            "Repository '{}' contains no files matching the mlxcel allow-list \
             (config.json, tokenizer*, *.safetensors, ...). Nothing to download.",
            opts.repo_id
        ));
    }

    println!(
        "[mlxcel download] {} files queued (filtered from {} total siblings)",
        wanted.len(),
        info.siblings.len(),
    );

    fs::create_dir_all(&local_dir).with_context(|| {
        format!(
            "Failed to create destination directory {}",
            local_dir.display()
        )
    })?;

    // L5 — opportunistic cleanup of stale `.mlxcel-partial.*` orphans
    // Best-effort: any I/O error here is logged but does
    // not fail the download. Only files older than `PARTIAL_TEMPFILE_STALE_AGE`
    // are removed so concurrent in-flight downloads from a sibling process
    // are not disturbed.
    cleanup_stale_partials(&local_dir);

    if opts.force {
        println!("[mlxcel download] --force: refreshing every file");
    } else if snapshot_complete(&local_dir, &wanted) {
        println!(
            "[mlxcel download] all expected files already present at {}, skipping (use --force to refresh)",
            local_dir.display(),
        );
        return Ok(());
    }

    // Canonicalize `local_dir` once. We compare every per-file destination
    // parent against this prefix to refuse writes that escape the snapshot
    // directory (defense in depth on top of the basename allow-list and the
    // `is_safe_relative_path` filter in `is_wanted_file`).
    let canonical_local = fs::canonicalize(&local_dir)
        .with_context(|| format!("Failed to canonicalize destination {}", local_dir.display()))?;

    let show_bars = should_show_progress();

    // Build the reqwest client once and share across all file downloads.
    //
    // Timeouts (M2): `connect_timeout(10s)` aborts the TCP/TLS
    // handshake if a mirror is unreachable. `read_timeout(30s)` aborts when
    // the response body stalls for 30s — the correct semantics for a long
    // download where total elapsed time is unbounded but any 30s window of
    // dead air is a clear stall. Total `timeout(...)` is intentionally NOT
    // set because legitimate large weight files take more than the global
    // default.
    //
    // Redirect downgrade defense (M1 reinforcement): reqwest's
    // default `remove_sensitive_headers` only strips `Authorization` on
    // cross-host redirects, NOT on same-host scheme downgrades. Without
    // `https_only(true)` a malicious HTTPS->HTTP 302 on the same host would
    // forward the bearer token over plaintext. The `require_secure_endpoint_*`
    // guard above only validates the initial scheme; `https_only(true)` makes
    // the redirect path enforce HTTPS too. Operators who opt out via
    // `MLXCEL_ALLOW_INSECURE_ENDPOINT` already accepted the plaintext risk,
    // so we honor their decision here as well.
    let enforce_https = token.is_some() && !is_insecure_endpoint_opt_out();
    let rt = tokio::runtime::Runtime::new().context("Failed to create tokio runtime")?;
    let client = rt.block_on(async {
        let mut headers = reqwest::header::HeaderMap::new();
        if let Some(ref tok) = token {
            // L3: reject non-ASCII or control-char tokens with
            // a domain-specific error instead of the panic that the prior
            // `.expect("token must be ASCII")` would produce. `validate_token`
            // also rejects more characters than `HeaderValue::from_str` would
            // strictly need (e.g., embedded `\r\n`), making it harder to
            // smuggle header-injection payloads through a malformed env var.
            validate_token(tok)?;
            let auth_val = format!("Bearer {tok}");
            headers.insert(
                reqwest::header::AUTHORIZATION,
                reqwest::header::HeaderValue::from_str(&auth_val).with_context(
                    || "HF token contains invalid characters (must be ASCII, no control chars)",
                )?,
            );
        }
        let mut builder = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .read_timeout(Duration::from_secs(30))
            .default_headers(headers);
        if enforce_https {
            builder = builder.https_only(true);
        }
        builder.build().context("Failed to create HTTP client")
    })?;

    let revision = opts.revision.as_deref().unwrap_or("main");

    // Build per-file sizes map for accurate bar lengths. `hf-hub 0.5` does
    // not expose per-file sizes in the manifest (Siblings only has `rfilename`),
    // so we issue concurrent HEAD requests here. L6: previously
    // sequential (N RTTs before the first byte streamed); now bounded-parallel
    // via `futures::stream::iter(...).buffer_unordered(8)` so total pre-stream
    // wallclock is roughly one RTT for a small repo. 8 is well below any HF
    // rate limit. Sizes are best-effort: if a HEAD fails we fall back to 0
    // (indeterminate bar). We skip the HEAD pass entirely when progress bars
    // are suppressed.
    let size_map: std::collections::HashMap<String, u64> = if show_bars {
        rt.block_on(async {
            let client_ref = &client;
            let endpoint_ref = &endpoint;
            let repo_id_ref = &opts.repo_id;
            futures::stream::iter(wanted.iter().cloned())
                .map(move |filename| async move {
                    let url = file_url(endpoint_ref, repo_id_ref, revision, &filename);
                    let size = client_ref
                        .head(&url)
                        .send()
                        .await
                        .ok()
                        .and_then(|r| {
                            r.headers()
                                .get(reqwest::header::CONTENT_LENGTH)
                                .and_then(|v| v.to_str().ok())
                                .and_then(|s| s.parse::<u64>().ok())
                        })
                        .unwrap_or(0);
                    (filename, size)
                })
                .buffer_unordered(8)
                .collect::<std::collections::HashMap<String, u64>>()
                .await
        })
    } else {
        std::collections::HashMap::new()
    };

    let total_known_bytes: u64 = size_map.values().sum();

    let mp = progress::create_multi_progress();
    let aggregate_pb = progress::add_aggregate_bar(&mp, total_known_bytes);

    let total = wanted.len();
    let mut downloaded = 0usize;
    let mut skipped = 0usize;
    let mut total_bytes: u64 = 0;

    for (idx, filename) in wanted.iter().enumerate() {
        let dest = local_dir.join(filename);

        // Defense in depth: even if a malicious sibling slipped through the
        // allow-list, refuse to touch any path whose parent does not resolve
        // inside `canonical_local`. This catches symlink shenanigans and any
        // future regression in the basename filter.
        let dest_parent = dest.parent().unwrap_or(&local_dir);
        fs::create_dir_all(dest_parent).with_context(|| {
            format!(
                "Failed to create directory {} for {filename}",
                dest_parent.display()
            )
        })?;
        let canonical_parent = fs::canonicalize(dest_parent).with_context(|| {
            format!(
                "Failed to canonicalize destination parent {}",
                dest_parent.display()
            )
        })?;
        if !canonical_parent.starts_with(&canonical_local) {
            return Err(anyhow!(
                "Refusing to write '{filename}' outside of '{}': resolved to '{}'.",
                local_dir.display(),
                canonical_parent.display(),
            ));
        }

        if !opts.force && file_exists_nonempty(&dest) {
            // Cached-file fast path: emit a single line, do NOT animate a bar.
            // The aggregate bar is not ticked since no bytes are transferred;
            // instead we advance it by the expected file size to keep the total
            // accurate.
            let cached_size = fs::metadata(&dest).map(|m| m.len()).unwrap_or(0);
            println!("[{}/{total}] cached: {filename}", idx + 1,);
            // Advance aggregate bar by the cached file size so total progress
            // reflects what is on disk, not just what was downloaded this session.
            aggregate_pb.inc(cached_size);
            total_bytes += cached_size;
            skipped += 1;
            continue;
        }

        let file_size = *size_map.get(filename.as_str()).unwrap_or(&0);
        let file_pb = progress::add_file_bar(&mp, filename, file_size);

        if !show_bars {
            println!("[{}/{total}] downloading: {filename}", idx + 1,);
        }

        let started = Instant::now();
        let url = file_url(&endpoint, &opts.repo_id, revision, filename);

        let result = rt.block_on(stream_file(
            &client,
            &url,
            &dest,
            filename,
            &file_pb,
            &aggregate_pb,
        ));

        match result {
            Ok(bytes) => {
                total_bytes += bytes;
                downloaded += 1;
                let elapsed = started.elapsed();
                file_pb.finish_and_clear();
                println!(
                    "[{}/{total}] done: {filename} ({size} in {secs:.1}s)",
                    idx + 1,
                    size = format_bytes(bytes),
                    secs = elapsed.as_secs_f64(),
                );
            }
            Err(err) => {
                // Red-finish the per-file bar so users see which file failed.
                file_pb.abandon_with_message(format!("FAILED: {filename}"));
                return Err(err);
            }
        }
    }

    aggregate_pb.finish_and_clear();
    drop(mp);

    println!(
        "[mlxcel download] complete: downloaded={} cached={} total_size={} dest={}",
        downloaded,
        skipped,
        format_bytes(total_bytes),
        local_dir.display(),
    );
    Ok(())
}

/// True when every wanted file is present in `local_dir` with non-zero size.
///
/// A simple presence + non-empty check is sufficient because we write to a
/// temp path and rename atomically, so partial files do not normally remain.
/// `--force` is the documented escape hatch when this heuristic is not enough.
fn snapshot_complete(local_dir: &Path, wanted: &[String]) -> bool {
    if !local_dir.join("config.json").exists() {
        return false;
    }
    wanted
        .iter()
        .all(|name| file_exists_nonempty(&local_dir.join(name)))
}

fn file_exists_nonempty(path: &Path) -> bool {
    fs::metadata(path)
        .map(|m| m.is_file() && m.len() > 0)
        .unwrap_or(false)
}

fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KiB", "MiB", "GiB", "TiB"];
    if bytes == 0 {
        return "0 B".to_string();
    }
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{} {}", bytes, UNITS[0])
    } else {
        format!("{value:.2} {}", UNITS[unit])
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
