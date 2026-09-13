// Copyright 2025-2026 Lablup Inc.
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
//! - **Extra CA certificates** — set `MLXCEL_EXTRA_CA_CERTS=/path/to/bundle.pem`
//!   to additionally trust one or more custom root/intermediate CAs (PEM,
//!   concatenated) for this module's `reqwest::Client`, on top of its default
//!   trust store. Needed behind a TLS-inspecting corporate proxy (Cloudflare
//!   Zero Trust / WARP Gateway, Netskope, Zscaler, ...) when its root CA is
//!   not available to the active TLS backend (for example in a container).
//!   [`load_extra_ca_certificates`] is also used by the server's
//!   `http_image_client` (`src/server/media.rs`), which fetches remote media
//!   URLs through an independently-built client with the same trust gap.
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
mod model_format;
mod progress;
mod resolver;
mod store;

pub use cli::DownloadArgs;
pub use errors::map_hf_error;
pub use filters::{is_wanted_file, repo_basename};
pub use model_format::{
    UnsupportedModelReference, classify_model_reference, ensure_mlx_model_reference,
    huggingface_repo_from_url, redact_url_userinfo,
};
pub use progress::should_show_progress;
pub use resolver::ModelSourceOptions;
pub use resolver::normalize_repo_id;
pub use resolver::resolve_model_source;
pub use resolver::resolve_model_source_quietly_with_options;
pub use resolver::resolve_model_source_with_options;
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
use sha2::{Digest, Sha256};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
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
const MAX_HF_SIBLINGS: usize = 16_384;
const MAX_SELECTED_DOWNLOAD_FILES: usize = 8_192;
const MAX_REPO_FILENAME_BYTES: usize = 512;
const MAX_HF_METADATA_BYTES: usize = 8 * 1024 * 1024;

/// Resolved options for a download invocation.
///
/// Constructed from CLI arguments via [`DownloadOptions::from_args`] (the
/// shared adapter both binaries use). The struct exists so that programmatic
/// callers (and unit tests) can drive [`download_repo`] without going through
/// clap parsing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenMode {
    /// Resolve an explicit token first, then HF_TOKEN / HUGGING_FACE_HUB_TOKEN.
    Environment,
    /// Use anonymous HuggingFace requests; explicit and ambient tokens are ignored.
    Anonymous,
}

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
    /// Authentication token override. When `None` and `token_mode` is
    /// [`TokenMode::Environment`], falls back to environment variables
    /// (`HF_TOKEN`, then `HUGGING_FACE_HUB_TOKEN`). WebUI-managed library
    /// downloads set [`TokenMode::Anonymous`] so browser actions never
    /// silently consume ambient Hub credentials.
    pub token: Option<String>,
    pub token_mode: TokenMode,
    /// Optional repository-relative glob allow-list applied after the built-in
    /// safe file-type filter. Empty means every built-in-allowed file.
    pub include: Vec<String>,
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
            token_mode: TokenMode::Environment,
            include: args.include.clone(),
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
    resolve_token_with_mode(explicit, TokenMode::Environment)
}

pub fn resolve_token_with_mode(explicit: Option<&str>, mode: TokenMode) -> Option<String> {
    if mode == TokenMode::Anonymous {
        return None;
    }
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
    finish_api_builder(builder)
}

fn finish_api_builder(builder: ApiBuilder) -> Result<Api> {
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

/// Env var pointing at a PEM file of additional CA certificates to trust,
/// on top of (not instead of) the active TLS backend's default roots.
///
/// The direct download and media clients are built independently from the
/// `hf-hub` manifest client. Cargo feature unification currently makes both
/// native-tls and rustls available to `reqwest`, so callers must not depend on
/// a particular backend discovering a custom OS root. This explicit bundle
/// gives both direct clients deterministic, scoped trust without weakening or
/// replacing their default trust anchors.
const EXTRA_CA_CERTS_ENV: &str = "MLXCEL_EXTRA_CA_CERTS";

/// Generous upper bound for an operator-provided CA bundle.
///
/// Public root bundles are typically a few hundred KiB. Limiting this input
/// prevents a mistaken path to a large file from allocating unbounded memory
/// in the downloader or performing unbounded work on a server blocking-pool
/// worker.
const MAX_EXTRA_CA_CERTS_BYTES: usize = 1024 * 1024;

/// Load additional trust anchors from [`EXTRA_CA_CERTS_ENV`], if set.
///
/// Returns an empty `Vec` when the env var is unset or empty — the default,
/// zero-config path. When set, the file at that path is read and parsed as a
/// PEM bundle (one or more concatenated `-----BEGIN CERTIFICATE-----` blocks,
/// e.g. `cat corp-ca.pem >> extra-ca-certs.pem` to add more than one). A
/// missing file, unreadable file, oversized bundle, or bundle with zero valid
/// certificates is a hard error rather than a silent no-op: an operator who
/// set this var intended for it to take effect, and downloads would otherwise
/// fail later with a much less actionable TLS error.
pub(crate) fn load_extra_ca_certificates() -> Result<Vec<reqwest::Certificate>> {
    let path = match std::env::var(EXTRA_CA_CERTS_ENV) {
        Ok(val) if !val.trim().is_empty() => val.trim().to_string(),
        Ok(_) | Err(std::env::VarError::NotPresent) => return Ok(Vec::new()),
        Err(std::env::VarError::NotUnicode(raw)) => {
            return Err(anyhow!(
                "{EXTRA_CA_CERTS_ENV} is set but is not valid UTF-8: {raw:?}"
            ));
        }
    };
    let file = fs::File::open(&path).with_context(|| {
        format!("{EXTRA_CA_CERTS_ENV} is set to '{path}' but the file could not be read")
    })?;
    let mut bytes = Vec::new();
    file.take((MAX_EXTRA_CA_CERTS_BYTES + 1) as u64)
        .read_to_end(&mut bytes)
        .with_context(|| {
            format!("{EXTRA_CA_CERTS_ENV} is set to '{path}' but the file could not be read")
        })?;
    if bytes.len() > MAX_EXTRA_CA_CERTS_BYTES {
        return Err(anyhow!(
            "{EXTRA_CA_CERTS_ENV} points to '{path}', but the bundle exceeds the \
             {MAX_EXTRA_CA_CERTS_BYTES}-byte safety limit"
        ));
    }
    let certs = reqwest::Certificate::from_pem_bundle(&bytes).with_context(|| {
        format!(
            "{EXTRA_CA_CERTS_ENV} points to '{path}', but it could not be parsed as a PEM \
             certificate bundle"
        )
    })?;
    if certs.is_empty() {
        return Err(anyhow!(
            "{EXTRA_CA_CERTS_ENV} points to '{path}', but it contains no certificates"
        ));
    }
    Ok(certs)
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

fn repo_info_url(endpoint: &str, repo_id: &str, revision: &str) -> String {
    let endpoint = endpoint.trim_end_matches('/');
    let repo_enc = encode_path_segments(repo_id);
    let rev_enc = encode_path_segments(revision);
    format!("{endpoint}/api/models/{repo_enc}/revision/{rev_enc}?blobs=true")
}

#[derive(Debug, Clone, serde::Deserialize)]
struct HubRepoInfo {
    sha: String,
    #[serde(default)]
    siblings: Vec<HubSibling>,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct HubSibling {
    rfilename: String,
    #[serde(default)]
    size: Option<u64>,
    #[serde(default)]
    lfs: Option<HubLfs>,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct HubLfs {
    #[serde(default)]
    sha256: Option<String>,
    #[serde(default)]
    size: Option<u64>,
}

#[derive(Debug, Clone)]
struct SelectedDownloadFile {
    filename: String,
    expected_size: Option<u64>,
    sha256: Option<String>,
}

impl SelectedDownloadFile {
    fn from_sibling(repo_id: &str, sibling: &HubSibling) -> Result<Self> {
        validate_manifest_filename(repo_id, &sibling.rfilename)?;
        let sha256 = sibling
            .lfs
            .as_ref()
            .and_then(|lfs| lfs.sha256.as_deref())
            .map(|hash| validate_sha256_hex(&sibling.rfilename, hash))
            .transpose()?;
        if sibling.rfilename.ends_with(".safetensors") && sha256.is_none() {
            return Err(anyhow!(
                "Repository metadata for '{}' is missing an LFS SHA-256 digest; refusing to publish unverifiable weights",
                sibling.rfilename
            ));
        }
        Ok(Self {
            filename: sibling.rfilename.clone(),
            expected_size: sibling
                .lfs
                .as_ref()
                .and_then(|lfs| lfs.size)
                .or(sibling.size),
            sha256,
        })
    }
}

fn build_reqwest_client(token: Option<&str>, enforce_https: bool) -> Result<reqwest::Client> {
    let mut headers = reqwest::header::HeaderMap::new();
    if let Some(tok) = token {
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
    for cert in load_extra_ca_certificates()? {
        builder = builder.add_root_certificate(cert);
    }
    builder.build().context("Failed to create HTTP client")
}

async fn fetch_anonymous_repo_info(
    client: &reqwest::Client,
    endpoint: &str,
    repo_id: &str,
    revision: &str,
) -> Result<HubRepoInfo> {
    let url = repo_info_url(endpoint, repo_id, revision);
    let response = client
        .get(&url)
        .send()
        .await
        .with_context(|| format!("HTTP request failed for repository metadata '{repo_id}'"))?;
    let status = response.status();
    if !status.is_success() {
        let code = status.as_u16();
        return Err(match code {
            401 | 403 => anyhow!(
                "Repository '{repo_id}' requires authentication, is gated, or is private; WebUI downloads use anonymous public Hub access only"
            ),
            404 => anyhow!("Repository '{repo_id}' or revision '{revision}' was not found"),
            _ => anyhow!("HTTP {code} while fetching repository metadata for '{repo_id}'"),
        });
    }

    let mut body = Vec::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.with_context(|| {
            format!("Stream error while reading repository metadata for '{repo_id}'")
        })?;
        if body.len().saturating_add(chunk.len()) > MAX_HF_METADATA_BYTES {
            return Err(anyhow!(
                "Repository metadata for '{repo_id}' exceeds the mlxcel {MAX_HF_METADATA_BYTES}-byte safety limit"
            ));
        }
        body.extend_from_slice(&chunk);
    }
    serde_json::from_slice::<HubRepoInfo>(&body)
        .with_context(|| format!("Failed to parse HuggingFace repository metadata for '{repo_id}'"))
}

fn fetch_anonymous_repo_info_blocking(
    repo_id: &str,
    revision: Option<&str>,
) -> Result<HubRepoInfo> {
    let endpoint = hf_endpoint();
    let requested_revision = revision.unwrap_or("main");
    let rt = tokio::runtime::Runtime::new().context("Failed to create tokio runtime")?;
    let client = build_reqwest_client(None, false)?;
    rt.block_on(fetch_anonymous_repo_info(
        &client,
        &endpoint,
        repo_id,
        requested_revision,
    ))
}

/// Download a single file via reqwest streaming, ticking the per-file and
/// aggregate progress bars as each chunk arrives.
///
/// Writes to a sibling tempfile first, then atomically renames to `dest`.
/// On error, the tempfile is removed and both progress bars are abandoned.
#[allow(clippy::too_many_arguments)]
async fn stream_file(
    client: &reqwest::Client,
    url: &str,
    dest: &Path,
    filename: &str,
    file_pb: &indicatif::ProgressBar,
    aggregate_pb: &indicatif::ProgressBar,
    expected_size: u64,
    hooks: &DownloadHooks,
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

    let result = stream_to_tempfile(
        client,
        url,
        &tmp,
        dest,
        filename,
        file_pb,
        aggregate_pb,
        expected_size,
        hooks,
    )
    .await;
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
#[allow(clippy::too_many_arguments)]
async fn stream_to_tempfile(
    client: &reqwest::Client,
    url: &str,
    tmp: &Path,
    dest: &Path,
    filename: &str,
    file_pb: &indicatif::ProgressBar,
    aggregate_pb: &indicatif::ProgressBar,
    expected_size: u64,
    hooks: &DownloadHooks,
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

    // Prefer a known metadata/HEAD size for stable UI totals, then fall back to
    // the response's own Content-Length for validation when no manifest size is
    // available.
    let response_size = response.content_length();
    let expected_size = (expected_size > 0).then_some(expected_size);
    let total_size = expected_size.or(response_size).unwrap_or(0);

    let mut stream = response.bytes_stream();
    let mut bytes_written: u64 = 0;

    while let Some(chunk) = stream.next().await {
        check_cancelled(hooks)
            .with_context(|| format!("Cancelled while downloading {filename}"))?;
        let chunk = chunk.with_context(|| format!("Stream error while downloading {filename}"))?;
        out.write_all(&chunk)
            .await
            .with_context(|| format!("Write error while downloading {filename}"))?;
        let chunk_len = chunk.len() as u64;
        bytes_written += chunk_len;
        file_pb.inc(chunk_len);
        aggregate_pb.inc(chunk_len);
        if let Some(progress) = &hooks.progress {
            progress(url, bytes_written, total_size);
        }
    }

    out.flush()
        .await
        .with_context(|| format!("Flush error for {filename}"))?;
    drop(out);
    validate_downloaded_size(filename, expected_size.or(response_size), bytes_written)?;

    tokio::fs::rename(tmp, dest)
        .await
        .with_context(|| format!("Failed to atomically install {filename}"))?;

    Ok(bytes_written)
}

/// Process-wide offline mode (`--offline` / `LLAMA_ARG_OFFLINE`, issue #1434).
///
/// b10621 carries offline mode as one `params.offline` boolean consulted at
/// every download site, and mlxcel mirrors that rather than threading a
/// parameter through every loader: the resolver takes it explicitly through
/// [`ModelSourceOptions`], and the sites that are reached from deep inside a
/// loader (the moondream starmie tokenizer fetch, the request-path media
/// fetch) read it here. The flag only ever turns network access off, so a
/// caller that forgets to pass it cannot re-enable a fetch this forbids.
static OFFLINE: AtomicBool = AtomicBool::new(false);

/// Turn process-wide offline mode on (or off, for a test that set it).
///
/// Called once per process by each server entry point after `--offline` and
/// `LLAMA_ARG_OFFLINE` have been resolved, before any model is loaded.
pub fn set_offline_mode(offline: bool) {
    OFFLINE.store(offline, Ordering::Relaxed);
}

/// True when this process is in offline mode.
#[must_use]
pub fn offline_mode() -> bool {
    OFFLINE.load(Ordering::Relaxed)
}

/// `Ok(())` unless offline mode forbids the fetch `what` describes.
///
/// `what` is a short noun phrase naming the artifact, and lands in the
/// diagnostic as "offline mode is on, so mlxcel will not fetch {what}". Use it
/// at every site that would otherwise reach the network.
///
/// # Errors
///
/// Returns an error naming the artifact and how to pre-populate it whenever
/// [`offline_mode`] is true.
pub fn ensure_online(what: &str) -> Result<()> {
    if offline_mode() {
        return Err(anyhow!(
            "offline mode is on (--offline / LLAMA_ARG_OFFLINE), so mlxcel will \
             not fetch {what}. Pre-populate it on a host with network access, or \
             re-run without --offline."
        ));
    }
    Ok(())
}

/// Programmatic observation and control of one [`download_repo_with_hooks`]
/// run (issue #1438).
///
/// The CLI download surface reports progress through indicatif bars on
/// stderr; the router server's `POST /models` flow instead forwards progress
/// into its `GET /models/sse` event stream and needs to cancel an in-flight
/// download when `DELETE /models` removes the model. Both needs are optional
/// observations of the same download loop, so they ride along as hooks rather
/// than forking the implementation.
#[derive(Debug, Clone)]
pub struct DownloadPlan {
    pub repo_id: String,
    pub requested_revision: String,
    pub resolved_revision: String,
    pub destination: PathBuf,
    pub selected_files: usize,
    pub total_bytes: Option<u64>,
}

#[derive(Clone, Default)]
pub struct DownloadHooks {
    /// Called as bytes arrive for each file: `(url, downloaded, total)`.
    /// `total` is `0` when the size is unknown. Files already present on disk
    /// report one terminal call with `downloaded == total`. Called from the
    /// download worker thread; keep it fast and non-blocking.
    pub progress: Option<std::sync::Arc<dyn Fn(&str, u64, u64) + Send + Sync>>,
    /// Called after repository metadata and selected file sizes are known but
    /// before the first transfer starts. `total_bytes` is `None` when one or
    /// more selected files have unknown length.
    pub plan: Option<std::sync::Arc<dyn Fn(DownloadPlan) + Send + Sync>>,
    /// Called immediately before an already-complete staged snapshot is made
    /// visible. Returning `false` aborts publication as a cooperative cancel;
    /// returning `true` is the linearization point after which cancellation may
    /// be refused by the coordinator.
    pub begin_publish: Option<std::sync::Arc<dyn Fn() -> bool + Send + Sync>>,
    /// Cooperative cancellation flag, checked between streamed chunks and
    /// between files. When it becomes `true`, the download aborts with an
    /// error wrapping [`DownloadCancelled`], leaving no partial tempfiles
    /// behind (completed files stay, exactly like any other mid-run failure).
    pub cancel: Option<std::sync::Arc<AtomicBool>>,
}

impl std::fmt::Debug for DownloadHooks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DownloadHooks")
            .field("progress", &self.progress.is_some())
            .field("plan", &self.plan.is_some())
            .field("begin_publish", &self.begin_publish.is_some())
            .field("cancel", &self.cancel.is_some())
            .finish()
    }
}

/// Marker error produced when a [`DownloadHooks::cancel`] flag aborts a
/// download. Callers distinguish cancellation from genuine failures with
/// `err.chain().any(|c| c.is::<DownloadCancelled>())` (the marker may sit
/// below added context) rather than by matching message text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DownloadCancelled;

impl std::fmt::Display for DownloadCancelled {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "download cancelled")
    }
}

impl std::error::Error for DownloadCancelled {}

/// True when `err`'s chain contains the [`DownloadCancelled`] marker.
#[must_use]
pub fn is_download_cancelled(err: &anyhow::Error) -> bool {
    err.chain().any(|cause| cause.is::<DownloadCancelled>())
}

fn check_cancelled(hooks: &DownloadHooks) -> Result<()> {
    if let Some(cancel) = &hooks.cancel
        && cancel.load(Ordering::Relaxed)
    {
        return Err(anyhow::Error::new(DownloadCancelled));
    }
    Ok(())
}

fn validate_manifest_filename(repo_id: &str, filename: &str) -> Result<()> {
    if filename.len() > MAX_REPO_FILENAME_BYTES {
        return Err(anyhow!(
            "Repository '{repo_id}' contains a filename longer than {MAX_REPO_FILENAME_BYTES} bytes; refusing to download it"
        ));
    }
    Ok(())
}

fn validate_manifest_bounds(
    repo_id: &str,
    sibling_count: usize,
    selected_count: usize,
) -> Result<()> {
    if sibling_count > MAX_HF_SIBLINGS {
        return Err(anyhow!(
            "Repository '{repo_id}' exposes {sibling_count} files, above the mlxcel limit of {MAX_HF_SIBLINGS}"
        ));
    }
    if selected_count > MAX_SELECTED_DOWNLOAD_FILES {
        return Err(anyhow!(
            "Repository '{repo_id}' selected {selected_count} files for download, above the mlxcel limit of {MAX_SELECTED_DOWNLOAD_FILES}"
        ));
    }
    Ok(())
}

fn validate_downloaded_size(
    filename: &str,
    expected_size: Option<u64>,
    bytes_written: u64,
) -> Result<()> {
    if let Some(expected_size) = expected_size
        && bytes_written != expected_size
    {
        return Err(anyhow!(
            "Downloaded size mismatch for '{filename}': expected {expected_size} bytes, wrote {bytes_written} bytes"
        ));
    }
    Ok(())
}

fn validate_sha256_hex(filename: &str, hash: &str) -> Result<String> {
    if hash.len() != 64 || !hash.as_bytes().iter().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(anyhow!(
            "Repository metadata for '{filename}' contains an invalid SHA-256 digest"
        ));
    }
    Ok(hash.to_ascii_lowercase())
}

fn sha256_hex(digest: impl AsRef<[u8]>) -> String {
    let bytes = digest.as_ref();
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        let _ = write!(&mut out, "{byte:02x}");
    }
    out
}

fn verify_downloaded_sha256(filename: &str, expected: Option<&str>, actual: &[u8]) -> Result<()> {
    let Some(expected) = expected else {
        return Ok(());
    };
    let actual = sha256_hex(actual);
    if actual != expected {
        return Err(anyhow!(
            "Downloaded checksum mismatch for '{filename}': expected SHA-256 {expected}, got {actual}"
        ));
    }
    Ok(())
}

/// Probe that `repo_id` names a reachable HuggingFace model repository
/// (issue #1438). Resolves the ambient token, builds the API client, and
/// fetches the repository manifest without downloading any file. The router
/// server's `POST /models` validates a requested name with this before
/// starting the background download, mirroring b10621's synchronous metadata
/// fetch in its own handler. Blocking: call from a blocking-capable thread.
pub fn probe_repo(repo_id: &str, revision: Option<&str>) -> Result<()> {
    probe_repo_with_token_mode(repo_id, revision, TokenMode::Environment)
}

pub fn probe_repo_with_token_mode(
    repo_id: &str,
    revision: Option<&str>,
    token_mode: TokenMode,
) -> Result<()> {
    ensure_online(&format!("the repository manifest for '{repo_id}'"))?;
    if token_mode == TokenMode::Anonymous {
        let repo_id = normalize_repo_id(repo_id)?;
        return fetch_anonymous_repo_info_blocking(&repo_id, revision).map(|_| ());
    }
    let token = resolve_token_with_mode(None, token_mode);
    let api = build_api(token)?;
    let repo = build_repo_handle(repo_id, revision);
    api.repo(repo)
        .info()
        .map(|_| ())
        .map_err(|err| map_hf_error(err, repo_id, revision, None))
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
    download_repo_with_hooks(opts, DownloadHooks::default())
}

/// [`download_repo`] with progress observation and cooperative cancellation
/// (issue #1438). See [`DownloadHooks`].
pub fn download_repo_with_hooks(opts: DownloadOptions, hooks: DownloadHooks) -> Result<()> {
    // Offline mode forbids every snapshot fetch (issue #1434). The resolver
    // already reports a cache miss with a repo-specific diagnostic before
    // reaching here; this is the backstop for any other caller, including
    // `mlxcel download` itself.
    ensure_online(&format!("the model snapshot '{}'", opts.repo_id))?;

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
            .spawn(move || download_repo_blocking(opts, hooks))
            .context("Failed to spawn the download worker thread")?;
        return match handle.join() {
            Ok(result) => result,
            Err(panic) => std::panic::resume_unwind(panic),
        };
    }
    download_repo_blocking(opts, hooks)
}

#[cfg(unix)]
pub fn download_repo_to_existing_dir_fd(
    repo_id: &str,
    revision: Option<&str>,
    dir_fd: std::os::fd::RawFd,
    reported_destination: PathBuf,
    hooks: DownloadHooks,
) -> Result<()> {
    ensure_online(&format!("the model snapshot '{repo_id}'"))?;
    let repo_id = normalize_repo_id(repo_id)?;
    let requested_revision = revision.unwrap_or("main").to_string();
    let endpoint = hf_endpoint();
    let rt = tokio::runtime::Runtime::new().context("Failed to create tokio runtime")?;
    let client = build_reqwest_client(None, false)?;
    let info = rt.block_on(fetch_anonymous_repo_info(
        &client,
        &endpoint,
        &repo_id,
        &requested_revision,
    ))?;
    let resolved_revision = info.sha.clone();
    if info.siblings.len() > MAX_HF_SIBLINGS {
        validate_manifest_bounds(&repo_id, info.siblings.len(), 0)?;
    }
    let mut wanted: Vec<SelectedDownloadFile> = info
        .siblings
        .iter()
        .map(|sibling| SelectedDownloadFile::from_sibling(&repo_id, sibling))
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .filter(|file| is_wanted_file(&file.filename))
        .collect();
    validate_manifest_bounds(&repo_id, info.siblings.len(), wanted.len())?;
    if wanted.is_empty() {
        return Err(anyhow!(
            "Repository '{repo_id}' does not expose any supported model files. Expected config/tokenizer JSON and safetensors/weights files."
        ));
    }
    if !wanted
        .iter()
        .any(|file| file.filename.ends_with(".safetensors"))
    {
        return Err(anyhow!(
            "Repository '{repo_id}' metadata is incomplete: no safetensors weight files were selected"
        ));
    }
    let revision = resolved_revision.as_str();
    if (hooks.progress.is_some() || hooks.plan.is_some())
        && wanted.iter().any(|file| file.expected_size.is_none())
    {
        let size_map = rt.block_on(async {
            let client_ref = &client;
            let endpoint_ref = &endpoint;
            let repo_id_ref = &repo_id;
            futures::stream::iter(
                wanted
                    .iter()
                    .filter(|file| file.expected_size.is_none())
                    .map(|file| file.filename.clone()),
            )
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
                    .filter(|size| *size > 0);
                (filename, size)
            })
            .buffer_unordered(8)
            .collect::<std::collections::HashMap<String, Option<u64>>>()
            .await
        });
        for file in &mut wanted {
            if file.expected_size.is_none()
                && let Some(size) = size_map.get(&file.filename).and_then(|size| *size)
            {
                file.expected_size = Some(size);
            }
        }
    }
    let total_bytes = wanted
        .iter()
        .try_fold(0u64, |acc, file| {
            file.expected_size
                .map(|size| acc.saturating_add(size))
                .ok_or(())
        })
        .ok();
    if let Some(plan) = &hooks.plan {
        plan(DownloadPlan {
            repo_id: repo_id.clone(),
            requested_revision: requested_revision.clone(),
            resolved_revision: resolved_revision.clone(),
            destination: reported_destination,
            selected_files: wanted.len(),
            total_bytes,
        });
    }
    for file in &wanted {
        check_cancelled(&hooks)?;
        let url = file_url(&endpoint, &repo_id, revision, &file.filename);
        rt.block_on(stream_file_to_dir_fd(
            &client,
            &url,
            dir_fd,
            &file.filename,
            file.expected_size,
            file.sha256.as_deref(),
            &hooks,
        ))?;
    }
    if !file_exists_nonempty_at(dir_fd, "config.json")? {
        return Err(anyhow!(
            "Downloaded files for '{repo_id}' are incomplete: missing config.json"
        ));
    }
    if !wanted
        .iter()
        .filter(|file| file.filename.ends_with(".safetensors"))
        .all(|file| file_exists_nonempty_at(dir_fd, &file.filename).unwrap_or(false))
    {
        return Err(anyhow!(
            "Downloaded files for '{repo_id}' are incomplete: one or more safetensors files are missing or empty"
        ));
    }
    unsafe {
        libc::fsync(dir_fd);
    }
    Ok(())
}

#[cfg(unix)]
async fn stream_file_to_dir_fd(
    client: &reqwest::Client,
    url: &str,
    root_fd: std::os::fd::RawFd,
    filename: &str,
    expected_size: Option<u64>,
    expected_sha256: Option<&str>,
    hooks: &DownloadHooks,
) -> Result<u64> {
    use std::os::fd::{AsRawFd, FromRawFd};

    let (parent, leaf) = open_parent_dir_fd(root_fd, filename)?;
    let tmp_name = std::ffi::CString::new(format!(
        ".mlxcel-partial.{}.{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0),
    ))?;
    let tmp_fd = unsafe {
        libc::openat(
            parent.as_raw_fd(),
            tmp_name.as_ptr(),
            libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL | libc::O_CLOEXEC | libc::O_NOFOLLOW,
            0o600,
        )
    };
    if tmp_fd < 0 {
        return Err(std::io::Error::last_os_error())
            .with_context(|| format!("Failed to create tempfile for {filename}"));
    }
    let std_file = unsafe { std::fs::File::from_raw_fd(tmp_fd) };
    let mut out = tokio::fs::File::from_std(std_file);
    let result = async {
        let response = client
            .get(url)
            .send()
            .await
            .with_context(|| format!("HTTP request failed for {filename}"))?;
        let status = response.status();
        if !status.is_success() {
            let code = status.as_u16();
            return Err(anyhow!("HTTP {code} downloading '{filename}'"));
        }
        let response_size = response.content_length();
        let total_size = expected_size.or(response_size).unwrap_or(0);
        let mut stream = response.bytes_stream();
        let mut bytes_written = 0u64;
        let mut hasher = expected_sha256.map(|_| Sha256::new());
        while let Some(chunk) = stream.next().await {
            check_cancelled(hooks)
                .with_context(|| format!("Cancelled while downloading {filename}"))?;
            let chunk =
                chunk.with_context(|| format!("Stream error while downloading {filename}"))?;
            out.write_all(&chunk)
                .await
                .with_context(|| format!("Write error while downloading {filename}"))?;
            if let Some(hasher) = hasher.as_mut() {
                hasher.update(&chunk);
            }
            bytes_written += chunk.len() as u64;
            if let Some(progress) = &hooks.progress {
                progress(url, bytes_written, total_size);
            }
        }
        out.flush()
            .await
            .with_context(|| format!("Flush error for {filename}"))?;
        drop(out);
        validate_downloaded_size(filename, expected_size.or(response_size), bytes_written)?;
        if let Some(hasher) = hasher {
            let digest = hasher.finalize();
            verify_downloaded_sha256(filename, expected_sha256, &digest)?;
        }
        let rc = unsafe {
            libc::renameat(
                parent.as_raw_fd(),
                tmp_name.as_ptr(),
                parent.as_raw_fd(),
                leaf.as_ptr(),
            )
        };
        if rc != 0 {
            return Err(std::io::Error::last_os_error())
                .with_context(|| format!("Rename error for {filename}"));
        }
        Ok(bytes_written)
    }
    .await;
    if result.is_err() {
        unsafe {
            libc::unlinkat(parent.as_raw_fd(), tmp_name.as_ptr(), 0);
        }
    }
    result
}

#[cfg(unix)]
fn open_parent_dir_fd(
    root_fd: std::os::fd::RawFd,
    filename: &str,
) -> Result<(std::fs::File, std::ffi::CString)> {
    use std::os::fd::{AsRawFd, FromRawFd};
    let mut parts = filename.split('/').collect::<Vec<_>>();
    let leaf = parts
        .pop()
        .ok_or_else(|| anyhow!("invalid empty filename in repository manifest"))?;
    let leaf = cstring_repo_component(leaf)?;
    let dup = unsafe { libc::dup(root_fd) };
    if dup < 0 {
        return Err(std::io::Error::last_os_error()).context("failed to duplicate root fd");
    }
    let mut current = unsafe { std::fs::File::from_raw_fd(dup) };
    for part in parts {
        let name = cstring_repo_component(part)?;
        let rc = unsafe { libc::mkdirat(current.as_raw_fd(), name.as_ptr(), 0o700) };
        if rc != 0 {
            let err = std::io::Error::last_os_error();
            if err.raw_os_error() != Some(libc::EEXIST) {
                return Err(err)
                    .with_context(|| format!("failed to create directory for {filename}"));
            }
        }
        let fd = unsafe {
            libc::openat(
                current.as_raw_fd(),
                name.as_ptr(),
                libc::O_RDONLY | libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOFOLLOW,
            )
        };
        if fd < 0 {
            return Err(std::io::Error::last_os_error())
                .with_context(|| format!("failed to open directory for {filename}"));
        }
        current = unsafe { std::fs::File::from_raw_fd(fd) };
    }
    Ok((current, leaf))
}

#[cfg(unix)]
fn cstring_repo_component(component: &str) -> Result<std::ffi::CString> {
    if component.is_empty() || component == "." || component == ".." || component.contains('\\') {
        return Err(anyhow!("unsafe repository filename component"));
    }
    std::ffi::CString::new(component).map_err(|_| anyhow!("repository filename contains NUL byte"))
}

#[cfg(unix)]
fn file_exists_nonempty_at(root_fd: std::os::fd::RawFd, filename: &str) -> Result<bool> {
    use std::os::fd::AsRawFd;
    let (parent, leaf) = open_parent_dir_fd(root_fd, filename)?;
    let mut stat = std::mem::MaybeUninit::<libc::stat>::uninit();
    let rc = unsafe {
        libc::fstatat(
            parent.as_raw_fd(),
            leaf.as_ptr(),
            stat.as_mut_ptr(),
            libc::AT_SYMLINK_NOFOLLOW,
        )
    };
    if rc != 0 {
        let err = std::io::Error::last_os_error();
        if err.raw_os_error() == Some(libc::ENOENT) {
            return Ok(false);
        }
        return Err(err).context("failed to stat downloaded file");
    }
    let stat = unsafe { stat.assume_init() };
    Ok((stat.st_mode & libc::S_IFMT) == libc::S_IFREG && stat.st_size > 0)
}

fn download_repo_blocking(opts: DownloadOptions, hooks: DownloadHooks) -> Result<()> {
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
    let include_patterns = opts
        .include
        .iter()
        .map(|pattern| {
            glob::Pattern::new(pattern)
                .with_context(|| format!("Invalid --include glob pattern {pattern:?}"))
        })
        .collect::<Result<Vec<_>>>()?;

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
    let token = resolve_token_with_mode(opts.token.as_deref(), opts.token_mode);
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
    let requested_revision = opts.revision.as_deref().unwrap_or("main").to_string();
    let resolved_revision = info.sha.clone();
    if info.siblings.len() > MAX_HF_SIBLINGS {
        validate_manifest_bounds(&opts.repo_id, info.siblings.len(), 0)?;
    }
    let wanted: Vec<String> = info
        .siblings
        .iter()
        .map(|sibling| {
            validate_manifest_filename(&opts.repo_id, &sibling.rfilename)?;
            Ok(sibling.rfilename.clone())
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .filter(|name| is_wanted_file(name))
        .filter(|name| matches_include_patterns(name, &include_patterns))
        .collect();
    validate_manifest_bounds(&opts.repo_id, info.siblings.len(), wanted.len())?;

    if wanted.is_empty() {
        return Err(anyhow!(
            "Repository '{}' contains no files matching the mlxcel allow-list{}.",
            opts.repo_id,
            if opts.include.is_empty() {
                " (config.json, tokenizer*, *.safetensors, ...). Nothing to download".to_string()
            } else {
                format!(
                    " and --include patterns {:?}. Nothing to download",
                    opts.include
                )
            }
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
    let client = build_reqwest_client(token.as_deref(), enforce_https)?;

    let revision = resolved_revision.as_str();

    // Build per-file sizes map for accurate bar lengths. `hf-hub 0.5` does
    // not expose per-file sizes in the manifest (Siblings only has `rfilename`),
    // so we issue concurrent HEAD requests here. L6: previously
    // sequential (N RTTs before the first byte streamed); now bounded-parallel
    // via `futures::stream::iter(...).buffer_unordered(8)` so total pre-stream
    // wallclock is roughly one RTT for a small repo. 8 is well below any HF
    // rate limit. Sizes are best-effort: if a HEAD fails we fall back to 0
    // (indeterminate bar). We skip the HEAD pass entirely when progress bars
    // are suppressed.
    // The HEAD-request size pass also runs when a progress hook is installed
    // (issue #1438): the router's SSE `download_progress` events carry
    // per-file totals, which come from nowhere else.
    let size_map: std::collections::HashMap<String, u64> =
        if show_bars || hooks.progress.is_some() || hooks.plan.is_some() {
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
    let total_bytes = if wanted
        .iter()
        .all(|name| size_map.get(name.as_str()).copied().unwrap_or(0) > 0)
    {
        Some(total_known_bytes)
    } else {
        None
    };
    if let Some(plan) = &hooks.plan {
        plan(DownloadPlan {
            repo_id: opts.repo_id.clone(),
            requested_revision: requested_revision.clone(),
            resolved_revision: resolved_revision.clone(),
            destination: local_dir.clone(),
            selected_files: wanted.len(),
            total_bytes,
        });
    }

    let mp = progress::create_multi_progress();
    let aggregate_pb = progress::add_aggregate_bar(&mp, total_known_bytes);

    let total = wanted.len();
    let mut downloaded = 0usize;
    let mut skipped = 0usize;
    let mut total_bytes: u64 = 0;

    for (idx, filename) in wanted.iter().enumerate() {
        check_cancelled(&hooks)?;
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
            if let Some(progress) = &hooks.progress {
                let url = file_url(&endpoint, &opts.repo_id, revision, filename);
                progress(&url, cached_size, cached_size);
            }
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
            file_size,
            &hooks,
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

fn matches_include_patterns(name: &str, patterns: &[glob::Pattern]) -> bool {
    patterns.is_empty()
        || patterns
            .iter()
            .any(|pattern| pattern.matches_path(Path::new(name)))
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
