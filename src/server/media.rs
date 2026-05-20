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

//! Shared request-media helpers for server routes.
//!
//! Keeping image-source parsing at the HTTP edge makes it easier to add new
//! request formats without growing individual route handlers. The helpers stay
//! async so local file reads and remote URL fetches do not block Axum workers.

use anyhow::{Context, Result, anyhow, bail};
use base64::Engine;
use futures::StreamExt;
use std::{
    path::{Path, PathBuf},
    sync::{
        OnceLock,
        atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering},
    },
    time::Duration,
};

use crate::multimodal::video::{TempFile, VideoSource, is_video_file};

use super::types::ChatCompletionRequest;
use super::types::request::{InputAudio, VideoUrl};

/// Resolved video request item produced by
/// [`extract_chat_video_paths_with_allowlist`] (issue #601).
///
/// Carries:
/// * the [`VideoSource`] handle that downstream consumers must pass to
///   [`crate::multimodal::video::load_video_source`]. The fd-backed variant
///   is what closes the TOCTOU race against an attacker swapping the
///   resolved file for a symlink between the resolver's `canonicalize`
///   and ffmpeg's `open` (issue #601);
/// * the optional per-video FPS override from
///   [`crate::server::types::request::VideoUrl::fps`];
/// * an optional [`TempFile`] guard that owns the cleanup of any
///   server-allocated temporary file (data-URI decode, HTTP fetch). The
///   guard is held until the response handler drops `ResolvedVideo`.
///
/// **fd lifetime = request lifetime.** On Unix each `ResolvedVideo` holds
/// an open file descriptor for the duration of the request. Operators
/// sizing `--max-queue-depth` (or the equivalent `LLAMA_ARG_*` env var)
/// should account for this: at peak concurrency the process holds up to
/// `max_queue_depth × videos_per_request` additional file descriptors
/// beyond the model and tokenizer fds. Ensure the per-process fd ulimit
/// (`ulimit -n` / `LimitNOFILE`) is large enough — a minimum of 4096 is
/// recommended; 65536 is typical for production deployments.
///
/// `pub(crate)` because every consumer lives inside the `mlxcel` crate.
#[derive(Debug)]
pub(crate) struct ResolvedVideo {
    /// The video handle the worker thread feeds to ffmpeg/ffprobe. For
    /// `file://` and bare-local-path resolutions on Unix this is the
    /// fd-backed [`VideoSource::Fd`] variant; for data-URI / HTTP fetches
    /// the resolver materialises the bytes to a temp file and then opens
    /// that temp file as an fd. On non-Unix targets every variant collapses
    /// to [`VideoSource::Path`].
    pub(crate) source: VideoSource,
    /// Per-video FPS override from
    /// [`crate::server::types::request::VideoUrl::fps`].
    pub(crate) fps: Option<f64>,
    /// `Some(TempFile)` when the resolver allocated a server-owned
    /// temp file (data URI / HTTP fetch). The guard is dropped — and the
    /// file unlinked — when this struct drops.
    ///
    /// `dead_code` suppression is intentional: the guard is read only for
    /// its `Drop` impl. Any other access path would defeat the point of
    /// the guard.
    #[allow(dead_code)]
    pub(crate) temp_guard: Option<TempFile>,
}

impl ResolvedVideo {
    /// Borrow the canonical path identity used by diagnostics and
    /// request preparation logging. Subprocesses must still consume the
    /// [`VideoSource`] via [`Self::source`], not the path — passing the
    /// path to ffmpeg would re-introduce the canonicalise → ffmpeg-open
    /// TOCTOU window the fd-based design closed (issue #601).
    #[allow(dead_code)]
    pub(crate) fn canonical_path(&self) -> &Path {
        self.source.canonical_path()
    }
}

/// Maximum encoded image payload size after base64/HTTP/file resolution: 64 MiB.
///
/// This keeps common VLM uploads well inside the default while bounding the
/// server memory committed before the image decoder can apply format-aware
/// limits.
pub const DEFAULT_MAX_IMAGE_PAYLOAD_SIZE: usize = 64 * 1024 * 1024;

/// Maximum number of image content blocks accepted in one request.
pub const DEFAULT_MAX_IMAGES_PER_REQUEST: usize = 16;

/// Maximum decoded image width accepted by the server image decoder.
pub const DEFAULT_MAX_IMAGE_WIDTH: u32 = 16_384;

/// Maximum decoded image height accepted by the server image decoder.
pub const DEFAULT_MAX_IMAGE_HEIGHT: u32 = 16_384;

/// Maximum image-decoder allocation budget in bytes.
pub const DEFAULT_MAX_IMAGE_DECODE_ALLOC_BYTES: u64 = 512 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageInputLimits {
    pub max_payload_bytes: usize,
    pub max_images_per_request: usize,
    pub max_width: u32,
    pub max_height: u32,
    pub max_decode_alloc_bytes: u64,
}

impl Default for ImageInputLimits {
    fn default() -> Self {
        Self {
            max_payload_bytes: DEFAULT_MAX_IMAGE_PAYLOAD_SIZE,
            max_images_per_request: DEFAULT_MAX_IMAGES_PER_REQUEST,
            max_width: DEFAULT_MAX_IMAGE_WIDTH,
            max_height: DEFAULT_MAX_IMAGE_HEIGHT,
            max_decode_alloc_bytes: DEFAULT_MAX_IMAGE_DECODE_ALLOC_BYTES,
        }
    }
}

impl ImageInputLimits {
    #[must_use]
    pub(crate) fn image_decode_limits(self) -> image::Limits {
        let mut limits = image::Limits::default();
        limits.max_image_width = Some(self.max_width);
        limits.max_image_height = Some(self.max_height);
        limits.max_alloc = Some(self.max_decode_alloc_bytes);
        limits
    }
}

static MAX_IMAGE_PAYLOAD_BYTES: AtomicUsize = AtomicUsize::new(DEFAULT_MAX_IMAGE_PAYLOAD_SIZE);
static MAX_IMAGES_PER_REQUEST: AtomicUsize = AtomicUsize::new(DEFAULT_MAX_IMAGES_PER_REQUEST);
static MAX_IMAGE_WIDTH: AtomicU32 = AtomicU32::new(DEFAULT_MAX_IMAGE_WIDTH);
static MAX_IMAGE_HEIGHT: AtomicU32 = AtomicU32::new(DEFAULT_MAX_IMAGE_HEIGHT);
static MAX_IMAGE_DECODE_ALLOC_BYTES: AtomicU64 =
    AtomicU64::new(DEFAULT_MAX_IMAGE_DECODE_ALLOC_BYTES);

pub(crate) fn configure_image_input_limits(limits: ImageInputLimits) {
    MAX_IMAGE_PAYLOAD_BYTES.store(limits.max_payload_bytes, Ordering::Relaxed);
    MAX_IMAGES_PER_REQUEST.store(limits.max_images_per_request, Ordering::Relaxed);
    MAX_IMAGE_WIDTH.store(limits.max_width, Ordering::Relaxed);
    MAX_IMAGE_HEIGHT.store(limits.max_height, Ordering::Relaxed);
    MAX_IMAGE_DECODE_ALLOC_BYTES.store(limits.max_decode_alloc_bytes, Ordering::Relaxed);
}

#[must_use]
pub(crate) fn current_image_input_limits() -> ImageInputLimits {
    ImageInputLimits {
        max_payload_bytes: MAX_IMAGE_PAYLOAD_BYTES.load(Ordering::Relaxed),
        max_images_per_request: MAX_IMAGES_PER_REQUEST.load(Ordering::Relaxed),
        max_width: MAX_IMAGE_WIDTH.load(Ordering::Relaxed),
        max_height: MAX_IMAGE_HEIGHT.load(Ordering::Relaxed),
        max_decode_alloc_bytes: MAX_IMAGE_DECODE_ALLOC_BYTES.load(Ordering::Relaxed),
    }
}

#[cfg(test)]
pub(crate) async fn extract_chat_image_data(request: &ChatCompletionRequest) -> Vec<Vec<u8>> {
    match try_extract_chat_image_data(request).await {
        Ok(images) => images,
        Err(err) => {
            tracing::warn!("Failed to collect request images: {err:#}");
            Vec::new()
        }
    }
}

pub(crate) async fn try_extract_chat_image_data(
    request: &ChatCompletionRequest,
) -> Result<Vec<Vec<u8>>> {
    try_collect_image_data_with_limits(request.image_urls(), current_image_input_limits()).await
}

/// Environment variable holding the comma-separated list of canonical
/// directories permitted to be referenced via `file://` URIs and bare local
/// paths in `video_url` content blocks (issue #596).
///
/// Empty or unset = fail-closed: every local-filesystem video reference is
/// rejected. Operators must opt in explicitly to enable file uploads.
///
/// Issue #601 closed the canonicalise → ffmpeg-open TOCTOU window by
/// passing an opened fd down to ffmpeg. The startup helper
/// [`scan_insecure_allowlist_dirs`] still flags world- or group-writable
/// allowlist directories because a writable allowlist directory remains
/// a policy red flag — the operator should restrict it to mode 0750 or
/// stricter regardless.
pub(crate) const VIDEO_DIR_ALLOWLIST_ENV: &str = "MLXCEL_VIDEO_DIR_ALLOWLIST";

/// Resolve `video_url` content blocks from a chat request to opened video
/// handles (issue #553, wired up in #596, hardened in #601).
///
/// For `file://` and bare local paths the path is canonicalised and
/// validated against the directory allowlist from
/// `MLXCEL_VIDEO_DIR_ALLOWLIST`, then **opened read-only with `O_NOFOLLOW`
/// and the fd retained** (Unix only). The fd is what subprocesses ultimately
/// read from — never the path — so the dominant TOCTOU window
/// (canonicalise → allowlist check → metadata → open) is closed at the
/// kernel level (issue #601). `O_NOFOLLOW` hardens the narrowed
/// metadata→open window further: a symlink swap that occurs in that
/// microsecond gap causes the open to return `ELOOP` rather than silently
/// following the swapped link.
///
/// For `data:video/...;base64,...` and `http(s)://` we materialise the
/// bytes to a temporary file (because `ffmpeg` needs a seekable handle),
/// then open that file as an fd as well. The temp file is unlinked when
/// the [`ResolvedVideo::temp_guard`] drops.
///
/// On non-Unix targets the fd path is unavailable, so the resolver falls
/// back to the path-only variant. The chat-completion server is currently
/// only supported on Linux and macOS; this fallback exists for forward
/// compatibility, not as a current deployment target.
///
/// Stash the returned vector in a struct that lives for the full duration
/// of the request handler — when that struct drops, every fd is closed
/// and every temp file is removed automatically.
pub(crate) async fn extract_chat_video_paths(
    request: &ChatCompletionRequest,
) -> Vec<ResolvedVideo> {
    let allowlist = video_dir_allowlist_from_env();
    extract_chat_video_paths_with_allowlist(request, &allowlist).await
}

/// Test/internal-friendly variant of [`extract_chat_video_paths`] that
/// accepts the allowlist directly. Production code reads the env var via
/// the public wrapper above; tests inject a controlled allowlist so they
/// don't depend on process-wide state.
///
/// Returns a vector of [`ResolvedVideo`] records — one per `video_url`
/// content block that resolved successfully. Failed resolutions are
/// dropped silently with a `tracing::warn!` (the resolver is fail-closed
/// so the chat handler simply sees an empty `videos` list and short-circuits
/// to a 400 with "no videos to process").
pub(crate) async fn extract_chat_video_paths_with_allowlist(
    request: &ChatCompletionRequest,
    allowlist: &[PathBuf],
) -> Vec<ResolvedVideo> {
    let mut resolved = Vec::new();
    for video in request.video_urls() {
        if let Some(item) = resolve_video_url(&video, allowlist).await {
            resolved.push(item);
        }
    }
    resolved
}

/// Read [`VIDEO_DIR_ALLOWLIST_ENV`] and canonicalise each entry. Entries
/// that fail to canonicalise (typo, missing directory) are dropped with a
/// warning rather than poisoning the whole list.
///
/// An empty/unset env yields an empty `Vec`, which is the deliberate
/// fail-closed default — no `file://` URI or bare local path can resolve
/// until an operator opts in.
///
/// Callable from sync contexts (used by [`crate::server::startup`] for the
/// startup-time writability check). The hot path on the request side reads
/// the result via the async [`extract_chat_video_paths`] wrapper which then
/// passes it down by reference, so the blocking `std::fs::canonicalize` here
/// runs once at startup and once per request handler — both off the Tokio
/// hot loop, so it is acceptable to keep this synchronous.
pub(crate) fn video_dir_allowlist_from_env() -> Vec<PathBuf> {
    let raw = std::env::var(VIDEO_DIR_ALLOWLIST_ENV).unwrap_or_default();
    if raw.trim().is_empty() {
        return Vec::new();
    }
    raw.split(',')
        .filter_map(|entry| {
            let trimmed = entry.trim();
            if trimmed.is_empty() {
                return None;
            }
            match std::fs::canonicalize(trimmed) {
                Ok(canonical) => Some(canonical),
                Err(err) => {
                    tracing::warn!(
                        "{VIDEO_DIR_ALLOWLIST_ENV} entry {trimmed:?} could not be canonicalised: \
                         {err}; dropping from allowlist"
                    );
                    None
                }
            }
        })
        .collect()
}

/// Walk every directory in `allowlist` and return the subset whose Unix
/// permissions allow group or world write access (mode bits `0o022`).
///
/// Used by the server startup hook to surface a `tracing::warn!` when an
/// operator opts into the video-URL feature with a loose-mode directory.
/// The resolver in this module canonicalises, stats, and then **opens**
/// the file (issue #601), passing the fd down to ffmpeg via `/dev/fd/N`,
/// so the canonicalise → ffmpeg-open TOCTOU race is closed at the kernel
/// level. The startup warning remains as defence-in-depth: a writable
/// upload directory is still an operator-policy red flag (anyone with
/// shell access on the host can drop arbitrary files into the sandbox),
/// and restricting to mode `0750` or stricter is the recommended posture.
///
/// On non-Unix targets the file mode is unavailable, so this function
/// returns an empty `Vec`.
///
/// Returning the offending directories rather than emitting the warning
/// directly keeps the helper unit-testable: tests can construct a
/// world-writable temp directory and assert the helper detects it.
#[must_use]
pub(crate) fn scan_insecure_allowlist_dirs(allowlist: &[PathBuf]) -> Vec<PathBuf> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut insecure = Vec::new();
        for dir in allowlist {
            match std::fs::metadata(dir) {
                Ok(meta) => {
                    let mode = meta.permissions().mode();
                    // Group or other write bits (0o020, 0o002) are the
                    // dangerous flags. Owner-only writability (0o200) is fine.
                    if mode & 0o022 != 0 {
                        insecure.push(dir.clone());
                    }
                }
                Err(err) => {
                    tracing::warn!(
                        "Allowlist directory {dir:?} could not be stat'd for the security check: {err}",
                    );
                }
            }
        }
        insecure
    }
    #[cfg(not(unix))]
    {
        let _ = allowlist;
        Vec::new()
    }
}

/// Resolve a single [`VideoUrl`] to a [`ResolvedVideo`] record.
///
/// Returns `None` when the URL is unsupported, the file is missing, the
/// allowlist guard rejects it, or the open(2) syscall fails. The caller
/// drops the failure silently — the chat handler short-circuits with a
/// 400 when `videos` is empty.
async fn resolve_video_url(video: &VideoUrl, allowlist: &[PathBuf]) -> Option<ResolvedVideo> {
    let url = &video.url;
    let fps = video.fps;

    if url.starts_with("data:video/") {
        let (path, guard) = decode_video_data_uri(url).await?;
        let source = open_video_source(&path, url).await?;
        return Some(ResolvedVideo {
            source,
            fps,
            temp_guard: Some(guard),
        });
    }
    if let Some(path) = url.strip_prefix("file://") {
        let canonical = resolve_local_video_path(path, allowlist).await?;
        let source = open_video_source(&canonical, url).await?;
        return Some(ResolvedVideo {
            source,
            fps,
            temp_guard: None,
        });
    }
    if is_http_url(url) {
        let (path, guard) = fetch_remote_video(url).await?;
        let source = open_video_source(&path, url).await?;
        return Some(ResolvedVideo {
            source,
            fps,
            temp_guard: Some(guard),
        });
    }
    if let Some(canonical) = resolve_local_video_path(url, allowlist).await {
        let source = open_video_source(&canonical, url).await?;
        return Some(ResolvedVideo {
            source,
            fps,
            temp_guard: None,
        });
    }
    tracing::warn!("Unsupported video URL scheme or missing file: {}", url);
    None
}

/// Open `canonical` read-only and wrap the resulting fd in a
/// [`VideoSource`]. On non-Unix targets the resolver falls back to the
/// path-only variant.
///
/// This is the heart of the issue #601 TOCTOU fix: the resolver opens the
/// file once, retains the fd, and surrenders that fd (via `/dev/fd/N`) to
/// every subsequent ffmpeg/ffprobe invocation. Even if an attacker swaps
/// the underlying path for a symlink to `/etc/passwd` between this open
/// and the ffmpeg spawn, ffmpeg never re-reads the path — it consumes the
/// open file description we already validated.
///
/// On Unix the open is performed with `O_NOFOLLOW` so that a symlink swap
/// that occurs between [`resolve_local_video_path`]'s final `metadata`
/// call and this `open` causes the open to return `ELOOP` rather than
/// silently following the swapped-in symlink. The dominant TOCTOU window
/// (canonicalise → allowlist check → metadata → open) is closed at the
/// kernel level by `O_NOFOLLOW`; the residual window is now a narrow
/// kernel-internal race that requires an attacker to win a
/// compare-and-swap on the dentry simultaneously with the kernel's
/// `namei` walk, which is not feasible in practice.
///
/// The `original_url` parameter is solely for log diagnostics: when the
/// open syscall fails (race window between canonicalise and open, file
/// removed mid-request, etc.) the warning includes the originating URL so
/// operators can tie the failure back to a request body.
async fn open_video_source(canonical: &Path, original_url: &str) -> Option<VideoSource> {
    #[cfg(unix)]
    {
        let open_result = tokio::fs::OpenOptions::new()
            .read(true)
            // O_NOFOLLOW: if `canonical` is itself a symlink when the open
            // syscall reaches it, the kernel returns ELOOP instead of
            // following the link. Combined with the canonicalize + allowlist
            // + metadata checks above, this closes the metadata→open TOCTOU
            // window: a symlink swap that happens after the metadata call
            // causes a hard open failure rather than a silent misdirect.
            // `custom_flags` is available via the std::os::unix::fs::OpenOptionsExt
            // trait, which tokio::fs::OpenOptions delegates to internally.
            .custom_flags(libc::O_NOFOLLOW)
            .open(canonical)
            .await;
        match open_result {
            Ok(file) => {
                // Convert tokio File -> std File -> OwnedFd. The std File
                // owns the underlying OS handle; `into_std()` blocks
                // briefly waiting for any in-flight async I/O on the
                // tokio handle to drain (we issued no I/O yet, so this
                // is effectively a sync conversion).
                let std_file = file.into_std().await;
                let owned_fd = std::os::fd::OwnedFd::from(std_file);
                Some(VideoSource::from_fd(owned_fd, canonical.to_path_buf()))
            }
            Err(err) => {
                tracing::warn!(
                    "Failed to open canonicalised video path {:?} (from URL {}): {err}",
                    canonical,
                    original_url
                );
                None
            }
        }
    }
    #[cfg(not(unix))]
    {
        // No fd-based primitive on non-Unix; fall back to path. The
        // canonicalise / allowlist guards already validated the path,
        // and the residual TOCTOU window is unavoidable on platforms
        // without `/dev/fd/N` or `O_NOFOLLOW`. mlxcel server's threat
        // model targets Linux + macOS deployments, so this code path is
        // forward-compatibility scaffolding rather than an active
        // configuration.
        let _ = original_url;
        Some(VideoSource::from_path(canonical.to_path_buf()))
    }
}

/// Resolve a bare local path or the path component of a `file://` URI to a
/// canonical, allowlisted, regular video file (issue #596).
///
/// Async-friendly variant (#596 follow-up): uses `tokio::fs::canonicalize` and
/// `tokio::fs::metadata` so a slow disk or NFS mount cannot stall a Tokio
/// worker thread. Each `.await` boundary lets the runtime schedule other
/// requests while the lookup is in flight.
///
/// Rejection rules — every check is fail-closed:
/// * Empty allowlist → reject everything (operator opt-in required).
/// * `canonicalize` failure (missing file, broken symlink, permission) → reject.
/// * Canonical path not under any allowlist directory → reject. Symlinks pointing
///   outside the allowlist canonicalise to their target and are caught by this
///   check.
/// * Resolved target is not a regular file (directory, FIFO, device, socket) → reject.
/// * Filename extension is not in [`crate::multimodal::video::VIDEO_EXTENSIONS`] → reject.
async fn resolve_local_video_path(raw: &str, allowlist: &[PathBuf]) -> Option<PathBuf> {
    if allowlist.is_empty() {
        tracing::warn!(
            "{VIDEO_DIR_ALLOWLIST_ENV} is empty/unset; rejecting local video reference {raw:?}. \
             Set the env var to a comma-separated list of trusted directories to enable file uploads."
        );
        return None;
    }

    let path = Path::new(raw);

    let canonical = match tokio::fs::canonicalize(path).await {
        Ok(p) => p,
        Err(err) => {
            tracing::warn!("Failed to canonicalise video path {raw:?}: {err}");
            return None;
        }
    };

    if !allowlist.iter().any(|dir| canonical.starts_with(dir)) {
        tracing::warn!(
            "Video path {canonical:?} (from {raw:?}) is outside the allowlist; rejecting"
        );
        return None;
    }

    let metadata = match tokio::fs::metadata(&canonical).await {
        Ok(m) => m,
        Err(err) => {
            tracing::warn!("Failed to stat video path {canonical:?}: {err}");
            return None;
        }
    };
    if !metadata.is_file() {
        tracing::warn!("Video path {canonical:?} is not a regular file; rejecting");
        return None;
    }

    if !is_video_file(&canonical) {
        tracing::warn!(
            "Rejecting local video path with unsupported extension: {canonical:?} (from {raw:?})"
        );
        return None;
    }

    Some(canonical)
}

/// Maximum decoded video payload size: 1 GB. Mirrors the audio cap to
/// prevent OOM from extremely large base64 attachments.
pub(crate) const MAX_VIDEO_PAYLOAD_SIZE: usize = 1024 * 1024 * 1024;

async fn decode_video_data_uri(url: &str) -> Option<(PathBuf, TempFile)> {
    let Some((metadata, encoded_data)) = url.split_once(',') else {
        tracing::warn!("Invalid video data URI format");
        return None;
    };
    if !metadata.ends_with(";base64") {
        tracing::warn!("Unsupported video data URI encoding: {}", metadata);
        return None;
    }
    let bytes = match base64::engine::general_purpose::STANDARD.decode(encoded_data) {
        Ok(b) => b,
        Err(err) => {
            tracing::warn!("Failed to decode base64 video: {}", err);
            return None;
        }
    };
    if bytes.len() > MAX_VIDEO_PAYLOAD_SIZE {
        tracing::warn!(
            "Video payload too large ({} bytes, max {}); rejecting",
            bytes.len(),
            MAX_VIDEO_PAYLOAD_SIZE
        );
        return None;
    }
    write_video_temp_file(&bytes, infer_video_extension(metadata)).await
}

/// Fetch a remote video URL with size enforcement applied **incrementally**
/// (issue #596 hardening).
///
/// Implementation note — buffer-then-check is a DoS vector. `response.bytes()`
/// would read the entire response body into memory before we could enforce
/// `MAX_VIDEO_PAYLOAD_SIZE`. A hostile server could advertise no
/// `Content-Length` (or lie about it) and stream until our process OOMs.
///
/// Instead we use `bytes_stream()` and accumulate into a `Vec<u8>`, checking
/// the length cap after every chunk. The moment the accumulated total exceeds
/// the cap we drop everything and return `None`. This keeps peak memory
/// bounded by `MAX_VIDEO_PAYLOAD_SIZE` regardless of what the remote server
/// does with the wire protocol.
///
/// Connect timeout, total request timeout, and redirect cap are configured
/// on the shared client (see [`http_image_client`]).
async fn fetch_remote_video(url: &str) -> Option<(PathBuf, TempFile)> {
    let response = match http_image_client().get(url).send().await {
        Ok(r) => r,
        Err(err) => {
            tracing::warn!("Failed to fetch video URL {}: {}", url, err);
            return None;
        }
    };
    let response = match response.error_for_status() {
        Ok(r) => r,
        Err(err) => {
            tracing::warn!("Video URL returned error status {}: {}", url, err);
            return None;
        }
    };

    // Streaming accumulation with per-chunk size enforcement.
    let mut accumulated: Vec<u8> = Vec::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk_result) = stream.next().await {
        let chunk = match chunk_result {
            Ok(c) => c,
            Err(err) => {
                tracing::warn!("Failed to read video response body {}: {}", url, err);
                return None;
            }
        };
        // Reject *before* extending the buffer if the new chunk would exceed
        // the cap; this keeps peak memory bounded by MAX_VIDEO_PAYLOAD_SIZE +
        // one chunk size rather than 2x cap.
        if accumulated.len().saturating_add(chunk.len()) > MAX_VIDEO_PAYLOAD_SIZE {
            tracing::warn!(
                "Remote video too large (>{} bytes after streaming chunk, max {}); rejecting",
                accumulated.len() + chunk.len(),
                MAX_VIDEO_PAYLOAD_SIZE
            );
            // Drop the partial buffer; do not retain partially fetched bytes.
            drop(accumulated);
            return None;
        }
        accumulated.extend_from_slice(&chunk);
    }

    let ext = url
        .rsplit_once('.')
        .map(|(_, ext)| ext.split('?').next().unwrap_or(ext).to_string())
        .unwrap_or_else(|| "mp4".to_string());
    write_video_temp_file(&accumulated, sanitize_video_extension(&ext)).await
}

fn infer_video_extension(metadata: &str) -> &str {
    if metadata.contains("video/mp4") {
        "mp4"
    } else if metadata.contains("video/webm") {
        "webm"
    } else if metadata.contains("video/x-matroska") {
        "mkv"
    } else if metadata.contains("video/quicktime") {
        "mov"
    } else {
        "mp4"
    }
}

/// Write `bytes` to a fresh temp file and wrap the path in a [`TempFile`]
/// drop guard. The guard's `Drop` impl removes the file when the caller
/// drops it, so callers MUST keep the guard alive for as long as the file
/// must remain on disk (e.g., until ffmpeg has finished probing it).
///
/// Returning the guard alongside the path is the fix for the temp-file leak
/// reported during PR #600 review: every previous version returned a bare
/// `PathBuf` and relied on no-one to call cleanup explicitly. With the guard
/// in place a panic, an early-return error path, or a normal completion all
/// converge on the same cleanup behaviour.
async fn write_video_temp_file(bytes: &[u8], ext: &str) -> Option<(PathBuf, TempFile)> {
    let dir = std::env::temp_dir();
    let unique = uuid::Uuid::new_v4();
    let ext = sanitize_video_extension(ext);
    let path = dir.join(format!("mlxcel-video-{unique}.{ext}"));
    match tokio::fs::write(&path, bytes).await {
        Ok(()) => {
            let guard = TempFile::new(path.clone());
            Some((path, guard))
        }
        Err(err) => {
            tracing::warn!(
                "Failed to write video temp file {}: {}",
                path.display(),
                err
            );
            None
        }
    }
}

fn sanitize_video_extension(ext: &str) -> &'static str {
    match ext.trim_start_matches('.').to_ascii_lowercase().as_str() {
        "mp4" => "mp4",
        "webm" => "webm",
        "mkv" => "mkv",
        "mov" => "mov",
        "avi" => "avi",
        "m4v" => "m4v",
        "mpg" => "mpg",
        "mpeg" => "mpeg",
        _ => "mp4",
    }
}

/// Extract raw audio bytes from chat request audio inputs.
///
/// Supports base64-encoded inline data, `data:audio/...;base64,...` URIs,
/// `file://` paths, bare local paths, and `http(s)` URLs.
///
/// Only WAV format is currently supported; other formats are rejected early.
pub(crate) async fn extract_chat_audio_data(request: &ChatCompletionRequest) -> Vec<Vec<u8>> {
    let audio_inputs = request.audio_inputs();
    let mut audio_data = Vec::new();
    for input in &audio_inputs {
        if let Some(bytes) = read_audio_input(input).await {
            audio_data.push(bytes);
        }
    }
    audio_data
}

/// Maximum raw audio payload size after decoding: 500 MB.
/// This prevents OOM from extremely large base64 payloads before WAV
/// parsing can apply its own data-chunk limit.
const MAX_AUDIO_PAYLOAD_SIZE: usize = 500 * 1024 * 1024;

async fn read_audio_input(input: &InputAudio) -> Option<Vec<u8>> {
    // Validate format early -- only WAV is supported for now.
    if input.format != "wav" {
        tracing::warn!(
            "Unsupported audio format \'{}\'; only \'wav\' is currently supported",
            input.format
        );
        return None;
    }

    let data = &input.data;

    // data:audio/...;base64,... URI
    if data.starts_with("data:audio/") {
        return validate_audio_size(
            decode_data_uri_with_limit(data, "audio", MAX_AUDIO_PAYLOAD_SIZE)
                .map_err(|err| {
                    tracing::warn!("Failed to read audio data URI: {err:#}");
                    err
                })
                .ok(),
        );
    }

    // file:// prefix
    if let Some(path) = data.strip_prefix("file://") {
        return validate_audio_size(
            read_local_bytes_with_limit(Path::new(path), "audio", MAX_AUDIO_PAYLOAD_SIZE)
                .await
                .map_err(|err| {
                    tracing::warn!("Failed to read audio file {}: {err:#}", path);
                    err
                })
                .ok(),
        );
    }

    // HTTP(S) URL
    if is_http_url(data) {
        return validate_audio_size(
            fetch_remote_bytes_with_limit(data, "audio", MAX_AUDIO_PAYLOAD_SIZE)
                .await
                .map_err(|err| {
                    tracing::warn!("Failed to fetch audio URL {data}: {err:#}");
                    err
                })
                .ok(),
        );
    }

    // Bare local path
    if Path::new(data).is_file() {
        return validate_audio_size(
            read_local_bytes_with_limit(Path::new(data), "audio", MAX_AUDIO_PAYLOAD_SIZE)
                .await
                .map_err(|err| {
                    tracing::warn!("Failed to read audio file {data}: {err:#}");
                    err
                })
                .ok(),
        );
    }

    // Try as raw base64 data
    if data.len() > max_base64_encoded_len(MAX_AUDIO_PAYLOAD_SIZE) {
        tracing::warn!(
            "Audio base64 payload is too large to fit the {} byte limit; rejecting",
            MAX_AUDIO_PAYLOAD_SIZE
        );
        return None;
    }
    match base64::engine::general_purpose::STANDARD.decode(data) {
        Ok(bytes) if !bytes.is_empty() => validate_audio_size(Some(bytes)),
        _ => {
            tracing::warn!("Could not decode audio input data");
            None
        }
    }
}

/// Reject audio payloads that exceed `MAX_AUDIO_PAYLOAD_SIZE`.
fn validate_audio_size(data: Option<Vec<u8>>) -> Option<Vec<u8>> {
    match data {
        Some(bytes) if bytes.len() > MAX_AUDIO_PAYLOAD_SIZE => {
            tracing::warn!(
                "Audio payload too large ({} bytes, max {}); rejecting",
                bytes.len(),
                MAX_AUDIO_PAYLOAD_SIZE
            );
            None
        }
        other => other,
    }
}

#[cfg(test)]
pub(crate) async fn collect_image_data<I, S>(urls: I) -> Vec<Vec<u8>>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    match try_collect_image_data_with_limits(urls, current_image_input_limits()).await {
        Ok(images) => images,
        Err(err) => {
            tracing::warn!("Failed to collect image data: {err:#}");
            Vec::new()
        }
    }
}

pub(crate) async fn try_collect_image_data_with_limits<I, S>(
    urls: I,
    limits: ImageInputLimits,
) -> Result<Vec<Vec<u8>>>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut images = Vec::new();

    for (index, url) in urls.into_iter().enumerate() {
        if index >= limits.max_images_per_request {
            bail!(
                "Too many image inputs: received at least {}, maximum is {}",
                index + 1,
                limits.max_images_per_request
            );
        }
        match try_read_image_url_with_limits(url.as_ref(), limits).await {
            Ok(Some(bytes)) => images.push(bytes),
            Ok(None) => {}
            Err(err) => {
                tracing::warn!("Skipping invalid image input: {err:#}");
            }
        }
    }

    Ok(images)
}

#[cfg(test)]
pub(crate) async fn read_image_url(url: &str) -> Option<Vec<u8>> {
    match try_read_image_url_with_limits(url, current_image_input_limits()).await {
        Ok(bytes) => bytes,
        Err(err) => {
            tracing::warn!("Failed to read image URL: {err:#}");
            None
        }
    }
}

pub(crate) async fn try_read_image_url_with_limits(
    url: &str,
    limits: ImageInputLimits,
) -> Result<Option<Vec<u8>>> {
    if url.starts_with("data:image/") {
        return decode_data_uri_with_limit(url, "image", limits.max_payload_bytes).map(Some);
    }

    if let Some(path) = url.strip_prefix("file://") {
        return read_local_bytes_with_limit(Path::new(path), "image", limits.max_payload_bytes)
            .await
            .map(Some);
    }

    if is_http_url(url) {
        return fetch_remote_bytes_with_limit(url, "image", limits.max_payload_bytes)
            .await
            .map(Some);
    }

    if Path::new(url).is_file() {
        return read_local_bytes_with_limit(Path::new(url), "image", limits.max_payload_bytes)
            .await
            .map(Some);
    }

    tracing::warn!("Unsupported image URL scheme: {}", url);
    Ok(None)
}

fn decode_data_uri_with_limit(url: &str, kind: &str, max_size: usize) -> Result<Vec<u8>> {
    let Some((metadata, encoded_data)) = url.split_once(',') else {
        bail!("Invalid {kind} data URI format");
    };

    if !metadata.ends_with(";base64") {
        bail!("Unsupported {kind} data URI encoding: {metadata}");
    }

    let max_encoded_len = max_base64_encoded_len(max_size);
    if encoded_data.len() > max_encoded_len {
        bail!(
            "{kind} payload too large: base64 body is {} bytes, maximum decoded size is {} bytes",
            encoded_data.len(),
            max_size
        );
    }

    match base64::engine::general_purpose::STANDARD.decode(encoded_data) {
        Ok(bytes) => validate_payload_size(bytes, kind, max_size),
        Err(err) => Err(anyhow!("Failed to decode base64 {kind}: {err}")),
    }
}

fn max_base64_encoded_len(max_decoded_len: usize) -> usize {
    max_decoded_len.div_ceil(3).saturating_mul(4)
}

fn is_http_url(url: &str) -> bool {
    url.starts_with("http://") || url.starts_with("https://")
}

async fn fetch_remote_bytes_with_limit(url: &str, kind: &str, max_size: usize) -> Result<Vec<u8>> {
    let response = match http_image_client().get(url).send().await {
        Ok(response) => response,
        Err(err) => bail!("Failed to fetch {kind} URL {url}: {err}"),
    };

    let response = match response.error_for_status() {
        Ok(response) => response,
        Err(err) => bail!("{kind} URL returned error status {url}: {err}"),
    };

    if let Some(content_length) = response.content_length()
        && content_length > max_size as u64
    {
        bail!(
            "{kind} payload too large: HTTP content-length is {} bytes, maximum is {} bytes",
            content_length,
            max_size
        );
    }

    let mut bytes = Vec::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.with_context(|| format!("Failed to read {kind} response body {url}"))?;
        if bytes.len().saturating_add(chunk.len()) > max_size {
            bail!(
                "{kind} payload too large: response body exceeded {} bytes",
                max_size
            );
        }
        bytes.extend_from_slice(&chunk);
    }
    Ok(bytes)
}

async fn read_local_bytes_with_limit(path: &Path, kind: &str, max_size: usize) -> Result<Vec<u8>> {
    match tokio::fs::metadata(path).await {
        Ok(meta) if meta.len() > max_size as u64 => {
            bail!(
                "{kind} payload too large: file {} is {} bytes, maximum is {} bytes",
                path.display(),
                meta.len(),
                max_size
            );
        }
        Ok(_) => {}
        Err(err) => bail!("Failed to stat {kind} file {}: {err}", path.display()),
    }

    match tokio::fs::read(path).await {
        Ok(bytes) => validate_payload_size(bytes, kind, max_size),
        Err(err) => bail!("Failed to read {kind} file {}: {err}", path.display()),
    }
}

fn validate_payload_size(bytes: Vec<u8>, kind: &str, max_size: usize) -> Result<Vec<u8>> {
    if bytes.len() > max_size {
        bail!(
            "{kind} payload too large: {} bytes, maximum is {} bytes",
            bytes.len(),
            max_size
        );
    }
    Ok(bytes)
}

fn http_image_client() -> &'static reqwest::Client {
    static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
    CLIENT.get_or_init(|| {
        reqwest::Client::builder()
            // Total deadline for any single fetch.
            .timeout(Duration::from_secs(10))
            // Cap the time spent dialling a hostile or unreachable origin so
            // the request as a whole cannot stall longer than necessary.
            .connect_timeout(Duration::from_secs(5))
            // Bound redirect chains so a malicious origin cannot bounce the
            // client through unbounded hops.
            .redirect(reqwest::redirect::Policy::limited(5))
            .build()
            .expect("server image client should build")
    })
}

#[cfg(test)]
#[path = "media_tests.rs"]
mod tests;
