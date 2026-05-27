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

//! Generic VLM video utilities (issue #553, perf follow-up #597).
//!
//! Translates the upstream Python `mlx_vlm/video_generate.py` pipeline into
//! Rust so that any vision-language model can accept `--video` inputs through
//! the same code path used by `--image`. The module is intentionally
//! decoder-agnostic at the Rust level: frame extraction is delegated to the
//! `ffmpeg` system binary via a subprocess. We chose subprocess over the
//! `ffmpeg-next` Rust binding because:
//!
//! - `ffmpeg-next` requires `libavcodec` / `libavformat` dev headers at build
//!   time, which complicates Linux + CUDA CI containers and introduces a
//!   non-trivial system dependency.
//! - The runtime binary `ffmpeg` ships on every macOS Apple Silicon
//!   developer machine via Homebrew and on every standard Linux GPU image,
//!   so the deployment surface is already covered.
//! - We only need uniform frame sampling (no per-codec hot path), so the
//!   subprocess overhead is amortised across all frames in a single invocation.
//!
//! ## Single-pass extraction (issue #597)
//!
//! [`load_video`] now invokes `ffmpeg` exactly **once** per video, streaming
//! all sampled frames as a concatenated PNG sequence through stdout
//! (`-f image2pipe -vcodec png`). The PNG frame boundaries are detected by
//! parsing the PNG file format: each PNG begins with the 8-byte signature
//! `\x89PNG\r\n\x1a\n` and ends with the 12-byte IEND chunk. The parser
//! accumulates bytes until it sees a complete PNG and then decodes it in
//! place before moving on to the next frame. Peak memory during extraction
//! is bounded by approximately one decoded frame at a time.
//!
//! ## Resolution and duration caps (issue #597)
//!
//! Before decoding, [`probe_video`] checks:
//!
//! - `MLXCEL_VIDEO_MAX_PIXELS` (default `16_777_216` = 4096×4096): rejects
//!   source videos whose `width × height` exceeds the cap.
//! - `MLXCEL_VIDEO_MAX_DURATION_SEC` (default `600`): rejects videos whose
//!   FFprobe-reported duration exceeds the cap.
//!
//! Both caps are checked after the ffprobe call and before any ffmpeg decode
//! work starts. Rejection surfaces as [`VideoError::ResolutionTooLarge`] or
//! [`VideoError::DurationTooLong`] with the measured and allowed values
//! embedded in the message.
//!
//! ## Runtime requirements
//!
//! Both `ffmpeg` and `ffprobe` must be on PATH at runtime. They are not
//! build-time dependencies. Missing binaries produce a clear error via
//! [`VideoError::FfmpegMissing`] / [`VideoError::FfprobeMissing`].
//!
//! ## Drop guard for temp files
//!
//! Callers that create temp files for HTTP-fetched or base64-inline video
//! (currently in `src/server/media.rs`) should wrap the path in
//! [`TempFile`], which implements `Drop` with `fs::remove_file`. This
//! ensures cleanup even when the frame extraction panics partway through.
//! The single-pass implementation in this module does not write any temp
//! files itself, so no Drop guard is needed here.
//!
//! The module surfaces four primitives:
//!
//! - [`smart_nframes`] — uniform-sample target frame count per upstream
//!   [`smart_nframes`](https://github.com/Blaizzy/mlx-vlm) heuristics.
//! - [`load_video`] — decode a video file, sample frames, return a
//!   `Vec<DynamicImage>` ready for the existing image-tower preprocess.
//! - [`load_videos`] — multi-video convenience wrapper.
//! - [`is_video_file`] — extension-based detection (mp4/mov/webm/mkv/...).
//! - [`TempFile`] — RAII drop guard for transient temp-file paths.
//!
//! When `ffmpeg` is not present on the runtime PATH, [`load_video`] returns
//! a clear error so callers can degrade gracefully or surface a
//! configuration message to the user. Tests that exercise the subprocess
//! path are gated by [`ffmpeg_available`] and skip cleanly on machines
//! without `ffmpeg`.

use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

#[cfg(unix)]
use std::os::fd::{AsRawFd, OwnedFd};

use image::DynamicImage;

/// Default FPS target when callers do not specify `--fps`. Mirrors the
/// upstream Python `FPS = 2.0` constant.
pub const DEFAULT_FPS: f64 = 2.0;

/// Frame count must be an even multiple — vision towers downstream pair
/// frames. Mirrors upstream `FRAME_FACTOR = 2`.
pub const FRAME_FACTOR: usize = 2;

/// Lower bound on sampled frame count (mirrors upstream `FPS_MIN_FRAMES`).
pub const FPS_MIN_FRAMES: usize = 4;

/// Upper bound on sampled frame count (mirrors upstream `FPS_MAX_FRAMES`).
pub const FPS_MAX_FRAMES: usize = 768;

/// Default cap on source-video pixel count (width × height). Env var
/// `MLXCEL_VIDEO_MAX_PIXELS` overrides. Default = 4096 × 4096 = 16 777 216.
const DEFAULT_MAX_PIXELS: u64 = 4096 * 4096;

/// Default cap on source-video duration in seconds. Env var
/// `MLXCEL_VIDEO_MAX_DURATION_SEC` overrides. Default = 600 s (10 min).
const DEFAULT_MAX_DURATION_SEC: f64 = 600.0;

/// Default per-frame size cap for the PNG stream splitter. A single PNG frame
/// larger than this is almost certainly a malformed or malicious stream.
/// Env var `MLXCEL_VIDEO_MAX_PNG_FRAME_BYTES` overrides.
/// Default = 256 MiB.
const DEFAULT_MAX_PNG_FRAME_BYTES: usize = 256 * 1024 * 1024;

/// Resource caps for video decoding (resolution, duration, per-PNG-frame size).
///
/// Resolved once from the environment at the public entry point via
/// [`VideoLimits::from_env`] and threaded through the probe/extract path, so the
/// decode hot path never reads process-global `std::env`. This lets tests inject
/// explicit caps instead of calling `std::env::set_var`, which mutates the
/// process-global environment and races across the threaded test runner (and is
/// `unsafe` for exactly that reason) — see issue #103.
#[derive(Debug, Clone, Copy)]
pub struct VideoLimits {
    /// Max source pixel count (width × height).
    pub max_pixels: u64,
    /// Max source duration in seconds.
    pub max_duration_sec: f64,
    /// Max bytes accumulated for a single PNG frame before the stream is
    /// rejected as malformed.
    pub max_png_frame_bytes: usize,
}

impl Default for VideoLimits {
    /// The compile-time defaults (`DEFAULT_MAX_*`), used when the corresponding
    /// env var is unset or unparseable.
    fn default() -> Self {
        Self {
            max_pixels: DEFAULT_MAX_PIXELS,
            max_duration_sec: DEFAULT_MAX_DURATION_SEC,
            max_png_frame_bytes: DEFAULT_MAX_PNG_FRAME_BYTES,
        }
    }
}

impl VideoLimits {
    /// Resolve caps from `MLXCEL_VIDEO_MAX_{PIXELS,DURATION_SEC,PNG_FRAME_BYTES}`,
    /// each falling back to its compile-time default on a missing or unparseable
    /// value. This is the single place these env vars are read.
    pub fn from_env() -> Self {
        Self::from_raw(
            std::env::var("MLXCEL_VIDEO_MAX_PIXELS").ok().as_deref(),
            std::env::var("MLXCEL_VIDEO_MAX_DURATION_SEC")
                .ok()
                .as_deref(),
            std::env::var("MLXCEL_VIDEO_MAX_PNG_FRAME_BYTES")
                .ok()
                .as_deref(),
        )
    }

    /// Pure parser shared by [`from_env`](Self::from_env): maps already-fetched
    /// env strings to caps, falling back to the defaults. Kept env-free so it is
    /// unit-testable without mutating the process environment.
    fn from_raw(
        pixels: Option<&str>,
        duration_sec: Option<&str>,
        png_frame_bytes: Option<&str>,
    ) -> Self {
        let defaults = Self::default();
        Self {
            max_pixels: pixels
                .and_then(|v| v.parse().ok())
                .unwrap_or(defaults.max_pixels),
            max_duration_sec: duration_sec
                .and_then(|v| v.parse().ok())
                .unwrap_or(defaults.max_duration_sec),
            max_png_frame_bytes: png_frame_bytes
                .and_then(|v| v.parse().ok())
                .unwrap_or(defaults.max_png_frame_bytes),
        }
    }
}

/// PNG file format constants used by the single-pass stream splitter.
const PNG_SIGNATURE: &[u8; 8] = b"\x89PNG\r\n\x1a\n";
const PNG_IEND_CHUNK_LEN: usize = 12; // 4-byte length + 4-byte "IEND" + 4-byte CRC

/// Recognised video extensions for [`is_video_file`]. Lowercased.
const VIDEO_EXTENSIONS: &[&str] = &[
    ".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v", ".mpg", ".mpeg",
];

/// Errors surfaced by the video pipeline.
#[derive(Debug, thiserror::Error)]
pub enum VideoError {
    /// `ffmpeg` is not on PATH. Callers can fall back gracefully.
    #[error("ffmpeg binary not found on PATH; install ffmpeg to enable video inputs")]
    FfmpegMissing,
    /// `ffprobe` is not on PATH (ships with ffmpeg).
    #[error("ffprobe binary not found on PATH; install ffmpeg to enable video inputs")]
    FfprobeMissing,
    /// Video file does not exist on disk.
    #[error("video file not found: {0}")]
    FileNotFound(PathBuf),
    /// `ffprobe` could not parse the video container.
    #[error("ffprobe failed for {path}: {message}")]
    Probe { path: PathBuf, message: String },
    /// `ffmpeg` failed to extract frames.
    #[error("ffmpeg frame extraction failed for {path}: {message}")]
    Extract { path: PathBuf, message: String },
    /// No frames were decoded (corrupt file or empty stream).
    #[error("no frames decoded from video {0}")]
    EmptyVideo(PathBuf),
    /// Generic I/O failure surfacing from an underlying `Command`.
    #[error("video I/O error for {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    /// Decoded a frame but the bytes did not parse as a valid image.
    #[error("decoded frame from {path} is not a valid image: {source}")]
    Decode {
        path: PathBuf,
        #[source]
        source: image::ImageError,
    },
    /// Source video resolution exceeds the configured cap.
    ///
    /// Set env var `MLXCEL_VIDEO_MAX_PIXELS` to raise or lower the limit.
    #[error(
        "video source resolution {width}x{height} ({pixels} pixels) exceeds the cap of \
         {max_pixels} pixels (set MLXCEL_VIDEO_MAX_PIXELS to override)"
    )]
    ResolutionTooLarge {
        width: u32,
        height: u32,
        pixels: u64,
        max_pixels: u64,
    },
    /// Source video duration exceeds the configured cap.
    ///
    /// Set env var `MLXCEL_VIDEO_MAX_DURATION_SEC` to raise or lower the limit.
    #[error(
        "video duration {seconds:.1}s exceeds the cap of {max_seconds:.1}s \
         (set MLXCEL_VIDEO_MAX_DURATION_SEC to override)"
    )]
    DurationTooLong { seconds: f64, max_seconds: f64 },
}

// ─── RAII drop guard ────────────────────────────────────────────────────────

/// Panic-safe RAII guard that deletes a temporary file when dropped.
///
/// Construct with [`TempFile::new`] after the file has been created. The
/// guard is intentionally transparent — callers borrow the inner path via
/// [`TempFile::path`] to pass it to `ffmpeg` or other callers, then let
/// the guard go out of scope (or be dropped explicitly) when done.
///
/// If `fs::remove_file` fails on drop a warning is emitted but the error is
/// otherwise swallowed — temp-file cleanup failure is not fatal.
///
/// # Example
/// ```no_run
/// use std::path::PathBuf;
/// use mlxcel::multimodal::video::TempFile;
///
/// let tmp = TempFile::new(PathBuf::from("/tmp/mlxcel-video-abc123.mp4"));
/// let path = tmp.path(); // borrow to ffmpeg
/// // drop(tmp) removes the file, even if earlier code panicked.
/// ```
#[derive(Debug)]
pub struct TempFile {
    path: PathBuf,
}

impl TempFile {
    /// Wrap an already-created temporary file path in the guard.
    #[must_use]
    pub fn new(path: PathBuf) -> Self {
        Self { path }
    }

    /// Borrow the inner path for passing to subprocesses or I/O functions.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        if let Err(err) = std::fs::remove_file(&self.path) {
            // Only warn if the file still exists — it may have been cleaned up
            // already by an earlier success path.
            if self.path.exists() {
                tracing::warn!("TempFile: failed to remove {:?}: {}", self.path, err);
            }
        }
    }
}

// ─── VideoSource (issue #601): TOCTOU-safe video handle ─────────────────────

/// A video input that can be safely passed to `ffmpeg` / `ffprobe` without
/// re-opening by path.
///
/// On Unix, the [`VideoSource::Fd`] variant carries an owned read-only file
/// descriptor produced by [`crate::server::media::extract_chat_video_paths`]
/// after canonicalisation, allowlist prefix checks, regular-file stats, and
/// extension filtering have all succeeded. The fd is then surrendered to
/// ffmpeg via `/dev/fd/N` so ffmpeg never re-opens the underlying path.
/// This closes the TOCTOU window where an attacker with write access inside
/// an allowlisted directory could swap the validated file for a symlink to
/// `/etc/passwd` (or another out-of-sandbox secret) between the resolver's
/// `canonicalize` call and ffmpeg's `open`.
///
/// The [`VideoSource::Path`] variant remains for callers that legitimately
/// need path-based access — most importantly the `mlxcel` CLI, which receives
/// paths from `--video` flags chosen by the operator at the local terminal
/// and is not running in a multi-tenant, untrusted-input setting.
///
/// On non-Unix platforms only the `Path` variant is constructible; the fd
/// path is gated behind `#[cfg(unix)]` because `/dev/fd/N` is a Unix-only
/// kernel interface. This is acceptable because mlxcel server's threat
/// model targets Linux and macOS deployments only.
#[derive(Debug)]
pub enum VideoSource {
    /// A bare filesystem path. Subprocesses re-open this path. Suitable for
    /// trusted-input contexts (CLI arguments, internal callers).
    Path(PathBuf),

    /// An owned read-only file descriptor plus the canonical path it was
    /// opened from (the path is retained for diagnostics only — subprocesses
    /// receive `/dev/fd/N`, never the canonical path). Available on Unix
    /// targets only (Linux, macOS).
    #[cfg(unix)]
    Fd {
        /// The owned read-only file descriptor. Closed when [`VideoSource`]
        /// drops. The fd is opened with `O_CLOEXEC` set on the parent side;
        /// the cloexec flag is cleared inside [`Command::pre_exec`] before
        /// the child execs ffmpeg/ffprobe so the fd is inherited by the
        /// child and `/dev/fd/N` resolves successfully.
        fd: OwnedFd,
        /// Canonical path the fd was opened from. Used solely for error
        /// messages and the resolved-path piece of `(path, fps)` tuples
        /// returned to callers that still need a path-shaped identity. The
        /// path is never opened again.
        canonical: PathBuf,
    },
}

impl VideoSource {
    /// Construct a path-only video source. Used by CLI / non-allowlist
    /// callers and on non-Unix targets.
    #[must_use]
    pub fn from_path(path: PathBuf) -> Self {
        VideoSource::Path(path)
    }

    /// Construct an fd-backed video source from an already-opened
    /// [`OwnedFd`] and its canonical path. Unix only.
    #[cfg(unix)]
    #[must_use]
    pub fn from_fd(fd: OwnedFd, canonical: PathBuf) -> Self {
        VideoSource::Fd { fd, canonical }
    }

    /// Return the canonical path identity of this source.
    ///
    /// For [`VideoSource::Path`] this is the input path verbatim. For the
    /// Unix fd variant this is the canonical path the fd was originally
    /// opened from — used for error messages and as a stable identifier
    /// the rest of the request handler can carry around. **Do not pass
    /// this to a subprocess** when the source is fd-backed: the subprocess
    /// must receive `/dev/fd/N` (see [`VideoSource::ffmpeg_input`]) so it
    /// reads from the validated open file description rather than the
    /// path, which an attacker could have swapped post-resolution.
    #[must_use]
    pub fn canonical_path(&self) -> &Path {
        match self {
            VideoSource::Path(p) => p.as_path(),
            #[cfg(unix)]
            VideoSource::Fd { canonical, .. } => canonical.as_path(),
        }
    }

    /// Return the input string to pass to `ffmpeg -i` / `ffprobe`.
    ///
    /// For [`VideoSource::Path`] this is the path as a [`PathBuf`]. For the
    /// Unix fd variant this is `/dev/fd/N` where `N` is the integer fd. On
    /// Linux the kernel resolves `/dev/fd/N` via the magic-symlink in
    /// `/proc/self/fd`; on macOS `/dev/fd/N` is a built-in special device.
    /// Both yield a new file description bound to the same underlying open
    /// file, so the child ffmpeg process reads the bytes of the file the
    /// parent already validated.
    fn ffmpeg_input(&self) -> PathBuf {
        match self {
            VideoSource::Path(p) => p.clone(),
            #[cfg(unix)]
            VideoSource::Fd { fd, .. } => PathBuf::from(format!("/dev/fd/{}", fd.as_raw_fd())),
        }
    }

    /// Return `true` when this source carries a writable fd whose offset
    /// must be reset before each subprocess. The path-variant has no such
    /// requirement (each subprocess opens the file fresh, with its own
    /// offset).
    #[cfg(unix)]
    fn needs_offset_reset(&self) -> bool {
        matches!(self, VideoSource::Fd { .. })
    }

    /// Reset the file offset of the fd (if any) to 0 before invoking a
    /// subprocess.
    ///
    /// On Linux, opening `/dev/fd/N` from a child process duplicates the
    /// **same open file description** as the parent's master fd (per
    /// `proc(5)`: "the new file descriptor refers to the same OFD as the
    /// corresponding entry in /proc/[pid]/fd, and thus shares the same file
    /// offset"). macOS `/dev/fd/N` follows the same dup-style semantics.
    /// Consequently a previous ffprobe invocation that left the offset at
    /// the end of the moov atom would cause the next ffmpeg invocation to
    /// read from that offset rather than from byte 0.
    ///
    /// We seek the master fd back to 0 in the parent before each spawn so
    /// each ffmpeg/ffprobe child sees the file from the start.
    #[cfg(unix)]
    fn rewind(&self) -> io::Result<()> {
        if let VideoSource::Fd { fd, .. } = self {
            // SAFETY: `lseek` on a valid raw fd is async-signal-safe and
            // does not invalidate the OwnedFd. The fd is still owned by
            // `self`; the libc call only adjusts kernel-side offset state.
            let raw = fd.as_raw_fd();
            let r = unsafe { libc::lseek(raw, 0, libc::SEEK_SET) };
            if r < 0 {
                return Err(io::Error::last_os_error());
            }
        }
        Ok(())
    }

    /// Configure a [`Command`] so the child can read from this source.
    ///
    /// For [`VideoSource::Path`] this is a no-op — the child receives the
    /// path on the command line and opens it via the normal path-lookup
    /// rules. For the Unix fd variant this installs a `pre_exec` hook
    /// that clears `FD_CLOEXEC` on the inherited fd so the kernel keeps
    /// the fd open across `execve`, and the child can then `open(/dev/fd/N)`
    /// successfully.
    #[cfg(unix)]
    fn configure_child(&self, command: &mut Command) {
        if let VideoSource::Fd { fd, .. } = self {
            use std::os::unix::process::CommandExt;
            let raw_fd = fd.as_raw_fd();
            // SAFETY: This closure is executed in the child process after
            // `fork()` and before `execve()`. In this single-threaded
            // post-fork environment, only async-signal-safe functions may
            // be called (POSIX.1-2017 §2.4.3). The contract is met:
            //
            // 1. `libc::fcntl(F_GETFD)` / `libc::fcntl(F_SETFD, ...)` are
            //    listed in the POSIX async-signal-safe table (they perform a
            //    pure kernel-side fd-flags read/write with no heap interaction).
            //
            // 2. `io::Error::last_os_error()` reads `errno` and stores the
            //    integer value in an `io::Error` struct. Despite the `Error`
            //    wrapper, this particular constructor **does not allocate**:
            //    `io::Error::from_raw_os_error` stores the `i32` inline and
            //    only wraps it in a heap-allocated message on `Display`/`Debug`
            //    formatting, which does not happen in the pre-exec closure.
            //    The return path therefore satisfies the no-malloc constraint.
            //
            // 3. The captured `raw_fd` is a plain `i32` (copy type). No Rust
            //    drop glue runs for it in the child, so there is no risk of a
            //    double-free or destructor calling allocator functions.
            //
            // 4. Only the child's fd table is mutated; `fork` copies the fd
            //    table by value so the parent's `OwnedFd` is unaffected.
            unsafe {
                command.pre_exec(move || {
                    let flags = libc::fcntl(raw_fd, libc::F_GETFD);
                    if flags < 0 {
                        return Err(io::Error::last_os_error());
                    }
                    let new_flags = flags & !libc::FD_CLOEXEC;
                    if libc::fcntl(raw_fd, libc::F_SETFD, new_flags) < 0 {
                        return Err(io::Error::last_os_error());
                    }
                    Ok(())
                });
            }
        }
    }

    /// Configure a child Command on non-Unix platforms.
    ///
    /// No-op shim so the call sites in [`probe_video`] and
    /// [`extract_frames_single_pass`] do not need explicit cfg gating.
    /// Constructing a [`VideoSource::Fd`] is impossible on non-Unix targets,
    /// so all paths reduce to the path-only variant.
    #[cfg(not(unix))]
    fn configure_child(&self, _command: &mut Command) {}
}

impl From<PathBuf> for VideoSource {
    fn from(path: PathBuf) -> Self {
        VideoSource::Path(path)
    }
}

impl From<&Path> for VideoSource {
    fn from(path: &Path) -> Self {
        VideoSource::Path(path.to_path_buf())
    }
}

// ─── Helper functions ────────────────────────────────────────────────────────

/// Round `value` to the nearest multiple of `factor`. Mirrors
/// `round_by_factor` from `video_generate.py`.
#[inline]
fn round_by_factor(value: f64, factor: usize) -> usize {
    let factor_f = factor as f64;
    ((value / factor_f).round() as usize) * factor
}

/// Floor `value` to the nearest multiple of `factor` from below. Mirrors
/// `floor_by_factor`.
#[inline]
fn floor_by_factor(value: f64, factor: usize) -> usize {
    let factor_f = factor as f64;
    ((value / factor_f).floor() as usize) * factor
}

/// Ceil `value` to the nearest multiple of `factor` from above. Mirrors
/// `ceil_by_factor`.
#[inline]
fn ceil_by_factor(value: f64, factor: usize) -> usize {
    let factor_f = factor as f64;
    ((value / factor_f).ceil() as usize) * factor
}

/// Return true when `path` has a video file extension.
///
/// Detection is purely extension-based and does not open the file. Used by
/// the CLI to dispatch between image and video preprocessing without a
/// container probe round-trip.
#[must_use]
pub fn is_video_file(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
        return false;
    };
    let lower = name.to_lowercase();
    VIDEO_EXTENSIONS.iter().any(|ext| lower.ends_with(ext))
}

/// Compute the number of frames to sample given the source video's frame
/// count and FPS, with optional caller-supplied target FPS / explicit
/// frame count overrides. Mirrors upstream `smart_nframes`.
///
/// Behavior:
/// - When `target_nframes == Some(n)`, returns `round_by_factor(n,
///   FRAME_FACTOR)` clamped to `[FRAME_FACTOR, total_frames]`.
/// - Otherwise computes `total_frames / video_fps * target_fps`, clamps
///   to `[FPS_MIN_FRAMES, min(FPS_MAX_FRAMES, total_frames)]`, then
///   floors to a multiple of `FRAME_FACTOR`.
///
/// # Errors
/// Returns `Err` when:
/// - `total_frames < FRAME_FACTOR` (can't sample at least 2 frames).
/// - The clamped result still does not fit in `[FRAME_FACTOR, total_frames]`.
pub fn smart_nframes(
    total_frames: usize,
    video_fps: f64,
    target_fps: Option<f64>,
    target_nframes: Option<usize>,
) -> Result<usize, VideoError> {
    if total_frames < FRAME_FACTOR {
        return Err(VideoError::Extract {
            path: PathBuf::new(),
            message: format!(
                "video has only {total_frames} frame(s); need at least {FRAME_FACTOR}"
            ),
        });
    }

    if let Some(n) = target_nframes {
        let nframes = round_by_factor(n as f64, FRAME_FACTOR);
        if nframes < FRAME_FACTOR || nframes > total_frames {
            return Err(VideoError::Extract {
                path: PathBuf::new(),
                message: format!("nframes={nframes} out of range [{FRAME_FACTOR}, {total_frames}]"),
            });
        }
        return Ok(nframes);
    }

    let fps = target_fps.unwrap_or(DEFAULT_FPS);
    if fps <= 0.0 {
        return Err(VideoError::Extract {
            path: PathBuf::new(),
            message: format!("target fps must be > 0; got {fps}"),
        });
    }
    let video_fps_safe = if video_fps > 0.0 { video_fps } else { 1.0 };

    let raw = total_frames as f64 / video_fps_safe * fps;
    let max_cap = ceil_by_factor(FPS_MIN_FRAMES as f64, FRAME_FACTOR).max(FRAME_FACTOR);
    let upper = floor_by_factor(FPS_MAX_FRAMES.min(total_frames) as f64, FRAME_FACTOR);
    let upper = upper.max(max_cap);

    let bounded = raw
        .max(max_cap as f64)
        .min(upper as f64)
        .min(total_frames as f64);
    let nframes = floor_by_factor(bounded, FRAME_FACTOR);
    let nframes = nframes.max(FRAME_FACTOR).min(total_frames);
    if !(FRAME_FACTOR..=total_frames).contains(&nframes) {
        return Err(VideoError::Extract {
            path: PathBuf::new(),
            message: format!("nframes={nframes} out of range [{FRAME_FACTOR}, {total_frames}]"),
        });
    }
    Ok(nframes)
}

/// Return true if `ffmpeg` and `ffprobe` are both invokable. Cached with
/// `OnceLock` so repeated calls do not re-fork.
pub fn ffmpeg_available() -> bool {
    use std::sync::OnceLock;
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        let ff = Command::new("ffmpeg")
            .arg("-version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        let fp = Command::new("ffprobe")
            .arg("-version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        ff && fp
    })
}

/// Full metadata returned by [`probe_video`].
///
/// Fields beyond `total_frames` and `fps` are used for cap checks during
/// probing; `width`, `height`, and `duration_sec` are stored for potential
/// caller inspection but are currently only read inside `probe_video` itself.
#[derive(Debug, Clone)]
struct VideoMeta {
    total_frames: usize,
    fps: f64,
    /// Source width in pixels. Used for the resolution cap check.
    #[allow(dead_code)]
    width: u32,
    /// Source height in pixels. Used for the resolution cap check.
    #[allow(dead_code)]
    height: u32,
    /// Source duration in seconds. Used for the duration cap check.
    #[allow(dead_code)]
    duration_sec: f64,
}

/// Probe a video container with `ffprobe`, returning full metadata.
///
/// Rejects inputs whose resolution or duration exceed the configured caps
/// before returning. The caps are read from environment variables once per
/// process:
///
/// - `MLXCEL_VIDEO_MAX_PIXELS` — product of width × height (default 16 777 216)
/// - `MLXCEL_VIDEO_MAX_DURATION_SEC` — duration in seconds (default 600)
///
/// Accepts a [`VideoSource`] so the resolver in `src/server/media.rs` can
/// pass an opened fd (issue #601) instead of the canonical path. The fd
/// path closes the TOCTOU window between the resolver's `canonicalize`
/// and ffmpeg's `open`.
fn probe_video(source: &VideoSource, limits: &VideoLimits) -> Result<VideoMeta, VideoError> {
    let canonical_path = source.canonical_path();
    // For path-variant sources we still verify the path exists; for fd-variant
    // sources the fd was opened by the resolver and the file is guaranteed
    // accessible (we are reading from the open file description, not the path).
    if matches!(source, VideoSource::Path(_)) && !canonical_path.exists() {
        return Err(VideoError::FileNotFound(canonical_path.to_path_buf()));
    }
    // Reset the master fd offset to 0 (no-op for path sources) so ffprobe
    // sees the file from the beginning even if a previous probe invocation
    // had left the offset advanced. See `VideoSource::rewind` for rationale.
    #[cfg(unix)]
    if source.needs_offset_reset()
        && let Err(err) = source.rewind()
    {
        return Err(VideoError::Io {
            path: canonical_path.to_path_buf(),
            source: err,
        });
    }

    let mut cmd = Command::new("ffprobe");
    cmd.args([
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_frames,avg_frame_rate,r_frame_rate,duration,width,height",
        "-of",
        "default=noprint_wrappers=1",
    ])
    .arg(source.ffmpeg_input());
    source.configure_child(&mut cmd);
    let output = cmd.output().map_err(|err| {
        if err.kind() == std::io::ErrorKind::NotFound {
            VideoError::FfprobeMissing
        } else {
            VideoError::Io {
                path: canonical_path.to_path_buf(),
                source: err,
            }
        }
    })?;
    if !output.status.success() {
        return Err(VideoError::Probe {
            path: canonical_path.to_path_buf(),
            message: String::from_utf8_lossy(&output.stderr).into_owned(),
        });
    }
    let text = String::from_utf8_lossy(&output.stdout);

    let mut nb_frames: Option<usize> = None;
    let mut avg_fps: Option<f64> = None;
    let mut r_fps: Option<f64> = None;
    let mut duration: Option<f64> = None;
    let mut width: Option<u32> = None;
    let mut height: Option<u32> = None;

    for line in text.lines() {
        let Some((key, raw_value)) = line.split_once('=') else {
            continue;
        };
        let value = raw_value.trim();
        match key.trim() {
            "nb_frames" => {
                if let Ok(n) = value.parse::<usize>() {
                    nb_frames = Some(n);
                }
            }
            "avg_frame_rate" => avg_fps = parse_rational(value),
            "r_frame_rate" => r_fps = parse_rational(value),
            "duration" => {
                if let Ok(d) = value.parse::<f64>() {
                    duration = Some(d);
                }
            }
            "width" => {
                if let Ok(w) = value.parse::<u32>() {
                    width = Some(w);
                }
            }
            "height" => {
                if let Ok(h) = value.parse::<u32>() {
                    height = Some(h);
                }
            }
            _ => {}
        }
    }

    let fps = avg_fps
        .or(r_fps)
        .filter(|f| f.is_finite() && *f > 0.0)
        .unwrap_or(1.0);

    apply_probe_caps(
        canonical_path,
        nb_frames,
        fps,
        width,
        height,
        duration,
        limits,
    )
}

/// Enforce resolution and duration caps on raw ffprobe field values, returning
/// a validated [`VideoMeta`].
///
/// This function is separated from [`probe_video`] so that unit tests can
/// exercise the fail-closed sentinel behaviour with hand-crafted inputs,
/// without needing a real video file or a real `ffprobe` invocation.
///
/// ## Fail-closed defaults
///
/// - `width` / `height` missing → `u32::MAX` so the pixel cap trips.
/// - `duration` missing or non-finite → `+∞` so the duration cap trips.
///
/// Pixel overflow is prevented by `saturating_mul` so `u32::MAX × u32::MAX`
/// saturates to `u64::MAX` rather than wrapping back to `0`.
fn apply_probe_caps(
    path: &Path,
    nb_frames: Option<usize>,
    fps: f64,
    width: Option<u32>,
    height: Option<u32>,
    duration: Option<f64>,
    limits: &VideoLimits,
) -> Result<VideoMeta, VideoError> {
    // If ffprobe did not report duration, default to +∞ so the duration cap
    // trips immediately rather than silently bypassing the check.
    let duration_sec = duration
        .filter(|d| d.is_finite() && *d > 0.0)
        .unwrap_or(f64::INFINITY);

    // Some containers return nb_frames="N/A". Fall back to duration*fps.
    // When duration_sec is +∞ (missing field) this branch is skipped and the
    // error below fires, but the guard keeps the logic explicit.
    let total_frames = if let Some(n) = nb_frames.filter(|n| *n > 0) {
        n
    } else if duration_sec.is_finite() && duration_sec > 0.0 {
        ((duration_sec * fps).round() as usize).max(1)
    } else {
        return Err(VideoError::Probe {
            path: path.to_path_buf(),
            message: "could not determine frame count from ffprobe".to_string(),
        });
    };

    // If ffprobe did not report width or height, default to u32::MAX so that
    // the pixel cap (default 16M) trips immediately rather than silently
    // bypassing the resolution guard.
    let w = width.unwrap_or(u32::MAX);
    let h = height.unwrap_or(u32::MAX);

    // ── Resolution cap check ─────────────────────────────────────────────
    let max_pixels = limits.max_pixels;
    // Use saturating_mul so that u32::MAX * u32::MAX saturates to u64::MAX
    // instead of overflowing back to 0 and silently bypassing the cap.
    let pixels = (w as u64).saturating_mul(h as u64);
    if pixels > max_pixels {
        return Err(VideoError::ResolutionTooLarge {
            width: w,
            height: h,
            pixels,
            max_pixels,
        });
    }

    // ── Duration cap check ───────────────────────────────────────────────
    let max_duration = limits.max_duration_sec;
    if duration_sec > max_duration {
        return Err(VideoError::DurationTooLong {
            seconds: duration_sec,
            max_seconds: max_duration,
        });
    }

    Ok(VideoMeta {
        total_frames,
        fps,
        width: w,
        height: h,
        duration_sec,
    })
}

/// Parse `"30000/1001"` style rationals returned by ffprobe.
fn parse_rational(value: &str) -> Option<f64> {
    if let Some((num, den)) = value.split_once('/') {
        let num: f64 = num.trim().parse().ok()?;
        let den: f64 = den.trim().parse().ok()?;
        if den.abs() < f64::EPSILON {
            None
        } else {
            Some(num / den)
        }
    } else {
        value.parse().ok()
    }
}

/// Decode a video file and return uniformly-sampled frames as RGB images.
///
/// This function invokes `ffmpeg` exactly **once** per video. Frames are
/// streamed out of ffmpeg's stdout as a concatenated PNG sequence (using
/// `-f image2pipe -vcodec png`). The PNG frame boundaries are detected by
/// the [`split_png_stream`] parser, which reads the stream byte-by-byte
/// looking for PNG IEND chunk terminators. Each decoded [`DynamicImage`] is
/// pushed into the result vector before the next frame is decoded, so peak
/// memory during extraction is approximately one frame at a time.
///
/// Sampling matches the upstream Python pipeline:
/// 1. Probe `total_frames`, `video_fps`, `width`, `height`, and `duration`
///    via `ffprobe`, applying resolution and duration caps.
/// 2. Compute `nframes = smart_nframes(total_frames, video_fps, target_fps, ..)`.
/// 3. Build a `select` filter expression to extract exactly the desired frames.
/// 4. Pipe all selected frames through a single `ffmpeg` invocation.
///
/// # Resolution and duration caps
///
/// The input is rejected early (before any decode work) if:
/// - `width × height > MLXCEL_VIDEO_MAX_PIXELS` (default 16 777 216)
/// - `duration > MLXCEL_VIDEO_MAX_DURATION_SEC` (default 600)
///
/// Both caps are configurable via environment variables.
///
/// # Errors
/// Returns [`VideoError`] when `ffmpeg`/`ffprobe` is missing, the file
/// does not exist, the container cannot be probed, a cap is exceeded, or
/// no frames can be decoded.
///
/// # Async warning
/// This function blocks the calling thread while running `ffprobe` and
/// `ffmpeg` subprocesses and reading their stdout. Callers running inside an
/// async runtime (e.g. Tokio) **must** wrap this call in
/// `tokio::task::spawn_blocking` to avoid starving the executor:
///
/// ```ignore
/// let frames = tokio::task::spawn_blocking(move || {
///     mlxcel::multimodal::video::load_video(&path, target_fps, target_nframes)
/// }).await??;
/// ```
///
/// # Path vs fd input (issue #601)
///
/// This function operates on a [`Path`]. Server callers that source the
/// video from an untrusted user request should instead use
/// [`load_video_source`] with a [`VideoSource::Fd`] produced by the media
/// resolver — that closes the canonicalize → ffmpeg-open TOCTOU window
/// described in `extract_chat_video_paths_with_allowlist`.
pub fn load_video(
    path: &Path,
    target_fps: Option<f64>,
    target_nframes: Option<usize>,
) -> Result<Vec<DynamicImage>, VideoError> {
    let source = VideoSource::from_path(path.to_path_buf());
    load_video_source(&source, target_fps, target_nframes)
}

/// Like [`load_video`] but with explicit resource [`VideoLimits`] instead of
/// resolving them from the environment.
pub fn load_video_with_limits(
    path: &Path,
    target_fps: Option<f64>,
    target_nframes: Option<usize>,
    limits: &VideoLimits,
) -> Result<Vec<DynamicImage>, VideoError> {
    let source = VideoSource::from_path(path.to_path_buf());
    load_video_source_with_limits(&source, target_fps, target_nframes, limits)
}

/// Decode a [`VideoSource`] into uniformly-sampled frames.
///
/// Equivalent to [`load_video`] but accepts a [`VideoSource`] so the
/// fd-backed variant (used by the chat-completion video resolver, issue
/// #601) can pipe the file through `/dev/fd/N` instead of re-opening the
/// canonical path. See [`VideoSource`] for the security rationale.
pub fn load_video_source(
    source: &VideoSource,
    target_fps: Option<f64>,
    target_nframes: Option<usize>,
) -> Result<Vec<DynamicImage>, VideoError> {
    load_video_source_with_limits(source, target_fps, target_nframes, &VideoLimits::from_env())
}

/// Like [`load_video_source`] but with explicit resource [`VideoLimits`]
/// instead of resolving them from the environment. Lets callers (and tests)
/// pin the caps deterministically; [`load_video_source`] is the env-resolving
/// convenience wrapper over this.
pub fn load_video_source_with_limits(
    source: &VideoSource,
    target_fps: Option<f64>,
    target_nframes: Option<usize>,
    limits: &VideoLimits,
) -> Result<Vec<DynamicImage>, VideoError> {
    if !ffmpeg_available() {
        return Err(VideoError::FfmpegMissing);
    }
    let canonical = source.canonical_path().to_path_buf();
    let meta = probe_video(source, limits)?;
    let nframes =
        smart_nframes(meta.total_frames, meta.fps, target_fps, target_nframes).map_err(|err| {
            match err {
                VideoError::Extract { message, .. } => VideoError::Extract {
                    path: canonical.clone(),
                    message,
                },
                other => other,
            }
        })?;

    let indices = uniform_indices(meta.total_frames, nframes);
    let frames = extract_frames_single_pass(source, &indices, meta.fps, limits)?;

    if frames.is_empty() {
        return Err(VideoError::EmptyVideo(canonical));
    }
    Ok(frames)
}

/// Convenience: load multiple videos by path, returning one frame vector per video.
///
/// Errors short-circuit on the first failure. Callers that need
/// per-video error tolerance should call [`load_video`] in a loop and
/// handle the [`Result`] themselves.
pub fn load_videos(
    paths: &[PathBuf],
    target_fps: Option<f64>,
    target_nframes: Option<usize>,
) -> Result<Vec<Vec<DynamicImage>>, VideoError> {
    let mut all = Vec::with_capacity(paths.len());
    for path in paths {
        all.push(load_video(path, target_fps, target_nframes)?);
    }
    Ok(all)
}

/// Compute `nframes` evenly-spaced frame indices across
/// `[0, total_frames - 1]`. Mirrors the Python `np.linspace(...).round()
/// .astype(int)` call used in `load_video`.
fn uniform_indices(total_frames: usize, nframes: usize) -> Vec<usize> {
    if nframes <= 1 {
        return vec![0];
    }
    let last = total_frames.saturating_sub(1) as f64;
    let step = last / (nframes as f64 - 1.0);
    (0..nframes)
        .map(|i| (i as f64 * step).round() as usize)
        .map(|i| i.min(total_frames.saturating_sub(1)))
        .collect()
}

/// Extract the requested frames in a **single** `ffmpeg` invocation.
///
/// Uses the ffmpeg `select` filter to pick exactly the frames at `indices`
/// and pipes the output as a concatenated PNG sequence to stdout. The
/// stdout stream is split into individual PNG files by [`split_png_stream`]
/// and each is decoded in turn by the `image` crate.
///
/// ## Single-pass vs seek approach
///
/// The previous implementation issued one `ffmpeg -ss <ts> -frames:v 1`
/// per frame. At 768 frames the per-fork overhead was ~3.8 s. The
/// `select` filter approach evaluates every frame in the source up to the
/// last requested index (O(last_index)) but issues only one process, so
/// the net win is large for any non-trivial frame count.
///
/// ## Filter expression
///
/// The `select` expression is `eq(n\,0)+eq(n\,5)+...` where each `n` is a
/// 0-based frame index. ffmpeg's `select` filter compares the frame
/// presentation number `n` against each operand and outputs a frame when
/// any equality holds. The `+` operator is a logical OR.
///
/// ## Issue #601: fd-bearing source support
///
/// Accepts a [`VideoSource`] so the resolver in `src/server/media.rs` can
/// pass an opened fd instead of the canonical path. When the source is
/// fd-backed, the parent's master fd is rewound to offset 0, the
/// `pre_exec` hook clears `FD_CLOEXEC` so the fd is inherited by the
/// ffmpeg child, and ffmpeg opens `/dev/fd/N` (Linux/macOS). This closes
/// the TOCTOU window between the resolver's `canonicalize` and ffmpeg's
/// `open` call.
fn extract_frames_single_pass(
    source: &VideoSource,
    indices: &[usize],
    _video_fps: f64,
    limits: &VideoLimits,
) -> Result<Vec<DynamicImage>, VideoError> {
    if indices.is_empty() {
        return Ok(Vec::new());
    }

    let canonical_path = source.canonical_path().to_path_buf();

    // Reset the master fd offset to 0 (no-op for path sources) so the new
    // ffmpeg invocation reads the file from the beginning even after a
    // previous ffprobe call advanced the offset. See `VideoSource::rewind`
    // for the OFD-sharing rationale.
    #[cfg(unix)]
    if source.needs_offset_reset()
        && let Err(err) = source.rewind()
    {
        return Err(VideoError::Io {
            path: canonical_path,
            source: err,
        });
    }

    // Build the select filter: "eq(n\,0)+eq(n\,5)+eq(n\,30)+..."
    // Backslash-escape the comma inside eq() because the filter graph
    // parser would otherwise interpret the comma as a filter separator.
    let select_expr: String = indices
        .iter()
        .map(|idx| format!("eq(n\\,{idx})"))
        .collect::<Vec<_>>()
        .join("+");

    // Spawn ffmpeg with stdout piped; stderr is piped so we can surface
    // error messages on failure.
    let mut cmd = Command::new("ffmpeg");
    cmd.args(["-loglevel", "error", "-i"])
        .arg(source.ffmpeg_input())
        .args([
            "-vf",
            &format!("select='{select_expr}',setpts=N/FRAME_RATE/TB"),
            "-vsync",
            "vfr",
            "-f",
            "image2pipe",
            "-vcodec",
            "png",
            "-",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    source.configure_child(&mut cmd);
    let mut child = cmd.spawn().map_err(|err| {
        if err.kind() == io::ErrorKind::NotFound {
            VideoError::FfmpegMissing
        } else {
            VideoError::Io {
                path: source.canonical_path().to_path_buf(),
                source: err,
            }
        }
    })?;

    // We must read stdout and wait for the process concurrently — if we
    // call child.wait() before draining stdout the process blocks on a
    // full pipe buffer and we deadlock.
    let stdout = child
        .stdout
        .take()
        .expect("stdout was piped but is None — this is a bug");

    let frames = split_png_stream(
        stdout,
        source.canonical_path(),
        indices.len(),
        limits.max_png_frame_bytes,
    )?;

    // Collect stderr and wait for the process exit status.
    let output = child.wait_with_output().map_err(|err| VideoError::Io {
        path: source.canonical_path().to_path_buf(),
        source: err,
    })?;

    if !output.status.success() && frames.is_empty() {
        // Only treat non-zero exit as fatal when we got no frames at all.
        // ffmpeg sometimes exits non-zero for harmless reasons (e.g.,
        // stream ended cleanly but signals EIO on stdout close).
        return Err(VideoError::Extract {
            path: source.canonical_path().to_path_buf(),
            message: String::from_utf8_lossy(&output.stderr).into_owned(),
        });
    }

    Ok(frames)
}

/// Split a concatenated PNG byte stream into individual frames and decode
/// each one.
///
/// PNG files always start with the 8-byte signature `\x89PNG\r\n\x1a\n`
/// and end with a 12-byte IEND chunk (4-byte zero length + `IEND` + 4-byte
/// CRC). By scanning for IEND we can locate each frame boundary without a
/// full recursive PNG parser.
///
/// The algorithm:
/// 1. Read bytes from `reader` into a rolling buffer.
/// 2. Scan for the IEND terminator sequence (`\x00\x00\x00\x00IEND…CRC`).
///    The IEND chunk is always 12 bytes: 4-byte `0x00000000` length, 4-byte
///    `IEND` type, 4-byte CRC (always `\xaeB`\x82`).
/// 3. When found, everything from the start of the buffer to (and including)
///    the IEND chunk is one complete PNG file. Decode it and push the result.
/// 4. Advance the buffer past the IEND chunk and continue from the next PNG
///    signature.
fn split_png_stream<R: Read>(
    mut reader: R,
    path: &Path,
    expected_frames: usize,
    max_png_frame_bytes: usize,
) -> Result<Vec<DynamicImage>, VideoError> {
    // IEND chunk bytes: length (4 bytes = 0) + type "IEND" (4) + CRC (4).
    // The CRC of IEND with no data is always 0xAE426082.
    const IEND_MARKER: &[u8] = &[
        0x00, 0x00, 0x00, 0x00, b'I', b'E', b'N', b'D', 0xAE, 0x42, 0x60, 0x82,
    ];

    // Per-frame accumulation cap (resolved by the caller from `VideoLimits`).
    // A stream that never emits IEND would grow buf without bound; reject it
    // once the cap is exceeded.

    let mut frames = Vec::with_capacity(expected_frames);
    let mut buf: Vec<u8> = Vec::new();

    // Incremental read buffer to avoid a single giant allocation.
    let mut read_chunk = [0u8; 65536];

    loop {
        // Try to read more data. We break on EOF when the buffer is drained.
        let n = match reader.read(&mut read_chunk) {
            Ok(0) => 0,
            Ok(n) => {
                buf.extend_from_slice(&read_chunk[..n]);
                n
            }
            Err(ref e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => {
                return Err(VideoError::Io {
                    path: path.to_path_buf(),
                    source: e,
                });
            }
        };

        // Reject a frame that has grown beyond the configured cap. This
        // catches malformed or malicious streams that never emit IEND.
        if buf.len() > max_png_frame_bytes {
            return Err(VideoError::Extract {
                path: path.to_path_buf(),
                message: format!(
                    "PNG frame exceeded {max_png_frame_bytes} bytes without an IEND marker; \
                     stream may be malformed (set MLXCEL_VIDEO_MAX_PNG_FRAME_BYTES to override)"
                ),
            });
        }

        // Scan the accumulated buffer for IEND terminators.
        while let Some(iend_pos) = find_subsequence(&buf, IEND_MARKER) {
            let frame_end = iend_pos + PNG_IEND_CHUNK_LEN;
            if frame_end > buf.len() {
                // IEND marker spans into unread territory — wait for more bytes.
                break;
            }
            let png_bytes = buf[..frame_end].to_vec();
            buf.drain(..frame_end);

            // Verify the PNG signature so we don't try to decode garbage.
            if png_bytes.len() < PNG_SIGNATURE.len()
                || &png_bytes[..PNG_SIGNATURE.len()] != PNG_SIGNATURE
            {
                // Skip malformed data and try to recover at the next signature.
                continue;
            }

            let img = image::load_from_memory(&png_bytes).map_err(|err| VideoError::Decode {
                path: path.to_path_buf(),
                source: err,
            })?;
            frames.push(img);
        }

        // No more data from ffmpeg — stop reading.
        if n == 0 {
            break;
        }
    }

    Ok(frames)
}

/// Return the byte offset of the first occurrence of `needle` in `haystack`,
/// or `None` if not found.
fn find_subsequence(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() {
        return Some(0);
    }
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

#[cfg(test)]
#[path = "video_tests.rs"]
mod tests;
