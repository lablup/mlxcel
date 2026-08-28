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

//! The confined root local media URLs are resolved against (`--media-path`).
//!
//! This is the mlxcel side of llama-server b10621's `--media-path` (issue
//! #1451). Upstream keeps the value on `common_params::media_path`, appends the
//! platform separator to it at parse time, and resolves a request's `file://`
//! URL by string concatenation in
//! [`handle_media`](https://github.com/ggml-org/llama.cpp/blob/c1d0e7a004015f23bc0233470b747b596f29b264/tools/server/server-common.cpp):
//!
//! ```text
//! if (media_path.empty())            -> "file:// URLs are not allowed unless --media-path is specified"
//! file_path = url.substr(7)          // everything after "file://"
//! if (!fs_validate_filename(file_path, true)) -> "file path is not allowed: ..."
//! std::ifstream file(media_path + file_path, std::ios::binary)
//! ```
//!
//! Three properties of that upstream code decide this module's shape:
//!
//! 1. **It concatenates, it does not join.** `file:///etc/passwd` becomes
//!    `<root>//etc/passwd`, not `/etc/passwd`. A Rust [`Path::join`] with a
//!    leading-slash relative part would replace the root instead of extending
//!    it, so leading separators are stripped before joining. Getting this wrong
//!    turns a compatibility feature into an arbitrary-file read.
//! 2. **It never percent-decodes.** `%2e%2e` names a file literally called
//!    `%2e%2e`. Not decoding is therefore both compatible and safe, and this
//!    module additionally refuses the traversal-shaped escapes outright so the
//!    property is asserted rather than inherited from an absent call.
//! 3. **It is a pure string check.** `fs_validate_filename` rejects `..`,
//!    control characters, the Windows-illegal punctuation, and a few Unicode
//!    look-alikes, but it cannot see the filesystem: a symlink inside the media
//!    root pointing at `/etc/shadow` passes every one of its rules and upstream
//!    reads the target. mlxcel canonicalizes and requires the result to stay
//!    inside the canonical root, then opens with `O_NOFOLLOW` so the
//!    canonicalize-to-open window cannot be won by swapping the last component.
//!    That is deliberately stricter than b10621 and is recorded as a divergence
//!    on the `--media-path` manifest entry.
//!
//! Before #1451 mlxcel had no root at all: `try_read_image_url_with_limits`
//! opened any `file://` path, and any bare string that happened to name an
//! existing file, straight off the request. Every local read now goes through
//! [`resolve_media_file`].
//!
//! Used by: chat / responses / anthropic image parts, chat audio parts,
//! embeddings and rerank image inputs.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// b10621's own wording for a `file://` URL with no configured root.
pub(crate) const NO_MEDIA_ROOT_MESSAGE: &str =
    "file:// URLs are not allowed unless --media-path is specified";

/// Why a local media reference was refused.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub(crate) enum MediaPathError {
    /// No `--media-path` is configured, so local files are disabled.
    #[error("{NO_MEDIA_ROOT_MESSAGE}")]
    NoRoot,
    /// The relative path failed the b10621-compatible name validation.
    #[error("file path is not allowed: {path}")]
    NotAllowed { path: String },
    /// The path does not resolve to an existing file under the root.
    #[error("file does not exist or cannot be opened: {path}")]
    Unresolvable { path: String },
    /// The resolved path left the configured root.
    #[error("file path escapes the --media-path root: {path}")]
    Escape { path: String },
    /// The resolved path is not a regular file.
    #[error("file path is not a regular file: {path}")]
    NotRegular { path: String },
}

/// The process-wide media root, installed once at startup.
static MEDIA_ROOT: OnceLock<Option<PathBuf>> = OnceLock::new();

/// Install the confined root before the server starts accepting requests.
///
/// Installing the same value twice is accepted so the model-switch path, which
/// re-enters startup, is not an error.
///
/// # Errors
///
/// Returns `Err` when a different root was already installed.
pub fn install_media_root(root: Option<PathBuf>) -> Result<(), String> {
    match MEDIA_ROOT.set(root.clone()) {
        Ok(()) => Ok(()),
        Err(_) if MEDIA_ROOT.get() == Some(&root) => Ok(()),
        Err(_) => Err(format!(
            "a --media-path root is already installed ({:?}); it must be set once, before the \
             server starts",
            MEDIA_ROOT.get()
        )),
    }
}

/// The installed root, if the operator configured one.
pub(crate) fn media_root() -> Option<&'static Path> {
    MEDIA_ROOT.get().and_then(|slot| slot.as_deref())
}

/// b10621's `fs_validate_filename(path, /*allow_subdirs=*/true)`, plus the
/// percent-encoded traversal refusal described in the module documentation.
///
/// Upstream's rules, in its own order: non-empty, at most 255 bytes, valid
/// UTF-8 with no overlong encodings, no C0/C1 control characters or `DEL`, none
/// of the Unicode separator look-alikes (`U+FF0E`, `U+2215`, `U+2216`), no
/// UTF-16 surrogate, no `U+FFFD` or `U+FEFF`, none of `: * ? " < > |`, no
/// leading or trailing ASCII space, no trailing `.`, no `..` anywhere, and not
/// exactly `.`.
pub(crate) fn validate_media_filename(path: &str) -> Result<(), MediaPathError> {
    let refuse = || {
        Err(MediaPathError::NotAllowed {
            path: path.to_owned(),
        })
    };
    if path.is_empty() || path.len() > 255 {
        return refuse();
    }
    for c in path.chars() {
        // `char` is already a scalar value, so Rust's own UTF-8 decoding has
        // ruled out overlong encodings and surrogates before this loop; the
        // remaining codepoint rules are upstream's.
        if c <= '\u{1F}'
            || c == '\u{7F}'
            || ('\u{80}'..='\u{9F}').contains(&c)
            || c == '\u{FF0E}'
            || c == '\u{2215}'
            || c == '\u{2216}'
            || c == '\u{FFFD}'
            || c == '\u{FEFF}'
            || matches!(c, ':' | '*' | '?' | '"' | '<' | '>' | '|')
        {
            return refuse();
        }
    }
    if path.starts_with(' ') || path.ends_with(' ') || path.ends_with('.') {
        return refuse();
    }
    if path.contains("..") || path == "." {
        return refuse();
    }
    // Upstream never percent-decodes, so `%2e%2e` is a literal filename there
    // and cannot traverse. Refusing the traversal-shaped escapes makes that a
    // checked property here rather than one inherited from an absent call, and
    // costs only filenames that literally spell a separator or a dot in
    // percent-encoding. An ordinary `%20` is untouched.
    if contains_encoded_traversal(path) {
        return refuse();
    }
    Ok(())
}

/// True when `path` carries a percent escape for `.`, `/`, `\` or NUL.
fn contains_encoded_traversal(path: &str) -> bool {
    const ENCODED: [&str; 4] = ["%2e", "%2f", "%5c", "%00"];
    let lowered = path.to_ascii_lowercase();
    ENCODED.iter().any(|needle| lowered.contains(needle))
}

/// Strip the leading separators upstream's concatenation makes inert.
///
/// `file:///etc/passwd` reaches upstream as `/etc/passwd` and is concatenated
/// onto the root, landing at `<root>//etc/passwd`. Stripping the leading
/// separators and joining reproduces that; joining without stripping would let
/// the absolute path replace the root.
fn relative_component(path: &str) -> &str {
    path.trim_start_matches(['/', '\\'])
}

/// Resolve a request's local media reference to a canonical file inside the
/// configured root.
///
/// `reference` is either the whole `file://...` URL or a bare relative path;
/// both are resolved identically, because a bare path is what an operator who
/// configured `--media-path` most naturally writes and neither may leave the
/// root.
///
/// # Errors
///
/// Every rejection is fail-closed: no root configured, a name upstream would
/// refuse, a path that does not resolve, a resolved path outside the root, or a
/// resolved path that is not a regular file.
pub(crate) async fn resolve_media_file(reference: &str) -> Result<PathBuf, MediaPathError> {
    let Some(root) = media_root() else {
        return Err(MediaPathError::NoRoot);
    };
    resolve_media_file_in(root, reference).await
}

/// [`resolve_media_file`] against an explicit root.
///
/// The root must already be canonical; [`crate::cli::multimodal_compat_args::resolve_media_root_path`]
/// is what makes it so at startup. Separated from the global-reading wrapper so
/// the containment rules can be tested against a temporary directory without
/// installing a process-wide root.
pub(crate) async fn resolve_media_file_in(
    root: &Path,
    reference: &str,
) -> Result<PathBuf, MediaPathError> {
    let raw = reference.strip_prefix("file://").unwrap_or(reference);
    validate_media_filename(raw)?;
    let relative = relative_component(raw);
    if relative.is_empty() {
        return Err(MediaPathError::NotAllowed {
            path: raw.to_owned(),
        });
    }

    let canonical = tokio::fs::canonicalize(root.join(relative))
        .await
        .map_err(|_| MediaPathError::Unresolvable {
            path: raw.to_owned(),
        })?;
    // The root was canonicalized at startup, so this prefix test compares two
    // fully resolved paths: a symlink anywhere in `relative` has already been
    // followed and shows up here as a canonical path outside the root.
    if !canonical.starts_with(root) {
        return Err(MediaPathError::Escape {
            path: raw.to_owned(),
        });
    }
    let metadata =
        tokio::fs::metadata(&canonical)
            .await
            .map_err(|_| MediaPathError::Unresolvable {
                path: raw.to_owned(),
            })?;
    if !metadata.is_file() {
        return Err(MediaPathError::NotRegular {
            path: raw.to_owned(),
        });
    }
    Ok(canonical)
}

/// Open a resolved media file without following a final symlink.
///
/// `O_NOFOLLOW` closes the canonicalize-to-open window: a swap of the last
/// component for a symlink after [`resolve_media_file`] returned makes the
/// `open` fail with `ELOOP` instead of silently reading the new target. The
/// same primitive already guards the video path (`media.rs::open_video_source`).
///
/// # Errors
///
/// Returns [`MediaPathError::Unresolvable`] when the file cannot be opened,
/// which includes the symlink-swap case.
pub(crate) async fn open_confined(canonical: &Path) -> Result<tokio::fs::File, MediaPathError> {
    #[cfg(unix)]
    {
        // `custom_flags` comes from `std::os::unix::fs::OpenOptionsExt`, which
        // `tokio::fs::OpenOptions` re-exposes on its own type.
        tokio::fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW)
            .open(canonical)
            .await
            .map_err(|_| MediaPathError::Unresolvable {
                path: canonical.display().to_string(),
            })
    }
    #[cfg(not(unix))]
    {
        tokio::fs::File::open(canonical)
            .await
            .map_err(|_| MediaPathError::Unresolvable {
                path: canonical.display().to_string(),
            })
    }
}

/// Install a shared, process-wide media root for the tests that need a
/// readable local file.
///
/// The real root is a `OnceLock`, so the first installer in a test binary wins
/// and every later identical call is accepted. Routing every test that wants a
/// root through this one helper is what keeps that first-writer-wins rule from
/// depending on test order.
#[cfg(test)]
pub(crate) fn install_test_root_once() -> &'static Path {
    static TEST_ROOT: OnceLock<PathBuf> = OnceLock::new();
    let root = TEST_ROOT.get_or_init(|| {
        let dir = std::env::temp_dir().join(format!("mlxcel-media-root-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).expect("create the shared test media root");
        std::fs::canonicalize(&dir).expect("canonicalize the shared test media root")
    });
    install_media_root(Some(root.clone())).expect("install the shared test media root");
    media_root().expect("the shared test media root is installed")
}

#[cfg(test)]
#[path = "media_root_tests.rs"]
mod tests;
