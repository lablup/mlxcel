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

//! Owner-private locations for every compiled-artifact cache of the OpenXLA
//! backend (issue #3248).
//!
//! The caches used to be fixed names directly under the shared temp directory,
//! where another local user could create the directory first or plant a VMFB
//! (native code for a CPU target) next to a manifest whose fields are all
//! derivable from public inputs. They now live under the mlxcel cache root,
//! `${MLXCEL_CACHE_DIR:-$HOME/.cache/mlxcel}/xla/<cache>`, and nothing is read
//! from there unless the root, `xla`, and the cache directory are real
//! directories owned by the current user that neither the group nor others can
//! write, and the artifact itself is a regular file with the same properties
//! (checked on the opened descriptor, see [`mlxcel_core::private_dir`]).
//!
//! When a check fails the cache is refused, one warning names the path, and the
//! artifact is compiled into a fresh owner-only directory created with
//! `mkdtemp` under the temp directory instead. Nothing from the refused
//! location is read, moved, or deleted.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use mlxcel_core::private_dir::{
    PrivateDirError, create_fresh_private_dir, current_euid, ensure_private_dir, open_private_file,
    seal_private_file, verify_private_dir,
};

/// One compiled-artifact cache of the OpenXLA backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(not(feature = "iree"), allow(dead_code))]
pub(crate) enum XlaCache {
    /// Text prefill/decode bundles, the ROCm tuning spec, and Phi4MM audio.
    Vmfb,
    /// LLaVA-family vision towers.
    Vision,
    /// Qwen2-VL and Qwen2.5-VL vision towers.
    Qwen2Vl,
    /// Qwen3-VL vision towers.
    Qwen3Vl,
    /// Molmo vision towers.
    MolmoVision,
    /// Molmo2 vision towers.
    Molmo2Vision,
    /// Youtu-VL vision towers.
    YoutuVl,
    /// Gemma3n audio encoder and merge graphs.
    Gemma3nAudio,
    /// Native 4-bit GEMV code objects (HSACO).
    Qmv4,
    /// The auxiliary ABI smoke module.
    AuxSmoke,
    /// The bounded numeric-oracle probes.
    #[cfg_attr(not(feature = "micro-oracle"), allow(dead_code))]
    NumericProbes,
}

impl XlaCache {
    /// The directory name under `<cache root>/xla`.
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Vmfb => "vmfb",
            Self::Vision => "vision",
            Self::Qwen2Vl => "qwen2-vl",
            Self::Qwen3Vl => "qwen3-vl",
            Self::MolmoVision => "molmo-vision",
            Self::Molmo2Vision => "molmo2-vision",
            Self::YoutuVl => "youtu-vl",
            Self::Gemma3nAudio => "gemma3n-audio",
            Self::Qmv4 => "qmv4",
            Self::AuxSmoke => "aux-smoke",
            Self::NumericProbes => "numeric-probes",
        }
    }
}

/// The verified directory for `cache`, or a fresh private one when the
/// persistent location cannot be verified.
pub(crate) fn xla_cache_dir(cache: XlaCache) -> Result<PathBuf, String> {
    let explicit = std::env::var_os("MLXCEL_CACHE_DIR").is_some_and(|value| !value.is_empty());
    resolve_cache_dir(mlxcel_core::cache_root(), !explicit, cache, current_euid())
}

/// [`xla_cache_dir`] with the root, the root-repair policy, and the effective
/// uid as parameters, so a test can model a directory owned by another user.
pub(crate) fn resolve_cache_dir(
    root: Option<PathBuf>,
    default_root: bool,
    cache: XlaCache,
    euid: u32,
) -> Result<PathBuf, String> {
    let Some(root) = root else {
        warn_once(
            Path::new("<unset>"),
            "neither MLXCEL_CACHE_DIR nor a home directory is available",
        );
        return fresh_dir();
    };
    let subpath = Path::new("xla").join(cache.name());
    match ensure_private_dir(&root, &subpath, default_root, euid) {
        Ok(dir) => Ok(dir),
        Err(error) => {
            warn_refused(&error);
            fresh_dir()
        }
    }
}

/// The path to use for an artifact the caller would place at `path`.
///
/// Returns `path` when its directory is private to the current user and every
/// existing entry among `path` and `companions` (for example the manifest next
/// to a VMFB) passes [`open_private_file`]; an absent entry is fine and will be
/// written there. Otherwise warns once and returns the same file name in a
/// fresh private directory, so nothing at the refused location is reused. A
/// returned path that exists is safe to load.
pub(crate) fn artifact_slot(path: &Path, companions: &[PathBuf]) -> Result<PathBuf, String> {
    artifact_slot_for(path, companions, current_euid())
}

pub(crate) fn artifact_slot_for(
    path: &Path,
    companions: &[PathBuf],
    euid: u32,
) -> Result<PathBuf, String> {
    match check_slot(path, companions, euid) {
        Ok(()) => Ok(path.to_path_buf()),
        Err(error) => {
            warn_refused(&error);
            let name = path
                .file_name()
                .ok_or_else(|| format!("{} has no file name", path.display()))?;
            Ok(fresh_dir()?.join(name))
        }
    }
}

fn check_slot(path: &Path, companions: &[PathBuf], euid: u32) -> Result<(), PrivateDirError> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .ok_or_else(|| PrivateDirError::InvalidSubpath(path.to_path_buf()))?;
    if std::fs::symlink_metadata(parent).is_err() {
        // A schema-scoped subdirectory that does not exist yet: create it
        // owner-only inside the (already verified) cache directory.
        let mut builder = std::fs::DirBuilder::new();
        builder.recursive(true);
        #[cfg(unix)]
        std::os::unix::fs::DirBuilderExt::mode(&mut builder, 0o700);
        builder
            .create(parent)
            .map_err(|source| PrivateDirError::Io {
                path: parent.to_path_buf(),
                source,
            })?;
    }
    verify_private_dir(parent, euid)?;
    for entry in std::iter::once(path).chain(companions.iter().map(PathBuf::as_path)) {
        open_private_file(entry, euid)?;
    }
    Ok(())
}

/// Read a cached artifact if it exists and is private to the current user.
/// `Ok(None)` when absent; an untrusted entry is an error the caller handles by
/// rebuilding into [`artifact_slot`]'s fresh path.
pub(crate) fn read_private(path: &Path) -> Result<Option<Vec<u8>>, String> {
    use std::io::Read;
    let Some(mut file) = open_private_file(path, current_euid()).map_err(|e| e.to_string())? else {
        return Ok(None);
    };
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)
        .map_err(|error| format!("read {}: {error}", path.display()))?;
    Ok(Some(bytes))
}

/// A unique sibling of `path` for writing before an atomic rename.
pub(crate) fn temporary_sibling(path: &Path, purpose: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let mut name = path
        .file_name()
        .map_or_else(|| "artifact".into(), |name| name.to_os_string());
    name.push(format!(
        ".{purpose}.{}.{}.tmp",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    path.with_file_name(name)
}

/// Make `temporary` owner-only and rename it onto `path`, so a concurrent
/// reader never sees a partial artifact and the published file passes the
/// private-file check under any umask.
pub(crate) fn publish_private(temporary: &Path, path: &Path) -> Result<(), String> {
    let installed = seal_private_file(temporary)
        .map_err(|error| error.to_string())
        .and_then(|()| {
            std::fs::rename(temporary, path).map_err(|error| {
                format!(
                    "install {} as {}: {error}",
                    temporary.display(),
                    path.display()
                )
            })
        });
    if installed.is_err() {
        std::fs::remove_file(temporary).ok();
    }
    installed
}

/// Write `bytes` to `path` atomically, owner-only (0600). The temporary file is
/// created with `O_EXCL`, so a planted entry of that name is never written
/// through.
pub(crate) fn write_private_atomic(path: &Path, bytes: &[u8]) -> Result<(), String> {
    use std::io::Write;
    let temporary = temporary_sibling(path, "write");
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
    let written = options
        .open(&temporary)
        .and_then(|mut file| file.write_all(bytes))
        .map_err(|error| format!("write {}: {error}", temporary.display()));
    if let Err(error) = written {
        std::fs::remove_file(&temporary).ok();
        return Err(error);
    }
    publish_private(&temporary, path)
}

/// This process's fallback root: one `mkdtemp` directory under the temp
/// directory, created on first use. Each call returns a new subdirectory of it,
/// so two refused locations never share a fallback.
fn fresh_dir() -> Result<PathBuf, String> {
    static ROOT: OnceLock<Result<PathBuf, String>> = OnceLock::new();
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let root = ROOT
        .get_or_init(|| {
            create_fresh_private_dir(&std::env::temp_dir(), "mlxcel-xla-")
                .map_err(|error| format!("create a private XLA cache directory: {error}"))
        })
        .clone()?;
    let dir = root.join(NEXT.fetch_add(1, Ordering::Relaxed).to_string());
    let mut builder = std::fs::DirBuilder::new();
    #[cfg(unix)]
    std::os::unix::fs::DirBuilderExt::mode(&mut builder, 0o700);
    builder
        .create(&dir)
        .map_err(|error| format!("mkdir {}: {error}", dir.display()))?;
    Ok(dir)
}

fn warn_refused(error: &PrivateDirError) {
    warn_once(error.path(), &error.to_string());
}

/// Print one warning per refused path for the life of the process.
fn warn_once(path: &Path, reason: &str) {
    static WARNED: Mutex<Option<HashSet<PathBuf>>> = Mutex::new(None);
    let mut warned = WARNED
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if warned
        .get_or_insert_with(HashSet::new)
        .insert(path.to_path_buf())
    {
        eprintln!(
            "[mlxcel-xla] refusing the compiled-artifact cache at {}: {reason}; compiling into a \
             fresh private directory instead (fix the ownership or permissions, or remove it)",
            path.display()
        );
    }
}

#[cfg(all(test, unix))]
#[path = "cache_dir_tests.rs"]
mod tests;
