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

//! Owner-private directories and files for caches whose contents are loaded
//! as code (issue #3248).
//!
//! A compiled artifact read back from disk (an IREE VMFB, a GPU code object)
//! is only as trustworthy as whoever could write it. These checks accept a
//! directory or file only when it is owned by the given effective uid, is not
//! a symlink, and is not group- or world-writable. Every check is made on an
//! opened descriptor (`O_NOFOLLOW`, then `fstat`), not on a path that could be
//! swapped between the check and the use. The effective uid is a parameter so
//! a test can model a directory owned by another user without root.
//!
//! The group-write repair of directories mlxcel owns is the PTX cache's
//! (`ptx_cache_dirs::make_private_dir`, #2765 and #2797), shared here rather
//! than reimplemented.

use std::fmt;
use std::fs::File;
use std::io;
use std::path::{Component, Path, PathBuf};

#[cfg(unix)]
use std::os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt, PermissionsExt};

/// Group- and world-write bits: either one lets another user plant or replace
/// an entry.
#[cfg(unix)]
const GROUP_OR_WORLD_WRITE: u32 = 0o022;

/// Why a cache directory or file was refused.
#[derive(Debug)]
#[non_exhaustive]
pub enum PrivateDirError {
    /// The cache root is relative, so it would resolve against the current
    /// directory.
    RelativeRoot(PathBuf),
    /// A path below the root is absolute or climbs out of it.
    InvalidSubpath(PathBuf),
    /// The entry is a symlink.
    Symlink(PathBuf),
    /// A directory was expected.
    NotADirectory(PathBuf),
    /// A regular file was expected.
    NotARegularFile(PathBuf),
    /// The entry belongs to another user.
    ForeignOwner {
        /// The refused entry.
        path: PathBuf,
        /// Its owner.
        owner: u32,
        /// The effective uid it was checked against.
        euid: u32,
    },
    /// The entry is group- or world-writable.
    Writable {
        /// The refused entry.
        path: PathBuf,
        /// Its permission bits.
        mode: u32,
    },
    /// The entry could not be created, opened, or inspected.
    Io {
        /// The entry.
        path: PathBuf,
        /// The underlying error.
        source: io::Error,
    },
}

impl PrivateDirError {
    /// The path this error is about.
    #[must_use]
    pub fn path(&self) -> &Path {
        match self {
            Self::RelativeRoot(path)
            | Self::InvalidSubpath(path)
            | Self::Symlink(path)
            | Self::NotADirectory(path)
            | Self::NotARegularFile(path)
            | Self::ForeignOwner { path, .. }
            | Self::Writable { path, .. }
            | Self::Io { path, .. } => path,
        }
    }
}

impl fmt::Display for PrivateDirError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RelativeRoot(path) => write!(f, "{} is not an absolute path", path.display()),
            Self::InvalidSubpath(path) => {
                write!(f, "{} is not a plain relative path", path.display())
            }
            Self::Symlink(path) => write!(f, "{} is a symlink", path.display()),
            Self::NotADirectory(path) => write!(f, "{} is not a directory", path.display()),
            Self::NotARegularFile(path) => write!(f, "{} is not a regular file", path.display()),
            Self::ForeignOwner { path, owner, euid } => write!(
                f,
                "{} is owned by uid {owner}, not the current user (uid {euid})",
                path.display()
            ),
            Self::Writable { path, mode } => write!(
                f,
                "{} is group- or world-writable (mode {mode:o})",
                path.display()
            ),
            Self::Io { path, source } => write!(f, "{}: {source}", path.display()),
        }
    }
}

impl std::error::Error for PrivateDirError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// The effective uid of this process (0 where uids do not exist).
#[must_use]
pub fn current_euid() -> u32 {
    #[cfg(unix)]
    {
        // SAFETY: geteuid has no preconditions and cannot fail.
        unsafe { libc::geteuid() }
    }
    #[cfg(not(unix))]
    {
        0
    }
}

/// Clear the group-write bit of a real directory owned by `euid`, keeping
/// every other bit; anything else is left alone. This is the PTX cache's
/// repair: a umask of 002 leaves directories group-writable, and only group
/// write is cleared because an other-writable directory may already hold
/// someone else's files and must stay refused.
pub fn clear_group_write(path: &Path, euid: u32) {
    #[cfg(unix)]
    crate::ptx_cache_dirs::make_private_dir(path, euid);
    #[cfg(not(unix))]
    let _ = (path, euid);
}

/// Open `path` with `flags | O_NOFOLLOW | O_NONBLOCK`, mapping a symlink to
/// [`PrivateDirError::Symlink`]. `O_NONBLOCK` keeps a FIFO swapped in for the
/// entry from blocking the open (#2765).
#[cfg(unix)]
fn open_nofollow(path: &Path, flags: i32) -> Result<File, PrivateDirError> {
    std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(flags | libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)
        .map_err(|source| match source.raw_os_error() {
            Some(libc::ELOOP) => PrivateDirError::Symlink(path.to_path_buf()),
            Some(libc::ENOTDIR)
                if std::fs::symlink_metadata(path).is_ok_and(|m| m.file_type().is_symlink()) =>
            {
                PrivateDirError::Symlink(path.to_path_buf())
            }
            Some(libc::ENOTDIR) => PrivateDirError::NotADirectory(path.to_path_buf()),
            _ => PrivateDirError::Io {
                path: path.to_path_buf(),
                source,
            },
        })
}

/// Owner and mode checks on an opened descriptor's metadata.
#[cfg(unix)]
fn check_owner_and_mode(
    path: &Path,
    metadata: &std::fs::Metadata,
    euid: u32,
) -> Result<(), PrivateDirError> {
    if metadata.uid() != euid {
        return Err(PrivateDirError::ForeignOwner {
            path: path.to_path_buf(),
            owner: metadata.uid(),
            euid,
        });
    }
    let mode = metadata.mode() & 0o7777;
    if mode & GROUP_OR_WORLD_WRITE != 0 {
        return Err(PrivateDirError::Writable {
            path: path.to_path_buf(),
            mode,
        });
    }
    Ok(())
}

/// Verify that `path` is a real directory (not a symlink) owned by `euid`
/// that neither the group nor other users can write.
pub fn verify_private_dir(path: &Path, euid: u32) -> Result<(), PrivateDirError> {
    #[cfg(unix)]
    {
        let dir = open_nofollow(path, libc::O_DIRECTORY)?;
        let metadata = dir.metadata().map_err(|source| PrivateDirError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        if !metadata.is_dir() {
            return Err(PrivateDirError::NotADirectory(path.to_path_buf()));
        }
        check_owner_and_mode(path, &metadata, euid)
    }
    #[cfg(not(unix))]
    {
        let _ = euid;
        let metadata = std::fs::symlink_metadata(path).map_err(|source| PrivateDirError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        if metadata.file_type().is_symlink() {
            return Err(PrivateDirError::Symlink(path.to_path_buf()));
        }
        if !metadata.is_dir() {
            return Err(PrivateDirError::NotADirectory(path.to_path_buf()));
        }
        Ok(())
    }
}

/// Create `root/subpath` (mode 0700 for every directory created) and verify
/// that `root` and every directory below it on the way to the leaf is private
/// to `euid` (see [`verify_private_dir`]), and that no one else can replace
/// `root` through a directory above it (see [`verify_trusted_ancestors`]).
/// Returns the leaf, spelled through the canonical parent of `root`, so a
/// symlink above `root` swapped after the check cannot redirect later opens
/// by path.
///
/// Group write is first cleared ([`clear_group_write`]) on the directories
/// below `root`, which mlxcel owns, and on `root` itself only when
/// `repair_root` is set (the default cache root; an explicit root is the
/// user's choice and is checked as found). Nothing above `root` is modified.
pub fn ensure_private_dir(
    root: &Path,
    subpath: &Path,
    repair_root: bool,
    euid: u32,
) -> Result<PathBuf, PrivateDirError> {
    if !root.is_absolute() {
        return Err(PrivateDirError::RelativeRoot(root.to_path_buf()));
    }
    if root
        .components()
        .any(|component| matches!(component, Component::ParentDir))
    {
        return Err(PrivateDirError::InvalidSubpath(root.to_path_buf()));
    }
    let mut chain = vec![root.to_path_buf()];
    for component in subpath.components() {
        let Component::Normal(name) = component else {
            return Err(PrivateDirError::InvalidSubpath(subpath.to_path_buf()));
        };
        let next = chain[chain.len() - 1].join(name);
        chain.push(next);
    }
    let requested_leaf = chain[chain.len() - 1].clone();
    let mut builder = std::fs::DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    builder.mode(0o700);
    builder
        .create(&requested_leaf)
        .map_err(|source| PrivateDirError::Io {
            path: requested_leaf.clone(),
            source,
        })?;
    // Respell the chain through the canonical parent of `root` and check that
    // parent and everything above it: whoever can write a directory above the
    // root can rename the verified root away and put their own in its place
    // after the check, and the IREE runtime opens a VMFB by path.
    if let (Some(parent), Some(name)) = (root.parent(), root.file_name()) {
        let parent = std::fs::canonicalize(parent).map_err(|source| PrivateDirError::Io {
            path: parent.to_path_buf(),
            source,
        })?;
        verify_trusted_ancestors(&parent, euid)?;
        let canonical_root = parent.join(name);
        chain = chain
            .iter()
            .map(|dir| match dir.strip_prefix(root) {
                Ok(rest) if rest.as_os_str().is_empty() => canonical_root.clone(),
                Ok(rest) => canonical_root.join(rest),
                Err(_) => dir.clone(),
            })
            .collect();
    }
    let leaf = chain[chain.len() - 1].clone();
    if repair_root {
        clear_group_write(&chain[0], euid);
    }
    for dir in &chain[1..] {
        clear_group_write(dir, euid);
    }
    for dir in &chain {
        verify_private_dir(dir, euid)?;
    }
    Ok(leaf)
}

/// Verify that `dir` and every directory above it up to `/` cannot be used by
/// another user to swap an entry below them: each is a real directory (not a
/// symlink) owned by `euid` or by root, and is not other-writable unless it is
/// a root-owned sticky directory such as `/tmp`. Group write is accepted only
/// when the directory's group is this process's effective gid (the user's own
/// group under the user-private-group convention, where a umask of 002 leaves
/// `~/.cache` group-writable); any other group-writable ancestor is refused.
/// This is the PTX cache overlay's ancestor rule (#2539), except that the
/// user's own group is trusted, as `ensure_private_dir` never modifies a
/// directory above the cache root. `dir` should be canonical; it is not
/// resolved here.
pub fn verify_trusted_ancestors(dir: &Path, euid: u32) -> Result<(), PrivateDirError> {
    #[cfg(unix)]
    {
        // SAFETY: getegid has no preconditions and cannot fail.
        let egid = unsafe { libc::getegid() };
        let mut current = Some(dir);
        while let Some(component) = current {
            let metadata =
                std::fs::symlink_metadata(component).map_err(|source| PrivateDirError::Io {
                    path: component.to_path_buf(),
                    source,
                })?;
            if metadata.file_type().is_symlink() {
                return Err(PrivateDirError::Symlink(component.to_path_buf()));
            }
            if !metadata.is_dir() {
                return Err(PrivateDirError::NotADirectory(component.to_path_buf()));
            }
            if metadata.uid() != euid && metadata.uid() != 0 {
                return Err(PrivateDirError::ForeignOwner {
                    path: component.to_path_buf(),
                    owner: metadata.uid(),
                    euid,
                });
            }
            let mode = metadata.mode() & 0o7777;
            let root_sticky = metadata.uid() == 0 && mode & 0o1000 != 0;
            let other_write = mode & 0o002 != 0;
            let foreign_group_write = mode & 0o020 != 0 && metadata.gid() != egid;
            if (other_write || foreign_group_write) && !root_sticky {
                return Err(PrivateDirError::Writable {
                    path: component.to_path_buf(),
                    mode,
                });
            }
            current = component.parent();
        }
        Ok(())
    }
    #[cfg(not(unix))]
    {
        let _ = (dir, euid);
        Ok(())
    }
}

/// Open an existing cache file for reading after checking it on the opened
/// descriptor: a regular file (not a symlink), owned by `euid`, neither
/// group- nor world-writable. `Ok(None)` when nothing is at `path`.
pub fn open_private_file(path: &Path, euid: u32) -> Result<Option<File>, PrivateDirError> {
    #[cfg(unix)]
    let opened = open_nofollow(path, 0);
    #[cfg(not(unix))]
    let opened = {
        let _ = euid;
        if std::fs::symlink_metadata(path).is_ok_and(|m| m.file_type().is_symlink()) {
            return Err(PrivateDirError::Symlink(path.to_path_buf()));
        }
        File::open(path).map_err(|source| PrivateDirError::Io {
            path: path.to_path_buf(),
            source,
        })
    };
    let file = match opened {
        Ok(file) => file,
        Err(PrivateDirError::Io { source, .. }) if source.kind() == io::ErrorKind::NotFound => {
            return Ok(None);
        }
        Err(error) => return Err(error),
    };
    let metadata = file.metadata().map_err(|source| PrivateDirError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    if !metadata.is_file() {
        return Err(PrivateDirError::NotARegularFile(path.to_path_buf()));
    }
    #[cfg(unix)]
    check_owner_and_mode(path, &metadata, euid)?;
    Ok(Some(file))
}

/// Make a file this process just wrote owner-only (0600) through its
/// descriptor, so a umask of 002 cannot leave a cache entry group-writable
/// and later refused. A symlink or a non-regular file is an error.
pub fn seal_private_file(path: &Path) -> Result<(), PrivateDirError> {
    #[cfg(unix)]
    {
        let file = open_nofollow(path, 0)?;
        let metadata = file.metadata().map_err(|source| PrivateDirError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        if !metadata.is_file() {
            return Err(PrivateDirError::NotARegularFile(path.to_path_buf()));
        }
        file.set_permissions(std::fs::Permissions::from_mode(0o600))
            .map_err(|source| PrivateDirError::Io {
                path: path.to_path_buf(),
                source,
            })
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        Ok(())
    }
}

/// Create a new, uniquely named, owner-only directory under `parent` whose
/// name starts with `prefix` (`mkdtemp`, so it cannot already exist).
pub fn create_fresh_private_dir(parent: &Path, prefix: &str) -> Result<PathBuf, PrivateDirError> {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::{OsStrExt, OsStringExt};
        let template = parent.join(format!("{prefix}XXXXXX"));
        let mut bytes = template.as_os_str().as_bytes().to_vec();
        if bytes.contains(&0) {
            return Err(PrivateDirError::InvalidSubpath(template));
        }
        bytes.push(0);
        // SAFETY: bytes is a NUL-terminated buffer that mkdtemp rewrites in
        // place (only the trailing XXXXXX) and that outlives the call.
        let created = unsafe { libc::mkdtemp(bytes.as_mut_ptr().cast()) };
        if created.is_null() {
            return Err(PrivateDirError::Io {
                path: template,
                source: io::Error::last_os_error(),
            });
        }
        bytes.pop();
        Ok(PathBuf::from(std::ffi::OsString::from_vec(bytes)))
    }
    #[cfg(not(unix))]
    {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |duration| duration.as_nanos());
        let path = parent.join(format!("{prefix}{}-{nonce}", std::process::id()));
        std::fs::create_dir(&path).map_err(|source| PrivateDirError::Io {
            path: path.clone(),
            source,
        })?;
        Ok(path)
    }
}

#[cfg(all(test, unix))]
#[path = "private_dir_tests.rs"]
mod tests;
