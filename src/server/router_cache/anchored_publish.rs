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

use super::{STAGING_DIR, anchored_delete::delete_tree_at};
use anyhow::{Context, anyhow};
use std::ffi::{CStr, CString};
use std::fs::{self, File, OpenOptions};
use std::io;
use std::mem::MaybeUninit;
use std::os::fd::{AsRawFd, FromRawFd};
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

static STAGE_COUNTER: AtomicU64 = AtomicU64::new(0);

pub(super) struct AnchoredStage {
    root: File,
    staging_parent: File,
    final_parent: File,
    stage: File,
    owner: CString,
    final_name: CString,
    stage_name: CString,
    published: bool,
}

impl AnchoredStage {
    pub(super) fn create(root_path: &Path, repo_id: &str) -> anyhow::Result<Self> {
        let (owner, final_name) = repo_segments(repo_id)?;
        fs::create_dir_all(root_path).with_context(|| {
            format!("failed to create model-store root {}", root_path.display())
        })?;
        let root = open_dir_path(root_path).with_context(|| {
            format!(
                "model-store root {} must be a real directory, not a symlink",
                root_path.display()
            )
        })?;
        let staging_name = cstring_segment(STAGING_DIR)?;
        ensure_child_dir(&root, &staging_name, 0o700)
            .context("failed to create router staging directory")?;
        let staging_parent = open_child_dir(&root, &staging_name)
            .context("router staging path must be a real directory")?;
        verify_private_staging_parent(&staging_parent)
            .context("router staging directory must be private to the server user")?;
        ensure_child_dir(&root, &owner, 0o755)
            .with_context(|| format!("failed to create owner directory for repo '{repo_id}'"))?;
        let final_parent = open_child_dir(&root, &owner).with_context(|| {
            format!("cache owner directory for repo '{repo_id}' must be a real directory")
        })?;
        if child_exists(&final_parent, &final_name)? {
            anyhow::bail!(
                "cache destination for '{repo_id}' already exists; refusing to overwrite it"
            );
        }
        let stage_name = create_unique_stage_dir(&staging_parent, repo_id)?;
        let stage = open_child_dir(&staging_parent, &stage_name)
            .context("created staging directory vanished before it could be opened")?;
        Ok(Self {
            root,
            staging_parent,
            final_parent,
            stage,
            owner,
            final_name,
            stage_name,
            published: false,
        })
    }

    pub(super) fn stage_fd(&self) -> std::os::fd::RawFd {
        use std::os::fd::AsRawFd;
        self.stage.as_raw_fd()
    }

    pub(super) fn publish(&mut self) -> anyhow::Result<()> {
        self.verify_anchors()?;
        if child_exists(&self.final_parent, &self.final_name)? {
            anyhow::bail!("cache destination appeared before publish; refusing to overwrite it");
        }
        rename_no_replace(
            &self.staging_parent,
            &self.stage_name,
            &self.final_parent,
            &self.final_name,
        )
        .context("atomic no-replace rename failed")?;
        self.published = true;
        Ok(())
    }

    pub(super) fn cleanup(&mut self) {
        if self.published {
            return;
        }
        if self.verify_staging_anchor().is_ok() && self.verify_stage_identity().is_ok() {
            let _ = delete_tree_at(&self.staging_parent, &self.stage_name);
        }
    }

    fn verify_anchors(&self) -> anyhow::Result<()> {
        self.verify_staging_anchor()?;
        self.verify_final_parent_anchor()?;
        self.verify_stage_identity()?;
        Ok(())
    }

    fn verify_staging_anchor(&self) -> anyhow::Result<()> {
        verify_child_identity(
            &self.root,
            &cstring_segment(STAGING_DIR)?,
            &self.staging_parent,
        )
        .context("router staging directory identity changed during download")
    }

    fn verify_final_parent_anchor(&self) -> anyhow::Result<()> {
        verify_child_identity(&self.root, &self.owner, &self.final_parent)
            .context("cache owner directory identity changed during download")
    }

    fn verify_stage_identity(&self) -> anyhow::Result<()> {
        verify_child_identity(&self.staging_parent, &self.stage_name, &self.stage)
            .context("staging directory identity changed during download")
    }
}

impl Drop for AnchoredStage {
    fn drop(&mut self) {
        self.cleanup();
    }
}

pub(super) fn repo_segments(repo_id: &str) -> anyhow::Result<(CString, CString)> {
    let mut parts = repo_id.split('/');
    let Some(owner) = parts.next() else {
        anyhow::bail!("repo id must be owner/name");
    };
    let Some(name) = parts.next() else {
        anyhow::bail!("repo id must be owner/name");
    };
    if parts.next().is_some() {
        anyhow::bail!("repo id must be owner/name");
    }
    Ok((cstring_segment(owner)?, cstring_segment(name)?))
}

pub(super) fn cstring_segment(segment: &str) -> anyhow::Result<CString> {
    let valid = !segment.is_empty()
        && segment != "."
        && segment != ".."
        && segment
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'));
    if !valid {
        anyhow::bail!("unsafe repository path segment '{segment}'");
    }
    CString::new(segment).map_err(|_| anyhow!("path segment contains NUL byte"))
}

fn safe_stage_name(repo_id: &str, seq: u64) -> CString {
    let safe = repo_id
        .bytes()
        .map(|b| {
            if b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-') {
                b as char
            } else {
                '_'
            }
        })
        .collect::<String>();
    CString::new(format!(
        "{}-{}-{}-{safe}",
        std::process::id(),
        chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default(),
        seq
    ))
    .expect("generated stage name has no NUL")
}

fn create_unique_stage_dir(parent: &File, repo_id: &str) -> anyhow::Result<CString> {
    for _ in 0..64 {
        let seq = STAGE_COUNTER.fetch_add(1, Ordering::Relaxed);
        let name = safe_stage_name(repo_id, seq);
        match mkdir_child(parent, &name, 0o700) {
            Ok(()) => return Ok(name),
            Err(err) if err.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(err) => return Err(err).context("failed to create private staging directory"),
        }
    }
    anyhow::bail!("failed to allocate a unique private staging directory")
}

pub(super) fn open_dir_path(path: &Path) -> io::Result<File> {
    OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOFOLLOW)
        .open(path)
}

pub(super) fn open_child_dir(parent: &File, name: &CStr) -> io::Result<File> {
    let fd = unsafe {
        libc::openat(
            parent.as_raw_fd(),
            name.as_ptr(),
            libc::O_RDONLY | libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOFOLLOW,
        )
    };
    if fd < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(unsafe { File::from_raw_fd(fd) })
    }
}

pub(super) fn ensure_child_dir(parent: &File, name: &CStr, mode: libc::mode_t) -> io::Result<()> {
    match mkdir_child(parent, name, mode) {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == io::ErrorKind::AlreadyExists => Ok(()),
        Err(err) => Err(err),
    }
}

fn mkdir_child(parent: &File, name: &CStr, mode: libc::mode_t) -> io::Result<()> {
    let rc = unsafe { libc::mkdirat(parent.as_raw_fd(), name.as_ptr(), mode) };
    if rc == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

pub(super) fn child_exists(parent: &File, name: &CStr) -> io::Result<bool> {
    let mut stat = MaybeUninit::<libc::stat>::uninit();
    let rc = unsafe {
        libc::fstatat(
            parent.as_raw_fd(),
            name.as_ptr(),
            stat.as_mut_ptr(),
            libc::AT_SYMLINK_NOFOLLOW,
        )
    };
    if rc == 0 {
        Ok(true)
    } else {
        let err = io::Error::last_os_error();
        if err.raw_os_error() == Some(libc::ENOENT) {
            Ok(false)
        } else {
            Err(err)
        }
    }
}

pub(super) fn verify_private_staging_parent(dir: &File) -> anyhow::Result<()> {
    let meta = dir.metadata()?;
    let mode = meta.mode() & 0o777;
    let uid = meta.uid();
    let euid = unsafe { libc::geteuid() };
    if mode == 0o700 && uid == euid {
        Ok(())
    } else {
        anyhow::bail!(
            "staging directory must be owned by uid {euid} with mode 0700; found uid {uid} mode {mode:o}"
        )
    }
}

pub(super) fn verify_child_identity(parent: &File, name: &CStr, held: &File) -> anyhow::Result<()> {
    let current = open_child_dir(parent, name)?;
    verify_same_file_identity(held, &current)
}

pub(super) fn verify_same_file_identity(expected: &File, actual: &File) -> anyhow::Result<()> {
    let expected_meta = expected.metadata()?;
    let actual_meta = actual.metadata()?;
    if expected_meta.dev() == actual_meta.dev() && expected_meta.ino() == actual_meta.ino() {
        Ok(())
    } else {
        anyhow::bail!("directory identity mismatch")
    }
}

pub(super) fn regular_file_nonempty_at(parent: &File, name: &CStr) -> io::Result<bool> {
    let mut stat = MaybeUninit::<libc::stat>::uninit();
    let rc = unsafe {
        libc::fstatat(
            parent.as_raw_fd(),
            name.as_ptr(),
            stat.as_mut_ptr(),
            libc::AT_SYMLINK_NOFOLLOW,
        )
    };
    if rc != 0 {
        let err = io::Error::last_os_error();
        if err.raw_os_error() == Some(libc::ENOENT) {
            return Ok(false);
        }
        return Err(err);
    }
    let stat = unsafe { stat.assume_init() };
    Ok(is_regular_mode(stat.st_mode) && stat.st_size > 0)
}

#[cfg(target_os = "linux")]
pub(super) fn rename_no_replace(
    from_parent: &File,
    from_name: &CStr,
    to_parent: &File,
    to_name: &CStr,
) -> io::Result<()> {
    let rc = unsafe {
        libc::syscall(
            libc::SYS_renameat2,
            from_parent.as_raw_fd(),
            from_name.as_ptr(),
            to_parent.as_raw_fd(),
            to_name.as_ptr(),
            libc::RENAME_NOREPLACE,
        )
    };
    if rc == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
pub(super) fn rename_no_replace(
    from_parent: &File,
    from_name: &CStr,
    to_parent: &File,
    to_name: &CStr,
) -> io::Result<()> {
    let rc = unsafe {
        libc::renameatx_np(
            from_parent.as_raw_fd(),
            from_name.as_ptr(),
            to_parent.as_raw_fd(),
            to_name.as_ptr(),
            libc::RENAME_EXCL,
        )
    };
    if rc == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "ios")))]
pub(super) fn rename_no_replace(
    _from_parent: &File,
    _from_name: &CStr,
    _to_parent: &File,
    _to_name: &CStr,
) -> io::Result<()> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "atomic no-replace directory rename is not available on this platform",
    ))
}

fn is_regular_mode(mode: libc::mode_t) -> bool {
    (mode & libc::S_IFMT) == libc::S_IFREG
}

#[cfg(test)]
#[path = "anchored_publish_tests.rs"]
mod tests;
